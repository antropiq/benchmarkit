"""Tests for language-server discovery, the LSP hub, its API and chat wiring.

Mirrors tests/test_mcp.py. The hub is exercised against
``tests/fake_langserver.py`` (a stdlib stdio JSON-RPC server), never against
a real pyright/tsserver — exactly like the MCP tests never spawn real
connectors.
"""

import asyncio
import inspect
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.core.lsphub import constants as lsp_constants
from src.core.lsphub import (
    LspManager,
    action_of_tool_name,
    all_servers,
    TOOL_SPECS,
    find_project_config,
    lsp_reach_is_refused,
    parse_tsserver_request,
    restricted_lsp_roots,
    tsserver_project_info,
    tsserver_response_params,
    load_lsp_servers,
    set_lsp_enabled,
    tool_name_for,
    tools_for_server,
)

from src.core import proctree, proctree_jobs
from src.core import mcphub
from src.core.mcphub import tools_to_openai_format
from src.core.mcphub.validation import expand_env_list, expand_env_map

from tests.conftest import skip_unless, proc_fs_probe

FAKE_SERVER = str(Path(__file__).parent / "fake_langserver.py")
# The loader requires an existing workspaceRoot; the repo itself always does.
REPO_ROOT = str(Path(FAKE_SERVER).resolve().parents[2])


def _entry(command: Any, **overrides: Any) -> Dict[str, Any]:
    entry = {
        "enable": True,
        "extensions": [".py"],
        "command": command,
        "workspaceRoot": REPO_ROOT,
    }
    entry.update(overrides)
    return entry


# Values are typed loosely on purpose: the validation tests feed deliberately
# malformed entries (a bare string, ...) to check the loader rejects them.
def _servers_doc(**entries: Any) -> str:
    return json.dumps(entries, indent=2)


# No reach-seam fixture here any more (lspfinal-04). One used to pin the seam
# open on a platform that refused every declared server, so the lifecycle,
# tool-call and tsserver-relay paths had a hub to exercise. The refusal is gone
# — it answers only whether the hub's guardrail list speaks this filesystem — so
# the real answers are the ones this suite was written against, on both
# platforms, and a fixture that patched them would be asserting a state the app
# can no longer be in. Where the refusal *is* the subject —
# ``TestLocalReachRefusal``, the reach-note cases in ``TestLoadLspServers`` —
# the test forces its own ``guardrails_speak_here`` to ``False``.

# ---------------------------------------------------------------------------
# Config discovery / validation
# ---------------------------------------------------------------------------


class TestLoadLspServers:
    def test_missing_file_yields_nothing(self, tmp_path):
        assert load_lsp_servers(tmp_path / "nope.json") == ([], [])

    def test_invalid_json_is_one_file_level_error(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text("{oops", encoding="utf-8")
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert len(errors) == 1 and "not readable JSON" in errors[0]

    def test_top_level_must_be_an_object(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text("[1, 2]", encoding="utf-8")
        _servers, errors = load_lsp_servers(path)
        assert len(errors) == 1 and "object of servers" in errors[0]

    def test_valid_entry_is_normalized(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                pyright=_entry(
                    ["pyright-langserver", "--stdio"],
                    label="Pyright",
                    description="Python.",
                    extensions=[".PY", ".py"],
                    tools=["hover", "bogus_action"],
                    env={"PATH": "/x"},
                    initializationOptions={"typeCheckingMode": "standard"},
                    unknown_key="ignored",
                )
            ),
            encoding="utf-8",
        )
        [server], errors = load_lsp_servers(path)
        assert errors == []
        assert server["id"] == "pyright"
        assert server["label"] == "Pyright"
        assert server["enable"] is True
        assert server["lazy"] is False
        assert server["extensions"] == [".py"]  # deduplicated, lowercased
        assert server["command"] == ["pyright-langserver", "--stdio"]
        assert server["env"] == {"PATH": "/x"}
        assert server["tools"] == ["hover", "bogus_action"]  # filtered at synthesis
        assert server["maxDiagnostics"] == 40
        assert server["idleShutdownSeconds"] == 300
        assert "unknown_key" not in server

    def test_command_string_is_parsed_like_externals(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry("vue-language-server --stdio --tsdk=/ts/lib")),
            encoding="utf-8",
        )
        [server], errors = load_lsp_servers(path)
        assert errors == []
        assert server["command"] == [
            "vue-language-server",
            "--stdio",
            "--tsdk=/ts/lib",
        ]

    def test_bad_entry_skipped_with_note_but_others_survive(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                good=_entry(["x"]),
                bad_command=_entry([]),
                bad_ext=_entry(["x"], extensions=["py"]),
                bad_root=_entry(["x"], workspaceRoot="relative/path"),
                bad_env=_entry(["x"], env=["not", "a", "map"]),
                bad_init=_entry(["x"], initializationOptions=[1]),
                bad_tools=_entry(["x"], tools=["hover", 4]),
                bad_max=_entry(["x"], maxDiagnostics=0),
                bad_idle=_entry(["x"], idleShutdownSeconds=-1),
                not_object="just a string",
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert [server["id"] for server in servers] == ["good"]
        assert len(errors) == 9
        assert all("'bad_" in error or "not_object" in error for error in errors)
        assert all(error.endswith("— skipped") for error in errors)

    def test_every_declared_server_gets_the_reach_note(self, tmp_path, monkeypatch):
        """win-07 / A-D4: a server is refused where its guardrails match nothing.

        The seam is forced through ``lsphub.constants`` — the hub's single named
        place for it — not by pretending to be another OS.
        """
        monkeypatch.setattr(
            lsp_constants, "guardrails_speak_here", lambda prefixes=None: False
        )
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                pyright=_entry([sys.executable, FAKE_SERVER]),
                vue=_entry([sys.executable, FAKE_SERVER], extensions=[".vue"]),
                off=_entry([sys.executable, FAKE_SERVER], enable=False),
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        # All three are still loaded — a card naming the gap beats a vanished
        # server — but only the two that ask to run are noted.
        assert [server["id"] for server in servers] == ["off", "pyright", "vue"]
        assert len(errors) == 2
        for note in errors:
            assert lsp_constants.LOCAL_REACH_REFUSAL in note
            assert "lsp_servers.json: " in note
        assert not any("'off'" in note for note in errors)

    def test_workspace_root_must_exist(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot=str(tmp_path / "gone"))),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "workspaceRoot does not exist" in errors[0]

    def test_denied_workspace_root_is_skipped(self, tmp_path, monkeypatch):
        deny = tmp_path / "restricted-paths.txt"
        deny.write_text(f"{tmp_path}/secret\n# comment\n", encoding="utf-8")
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)
        secret = tmp_path / "secret"
        secret.mkdir()  # exists — the deny-list, not existence, must reject it
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot=str(secret))),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "deny-list" in errors[0]

    @pytest.mark.case_sensitive_volume(
        "the deny-list names the *other* spelling of a real folder, whose "
        "``realpath`` has to keep the text as written; on a volume where the "
        "two spellings are one folder the ``fold=False`` half is no longer the "
        "pair's other answer"
    )
    def test_the_deny_list_compare_folds_when_the_workspace_volume_folds(
        self, tmp_path, monkeypatch
    ):
        r"""win-06 §4: this site asks the shared ``_is_within`` with the fold.

        The note above has to fire for *either* spelling of a denied folder on a
        volume where they are one folder — a line naming ``Secret`` must refuse a
        ``workspaceRoot`` of ``secret`` there, and must not here, where the two
        names are two directories. This runner can hold both, so the pair is
        assertable by forcing the probe, and the exact half is the Linux answer
        unchanged.
        """
        from src.core.editorfs import fscase

        secret = tmp_path / "secret"
        secret.mkdir()
        deny = tmp_path / "restricted-paths.txt"
        # The other spelling: not a path that exists, so its realpath keeps the
        # text — which is what a deny-list line written on a folded volume looks
        # like from here.
        deny.write_text(f"{tmp_path}/Secret\n", encoding="utf-8")
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot=str(secret))),
            encoding="utf-8",
        )

        monkeypatch.setattr(fscase, "case_folded", lambda _root: False)
        servers, errors = load_lsp_servers(path)
        assert [server["id"] for server in servers] == ["srv"]
        assert errors == []

        monkeypatch.setattr(fscase, "case_folded", lambda _root: True)
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "workspaceRoot is on the deny-list — skipped" in errors[0]

    def test_the_list_is_expanded_at_read_time_so_one_file_serves_both_platforms(
        self, tmp_path, monkeypatch
    ):
        """B-D1: the deny-list itself takes ``${NAME}`` / ``${NAME:-fallback}``.

        The committed file carries both platforms' forms and no machine path, so
        what a *deployment* writes is a variable name — and the effective list is
        what that name states on this runner. Same helper the declaration values
        go through, so the two halves cannot expand differently.
        """
        from src.core.lsphub.discovery import restricted_lsp_roots

        denied = tmp_path / "profile-store"
        denied.mkdir()
        monkeypatch.setenv("LSP_TEST_PROFILE", str(tmp_path))
        monkeypatch.delenv("LSP_TEST_ABSENT", raising=False)
        deny = tmp_path / "restricted-paths.txt"
        deny.write_text(
            "# comment\n"
            "${LSP_TEST_PROFILE}/profile-store\n"
            "${LSP_TEST_ABSENT:-/opt/scratch-store}\n"
            "/etc\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot=str(denied))),
            encoding="utf-8",
        )
        assert restricted_lsp_roots() == [
            f"{tmp_path}/profile-store",
            "/opt/scratch-store",
            "/etc",
        ]
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "workspaceRoot is on the deny-list — skipped" in errors[0]

    def test_a_guardrail_whose_reference_does_not_resolve_is_absent_not_guessed(
        self, tmp_path, monkeypatch, caplog
    ):
        r"""An un-expanded ``${NAME}`` must not count as a prefix that speaks here.

        A declaration keeps such a value verbatim and notes it, so a typo stays
        visible in that entry's note. A *deny-list* line cannot be kept: the text
        keeps a rooted shape, so it answers ``guardrails_speak_here`` True while
        protecting nothing — the reach seam would be lifted on a bound that is not
        in force, and the entries that did resolve would go unenforced. A line with
        a reference that resolves to nothing is therefore dropped and named: a
        leading reference leaves relative text, a mid-path one leaves text that
        still looks rooted, and both are the same absence of a guardrail.
        """
        import logging

        from src.core.lsphub.discovery import restricted_lsp_roots

        monkeypatch.delenv("LSP_TEST_NO_STORE", raising=False)
        deny = tmp_path / "restricted-paths.txt"
        deny.write_text(
            # A leading reference leaves *relative* text, and a mid-path one
            # leaves text that still looks rooted — both must be dropped.
            "${LSP_TEST_NO_STORE}/.ssh\n"
            "/home/${LSP_TEST_NO_STORE}/.aws\n"
            "${LSP_TEST_KNOWN}/.config/gcloud\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)
        monkeypatch.setenv("LSP_TEST_KNOWN", str(tmp_path / "known"))

        with caplog.at_level(logging.WARNING):
            entries = restricted_lsp_roots()
        assert entries == [f"{tmp_path / 'known'}/.config/gcloud"]
        assert not any("${" in entry for entry in entries)
        assert "${LSP_TEST_NO_STORE}" in caplog.text
        assert "not set in the environment" in caplog.text
        assert "not in force" in caplog.text

    def test_a_relative_line_is_absent_from_the_list_too(self, tmp_path, monkeypatch):
        """A prefix that is not rooted protects nothing, so it is not a prefix."""
        from src.core.lsphub.discovery import restricted_lsp_roots

        deny = tmp_path / "restricted-paths.txt"
        deny.write_text(
            "~/.ssh\nrelative/store\n\n   \n# ${HOME}/.ssh\n", encoding="utf-8"
        )
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)
        assert restricted_lsp_roots() == []

    def test_the_committed_list_is_machine_neutral_and_names_both_platforms(self):
        """The file every deployment reads spells no drive letter and no home.

        ``/home/vgr/.ssh`` was the thing B-D1 called *machine*-shaped: correct
        for one box, inert for every other. The forms that reach a Windows tree
        arrive from the environment the OS states them in, exactly as
        ``path_policy`` derives its connector prefixes (G-D8).
        """
        from mcp_connectors import path_policy

        from src.core.lsphub.discovery import restricted_lsp_roots

        text = lsp_constants.LSP_RESTRICTED_PATHS.read_text(encoding="utf-8")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert not any(re.search(r"[A-Za-z]:", line) for line in lines)
        assert not any(line.startswith(("/home/", "/root/")) for line in lines)
        # The POSIX half stays (it is load-bearing on the other platform), and
        # the profile stores are stated on both spellings of a profile home.
        assert "/etc" in lines
        assert "${HOME}/.ssh" in lines and "${USERPROFILE}/.ssh" in lines
        # What this runner's environment states is what the list answers with,
        # and every line that survives is a real prefix: rooted by shape.
        entries = restricted_lsp_roots()
        assert entries
        assert all(
            path_policy.root_form(entry) in ("absolute", "drive", "unc")
            for entry in entries
        )
        assert "/etc" in entries
        if os.environ.get("SystemRoot"):
            assert os.environ["SystemRoot"] in entries
        if os.environ.get("USERPROFILE"):
            assert f"{os.environ['USERPROFILE']}/.ssh" in entries

    def test_the_guardrail_half_speaks_this_runners_working_location(self):
        r"""B-D1's own proof: the hub's committed list answers True on its own.

        Nothing is patched, because there is nothing left to un-patch: the seam
        asks this one question. Both answers are asserted on the runner that runs
        them — the list is True everywhere (on Windows by the environment-derived
        entries B-D1 added, on Linux by the POSIX ones that always were), so the
        reach refusal is ``False`` everywhere the committed file is in place.
        """
        from src.core.lsphub.discovery import lsp_reach_is_refused, restricted_lsp_roots

        entries = restricted_lsp_roots()
        assert entries
        assert lsp_constants.guardrails_speak_here(entries or None) is True
        assert lsp_reach_is_refused() is False

    def test_a_denied_prefix_refuses_the_tree_and_what_is_inside_it(
        self, tmp_path, monkeypatch
    ):
        r"""The prefix boundary, stated as components rather than as text.

        The case this task exists for is a ``workspaceRoot`` the *old* list let
        through — a Windows system tree or a credential store — and the shape of
        that verdict is askable on either runner: the exact prefix, something
        deep inside it, and the neighbour whose name merely starts with the same
        letters (``C:\Windows.old`` against ``C:\Windows`` — a text-prefix test
        called that a child of the OS tree, B-D1's named mistake).
        """
        monkeypatch.setenv("LSP_TEST_DENY_ROOT", str(tmp_path / "system"))
        for name in ("system", "system/System32/config", "system-old"):
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        deny = tmp_path / "restricted-paths.txt"
        deny.write_text("${LSP_TEST_DENY_ROOT}\n", encoding="utf-8")
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                exact=_entry(
                    [sys.executable, FAKE_SERVER],
                    workspaceRoot=str(tmp_path / "system"),
                ),
                inside=_entry(
                    [sys.executable, FAKE_SERVER],
                    workspaceRoot=str(tmp_path / "system" / "System32" / "config"),
                ),
                neighbour=_entry(
                    [sys.executable, FAKE_SERVER],
                    workspaceRoot=str(tmp_path / "system-old"),
                ),
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert [server["id"] for server in servers] == ["neighbour"]
        assert len(errors) == 2
        assert all("workspaceRoot is on the deny-list — skipped" in n for n in errors)

    def test_the_committed_list_refuses_a_declaration_pointed_at_a_system_tree(
        self, tmp_path, monkeypatch
    ):
        r"""The literal acceptance, through the real loader and the committed file.

        Here (Windows) that is ``C:/Windows``, the registry-hive directory inside
        it, and this profile's ``.ssh`` — the three roots the POSIX-only list let
        through, and the reason the reach seam was raised. Nothing is patched
        except the profile home the list derives its own entries from, so the file
        every deployment reads is what answers. A POSIX runner asks the question
        of ``/etc`` and its own ``${HOME}`` store instead: the same two halves of
        the list, the same verdict, no new skip.
        """
        profile = tmp_path / "profile"
        (profile / ".ssh").mkdir(parents=True)
        for name in ("USERPROFILE", "HOME"):
            monkeypatch.setenv(name, str(profile))

        system_root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT")
        candidates = [
            system_root,
            os.path.join(system_root, "System32", "config") if system_root else None,
            "/etc",
            str(profile / ".ssh"),
        ]
        probes = [item for item in candidates if item and os.path.isdir(item)]
        assert probes, "this runner states no system tree and no profile home"

        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                **{
                    f"srv{index}": _entry(
                        [sys.executable, FAKE_SERVER], workspaceRoot=probe
                    )
                    for index, probe in enumerate(probes)
                }
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert len(errors) == len(probes)
        assert all("workspaceRoot is on the deny-list — skipped" in n for n in errors)

    def test_the_working_location_itself_is_not_on_the_committed_list(self):
        """Consumer 3/4: the app's own location must survive the reworked list.

        ``editorfs.policy._check_workspace_root`` refuses a workspace that sits on
        this file, and ``POST /path-location`` asks the same question — so if the
        Windows half of the list ever named the tree the app runs in, the Editor
        would go dark and the settings route would refuse the only location that
        exists. Asserted, not assumed.
        """
        from src.core.editorfs import fscase
        from src.core.editorfs.policy import _check_workspace_root, _is_within
        from src.core.editorfs.policy import workspace_root
        from src.core.lsphub.discovery import restricted_lsp_roots

        assert _check_workspace_root() == workspace_root()
        fold = fscase.case_folded(workspace_root())
        real = os.path.realpath(workspace_root())
        assert not any(
            _is_within(os.path.realpath(entry), real, fold=fold)
            for entry in restricted_lsp_roots()
        )

    def test_a_working_location_on_the_list_is_refused_by_the_tree_gate(
        self, tmp_path, monkeypatch
    ):
        """Consumer 3, other direction: a location the list names stays refused.

        The reworked list must not have stopped binding anything — the gate that
        refuses to *browse* such a location is where that would show.
        """
        from src.core.editorfs.policy import WorkspacePathError, _check_workspace_root
        from src.core.lsphub.discovery import restricted_lsp_roots

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(workspace))
        # The list names a tree, the workspace sits inside it. The variable is
        # the realpath, because the gate compares realpaths on both sides.
        monkeypatch.setenv("LSP_TEST_DENIED_TREE", os.path.realpath(tmp_path))
        deny = tmp_path / "restricted-paths.txt"
        deny.write_text("${LSP_TEST_DENIED_TREE}\n", encoding="utf-8")
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)

        assert restricted_lsp_roots() == [os.path.realpath(tmp_path)]
        with pytest.raises(WorkspacePathError) as caught:
            _check_workspace_root()
        assert caught.value.status_code == 422
        assert "deny-list" in caught.value.detail

    def test_env_references_are_expanded_everywhere(self, tmp_path, monkeypatch):
        """command, env, workspaceRoot and the option objects all expand.

        ``workspaceRoot`` is expanded *before* the existence/deny-list checks,
        so a per-machine path can be written as one ``${VAR}`` and the entry
        still loads on every checkout. A value that expands to something with
        a space stays a single argv element (no shell re-splitting).
        """
        root = tmp_path / "ws dir"
        root.mkdir()
        monkeypatch.setenv("LSP_TEST_BIN", str(tmp_path / "bin"))
        monkeypatch.setenv("LSP_TEST_ROOT", str(root))
        monkeypatch.setenv("LSP_TEST_PATH", "/usr/bin:/bin")
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                srv=_entry(
                    ["${LSP_TEST_BIN}/langserver", "--stdio", "--root=${LSP_TEST_ROOT}"],
                    env={"PATH": "${LSP_TEST_PATH}"},
                    workspaceRoot="${LSP_TEST_ROOT}",
                    initializationOptions={"tsdk": "${LSP_TEST_ROOT}/lib", "n": 3, "on": True},
                    settings={"html.format": {"indentScriptTags": False}},
                )
            ),
            encoding="utf-8",
        )
        [server], errors = load_lsp_servers(path)
        assert errors == []
        assert server["command"] == [
            f"{tmp_path / 'bin'}/langserver",
            "--stdio",
            f"--root={root}",
        ]
        assert server["workspaceRoot"] == str(root)
        assert server["env"] == {"PATH": "/usr/bin:/bin"}
        assert server["initializationOptions"] == {"tsdk": f"{root}/lib", "n": 3, "on": True}

    def test_unresolved_env_reference_reports_and_explains_the_shape_error(
        self, tmp_path, monkeypatch
    ):
        """One note names the unset variable, and it comes before the error it causes."""
        monkeypatch.delenv("LSP_TEST_NOPE", raising=False)
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot="${LSP_TEST_NOPE}/web")),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "LSP_TEST_NOPE" in errors[0] and "not set in the environment" in errors[0]
        assert "workspaceRoot must be an absolute path" in errors[1]

    def test_reserved_id_and_underscore_comment_keys(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                taken=_entry(["x"]),
                _comment="not a server",
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path, reserved_ids={"taken"})
        assert servers == []
        assert errors == ["lsp_servers.json: 'taken': id is reserved — skipped"]

    def test_all_servers_reads_the_local_declaration_file(self):
        # .nojae/lsp-settings.json is gitignored (per-machine, may hold env
        # secrets), so this only runs where one is configured — like the real
        # app.
        if not lsp_constants.lsp_settings_path().is_file():
            pytest.skip("no local .nojae/lsp-settings.json at this working location")
        servers, errors = all_servers()
        assert errors == []
        assert servers
        assert all("env" in server for server in servers)

    def test_the_committed_example_loads_and_reports_only_its_placeholders(self):
        """The template every deployment copies is loaded by the loader, not
        only read by a human (lspfinal-05).

        It cannot be a *clean* load: its ``workspaceRoot`` values are the
        deliberate ``/absolute/path/to/…`` placeholders, which no runner has, so
        those entries are skipped. Asserting a clean load would push a future
        edit to turn a placeholder into a real machine path — worse than the
        note. What is pinned instead: the file parses, the three server ids are
        unchanged, the ``_…_note`` keys are still comments (never reported as
        malformed servers), and no note is an unresolved ``${…}`` or the reach
        refusal — either of those would make the file the app hands out one the
        app refuses.
        """
        example = (
            lsp_constants.LSP_CONNECTORS_DIR / "lsp_servers.example.json"
        )
        declared = json.loads(example.read_text(encoding="utf-8"))
        ids = [key for key in declared if not key.startswith("_")]
        assert ids == ["pyright", "typescript", "vue"]
        note_keys = [key for key in declared if key.startswith("_")]
        assert note_keys  # documented from inside the file, as comment keys

        servers, errors = load_lsp_servers(example)
        assert {server["id"] for server in servers} <= set(ids)
        # A note names one declared server and never a comment key, and the only
        # thing that can skip one of these entries is its placeholder root: an
        # unresolved ``${…}`` or the reach refusal here would make the file the
        # app hands out one the app refuses.
        assert errors
        for note in errors:
            assert sum(f"'{server_id}'" in note for server_id in ids) == 1
            assert not any(f"'{key}'" in note for key in note_keys)
            assert "workspaceRoot" in note
            assert "not set in the environment" not in note
            assert "not started" not in note

    def test_the_examples_values_stay_the_posix_form_on_any_runner(self, monkeypatch):
        """The template cannot carry one ``PATH`` for both systems, and its note
        says so (lspfinal-05) — this is that sentence made executable.

        Asked of the same helper a real declaration goes through, with
        ``VOLTA_BIN`` unset: ``${NAME:-fallback}`` substitutes that one variable
        or its fallback, with no conditional and no platform test, so the
        POSIX fallback and its ':' separators land on a Windows runner too. The
        Windows spelling is therefore a note to copy, not a value — and a future
        edit that made the values "adapt themselves" would be changing the
        expansion contract, which is code, not a data file.
        """
        monkeypatch.delenv("VOLTA_BIN", raising=False)
        example = (
            lsp_constants.LSP_CONNECTORS_DIR / "lsp_servers.example.json"
        )
        entry = json.loads(example.read_text(encoding="utf-8"))["pyright"]
        command, missing = expand_env_list(entry["command"])
        env, more = expand_env_map(entry["env"])
        assert missing == more == []  # resolves, or falls back — never left blank
        assert command == ["/usr/local/bin/pyright-langserver", "--stdio"]
        assert env == {"PATH": "/usr/local/bin:/usr/local/bin:/usr/bin:/bin"}


class TestSetLspEnabled:
    def test_writes_enable_back_and_preserves_everything_else(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            json.dumps(
                {
                    "zeta": _entry(["z"], label="Z"),
                    "alpha": _entry(["a"], label="A"),
                    "_note": "keep me",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert set_lsp_enabled({"alpha": True, "zeta": False}, path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["alpha"]["enable"] is True
        assert data["zeta"]["enable"] is False
        assert list(data) == ["zeta", "alpha", "_note"]  # entry order kept
        assert data["_note"] == "keep me"
        assert data["alpha"]["label"] == "A"

    def test_unknown_id_is_ignored_and_missing_file_is_a_noop(self, tmp_path):
        assert set_lsp_enabled({"ghost": True}, tmp_path / "nope.json") == []
        path = tmp_path / "lsp_servers.json"
        path.write_text(_servers_doc(real=_entry(["x"])), encoding="utf-8")
        assert set_lsp_enabled({"ghost": True}, path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "ghost" not in data

    def test_unparsable_file_is_reported_and_left_untouched(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text("{oops", encoding="utf-8")
        errors = set_lsp_enabled({"a": True}, path)
        assert len(errors) == 1 and "Could not update" in errors[0]
        assert path.read_text(encoding="utf-8") == "{oops"


# ---------------------------------------------------------------------------
# Tool synthesis
# ---------------------------------------------------------------------------


class TestToolSynthesis:
    def test_declared_tools_are_namespaced_and_openai_shaped(self):
        server = {"id": "pyright", "tools": ["diagnostics", "hover"]}
        defs = tools_for_server(server)
        assert [d["function"]["name"] for d in defs] == [
            "lsp_pyright_diagnostics",
            "lsp_pyright_hover",
        ]
        # Matches the exact shape mcphub.tools_to_openai_format() produces,
        # so the routes merge is trivial.
        assert set(defs[0]) == {"type", "function"}
        assert set(defs[0]["function"]) == {"name", "description", "parameters"}

    def test_undeclared_tools_default_to_the_full_catalogue(self):
        server = {"id": "s", "tools": None}
        names = [d["function"]["name"] for d in tools_for_server(server)]
        assert names == [f"lsp_s_{action}" for action in TOOL_SPECS]

    def test_unknown_declared_names_are_dropped(self):
        defs = tools_for_server({"id": "s", "tools": ["hover", "write_file"]})
        assert [d["function"]["name"] for d in defs] == ["lsp_s_hover"]

    def test_capabilities_intersect_the_catalogue(self):
        caps = SimpleNamespace(hover_provider=True, definition_provider=None)
        server = {"id": "s", "tools": ["hover", "definition", "diagnostics"]}
        defs = tools_for_server(server, caps)
        # diagnostics has no capability (push/pull) so it is always offered;
        # the declared order is what survives.
        assert [d["function"]["name"] for d in defs] == [
            "lsp_s_hover",
            "lsp_s_diagnostics",
        ]

    def test_name_round_trip(self):
        assert tool_name_for("vue", "hover") == "lsp_vue_hover"
        assert action_of_tool_name("vue", "lsp_vue_document_symbols") == "document_symbols"
        assert action_of_tool_name("vue", "execute_command") == ""

    def test_openai_format_compatibility(self):
        # A def from the hub and one from the MCP hub must be interchangeable.
        [lsp_def] = tools_for_server({"id": "s", "tools": ["hover"]})
        [mcp_def] = tools_to_openai_format(
            [SimpleNamespace(name="execute_command", description="d", input_schema={})]
        )
        assert set(lsp_def) == set(mcp_def)
        assert set(lsp_def["function"]) == set(mcp_def["function"])


# ---------------------------------------------------------------------------
# LspManager lifecycle against the fake language server
# ---------------------------------------------------------------------------


def _fake_server_entry(tmp_path: Path, server_id: str = "fake", *flags: str,
                       lazy: bool = False, enable: bool = True) -> Dict[str, Any]:
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return {
        "id": server_id,
        "label": server_id,
        "description": "",
        "command": [sys.executable, FAKE_SERVER, *flags],
        "workspaceRoot": str(root),
        "extensions": [".py"],
        "languageId": "python",
        "enable": enable,
        "lazy": lazy,
        "initializationOptions": {},
        "tools": ["diagnostics", "hover", "definition", "document_symbols"],
        "env": {},
        "maxDiagnostics": 10,
        "idleShutdownSeconds": 0,
    }


@pytest.fixture
async def hub(tmp_path):
    """LspManager whose private hub thread is up, torn down after the test.

    Started with no servers declared (``start_all`` boots the thread), then
    each test declares its fake server and starts it on the hub thread —
    the same path the app uses, so ``call_tool_sync`` bridges correctly.
    """
    manager = LspManager([], logs_dir=tmp_path / "logs")
    await manager.start_all()
    try:
        yield manager
    finally:
        await manager.stop_all()


def _declare(hub: LspManager, entry: Dict[str, Any]) -> None:
    hub._servers[entry["id"]] = {**entry, "status": "stopped", "error": None,
                                 "last_used": 0.0}


async def _run(hub: LspManager, coroutine: Any) -> None:
    """Run one hub coroutine on the hub's own thread (like the API does)."""
    # ``start_all`` (the hub fixture) always set the loop before this runs.
    assert hub._loop is not None
    await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(coroutine, hub._loop)
    )


async def _start(hub: LspManager, entry: Dict[str, Any]) -> None:
    _declare(hub, entry)
    await _run(hub, hub._start_one(entry["id"]))


class TestLspLifecycle:
    @pytest.mark.anyio
    async def test_handshake_then_running_with_capability_tools(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        info = hub._servers["fake"]
        assert info["status"] == "running", info["error"]
        assert info["error"] is None
        # Only the capabilities the fake server advertises became tools:
        # references/completion/formatting/signature_help/workspace_symbols
        # are absent.
        tools, mapping = hub.active_tool_defs(["fake"])
        assert [d["function"]["name"] for d in tools] == [
            "lsp_fake_diagnostics",
            "lsp_fake_hover",
            "lsp_fake_definition",
            "lsp_fake_document_symbols",
        ]
        assert set(mapping.values()) == {"fake"}
        assert hub.server_status("fake") == "running"
        assert hub.server_status("nope") is None

    @pytest.mark.anyio
    async def test_stop_is_idempotent_and_restart_respawns(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        first_pid = hub._sessions["fake"].process.pid
        await _run(hub, hub._stop_one("fake"))
        assert hub.server_status("fake") == "stopped"
        await _run(hub, hub._stop_one("fake"))  # idempotent
        assert hub.server_status("fake") == "stopped"
        await _run(hub, hub._restart_one("fake"))
        assert hub.server_status("fake") == "running"
        assert hub._sessions["fake"].process.pid != first_pid

    @pytest.mark.anyio
    async def test_restart_rereads_the_declaration_file(self, hub, tmp_path):
        # The whole point of hub restart: an edited .nojae/lsp-settings.json
        # (here, a
        # different workspaceRoot and a changed tool list) applies without
        # restarting the web app.
        first = _fake_server_entry(tmp_path, "fake")
        await _start(hub, first)
        old_process = hub._sessions["fake"].process
        (Path(first["workspaceRoot"]) / "in_root1.py").write_text(
            "x = 1\n", encoding="utf-8"
        )
        other = tmp_path / "other_root"
        other.mkdir()
        (other / "in_root2.py").write_text("x = 1\n", encoding="utf-8")
        second = {**first, "workspaceRoot": str(other),
                  "tools": ["diagnostics"]}

        await hub.restart([second], ["fake"])

        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        assert hub._sessions["fake"].process.pid != old_process.pid
        for _ in range(100):
            if old_process.returncode is not None:
                break
            await asyncio.sleep(0.05)
        assert old_process.returncode is not None
        # The new workspaceRoot is in force: its file resolves, the old
        # root's file is gone.
        assert hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "in_root2.py"}
        ) == "No diagnostics."
        assert hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "in_root1.py"}
        ).startswith("Error: not a file:")
        # Tool defs come from the new entry too (hover dropped).
        tools, _mapping = hub.active_tool_defs(["fake"])
        assert [d["function"]["name"] for d in tools] == ["lsp_fake_diagnostics"]

    @pytest.mark.anyio
    async def test_restart_can_drop_and_add_servers(self, hub, tmp_path):
        await _start(hub, _fake_server_entry(tmp_path, "gone"))
        assert hub.server_status("gone") == "running"
        incoming = _fake_server_entry(tmp_path, "fresh")

        await hub.restart([incoming], ["fresh"])

        assert hub.server_status("gone") is None  # no longer declared
        assert hub.server_status("fresh") == "running", hub._servers["fresh"]["error"]

    @pytest.mark.anyio
    async def test_restart_respects_the_enabled_gate(self, hub, tmp_path):
        # A dormant file stays dormant: declared but not enabled means no
        # spawn (the two-gate rule the startup lifespan applies).
        entry = _fake_server_entry(tmp_path)
        await hub.restart([entry], [])
        assert hub.server_status("fake") == "stopped"
        assert "fake" not in hub._sessions

    @pytest.mark.anyio
    async def test_initialize_failure_lands_in_status_verbatim(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--fail-init")
        await _start(hub, entry)
        info = hub._servers["fake"]
        assert info["status"] == "error"
        assert "Could not find a valid TypeScript installation" in info["error"]
        assert "fake" not in hub._sessions

    @pytest.mark.anyio
    async def test_spawn_failure_lands_in_status(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake")
        entry["command"] = ["/nonexistent/langserver"]
        await _start(hub, entry)
        assert hub._servers["fake"]["status"] == "error"
        assert "Could not spawn" in hub._servers["fake"]["error"]

    @pytest.mark.anyio
    async def test_lazy_server_spawns_on_first_tool_call(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, lazy=True)
        _declare(hub, entry)
        await _run(hub, hub._start_all_matching({"fake"}))
        assert hub.server_status("fake") == "stopped"
        # Still represented in the catalogue, from the declared tools.
        tools, _mapping = hub.active_tool_defs(["fake"])
        assert tools
        hub.call_tool_sync("fake", "lsp_fake_hover",
                           {"path": "x.py", "line": 0, "character": 0})
        assert hub.server_status("fake") == "running"

    @pytest.mark.anyio
    async def test_start_server_recovers_a_server_stuck_in_starting(self, hub, tmp_path):
        # The regression: a hot toggle publishes ``starting`` synchronously and
        # then dispatches the start onto the hub thread. A guard on *status*
        # used to reject that very start, so the server stayed in ``starting``
        # forever — no process, toolCount 0, and every other path (catalogue,
        # lazy wake-up, the settings toggle) refusing to touch it.
        entry = _fake_server_entry(tmp_path, "fake", lazy=True)
        _declare(hub, entry)
        hub._servers["fake"]["status"] = "starting"
        hub._servers["fake"]["startingSince"] = hub._loop.time()

        hub.start_server("fake")
        for _ in range(200):
            if hub.server_status("fake") in ("running", "error"):
                break
            await asyncio.sleep(0.05)

        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        tools, _mapping = hub.active_tool_defs(["fake"])
        assert tools

    @pytest.mark.anyio
    async def test_start_server_is_idempotent_while_an_attempt_is_in_flight(
        self, hub, tmp_path
    ):
        # A second click must not tear down the handshake the first one began.
        #
        # The delay matters: the hub records the session *before* it awaits the
        # ``initialize`` answer, so with a server that answers instantly the
        # first attempt is over before this test can even see the session — and
        # a second ``start_server`` on a running server is a restart, which
        # would prove the opposite of what is being asserted. ``--slow-init``
        # keeps the attempt genuinely in flight across the second call.
        entry = _fake_server_entry(tmp_path, "fake", "--slow-init", "1.5", lazy=True)
        _declare(hub, entry)
        hub.start_server("fake")
        first_pid: Optional[int] = None
        for _ in range(200):
            session = hub._sessions.get("fake")
            if session is not None:
                first_pid = session.process.pid
                break
            await asyncio.sleep(0.05)
        assert first_pid is not None
        assert "fake" in hub._starting  # still mid-handshake, by construction
        hub.start_server("fake")  # in flight: a no-op
        for _ in range(200):
            if hub.server_status("fake") == "running":
                break
            await asyncio.sleep(0.05)
        assert hub._sessions["fake"].process.pid == first_pid

    @pytest.mark.anyio
    async def test_lazy_server_is_published_while_starting(self, hub, tmp_path):
        # tsserver's cold start is multi-second: a lazy server mid-handshake
        # must not vanish from the next request's tool list.
        entry = _fake_server_entry(tmp_path, "fake", lazy=True)
        _declare(hub, entry)
        hub._servers["fake"]["status"] = "starting"
        tools, mapping = hub.active_tool_defs(["fake"])
        assert [d["function"]["name"] for d in tools] == [
            "lsp_fake_diagnostics",
            "lsp_fake_hover",
            "lsp_fake_definition",
            "lsp_fake_document_symbols",
        ]
        assert set(mapping.values()) == {"fake"}

    @pytest.mark.anyio
    async def test_starting_without_a_handshake_expires_as_error(self, hub, tmp_path):
        # Whatever left a server ``starting`` with no session behind it, the
        # reaper must name it (and free the status for the toggle to retry).
        entry = _fake_server_entry(tmp_path, "fake", lazy=True)
        _declare(hub, entry)
        hub._servers["fake"]["status"] = "starting"
        hub._servers["fake"]["startingSince"] = (
            hub._loop.time() - lsp_constants.STARTING_STUCK_SECONDS - 1.0
        )

        hub._reap()

        info = hub._servers["fake"]
        assert info["status"] == "error"
        assert "without completing its handshake" in info["error"]
        assert "fake" not in hub._sessions

    @pytest.mark.anyio
    async def test_a_reaped_non_lazy_server_revives_on_next_call(self, hub, tmp_path):
        # pyright is not lazy, so the idle reaper still tears it down after
        # idleShutdownSeconds. Before, the catalogue published a non-lazy
        # server only while it was *running*: a reaped pyright vanished from
        # the next request's tools with no path back, mid-conversation.
        entry = _fake_server_entry(tmp_path, "fake")
        await _start(hub, entry)
        assert hub.server_status("fake") == "running"

        await _run(hub, hub._reap_one("fake"))

        assert hub.server_status("fake") == "stopped"
        assert "fake" not in hub._sessions
        tools, mapping = hub.active_tool_defs(["fake"])
        assert [d["function"]["name"] for d in tools] == [
            "lsp_fake_diagnostics",
            "lsp_fake_hover",
            "lsp_fake_definition",
            "lsp_fake_document_symbols",
        ]
        assert set(mapping.values()) == {"fake"}
        # And its next tool call brings the process back rather than failing.
        (Path(entry["workspaceRoot"]) / "x.py").write_text("x = 1\n", encoding="utf-8")
        result = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "x.py", "line": 0, "character": 0}
        )
        assert not result.startswith("Error:")
        assert hub.server_status("fake") == "running"

    @pytest.mark.anyio
    async def test_a_healthy_start_is_never_expired_by_the_reaper(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", lazy=True)
        await _start(hub, entry)
        assert hub.server_status("fake") == "running"
        hub._reap()
        assert hub.server_status("fake") == "running"

    @pytest.mark.anyio
    async def test_crashed_server_surfaces_as_error_status(self, hub, tmp_path):
        # The fake dies instead of answering its first post-init request
        # (hover): the pending call fails fast (pygls cancels it on process
        # exit) and the exit watcher turns the server's status into 'error'
        # with the log tail, instead of a stale 'running'.
        entry = _fake_server_entry(tmp_path, "fake", "--crash-after", "1")
        await _start(hub, entry)
        assert hub.server_status("fake") == "running"
        (Path(entry["workspaceRoot"]) / "x.py").write_text("x = 1\n", encoding="utf-8")
        result = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "x.py", "line": 0, "character": 0}
        )
        assert result.startswith("Error:")
        for _ in range(100):
            if hub.server_status("fake") == "error":
                break
            await asyncio.sleep(0.05)
        assert hub.server_status("fake") == "error"
        assert "exited with code" in hub._servers["fake"]["error"]

    @pytest.mark.anyio
    async def test_close_session_terminates_the_process(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        process = hub._sessions["fake"].process
        await _run(hub, hub._close_session("fake"))
        for _ in range(100):
            if process.returncode is not None:
                break
            await asyncio.sleep(0.05)
        assert process.returncode is not None

    @pytest.mark.posix_behaviour("process groups are POSIX")
    @pytest.mark.anyio
    async def test_server_leads_its_own_process_group(self, hub, tmp_path):
        # The teardown signals a *group*, which only works because the spawn
        # put the server in a group of its own. Without start_new_session the
        # pgid would be the app's, and a teardown would aim at the app itself.
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        pid = hub._sessions["fake"].process.pid
        assert os.getpgid(pid) == pid

    @skip_unless(proc_fs_probe,
                 "uses /proc to tell a zombie from a live process")
    @pytest.mark.anyio
    async def test_close_session_kills_the_whole_process_tree(self, hub, tmp_path,
                                                             monkeypatch):
        # The ghost this prevents: typescript-language-server spawns tsserver,
        # and a SIGTERM sent to the wrapper alone leaves that child reparented
        # to init, holding a project index the next hub knows nothing about.
        # The fake's --child is a grandchild that ignores SIGTERM on purpose,
        # so only reaching the group and escalating to SIGKILL can clear it.
        monkeypatch.setattr(lsp_constants, "STOP_GRACE_SECONDS", 1.0)
        pidfile = tmp_path / "grandchild.pid"
        entry = _fake_server_entry(tmp_path, "fake", "--child", str(pidfile))
        await _start(hub, entry)

        grandchild: Optional[int] = None
        for _ in range(100):
            if pidfile.exists():
                grandchild = int(pidfile.read_text(encoding="utf-8").strip())
                break
            await asyncio.sleep(0.05)
        assert grandchild is not None, "the fake never reported its grandchild"
        assert _process_alive(grandchild), f"grandchild {grandchild} never came up"

        await _run(hub, hub._close_session("fake"))

        for _ in range(100):
            if not _process_alive(grandchild):
                break
            await asyncio.sleep(0.05)
        assert not _process_alive(grandchild), (
            f"grandchild {grandchild} survived the teardown — the hub killed "
            "the wrapper but not its descendants"
        )


class TestJobContainmentAtSpawn:
    """B-D5's hub half: the child is born inside a boundary the OS maintains,
    and the teardown is the one shared escalation on **either** platform.

    The process-group test above is what POSIX has always done. What this class
    adds is that the hub stops branching on the platform, and stops killing only
    the wrapper where there is no group to signal.
    """

    class RecordingJob(proctree_jobs.ProcessJob):
        """The real job class, recording what the hub asked and what it answered.

        A subclass rather than a stand-in, so the mechanism underneath is the one
        the runner really has (or really does not) and only the *conversation* is
        observed.
        """

        def __init__(self, name: Optional[str] = None) -> None:
            super().__init__(name)
            self.assignments: List[Any] = []
            self.close_calls = 0

        def assign(self, pid: Optional[int], *, leader: bool = False):
            answer = super().assign(pid, leader=leader)
            self.assignments.append((pid, leader, answer))
            return answer

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    @pytest.mark.anyio
    async def test_the_job_is_opened_before_the_spawn_and_leads_the_child(
        self, hub, tmp_path, monkeypatch
    ):
        """A job is a spawn-time boundary, so the handle must pre-date the child.

        The hub does not spawn the child — pygls does — so the ordering is
        asserted at both ends: the job exists before ``start_io`` is entered, and
        the child that comes back from it is registered as that job's **leader**,
        the pid the shared teardown is later handed.
        """
        from pygls.lsp.client import LanguageClient

        order: List[str] = []

        class OrderingJob(self.RecordingJob):
            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                order.append("job")

        real_start_io = LanguageClient.start_io
        spawn_kwargs: Dict[str, Any] = {}

        async def spy_start_io(client, *args, **kwargs):
            order.append("spawn")
            spawn_kwargs.update(kwargs)
            return await real_start_io(client, *args, **kwargs)

        monkeypatch.setattr(proctree_jobs, "ProcessJob", OrderingJob)
        monkeypatch.setattr(LanguageClient, "start_io", spy_start_io)

        await _start(hub, _fake_server_entry(tmp_path))
        session = hub._sessions["fake"]
        assert order == ["job", "spawn"], (
            f"the job was opened after the spawn ({order}): a process created "
            "before its parent is contained is never migrated into the job"
        )
        # One call site, one shape: the flag that leads the POSIX group stays on
        # both platforms (it is accepted and ignored on Windows, where the job
        # above is what bounds the tree).
        assert spawn_kwargs.get("start_new_session") is True
        [(assigned_pid, is_leader, _answer)] = session.job.assignments
        assert assigned_pid == session.process.pid and is_leader

    @pytest.mark.anyio
    async def test_where_there_are_no_jobs_the_spawn_site_still_asks_once(
        self, hub, tmp_path, monkeypatch
    ):
        """The Linux path, forced on either runner through the seam's own
        patch-point (roadmap §9 rule 2): the job is inert, ``assign`` answers
        ``None`` — the platform's answer, not a failure — nothing is reported,
        and the group teardown is untouched.
        """
        monkeypatch.setattr(proctree_jobs, "JOBS_SUPPORTED", False)
        probe = self.RecordingJob("inert")
        assert probe.available is False
        assert probe.assign(os.getpid(), leader=True) is None
        probe.close()

        jobs: List[Any] = []

        class ObservingJob(self.RecordingJob):
            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                jobs.append(self)

        monkeypatch.setattr(proctree_jobs, "ProcessJob", ObservingJob)
        await _start(hub, _fake_server_entry(tmp_path))

        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        [job] = jobs
        assert job.handle is None
        assert [answer for _pid, _leader, answer in job.assignments] == [None]
        assert "not contained in a job" not in hub.log_tail("fake")

    @pytest.mark.anyio
    async def test_a_child_that_cannot_be_contained_is_reported(
        self, hub, tmp_path, monkeypatch
    ):
        """A ``False`` from ``assign`` fails neither the spawn nor the silence.

        ``ProcessJob.assign`` keeps the caller's process either way, so the
        server still comes up; but the hub must not then claim a bounded tree,
        which is what the server's own log line is for — the file the settings
        panel shows. Ignoring it is the "spawn site that forgot the call" that
        the teardown's own warning exists to find.
        """
        class RefusingJob(self.RecordingJob):
            def assign(self, pid: Optional[int], *, leader: bool = False):
                self.assignments.append((pid, leader, False))
                return False

        monkeypatch.setattr(proctree_jobs, "ProcessJob", RefusingJob)
        await _start(hub, _fake_server_entry(tmp_path))

        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        assert "not contained in a job" in hub.log_tail("fake")
        assert proctree_jobs.job_for(hub._sessions["fake"].process.pid) is None

    @pytest.mark.anyio
    async def test_the_teardown_is_asked_of_the_shared_entry_point(
        self, hub, tmp_path, monkeypatch
    ):
        """No second escalation in the hub: it asks ``proctree`` and nothing else.

        ``proctree`` chooses the mechanism by capability, so the hub has one call
        site for both platforms — which is how the hubs and the terminal stay one
        implementation. The handle is still released by the hub, because the
        entry point only closes a job it *found*.
        """
        asked: List[Any] = []

        async def fake_kill(pid, reap_leader, grace_seconds):
            asked.append(pid)

        monkeypatch.setattr(proctree, "kill_process_tree", fake_kill)
        jobs: List[Any] = []

        class ObservingJob(self.RecordingJob):
            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                jobs.append(self)

        monkeypatch.setattr(proctree_jobs, "ProcessJob", ObservingJob)
        await _start(hub, _fake_server_entry(tmp_path))
        pid = hub._sessions["fake"].process.pid
        await _run(hub, hub._close_session("fake"))

        assert asked == [pid], "the hub escalated by some route of its own"
        [job] = jobs
        assert job.close_calls == 1, (
            "the child was dropped without releasing its job handle — the "
            "kill-on-close guarantee only fires on the last handle"
        )

    def test_the_teardown_has_no_platform_branch(self):
        """The acceptance line, asserted rather than trusted.

        ``_close_session`` used to read ``os.name == "posix"`` and, on the other
        side of that branch, terminate the wrapper alone — the private copy of
        the escalation that let tsserver outlive the hub. The choice of mechanism
        lives inside ``proctree`` now.
        """
        source = inspect.getsource(LspManager._close_session)
        assert "os.name" not in source
        assert "proctree.kill_process_tree" in source
        assert ".terminate()" not in source and ".kill()" not in source
        # …and the escalation is reached only through that entry point: a hub
        # driving the job primitives itself is how the hubs and the terminal
        # drift (the terminal calls them directly because its driver seam needs
        # that; a hub copying the shape is the failure).
        hub_source = inspect.getsource(LspManager)
        assert "kill_job_tree" not in hub_source
        assert "terminate_and_drain" not in hub_source

    @pytest.mark.anyio
    async def test_replacing_a_child_closes_the_old_job(self, hub, tmp_path,
                                                        monkeypatch):
        """A restart replaces a child, so it must close the *old* job rather
        than drop the reference."""
        jobs: List[Any] = []

        class ObservingJob(self.RecordingJob):
            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                jobs.append(self)

        monkeypatch.setattr(proctree_jobs, "ProcessJob", ObservingJob)
        await _start(hub, _fake_server_entry(tmp_path))
        await _run(hub, hub._restart_one("fake"))
        assert hub.server_status("fake") == "running"
        assert len(jobs) == 2
        # The old child's handle is gone whatever route closed it (the shared
        # teardown closes a job it finds, the hub closes it again on the way
        # out, and ``close`` is idempotent — what must not happen is zero).
        assert jobs[0].close_calls >= 1 and jobs[0].handle is None, (
            "the replaced child's job handle was dropped, not closed"
        )
        assert jobs[1].close_calls == 0
        await _run(hub, hub._stop_one("fake"))
        assert jobs[1].close_calls >= 1 and jobs[1].handle is None

    @pytest.mark.anyio
    async def test_a_hub_reload_closes_a_job_its_teardown_left_behind(
        self, hub, tmp_path, monkeypatch
    ):
        """``restart`` replaces every child, so it releases what it replaces.

        A reload normally tears each session down first; when its bounded wait
        runs out the session is still in the dict as that dict is cleared, and
        that is the one place a handle could be dropped rather than closed. The
        missed teardown is the *premise* here, and it is stated narrowly: the
        graceful ``shutdown``/``exit`` still runs (so the child exits and is
        reaped and no ghost is left for the next test), the process-tree step
        that would have closed the job does not. What is under test is what the
        reload does with the session that is still registered.
        """
        async def graceful_stop_only() -> None:
            for session in list(hub._sessions.values()):
                client = session.client
                if client is None or client.stopped:
                    continue
                try:
                    await asyncio.wait_for(
                        client.shutdown_async(None),
                        timeout=lsp_constants.STOP_GRACE_SECONDS,
                    )
                    client.exit(None)
                    await client.stop()
                except Exception:  # noqa: BLE001 — the premise, not the rule
                    pass

        jobs: List[Any] = []

        class ObservingJob(self.RecordingJob):
            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                jobs.append(self)

        monkeypatch.setattr(proctree_jobs, "ProcessJob", ObservingJob)
        await _start(hub, _fake_server_entry(tmp_path))
        session = hub._sessions["fake"]
        monkeypatch.setattr(hub, "_stop_on_loop", graceful_stop_only)
        await hub.restart([_fake_server_entry(tmp_path)])

        assert jobs[0].close_calls >= 1 and jobs[0].handle is None, (
            "the reload cleared the session and leaked its job handle"
        )
        # The premise left no ghost behind: the child took the graceful
        # shutdown, so the one thing the reload had to do was the handle.
        assert session.process.returncode is not None

    @pytest.mark.anyio
    async def test_a_spawn_that_never_happened_closes_its_job(
        self, hub, tmp_path, monkeypatch
    ):
        """Every path out of the spawn site releases the handle."""
        from pygls.lsp.client import LanguageClient

        jobs: List[Any] = []

        class ObservingJob(self.RecordingJob):
            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                jobs.append(self)

        async def failing_start_io(client, *args, **kwargs):
            raise OSError("no such program")

        monkeypatch.setattr(proctree_jobs, "ProcessJob", ObservingJob)
        monkeypatch.setattr(LanguageClient, "start_io", failing_start_io)
        await _start(hub, _fake_server_entry(tmp_path))

        assert hub.server_status("fake") == "error"
        assert "Could not spawn language server" in hub._servers["fake"]["error"]
        assert jobs and jobs[0].close_calls == 1
        assert not hub._sessions

    @pytest.mark.anyio
    async def test_a_crashed_child_has_its_job_closed(self, hub, tmp_path,
                                                       monkeypatch):
        """The crash path drops a session with no teardown, so it releases the
        handle too — with kill-on-close, that release is what takes whatever the
        dead server left running."""
        jobs: List[Any] = []

        class ObservingJob(self.RecordingJob):
            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                jobs.append(self)

        monkeypatch.setattr(proctree_jobs, "ProcessJob", ObservingJob)
        entry = _fake_server_entry(tmp_path, "fake", "--crash-after", "1")
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "x.py").write_text("x = 1\n", encoding="utf-8")
        hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "x.py", "line": 0, "character": 0}
        )
        for _ in range(100):
            if hub.server_status("fake") == "error":
                break
            await asyncio.sleep(0.05)
        assert hub.server_status("fake") == "error"
        assert jobs and jobs[0].close_calls == 1

    @pytest.mark.windows_behaviour(
        "the boundary under test is the Job Object acting on a real language-"
        "server tree; on POSIX the boundary is the process group and its own "
        "test is above"
    )
    @pytest.mark.anyio
    async def test_the_whole_tree_is_inside_the_job_and_dies_with_it(
        self, hub, tmp_path, monkeypatch
    ):
        """Tree, not wrapper — measured against the OS, not a return code.

        The fake's ``--child`` grandchild takes no notice of a polite shutdown,
        so a teardown that reaches only the leader provably leaves it behind. Its
        *membership* of the job is the claim this task makes — the hub does not
        spawn the child, pygls does, and the assignment lands a turn after the
        creation — and an empty member list is the Windows-side evidence a unit
        test can produce (the Task-Manager check is a manual one).
        """
        monkeypatch.setattr(lsp_constants, "STOP_GRACE_SECONDS", 3.0)
        pidfile = tmp_path / "grandchild.pid"
        entry = _fake_server_entry(tmp_path, "fake", "--child", str(pidfile))
        await _start(hub, entry)

        job = hub._sessions["fake"].job
        assert job is not None and job.available, (
            "the hub opened no real job for its child — the wiring is missing"
        )
        leader = hub._sessions["fake"].process.pid
        assert leader in job.leaders

        grandchild: Optional[int] = None
        for _ in range(100):
            if pidfile.exists():
                grandchild = int(pidfile.read_text(encoding="utf-8").strip())
                break
            await asyncio.sleep(0.05)
        assert grandchild is not None, "the fake never reported its grandchild"
        assert not proctree_jobs.process_is_gone(grandchild), (
            f"grandchild {grandchild} never came up"
        )

        members = job.members()
        assert members is not None and grandchild in members, (
            f"the descendant the language server created is not itself a job "
            f"member (members={members}): the boundary is the wrapper, not the "
            "tree, so terminating it cannot reach one either"
        )

        # What the job answers *while* the teardown runs, so the last answer is
        # evidence of a drained tree rather than of a handle already closed.
        answers: List[Any] = []
        real_member_pids = proctree_jobs.job_member_pids

        def spy(handle):
            answer = real_member_pids(handle)
            answers.append(answer)
            return answer

        monkeypatch.setattr(proctree_jobs, "job_member_pids", spy)
        await _run(hub, hub._close_session("fake"))

        assert () in answers, (
            f"the job never answered 'no members' during the teardown "
            f"(answers={answers}) — part of the tree outlived it"
        )
        assert job.handle is None, "the job handle outlived its child"
        assert proctree_jobs.job_for(leader) is None
        assert proctree_jobs.process_is_gone(grandchild), (
            f"grandchild {grandchild} survived the job teardown"
        )

class TestDeclaredCommandResolution:
    """B-D6's hub half: the declared command is resolved before it is spawned,
    and what a failed lookup leaves behind is a sentence naming the search.

    The failure this exists for is the obvious declaration — ``pyright-
    langserver``, the extensionless program a POSIX user would write — which on
    Windows is a ``#!/bin/bash`` shim and not an executable image at all. Asked
    of the OS directly it answers ``[WinError 193]``, a code with nothing to act
    on; asked of ``shutil.which`` it resolves to the ``.cmd`` beside it. Neither
    half of that is a platform branch: the search is one call with one answer on
    either side, and what differs is what the answer looks like.
    """

    @staticmethod
    def _shim(bin_dir: Path, server_id: str = "fake-server") -> Path:
        """A folder holding both spellings of one command, Volta's shape.

        The extensionless file is the script a POSIX shell runs (and the one a
        Windows user's ``PATH`` carries); the ``.cmd`` beside it is what Windows
        can actually start. Both launch the suite's fake language server, so a
        resolution that picked the wrong one is visible as a session that never
        handshakes. Nothing here depends on a toolchain being installed.
        """
        bin_dir.mkdir(parents=True, exist_ok=True)
        script = bin_dir / server_id
        script.write_text(
            "#!/bin/sh\n"
            f"exec {json.dumps(sys.executable)} {json.dumps(FAKE_SERVER)}\n",
            encoding="utf-8",
        )
        os.chmod(script, 0o755)  # the POSIX half; a no-op on NTFS
        (bin_dir / f"{server_id}.cmd").write_text(
            f"@echo off\r\n{json.dumps(sys.executable)} "
            f"{json.dumps(FAKE_SERVER)}\r\n",
            encoding="utf-8",
        )
        return script

    def test_an_absolute_command_that_is_the_program_resolves_to_itself(
        self, tmp_path
    ):
        """The unchanged answer, asked where both platforms give it.

        ``sys.executable`` is an absolute, existing, executable program here and
        on Linux, and the search hands it back verbatim — the hub never rewrites
        a declaration that already names a program. (The POSIX shape of the
        extensionless case below, where the declared file *is* the program, is
        the lifecycle every other test in this file already runs.)
        """
        hub = LspManager([], logs_dir=tmp_path / "logs")
        assert hub._resolve_command(sys.executable, dict(os.environ)) == sys.executable

    def test_a_declared_folder_is_the_search_a_named_command_gets(self, tmp_path):
        """A command that spells a path is looked up *there* and nowhere else.

        ``shutil.which`` searches the spelled directory for a bare name's
        ``PATHEXT`` forms, so the folders the refusal reports are that directory,
        not the ``PATH`` the operator would otherwise go and edit.
        """
        hub = LspManager([], logs_dir=tmp_path / "logs")
        script = self._shim(tmp_path / "bin")
        # An env PATH this folder is not on, so a report of it is a bug caught.
        env = {"PATH": str(script.parent), "PATHEXT": os.environ.get("PATHEXT", "")}
        resolved = hub._resolve_command(str(script), env)
        # On Windows that is the ``.cmd`` beside the script; on POSIX the script
        # itself, which is what its executable bit makes it.
        assert resolved is not None
        assert Path(resolved).parent == script.parent
        assert Path(resolved).name.lower() in ("fake-server", "fake-server.cmd")
        assert hub._searched_paths(str(script), env) == str(script.parent)
        # A bare name is the other half: then the searched folders are the PATH.
        assert hub._searched_paths("fake-server", env) == str(script.parent)

    @pytest.mark.anyio
    async def test_a_command_that_resolves_to_nothing_is_refused_not_spawned(
        self, hub, tmp_path, monkeypatch
    ):
        """No child, and a sentence rather than the code a spawn would give.

        ``_start_session`` is the only place a language-server process is
        spawned, so patching it out answers "did the hub try" directly — the
        same pattern the reach-refusal tests use, for the same reason: a hub
        that reports a status while spawning anyway has reported nothing.
        """
        async def no_spawn(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("spawn attempted")

        monkeypatch.setattr(hub, "_start_session", no_spawn)
        entry = _fake_server_entry(tmp_path, "fake")
        bin_dir = tmp_path / "empty-bin"
        bin_dir.mkdir()
        entry["command"] = ["no-such-langserver"]
        # The declaration's own PATH, so the refusal has one folder to name and
        # the assertion below cannot be satisfied by the app's environment.
        entry["env"] = {"PATH": str(bin_dir)}
        _declare(hub, entry)
        await _run(hub, hub._start_one("fake"))

        info = hub._servers["fake"]
        assert info["status"] == "error"
        assert lsp_constants.SPAWN_FAILURE_LEAD in info["error"]
        assert "no-such-langserver" in info["error"]
        assert str(bin_dir) in info["error"]
        assert "fake" not in hub._sessions
        assert hub._defs.get("fake") is None
        # The card is not the only report: the settings panel shows the log too.
        assert "refused" in hub.log_tail("fake")

    @pytest.mark.anyio
    async def test_the_search_is_the_childs_own_path_and_reports_it(
        self, hub, tmp_path
    ):
        """The trap: resolve with the app's PATH, spawn with the child's, and
        the second search fails on a declaration the first one accepted.

        ``_environment`` replaces ``PATH`` wholesale with the entry's
        ``env.PATH``, so the lookup and the spawn are one environment or they are
        two verdicts. Both halves are asked here, from folders the app's own PATH
        does not carry — otherwise the test passes because the app's PATH happens
        to answer, for the wrong reason.
        """
        carry = tmp_path / "bin-carries-it"
        absent = tmp_path / "bin-lacks-it"
        self._shim(carry)
        absent.mkdir()

        entry = _fake_server_entry(tmp_path, "fake")
        # A bare name: nothing about it spells a folder, so only a search of the
        # child's PATH can find it.
        entry["command"] = ["fake-server"]
        entry["env"] = {"PATH": str(carry)}
        await _start(hub, entry)
        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]

        other = _fake_server_entry(tmp_path, "fake2")
        other["command"] = ["fake-server"]
        other["env"] = {"PATH": str(absent)}
        await _start(hub, other)
        info = hub._servers["fake2"]
        assert info["status"] == "error"
        assert str(absent) in info["error"], (
            f"the refusal named something other than the path that was actually "
            f"searched: {info['error']}"
        )
        # And it did not quietly fall back to the app's own PATH, which carries
        # the interpreter's folder — the one folder this declaration excludes.
        assert str(Path(sys.executable).parent) not in info["error"]

    def test_an_os_spawn_failure_is_a_sentence_naming_the_search(self, tmp_path):
        """Both answers the OS gives at the spawn, reported by name.

        The pair is built rather than caught: ``OSError(193, …)`` is exactly what
        ``CreateProcess`` raises for a file that is a script instead of an
        executable image, and a real such spawn is exercised by the Windows-only
        test below. A translated sentence must not carry the bare code; an OS
        answer that *is* the diagnosis must keep its own words beside the named
        command and folders.
        """
        hub = LspManager([], logs_dir=tmp_path / "logs")
        declared = r"C:\Users\vgr\AppData\Local\Volta\bin\pyright-langserver"
        searched = r"C:\Users\vgr\AppData\Local\Volta\bin"

        not_an_image = OSError(193, "%1 is not a valid Win32 application")
        not_an_image.winerror = 193
        sentence = hub._spawn_failure_sentence(
            not_an_image, declared, declared, searched
        )
        assert sentence.startswith(lsp_constants.SPAWN_FAILURE_LEAD)
        # Both are spelled the way an operator pastes them, not ``repr``-ed.
        assert declared in sentence and searched in sentence
        assert "not a valid Win32 application" not in sentence, (
            "the OS code reached the operator untranslated"
        )
        assert "PATHEXT" in sentence, (
            "the sentence names no fix: it must point at the shim it could run"
        )

        not_a_file = FileNotFoundError(2, "The system cannot find the file specified")
        not_a_file.winerror = 2
        sentence = hub._spawn_failure_sentence(
            not_a_file, declared, declared, searched
        )
        assert sentence.startswith(lsp_constants.SPAWN_FAILURE_LEAD)
        assert declared in sentence and searched in sentence
        assert repr(declared) not in sentence
        # Not a code this hub can improve on: the OS's words are kept.
        assert "cannot find the file specified" in sentence

    @pytest.mark.anyio
    async def test_only_the_program_is_resolved_and_no_spawn_uses_a_shell(
        self, hub, tmp_path, monkeypatch
    ):
        """The rest of the declaration's argv reaches the child as declared.

        An argument is not a program name, so it is never handed to the search.
        And nothing about a command that will not resolve is solved by running it
        through a shell: the resolved program is what is exec'd, and ``shell=True``
        would turn a declared command into a command line — the rule every
        spawned process in this app holds to.
        """
        from pygls.lsp.client import LanguageClient

        execed: List[Any] = []
        real_start_io = LanguageClient.start_io

        async def spy_start_io(client, cmd, *args, **kwargs):
            execed.append((cmd, args))
            return await real_start_io(client, cmd, *args, **kwargs)

        monkeypatch.setattr(LanguageClient, "start_io", spy_start_io)
        entry = _fake_server_entry(tmp_path, "fake")
        await _start(hub, entry)

        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        [(cmd, args)] = execed
        # A command that already names a program is exec'd as it was declared.
        assert cmd == sys.executable
        assert args == (FAKE_SERVER,), "an argument was treated as a program name"

        source = inspect.getsource(LspManager._start_session)
        assert "shell=" not in source
        # And the resolution is on the one gate every start attempt passes, not
        # beside it: a second route into the session skips it.
        assert "_resolve_spawn" in inspect.getsource(LspManager._start_one)

    @pytest.mark.windows_behaviour(
        "WinError 193 is what CreateProcess answers for a script that is not an "
        "executable image; on POSIX the same file is an executable program and "
        "there is no such refusal to translate"
    )
    @pytest.mark.anyio
    async def test_a_script_that_is_not_an_image_is_reported_not_a_winerror(
        self, hub, tmp_path
    ):
        """The verified failure, end to end: a ``.cmd``-less shim.

        With no ``PATHEXT`` form beside it, the search can do nothing with the
        file — it exists, so refusing it here would replace the OS's fact with a
        guess. The spawn is where the fact lands, and it must land as a sentence.
        """
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        script = bin_dir / "fake-server"
        script.write_text(
            "#!/bin/bash\nvolta run \"$(basename $0)\" \"$@\"\n", encoding="utf-8"
        )
        entry = _fake_server_entry(tmp_path, "fake")
        entry["command"] = [str(script)]
        await _start(hub, entry)

        info = hub._servers["fake"]
        assert info["status"] == "error"
        assert str(script) in info["error"]
        assert str(bin_dir) in info["error"]
        # The operator's sentence, not the code: what CreateProcess said is
        # gone, and the fix it cannot see is named in its place.
        assert "Win32 application" not in info["error"]
        assert "PATHEXT" in info["error"]
        assert "fake" not in hub._sessions

    @pytest.mark.windows_behaviour(
        "the fact under test is ``PATHEXT`` — a Windows-only search rule; the "
        "POSIX twin, an absolute existing command answering itself, is the "
        "unmarked test above"
    )
    @pytest.mark.anyio
    async def test_a_windows_path_with_no_extension_resolves_to_the_cmd(
        self, hub, tmp_path, monkeypatch
    ):
        """The declaration that used to fail, and what is exec'd instead.

        Asserted at the spawn, not only at the predicate: the hub must exec the
        ``.CMD`` (``which`` answers whatever case ``PATHEXT`` carries) and leave
        the declaration itself unrewritten, so a reload re-searches rather than
        inheriting a path this run found.
        """
        from pygls.lsp.client import LanguageClient

        script = self._shim(tmp_path / "bin")
        execed: List[str] = []
        real_start_io = LanguageClient.start_io

        async def spy_start_io(client, cmd, *args, **kwargs):
            execed.append(cmd)
            return await real_start_io(client, cmd, *args, **kwargs)

        monkeypatch.setattr(LanguageClient, "start_io", spy_start_io)
        entry = _fake_server_entry(tmp_path, "fake")
        entry["command"] = [str(script)]
        await _start(hub, entry)

        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        assert Path(execed[0]).name.lower() == "fake-server.cmd"
        assert Path(execed[0]).parent == script.parent
        assert hub._servers["fake"]["command"] == [str(script)]
        # The rest of the declaration's argv stays as declared: an argument is
        # not a program name, so it is never resolved.
        assert len(execed) == 1


class TestLocalReachRefusal:
    """win-07 / A-D4 on the hub side: refused, named, and never spawned."""

    def test_the_guardrails_speak_here_so_nothing_is_refused(self):
        """The answer on **both** platforms: the lists speak, nothing is refused.

        lspfinal-04 took the seam's platform half away, so this is no longer the
        Linux-only answer it was: on this runner the hub's own committed list
        speaks this working location and the refusal is ``False``. The declared
        servers are therefore loaded, started and *spawned* here — the first
        point in target B where the app does not refuse its own hub.
        """
        assert lsp_constants.guardrails_speak_here(restricted_lsp_roots()) is True
        assert lsp_reach_is_refused() is False

    def test_enabled_servers_get_no_reach_note_from_the_committed_list(
        self, tmp_path
    ):
        """The acceptance sentence, asked with the committed guardrails in force.

        Three enabled entries through the loader, with nothing patched but the
        declaration file: the hub's real ``restricted-paths.txt`` is what answers,
        so the reach note this hub used to give *every* enabled entry is gone on
        either runner. (What the workspace's own file yields is the other test's
        business — that file is per-machine and may be absent here.)
        """
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                pyright=_entry([sys.executable, FAKE_SERVER]),
                typescript=_entry(
                    [sys.executable, FAKE_SERVER], extensions=[".ts"]
                ),
                vue=_entry([sys.executable, FAKE_SERVER], extensions=[".vue"]),
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert [server["id"] for server in servers] == ["pyright", "typescript", "vue"]
        assert not any(lsp_constants.LOCAL_REACH_REFUSAL in note for note in errors)

    def test_both_hubs_refuse_in_the_same_words(self):
        """The symmetry test: one sentence for one missing capability."""
        from src.core.mcphub import constants as mcp_constants

        assert lsp_constants.LOCAL_REACH_REFUSAL == mcp_constants.LOCAL_REACH_REFUSAL
        assert "bounded" in lsp_constants.LOCAL_REACH_REFUSAL
        # lspfinal-06: the sentence names the *cause* an operator can fix (a
        # guardrail list with no entry in this filesystem's path form), not a
        # roadmap target that has since closed.
        assert "guardrail" in lsp_constants.LOCAL_REACH_REFUSAL
        assert "target B" not in lsp_constants.LOCAL_REACH_REFUSAL

    def test_a_posix_shaped_list_speaks_a_posix_shaped_location(self):
        """The Linux half of §6's pair, asked as text so it runs everywhere.

        ``/etc``-shaped guardrails are load-bearing on a ``/``-rooted tree and
        inert on a drive-rooted one. The predicate answers that from the *shape*
        of the entries against the shape of the path (roadmap §9 rule 2), so the
        statement is pure text: correct on either runner, no skip on either.
        """
        from mcp_connectors import path_policy

        assert path_policy.lists_are_meaningful(
            "/home/dev/app", prefixes=("/etc", "/root")
        ) is True
        assert path_policy.lists_are_meaningful(
            "C:/src/app", prefixes=("/etc", "/root")
        ) is False

    def test_a_windows_shaped_list_speaks_a_drive_shaped_location(self):
        """The Windows half of the same pair — the other platform's inertness.

        Its twin above, and the reason ``restricted-paths.txt`` carries both
        forms rather than one (B-D1): a ``C:``-shaped entry protects nothing on a
        ``/``-rooted tree, exactly as ``/etc`` protects nothing here.
        """
        from mcp_connectors import path_policy

        assert path_policy.lists_are_meaningful(
            "C:/src/app", prefixes=("C:/Users/dev/.ssh",)
        ) is True
        assert path_policy.lists_are_meaningful(
            "/home/dev/app", prefixes=("C:/Users/dev/.ssh",)
        ) is False

    def test_its_guardrail_list_is_the_one_it_is_asked_about(self, tmp_path, monkeypatch):
        """The LSP hub asks about *its* deny-list, not the connector prefixes.

        A list of the *other* platform's shape matches nothing this working
        location is spelled with, so this hub refuses — while the MCP hub, whose
        connector lists the environment states in this platform's own form, does
        not. Which shape is "the other one" is the same shape question the
        predicate asks, never ``sys.platform``: this is the live version of the
        two text pairs above, and it runs on either runner.
        """
        from mcp_connectors import path_policy

        from src.config import path_location

        own_form = path_policy.root_form(str(path_location()))
        other_form_list = (
            "/etc/credential-store" if own_form == "drive" else "C:/Users/other/.ssh"
        )
        deny = tmp_path / "restricted-paths.txt"
        deny.write_text(f"{other_form_list}\n", encoding="utf-8")
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)

        assert restricted_lsp_roots() == [other_form_list]
        assert lsp_constants.guardrails_speak_here(restricted_lsp_roots()) is False
        assert lsp_reach_is_refused() is True

        from src.core import mcphub

        assert mcphub.constants.guardrails_speak_here() is True
        assert mcphub.discovery.mcp_reach_is_refused() is False

    @pytest.mark.anyio
    async def test_a_refused_server_is_error_and_is_never_spawned(
        self, hub, tmp_path, monkeypatch
    ):
        """The refusal happens before pygls is asked to start a child.

        ``_start_session`` is the only place a language-server process is
        spawned, so patching it out is the direct answer to "did the hub start
        a server whose path rules match nothing" — and the reason a *spawn*
        would be the bug: a running server with an inert deny-list is the
        silently-permissive shape A exists to remove.
        """
        monkeypatch.setattr(
            lsp_constants, "guardrails_speak_here", lambda prefixes=None: False
        )

        async def no_spawn(server_id: str, info: Dict[str, Any]) -> None:
            raise AssertionError(f"spawn attempted for {server_id}")

        monkeypatch.setattr(hub, "_start_session", no_spawn)
        _declare(hub, _fake_server_entry(tmp_path))
        await _run(hub, hub._start_one("fake"))

        info = hub._servers["fake"]
        assert info["status"] == "error"
        assert info["error"] == lsp_constants.LOCAL_REACH_REFUSAL
        assert "fake" not in hub._sessions
        assert hub._defs.get("fake") is None

    @pytest.mark.anyio
    async def test_a_declared_refused_server_starts_as_error_without_being_started(
        self, monkeypatch, tmp_path
    ):
        """A refusal is applied when the declaration is read, not at start time.

        A ``stopped`` entry with ``enable: true`` is what a *lazy* server looks
        like before its first call, and the catalogue publishes that state from
        the declared tools — so a refused server has to be ``error`` the moment
        ``.nojae/lsp-settings.json`` is read, or its tools reach chat and its
        first call spawns it.
        """
        monkeypatch.setattr(
            lsp_constants, "guardrails_speak_here", lambda prefixes=None: False
        )
        manager = LspManager(
            [
                _fake_server_entry(tmp_path, "eager"),
                _fake_server_entry(tmp_path, "dormant", enable=False),
            ],
            logs_dir=tmp_path / "logs",
        )
        status = {entry["id"]: entry for entry in manager.status()}
        assert status["eager"]["status"] == "error"
        assert status["eager"]["error"] == lsp_constants.LOCAL_REACH_REFUSAL
        # A disabled entry is the user's own switch, not a missing capability.
        assert status["dormant"]["status"] == "stopped"
        assert status["dormant"]["error"] is None

        # And neither is offered to chat, eager or lazy.
        tools, mapping = manager.active_tool_defs(["eager", "dormant"])
        assert tools == []
        assert mapping == {}

    @pytest.mark.anyio
    async def test_a_refused_lazy_server_is_out_of_the_catalogue_and_the_call(
        self, hub, tmp_path, monkeypatch
    ):
        """Both halves of "never invisible" exclusions, verified not assumed.

        A lazy server with no session is normally published from its declared
        tools and revived by its first call. Refusing it at index time is what
        takes it out of both: the catalogue skips ``error``, and the call
        reaches the refusal instead of a spawn.
        """
        monkeypatch.setattr(
            lsp_constants, "guardrails_speak_here", lambda prefixes=None: False
        )
        hub._servers = hub._index([_fake_server_entry(tmp_path, lazy=True)])

        tools, mapping = hub.active_tool_defs(["fake"])
        assert tools == []
        assert mapping == {}

        result = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "x.py", "line": 0, "character": 0}
        )
        assert result.startswith("Error:")
        assert lsp_constants.LOCAL_REACH_REFUSAL in result
        assert "fake" not in hub._sessions
        assert hub.server_status("fake") == "error"

    @pytest.mark.anyio
    async def test_the_hot_toggle_cannot_spawn_a_refused_server(
        self, hub, tmp_path, monkeypatch
    ):
        """``start_server`` from the settings panel answers the refusal itself."""
        monkeypatch.setattr(
            lsp_constants, "guardrails_speak_here", lambda prefixes=None: False
        )
        _declare(hub, _fake_server_entry(tmp_path))
        hub.start_server("fake")
        assert hub.server_status("fake") == "error"
        assert hub._servers["fake"]["error"] == lsp_constants.LOCAL_REACH_REFUSAL
        assert "fake" not in hub._starting

    def test_a_booting_app_does_not_spawn_a_refused_server(self, tmp_path, monkeypatch):
        """With the seam forced, ``BENCHMARKIT_DISABLE_LSP`` is not what keeps it quiet.

        The lifespan loads ``.nojae/lsp-settings.json`` only when that switch is
        unset, so
        clearing it puts the test on the real boot path, and the declared command
        *is* a program that leaves a marker file behind — the assertion is that
        the marker never appears. A refusal that reported the right status while
        spawning the child anyway — the silently-permissive shape A-D4 exists to
        remove — fails here.
        """
        from src import config as config_module
        from src.core import mcphub

        marker = tmp_path / "spawned.flag"
        monkeypatch.delenv("BENCHMARKIT_DISABLE_LSP", raising=False)
        for key in config_module.SETTING_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        monkeypatch.setattr(
            lsp_constants, "guardrails_speak_here", lambda prefixes=None: False
        )
        servers_dir = tmp_path / "lsp_connectors"
        servers_dir.mkdir()
        (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.setattr(lsp_constants, "LSP_CONNECTORS_DIR", servers_dir)
        monkeypatch.setattr(
            lsp_constants, "LSP_SETTINGS_PATH", servers_dir / "lsp-settings.json"
        )
        monkeypatch.setattr(
            lsp_constants, "LSP_RESTRICTED_PATHS", servers_dir / "restricted-paths.txt"
        )
        (servers_dir / "lsp-settings.json").write_text(
            _servers_doc(
                pyright=_entry(
                    [
                        sys.executable,
                        "-c",
                        f"open({str(marker)!r}, 'w').close()",
                    ],
                    workspaceRoot=str(workspace),
                )
            ),
            encoding="utf-8",
        )
        # Keep the MCP hub off the developer's own mcp-settings.json, since this app
        # boots without the LSP switch and the MCP one is left as conftest set it.
        monkeypatch.setattr(
            mcphub.constants, "MCP_SETTINGS_PATH", servers_dir / "mcp-settings.json"
        )
        monkeypatch.setattr(
            routes,
            "load_endpoints_config",
            lambda: (
                SimpleNamespace(
                    serverUrl="http://x/v1", endpoint="m", model_list=lambda: ["m"],
                    to_kwargs=lambda: {}, apiKey="", sysprompt=None,
                ),
                None,
            ),
        )
        app = create_app()
        with TestClient(app) as test_client:
            body = test_client.get("/api/lsp").json()
            assert lsp_constants.LOCAL_REACH_REFUSAL in body["serversError"]
            entry = {item["id"]: item for item in app.state.lsp.status()}["pyright"]
            assert entry["status"] == "error"
            assert entry["error"] == lsp_constants.LOCAL_REACH_REFUSAL
        assert not marker.exists(), "the refused language server was spawned anyway"


def _process_alive(pid: int) -> bool:
    """True while ``pid`` names a live (not merely unreaped) process.

    A descendant the hub did not spawn itself is reparented to init, so it can
    linger as a zombie: signal 0 still answers 'there', while /proc reports the
    truth.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    # 'pid (comm) state …' — comm may contain spaces, hence the rsplit.
    state = stat.rsplit(")", 1)[1].split()[0]
    return state != "Z"


class TestLspToolCalls:
    @pytest.mark.anyio
    async def test_diagnostics_open_then_receive_the_push(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "a.py").write_text(
            "clean line\nx = 1  # BUG here\n", encoding="utf-8"
        )
        result = hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "a.py"}
        )
        assert "a.py:2:10 error: found a BUG here (fakeserverfake001)" in result

    @pytest.mark.anyio
    async def test_clean_file_reports_no_diagnostics(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "b.py").write_text("fine\n", encoding="utf-8")
        result = hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "b.py"}
        )
        assert result == "No diagnostics."

    @pytest.mark.anyio
    async def test_didChange_republish_analyses_current_content(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        document = Path(entry["workspaceRoot"]) / "c.py"
        document.write_text("fine\n", encoding="utf-8")
        assert hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "c.py"}
        ) == "No diagnostics."
        document.write_text("fine\nBUG crept in\n", encoding="utf-8")
        result = hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "c.py"}
        )
        assert "c.py:2:1 error: found a BUG here" in result

    @pytest.mark.anyio
    async def test_hover_and_definition_and_symbols(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "d.py").write_text("x = 1\n", encoding="utf-8")
        hover = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "d.py", "line": 3, "character": 0}
        )
        assert hover == "hover at 3"
        definition = hub.call_tool_sync(
            "fake", "lsp_fake_definition", {"path": "d.py", "line": 0, "character": 0}
        )
        assert definition == "d.py:3:1"
        symbols = hub.call_tool_sync(
            "fake", "lsp_fake_document_symbols", {"path": "d.py"}
        )
        assert symbols == "class Thing (line 5)"

    @pytest.mark.anyio
    async def test_path_policy(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        root = Path(entry["workspaceRoot"])
        (root / "notes.txt").write_text("hi", encoding="utf-8")
        outside = root.parent / "outside.py"
        outside.write_text("x = 1  # BUG\n", encoding="utf-8")

        escaped = hub.call_tool_sync("fake", "lsp_fake_diagnostics",
                                     {"path": str(outside)})
        assert escaped.startswith("Error: path must live inside the workspace root")
        wrong_ext = hub.call_tool_sync("fake", "lsp_fake_diagnostics",
                                       {"path": "notes.txt"})
        assert wrong_ext.startswith("Error: .txt is not handled by fake")
        missing = hub.call_tool_sync("fake", "lsp_fake_diagnostics",
                                     {"path": "nope.py"})
        assert missing.startswith("Error: not a file:")
        empty = hub.call_tool_sync("fake", "lsp_fake_diagnostics", {"path": ""})
        assert empty.startswith("Error: path is required")

    @pytest.mark.anyio
    async def test_unknown_or_unsupported_call_shapes(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        assert hub.call_tool_sync("ghost", "lsp_ghost_hover", {}) == (
            "Error: language server 'ghost' is not declared"
        )
        (Path(entry["workspaceRoot"]) / "e.py").write_text("x\n", encoding="utf-8")
        bad_position = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "e.py", "line": -1, "character": 0}
        )
        assert "non-negative integers" in bad_position


# ---------------------------------------------------------------------------
# The tsserver relay (Volar's private 'tsserver/request' protocol)
# ---------------------------------------------------------------------------


class TestTsserverRelayShapes:
    """Pure-shape tests: the protocol is doubly nested and easy to get wrong.

    Getting either direction wrong is silent — Volar just never resolves its
    ``getLanguageService()`` promise — so the shapes are pinned here as well
    as they can be against the fake server below.
    """

    def test_request_is_unwrapped_from_its_positional_argument_array(self):
        parsed = parse_tsserver_request(
            [[1, "_vue:projectInfo", {"file": "/w/src/App.vue"}]]
        )
        assert parsed == (1, "_vue:projectInfo", {"file": "/w/src/App.vue"})

    def test_request_flat_and_object_forms_are_also_accepted(self):
        assert parse_tsserver_request([2, "_vue:getComponentNames", None]) == (
            2, "_vue:getComponentNames", None,
        )
        assert parse_tsserver_request(
            {"id": 3, "command": "_vue:projectInfo", "args": {"file": "x"}}
        ) == (3, "_vue:projectInfo", {"file": "x"})

    def test_unrecognisable_request_is_reported_not_raised(self):
        # A payload the hub cannot make sense of must not kill the session:
        # it is logged and skipped.
        assert parse_tsserver_request(["not-an-int", 7]) is None
        assert parse_tsserver_request(None) is None
        assert parse_tsserver_request([]) is None

    def test_response_pairs_the_id_inside_the_params_array(self):
        # Volar's handler is onNotification('tsserver/response', ([id, res]) …)
        # so the pair must be nested one level inside params.
        assert tsserver_response_params(4, {"a": 1}) == [[4, {"a": 1}]]
        assert tsserver_response_params(5, None) == [[5, None]]

    def test_project_config_prefers_tsconfig_over_jsconfig(self, tmp_path):
        (tmp_path / "jsconfig.json").write_text("{}", encoding="utf-8")
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
        assert find_project_config(tmp_path) == str(tmp_path / "tsconfig.json")

    def test_project_config_missing_is_none(self, tmp_path):
        assert find_project_config(tmp_path) is None

    def test_project_info_answer_names_the_config_as_external(self):
        info = tsserver_project_info("/w/a.vue", "/w/tsconfig.json")
        # The config path was supplied, so an answer object is expected.
        assert info is not None
        assert info["configFileName"] == "/w/tsconfig.json"
        assert info["projectKind"] == "external"
        assert info["languageServiceEnabled"] is True

    def test_project_info_without_a_config_is_an_explicit_null(self):
        assert tsserver_project_info("/w/a.vue", None) is None

    @pytest.mark.anyio
    async def test_config_walk_stops_at_the_workspace_root(self, hub, tmp_path):
        # A config above the root must never leak into the answer: the root
        # is the boundary the hub promises not to read past.
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]).parent / "tsconfig.json").write_text(
            "{}", encoding="utf-8"
        )
        session = hub._sessions["fake"]
        nested = Path(entry["workspaceRoot"]) / "src" / "components" / "A.vue"
        assert hub._project_config_for(session, str(nested)) is None

    @pytest.mark.anyio
    async def test_config_is_found_from_a_nested_file(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        root = Path(entry["workspaceRoot"])
        (root / "jsconfig.json").write_text("{}", encoding="utf-8")
        session = hub._sessions["fake"]
        nested = root / "src" / "components" / "A.vue"
        assert hub._project_config_for(session, str(nested)) == str(
            root / "jsconfig.json"
        )
        # A file URI is normalised too (Volar sends a plain path, but the
        # hub should not be fooled by one that is not).
        assert hub._project_config_for(session, nested.as_uri()) == str(
            root / "jsconfig.json"
        )


class TestTsserverRelayAgainstTheFakeServer:
    """The regression that matters: without the relay, every call hangs.

    ``--relay`` makes the fake block ``documentSymbol`` and suppress the
    diagnostics publish until its client answers the ``_vue:projectInfo``
    question it asks on ``didOpen`` — the real Volar behaviour.
    """

    @pytest.mark.anyio
    async def test_relay_answer_unblocks_symbols_and_diagnostics(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        root = Path(entry["workspaceRoot"])
        (root / "tsconfig.json").write_text("{}", encoding="utf-8")
        (root / "v.py").write_text("clean line\nx = 1  # BUG\n", encoding="utf-8")
        await _start(hub, entry)

        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert "Thing" in symbols
        # The fake echoes back the configFileName the hub answered with, so
        # this also proves the response survived the wire in the right shape.
        assert f"config:{root / 'tsconfig.json'}" in symbols

        diagnostics = hub.call_tool_sync("fake", "lsp_fake_diagnostics", {"path": "v.py"})
        assert "v.py:2:10 error: found a BUG here" in diagnostics
        assert hub._sessions["fake"].ts_relay_counts.get("_vue:projectInfo") == 1

    @pytest.mark.anyio
    async def test_relay_answers_null_when_the_workspace_has_no_config(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        (Path(entry["workspaceRoot"]) / "v.py").write_text("x = 1  # BUG\n", encoding="utf-8")
        await _start(hub, entry)
        # No tsconfig/jsconfig: the hub answers null, which is Volar's own
        # "use the inferred project" signal — the call must still resolve.
        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert "config:None" in symbols
        tail = hub.log_tail("fake")
        assert "answering null" in tail

    @pytest.mark.anyio
    async def test_unparsable_relay_request_is_logged_and_survived(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay", "--relay-unknown")
        root = Path(entry["workspaceRoot"])
        (root / "jsconfig.json").write_text("{}", encoding="utf-8")
        (root / "v.py").write_text("x = 1  # BUG\n", encoding="utf-8")
        await _start(hub, entry)
        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        # The garbage request must not have broken the session.
        assert "Thing" in symbols
        assert "unrecognised tsserver/request" in hub.log_tail("fake")

    @pytest.mark.anyio
    async def test_started_session_records_a_log_line(self, hub, tmp_path):
        # A quiet server otherwise leaves no log file at all, which is when
        # the log endpoint is wanted most.
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        await _start(hub, entry)
        tail = hub.log_tail("fake")
        assert "[hub] fake running" in tail
        assert "no tsconfig/jsconfig" in tail

    @pytest.mark.anyio
    async def test_configuration_answers_are_null_not_empty_objects(self, hub, tmp_path):
        # The hub used to answer workspace/configuration with {} per item.
        # Volar does `getConfiguration('html.customData') ?? []` and then
        # iterates it: {} is not nullish and not iterable, so the language
        # service plugin threw 'TypeError: paths is not iterable' and the
        # whole vue-language-server process died on the first completion
        # request. In relay mode the fake applies that same logic and exits
        # with code 4 if the hub ever answers {} again.
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "v.py").write_text("x = 1\n", encoding="utf-8")
        hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        assert "not iterable" not in hub.log_tail("fake")

    @pytest.mark.anyio
    async def test_declared_settings_are_returned_verbatim(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        entry["settings"] = {"html.customData": ["/custom/data.json"]}
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "v.py").write_text("x = 1\n", encoding="utf-8")
        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert "Thing" in symbols


# ---------------------------------------------------------------------------
# /api/lsp endpoints
# ---------------------------------------------------------------------------


class FakeLspManager:
    installed = True

    def __init__(self, status: Optional[List[Dict[str, Any]]] = None) -> None:
        self._status = status or []
        self.started: List[str] = []
        self.stopped: List[str] = []
        self.restarts: List[Any] = []

    async def restart(self, servers, enabled_ids=None) -> None:
        self.restarts.append((servers, enabled_ids))

    def status(self):
        return self._status

    def active_tool_defs(self, ids):
        return [], {}

    def call_tool_sync(self, sid, name, args, timeout=30.0):
        return "ok"

    def server_status(self, sid):
        return next((s["status"] for s in self._status if s["id"] == sid), None)

    def start_server(self, sid):
        self.started.append(sid)

    def stop_server(self, sid):
        self.stopped.append(sid)

    def log_tail(self, sid, chars=4000):
        return f"log tail of {sid}"


@pytest.fixture
def lsp_client(monkeypatch, tmp_path):
    from src import config as config_module

    for key in config_module.SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
    # Point the hub at a temp declaration file (fake servers), so the real
    # .nojae/lsp-settings.json stays untouched.
    servers_dir = tmp_path / "lsp_connectors"
    servers_dir.mkdir()
    workspace = tmp_path / "web"
    workspace.mkdir()
    servers_path = servers_dir / "lsp-settings.json"
    servers_path.write_text(
        _servers_doc(
            pyright=_entry(
                [sys.executable, FAKE_SERVER],
                label="Pyright (Python)",
                env={"SECRET": "do-not-leak"},
            ),
            vue=_entry(
                [sys.executable, FAKE_SERVER],
                label="Vue",
                extensions=[".vue"],
                enable=False,
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lsp_constants, "LSP_CONNECTORS_DIR", servers_dir)
    monkeypatch.setattr(lsp_constants, "LSP_SETTINGS_PATH", servers_path)
    monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", servers_dir / "restricted-paths.txt")
    # Keep the MCP hub off the developer's own mcp-settings.json: no connector is
    # declared in these tests unless they write that file themselves.
    monkeypatch.setattr(
        mcphub.constants, "MCP_SETTINGS_PATH", servers_dir / "mcp-settings.json"
    )
    monkeypatch.setattr(
        routes,
        "load_endpoints_config",
        lambda: (
            SimpleNamespace(
                serverUrl="http://x/v1", endpoint="m", model_list=lambda: ["m"],
                to_kwargs=lambda: {}, apiKey="", sysprompt=None,
            ),
            None,
        ),
    )
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, app


class TestLspEndpoints:
    def test_get_lists_servers_with_stopped_status(self, lsp_client):
        client, _ = lsp_client
        body = client.get("/api/lsp").json()
        assert [s["id"] for s in body["servers"]] == ["pyright", "vue"]
        assert body["servers"][0]["status"] == "stopped"
        assert body["servers"][0]["active"] is False
        assert body["servers"][1]["active"] is False  # enable: false
        assert body["maxIterations"] == 25

    def test_get_never_leaks_env_or_initialization_secrets(self, lsp_client):
        client, _ = lsp_client
        text = client.get("/api/lsp").text
        assert "do-not-leak" not in text
        assert "SECRET" not in text

    def test_active_requires_both_gates(self, lsp_client, monkeypatch):
        import os

        client, _ = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        body = client.get("/api/lsp").json()
        by_id = {s["id"]: s for s in body["servers"]}
        assert by_id["pyright"]["active"] is True
        assert by_id["vue"]["active"] is False  # enable flag still false
        monkeypatch.setenv("LSP_ENABLED", "vue")
        body = client.get("/api/lsp").json()
        assert {s["id"]: s["active"] for s in body["servers"]} == {
            "pyright": False,
            "vue": False,
        }

    def test_post_persists_to_env_file_and_writes_enable_back(self, lsp_client):
        client, _ = lsp_client
        response = client.post(
            "/api/lsp", json={"enabled": ["pyright"], "maxIterations": 7}
        )
        assert response.status_code == 200
        body = response.json()
        assert [s["id"] for s in body["servers"] if s["active"]] == ["pyright"]
        assert body["maxIterations"] == 7
        assert os.environ["LSP_ENABLED"] == "pyright"
        assert os.environ["MAX_TOOL_ITERATION"] == "7"
        from src import config as config_module

        file_values = dotenv_values(config_module.ENV_PATH)
        assert file_values["LSP_ENABLED"] == "pyright"
        # enable flag written back for every declared server, both gates move.
        data = json.loads(lsp_constants.LSP_SETTINGS_PATH.read_text(encoding="utf-8"))
        assert data["pyright"]["enable"] is True
        assert data["vue"]["enable"] is False
        assert data["pyright"]["env"] == {"SECRET": "do-not-leak"}  # keys kept

    def test_post_empty_list_deactivates_all(self, lsp_client):
        client, _ = lsp_client
        client.post("/api/lsp", json={"enabled": ["pyright", "vue"]})
        response = client.post("/api/lsp", json={"enabled": []})
        assert response.status_code == 200
        assert "LSP_ENABLED" not in os.environ
        assert not [s for s in response.json()["servers"] if s["active"]]
        data = json.loads(lsp_constants.LSP_SETTINGS_PATH.read_text(encoding="utf-8"))
        assert data["pyright"]["enable"] is False

    def test_post_unknown_server_returns_422(self, lsp_client):
        client, _ = lsp_client
        response = client.post("/api/lsp", json={"enabled": ["ghost"]})
        assert response.status_code == 422
        assert "ghost" in response.text

    def test_post_invalid_max_iterations_returns_422(self, lsp_client):
        client, _ = lsp_client
        assert client.post("/api/lsp", json={"maxIterations": 0}).status_code == 422
        assert client.post("/api/lsp", json={"maxIterations": 9999}).status_code == 422

    def test_post_during_a_run_returns_409(self, lsp_client):
        client, app = lsp_client
        app.state.run_state.status = "running"
        response = client.post("/api/lsp", json={"enabled": ["pyright"]})
        assert response.status_code == 409

    def test_post_reconciles_running_servers(self, lsp_client):
        client, app = lsp_client
        hub = FakeLspManager(
            [{"id": "pyright", "status": "running", "error": None},
             {"id": "vue", "status": "stopped", "error": None}]
        )
        app.state.lsp = hub
        # vue asked for: start it; pyright stays (already running).
        client.post("/api/lsp", json={"enabled": ["pyright", "vue"]})
        assert hub.started == ["vue"]
        assert hub.stopped == []
        # pyright dropped while running: stop it.
        client.post("/api/lsp", json={"enabled": ["vue"]})
        assert hub.stopped == ["pyright"]

    def test_merges_manager_status_into_the_payload(self, lsp_client):
        client, app = lsp_client
        app.state.lsp = FakeLspManager(
            [{"id": "pyright", "status": "error", "error": "boom", "toolCount": 3}]
        )
        by_id = {s["id"]: s for s in client.get("/api/lsp").json()["servers"]}
        assert by_id["pyright"]["status"] == "error"
        assert by_id["pyright"]["error"] == "boom"
        assert by_id["pyright"]["toolCount"] == 3
        assert by_id["vue"]["status"] == "stopped"

    def test_restart_rereads_the_file_and_reapplies_the_gate(
        self, lsp_client, monkeypatch
    ):
        client, app = lsp_client
        # The hub under test is a fake, so spawning is allowed here: the
        # autouse switch would otherwise blank the scan (tested separately).
        monkeypatch.delenv("BENCHMARKIT_DISABLE_LSP", raising=False)
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        hub = FakeLspManager()
        app.state.lsp = hub

        response = client.post("/api/lsp/restart")

        assert response.status_code == 200
        body = response.json()
        assert {server["id"] for server in body["servers"]} == {"pyright", "vue"}
        assert body["serversError"] is None
        # Fresh file scan + the two-gate intersection (vue is enable:false).
        assert len(hub.restarts) == 1
        servers, enabled_ids = hub.restarts[0]
        assert [server["id"] for server in servers] == ["pyright", "vue"]
        assert enabled_ids == ["pyright"]

    def test_restart_keeps_a_dormant_file_dormant(self, lsp_client, monkeypatch):
        client, app = lsp_client
        monkeypatch.delenv("BENCHMARKIT_DISABLE_LSP", raising=False)
        monkeypatch.delenv("LSP_ENABLED", raising=False)  # env gate closed
        hub = FakeLspManager()
        app.state.lsp = hub
        client.post("/api/lsp/restart")
        servers, enabled_ids = hub.restarts[0]
        assert [server["id"] for server in servers] == ["pyright", "vue"]
        assert enabled_ids == []

    def test_restart_honours_the_disable_switch(self, lsp_client, monkeypatch):
        # BENCHMARKIT_DISABLE_LSP keeps the endpoints alive but spawns
        # nothing — the same rule as the app lifespan (tests rely on it).
        client, app = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        hub = FakeLspManager()
        app.state.lsp = hub
        client.post("/api/lsp/restart")
        servers, enabled_ids = hub.restarts[0]
        assert servers == []
        assert enabled_ids == []

    def test_restart_during_a_run_returns_409(self, lsp_client):
        client, app = lsp_client
        hub = FakeLspManager()
        app.state.lsp = hub
        app.state.run_state.status = "running"
        response = client.post("/api/lsp/restart")
        assert response.status_code == 409
        assert hub.restarts == []
        app.state.run_state.status = "idle"

    def test_restart_without_a_hub_returns_503(self, lsp_client):
        client, app = lsp_client
        app.state.lsp = None
        assert client.post("/api/lsp/restart").status_code == 503

    def test_log_endpoint_returns_the_tail(self, lsp_client):
        client, app = lsp_client
        app.state.lsp = FakeLspManager()
        body = client.get("/api/lsp/log", params={"server": "pyright"}).json()
        assert body == {"server": "pyright", "log": "log tail of pyright"}
        assert client.get("/api/lsp/log").status_code == 404


class TestActiveServerIds:
    def test_both_gates_must_agree(self, lsp_client, monkeypatch):
        monkeypatch.setenv("LSP_ENABLED", "pyright,vue")
        assert routes._active_server_ids() == ["pyright"]
        monkeypatch.delenv("LSP_ENABLED")
        assert routes._active_server_ids() == []


# ---------------------------------------------------------------------------
# /api/chat wiring: the merged tool loop
# ---------------------------------------------------------------------------


def _lsp_def(name: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": f"d {name}", "parameters": {}},
    }


class TestChatLspWiring:
    def _capture_stream(self, monkeypatch, probe: tuple = ()):
        """Stub ``stream_chat`` capturing the merged tool-loop kwargs.

        ``probe`` names are the tools the stub executes through the merged
        router (only tools actually offered by a hub may be probed).
        """
        captured: Dict[str, Any] = {}

        def fake_stream(messages, model, server_url, emit, sampling=None,
                        cancel_event=None, pause_event=None, api_key="none", sysprompt=None,
                        tools=None, tool_executor=None, tool_servers=None,
                        max_tool_iterations=25, agent_instructions=None):
            captured["tools"] = tools
            captured["tool_servers"] = tool_servers
            captured["max"] = max_tool_iterations
            if tool_executor:
                for tool_name in probe:
                    captured[f"exec_{tool_name}"] = tool_executor(tool_name, {})
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake_stream)
        return captured

    def test_lsp_only_keeps_one_shared_iteration_cap(self, lsp_client, monkeypatch):
        client, app = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        monkeypatch.setenv("MAX_TOOL_ITERATION", "9")

        class Hub(FakeLspManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

            def call_tool_sync(self, sid, name, args, timeout=30.0):
                return f"{sid}:{name}"

        app.state.lsp = Hub()
        captured = self._capture_stream(monkeypatch, ("lsp_pyright_hover",))
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert app.state.lsp.ids == ["pyright"]
        assert [d["function"]["name"] for d in captured["tools"]] == ["lsp_pyright_hover"]
        assert captured["tool_servers"] == {"lsp_pyright_hover": "pyright"}
        assert captured["max"] == 9
        assert captured["exec_lsp_pyright_hover"] == "pyright:lsp_pyright_hover"

    def test_mcp_and_lsp_merge_into_one_loop(self, lsp_client, monkeypatch):
        client, app = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        # The MCP connector comes from the declaration file, whose ``enable``
        # flag is the only activation source.
        externals_path = mcphub.constants.mcp_settings_path()
        externals_path.parent.mkdir(parents=True, exist_ok=True)
        externals_path.write_text(
            json.dumps({"console_mcp": {"type": "local", "enable": True, "command": ["x"]}}),
            encoding="utf-8",
        )

        mcp_tools = [_lsp_def("execute_command")]

        class McpHub(FakeLspManager):
            def active_tool_defs(self, ids):
                return mcp_tools, {"execute_command": "console_mcp"}

            def call_tool_sync(self, sid, name, args, timeout=30.0):
                return f"mcp:{name}"

        class LspHub(FakeLspManager):
            def active_tool_defs(self, ids):
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

            def call_tool_sync(self, sid, name, args, timeout=30.0):
                return f"lsp:{name}"

        app.state.mcp = McpHub()
        app.state.lsp = LspHub()
        captured = self._capture_stream(
            monkeypatch, ("execute_command", "lsp_pyright_hover")
        )
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert [d["function"]["name"] for d in captured["tools"]] == [
            "execute_command",
            "lsp_pyright_hover",
        ]
        assert captured["tool_servers"] == {
            "execute_command": "console_mcp",
            "lsp_pyright_hover": "pyright",
        }
        # The router sends each tool to its own hub.
        assert captured["exec_execute_command"] == "mcp:execute_command"
        assert captured["exec_lsp_pyright_hover"] == "lsp:lsp_pyright_hover"

    def test_body_filter_cannot_revive_a_settings_disabled_server(
        self, lsp_client, monkeypatch
    ):
        client, app = lsp_client

        class Hub(FakeLspManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

        hub = Hub()
        app.state.lsp = hub
        self._capture_stream(monkeypatch)
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "lspEnabled": ["pyright"],  # user toggle on...
            },
        )
        assert response.status_code == 200
        # ...but LSP_ENABLED is empty, so no server id ever reaches the hub.
        assert hub.ids == []

    def test_body_filter_intersects_the_activated_set(self, lsp_client, monkeypatch):
        client, app = lsp_client
        import os

        os.environ["LSP_ENABLED"] = "pyright,vue"

        class Hub(FakeLspManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

        hub = Hub()
        app.state.lsp = hub
        self._capture_stream(monkeypatch)
        client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}],
                  "lspEnabled": ["vue"]},
        )
        # vue is Settings-active but toggled off for this conversation;
        # pyright is on in Settings but not listed in the body filter.
        assert hub.ids == []


# ---------------------------------------------------------------------------
# The app lifespan must never spawn language servers in tests
# (conftest sets BENCHMARKIT_DISABLE_LSP, mirroring the MCP rule).
# ---------------------------------------------------------------------------


def test_lifespan_starts_no_servers_without_the_env_switch(lsp_client):
    _client, app = lsp_client
    # create_app's lifespan ran with BENCHMARKIT_DISABLE_LSP=1 (autouse
    # conftest fixture): an empty LspManager, so nothing was ever spawned.
    assert app.state.lsp.status() == []
