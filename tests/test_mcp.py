"""Tests for MCP connector declaration, settings API, and the chat tool loop."""

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.config import max_tool_iteration
from src.core import chat as chat_core
from src.core import mcphub
from src.core.mcphub import (
    McpManager,
    connector_environment,
    extract_tool_result_text,
    load_externals,
    set_externals_enabled,
    tools_to_openai_format,
)

# ---------------------------------------------------------------------------
# externals.json loading and write-back
# ---------------------------------------------------------------------------


def _write_externals(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


class TestLoadExternals:
    def test_missing_file_is_empty_and_silent(self, tmp_path):
        assert load_externals(tmp_path / "externals.json") == ([], [])

    def test_parses_remote_and_local_entries(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "context7": {
                    "type": "remote",
                    "enable": True,
                    "url": "https://mcp.context7.com/mcp",
                    "headers": {"CONTEXT7_API_KEY": "ctx7sk-1"},
                },
                "chrome-devtools": {
                    "type": "local",
                    "command": ["npx", "-y", "chrome-devtools-mcp@latest"],
                },
            },
        )
        connectors, errors = load_externals(path)
        assert errors == []
        by_id = {c["id"]: c for c in connectors}
        remote = by_id["context7"]
        assert remote["kind"] == "remote"
        assert remote["enable"] is True
        assert remote["endpoint"] == "https://mcp.context7.com/mcp"
        assert remote["headers"] == {"CONTEXT7_API_KEY": "ctx7sk-1"}
        local = by_id["chrome-devtools"]
        assert local["enable"] is False  # unset enable defaults to off
        assert local["command"] == ["npx", "-y", "chrome-devtools-mcp@latest"]
        assert local["endpoint"] == ""

    def test_string_command_is_split_and_args_appended(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {"devtools": {"type": "local", "command": "npx -y 'chrome devtools'", "args": ["--stdio"]}},
        )
        connectors, errors = load_externals(path)
        assert errors == []
        assert connectors[0]["command"] == ["npx", "-y", "chrome devtools", "--stdio"]

    def test_invalid_json_reports_and_yields_nothing(self, tmp_path):
        path = tmp_path / "externals.json"
        path.write_text("{oops", encoding="utf-8")
        connectors, errors = load_externals(path)
        assert connectors == [] and len(errors) == 1 and "JSON" in errors[0]

    def test_non_object_root_reports(self, tmp_path):
        path = tmp_path / "externals.json"
        path.write_text("[]", encoding="utf-8")
        _, errors = load_externals(path)
        assert "object" in errors[0]

    def test_bad_entries_skipped_with_notes(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "no_type": {"enable": True},
                "bad_remote": {"type": "remote", "url": "ftp://x"},
                "bad_local": {"type": "local"},
                "bad_headers": {"type": "remote", "url": "http://x", "headers": "nope"},
                "not_an_object": [],
            },
        )
        connectors, errors = load_externals(path)
        assert connectors == []
        assert len(errors) == 5

    def test_valid_entries_survive_bad_neighbours(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "oops": {"type": "sideways"},
                "good": {"type": "remote", "url": "https://x/mcp", "description": "Docs."},
            },
        )
        connectors, errors = load_externals(path)
        assert [c["id"] for c in connectors] == ["good"]
        assert connectors[0]["description"] == "Docs."
        assert len(errors) == 1

    def test_env_references_are_expanded(self, tmp_path, monkeypatch):
        """``${NAME}`` in url/headers/env/command comes from the environment.

        ``${NAME:-fallback}`` covers the optional-key case; an expanded value
        never re-splits, so the ``--dir`` argument below stays one argument
        even though it contains a space.
        """
        monkeypatch.setenv("CTX_KEY", "ctx7sk-real")
        monkeypatch.setenv("VOLTA_BIN", "/home/vgr/.volta/bin")
        monkeypatch.setenv("WS_DIR", str(tmp_path / "a dir"))
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "remote": {
                    "type": "remote",
                    "url": "https://mcp.example.com/${CTX_PATH:-mcp}",
                    "headers": {"Authorization": "Bearer ${CTX_KEY}"},
                    "env": {"TOKEN": "${CTX_KEY}"},
                },
                "local": {
                    "type": "local",
                    "command": ["${VOLTA_BIN}/npx", "-y", "pkg@$latest", "--dir=${WS_DIR}/sub"],
                },
            },
        )
        connectors, errors = load_externals(path)
        assert errors == []
        by_id = {c["id"]: c for c in connectors}
        assert by_id["remote"]["endpoint"] == "https://mcp.example.com/mcp"
        assert by_id["remote"]["headers"] == {"Authorization": "Bearer ctx7sk-real"}
        assert by_id["remote"]["env"] == {"TOKEN": "ctx7sk-real"}
        assert by_id["local"]["command"] == [
            "/home/vgr/.volta/bin/npx",
            "-y",
            "pkg@$latest",
            f"--dir={tmp_path / 'a dir'}/sub",
        ]

    def test_whole_url_can_be_an_env_reference(self, tmp_path, monkeypatch):
        """Expansion runs before the http(s) check, so ``${VAR}`` may hold a URL."""
        monkeypatch.setenv("MCP_BASE_URL", "https://mcp.example.com")
        path = _write_externals(
            tmp_path / "externals.json",
            {"ctx": {"type": "remote", "url": "${MCP_BASE_URL}/mcp"}},
        )
        connectors, errors = load_externals(path)
        assert errors == []
        assert connectors[0]["endpoint"] == "https://mcp.example.com/mcp"

    def test_unresolved_env_reference_is_kept_and_reported(self, tmp_path, monkeypatch):
        """An unset name is left written verbatim plus one note, never blanked."""
        monkeypatch.delenv("NO_SUCH_CONNECTOR_KEY", raising=False)
        path = _write_externals(
            tmp_path / "externals.json",
            {"remote": {"type": "remote", "url": "https://x/mcp", "headers": {"K": "${NO_SUCH_CONNECTOR_KEY}"}}},
        )
        connectors, errors = load_externals(path)
        assert len(connectors) == 1
        assert connectors[0]["headers"] == {"K": "${NO_SUCH_CONNECTOR_KEY}"}
        assert len(errors) == 1
        assert "NO_SUCH_CONNECTOR_KEY" in errors[0]
        assert "not set in the environment" in errors[0]

    def test_secrets_survive_the_write_back_untouched(self, tmp_path, monkeypatch):
        """Toggling ``enable`` must not persist an expanded value into the file."""
        monkeypatch.setenv("CTX_KEY", "ctx7sk-real")
        path = _write_externals(
            tmp_path / "externals.json",
            {"a": {"type": "remote", "url": "https://a", "enable": False, "headers": {"K": "${CTX_KEY}"}}},
        )
        assert set_externals_enabled({"a": True}, path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["a"]["headers"] == {"K": "${CTX_KEY}"}
        assert data["a"]["enable"] is True


class TestSetExternalsEnabled:
    def test_missing_file_is_noop(self, tmp_path):
        path = tmp_path / "externals.json"
        assert set_externals_enabled({"a": True}, path) == []
        assert not path.exists()

    def test_flips_flags_and_preserves_everything_else(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "a": {"type": "remote", "url": "https://a", "enable": False, "custom": "keep"},
                "b": {"type": "local", "command": ["x"], "enable": True},
                "untouched": {"type": "remote", "url": "https://u", "enable": True},
            },
        )
        assert set_externals_enabled({"a": True, "b": False}, path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["a"] == {"type": "remote", "url": "https://a", "enable": True, "custom": "keep"}
        assert data["b"]["enable"] is False
        assert data["untouched"]["enable"] is True

    def test_unknown_id_is_not_added(self, tmp_path):
        path = _write_externals(tmp_path / "externals.json", {})
        set_externals_enabled({"ghost": True}, path)
        assert json.loads(path.read_text(encoding="utf-8")) == {}

    def test_broken_file_is_left_untouched(self, tmp_path):
        path = tmp_path / "externals.json"
        path.write_text("{oops", encoding="utf-8")
        errors = set_externals_enabled({"a": True}, path)
        assert errors and "Could not update" in errors[0]
        assert path.read_text(encoding="utf-8") == "{oops"


# ---------------------------------------------------------------------------
# Tool formatting helpers (ported from the agentic-tool runner)
# ---------------------------------------------------------------------------


class TestToolHelpers:
    def test_tools_to_openai_format(self):
        tool = SimpleNamespace(
            name="execute_command",
            description="Run a command",
            input_schema={"type": "object", "properties": {}},
        )
        [entry] = tools_to_openai_format([tool])
        assert entry["function"]["name"] == "execute_command"
        assert entry["function"]["description"] == "Run a command"

    def test_extract_tool_result_text_joins_and_truncates(self):
        result = SimpleNamespace(
            is_error=False,
            content=[SimpleNamespace(text="a"), SimpleNamespace(text="b")],
        )
        assert extract_tool_result_text(result) == "a\nb"
        big = SimpleNamespace(is_error=False, content=[SimpleNamespace(text="x" * 100)])
        assert extract_tool_result_text(big, max_length=10).endswith("[truncated]")

    def test_extract_tool_result_error_flag(self):
        result = SimpleNamespace(is_error=True, content=[SimpleNamespace(text="boom")])
        assert "boom" in extract_tool_result_text(result)


# ---------------------------------------------------------------------------
# McpManager (no real processes or clients)
# ---------------------------------------------------------------------------


def _declared_manager() -> McpManager:
    """A manager fed the dicts ``load_externals`` would produce."""
    return McpManager(
        [
            {
                "id": "on_ext",
                "label": "on_ext",
                "description": "",
                "endpoint": "https://example.com/mcp",
                "command": None,
                "kind": "remote",
                "enable": True,
                "headers": {"SECRET": "leaky"},
                "env": {"TOKEN": "leaky"},
            },
            {
                "id": "off_ext",
                "label": "off_ext",
                "description": "",
                "endpoint": "https://example.com/other",
                "command": None,
                "kind": "remote",
                "enable": False,
                "headers": {},
                "env": {},
            },
            {
                "id": "local_ext",
                "label": "local_ext",
                "description": "",
                "endpoint": "",
                "command": ["npx", "-y", "some-mcp"],
                "kind": "local",
                "enable": True,
                "headers": {},
                "env": {},
            },
        ]
    )


class TestMcpManager:
    def test_active_tool_defs_only_running_connectors(self):
        manager = _declared_manager()
        manager._tools["on_ext"] = [
            SimpleNamespace(name="execute_command", description="", input_schema=None)
        ]
        manager._tools["local_ext"] = [
            SimpleNamespace(name="browser_navigate", description="", input_schema=None)
        ]
        tools, mapping = manager.active_tool_defs(["on_ext", "local_ext"])
        assert tools == []  # nothing marked running yet
        manager._connectors["on_ext"]["status"] = "running"
        manager._connectors["local_ext"]["status"] = "error"
        tools, mapping = manager.active_tool_defs(["on_ext", "local_ext"])
        assert [t["function"]["name"] for t in tools] == ["execute_command"]
        assert mapping == {"execute_command": "on_ext"}

    def test_status_lists_all_connectors_without_secrets(self):
        manager = _declared_manager()
        entries = manager.status()
        status = {entry["id"]: entry for entry in entries}
        assert status["on_ext"]["status"] == "stopped"
        assert status["on_ext"]["kind"] == "remote"
        assert status["local_ext"]["command"] == ["npx", "-y", "some-mcp"]
        for entry in entries:
            assert "headers" not in entry
            assert "env" not in entry
        assert "leaky" not in repr(entries)

    def test_start_all_skips_disabled_entries(self):
        import asyncio

        manager = _declared_manager()
        started: List[str] = []

        async def fake_start(connector_id: str) -> None:
            started.append(connector_id)

        manager._installed = True
        manager._start_one = fake_start
        asyncio.run(manager.start_all())
        asyncio.run(manager.stop_all())
        assert sorted(started) == ["local_ext", "on_ext"]

    def test_start_stop_connector_are_noops_before_start(self):
        manager = _declared_manager()
        manager.start_connector("on_ext")  # must not raise without a loop
        manager.stop_connector("on_ext")
        assert manager.connector_status("on_ext") == "stopped"
        assert manager.connector_status("ghost") is None

    def test_stop_all_is_idempotent_without_start(self):
        manager = _declared_manager()
        import asyncio

        asyncio.run(manager.stop_all())
        asyncio.run(manager.stop_all())  # second call must not raise


# ---------------------------------------------------------------------------
# win-07 / A-D4: a hub refuses what its guardrails cannot bound
# ---------------------------------------------------------------------------


def _unbounded_guardrails(monkeypatch, hub_module) -> None:
    """Force one hub's reach seam to ``False``.

    Patched on the hub's own ``constants`` module — the single named seam both
    hubs read through — never by pretending to be another OS. It is the whole
    predicate: the guardrails of a deployment that wrote the other platform's
    values are what answer "no" there, and the refusal is one state with one
    consequence downstream.
    """
    monkeypatch.setattr(
        hub_module.constants, "guardrails_speak_here", lambda prefixes=None: False
    )


def _declaration(kind: str, connector_id: str, *, enable: bool = True):
    """One connector dict as ``load_externals`` would hand it to the hub."""
    return {
        "id": connector_id,
        "label": connector_id,
        "description": "",
        "endpoint": "https://example.com/mcp" if kind == "remote" else "",
        "command": None if kind == "remote" else ["python", "mcp_connectors/x_mcp.py"],
        "kind": kind,
        "enable": enable,
        "headers": {},
        "env": {},
    }


class TestLocalReachRefusal:
    """win-07 / A-D4 on the hub side: refused, named, and never spawned."""

    def test_the_seam_is_bounded_here_so_nothing_is_refused(self):
        """This runner bounds a local connector: guardrails speak, no refusal.

        After target B1 this is the answer on **both** platforms for the MCP
        hub — the connector lists speak both path forms. There is no second
        half to ask any more: the LSP hub's platform fact went with lspfinal-04.
        """
        assert mcphub.constants.guardrails_speak_here() is True
        assert mcphub.discovery.mcp_reach_is_refused() is False

    def test_the_mcp_refusal_asks_only_the_guardrail_half(self):
        """The one predicate, stated as a forcing pair rather than assumed.

        Nothing is patched: the guardrails speak here, so nothing is refused.
        Forcing that answer to ``False`` — the state a deployment with an inert
        list is really in — refuses. This is the whole of the seam now.
        """
        monkeypatch = pytest.MonkeyPatch()
        assert mcphub.discovery.mcp_reach_is_refused() is False
        monkeypatch.setattr(
            mcphub.constants, "guardrails_speak_here", lambda prefixes=None: False
        )
        assert mcphub.discovery.mcp_reach_is_refused() is True
        monkeypatch.undo()

    def test_the_lsp_hub_asks_the_same_single_half(self):
        """The symmetry restored (AGENTS.md: the two hubs mirror each other).

        lspfinal-04 took the LSP hub's platform half away, so its predicate is
        now the shape this hub's already had: forcing *its* guardrail list to
        speak nothing refuses, forcing nothing does not. A deployment fixes
        itself by writing values into ``restricted-paths.txt``, not by the code
        knowing which OS it runs on.
        """
        from src.core.lsphub import constants as lsp_constants
        from src.core.lsphub.discovery import lsp_reach_is_refused

        monkeypatch = pytest.MonkeyPatch()
        assert lsp_reach_is_refused() is False
        monkeypatch.setattr(
            lsp_constants, "guardrails_speak_here", lambda prefixes=None: False
        )
        assert lsp_reach_is_refused() is True
        monkeypatch.undo()

    def test_the_lsp_guardrail_half_speaks_this_runners_path_form(self):
        """B-D1 + lspfinal-04: an inert deny-list is the only thing that refuses.

        Asked of the unpinned hub (nothing is patched here, unlike the two
        seam-forcing tests above), the LSP hub's own list answers the shape
        question True on either runner: on Windows because the entries B-D1
        added speak a drive-rooted working location, on Linux because the POSIX
        ones always did. Nothing is left that could refuse, so the refusal is
        ``False`` here on both platforms.
        """
        from src.core.lsphub import constants as lsp_constants
        from src.core.lsphub.discovery import lsp_reach_is_refused, restricted_lsp_roots

        entries = restricted_lsp_roots()
        assert entries
        assert lsp_constants.guardrails_speak_here(entries) is True
        assert lsp_reach_is_refused() is False

    def test_both_hubs_refuse_in_the_same_words(self):
        """The symmetry test: one sentence for one missing capability.

        The two hubs mirror each other, and a connector card explaining the same
        gap differently from a language-server card is the first thing a reader
        would distrust.
        """
        from src.core.lsphub import constants as lsp_constants

        assert mcphub.constants.LOCAL_REACH_REFUSAL == lsp_constants.LOCAL_REACH_REFUSAL
        # And the sentence names the gap and the capability, not a symptom.
        assert "bounded" in mcphub.constants.LOCAL_REACH_REFUSAL
        # lspfinal-06: it names the cause an operator can fix — a guardrail list
        # with no entry in this filesystem's path form — and no longer points at
        # target B, which is finished.
        assert "guardrail" in mcphub.constants.LOCAL_REACH_REFUSAL
        assert "target B" not in mcphub.constants.LOCAL_REACH_REFUSAL

    def test_a_local_entry_gets_the_note_and_a_remote_one_does_not(
        self, tmp_path, monkeypatch
    ):
        _unbounded_guardrails(monkeypatch, mcphub)
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "remote_live": {
                    "type": "remote",
                    "enable": True,
                    "url": "https://example.com/mcp",
                },
                "local_live": {
                    "type": "local",
                    "enable": True,
                    "command": ["python", "mcp_connectors/console_mcp.py"],
                },
                "local_off": {
                    "type": "local",
                    "enable": False,
                    "command": ["python", "mcp_connectors/fedit_mcp.py"],
                },
            },
        )
        connectors, errors = load_externals(path)
        # The entries stay loaded — a card naming the gap beats a vanished
        # connector — and only the one that *asks to run* is noted.
        assert [connector["id"] for connector in connectors] == [
            "local_live",
            "local_off",
            "remote_live",
        ]
        assert len(errors) == 1
        assert "'local_live'" in errors[0]
        assert mcphub.constants.LOCAL_REACH_REFUSAL in errors[0]

    def test_an_unresolved_env_reference_and_the_note_are_both_reported(
        self, tmp_path, monkeypatch
    ):
        """The new note joins the existing ones, it never replaces them."""
        _unbounded_guardrails(monkeypatch, mcphub)
        monkeypatch.delenv("NOT_SET_ANYWHERE_MARKER", raising=False)
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "local_live": {
                    "type": "local",
                    "enable": True,
                    "command": ["python", "${NOT_SET_ANYWHERE_MARKER}/x_mcp.py"],
                },
            },
        )
        connectors, errors = load_externals(path)
        assert connectors[0]["command"] == ["python", "${NOT_SET_ANYWHERE_MARKER}/x_mcp.py"]
        assert len(errors) == 2
        assert any("is not set in the environment" in note for note in errors)
        assert any(mcphub.constants.LOCAL_REACH_REFUSAL in note for note in errors)

    def test_the_endpoint_reports_the_note_and_names_no_secret(
        self, mcp_client, tmp_path, monkeypatch
    ):
        client, _ = mcp_client
        _unbounded_guardrails(monkeypatch, mcphub)
        _write_externals(
            tmp_path / "mcp_connectors" / "externals.json",
            {
                "console": {
                    "type": "local",
                    "enable": True,
                    "command": [
                        "python",
                        "mcp_connectors/console_mcp.py",
                        "--token",
                        "a-secret-in-the-argv",
                    ],
                    "env": {"API_KEY": "a-secret-in-the-env"},
                }
            },
        )
        body = client.get("/api/mcp").json()
        note = body["externalsError"]
        assert mcphub.constants.LOCAL_REACH_REFUSAL in note
        # The refusal names the gap, never the command line it refused to run
        # (``env`` never leaves the server at all, and a note that quoted the
        # argv would carry whatever the declaration put in it).
        assert "a-secret-in-the-argv" not in note
        assert "console_mcp" not in note
        for secret in ("a-secret-in-the-env", "API_KEY"):
            assert secret not in json.dumps(body)

    def test_a_refused_local_entry_is_error_and_a_remote_one_still_runs(
        self, monkeypatch
    ):
        """``start_all`` reaches the remote entry and never touches the local one.

        The stand-in for ``_start_connector`` is where a stdio child would be
        spawned (or an HTTP session opened), so ``spawned`` is the direct answer
        to "did the hub start a process it should have refused" — and the
        remote entry ending up ``running`` is what pins that the refusal costs
        a connector with no local path reach nothing.
        """
        import asyncio

        _unbounded_guardrails(monkeypatch, mcphub)
        spawned: List[str] = []

        async def pretend_connect(connector_id: str, info: Dict[str, Any]) -> None:
            spawned.append(connector_id)
            info["status"] = "running"
            info["error"] = None

        manager = McpManager(
            [_declaration("local", "local_live"), _declaration("remote", "remote_live")]
        )
        monkeypatch.setattr(manager, "_start_connector", pretend_connect)

        async def scenario():
            await manager.start_all()
            statuses = {entry["id"]: entry for entry in manager.status()}
            await manager.stop_all()
            return statuses

        status = asyncio.run(scenario())
        assert spawned == ["remote_live"]
        assert status["local_live"]["status"] == "error"
        assert status["local_live"]["error"] == mcphub.constants.LOCAL_REACH_REFUSAL
        assert status["remote_live"]["status"] == "running"
        assert status["remote_live"]["error"] is None

    def test_a_refused_entry_is_error_as_soon_as_it_is_declared(
        self, monkeypatch
    ):
        """No start needs to be attempted for the card to say so.

        The entry is named at index time — boot and every hub reload — so the
        panel and the tool catalogue see the same answer the validation note
        gives, and a refused connector never sits in ``stopped``, the state whose
        tools are still offered to chat.
        """
        _unbounded_guardrails(monkeypatch, mcphub)
        manager = McpManager(
            [_declaration("local", "local_live"), _declaration("remote", "remote_live")]
        )
        status = {entry["id"]: entry for entry in manager.status()}
        assert status["local_live"]["status"] == "error"
        assert status["local_live"]["error"] == mcphub.constants.LOCAL_REACH_REFUSAL
        # A remote entry keeps its ordinary starting state, and a disabled local
        # one is the user's own switch rather than a missing capability.
        assert status["remote_live"]["status"] == "stopped"
        disabled = McpManager([_declaration("local", "local_off", enable=False)])
        assert disabled.status()[0]["status"] == "stopped"
        assert disabled.status()[0]["error"] is None

    def test_the_hot_toggle_does_not_spawn_a_refused_entry(self, monkeypatch):
        """``start_connector`` from the settings panel cannot bypass the refusal.

        The dispatch is what would reach ``_start_connector`` and therefore the
        subprocess, so the assertion is on the *dispatch list* being empty as
        well as on the status: a refusal that only arrived after the child was
        spawned would still show ``error`` and pass a status-only check.
        """
        import asyncio

        _unbounded_guardrails(monkeypatch, mcphub)
        manager = McpManager([_declaration("local", "local_live")])
        # A live hub thread, so the synchronous toggle is not a no-op for
        # having no loop; nothing is ever connected on it.
        asyncio.run(manager.start_all())
        dispatched: List[str] = []

        async def record_restart(connector_id: str) -> None:
            dispatched.append(connector_id)

        monkeypatch.setattr(manager, "_restart_one", record_restart)
        manager.start_connector("local_live")
        assert manager.connector_status("local_live") == "error"
        assert (
            manager._connectors["local_live"]["error"]
            == mcphub.constants.LOCAL_REACH_REFUSAL
        )
        asyncio.run(manager.stop_all())
        assert dispatched == []

    def test_its_tools_are_not_offered_to_chat(self, monkeypatch):
        """``active_tool_defs`` publishes only running connectors — unchanged rule."""
        _unbounded_guardrails(monkeypatch, mcphub)
        manager = McpManager([_declaration("local", "local_live")])
        manager._tools["local_live"] = [
            SimpleNamespace(name="execute_command", description="", input_schema=None)
        ]
        tools, mapping = manager.active_tool_defs(["local_live"])
        assert tools == []
        assert mapping == {}

    def test_a_booting_app_does_not_spawn_the_local_connector(
        self, tmp_path, monkeypatch, mcp_client
    ):
        """With the seam forced, the hub's own test switch is not what keeps it quiet.

        The lifespan normally loads ``externals.json`` only when
        ``BENCHMARKIT_DISABLE_MCP`` is unset, so clearing it here puts the test on
        the real boot path: the connector's command *is* a program that leaves a
        marker file behind, and the assertion is that the marker never appears.
        A refusal that only avoided the ``error`` status while still spawning the
        child — the exact bug A-D4 exists to prevent — fails here.
        """
        marker = tmp_path / "spawned.flag"
        _, original = mcp_client
        del original  # the fixture's app already booted with the switch set
        monkeypatch.delenv("BENCHMARKIT_DISABLE_MCP", raising=False)
        _unbounded_guardrails(monkeypatch, mcphub)
        _write_externals(
            tmp_path / "mcp_connectors" / "externals.json",
            {
                "console": {
                    "type": "local",
                    "enable": True,
                    "command": [
                        sys.executable,
                        "-c",
                        f"open({str(marker)!r}, 'w').close()",
                    ],
                }
            },
        )
        app = create_app()
        with TestClient(app) as test_client:
            body = test_client.get("/api/mcp").json()
            assert mcphub.constants.LOCAL_REACH_REFUSAL in body["externalsError"]
            hub = app.state.mcp
            entry = {item["id"]: item for item in hub.status()}["console"]
            assert entry["status"] == "error"
            assert entry["error"] == mcphub.constants.LOCAL_REACH_REFUSAL
        assert not marker.exists(), "the refused connector was spawned anyway"


class TestMcpEnvHelpers:
    def test_max_tool_iteration_default_and_fallbacks(self, monkeypatch):
        monkeypatch.delenv("MAX_TOOL_ITERATION", raising=False)
        monkeypatch.delenv("MAX_MCP_ITERATION", raising=False)
        assert max_tool_iteration() == 25
        monkeypatch.setenv("MAX_TOOL_ITERATION", "7")
        assert max_tool_iteration() == 7
        monkeypatch.setenv("MAX_TOOL_ITERATION", "nope")
        assert max_tool_iteration() == 25
        monkeypatch.setenv("MAX_TOOL_ITERATION", "0")
        assert max_tool_iteration() == 1


class TestConnectorEnvironment:
    def test_relative_temp_vars_never_reach_the_connector(self, tmp_path):
        # A relative TEMP=1 in the environment must not be taken by the
        # Node playwright driver as its temp directory (ENOENT mkdtemp '1/…').
        # The values are relative on *every* platform (gb-01b): what the rule
        # says is "a relative temp value is dropped", and the pinned TMPDIR is
        # absolute in the platform's own spelling.
        env = connector_environment(
            tmp_path, {"TEMP": "1", "TMP": "rel", "TMPDIR": "reltmp"}
        )
        assert "TEMP" not in env
        assert "TMP" not in env
        assert Path(env["TMPDIR"]).is_absolute()

    def test_absolute_temp_vars_are_kept_and_pythonpath_prepended(self, tmp_path):
        # Absolute in the runner's own spelling — a POSIX-looking "/fast/tmp"
        # is not an absolute Windows path and would be dropped, which is the
        # honest verdict, not the invariant (gb-01b).
        absolute = str(tmp_path / "fast-tmp")
        env = connector_environment(
            tmp_path, {"TEMP": absolute, "PYTHONPATH": "other"}
        )
        assert env["TEMP"] == absolute
        entries = env["PYTHONPATH"].split(os.pathsep)
        assert entries[0] == str(tmp_path)
        assert "other" in entries[1:]

    def test_defaults_to_python_tempdir(self, tmp_path):
        import tempfile

        env = connector_environment(tmp_path, {})
        assert env["TMPDIR"] == tempfile.gettempdir()


# ---------------------------------------------------------------------------
# /api/mcp endpoints
# ---------------------------------------------------------------------------


class FakeManager:
    installed = True

    def __init__(self, status: Optional[List[Dict[str, Any]]] = None) -> None:
        self._status = status or []

    def status(self):
        return self._status

    def active_tool_defs(self, ids):
        return [], {}

    def call_tool_sync(self, cid, name, args):
        return "ok"


@pytest.fixture
def mcp_client(monkeypatch, tmp_path):
    from src import config as config_module

    for key in config_module.SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MAX_TOOL_ITERATION", "25")
    monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
    # Point the hub at a temp declaration file, so the developer's own
    # mcp_connectors/externals.json stays untouched. No file is written: the
    # tests that need entries write one themselves.
    connectors_dir = tmp_path / "mcp_connectors"
    connectors_dir.mkdir()
    monkeypatch.setattr(mcphub.constants, "MCP_CONNECTORS_DIR", connectors_dir)
    monkeypatch.setattr(mcphub.constants, "EXTERNALS_PATH", connectors_dir / "externals.json")
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


EXTERNALS_DOC = {
    "context7": {
        "type": "remote",
        "enable": True,
        "url": "https://mcp.context7.com/mcp",
        "headers": {"CONTEXT7_API_KEY": "ctx7sk-secret-value"},
    },
    "chrome-devtools": {
        "type": "local",
        "enable": False,
        "command": ["npx", "-y", "chrome-devtools-mcp@latest"],
    },
}


class RecordingManager:
    installed = True

    def __init__(self, statuses: Dict[str, str]) -> None:
        self._statuses = dict(statuses)
        self.started: List[str] = []
        self.stopped: List[str] = []
        self.restarts: List[Any] = []

    async def restart(self, connectors) -> None:
        self.restarts.append(connectors)

    def status(self):
        return [
            {"id": cid, "status": status, "error": None}
            for cid, status in self._statuses.items()
        ]

    def active_tool_defs(self, ids):
        return [], {}

    def connector_status(self, connector_id):
        return self._statuses.get(connector_id)

    def start_connector(self, connector_id):
        self.started.append(connector_id)
        self._statuses[connector_id] = "starting"

    def stop_connector(self, connector_id):
        self.stopped.append(connector_id)
        self._statuses[connector_id] = "stopped"


class TestMcpEndpoints:
    def _externals_file(self, tmp_path, doc=None) -> Path:
        return _write_externals(
            tmp_path / "mcp_connectors" / "externals.json",
            EXTERNALS_DOC if doc is None else doc,
        )

    def test_get_without_declaration_file_lists_nothing(self, mcp_client):
        client, _ = mcp_client
        body = client.get("/api/mcp").json()
        assert body["connectors"] == []
        assert body["maxIterations"] == 25
        assert body["externalsError"] is None

    def test_get_lists_declared_connectors_with_enable_as_active(
        self, mcp_client, tmp_path
    ):
        self._externals_file(tmp_path)
        client, _ = mcp_client
        body = client.get("/api/mcp").json()
        # Sorted by id, only what externals.json declares.
        assert [c["id"] for c in body["connectors"]] == ["chrome-devtools", "context7"]
        by_id = {c["id"]: c for c in body["connectors"]}
        remote = by_id["context7"]
        assert remote["kind"] == "remote"
        assert remote["status"] == "stopped"
        assert remote["active"] is True  # enable: true
        assert by_id["chrome-devtools"]["active"] is False  # enable: false
        assert by_id["chrome-devtools"]["command"] == [
            "npx",
            "-y",
            "chrome-devtools-mcp@latest",
        ]

    def test_get_merges_manager_status(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, app = mcp_client
        app.state.mcp = FakeManager(
            [{"id": "context7", "status": "running", "error": None}]
        )
        body = client.get("/api/mcp").json()
        assert body["installed"] is True
        by_id = {c["id"]: c for c in body["connectors"]}
        assert by_id["context7"]["status"] == "running"
        assert by_id["chrome-devtools"]["status"] == "stopped"

    def test_get_never_leaks_external_secrets(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, _ = mcp_client
        response = client.get("/api/mcp")
        assert "ctx7sk-secret-value" not in response.text
        assert "CONTEXT7_API_KEY" not in response.text

    def test_post_writes_enable_back_and_only_the_cap_to_env(self, mcp_client, tmp_path):
        import os

        from dotenv import dotenv_values

        from src import config as config_module

        path = self._externals_file(tmp_path)
        client, _ = mcp_client
        response = client.post(
            "/api/mcp", json={"enabled": ["context7"], "maxIterations": 9}
        )
        assert response.status_code == 200
        # The enable flag is the only activation source: no MCP_ENABLED is
        # ever written back to .env or the environment.
        assert "MCP_ENABLED" not in os.environ
        file_values = dotenv_values(config_module.ENV_PATH)
        assert "MCP_ENABLED" not in file_values
        assert file_values["MAX_TOOL_ITERATION"] == "9"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["context7"]["enable"] is True
        assert data["chrome-devtools"]["enable"] is False
        assert data["context7"]["url"] == "https://mcp.context7.com/mcp"  # preserved
        by_id = {c["id"]: c for c in response.json()["connectors"]}
        assert by_id["context7"]["active"] is True

    def test_post_enabling_only_one_connector_disables_the_other(self, mcp_client, tmp_path):
        path = self._externals_file(tmp_path)
        client, _ = mcp_client
        client.post("/api/mcp", json={"enabled": ["chrome-devtools"]})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["chrome-devtools"]["enable"] is True
        assert data["context7"]["enable"] is False

    def test_post_unknown_connector_returns_422(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, _ = mcp_client
        response = client.post("/api/mcp", json={"enabled": ["ghost"]})
        assert response.status_code == 422
        assert "ghost" in response.text

    def test_post_invalid_max_iterations_returns_422(self, mcp_client):
        client, _ = mcp_client
        assert client.post("/api/mcp", json={"maxIterations": 0}).status_code == 422
        assert client.post("/api/mcp", json={"maxIterations": 9999}).status_code == 422

    def test_post_empty_list_deactivates_all(self, mcp_client, tmp_path):
        path = self._externals_file(tmp_path)
        client, _ = mcp_client
        response = client.post("/api/mcp", json={"enabled": [], "maxIterations": 5})
        assert response.status_code == 200
        assert not [c for c in response.json()["connectors"] if c["active"]]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert not any(entry["enable"] for entry in data.values())

    def test_corrupt_file_surfaces_externals_error_and_survives_post(self, mcp_client, tmp_path):
        path = tmp_path / "mcp_connectors" / "externals.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{oops", encoding="utf-8")
        client, _ = mcp_client
        body = client.get("/api/mcp").json()
        assert "not readable JSON" in body["externalsError"]
        response = client.post("/api/mcp", json={"enabled": [], "maxIterations": 9})
        assert response.status_code == 200
        assert "not readable JSON" in response.json()["externalsError"]
        assert path.read_text(encoding="utf-8") == "{oops"

    def test_enable_start_and_disable_stop_are_reconciled(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, app = mcp_client
        hub = RecordingManager(
            {"context7": "running", "chrome-devtools": "stopped"}
        )
        app.state.mcp = hub
        client.post(
            "/api/mcp",
            json={"enabled": ["context7", "chrome-devtools"]},
        )
        assert hub.started == ["chrome-devtools"]  # newly enabled
        assert hub.stopped == []

        client.post("/api/mcp", json={"enabled": []})
        # Both go down: context7 was running, chrome starting.
        assert hub.stopped == ["chrome-devtools", "context7"]
        assert hub.started == ["chrome-devtools"]  # unchanged set not restarted

    def test_restart_rereads_the_declaration_file(
        self, mcp_client, tmp_path, monkeypatch
    ):
        self._externals_file(tmp_path)
        client, app = mcp_client
        # The hub under test is a fake, so connecting is allowed here: the
        # autouse switch would otherwise blank the scan (tested separately).
        monkeypatch.delenv("BENCHMARKIT_DISABLE_MCP", raising=False)
        hub = RecordingManager({"context7": "running"})
        app.state.mcp = hub

        response = client.post("/api/mcp/restart")

        assert response.status_code == 200
        body = response.json()
        assert [c["id"] for c in body["connectors"]] == [
            "chrome-devtools",
            "context7",
        ]
        assert body["externalsError"] is None
        assert len(hub.restarts) == 1
        # The hub is fed the fresh scan, in the payload's own (id-sorted) order.
        assert [c["id"] for c in hub.restarts[0]] == ["chrome-devtools", "context7"]

    def test_restart_honours_the_disable_switch(self, mcp_client, tmp_path):
        # BENCHMARKIT_DISABLE_MCP keeps the endpoints alive but connects
        # nothing — the same rule as the app lifespan (tests rely on it).
        self._externals_file(tmp_path)
        client, app = mcp_client
        hub = RecordingManager({})
        app.state.mcp = hub
        assert client.post("/api/mcp/restart").status_code == 200
        assert hub.restarts == [[]]

    def test_restart_without_a_hub_returns_503(self, mcp_client):
        client, app = mcp_client
        app.state.mcp = None
        assert client.post("/api/mcp/restart").status_code == 503

    def test_restart_during_a_run_returns_409(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, app = mcp_client
        hub = RecordingManager({"context7": "running"})
        app.state.mcp = hub
        app.state.run_state.status = "running"
        response = client.post("/api/mcp/restart")
        assert response.status_code == 409
        assert hub.restarts == []
        app.state.run_state.status = "idle"

    def test_restart_during_a_streaming_chat_returns_409(self, mcp_client, tmp_path):
        # A hub restart closes the client sessions a tool call is waiting on,
        # so it must not race an in-flight chat reply.
        self._externals_file(tmp_path)
        client, app = mcp_client
        hub = RecordingManager({"context7": "running"})
        app.state.mcp = hub
        guard = object()
        routes._CHAT_TASKS.add(guard)
        try:
            response = client.post("/api/mcp/restart")
        finally:
            routes._CHAT_TASKS.discard(guard)
        assert response.status_code == 409
        assert hub.restarts == []
        # And the guard released: the next restart goes through.
        assert client.post("/api/mcp/restart").status_code == 200

    def test_chat_uses_the_enabled_declaration_entries(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        assert routes._active_connector_ids() == ["context7"]


# ---------------------------------------------------------------------------
# stream_chat tool loop
# ---------------------------------------------------------------------------


def content_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def tool_call_chunk(call_id, name, arguments, index=0):
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                }
            }
        ]
    }


def usage_chunk(completion=None, prompt=None, total=None):
    usage: Dict[str, int] = {}
    if completion is not None:
        usage["completion_tokens"] = completion
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if total is not None:
        usage["total_tokens"] = total
    return {"choices": [], "usage": usage}


class TestChatToolLoop:
    # Declared here so the fake `completion` below can record the tool
    # definitions it was handed (also satisfies the type checker).
    tools_arg: Optional[List[Dict[str, Any]]] = None

    def _run(self, monkeypatch, completions, executor=None, **kwargs):
        calls: List[List[Dict[str, Any]]] = []
        generators = iter(completions)

        def fake_completion(**call_kwargs):
            calls.append(call_kwargs["messages"])
            self.tools_arg = call_kwargs.get("tools")
            return iter(next(generators))

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        events: List[Dict[str, Any]] = []
        chat_core.stream_chat(
            [{"role": "user", "content": "run it"}],
            "model-a",
            "http://server:8080/v1",
            events.append,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "execute_command", "description": "",
                                 "parameters": {}},
                }
            ],
            tool_executor=executor or (lambda name, args: "42"),
            tool_servers={"execute_command": "console_mcp"},
            **kwargs,
        )
        return events, calls

    def test_tool_call_then_final_answer(self, monkeypatch):
        seen_args = {}

        def executor(name, args):
            seen_args["call"] = (name, args)
            return "42"

        events, calls = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", '{"command": "ls"}'),
                 usage_chunk(completion=8)],
                [content_chunk("The answer is 42"), usage_chunk(completion=5)],
            ],
            executor=executor,
        )
        names = [e["event"] for e in events]
        # `usage` closes the round's *generation*, so it precedes the tool
        # execution: the badges tick as soon as generation stops.
        assert names == ["usage", "tool_call", "tool_result", "token", "done"]
        assert seen_args["call"] == ("execute_command", {"command": "ls"})
        tool_call = events[1]
        assert tool_call["tool"] == "execute_command"
        assert tool_call["server"] == "console_mcp"
        assert tool_call["iteration"] == 1
        assert tool_call["maxIterations"] == 25
        assert events[2]["ok"] is True
        # Cumulative usage of the completed round, before the final answer.
        assert events[0]["tokens"] == 8
        # usage.contextTokens: the just-ended round's prompt + completion
        # (its total_tokens, char-estimated here since the stub reports
        # completion only) — the round's own context occupancy figure.
        assert events[0]["contextTokens"] >= 8
        assert events[0]["contextEstimated"] is True
        # done carries the cumulative decode window for an accurate tok/s.
        assert events[-1]["decode"] >= 0
        # Second request carries the assistant tool_calls + tool result turn.
        second = calls[1]
        assistant = [m for m in second if m.get("tool_calls")]
        assert assistant and assistant[0]["tool_calls"][0]["function"]["name"] == (
            "execute_command"
        )
        tool_turn = [m for m in second if m.get("role") == "tool"]
        assert tool_turn[0]["content"] == "42"
        assert events[-1]["tokens"] == 13  # summed across iterations

    def test_split_tool_call_deltas_reassembled(self, monkeypatch):
        events, _ = self._run(
            monkeypatch,
            [
                [
                    tool_call_chunk("c1", "execute", "", index=0),
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "_command",
                                                "arguments": '{"a": ',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    tool_call_chunk(None, None, '1}', index=0),
                ],
                [content_chunk("done")],
            ],
            executor=lambda name, args: "x",
        )
        call = next(e for e in events if e["event"] == "tool_call")
        assert call["tool"] == "execute_command"
        assert call["id"] == "c1"

    def test_invalid_tool_args_feedback(self, monkeypatch):
        events, calls = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{oops")],
                [content_chunk("fixed")],
            ],
        )
        result = next(e for e in events if e["event"] == "tool_result")
        assert result["ok"] is False
        tool_turn = [m for m in calls[1] if m.get("role") == "tool"][0]
        assert "not valid JSON" in tool_turn["content"]

    def test_executor_exception_becomes_tool_result_error(self, monkeypatch):
        def boom(name, args):
            raise RuntimeError("connector down")

        events, _ = self._run(
            monkeypatch,
            [[tool_call_chunk("c1", "execute_command", "{}")], [content_chunk("ok")]],
            executor=boom,
        )
        result = next(e for e in events if e["event"] == "tool_result")
        assert result["ok"] is False
        assert "connector down" in result["preview"]

    def test_iteration_and_usage_events_across_rounds(self, monkeypatch):
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(completion=8)],
                [tool_call_chunk("c2", "execute_command", "{}"),
                 usage_chunk(completion=5)],
                [content_chunk("done")],
            ],
            max_tool_iterations=10,
        )
        names = [e["event"] for e in events]
        assert names == [
            "usage", "tool_call", "tool_result",
            "usage", "tool_call", "tool_result",
            "token", "done",
        ]
        calls = [e for e in events if e["event"] == "tool_call"]
        assert [e["iteration"] for e in calls] == [1, 2]
        assert all(e["maxIterations"] == 10 for e in calls)
        usages = [e for e in events if e["event"] == "usage"]
        assert [e["tokens"] for e in usages] == [8, 13]  # cumulative
        assert all(e["decode"] >= 0 for e in usages)
        # Occupancy is NOT a cumulative sum: each usage event carries its
        # OWN round's prompt + completion, never a running total — and is
        # char-estimated here, since the stub reports completion only.
        contexts = [e["contextTokens"] for e in usages]
        assert all(e["contextEstimated"] is True for e in usages)  # no totals
        # Each figure covers its OWN round (round 1: 8 generated, round 2:
        # 5 — not the cumulative 13), plus that round's estimated prompt.
        assert all(c > t for c, t in zip(contexts, [8, 5]))

    def test_context_tokens_is_last_round_not_cumulative(self, monkeypatch):
        # Both rounds report full usage: contextTokens is ONLY the final
        # round's total — the figure comparable to the server's context
        # window, never the sum of the rounds' (re-billed) totals.
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(prompt=100, completion=8, total=108)],
                [content_chunk("done"),
                 usage_chunk(prompt=120, completion=5, total=125)],
            ],
        )
        usage_event = next(e for e in events if e["event"] == "usage")
        assert usage_event["contextTokens"] == 108
        assert usage_event["contextEstimated"] is False
        done = events[-1]
        assert done["contextTokens"] == 125
        assert done["contextEstimated"] is False

    def test_iteration_cap_stops_with_note(self, monkeypatch):
        # Loose element type: the rounds mix tool-call and content chunk shapes.
        forever: List[List[Any]] = [
            [tool_call_chunk(f"c{i}", "execute_command", "{}")] for i in range(5)
        ]
        forever.append([content_chunk("never reached")])
        events, calls = self._run(
            monkeypatch,
            forever,
            max_tool_iterations=2,
        )
        # Two iterations of one tool call each, then the cap note.
        tool_calls = [e for e in events if e["event"] == "tool_call"]
        assert len(tool_calls) == 2
        assert tool_calls[-1]["iteration"] == 2  # gauge reads max on the cap round
        assert len(calls) == 2
        note = next(e for e in events if e["event"] == "token")
        assert "tool iterations reached" in note["text"]
        assert events[-1]["event"] == "done"

    def test_cancel_between_iterations_stops_silently(self, monkeypatch):
        cancel = threading.Event()

        def executor(name, args):
            cancel.set()  # user stopped while the tool was running
            return "42"

        events, calls = self._run(
            monkeypatch,
            [[tool_call_chunk("c1", "execute_command", "{}")], [content_chunk("x")]],
            executor=executor,
            cancel_event=cancel,
        )
        assert all(
            e["event"] in ("usage", "tool_call", "tool_result") for e in events
        )
        assert "done" not in [e["event"] for e in events]
        assert len(calls) == 1

    def test_context_is_round_total_with_full_usage(self, monkeypatch):
        # Each round reports usage with a total: its occupancy figure is
        # exactly that round's total (prompt re-billed every round, because
        # each tool round re-sends the whole history). Round totals here:
        # 18, then 25, then 25 — never the 68 sum across rounds.
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(completion=8, prompt=10, total=18)],
                [tool_call_chunk("c2", "execute_command", "{}"),
                 usage_chunk(completion=5, prompt=20, total=25)],
                [content_chunk("done"), usage_chunk(completion=0, prompt=25,
                                                   total=25)],
            ],
        )
        usages = [e for e in events if e["event"] == "usage"]
        assert [e["contextTokens"] for e in usages] == [18, 25]
        assert all(e["contextEstimated"] is False for e in usages)
        done = events[-1]
        assert done["tokens"] == 13
        assert done["contextTokens"] == 25
        assert done["contextEstimated"] is False

    def test_tool_call_arguments_extend_decode_window(self, monkeypatch):
        # A pure tool-call round streams no content and no reasoning: the
        # argument deltas must still open/extend the decode window (their
        # tokens are in completion_tokens), and the cumulative done.decode
        # must sum the rounds' windows without any tool-execution time.
        clock = [
            0.0,   # t0
            1.0,   # first tool-call argument delta (round 1 opens)
            3.0,   # second argument delta (round 1 closes at 2.0s)
            20.0,  # first content delta of the final round (opens)
            22.0,  # second content delta (final round closes at 2.0s)
        ]
        monkeypatch.setattr("src.core.chat.time.monotonic", lambda: clock.pop(0))
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", '{"a"'),
                 tool_call_chunk("c1", None, ': 1}')],
                [content_chunk("He"), content_chunk("llo")],
            ],
            # The 17 s gap between t=3 and t=20 stands for the tool run +
            # re-prefill: the clock is never sampled there, so none of it
            # may leak into the decode window.
            executor=lambda name, args: "42",
        )
        assert clock == []
        done = events[-1]
        assert done["decode"] == 4.0  # (3-1) + (22-20), tool time excluded
        assert done["took"] == 22.0  # wall clock does include the tool run
        assert done["ttft"] == 1.0

    def test_reasoning_deltas_interleave_with_tool_frames(self, monkeypatch):
        # The Chat tab's left panel renders a chronological timeline: the
        # panel relies on reasoning_delta frames arriving between the rounds'
        # tool_call frames, in stream order.
        events, _ = self._run(
            monkeypatch,
            [
                [{"choices": [{"delta": {"reasoning_content": "plan"}}]},
                 tool_call_chunk("c1", "execute_command", "{}")],
                [{"choices": [{"delta": {"reasoning_content": "again"}}]},
                 tool_call_chunk("c2", "execute_command", "{}")],
                [content_chunk("ok")],
            ],
        )
        names = [e["event"] for e in events]
        assert names == [
            "reasoning",            # one-shot flag (first thinking delta)
            "reasoning_delta",      # round 1 flow
            "usage",
            "tool_call", "tool_result",
            "reasoning_delta",      # round 2 flow, after the tool round
            "usage",
            "tool_call", "tool_result",
            "token", "done",
        ]
        texts = [e["text"] for e in events if e["event"] == "reasoning_delta"]
        assert texts == ["plan", "again"]

    def test_usage_precedes_tool_execution(self, monkeypatch):
        # The usage frame is emitted when the round's generation ends, so a
        # slow tool cannot freeze the token badge for its whole execution.
        order = []

        def executor(name, args):
            order.append("execute")
            return "42"

        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}")],
                [content_chunk("ok")],
            ],
            executor=executor,
        )
        names = [e["event"] for e in events]
        assert names.index("usage") < names.index("tool_result")

    def test_multi_round_usage_tokens_summed_without_completion(self, monkeypatch):
        # Servers omitting completion_tokens: the per-round (total - prompt)
        # deltas must add up across rounds — the last round's usage alone
        # would drop the tool rounds' tokens.
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(prompt=10, total=18)],
                [tool_call_chunk("c2", "execute_command", "{}"),
                 usage_chunk(prompt=20, total=25)],
                [content_chunk("done"), usage_chunk(prompt=27, total=32)],
            ],
        )
        done = events[-1]
        assert done["tokens"] == 8 + 5 + 5  # per-round deltas summed
        assert done["tokensEstimated"] is False

    def test_char_estimate_covers_tool_arguments_without_usage(self, monkeypatch):
        # No usage anywhere: the fallback estimate counts answer text AND
        # tool-call argument text (16 + 16 chars here -> 8 tokens).
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "x" * 16)],
                [content_chunk("y" * 16)],
            ],
        )
        done = events[-1]
        assert done["tokens"] == 8
        assert done["tokensEstimated"] is True

    def test_tool_notice_appended_to_system_prompt(self, monkeypatch):
        events, calls = self._run(monkeypatch, [[content_chunk("hi")]])
        assert chat_core.TOOL_NOTICE.strip() in calls[0][0]["content"]
        assert calls[0][0]["role"] == "system"

    def test_tool_notice_skips_prompt_that_already_mentions_tools(self, monkeypatch):
        events, calls = self._run(
            monkeypatch,
            [[content_chunk("hi")]],
            sysprompt="You may use your tools freely.",
        )
        assert calls[0][0]["content"] == "You may use your tools freely."

    def test_no_tools_keeps_single_shot_shape(self, monkeypatch):
        monkeypatch.setattr(
            "src.core.chat.completion",
            lambda **kwargs: iter([content_chunk("hi")])
            if "tools" not in kwargs
            else iter([]),
        )
        events: List[Dict[str, Any]] = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s", events.append
        )
        assert [e["event"] for e in events] == ["token", "done"]


# ---------------------------------------------------------------------------
# /api/chat wiring
# ---------------------------------------------------------------------------


class TestChatMcpWiring:
    def test_active_connectors_pass_tools_to_stream_chat(
        self, mcp_client, tmp_path, monkeypatch
    ):
        client, app = mcp_client
        _write_externals(
            tmp_path / "mcp_connectors" / "externals.json",
            {
                "context7": {
                    "type": "remote",
                    "enable": True,
                    "url": "https://mcp.context7.com/mcp",
                }
            },
        )

        tools = [
            {"type": "function", "function": {"name": "execute_command",
                                              "description": "", "parameters": {}}}
        ]

        class Hub(FakeManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return tools, {"execute_command": "context7"}

            def call_tool_sync(self, cid, name, args):
                return f"{cid}:{name}"

        hub = Hub()
        app.state.mcp = hub

        captured: Dict[str, Any] = {}

        def fake_stream(messages, model, server_url, emit, sampling=None,
                        cancel_event=None, pause_event=None, api_key="none", sysprompt=None,
                        tools=None, tool_executor=None, tool_servers=None,
                        max_tool_iterations=25, agent_instructions=None):
            captured["tools"] = tools
            captured["tool_servers"] = tool_servers
            captured["max"] = max_tool_iterations
            if tool_executor:
                captured["exec"] = tool_executor("execute_command", {})
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake_stream)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert hub.ids == ["context7"]
        assert captured["tools"] == tools
        assert captured["tool_servers"] == {"execute_command": "context7"}
        assert captured["exec"] == "context7:execute_command"

    def test_nothing_enabled_keeps_call_shape(self, mcp_client, monkeypatch):
        client, app = mcp_client
        monkeypatch.setattr(app.state, "mcp", None)
        captured = {}

        def fake_stream(messages, model, server_url, emit, sampling=None,
                        cancel_event=None, pause_event=None, api_key="none", sysprompt=None,
                        agent_instructions=None):
            captured["ok"] = True
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake_stream)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert captured["ok"] is True
