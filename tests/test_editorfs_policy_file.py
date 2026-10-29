"""Tests for pol-04 — level B: read, validate and cache the workspace's policy file.

Everything runs against a ``tmp_path`` workspace whose
``.nojae/read-write-policy.json`` the test writes itself, so the file, the
reserved folder and the cache key are all real paths on disk — and nothing is
wired to a surface yet (pol-06 does the wiring), so the assertions are on the
loader's three accessors and on ``mode_for``, not on HTTP.

The two module boundaries the task draws are both pinned here, because both are
the kind that "gets consolidated" later:

* the **parser** lives in ``mcp_connectors/`` and is stdlib-only (pol-08's two
  stdio connector processes have to reach it);
* the **cache and the level-A notes** live in ``src/core/editorfs/``, which is
  the only thing in the app that opens ``.nojae`` at all.
"""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from mcp_connectors import policy_file as parser
from mcp_connectors import policy_shared as shared

from src.core.editorfs import builtin_policy, constants, policy_file, policy_modes


def policy_json(
    # ``List[Any]`` and not ``List[Dict[str, Any]]`` on purpose: half of the
    # malformed-file suite hands this helper entries that are not objects at all,
    # which is exactly what the loader is supposed to notice — the helper builds
    # the JSON, it does not vouch for the rules inside it.
    rules: List[Any],
    *,
    version: Optional[int] = constants.POLICY_VERSION,
    readme: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    document: Dict[str, Any] = {}
    if version is not None:
        document["version"] = version
    if readme is not None:
        document["readme"] = readme
    if extra:
        document.update(extra)
    document["policies"] = rules
    return json.dumps(document, indent=2)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch workspace on the shape pol-00's example file is written about."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (tmp_path / "a" / "b" / "__pycache__").mkdir(parents=True)
    (tmp_path / "a" / "b" / "__pycache__" / "c.pyc").write_text("x", encoding="utf-8")
    (tmp_path / "web" / "node_modules" / "react").mkdir(parents=True)
    (tmp_path / "web" / "node_modules" / "react" / "x.js").write_text("x", encoding="utf-8")
    (tmp_path / "web" / "src").mkdir(parents=True)
    (tmp_path / "web" / "src" / "main.js").write_text("x", encoding="utf-8")
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")

    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    policy_file.invalidate_policy_cache()
    yield tmp_path
    policy_file.invalidate_policy_cache()


def write_policy(root: Path, text: str) -> Path:
    folder = root / constants.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = folder / constants.POLICY_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def mode(rel: str) -> str:
    return policy_modes.mode_for(rel).mode


class TestAbsentFile:
    def test_no_rules_no_findings_and_nothing_blocks(self, workspace: Path):
        assert policy_file.workspace_rules() == ()
        assert policy_file.workspace_findings() == ()
        assert policy_file.policy_blocking_error() is None

    def test_the_workspace_root_itself_is_where_the_lookup_starts(self, workspace: Path):
        """``policy_path()`` is built from the *active* root, so a switch moves it."""
        assert policy_file.policy_path() == str(
            workspace / constants.POLICY_DIRNAME / constants.POLICY_FILENAME
        )


class TestTheExampleFile:
    """pol-04 §1's file, resolving as written."""

    EXAMPLE = [
        {"path": ".git", "mode": "hidden"},
        {"path": "**/__pycache__", "mode": "hidden", "reason": "bytecode"},
        {"path": "**/node_modules", "mode": "ro", "reason": "browse docs, edit nothing"},
    ]

    def test_each_of_the_three_rules_reaches_its_own_path(self, workspace: Path):
        write_policy(workspace, policy_json(self.EXAMPLE))
        assert mode(".git/HEAD") == shared.HIDDEN
        assert mode("a/b/__pycache__/c.pyc") == shared.HIDDEN
        assert mode("web/node_modules/react/x.js") == shared.READ_ONLY

    def test_a_path_no_rule_names_stays_rw(self, workspace: Path):
        write_policy(workspace, policy_json(self.EXAMPLE))
        assert mode("web/src/main.js") == shared.READ_WRITE
        assert policy_modes.mode_for("web/src/main.js").level is None

    def test_the_verdict_carries_the_level_the_pattern_and_the_reason(self, workspace: Path):
        write_policy(workspace, policy_json(self.EXAMPLE))
        verdict = policy_modes.mode_for("web/node_modules/react/x.js")
        assert verdict.level == shared.LEVEL_WORKSPACE
        assert verdict.pattern == "**/node_modules"
        assert verdict.reason == "browse docs, edit nothing"
        assert "browse docs, edit nothing" in policy_modes.refusal_message(
            verdict, for_write=True
        )

    def test_the_readme_is_accepted_and_never_reaches_a_rule(self, workspace: Path):
        write_policy(
            workspace,
            policy_json(
                self.EXAMPLE,
                readme=["Rules are matched in order; the last match wins."],
            ),
        )
        assert len(policy_file.workspace_rules()) == 3
        assert policy_file.workspace_findings() == ()

    def test_an_empty_policies_list_is_a_choice_not_a_problem(self, workspace: Path):
        write_policy(workspace, policy_json([]))
        assert policy_file.workspace_rules() == ()
        assert policy_file.workspace_findings() == ()
        assert policy_file.policy_blocking_error() is None
        assert mode(".git/HEAD") == shared.READ_WRITE


class TestLastMatchWins:
    """In both directions — the idiom the file is written in."""

    def test_a_later_rw_loosens_an_earlier_ro_of_the_same_path(self, workspace: Path):
        write_policy(workspace, policy_json([
            {"path": "notes.md", "mode": "ro"},
            {"path": "notes.md", "mode": "rw"},
        ]))
        verdict = policy_modes.mode_for("notes.md")
        assert verdict.mode == shared.READ_WRITE
        # The rule that answered is the one quoted, so the second line is the one
        # a refusal names.
        assert verdict.pattern == "notes.md"
        assert policy_file.workspace_rules()[1].index == 1

    def test_a_later_ro_tightens_an_earlier_rw_of_the_same_path(self, workspace: Path):
        write_policy(workspace, policy_json([
            {"path": "notes.md", "mode": "rw"},
            {"path": "notes.md", "mode": "ro"},
        ]))
        assert policy_modes.mode_for("notes.md").mode == shared.READ_ONLY

    def test_a_carve_out_below_a_folder_rule_works(self, workspace: Path):
        write_policy(workspace, policy_json([
            {"path": "**/node_modules", "mode": "ro"},
            {"path": "**/node_modules/**/README.md", "mode": "rw"},
        ]))
        (workspace / "web" / "node_modules" / "README.md").write_text("x", encoding="utf-8")
        assert mode("web/node_modules/README.md") == shared.READ_WRITE
        assert mode("web/node_modules/react/x.js") == shared.READ_ONLY

    def test_a_repeated_rule_is_a_note_naming_both_indices(self, workspace: Path):
        write_policy(workspace, policy_json([
            {"path": ".git", "mode": "hidden"},
            {"path": ".git", "mode": "hidden"},
        ]))
        notes = [f for f in policy_file.workspace_findings() if f.kind == shared.KIND_NOTE]
        assert len(notes) == 1
        assert notes[0].index == 1
        assert "policies[0]" in notes[0].message
        assert "policies[1]" in notes[0].message
        assert policy_file.policy_blocking_error() is None

    def test_a_carve_out_of_a_hidden_folder_is_a_note(self, workspace: Path):
        """A hidden folder is pruned, not walked — so the rule below it is dead."""
        write_policy(workspace, policy_json([
            {"path": ".git", "mode": "hidden"},
            {"path": ".git/HEAD", "mode": "rw"},
        ]))
        notes = [f for f in policy_file.workspace_findings() if f.kind == shared.KIND_NOTE]
        assert len(notes) == 1
        assert notes[0].index == 1
        assert "pruned" in notes[0].message
        assert mode(".git/HEAD") == shared.HIDDEN


class TestMalformedFile:
    """Every row of pol-04 §3 that lands in the file itself.

    Each one is checked twice, because D4 has two halves: the finding names the
    file and the rule, *and* the rules that were fine contribute nothing.
    """

    def check(self, workspace: Path, text: str, *fragments: str) -> shared.PolicyFinding:
        write_policy(workspace, text)
        findings = policy_file.workspace_findings()
        errors = [f for f in findings if f.kind == shared.KIND_ERROR]
        assert errors, text
        message = " ".join(f.message for f in errors)
        for fragment in fragments:
            assert fragment in message, (text, message)
        assert policy_file.workspace_rules() == ()
        return errors[0]

    def test_malformed_json_names_the_file_and_the_line_and_column(self, workspace: Path):
        finding = self.check(
            workspace, '{"version": 1, "policies": [', "read-write-policy.json",
            "line", "column",
        )
        assert finding.index is None

    def test_a_top_level_that_is_not_an_object_is_an_error(self, workspace: Path):
        self.check(workspace, '[".git"]', "read-write-policy.json", "not an object")

    def test_an_unknown_top_level_key_names_the_key(self, workspace: Path):
        self.check(
            workspace,
            policy_json([{"path": ".git", "mode": "hidden"}], extra={"policyies": []}),
            "policyies",
        )

    @pytest.mark.parametrize("version,fragment", [
        (2, "is 2"),
        ("1", "not a number"),
        (True, "not a number"),
    ])
    def test_a_version_that_is_not_the_supported_one_says_both(
        self, workspace: Path, version: Any, fragment: str
    ):
        self.check(
            workspace,
            policy_json([{"path": ".git", "mode": "hidden"}], version=version),
            fragment,
            str(constants.POLICY_VERSION),
        )

    def test_a_missing_version_is_an_error(self, workspace: Path):
        self.check(
            workspace,
            json.dumps({"policies": [{"path": ".git", "mode": "hidden"}]}),
            "version",
        )

    @pytest.mark.parametrize("readme", [42, ["a", 4], {"a": 1}])
    def test_a_readme_that_cannot_be_shown_as_text(self, workspace: Path, readme: Any):
        self.check(
            workspace,
            policy_json([{"path": ".git", "mode": "hidden"}], readme=readme),
            "readme",
        )

    def test_policies_that_is_not_a_list(self, workspace: Path):
        self.check(workspace, '{"version": 1, "policies": {"a": 1}}', "policies", "not a list")

    def test_a_missing_policies_list(self, workspace: Path):
        self.check(workspace, '{"version": 1}', "policies")

    def test_a_rule_that_is_not_an_object_names_its_index(self, workspace: Path):
        self.check(workspace, policy_json([".git"]), "policies[0]")

    def test_a_rule_with_no_path_names_its_index(self, workspace: Path):
        self.check(workspace, policy_json([{"mode": "hidden"}]), "policies[0]", "path")

    def test_a_path_of_the_wrong_type(self, workspace: Path):
        self.check(workspace, policy_json([{"path": 7, "mode": "hidden"}]), "policies[0]", "path")

    def test_a_mode_that_is_not_spelled_right_names_the_three(
        self, workspace: Path
    ):
        finding = self.check(
            workspace,
            policy_json([{"path": ".git", "mode": "readonly"}]),
            "policies[0]", "hidden", "ro", "rw",
        )
        assert finding.index == 0

    @pytest.mark.parametrize("mode_value", ["HIDDEN", "Read", "Hidden", ["ro"], {"m": "ro"}])
    def test_a_mode_is_case_sensitive_and_must_be_a_string(
        self, workspace: Path, mode_value: Any
    ):
        self.check(
            workspace, policy_json([{"path": ".git", "mode": mode_value}]), "policies[0]"
        )

    def test_an_unknown_field_in_a_rule(self, workspace: Path):
        self.check(
            workspace,
            policy_json([{"path": ".git", "mode": "hidden", "reasons": "x"}]),
            "policies[0]", "reasons",
        )

    def test_a_reason_of_the_wrong_type(self, workspace: Path):
        self.check(
            workspace, policy_json([{"path": ".git", "mode": "hidden", "reason": 3}]),
            "policies[0]", "reason",
        )

    @pytest.mark.parametrize("pattern", ["", "/abs", "~/x", "trailing/", "a//b", "../out", "a/./b", "**"])
    def test_a_pattern_the_matcher_refuses_quotes_its_reason(
        self, workspace: Path, pattern: str
    ):
        """The *why* comes from pol-02 — one vocabulary for both halves.

        The empty string is the exception: a blank ``path`` is a *field* problem
        and is reported as one, before the pattern is ever read.
        """
        from mcp_connectors import policy_glob

        if not pattern:
            self.check(workspace, policy_json([{"path": pattern, "mode": "hidden"}]),
                       "policies[0]", "path")
            return
        reason = policy_glob.pattern_error(pattern)
        assert reason is not None
        finding = self.check(
            workspace, policy_json([{"path": pattern, "mode": "hidden"}]), "policies[0]"
        )
        assert reason in finding.message

    def test_the_findings_are_quoted_into_the_blocking_error(self, workspace: Path):
        self.check(workspace, policy_json([".git"]), "policies[0]")
        error = policy_file.policy_blocking_error()
        assert error is not None
        assert error.status_code == 422
        assert "read-write-policy.json" in error.detail
        assert "policies[0]" in error.detail


class TestReservedFolder:
    def test_a_rule_naming_the_reserved_folder_is_an_error(self, workspace: Path):
        self_note = policy_json([{"path": ".nojae/x", "mode": "rw"}])
        write_policy(workspace, self_note)
        errors = [f for f in policy_file.workspace_findings() if f.kind == shared.KIND_ERROR]
        assert errors
        assert constants.POLICY_DIRNAME in errors[0].message

    def test_and_the_reservation_survives_its_own_file(self, workspace: Path):
        """The rule does not apply, and ``.nojae/x`` is still hidden."""
        write_policy(workspace, policy_json([{"path": ".nojae/x", "mode": "rw"}]))
        assert mode(f"{constants.POLICY_DIRNAME}/x") == shared.HIDDEN
        assert policy_modes.mode_for(".nojae/x").level == shared.LEVEL_BUILTIN

    def test_the_reserved_name_is_patchable_for_this_check_too(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(constants, "POLICY_DIRNAME", ".policydir")
        policy_file.invalidate_policy_cache()
        write_policy(workspace, policy_json([{"path": ".policydir/x", "mode": "rw"}]))
        assert policy_file.workspace_findings()[0].kind == shared.KIND_ERROR
        assert mode(".policydir/x") == shared.HIDDEN


class TestUnreachablePattern:
    def test_a_typoed_prefix_is_an_error_with_a_sibling_suggestion(
        self, workspace: Path
    ):
        """``web/node_module`` next to a real ``web/node_modules``: say which one."""
        write_policy(workspace, policy_json([{"path": "web/node_module", "mode": "ro"}]))
        errors = [f for f in policy_file.workspace_findings() if f.kind == shared.KIND_ERROR]
        assert errors
        assert "node_modules" in errors[0].message
        assert "did you mean" in errors[0].message
        assert policy_file.policy_blocking_error() is not None

    def test_a_missing_prefix_with_no_near_name_is_not_a_typo(self, workspace: Path):
        """A rule about a folder that is simply not here is a legitimate rule.

        This is what lets pol-05's seed load at all: it names ``.hg`` and ``.svn``
        in every workspace, and treating absence as an error would refuse writes
        everywhere the app goes. Only a *near miss* is evidence of a typo.
        """
        write_policy(workspace, policy_json([
            {"path": "zzzthing", "mode": "ro"},
            {"path": ".hg", "mode": "hidden"},
            {"path": "src/deeply/missing", "mode": "ro"},
        ]))
        assert policy_file.workspace_findings() == ()
        assert len(policy_file.workspace_rules()) == 3
        assert policy_file.policy_blocking_error() is None

    def test_a_tool_directory_is_not_claimed_as_the_typo_of_another(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """``.svn`` beside a real ``.venv`` is not a misspelling of ``.venv``.

        difflib scores the two 0.67 — above its default cutoff — so a ratio alone
        cannot tell a typo from an unrelated short name. The guard matters
        because pol-05's seeded file writes ``.svn`` into every workspace it
        touches: without it, opening any Python project with a virtualenv would
        reject the whole policy as a typo the author never typed and refuse every
        write in it (D4). Distance, not resemblance, is the evidence of a typo.
        """
        root = tmp_path / "pyproject"
        (root / ".venv" / "lib").mkdir(parents=True)
        (root / "src").mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
        policy_file.invalidate_policy_cache()
        try:
            write_policy(root, policy_json([
                {"path": ".svn", "mode": "hidden"},
                {"path": ".hg", "mode": "hidden"},
                {"path": ".venv", "mode": "ro"},
            ]))
            assert policy_file.workspace_findings() == ()
            assert len(policy_file.workspace_rules()) == 3
            assert policy_file.policy_blocking_error() is None
            # And one real typo in the same file is still caught, so the guard
            # above is a veto on unrelated names, not a switch-off.
            write_policy(root, policy_json([{"path": ".vnv", "mode": "hidden"}]))
            errors = [
                f for f in policy_file.workspace_findings() if f.kind == shared.KIND_ERROR
            ]
            assert errors and ".venv" in errors[0].message
        finally:
            policy_file.invalidate_policy_cache()

    def test_a_deep_missing_prefix_still_suggests_at_its_own_level(self, workspace: Path):
        """The suggestion comes from the directory the prefix sits in, not the root."""
        write_policy(workspace, policy_json([{"path": "web/scr", "mode": "ro"}]))
        errors = [f for f in policy_file.workspace_findings() if f.kind == shared.KIND_ERROR]
        assert "web/src" in errors[0].message

    def test_a_leading_wildcard_has_no_prefix_to_check(self, workspace: Path):
        """``**/node_modules`` and ``.env*`` name nothing literal: never reported."""
        write_policy(workspace, policy_json([
            {"path": "**/node_modules", "mode": "ro"},
            {"path": ".env*", "mode": "hidden"},
            {"path": "web/**/deep", "mode": "ro"},
        ]))
        assert policy_file.workspace_findings() == ()
        assert len(policy_file.workspace_rules()) == 3

    def test_a_broken_symlink_in_the_prefix_is_not_the_authors_typo(self, workspace: Path):
        link = workspace / "dangling"
        link.symlink_to(workspace / "not-there")
        write_policy(workspace, policy_json([{"path": "dangling/x", "mode": "ro"}]))
        # The link is a name the author can see in the folder, so the rule is not
        # a typo and the file loads: the check judges *names*, and a dangling
        # target is the filesystem's business, not the pattern's.
        assert policy_file.workspace_findings() == ()
        assert len(policy_file.workspace_rules()) == 1
        assert policy_file.policy_blocking_error() is None
        assert mode("web/src/main.js") == shared.READ_WRITE

    def test_a_name_close_to_the_dangling_link_is_still_a_typo(self, workspace: Path):
        """The leniency above is about the link, not a refusal to check."""
        (workspace / "dangling").symlink_to(workspace / "not-there")
        write_policy(workspace, policy_json([{"path": "danglin/x", "mode": "ro"}]))
        errors = [f for f in policy_file.workspace_findings() if f.kind == shared.KIND_ERROR]
        assert errors and "dangling" in errors[0].message
        assert policy_file.policy_blocking_error() is not None



class TestUnreadableFile:
    def test_a_folder_where_the_file_should_be_names_the_path_and_the_os_error(
        self, workspace: Path
    ):
        (workspace / constants.POLICY_DIRNAME).mkdir(exist_ok=True)
        (workspace / constants.POLICY_DIRNAME / constants.POLICY_FILENAME).mkdir()
        errors = [f for f in policy_file.workspace_findings() if f.kind == shared.KIND_ERROR]
        assert errors
        assert constants.POLICY_FILENAME in errors[0].message
        assert policy_file.policy_blocking_error() is not None

    def test_a_file_the_process_cannot_open_names_the_os_error(self, workspace: Path):
        """``EACCES`` is a finding the user can act on, not a silent "no rules"."""
        if os.geteuid() == 0:
            pytest.skip("root ignores the mode bits")
        path = write_policy(workspace, policy_json([{"path": "notes.md", "mode": "ro"}]))
        path.chmod(0o000)
        try:
            policy_file.invalidate_policy_cache()
            errors = [
                f for f in policy_file.workspace_findings() if f.kind == shared.KIND_ERROR
            ]
            assert errors
            assert constants.POLICY_FILENAME in errors[0].message
            assert "read" in errors[0].message
            assert policy_file.policy_blocking_error() is not None
        finally:
            path.chmod(0o600)

    def test_a_policy_folder_that_cannot_be_entered_is_not_an_error(
        self, workspace: Path
    ):
        """Absence of the file is a normal state; only reading it can fail."""
        assert (workspace / constants.POLICY_DIRNAME).exists() is False
        assert policy_file.workspace_findings() == ()


class TestLevelAShadowing:
    """The note half of the table: a user rule that cannot win (D2)."""

    @pytest.fixture
    def self_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[Path]:
        """A scratch directory that is *also* "the app", so level A is live.

        Both sides of :func:`builtin_policy.is_self_workspace` are read at call
        time, so moving ``constants.PROJECT_ROOT`` alongside the location makes a
        self workspace out of a temp folder — which is what lets this test write a
        policy file at all, instead of editing the real checkout.
        """
        (tmp_path / "logs").mkdir()
        (tmp_path / "notes.md").write_text("x\n", encoding="utf-8")
        monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        policy_file.invalidate_policy_cache()
        assert builtin_policy.is_self_workspace() is True
        yield tmp_path
        policy_file.invalidate_policy_cache()

    def test_a_rw_rule_for_an_app_hidden_path_is_a_note_not_an_error(
        self, self_workspace: Path
    ):
        write_policy(self_workspace, policy_json([{"path": ".env**", "mode": "rw"}]))
        findings = policy_file.workspace_findings()
        assert [f.kind for f in findings] == [shared.KIND_NOTE]
        assert "benchmarkit" in findings[0].message
        assert "policies[0]" in findings[0].message
        assert policy_file.policy_blocking_error() is None

    def test_the_effective_mode_is_still_hidden(self, self_workspace: Path):
        write_policy(self_workspace, policy_json([{"path": ".env", "mode": "rw"}]))
        verdict = policy_modes.mode_for(".env")
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_BUILTIN

    def test_a_corrupt_file_in_the_apps_own_workspace_changes_nothing_to_look_at(
        self, self_workspace: Path
    ):
        """pol-04 §7's acceptance, in the one workspace where level A is live."""
        write_policy(self_workspace, '{"version": 1, "policies": [ {"path" ')
        assert mode(".env") == shared.HIDDEN
        assert policy_modes.mode_for(".env").level == shared.LEVEL_BUILTIN
        error = policy_file.policy_blocking_error()
        assert error is not None and "line" in error.detail

    def test_a_workspace_rule_may_still_tighten_an_app_path(self, self_workspace: Path):
        (self_workspace / "logs" / "keep").mkdir(parents=True)
        write_policy(self_workspace, policy_json([
            {"path": "logs/keep", "mode": "rw"},
            {"path": "notes.md", "mode": "hidden"},
        ]))
        assert mode("notes.md") == shared.HIDDEN
        assert mode("logs/keep") == shared.HIDDEN  # the app rule still holds
        assert mode("logs") == shared.HIDDEN

    def test_no_note_is_made_in_a_foreign_workspace(self, workspace: Path):
        """Level A is dormant here, so nothing of the user's can lose to it."""
        (workspace / ".env").write_text("A=1\n", encoding="utf-8")
        write_policy(workspace, policy_json([{"path": ".env", "mode": "rw"},
                                             {"path": "notes.md", "mode": "hidden"}]))
        assert policy_file.workspace_findings() == ()
        assert len(policy_file.workspace_rules()) == 2
        assert mode(".env") == shared.READ_WRITE


class TestCache:
    def test_an_edit_at_the_same_size_with_a_new_mtime_is_seen(self, workspace: Path):
        """Same length, different rules — a size-only cache would miss it."""
        path = write_policy(workspace, policy_json([{"path": "notes.md", "mode": "ro"}]))
        assert mode("notes.md") == shared.READ_ONLY
        path.write_text(policy_json([{"path": "notes.md", "mode": "rw"}]), encoding="utf-8")
        assert path.stat().st_size == len(policy_json([{"path": "notes.md", "mode": "ro"}]))
        future = path.stat().st_atime_ns + 1_000_000_000
        os.utime(path, ns=(future, future))
        assert mode("notes.md") == shared.READ_WRITE

    def test_a_second_root_gets_its_own_rules_not_the_firsts(
        self, workspace: Path, tmp_path_factory
    ):
        """The regression the root in the cache key exists to prevent.

        The two files are made to **collide on the stamp** — ``"alpha"`` and
        ``"betab"`` are the same length, and the second file's ``mtime_ns`` is set
        to the first's — so a cache keyed on ``(mtime, size)`` alone has nothing
        left to tell them apart and answers the new workspace with the old one's
        rules. The two equal-length patterns are the whole point.
        """
        (workspace / "alpha").mkdir()
        first = write_policy(workspace, policy_json([{"path": "alpha", "mode": "hidden"}]))
        assert policy_file.workspace_rules()[0].pattern == "alpha"

        second: Path = tmp_path_factory.mktemp("other-root")
        (second / "betab").mkdir()
        stamp = (first.stat().st_mtime_ns, first.stat().st_size)
        os.environ["BENCHMARKIT_PATH_LOCATION"] = str(second)
        try:
            second_file = write_policy(
                second, policy_json([{"path": "betab", "mode": "hidden"}])
            )
            # ``ns=`` wants (atime_ns, mtime_ns); the two files already share a
            # size, so copying the first file's mtime completes the collision.
            os.utime(second_file, ns=(stamp[0], stamp[0]))
            assert (second_file.stat().st_mtime_ns, second_file.stat().st_size) == stamp
            assert policy_file.workspace_rules()[0].pattern == "betab"
            assert mode("betab/inner") == shared.HIDDEN
            assert mode("alpha/inner") == shared.READ_WRITE
        finally:
            os.environ.pop("BENCHMARKIT_PATH_LOCATION", None)

    def test_an_unchanged_file_is_read_once_per_workspace(self, workspace: Path):
        write_policy(workspace, policy_json([{"path": "notes.md", "mode": "ro"}]))
        reads: List[str] = []
        real = parser.load_policy_file

        def counting(path: str, **kwargs: Any) -> Tuple[Any, Any]:
            reads.append(path)
            return real(path, **kwargs)

        parser.load_policy_file = counting  # type: ignore[assignment]
        try:
            policy_file.invalidate_policy_cache()
            for _ in range(5):
                assert mode("notes.md") == shared.READ_ONLY
            assert len(reads) == 1
        finally:
            parser.load_policy_file = real  # type: ignore[assignment]


class TestModeForNeverRaisesAboutTheFile:
    """D4's first half: the read side keeps answering while the file is broken."""

    def test_a_broken_file_leaves_the_tree_and_the_read_on_level_a_only(
        self, workspace: Path
    ):
        # A file with one perfectly good rule and a syntax error: the rule is
        # dropped with the file, because a policy that half loaded protects
        # nothing — and the two halves below are pol-00 D4 asserted together.
        write_policy(
            workspace, '{"version": 1, "policies": [{"path": "web/src", "mode": "hidden"}'
        )
        assert mode("web/src/main.js") == shared.READ_WRITE
        assert mode("notes.md") == shared.READ_WRITE
        assert policy_file.policy_blocking_error() is not None
        # And the live tree still lists the workspace: the broken file closes
        # nothing that a user could use to go look at it (pol-06 makes the tree
        # answer from ``mode_for``, and this assertion is about it staying open).
        from src.core.editorfs import tree

        entries = tree.list_dir("")
        listed = {entry["name"] for entry in entries["entries"]}
        assert {"web", "a", "notes.md"} <= listed

    def test_the_blocking_error_names_the_file_the_user_has_to_fix(self, workspace: Path):
        write_policy(workspace, '{"version": 1, "policies": [')
        error = policy_file.policy_blocking_error()
        assert error is not None
        assert str(workspace / constants.POLICY_DIRNAME) in error.detail


class TestTheTwoHalvesStayApart:
    """The split pol-04 §2 draws, pinned so it survives the next refactor.

    The parser has to stay importable from a stdio connector process that cannot
    reach ``src`` (that is what pol-08 depends on), and the app-side half has to
    stay the only thing that opens ``.nojae`` — which is what makes pol-03's
    reservation and pol-05's seeding legal in the first place.
    """

    def test_the_shared_parser_imports_nothing_from_src(self):
        """A ``src`` import in ``policy_file`` would break pol-08 at startup."""
        # Parsed, not grepped: a docstring line that happens to start with
        # ``from src ...`` is prose and must not fail this, and a real import
        # hidden inside a function must not pass it.
        tree = ast.parse(Path(parser.__file__).read_text(encoding="utf-8"))
        offenders: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("src"):
                    offenders.append(module)
            elif isinstance(node, ast.Import):
                offenders.extend(
                    alias.name for alias in node.names if alias.name.startswith("src")
                )
        assert offenders == []

    def test_the_shared_parser_needs_no_workspace_to_be_active(self, tmp_path: Path):
        """Its only filesystem question is the one its caller hands it."""
        rules, findings = parser.load_policy_text(
            policy_json([{"path": "anything", "mode": "ro"}]),
            location=str(tmp_path / "policy.json"),
        )
        assert [rule.pattern for rule in rules] == ["anything"]
        assert findings == []

    def test_only_the_verdict_and_the_gate_ask_the_loader(self):
        """pol-06's shape: the file is consulted in two places, never on a surface.

        :mod:`editorfs.policy_modes` asks it for the rules and
        :mod:`editorfs.policy` asks it for the write block; the tree, the read,
        the mutations and git status ask *them*. A surface that went straight to
        the loader is a surface that can disagree with its neighbours about the
        same folder, which is the half-wired state this lot forbids.

        Checked on imports rather than on text, so a docstring naming a function
        is prose and does not pass or fail the rule.
        """
        here = Path(constants.__file__).parent

        def imports_loader(path: Path) -> bool:
            """``import x.policy_file`` or ``from . import policy_file``."""
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    if any(alias.name.endswith("policy_file") for alias in node.names):
                        return True
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").endswith("policy_file"):
                        return True
                    if any(alias.name == "policy_file" for alias in node.names):
                        return True
            return False

        assert sorted(
            path.name
            for path in here.glob("*.py")
            if path.name != "policy_file.py" and imports_loader(path)
        ) == ["policy.py", "policy_modes.py", "policy_seed.py"]
