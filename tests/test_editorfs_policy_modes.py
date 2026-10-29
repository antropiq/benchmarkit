"""Tests for the pol-01 verdict primitive: ``mode_for``, ``reserved_hit`` and
``refusal_message`` — pure, no HTTP, no app, no workspace on disk.

The two rule sources are still empty (pol-03 and pol-04 fill them), so the
combination tests feed rules through ``monkeypatch`` on the two ``list_*_rules``
functions — that is the seam the later tasks plug into, and patching it here is
what pins the *merge* semantics (strictest wins, ties to the later rule)
independently of where the rules eventually come from.
"""

import ast
from pathlib import Path
from typing import List, Tuple

import pytest

from mcp_connectors import policy_shared as shared

from src.core import editorfs
from src.core.editorfs import constants
from src.core.editorfs import policy_modes


@pytest.fixture
def no_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both sources empty, so every test decides the candidate list itself."""
    monkeypatch.setattr(policy_modes, "list_builtin_rules", lambda rel: [])
    monkeypatch.setattr(policy_modes, "list_workspace_rules", lambda rel: [])


def rules(monkeypatch: pytest.MonkeyPatch, level: str, *pairs: Tuple[str, str]) -> None:
    """Make one level's source return ``[(pattern, mode), ...]`` in file order."""
    def source(rel: str) -> List[shared.Rule]:
        return [
            shared.Rule(mode=mode, pattern=pattern, reason=None, index=index, level=level)
            for index, (pattern, mode) in enumerate(pairs)
        ]

    name = "list_builtin_rules" if level == shared.LEVEL_BUILTIN else "list_workspace_rules"
    monkeypatch.setattr(policy_modes, name, source)


class TestModeRank:
    def test_modes_are_ordered_rw_ro_hidden(self):
        assert (
            shared.MODE_RANK[shared.READ_WRITE]
            < shared.MODE_RANK[shared.READ_ONLY]
            < shared.MODE_RANK[shared.HIDDEN]
        )

    def test_every_mode_is_ranked_and_named(self):
        assert set(shared.MODE_RANK) == set(shared.MODES)
        assert set(shared.MODE_LABELS) == set(shared.MODES)
        assert set(shared.LEVEL_LABELS) == set(shared.LEVELS)


class TestModeFor:
    def test_unmatched_path_is_read_write_with_no_level(self, no_rules):
        verdict = policy_modes.mode_for("src/config.py")
        assert verdict == shared.Verdict(shared.READ_WRITE, None, None, None)
        assert verdict.level is None

    def test_a_match_carries_its_mode_level_and_pattern_verbatim(
        self, monkeypatch, no_rules
    ):
        rules(monkeypatch, shared.LEVEL_WORKSPACE, ("**/__pycache__", shared.HIDDEN))
        verdict = policy_modes.mode_for("anything")
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_WORKSPACE
        assert verdict.pattern == "**/__pycache__"

    def test_the_tie_goes_to_the_later_index(self, monkeypatch, no_rules):
        """Same strictness twice: the later rule wins, and it is the one quoted.

        Both stubbed sources ignore ``rel`` (matching is pol-02's business), so
        every rule is a candidate for every path — which is exactly the tie the
        merge has to break, and why a carve-out of the same mode replaces the
        rule it sits under.
        """
        rules(monkeypatch, shared.LEVEL_WORKSPACE,
              ("earlier", shared.READ_ONLY), ("later", shared.READ_ONLY))
        verdict = policy_modes.mode_for("anything")
        assert verdict.mode == shared.READ_ONLY
        assert verdict.pattern == "later"

    def test_the_last_match_amends_the_one_above_it(self, monkeypatch, no_rules):
        """``ro`` then ``rw`` -> ``rw``; ``rw`` then ``ro`` -> ``ro``.

        File order, inside one level: the last match wins, which is what lets a
        later line carve an exception out of the folder rule it sits under
        (pol-00's ``node_modules`` / ``README.md`` example).
        """
        rules(monkeypatch, shared.LEVEL_WORKSPACE,
              ("d", shared.READ_ONLY), ("d", shared.READ_WRITE))
        assert policy_modes.mode_for("d").mode == shared.READ_WRITE
        rules(monkeypatch, shared.LEVEL_WORKSPACE,
              ("d", shared.READ_WRITE), ("d", shared.READ_ONLY))
        assert policy_modes.mode_for("d").mode == shared.READ_ONLY

    def test_an_earlier_hidden_beats_a_later_rw(self, monkeypatch, no_rules):
        """The one order rule that does not bend: ``hidden`` is never loosened.

        A hidden folder is pruned rather than walked, so nothing below it is
        reachable "not even by carve-out" — and the same word applies to the
        folder's own name.
        """
        rules(monkeypatch, shared.LEVEL_WORKSPACE,
              ("d", shared.HIDDEN), ("d", shared.READ_WRITE))
        verdict = policy_modes.mode_for("d")
        assert verdict.mode == shared.HIDDEN
        assert verdict.pattern == "d"

    def test_a_looser_workspace_rule_cannot_unhide_an_app_rule(self, monkeypatch, no_rules):
        rules(monkeypatch, shared.LEVEL_WORKSPACE, (".env", shared.READ_WRITE))
        rules(monkeypatch, shared.LEVEL_BUILTIN, (".env**", shared.HIDDEN))
        verdict = policy_modes.mode_for(".env")
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_BUILTIN
        assert verdict.pattern == ".env**"

    def test_a_stricter_workspace_rule_beats_a_looser_app_rule(self, monkeypatch, no_rules):
        rules(monkeypatch, shared.LEVEL_WORKSPACE, ("build", shared.HIDDEN))
        rules(monkeypatch, shared.LEVEL_BUILTIN, ("build", shared.READ_WRITE))
        verdict = policy_modes.mode_for("build")
        assert verdict.mode == shared.HIDDEN
        assert verdict.level == shared.LEVEL_WORKSPACE

    def test_a_looser_workspace_rule_cannot_lift_an_app_read_only_rule(
        self, monkeypatch, no_rules
    ):
        """D2 in its general form, not only where ``hidden`` is involved."""
        rules(monkeypatch, shared.LEVEL_WORKSPACE, ("resources", shared.READ_WRITE))
        rules(monkeypatch, shared.LEVEL_BUILTIN, ("resources", shared.READ_ONLY))
        verdict = policy_modes.mode_for("resources")
        assert verdict.mode == shared.READ_ONLY
        assert verdict.level == shared.LEVEL_BUILTIN

    def test_two_levels_agreeing_on_a_mode_quote_the_app_rule(self, monkeypatch, no_rules):
        """Equal strictness across levels: the message names the rule that holds.

        Both levels say ``hidden`` for ``.env``, so either is true — but a
        refusal quoting only the user's own line would send them to a file whose
        deletion changes nothing, which is the lie pol-07's note exists to avoid.
        """
        rules(monkeypatch, shared.LEVEL_WORKSPACE, (".env", shared.HIDDEN))
        rules(monkeypatch, shared.LEVEL_BUILTIN, (".env**", shared.HIDDEN))
        verdict = policy_modes.mode_for(".env")
        assert verdict.level == shared.LEVEL_BUILTIN
        assert verdict.pattern == ".env**"


class TestReservedHit:
    @pytest.mark.parametrize(
        "rel",
        [".nojae", ".nojae/read-write-policy.json", "sub/.nojae/x", "a/b/.nojae/c/d"],
    )
    def test_the_reserved_component_is_reported_at_any_depth(self, rel):
        assert policy_modes.reserved_hit(rel) == constants.POLICY_DIRNAME

    @pytest.mark.parametrize("rel", [".nojaeer", "nojae", "x.nojae", ".noja", "src/.nojae.bak/y"])
    def test_a_name_only_similar_to_it_is_not_reserved(self, rel):
        assert policy_modes.reserved_hit(rel) is None

    def test_the_root_path_itself_is_not_reserved(self):
        assert policy_modes.reserved_hit("") is None

    def test_it_answers_without_touching_the_disk(self, tmp_path: Path):
        """The answer comes from the component list, so absence is irrelevant."""
        assert not (tmp_path / constants.POLICY_DIRNAME).exists()
        rel = f"{constants.POLICY_DIRNAME}/whatever"
        assert policy_modes.reserved_hit(rel) == constants.POLICY_DIRNAME

    def test_the_name_is_patchable_through_constants(self, monkeypatch: pytest.MonkeyPatch):
        """App-side tests move the lookup; the connectors keep the default."""
        monkeypatch.setattr(constants, "POLICY_DIRNAME", ".policydir")
        assert policy_modes.reserved_hit("sub/.policydir/x") == ".policydir"
        assert policy_modes.reserved_hit("sub/.nojae/x") is None


class TestRefusalMessage:
    def test_a_hidden_path_read_names_the_level_mode_and_pattern(self):
        verdict = shared.Verdict(shared.HIDDEN, shared.LEVEL_WORKSPACE, ".git", None)
        assert policy_modes.refusal_message(verdict, for_write=False) == (
            'Hidden by the workspace policy: ".git" (hidden) '
            "— it is not shown and not reachable"
        )

    def test_a_read_only_mutation_says_what_may_not_change(self):
        verdict = shared.Verdict(
            shared.READ_ONLY, shared.LEVEL_WORKSPACE, "web/node_modules", None
        )
        assert policy_modes.refusal_message(verdict, for_write=True) == (
            'Refused by the workspace policy: "web/node_modules" (read-only) '
            "— nothing under it may change"
        )

    def test_an_app_level_rule_says_benchmarkit_did_it(self):
        verdict = shared.Verdict(shared.HIDDEN, shared.LEVEL_BUILTIN, ".env**", None)
        assert policy_modes.refusal_message(verdict, for_write=True) == (
            "Refused by benchmarkit's own rules for this workspace: \".env**\" (hidden)"
        )

    def test_the_pattern_is_quoted_verbatim(self):
        verdict = shared.Verdict(shared.READ_ONLY, shared.LEVEL_WORKSPACE, "**/node_modules", None)
        assert '"**/node_modules"' in policy_modes.refusal_message(verdict, for_write=True)

    def test_a_reason_from_the_rule_replaces_the_stock_clause(self):
        verdict = shared.Verdict(
            shared.HIDDEN, shared.LEVEL_WORKSPACE, "**/__pycache__", "bytecode"
        )
        message = policy_modes.refusal_message(verdict, for_write=False)
        assert message.endswith("— bytecode")
        assert "not shown" not in message

    def test_every_message_names_its_level_and_its_mode(self):
        for level in shared.LEVELS:
            for mode in (shared.HIDDEN, shared.READ_ONLY):
                verdict = shared.Verdict(mode, level, "p", None)
                for write in (True, False):
                    message = policy_modes.refusal_message(verdict, for_write=write)
                    assert shared.LEVEL_LABELS[level] in message
                    assert f"({shared.MODE_LABELS[mode]})" in message


class TestTheVerdictIsTheOnlyRouteToTheRules:
    """pol-01's pin, inverted by pol-06 — which is what it was pinned *for*.

    The verdict is now wired to every surface, so what has to hold instead is
    that it is the **only** door: the tree, the read, the writes and git status
    import ``mode_for`` and nothing else that could answer about a path. A
    surface that reached the rules another way is a surface that can disagree
    with its neighbours, and that is the one state this lot must never ship.
    """

    def test_the_package_exports_the_verdict_and_no_old_predicate(self):
        for name in ("mode_for", "refusal_message", "Verdict", "MODES"):
            assert hasattr(editorfs, name), name
        # The three the verdict replaced stay gone: no second spelling of the
        # question, so no way to answer it inconsistently.
        for name in ("listable", "writable", "refusal"):
            assert not hasattr(editorfs, name), name

    def test_every_surface_reaches_the_rules_through_one_import(self):
        """An import line, not a mention: prose about the verdict is not wiring."""
        here = Path(editorfs.__file__).parent
        importers: List[str] = []
        for path in sorted(here.glob("*.py")):
            if path.name == "policy_modes.py":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and (
                    "policy_modes" in stripped
                ):
                    importers.append(path.name)
                    break
        # The gate — which asks the verdict for every path it resolves, so no
        # other module has to — the package that re-exports it, and the three
        # surfaces that ask directly (the tree's prune, the delete's subtree
        # walk, git status's one omission rule).
        assert importers == [
            "__init__.py", "gitstatus.py", "mutate.py", "policy.py", "tree.py",
        ]

    def test_no_surface_consults_the_connector_deny_list(self):
        """pol-06's other half: the rule-list import is gone from ``editorfs``.

        ``path_policy`` still guards the connectors' own surface (and
        ``core/batch.py`` still uses it for the batch lot), so the check is on
        this package's modules — the pane must answer from the policy, in one
        place, or the tree and the write gate drift apart again.
        """
        here = Path(editorfs.__file__).parent
        offenders: List[str] = []
        for path in here.glob("*.py"):
            parsed = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(parsed):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "path_policy"
                ):
                    offenders.append(path.name)
                elif isinstance(node, ast.Import):
                    offenders.extend(
                        alias.name for alias in node.names
                        if alias.name.endswith("path_policy")
                    )
        assert offenders == []
