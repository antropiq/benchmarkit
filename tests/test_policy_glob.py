"""Tests for pol-02, the path-aware matcher — pure: no app, no HTTP, no rules.

The table in the task is the contract, so it is parametrised here rather than
paraphrased: each row states one thing ``*``, ``?`` and ``**`` do and the
half-row states the over-match that would otherwise be invisible (``web/node``
locking ``web/node_modules``, ``fnmatch``'s ``*`` eating a ``/``). The matcher
is wired to nothing yet, so nothing here needs a workspace on disk — except the
two tests that use a real ``tmp_path`` listing to show what pol-04 does with
``literal_prefix``.
"""

from pathlib import Path
from typing import List, Tuple

import pytest

from mcp_connectors import policy_glob as glob


# (pattern, matching paths, non-matching paths) — pol-00's table, verbatim.
TABLE: List[Tuple[str, List[str], List[str]]] = [
    (".env", [".env"], [".env.local", "sub/.env"]),
    ("web/dist", ["web/dist", "web/dist/index.html"], ["web/Dist", "webx/dist"]),
    (".env*", [".env", ".env.test", ".env.local"], ["sub/.env"]),
    ("*/x", ["a/x", "b/x"], ["x", "a/b/x"]),
    ("**/x", ["x", "a/x", "a/b/x"], ["xa"]),
    ("**/__pycache__/*", ["a/__pycache__/b.pyc"], ["a/__pycache__"]),
    ("a/**/b", ["a/b", "a/x/b", "a/x/y/b"], ["a", "x/b"]),
    ("web/**", ["web", "web/index.html", "web/src/api/routes.py"], ["webby"]),
]


@pytest.mark.parametrize(
    "pattern,hits,misses",
    TABLE,
    ids=[row[0] for row in TABLE],
)
class TestTable:
    def test_it_matches_the_paths_it_means(self, pattern, hits, misses):
        for path in hits:
            assert glob.pattern_matches(pattern, path), (pattern, path)

    def test_it_misses_the_paths_it_must_not(self, pattern, hits, misses):
        for path in misses:
            assert not glob.pattern_matches(pattern, path), (pattern, path)


class TestComponentBoundary:
    def test_a_literal_folder_name_does_not_match_a_longer_name(self):
        # The bug the matcher exists to prevent: ``web/node`` locking
        # ``web/node_modules`` because a comparison stopped at a prefix.
        assert not glob.pattern_matches("web/node", "web/node_modules/x")
        assert not glob.pattern_matches("node", "node_modules/x")
        assert not glob.pattern_matches("**/mod", "src/models.py")

    def test_a_wildcard_may_end_mid_name_when_the_pattern_says_so(self):
        # The other half of the rule: a boundary is implicit only where the
        # pattern is a bare name.
        assert glob.pattern_matches("node*", "node_modules/x")
        assert glob.pattern_matches("web/node*", "web/node_modules/x")

    def test_a_single_character_wildcard_stays_in_one_name(self):
        assert glob.pattern_matches("a?c", "abc")
        assert not glob.pattern_matches("a?c", "ac")
        assert not glob.pattern_matches("a?c", "a/c")


class TestFolderRulesCoverTheirSubtree:
    @pytest.mark.parametrize("path", ["web/node_modules", "web/node_modules/react/index.js"])
    def test_a_folder_rule_reaches_any_depth_below_it(self, path):
        assert glob.pattern_matches("web/node_modules", path)

    def test_the_seeded_git_rule_hides_the_whole_internals_tree(self):
        for path in (".git", ".git/HEAD", ".git/refs/heads/main"):
            assert glob.pattern_matches(".git", path)

    def test_a_wildcard_folder_rule_covers_its_subtree_too(self):
        assert glob.pattern_matches("**/node_modules", "a/b/node_modules/react/index.js")
        assert glob.pattern_matches("build*", "build2/tmp/x.o")

    def test_what_a_wildcard_matched_as_a_folder_covers_its_subtree_too(self):
        # The consequence of the folder rule, stated rather than hoped away:
        # ``logs/*`` matches ``logs/artifacts``, so the subtree rule carries
        # everything below it as well. A single-level rule is only single-level
        # when the name it matches is a file -- the matcher cannot know, and
        # ``**`` in the middle (``*/x``) is still bounded to one folder.
        assert glob.pattern_matches("logs/*", "logs/run.txt")
        assert glob.pattern_matches("logs/*", "logs/artifacts/run.txt")
        assert not glob.pattern_matches("logs/*", "other/run.txt")
        assert not glob.pattern_matches("*/x", "a/b/x")


class TestLeadingDoubleStar:
    @pytest.mark.parametrize("path", ["x", "a/x", "a/b/c/x"])
    def test_it_matches_at_every_depth_including_the_root(self, path):
        assert glob.pattern_matches("**/x", path)

    def test_it_absorbs_zero_components_not_just_some(self):
        assert glob.pattern_matches("a/**/b", "a/b")
        assert not glob.pattern_matches("a/**/b", "b")

    def test_it_never_reaches_outside_its_anchor(self):
        assert not glob.pattern_matches("a/**", "b/a/x")


class TestSingleStarDoesNotCrossDirectories:
    def test_a_star_between_two_literals_never_spans_a_directory(self):
        # The ``fnmatch`` over-match this module exists to remove: a lone ``*``
        # there is ``.*``, so ``fnmatch`` answers True for all three.
        assert not glob.pattern_matches("*/x", "a/b/x")
        assert not glob.pattern_matches("a/*/c", "a/b/d/c")
        assert not glob.pattern_matches("*.py", "src/x.py")

    def test_a_star_never_spans_a_slash_inside_itself(self):
        # ``src/*`` is about the names of ``src``; it can never match a name two
        # folders down, because the subtree rule only adds components *after*
        # the whole pattern has been consumed -- and here the path's extra
        # components sit before the point the pattern reaches.
        assert glob.pattern_matches("src/*", "src/config.py")
        assert not glob.pattern_matches("src/*/x", "src/a/b/x")
        assert not glob.pattern_matches("logs/*", "other/logs/run.txt")


class TestCarveOutShape:
    """The example from pol-00: a ``rw`` rule inside an ``ro`` folder."""

    CARVE = "**/node_modules/**/README.md"

    @pytest.mark.parametrize(
        "path",
        ["node_modules/x/README.md", "a/b/node_modules/README.md"],
    )
    def test_it_reaches_the_file_it_was_written_for(self, path):
        assert glob.pattern_matches(self.CARVE, path)
        assert glob.pattern_matches("**/node_modules", path)

    def test_it_does_not_loosen_the_folder_rule_it_carves_out_of(self):
        assert not glob.pattern_matches(self.CARVE, "node_modules/x/other.js")
        assert not glob.pattern_matches(self.CARVE, "src/README.md")


class TestLiteralPrefix:
    @pytest.mark.parametrize(
        "pattern,prefix",
        [
            ("web/node_module/**", "web/node_module"),
            ("**/x", ""),
            ("**/__pycache__", ""),
            ("a/*/b", "a"),
            (".env*", ""),
            (".env", ".env"),
            (".git", ".git"),
            ("web/src/components/*.vue", "web/src/components"),
            ("a/b/c", "a/b/c"),
            ("src/?.py", "src"),
        ],
    )
    def test_it_stops_at_the_first_component_with_a_wildcard(self, pattern, prefix):
        assert glob.literal_prefix(pattern) == prefix

    def test_a_prefix_that_exists_is_the_thing_pol_04_stats(self, tmp_path: Path):
        # pol-04 reports a rule as unreachable when this does not exist, so the
        # value has to be usable as a path under the workspace root as-is.
        (tmp_path / "web" / "node_module").mkdir(parents=True)
        prefix = glob.literal_prefix("web/node_module/**")
        assert (tmp_path / prefix).exists()
        assert not (tmp_path / glob.literal_prefix("web/node_modules/**")).exists()

    def test_an_empty_prefix_means_there_is_nothing_to_check(self, tmp_path: Path):
        # Not a failure: ``**`` rules are anchored by their last component, so
        # pol-04 must skip the reachability test rather than report one.
        assert glob.literal_prefix("**/x") == ""
        assert (tmp_path / "").exists()


class TestPatternError:
    @pytest.mark.parametrize(
        "pattern,fraction_of_the_message",
        [
            ("/etc", "absolute"),
            ("~/secrets", "'~'"),
            ("../outside", "'..'"),
            ("a/../b", "'..'"),
            ("a//b", "empty component"),
            ("web/", "ends with '/'"),
            ("", "empty"),
            ("**", "bare '**'"),
            ("a/./b", "'.'"),
            ("x\x00y", "NUL"),
        ],
    )
    def test_each_rejected_shape_says_why(self, pattern, fraction_of_the_message):
        reason = glob.pattern_error(pattern)
        assert reason is not None
        assert fraction_of_the_message in reason

    @pytest.mark.parametrize(
        "pattern",
        [
            "**/x",
            "**/__pycache__",
            ".env",
            ".env*",
            "web/**",
            "web/node_modules",
            "**/node_modules/**/README.md",
            "a?c",
            "src/[old]/*.py",  # no character classes: the brackets are literal
            "weird name/*.txt",
        ],
    )
    def test_a_pattern_that_can_match_has_no_error(self, pattern):
        assert glob.pattern_error(pattern) is None

    def test_the_rejected_shapes_match_nothing_rather_than_raising(self):
        # ``pattern_matches`` is total: a caller that forgot to validate first
        # gets "no match", never an exception in the middle of a tree walk.
        for pattern in ("/etc", "../x", "a//b", "**", "web/"):
            assert not glob.pattern_matches(pattern, "a/b/x")
            assert not glob.pattern_matches(pattern, "")

    def test_pol_04_can_read_every_message_from_this_function_alone(self):
        # Acceptance for the task: no separate error vocabulary to keep in sync.
        assert glob.pattern_error("**/x") is None
        assert "absolute" in (glob.pattern_error("/etc") or "")


class TestWorkspaceRoot:
    @pytest.mark.parametrize(
        "pattern",
        ["*", "**", ".env", "web/**", "*/x", "**/x", "."],
    )
    def test_no_pattern_matches_the_root_itself(self, pattern):
        # The root is the workspace: an accidental match would hide or lock the
        # folder the whole tab is showing, and pol-05's own file lives under it.
        assert not glob.pattern_matches(pattern, "")

    def test_an_un_normalised_path_is_not_matched(self):
        # Callers normalise first (``policy.normalise_rel``); matching ``"./a"``
        # would let a caller's mistake read as a rule that works.
        assert not glob.pattern_matches("a", "./a")
        assert not glob.pattern_matches("a", "a/")


class TestCaseSensitivity:
    """Pinned so a future "helpful" case-fold cannot slip in (pol-00 §3)."""

    def test_a_rule_never_reaches_a_name_that_differs_only_in_case(self):
        for pattern, path in (
            ("Dist", "dist"),
            ("dist", "Dist"),
            ("web/dist", "web/Dist/index.html"),
            ("**/README.md", "docs/readme.md"),
            (".GIT", ".git"),
        ):
            assert not glob.pattern_matches(pattern, path), (pattern, path)

    def test_a_wildcard_is_case_sensitive_like_a_literal(self):
        assert glob.pattern_matches("D*", "Dist")
        assert not glob.pattern_matches("D*", "dist")


class TestReservedComponentHit:
    """The component test for the folder holding the policy (pol-00 D5)."""

    @pytest.mark.parametrize(
        "rel",
        [".nojae", ".nojae/read-write-policy.json", "a/.nojae/b", "x/y/.nojae/z"],
    )
    def test_the_reserved_name_is_reported_at_every_depth(self, rel: str):
        assert glob.reserved_component_hit(rel, ".nojae") == ".nojae"

    @pytest.mark.parametrize("rel", [".nojaeer", "nojae/x", "x.nojae/y", "", "a/b"])
    def test_anything_that_is_not_the_component_misses(self, rel: str):
        assert glob.reserved_component_hit(rel, ".nojae") is None

    def test_it_never_looks_at_the_disk(self, tmp_path: Path):
        # The answer is a name test, so absence must not change it — a broken
        # policy file cannot open the folder that holds it.
        assert not (tmp_path / ".nojae").exists()
        assert glob.reserved_component_hit(".nojae/x", ".nojae") == ".nojae"

    def test_an_empty_name_reserves_nothing(self):
        # A caller that moved the reserved name away must not get every path.
        assert glob.reserved_component_hit("anything", "") is None
