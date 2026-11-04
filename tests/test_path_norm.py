r"""Unit tests for win-02's shared path-text layer — pure text, so every case
here runs on the Linux runner (roadmap §9 rule 2: the Windows shapes are tested
by feeding `\`-shaped input and a drive spec, not by pretending to be Windows).

The tables are the contract, so they are parametrised rather than paraphrased:
each row states one thing ``canonical_rel`` / ``is_qualified`` / ``path_parts``
must answer, and the neighbouring row states the mistake it must not make
(``a\b`` folded into one name, ``foo:bar`` refused as a drive spec, a leading
``..`` quietly dropped).

Nothing imports the module from ``src`` yet — win-03 wires it into the inbound
lock and win-04 into the consumers — so these call it directly.
"""

from typing import List, Tuple

import pytest

from mcp_connectors import path_norm


# (input, expected canonical form) — A-D1's list of accepted spellings.
CANONICAL: List[Tuple[str, str]] = [
    # The §6 test-1 inputs, at this layer.
    (r"src\config.py", "src/config.py"),
    (r"a\..\b", "b"),
    ("a//b", "a/b"),
    (r"a\/b", "a/b"),
    (r"\Users\vgr", "/Users/vgr"),  # leading `\` -> the leading `/` form
    (r"/etc/x", "/etc/x"),  # a leading separator is kept, not translated away
    # Mixed and redundant spelling.
    (r"web\dist", "web/dist"),
    ("web/dist", "web/dist"),
    (r"todos\drafts", "todos/drafts"),
    ("./a/./b", "a/b"),
    ("a/b/", "a/b"),
    ("a/./", "a"),
    (r"a\\b", "a/b"),
    ("a///b", "a/b"),
    (".", ""),
    ("", ""),
    ("./", ""),
    # `..` applied lexically, surviving when it escapes the start — the
    # property policy.py:201's refusal depends on.
    ("a/../b", "b"),
    ("a/b/../c", "a/c"),
    ("a/../../b", "../b"),
    ("../a", "../a"),
    ("..", ".."),
    ("../..", "../.."),
    ("a/..", ""),
    ("a/../..", ".."),
    (r"a\..\..\b", "../b"),
    (r"a\b\..\..\c\d", "c/d"),
    # A component that only looks like a traversal stays a name.
    ("...", "..."),
    ("a/.../b", "a/.../b"),
    ("a/..b", "a/..b"),
    # Whitespace is the lock's business, not this module's.
    (" a ", " a "),
    ("a b/c d", "a b/c d"),
    # A colon after the first component is an ordinary character.
    ("src/a:b", "src/a:b"),
    ("foo:bar/src", "foo:bar/src"),
]

QUALIFIED = [
    # The three §6 test-4 shapes that slip past ``os.path.isabs`` on Windows.
    "/etc/x",
    "//srv/share",
    "C:x",
    # The rest of the qualified set.
    r"C:\x",
    "c:/x",
    r"\Users",
    "C:",
    "x:C",
    r"\\srv\share",
    "\\",  # one bare backslash: the root of a Windows drive
    "/",
    "//",
    # "The residue": a leading one-letter component plus a colon is a drive
    # spec on both platforms, so a legal POSIX filename is refused here. No
    # rule can refuse `C:x` without refusing `a:b` too — see win-02's note and
    # the lock's refusal message (win-03).
    "a:b",
    "x:y/z",
    "A:",
    "x:",
    # `..` cancellation can *create* a leading drive spec, and the canonical
    # form is what a later join would receive.
    r"a\..\C:x",
    # A single non-ASCII letter is what ntpath.splitdrive calls a drive but the
    # Windows parser is not: kept out by the ASCII constraint, so `é:x` stays
    # legal (and `é:` is not in the qualified set below).
]

NOT_QUALIFIED = [
    "src/config.py",
    r"a\b",
    "",
    ".",
    "..",
    "a/..",  # the root, spelled as a traversal
    "src/a:b",
    "foo:bar/src",
    "http://host:8080/v1",  # multi-letter first component with a colon
    "foo:",
    "foo:bar",
    "a/b:c/d",
    "é:x",
    "é:",
    ":",
    "::",
    "1:x",
    ".env:C",
    "a/../b",
    "../a",
]


class TestCanonicalRel:
    IDS = [row[0] for row in CANONICAL]

    @pytest.mark.parametrize("text,expected", CANONICAL, ids=IDS)
    def test_it_emits_the_slash_joined_form(self, text: str, expected: str) -> None:
        assert path_norm.canonical_rel(text) == expected

    @pytest.mark.parametrize("text", IDS)
    def test_the_result_never_carries_a_backslash(self, text: str) -> None:
        assert "\\" not in path_norm.canonical_rel(text)

    @pytest.mark.parametrize("text", IDS)
    def test_it_is_idempotent(self, text: str) -> None:
        once = path_norm.canonical_rel(text)
        assert path_norm.canonical_rel(once) == once

    @pytest.mark.parametrize("text", IDS + QUALIFIED + NOT_QUALIFIED)
    def test_idempotence_holds_over_the_qualified_shapes_too(self, text: str) -> None:
        # batch.py may hand this function text it normalised itself.
        once = path_norm.canonical_rel(text)
        assert path_norm.canonical_rel(once) == once

    def test_a_surviving_traversal_is_visible_to_the_caller(self) -> None:
        # The invariant the lock leans on: what stays inside collapses, what
        # escapes stays, so `..`-refusal keeps its meaning after this call.
        assert path_norm.canonical_rel("a/../../b").startswith("..")
        assert ".." not in path_norm.canonical_rel("a/../b")

    def test_a_leading_separator_survives_as_one_slash(self) -> None:
        # Not stripped (the lock refuses it), and not doubled into a UNC form
        # this module would have to invent.
        assert path_norm.canonical_rel("//srv/share") == "/srv/share"
        assert path_norm.canonical_rel(r"\\srv\share") == "/srv/share"
        assert path_norm.canonical_rel("/") == "/"


class TestIsQualified:
    @pytest.mark.parametrize("text", QUALIFIED, ids=QUALIFIED)
    def test_it_refuses_what_cannot_be_relative(self, text: str) -> None:
        assert path_norm.is_qualified(text)

    @pytest.mark.parametrize("text", NOT_QUALIFIED, ids=NOT_QUALIFIED)
    def test_it_accepts_what_can_be_relative(self, text: str) -> None:
        assert not path_norm.is_qualified(text)

    def test_the_drive_rule_is_its_own_check_not_a_separator_one(self) -> None:
        # `C:x` has no leading separator, and on Windows `ntpath.isabs` says it
        # is relative while `ntpath.join` sends it to another drive.
        assert not path_norm.canonical_rel("C:x").startswith("/")
        assert path_norm.is_qualified("C:x")

    def test_the_unc_root_is_caught_by_the_leading_separator_rule(self) -> None:
        # `ntpath.splitdrive("//srv/share")` returns the whole share as the
        # drive, so a third case would be a second definition of the same thing.
        assert path_norm.is_qualified("//srv/share")
        # The same share spelled without its leading shape is an ordinary pair
        # of names, so the verdict above comes from the separator rule.
        assert not path_norm.is_qualified("srv/share")


class TestPathParts:
    @pytest.mark.parametrize("text,expected", CANONICAL, ids=TestCanonicalRel.IDS)
    def test_it_agrees_with_the_canonical_split(self, text: str, expected: str) -> None:
        # The agreement win-04 relies on: the ancestor walk and the matcher
        # cannot disagree by coincidence of separator.
        canonical = path_norm.canonical_rel(text)
        assert path_norm.path_parts(text) == (
            tuple(canonical.split("/")) if canonical else ()
        )

    @pytest.mark.parametrize("text,expected", CANONICAL, ids=TestCanonicalRel.IDS)
    def test_the_parts_match_the_expected_form(self, text: str, expected: str) -> None:
        # Same table, checked against the expected string rather than the
        # function's own output, so a wrong canonical form cannot self-justify.
        assert path_norm.path_parts(text) == (
            tuple(expected.split("/")) if expected else ()
        )

    def test_the_root_has_no_components(self) -> None:
        assert path_norm.path_parts("") == ()
        assert path_norm.path_parts(".") == ()
        assert path_norm.path_parts("a/..") == ()

    def test_it_splits_at_every_component(self) -> None:
        deep = r"web\src\api\routes.py"
        assert path_norm.path_parts(deep) == ("web", "src", "api", "routes.py")
        assert path_norm.path_parts(r"a\..\b\c") == ("b", "c")

    def test_it_never_raises_on_input_the_lock_should_have_refused(self) -> None:
        # Total by contract: answering is this module's job, raising is the
        # lock's. A leading separator or a surviving `..` still gets an answer.
        assert path_norm.path_parts("/etc/x") == ("", "etc", "x")
        assert path_norm.path_parts("../a") == ("..", "a")
        assert path_norm.path_parts(r"C:\x") == ("C:", "x")
