"""The workspace search: the core's two halves, and the route in front of them.

The **byte supply** — which files are opened and what reaches a matcher — is
asserted on the *sequence of yielded paths* and on the counts, because both are
what the payload is assembled from: the order is what makes a capped scan the
same scan twice, and the counts are what the footer will say. The **matcher** —
:func:`search_workspace` — is then asserted on the rows (line/column/matchLength/
snippet), the payload's exact key set, and the one footer sentence.

``GET /api/fs/search`` is asserted last, and only on what the core cannot assert:
the **transport**. The wire form of the parameters, which status each core error
arrives as, that an unset query is the core's 400 rather than a framework
validation error, and that the route answers no write verb at all.

Visibility is exercised by **writing rules**, never by patching a name list (the
tree's own suite's rule, and the search's too — a search has no visibility
mechanism of its own, which is the point). The two answers that must not be
confused: a ``hidden`` path is absent **and uncounted** (counting it would be an
observation of the name), while an ``oversized`` or binary one is absent *and*
counted. The scope root alone raises; a file inside the scope never does.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pytest
from fastapi.testclient import TestClient

from src.core import editorfs
from src.api.server import create_app
from src.core.editorfs import policy_file, policy_seed
from src.core.editorfs import search as search_module
from src.core.editorfs.policy import WorkspacePathError
from src.core.editorfs.search import (
    ScanCounts,
    SearchQueryError,
    iter_scope_text,
    search_workspace,
)

from tests.test_editorfs import write_policy


def scan(rel: str = "") -> Tuple[List[Tuple[str, str]], ScanCounts]:
    """Drain the supply for ``rel``: every ``(path, text)`` pair, plus the counts."""
    counts = ScanCounts()
    return list(iter_scope_text(rel, counts=counts)), counts


def paths(rel: str = "") -> List[str]:
    """Just the yielded paths, in the order the walk supplied them."""
    return [path for path, _ in scan(rel)[0]]


def write(root: Path, rel: str, body: str) -> None:
    """One file of the throw-away workspace, with its folders."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty tree pointed at as the working location, seeded like a real one.

    Re-declared rather than imported: pytest fixtures do not travel between test
    modules. Each test writes exactly the files its assertion is about, so an
    expected sequence here is a complete answer rather than a subset of a
    decoration. The seeded policy is what makes ``node_modules`` ``ro`` and
    ``.git`` ``hidden`` the way a project a user opened would be.
    """
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    policy_seed.ensure_policy_file(str(tmp_path))
    policy_file.invalidate_policy_cache()
    return tmp_path


@pytest.fixture
def client(workspace: Path):
    """The app over the same throw-away workspace the core rows use.

    Re-declared rather than imported from ``tests/test_editorfs.py`` for the same
    reason as the fixture above: pytest does not move fixtures between modules.
    """
    with TestClient(create_app()) as test_client:
        yield test_client


def search(client: TestClient, **params: Any):
    """One ``GET /api/fs/search``, params spelled as they go on the wire."""
    return client.get("/api/fs/search", params=params)


class TestScopeSelection:
    def test_root_scope_reaches_a_nested_file(self, workspace: Path):
        write(workspace, "src/core/job.py", "needle\n")
        write(workspace, "top.py", "needle\n")
        # Dirs before files at the root, so the nested one is supplied first.
        assert paths("") == ["src/core/job.py", "top.py"]

    def test_folder_scope_reaches_a_descendant_not_a_sibling_folder(
        self, workspace: Path
    ):
        write(workspace, "docs/old/deep/notes.md", "needle\n")
        write(workspace, "docs/old/notes.md", "needle\n")
        write(workspace, "docs/new/notes.md", "needle\n")
        write(workspace, "docs/notes.md", "needle\n")
        write(workspace, "notes.md", "needle\n")
        assert paths("docs/old") == [
            "docs/old/deep/notes.md",
            "docs/old/notes.md",
        ]

    def test_file_scope_yields_exactly_that_path(self, workspace: Path):
        write(workspace, "src/config.py", "needle\n")
        write(workspace, "src/config.test.py", "needle\n")
        supplied, _ = scan("src/config.py")
        assert supplied == [("src/config.py", "needle\n")]

    def test_collapsed_and_capped_branches_are_still_walked(self, workspace: Path,
                                                            monkeypatch):
        """The tree's *listing* cap is not a search cap (spec 2.5).

        Three files under a folder the tree would list two of: a scan reads the
        subtree in full, because ``MAX_TREE_ENTRIES_PER_DIR`` bounds one screenful
        of rows and says nothing about access.
        """
        monkeypatch.setattr(editorfs.constants, "MAX_TREE_ENTRIES_PER_DIR", 2)
        for name in ("a.py", "b.py", "c.py"):
            write(workspace, f"src/{name}", "needle\n")
        assert paths("src") == ["src/a.py", "src/b.py", "src/c.py"]


class TestSizeAndEncoding:
    def test_file_scope_over_the_editor_cap_is_counted_not_a_413(
        self, workspace: Path
    ):
        """A 1.5 MB file is a *skip*, where ``read_file`` of the same path is a 413.

        The size is the spec's, not a patched cap, because the row is about the
        difference between the two surfaces over the *real* ``MAX_EDITOR_BYTES``:
        a human who asked for that one file deserves to know why it did not open,
        and a scan has no right to fail the whole search because one artefact is
        large.
        """
        cap = editorfs.constants.MAX_EDITOR_BYTES
        (workspace / "big.py").write_bytes(b"x" * (cap + 500_000))
        supplied, counts = scan("big.py")
        assert supplied == []
        assert counts.oversized == 1
        assert counts.files_scanned == 0
        # The contrast the row exists for: the *read* of that same path answers
        # 413, so the difference is the scan's choice, not a missing cap.
        with pytest.raises(WorkspacePathError) as excinfo:
            editorfs.read_file("big.py")
        assert excinfo.value.status_code == 413

    def test_folder_scope_over_the_editor_cap_is_counted_too(self, workspace: Path,
                                                            monkeypatch):
        """Same answer from a folder scope — the cap is the reader's, not the gate's."""
        monkeypatch.setattr(editorfs.constants, "MAX_EDITOR_BYTES", 8)
        (workspace / "big.py").write_bytes(b"x" * 9)
        write(workspace, "small.py", "needle\n")
        supplied, counts = scan("")
        assert [path for path, _ in supplied] == ["small.py"]
        assert counts.oversized == 1

    def test_nul_in_the_head_is_binary(self, workspace: Path):
        (workspace / "blob.txt").write_bytes(b"needle\x00def\n")
        supplied, counts = scan("")
        assert supplied == []
        assert counts.binary == 1
        # Read and then rejected: the scan paid for it, so the footer says so.
        assert counts.files_scanned == 1

    def test_invalid_utf8_is_binary_like_the_read_route_does(self, workspace: Path):
        (workspace / "latin.txt").write_bytes(b"needle caf\xe9\n")
        supplied, counts = scan("")
        assert supplied == []
        assert counts.binary == 1

    def test_the_bytes_are_the_whole_file(self, workspace: Path):
        write(workspace, "a.py", "one\ntwo\nthree\n")
        supplied, _ = scan("")
        assert supplied == [("a.py", "one\ntwo\nthree\n")]


class TestPolicyIsTheTreesPolicy:
    def test_hidden_folder_is_pruned_and_nothing_is_counted(self, workspace: Path):
        write(workspace, "secret/inner.md", "needle\n")
        write(workspace, "secret/deeper/inner.md", "needle\n")
        write_policy(workspace, [{"path": "secret", "mode": "hidden"}])
        supplied, counts = scan("")
        assert supplied == []
        # The loud half of D2: not one counter moved, because a count of a
        # policy-skip would confirm the hidden name exists.
        assert counts == ScanCounts()

    def test_hidden_file_is_omitted_and_nothing_is_counted(self, workspace: Path):
        write(workspace, "hidden.md", "needle\n")
        write(workspace, "plain.md", "needle\n")
        write_policy(workspace, [{"path": "hidden.md", "mode": "hidden"}])
        supplied, counts = scan("")
        assert [path for path, _ in supplied] == ["plain.md"]
        # One read — the visible file — and no skip counter for the hidden one.
        assert counts == ScanCounts(files_scanned=1)

    def test_a_hidden_folder_costs_no_scandir_below_it(self, workspace: Path,
                                                       monkeypatch):
        """The prune is a *prune*, not a filter applied after the descent.

        The reason the tree prunes rather than skips is that nothing under a
        hidden name may be reached at all — a search that listed the folder and
        then dropped its rows would still have read the bytes of a secret file
        into memory on the way. So the counter here is ``os.scandir`` itself: the
        hidden folder is seen as a name, and never opened.
        """
        write(workspace, "secret/deeper/inner.md", "needle\n")
        write(workspace, "plain.md", "needle\n")
        write_policy(workspace, [{"path": "secret", "mode": "hidden"}])

        listed: List[str] = []
        real_scandir = os.scandir

        def spy(path: Union[str, "os.PathLike[str]"]):
            listed.append(str(path))
            return real_scandir(path)

        monkeypatch.setattr(search_module.os, "scandir", spy)
        assert [path for path, _ in scan("")[0]] == ["plain.md"]
        # Exactly one directory was opened — the root. The hidden name was seen
        # in its listing and never descended into.
        assert len(listed) == 1, listed
        assert Path(listed[0]).resolve() == workspace.resolve()

    def test_a_hidden_path_cannot_be_searched_by_name(self, workspace: Path):
        """The one difference a file scope has: its *path* still goes through the
        gate, so 404 for a hidden scope exactly as for a hidden read."""
        write(workspace, "hidden.md", "needle\n")
        write_policy(workspace, [{"path": "hidden.md", "mode": "hidden"}])
        with pytest.raises(WorkspacePathError) as excinfo:
            paths("hidden.md")
        assert excinfo.value.status_code == 404

    def test_ro_is_searched(self, workspace: Path):
        """``ro`` constrains what changes, not what is seen — the tree promised
        this row, so the search owes it a look at the bytes."""
        write(workspace, "vendor/lib.js", "needle\n")
        write_policy(workspace, [{"path": "vendor", "mode": "ro"}])
        assert paths("") == ["vendor/lib.js"]

    def test_the_reserved_folder_is_never_searched(self, workspace: Path):
        (workspace / editorfs.constants.POLICY_DIRNAME / "notes.md").write_text(
            "needle\n", encoding="utf-8"
        )
        write(workspace, "visible.md", "needle\n")
        supplied, counts = scan("")
        assert [path for path, _ in supplied] == ["visible.md"]
        # The visible file is the only read: the reserved folder costs no skip.
        assert counts == ScanCounts(files_scanned=1)

    def test_a_carve_back_out_of_a_hidden_folder_reopens_nothing(
        self, workspace: Path
    ):
        """``hidden`` is absolute, so the walk prunes on the *folder's own* verdict
        and a later ``rw`` rule below it is never reached — the tree's rule, and
        the reason the prune is a prune: a search that asked each child would find
        this row readable and quote the secret anyway."""
        write(workspace, "docs/old/notes.md", "needle\n")
        write_policy(workspace, [
            {"path": "docs/old", "mode": "hidden"},
            {"path": "docs/old/notes.md", "mode": "rw"},
        ])
        supplied, counts = scan("")
        assert supplied == []
        assert counts == ScanCounts()


class TestSymlinksStayOutOfTheSupply:
    """*Traps* 1 — the security-relevant rows: the walk never resolves a link."""

    def test_symlinked_directory_is_not_followed(
        self, workspace: Path, tmp_path: Path
    ):
        outside = tmp_path.parent / "search-outside-dir"
        outside.mkdir(exist_ok=True)
        (outside / "leak.md").write_text("needle\n", encoding="utf-8")
        (workspace / "link").symlink_to(outside, target_is_directory=True)
        supplied, counts = scan("")
        assert supplied == []
        assert counts == ScanCounts()

    def test_symlinked_file_is_never_opened_and_never_counted(
        self, workspace: Path, tmp_path: Path
    ):
        """The row the spec's table does not have.

        A link to a file outside the working location is listed by the tree as an
        ordinary file, and ``open()`` on it would read those outside bytes with
        nothing in the way — the ``realpath`` lock only guards paths that go
        through the gate, and a walk does not resolve per child. So the name is
        skipped in silence: no text, and no counter naming it.
        """
        outside = tmp_path.parent / "search-outside-target"
        outside.mkdir(exist_ok=True)
        (outside / "secret.txt").write_text("needle\n", encoding="utf-8")
        (workspace / "link.txt").symlink_to(outside / "secret.txt")
        supplied, counts = scan("")
        assert supplied == []
        assert counts == ScanCounts()

    def test_a_symlink_to_a_file_inside_the_root_is_skipped_with_the_others(
        self, workspace: Path
    ):
        """Still skipped, even where following it would read an allowed file: the
        rule is one shape (no symlink is opened), not a per-target judgement."""
        write(workspace, "real.txt", "needle\n")
        (workspace / "alias.txt").symlink_to(workspace / "real.txt")
        supplied, counts = scan("")
        assert [path for path, _ in supplied] == ["real.txt"]
        assert counts.files_scanned == 1


class TestWalkOrder:
    def test_dirs_first_then_files_each_sorted_case_insensitively(
        self, workspace: Path
    ):
        """The tree's order, exact-name tie-break included (D6).

        Two names differing only in case sort arbitrarily by inode otherwise, which
        would make a capped scan a different N rows each run.
        """
        write(workspace, "B/a.py", "needle\n")
        write(workspace, "b/a.py", "needle\n")
        write(workspace, "Z.py", "needle\n")
        write(workspace, "A.py", "needle\n")
        write(workspace, "a.py", "needle\n")
        assert paths("") == [
            "B/a.py",
            "b/a.py",
            "A.py",
            "a.py",
            "Z.py",
        ]


class TestCapsStopTheSupply:
    def test_max_search_files_supplies_the_first_two_in_walk_order(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_FILES", 2)
        write(workspace, "B/a.py", "needle\n")
        write(workspace, "Z.py", "needle\n")
        write(workspace, "A.py", "needle\n")
        write(workspace, "a.py", "needle\n")
        supplied, counts = scan("")
        # Not the two biggest or the two newest: the first two of the tree's own
        # order — the only folder's file, then the first root file.
        assert [path for path, _ in supplied] == ["B/a.py", "A.py"]
        assert counts.files_scanned == 2
        assert counts.files_cut is True
        assert counts.bytes_cut is False

    def test_max_search_bytes_cuts_the_supply_never_a_fragment(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The budget is a per-scan total, and a file that does not fit in what is
        left is not searched at all — a fragment would be reported as the file."""
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_BYTES", 8)
        for name, body in (("a.txt", b"111\n"), ("b.txt", b"222\n"), ("c.txt", b"333\n")):
            (workspace / name).write_bytes(body)
        supplied, counts = scan("")
        assert supplied == [("a.txt", "111\n"), ("b.txt", "222\n")]
        assert counts.files_scanned == 2
        assert counts.bytes_cut is True
        assert counts.files_cut is False

    def test_the_budget_is_not_shared_between_scans(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_BYTES", 4)
        write(workspace, "a.txt", "111\n")
        first, first_counts = scan("")
        second, second_counts = scan("")
        assert first == second
        assert first_counts.bytes_cut is False
        assert second_counts.bytes_cut is False


class TestScopeGate:
    @pytest.mark.parametrize("raw", ["../elsewhere", "/etc/hosts", "~/x", "a/../../b"])
    def test_an_escaping_scope_is_422(self, workspace: Path, raw: str):
        with pytest.raises(WorkspacePathError) as excinfo:
            paths(raw)
        assert excinfo.value.status_code == 422

    def test_an_absent_scope_is_404(self, workspace: Path):
        with pytest.raises(WorkspacePathError) as excinfo:
            paths("nope")
        assert excinfo.value.status_code == 404


# -- the matcher half: the rows, the payload, the footer sentence ----------------

PAYLOAD_KEYS = {
    "scope",
    "isFile",
    "matches",
    "matchCount",
    "filesScanned",
    "truncated",
    "skipped",
    "footerNote",
}


def rows(payload: Dict[str, Any]) -> List[Tuple[str, int, int, int]]:
    """The result rows as ``(path, line, column, matchLength)`` — the four numbers
    the caret depends on, without the snippet noise around them."""
    return [
        (row["path"], row["line"], row["column"], row["matchLength"])
        for row in payload["matches"]
    ]


class TestQueryIsAQuery:
    def test_literal_query_is_case_insensitive_by_default(self, workspace: Path):
        write(workspace, "a.py", "Foo\nfoo\nFOO\nbar\n")
        payload = search_workspace("", query="foo")
        assert payload["matchCount"] == 3
        assert [row["line"] for row in payload["matches"]] == [1, 2, 3]

    def test_case_sensitive_matches_only_the_exact_case(self, workspace: Path):
        write(workspace, "a.py", "Foo\nfoo\nFOO\n")
        payload = search_workspace("", query="foo", case_sensitive=True)
        assert payload["matchCount"] == 1
        assert payload["matches"][0]["line"] == 2

    def test_a_literal_query_is_not_a_pattern(self, workspace: Path):
        """``a.c`` is a needle: the dot does not stand for a character unless the
        user asked for regex — the spec's own row for the two modes."""
        write(workspace, "a.py", "abc\na.c\n")
        literal = search_workspace("", query="a.c")
        assert rows(literal) == [("a.py", 2, 1, 3)]
        as_pattern = search_workspace("", query="a.c", regex=True)
        assert rows(as_pattern) == [("a.py", 1, 1, 3), ("a.py", 2, 1, 3)]

    def test_regex_meta_chars_in_a_literal_query_are_the_needle(self, workspace: Path):
        """``(`` must not raise when regex is off, and must find the parenthesis."""
        write(workspace, "a.py", "if (x) {\n")
        payload = search_workspace("", query="(")
        assert payload["matchCount"] == 1
        assert payload["matches"][0]["column"] == 4
        assert payload["matches"][0]["snippet"] == "if (x) {"

    def test_an_invalid_pattern_is_400_carrying_re_message(self, workspace: Path):
        write(workspace, "a.py", "foo(bar\n")
        with pytest.raises(SearchQueryError) as excinfo:
            search_workspace("", query="foo(bar", regex=True)
        assert excinfo.value.status_code == 400
        # re's own words, caret position included — the user wrote this pattern.
        assert "missing ), unterminated subpattern" in excinfo.value.detail

    def test_an_invalid_pattern_is_compiled_before_anything_is_walked(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """D5 — asserted on the **count**, not on an empty payload: a scan that
        ran and matched nothing answers the same way, so the payload proves
        nothing. A file whose contents match the prefix of the bad pattern sits
        where a walk would find it.

        The same spies are run over a *valid* query first, so the zero cannot come
        from the file being invisible or from the spy being wired wrong — it is
        only about the order of the function body: compile, then walk.
        """
        write(workspace, "a.py", "foo(bar\n")

        listed: List[str] = []
        real_scandir = os.scandir

        def spy_scandir(path: Union[str, "os.PathLike[str]"]):
            listed.append(str(path))
            return real_scandir(path)

        monkeypatch.setattr(search_module.os, "scandir", spy_scandir)

        seen = ScanCounts()
        monkeypatch.setattr(search_module, "ScanCounts", lambda: seen)
        assert search_workspace("", query="foo")["matchCount"] == 1
        assert seen.files_scanned == 1
        assert listed, "the spies must be live for the zero below to mean anything"

        listed.clear()
        untouched = ScanCounts()
        monkeypatch.setattr(search_module, "ScanCounts", lambda: untouched)
        with pytest.raises(SearchQueryError):
            search_workspace("", query="foo(bar", regex=True)
        # Not one counter moved, and not one directory was listed.
        assert untouched.files_scanned == 0
        assert untouched == ScanCounts()
        assert listed == []

    def test_an_empty_query_is_400_and_not_a_scan_of_every_line(
        self, workspace: Path
    ):
        """An empty needle would match every line, and the payload would look like
        a working search — so it is refused with its own sentence."""
        write(workspace, "a.py", "one\ntwo\n")
        with pytest.raises(SearchQueryError) as excinfo:
            search_workspace("", query="")
        assert excinfo.value.status_code == 400
        assert "empty" in excinfo.value.detail


class TestRows:
    def test_line_column_and_match_length_are_one_based(self, workspace: Path):
        write(workspace, "src/a.py", "the verb and a verb\n")
        payload = search_workspace("", query="verb")
        assert rows(payload) == [("src/a.py", 1, 5, 4), ("src/a.py", 1, 16, 4)]
        # ``column`` is the match start, so the caret lands on what was searched.
        for row in payload["matches"]:
            start = row["column"] - 1
            span = row["snippet"][start : start + row["matchLength"]]
            assert span == "verb"

    def test_a_zero_width_pattern_terminates(self, workspace: Path):
        """*Traps* 6 — ``finditer`` advances on a zero-width match where a manual
        ``while m := search()`` loop would spin."""
        write(workspace, "a.py", "ab\n")
        payload = search_workspace("", query="x*", regex=True)
        assert payload["matchCount"] == 3
        assert rows(payload) == [
            ("a.py", 1, 1, 0),
            ("a.py", 1, 2, 0),
            ("a.py", 1, 3, 0),
        ]

    def test_the_row_cap_keeps_counting_what_it_stops_listing(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_MATCHES", 3)
        write(workspace, "a.py", "hit\nhit\nhit\nhit\nhit\n")
        payload = search_workspace("", query="hit")
        assert len(payload["matches"]) == 3
        assert payload["matchCount"] == 5
        assert payload["truncated"] is True
        assert payload["footerNote"] == (
            "showing the first 3 matches — refine the query"
        )

    def test_snippet_is_trimmed_at_the_cap_with_an_ellipsis(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_SNIPPET_BYTES", 8)
        write(workspace, "a.py", "needle and a very long tail\nshort needle\n")
        payload = search_workspace("", query="needle")
        assert [row["snippet"] for row in payload["matches"]] == [
            "needle a…",
            "short ne…",
        ]

    def test_a_short_line_is_not_trimmed(self, workspace: Path):
        write(workspace, "a.py", "short needle\n")
        payload = search_workspace("", query="needle")
        assert payload["matches"][0]["snippet"] == "short needle"

    def test_the_cut_is_on_characters_not_bytes(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A multi-byte character at the boundary comes out intact: the cap bounds
        a *display*, and slicing bytes mid-character would put a broken glyph in a
        DOM node."""
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_SNIPPET_BYTES", 5)
        write(workspace, "a.py", "abcéfg needle\n")
        payload = search_workspace("", query="needle")
        # ``abcé`` is 5 *bytes* but only 4 characters: a byte-slice would cut the
        # é in half and put a broken glyph in the cell.
        assert payload["matches"][0]["snippet"] == "abcéf…"

    def test_a_crlf_file_reports_the_columns_of_the_text_without_the_cr(
        self, workspace: Path
    ):
        (workspace / "a.txt").write_bytes(b"first\r\nsecond needle\r\n")
        payload = search_workspace("", query="needle")
        assert rows(payload) == [("a.txt", 2, 8, 6)]
        assert payload["matches"][0]["snippet"] == "second needle"


class TestPayload:
    def test_the_key_set_is_spelled_out(self, workspace: Path):
        """The exact set, not ``>=`` containment: a new key is a deliberate act."""
        write(workspace, "a.py", "needle\n")
        (workspace / "blob.bin").write_bytes(b"needle\x00\n")
        payload = search_workspace("", query="needle")
        assert set(payload) == PAYLOAD_KEYS
        assert set(payload["skipped"]) == {"oversized", "binary"}
        assert "policy" not in payload["skipped"]

    def test_a_file_scope_reports_itself_as_a_file(self, workspace: Path):
        write(workspace, "src/config.py", "needle\n")
        payload = search_workspace("src/config.py", query="needle")
        assert payload["isFile"] is True
        assert payload["scope"] == "src/config.py"
        assert rows(payload) == [("src/config.py", 1, 1, 6)]

    def test_the_root_scope_is_the_empty_string(self, workspace: Path):
        write(workspace, "src/core/job.py", "needle\n")
        payload = search_workspace("", query="needle")
        assert payload["isFile"] is False
        assert payload["scope"] == ""
        assert payload["matches"][0]["path"] == "src/core/job.py"

    def test_a_folder_scope_normalises_its_path(self, workspace: Path):
        write(workspace, "docs/old/notes.md", "the verb here\n")
        payload = search_workspace("./docs/old/", query="verb")
        assert payload["scope"] == "docs/old"
        assert payload["isFile"] is False

    def test_the_scope_errors_reach_the_caller_through_the_matcher(
        self, workspace: Path
    ):
        """The route only re-raises, so the supply's two answers must survive the
        generator: ``iter_resolved_scope`` is lazy, and a 404 swallowed until the
        rows were consumed would still be a 404 here — the loop always consumes."""
        write(workspace, "hidden.md", "needle\n")
        write_policy(workspace, [{"path": "hidden.md", "mode": "hidden"}])
        for rel, status in (("nope", 404), ("hidden.md", 404), ("../out", 422)):
            with pytest.raises(WorkspacePathError) as excinfo:
                search_workspace(rel, query="needle")
            assert excinfo.value.status_code == status, rel


class TestFooter:
    def test_the_builder_names_every_cut_in_the_maps_order(self):
        """The builder is a pure function of the counts, so the *whole* sentence
        — every part at once — is testable without a filesystem, and a real scan
        never has to be bent into cutting two walk caps at the same time.

        Order: walk caps first (they bound completeness), then the row cap (which
        bounds only what is listed), then the counted skips, then the closing
        clause.
        """
        assert search_module._footer_note(
            files_cut=True,
            bytes_cut=True,
            rows_cut=True,
            oversized=3,
            binary=4,
        ) == (
            "searched the first 5000 files; read the first 32 MB; "
            "showing the first 500 matches; 3 files too large to search; "
            "4 binary files skipped — refine the query"
        )

    def test_one_walk_cap_end_to_end(self, workspace: Path,
                                     monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_FILES", 1)
        for name in ("a.txt", "b.txt"):
            write(workspace, name, "hit\n")
        payload = search_workspace("", query="hit")
        assert payload["truncated"] is True
        assert payload["skipped"] == {"oversized": 0, "binary": 0}
        assert payload["footerNote"] == (
            "searched the first 1 file — refine the query"
        )

    def test_skips_without_a_cap_cut_are_named_and_are_not_truncation(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """D9 — a scan that read every readable file is complete even if it
        skipped three big ones and a couple of binaries."""
        monkeypatch.setattr(editorfs.constants, "MAX_EDITOR_BYTES", 8)
        (workspace / "big.py").write_bytes(b"hit123456789")
        (workspace / "blob.bin").write_bytes(b"hit\x00\n")
        write(workspace, "small.py", "hit\n")
        payload = search_workspace("", query="hit")
        assert payload["truncated"] is False
        assert payload["skipped"] == {"oversized": 1, "binary": 1}
        assert payload["footerNote"] == (
            "1 file too large to search; 1 binary file skipped"
        )

    def test_a_clean_search_says_nothing(self, workspace: Path):
        write(workspace, "a.py", "hit\n")
        payload = search_workspace("", query="hit")
        assert payload["truncated"] is False
        assert payload["footerNote"] == ""

    def test_the_footer_names_the_caps_as_they_are_patched(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The sentence is built from ``constants`` at call time — the same patch
        that cuts the scan is the number the footer shows."""
        monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_FILES", 2)
        for name in ("a.txt", "b.txt", "c.txt"):
            write(workspace, name, "hit\n")
        payload = search_workspace("", query="hit")
        assert payload["footerNote"] == (
            "searched the first 2 files — refine the query"
        )

    def test_a_policy_skip_never_appears_in_the_footer(self, workspace: Path):
        """D2 again, from the payload side: no count, no hint, no extra key — the
        hidden name is not observable through the footer any more than through
        the rows."""
        write(workspace, "secret/inner.md", "hit\n")
        write(workspace, "plain.md", "hit\n")
        write_policy(workspace, [{"path": "secret", "mode": "hidden"}])
        payload = search_workspace("", query="hit")
        assert payload["filesScanned"] == 1
        assert payload["skipped"] == {"oversized": 0, "binary": 0}
        assert payload["footerNote"] == ""
        assert payload["matches"] == [
            {
                "path": "plain.md",
                "line": 1,
                "column": 1,
                "snippet": "hit",
                "matchLength": 3,
            }
        ]


class TestPackageExport:
    """What search-03 will actually call — through the package, not the module."""

    def test_the_entry_point_and_the_error_are_attributes_of_the_package(self):
        assert editorfs.search_workspace is search_workspace
        assert editorfs.SearchQueryError is SearchQueryError

    def test_the_caps_still_resolve_through_the_package_getattr(self):
        assert editorfs.MAX_SEARCH_MATCHES == editorfs.constants.MAX_SEARCH_MATCHES
        assert editorfs.MAX_SEARCH_SNIPPET_BYTES == (
            editorfs.constants.MAX_SEARCH_SNIPPET_BYTES
        )


# -- the route: GET /api/fs/search, transport only ------------------------------


class TestRoute:
    """``GET /api/fs/search`` — the payload of the Contract, over HTTP.

    The rules themselves are asserted on the core above; these rows are about the
    seam: the **wire form** of the parameters, the status each core error arrives
    as, and that this route answers no verb but ``GET``. A handler that grew a
    rule would show up here as a row the core section already covers, which is
    what makes this class short on purpose.
    """

    def test_a_folder_scope_answers_the_contracts_exact_key_set(
        self, workspace: Path, client: TestClient
    ):
        write(workspace, "docs/old/notes.md", "the verb of the sentence\n")
        write(workspace, "docs/new/notes.md", "the verb of the sentence\n")
        response = search(client, path="docs/old", q="verb")
        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == PAYLOAD_KEYS
        assert set(payload["skipped"]) == {"oversized", "binary"}
        assert payload["scope"] == "docs/old"
        assert payload["isFile"] is False
        assert payload["matches"] == [
            {
                "path": "docs/old/notes.md",
                "line": 1,
                "column": 5,
                "snippet": "the verb of the sentence",
                "matchLength": 4,
            }
        ]
        assert payload["matchCount"] == 1
        assert payload["filesScanned"] == 1
        assert payload["truncated"] is False
        assert payload["footerNote"] == ""

    def test_q_alone_searches_the_whole_workspace_with_both_flags_off(
        self, workspace: Path, client: TestClient
    ):
        """The defaults, and the one that proves them: ``(`` is a needle.

        A literal ``(`` reaching the matcher is only possible when ``regex``
        defaulted to **false** — as a pattern it is a 400. So this row is the
        proof, not ``payload["isFile"] is False`` alone.
        """
        write(workspace, "src/config.py", "if (verb) {\n")
        response = search(client, q="verb")
        assert response.status_code == 200
        payload = response.json()
        assert payload["scope"] == ""
        assert payload["isFile"] is False
        assert [row["path"] for row in payload["matches"]] == ["src/config.py"]

        paren = search(client, q="(")
        assert paren.status_code == 200
        assert paren.json()["matches"][0]["column"] == 4

    def test_case_sensitive_and_regex_are_the_camel_case_names(
        self, workspace: Path, client: TestClient
    ):
        write(workspace, "a.py", "Foo\nfoo\nbar(baz)\n")
        assert search(client, q="foo").json()["matchCount"] == 2
        assert search(client, q="foo", caseSensitive="true").json()["matchCount"] == 1
        assert search(client, q="bar.*").json()["matchCount"] == 0
        assert search(client, q="bar.*", regex="true").json()["matchCount"] == 1

    def test_a_snake_case_name_is_ignored_not_an_error(
        self, workspace: Path, client: TestClient
    ):
        """The negative half of the row above, and the reason it is pinned.

        FastAPI ignores an unknown query parameter rather than rejecting it, so a
        rename of the handler's arguments to snake_case would otherwise answer
        with the case-insensitive default and look like a passing search. The
        contract is camelCase on the wire, and only a red test keeps it there.
        """
        write(workspace, "a.py", "Foo\nfoo\n")
        payload = search(client, q="foo", case_sensitive="true").json()
        assert payload["matchCount"] == 2

    def test_an_invalid_pattern_is_400_carrying_re_owns_words(
        self, workspace: Path, client: TestClient
    ):
        write(workspace, "a.py", "foo[bar\n")
        response = search(client, q="foo[", regex="true")
        assert response.status_code == 400
        assert "unterminated character set" in response.json()["detail"]

    def test_an_unset_query_is_the_cores_sentence_not_a_validation_error(
        self, workspace: Path, client: TestClient
    ):
        """``q`` defaults to "" so this is a 400 with a *string* ``detail``.

        FastAPI's own validation failure is a 422 whose ``detail`` is a list of
        error objects — the shape the dialog cannot show, and the reason the
        handler declares the default instead of a required parameter.
        """
        response = search(client)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert isinstance(detail, str)
        assert "empty" in detail

    @pytest.mark.parametrize("raw", ["../elsewhere", "/etc"])
    def test_an_escaping_scope_is_422(
        self, workspace: Path, client: TestClient, raw: str
    ):
        assert search(client, path=raw, q="verb").status_code == 422

    def test_a_hidden_scope_is_404_and_the_body_does_not_name_the_rule(
        self, workspace: Path, client: TestClient
    ):
        """pol-00 D8 over the wire: a search of a hidden folder is *absence*.

        The 422 a write gets would confirm the name exists, which is the leak
        hiding the row exists to prevent — so the body has to read exactly like
        the answer for a path that was never there.
        """
        write(workspace, "secret-vault/ledger.txt", "the verb\n")
        write_policy(workspace, [{"path": "secret-vault", "mode": "hidden"}])
        response = search(client, path="secret-vault", q="verb")
        assert response.status_code == 404
        body = response.json()["detail"]
        assert "secret-vault" in body  # the same sentence an absent path gets
        for giveaway in ("hidden", "policy", "rule", ".nojae"):
            assert giveaway not in body
        # And from the root scope, no row from inside it either.
        assert search(client, q="verb").json()["matches"] == []

    def test_an_absent_scope_is_404(self, workspace: Path, client: TestClient):
        response = search(client, path="nope", q="verb")
        assert response.status_code == 404
        body = response.json()["detail"]
        assert "nope" in body
        for giveaway in ("hidden", "policy", "rule", ".nojae"):
            assert giveaway not in body
        # The two sentences differ in wording (a hidden name gets the read
        # route's "Path does not exist", an absent folder the walk's), and both
        # are indistinguishable from absence to the caller.

    @pytest.mark.parametrize("verb", ["post", "put", "delete"])
    def test_the_route_answers_no_write_verb(
        self, workspace: Path, client: TestClient, verb: str
    ):
        """405, not a route that was quietly added: this is a read (D16)."""
        response = getattr(client, verb)("/api/fs/search", params={"q": "verb"})
        assert response.status_code == 405
