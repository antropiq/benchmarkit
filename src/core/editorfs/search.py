"""One workspace search: which bytes a query may see, and what a match is.

Two halves in one module. The **byte supply** (:func:`iter_scope_text`) answers
*"which files are read?"* — the scope root, the walk, one policy verdict per
child, the caps and the reader — yielding ``(path, text)`` pairs and reporting
its counts as it goes. The **matcher** (:func:`search_workspace`) answers *"what
a match is?"*: it compiles the query, turns the supplied text into result rows
and folds every count into the payload dict the search route sends. Neither half
touches FastAPI: the status carried by :class:`SearchQueryError` and by
:class:`~src.core.editorfs.policy.WorkspacePathError` *is* the contract.

What it promises (each rule mirrored by a test in ``tests/test_editorfs_search.py``):

* **a search's visible set is the tree's visible set.** One
  :func:`editorfs.policy_modes.mode_for` verdict per child, in the same three
  lines :func:`editorfs.tree.list_dir` uses: ``hidden`` is *pruned* — a directory
  costs no ``scandir`` below it — and a file is omitted; ``ro`` is searched,
  because it constrains what changes, never what is read. A path skipped by
  policy is **not counted anywhere**: a count of it would be an observation of a
  hidden name, which is the leak hiding the row exists to prevent. The scan does
  not inherit the tree's *listing* cap — that bounds one screenful of rows, not
  access — so a collapsed branch is walked in full.
* **and that visible set is then narrowed by the repository's ignore rules.**
  :func:`editorfs.gitstatus.ignored_paths` asks git (``status --ignored=matching``)
  what this workspace excludes, and the walk prunes those names the same way it
  prunes a ``hidden`` one: an ignored *folder* costs no ``scandir`` below it and
  an ignored file is never opened. Two filters, in that order, and neither one is
  the other's substitute — the policy says what this surface may see, the ignore
  rules say what the project calls its own bytes. A path with no repository (or a
  git that cannot be asked) simply has no ignore filter; the policy still bounds
  the scan, so this costs pruning of caches and build output and never a refusal.
  Ignored paths are counted nowhere either: they are not a cap the user can hit.
* the scope **root** goes through :func:`editorfs.policy.resolve_in_workspace`
  once: 422 for a path outside the workspace, 404 for one that is hidden or
  absent. A hidden *file* therefore cannot be searched by name even though no row
  of it was ever listed.
* **a symlink is never opened and never descended.** A link to a file outside the
  working location looks like an ordinary row in the tree, but ``open()`` follows
  it, and the ``realpath`` lock only guards paths that pass through
  ``resolve_in_workspace`` — which a walk does not do per child. So the walk
  reads only non-symlink regular files, silently and counted nowhere; clicking
  that row still gets the read route's 422, unchanged.
* the caps **stop the supply and are reported, never raised**: over
  ``MAX_EDITOR_BYTES`` a file is skipped and counted in ``oversized`` (a file
  scope and a folder scope agree, so a big file is never a 413), and a file past
  ``MAX_SEARCH_FILES`` or past the ``MAX_SEARCH_BYTES`` budget ends the scan with
  its cut flag. A file that does not fit in what is left of the budget is not
  searched *partially* — a fragment would be reported as if it were the file.
* the walk's **order is the tree's order** (directories first, then files, each
  sorted by :func:`editorfs.tree._name_key`), so a truncated scan hands over the
  same first N files every run rather than whichever inode answered first.
* text is decoded **strict** UTF-8 after the same NUL-in-``BINARY_SNIFF_BYTES``
  sniff the read route uses; anything else is binary, skipped and counted.
* **a query that is not a query costs no scan**: the pattern is compiled before
  the walk is asked, an invalid one answers 400 with ``re``'s own message (the
  user wrote it and needs the caret position back), and an empty one is refused
  rather than run against every line of the workspace.
* **the counts stay true when the rows are not**: ``matchCount`` includes matches
  the row cap dropped, ``truncated`` is set only by a **cap** (not by files that
  were merely skipped), and ``footerNote`` is this module's one sentence about
  which ones cut — the caps are backend numbers, so the dialog renders that
  sentence and never reassembles it.
"""

import os
import re
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from . import constants, gitstatus, tree
from .policy import WorkspacePathError, resolve_in_workspace
from .policy_modes import mode_for
from .constants import policy_shared as shared


@dataclass
class ScanCounts:
    """What one scan knows about itself, accumulated as the caller consumes it.

    A fixed small record with defaults (the shape ``mcp_connectors/policy_shared``
    uses) rather than a dict, because the two halves of the scan write different
    fields of it and a typo must be an ``AttributeError`` at once. The search's
    payload folds these into ``filesScanned`` and ``skipped``; there is
    deliberately no policy field, for the reason in the module docstring.
    """

    files_scanned: int = 0
    oversized: int = 0
    binary: int = 0
    files_cut: bool = False  # MAX_SEARCH_FILES ended the scan
    bytes_cut: bool = False  # MAX_SEARCH_BYTES ended the scan


def _ignored_hit(rel: str, names: Set[str]) -> bool:
    """Whether ``rel`` is ignored — itself, or because a folder above it is.

    git answers with whatever names its own rules match, and it *collapses* a
    folder whose whole content is excluded to that folder's name (``build/``),
    while an ignore rule that only matches files arrives as the files
    (``src/__pycache__/x.pyc``). Checking the name and each of its ancestors
    makes one rule of both shapes, and it is what lets a *scope* that sits inside
    an ignored folder come back empty even though git only names the folder above
    it.
    """
    if not names:
        return False
    cut = len(rel)
    while cut > 0:
        if rel[:cut] in names:
            return True
        cut = rel.rfind("/", 0, cut)
    return False


class SearchQueryError(Exception):
    """A query that cannot be run, carrying its HTTP status.

    The shape of :class:`~src.core.editorfs.policy.WorkspacePathError` and of
    :class:`core.batch.BatchListingError`: the status belongs to the core
    because these rules *are* the endpoint's contract — 400 is "the pattern you
    wrote is not a pattern", and :attr:`detail` is what the dialog shows inline
    under the field. ``routes`` only re-raises this as an ``HTTPException``.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def iter_scope_text(rel: str, *, counts: ScanCounts) -> Iterator[Tuple[str, str]]:
    """Yield ``(child_rel, decoded_text)`` for every searchable file in scope.

    ``rel`` is workspace-relative ("" is the whole workspace). Lazy on purpose:
    the byte budget can only be honoured while matching *consumes* it, and an
    eager list of every decoded file in a huge checkout is gigabytes. Raises
    :class:`~src.core.editorfs.policy.WorkspacePathError` for the scope **root**
    only — 422 outside the workspace, 404 hidden or absent — and raises nothing
    for a file inside the scope: it is skipped and counted, or it ends the scan.

    Every cap here is reached as ``constants.MAX_…`` at call time, never bound
    into this module's namespace: a bound copy is invisible to the patch that
    exercises it (the same reason ``tree.py`` reads
    ``constants.MAX_TREE_ENTRIES_PER_DIR``).
    """
    normalised, resolved = resolve_in_workspace(rel)
    yield from iter_resolved_scope(normalised, resolved, counts=counts)


def iter_resolved_scope(
    normalised: str, resolved: Path, *, counts: ScanCounts
) -> Iterator[Tuple[str, str]]:
    """The supply for a scope root that is already through the gate.

    Split out because a caller that needs the resolved root for an answer of its
    own — :func:`search_workspace` reports ``scope`` and ``isFile`` from it —
    would otherwise ask the gate twice for one path: the same rules, the same
    errors, and a second read of the deny-list file for nothing.
    """
    ignored = gitstatus.ignored_paths(normalised) or set()
    if _ignored_hit(normalised, ignored):
        # The scope itself is one of the project's ignored names, so *every* file
        # in it is out of the supply. This answers an empty scan rather than the
        # 404 a hidden scope gets: an ignored folder is a row the tree shows, and
        # "nothing matched here" is the truth about it, not an observation.
        return

    if resolved.is_file():
        # A file scope: the same reader a walk would use for this path, and *not*
        # ``files.read_file`` — that raises 413 past ``MAX_EDITOR_BYTES``, which
        # would make one big file an error for a search instead of a skip. The
        # path already passed the gate, so it is inside the workspace, is not
        # hidden, and is a ``realpath``: no symlink is left to worry about here.
        text = _Scanner(counts, ignored).from_resolved_path(resolved)
        if text is not None:
            yield (normalised, text)
        return

    if not resolved.is_dir():
        raise WorkspacePathError(
            404, f"Directory does not exist: {normalised or '.'}"
        )
    yield from _Scanner(counts, ignored).walk(normalised, resolved)


class _Scanner:
    """The running state of one scan: the counts, and the bytes it has spent.

    An object rather than free functions because ``MAX_SEARCH_BYTES`` is a
    *per-scan* total, so that number lives exactly as long as one scan and no
    longer — it is not a field of the payload and must not leak between searches.
    """

    def __init__(self, counts: ScanCounts, ignored: Set[str]) -> None:
        self.counts = counts
        self.ignored = ignored
        self.bytes_read = 0

    def stopped(self) -> bool:
        """True once a cap has ended this scan — checked after every child."""
        return self.counts.files_cut or self.counts.bytes_cut

    def walk(self, rel: str, folder: Path) -> Iterator[Tuple[str, str]]:
        """Depth-first supply of one directory, in :func:`tree.list_dir`'s order."""
        try:
            with os.scandir(folder) as scan:
                entries: List[os.DirEntry] = list(scan)
        except OSError:
            # Unreadable directory: end this branch rather than fail the search —
            # the same answer ``tree._has_children`` gives for the same condition.
            return

        dirs: List[os.DirEntry] = []
        files: List[os.DirEntry] = []
        for scan_entry in entries:
            child_rel = tree._rel_of(rel, scan_entry.name)
            # One verdict, and it is a *prune*, exactly the three lines of
            # ``tree.list_dir``: a hidden folder is neither listed nor walked, so
            # no snippet from inside it can reach the payload. Nothing is counted.
            if mode_for(child_rel).mode == shared.HIDDEN:
                continue
            # The second filter, applied the same way and for the same reason: a
            # name the project's ignore rules exclude is not walked and not
            # opened, so the scan never spends its byte budget on build output or
            # caches. It is not a *policy* answer — an ignored folder is a row the
            # tree shows — so the two stay separate checks, and neither one counts
            # what it pruned.
            if _ignored_hit(child_rel, self.ignored):
                continue
            # ``follow_symlinks=False`` for the same reason the tree uses it: a
            # link to a folder is a leaf here, so the walk cannot leave the
            # workspace one hop at a time.
            if scan_entry.is_dir(follow_symlinks=False):
                dirs.append(scan_entry)
            else:
                files.append(scan_entry)

        # ``tree._name_key`` reads the ``name`` of an entry dict, so it is handed
        # that one field. Re-deriving a key here would give the same tree two
        # orderings, which is what the deterministic-cap rule exists to prevent.
        dirs.sort(key=lambda entry: tree._name_key({"name": entry.name}))
        files.sort(key=lambda entry: tree._name_key({"name": entry.name}))

        for scan_entry in dirs:
            yield from self.walk(
                tree._rel_of(rel, scan_entry.name), Path(scan_entry.path)
            )
            if self.stopped():
                return
        for scan_entry in files:
            text = self.from_entry(scan_entry)
            if text is not None:
                yield (tree._rel_of(rel, scan_entry.name), text)
            if self.stopped():
                return

    def from_entry(self, scan_entry: os.DirEntry) -> Optional[str]:
        """One candidate the walk found: its text, or None when it is not read."""
        if scan_entry.is_symlink():
            # Its target may be outside the working location and ``open()``
            # follows it with nothing in the way: skipped, and counted nowhere.
            return None
        try:
            info = scan_entry.stat(follow_symlinks=False)
        except OSError:
            return None
        if not stat_module.S_ISREG(info.st_mode):
            # A fifo, socket or device: nothing to read, and reading one would
            # block the request that asked for it.
            return None
        if info.st_size > constants.MAX_EDITOR_BYTES:
            self.counts.oversized += 1
            return None
        return self.read(scan_entry.path, size=info.st_size)

    def from_resolved_path(self, path: Path) -> Optional[str]:
        """The same rules for the one path of a *file* scope.

        The path came from :func:`resolve_in_workspace`, so it is already a
        ``realpath`` inside the workspace — which is why this side needs no
        symlink check: a link pointing out of the workspace was refused by the
        gate (422) before anything here ran.
        """
        try:
            info = os.stat(path)
        except OSError:
            return None
        if not stat_module.S_ISREG(info.st_mode):
            return None
        if info.st_size > constants.MAX_EDITOR_BYTES:
            self.counts.oversized += 1
            return None
        return self.read(str(path), size=info.st_size)

    def read(self, path: str, *, size: int) -> Optional[str]:
        """The budget, the read, the sniff and the decode, for one proven file."""
        budget = constants.MAX_SEARCH_BYTES
        if size > budget - self.bytes_read:
            # Nothing (or not enough) left: the scan is over, and this file is
            # not searched in a fragment — a partial file would be reported as if
            # it were the file.
            self.counts.bytes_cut = True
            return None

        try:
            with open(path, "rb") as handle:
                # One byte past the size the stat proved: what makes a file that
                # grew while it was read detectable as a fragment rather than
                # silently searched up to where the read happened to stop.
                raw = handle.read(size + 1)
        except OSError:
            # Vanished or unreadable between the listing and the open. That name
            # was real a moment ago and is not a policy answer, so it is skipped
            # in silence like a symlink is.
            return None

        self.bytes_read += len(raw)
        # Counted when the file is **opened**, not when it yields: a binary file
        # cost the scan its read too, and the footer says how many were read.
        self.counts.files_scanned += 1
        if self.counts.files_scanned >= constants.MAX_SEARCH_FILES:
            self.counts.files_cut = True
        if len(raw) > size:
            # The file grew while it was read: what came back is a fragment.
            self.counts.bytes_cut = True
            return None

        # The same two answers as the read route's head-sniff and decode —
        # deliberately *not* ``files._decode_utf8``: that function's tail-repair
        # branch exists only because a read may be cut mid-character by the editor
        # cap, and a scan never accepts a cut file, so strict is the equivalent.
        if b"\0" in raw[: constants.BINARY_SNIFF_BYTES]:
            self.counts.binary += 1
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            self.counts.binary += 1
            return None


# -- the matcher half: what a match is, and what the payload says ----------------


def _compile(query: str, *, case_sensitive: bool, regex: bool) -> "re.Pattern[str]":
    """The query as a compiled pattern, or a :class:`SearchQueryError`.

    A **literal** query goes through :func:`re.escape`, so ``(``, ``a.c`` and
    ``+`` are needles rather than syntax — and the flags do the rest of the work:
    :data:`re.IGNORECASE` *is* "lower-case both sides", which is why there is no
    ``.lower()`` anywhere in this module. An empty query is refused with its own
    sentence instead of matching every line, which would hand back the first 500
    lines of the workspace and read like a bug.
    """
    if not query:
        raise SearchQueryError(400, "Search query must not be empty")
    pattern_src = query if regex else re.escape(query)
    try:
        return re.compile(pattern_src, 0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        # ``str(exc)`` is re's own message, caret position and all: the user
        # wrote this pattern, so it goes back verbatim rather than as a generic
        # "invalid pattern" (spec 2.3).
        raise SearchQueryError(400, str(exc)) from exc


def _snippet(line: str) -> str:
    """One matched line, trimmed to the snippet cap with a trailing ``…``.

    The cap is named ``MAX_SEARCH_SNIPPET_BYTES`` because that is the unit the
    sibling caps are written in, but the bound here is a *display* one, so the
    cut is on characters: a plain slice is code-point-safe in Python, and what
    would break is encoding to bytes and cutting there.
    """
    limit = constants.MAX_SEARCH_SNIPPET_BYTES
    if len(line) <= limit:
        return line
    return line[:limit] + "…"


def _format_cap(num_bytes: int) -> str:
    """A byte cap in the footer's own words — ``32 MB``, not ``33554432``."""
    for unit, size in (("MB", 1024 * 1024), ("KB", 1024)):
        if num_bytes >= size and num_bytes % size == 0:
            return f"{num_bytes // size} {unit}"
    return f"{num_bytes} bytes"


def _count_word(count: int, singular: str, plural: str) -> str:
    """A count with its noun — the plural is spelled out, not guessed: ``matches``
    is not ``matchs``, and the footer is a sentence a person reads."""
    return f"{count} {singular if count == 1 else plural}"


def _footer_note(
    *,
    files_cut: bool,
    bytes_cut: bool,
    rows_cut: bool,
    oversized: int,
    binary: int,
) -> str:
    """The one sentence about what the payload could not give — ``""`` when nothing.

    Written here and **only** here, because the numbers are the caps in
    ``constants`` and a frontend that composed the sentence would have to know
    them (D8): the dialog renders this text verbatim. Fixed order, walk caps
    first, because they bound *completeness* while the row cap bounds only what
    is *listed*; then the counted skips, which are not truncation at all. The
    closing clause appears only when a cap cut, so "3 files were too big" never
    reads as "the search stopped early".

    Cap values are read at call time from ``constants`` for the same reason the
    scan reads them there: a patched cap must be the one the sentence names.
    """
    parts: List[str] = []
    if files_cut:
        parts.append(
            "searched the first "
            + _count_word(constants.MAX_SEARCH_FILES, "file", "files")
        )
    if bytes_cut:
        parts.append(f"read the first {_format_cap(constants.MAX_SEARCH_BYTES)}")
    if rows_cut:
        parts.append(
            "showing the first "
            + _count_word(constants.MAX_SEARCH_MATCHES, "match", "matches")
        )
    if oversized:
        parts.append(f"{_count_word(oversized, 'file', 'files')} too large to search")
    if binary:
        parts.append(
            f"{_count_word(binary, 'binary file', 'binary files')} skipped"
        )
    if not parts:
        return ""
    note = "; ".join(parts)
    if files_cut or bytes_cut or rows_cut:
        note = f"{note} — refine the query"
    return note


def search_workspace(
    rel: str, *, query: str, case_sensitive: bool = False, regex: bool = False
) -> Dict[str, Any]:
    """Search the scope ``rel`` for ``query``: the whole payload, in one dict.

    The route hands it over unchanged, so the keys and their meanings are the
    contract between this module and the dialog (spec 4): 1-based ``line`` and
    ``column``, ``column`` being the **start of the match** so the caret lands on
    what was searched for; ``matchCount`` counting every match, including rows
    the cap dropped; ``skipped`` carrying exactly ``oversized`` and ``binary``,
    never a policy count (D2/D4 — a count of a hidden name is an observation of
    it, which is what hiding the row exists to prevent).

    The order of the body is the rule: **compile, then walk.** An invalid pattern
    raises before a single file is opened, so a typo costs nothing (D5).
    """
    pattern = _compile(query, case_sensitive=case_sensitive, regex=regex)
    normalised, resolved = resolve_in_workspace(rel)
    is_file = resolved.is_file()

    counts = ScanCounts()
    matches: List[Dict[str, Any]] = []
    match_count = 0
    rows_cut = False
    row_cap = constants.MAX_SEARCH_MATCHES

    for child_rel, text in iter_resolved_scope(normalised, resolved, counts=counts):
        # Line by line, so ``line``/``column`` are what the model Monaco builds
        # from the same text — ``splitlines`` strips the terminator, which is why
        # a CRLF file's columns are not off by one. A needle containing a newline
        # never matches, and that is v1's scope (spec 4 step 5).
        for line_no, line in enumerate(text.splitlines(), start=1):
            # ``finditer`` and not a ``while m := search(...)`` loop: a
            # zero-width pattern (``x*``) makes the manual one spin forever.
            for match in pattern.finditer(line):
                match_count += 1
                if len(matches) >= row_cap:
                    # Rows are dropped, knowledge is not: keep counting.
                    rows_cut = True
                    continue
                matches.append(
                    {
                        "path": child_rel,
                        "line": line_no,
                        "column": match.start() + 1,
                        "snippet": _snippet(line),
                        "matchLength": match.end() - match.start(),
                    }
                )

    truncated = counts.files_cut or counts.bytes_cut or rows_cut
    return {
        "scope": normalised,
        "isFile": is_file,
        "matches": matches,
        "matchCount": match_count,
        "filesScanned": counts.files_scanned,
        "truncated": truncated,
        "skipped": {"oversized": counts.oversized, "binary": counts.binary},
        "footerNote": _footer_note(
            files_cut=counts.files_cut,
            bytes_cut=counts.bytes_cut,
            rows_cut=rows_cut,
            oversized=counts.oversized,
            binary=counts.binary,
        ),
    }
