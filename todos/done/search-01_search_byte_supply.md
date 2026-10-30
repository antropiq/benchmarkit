# search-01 — The scan's byte supply: walk, verdicts, caps, reader

Build the half of the search that decides **which files are read and which bytes
reach a matcher**: resolve the scope root, walk the subtree in the tree's own order,
ask one policy verdict per child, prune hidden names, skip what is too big or not
text, and honour the byte budget. No matching yet, no payload, no route — search-02
consumes what this yields.

Depends on: nothing (first task of the lot). Read `docs/tree-search-spec.md` §2.4,
§2.5 and §4 steps 1-4 first; decisions D2, D3, D4, D6, D7 and *Traps* §1 come from
`todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `src/core/editorfs/constants.py` | the four new caps, beside the existing ones |
| `src/core/editorfs/search.py` | **new** — the scanner |
| `tests/test_editorfs_search.py` | **new** — the rows of §6 that are about bytes |

Do **not** touch `editorfs/__init__.py`, `api/routes.py` or any frontend file: the
export and the route are search-02's and search-03's. `search.py` is importable as
`from src.core.editorfs import search` — that is enough for this task's tests.

## What to deliver

### 1. The caps (`constants.py`)

Add four module attributes with the file's own commenting style (each says *what* it
bounds and *why*, in the voice of `MAX_TREE_ENTRIES_PER_DIR` / `MAX_STATUS_ENTRIES`,
which are named as the siblings in §2.4):

```python
MAX_SEARCH_MATCHES = 500      # rows in the payload; extra matches counted, not sent
MAX_SEARCH_FILES = 5000       # files opened by one scan, whatever their size
MAX_SEARCH_BYTES = 32 * 1024 * 1024   # total bytes read by one scan
MAX_SEARCH_SNIPPET_BYTES = 240  # one snippet, before the ellipsis
```

Values exactly as §2.4's table. They belong here, not in `search.py`, because a test
patches the *name* and `editorfs/__init__.py`'s PEP-562 `__getattr__` already resolves
cap names out of `constants` for callers.

### 2. The scanner (`search.py`)

Core-only: plain values in, no FastAPI, no `HTTPException`. One public entry point for
this task, plus the counts object search-02 will fold into its payload:

```python
@dataclass
class ScanCounts:
    files_scanned: int = 0
    oversized: int = 0
    binary: int = 0
    files_cut: bool = False    # MAX_SEARCH_FILES stopped the walk
    bytes_cut: bool = False    # MAX_SEARCH_BYTES stopped the read

def iter_scope_text(rel: str, *, counts: ScanCounts) -> Iterator[Tuple[str, str]]:
    """Yield ``(child_rel, decoded_text)`` for every searchable file in scope."""
```

(`dataclass` is used by `mcp_connectors/policy_shared.py` for the same reason — a
fixed small record with defaults. A plain dict is equally fine; pick one and annotate
it.)

Behaviour, in §4's order:

1. `resolve_in_workspace(rel)` **once**, for the scope root: 422 outside the workspace,
   404 for a hidden or absent scope. If the root `is_file()`, yield that one path
   through the same reader as step 4 — *not* through `files.read_file`, which raises
   413 past `MAX_EDITOR_BYTES` and would answer a file-scope search of a big file with
   an error. A file scope and a folder scope must agree: in both, the file is skipped
   and counted in `oversized`. The one difference is the gate: the *path* of a file
   scope still 404s when hidden, so a hidden file cannot be searched by name.
2. A depth-first `os.scandir` walk, in `tree.py`'s order: directories first, then
   files, each group sorted by `tree._name_key`. Import and reuse that helper — the
   exact-name tie-break is not optional, it is what makes a truncated result the *same*
   N rows every run (D6).
3. One `mode_for(child_rel)` verdict per child, with the **same three lines as
   `tree.py:102`**: `hidden` ⇒ `continue` (a directory is pruned — no `scandir` below
   it — and a file is omitted), and *no counter* is touched for either (D2/D4).
   `ro` falls through and is searched.
4. Per candidate file (`*Traps* §1` — this is the security-relevant line):
   - `scan_entry.is_symlink()` ⇒ skip it entirely, silently, counted nowhere (D7).
     A symlink's target may be outside the working location, and `open()` follows it
     with nothing in the way; the tree's `realpath` lock only protects paths that go
     through `resolve_in_workspace`, which a walk does not do per child.
   - `stat(follow_symlinks=False)`; a non-regular file (fifo, socket, device) is
     skipped like a symlink is.
   - size over `constants.MAX_EDITOR_BYTES` ⇒ `counts.oversized += 1`, next file. No
     exception, ever.
   - budget: `MAX_SEARCH_BYTES` is a per-scan total. Read at most what is left; once
     nothing is left, set `counts.bytes_cut = True` and **stop supplying files** (the
     caller's loop ends). A file that would overrun the remaining budget is not
     partially searched — a fragment of a file would be reported as if it were the
     file.
   - open, read, NUL-sniff the first `constants.BINARY_SNIFF_BYTES` bytes (the same
     window `files.py:176` uses), then `bytes.decode("utf-8")` **strict**; a
     `UnicodeDecodeError` ⇒ `counts.binary += 1`, next file. This is deliberately the
     same two answers as `files._decode_utf8` on a *untruncated* read (that function's
     tail-repair branch only exists because a read can be cut mid-character by the
     editor cap; a scan never cuts a file it accepts, so a strict `decode` is the
     honest equivalent — say so in one comment line, because the next reader will
     otherwise wonder why `_decode_utf8` was not simply called).
   - yield `(child_rel, text)` — `child_rel` built with `tree._rel_of`, i.e. the
     normalised workspace-relative form, never an absolute path.
   - `counts.files_scanned += 1` when the file is opened (not when it yields), and
     `counts.files_cut = True` when `files_scanned` reaches `MAX_SEARCH_FILES`.

Read the caps as `constants.MAX_…` **at call time**, never
`from .constants import MAX_…` — a bound copy is invisible to a patch (D3).

## Tests — `tests/test_editorfs_search.py`

New file, modelled on `tests/test_editorfs.py`.

The `workspace` and `client` fixtures there **cannot be imported** — pytest fixtures do
not travel between test modules — so re-declare small local ones here (a `tmp_path`
pointed at by `BENCHMARKIT_PATH_LOCATION`, seeded with `policy_seed.ensure_policy_file`
and the cache invalidated). That is expected, not duplication to avoid.
`write_policy` and the `tree`/`read`/`names` helpers **are** plain functions and can be
imported from `tests.test_editorfs`, or copied — either is fine.

Assert on the **sequence of yielded paths** (`[p for p, _ in iter_scope_text(…)]`) and
on the counts. Cover, one test each:

| test | asserts |
|---|---|
| root scope | a file at the root **and** one two folders down are both yielded |
| folder scope | a descendant two levels deep is found; a sibling folder's file is not |
| file scope | yields exactly that path; a sibling file with the same name pattern is absent |
| file scope, oversized | a 1.5 MB file answers `oversized == 1` and yields nothing — **not** a 413 |
| hidden folder pruned | write a `hidden` rule with the `write_policy` helper's shape; a file matching the query inside it is absent **and every counter is still zero** — this is D2's "never counted", assert it as loudly as the absence |
| hidden file omitted | a `hidden` file yields nothing, counters zero |
| `ro` searched | an `ro` file's path **is** yielded |
| `.nojae` never searched | a match inside `.nojae/` is absent (seeded workspace, no rule written) |
| symlinked dir not followed | a link to a folder outside the root is not descended |
| symlinked file not opened | a link to a file outside the root whose target contains the needle is **not yielded** and is **not counted anywhere** (`*Traps* §1`; this is the row the spec does not have) |
| binary skipped | a NUL-leading file is absent, `binary == 1` |
| non-UTF-8 skipped | the same answer as `read_file`'s `binary: true` — a latin-1 file is skipped, `binary == 1` |
| walk order | `B/a.py`, `b/a.py`, `Z.py`, `a.py` come out dirs-first, each sorted case-insensitively, with the exact-name tie-break visible (`A.py` before `a.py`) |
| `MAX_SEARCH_FILES` | patch it to 2, assert two files scanned, `files_cut` true, and that the *first two in walk order* are the ones supplied |
| `MAX_SEARCH_BYTES` | patch it small, assert the cut flag, that every yielded file is whole, and that no yielded file is a fragment |

Patch caps as `monkeypatch.setattr(editorfs.constants, "MAX_SEARCH_FILES", 2)` — never
by writing 5000 files. Patch `MAX_EDITOR_BYTES` for the oversized rows, since that is
the cap a scan respects (§2.4).

## Acceptance

- `pytest tests/test_editorfs_search.py tests/test_editorfs.py` green. The whole
  existing editorfs suite must be untouched: this task adds caps and one module, and
  changes no existing behaviour.
- `git diff --stat` shows exactly three files: `constants.py`, the new `search.py`,
  the new test file. Nothing else.
- `grep -n "from .constants import\|from src.core.editorfs.constants import" src/core/editorfs/search.py`
  finds no import of a cap name. (The two `policy_shared`/`policy_glob` mirror imports
  `tree.py` uses are fine; a *cap* must not be bound.)
- `grep -c "PUT /api/fs\|POST /api/fs\|DELETE /api/fs" src/core/editorfs/search.py` is
  `0`, and `pytest tests/test_write_surface_isolation.py` is green (D16: the marker
  scan walks all of `src/core`, so the module's prose is part of the contract).
- No route, no export, no frontend file: the feature is unreachable from the app at
  this point. Say so in the final note rather than wiring something to prove it works.
- Move this file to `todos/done/` when the above is true.
