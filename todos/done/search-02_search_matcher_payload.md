# search-02 — The matcher, the snippet and the payload dict

Turn search-01's supply of `(path, text)` pairs into **rows**: compile the pattern,
find every match with its line and column, build the snippet, fold the counts into the
payload of the lot's Contract, and expose one core entry point
`search_workspace(rel, *, query, case_sensitive, regex) -> Dict[str, Any]`.

Still no route: search-03 puts this behind `GET /api/fs/search`. A test can call the
core directly — that is how it should be tested, and it is what the rest of
`editorfs/` does.

Depends on: **search-01** (the caps in `constants.py`, `iter_scope_text`, `ScanCounts`).
Read `docs/tree-search-spec.md` §2.3, §2.4 and §4 steps 5-6; decisions D5, D8, D9,
D16 and *Traps* §6 come from `todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `src/core/editorfs/search.py` | the pattern, the rows, the payload, `SearchQueryError`, `search_workspace` |
| `src/core/editorfs/__init__.py` | export both names, and one bullet in the module docstring's per-file list |
| `tests/test_editorfs_search.py` | extend with the matching and payload rows |

## What to deliver

### 1. `SearchQueryError`

Follows `WorkspacePathError` exactly — same two attributes, same reason (the status
*is* the core's contract, so `routes` only re-raises; `batch.BatchListingError` is the
same shape and worth reading as the model). 400 for an invalid pattern.

### 2. The pattern, compiled **before anything is walked**

```python
pattern_src = query if regex else re.escape(query)
pattern = re.compile(pattern_src, 0 if case_sensitive else re.IGNORECASE)
```

`re.error` is caught around the compile and re-raised as
`SearchQueryError(400, str(exc))` — `str(exc)` is `re`'s own message and must reach the
client verbatim, because the user wrote the pattern and needs the caret position back
(§2.3). A bad pattern costs **no scan**: compile first, walk second, in that order in
the function body, and a test proves the order rather than trusting it (see Tests).

Two notes on the two flags:

- `caseSensitive: false` means lower-case both sides — `re.IGNORECASE` is exactly that,
  and it is why there is no `.lower()` anywhere in this module.
- a **literal** query goes through `re.escape`, so `(`, `a.c` and `+` are needles, not
  syntax. The spec's own row for this: a query of `a.c` must not match `abc` with
  `regex: false`. An empty `query` is a 400 from the same error type with its own
  sentence — not a scan that matches every line, which would return the whole
  workspace's first 500 lines and be indistinguishable from a bug.

### 3. The rows

Per file from `iter_scope_text`, per line (split the text with `splitlines()`), per
`pattern.finditer(line)` (never a manual search loop — *Traps* §6):

```python
{ "path": child_rel, "line": line_no, "column": match.start() + 1,
  "snippet": snippet, "matchLength": match.end() - match.start() }
```

- `line` and `column` are **1-based**; `column` is the **start of the match**, not the
  start of the line — the caret must land on what was searched for (see the Contract).
- **Multiple matches on one line are separate rows**, each with its own column.
- `matchCount` counts every match found, **including the ones that never become rows**
  (the row cap drops rows, not knowledge).
- rows past `constants.MAX_SEARCH_MATCHES` are not built — but keep counting matches,
  and set the row-cut flag. Do not `break` out of everything at the cap: `matchCount`
  would then be a lie.
- `snippet` is the whole line with its terminator stripped, `…` (U+2026) appended when
  it is longer than `constants.MAX_SEARCH_SNIPPET_BYTES`. Trim on **characters**, and
  say in the comment that the constant is named `…_BYTES` because it is the same unit
  the sibling caps use — a snippet is a *display* bound, not a byte bound on a file.
  Never slice mid-surrogate-pair: cut at the last code point boundary at or below the
  limit (a plain `snippet[:n]` is already code-point-safe in Python; what is *not*
  safe is encoding to bytes and cutting there — do not do that).
- the full line is what the row's `title` will show in the dialog (search-06's job);
  the payload sends the trimmed snippet only, plus — if you want to keep search-06
  honest — nothing else. Do not add a `fullLine` key: §4's payload does not have one,
  and a `title` that repeats the snippet is an accepted loss (§6's "not tested, and
  why" calls the tooltip cosmetic).

### 4. The payload and the footer

Return exactly the Contract dict — `scope` (the normalised root, `""` for the whole
workspace), `isFile`, `matches`, `matchCount`, `filesScanned`, `truncated`, `skipped`
(only `oversized` and `binary`; **no policy key**, D2/D4), `footerNote`.

`truncated` is true when any **cap** cut: the row cap, `MAX_SEARCH_FILES`, or
`MAX_SEARCH_BYTES` (D9). Counted skips alone do not set it — a scan that read every
file and skipped three big ones is complete.

`footerNote` is one sentence, built in the fixed order the map gives (walk caps, then
the row cap, then the two skip counts, then `— refine the query` when a cap cut), from
the `constants` values read at call time. This module owns that sentence **once**;
search-06 renders it verbatim and must not reassemble it (D8). Keep the builder a small
pure function of the counts so it is testable without a filesystem.

### 5. The export

`editorfs/__init__.py`: add `search_workspace` and `SearchQueryError` to the
`from .search import …` line, **and** one bullet to that module docstring's per-file
list, which enumerates one per file (`tree`, `files`, `mutate`, `gitstatus`, …). Write
the bullet in that file's voice: what the module answers and which invariants it
inherits ("one verdict per child, the same prune as `tree`", "the caps it respects"),
not a list of its functions.

The caps reach outside callers through `__init__`'s existing PEP-562 `__getattr__`, so
no cap name is added to the `TYPE_CHECKING` block unless you annotate against it —
leave that block alone.

## Tests — extend `tests/test_editorfs_search.py`

| test | asserts |
|---|---|
| literal, case-insensitive | `Foo`, `foo`, `FOO` all match, `matchCount` right |
| case-sensitive | only the exact case matches |
| literal ≠ regex | `a.c` does not match `abc` with `regex: false`, and does with `regex: true` |
| regex meta-chars in a literal query | `(` does not raise, and matches a literal `(` |
| invalid regex | 400, `re`'s message in the error's `detail`, **and no scan happened**: put a file whose contents match the *prefix* of the pattern somewhere a walk would find it, then assert the raised error is `SearchQueryError` and that `ScanCounts.files_scanned` is `0` (the spec's sentinel row — assert the count, not the absence of results, because an absent result proves nothing) |
| empty query | 400 from the same type, no scan |
| line/column/matchLength | 1-based; `column` is the match start; a second match on the same line is a second row with its own column; a zero-width pattern gives `matchLength: 0` and one row per line position, without hanging (*Traps* §6) |
| `matchCount` vs rows | patch `MAX_SEARCH_MATCHES` to 3 in a workspace with 5 matches ⇒ 3 rows, `matchCount: 5`, `truncated: true` |
| snippet | trimmed at `MAX_SEARCH_SNIPPET_BYTES` with `…` present; a short line is untrimmed; a line with a non-ASCII character near the cut keeps its characters intact |
| CRLF file | columns match the text with `\r` stripped — a match after a CRLF line is not off by one |
| footer order | two caps cut ⇒ both named, in the map's order, and `— refine the query` present; skips with no cap cut ⇒ footer names the skips and `truncated` is **false** (D9); nothing cut and nothing skipped ⇒ `footerNote == ""` |
| payload keys | `set(payload) == {"scope", "isFile", "matches", "matchCount", "filesScanned", "truncated", "skipped", "footerNote"}` and `"policy" not in payload["skipped"]` — assert the exact key set spelled out, not `>=`-style containment, so a future key is a deliberate act |
| `isFile` / `scope` | a file scope answers `isFile: true` and `scope` equal to the normalised path; the root scope answers `scope: ""` |

## Acceptance

- `pytest tests/test_editorfs_search.py tests/test_editorfs.py tests/test_api.py` green.
- `editorfs.search_workspace` and `editorfs.SearchQueryError` are reachable as
  attributes of the package (that is what search-03 will call), and
  `editorfs.MAX_SEARCH_MATCHES` still resolves through the PEP-562 `__getattr__`.
- `pytest tests/test_write_surface_isolation.py` green, and
  `grep -c "PUT /api/fs\|POST /api/fs\|DELETE /api/fs" src/core/editorfs/search.py`
  is still `0` (D16). Note `TestSurfaceShape` in `test_editorfs.py` pins the *exported
  write* names as an allowlist of the four write helpers and asserts a list of names
  absent — `search_workspace` is neither, so it passes; if it does not, you have named
  the entry point like a writer and must fix the name, not the test.
- `git diff --stat` shows exactly three files: `search.py`, `__init__.py`, the test
  file. No route, no frontend file.
- One line in the final note giving the exact `footerNote` wording you settled on,
  since search-06 asserts against it.
- Move this file to `todos/done/` when the above is true.
