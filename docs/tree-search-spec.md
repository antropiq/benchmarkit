# Search inside the workspace tree (Editor tab)

**Status**: spec written, implementation not started.
**Surface**: the Editor tab only — the tree's context menu, one modal, one new
read-only route, one new core module. No LLM, no hub, no SSE.

## 1. The need

The Editor tab can browse a workspace and open one file at a time, but it cannot
answer "where is this string?". The only way to find out today is to expand
branches until the right file is visible and open it. For a working location
that is a real checkout, that is slower than the terminal the tab already
offers — and the terminal is a human surface, not something the results can
link back into.

So: pick a row, search for text, get a list of matches, click one and land on
the line in a buffer.

The three scopes come from the row the menu was opened on, which is the row the
menu already acts on:

| row clicked | searched |
|---|---|
| a folder (`docs/old`) | that folder's whole subtree |
| the root row | every accessible file of the project |
| a file (`src/config.py`) | that file's content |

## 2. Decisions taken

### 2.1 Results live in the modal, not in a tab

The first instinct was a fourth tab kind in the centre strip ("Search results"
between a file, its diff and its preview). Rejected.

The strip's tabs are *records* in `useEditorFiles.js`, and every one of them is
branched on by kind in five places — `keyFor`/`labelFor`, `dirtyCount`, the
freshness `probeIsUseful`, `repointPath` (rename re-points open tabs) and
`forgetWorkspaceBuffers` (a working-location switch closes them all). A search
record would have to be added to all five, and would still be wrong: a tab sits
there showing hits from before a rename or a delete, and `repointPath` does not
know to fix them.

A `<dialog>` is a snapshot the user dismisses and re-opens, so it cannot go
stale the way a tab does, and it costs none of that plumbing. A **top-level**
tab beside Benchmark / WorkBench / Editor was rejected for a different reason:
every top-level tab in this app shares one LLM configuration and the SSE layer.
Search shares neither.

Clicking a hit therefore reuses the existing `openFile(path)`
(`useEditorFiles.js:261`), with one addition — see 2.6.

**Rejected alternative**: one row per *file* (with a hit count) instead of one
row per *match*. Rejected because a file list the user cannot jump into is not a
search: the click would open a tab scrolled nowhere. One match per row, capped.

### 2.2 One blocking request, no stream

`/api/fs/*` are plain request/response today, and that stays true. Streaming
live hits would need its own `EventBus`, its own terminal event and a decision
about the "one run at a time" rule in `RunState` — all for a scan that is
bounded by the caps in 2.4 and finishes in well under a second on a real
checkout. The price is that a large scan shows a spinner and no partial
results; 2.4's footer is what makes that honest rather than silent.

### 2.3 The form: query, case, regex — nothing else

Three controls, exactly:

| control | widget | server param |
|---|---|---|
| the text | one `<input>`, autofocused | `q` |
| case sensitive | checkbox, default **off** | `caseSensitive` |
| regex | checkbox, default **off** | `regex` |

`caseSensitive` off means the server lowercases both needle and haystack — the
meaning of "Uppercase, mixed or lowercase all match".

**Regex is evaluated by Python `re`, not by JavaScript's.** The search is
server-side, so the honest thing is to say where the pattern runs (a tooltip:
"regex — Python syntax") rather than to promise parity with a JS engine the
result did not come from. An invalid pattern is a **400 carrying `re`'s own
message**, shown inline under the field — not a generic error — because the
user wrote that pattern and needs the caret position back.

**Rejected: a whole-word toggle.** Rejected for v1: it is one more control for a
rarer need, and with `regex` on the user can write `\b…\b` themselves.

**Rejected: searching file *names*.** Content only for v1. A name search is a
different question ("which file is this?"), answered mostly by the tree the user
is already standing in, and mixing the two into one result table makes the table
ambiguous.

### 2.4 Two different truncations, and the caps behind them

The word "truncated" in the sketch covers two unrelated events. They get two
different treatments, because one is about a *cell* and the other about the
*result set*:

| | meaning | how it shows |
|---|---|---|
| snippet trim | the matched line is longer than the column | `…` inside the cell; the full line is in the row's `title` |
| result set cut | the scan hit a cap and stopped early | one footer line: "showing the first 500 matches — refine the query" |

The caps live in `core/editorfs/constants.py` beside `MAX_TREE_ENTRIES_PER_DIR`,
`MAX_EDITOR_BYTES` and `MAX_STATUS_ENTRIES` — not in the feature module. That is
where every other Editor cap is defined, and the reason is stated in that file's
docstring: a test patches the *name*, so `search.py` must read them as
`constants.MAX_SEARCH_MATCHES` at call time (the way `tree.py` reads
`constants.MAX_TREE_ENTRIES_PER_DIR`), never bind them with
`from .constants import …`, which would freeze a copy a patch cannot reach.

| constant | value | bounds |
|---|---|---|
| `MAX_SEARCH_MATCHES` | 500 | rows in the payload; extra matches counted, not sent |
| `MAX_SEARCH_FILES` | 5000 | files opened, whatever their size |
| `MAX_SEARCH_BYTES` | 32 MB | total bytes read by one scan |
| `MAX_SEARCH_SNIPPET_BYTES` | 240 | one snippet, before the `…` |

Two numbers a file scan must respect but must **not** inherit as errors:

- **A file over `MAX_EDITOR_BYTES` (1 MB, `editorfs.constants`) is skipped and
  counted in the footer.** It is *not* a 413. `read_file` raises 413 because a
  human asked for that one file and deserves to know why it did not open; a scan
  has no right to fail the whole search because one artefact is large.
- **A binary file is skipped**, using the existing NUL-in-first-
  `BINARY_SNIFF_BYTES` (8192) sniff and the same "not valid UTF-8 ⇒ binary"
  answer as `files._decode_utf8`. Its name is never in the results.

These caps are the sibling rule of `MAX_TREE_ENTRIES_PER_DIR` (2000) and
`MAX_STATUS_ENTRIES` (5000), which already answer `truncated: true` instead of
dropping rows silently — the same honesty rule as `tokensEstimated` on a stream.

### 2.5 A search sees exactly what the tree sees — no more

This is the part with real consequences, because a search returns *content*, and
content is what the policy exists to protect.

The rule, stated once: **a search's visible set is the tree's visible set, and
its readable set is the read route's readable set.** No new verdict, no new
matcher — the scan walks with the same two calls every other Editor surface
uses.

"Visible" here means the policy's answer about a path, not the rows a listing
happened to render. `list_dir` additionally caps one directory at
`MAX_TREE_ENTRIES_PER_DIR` (2000) and flags the listing `truncated`; that is a
bound on what a *screenful* of rows costs, not a statement about access. The
scan does not inherit it — its own caps (2.4) govern instead — so a folder with
3000 files is searched in full even though the tree showed 2000 of them. Same
rule for a collapsed branch: the tree has not listed it, the search still walks
it.

| path mode (`policy_modes.mode_for`) | searched? | why |
|---|---|---|
| normal | yes | |
| `ro` | **yes** | `ro` constrains what *changes*, not what is seen — `read_file` reads it, so the tree promised this row |
| `hidden` | **no — pruned, never listed, never matched** | see below |

Two promises that follow, and both are already this app's law elsewhere:

1. **A `hidden` directory is pruned, not walked** — one verdict on its own name
   and no `scandir` below it (`tree.py`, pol-00). The reason a *search* has to
   obey it harder than the tree does: a tree row is a name, but a search result
   is **the bytes from inside the file**. Returning a snippet from `.env` would
   put a secret in a payload and then in a DOM node. Hiding the row is worthless
   if the content is still quotable.
2. **An observation of a hidden path is indistinguishable from absence** (pol-00
   D8, `policy.resolve_in_workspace` ⇒ 404). So a search of `docs/old` where
   `.env` sits inside answers as if the file were never there — no "3 files
   skipped by policy" counter that quietly confirms a hidden name exists.
   Skipped-by-size and skipped-by-binary are counted; **skipped-by-policy is not
   counted at all**, because a count of them is an observation.

That covers `.nojae/` (reserved, hidden in every workspace, no rule can unlock
it), level A's `**/.env**` and `logs/**` (live only while the workspace *is* the
install dir — another project's `.env` is an ordinary file and *is* searchable,
by design), and every level-B rule in `<workspace>/.nojae/read-write-policy.json`.

**Symlinked directories are listed at one level and never followed** — the
`tree.py` rule, inherited verbatim, so a scan cannot crawl out of the working
location one hop at a time.

**Rejected: `os.walk` + the connector deny-list** (what `core/batch.py` does for
`todos/`). Rejected because it would be a *second* policy answer for the same
path, and the two levels disagree by construction: level A is the app's own rule
about its own generated files, the connector deny-list is per-machine. The
Editor surface has one verdict (`policy_modes.mode_for`) and search must use it.

**Rejected: `rg`/`grep` as a subprocess.** Rejected: the subprocess would answer
the policy question itself, in a second implementation, and its *file names*
would reach the payload before any filter could prune them. The in-process scan
is slower and correct.

### 2.6 A hit opens the file *at the line*

Clicking a row does two things, in this order:

1. publishes the path as the tree's selection — the existing
   `select(entryFor(path))` flow, so `EditorArea`'s watcher on `tree.state.selected`
   (`EditorArea.vue:116`) opens or focuses the buffer and its refused-open answers
   (binary, over the cap) are already the pane's own words;
2. asks the pane to move the caret to that line.

Step 2 is the only genuinely new frontend plumbing, and it is small:
`MonacoEditor.vue` has **no `defineExpose` today** — it is a props-in /
`content-change`-out component with no imperative surface. It gains one exposed
method, `revealPosition({ line, column })`, calling
`editor.revealLineInCenter(line)` + `editor.setPosition({ lineNumber, column })`,
plus a `pendingReveal` ref, because the timing is not the click's timing:

- the file may not be open yet, so the surface may not exist when the row is
  clicked — the reveal is queued and applied in `onMounted`;
- when the file *is* already open, `EditorArea` keeps **one** `MonacoEditor`
  instance across tab switches (`:content="tab.text"`, no `:key`), and the
  component's watcher on `props.content` calls `model.setValue`, which the
  component's own comment says **clears the undo stack and the cursor**. So a
  reveal must be applied *after* that watcher has run (a `nextTick`-flushed
  queue), never synchronously from the click, or the swap erases it.

The column is the **start of the match**, not the start of the line — so the
caret lands on what was searched for, and Monaco's own current-line highlight
(`renderLineHighlight: 'line'`) does the rest. No decorations are drawn: a
highlighted-match overlay across an entire file is Monaco's own find widget's
job, not this one's.

**The dialog closes on the click.** A `showModal()` dialog covers the pane, so a
result the user cannot yet look at is a result they cannot use. Rejected:
keeping the dialog open as a side panel — it needs a non-modal placement, a
second resize axis and a way to dismiss it that is not Escape.

### 2.7 The popup keeps its last answer

Results live in a composable singleton (`useWorkspaceSearch`), not in the
component, so they survive the dialog closing, a re-open, and a switch to the
WorkBench tab and back — the same reason `useWorkspaceTree` holds the tree
state ("the selection lives *here*, not in the component, so a tab switch
cannot lose it"). Re-opening the popup shows the previous query (still in the
field), the previous scope (still in the label) and the previous table.
Nothing is re-fetched on re-open: the answer is only replaced by a *new* search.

Two consequences, both wanted:

- a stale result is allowed. If the file was renamed or deleted meanwhile, the
  click produces the pane's **existing** refused-open answer (the 404 reason
  echoed with the path, `useEditorFiles.js:278`). No staleness machinery,
  because the flow already reports it honestly.
- a **working-location switch clears it**. The results are paths relative to a
  root that no longer exists; showing them under the new root's label would be a
  lie. This reuses the one-way registry the buffers already use:
  `registerReRootHandler` (`useWorkspaceTree.js:151`), which
  `useEditorFiles.js:755` registers the buffers through — so the search
  composable forgets query, scope and matches on the same event that closes the
  tabs.

### 2.8 The scope label is read-only, and it names what is being searched

One line under the title, before the field, not an input:

```
Search in: docs/old            (folder)
Search in: src/config.py       (file)
Search in: benchmarkit         (whole workspace)
```

The root row has no relative path — it *is* `""` — so it cannot render as an
empty label. It renders the workspace's own name, which is what the tree's root
row already does via `rootLabel()` (`useWorkspaceTree.js:683`, basename with the
absolute path in `title`), and the kind suffix says which of the three scopes is
in play. The label is derived from the row the gesture came from, never typed.

### 2.9 Two entry points, one subject

**Context menu (primary).** The tree's menu is data — one ordered `ITEMS` table
in `web/src/lib/tree-menu.js` where `scope` says which rows an item exists for.
Search is one row:

```js
{
  id: 'search',
  label: 'Search…',
  title: 'Search inside this folder, this file, or the whole workspace',
  scope: 'row',
}
```

`scope: 'row'` is already file **and** folder **and** root (`inScope` returns
true for every row), and there is no `needs` flag because search is honest for
every row — unlike `Close` (needs a buffer) or `View diff…` (needs a committed
side). So the three scope cases from §1 need no new menu machinery; they are the
existing scope rule. The `…` follows the table's own convention for "asks for
something" (`Rename…`, `View diff…`).

**Icon (secondary), in the tree header — not beside the tab strip.** The tab
strip's right edge does have the shape remembered (`.editor-save-actions`,
`EditorArea.vue:348`, icon+title+aria-label buttons kept outside the scrolling
strip on purpose). But the tree header already has its own pair of the same
buttons — `.editor-nav-actions` with refresh and collapse
(`WorkspaceNavigator.vue:362`) — and Search belongs there: the dialog's scope
label needs the navigator's own subject (`commandTarget()` / the focused row),
and mounting the dialog next to the button that raises it avoids one component
reaching into another's state. It is enabled only when a row is focused, with
the same disabled-when-nothing-to-do shape as the collapse button.

**No keyboard chord in v1.** `lib/editor-keys.js` owns exactly the chords the
app claims (Ctrl+S, Ctrl+W) and refuses to take keys it has no answer for.
Ctrl+F is the browser's own find; taking it would be a chord the user cannot
turn off. Rejected for v1, revisit if a `Ctrl+Shift+F` shape feels needed.

## 3. What it builds on

| existing piece | reused for |
|---|---|
| `lib/tree-menu.js` `ITEMS` / `inScope('row')` | the menu entry, all three scopes |
| `WorkspaceNavigator.vue` `MENU_ACTIONS`, `commandTarget()` | the action receives the clicked row, same subject as every other item |
| `EditorTreeDialogs.vue` / `BatchModeDialog.vue` | `<dialog>` + `showModal()` over composable state |
| `useEditorFiles.openFile(path)` | opening/focusing the buffer for a hit, incl. its existing refused-open answers |
| `useWorkspaceTree.registerReRootHandler` | clearing results on a working-location switch |
| `useWorkspaceTree.rootLabel()` | the whole-workspace form of the scope label |
| `policy.resolve_in_workspace` / `policy_modes.mode_for` | every path decision the scan makes |
| `files._decode_utf8`, `BINARY_SNIFF_BYTES` | binary detection, with the same answer |
| `.editor-nav-actions` / `.icon-btn` | the icon button, its title/aria-label shape |

## 4. Shape and files of the new pieces

### `src/core/editorfs/search.py` (new)

Core-only, no FastAPI — the same discipline as the rest of `editorfs/`: it takes
plain values and returns a dict.

```python
def search_workspace(
    rel: str, *, query: str, case_sensitive: bool, regex: bool,
) -> Dict[str, Any]: ...
```

Algorithm:

1. `resolve_in_workspace(rel)` once for the scope root — 422 outside the
   workspace, 404 for a hidden scope. A **file** scope skips the walk and scans
   that one path — **not** through `files.read_file`, which raises 413 past
   `MAX_EDITOR_BYTES` and would answer a file-scope search of a big file with an
   error. It uses the scan's own reader (the step-4 rules), so a file scope and a
   folder scope agree about what is too big to read: in both cases the file is
   skipped, counted in `skipped.oversized`, and the search answers. The one
   difference is the gate: the *path* of a file scope still 404s if hidden, so a
   hidden file cannot be searched by name even when its row is not visible.
2. A depth-first `os.scandir` walk of the subtree, in the tree's own order
   (directories first, then files, case-insensitive alphabetical — a truncated
   result must be the *same* first 500 rows every time, not whichever inode
   answered first). That order is `tree.py`'s: `dirs + files` per directory,
   each sorted by `_name_key` = `(name.lower(), name)` — the exact-name tie-break
   matters, because two files differing only in case otherwise sort
   arbitrarily and the cap would become non-deterministic. Reuse the existing
   helper rather than re-deriving a key.
3. One `mode_for(child_rel)` verdict per child. `hidden` ⇒ pruned if a
   directory, omitted if a file — **the same three lines as `tree.py`**, and a
   symlinked directory is never descended.
4. Per candidate file: `stat` size — skip over `MAX_EDITOR_BYTES`; open, read at
   most the bytes left in the `MAX_SEARCH_BYTES` budget; NUL sniff; decode
   strict UTF-8 (failure ⇒ binary ⇒ skip).
5. Match line by line: `pattern = query if regex else re.escape(query)`,
   compiled once as `re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)`.
   Invalid pattern ⇒ `re.error` caught at the top, before any walk, re-raised as
   `SearchQueryError(400, str(exc))` so a bad pattern costs no scan.
6. Per match: `{path, line (1-based), column (1-based), snippet, matchLength}`.
   Snippet is the whole trimmed line, `…`-appended at `MAX_SEARCH_SNIPPET_BYTES`.
   Multiple matches on one line are separate rows.

`SearchQueryError` follows `WorkspacePathError` exactly: it carries its HTTP
status because the status *is* the core's contract, and `routes` only re-raises
it (`batch.BatchListingError` is the same shape).

Exported through `core/editorfs/__init__.py` alongside `list_dir` and
`read_file` (and given a bullet in that module's docstring list, which enumerates
one per file), so `routes` calls `editorfs.search_workspace(...)`. The caps reach
callers through `__init__`'s existing PEP-562 `__getattr__`, which resolves names
from `constants` — another reason they belong there and not in `search.py`.

### `GET /api/fs/search` (new route in `api/routes.py`)

```
GET /api/fs/search?path=docs/old&q=verb&caseSensitive=false&regex=false
```

```json
{
  "scope": "docs/old",
  "isFile": false,
  "matches": [
    { "path": "docs/old/notes.md", "line": 12, "column": 8,
      "snippet": "…the verb of the sentence…", "matchLength": 4 }
  ],
  "matchCount": 517,
  "filesScanned": 61,
  "truncated": true,
  "skipped": { "oversized": 3, "binary": 4 }
}
```

- `truncated` is true when *any* cap cut the scan; the footer names which one.
- `skipped` counts only the two honest-by-size reasons. There is deliberately no
  `skipped.policy` key — see 2.5.
- Query params use the camelCase the rest of `/api/fs` uses (`ifUnmodifiedSince`).
- **A `GET`, so it is not a write.** `tests/test_write_surface_isolation.py`
  scans for the markers `PUT /api/fs`, `POST /api/fs`, `DELETE /api/fs` and
  `GET /api/fs` is *deliberately* not one of them (its own comment: chosen so a
  docstring naming a route method stays out of the scan). So no `ALLOWED_HOMES`
  entry is needed and the guard keeps passing — with one caveat for the
  implementer: `scan_roots()` walks all of `src/core`, so the *prose* of
  `search.py` and its docstring must not spell a write marker either.

### `web/src/composables/useWorkspaceSearch.js` (new)

One singleton, holding `query`, `caseSensitive`, `regex`, `scope` (the
`commandTarget()` shape it was opened on), `state` (`idle | running | done |
error`), `result`, the `reveal` it hands the pane (2.6), and an `openFor(target)`
that raises the dialog **without clearing the last answer** (2.7). Scope is
derived from the target: `dir` true means folder, false means file, and
`path === ROOT` means the whole workspace. It registers itself through
`registerReRootHandler` to forget query, scope, results and pending reveal on the
same event that closes the tabs.

### `web/src/lib/search-scope.js` (new)

Kebab-case, the way the other `lib/` modules are named. One pure function,
`scopeFor(target, rootLabel)` returning `{ kind, label }`, mapping the
`commandTarget()` shape to the dialog's scope: `path === ROOT` means
`{ kind: 'workspace', label: rootLabel() }`, `dir` means `folder`, otherwise
`file` (2.8). It lives in `lib/` rather than inside the composable for the reason
`tab-strip-wheel.js` and `editor-keys.js` exist: it is the rule two components
must not disagree about, and jsdom can test it without mounting either. The
`isFile` the request needs is `kind === 'file'`, derived here instead of passed
around as a second flag.

### `web/src/components/SearchDialog.vue` (new)

`<dialog>` + `showModal()`, the house style. Markup, top to bottom:

1. read-only scope label (2.8);
2. the query `<input>` (autofocus, `Enter` submits);
3. the two checkboxes, on one line;
4. the results `<table>` — columns **path**, **line**, **snippet** — or the
   empty/inline-error state;
5. the footer line when `truncated`.

Snippet text goes through **text interpolation, never `v-html`**: the app's rule
is that rendered HTML always passes `lib/markdown.js`, and a search snippet is
not rendered HTML — it is a line of someone's file, so it is bound as a string
and gets no markup at all. Highlighting the matched span is done with a separate
`<span>` around a slice of the string, never by injecting the file's bytes.

The row is a `<button>` in the cell (the strip's chips use the same reason: a
row is a click target, and nesting needs care), or a `<tr>` with a keypress
handler. Path column shows the workspace-relative path verbatim — the same
string the tree would show, and the same string `openFile` takes.

### `web/src/lib/tree-menu.js`, `WorkspaceNavigator.vue`, `EditorArea.vue`, `MonacoEditor.vue` (edits)

| file | change |
|---|---|
| `lib/tree-menu.js` | one `ITEMS` row (2.9). Nothing else: `treeMenuItems`/`inScope` already cover it. |
| `WorkspaceNavigator.vue` | `search: (target) => search.openFor(target)` in `MENU_ACTIONS`, taking the whole `commandTarget()` shape (`{ path, name, dir, parent }`, `useWorkspaceTree.js:450`) — so `dir` decides file vs folder and `path === ROOT` decides the workspace case, exactly as `menuFlags` already reads the same row. The header icon button, and `<SearchDialog />` mounted here. |
| `EditorArea.vue` | a `watch` on the search composable's pending reveal, calling the pane's `revealPosition` on the active Monaco surface. |
| `MonacoEditor.vue` | `defineExpose({ revealPosition })` + the `pendingReveal` ref (2.6). |

## 5. Out of scope for v1

- Searching file **names**, globs, or "find in open files".
- **Whole-word** toggle; `\b…\b` by hand with regex on.
- **Filters** by extension, by folder, exclude patterns. The policy file is the
  only exclusion mechanism this surface has, and that is deliberate — a search
  box with its own ignore list would be a second, temporary policy that the
  workspace's rules do not know about.
- A **replace** affordance of any kind. This feature is read-only by design and
  must stay out of the guarded write surface.
- Result **grouping by file**, file headers, and a file-level hit count. The
  table is flat.
- **Streaming/partial** results, progress events, cancel mid-scan.
- Remembering the query **across reloads** (no persistence layer is touched);
  "retained" in 2.7 means *within the page session*.
- Keyboard chords (2.9) and a search **history** dropdown.
- Search inside the **terminal** or the **WorkBench** — the Editor tab only.

## 6. Verification

### Backend (`tests/test_editorfs_search.py`, new)

Hermetic, `tmp_path` workspace, no network — the pattern of `tests/test_editorfs.py`.
Each rule above is one test:

| test | asserts |
|---|---|
| literal, case-insensitive | `Foo`, `foo`, `FOO` all match; `matchCount` right |
| case-sensitive | only the exact case matches |
| literal ≠ regex | a query of `a.c` does not match `abc` with `regex: false` |
| invalid regex | 400, `re`'s message in `detail`, **and no scan happened** (patch a sentinel file so a match there proves nothing) |
| regex meta-chars in a literal query | `(` does not raise |
| file scope | one file searched; a sibling file with the same text is not in the results |
| folder scope | descendant two levels deep is found |
| root scope | a file at the workspace root *and* in a nested folder are found |
| hidden folder pruned | a `hidden` directory's content is **not** matched, and no payload key counts it |
| hidden file omitted | a `hidden` file matching the query returns nothing |
| `ro` searched | an `ro` file's content *is* in the results |
| `.nojae` never searched | seeded workspace: a match inside `.nojae/` is absent |
| symlinked dir not followed | a link out of the workspace is not descended |
| oversized skipped | a 1.5 MB file with a match is absent; `skipped.oversized` is 1 |
| binary skipped | a NUL-leading file with the bytes is absent; `skipped.binary` is 1 |
| non-UTF-8 skipped | the same answer as `read_file`'s `binary: true` |
| match cap | 600 matches ⇒ 500 rows, `matchCount: 600`, `truncated: true` |
| deterministic order | a truncated result is the same 500 rows twice, and matches the tree's ordering |
| line/column | 1-based, column is the match start, `matchLength` is the match width |
| snippet trimmed | `…` present at `MAX_SEARCH_SNIPPET_BYTES`, and the row is not the whole line |
| path escapes | `?path=../elsewhere` ⇒ 422; a hidden scope path ⇒ 404 |

The caps are patched as `editorfs.constants.MAX_SEARCH_MATCHES = 3`-style module
attributes (which is why `search.py` reads them qualified), never by writing 500
files.

### Frontend (`web/tests/`, Vitest + jsdom)

- `searchScope.test.js` — `lib/search-scope.js` in isolation: the root row gives
  `workspace` + the workspace label, a folder row gives `folder` + its path, a
  file row gives `file` + its path, and the file case is decided by `dir` being
  false rather than by the name looking like one.
- `workspaceTreeMenu.test.js` (existing) — extend: every row kind, root included,
  now offers `search`; and a row that offers *only* search still gets a menu.
- `useWorkspaceSearch.test.js` — the retention rule: stub one search, close the
  dialog, re-open, and the query/scope/table are still there with **no second
  fetch** on the stub; then a working-location switch empties all three.
- `searchDialog.test.js` — the scope label renders the folder path, the file
  path, and the workspace basename for the root; a 400 renders `re`'s message
  inline rather than as a notice; a snippet is never injected as HTML.
- Monaco reveal is **not** unit-tested: `web/tests/probe-monaco.test.js` is a
  browser probe, and the jsdom suite cannot drive Monaco's viewport. It is
  covered by e2e instead.

### e2e (`web/e2e/`, Playwright against the real app)

A new spec alongside `02-editor-write.spec.js`, its own server on its own port,
endpoint from `.env.test` — never the developer's `.env`:

1. right-click a folder row → **Search…** → label names that folder → type a
   word only a nested file contains → the row appears → click it → the tab strip
   holds that file and the caret is on the matching line
   (assert on the buffer's revealed line, not on a screenshot);
2. a file row → search one word → every result path is that file;
3. the root row → the label names the workspace, not an empty string;
4. regex on, `(` typed → the inline error, and the table still holds the
   previous answer;
5. re-open the dialog without searching → the previous table is on screen.

### What is not tested, and why

- **Performance on a real huge tree.** The caps are what bound it; a timing test
  would fail on the CI machine rather than on the code.
- **Regex parity with any JS engine** — there is no JS regex in this feature
  (2.3), so there is nothing to compare against.
- **The `title` tooltip of a full snippet line** — cosmetic.

## 7. Note for `AGENTS.md` when this lands

One clause, in the Editor's file-surface paragraph, and nothing longer:

> The Editor's search answers about the same set of paths the tree lists: a
> `hidden` folder is pruned and its bytes never quoted, `ro` is searched, and
> no count of policy-skipped files is reported — an observation of a hidden
> path is still indistinguishable from absence.

The route list, the caps and the column layout belong in the code. The invariant
worth six lines is the one about *content*, because that is the guarantee a
future reader is most likely to break by accident.
