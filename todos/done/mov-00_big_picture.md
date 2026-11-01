# Move a batch of same-folder files to another folder — big picture

Teach the Editor tree one new gesture: **mark several file rows, right-click a
file row, `Move to…`, type a destination *folder*, press Move** — one request, one
refresh, and the set either moves whole or nothing moves at all. Today the only way
to move a file is `Rename…` with a `/` in the name field, so regrouping six files
costs six right-clicks, six typed full paths and six re-lists.

The design — the need, the five rules, every rejected alternative, the verification
list — is **`docs/tree-move-batch-spec.md`** (510 lines, English) and it is the
authority. This file is the map, the decision list and the cross-task contract. Read
the spec's §2.2b, §2.4, §2.5 and §5.2 before touching either side. Nothing here
restates it except what a task cannot re-derive, or must agree with another task about.

## What is new

| piece | role |
|---|---|
| `move_paths_batch(paths, to_parent)` in `src/core/editorfs/mutate.py` | core-only batch move: plain values in, dict out, no FastAPI. Reuses `rename_path`'s gates; it is **not** a second rename. |
| `MAX_MOVE_BATCH = 200` in `src/core/editorfs/constants.py` | the entry cap, patchable **by name** (sibling of `MAX_TREE_ENTRIES_PER_DIR`). |
| `FileMoveBatchIn` + `POST /api/fs/move-batch` in `src/api/routes.py` | transport only: the streaming refusal, the core call, the two re-raises. |
| `state.marked` / `state.markAnchor` in `useWorkspaceTree.js` | the batch as **new tree state** — never an overloading of `state.focused` or `state.selected` (§2.1). |
| `web/src/composables/useEditorMove.js` | the flow: `startMove`, `cancelMove`, `submitMove`, `validateMoveDestination`, `state.movePrompt`. |
| `web/src/components/EditorMoveDialog.vue` | one `<dialog>`; decides nothing. |

Edits: `useWorkspaceTree.js`, `WorkspaceTreeItem.vue`, `WorkspaceNavigator.vue`,
`lib/tree-menu.js`, `EditorArea.vue`, `styles/theme.css`,
`src/core/editorfs/__init__.py`, and the guard/test files named in *Traps*.

## The five rules, and what each one costs the code

| # | rule | where it is enforced |
|---|---|---|
| R1 | a batch's files share one parent folder | `batchFor()` (mov-02): a cross-parent selection narrows to the clicked row, and the narrowing is **visible** in the dialog's list before anything is sent |
| R2 | all-or-nothing | `move_paths_batch` (mov-01): validate every entry, then apply none until all are valid. It is one request — N browser-side renames could not be undone |
| R3 | the destination folder must already exist | the core's step 5, answering the existing `_parent_missing_error` sentence verbatim. No implicit `mkdir`, no create-chaining |
| R4 | only files join a selection | the tree refuses a folder row entry (mov-02) **and** the core 400s a folder in `paths` (mov-01) |
| R5 | `Ctrl`/`Cmd`+click toggles, `Shift`+click/arrows range-extends | `toggleMark` / `rangeMark` over `visiblePaths()` (mov-02), so a range can never disagree with what is rendered |

## Decisions

| # | decision | why |
|---|---|---|
| D1 | `marked` is a **third** tree field, not a use of `selected` | §2.1. `selected` carries `size`/`mtime`/`source` for the pane and is cleared by `dropBranch`; a batch needs none of it and would inherit all of it. `marked` is files-only by construction, which is why no cycle check is needed anywhere. |
| D2 | the menu item is gated by `scope: 'file'` alone — **no `needs` flag** | §2.2. That scope is exactly the set of rows the item can honour. A flag would have to answer "is this writable", which that file's header says is never its business; the server is the authority, as it already is for `Rename…`. |
| D3 | the new flag `moveCount` feeds **hint text only**, never presence | §2.2: presence is D2's, so the item sits on every file row and the single move is the N=1 case of one code path, not a second feature. |
| D4 | the subject is resolved **once**, when the menu opens, via `batchFor(clickedPath)` | §2.2b. By the time an item is clicked the marks may have moved; `startMove` copies the list into `state.movePrompt` immediately, so no later click can empty the list the dialog is about to show. |
| D5 | a range **may cross parents**; the batch narrows, the marks stay highlighted | §2.1. Clamping the range to the anchor's folder was rejected: the tree would refuse to highlight rows the user dragged across, and the reason would be invisible. |
| D6 | the field asks for a **destination folder**, never a full target path | §2.4. Renaming while moving stays `Rename…`'s job; a second way to rename means a second basename rule to keep in step. |
| D7 | the field is prefilled with the sources' `parentOf(path)`, so N=1 opens with Move **disabled** and "That is where they already are" showing | §2.4. The common case becomes an edit instead of a path typed from memory, and the no-op case explains itself. |
| D8 | the prefill and the typed value are **workspace-relative**, never absolute | §2.3. Every path the app holds is relative to the working location, so a working-location switch needs nothing here: the value shown is already re-rooted. |
| D9 | the client refuses a leading `/` or `~` and **any** `.`/`..` segment; the server refuses only a `..` that leaves the workspace | §2.4. Inherited asymmetry, deliberate: `notes/tmp/../archive` is legal server-side, and a field that resolves `..` would need a second path model in the browser to explain what was typed. |
| D10 | **no copy of the deny-list in the client**, and no folder picker in the modal | §2.4, §2.3. The server's sentence names the folder it looked for; the tree is lazy, so a complete folder list does not exist without N listings, and a half list would hide policy-denied folders and invite picking one. |
| D11 | the server joins the target (`to_parent + "/" + basename(p)`) | §2.5. Same argument `FileCreateIn`'s docstring makes in reverse: a client that joined could hide a target the ancestor gate never saw. |
| D12 | the rename loop is **not** atomic across N paths, and the payload reports what landed | §2.5 step 7. There is no filesystem transaction to borrow; the loop stops at the first `OSError` and must never silently continue. The window is the one `rename_path` already accepts. |
| D13 | no two sources of one parent can collide at the target, so **no collision table** | §2.4. Same parent + one destination ⇒ distinct basenames ⇒ distinct targets; and destination == source parent is already refused by D7's check. Recorded so no implementation adds the table. |
| D14 | each affected listing is refreshed **once**, and there is **one** notice | §5.2. The batch has exactly two affected listings; per-file notices would bury the tab strip. |
| D15 | `aria-selected` follows `marked` when there are marks, else the single pane file | §5.3. The one accessibility behaviour change in the feature, required by the `role="tree"` multi-select pattern, not incidental. |
| D16 | **no keyboard chord** for Move | The spec gives `Shift` only on Arrow/Home/End and offers no `M`-style chord (unlike `F2`/`Delete`). Do not invent one. |
| D17 | the Move button is gated only by `useEditorMode.isEditable` — no `ro` pre-check | §3, and the same answer `Rename…` gives: the mode badge is the only client-side claim about writability; whether *this path* is writable stays the server's rule. |

## The two contracts every task must agree on

**The request** — `POST /api/fs/move-batch`:

```json
{"paths": ["notes/a.md", "notes/c.md"], "to_parent": "notes/archive"}
```

`""` for `to_parent` is the working location. `paths` are whole workspace-relative
source paths; the parent is a separate field (D11).

**The response** — shaped so the client's reconciliation is the rename's loop:

```json
{"moved": [{"from": "notes/a.md", "to": "notes/archive/a.md", "mtime": 1712345.0}],
 "to_parent": "notes/archive"}
```

**The refusal statuses, in the core's step order** (the validation order *is* the
test list, §2.5):

| condition | status / body |
|---|---|
| `paths` empty, or over `MAX_MOVE_BATCH` | **400**; the over-cap one names the cap |
| a source is absent | **404** (`missing_path_error`) |
| a source is a folder | **400** naming the path — R4 enforced server-side too |
| a source is a top-level symlink | **400** (the delete's argument, unchanged) |
| a source, or any ancestor of a target, is refused by a rule | **422** in the policy's own words |
| `to_parent` is missing, or is a file | **409** `_parent_missing_error`, naming it (R3) |
| a target is occupied — file, folder, dangling link | **409** `{"reason":"exists","path","dir"}`, the existing object |
| a source and its target normalise to the same path | **409** through "occupied", never a 200 no-op (§7: stated so nobody "fixes" it by skipping) |
| `EXDEV` / `ENOENT` during the renames | the existing 409 sentences |
| a chat reply is streaming | **409**, named reason, before any core call |
| malformed body | **422** from FastAPI, never a 500 from the core |

## Traps — found by reading the code, not stated as warnings in the spec

1. **Two guards go red unless they are edited, and they are edited in *different*
   tasks.** `tests/test_write_surface_isolation.py` holds both halves: the `writers`
   tuple in `test_the_write_core_is_called_from_the_routes_and_nowhere_else` — which
   the same test duplicates verbatim in
   `tests/test_editorfs_write.py::TestWritesStayBrowserOnly` — must gain
   `move_paths_batch`, and `ALLOWED_HOMES` must gain
   `web/src/composables/useEditorMove.js`. **mov-01 makes all four list edits** (the
   marker scan walks `web/src/composables`, so a missing `ALLOWED_HOMES` entry flags
   mov-03's file as soon as it lands); **mov-03 appends the `POST /api/fs/move-batch`
   marker to the body of its own `submitMove`** — which is exactly why
   `useEditorRename.js` names the rename route in a `fetch` string.
2. **`tests/test_editorfs.py:~677` (`TestSurfaceShape`) pins the whole `/api/fs` verb
   map** as an exact dict, whose docstring says a new mutation "has to appear here on
   purpose". mov-01 adds `"/api/fs/move-batch": ["post"]` as part of *its* acceptance:
   that entry **is** the review of "this route is a write".
3. **Several existing tests pin menu lists verbatim and go red the moment the item
   lands**: `web/e2e/02-editor-write.spec.js`
   (`each row offers exactly what it can honour` — a file row's array at ~`:310` and a
   markdown row's at ~`:318`; the **root and folder arrays do not change**),
   `web/tests/useWorkspaceTreeCommands.test.js` (the `treeMenuItemIds(...).toEqual([...])`
   block around lines 417-507, and the leanest-shape `['search', 'copy-path']`
   assertion), and `web/tests/workspaceTreeMenu.test.js` (the `menuFor(...)` label
   arrays around lines 131-165, plus its own `['search', 'copy-path']` one at ~`:340`).
   **mov-05 updates every one of them in the same task** — otherwise it cannot be
   acceptance-checked and each later task inherits a red suite.
4. **`tests/test_editorfs_write.py`'s cross-cutting matrix** (`WRITE_KINDS` +
   `call_write`) is that file's own statement that "a property asserted over every
   route is written once here, so adding a route means adding a branch and nothing
   else". mov-01 adds the source-side and destination-side kinds there. **Do not add a
   batch row to
   `TestWriteMatrix.test_no_route_silently_replaces_or_reduces_what_is_already_there`**:
   that row would point the batch at an occupied destination inside the source's own
   folder, which is the same-path case the spec deliberately does not exempt; the
   property it would prove is already `TestMoveBatch`'s occupied-target test, which
   asserts both sources still at home.
5. **jsdom has no `showModal`/`close` on `<dialog>`.**
   `web/tests/useEditorWriteMatrix.test.js` (~`:385`) patches `Element.prototype.showModal`
   and `.close` to set `el.open` — mov-03's flow test, and any dialog test mov-04 adds,
   need the same two stubs.
   `web/vitest.config.js` sets `globals: false`, so every test file imports
   `describe/it/expect/beforeEach` from `'vitest'` explicitly.
6. **`handleKey` returns early on `event.ctrlKey || event.metaKey || event.altKey`,
   and that stays true** (§4: "`Ctrl`/`Cmd` stays out of the chord chain, as today",
   plus D16). `Ctrl`+click is a **mouse** gesture reaching the composable through
   `select(entry, event)`, never a chord. `Shift` joins only the Arrow/Home/End chords.
7. **`marked` must be dropped where its paths stop meaning anything**: `dropBranch`
   forgets marks under the removed path (next to its existing `focused`/`selected`
   lines, ~`:602`), and `reRoot` clears all of them (next to `focused`, `selected`,
   `editing`, ~`:332`). Forgetting either is how a batch outlives its own rows.
8. **A row's `@click` today calls `select(entry)` with no event**
   (`WorkspaceTreeItem.vue:100`). The modifier pass-through is what makes R5 reach the
   composable at all — without it the whole gesture is dead.
9. **Two listings, not N.** In the reconciliation, the folder set is
   `{parentOf(from)} ∪ {parentOf(to)}` across the whole batch (normally two), and
   `refreshListingOfParent` runs at both ends for the `children` flag promise. A loop
   that re-lists once per file passes every assertion in `useEditorMove.test.js` unless
   that test counts the requests — so it must count them (§5.2's words).

## Not in this lot (§6 — do not add these)

drag-and-drop in either direction; moving **folders**; cross-parent batches (several
parents, one destination each); copy, undo, a batch rename; a destination picker,
breadcrumbs or any folder list; creating the destination from the modal; batch
selection through the terminal or the WorkBench; any connector reaching this route;
checkboxes — the marks are a row style.

## Project rules that bind these tasks (from `AGENTS.md`)

- *Core owns logic, API owns transport*: `move_paths_batch` takes plain values and
  returns a dict and touches no FastAPI; the handler is the streaming refusal, the call
  and the re-raises.
- *A `/api/fs` write route is reachable from the SPA only*: no connector, LSP tool or
  chat code path may call one, and every write route is refused while a chat reply
  streams. `tests/test_write_surface_isolation.py` asserts that instead of trusting the
  sentence — Trap 1 is that file's two lists.
- *Path rules come from two levels, one verdict*: a hidden path answers 404 to a read
  and 422 naming the rule to a write, `ro` answers 422 to a write only. The batch asks
  the **same** verdict through `rename_path`'s helpers and grows no second list of
  protected paths.
- *Every path the app holds is relative to that location* (D8).
- *Frontend invariants*: rendered HTML always goes through `lib/markdown.js`, never raw
  `v-html` — so the dialog's source list is `{{ }}` text in `.caption.mono`, exactly as
  the delete confirm renders its path; and *run/stream state lives in the composables*,
  which is why `marked` and `movePrompt` are composable state.
- *Python style*: explicit type annotations, `typing` imports, no bare containers.
- *Tests stay hermetic*: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1` (set by
  `tests/conftest.py`); a backend test's workspace is a throw-away `tmp_path` pointed at
  by `BENCHMARKIT_PATH_LOCATION`, never this checkout; e2e takes its endpoint from
  `.env.test`, builds `web/dist` itself, and runs `fullyParallel: false`, `workers: 1`.
- No lint step. `pytest` for Python; `cd web && npm test` for Vitest;
  `cd web && npm run test:e2e` for Playwright; `cd web && npm run build` for `web/dist`,
  which must exist for FastAPI to serve the UI.

## Task order

| # | task | depends on |
|---|---|---|
| mov-01 | the batch core and its route: `move_paths_batch`, `MAX_MOVE_BATCH`, the package export, `POST /api/fs/move-batch`, `TestMoveBatch`, and every guard/allowlist/matrix edit | — |
| mov-02 | marked rows: `state.marked`/`markAnchor`, `toggleMark`/`rangeMark`/`clearMarks`/`batchFor`, the click pass-through, the `Shift` chords, `tree-row--marked` + `aria-selected`, the "N selected" footer line, and their unit tests | — |
| mov-03 | `useEditorMove.js`: `movePrompt` state, `validateMoveDestination`, the request, the §5.2 reconciliation — plus the new `web/tests/useEditorMove.test.js` | mov-01, mov-02 |
| mov-04 | `EditorMoveDialog.vue`, mounted in `EditorArea.vue`, and its `theme.css` block | mov-02, mov-03 |
| mov-05 | the `move-batch` menu item: `tree-menu.js`, `menuFlags`, `MENU_ACTIONS` — **and every pinned menu list five tests assert (Trap 3)** | mov-02, mov-03, mov-04 |
| mov-06 | the Playwright sweep (the new §7 flows in `02-editor-write.spec.js`), the README paragraph about `todos/`, and the AGENTS.md decision | all of the above |

mov-01 is backend; mov-02 → mov-05 are frontend (their tests stub `fetch`, so the
frontend half needs no live route); mov-06 is what only a real browser can show.
mov-01 and mov-02 are independent of each other, so either may run first. Every task
leaves the app working and the suites it did not touch green. Two tasks edit
`web/e2e/02-editor-write.spec.js` on purpose (mov-05 fixes its pinned menu arrays,
mov-06 adds the new flows) — the numbering keeps them from racing.
