# Move a batch of same-folder files to another folder (Editor tree)

**Status**: spec written, implementation not started.
**Surface**: the Editor tab's tree only — a new multi-selection model, one
context-menu item, one modal, one new write route, one new core function.
No LLM, no hub, no SSE, no drag-and-drop, no terminal.

## 1. The need

The tree can move exactly one path per gesture: `Rename…` opens the inline name
input and a `/` typed in it is a move (`validateRenameTarget` in
`web/src/composables/useEditorRename.js:42`). Regrouping six files of `notes/`
into `notes/archive/` therefore costs six right-clicks, six typed full paths,
and six re-lists.

So: select several files that live in the same folder, right-click, choose
**Move to…**, type the destination folder in a modal, press Move. One request,
one refresh, and — the point of the batch — the set either moves whole or
nothing moves at all.

The same item answers a **plain right-click on one file**. Multi-selection is
an enlargement of the gesture, never a precondition for it: the item lives on
the file row, so the single move is the N=1 case of one code path rather than a
second feature.

The five rules this feature is built on, all settled before design:

| # | rule |
|---|---|
| R1 | A batch's files must share one parent folder. |
| R2 | All-or-nothing: any refusal cancels every move in the batch. |
| R3 | The destination folder must already exist — never an implicit `mkdir`. |
| R4 | Only files take part. A folder row never joins a selection. |
| R5 | `Ctrl`/`Cmd`+click adds single rows, `Shift`+click extends a contiguous range. |

## 2. Decisions taken

### 2.1 Multi-selection is new tree state, not a use of `state.selected`

`useWorkspaceTree.js` has no set of rows today. `state.focused` is the roving
tabindex and `state.selected` is *"the code pane should open this file"* — one
object, consumed by `EditorArea`'s watcher, and never set for a folder row
(`select()` at `:496` toggles instead). Overloading either would make every
folder click try to open a buffer, or would give the pane N files at once.

A third field is added:

```js
marked: [],   // ordered paths of the rows the user put in the batch (files only)
markAnchor: null, // path the Shift-range grows from
```

It lives in the composable, like every other tree fact, so the selection
survives a tab switch (the navigator sits in a `v-show` view) and so the new
write composable can read it without going through the component.

- A **plain click** replaces the marks with nothing (focus and `selected`
  behave exactly as today).
- **`Ctrl`/`Cmd`+click** toggles one path in/out of `marked`. `select()` still
  runs, so a Ctrl+click also opens the file — the row under the cursor is both
  the row you look at and the row you add. A folder row is refused entry (R4)
  and keeps toggling.
- **`Shift`+click / `Shift`+arrows / `Shift`+Home/End** selects every *file*
  row in the visual span, taken from `visiblePaths()` (`:95`) between
  `markAnchor` and the new focus — the same list arrow navigation already uses,
  so a range can never disagree with what is rendered.
- A range **may** cross parents. That is not softened: a cross-folder selection
  is legal, it only narrows what a `Move to…` on any of those rows covers (see
  2.2b). Deciding instead to clamp the range to the anchor's folder was
  rejected — the tree would then refuse to highlight rows the user dragged
  across, and the reason would be invisible.

**Rejected alternative**: reuse `state.selected` as a list. It carries `size`
and `mtime` for the pane, is stamped `source: 'tab'` by `reveal()`, and is
cleared by `dropBranch()`. A batch needs none of that and would inherit all of
it.

### 2.2 The `Move to…` item is on every file row; its subject is computed once

`web/src/lib/tree-menu.js` is one ordered table; a new item is a new row:

```js
{
  id: 'move-batch',
  label: 'Move to…',
  scope: 'file',
  title: (flags) =>
    flags.moveCount > 1
      ? `Move these ${flags.moveCount} files into another folder`
      : 'Move this file into another folder',
}
```

placed directly **after `rename`** — same mutation family, and `Delete` stays
the only `danger` row.

**No `needs` flag gates it.** `scope: 'file'` is the whole condition, because
that is exactly the set of rows the item can honour: a folder row can never
join a batch (R4) and the root is not a file row. A flag would have to answer
"is this path writable", which this file's header says is never its business —
the server is the authority, as it already is for `Rename…`. The one new flag,
`moveCount`, feeds the hint text only, never presence.

`Rename…` and `Move to…` therefore coexist on a single file, and that is not
redundancy: `Rename…` asks for the **full target path** in a field labelled as
a name, while `Move to…` asks for the **destination folder** alone and lists
every path it is about to move. The modal is the reason the batch exists.

R1 is now answered by the modal's list rather than by the menu: a cross-folder
selection does not hide the item, it moves the clicked file alone, and the user
sees `Move 1 file` before pressing anything.

### 2.2b The subject: `batchFor(path)`, decided when the menu opens

The action must not read `commandTarget()` on click — by then the marks may have
moved. It resolves the subject once, from the row the menu was opened on:

```
batchFor(clickedPath):
  marks = state.marked                     // files only, by construction
  if clickedPath in marks and marks.length >= 2
     and every mark shares parentOf(clickedPath)
        -> the whole mark set
  otherwise
        -> [clickedPath]                   // the N=1 case
```

- Right-click with **no marks at all** → one file, `moveCount` 1.
- `Ctrl`+click two files of `notes/`, right-click one of them → both.
- Right-click an **unmarked** file while a batch is live, or a batch whose marks
  cross parents → the clicked file alone. The marks stay highlighted, but the
  modal's list is the truth: `Move 1 file` and that one path. The narrowing is
  visible before anything is sent, which is what makes it acceptable — a silent
  batch of the wrong files would not be.

`useEditorMove.startMove(batch)` copies the list into `state.movePrompt`
immediately, so no later click can empty the list the dialog is about to show.

### 2.3 The destination is typed in a modal, not in the inline name input

`TreeNameInput` is one-at-a-time: `state.editing` is a single object, it
renders *inside* one listing, and its value is a **name** (a create) or a
**full target path of one row** (a rename). A batch has no single row and no
single target name, so it gets its own `<dialog>`.

New component `web/src/components/EditorMoveDialog.vue`, following the house
pattern (`EditorTreeDialogs.vue`, `SearchDialog.vue`): `<dialog>` +
`showModal()` driven by composable state, reconciled by a `watch`, Cancel the
**first** button so Esc / backdrop / Enter all mean Cancel. Mounted once inside
`EditorArea.vue` beside `<EditorTreeDialogs />`.

Its content, and nothing more:

| part | text |
|---|---|
| title | `Move N files` |
| list | every source path, one per line, in `.caption.mono` (the delete confirm's shape) |
| field | `<input>` — destination folder, workspace-relative, prefilled with the sources' parent so the common case is an edit not a type (for one file, that is its own folder) |
| hint | `The folder must already exist.` |
| buttons | `Cancel`, `Move N files` |

**Worked example.** A right-click on `todos/done/a-file-any.any` opens the modal
with the field holding `todos/done` — the file's own parent, spelled the way the
tree spells it. Note what that value is **not**: not
`$BENCHMARKIT_PATH_LOCATION/todos/done`, not
`/mnt/bigfoot/dev/benchmarkit/todos/done`. The tree's whole path model is
root-relative (`state.listings` keys, the `path` field of every
`POST /api/fs/…` body), so the prefill is just `parentOf(path)` — the helper
`useEditorRename.js:30` already has — and nothing in this feature ever needs the
absolute location. Which is also why a working-location switch needs no change
here: the value shown is already re-rooted.

A **bare name** typed into the field is not "relative to the sources' parent": it
is `todos` resolved from the workspace root, like every other path in the app. A
one-segment relative path cannot be told apart from a bare name in that field,
and guessing would make the same keystroke mean two things in two dialogs — so it
is answered by the server's existing `_parent_missing_error`, naming the folder it
looked for, which the hint under the field has already promised.

A textarea was rejected: the value is one path, never multiline, and a newline
inside it is a character the server's path rules would answer — the input's own
validation would have to lie about it.

The dialog **does not** grow a file picker or a folder browser. Rejected: the
tree is lazy, so a complete folder list the browser could offer does not exist
without N listings, and a half list that hides policy-denied folders would be
an invitation to pick one. Typing is honest, and the server's refusal names the
folder that is not there.

### 2.4 The fast half of the answer is client-side, the real half is the server's

`validateMoveDestination(value)` (new, in the move composable) mirrors
`validateRenameTarget`'s shape checks and nothing else: empty, `\`, NUL, a
leading `/` or `~`, a `.`/`..` segment, and — the one new check — a destination
equal to the sources' own parent (`That is where they already are`). A
destination *under one of the selected files* is impossible by construction
(R4: no folder rows), so no cycle check is needed and none is written.

That last check is what the **N=1** dialog meets first, since the field is
prefilled with the file's own folder: it opens with Move disabled and that
sentence showing, and enabling the button is the edit. Accepted deliberately —
prefilling the current folder makes the common case an edit and the no-op case
self-explanatory. Rejected: leaving the field empty instead, which would make
the first keystroke a full path from memory.

The field is a **folder only**, at every N. Renaming while moving stays
`Rename…'s` job (its input takes the full target path); a second way to rename
would mean a second basename rule to keep in step.

Everything else stays the server's rule: `.env`, `.git/…`, an `ro` or `hidden`
folder, a taken target name. There is deliberately **no copy of the deny-list
in the client**, exactly as `useEditorRename.js:22-24` states for renames.

**Relativity is not a new rule, it is the existing one.** The value is always
resolved against the active working location — the same `$BENCHMARKIT_PATH_LOCATION`
every other file surface is rooted at — and the backstop is
`normalise_rel()` (`src/core/editorfs/policy.py:180`), which `resolve_write_target`
already runs for the target's every ancestor: an absolute path or a leading `~`
is a 422 naming the workspace it must be relative to, and a `..` that survives
`os.path.normpath` is a 422 too. So the route needs **no relativity code of its
own**; the client's leading-`/` check above is a round-trip saving, nothing more.

One asymmetry, inherited and deliberate: the client refuses *any* `.`/`..`
segment, while the server only refuses a `..` that would leave the workspace —
`notes/tmp/../archive` is legal there. Rejected: matching the server in the
client, because a destination field that resolves `..` needs a second path
model in the browser to explain what it typed, and the prefill already gives the
user a path they can edit instead. A destination typed with `..` is simply
refused up front, in the client's own words.

One client-side check the modal *does* run, because it saves the batch from a
guaranteed 409: two selected files cannot land on the same target name when the
destination equals the source parent — already excluded above — and, in
general, the target name of each move is its basename, so no two sources of one
parent can collide. Recorded here so that no implementation adds a collision
table that the same-parent rule makes unreachable.

### 2.5 One new route, one core function that reuses `rename_path`'s rules

All-or-nothing (R2) cannot be done by firing N `POST /api/fs/rename` calls from
the browser: the first successes would already be on disk when the fifth is
refused. So the batch is **one request**, and the transaction is the ordering
itself — validate every entry, then apply none until all are valid.

**Route**: `POST /api/fs/move-batch`, body `FileMoveBatchIn` beside
`FileRenameIn` (`src/api/routes.py:199`):

```python
class FileMoveBatchIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    paths: List[str]        # the whole workspace-relative source paths
    to_parent: str          # "" is the working location
```

`paths` are full paths, not names, and the parent is a separate field: the
server joins them (`to = to_parent + "/" + basename(p)`), because a client that
joined them could hide a target the ancestor gate would never see — the same
argument `FileCreateIn`'s docstring makes in reverse. The route calls
`_refuse_if_chat_streaming("move files")` first, like every other write.

**Core**: `move_paths_batch(paths, to_parent) -> Dict[str, Any]` in
`src/core/editorfs/mutate.py`, next to `rename_path`. It is a *reuse of
rename's rules, not a second rename*: the same helpers in the same order —
`resolve_write_target` for each source (and its whole chain), the top-level
symlink refusal, `resolve_parent_in_workspace` for each computed target,
`os.path.lexists` on each target — and then `os.rename` for each entry only
after every entry passed.

Validation order, which is the test list:

1. `paths` empty → 400; over `MAX_MOVE_BATCH` entries → 400 naming the cap
   (a new number in `editorfs/constants.py`, patchable by name, sibling of
   `MAX_TREE_ENTRIES_PER_DIR`; start at **200**).
2. `to_parent` normalised (the client refuses the no-op case, the server has no
   reason to).
3. every source gated and existing (404 for the first absent one), every source
   a file and not a folder (**400** naming the path — R4 is enforced
   server-side too, so the route cannot be talked into moving a tree), every
   source not a symlink;
4. every target gated through its whole ancestor chain;
5. `to_parent` must already be a directory → 409 with `_parent_missing_error`
   (R3 — the existing sentence, reused verbatim);
6. no target occupied → 409 `{"reason": "exists", "path", "dir"}` (the existing
   `_exists_answer`), and no two sources resolving to the same target name;
7. **then** the renames, in the given order.

Step 7 is not atomic across N paths — there is no filesystem transaction to
borrow — and that residual risk is owned out loud rather than papered over: the
loop stops at the first `OSError`, and the payload reports what landed. The
window between step 6 and step 7 is the same window a single `rename_path`
already accepts, and `EXDEV`/`ENOENT` map to the existing 409 sentences. An
implementation must not silently continue past a failure.

Response, shaped so the client's reconciliation is the rename's loop:

```json
{"moved": [{"from": "notes/a.md", "to": "notes/archive/a.md",
            "mtime": 1712345.0}],
 "to_parent": "notes/archive"}
```

**Two project guards break unless they are updated in the same change** — this
is the part most likely to be missed:

- `tests/test_write_surface_isolation.py::test_the_write_helpers_are_called_from_the_routes_and_nowhere_else`
  scans `src` for the names in its `writers` tuple. `move_paths_batch` is a
  write core function, so it **must** be added to that tuple (and to the
  sibling guard's list), or the new route is invisible to the invariant that
  keeps writes out of the connectors.
- The same file's `ALLOWED_HOMES` set lists the frontend files allowed to
  contain a `POST /api/fs` marker. `web/src/composables/useEditorMove.js` must
  be added to it, or the marker scan fails.

## 3. What it builds on

| existing piece | reused for |
|---|---|
| `rename_path`'s helpers: `resolve_write_target`, `resolve_parent_in_workspace`, `_exists_answer`, `_parent_missing_error`, `missing_path_error`, `fsync_dir` | every gate and refusal sentence, so the policy wording stays one sentence everywhere |
| `useEditorRename.submitRename()` (`:157-198`) | the exact success reconciliation, run once per moved path |
| `files.repointPath(from, to)` | re-pointing an open buffer, its key, its label and `state.active` |
| `tree.dropBranch` / `refreshBranch` / `refreshListingOfParent` / `reveal` | cache + listing reconciliation |
| `git.load({ force: true })` | a move is a git event |
| `notices.notice('ok'/'error', …)` | the copy that travels when the eyes are on the pane |
| `visiblePaths()` | the Shift range and the keyboard, so navigation and `aria-posinset` cannot disagree |
| `useEditorMode.isEditable` | the modal's Move button is disabled while the badge says read-only — the server refuses the write anyway |

## 4. Files, and what each new one owns

| file | change | owns |
|---|---|---|
| `web/src/composables/useWorkspaceTree.js` | edit | `state.marked`, `state.markAnchor`; `toggleMark(path)`, `rangeMark(toPath)`, `clearMarks()`, `batchFor(path)` → `{ paths, parent }` for the row a menu was opened on (2.2b). `select()` clears the marks on a plain click. `dropBranch` and `reRoot` drop marks under the removed path / everything. The keyboard model gains `Shift` on the arrow/Home/End chords; `Ctrl`/`Cmd` stays out of the chord chain, as today. |
| `web/src/components/WorkspaceTreeItem.vue` | edit | `tree-row--marked` class from `state.marked.includes(path)`; `@click` passes the modifier through to `select(entry, event)`; `aria-selected` reports the **batch** (`marked`), not only the pane's file — see 5.3. |
| `web/src/lib/tree-menu.js` | edit | one `ITEMS` row (`move-batch`, `scope: 'file'`, after `rename`, no `needs`); its `title` reads the new `moveCount` flag. |
| `web/src/components/WorkspaceNavigator.vue` | edit | `menuFlags()` gains `moveCount` (= the size of the batch `batchFor` would answer for this row); `MENU_ACTIONS['move-batch']` calls `move.startMove(tree.batchFor(target.path))`; a "N selected" line in the footer while a batch is live, so the invisible state is visible. |
| `web/src/composables/useEditorMove.js` | **new** | the flow: `startMove`, `cancelMove`, `submitMove`, `validateMoveDestination`, `state.movePrompt = { paths, parent, value, error, busy }`. One `inFlight` guard, the same reason `useEditorRename` has one. |
| `web/src/components/EditorMoveDialog.vue` | **new** | the `<dialog>` and its buttons. Decides nothing. |
| `web/src/components/EditorArea.vue` | edit | mounts `<EditorMoveDialog />` beside `<EditorTreeDialogs />`. |
| `web/src/styles/theme.css` | edit | `.tree-row--marked` (a third row tone, distinct from `--focused` and `--selected`, in both themes) and the dialog's class, beside the existing `editor-*-dialog` block. |
| `src/api/routes.py` | edit | `FileMoveBatchIn` + `@router.post("/fs/move-batch")`, docstring naming where the rules live. |
| `src/core/editorfs/mutate.py` | edit | `move_paths_batch`. |
| `src/core/editorfs/constants.py` | edit | `MAX_MOVE_BATCH = 200`, patchable by name. |
| `src/core/editorfs/__init__.py` | edit | re-export `move_paths_batch` the way `rename_path` is re-exported. |
| `tests/test_write_surface_isolation.py` | edit | `writers` tuple + `ALLOWED_HOMES` (see 2.5). |

No new module under `src/` outside `editorfs/`, and no connector is touched:
the write surface stays SPA-only.

## 5. Interaction details

### 5.1 The flow, end to end

1. `Ctrl`+click `notes/a.md`, `Ctrl`+click `notes/c.md` (or `Shift`+click the
   last one). Both rows wear `tree-row--marked`.
2. Right-click **any** file row — marked or not; the item is on every file row.
   `batchFor()` decides the subject here: the marks when the row is in a legal
   batch of them, otherwise that one row.
3. Click it. `state.movePrompt` is raised with the batch `batchFor()` returned; the dialog opens
   with the field holding the shared parent (`notes`) and the caret there.
4. Edit it to `notes/archive`, press Enter (or Move).
5. `POST /api/fs/move-batch {"paths": ["notes/a.md","notes/c.md"],
   "to_parent": "notes/archive"}`.
6. Success → §5.2. Refusal → the dialog **stays open**, the field keeps what
   was typed, the server's sentence appears under the field and as an error
   notice. The rows do not move. This is the rename's rule (a refusal is an
   edit, not a restart), and it is why the modal, not a toast, owns the answer.

### 5.2 Success reconciliation — the refresh you asked for, done per path

For each `{"from", "to"}` in the response, in order, the rename's sequence:

1. `files.repointPath(from, to)` — first, for the reason `useEditorRename.js:90-93`
   states (a tick of the freshness probe must never see a tab whose path just
   vanished);
2. `tree.dropBranch(from)` — a file has no subtree, so this is the cheap case;
3. `refreshBranch(sourceParent)` and `refreshBranch(destinationParent)`
   **once each**, not once per file — the batch has exactly two affected
   listings, which is the whole efficiency the batch buys on this side;
4. `refreshListingOfParent` at both ends, for the `children` flag lie;
5. after the loop: `clearMarks()`, one `git.load({ force: true })`, and **one**
   notice — `Moved 3 files → notes/archive`. Per-file notices would bury the
   tab strip; the paths are in the payload for a future detail view.

The destination branch is expanded first (`expand(destinationParent)`) so the
user sees where the rows went; the last moved row is `reveal`ed.

### 5.3 What the selection costs the rest of the pane

- A marked file that is open in a tab keeps its tab: marking never opens or
  closes anything, only clicking does.
- `marked` is **not** the code pane's input. `state.selected` stays the single
  file the pane follows. Rejected: opening every marked file at once, which
  would make `Ctrl`+click a way to flood the strip with buffers.
- `aria-selected`: the tree is a `role="tree"`, and today `selected` is
  overloaded onto `state.selected`. A multi-selectable tree must report the
  selected set, so `aria-selected` follows `marked` (or the single pane file
  when the marks are empty). This is the one accessibility behaviour change in
  the feature, and it is required by the pattern, not incidental.
- A working-location switch (`reRoot`) clears the marks: their paths mean
  nothing in the new root, which is the argument `reRoot` already makes for
  `selected` and `editing`.

**One consequence of moving files in this workspace, not a feature.** The example
above sits in `todos/`, which is the batch lot: a task is *finished* when its
file left `todos/`, and `todos/done/` is where the loop's own prompt tells the
agent to put it (`web/src/composables/useBatch.js:41`, and `core/batch.py:262`
never descends into a done folder). So `Move to…` pointing at `todos/` re-arms a
finished task on the next listing, and pointing out of `todos/` retires one. That
is the correct behaviour of a file mover — the loop reads the filesystem, it does
not trust a flag — but it should be in the README sentence for this feature, so
the gesture is not mistaken for a no-op. No guard is added: the app has no way to
know a `.md` is a task, and `todos/` is an ordinary folder to every other surface.

## 6. Out of scope

- Drag-and-drop, in either direction. Same mutation core, different gesture
  layer, and a tree of `overflow: auto` rows is its own problem.
- Moving **folders**. R4. A folder in the batch would make "same parent" a
  question about the folder's own row, and `dropBranch`/re-expansion the
  per-entry job it is in the rename flow — a second spec.
- Cross-parent batches (choosing several parents and one destination each).
  Rejected outright for R1: one field, one answer.
- Copy (as opposed to move), undo, and a rename dialog for the batch.
- A destination **picker** (folder list, breadcrumbs). See 2.3.
- Creating the destination folder from the modal. R3; `POST /api/fs/create`
  exists, chaining it silently is the implicit-`mkdir` the create rule refuses.
- Batch selection through the terminal or the WorkBench, and any connector
  reaching this route.
- Column/checkbox selection UI: the marks are a row style, no checkboxes.

## 7. How it is verified

**Backend** — `tests/test_editorfs_write.py`, a `TestMoveBatch` class beside the
rename one, mirroring its list (each rule above is one test):

- two files land, both are readable at the new path, no backup is written (a
  move changes no bytes — the rename's argument, restated once);
- one occupied target → 409 with the `{reason: "exists"}` object **and both
  files still at their original paths** (this is the R2 test, and the only one
  that proves the ordering);
- one `ro`/`hidden` source, one `ro`/`hidden` destination → refused in the
  policy's own words, nothing moved;
- destination missing → 409 naming it; destination is a file → 400;
- a folder in `paths` → 400 and the folder survives whole;
- a symlink in `paths` → 400; an escape attempt on either field → 422;
- empty `paths` → 400, over-cap → 400 naming `MAX_MOVE_BATCH` (patched by name);
- a chat reply streaming → 409, the same message the rename test asserts;
- the source and destination are the same folder → 200 no-ops are **not**
  produced: the target equals the source, so it is occupied → 409. Stated so
  nobody "fixes" it by skipping;
- malformed bodies → 422, not 500.

**Frontend unit** — `web/tests/useEditorMove.test.js` (**new**; the refusal keeps
the prompt open with the typed value; success re-points an open buffer's key and
`state.active`; two re-lists, not N; one notice) and
`web/tests/useWorkspaceTreeCommands.test.js` additions (marks: toggle, range,
clear on plain click, `batchFor()` across parents, marks dropped by
`dropBranch`/`reRoot`, `Shift`+arrow chord).

The subject rule is tested where it lives — `batchFor()` in
`useWorkspaceTreeCommands.test.js`, and the menu table in
`web/tests/workspaceTreeMenu.test.js` (which mounts `WorkspaceNavigator` and
asserts verbatim label arrays) **plus** the table assertions in
`useWorkspaceTreeCommands.test.js` (which assert verbatim id arrays):

| case | expected |
|---|---|
| file row, no marks | `move-batch` present, `moveCount` 1 |
| folder row, root row | absent (`scope: 'file'`) — unchanged |
| 2 marks same parent, menu on a marked row | present, `moveCount` 2 |
| 2 marks same parent, menu on an unmarked file | present, `moveCount` 1 |
| 2 marks across parents, menu on a marked row | present, `moveCount` 1 |

Marks containing a folder cannot happen by construction (R4) and is asserted
anyway, so a future gesture that marks folders fails here rather than shipping a
batch that moves a tree.

**e2e** — `web/e2e/02-editor-write.spec.js`. Note this correction: the existing
`each row offers exactly what it can honour` test (`:281`) asserts a file row's
item list **verbatim** at `:310` and a markdown row's at `:318`, and the new
item is present for both — so **both arrays change**, gaining `Move to…` after
`Rename…`. The root and folder arrays do not. The same holds for the verbatim
id arrays in the two unit files above. Any implementation that leaves those
tests alone is asserting an item the feature removed.

Then these new cases:

- `Ctrl`+click two seeded scratch files → right-click one → `Move to…` → type the
  destination → both rows in the new folder, both on disk, no error notice, and
  a dirty buffer of one of them still dirty and still saving to its new path (the
  rename test's proof, at batch scale).
- A single right-click with no marks → `Move to…` → the field is prefilled with
  the file's own folder → move it → one row moved, its tab re-pointed.
- A refusal: destination does not exist → dialog still open, the server's
  sentence visible, nothing moved.
- The prefill's exact text, asserted rather than eyeballed — in the suite's own
  scratch tree, never the real lot (the `todos/` paths in 2.3 illustrate the
  gesture only): a scratch file at `<scratch>/nested/a.any` → the field holds
  `<scratch>/nested` (not the absolute location, not the bare `nested`). Then a
  destination typed with a **leading slash** is refused in the field before any
  request goes out, and one typed as a **bare segment** reaches the server and
  comes back with the not-there sentence naming the folder it looked for.
- The narrowing case from 2.2b: two marks across parents, right-click one, and
  assert the dialog's list names **one** path.

## 8. AGENTS.md impact

One sentence, in the SSE/write-surface area, only if the wording of the
existing invariant needs widening — the file already says a `/api/fs` write
route is SPA-only and names the test that asserts it, and that test now covers
`move_paths_batch` too. No counts, no route list.
