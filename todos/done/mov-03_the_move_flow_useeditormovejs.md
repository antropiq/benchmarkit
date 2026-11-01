# mov-03 — `useEditorMove.js`: the flow, the field rule, the reconciliation

Write the composable that owns the whole move: the prompt state a dialog will render,
the shape check on the destination field, the one `POST /api/fs/move-batch`, and the
success reconciliation that re-points buffers and re-lists exactly two folders. Plus its
own new unit test file. This task adds no visible affordance — a caller does not exist
until mov-05 — but it is the piece the dialog and the menu both hang on.

Depends on: **mov-01** (the route and its payload contract) and **mov-02**
(`batchFor`, `clearMarks`, `marked`).
Read `docs/tree-move-batch-spec.md` §2.4 (the field rule and its two asymmetries),
§5.1 (the flow), §5.2 (the reconciliation, step by step) and §7's frontend-unit list.
Decisions D6, D7, D8, D9, D10, D13, D14, D17 and *Traps* 1, 9 come from this task.

## Files

| file | change |
|---|---|
| `web/src/composables/useEditorMove.js` | **new** — `startMove`, `cancelMove`, `submitMove`, `validateMoveDestination`, `state.movePrompt = { paths, parent, value, error, busy }`, one `inFlight` guard |
| `web/tests/useEditorMove.test.js` | **new** — the flow: refusal, success, the two-re-lists rule, one notice, a dirty buffer surviving the move |

Two files. Nothing else: the dialog is mov-04, the menu item is mov-05.

## What to deliver

### The state, in the composable

`movePrompt` is this composable's own singleton reactive state — one
`{ paths, parent, value, error, busy }` at a time, so a tab switch or a remount cannot
drop an open prompt (the rename keeps its slot in the tree because its input renders
*inside* a listing; a modal has no such host). `startMove(batch)` takes the
`{ paths, parent }` object `batchFor` returned and **copies the list** into the prompt
immediately (D4: no later click can empty the list the dialog is about to show), with
`value` prefilled to `batch.parent`.

### `validateMoveDestination(value, { parent })` — the shape half only

Mirror `validateRenameTarget`'s shape checks and nothing else: empty; a `\`; a NUL; a
leading `/` or `~`; a `.` or `..` segment; and the one new check — a destination equal
to the sources' own parent, answered `That is where they already are`. Return `null`
when the value may be sent, a sentence when it may not.

- The **prefill makes N=1 open with Move disabled** and that sentence already showing
  (D7). That is accepted deliberately — do not "fix" it by leaving the field empty.
- A destination *under one of the selected files* is impossible by construction (R4: no
  folder rows), so **no cycle check is written**. If you feel one is missing, the
  invariant it depends on is in mov-02.
- The client refuses **any** `.`/`..` segment while the server only refuses a `..` that
  leaves the workspace (D9). That asymmetry is deliberate: do not relax the client to
  match, and do not add a second path model to explain `..`.
- **No copy of the deny-list in the client** (D10) — no `.env`, no `.git`, no `ro`, no
  `hidden` check here. `useEditorRename.js`'s header states the same split of rules;
  keep the wording of this file's header consistent with it.

### The request

`submitMove()` posts `{"paths": [...], "to_parent": value}` to
`POST /api/fs/move-batch` — the contract both halves agreed on in the big picture. Send
the **workspace-relative** value verbatim (D8): no absolute path, no location prefix, no
joining of the target (the server joins it, D11).

A **busy guard**: one `inFlight` promise, the same reason `useEditorRename` has one
(a second Enter for one batch makes the second answer a 409 about the first).

On refusal: the prompt **stays open**, `value` keeps what was typed, `error` holds the
server's sentence, and a matching `notices.notice('error', …)` is raised — the same
pair the rename raises, with `from → to` replaced by `N files → destination`. Read the
`{reason: "exists", path, dir}` object the way `submitRename` does (it is an object, not
a string) and phrase it from the server's reason. **Nothing moves on refusal**: no
re-point, no re-list, no clear.

### The success reconciliation (§5.2) — the exact order

For each `{"from","to"}` in the response's `moved` array, in order:

1. `files.repointPath(from, to)` — **first**, for the reason `useEditorRename.js` states:
   a tick of the freshness probe must never see a tab whose path just vanished;
2. `tree.dropBranch(from)` — the cheap case, a file has no subtree.

Then, **once each across the whole batch** (*Trap 9*, and the point of D14):

3. `refreshBranch(folder)` for the set `{parentOf(from)} ∪ {parentOf(to)}` — normally
   exactly two folders, awaited together the way `submitRename` does it;
4. `refreshListingOfParent` at both ends, for the `children` flag promise a collapsed
   ancestor carries.

Then, after the loop: `tree.clearMarks()`, **one** `git.load({ force: true })`, and
**one** notice — `Moved 3 files → notes/archive` (per-file notices would bury the tab
strip). Then `tree.expand(destinationParent)` so the user sees where the rows went, and
`reveal` the last moved row. Clear `movePrompt` on success.

### And on a working-location switch

`movePrompt` carries paths from the root being left, so a `reRoot` closes it the way the
tree forgets `selected` and `editing`: subscribe through the tree's **re-root subscriber
registry** (it takes several subscribers since the search lot), do not claim a slot of
your own and do not add a second registry. Cancel means `cancelMove()` — drop the prompt,
issue nothing.

### Tests — `web/tests/useEditorMove.test.js` (new)

Stub `fetch` the way `useEditorRename.test.js` and `useEditorWriteMatrix.test.js` do,
and record every request the flow issues. The assertions §7 names:

- a refusal keeps the prompt open with the typed value, sets `error` to the server's
  sentence, and issues **no** re-list (nothing moved);
- success re-points an open buffer's key and the active tab; a buffer with unsaved edits
  is still dirty after the move and its next save goes to the **new** path;
- **two re-lists, not N** — with three files moved, count the `/api/fs/tree` requests:
  this is the assertion that catches the per-file loop;
- **one** notice, and the marks cleared afterwards;
- `validateMoveDestination`: the empty field, a leading `/`, a bare `..` segment, a `\`,
  and the destination-equals-parent sentence that the prefill produces at N=1;
- the `inFlight` guard: two `submitMove()` calls issue one request.

Import `describe/it/expect/beforeEach` from `'vitest'` explicitly (`globals: false`).

## Rules this task must not break

- *A `/api/fs` write route is reachable from the SPA only*: this file is one of the few
  allowed homes for a write marker — mov-01 already added
  `web/src/composables/useEditorMove.js` to `ALLOWED_HOMES`, so name the route in the
  `fetch` body and nothing else may name it. `pytest tests/test_write_surface_isolation.py`
  must still pass after this file exists.
- *Run/stream state lives in the composables*: the dialog (mov-04) renders
  `movePrompt`, it does not hold it.
- The reconciliation sequence belongs to `useEditorRename.submitRename()` and must stay
  identical here per path — buffers first, then the tree forgets, then the two listings,
  then git, then the notice. Re-deriving a *different* order is how a moved buffer starts
  offering "reload a file that was only moved".
- Reuse, do not re-implement: `repointPath`, `dropBranch`, `refreshBranch`,
  `refreshListingOfParent`, `reveal`, `expand`, `git.load`, `notices.notice`. No new
  refresh machinery, and no new notice channel.
- No `v-html`, no raw HTML (this file renders nothing).
- No `useEditorMode.isEditable` check inside the composable: that gates the **button**
  in mov-04 (D17); the server is the authority on the write.

## Done when

```bash
cd web && npm test         # useEditorMove.test.js green; the rest of the suite untouched
cd web && npm run build    # the new module compiles
pytest tests/test_write_surface_isolation.py -q   # the marker scan accepts the new file
```

The feature is still unreachable from the UI at this point. That is expected — say so in
the task report rather than wiring an entry point in.

## Out of scope here

any `.vue` file; the menu item and `menuFlags`; the dialog; drag-and-drop; copy; undo;
creating the destination folder when it is missing (R3 — the refusal stays a refusal); a
collision table (D13); a keyboard chord; any retry or re-post of a refused batch.
