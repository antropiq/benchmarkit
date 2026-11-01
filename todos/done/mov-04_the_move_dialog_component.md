# mov-04 — `EditorMoveDialog.vue`: the modal, mounted and styled

Give the batch a face: one `<dialog>` that lists what it is about to move, holds the
destination field, and offers `Cancel` / `Move N files`. It decides **nothing** — every
value it shows and every rule that enables its button comes from `movePrompt` and
`validateMoveDestination` in mov-03. Mount it beside the other Editor dialogs and give it
its `theme.css` block.

Depends on: **mov-02** (the marks the footer/labels talk about) and **mov-03**
(`movePrompt`, `submitMove`, `cancelMove`, `validateMoveDestination`).
Read `docs/tree-move-batch-spec.md` §2.3 in full (the content table, the worked prefill,
the two rejected alternatives) and §5.1's steps 3-6. Decisions D6, D7, D10, D17 and
*Trap 5* come from this task.

## Files

| file | change |
|---|---|
| `web/src/components/EditorMoveDialog.vue` | **new** — the `<dialog>` and its buttons |
| `web/src/components/EditorArea.vue` | import + `<EditorMoveDialog />` beside `<EditorTreeDialogs />` |
| `web/src/styles/theme.css` | the dialog's class, beside the existing `editor-*-dialog` block |

Three files. If the component starts importing the tree or the file layer to answer a
question, it has grown past this task.

## What to deliver

### The content, and nothing more (§2.3's table)

| part | text |
|---|---|
| title | `Move N files` (N = `movePrompt.paths.length`; at N=1 the honest singular, `Move 1 file`) |
| list | every source path, one per line, in `.caption.mono` — the delete confirm's shape |
| field | one `<input>`: the destination **folder**, workspace-relative, prefilled with the sources' parent (mov-03 put that value there) |
| hint | `The folder must already exist.` |
| buttons | `Cancel`, `Move N files` |

- The list is the answer to the narrowing case (D4/R1): a cross-parent selection shows
  **one** path and says `Move 1 file`, so the narrowing is visible before anything is
  sent. Render `movePrompt.paths` verbatim; never re-derive the subject here.
- Plain `{{ }}` text for the paths. No `v-html`, no `lib/markdown.js` — this is a path
  from the user's own tree, not rendered HTML (the project invariant is about *rendered
  HTML*, and a path is not any).
- One `<input>`, never a `<textarea>`: the value is one path, never multiline (§2.3).
- **No folder picker, no breadcrumbs, no folder list** (D10). Typing is the whole
  affordance, and the hint under the field is what makes the resulting refusal readable.

### Behaviour, all of it borrowed

- `<dialog>` + `showModal()` driven by composable state and reconciled by a `watch` —
  the house pattern this repo already has three times over (`EditorTreeDialogs.vue`,
  `SearchDialog.vue`).
- `Cancel` is the **first** button, so Esc, the backdrop and Enter all mean Cancel or the
  field's own submit — the shape `EditorTreeDialogs` documents. Enter in the field calls
  `submitMove()`; while `busy`, both buttons are inert and neither closes the dialog.
- Field `input` → write the value into `movePrompt.value`, then set `movePrompt.error` to
  `validateMoveDestination(value, …) ?? ''`. The **Move button is disabled while that
  returns a sentence** — which is how the N=1 prefill opens with Move disabled and the
  no-op sentence showing (D7): the edit *is* the enable.
- Move is additionally disabled when `useEditorMode.isEditable` is false (D17) — the
  badge's answer, and nothing more; the server still refuses the write itself.
- A **refusal leaves the dialog open** with the typed value and the server's sentence
  under the field (§5.1 step 6). Only `cancelMove()` or a success closes it. Do not close
  on a failed submit — a refusal is an edit, not a restart.
- Focus the field with its content selected on open, so the common case (edit the
  prefill) is two keystrokes.

### Styles

`.editor-move-dialog > article` and `.editor-move-dialog .caption`, beside the existing
`editor-conflict-dialog` / `editor-delete-dialog` / `editor-search-dialog` block, matching
those rules' shape (they are the same card). Light and dark both. If the source list needs
a scroll cap for a 40-file batch, that cap belongs here in CSS, not in the component's
logic.

## Rules this task must not break

- *Run/stream state lives in the composables*: the component holds only local view state
  (the element ref, at most a busy echo). `movePrompt` is imported, never duplicated — a
  second copy of the paths is how the dialog and the request disagree.
- *Rendered HTML always goes through `lib/markdown.js`, never raw `v-html`*: this file
  uses no `v-html` at all.
- One `<dialog>` per concern, mounted **once**. `<EditorTreeDialogs />` is already the
  single mount for its three dialogs; do not add this one to that file as well as to
  `EditorArea.vue`, and do not mount it in `EditorView.vue` too — a second `showModal()`
  host means two dialogs answering one prompt.
- It must not call `fetch`, must not import the composable's `submitMove` internals
  beyond the public functions, and must not know what `MAX_MOVE_BATCH` is: a too-large
  batch is the server's 400 and the dialog shows the sentence like any other refusal.
- No menu item and no `MENU_ACTIONS` entry in this task — mov-05 owns the entry point. The
  dialog is reachable only from a test until then.

## Done when

```bash
cd web && npm test      # nothing regressed; a small mount test for the dialog passes
                       # (stub showModal/close, per Trap 5)
cd web && npm run build # the component compiles into the Editor chunk
```

Asserted in the test: opening with `paths` of length 3 titles `Move 3 files` and lists all
three paths verbatim; the prefill's exact text is `movePrompt.parent` (not a basename, not
an absolute path); the button is disabled while the field equals the parent; a refusal
keeps the dialog open with the typed value.

## Out of scope here

the menu item, `menuFlags`, `MENU_ACTIONS`, any `tree-menu.js` change (mov-05); a folder
browser or autocomplete; a file picker; a "create this folder" affordance (R3); a detail
view of what moved; remembering the last destination typed; a rename affordance in the
same dialog (D6 — renaming stays `Rename…`'s job).
