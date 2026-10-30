# search-06 — `SearchDialog.vue` and its card in `theme.css`

The modal itself: the scope label, the query field, the two checkboxes, the results
table, the inline error and the truncation footer — all rendered off
`useWorkspaceSearch`'s state. Nothing raises it yet (search-07 wires the gesture) and
nothing consumes the reveal yet (search-08), so the row click is wired to
`openHit` and its side effects can be asserted on the state.

Depends on: **search-05** (the composable), and search-02's `footerNote` wording (this
task asserts against it verbatim).
Read `docs/tree-search-spec.md` §2.3, §2.8, §4's `SearchDialog.vue` section and §6's
frontend rows; decisions D8, D12, D13 come from `todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `web/src/components/SearchDialog.vue` | **new** |
| `web/src/styles/theme.css` | one new dialog block, in the family of the other Editor dialogs |
| `web/tests/searchDialog.test.js` | **new** |

## What to deliver

`<script setup>` + a top-level `<dialog>`, `showModal()`/`close()` driven by
`state.open` through a `watch` — copy the shape from `EditorTreeDialogs.vue:153-159`
(the close confirm): `if (isOpen && !el.open) el.showModal()`,
`else if (!isOpen && el.open) el.close()`. That file is the house pattern; do not invent
a `v-if` on the dialog (a `v-if` unmounts it and loses the input's caret position and
any scrolling in the results).

Markup, top to bottom, exactly §4's five parts:

1. **the scope label** — read-only, before the field, not an input:
   `Search in: {{ scope.label }}` plus the kind as a caption
   (`folder` / `file` / `whole workspace`, from `scope.kind`). The label is derived from
   the row the gesture came from and is never typed (§2.8).
2. **the query `<input>`** — `autofocus`, and `Enter` submits (a bare `<input>` in a
   form-less dialog needs an explicit keydown; do not wrap everything in a `<form>`
   just to get that, and do not let `Enter` in the field submit a *form* that would
   reload). Bound to `state.query`; `run()` on submit.
3. **the two checkboxes on one line** — `Case sensitive` and `Regex`, defaults off, the
   regex one carrying a `title` that names where the pattern runs: **`regex — Python
   syntax`** (§2.3: the honest thing is to say the pattern did not come from a JS
   engine).
4. **the results table** — columns **path**, **line**, **snippet**; or the empty state
   ("no matches"), or the inline error line.
5. **the footer** — `state.result.footerNote`, rendered verbatim when non-empty
   (D8: the *server* wrote that sentence from the caps in `constants`, the dialog does
   not reassemble it and must not add counts of its own).

### The snippet cell (D12 — the invariant of this task)

The snippet is a line of someone's file. It is bound as a **string**:

- never `v-html`; never imported through `lib/markdown.js` (that module is for rendered
  HTML, and this is not HTML — the project rule about markdown does not extend here and
  must not be satisfied by accident by piping file bytes through a sanitiser);
- the matched span is marked with a **separate `<span>` around a slice of the string** —
  three text nodes: `snippet.slice(0, i)`, `snippet.slice(i, i + matchLength)`,
  `snippet.slice(i + matchLength)`, where `i` is the offset of the match *within the
  snippet*. Never build HTML from the bytes;
- **`matchLength: 0` is legal** (a zero-width regex match, or a snippet cut before the
  match): the highlight span must be empty, not the whole line. Clamp
  `min(column - 1, snippet.length)` and clamp the end too — a snippet trimmed at 240
  chars can end *before* the match starts, and the highlight then belongs at the end
  with zero width. A wrong slice here shows the highlight on the wrong text, which is
  worse than no highlight;
- the path cell shows the workspace-relative path **verbatim**, the same string the tree
  shows and the same string `openFile` takes.

### Rows are buttons

§4 allows either a `<button>` in the row or a `<tr>` with a keypress handler. Use the
button — the tab strip's chips are the precedent, and a real button is keyboard
reachable and carries the `title` without a handler. Clicking calls
`search.openHit(hit)`, which closes the dialog itself (D13); do not also call
`closeDialog()` here, or the close has two owners.

### Error and busy states

- `status === 'error'`: **one line under the field**, in the dialog, showing
  `error.detail`. §6 is explicit: a 400 renders `re`'s message *inline rather than as a
  notice* — so do not route it through `useEditorNotices` / the app's notice stack. The
  table below it keeps showing the previous answer (the composable left `result`
  alone); do not blank it.
- `status === 'running'`: a busy marker (a disabled submit, one caption "searching…")
  **while the old table stays on screen** — same reason as the error: the previous
  answer is the thing being compared against.
- Never show `error` and a footer at the same time as though both were current: the
  footer belongs to `result`, and `result` is by definition the last *successful*
  search. Say so in a comment on the footer element, because the pairing looks like a
  bug.

### Escape and the backdrop

Escape on a `showModal()` dialog fires `cancel`/closes the element natively, and the
`watch` above would then disagree with `state.open`. Add `@close` →
`search.closeDialog()` (the same handler `EditorTreeDialogs` uses as
`onNativeClose`) so the native close writes the state rather than fighting it, and a
backdrop click → `closeDialog()` via the `event.target === dialogEl` test that file also
uses. `closeDialog()` keeps query, scope and result (D14) — the native-close handler
must not clear them.

### `theme.css`

One block beside the other Editor dialogs (`:3112-3176`), same shape:
`.editor-search-dialog > article { width: …; max-width: 94vw; border; border-radius }`.
§4's table needs more width than a confirm — the path and snippet columns both want to
breathe, and a 480 px card makes the snippet column useless. Pick a card width in the
700-860 px range and a max-height on the results area so 500 rows cannot make the dialog
taller than the window (the table scrolls, the header and footer stay). No inline
styles; the highlight span's colour comes from the theme's existing variables.

## Tests — `web/tests/searchDialog.test.js`

Mount with `@vue/test-utils` (the suite already does this — `workbenchGutter.test.js`,
`settingsSyspromptHelp.test.js`). **jsdom has no `showModal`/`close` on `<dialog>`**, so
patch both on `Element.prototype` the way `useEditorWriteMatrix.test.js:385-390` does
(`this.open = true` / `false`), and remove the mounted host in `afterEach`. Import
`describe/it/expect` from `'vitest'` explicitly.

1. the scope label renders the **folder** path, the **file** path, and the workspace
   **basename** for the root case — three mounts, three labels, and the root one is not
   the empty string (§6's row, verbatim);
2. an invalid-pattern 400 renders `re`'s message inside the dialog and **not** in
   `.editor-notice--*` (assert the notice stack has no such entry — that is the
   difference §6 calls out), while the previous table is still in the DOM;
3. **a snippet is never injected as HTML**: put `<img src=x onerror=alert(1)>` in a
   fixture snippet, assert the cell's `textContent` contains it literally and its
   `innerHTML` contains no `<img`;
4. the highlight: a match at column 8 with `matchLength 4` wraps exactly those four
   characters in the highlight span, and the surrounding text is in separate nodes;
   `matchLength: 0` produces an empty highlight span, not a whole-line one; a snippet
   trimmed before the match position still renders without throwing;
5. the footer renders `footerNote` verbatim when present, and is absent when it is `""`;
6. `status === 'running'` shows the busy marker with the old table still present;
7. clicking a row's button calls `openHit` → `state.open` is false and
   `state.pendingReveal` carries that hit's `{ path, line, column }` — and the
   component did **not** call `closeDialog()` itself (the composable owns the close);
8. Escape (the native `close` event) ends with `state.open === false` **and**
   `state.query`/`state.result` still set — the retention rule seen from the element;
9. `Enter` in the query field triggers one request and a `<form>` is not submitted
   (no page navigation: assert no `submit` handler ran and the stubbed `fetch` fired once).

## Acceptance

- `cd web && npm test` green; `cd web && npm run build` green.
- `grep -n "v-html\|markdown" web/src/components/SearchDialog.vue` finds nothing (D12).
- `grep -c "footerNote" web/src/components/SearchDialog.vue` is 1 — the string is
  rendered, never reassembled (D8). No `${result.matchCount} matches`-style client
  composition of cap numbers.
- `git diff --stat` shows exactly the three files above. No `tree-menu.js`, no
  `WorkspaceNavigator.vue`, no `MonacoEditor.vue` — the dialog is not reachable from the
  app yet, and that is expected: say so rather than adding a button to see it.
- Move this file to `todos/done/` when the above is true.
