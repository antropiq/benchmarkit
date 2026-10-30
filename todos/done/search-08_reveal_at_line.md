# search-08 — Reveal at the line: `revealPosition`, the queue, the watcher

Make the caret land on the match. Clicking a hit already opens the file (search-07);
after this task it opens **there**. One exposed method on the editor surface, one queue
inside it, one watcher in the pane. This is the only genuinely new frontend plumbing in
the feature, and all of its difficulty is timing.

Depends on: **search-05** (`state.pendingReveal`, set by `openHit`). Not on search-07:
this task can be built and reviewed without the gesture existing.
Read `docs/tree-search-spec.md` §2.6 and §4's two edit rows for these components;
decision D11 and *Traps* §5 come from `todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `web/src/components/MonacoEditor.vue` | `defineExpose({ revealPosition })` + the internal queue |
| `web/src/components/EditorArea.vue` | one `watch` on `pendingReveal` |
| `web/src/lib/monaco-surface.js` | the pure `clampReveal` helper (see Tests) |
| `web/tests/monacoRevealClamp.test.js`, `web/tests/editorRevealWiring.test.js` | **new** — the clamp, and the watcher wiring |

## The problem, stated exactly

`MonacoEditor.vue` has **no `defineExpose` today** — it is props-in /
`content-change`-out with no imperative surface. And the reveal cannot be applied at the
moment of the click, for two independent reasons this component already documents:

1. the file may not be open yet, so the Monaco surface may not exist when the row is
   clicked — the reveal has to wait for `onMounted`;
2. when the file *is* already open, `EditorArea` keeps **one** `MonacoEditor` instance
   across tab switches (`:content="tab.text"`, **no `:key`**, `EditorArea.vue:506-512`),
   and this component's own watcher on `props.content` calls `model.setValue` — whose
   comment (`MonacoEditor.vue:130-133`) says `setValue` **clears the undo stack and the
   cursor**. So a reveal applied *before* that watcher has run is erased by the swap.

The queue lives **inside `MonacoEditor.vue`**, not in `EditorArea`, because only this
component knows when its model swap has finished. `EditorArea` only says "reveal this".

## What to deliver

### `MonacoEditor.vue`

```js
const pendingReveal = ref(null)   // { line, column }

function revealPosition({ line, column } = {}) { … }
defineExpose({ revealPosition })
```

- `revealPosition(position)` validates (positive integers; anything else is ignored
  silently — a reveal that cannot be applied is a no-op, never an error the pane has to
  handle), stores it in `pendingReveal`, and applies it **if the surface exists**.
- Applying means exactly §2.6's two calls:
  `editor.revealLineInCenter(line)` then `editor.setPosition({ lineNumber: line, column })`
  — in that order, because `setPosition` on an off-screen line does not scroll.
- **Apply it on the next tick after the content watcher ran**, not synchronously:
  `nextTick` inside `revealPosition`, and *also* drain `pendingReveal` at the end of the
  `props.content` watcher (after the `setValue`) and at the end of `onMounted`. Those
  three drains are the whole fix for reason 2 and reason 1 respectively — write them as
  one small `flushPendingReveal()` helper with a comment naming the `setValue` clears,
  so the next reader understands why a drain belongs in a content watcher.
- Clamping: a stale hit may name a line that no longer exists in the buffer (the file
  changed since the scan). Clamp `line` to `model.getLineCount()` and `column` to
  `model.getLineMaxColumn(line)` before calling Monaco — Monaco's own behaviour on an
  out-of-range position is not part of this contract, and a silent clamp beats a throw
  from inside a click handler.
- **No decorations.** No match highlight, no overlay: §2.6 says a highlighted-match view
  across a file is Monaco's own find widget's job. Monaco's `renderLineHighlight: 'line'`
  (already set at `:103`) is what shows the user where it landed.
- Disposing: `pendingReveal` is a ref, nothing to dispose. Do not add a `monaco.editor`
  API call outside the existing `editor`/`model` pair.

### `EditorArea.vue`

```js
watch(
  () => search.state.pendingReveal,
  (reveal) => {
    if (!reveal) return
    editorSurface.value?.revealPosition({ line: reveal.line, column: reveal.column })
    search.state.pendingReveal = null
  },
)
```

- a `ref` on the `<MonacoEditor>` element at `:506` to reach the exposed method. **A
  template ref on a component whose instance may be absent is the whole reason for the
  `?.`** — when the active tab is a diff or a preview, that `<template v-else-if="tab">`
  branch is not rendered and there is no surface at all. (Note the ref is also `null`
  while the Editor tab has no active tab at all — same `?.`, same reason.)
- clearing `pendingReveal` here (the consumer owns the clear) so a re-render does not
  re-apply an old reveal and the queue has one owner.
- **Order with the open.** `openHit` sets `pendingReveal` *then* publishes the selection,
  so this watcher can fire before `openFile` has fetched the text. That is correct and
  must not be "fixed" by reordering: the queue inside `MonacoEditor` is what absorbs it,
  because the drain at the end of the content watcher runs when the new text lands. If
  you find yourself wanting to await the open here, stop — that is the synchronous-reveal
  bug D11 exists to prevent.
- A reveal for a path that fails to open (binary, over the cap, gone) simply never finds
  a surface with those coordinates; the pane's existing refused-open message is the
  answer and this task adds nothing to that path.
- Keep the pane's existing watchers untouched: the `tree.state.selected` watcher
  (`:116-123`) and the `state.active` → `reveal()` watcher (`:133-139`) both stay
  byte-identical. This is a third, independent watcher.

## Tests

Monaco cannot be driven in jsdom — `web/tests/probe-monaco.test.js` is a browser probe
and `vitest.config.js` excludes `MonacoEditor.vue` from coverage for exactly this
reason — so **do not assert on Monaco**. §6 says the reveal is not unit-tested and is
covered by e2e instead (search-09). What *is* unit-testable is the queue's decision, and
the way to test it without a Monaco instance is to extract the queue as a tiny pure
helper in the component's own file or in `lib/monaco-surface.js` (the module both Monaco
surfaces already share):

```js
// lib/monaco-surface.js
export function clampReveal(position, lineCount, maxColumnFor) { … }
```

New `web/tests/monacoRevealClamp.test.js` (or add to an existing `monaco-surface` test
if one exists — check first): a position inside the file passes through unchanged; a
line past the end clamps to the last line; a column past the line's end clamps to
`maxColumn`; a zero/negative/NaN/fractional line or a non-object is rejected (`null`).

If extracting a helper feels like too much indirection for one clamp, the alternative is
to unit-test only `EditorArea`'s side: mount the pane with a stub child exposing a
spy `revealPosition`, set `pendingReveal`, `flushPromises`, and assert the spy was called
once with `{ line, column }` and that `state.pendingReveal` is now `null`. That case is
worth having **whatever you choose** — it is the wiring, and it is the half that can
break silently:

1. the spy is called with the hit's line and column, exactly once;
2. `pendingReveal` is cleared afterwards;
3. with the active tab a **diff** (no Monaco branch rendered), setting `pendingReveal`
   throws nothing and leaves it cleared — a stale hit on a diff tab must not error.
   Mounting the real `EditorArea` in jsdom pulls Monaco in, so stub the child (a
   `vi.mock` of `./MonacoEditor.vue` exposing a spy) rather than mounting the real one.

## Acceptance

- `cd web && npm test` green; `cd web && npm run build` green.
- `grep -n "defineExpose" web/src/components/MonacoEditor.vue` finds it, and
  `revealPosition` is the **only** thing exposed — this component gains no other
  imperative surface.
- `grep -n "revealLineInCenter\|setPosition" web/src/components/MonacoEditor.vue` shows
  both, in that order, inside one function.
- `git diff --stat` shows `MonacoEditor.vue`, `EditorArea.vue`, `lib/monaco-surface.js`
  (the helper only) and the two new test files.
  **No** change to the `props.content` watcher's `setValue` logic itself (its
  `applyingExternal` guard and the EOL comment stay as they are — the drain is appended
  after it), and no change to `web/vitest.config.js`.
- Real behaviour is verified in the browser by search-09, not here; if you can run
  `cd web && npm run test:e2e` (needs `.env.test` reachable), say in the final note
  whether the caret landed, and do not add a Playwright spec of your own — that file is
  search-09's deliverable.
- Move this file to `todos/done/` when the above is true.
