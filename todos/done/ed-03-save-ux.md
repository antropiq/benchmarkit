# ed-03 — Save, wired to the hand

ed-01 made saving possible on the server; this task makes it something a human
actually does, and closes the three ways the app currently lets unsaved work
vanish: a silent `Ctrl+W`, a silent middle-click close, and a page reload with no
warning.

Depends on: ed-01 (the route), ed-02 (a save can now come back as a conflict, so
the affordances have to handle that answer).

## 1. `saveFile(key)` in the composable

The single entry point; every affordance calls it, nothing else writes.

```
saveFile(key)
  record = recordForKey(key)              // diff records are never saved
  if (!record || record.kind !== 'file') return { ok: false, reason: 'not-savable' }
  if (record.binary || record.truncated)  return { ok: false, reason: 'partial' }
  if (record.saving)                      return { ok: false, reason: 'in-flight' }
  record.saving = true
  PUT /api/fs/file { path, text: record.text, ifUnmodifiedSince: record.mtime }
    200 → record.baseline = record.text; record.dirty = false
           record.mtime = body.mtime; record.size = body.size
           record.stale = false; record.conflict = null
           → ask useGitStatus().load({ force: true })  // the tree's marks must
             follow the write, or a saved file keeps looking unmodified
    409 (stale) → record.stale = true, record.conflict = 'stale',
                  state.conflict = {…}            // the dialog from ed-02
    409 (chat)  → notice, buffer stays dirty
    4xx/5xx     → notice with the server's detail, buffer stays dirty
  finally record.saving = false
```

Rules worth restating because they are the ones a later refactor breaks:

* **`baseline` moves only on a confirmed 200.** A failed save leaves the dot.
* `saving` is per-record state so two chips can both look busy; it is also the
  re-entrancy guard for a held `Ctrl+S`.
* `saveAllDirty()` iterates the dirty file records sequentially (not `Promise.all`):
  parallel writes to the same file from one client are how a self-inflicted 409
  happens, and the order the user sees is the order of the tab strip.

## 2. Keybindings and buttons

| Chord | Where | Action |
|---|---|---|
| `Ctrl/Cmd+S` | `EditorArea.vue`'s existing `onKeydown` (it already gates on `bench.state.activeTab === 'editor'`, `EditorArea.vue:135`) | `saveFile(state.active)` |
| `Ctrl/Cmd+Shift+S` | same handler | `saveAllDirty()` |
| `Ctrl/Cmd+W` | same handler | **confirm first** when dirty (§3) |

A Monaco `addCommand`/keybinding is the other way to do `Ctrl+S`; do **not** use
it: the app already owns one keydown handler for the pane (the `Ctrl+W` rule) and
splitting the chord handling across two owners is how one of them stops working
when the pane is read-only. Keep the window listener, and `preventDefault()` so
the browser's own Save-page dialog does not appear.

The header of the pane gains a small save affordance — a Save button that is
disabled unless the active record is dirty, plus a Save-all button that appears
only when more than one buffer is dirty. Both in the `.editor-area-head` row
beside the tab strip (the strip already scrolls with the wheel rule, so the
buttons live outside the scroller). Icon + `title` + `aria-label`, the same
conventions as the chip's ×.

`MonacoEditor.vue:35`'s comment ("read-only by default: the file surface has no
write route to disk yet") gets its reason updated here — the default is still
safe, but the reason is now "the pane decides", not "there is no route".

## 3. Never lose work silently

* **Confirm-on-close.** A dirty record cannot be closed by one gesture. The chip's
  ×, middle-click and `Ctrl+W` all route through `requestCloseTab(key)`, which
  saves-or-discards-or-cancels through a dialog with the file's name in the
  question and **three** answers: Save and close / Discard / Cancel. The default
  is Cancel — a destructive default in a confirm dialog is a bug.
* **A closed dirty tab is recoverable only in this session**: closing drops the
  buffer, so the dialog is the last line of defence, not a trash can. Say so in
  the Discard label ("Discard — the changes are lost").
* **`beforeunload`.** Uncomment and fix the listener at `useTerminal.js:366-377`:
  move it out of the terminal's file (it belongs to the buffers) and gate it on
  `useEditorFiles().state.open.some((item) => item.dirty)`. Keep the two listeners
  separate exactly as that comment instructs — cancelling this prompt must not
  have closed the terminal's PTY via the `pagehide` teardown beside it.
* **Editor-tab switch is *not* a close.** Buffers survive `v-show`; do not add a
  confirm there, it would fight the invariant that keeps the pane's state alive.

## 4. What a save looks like

* the dot disappears on the 200, not on the click;
* while `saving`, the chip shows a spinner-ish class (`.editor-tab-saving`) —
  a save of a large file is not instantaneous;
* the pane's `state.dirtyCount` (a small computed) drives the Save-all button and
  a title on the tab strip ("3 unsaved");
* the git marks update (the `load({ force: true })` above), so the row the user
  just saved gets its modified marker.

## 5. Tests

Extend `web/tests/useEditorWrite.test.js`:

1. a 200 re-baselines (`dirty === false`, new `mtime`/`size`) and the next save
   sends the **new** token;
2. a 409-stale sets `conflict` and leaves `dirty` true;
3. a 4xx leaves everything but the notice;
4. `saving` guards re-entrancy (two calls → one request);
5. a `truncated` or binary record is refused client-side, no request issued;
6. a diff record cannot be saved;
7. `saveAllDirty()` issues one request per dirty record, in tab order, and stops
   on the first failure without marking the rest saved;
8. a successful save calls `useGitStatus().load({ force: true })`;
9. `requestCloseTab` on a dirty record does not remove it until Discard/Save.

For the chord: prefer extracting the decision (which key, which tab is visible,
what it means) into a pure `lib/editor-keys.js`, like `lib/tab-strip-wheel.js`
already does for the strip, and test the module. Mounting `EditorArea` with
`@vue/test-utils` (a devDependency, jsdom is configured) to dispatch a `keydown`
on `window` is the fallback if the pure module feels like over-abstraction — it
is the slower test, not the better one.

## 6. Acceptance

* `Ctrl+S` saves; the dot goes away; a second `Ctrl+S` is a no-op (clean).
* Two dirty buffers, `Ctrl+Shift+S` → both saved in order.
* `Ctrl+W` on a dirty buffer asks; Cancel keeps the buffer; Discard drops it;
  Save-and-close saves then drops.
* Reloading the page with a dirty buffer prompts.
* A save while a chat reply streams is refused, and the refusal is visible —
  nothing about the buffer changed.
* After saving, the tree shows the file as modified (git marks updated).
* `cd web && npm test` and `pytest` green.
