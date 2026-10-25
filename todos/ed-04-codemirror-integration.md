# Step 4 — CodeMirror and tabs (**read-only surface**, decision D4)

Replace the `<pre>` preview with a real CodeMirror 6 editor: language support,
tab strip, cursor position, search. The editor is configured read-only and
there is **no save path** — this lot ships no write endpoint at all.

Depends on: ed-03 (open-file state, read endpoint). Needs ed-05's packages
installed — do them in that order.

## Why read-only, stated in the component too

Decision D4: the create / edit / delete behaviour of a standard editor belongs
to a **future version** (specced in ed-10), not to this lot. What this step
deliberately does *not* build:

* no `PUT /api/fs/file` (so no conflict rule, no backup, no atomic replace —
  all of that is ed-10, where it is designed together with the guards);
* no dirty tracking, no `Ctrl+S`, no `Save` button, no `beforeunload` warning;
* no mutation endpoints (create / rename / delete) either.

Two reasons worth keeping as a code comment, because they explain why read-only
is not a mere placeholder:

1. **it removes the worst data-loss race.** While a WorkBench conversation runs,
   the model's own tools (`apply_edit`, `revert_edit`, `console` commands) do
   write inside the workspace. A human buffer held over the same file would
   silently overwrite one side. Refusing the edit is the honest answer until
   there is a defined conflict protocol — which is what ed-10 specifies.
2. **it makes AGENTS.md's read-only-by-design invariant unbreakable here.** With
   no write endpoint, there is nothing for an MCP or LSP tool to reach, so no
   later refactor can accidentally expose a write path to the agent.

The read endpoints of ed-01/ed-03 stay the only file surface this tab uses, and
ed-09 makes the open buffers follow the model's writes.

## Packages (installed in ed-05, used here)

| Package | Why |
|---|---|
| `codemirror` | the preset: `basicSetup` (line numbers, history, search, autocomplete, closeBrackets, default keymaps) |
| `@codemirror/state`, `@codemirror/view` | explicit `EditorView`/`EditorState` control, so nothing depends on the preset's re-exports |
| `@codemirror/lang-python` | Python |
| `@codemirror/lang-javascript` | JS **and** TS (`{ typescript: true }`) |
| `@codemirror/lang-vue` | SFC: HTML plus a nested JS/TS `<script>`; pulls only `lang-html`/`lang-javascript` |
| `@codemirror/lang-json`, `@codemirror/lang-markdown` | the two most common extras in this repo |
| a hand-written `EditorView.theme` | colours bound to the Pico CSS variables |

No `@codemirror/merge`, no Vim keymap, no `@codemirror/lint` panel: diagnostics
belong to the existing LSP hub, and that integration is parked in ed-08. No
extra `@codemirror/commands` either — `basicSetup` already pulls what search
needs, and the edit commands go unused while the pane is read-only.

### Lazy language loading

One dynamic import per language, keyed by the `language` id from ed-03:

```js
const loaders = {
  python:     () => import('@codemirror/lang-python').then(m => m.python()),
  typescript: () => import('@codemirror/lang-javascript').then(m => m.javascript({ typescript: true })),
  javascript: () => import('@codemirror/lang-javascript').then(m => m.javascript()),
  vue:        () => import('@codemirror/lang-vue').then(m => m.vue()),
  json:       () => import('@codemirror/lang-json').then(m => m.json()),
  markdown:   () => import('@codemirror/lang-markdown').then(m => m.markdown()),
}
```

Opening a `.py` file therefore never downloads the Vue grammar. That is what
keeps decision D7 ("code-split harder") true: `vite build` emits one small
chunk per language instead of folding them into the app chunk.

## Component: `components/CodeMirrorEditor.vue`

Deliberately thin — state in the composable, transport in the component:

```
props: { model }             // the open-file record from useEditorFiles
onMounted:        new EditorView({ state, parent })   // state built with readOnly
watch(model.path): reconfigure extensions (language chunk)
watch(model.reloadedAt): swap in the document reloaded by ed-09
updateListener:   recompute Ln/Col for the status bar
onBeforeUnmount:  view.destroy()
```

Rules that matter in Vue specifically:

* **Never `ref()` the `EditorView`** — Vue's reactive proxy around CM6
  internals breaks its change handling. Keep it a plain `let view`.
* Extensions are rebuilt on tab switch (`StateEffect.appendConfig`) instead of
  recreating the view, and each open file caches its own `EditorState`, so
  switching tabs keeps each file's scroll, selection and (once ed-10 enables
  edits) undo history intact.
* `lineSeparator` is derived from the loaded text (`text.includes('\r\n')`) so a
  CRLF file renders as one line per row instead of one row with embedded CRs.
  Nothing writes it back in this lot; setting it now keeps ed-10 from changing
  how an existing buffer looks.
* `indentUnit` per language: 4 for Python (PEP 8), 2 elsewhere, matching this
  repo's own style (4-space Python, 2-space Vue).
* **Both** `EditorView.editable.of(false)` and `EditorView.readOnly.of(true)`
  are set, once per view rather than per file. They are different levers —
  `editable: false` also shuts the IME/clipboard mutation path — and ed-10
  relaxes them together, never one at a time. A one-line comment naming D4 and
  ed-10 belongs next to them, so the pair is not mistaken for wiring left over
  from an unfinished step.
* `autocompletion: false` overrides `basicSetup`: offering completions in a
  buffer that cannot be edited is a promise the pane cannot keep. Search
  (`Ctrl+F`), selection, copy, folding and the `Ln, Col` readout all stay live —
  read-only means no mutation, not a disabled pane.
* The tab chip shows the file name, the path in `title`, and a `×`. There is no
  dirty dot and no Save control in this lot.

## What ed-10 adds here (not half-built now)

`ed-10-editor-write-layer.md` specifies the write half: `PUT /api/fs/file` under
the `for_write` policy, an `ifUnmodifiedSince` mtime precondition (409 on
conflict), backup-then-atomic-replace, the create / rename / delete endpoints,
and the rule refusing a write while a chat reply streams. None of it is stubbed
here — no dead `saveFile()`, no commented-out `Ctrl+S` — because a half-built
write path is precisely the thing that could later leak toward the hubs.

The one piece this step keeps that ed-10 builds on: the read endpoint returns
`mtime` (ed-03), which is what ed-09's probe compares and what a future conflict
precondition replays.

## Tests

No new Python surface here, so `tests/test_editorfs.py` grows by one assertion
only: **no `PUT`/`POST`/`DELETE` route exists under `/api/fs/`**. It locks this
lot's read-only promise, and ed-10 removes it deliberately rather than silently.

## Manual checklist

* Python, JS, TS, Vue, JSON, Markdown each highlight; a Vue `<script>` block is
  coloured and its template is not mis-parsed because of it.
* Type anywhere in the buffer: nothing changes. Paste, drag-and-drop and IME
  input are inert too.
* Select lines and copy; `Ctrl+F` finds and highlights; the gutter folds.
* Scroll position and cursor line survive a tab switch and a light/dark flip.
* Ask for `mcp_connectors/console_mcp.py` by hand in the URL: refused by policy
  (ed-03's guard, re-checked from this pane).
* Network panel while browsing ten files: one language chunk per kind, never the
  whole set.
