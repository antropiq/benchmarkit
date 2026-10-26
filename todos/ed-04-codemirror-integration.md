# Step 4 — CodeMirror and tabs (**read-only surface**, decision D4)

Replace the `<pre>` preview with a real CodeMirror 6 editor: language support,
tab strip, cursor position, search. There is **no save path**
this lot ships no write endpoint at all.


## Packages (installed)

| Package | Why |
|---|---|
| `codemirror` | the preset: `basicSetup` (line numbers, history, search, autocomplete, closeBrackets, default keymaps) |
| `@codemirror/state`, `@codemirror/view` | explicit `EditorView`/`EditorState` control, so nothing depends on the preset's re-exports |
| `@codemirror/lang-python` | Python |
| `@codemirror/lang-javascript` | JS **and** TS (`{ typescript: true }`) |
| `@codemirror/lang-vue` | SFC: HTML plus a nested JS/TS `<script>`; pulls only `lang-html`/`lang-javascript` |
| `@codemirror/lang-json`, `@codemirror/lang-markdown` | the two most common extras in this repo |


### Lazy language loading

One dynamic import per language, keyed by the `language`:

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
* The tab chip shows the file name, the path in `title`, and a `×`. There is no
  dirty dot and no Save control in this lot.
