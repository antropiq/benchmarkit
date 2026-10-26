<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

// The code pane's editor surface. Bundled with the Editor tab's lazy chunk
// (App.vue's defineAsyncComponent), never with the entry chunk — Monaco and
// its workers are heavy and the Benchmark tab is the default screen.
//
// Two things about this component are load-bearing and easy to break:
//
//   * Monaco owns the DOM inside `host`. It is created once and never
//     re-created on a prop change: content swaps go through the *model*
//     (`setValue`), the language through `setModelLanguage`. Re-creating the
//     editor per file would throw away scroll position, view state and the
//     worker session every click.
//   * Pico's bare-element rules must not reach Monaco's internals, which is
//     what `monaco-pico.css` scopes out. Its import rides this file so it
//     lands in the same chunk as the editor it protects.
import * as monaco from 'monaco-editor'
import '../styles/monaco-pico.css'

import EditorWorker from 'monaco-editor/editor/editor.worker.js?worker'
import JsonWorker from 'monaco-editor/language/json/json.worker.js?worker'
import CssWorker from 'monaco-editor/language/css/css.worker.js?worker'
import HtmlWorker from 'monaco-editor/language/html/html.worker.js?worker'
import TsWorker from 'monaco-editor/language/typescript/ts.worker.js?worker'

// Workers are configured once per page, before any editor exists: Monaco asks
// `MonacoEnvironment.getWorker(label)` at first use of each language service,
// and Vite's `?worker` suffix is what turns each worker source into a real
// same-origin module worker instead of a fetch of an unbundled file.
//
// The label is chosen by Monaco, not by us: `editorWorkerService` for the
// core worker, and otherwise the language id (`json`, `css`/`scss`/`less`,
// `html`/`handlebars`/`razor`, `typescript`/`javascript`). Anything unmapped
// falls back to the editor worker, which is the safe answer — a wrong worker
// fails the language service, a missing one fails the editor.
if (!self.MonacoEnvironment) {
  self.MonacoEnvironment = {
    getWorker(_workerId, label) {
      switch (label) {
        case 'json':
          return new JsonWorker()
        case 'css':
        case 'scss':
        case 'less':
          return new CssWorker()
        case 'html':
        case 'handlebars':
        case 'razor':
          return new HtmlWorker()
        case 'typescript':
        case 'javascript':
          return new TsWorker()
        default:
          return new EditorWorker()
      }
    },
  }
}

// The server sends the language id it chose for the file
// (`core/editorfs/files.py`), and an unknown extension is honestly
// `plaintext`. Monaco has no grammar for two of the ids it can receive, so
// they are mapped to what actually helps: a `.vue` file tokenizes as its
// template (`html`), and `toml` has no grammar in the bundle at all, so it
// stays plain rather than pretending to be `ini`.
const LANGUAGE_BY_SERVER_ID = {
  python: 'python',
  javascript: 'javascript',
  typescript: 'typescript',
  vue: 'html',
  json: 'json',
  markdown: 'markdown',
  html: 'html',
  css: 'css',
  scss: 'scss',
  yaml: 'yaml',
  shell: 'shell',
}

function monacoLanguageId(id) {
  return LANGUAGE_BY_SERVER_ID[id] ?? 'plaintext'
}

// Pico's theme is the single source of truth for light/dark: it is the
// `data-theme` attribute on `<html>`, set before first paint by index.html and
// flipped by useTheme. Monaco's two built-in themes are named `vs` (light) and
// `vs-dark`, so the mapping is one line — but it must be *read* at mount (the
// editor can be created long after boot) and re-read on every change.
function picoTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light'
    ? 'vs'
    : 'vs-dark'
}

const props = defineProps({
  content: { type: String, default: '' },
  language: { type: String, default: 'plaintext' },
  // Read-only by design for now: the file surface has no write route (AGENTS.md),
  // so the editor is a viewer, not an input. Kept as a prop so the write layer
  // turns it off rather than restructuring this component.
  readonly: { type: Boolean, default: true },
  ariaLabel: { type: String, default: 'File contents' },
})

const host = ref(null)

let editor = null
let model = null
let themeObserver = null

// The app's mono stack comes from the theme token rather than a literal here,
// so the editor cannot drift from the rest of the UI's monospace text.
function monoFontFamily() {
  const fromToken = getComputedStyle(document.documentElement)
    .getPropertyValue('--pico-font-family-monospace')
    .trim()
  return fromToken || "ui-monospace, SFMono-Regular, monospace"
}

onMounted(() => {
  model = monaco.editor.createModel(
    props.content,
    monacoLanguageId(props.language),
  )
  editor = monaco.editor.create(host.value, {
    model,
    automaticLayout: true,
    theme: picoTheme(),
    readOnly: props.readonly,
    domReadOnly: props.readonly,
    ariaLabel: props.ariaLabel,
    // The viewer shape: no minimap, no sticky scroll, a ruler-less overview —
    // none of it is decoration, each of them costs width this pane does not
    // have next to the tree and the terminal strip.
    minimap: { enabled: false },
    stickyScroll: { enabled: false },
    overviewRulerLanes: 0,
    hideCursorInOverviewRuler: true,
    overviewRulerBorder: false,
    scrollBeyondLastLine: false,
    renderLineHighlight: 'line',
    glyphMargin: false,
    folding: true,
    wordWrap: 'off',
    smoothScrolling: true,
    padding: { top: 8, bottom: 8 },
    fontFamily: monoFontFamily(),
    fontSize: 13,
    lineHeight: 20,
    renderWhitespace: 'none',
    scrollbar: {
      verticalScrollbarSize: 10,
      horizontalScrollbarSize: 10,
      useShadows: false,
    },
  })

  // Theme changes are an attribute mutation on `<html>`, not a store we can
  // import (useTheme writes the attribute directly and index.html sets it
  // before any module runs), so a MutationObserver is the honest listener.
  themeObserver = new MutationObserver(() => monaco.editor.setTheme(picoTheme()))
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
})

// Content: go through the model, never `create` again. The guard matters:
// `setValue` clears the undo stack and the cursor/selection, so it must not
// run when the text already matches (an unrelated re-render, a reload that
// changed nothing).
watch(
  () => props.content,
  (next) => {
    if (!model || next === model.getValue()) return
    model.setValue(next)
  },
)

watch(
  () => props.language,
  (next) => {
    if (model) monaco.editor.setModelLanguage(model, monacoLanguageId(next))
  },
)

watch(
  () => props.readonly,
  (next) => {
    if (!editor) return
    editor.updateOptions({ readOnly: next, domReadOnly: next })
  },
)

// `ariaLabel` is a construction-time option, so switching the active buffer
// (same editor, new file) would otherwise leave the screen-reader label stuck
// on the first file opened. It rides the same `updateOptions` path.
watch(
  () => props.ariaLabel,
  (next) => {
    if (editor) editor.updateOptions({ ariaLabel: next })
  },
)

// Both halves get disposed: the editor (its DOM, listeners, and the
// ResizeObserver behind `automaticLayout`) and the model (its worker-side
// mirror and tokenization caches). Disposing only the editor leaks the model.
onBeforeUnmount(() => {
  themeObserver?.disconnect()
  themeObserver = null
  editor?.dispose()
  editor = null
  model?.dispose()
  model = null
})
</script>

<template>
  <!-- `monaco-host` is the scope of the Pico overrides in monaco-pico.css and
       the element Monaco builds inside: it must stay empty of app markup. -->
  <div ref="host" class="monaco-host"></div>
</template>

<style scoped>
/* Monaco fills its host and never sizes it, so the host takes the pane and
   Monaco's own `automaticLayout` observes this box. */
.monaco-host {
  width: 100%;
  height: 100%;
  min-height: 0;
}
</style>
