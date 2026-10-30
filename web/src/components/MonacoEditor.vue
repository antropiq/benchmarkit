<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

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
import {
  clampReveal,
  ensureMonacoWorkers,
  monacoLanguageId,
  monoFontFamily,
  picoMonacoTheme,
} from '../lib/monaco-surface'

// The worker bootstrap, the theme mapping, the mono font and the server's
// language ids all live in `lib/monaco-surface`, because the diff viewer is a
// second Monaco surface with exactly the same needs — one source for both.
ensureMonacoWorkers()

const props = defineProps({
  content: { type: String, default: '' },
  language: { type: String, default: 'plaintext' },
  // Read-only by default: the safe shape is the one this component starts in,
  // and the *pane* decides when it changes (useEditorMode's rule — editable
  // while the WorkBench is idle). `readOnly`+`domReadOnly` gate the editing, and
  // the watcher below flips both hot when the mode changes. Saving is wired
  // either way (Ctrl+S and the header's buttons are window/pane level), so a
  // read-only pane can still be a pane with unsaved work in it.
  readonly: { type: Boolean, default: true },
  ariaLabel: { type: String, default: 'File contents' },
})

// Emits the new text whenever the user edits the buffer. Only fires for real
// edits (the model listener is guarded against the programmatic `setValue`
// below), and only matters while not `readonly` — a read-only surface cannot
// type. The pane routes this into the buffer record and the dirty flag.
const emit = defineEmits(['content-change'])

const host = ref(null)

let editor = null
let model = null
let themeObserver = null
// Set while this component drives the model itself (`setValue` on a buffer
// swap or a reload) so the content listener stays silent for those, and only
// real user edits reach `content-change`.
let applyingExternal = false

// --- The reveal: "put the caret on this search hit" -----------------------
//
// The pane owns *which* hit and when to ask; this surface owns how, and the
// queue lives here because only this component knows when its model swap has
// finished (search-08). A reveal cannot be applied at the moment of the row
// click, for two independent reasons:
//
//   * the file may not be open yet, so there is no `editor` to move — the
//     reveal waits for `onMounted` (drain at its end);
//   * when the file *is* already open, the pane keeps this one instance across
//     tab switches and the `props.content` watcher below calls `model.setValue`
//     — which clears the cursor. So the reveal drains at the end of that
//     watcher too, after the swap, which is what lets a reveal queued for a
//     buffer that is still being fetched land once its text arrives.
//
// `revealPosition` also drains on the next tick, covering the case where
// neither of those is scheduled: the right text is already on screen, so the
// watcher never runs and the mount long happened.
const pendingReveal = ref(null) // { line, column }

function flushPendingReveal() {
  if (!pendingReveal.value || !editor || !model) return
  const position = clampReveal(
    pendingReveal.value,
    model.getLineCount(),
    (line) => model.getLineMaxColumn(line),
  )
  pendingReveal.value = null
  if (!position) return
  // This order: `setPosition` on an off-screen line does not scroll.
  editor.revealLineInCenter(position.line)
  editor.setPosition({ lineNumber: position.line, column: position.column })
}

function revealPosition({ line, column } = {}) {
  if (!Number.isInteger(line) || !Number.isInteger(column) || line < 1 || column < 1) {
    return // a reveal that cannot be applied is a no-op, never an error
  }
  pendingReveal.value = { line, column }
  void nextTick(flushPendingReveal)
}

// The only imperative surface this component has.
defineExpose({ revealPosition })

onMounted(() => {
  model = monaco.editor.createModel(
    props.content,
    monacoLanguageId(props.language),
  )
  // One listener for the whole component: the model outlives nothing here, but
  // the guard keeps the programmatic `setValue` (a reload, a buffer swap) from
  // echoing back as an edit. Real edits are exactly the `content`-prop changes
  // we did *not* cause — so each one is surfaced to the pane once, with the
  // model's text. (The loop still closes: the pane stores what we emit, the
  // watcher below sees it already equals `model.getValue()` and skips.)
  model.onDidChangeContent(() => {
    if (applyingExternal) return
    emit('content-change', model.getValue())
  })
  editor = monaco.editor.create(host.value, {
    model,
    automaticLayout: true,
    theme: picoMonacoTheme(),
    readOnly: props.readonly,
    domReadOnly: props.readonly,
    ariaLabel: props.ariaLabel,
    // The viewer shape: no sticky scroll, a ruler-less overview — none of it is
    // decoration, each of them costs width this pane does not have next to the
    // tree and the terminal strip. The minimap is the exception worth paying
    // that width for: it is how you scrub a long file, so it never autohides
    // (an autohidden minimap is a slider you cannot grab) and its slider is
    // shown always rather than only on hover. It draws colour blocks instead
    // of glyphs — unreadable at this scale, and they cost render time.
    // `maxColumn` is the width budget for this narrow pane.
    minimap: {
      enabled: true,
      autohide: 'none',
      showSlider: 'always',
      renderCharacters: false,
      maxColumn: 80,
    },
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
  themeObserver = new MutationObserver(() => monaco.editor.setTheme(picoMonacoTheme()))
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
  flushPendingReveal()
})

// Content: go through the model, never `create` again. The guard matters:
// `setValue` clears the undo stack and the cursor/selection, so it must not
// run when the text already matches (an unrelated re-render, a reload that
// changed nothing).
//
// Line endings need no handling here beyond letting the model do it: `setValue`
// builds the model's EOL from the text it is given (the majority of `\r\n` over
// `\n`), so a CRLF file keeps CRLF, the caret's Enter inserts CRLF, and every
// `getValue()` the pane saves back is written with that EOL. The server never
// normalises (D5) — what the model says is what lands. The one thing that does
// not survive is a file with *mixed* endings, which the model resolves to its
// dominant form on load: the same normalising any editor with a single EOL per
// buffer does, and a reason the save goes through the mtime guard rather than
// trusting the buffer to be byte-identical.
watch(
  () => props.content,
  (next) => {
    if (!model || next === model.getValue()) return
    applyingExternal = true
    try {
      model.setValue(next)
    } finally {
      applyingExternal = false
    }
    // `setValue` cleared the cursor, so a reveal queued for this buffer (a hit
    // clicked while its file was still being fetched) is applied here — the
    // first moment the new text is in the model. Draining anywhere earlier is
    // the synchronous-reveal bug the queue exists to avoid.
    flushPendingReveal()
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
