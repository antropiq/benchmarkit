<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

// The diff surface for a "View diff…" tab: Monaco's own diff editor, showing
// the two sides the server sent (`core/editorfs/gitstatus.py`) — the file as of
// HEAD on the left, the file on disk on the right.
//
// It is a *second* Monaco surface, deliberately not the one in
// MonacoEditor.vue: a diff editor needs two models and cannot share the single
// model the code pane swaps between buffers. That is also why the shared pieces
// (workers, theme, font, language ids) moved to `lib/monaco-surface`.
//
// Nothing here decides what a diff *is* — no patch parsing, no local baseline.
// The two texts come from the server, and Monaco computes and renders the
// hunks (word-level inside a changed line, collapsible runs of unchanged code).
import * as monaco from 'monaco-editor'
import '../styles/monaco-pico.css'
import {
  ensureMonacoWorkers,
  monacoLanguageId,
  monoFontFamily,
  picoMonacoTheme,
} from '../lib/monaco-surface'

ensureMonacoWorkers()

const props = defineProps({
  original: { type: String, default: '' },
  modified: { type: String, default: '' },
  language: { type: String, default: 'plaintext' },
  ariaLabel: { type: String, default: 'File diff' },
})

const host = ref(null)

let diffEditor = null
let originalModel = null
let modifiedModel = null
let themeObserver = null

onMounted(() => {
  const language = monacoLanguageId(props.language)
  originalModel = monaco.editor.createModel(props.original, language)
  modifiedModel = monaco.editor.createModel(props.modified, language)
  // The two models are attached with `setModel` *after* construction, never as
  // a `model` option: the standalone diff editor forwards its own options into
  // each of the two editors it builds inside itself, and a code editor reads
  // `options.model` as a *single* text model. Handing it the {original,
  // modified} pair makes every inner editor treat it as a diff model and build
  // editors again — infinite recursion, i.e. a `RangeError: Maximum call stack
  // size exceeded` thrown out of this component's mount hook.
  diffEditor = monaco.editor.createDiffEditor(host.value, {
    automaticLayout: true,
    theme: picoMonacoTheme(),
    // A diff tab is a reader: neither side can be typed into, so there is no
    // half-applied "revert" to undo and no way to mistake a diff for an editor.
    readOnly: true,
    originalEditable: false,
    renderSideBySide: true,
    // A pane this narrow falls back to the inline (unified) view instead of
    // squeezing two columns, which is what makes the feature usable with the
    // tree and the terminal strip open.
    useInlineViewWhenSpaceIsLimited: true,
    renderSideBySideInlineBreakpoint: 620,
    compactMode: true,
    // Git's own default for a plain `git diff`: whitespace *is* a change.
    ignoreTrimWhitespace: false,
    renderIndicators: true,
    renderMarginRevertIcon: false,
    renderGutterMenu: false,
    renderOverviewRuler: false,
    hideUnchangedRegions: { enabled: true, contextLineCount: 3 },
    // The same viewer shape as the code pane: no minimap, no sticky scroll,
    // nothing that costs width this pane does not have.
    minimap: { enabled: false },
    stickyScroll: { enabled: false },
    overviewRulerLanes: 0,
    hideCursorInOverviewRuler: true,
    overviewRulerBorder: false,
    scrollBeyondLastLine: false,
    wordWrap: 'off',
    smoothScrolling: true,
    padding: { top: 8, bottom: 8 },
    fontFamily: monoFontFamily(),
    fontSize: 13,
    lineHeight: 20,
    renderWhitespace: 'none',
    ariaLabel: props.ariaLabel,
    originalAriaLabel: `${props.ariaLabel} — as of the last commit`,
    modifiedAriaLabel: `${props.ariaLabel} — as it is on disk`,
    scrollbar: {
      verticalScrollbarSize: 10,
      horizontalScrollbarSize: 10,
      useShadows: false,
    },
  })
  diffEditor.setModel({ original: originalModel, modified: modifiedModel })

  themeObserver = new MutationObserver(() => monaco.editor.setTheme(picoMonacoTheme()))
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
})

// Content and language changes go through the *models*, never by re-creating
// the editor: a new model pair would throw away the computed diff, the folded
// unchanged runs and the scroll position for what is usually one edited line.
watch(
  () => [props.original, props.modified],
  ([original, modified]) => {
    if (originalModel && original !== originalModel.getValue()) {
      originalModel.setValue(original)
    }
    if (modifiedModel && modified !== modifiedModel.getValue()) {
      modifiedModel.setValue(modified)
    }
  },
)

// The language of both sides is the file's own: the grammar is what colourises
// the changed lines, and both halves of one file's diff share it.
watch(
  () => props.language,
  (next) => {
    const language = monacoLanguageId(next)
    if (originalModel) monaco.editor.setModelLanguage(originalModel, language)
    if (modifiedModel) monaco.editor.setModelLanguage(modifiedModel, language)
  },
)

watch(
  () => props.ariaLabel,
  (next) => {
    diffEditor?.updateOptions({ ariaLabel: next })
  },
)

// Both halves again: the editor (its DOM, its two inner editors, the
// ResizeObserver behind `automaticLayout`) and the models (their worker-side
// mirrors and tokenization caches).
onBeforeUnmount(() => {
  themeObserver?.disconnect()
  themeObserver = null
  diffEditor?.dispose()
  diffEditor = null
  originalModel?.dispose()
  originalModel = null
  modifiedModel?.dispose()
  modifiedModel = null
})
</script>

<template>
  <!-- The same `monaco-host` scope MonacoEditor.vue uses: monaco-pico.css
       undoes Pico's bare-element rules inside it, and the diff editor builds
       two editors in there that need exactly that protection. -->
  <div ref="host" class="monaco-host monaco-diff-host"></div>
</template>

<style scoped>
/* Monaco fills its host and never sizes it, so the host takes the pane. */
.monaco-host {
  width: 100%;
  height: 100%;
  min-height: 0;
}
</style>
