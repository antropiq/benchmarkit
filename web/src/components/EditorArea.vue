<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useEditorFiles } from '../composables/useEditorFiles'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { useBenchmark } from '../composables/useBenchmark'

// The centre pane as ed-03 leaves it: the open-file tab strip, and the file's
// text as a raw `<pre>` with a line-number gutter. ed-04 replaces the body
// with CodeMirror; the header chrome and the tab chips below are what it keeps.
//
// The badge is deliberate, not decoration: a tab labelled "Editor" promises an
// editable surface, and nothing in this lot writes to disk — there is no write
// endpoint at all. It disappears when the write layer lands.
const files = useEditorFiles()
const tree = useWorkspaceTree()
const bench = useBenchmark()

const state = files.state

// The buffer the pane shows. The tree's selection and this are separate state:
// a click that is refused (binary, over the cap) selects a row and leaves the
// previous buffer on screen, which is what the user is still looking at.
const file = computed(() => state.open.find((f) => f.path === state.active) ?? null)

// One gutter line per line of text is a v-for of the file's size; past a few
// thousand rows that costs the tab its responsiveness, so the number column
// stops and the pane says why. The text itself stays: it is one text node.
const lineCount = computed(() => (file.value?.text ?? '').split('\n').length)
const gutters = computed(
  () => !!file.value && !file.value.binary && lineCount.value <= files.MAX_GUTTER_LINES,
)

// Which chip is the selected one of the tablist (the roving tabindex and the
// pane's aria-labelledby both key on it).
const activeIndex = computed(() => state.open.findIndex((f) => f.path === state.active))

// A tree click opens the file it selected (append-or-focus, never refetch).
watch(
  () => tree.state.selected,
  (selected) => {
    if (selected?.path) files.openFile(selected.path)
  },
  { immediate: true },
)

// The strip's keyboard model — a roving tabindex (only the selected chip is in
// the tab order) plus ← → / Home / End, the same contract the tree and
// TabNav.vue keep. Focus follows the selection here, as in the navigator.
const tabs = ref(null)
async function onTabsKeydown(event) {
  const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key]
  let target = null
  if (step !== undefined) target = activeIndex.value + step
  else if (event.key === 'Home') target = 0
  else if (event.key === 'End') target = state.open.length - 1
  else return
  const total = state.open.length
  if (!total) return
  target = (target + total) % total
  event.preventDefault()
  files.focusFile(state.open[target].path)
  await nextTick()
  tabs.value?.querySelector(`#editor-tab-${target}`)?.focus()
}

// Ctrl+W closes the active tab — but only while the Editor tab is the visible
// one, so the chord is never stolen from another screen. (A browser that
// consumes Ctrl+W itself cannot be overridden from the page; the chip's × is
// the path that always works.)
function onKeydown(event) {
  if (bench.state.activeTab !== 'editor') return
  if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 'w') return
  if (!state.active) return
  event.preventDefault()
  files.closeFile(state.active)
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <section class="editor-area" aria-label="File editor">
    <header class="editor-area-head">
      <div
        ref="tabs"
        class="editor-tabs"
        role="tablist"
        aria-label="Open files"
        @keydown="onTabsKeydown"
      >
        <!-- A chip is a span, not a button: the × is a button of its own, and
             nesting one button in another is invalid markup. Middle-click
             closes too — nothing is editable, so there is nothing to confirm. -->
        <span
          v-for="(item, index) in state.open"
          :key="item.path"
          class="editor-tab"
          :class="{ active: item.path === state.active }"
          :title="item.path"
          @mouseup.middle.prevent="files.closeFile(item.path)"
        >
          <button
            class="editor-tab-name"
            type="button"
            role="tab"
            :id="`editor-tab-${index}`"
            :aria-selected="item.path === state.active"
            aria-controls="editor-pane"
            :tabindex="item.path === state.active ? 0 : -1"
            @click="files.focusFile(item.path)"
          >
            {{ item.name }}
          </button>
          <button
            class="editor-tab-close"
            type="button"
            :aria-label="`Close ${item.name}`"
            title="Close (nothing to save)"
            @click="files.closeFile(item.path)"
          >
            ×
          </button>
        </span>
      </div>
      <span class="editor-badge badge gray" title="Files are opened read-only in this version">
        read-only
      </span>
    </header>

    <div class="editor-body">
      <div v-if="!file && !state.message" class="stream-empty">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <path d="M14 2v6h6" />
        </svg>
        <p class="stream-empty-title">{{ state.loading ? 'Loading…' : 'Open a file' }}</p>
        <p class="caption">
          {{
            state.loading
              ? 'Reading it from the working location.'
              : 'Pick one in the tree on the left — its contents appear here.'
          }}
        </p>
      </div>

      <div v-else class="editor-body-col">
        <!-- The path is echoed with the reason, so a policy refusal reads as
             "this file is protected", not as a broken tab. -->
        <p
          v-if="state.message"
          class="editor-raw-message"
          :class="state.message.tone === 'error' ? 'alert-error' : 'caption'"
          role="alert"
        >
          {{ state.message.text }}
          <span class="editor-raw-message-path">{{ state.message.path }}</span>
          <button
            class="editor-raw-message-dismiss"
            type="button"
            aria-label="Dismiss this message"
            title="Dismiss"
            @click="files.clearMessage()"
          >
            ×
          </button>
        </p>

        <template v-if="file">
          <p v-if="!gutters" class="caption editor-raw-note">
            large file — {{ lineCount.toLocaleString() }} lines; no line numbers, syntax
            view coming with the editor
          </p>
          <div class="editor-raw-scroll">
            <div v-if="gutters" class="editor-gutter" aria-hidden="true">
              <span v-for="line in lineCount" :key="line">{{ line }}</span>
            </div>
            <!-- The exact characters the server sent: CRLF stays CRLF, because
                 CodeMirror (ed-04) keeps its own lineSeparator and re-writes
                 what it received. `data-language` is what ed-04 keys its lazy
                 grammar import on — the id the server chose, not a guess. -->
            <pre
              class="editor-raw"
              id="editor-pane"
              role="tabpanel"
              :aria-labelledby="`editor-tab-${activeIndex}`"
              :data-language="file.language"
              tabindex="0"
            >{{ file.text }}</pre>
          </div>
        </template>
      </div>
    </div>
  </section>
</template>
