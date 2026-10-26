<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import MonacoEditor from './MonacoEditor.vue'
import MonacoDiffViewer from './MonacoDiffViewer.vue'
import { useEditorFiles } from '../composables/useEditorFiles'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { useBenchmark } from '../composables/useBenchmark'
import { useChat } from '../composables/useChat'

// The centre pane: the open-file tab strip and the Monaco surface that shows
// the active buffer. The header chrome and the tab chips are the pane's own
// markup; everything below them belongs to Monaco, which owns the DOM inside
// its host and the line-number gutter that used to be a hand-rolled `<pre>`
// column here.
//
// The badge is the editor's mode, live: "read-write" while the WorkBench is
// idle (the pane accepts edits, kept in the buffer only — nothing reaches disk
// until a save route lands), "read-only" while a reply is running. A tab
// labelled "Editor" promises an editable surface, and the badge says when that
// promise is currently in force.
const files = useEditorFiles()
const tree = useWorkspaceTree()
const bench = useBenchmark()
const chat = useChat()

const state = files.state

// The pane's editor state: read_write while the WorkBench is idle, read-only
// whenever a reply is in flight (`phase` is 'waiting' / 'reasoning' /
// 'answering'). A stopped conversation sits back at 'idle', so its files are
// editable again — which is the point: the model is the only thing that can
// write these files, and it must not race a human typing in the same buffer.
//
// Known edge, deliberate: batch mode owns the conversation between two task
// turns with `useBatch.state.active` while `phase` drops back to 'idle' (see
// useBatch.js), so on that gap this rule makes files briefly editable. Kept as
// specified.
// The buffer the pane shows — a file or a diff, whichever tab is active. The
// tree's selection and this are separate state: a click that is refused (binary,
// over the cap) selects a row and leaves the previous buffer on screen, which is
// what the user is still looking at.
//
// Matched on `key`, not `path`: a file buffer and a diff of that same file are
// two tabs about one path (useEditorFiles), and the pane has to tell them apart.
const tab = computed(() => state.open.find((item) => item.key === state.active) ?? null)
const isDiff = computed(() => tab.value?.kind === 'diff')

// A diff tab is a reader whatever the chat is doing: its two sides come from git
// and the disk and there is no buffer to type into, so the editable promise the
// badge makes belongs to a file buffer only.
const editorMode = computed(() => {
  if (isDiff.value) return 'read_only'
  return chat.state.phase === 'idle' ? 'read_write' : 'read_only'
})
const paneReadonly = computed(() => editorMode.value === 'read_only')

const badgeTitle = computed(() => {
  if (isDiff.value) {
    return 'Read-only — this tab is a diff: the committed side and the one on disk.'
  }
  return editorMode.value === 'read_write'
    ? 'Editable — the WorkBench is idle. Unsaved edits are kept in the tab, not written to disk yet.'
    : 'Read-only — a WorkBench reply is running and may be editing these files.'
})

// The one line that says what the two columns mean, so a reader never has to
// guess which side is which — and never reads the left side as the truth.
const diffNote = computed(() => {
  if (!isDiff.value) return ''
  if (tab.value.status === 'deleted') {
    return 'deleted since the last commit — the left side is what the file was'
  }
  if (tab.value.renamedFrom) {
    return `left: ${tab.value.renamedFrom} as of the last commit — right: ${tab.value.path} as it is on disk`
  }
  return 'left: as of the last commit — right: as it is on disk'
})

// Two Monaco surfaces share this pane, never at once: the code editor (one
// surface, one model, swapped between file buffers) and the diff editor (two
// models, and it cannot use the code pane's single model). Each is mounted only
// for the tab kind that needs it, so the last tab closing disposes the surface
// it belongs to and neither pays for the other.

// Which chip is the selected one of the tablist (the roving tabindex and the
// pane's aria-labelledby both key on it).
const activeIndex = computed(() => state.open.findIndex((item) => item.key === state.active))

// A tree click opens the file it selected (append-or-focus, never refetch).
//
// Selections published *by* this pane (`source: 'tab'`, see useWorkspaceTree
// .reveal) are skipped: they are the echo of the active-buffer watcher below,
// and replaying one would be a redundant open/focus of the buffer that caused
// it.
watch(
  () => tree.state.selected,
  (selected) => {
    if (selected?.source === 'tab') return
    if (selected?.path) files.openFile(selected.path)
  },
  { immediate: true },
)

// The other direction of the same link: whichever way the active buffer changes
// — a chip click, an arrow key on the strip, the neighbour a closed tab leaves —
// the tree selects that file, opening the collapsed branches its row hides
// under (`reveal` fetches the listings on the way down, since the tree is lazy).
// No DOM focus moves here: the chip the user just clicked keeps it.
//
// `state.active` is a tab *key*, so it is translated to the file path first: a
// diff tab belongs to the same row in the tree as the file it diffs.
watch(
  () => state.active,
  (key) => {
    const record = key ? files.recordForKey(key) : null
    if (record) tree.reveal(record.path)
  },
)

// A click on the chip that is *already* active changes nothing in `state.active`,
// so the watcher above cannot carry it to the tree. Reveal here as well: the
// tree's selection follows the tab the user pointed at, even when that buffer
// was already on screen.
function selectTab(key) {
  files.focusTab(key)
  const record = files.recordForKey(key)
  if (record) tree.reveal(record.path)
}

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
  files.focusTab(state.open[target].key)
  await nextTick()
  tabs.value?.querySelector(`#editor-tab-${target}`)?.focus()
}

// Ctrl+W closes the active tab — but only while the Editor tab is the visible
// one, so the chord is never stolen from another screen. (A browser that
// consumes Ctrl+W itself cannot be overridden from the page; the chip's × is
// the path that always works.) A diff tab closes the same way: `state.active`
// is the tab key, and closeTab works for either kind.
function onKeydown(event) {
  if (bench.state.activeTab !== 'editor') return
  if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 'w') return
  if (!state.active) return
  event.preventDefault()
  files.closeTab(state.active)
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
        <!-- Keyed by `item.key`: a file buffer and a diff of it share a path, so
             keying on the path would make Vue treat the two chips as one. -->
        <span
          v-for="(item, index) in state.open"
          :key="item.key"
          class="editor-tab"
          :class="{ active: item.key === state.active }"
          :title="item.path"
          @mouseup.middle.prevent="files.closeTab(item.key)"
        >
          <button
            class="editor-tab-name"
            type="button"
            role="tab"
            :id="`editor-tab-${index}`"
            :aria-selected="item.key === state.active"
            aria-controls="editor-pane"
            :tabindex="item.key === state.active ? 0 : -1"
            @click="selectTab(item.key)"
          >
            <!-- The unsaved-changes marker: a dot before the name, so it stays
                 visible when a long name is ellipsised at the end. A filled dot
                 is the conventional dirty state; it carries no text, so it gets
                 a label for screen readers instead of an asterisk. -->
            <span
              v-if="item.dirty"
              class="editor-tab-dirty"
              aria-hidden="true"
            ></span>
            <span class="editor-tab-label">{{ item.name }}</span>
            <span v-if="item.dirty" class="editor-tab-dirty-sr">
              (unsaved changes)
            </span>
          </button>
          <button
            class="editor-tab-close"
            type="button"
            :aria-label="`Close ${item.name}`"
            title="Close"
            @click="files.closeTab(item.key)"
          >
            ×
          </button>
        </span>
      </div>
      <span
        class="editor-badge"
        :class="editorMode === 'read_write' ? 'badge green' : 'badge gray'"
        :title="badgeTitle"
      >
        {{ editorMode === 'read_write' ? 'read-write' : 'read-only' }}
      </span>
    </header>

    <div class="editor-body">
      <div v-if="!tab && !state.message" class="stream-empty">
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

        <template v-if="tab && isDiff">
          <!-- What the two columns are, in the pane's own words rather than
               left to be guessed: left is the file as of the last commit, right
               is the file on disk. A rename names the old path, because the
               left side is the blob under that name. -->
          <p class="caption editor-raw-note">
            {{ diffNote }}
          </p>
          <div
            class="editor-monaco"
            id="editor-pane"
            role="tabpanel"
            :aria-labelledby="`editor-tab-${activeIndex}`"
          >
            <MonacoDiffViewer
              :original="tab.original"
              :modified="tab.modified"
              :language="tab.language"
              :aria-label="`Diff of ${tab.path}`"
            />
          </div>
        </template>

        <template v-else-if="tab">
          <!-- A partial is never presented as a whole: the server only sets
               `truncated` when the file grew past its cap mid-read, so this
               says which prefix is on screen rather than implying the file
               ends here. -->
          <p v-if="tab.truncated" class="caption editor-raw-note">
            file grew while it was being read — showing its first
            {{ files.humanSize(tab.truncatedAt) }}
          </p>
          <!-- The tabpanel wrapper is app-owned: `aria-controls` on the chips
               points here, and Monaco's host inside it stays empty of app
               markup (it is also the scope of the Pico overrides). -->
          <div
            class="editor-monaco"
            id="editor-pane"
            role="tabpanel"
            :aria-labelledby="`editor-tab-${activeIndex}`"
          >
            <MonacoEditor
              :content="tab.text"
              :language="tab.language"
              :readonly="paneReadonly"
              :aria-label="`Contents of ${tab.path}`"
              @content-change="(text) => files.updateText(tab.path, text)"
            />
          </div>
        </template>
      </div>
    </div>
  </section>
</template>
