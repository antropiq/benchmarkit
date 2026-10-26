<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import MonacoEditor from './MonacoEditor.vue'
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
const editorMode = computed(() => (chat.state.phase === 'idle' ? 'read_write' : 'read_only'))
const paneReadonly = computed(() => editorMode.value === 'read_only')

// The buffer the pane shows. The tree's selection and this are separate state:
// a click that is refused (binary, over the cap) selects a row and leaves the
// previous buffer on screen, which is what the user is still looking at.
const file = computed(() => state.open.find((f) => f.path === state.active) ?? null)

// One Monaco surface for the whole pane, not one per tab: `content`/`language`
// are the only props, and switching buffers goes through the editor's model —
// so the scroll position of a file is not what changes when a chip is clicked,
// and closing a tab costs nothing. The `v-if="file"` below unmounts (and so
// disposes) the surface once the last buffer is closed.

// Which chip is the selected one of the tablist (the roving tabindex and the
// pane's aria-labelledby both key on it).
const activeIndex = computed(() => state.open.findIndex((f) => f.path === state.active))

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
watch(
  () => state.active,
  (path) => {
    if (path) tree.reveal(path)
  },
)

// A click on the chip that is *already* active changes nothing in `state.active`,
// so the watcher above cannot carry it to the tree. Reveal here as well: the
// tree's selection follows the tab the user pointed at, even when that buffer
// was already on screen.
function selectTab(path) {
  files.focusFile(path)
  tree.reveal(path)
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
            @click="selectTab(item.path)"
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
            @click="files.closeFile(item.path)"
          >
            ×
          </button>
        </span>
      </div>
      <span
        class="editor-badge"
        :class="editorMode === 'read_write' ? 'badge green' : 'badge gray'"
        :title="
          editorMode === 'read_write'
            ? 'Editable — the WorkBench is idle. Unsaved edits are kept in the tab, not written to disk yet.'
            : 'Read-only — a WorkBench reply is running and may be editing these files.'
        "
      >
        {{ editorMode === 'read_write' ? 'read-write' : 'read-only' }}
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
          <!-- A partial is never presented as a whole: the server only sets
               `truncated` when the file grew past its cap mid-read, so this
               says which prefix is on screen rather than implying the file
               ends here. -->
          <p v-if="file.truncated" class="caption editor-raw-note">
            file grew while it was being read — showing its first
            {{ files.humanSize(file.truncatedAt) }}
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
              :content="file.text"
              :language="file.language"
              :readonly="paneReadonly"
              :aria-label="`Contents of ${file.path}`"
              @content-change="(text) => files.updateText(file.path, text)"
            />
          </div>
        </template>
      </div>
    </div>
  </section>
</template>
