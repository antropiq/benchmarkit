<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import MonacoEditor from './MonacoEditor.vue'
import MonacoDiffViewer from './MonacoDiffViewer.vue'
import MarkdownStream from './MarkdownStream.vue'
import EditorTreeDialogs from './EditorTreeDialogs.vue'
import { useEditorFiles } from '../composables/useEditorFiles'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { useBenchmark } from '../composables/useBenchmark'
import { useChat } from '../composables/useChat'
import { useEditorMode } from '../composables/useEditorMode'
import { scrollStripWithWheel } from '../lib/tab-strip-wheel'
import { applyEditorChord } from '../lib/editor-keys'

// The centre pane: the open-file tab strip and the surface that shows the
// active tab — a Monaco buffer for a file, the diff editor for a diff, or the
// rendered-markdown panel for a preview. The header chrome and the tab chips
// are the pane's own markup; inside `.monaco-host` (the two Monaco surfaces)
// everything belongs to Monaco, which owns the DOM there and the line-number
// gutter that used to be a hand-rolled `<pre>` column here.
//
// The pane's mode — editable while the WorkBench is idle, read-only while a
// reply streams — is decided once in useEditorMode and *displayed* in the file
// tree's header, where it always fits; this component only applies it to the
// surface below.
const files = useEditorFiles()
const tree = useWorkspaceTree()
const bench = useBenchmark()
const chat = useChat()
const { paneReadonly } = useEditorMode()

// The freshness tick belongs to the pane that shows its answer, and it runs
// only while that answer can be read: the Editor tab visible and something open
// (the watcher below). `probeIsUseful` re-checks both — plus the streaming rule
// — on every tick, so a poll can never slip through a stale timer.
function probeShouldRun() {
  // A streaming reply is a known writer, the pane is read-only meanwhile, and
  // a conflict the user cannot act on is noise — so the tick stops with it.
  return (
    bench.state.activeTab === 'editor' &&
    state.open.length > 0 &&
    chat.state.phase === 'idle'
  )
}

function syncProbe() {
  if (probeShouldRun()) files.startFreshnessProbe()
  else files.stopFreshnessProbe()
}

const state = files.state
// The buffer the pane shows — a file or a diff, whichever tab is active. The
// tree's selection and this are separate state: a click that is refused (binary,
// over the cap) selects a row and leaves the previous buffer on screen, which is
// what the user is still looking at.
//
// Matched on `key`, not `path`: a file buffer and a diff of that same file are
// two tabs about one path (useEditorFiles), and the pane has to tell them apart.
const tab = computed(() => state.open.find((item) => item.key === state.active) ?? null)
const isDiff = computed(() => tab.value?.kind === 'diff')
const isPreview = computed(() => tab.value?.kind === 'preview')

// The one line that says what the two columns mean, so a reader never has to
// guess which side is which — and never reads the left side as the truth.
const diffNote = computed(() => {
  if (!isDiff.value) return ''
  // The conflict comparison is the *opposite* pairing from a git diff: there
  // the right side is the disk, here the left side is the disk and the right
  // side is the unsaved buffer. Saying so is the whole point — a reader who
  // assumed the git polarity would keep the wrong side.
  if (tab.value.source === 'conflict') {
    return 'left: as it is on disk now — right: your unsaved buffer'
  }
  if (tab.value.status === 'deleted') {
    return 'deleted since the last commit — the left side is what the file was'
  }
  if (tab.value.renamedFrom) {
    return `left: ${tab.value.renamedFrom} as of the last commit — right: ${tab.value.path} as it is on disk`
  }
  return 'left: as of the last commit — right: as it is on disk'
})

// A buffer whose file is gone from disk says so in the pane, not only on its
// chip: what is on screen is all that is left of it, and presenting this
// surface as a view of a live file would be a lie.
const deletedNote = computed(() =>
  tab.value?.kind === 'file' && tab.value.deleted
    ? 'this file no longer exists on disk — what is here is the buffer only'
    : '',
)

// Three surfaces share this pane, never at once: the code editor (one surface,
// one model, swapped between file buffers), the diff editor (two models, and it
// cannot use the code pane's single model) and the rendered markdown preview —
// an app-owned sibling, because the bundled `monaco-editor` ESM build ships no
// markdown preview widget and `.monaco-host` must stay empty of app markup (it
// is the scope of the `monaco-pico.css` overrides and of e2e selectors). Each is
// mounted only for the tab kind that needs it, so the last tab closing disposes
// the surface it belongs to and none of them pays for the others.
//
// Known and accepted for v1: a preview shows no relative *image* (the app has no
// byte-serving route, so `<img src="diagram.png">` resolves against the app
// origin and renders broken), and a relative link navigates the SPA away.
// See the note in lib/markdown-preview.js.

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

// Everything that closes a tab — the ×, the middle-click, Ctrl+W — asks
// `requestCloseTab`, which is the one place that knows a dirty buffer cannot be
// closed by a single gesture.
function closeTabRequested(key) {
  files.requestCloseTab(key)
}

// The pane's save affordances, both outside the strip's scroller so they survive
// a strip full of chips. It is about the buffer on screen — `tab`, the same
// record the pane renders — so it goes enabled only for a dirty file buffer: a
// clean one has nothing to save, and a diff has no buffer at all.
const canSaveActive = computed(
  () => tab.value?.kind === 'file' && tab.value.dirty,
)
const dirtyCount = computed(() => files.dirtyCount.value)
// Names the tab's own kind, so the tooltip does not claim "diff" over a preview
// (the button itself is already disabled for every non-buffer kind).
const saveButtonTitle = computed(() => {
  const record = tab.value
  if (!record) return 'Nothing to save'
  if (record.kind !== 'file') {
    return `A ${record.kind} tab has no buffer to save`
  }
  if (record.saving) return 'Saving…'
  return record.dirty
    ? `Save ${record.path} (Ctrl+S)`
    : `${record.name} has no unsaved changes`
})

function saveActive() {
  if (state.active) void files.saveFile(state.active)
}

function saveAll() {
  void files.saveAllDirty()
}

// The strip's keyboard model — a roving tabindex (only the selected chip is in
// the tab order) plus ← → / Home / End, the same contract the tree and
// TabNav.vue keep. Focus follows the selection here, as in the navigator.
const tabs = ref(null)

// The wheel scrolls the strip sideways once its tabs overflow it (VS Code's
// behaviour); the rule itself lives in lib/tab-strip-wheel.js so it can be
// tested without a browser. It is bound by hand rather than `@wheel`, because
// the handler needs `passive: false` to cancel the event.
function onTabsWheel(event) {
  scrollStripWithWheel(tabs.value, event)
}

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

// The window-level chords: Ctrl+S saves the active buffer, Ctrl+Shift+S saves
// every dirty one, Ctrl+W closes the active tab (asking first when it is dirty).
// Which key means what — and when the pane may consume it at all — is decided in
// lib/editor-keys; this is the listener and its three effects.
//
// One owner for all three, deliberately: a Monaco keybinding is the other way to
// bind Ctrl+S, but the editor is read-only while a reply streams, and a chord
// that stops working depending on the pane's mode is worse than one more
// listener. (A browser that consumes Ctrl+W itself cannot be overridden from the
// page; the chip's × is the path that always works.)
function onKeydown(event) {
  applyEditorChord(event, { activeTab: bench.state.activeTab, active: state.active }, {
    save: saveActive,
    'save-all': saveAll,
    close: () => files.requestCloseTab(state.active),
  })
}

// Opening the first buffer or coming back to the tab starts the tick; emptying
// the strip or leaving it stops it. A background poll of a pane nobody is
// looking at is a request cost with no reader.
watch(
  () => [bench.state.activeTab, state.open.length, chat.state.phase],
  syncProbe,
  { immediate: true },
)

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  // `passive: false` so the handler above can call preventDefault (a passive
  // listener cannot) — the tablist element exists from the first render, so
  // this attaches once.
  tabs.value?.addEventListener('wheel', onTabsWheel, { passive: false })
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  tabs.value?.removeEventListener('wheel', onTabsWheel)
  files.stopFreshnessProbe()
})
</script>

<template>
  <section class="editor-area" aria-label="File editor">
    <header class="editor-area-head">
      <div
        ref="tabs"
        class="editor-tabs"
        role="tablist"
        aria-label="Open files"
        :title="dirtyCount ? `${dirtyCount} unsaved` : 'Open files'"
        @keydown="onTabsKeydown"
      >
        <!-- A chip is a span, not a button: the × is a button of its own, and
             nesting one button in another is invalid markup. Middle-click asks
             to close too; a dirty buffer answers with a prompt rather than
             silently dropping its edits. -->
        <!-- Keyed by `item.key`: a file buffer and a diff of it share a path, so
             keying on the path would make Vue treat the two chips as one. -->
        <span
          v-for="(item, index) in state.open"
          :key="item.key"
          class="editor-tab"
          :class="{ active: item.key === state.active, 'editor-tab-saving': item.saving }"
          :title="item.path"
          @mouseup.middle.prevent="closeTabRequested(item.key)"
        >
          <!-- The freshness marker: the disk disagrees with this buffer. It is
               a *button* of its own (nesting one in the tab would be invalid
               markup, and the × already sets that shape) and it sits first, so
               an ellipsised name cannot push it away. A conflict the user turned
               down is still unanswered, so this is how the answer set is reached
               again — the conflict itself lives on the record and is never lost,
               only un-shown. -->
          <button
            v-if="item.stale || item.deleted"
            class="editor-tab-stale"
            :class="{ 'editor-tab-stale--deleted': item.deleted }"
            type="button"
            :title="
              item.deleted
                ? 'Deleted on disk while it is open here — click to choose what to do'
                : 'Changed on disk while it is open here — click to choose what to do'
            "
            @click="selectTab(item.key); files.raiseConflict(item.key)"
          ></button>
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
            <!-- Same visually-hidden pattern for the freshness marker, whose
                 button carries no text of its own. -->
            <span v-if="item.deleted" class="editor-tab-stale-sr">(deleted on disk)</span>
            <span v-else-if="item.stale" class="editor-tab-stale-sr">(changed on disk)</span>
          </button>
          <button
            class="editor-tab-close"
            type="button"
            :aria-label="`Close ${item.name}`"
            title="Close"
            @click="closeTabRequested(item.key)"
          >
            ×
          </button>
        </span>
      </div>

      <!-- The save affordances, beside the scroller rather than inside it: a
           strip full of chips scrolls under its own rule, and a button that
           scrolls away with it is a button the user cannot find. Both are
           icon+title+aria-label, the same shape as a chip's ×. Save-all appears
           only when there is more than one dirty buffer — with one, the plain
           Save button *is* save-all. -->
      <div class="editor-save-actions">
        <button
          class="editor-save-button"
          type="button"
          :disabled="!canSaveActive"
          :aria-label="canSaveActive ? 'Save this file' : 'Nothing to save'"
          :title="saveButtonTitle"
          @click="saveActive"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
            <path d="M17 21v-8H7v8" />
            <path d="M7 3v5h8" />
          </svg>
        </button>
        <button
          v-if="dirtyCount > 1"
          class="editor-save-button editor-save-all"
          type="button"
          aria-label="Save every unsaved file"
          :title="`Save all — ${dirtyCount} unsaved (Ctrl+Shift+S)`"
          @click="saveAll"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
            <path d="M8 13l2.5 2.5L16 10" />
          </svg>
        </button>
      </div>
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
        <!-- A probe that could not answer is not evidence that a file changed,
             so no buffer is marked — but the pane should not imply its buffers
             are fresh either. A plain caption, not an alert and not dismissible:
             it is a state of the pane (probe broken), and it clears by itself
             as soon as a probe answers again. -->
        <p v-if="state.probeError" class="caption editor-probe-error" role="status">
          {{ state.probeError }}
        </p>

        <!-- A refused or confirmed *write* does not appear here: it goes to the
             notice stack (EditorNoticeStack.vue, mounted by EditorView), because
             it must survive the next click and must be readable while the user
             is looking at the tree. This line is only ever about a refused
             *open* — one sentence, cleared by the next focus or close. -->
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

        <template v-else-if="isPreview">
          <!-- The preview is a snapshot (useEditorFiles.openPreview), not a live
               binding to the buffer or the disk — this line is the honesty half
               of that bargain, and "Preview again to refresh" is the whole
               refresh affordance. -->
          <p class="caption editor-raw-note">
            rendered from the copy read when this tab opened — Preview again to refresh
          </p>
          <!-- A truncated `.md` renders a *fragment* (an unclosed fence swallows
               the rest of the file as one code block), so this caption is the
               only thing that says so. -->
          <p v-if="tab.truncated" class="caption editor-raw-note">
            file grew while it was being read — showing its first
            {{ files.humanSize(tab.truncatedAt) }}
          </p>
          <!-- App-owned, and it carries `id="editor-pane"` like the other two
               panels: exactly one element holds that id at a time, which is what
               makes the chips' `aria-controls` honest. It is *not* `.editor-monaco`
               — that is `overflow: hidden` because Monaco owns its own scroller,
               and reusing it would leave a rendered document that cannot scroll. -->
          <div
            class="editor-preview"
            id="editor-pane"
            role="tabpanel"
            :aria-labelledby="`editor-tab-${activeIndex}`"
          >
            <MarkdownStream :html="tab.html" mermaid />
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
          <p v-else-if="deletedNote" class="caption editor-raw-note">{{ deletedNote }}</p>
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

    <!-- The pane's dialogs. Mounted here rather than in EditorView so the
         conflict is scoped to the code pane it is about; the file stays one
         component for all of the Editor's confirmations, whatever the later
         write tasks add to it. -->
    <EditorTreeDialogs />
  </section>
</template>
