<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import BatchProgressBar from './BatchProgressBar.vue'
import AgentModesDialog from './AgentModesDialog.vue'
import MemoryBullets from './MemoryBullets.vue'
import RegistrationMemoryDlg from './RegistrationMemoryDlg.vue'
import {
  conversationJson,
  conversationMarkdown,
  humanFileSize,
  parseSessionFile,
  useChat,
} from '../composables/useChat'
import { useBenchmark } from '../composables/useBenchmark'
import { useBatch } from '../composables/useBatch'
import {
  findingsLine,
  modeHint,
  modeOptionDisabled,
  modeSelectTitle,
  modeSelectDisabled,
  setAgentMode,
  useAgentModes,
} from '../composables/useAgentModes'

const {
  state,
  send,
  stop,
  pause,
  reset,
  setDraft,
  registerDraftInput,
  attachImage,
  loadSession,
} = useChat()
const { state: benchmarkState } = useBenchmark()
const {
  state: batch,
  stopBatch,
  clearBatchRun,
  openDialog: openBatchDialog,
} = useBatch()
const agentModes = useAgentModes().state

// Let draft insertions (memory bullets) focus the textarea after appending.
const draftEl = ref(null)
onMounted(() => registerDraftInput(draftEl.value))

// Registration dialog for the clicked bullet slot (null index = closed).
const dialogOpen = ref(false)
const dialogIndex = ref(null)

function openBulletDialog(index) {
  // Unreachable once the bullets are inert, but the guard keeps a keyboard or
  // synthetic click from opening an editor for a draft nobody can send.
  if (batch.active) return
  dialogIndex.value = index
  dialogOpen.value = true
}

const streaming = computed(() => state.phase !== 'idle')
// A batch owns every surface but Stop and Restart: `batch.active` is true for
// the whole lot, including the momentary idle between two task turns, which is
// why the composer cannot key on `streaming` alone here.
const locked = computed(() => streaming.value || batch.active)
const canSend = computed(
  () =>
    !locked.value &&
    (state.draft.raw.trim().length > 0 || state.pendingImage !== null),
)

function submit() {
  if (!canSend.value) return
  send()
}

function onKeydown(event) {
  // Enter sends; Shift+Enter inserts a carriage return.
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    submit()
  }
}

// Restart stays live during a batch and means Stop + wipe (B15): the only way
// back to a launchable state after a stopped batch. clearBatchRun first, so the
// loop can never be left running on a conversation that no longer exists.
function restart() {
  clearBatchRun()
  stop()
  reset()
}

const canRestart = computed(
  () => state.messages.length > 0 || state.phase !== 'idle' || batch.active,
)

// --- Agent mode (the composer's dropdown; every rule lives in useAgentModes) ---

// The select's title, and the quiet findings line, come from the composable —
// this component only renders them. Switching mid-`streaming` is allowed and
// has no effect on the turn already in flight (that request carried the mode
// chosen when it was sent); the `:disabled` rule the composable owns is the
// only refusal, and it is D7's, not a streaming guard.
const modeTitle = computed(() => modeSelectTitle({ batchActive: batch.active }))
const modeFindingsLine = computed(() => findingsLine())

function onModeChange(event) {
  setAgentMode(event.target.value)
}

// --- The modes editor (the pencil beside the dropdown; eds-07) ---------------

// One button, one card: the pencil opens `AgentModesDialog`, which reads the
// modes **file** through its own document route rather than through the
// dropdown's verdict — so it stays reachable when the dropdown's list failed to
// load, which is precisely when a broken file needs fixing.
const modesDialogOpen = ref(false)

// D10's enablement, the dropdown's own window and nothing wider:
//  * inert while a reply streams — the same window in which the Editor's writes
//    are refused, and an editor whose Save will be refused is worse than no
//    editor;
//  * inert while a lot runs — the lot was launched on the modes the file had
//    then, so the pencil joins the rule that makes the `<select>` inert for a
//    running lot, and never one of its own;
//  * live with a failed list — deliberately not gated on `agentModes.loaded`.
// The click guard is the same verdict, not a second rule: a synthetic or
// keyboard activation cannot open the card in the window it must not open in.
//
// Not `modeSelectDisabled(batch.active)`: that predicate carries the D7 lock
// too, and a locked *conversation* is no reason to refuse an edit of the
// workspace's file — a save made while locked is exactly the case D9's
// `resetDecisionAfterSave` handles by leaving the selection where it is.
const pencilDisabled = computed(() => batch.active || streaming.value)
// Naming: neither string may contain the select's own accessible name, or the
// two controls in this column would answer the same `getByLabel` query (the
// pinned e2e spec asks for "Agent mode" and expects exactly the select). The
// wording is the composer's own — "modes", the short affordance the other
// buttons use ("Save session (JSON)"), and the disabled reason the batch
// button already states in the same row.
const pencilTitle = computed(() =>
  pencilDisabled.value
    ? 'Available when no reply is streaming and no batch runs'
    : 'Edit modes',
)

function openModesDialog() {
  if (pencilDisabled.value) return
  modesDialogOpen.value = true
}

function closeModesDialog() {
  modesDialogOpen.value = false
}

// --- Image attach (the hidden file input below, driven by the icon button) ---

// Only usable when the configured endpoint is declared vision-capable
// (SUPPORTS_VISION in .env, via /api/config).
const supportsVision = computed(
  () => benchmarkState.config?.supportsVision === true,
)

const fileInput = ref(null)

function pickImage() {
  if (locked.value) return
  fileInput.value?.click()
}

function onFilePicked(event) {
  const input = event.target
  attachImage(input.files?.[0])
  // Reset so picking the same file again still fires the change event.
  input.value = ''
}

// --- Batch mode (the dialog itself lives in App.vue, like every modal) ---

// Active only while this tab's state is IDLE: `locked` covers a reply streaming
// and the whole of a running lot (including the gap between two task turns),
// and a benchmark run keeps the shared chat endpoint busy, so a launch there
// would be refused anyway.
const canOpenBatch = computed(
  () => !locked.value && benchmarkState.status !== 'running',
)

const batchDisabledReason = computed(() => {
  if (batch.active) return 'A batch is already running'
  if (streaming.value) return 'Available when no reply is streaming'
  if (benchmarkState.status === 'running')
    return 'Available when the benchmark is idle'
  return ''
})

// --- Copy the conversation as markdown ---

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    const area = document.createElement('textarea')
    area.value = text
    area.style.position = 'fixed'
    area.style.opacity = '0'
    document.body.appendChild(area)
    area.select()
    let ok = false
    try {
      ok = document.execCommand('copy')
    } catch {
      ok = false
    }
    document.body.removeChild(area)
    return ok
  }
}

const canCopy = computed(
  () => state.messages.length > 0 && !batch.active,
)

const copyState = ref('idle') // idle | copied | failed
let copyTimer = null

function clearCopyTimer() {
  clearTimeout(copyTimer)
  copyTimer = null
}

async function copyConversation() {
  if (!canCopy.value) return
  const ok = await copyText(conversationMarkdown(state.messages))
  copyState.value = ok ? 'copied' : 'failed'
  clearCopyTimer()
  copyTimer = setTimeout(() => {
    copyState.value = 'idle'
  }, 1500)
}

// --- Save / load the session as a JSON file (see useChat's conversationJson) ---

// Refuse an oversized file before even reading it, so a stray blob cannot freeze
// the tab inside JSON.parse. Kept well above what a *valid* session can hold:
// an image is capped at MAX_IMAGE_BYTES (10 MB) decoded, i.e. ~13.4 MB of
// base64, and every per-turn image the conversation carries lives in this one
// file — so the guard stays a guard against junk, not against real saved
// images. An image that is itself too large is reported per turn by
// parseSessionFile instead.
const MAX_SESSION_FILE_BYTES = 64 * 1024 * 1024

const canSave = computed(() => canCopy.value && !streaming.value)

function sessionFileName() {
  const now = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  const stamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`
  return `chat-session-${stamp}.json`
}

function saveSession() {
  if (!canSave.value) return
  const blob = new Blob([JSON.stringify(conversationJson(state.messages), null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  // The browser's native Save-As lets the user rename it from here.
  link.download = sessionFileName()
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
  state.error = null
}

const sessionInput = ref(null)

function pickSessionFile() {
  if (locked.value) return
  sessionInput.value?.click()
}

async function onSessionFilePicked(event) {
  const input = event.target
  const file = input.files?.[0]
  // Reset so picking the same file again still fires the change event.
  input.value = ''
  if (!file || streaming.value) return
  if (file.size > MAX_SESSION_FILE_BYTES) {
    state.error = `Session file too large (${humanFileSize(file.size)}, max ${humanFileSize(MAX_SESSION_FILE_BYTES)}).`
    return
  }
  let text
  try {
    text = await file.text()
  } catch {
    state.error = 'Could not read the selected session file.'
    return
  }
  let parsed
  try {
    parsed = parseSessionFile(text)
  } catch (exc) {
    state.error = `Cannot load session: ${exc?.message ?? exc}`
    return
  }
  if (
    state.messages.length &&
    !window.confirm('Replace the current conversation with the loaded session?')
  ) {
    return
  }
  const result = loadSession(parsed)
  if (!result) {
    state.error = 'Cannot load a session while a reply is streaming.'
  }
}

onBeforeUnmount(() => {
  registerDraftInput(null)
  clearCopyTimer()
})
</script>

<template>
  <div class="chat-input">
    <input
      ref="fileInput"
      type="file"
      accept="image/png,image/jpeg,image/webp,image/gif"
      class="chat-file-input"
      tabindex="-1"
      aria-hidden="true"
      @change="onFilePicked"
    />
    <input
      ref="sessionInput"
      type="file"
      accept="application/json,.json"
      class="chat-file-input"
      tabindex="-1"
      aria-hidden="true"
      @change="onSessionFilePicked"
    />
    <div class="chat-textarea-col">
      <!-- The batch meter, above the composer row: it owns the column's full
           width now that no side panel stands next to the textarea, and it
           only renders while a lot runs (or just after it ended). -->
      <BatchProgressBar />
      <!-- One row above the textarea: the session icon buttons flanking the
           memory bullets, which stay centered between the two button groups. -->
      <div class="chat-bullet-row">
        <div class="chat-bullet-actions chat-bullet-actions--left">
          <button
            class="icon-btn chat-attach"
            type="button"
            :disabled="!supportsVision || locked"
            :class="{ 'chat-attach--pending': !!state.pendingImage }"
            :aria-label="
              supportsVision
                ? state.pendingImage
                  ? 'Replace attached image'
                  : 'Attach image'
                : 'Image input not supported'
            "
            :title="
              !supportsVision
                ? 'The configured model does not support image input (enable it in Settings → LLM)'
                : state.pendingImage
                  ? 'Replace attached image'
                  : 'Attach image'
            "
            @click="pickImage"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <path d="m21 15-5-5L5 21" />
            </svg>
            <span v-if="state.pendingImage" class="chat-attach-badge" aria-hidden="true"></span>
          </button>
          <button
            class="icon-btn"
            type="button"
            :disabled="!canCopy"
            :aria-label="copyState === 'copied' ? 'Conversation copied' : 'Copy conversation as markdown'"
            :title="
              copyState === 'copied'
                ? 'Copied'
                : copyState === 'failed'
                  ? 'Copy failed'
                  : 'Copy conversation (markdown)'
            "
            @click="copyConversation"
          >
            <template v-if="copyState === 'copied'">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M20 6 9 17l-5-5" />
              </svg>
            </template>
            <template v-else>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <rect x="9" y="9" width="13" height="13" rx="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            </template>
          </button>
          <button
            class="icon-btn"
            type="button"
            :disabled="!canSave"
            aria-label="Save session as JSON"
            title="Save session (JSON)"
            @click="saveSession"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M4 14v1a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-1" />
              <path d="M12 3v12" />
              <path d="m7 10 5 5 5-5" />
            </svg>
          </button>
          <button
            class="icon-btn"
            type="button"
            :disabled="locked"
            aria-label="Load session from a JSON file"
            title="Load session (JSON)"
            @click="pickSessionFile"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M4 14v1a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-1" />
              <path d="M12 15V3" />
              <path d="m7 8 5-5 5 5" />
            </svg>
          </button>
          <!-- Batch mode, right of the JSON session buttons: it belongs to this
               tab only, so it sits in the composer row instead of the app bar.
               Live only while the composer is fully idle (see canOpenBatch) —
               a lot always starts on an empty thread. -->
          <button
            class="icon-btn"
            type="button"
            :disabled="!canOpenBatch"
            aria-label="Batch mode"
            :title="canOpenBatch ? 'Batch mode' : batchDisabledReason"
            @click="openBatchDialog"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M4 6h10" />
              <path d="M4 12h16" />
              <path d="M4 18h7" />
              <path d="m17 15 3 3-3 3" />
            </svg>
          </button>
          <!-- The agent mode this conversation answers as (am-07), and the
               pencil that edits it (eds-07). The **column** always renders: the
               pencil is useful whatever the dropdown's verdict, and a modes
               file whose list could not be loaded is precisely the one worth
               opening. The `<select>` inside it is still rendered only once the
               list loaded — a control that cannot be used is worse than no
               control, and with a failed fetch the payload omits `agentMode`, so
               the server's own default applies (the pre-feature behaviour). The
               column sits in the left action group, after the buttons, so the
               memory bullets stay centered as the layout promises. The select is
               inert while a batch runs: the lot is one job executed under one
               mode the runner pins for the whole run (D13), so the dropdown can
               neither change it mid-lot nor pretend it owns it. -->
          <div class="chat-mode-col">
            <div class="chat-mode-row">
              <select
                v-if="agentModes.loaded"
                class="chat-mode-select"
                :value="agentModes.current"
                :disabled="modeSelectDisabled(batch.active)"
                :title="modeTitle"
                aria-label="Agent mode"
                @change="onModeChange"
              >
                <option
                  v-for="mode in agentModes.modes"
                  :key="mode.id"
                  :value="mode.id"
                  :disabled="modeOptionDisabled(mode.id)"
                  :title="modeHint(mode.id)"
                >
                  {{ mode.label }}
                </option>
              </select>
              <!-- The pencil (eds-07): this workspace's modes editor, immediately
                   after the select and on its line, drawn with the glyph the app
                   already uses for "Edit and resend". It opens the card that
                   edits the modes **file**, so it is *not* gated on the dropdown
                   having loaded — a file whose list could not be loaded is
                   exactly the one worth opening. Its accessible name deliberately
                   does not contain the select's ("Agent mode"), so the two
                   controls stay distinguishable to a query and to a reader. -->
              <button
                class="icon-btn chat-mode-edit"
                type="button"
                :disabled="pencilDisabled"
                aria-label="Edit modes"
                :title="pencilTitle"
                @click="openModesDialog"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
                </svg>
              </button>
            </div>
            <!-- The two quiet lines this control can carry, never a modal and
                 never a banner: a session asked for a mode this workspace does
                 not have (its substitution is said out loud, not chosen
                 silently), or the hand-edited modes file has findings the server
                 could not use (D11 — the defaults still answer, and the fix
                 lives outside the app). It is also the slot D9's skipped reset
                 speaks through, so it renders with or without the select: the
                 column is there for the pencil whatever the list's verdict. -->
            <p v-if="agentModes.notice" class="chat-mode-findings" role="status">
              {{ agentModes.notice }}
            </p>
            <p v-else-if="modeFindingsLine" class="chat-mode-findings" role="status">
              {{ modeFindingsLine }}
            </p>
          </div>
        </div>
        <MemoryBullets @configure="openBulletDialog" />
        <div class="chat-bullet-actions chat-bullet-actions--right">
          <button
            class="icon-btn"
            type="button"
            :disabled="!canRestart"
            aria-label="Restart conversation"
            title="Restart conversation"
            @click="restart"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
              <path d="M3 3v5h5" />
            </svg>
          </button>
          <!-- Pause: the non-destructive twin of Stop, and it only exists
               while a reply streams — the `pause_ack` returns the app to idle,
               so this button leaves the row exactly when Send comes back.
               Blinking means the request is armed and the button is not
               clickable; typing a comment and sending it is the only way back
               into inference. Hidden during a batch: pausing one turn and
               letting the loop carry on onto the next task would contradict
               "Stop is the only control", and a comment is meaningless between
               automated tasks. -->
          <button
            v-if="streaming && !batch.active"
            class="icon-btn chat-pause"
            type="button"
            :disabled="state.pausePending"
            :class="{ 'is-paused-blink': state.pausePending }"
            :aria-label="state.pausePending ? 'Pausing…' : 'Pause — keep the partial reply, add a comment'"
            :title="state.pausePending ? 'Pausing…' : 'Pause'"
            @click="pause"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="7" y="5" width="4" height="14" rx="1" />
              <rect x="14" y="5" width="4" height="14" rx="1" />
            </svg>
          </button>
          <!-- Send and Stop swap on `streaming`, except during a batch: the lot
               keeps Stop live across the microsecond its own loop spends between
               two task turns, and Send is inert for the whole lot anyway. -->
          <button
            v-if="!streaming && !batch.active"
            class="icon-btn chat-send"
            type="button"
            :disabled="!canSend"
            aria-label="Send message"
            title="Send"
            @click="submit"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M22 2 11 13" />
              <path d="M22 2 15 22l-4-9-9-4 20-7z" />
            </svg>
          </button>
          <button
            v-else
            class="icon-btn chat-send chat-stop"
            type="button"
            aria-label="Stop generating"
            :title="batch.active ? 'Stop the batch' : 'Stop'"
            @click="batch.active ? stopBatch() : stop()"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="5" y="5" width="14" height="14" rx="2" />
            </svg>
          </button>
        </div>
      </div>
      <textarea
        ref="draftEl"
        :value="state.draft.raw"
        class="chat-textarea"
        rows="3"
        :disabled="batch.active"
        :placeholder="
          batch.active
            ? 'A batch is running — only Stop and Restart are live.'
            : 'Ask the model something… (Enter to send, Shift+Enter for a new line)'
        "
        :aria-label="
          batch.active
            ? 'WorkBench input (unavailable while a batch runs)'
            : streaming
              ? 'WorkBench input (reply streaming)'
              : 'WorkBench input'
        "
        @input="setDraft($event.target.value)"
        @keydown="onKeydown"
      ></textarea>
    </div>
    <RegistrationMemoryDlg
      :open="dialogOpen"
      :index="dialogIndex"
      @close="dialogOpen = false"
    />
    <!-- The modes editor, mounted where it is used (the way the bullet dialog
         above is): `batchActive` is D10's enablement, handed to the card so its
         Save says the same thing the pencil already did.

         `v-if` rather than an always-mounted card: a closed `<dialog>` is still
         in the DOM, and this one is labelled "Agent modes" — a substring of the
         select's own accessible name, which the pinned e2e spec queries
         (`getByLabel('Agent mode')` must answer exactly the select, and a second
         match is a strict-mode failure). Mounting it only while open keeps the
         closed card out of every such query, and the card's own watch runs with
         `immediate`, so mounting it *is* opening it. -->
    <AgentModesDialog
      v-if="modesDialogOpen"
      :open="modesDialogOpen"
      :batch-active="batch.active"
      @close="closeModesDialog"
    />
  </div>
</template>
