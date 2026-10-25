<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import WorkBenchLspPanel from './WorkBenchLspPanel.vue'
import WorkBenchMcpPanel from './WorkBenchMcpPanel.vue'
import WorkBenchSidePanel from './WorkBenchSidePanel.vue'
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
const { state: batch, stopBatch, clearBatchRun } = useBatch()

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

// The left toggle column (MCP beside LSP) only takes horizontal space when
// at least one of the two panels has something to show - mirrors the
// per-panel `visible` conditions inside WorkBenchMcpPanel/WorkBenchLspPanel.
const mcpVisible = computed(
  () => state.mcp.loaded && state.mcp.connectors.length > 0,
)
const lspVisible = computed(
  () => state.lsp.loaded && state.lsp.servers.length > 0,
)
const showToggleCol = computed(() => mcpVisible.value || lspVisible.value)
// With only one of the two panels rendered the column shrinks to a single
// panel — WorkBenchToolPanel mirrors this so the two rows stay aligned.
const singlePanel = computed(() => mcpVisible.value !== lspVisible.value)

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
    <div
      v-if="showToggleCol"
      class="chat-toggle-col"
      :class="{ 'chat-toggle-col--single': singlePanel }"
    >
      <WorkBenchMcpPanel />
      <WorkBenchLspPanel />
    </div>
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
    <WorkBenchSidePanel />
  </div>
</template>
