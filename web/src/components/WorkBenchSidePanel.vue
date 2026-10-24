<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import {
  conversationJson,
  conversationMarkdown,
  humanFileSize,
  liveTokenRate,
  parseSessionFile,
  useChat,
} from '../composables/useChat'
import { useBenchmark } from '../composables/useBenchmark'
import ChatStopwatch from './ChatStopwatch.vue'

const props = defineProps({
  canSend: { type: Boolean, default: false },
  streaming: { type: Boolean, default: false },
})

const emit = defineEmits(['send'])

const { state, stop, reset, attachImage, loadSession } = useChat()
const { state: benchmarkState } = useBenchmark()

// The image attach button is only usable when the configured endpoint is
// declared vision-capable (SUPPORTS_VISION in .env, via /api/config).
const supportsVision = computed(
  () => benchmarkState.config?.supportsVision === true,
)

const fileInput = ref(null)

function pickImage() {
  fileInput.value?.click()
}

function onFilePicked(event) {
  const input = event.target
  attachImage(input.files?.[0])
  // Reset so picking the same file again still fires the change event.
  input.value = ''
}

const streamingMessage = computed(
  () => state.messages.find((m) => m.streaming) ?? null,
)

// Same convention as the benchmark tokenRate: tokens / decode window.
// The server-side `decode` (cumulative generation seconds across all tool
// rounds, tool execution and re-prefill gaps excluded) is the preferred
// denominator; `took - ttft` only backs up payloads without it (old
// servers — where there were no tool rounds, so the window is exact).
function exactRate(message) {
  const secs = message.decode > 0 ? message.decode : message.took - message.ttft
  return secs > 0 ? message.tokens / secs : null
}

const rate = computed(() => {
  const active = streamingMessage.value
  if (active) {
    // `done` already arrived but the stream is still closing: exact figures.
    if (active.tokens) {
      const value = exactRate(active)
      if (value !== null) return { value, estimated: active.tokensEstimated }
    }
    // Still generating: the live rate covers the reasoning phase as well as
    // the answer (see liveTokenRate), so the badge ticks while the model
    // thinks instead of freezing on the previous reply's figure.
    const live = liveTokenRate(active)
    if (live) return live
  }
  const finished = [...state.messages]
    .reverse()
    .find(
      (m) =>
        m.role === 'assistant' && !m.streaming && m.tokens > 0 && exactRate(m) !== null,
    )
  if (finished) {
    return { value: exactRate(finished), estimated: finished.tokensEstimated }
  }
  return null
})

const rateText = computed(() =>
  rate.value
    ? `${rate.value.estimated ? '≈ ' : ''}${rate.value.value.toFixed(1)} t/s`
    : '— t/s',
)

// Context-window occupancy: the LAST round's prompt + completion (exactly
// what the server slots into its window — template, tool schemas and
// thinking-round history included), never summed across rounds/turns. Fed
// by `done`, refreshed between tool rounds by `usage`; the cap comes from
// /api/config (meta.n_ctx).
const contextUsage = computed(() => {
  for (let i = state.messages.length - 1; i >= 0; i -= 1) {
    const message = state.messages[i]
    if (message.role === 'assistant' && message.contextTokens > 0) {
      return { tokens: message.contextTokens, estimated: message.contextEstimated }
    }
  }
  return null
})

const contextWindow = computed(
  () => benchmarkState.config?.contextWindow ?? null,
)

function formatTokens(n) {
  return n >= 1000 ? `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k` : `${n}`
}

const contextFill = computed(() => {
  const usage = contextUsage.value
  if (!usage) return null
  return contextWindow.value ? usage.tokens / contextWindow.value : null
})

const contextText = computed(() => {
  const usage = contextUsage.value
  if (!usage) return '— ctx'
  const cap = contextWindow.value
  return (
    `${usage.estimated ? '≈ ' : ''}ctx ${formatTokens(usage.tokens)}` +
    (cap ? ` / ${formatTokens(cap)}` : '')
  )
})

const contextTitle = computed(() => {
  const fill = contextFill.value
  const parts = [
    'Tokens in the model context window (last round: prompt + generation, tool schemas included).',
  ]
  if (fill !== null) {
    parts.push(
      fill > 0 && fill < 0.01
        ? '<1% of the window used.'
        : `${Math.round(fill * 100)}% of the window used.`,
    )
  }
  return parts.join(' ')
})

const contextClass = computed(() => {
  const fill = contextFill.value
  if (fill === null) return contextUsage.value ? 'green' : 'gray'
  if (fill >= 0.9) return 'red'
  if (fill >= 0.7) return 'violet'
  return 'green'
})

const canCopy = computed(() => state.messages.length > 0)
const canRestart = computed(
  () => state.messages.length > 0 || state.phase !== 'idle',
)

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

const copyState = ref('idle') // idle | copied | failed
let copyTimer = null

async function copyConversation() {
  if (!canCopy.value) return
  const ok = await copyText(conversationMarkdown(state.messages))
  copyState.value = ok ? 'copied' : 'failed'
  clearTimeout(copyTimer)
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

const canSave = computed(() => canCopy.value && !props.streaming)

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
  if (props.streaming) return
  sessionInput.value?.click()
}

async function onSessionFilePicked(event) {
  const input = event.target
  const file = input.files?.[0]
  // Reset so picking the same file again still fires the change event.
  input.value = ''
  if (!file || props.streaming) return
  if (file.size > MAX_SESSION_FILE_BYTES) {
    state.error = `Session file too large (${humanFileSize(file.size)}, max 2 MB).`
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

onBeforeUnmount(() => clearTimeout(copyTimer))

function restart() {
  stop()
  reset()
}
</script>

<template>
  <aside class="chat-side-panel" aria-label="Chat stats and actions">
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
    <span class="chat-stat-badges">
      <span class="badge" :class="rate ? 'violet' : 'gray'" role="status">
        {{ rateText }}
      </span>
      <span class="badge" :class="contextClass" role="status" :title="contextTitle">
        {{ contextText }}
      </span>
    </span>
    <div class="chat-side-actions">
      <button
        class="icon-btn chat-attach"
        type="button"
        :disabled="!supportsVision || streaming"
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
        :disabled="streaming"
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
      <button
        v-if="!streaming"
        class="icon-btn chat-send"
        type="button"
        :disabled="!canSend"
        aria-label="Send message"
        title="Send"
        @click="emit('send')"
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
        title="Stop"
        @click="stop"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <rect x="5" y="5" width="14" height="14" rx="2" />
        </svg>
      </button>
    </div>
    <ChatStopwatch />
  </aside>
</template>
