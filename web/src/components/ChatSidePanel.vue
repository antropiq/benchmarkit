<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import {
  conversationMarkdown,
  estimateCompletionTokens,
  useChat,
} from '../composables/useChat'
import { useBenchmark } from '../composables/useBenchmark'

const props = defineProps({
  canSend: { type: Boolean, default: false },
  streaming: { type: Boolean, default: false },
})

const emit = defineEmits(['send'])

const { state, stop, reset, attachImage } = useChat()
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

// Session badge: tokens actually CONSUMED (prompt + completion, summed per
// round — every tool round re-sends and re-processes the whole history),
// not the completion-only figure behind the tok/s badge.
function historyChars(messages) {
  // Rough char cost of the prompt the next request carries: the text of
  // every message so far (state messages keep their `raw` text; the OpenAI
  // content-parts shape is only built on send). Images are excluded, like
  // the backend's own prompt estimate, which is text-only too.
  let chars = 0
  for (const message of messages) chars += message.raw?.length ?? 0
  return chars
}

const totalTokens = computed(() => {
  let total = 0
  for (let i = 0; i < state.messages.length; i += 1) {
    const message = state.messages[i]
    if (message.role !== 'assistant') continue
    if (!message.streaming) {
      total += message.consumed || message.tokens
      continue
    }
    // Live: exact cumulative consumed tokens from the last `usage` event
    // (every ended round's prompt + generation, exact when the server
    // reports usage) plus a ~4 chars/token estimate of the in-flight round:
    // the prompt it carries — the conversation so far, the backend's own
    // fallback — and the answer text streamed so far (thinking and tool
    // arguments are invisible to the client either way).
    if (message.usageConsumed > 0) {
      total +=
        message.usageConsumed +
        estimateCompletionTokens(
          historyChars(state.messages.slice(0, i)) +
            message.raw.slice(message.usageRawLen || 0),
        )
    } else {
      total += estimateCompletionTokens(
        historyChars(state.messages.slice(0, i)) + message.raw,
      )
    }
  }
  return total
})

const tokensEstimated = computed(() => {
  for (const message of state.messages) {
    if (message.role !== 'assistant') continue
    if (message.streaming) {
      // The in-flight round always carries a client-side char estimate of
      // its prompt and answer tail, however small: flag the total.
      return true
    }
    if (
      (message.consumed || message.tokens) > 0 &&
      (message.consumedEstimated || message.tokensEstimated || !message.consumed)
    ) {
      return true
    }
  }
  return false
})

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
  if (active && !active.tokens) {
    // MCP tool loop: cumulative usage from the last `usage` event — exact
    // tokens over the exact decode window (reasoning and tool-argument
    // tokens included), so tool execution time never enters the rate.
    if (active.usageTokens > 0 && active.usageDecode > 0) {
      return {
        value: active.usageTokens / active.usageDecode,
        estimated: false,
      }
    }
    // Answer text streaming: continue from the usage baseline when tool
    // rounds ran (exact tokens + decode so far, extended by the estimated
    // tail over the client-measured window), else pure client estimate
    // over the answer window only — thinking tokens are unknown to the
    // client, so the window starts at the first answer token too.
    if (
      state.phase === 'answering' &&
      active.firstTokenAt &&
      active.lastTokenAt > active.firstTokenAt
    ) {
      const tail = estimateCompletionTokens(
        active.raw.slice(active.usageRawLen || 0),
      )
      if (active.usageTokens > 0 && active.usageAt) {
        return {
          value:
            (active.usageTokens + tail) /
            (active.usageDecode +
              (active.lastTokenAt - active.usageAt) / 1000),
          estimated: false,
        }
      }
      const secs = (active.lastTokenAt - active.firstTokenAt) / 1000
      return {
        value: estimateCompletionTokens(active.raw) / secs,
        estimated: true,
      }
    }
  } else if (active && active.tokens) {
    // done received but the stream is still closing: show the exact rate.
    const value = exactRate(active)
    if (value !== null) {
      return { value, estimated: active.tokensEstimated }
    }
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

const tokensText = computed(() =>
  totalTokens.value
    ? `${tokensEstimated.value ? '≈ ' : ''}${totalTokens.value} tokens`
    : '— tokens',
)

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
    <span class="badge" :class="rate ? 'violet' : 'gray'" role="status">
      {{ rateText }}
    </span>
    <span class="badge" :class="totalTokens ? 'green' : 'gray'" role="status">
      {{ tokensText }}
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
  </aside>
</template>
