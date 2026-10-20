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

const totalTokens = computed(() => {
  let total = 0
  for (const message of state.messages) {
    if (message.role !== 'assistant') continue
    // Live usage events (exact, tool-call argument tokens included) beat
    // the raw-text estimate; max() keeps the count ticking mid-round too.
    total += message.streaming
      ? Math.max(
          message.usageTokens || 0,
          estimateCompletionTokens(message.raw),
        )
      : message.tokens
  }
  return total
})

const tokensEstimated = computed(
  () =>
    streamingMessage.value !== null ||
    state.messages.some(
      (m) => m.role === 'assistant' && m.tokensEstimated && m.tokens > 0,
    ),
)

// Same convention as the benchmark tokenRate: tokens / decode window.
// Exact once the `done` event carried usage; live-estimated while answering.
const rate = computed(() => {
  const active = streamingMessage.value
  if (active && !active.tokens) {
    if (
      state.phase === 'answering' &&
      active.firstTokenAt &&
      active.lastTokenAt > active.firstTokenAt
    ) {
      const secs = (active.lastTokenAt - active.firstTokenAt) / 1000
      return {
        value: estimateCompletionTokens(active.raw) / secs,
        estimated: true,
      }
    }
    // MCP tool loop (no answer text streamed yet): cumulative usage from
    // the last `usage` event — exact tokens over the decode window, so
    // tool execution time never enters the rate.
    if (active.usageTokens > 0 && active.usageDecode > 0) {
      return {
        value: active.usageTokens / active.usageDecode,
        estimated: false,
      }
    }
  } else if (active && active.tokens && active.took > active.ttft) {
    // done received but the stream is still closing: show the exact rate.
    return {
      value: active.tokens / (active.took - active.ttft),
      estimated: active.tokensEstimated,
    }
  }
  const finished = [...state.messages]
    .reverse()
    .find(
      (m) =>
        m.role === 'assistant' &&
        !m.streaming &&
        m.tokens > 0 &&
        m.took > m.ttft,
    )
  if (finished) {
    return {
      value: finished.tokens / (finished.took - finished.ttft),
      estimated: finished.tokensEstimated,
    }
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
