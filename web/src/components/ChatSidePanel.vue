<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import {
  conversationMarkdown,
  estimateCompletionTokens,
  useChat,
} from '../composables/useChat'

const { state, stop, reset } = useChat()

const streamingMessage = computed(
  () => state.messages.find((m) => m.streaming) ?? null,
)

const totalTokens = computed(() => {
  let total = 0
  for (const message of state.messages) {
    if (message.role !== 'assistant') continue
    total += message.streaming
      ? estimateCompletionTokens(message.raw)
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
    ? `${rate.value.estimated ? '≈ ' : ''}${rate.value.value.toFixed(1)} tok/s`
    : '— tok/s',
)

const tokensText = computed(() =>
  totalTokens.value
    ? `${tokensEstimated.value ? '≈ ' : ''}${totalTokens.value} tok`
    : '— tok',
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
    <span class="badge" :class="rate ? 'violet' : 'gray'" role="status">
      {{ rateText }}
    </span>
    <span class="badge" :class="totalTokens ? 'green' : 'gray'" role="status">
      {{ tokensText }}
    </span>
    <div class="chat-side-actions">
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
    </div>
  </aside>
</template>
