<script setup>
import { computed } from 'vue'
import { liveTokenRate, useChat } from '../composables/useChat'
import { useBenchmark } from '../composables/useBenchmark'
import ChatStopwatch from './ChatStopwatch.vue'
import BatchProgressBar from './BatchProgressBar.vue'

const { state } = useChat()
const { state: benchmarkState } = useBenchmark()

// Only the stat badges and the stopwatch live here now: the composer's icon
// buttons (attach / copy / save / load / restart / send) sit in the bullet row
// of WorkBenchInput, above the textarea.
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
</script>

<template>
  <aside class="chat-side-panel" aria-label="Chat stats and actions">
    <span class="chat-stat-badges">
      <span class="badge" :class="rate ? 'violet' : 'gray'" role="status">
        {{ rateText }}
      </span>
      <span class="badge" :class="contextClass" role="status" :title="contextTitle">
        {{ contextText }}
      </span>
    </span>
    <ChatStopwatch />
    <BatchProgressBar />
  </aside>
</template>
