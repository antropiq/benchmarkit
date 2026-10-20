<script setup>
import { computed, ref } from 'vue'
import GaugeBar from './GaugeBar.vue'
import { useChat } from '../composables/useChat'
import { useAutoScroll } from '../composables/useAutoScroll'

const { state } = useChat()

// Chronological log of every MCP tool call in the conversation (reactive
// items, so tool_result state flips re-render the badges in place).
const tools = computed(() =>
  state.messages
    .filter((m) => m.role === 'assistant' && m.tools?.length)
    .flatMap((m) => m.tools),
)

// Iteration gauge tracks the latest assistant reply: 0 while it streams
// before its first tool round, holds its final level afterwards, and drops
// to the previous reply (or 0) when Stop/editLast splices messages away.
const lastAssistant = computed(() => {
  for (let i = state.messages.length - 1; i >= 0; i -= 1) {
    if (state.messages[i].role === 'assistant') return state.messages[i]
  }
  return null
})
const gaugeValue = computed(() => lastAssistant.value?.mcpIteration ?? 0)
const gaugeMax = computed(() => lastAssistant.value?.mcpMax ?? 0)

const list = ref(null)
const signature = computed(() => {
  const last = tools.value[tools.value.length - 1]
  return `${tools.value.length}:${last ? `${last.id}-${last.state}` : ''}`
})
const { onScroll } = useAutoScroll(list, signature)
</script>

<template>
  <aside class="chat-tool-panel" aria-label="MCP tool calls">
    <div
      ref="list"
      class="chat-tool-scroll"
      @scroll.passive="onScroll"
    >
      <ul class="chat-tool-list">
        <li
          v-for="tool in tools"
          :key="tool.id"
          class="chat-tool"
          :class="tool.state"
          :title="`${tool.server ? tool.server + ' · ' : ''}${tool.preview}${tool.resultPreview ? '\n→ ' + tool.resultPreview : ''}`"
        >
          <span class="chat-tool-state" aria-hidden="true">
            {{ tool.state === 'running' ? '…' : tool.state === 'ok' ? '✓' : '✗' }}
          </span>
          <span class="chat-tool-name">{{ tool.tool }}</span>
        </li>
      </ul>
    </div>
    <GaugeBar
      class="chat-tool-gauge"
      :value="gaugeValue"
      :max="gaugeMax"
      label="Tool-loop iterations of the current reply vs MAX_MCP_ITERATION — one iteration may run several calls (badges count individual calls)"
    />
  </aside>
</template>
