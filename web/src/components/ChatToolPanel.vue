<script setup>
import { computed, ref } from 'vue'
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

const list = ref(null)
const signature = computed(() => {
  const last = tools.value[tools.value.length - 1]
  return `${tools.value.length}:${last ? `${last.id}-${last.state}` : ''}`
})
const { onScroll } = useAutoScroll(list, signature)
</script>

<template>
  <aside
    ref="list"
    class="chat-tool-panel"
    aria-label="MCP tool calls"
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
  </aside>
</template>
