<script setup>
import { computed, ref } from 'vue'
import GaugeBar from './GaugeBar.vue'
import { useChat } from '../composables/useChat'
import { useAutoScroll } from '../composables/useAutoScroll'

const { state } = useChat()

// Chronological timeline of everything the right-hand chat stream does NOT
// show: the model's streamed reasoning flows and the MCP tool-call badges,
// in the order the backend emitted them (reasoning → calls → next round's
// reasoning…). Blocks are reactive objects, so tool_result state flips
// re-render the badges in place. Append-only, hence index keys are stable.
const blocks = computed(() =>
  state.messages
    .filter((m) => m.role === 'assistant')
    .flatMap((m) => m.blocks ?? []),
)
const reasoningChars = computed(() =>
  blocks.value.reduce(
    (sum, block) => sum + (block.kind === 'reasoning' ? block.text.length : 0),
    0,
  ),
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

// Width follows the input row's toggle column below (same `visible`
// conditions as WorkBenchInput.vue): two panels wide only when both the MCP and
// the LSP panel are rendered, one panel wide when just one of them is, and
// one panel wide when neither is (no column to align with).
const singlePanel = computed(
  () =>
    !(
      state.mcp.loaded &&
      state.mcp.connectors.length > 0 &&
      state.lsp.loaded &&
      state.lsp.servers.length > 0
    ),
)

const list = ref(null)
const signature = computed(() => {
  const last = blocks.value[blocks.value.length - 1]
  const tail = last ? `${last.kind}-${last.kind === 'tool' ? `${last.tool.id}-${last.tool.state}` : ''}` : ''
  return `${blocks.value.length}:${reasoningChars.value}:${tail}`
})
const { onScroll } = useAutoScroll(list, signature)

// Badge text: ``<tool>::<method>`` — the hub the call was routed to (MCP
// connector id, or the language-server id for `lsp_<id>_<action>` tools)
// then the callable name. Older payloads without a server fall back to the
// bare method name. LSP tool names already carry their server id
// (`lsp_vue_diagnostics`), so the redundant prefix is dropped there.
function toolLabel(tool) {
  if (!tool.server) return tool.tool
  const namespaced = `lsp_${tool.server}_`
  const method = tool.tool.startsWith(namespaced)
    ? tool.tool.slice(namespaced.length)
    : tool.tool
  return `${tool.server}::${method}`
}
</script>

<template>
  <aside
    class="chat-tool-panel"
    :class="{ 'chat-tool-panel--single': singlePanel }"
    aria-label="Model reasoning and MCP tool calls"
  >
    <div
      ref="list"
      class="chat-tool-scroll"
      @scroll.passive="onScroll"
    >
      <ul class="chat-tool-list">
        <template v-for="(block, i) in blocks" :key="i">
          <li
            v-if="block.kind === 'tool'"
            class="chat-tool"
            :class="block.tool.state"
            :title="`${block.tool.server ? block.tool.server + ' · ' : ''}${block.tool.preview}${block.tool.resultPreview ? '\n→ ' + block.tool.resultPreview : ''}`"
          >
            <span class="chat-tool-state" aria-hidden="true">
              {{ block.tool.state === 'running' ? '…' : block.tool.state === 'ok' ? '✓' : '✗' }}
            </span>
            <span class="chat-tool-name">
              {{ toolLabel(block.tool) }}
            </span>
          </li>
          <li v-else class="chat-reasoning">
            {{ block.text }}
          </li>
        </template>
      </ul>
    </div>
    <GaugeBar
      class="chat-tool-gauge"
      :value="gaugeValue"
      :max="gaugeMax"
      label="Tool-loop iterations of the current reply vs MAX_TOOL_ITERATION — MCP tools and language servers share the counter; one iteration may run several calls (badges count individual calls)"
    />
  </aside>
</template>
