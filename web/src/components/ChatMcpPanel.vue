<script setup>
import { computed } from 'vue'
import { useChat } from '../composables/useChat'

const { state, setChatMcp } = useChat()

// Status dot next to each switch: green when the connector serves its
// tools, red on a startup/configuration error, gray otherwise (stopped,
// disabled or still starting).
const DOT_CLASS = {
  running: 'chat-mcp-dot--green',
  error: 'chat-mcp-dot--red',
}

// Hidden while the snapshot is unavailable or nothing is activated in
// Settings: only Settings-launched connectors may ever appear here.
const visible = computed(
  () => state.mcp.loaded && state.mcp.connectors.length > 0,
)
const locked = computed(() => state.mcp.locked || state.phase !== 'idle')

function rowTitle(connector) {
  if (locked.value) {
    return 'Toggles apply to the next conversation — restart to change them'
  }
  if (connector.status !== 'running') {
    return `Connector is ${connector.status} — its tools are unavailable`
  }
  return `Use ${connector.label} tools in this conversation`
}
</script>

<template>
  <aside
    v-if="visible"
    class="chat-mcp-panel"
    aria-label="MCP connectors for this conversation"
  >
    <p class="caption chat-mcp-title">MCP tools</p>
    <label
      v-for="connector in state.mcp.connectors"
      :key="connector.id"
      class="chat-mcp-row"
      :class="{ 'chat-mcp-row--dim': connector.status !== 'running' }"
      :title="rowTitle(connector)"
    >
      <input
        type="checkbox"
        role="switch"
        :checked="state.mcp.enabled[connector.id]"
        :disabled="locked"
        :aria-label="`MCP ${connector.label} for this conversation`"
        @change="setChatMcp(connector.id, $event.target.checked)"
      />
      <span
        class="chat-mcp-dot"
        :class="DOT_CLASS[connector.status] ?? 'chat-mcp-dot--gray'"
        :title="`Status: ${connector.status}`"
      ></span>
      <span class="chat-mcp-label">{{ connector.label }}</span>
    </label>
  </aside>
</template>
