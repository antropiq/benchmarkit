<script setup>
import { computed } from 'vue'
import { useChat } from '../composables/useChat'

const { state, setChatLsp } = useChat()

// Status dot next to each switch: green when the language server serves its
// tools, red on a startup/configuration error, gray otherwise (stopped,
// disabled or still starting — a lazy server shows gray until its first
// tool call spawns it). Reuses the chat-mcp classes: same visual row.
const DOT_CLASS = {
  running: 'chat-mcp-dot--green',
  error: 'chat-mcp-dot--red',
}

// Hidden while the snapshot is unavailable or nothing is activated in
// Settings: only Settings-launched language servers may appear here.
const visible = computed(
  () => state.lsp.loaded && state.lsp.servers.length > 0,
)
const locked = computed(() => state.lsp.locked || state.phase !== 'idle')

function rowTitle(server) {
  if (locked.value) {
    return 'Toggles apply to the next conversation — restart to change them'
  }
  if (server.status !== 'running') {
    return `Language server is ${server.status} — its tools start on the first call`
  }
  return `Use ${server.label} tools in this conversation`
}
</script>

<template>
  <aside
    v-if="visible"
    class="chat-lsp-panel"
    aria-label="LSP language servers for this conversation"
  >
    <p class="caption chat-mcp-title">LSP tools</p>
    <label
      v-for="server in state.lsp.servers"
      :key="server.id"
      class="chat-mcp-row"
      :class="{ 'chat-mcp-row--dim': server.status !== 'running' }"
      :title="rowTitle(server)"
    >
      <input
        type="checkbox"
        role="switch"
        :checked="state.lsp.enabled[server.id]"
        :disabled="locked"
        :aria-label="`LSP ${server.label} for this conversation`"
        @change="setChatLsp(server.id, $event.target.checked)"
      />
      <span
        class="chat-mcp-dot"
        :class="DOT_CLASS[server.status] ?? 'chat-mcp-dot--gray'"
        :title="`Status: ${server.status}`"
      ></span>
      <span class="chat-mcp-label">{{ server.label }}</span>
    </label>
  </aside>
</template>
