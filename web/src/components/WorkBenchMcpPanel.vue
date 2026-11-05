<script setup>
import { computed } from 'vue'
import { useChat } from '../composables/useChat'
import { useBatch } from '../composables/useBatch'
import { hubGateReason, modeBlocksHub } from '../composables/useAgentModes'
import { mcpVisible } from '../lib/chat-hubs'

const { state, setChatMcp, restartMcp } = useChat()
const { state: batch } = useBatch()

// Status dot next to each switch: green when the connector serves its
// tools, red on a startup/configuration error, gray otherwise (stopped,
// disabled or still starting).
const DOT_CLASS = {
  running: 'chat-mcp-dot--green',
  error: 'chat-mcp-dot--red',
}

// Hidden while the snapshot is unavailable or nothing is activated in
// Settings: only Settings-launched connectors may ever appear here.
const visible = computed(() => mcpVisible(state))
// The toggles freeze with the first accepted send anyway; the batch flag covers
// the between-tasks window, so a lot really does run under one identical tool
// set (useChat's own guard refuses the change too, so the rule lives in both
// places it can be reached from).
const locked = computed(
  () => state.mcp.locked || state.phase !== 'idle' || batch.active,
)

// D4's coarsest switch: the active agent mode can switch the whole hub off. The
// panel then renders **disabled with a reason** rather than disappearing — the
// toggles stay real (Settings owns them) and a hidden control that reappears on
// a mode change reads as a bug. `modeBlocksHub` is false when the mode list
// never loaded, which is exactly the pre-feature rendering.
const gatedByMode = computed(() => modeBlocksHub('mcp'))
const gateReason = computed(() => hubGateReason('mcp'))
const switchesDisabled = computed(() => locked.value || gatedByMode.value)

function rowTitle(connector) {
  if (gatedByMode.value) return gateReason.value
  if (batch.active) {
    return 'Unavailable while a batch runs — stop it to change the tools'
  }
  if (locked.value) {
    return 'Toggles apply to the next conversation — restart to change them'
  }
  if (connector.status !== 'running') {
    return `Connector is ${connector.status} — its tools are unavailable`
  }
  return `Use ${connector.label} tools in this conversation`
}

// Restarting reconnects every connector from a fresh .nojae/mcp-settings.json
// scan, so
// a hub thread blocked in a chat tool call would be killed mid-call: the
// button is only usable on an idle conversation, and stays busy until the
// awaited server-side restart comes back.
const canRestart = computed(
  () => state.phase === 'idle' && !state.mcp.restarting && !batch.active,
)
const restartTitle = computed(() => {
  if (batch.active) return 'Unavailable while a batch runs'
  if (state.mcp.restarting) return 'Restarting MCP connectors…'
  return 'Restart the MCP hub: reconnect every connector from .nojae/mcp-settings.json'
})
</script>

<template>
  <aside
    v-if="visible"
    class="chat-mcp-panel"
    aria-label="MCP connectors for this conversation"
  >
    <div class="chat-hub-head">
      <p class="caption chat-mcp-title">MCP tools</p>
      <button
        class="icon-btn chat-hub-restart"
        :class="{ 'chat-hub-restart--busy': state.mcp.restarting }"
        type="button"
        :disabled="!canRestart"
        :aria-label="state.mcp.restarting ? 'Restarting MCP hub' : 'Restart MCP hub'"
        :title="restartTitle"
        @click="restartMcp"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
          <path d="M3 3v5h5" />
        </svg>
      </button>
    </div>
    <!-- The one sentence saying why the whole column went inert. The panel
         stays visible: the switches are real and Settings still owns them. -->
    <p v-if="gatedByMode" class="caption chat-hub-gate" role="status">
      {{ gateReason }}
    </p>
    <label
      v-for="connector in state.mcp.connectors"
      :key="connector.id"
      class="chat-mcp-row"
      :class="{ 'chat-mcp-row--dim': gatedByMode || connector.status !== 'running' }"
      :title="rowTitle(connector)"
    >
      <input
        type="checkbox"
        role="switch"
        :checked="state.mcp.enabled[connector.id]"
        :disabled="switchesDisabled"
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
