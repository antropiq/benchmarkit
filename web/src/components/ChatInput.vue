<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import ChatLspPanel from './ChatLspPanel.vue'
import ChatMcpPanel from './ChatMcpPanel.vue'
import ChatSidePanel from './ChatSidePanel.vue'
import MemoryBullets from './MemoryBullets.vue'
import RegistrationMemoryDlg from './RegistrationMemoryDlg.vue'
import { useChat } from '../composables/useChat'

const { state, send, setDraft, registerDraftInput } = useChat()

// Let draft insertions (memory bullets) focus the textarea after appending.
const draftEl = ref(null)
onMounted(() => registerDraftInput(draftEl.value))
onBeforeUnmount(() => registerDraftInput(null))

// Registration dialog for the clicked bullet slot (null index = closed).
const dialogOpen = ref(false)
const dialogIndex = ref(null)

function openBulletDialog(index) {
  dialogIndex.value = index
  dialogOpen.value = true
}

// The left toggle column (MCP beside LSP) only takes horizontal space when
// at least one of the two panels has something to show - mirrors the
// per-panel `visible` conditions inside ChatMcpPanel/ChatLspPanel.
const mcpVisible = computed(
  () => state.mcp.loaded && state.mcp.connectors.length > 0,
)
const lspVisible = computed(
  () => state.lsp.loaded && state.lsp.servers.length > 0,
)
const showToggleCol = computed(() => mcpVisible.value || lspVisible.value)
// With only one of the two panels rendered the column shrinks to a single
// panel — ChatToolPanel mirrors this so the two rows stay aligned.
const singlePanel = computed(() => mcpVisible.value !== lspVisible.value)

const streaming = computed(() => state.phase !== 'idle')
const canSend = computed(
  () =>
    !streaming.value &&
    (state.draft.raw.trim().length > 0 || state.pendingImage !== null),
)

function submit() {
  if (!canSend.value) return
  send()
}

function onKeydown(event) {
  // Enter sends; Shift+Enter inserts a carriage return.
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    submit()
  }
}
</script>

<template>
  <div class="chat-input">
    <div
      v-if="showToggleCol"
      class="chat-toggle-col"
      :class="{ 'chat-toggle-col--single': singlePanel }"
    >
      <ChatMcpPanel />
      <ChatLspPanel />
    </div>
    <div class="chat-textarea-col">
      <MemoryBullets @configure="openBulletDialog" />
      <textarea
        ref="draftEl"
        :value="state.draft.raw"
        class="chat-textarea"
        rows="3"
        placeholder="Ask the model something… (Enter to send, Shift+Enter for a new line)"
        :aria-label="streaming ? 'Chat input (reply streaming)' : 'Chat input'"
        @input="setDraft($event.target.value)"
        @keydown="onKeydown"
      ></textarea>
    </div>
    <RegistrationMemoryDlg
      :open="dialogOpen"
      :index="dialogIndex"
      @close="dialogOpen = false"
    />
    <ChatSidePanel
      :can-send="canSend"
      :streaming="streaming"
      @send="submit"
    />
  </div>
</template>
