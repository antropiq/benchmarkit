<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
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
    <ChatSidePanel
      :can-send="canSend"
      :streaming="streaming"
      @send="submit"
    />
    <ChatMcpPanel />
    <div class="chat-textarea-col">
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
      <MemoryBullets @configure="openBulletDialog" />
    </div>
    <RegistrationMemoryDlg
      :open="dialogOpen"
      :index="dialogIndex"
      @close="dialogOpen = false"
    />
  </div>
</template>
