<script setup>
import { computed } from 'vue'
import ChatMcpPanel from './ChatMcpPanel.vue'
import ChatSidePanel from './ChatSidePanel.vue'
import { useChat } from '../composables/useChat'

const { state, send, setDraft } = useChat()

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
    <textarea
      :value="state.draft.raw"
      class="chat-textarea"
      rows="3"
      placeholder="Ask the model something… (Enter to send, Shift+Enter for a new line)"
      :aria-label="streaming ? 'Chat input (reply streaming)' : 'Chat input'"
      @input="setDraft($event.target.value)"
      @keydown="onKeydown"
    ></textarea>
  </div>
</template>
