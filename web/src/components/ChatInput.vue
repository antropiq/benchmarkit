<script setup>
import { computed, ref } from 'vue'
import ChatSidePanel from './ChatSidePanel.vue'
import { useChat } from '../composables/useChat'

const { state, send } = useChat()

const text = ref('')

const streaming = computed(() => state.phase !== 'idle')
const canSend = computed(() => !streaming.value && text.value.trim().length > 0)

function submit() {
  if (!canSend.value) return
  send(text.value)
  text.value = ''
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
    <textarea
      v-model="text"
      class="chat-textarea"
      rows="3"
      placeholder="Ask the model something… (Enter to send, Shift+Enter for a new line)"
      :aria-label="streaming ? 'Chat input (reply streaming)' : 'Chat input'"
      @keydown="onKeydown"
    ></textarea>
  </div>
</template>
