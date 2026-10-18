<script setup>
import { computed, ref } from 'vue'
import { useChat } from '../composables/useChat'

const { state, send, stop } = useChat()

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
    <textarea
      v-model="text"
      class="chat-textarea"
      rows="3"
      placeholder="Ask the model something… (Enter to send, Shift+Enter for a new line)"
      :aria-label="streaming ? 'Chat input (reply streaming)' : 'Chat input'"
      @keydown="onKeydown"
    ></textarea>
    <button
      v-if="!streaming"
      class="icon-btn chat-send"
      type="button"
      :disabled="!canSend"
      aria-label="Send message"
      title="Send"
      @click="submit"
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M22 2 11 13" />
        <path d="M22 2 15 22l-4-9-9-4 20-7z" />
      </svg>
    </button>
    <button
      v-else
      class="icon-btn chat-send chat-stop"
      type="button"
      aria-label="Stop generating"
      title="Stop"
      @click="stop"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="5" y="5" width="14" height="14" rx="2" />
      </svg>
    </button>
  </div>
</template>
