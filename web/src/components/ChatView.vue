<script setup>
import { computed, ref } from 'vue'
import ChatInput from './ChatInput.vue'
import ChatToolPanel from './ChatToolPanel.vue'
import MarkdownStream from './MarkdownStream.vue'
import { useChat } from '../composables/useChat'
import { useAutoScroll } from '../composables/useAutoScroll'
import { useSplitter } from '../composables/useSplitter'

const { state, editLast } = useChat()

const lastUserIndex = computed(() => {
  for (let i = state.messages.length - 1; i >= 0; i -= 1) {
    if (state.messages[i].role === 'user') return i
  }
  return -1
})

const shell = ref(null)
const input = ref(null)
const panel = ref(null)
const contentVersion = computed(() =>
  state.messages
    .map((m) => `${m.role}:${m.raw.length}`)
    .concat(`draft:${state.draft.raw.length}`)
    .join(','),
)
const { onScroll } = useAutoScroll(panel, contentVersion)
const {
  dragging,
  topStyle,
  bottomStyle,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onKeyDown,
  onDoubleClick,
} = useSplitter(shell, input)
</script>

<template>
  <div ref="shell" class="chat-shell">
    <div class="chat-stream-row" :style="topStyle">
      <ChatToolPanel />
      <div
        ref="panel"
        class="chat-output stream-panel"
        aria-label="Chat conversation"
        @scroll.passive="onScroll"
      >
        <template v-for="(message, i) in state.messages" :key="i">
          <div v-if="message.role === 'user'" class="chat-user-row">
            <button
              v-if="i === lastUserIndex && state.phase === 'idle'"
              class="icon-btn chat-edit-btn"
              type="button"
              aria-label="Edit and resend"
              title="Edit and resend"
              @click="editLast"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
              </svg>
            </button>
            <div class="chat-bubble">
              <MarkdownStream :html="message.html" />
            </div>
          </div>
          <article v-else class="question-card">
            <MarkdownStream :html="message.html" mermaid />
            <p
              v-if="message.streaming && !message.raw"
              class="caption chat-typing"
              role="status"
            >
              {{ 'Thinking…' }}
            </p>
          </article>
        </template>
        <div
          v-if="state.draft.html"
          class="chat-bubble chat-bubble--draft"
          aria-label="Message preview (not sent yet)"
        >
          <MarkdownStream :html="state.draft.html" />
        </div>
        <p v-if="state.error" class="error-line" role="alert">{{ state.error }}</p>
        <div v-if="!state.messages.length && !state.draft.html" class="stream-empty">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <p class="stream-empty-title">Chat with the model</p>
          <p class="caption">
            Ask anything — answers stream from the endpoint configured for the benchmark.
          </p>
        </div>
      </div>
    </div>

    <div
      class="chat-resizer"
      :class="{ dragging }"
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize the message input (drag, arrow keys, double-click to reset)"
      tabindex="0"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
      @keydown="onKeyDown"
      @dblclick="onDoubleClick"
    ></div>

    <ChatInput ref="input" :style="bottomStyle" />
  </div>
</template>
