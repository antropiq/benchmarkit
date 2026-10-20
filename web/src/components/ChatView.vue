<script setup>
import { computed, ref } from 'vue'
import ChatInput from './ChatInput.vue'
import ChatToolPanel from './ChatToolPanel.vue'
import MarkdownStream from './MarkdownStream.vue'
import { useChat } from '../composables/useChat'
import { useAutoScroll } from '../composables/useAutoScroll'
import { useSplitter } from '../composables/useSplitter'

const { state } = useChat()

const shell = ref(null)
const input = ref(null)
const panel = ref(null)
const contentVersion = computed(() =>
  state.messages.map((m) => `${m.role}:${m.raw.length}`).join(','),
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
          <div v-if="message.role === 'user'" class="chat-bubble">
            <MarkdownStream :html="message.html" />
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
        <p v-if="state.error" class="error-line" role="alert">{{ state.error }}</p>
        <div v-if="!state.messages.length" class="stream-empty">
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
