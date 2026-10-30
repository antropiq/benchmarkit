<script setup>
import { computed, ref, watch } from 'vue'
import WorkBenchInput from './WorkBenchInput.vue'
import WorkBenchToolPanel from './WorkBenchToolPanel.vue'
import ImageLightbox from './ImageLightbox.vue'
import MarkdownStream from './MarkdownStream.vue'
import { humanFileSize, useChat } from '../composables/useChat.js'
import { useMemory } from '../composables/useMemory.js'
import { useAutoScroll } from '../composables/useAutoScroll.js'
import { useBenchmark } from '../composables/useBenchmark.js'
import { useChatLayout } from '../composables/useChatLayout.js'
import { useBatch } from '../composables/useBatch.js'
import { anyHub, bothHubs } from '../lib/chat-hubs.js'
import WorkBenchToolsDock from './WorkBenchToolsDock.vue'
import WorkBenchFocusGutter from './WorkBenchFocusGutter.vue'

const { state, editLast, clearImage, loadChatHubs } = useChat()
const { state: batch } = useBatch()
const { state: benchmarkState, loadConfig } = useBenchmark()
const { loadMemory } = useMemory()

// Keep config.supportsVision (which gates the attach button) and the
// conversation-scoped MCP snapshot fresh when the chat tab is opened:
// settings may have been saved or .env edited meanwhile (App.vue's initial
// loadConfig covers the first render). A locked (started) conversation
// keeps its frozen list — loadChatMcp ignores locked state. The memory
// bullets refresh the same way (they may have been edited meanwhile).
watch(
  () => benchmarkState.activeTab,
  (tab) => {
    if (tab === 'chat') {
      loadConfig()
      // The hub allowlists *and* the agent-mode list: the dropdown, the panels'
      // gray-out and the payload's agentMode all come from that one fetch.
      loadChatHubs()
      loadMemory()
    }
  },
)

const lastUserIndex = computed(() => {
  for (let i = state.messages.length - 1; i >= 0; i -= 1) {
    if (state.messages[i].role === 'user') return i
  }
  return -1
})

const input = ref(null)
const panel = ref(null)

// Image opened in the full-size lightbox popup (null = closed). Pure view
// state, deliberately outside the useChat singleton.
const preview = ref(null)
const contentVersion = computed(() =>
  state.messages
    .map((m) => `${m.role}:${m.raw.length}:${m.images?.length ?? 0}`)
    .concat(
      `draft:${state.draft.raw.length}`,
      `image:${state.pendingImage ? state.pendingImage.name : ''}`,
    )
    .join(','),
)
const { onScroll } = useAutoScroll(panel, contentVersion)
const leftCol = ref(null)
const rightCol = ref(null)
const dock = ref(null)

// Two splitters, one per column: the right one sizes the composer row (it used
// to span the whole shell and so sized the left column too), the left one
// sizes the tools dock under the reasoning timeline. See useChatLayout.
//
// `focus` / `leftHidden` / the two toggles drive the gutter between the
// columns: the state is the composable's, the template only renders it.
const {
  column: dockSplit,
  composer: composerSplit,
  dockCollapsed,
  focus,
  leftHidden,
  toggleFocus,
  toggleLeftHidden,
} = useChatLayout(leftCol, rightCol, input, dock)
</script>

<template>
  <div class="chat-shell">
    <div
      ref="leftCol"
      class="chat-col chat-col--left"
      :class="{ 'chat-col--single': !bothHubs(state) }"
    >
      <WorkBenchToolPanel />
      <div
        v-if="anyHub(state) && !dockCollapsed"
        class="pane-resizer chat-resizer chat-col-resizer"
        :class="{ dragging: dockSplit.dragging.value }"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize the tool panels (drag, arrow keys, double-click to reset)"
        tabindex="0"
        @pointerdown="dockSplit.onPointerDown"
        @pointermove="dockSplit.onPointerMove"
        @pointerup="dockSplit.onPointerUp"
        @pointercancel="dockSplit.onPointerUp"
        @keydown="dockSplit.onKeyDown"
        @dblclick="dockSplit.onDoubleClick"
      ></div>
      <WorkBenchToolsDock
        ref="dock"
        v-model:collapsed="dockCollapsed"
        :style="
          dockCollapsed || !anyHub(state) ? null : dockSplit.bottomStyle.value
        "
      />
    </div>

    <WorkBenchFocusGutter
      :focus="focus"
      :left-hidden="leftHidden"
      @toggle-focus="toggleFocus"
      @toggle-left-hidden="toggleLeftHidden"
    />
    <div ref="rightCol" class="chat-col chat-col--right">
      <div
        ref="panel"
        class="chat-output stream-panel"
        aria-label="WorkBench conversation"
        @scroll.passive="onScroll"
      >
        <template v-for="(message, i) in state.messages" :key="i">
          <div v-if="message.role === 'user'" class="chat-turn">
            <div class="chat-bubble">
              <span class="chat-turn-chip chat-turn-avatar" aria-hidden="true">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="8" r="4" />
                  <path d="M4 21a8 8 0 0 1 16 0" />
                </svg>
              </span>
              <button
                v-if="i === lastUserIndex && state.phase === 'idle' && !batch.active"
                class="chat-turn-chip chat-turn-edit"
                type="button"
                aria-label="Edit and resend"
                title="Edit and resend"
                @click="editLast"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
                </svg>
              </button>
              <button
                v-for="(image, j) in message.images ?? []"
                :key="j"
                type="button"
                class="chat-thumb-zoom"
                :aria-label="`View ${image.name} at full size`"
                :title="`View ${image.name} at full size`"
                @click="preview = image"
              >
                <img
                  :src="image.dataUrl"
                  :alt="image.name"
                  class="chat-thumb"
                  loading="lazy"
                />
              </button>
              <MarkdownStream v-if="message.html" :html="message.html" />
            </div>
          </div>
          <article v-else class="question-card">
            <MarkdownStream :html="message.html" mermaid />
            <!-- Sticky marker: the transient error line above the composer is
                 cleared by the next send, but the fragment stays in the
                 conversation (and is re-posted to the model on the next turn),
                 so the reply itself has to say it was cut short. -->
            <p v-if="message.truncated" class="caption chat-truncated" role="status">
              &#9888; Cut short by the server — this reply is incomplete.
            </p>
            <!-- The user's own interruption: unlike the line above this is not
                 a fault, and the fragment is deliberately part of the
                 conversation, so it says what happens next. -->
            <p v-if="message.paused" class="caption chat-paused" role="status">
              &#9208; Interrupted by you — the partial reply above is kept in
              the conversation.
            </p>
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
          v-if="state.pendingImage"
          class="chat-attachment chat-attachment--pending"
          aria-label="Attached image (not sent yet)"
        >
          <button
            type="button"
            class="chat-thumb-zoom"
            aria-label="View attached image at full size"
            title="View attached image at full size"
            @click="preview = state.pendingImage"
          >
            <img
              :src="state.pendingImage.dataUrl"
              :alt="state.pendingImage.name"
              class="chat-thumb"
            />
          </button>
          <span class="chat-attachment-meta">
            <span class="chat-attachment-name">{{ state.pendingImage.name }}</span>
            <span class="caption">{{ humanFileSize(state.pendingImage.size) }}</span>
          </span>
          <button
            class="icon-btn chat-attachment-remove"
            type="button"
            aria-label="Remove attached image"
            title="Remove image"
            @click="clearImage"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        </div>
        <div
          v-if="state.draft.html"
          class="chat-bubble chat-bubble--draft"
          aria-label="Message preview (not sent yet)"
        >
          <MarkdownStream :html="state.draft.html" />
        </div>
        <p v-if="state.error" class="error-line" role="alert">{{ state.error }}</p>
        <div
          v-if="!state.messages.length && !state.draft.html && !state.pendingImage"
          class="stream-empty"
        >
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

      <div
        class="pane-resizer chat-resizer"
        :class="{ dragging: composerSplit.dragging.value }"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize the message input (drag, arrow keys, double-click to reset)"
        tabindex="0"
        @pointerdown="composerSplit.onPointerDown"
        @pointermove="composerSplit.onPointerMove"
        @pointerup="composerSplit.onPointerUp"
        @pointercancel="composerSplit.onPointerUp"
        @keydown="composerSplit.onKeyDown"
        @dblclick="composerSplit.onDoubleClick"
      ></div>

      <WorkBenchInput ref="input" :style="composerSplit.bottomStyle.value" />
    </div>

    <ImageLightbox :image="preview" @close="preview = null" />
  </div>
</template>
