<script setup>
import { computed, onUnmounted, ref, watch } from 'vue'
import TabNav from './TabNav.vue'
import ThemeToggle from './ThemeToggle.vue'
import SoundToggle from './SoundToggle.vue'
import SessionClock from './SessionClock.vue'
import SessionStatBadges from './SessionStatBadges.vue'
import { useBenchmark } from '../composables/useBenchmark'
import { useChat } from '../composables/useChat'
import { useBatch } from '../composables/useBatch'

const { state, openSettings, setPathLocation } = useBenchmark()
const { state: chat } = useChat()
const { state: batch } = useBatch()

const models = computed(() => (state.config?.models ?? []).join(' · '))
const serverUrl = computed(() => state.config?.serverUrl || '')
const running = computed(() => state.status === 'running')

// One pill for both screens: a running benchmark wins; otherwise a chat
// reply in flight reports its own phase (including the reasoning flag).
const chatBusy = computed(() =>
  ['waiting', 'reasoning', 'answering'].includes(chat.phase),
)

// A running batch reads `Batch`, ahead of the chat branch. A benchmark run and
// a batch are mutually exclusive anyway (POST /api/chat 409s while a run is in
// progress, and POST /api/run now refuses while a reply streams), so the pill
// can never disagree with the server. Once the lot is over its own outcome
// keeps the pill for a couple of seconds — long enough to see which of
// Stop / finish / fail happened — then the chat/run state takes over again.
const BATCH_PILL_HOLD_MS = 2000
const BATCH_OUTCOME_CLASS = {
  stopped: 'status-cancelled',
  failed: 'status-failed',
  finished: 'status-finished',
}

// Transient, so it lives here rather than in useBatch: `batch.status` stays on
// screen (the progress bar reads it), but the pill must not stay coloured.
const batchOutcome = ref(null)
let batchPillTimer = null

function clearBatchPillTimer() {
  if (batchPillTimer) clearTimeout(batchPillTimer)
  batchPillTimer = null
}

watch(
  () => [batch.active, batch.status],
  ([active, status]) => {
    clearBatchPillTimer()
    batchOutcome.value = active || status === 'idle' ? null : status
    if (batchOutcome.value) {
      batchPillTimer = setTimeout(() => {
        batchOutcome.value = null
        batchPillTimer = null
      }, BATCH_PILL_HOLD_MS)
    }
  },
)
onUnmounted(clearBatchPillTimer)

// -- the working location, edited in place ---------------------------------
//
// ``state.pathLocation`` is what the user has typed, ``state.pathLocationCurrent``
// what the server applied (its realpath), so Set reads the same difference the
// field is about: a canonicalised path never leaves it spuriously enabled.
const pathLocationChanged = computed(
  () => (state.pathLocation || '').trim() !== state.pathLocationCurrent,
)

// A session must not have its root moved out from under it: the route refuses
// the switch with 409 while a benchmark run or a chat reply is in flight, and
// a batch would find its lot folder re-rooted mid-task. ``chatBusy`` is the
// WorkBench's own rule (it also covers the idle gap between two batch turns),
// so the button cannot disagree with a send being refused. The pill reads the
// same three states, so a locked Set always explains itself.
const locationLocked = computed(
  () => running.value || batch.active || chatBusy.value,
)
const locationTitle = computed(() =>
  locationLocked.value
    ? 'Unavailable while a session is running'
    : 'Point the MCP and LSP tools, the Editor and the terminal at this folder',
)

// What the switch answered that is worth a line under the field: the refusal
// (red), or the reserved-folder seeds that could not land. The latter is a
// warning and never a refusal — the location *did* move and both hubs
// restarted. The three seeds fail independently but are the outcome of one
// action, so they join into one sentence.
const locationNote = computed(() =>
  [
    state.pathLocationPolicyNote,
    state.pathLocationAgentModesNote,
    state.pathLocationPromptHelpersNote,
  ]
    .filter(Boolean)
    .join(' '),
)

// One gesture, two entries: the button and Enter in the field. The lock is
// re-checked here rather than left to the disabled button, because a key
// reaches the request without ever meeting the button.
function applyLocation() {
  if (locationLocked.value) return
  setPathLocation()
}

const pillLabel = computed(() => {
  if (!running.value && batch.active) return 'Batch'
  if (!running.value && batchOutcome.value) return `batch ${batchOutcome.value}`
  if (!running.value && chatBusy.value) {
    return chat.phase === 'reasoning' ? 'reasoning...' : 'thinking'
  }
  return state.status
})
const pillClass = computed(() => {
  if (!running.value && batch.active) return 'status-batch'
  if (!running.value && batchOutcome.value) {
    return BATCH_OUTCOME_CLASS[batchOutcome.value] ?? 'status-idle'
  }
  if (!running.value && chatBusy.value) {
    return chat.phase === 'reasoning' ? 'status-reasoning' : 'status-thinking'
  }
  return `status-${state.status}`
})
</script>

<template>
  <header class="appbar">
    <div class="appbar-brand">
      <span class="brand-mark" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 12h4l3 8 4-16 3 8h4" />
        </svg>
      </span>
      <span class="brand-text">
        <span class="brand-name">Benchmark'it</span>
        <span v-if="models" class="brand-sub">{{ models }}</span>
        <span v-if="serverUrl" class="brand-sub">on {{ serverUrl }}</span>
      </span>
    </div>

    <div class="appbar-center">
      <!-- The two session counters, left of the tab strip: they read whichever
           session is live (a run or a WorkBench reply) and blank out when
           nothing is generating. -->
      <SessionStatBadges />
      <TabNav />
      <div class="appbar-location">
        <div class="appbar-location-row">
          <input
            v-model="state.pathLocation"
            class="appbar-location-input"
            type="text"
            spellcheck="false"
            aria-label="Working location"
            :disabled="state.pathLocationLoading"
            :title="state.pathLocationCurrent"
            @keydown.enter.prevent="applyLocation"
          />
          <button
            class="appbar-location-set"
            type="button"
            :disabled="
              !pathLocationChanged || state.pathLocationSaving || locationLocked
            "
            :title="locationTitle"
            @click="applyLocation"
          >
            {{ state.pathLocationSaving ? 'Reloading…' : 'Set' }}
          </button>
        </div>
        <!-- One line, whichever half of the answer it is: the refusal, or the
             seeds that could not land. The bar has no room for a wrapped
             sentence, so it truncates and `title` carries the whole of it. -->
        <p
          v-if="state.pathLocationError"
          class="appbar-location-note appbar-location-note--error"
          role="alert"
          :title="state.pathLocationError"
        >
          {{ state.pathLocationError }}
        </p>
        <p
          v-else-if="locationNote"
          class="appbar-location-note"
          role="status"
          :title="locationNote"
        >
          {{ locationNote }}
        </p>
      </div>
    </div>

    <div class="appbar-actions">
      <!-- One clock for the whole app: a Benchmark run and a WorkBench reply
           are mutually exclusive, so the same readout serves both, and it
           keeps ticking while the Editor tab is open. -->
      <SessionClock />
      <span class="status-pill" :class="pillClass" role="status">
        <span class="status-dot" aria-hidden="true"></span>{{ pillLabel }}
      </span>
      <button
        class="icon-btn"
        type="button"
        :disabled="running || batch.active"
        :aria-label="running ? 'Settings (available when idle)' : 'Open settings'"
        :title="
          running || batch.active ? 'Available when idle' : 'Settings'
        "
        @click="openSettings"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </button>
      <SoundToggle />
      <ThemeToggle />
    </div>
  </header>
</template>
