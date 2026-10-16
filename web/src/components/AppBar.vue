<script setup>
import { computed } from 'vue'
import RunControls from './RunControls.vue'
import ThemeToggle from './ThemeToggle.vue'
import { useBenchmark } from '../composables/useBenchmark'

defineProps({
  canStart: { type: Boolean, default: false },
})
defineEmits(['start', 'stop'])

const { state, openSettings } = useBenchmark()

const models = computed(() => (state.config?.models ?? []).join(' · '))
const running = computed(() => state.status === 'running')
const canStop = computed(() =>
  ['running', 'finished', 'failed'].includes(state.status),
)
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
      </span>
    </div>

    <div class="appbar-actions">
      <span class="status-pill" :class="`status-${state.status}`" role="status">
        <span class="status-dot" aria-hidden="true"></span>{{ state.status }}
      </span>
      <RunControls
        :can-start="canStart"
        :running="running"
        :can-stop="canStop"
        :stopping="state.stopping"
        @start="$emit('start')"
        @stop="$emit('stop')"
      />
      <button
        class="icon-btn"
        type="button"
        :disabled="running"
        :aria-label="running ? 'Settings (available when idle)' : 'Open settings'"
        :title="running ? 'Available when idle' : 'Settings'"
        @click="openSettings"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </button>
      <ThemeToggle />
    </div>
  </header>
</template>
