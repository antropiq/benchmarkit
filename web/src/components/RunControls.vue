<script setup>
defineProps({
  canStart: { type: Boolean, default: false },
  running: { type: Boolean, default: false },
  canStop: { type: Boolean, default: false },
  stopping: { type: Boolean, default: false },
})
defineEmits(['start', 'stop'])
</script>

<template>
  <div class="run-controls">
    <button
      class="run-btn run-btn-play"
      :class="{ 'is-running': running }"
      type="button"
      style="color: #ffffff"
      :disabled="!canStart"
      :aria-label="running ? 'Benchmark running' : 'Run benchmark'"
      :title="running ? 'Running…' : 'Run benchmark'"
      @click="$emit('start')"
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M8 5.14v13.72a1 1 0 0 0 1.52.86l11.14-6.86a1 1 0 0 0 0-1.72L9.52 4.28A1 1 0 0 0 8 5.14z" />
      </svg>
    </button>
    <button
      class="run-btn run-btn-stop"
      type="button"
      style="color: var(--bi-danger)"
      :disabled="!canStop || stopping"
      :aria-label="running ? (stopping ? 'Stopping…' : 'Stop run') : 'Reset to initial state'"
      :title="running ? (stopping ? 'Stopping…' : 'Stop run') : 'Reset to initial state'"
      @click="$emit('stop')"
    >
      <svg
        v-if="running"
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="currentColor"
        aria-hidden="true"
      >
        <rect x="5" y="5" width="14" height="14" rx="2.5" />
      </svg>
      <svg
        v-else
        width="15"
        height="15"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
        <path d="M3 3v5h5" />
      </svg>
    </button>
  </div>
</template>
