<script setup>
import { computed, ref } from 'vue'
import QuestionCard from './QuestionCard.vue'
import { useBenchmark } from '../composables/useBenchmark'
import { useAutoScroll } from '../composables/useAutoScroll'

const { state } = useBenchmark()

const panel = ref(null)
const contentVersion = computed(() =>
  state.entries.map((e) => (e.type === 'question' ? e.version : e.type)).join(''),
)
const { onScroll } = useAutoScroll(panel, contentVersion)
</script>

<template>
  <div ref="panel" class="stream-panel" @scroll.passive="onScroll">
    <template v-for="(entry, i) in state.entries" :key="i">
      <QuestionCard v-if="entry.type === 'question'" :entry="entry" />
      <p v-else-if="entry.type === 'error'" class="error-line">{{ entry.message }}</p>
      <div v-else-if="entry.type === 'finished'" class="result-banner success">
        ✓ Benchmark finished
      </div>
      <div v-else-if="entry.type === 'cancelled'" class="result-banner warning">
        ⏹ Benchmark cancelled
      </div>
    </template>
    <div v-if="!state.entries.length" class="stream-empty">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M10 8.5v7l6-3.5-6-3.5z" fill="currentColor" stroke="none" />
      </svg>
      <p class="stream-empty-title">Ready to benchmark</p>
      <p class="caption">
        Press play to run all questions against the configured endpoints.
      </p>
    </div>
  </div>
</template>
