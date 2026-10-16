<script setup>
import { nextTick, ref, watch } from 'vue'
import QuestionCard from './QuestionCard.vue'
import { useBenchmark } from '../composables/useBenchmark'

const { state } = useBenchmark()

const panel = ref(null)
const stickToBottom = ref(true)

// Autoscroll while streaming, unless the user scrolled away.
function onScroll() {
  const el = panel.value
  if (!el) return
  stickToBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 48
}

watch(
  () => state.entries.map((e) => (e.type === 'question' ? e.version : e.type)).join(''),
  async () => {
    if (!stickToBottom.value) return
    await nextTick()
    const el = panel.value
    if (el) el.scrollTop = el.scrollHeight
  },
)
</script>

<template>
  <div ref="panel" class="stream-panel" @scroll.passive="onScroll">
    <template v-for="(entry, i) in state.entries" :key="i">
      <div v-if="entry.type === 'model'" class="model-badge-row">
        <span class="badge violet">▣ {{ entry.model }}</span>
      </div>
      <QuestionCard v-else-if="entry.type === 'question'" :entry="entry" />
      <p v-else-if="entry.type === 'error'" class="error-line">{{ entry.message }}</p>
      <div v-else-if="entry.type === 'finished'" class="finished-banner">
        ✓ Benchmark finished
      </div>
    </template>
    <p v-if="!state.entries.length" class="caption">
      Press “Start benchmark” to run all questions against the configured
      endpoints.
    </p>
  </div>
</template>
