<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps({
  entry: { type: Object, required: true },
})

const now = ref(Date.now())
let ticker = null

function stopTicker() {
  if (ticker) {
    clearInterval(ticker)
    ticker = null
  }
}

watch(
  () => props.entry.streaming,
  (streaming) => {
    stopTicker()
    if (streaming) {
      now.value = Date.now()
      ticker = setInterval(() => {
        now.value = Date.now()
      }, 100)
    }
  },
  { immediate: true },
)

onBeforeUnmount(stopTicker)

const caption = computed(() => {
  const base = `Question ${props.entry.number}`
  if (!props.entry.streaming) return base
  const elapsed = Math.max(0, (now.value - props.entry.startedAt) / 1000)
  return `${base} - ${elapsed.toFixed(1)}s`
})

const footer = computed(
  () => `${props.entry.took.toFixed(1)}s · ${props.entry.tokens.toLocaleString()} tokens`,
)
</script>

<template>
  <article class="question-card">
    <p class="caption mono question-meta">{{ caption }}</p>
    <p class="question-text">{{ entry.question }}</p>
    <div class="answer" v-html="entry.html"></div>
    <template v-if="!entry.streaming">
      <p class="question-verdict">
        <span v-if="entry.succeeded" class="badge green">✓ Correct</span>
        <span v-else class="badge red">✗ Incorrect</span>
      </p>
      <p class="caption">{{ footer }}</p>
    </template>
  </article>
</template>
