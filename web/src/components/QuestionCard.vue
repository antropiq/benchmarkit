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

const genRate = computed(() =>
  props.entry.genTook > 0 ? props.entry.tokens / props.entry.genTook : null,
)

const footer = computed(() => {
  const entry = props.entry
  const tokens = `${entry.tokensEstimated ? '~' : ''}${entry.tokens.toLocaleString()} tokens`
  const rate = genRate.value === null ? 'n/a' : `${genRate.value.toFixed(1)} tok/s`
  return `${entry.took.toFixed(1)}s · ${tokens} · ${rate}`
})

const ttftLabel = computed(() =>
  props.entry.ttft > 0 ? `first token in ${props.entry.ttft.toFixed(2)}s` : '',
)
</script>

<template>
  <article class="question-card">
    <p class="caption mono question-meta">{{ caption }}</p>
    <p class="question-text" v-html="entry.questionHtml"></p>
    <div class="answer" v-html="entry.html"></div>
    <template v-if="!entry.streaming">
      <p class="question-verdict">
        <span v-if="entry.succeeded" class="badge green">✓ Correct</span>
        <span v-else class="badge red">✗ Incorrect</span>
      </p>
      <p class="caption">{{ footer }}</p>
      <p v-if="ttftLabel" class="caption">{{ ttftLabel }}</p>
    </template>
  </article>
</template>
