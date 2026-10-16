<script setup>
import { computed } from 'vue'

const props = defineProps({
  entry: { type: Object, required: true },
})

const caption = computed(
  () => `${props.entry.number} · ${props.entry.model}`,
)

const footer = computed(
  () => `${props.entry.took.toFixed(1)}s · ${props.entry.tokens.toLocaleString()} tokens`,
)
</script>

<template>
  <div class="question-card">
    <p class="caption mono">Question {{ caption }}</p>
    <p class="question-text">{{ entry.question }}</p>
    <div class="answer" v-html="entry.html"></div>
    <template v-if="!entry.streaming">
      <p style="margin: 0.5rem 0 0.25rem">
        <span v-if="entry.succeeded" class="badge green">✓ Correct</span>
        <span v-else class="badge red">✗ Incorrect</span>
      </p>
      <p class="caption">{{ footer }}</p>
    </template>
  </div>
</template>
