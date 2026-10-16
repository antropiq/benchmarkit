<script setup>
defineProps({
  label: { type: String, required: true },
  value: { type: String, required: true },
  delta: { type: String, default: null },
})
</script>

<template>
  <div class="metric-card">
    <p class="metric-label">{{ label }}</p>
    <p class="metric-value">{{ value }}</p>
    <p v-if="delta !== null" class="metric-delta">{{ delta }}</p>
  </div>
</template>
