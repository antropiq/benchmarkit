<script setup>
import { computed } from 'vue'

// Reusable horizontal meter: a fixed green → yellow → orange → red gradient
// fills from the left, so the color at the fill tip reflects the level.
// `value`/`max` render empty (and the caption a dash) when max is unknown.
const props = defineProps({
  value: { type: Number, default: 0 },
  max: { type: Number, default: 0 },
  showValue: { type: Boolean, default: true },
  label: { type: String, default: 'Gauge' },
})

const pct = computed(() => {
  if (!props.max || props.max <= 0) return 0
  return Math.min(Math.max((props.value / props.max) * 100, 0), 100)
})

const caption = computed(() =>
  props.max > 0 ? `${props.value}/${props.max}` : '—',
)
</script>

<template>
  <div
    class="gauge"
    role="meter"
    :aria-label="label"
    :title="label"
    aria-valuemin="0"
    :aria-valuenow="value"
    :aria-valuemax="max > 0 ? max : null"
  >
    <div class="gauge-track">
      <div class="gauge-mask" :style="{ left: `${pct}%` }"></div>
    </div>
    <span v-if="showValue" class="gauge-caption" aria-hidden="true">
      {{ caption }}
    </span>
  </div>
</template>
