<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  clock: {
    type: Object,
    required: true, // { mode: idle|ticking|checking|frozen, baseMs, startedAt }
  },
})

const now = ref(Date.now())
let timer = null

onMounted(() => {
  timer = setInterval(() => {
    now.value = Date.now()
  }, 250)
})

onBeforeUnmount(() => clearInterval(timer))

function format(ms) {
  const total = Math.max(0, Math.floor(ms / 1000))
  const parts = [Math.floor(total / 3600), Math.floor((total % 3600) / 60), total % 60]
  return parts.map((n) => String(n).padStart(2, '0')).join(':')
}

const display = computed(() => {
  const { mode, baseMs, startedAt } = props.clock
  if (mode === 'checking') return 'Checking…'
  if (mode === 'idle') return '00:00:00'
  const elapsed = mode === 'ticking' ? baseMs + (now.value - startedAt) : baseMs
  return format(elapsed)
})
</script>

<template>
  <p class="chronometer">{{ display }}</p>
</template>
