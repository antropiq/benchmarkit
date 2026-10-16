<script setup>
defineProps({
  disabled: { type: Boolean, default: false },
  running: { type: Boolean, default: false },
})
defineEmits(['click'])
</script>

<template>
  <button class="primary" :disabled="disabled" @click="$emit('click')">
    ⚑ {{ running ? 'Running…' : 'Start benchmark' }}
  </button>
</template>
