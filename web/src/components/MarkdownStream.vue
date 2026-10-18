<script setup>
defineProps({
  html: { type: String, default: '' },
})
</script>

<template>
  <div class="answer" v-html="html"></div>
</template>
