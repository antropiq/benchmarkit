<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import { hydrateMermaid } from '../lib/mermaid'

const props = defineProps({
  html: { type: String, default: '' },
  mermaid: { type: Boolean, default: false },
})

const root = ref(null)
let renderToken = 0

function hydrate() {
  if (!props.mermaid || !root.value) return
  renderToken += 1
  const token = renderToken
  void hydrateMermaid(root.value, { isValid: () => token === renderToken })
}

onMounted(hydrate)

watch(
  () => props.html,
  async () => {
    await nextTick()
    hydrate()
  },
)
</script>

<template>
  <div ref="root" class="answer" v-html="html"></div>
</template>
