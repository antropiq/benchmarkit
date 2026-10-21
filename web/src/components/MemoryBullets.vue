<script setup>
import { useChat } from '../composables/useChat'
import { MEMORY_DEFAULT_COLOR, MEMORY_SLOT_COUNT, useMemory } from '../composables/useMemory'

const emit = defineEmits(['configure'])

const { state: memory } = useMemory()
const { appendDraft } = useChat()

// Tooltip preview cap (the full text stays on disk).
const TOOLTIP_MAX_CHARS = 150

function slot(index) {
  return memory.registration[index] ?? { color: MEMORY_DEFAULT_COLOR, content: '' }
}

// An empty slot always renders default gray; the stored color is kept but
// ignored until content comes back.
function isFilled(data) {
  return data.content !== ''
}

// Black or white number by background luminance (ITU-R BT.601 luma).
function textColor(color) {
  const r = parseInt(color.slice(1, 3), 16)
  const g = parseInt(color.slice(3, 5), 16)
  const b = parseInt(color.slice(5, 7), 16)
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) {
    return '#ffffff'
  }
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? '#000000' : '#ffffff'
}

function bulletStyle(data) {
  const background = isFilled(data) ? data.color : MEMORY_DEFAULT_COLOR
  return { background, color: textColor(background) }
}

function tooltip(data) {
  if (!isFilled(data)) return undefined
  return data.content.length > TOOLTIP_MAX_CHARS
    ? `${data.content.slice(0, TOOLTIP_MAX_CHARS)}…`
    : data.content
}

// Filled bullet: click appends its text to the draft, Ctrl/Cmd+click (or a
// click on an empty slot) opens the registration dialog.
function onClick(event, index) {
  const data = slot(index)
  if (isFilled(data) && !event.ctrlKey && !event.metaKey) {
    appendDraft(data.content)
    return
  }
  emit('configure', index)
}
</script>

<template>
  <div class="memory-bullets" role="group" aria-label="Memory bullets">
    <button
      v-for="n in MEMORY_SLOT_COUNT"
      :key="n"
      type="button"
      class="memory-bullet"
      :style="bulletStyle(slot(n - 1))"
      :title="tooltip(slot(n - 1))"
      :aria-label="
        isFilled(slot(n - 1))
          ? `Memory bullet ${n} (edit with Ctrl+click)`
          : `Memory bullet ${n} (empty — configure)`
      "
      @click="onClick($event, n - 1)"
    >
      {{ n }}
    </button>
  </div>
</template>
