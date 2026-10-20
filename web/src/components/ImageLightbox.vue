<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  image: { type: Object, default: null },
})
const emit = defineEmits(['close'])

const dialogEl = ref(null)

// Mirror SettingsDialog's native <dialog> pattern: the component stays
// mounted, the parent's `image` prop drives showModal()/close(). flush post
// so the <img> src is already rendered when the dialog opens (correct size).
watch(
  () => props.image,
  (image) => {
    const dialog = dialogEl.value
    if (!dialog) return
    if (image && !dialog.open) {
      dialog.showModal()
    } else if (!image && dialog.open) {
      dialog.close()
    }
  },
  { flush: 'post' },
)

function close() {
  const dialog = dialogEl.value
  if (dialog && dialog.open) dialog.close()
}

function onBackdropClick(event) {
  if (event.target === dialogEl.value) close()
}

function onNativeClose() {
  emit('close')
}
</script>

<template>
  <dialog
    ref="dialogEl"
    class="image-lightbox"
    :aria-label="image ? `Full-size preview of ${image.name}` : 'Image preview'"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <img
      v-if="image"
      :src="image.dataUrl"
      :alt="image.name"
    />
    <button
      class="icon-btn image-lightbox-close"
      type="button"
      aria-label="Close preview"
      title="Close preview"
      @click="close"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M18 6 6 18" />
        <path d="m6 6 12 12" />
      </svg>
    </button>
  </dialog>
</template>
