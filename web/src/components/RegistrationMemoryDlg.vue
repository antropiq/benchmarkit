<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useChat } from '../composables/useChat'
import { PROMPT_HELPER_DEFAULT_COLOR, usePromptHelpers } from '../composables/usePromptHelpers'

const props = defineProps({
  open: { type: Boolean, default: false },
  index: { type: Number, default: null },
})
const emit = defineEmits(['close'])

const { state, savePromptHelper } = usePromptHelpers()
const { appendDraft } = useChat()

const dialogEl = ref(null)
const content = ref('')
const color = ref(PROMPT_HELPER_DEFAULT_COLOR)
const error = ref('')

const busy = computed(() => state.saving)

// Prefill from the slot and show, following the SettingsDialog pattern
// (watch + showModal, backdrop click, native-close sync).
watch(
  () => [props.open, props.index, state.slots],
  async () => {
    await nextTick()
    const dialog = dialogEl.value
    if (!dialog) return
    if (props.open && props.index !== null && !dialog.open) {
      const slot = state.slots[props.index]
      content.value = slot?.content ?? ''
      color.value = slot?.color ?? PROMPT_HELPER_DEFAULT_COLOR
      error.value = ''
      dialog.showModal()
    } else if (!props.open && dialog.open) {
      dialog.close()
    }
  },
)

function onBackdropClick(event) {
  if (event.target === dialogEl.value) emit('close')
}

// Esc (or a programmatic close): sync the parent so Cancel/Esc discards and
// never inserts anything.
function onNativeClose() {
  if (props.open) emit('close')
}

async function save() {
  if (props.index === null || state.saving) return
  error.value = ''
  const saved = content.value
  const ok = await savePromptHelper(props.index, color.value, saved)
  if (!ok) {
    error.value = state.error || 'Could not save the memory bullet.'
    return
  }
  dialogEl.value?.close()
  // Saving empty content only clears the slot — nothing goes into the draft.
  if (saved !== '') appendDraft(saved)
}
</script>

<template>
  <dialog
    ref="dialogEl"
    class="memory-dialog"
    aria-labelledby="memory-dlg-title"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <article>
      <header>
        <h2 id="memory-dlg-title">Memory bullet {{ (props.index ?? 0) + 1 }}</h2>
      </header>
      <p class="caption">
        This message is appended to the chat draft when the bullet is clicked.
        Saving empty content clears the bullet.
      </p>

      <p v-if="error" class="alert-error" role="alert">{{ error }}</p>

      <form @submit.prevent="save">
        <label>
          Message
          <textarea
            v-model="content"
            rows="12"
            placeholder="Template message (markdown)…"
          ></textarea>
        </label>
        <label class="memory-color">
          Color
          <input v-model="color" type="color" />
        </label>
        <footer class="settings-actions">
          <button
            class="secondary contrast"
            type="button"
            :disabled="busy"
            @click="emit('close')"
          >
            Cancel
          </button>
          <button type="submit" :disabled="busy">
            {{ busy ? 'Saving…' : 'Save' }}
          </button>
        </footer>
      </form>
    </article>
  </dialog>
</template>
