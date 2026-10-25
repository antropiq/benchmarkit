<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { humanFileSize } from '../composables/useChat'
import { useBatch } from '../composables/useBatch'

// The lot-launch form. A thin view: every field it shows is useBatch state, and
// the count caption is what its blur-time fetch fills (the pattern
// RegistrationMemoryDlg and SettingsDialog both follow — watch + showModal +
// backdrop click + native-close sync).
const {
  state,
  canLaunch,
  countLabel,
  skippedLabel,
  lotLabel,
  validate,
  closeDialog,
  requestLaunch,
} = useBatch()

const dialogEl = ref(null)

watch(
  () => state.open,
  async () => {
    await nextTick()
    const dialog = dialogEl.value
    if (!dialog) return
    if (state.open && !dialog.open) dialog.showModal()
    else if (!state.open && dialog.open) dialog.close()
  },
)

function onBackdropClick(event) {
  if (event.target === dialogEl.value) closeDialog({ reset: true })
}

// Esc: sync the parent and drop the form — Esc must never launch anything.
function onNativeClose() {
  if (state.open) closeDialog({ reset: true })
}

// One hint per precondition the per-task instruction depends on, so a lot does
// not fail on task 1 for a reason the dialog could have stated up front.
const doneFolderMissing = computed(
  () => !!state.listing && state.listing.doneFolder === false,
)
const doneFolderHint =
  'No done folder in this lot directory — the instruction ends by moving each ' +
  'finished task file into <dir>/done, so create it before launching. Batch ' +
  'mode never creates a folder of its own.'

// The absolute folder the listing scanned: the same honesty the app bar applies
// to the working location.
const resolvedDir = computed(() => {
  const listing = state.listing
  if (!listing?.workspace) return ''
  return `${listing.workspace}/${listing.dir}`
})

const lotSize = computed(() => {
  const lot = state.listing?.lot
  if (!lot) return 'no big picture found'
  return `${humanFileSize(lot.size)} — re-sent on every task turn`
})

// E13: only the 00 file matched, so there is nothing to run.
const onlyLotFile = computed(
  () => !!state.listing && state.listing.taskCount === 0 && !!state.listing.lot,
)
const countHint = computed(() =>
  onlyLotFile.value ? ' — only the big-picture file matches' : '',
)

// Enter in a text field would submit the form (= Launch) before the listing it
// depends on has arrived, so the fields own Enter: it only re-lists.
function onFieldEnter(event) {
  event.preventDefault()
  validate()
}
</script>

<template>
  <dialog
    ref="dialogEl"
    class="batch-dialog"
    aria-labelledby="batch-dlg-title"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <article>
      <header>
        <h2 id="batch-dlg-title">Batch mode</h2>
      </header>
      <p class="caption">
        Runs a numbered lot of task files through WorkBench, one task per chat
        turn. The model reads each task file itself; the conversation context is
        reset between tasks, so what carries over is what is on disk. Choose the
        MCP and LSP connectors before launching — they freeze with the first
        task and cannot change mid-lot.
      </p>

      <form @submit.prevent="requestLaunch">
        <label>
          Task file prefix
          <input
            v-model="state.prefix"
            type="text"
            spellcheck="false"
            placeholder="ed-"
            :disabled="state.active"
            autocomplete="off"
            aria-describedby="batch-count"
            @blur="validate"
            @keydown.enter="onFieldEnter"
          />
        </label>
        <label>
          Lot folder (relative to the workspace)
          <input
            v-model="state.dir"
            type="text"
            spellcheck="false"
            placeholder="todos"
            :disabled="state.loading || state.active"
            autocomplete="off"
            @blur="validate"
            @keydown.enter="onFieldEnter"
          />
        </label>

        <p id="batch-count" class="caption" role="status" aria-live="polite">
          {{ countLabel }}{{ countHint }}
          <template v-if="state.loading"> · listing…</template>
        </p>
        <p v-if="skippedLabel" class="caption">{{ skippedLabel }}</p>
        <p v-if="state.listing" class="caption">{{ lotLabel }}</p>
        <p v-if="resolvedDir" class="caption">{{ resolvedDir }}</p>
        <p v-if="state.listing?.lotNote" class="caption">{{ state.listing.lotNote }}</p>
        <p v-if="doneFolderMissing" class="caption">{{ doneFolderHint }}</p>

        <label class="checkbox">
          <input v-model="state.includeLot" type="checkbox" :disabled="state.active" />
          Include the big picture ({{ lotSize }})
        </label>
        <p v-if="state.listing?.truncated" class="caption">
          More tasks than the app lists — the lot was truncated to
          {{ state.listing.tasks.length }} entries.
        </p>

        <p v-if="state.error" class="alert-error" role="alert">{{ state.error }}</p>

        <footer class="settings-actions">
          <button
            class="secondary contrast"
            type="button"
            :disabled="state.active"
            @click="closeDialog({ reset: true })"
          >
            Cancel
          </button>
          <button
            type="submit"
            :disabled="!canLaunch"
            :title="
              state.listing && state.listing.taskCount === 0
                ? 'No task matches this prefix'
                : 'Run every task of this lot, one turn each'
            "
          >
            Launch
          </button>
        </footer>
      </form>
    </article>
  </dialog>
</template>
