<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { TRASH_FOLDER, TRASH_RETENTION_DAYS, useEditorFiles } from '../composables/useEditorFiles'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'

// The Editor tab's dialogs, one file for all of them: the save conflict, the
// close confirm for a dirty buffer, and the delete confirm. They share one file
// rather than three because they are the same shape — a
// `<dialog>` + `showModal()` over composable state, the pattern
// RegistrationMemoryDlg.vue and BatchModeDialog.vue follow — and because an
// answer that belongs to the pane should not be spread over whichever component
// happened to trigger it.
//
// Nothing here decides what a conflict *is*: that is the probe's table in
// useEditorFiles; equally, nothing here decides whether closing a tab needs
// asking — `requestCloseTab` raised `state.closePrompt` only for a buffer that
// holds unsaved edits. This component names which copy or which file is at
// stake and offers the answers, all reachable by keyboard (they are plain
// buttons inside the modal, and `showModal()` puts the caret on the first one).
const {
  state,
  reloadFile,
  saveFile,
  openConflictedDiff,
  dismissConflict,
  cancelClose,
  discardClose,
  saveAndClose,
  cancelDelete,
  confirmDelete,
} = useEditorFiles()

// Only the *cached listing* is read from the tree, for the lower-bound line
// below — this component never asks the tree for anything and never walks it.
const tree = useWorkspaceTree()

const dialogEl = ref(null)
const busy = ref(false)

const conflict = computed(() => state.conflict)
// Dismissing is not answering: the record keeps its conflict, its marker and
// its edits, and only the modal stops. `conflictDismissedAt` (composable state,
// so it survives this component) is what keeps *this* news from returning as a
// modal on the next probe tick, while a genuinely new one still shows.
const isOpenNews = computed(
  () => !!conflict.value && conflict.value.raisedAt !== state.conflictDismissedAt,
)
const record = computed(() => {
  const current = conflict.value
  if (!current) return null
  return state.open.find((item) => item.key === current.key) ?? null
})

const isDeleted = computed(() => conflict.value?.kind === 'deleted')
const title = computed(() =>
  isDeleted.value ? 'The file was deleted on disk' : 'The file changed on disk',
)

// Which copy is at stake, and never both as "the truth": for a stale buffer the
// disk moved, for a deleted one the buffer is the only copy left.
const body = computed(() =>
  isDeleted.value
    ? 'Something outside this pane removed the file while your edits were ' +
      'unsaved, so this buffer holds the only copy of them left. Saving cannot ' +
      'bring the path back — a save never creates a file — so keep the tab open ' +
      'until the file exists again.'
    : 'Something outside this pane wrote the file while your edits were unsaved. ' +
      'One of the two copies is about to be lost, so choose which: take the ' +
      'version on disk, or keep your edits.',
)

// The disk's timestamp, when there is one to name — shown as a time, because
// the point is *when* the file moved, not the float itself.
const diskWhen = computed(() => {
  const at = conflict.value?.diskMtime
  return typeof at === 'number' ? new Date(at * 1000).toLocaleString() : ''
})

watch(
  () => [conflict.value?.key, conflict.value?.raisedAt],
  async () => {
    await nextTick()
    const dialog = dialogEl.value
    if (!dialog) return
    if (isOpenNews.value && !dialog.open) dialog.showModal()
    else if (!isOpenNews.value && dialog.open) dialog.close()
  },
)

function onBackdropClick(event) {
  if (event.target === dialogEl.value) dismiss()
}

// Esc: dismiss, answer nothing. The buffer keeps its edits and its marker —
// nothing here gets to discard either one.
function onNativeClose() {
  if (conflict.value) dismiss()
}

function dismiss() {
  dismissConflict()
  dialogEl.value?.close()
}

async function reload() {
  if (!conflict.value || busy.value) return
  busy.value = true
  await reloadFile(conflict.value.key)
  busy.value = false
  dialogEl.value?.close()
}

// Compare gets out of the way of what it shows: closing the dialog is what
// reveals the two sides. The answer set is one click on the chip away again.
async function compare() {
  if (!conflict.value || busy.value) return
  busy.value = true
  await openConflictedDiff(conflict.value.key)
  busy.value = false
  dismissConflict()
  dialogEl.value?.close()
}

// The escape hatch: save without the mtime token, past the guard that refused
// it. The least prominent button, and the only caller allowed to omit the
// token — the server logs a token-less save for exactly this reason.
async function overwrite() {
  if (!conflict.value || busy.value || isDeleted.value) return
  busy.value = true
  const result = await saveFile(conflict.value.key, { force: true })
  busy.value = false
  if (result.ok) dialogEl.value?.close()
}

// Nothing is exposed: re-opening after a dismissal goes through
// `raiseConflict` in the composable (the chip's marker calls it), which
// re-stamps the conflict and clears the dismissal, so the watch above sees news
// again without this component holding the only copy of that state.

// The close confirm. It exists only because a dirty buffer's edits live in this
// page and nowhere else: closing the tab drops them, so one gesture is never
// allowed to be the whole decision. `state.closePrompt` is raised by
// `requestCloseTab` for a dirty buffer alone, so the mere presence of this
// state means "there is work to lose".
const closeDialogEl = ref(null)
const closePrompt = computed(() => state.closePrompt)

const closeTitle = computed(() =>
  closePrompt.value ? `Close ${closePrompt.value.name}?` : 'Close this tab?',
)

watch(
  () => closePrompt.value?.key,
  async (key) => {
    await nextTick()
    const dialog = closeDialogEl.value
    if (!dialog) return
    if (key && !dialog.open) dialog.showModal()
    else if (!key && dialog.open) dialog.close()
  },
)

function onCloseBackdropClick(event) {
  if (event.target === closeDialogEl.value) cancelViaDialog()
}

// Esc answers Cancel, never Discard — the keyboard default of a confirm must not
// be the destructive one.
function onCloseNativeClose() {
  if (closePrompt.value) cancelViaDialog()
}

function cancelViaDialog() {
  cancelClose()
  closeDialogEl.value?.close()
}

async function saveAndCloseViaDialog() {
  if (busy.value) return
  busy.value = true
  await saveAndClose()
  busy.value = false
  closeDialogEl.value?.close()
}

function discardViaDialog() {
  discardClose()
  closeDialogEl.value?.close()
}

// The delete confirm. `state.deletePrompt` is raised by `requestDelete` for
// every path the tree or the `Delete` key asks about — a delete is destructive
// whatever the buffers hold, so one gesture is never the whole decision. The
// server has the rules (the gate, the subtree check, the trash); this dialog's
// job is to name *what* is about to go, where it will be kept, and what unsaved
// work is in the way.
const deleteDialogEl = ref(null)
const deletePrompt = computed(() => state.deletePrompt)

const deleteTitle = computed(() =>
  deletePrompt.value ? `Delete ${deletePrompt.value.name}` : 'Delete this path?',
)

// A folder means everything inside it, and the honest scope line says so.
const deleteScope = computed(() =>
  deletePrompt.value?.dir
    ? 'this folder and everything inside it'
    : 'this file',
)

// The listing the tree already holds for that folder may say how many entries
// are visible — a *lower bound* only, and labelled as one: the tree is lazy, so
// a collapsed branch was never listed, and the policy hides entries the row
// count cannot know about. Counting properly is the server's walk, and its
// number comes back in the answer, never guessed here from the browser.
const deleteShownAtLeast = computed(() => {
  const prompt = deletePrompt.value
  if (!prompt?.dir) return 0
  return tree.state.nodes[prompt.path]?.entries?.length ?? 0
})

// The one case worth a second sentence: unsaved edits are the only thing the
// trash cannot bring back.
const dirtyWarning = computed(() => {
  const count = deletePrompt.value?.dirtyBuffers ?? 0
  if (!count) return ''
  return count === 1
    ? 'An open buffer of this file has changes that are not on disk, and they will be lost.'
    : `${count} open buffers under this folder have changes that are not on disk, and they will be lost.`
})

watch(
  () => deletePrompt.value?.path,
  async (path) => {
    await nextTick()
    const dialog = deleteDialogEl.value
    if (!dialog) return
    if (path && !dialog.open) dialog.showModal()
    else if (!path && dialog.open) dialog.close()
  },
)

function onDeleteBackdropClick(event) {
  if (event.target === deleteDialogEl.value) cancelViaDelete()
}

// Esc answers Cancel, never Delete — the keyboard default of a confirm must not
// be the destructive one. The same reason Cancel is the *first* button: Enter
// runs the focused button, so focus is what decides the default here.
function onDeleteNativeClose() {
  if (deletePrompt.value) cancelViaDelete()
}

function cancelViaDelete() {
  cancelDelete()
  deleteDialogEl.value?.close()
}

async function deleteViaDialog() {
  if (busy.value) return
  busy.value = true
  await confirmDelete()
  busy.value = false
  deleteDialogEl.value?.close()
}
</script>

<template>
  <dialog
    ref="dialogEl"
    class="editor-conflict-dialog"
    aria-labelledby="editor-conflict-title"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <article>
      <header>
        <h2 id="editor-conflict-title">{{ title }}</h2>
      </header>
      <p v-if="record" class="caption mono">{{ record.path }}</p>
      <p>{{ body }}</p>
      <p v-if="diskWhen" class="caption">disk copy last written: {{ diskWhen }}</p>
      <footer class="settings-actions">
        <button class="secondary contrast" type="button" :disabled="busy" @click="dismiss">
          Not now
        </button>
        <button
          v-if="!isDeleted"
          type="button"
          :disabled="busy"
          title="Discard my edits and take the version on disk"
          @click="reload"
        >
          Reload
        </button>
        <button
          type="button"
          :disabled="busy || isDeleted"
          :title="
            isDeleted
              ? 'There is no copy on disk left to compare against'
              : 'Show the copy on disk beside my buffer, then choose'
          "
          @click="compare"
        >
          Compare
        </button>
        <button
          v-if="!isDeleted"
          class="secondary contrast editor-conflict-overwrite"
          type="button"
          :disabled="busy"
          title="Write my buffer over the disk copy, knowing it changed"
          @click="overwrite"
        >
          Overwrite
        </button>
      </footer>
    </article>
  </dialog>

  <!-- The close confirm: three answers, and Cancel is the default (a
       destructive default in a confirm is a bug — so Esc, the backdrop and the
       first button all mean Cancel). Discard names its own consequence because
       there is no undo and no trash behind a closed tab: the buffer *is* the
       only copy. -->
  <dialog
    ref="closeDialogEl"
    class="editor-close-dialog"
    aria-labelledby="editor-close-title"
    @click="onCloseBackdropClick"
    @close="onCloseNativeClose"
  >
    <article>
      <header>
        <h2 id="editor-close-title">{{ closeTitle }}</h2>
      </header>
      <p v-if="closePrompt" class="caption mono">{{ closePrompt.key }}</p>
      <p>
        This buffer has changes that are not on disk, and nothing else holds a
        copy of them — closing the tab throws them away.
      </p>
      <footer class="settings-actions">
        <button class="secondary contrast" type="button" :disabled="busy" @click="cancelViaDialog">
          Cancel
        </button>
        <button
          class="secondary contrast editor-close-discard"
          type="button"
          :disabled="busy"
          title="Close the tab and lose the changes — there is no undo and no trash for a closed buffer"
          @click="discardViaDialog"
        >
          Discard — the changes are lost
        </button>
        <button type="button" :disabled="busy" title="Write the buffer to disk, then close the tab" @click="saveAndCloseViaDialog">
          Save and close
        </button>
      </footer>
    </article>
  </dialog>

  <!-- The delete confirm. Nothing is unlinked: the path is *moved* into a
       batch under the workspace's own `.nojae/trash`, which is why the
       confirmation can name it and the retention out loud — a mistaken delete
       is two terminal commands away from being undone. Cancel is first and
       therefore the default; Delete is the destructive answer and wears the
       danger tone. -->
  <dialog
    ref="deleteDialogEl"
    class="editor-delete-dialog"
    aria-labelledby="editor-delete-title"
    @click="onDeleteBackdropClick"
    @close="onDeleteNativeClose"
  >
    <article>
      <header>
        <h2 id="editor-delete-title">{{ deleteTitle }}</h2>
      </header>
      <p v-if="deletePrompt">
        Deleting {{ deleteScope }}. It is not erased — it is moved to the trash
        and can be put back from there.
      </p>
      <p v-if="deletePrompt" class="caption mono">{{ deletePrompt.path }}</p>
      <p v-if="deleteShownAtLeast" class="caption">
        at least {{ deleteShownAtLeast }} items are shown in this folder —
        the full count is counted when the delete runs
      </p>
      <p v-if="dirtyWarning" class="alert-error" role="alert">{{ dirtyWarning }}</p>
      <footer class="settings-actions">
        <button class="secondary contrast" type="button" :disabled="busy" @click="cancelViaDelete">
          Cancel
        </button>
        <button
          class="secondary contrast editor-delete-confirm"
          type="button"
          :disabled="busy"
          :title="`Move ${deletePrompt?.path ?? 'it'} to the trash, kept for ${TRASH_RETENTION_DAYS} days`"
          @click="deleteViaDialog"
        >
          Delete
        </button>
      </footer>
      <p class="caption editor-delete-keeps">
        kept in <code>{{ TRASH_FOLDER }}/</code> for {{ TRASH_RETENTION_DAYS }} days —
        the exact batch is named once it has moved. That folder is reserved, so it
        is not shown in the tree: put it back with <code>mv</code> in a terminal.
      </p>
    </article>
  </dialog>
</template>
