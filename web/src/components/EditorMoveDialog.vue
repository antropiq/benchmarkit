<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useEditorMode } from '../composables/useEditorMode'
import { useEditorMove, validateMoveDestination } from '../composables/useEditorMove'

// The batch move's modal: what is about to move, the destination folder, and
// two buttons. It is the smallest of the Editor dialogs on purpose — every
// value on screen is `movePrompt` (composable state, raised by `startMove` with
// the subject `batchFor()` already settled), and every rule that enables its
// button is `validateMoveDestination`. Nothing here re-derives the batch, joins
// a target path, or knows whether a folder is allowed: the list is rendered
// verbatim, the server joins `to_parent` with each basename, and a protected
// destination comes back as a refusal in the policy's own words (§2.4, D10).
//
// `<dialog>` + `showModal()` over composable state, reconciled by a `watch` —
// the house pattern (`EditorTreeDialogs.vue`, `SearchDialog.vue`) — and, like
// those, no `v-if` on the element: unmounting it would drop the field's caret
// and its half-typed value, which is exactly the state a refusal is about to be
// edited into.
//
// Cancel is the **first** button, so Esc, the backdrop and Enter-on-focus all
// mean the non-destructive answer; Enter inside the field is the explicit
// submit. A refusal leaves the dialog open with what was typed and the server's
// sentence under the field (§5.1 step 6) — only `cancelMove()` or a success
// clears `movePrompt`, and it is the watch that closes the element. While the
// request is in flight nothing here is clickable and nothing closes: a second
// Enter for one batch would make the second answer a 409 about the first.
const { state, cancelMove, submitMove } = useEditorMove()
const { isEditable } = useEditorMode()

const dialogEl = ref(null)
const fieldEl = ref(null)

const prompt = computed(() => state.movePrompt)
const paths = computed(() => prompt.value?.paths ?? [])
const busy = computed(() => !!prompt.value?.busy)

// `Move 3 files`, and the honest singular at N=1 — the count the batch is
// about. It is the title *and* the button label, because both name the action
// the click performs.
const countLabel = computed(() =>
  paths.value.length === 1 ? 'Move 1 file' : `Move ${paths.value.length} files`,
)

// The field's answer, computed from what is in it rather than from what was
// last written to it: the prefill puts the sources' parent straight in, so the
// dialog opens with the no-op sentence showing and the Move button disabled —
// the edit *is* the enable (D7). `prompt.error` then carries the server's
// sentence for the same value after a refusal.
const shapeError = computed(() => {
  const pending = prompt.value
  if (!pending) return null
  return validateMoveDestination(pending.value, { parent: pending.parent })
})
const errorLine = computed(() => shapeError.value ?? prompt.value?.error ?? '')

// D17: the only client-side claim about writability is the mode badge's. Whether
// *this* path is writable stays the server's rule, so a too-large batch, a
// hidden destination and a taken target name all arrive as a refusal to show.
const canSubmit = computed(
  () => !!prompt.value && !shapeError.value && !busy.value && isEditable.value,
)
const submitTitle = computed(() => {
  if (!isEditable.value) {
    return 'The pane is read-only while a WorkBench reply runs — it is refused until it ends'
  }
  return errorLine.value || 'Move them there (Enter)'
})

// Raise and dismiss with the prompt, and put the caret on the field with the
// prefill *selected*: the common case is editing the folder that is already
// there, which should cost a keystroke rather than a select-all.
watch(
  () => prompt.value,
  async (key) => {
    await nextTick()
    const dialog = dialogEl.value
    if (dialog) {
      if (key && !dialog.open) dialog.showModal()
      else if (!key && dialog.open) dialog.close()
    }
    if (key) {
      const field = fieldEl.value
      field?.focus()
      field?.select()
    }
  },
)

function onInput(event) {
  const pending = prompt.value
  if (!pending) return
  pending.value = event.target.value
  // The task's shape: the sentence, or '' when there is none. The line below the
  // field still shows the computed one; this is what a later submit reads.
  pending.error = validateMoveDestination(pending.value, { parent: pending.parent }) ?? ''
}

async function onSubmit() {
  if (!canSubmit.value) return
  // Nothing is closed here: the success path clears `movePrompt` (composable
  // state) and the watch closes the element. A refusal leaves both where they
  // are, so the fix is an edit and not a restart.
  await submitMove()
}

function onCancel() {
  if (busy.value) return
  cancelMove()
  dialogEl.value?.close()
}

function onBackdropClick(event) {
  if (event.target === dialogEl.value) onCancel()
}

// Esc lands here through the native `close`. While a request is in flight the
// dialog may not disappear on its own — the answer belongs under the field that
// typed it — so it goes back up, and the watch closes it once the prompt itself
// is spent.
function onNativeClose() {
  if (busy.value) {
    dialogEl.value?.showModal()
    return
  }
  if (prompt.value) cancelMove()
}
</script>

<template>
  <dialog
    ref="dialogEl"
    class="editor-move-dialog"
    aria-labelledby="editor-move-title"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <article v-if="prompt">
      <header>
        <h2 id="editor-move-title">{{ countLabel }}</h2>
      </header>

      <!-- The subject, verbatim and one path per line — the answer to the
           narrowing case (D4/R1): a cross-parent selection shows *one* path and
           the title says `Move 1 file`, so what is about to happen is visible
           before a request is sent. Plain text, not markdown: a path from the
           user's own tree is a string, and nothing here renders one as HTML. -->
      <div class="editor-move-list">
        <p v-for="path in paths" :key="path" class="caption mono">{{ path }}</p>
      </div>

      <!-- One `<input>`, never a textarea: the value is one folder path, and a
           newline inside it would be a character only the server could answer.
           The value is workspace-relative and sent as typed — the *server* joins
           the target (D11), so nothing here builds a path out of it. -->
      <label class="editor-move-field">
        Destination folder
        <input
          ref="fieldEl"
          class="editor-move-destination"
          type="text"
          :value="prompt.value"
          placeholder="a folder that already exists"
          spellcheck="false"
          autocomplete="off"
          :aria-describedby="errorLine ? 'editor-move-error' : 'editor-move-hint'"
          @input="onInput"
          @keydown.enter.prevent="onSubmit"
        />
      </label>

      <!-- The promise R3 makes, said before the refusal that proves it: no
           folder is created here, and the server's sentence names the one it
           looked for. There is deliberately no folder picker — the tree is lazy,
           so a complete list does not exist in the browser, and a partial one
           would hide policy-denied folders (D10). -->
      <p id="editor-move-hint" class="caption editor-move-hint">
        The folder must already exist.
      </p>
      <p
        v-if="errorLine"
        id="editor-move-error"
        class="error-line editor-move-error"
        role="alert"
      >
        {{ errorLine }}
      </p>

      <footer class="settings-actions">
        <button class="secondary contrast" type="button" :disabled="busy" @click="onCancel">
          Cancel
        </button>
        <button
          class="editor-move-confirm"
          type="button"
          :disabled="!canSubmit"
          :title="submitTitle"
          @click="onSubmit"
        >
          {{ countLabel }}
        </button>
      </footer>
    </article>
  </dialog>
</template>
