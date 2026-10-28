<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { useEditorCreate } from '../composables/useEditorCreate'
import { useEditorRename } from '../composables/useEditorRename'
import { fileIcon } from '../lib/file-icons'

// The inline name row: where a new file or folder is named, *inside* the folder
// that will hold it — which is why it is a row of the tree rather than a modal
// form. The parent is whatever the user pointed at, and the answer lands in the
// same listing the name was typed into.
//
// It serves both name modes from one widget: `create` (a new file or folder
// inside the parent) and `rename` (a new name — possibly a new *path*, which is
// how a move is typed — for the row the input belongs to). The state is the
// tree's (`state.editing`: mode, parent, path, kind, value, error) and the
// request is the matching flow's (`useEditorCreate` / `useEditorRename`); this
// component is only the caret and the two keys. Nothing here decides whether a
// name is *allowed* — that is the server's rule, and its refusal is shown in
// `error`, not argued with. `Enter` submits, `Escape` cancels, and the listing
// the row lives in is already expanded before this mounts.
const { state, depthOf, cancelCreate, updateCreateValue } = useWorkspaceTree()
const { submitCreate } = useEditorCreate()
const { submitRename } = useEditorRename()

const input = ref(null)
// `state.editing` is the *one* name-input state, shared by both modes (see
// useWorkspaceTree): a create names a new thing inside `parent`, a rename gives
// the row at `path` a new name — and may name a whole path, which is how a move
// is typed. One widget, one state, one validation-and-refusal shape; the only
// difference is which flow a submitted value is sent to.
const pending = computed(() => state.editing)
const isRename = computed(() => pending.value?.mode === 'rename')
const glyph = computed(() => fileIcon(pending.value?.value ?? '', { dir: !!pending.value?.isDir }))
// The same indent as the rows of the listing it sits in, so the caret lands
// where the new row will appear: a folder's children are one level deeper than
// that folder, and a row's `--tree-depth` is the number of path segments in it.
const depth = computed(() => depthOf(pending.value?.parent ?? '') + 1)

// Focus once per input row, not once per keystroke: the row is keyed by its
// parent, kind and mode, so working elsewhere re-mounts it and gets the caret
// again. A *rename* preselects the current name but leaves the extension out of
// the selection, so typing replaces the stem and `.` stays where it was — the
// edit people actually mean when they ask to rename a file.
watch(
  () => [pending.value?.mode, pending.value?.path, pending.value?.parent, pending.value?.isDir],
  async () => {
    if (!pending.value) return
    await nextTick()
    const field = input.value
    if (!field) return
    field.focus()
    if (!isRename.value) return
    const value = pending.value.value ?? ''
    const dot = value.lastIndexOf('.')
    field.setSelectionRange(0, dot > 0 ? dot : value.length)
  },
  { immediate: true },
)

function onInput(event) {
  updateCreateValue(event.target.value)
}

// When the input goes away the caret must not fall to `<body>`: the row the tree
// considers focused carries the roving tabindex, so it is what takes the caret
// back. Rows are matched by `dataset.path` rather than by a selector because a
// path may hold any character (the same reason WorkspaceNavigator scans).
async function restoreRowFocus() {
  await nextTick()
  for (const candidate of document.querySelectorAll('.tree-row')) {
    if (candidate.dataset.path === state.focused) {
      candidate.focus()
      return
    }
  }
}

async function onSubmit() {
  // A second Enter while a request is in flight does nothing: `submitCreate`
  // joins the running one, and the row must not look like it is being asked
  // twice. The input stays open on a refusal (the fix is to type again), and the
  // flow clears the state itself when the change lands.
  const result = isRename.value ? await submitRename() : await submitCreate()
  if (result.ok) await restoreRowFocus()
  else input.value?.focus()
}

async function onCancel() {
  cancelCreate()
  await restoreRowFocus()
}
</script>

<template>
  <!-- Not a `role="treeitem"`: this row is not a file, and inventing a tree item
       for it would put a nameless entry in the tree's count and its reading
       order. It is a labelled textbox inside the group of its parent. -->
  <div
    class="tree-row tree-row--creating"
    :class="`depth-${depth - 1}`"
    :style="{ '--tree-depth': depth - 1 }"
  >
    <span class="tree-chevron" aria-hidden="true"></span>
    <span class="tree-icon" aria-hidden="true">{{ glyph }}</span>
    <input
      ref="input"
      class="tree-name-input"
      type="text"
      :value="pending.value"
      :aria-label="
        isRename
          ? 'New name — type a folder path to move it'
          : pending.isDir
            ? 'Name of the new folder'
            : 'Name of the new file'
      "
      :placeholder="
        isRename
          ? 'New name, or a path to move it'
          : pending.isDir
            ? 'New folder name'
            : 'New file name'
      "
      spellcheck="false"
      autocomplete="off"
      @input="onInput"
      @keydown.enter.prevent="onSubmit"
      @keydown.esc.prevent="onCancel"
    />
    <button
      class="tree-name-input-action"
      type="button"
      :aria-label="isRename ? 'Rename it' : 'Create it'"
      :title="isRename ? 'Rename (Enter)' : 'Create (Enter)'"
      @click="onSubmit"
    >
      ✓
    </button>
    <button
      class="tree-name-input-action"
      type="button"
      :aria-label="
        isRename
          ? `Cancel — do not rename ${pending.path}`
          : `Cancel — do not create anything in ${pending.parent || 'the workspace root'}`
      "
      title="Cancel (Escape)"
      @click="onCancel"
    >
      ×
    </button>
  </div>
  <!-- The refusal of the typed value, under the caret rather than in a message
       elsewhere: the fix is to edit what is already there, and both modes answer
       the same way. `.tree-row-error` is the shared name for that line;
       `.tree-name-input-error` rides along as the create's own hook, so a
       selector for one mode keeps working. One rule in theme.css styles both. -->
  <p
    v-if="pending.error"
    class="tree-row-error tree-name-input-error"
    role="alert"
  >
    {{ pending.error }}
  </p>
</template>
