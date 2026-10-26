import { computed } from 'vue'
import { useEditorFiles } from './useEditorFiles'
import { useChat } from './useChat'

// ---------------------------------------------------------------------------
// Editor mode: whether the code pane currently promises an editable surface.
//
// It is a composable rather than a computed inside EditorArea because the same
// answer is *shown* in the file tree's header and *enforced* on the Monaco
// surface in the code pane — two components, one rule. A pane labelled "Editor"
// is editable while the WorkBench is idle (edits stay in the buffer, nothing
// reaching disk until a save route lands) and read-only while a reply streams,
// because the model is the only other writer of these files and it must not
// race a human typing in the same buffer.
//
// The badge therefore lives in the navigator header, where there is always
// room for it: the tab strip is a row that shrinks as buffers pile up, so a
// status that belongs to the pane rather than to one tab has no business
// competing with the chips for its width.
//
// Known edge, deliberate: batch mode owns the conversation between two task
// turns with `useBatch.state.active` while `phase` drops back to 'idle' (see
// useBatch.js), so on that gap this rule makes files briefly editable. Kept as
// specified.
// ---------------------------------------------------------------------------

export function useEditorMode() {
  const files = useEditorFiles()
  const chat = useChat()

  // A diff tab is a reader whatever the chat is doing: its two sides come from
  // git and the disk and there is no buffer to type into, so the editable
  // promise the badge makes belongs to a file buffer only.
  const isDiffTab = computed(() => {
    const active = files.state.active
    if (!active) return false
    return files.state.open.find((item) => item.key === active)?.kind === 'diff'
  })

  const editorMode = computed(() => {
    if (isDiffTab.value) return 'read_only'
    return chat.state.phase === 'idle' ? 'read_write' : 'read_only'
  })

  const paneReadonly = computed(() => editorMode.value === 'read_only')
  const isEditable = computed(() => editorMode.value === 'read_write')

  const badgeLabel = computed(() => (isEditable.value ? 'read-write' : 'read-only'))

  const badgeTitle = computed(() => {
    if (isDiffTab.value) {
      return 'Read-only — this tab is a diff: the committed side and the one on disk.'
    }
    return isEditable.value
      ? 'Editable — the WorkBench is idle. Unsaved edits are kept in the tab, not written to disk yet.'
      : 'Read-only — a WorkBench reply is running and may be editing these files.'
  })

  return { editorMode, paneReadonly, isEditable, isDiffTab, badgeLabel, badgeTitle }
}
