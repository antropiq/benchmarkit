import { computed } from 'vue'
import { useBatch } from './useBatch'
import { useEditorFiles } from './useEditorFiles'
import { useEditorNotices } from './useEditorNotices'
import { useChat } from './useChat'

// ---------------------------------------------------------------------------
// Editor mode: whether the code pane currently promises an editable surface.
//
// It is a composable rather than a computed inside EditorArea because the same
// answer is *shown* in the file tree's header and *enforced* on the Monaco
// surface in the code pane — two components, one rule. A pane labelled "Editor"
// is editable while the WorkBench is idle (an edit stays in the buffer until an
// explicit save) and read-only while a reply streams, because the model is the
// only other writer of these files and it must not race a human typing in the
// same buffer. The server enforces the same rule on every write route, so this
// composable is the badge's source, not the boundary.
//
// The badge therefore lives in the navigator header, where there is always
// room for it: the tab strip is a row that shrinks as buffers pile up, so a
// status that belongs to the pane rather than to one tab has no business
// competing with the chips for its width. Its `title` has the last word on the
// two states a pane-wide label cannot otherwise explain: an unsaved buffer
// behind an editable badge, and a write that was just refused for this path.
//
// Batch mode owns the conversation between two task turns with
// `useBatch.state.active` while `phase` drops back to 'idle' (see useBatch.js),
// so the rule reads both: a lot that is running makes the pane read-only even
// in the gap, because the server refuses the write anyway and a badge inviting
// an edit that will be refused is a worse answer than one that explains itself.
// ---------------------------------------------------------------------------

export function useEditorMode() {
  const files = useEditorFiles()
  const batch = useBatch()
  const notices = useEditorNotices()
  const chat = useChat()

  // Which kind the active tab is: 'file' (an editable buffer) or the kind of a
  // *view* — a diff, a preview. A view is a reader whatever the chat is doing:
  // its contents come from the server and there is no buffer to type into, so
  // the editable promise the badge makes belongs to a file buffer only.
  const activeKind = computed(() => {
    const active = files.state.active
    if (!active) return null
    return files.state.open.find((item) => item.key === active)?.kind ?? null
  })

  // A diff tab is a reader whatever the chat is doing (kept as its own name: the
  // badge wording for a diff is a separate sentence below).
  const isDiffTab = computed(() => activeKind.value === 'diff')
  const isPreviewTab = computed(() => activeKind.value === 'preview')
  const isViewTab = computed(() => activeKind.value !== null && activeKind.value !== 'file')

  const editorMode = computed(() => {
    if (isViewTab.value) return 'read_only'
    // Idle *and* no lot in progress: between two task turns `phase` is 'idle'
    // but the next turn is already queued, and the model may be mid-edit of the
    // file under the caret.
    return chat.state.phase === 'idle' && !batch.state.active
      ? 'read_write'
      : 'read_only'
  })

  const paneReadonly = computed(() => editorMode.value === 'read_only')
  const isEditable = computed(() => editorMode.value === 'read_write')

  const badgeLabel = computed(() => (isEditable.value ? 'read-write' : 'read-only'))

  const badgeTitle = computed(() => {
    const parts = []
    if (isDiffTab.value) {
      parts.push('Read-only — this tab is a diff: the committed side and the one on disk.')
    } else if (isPreviewTab.value) {
      parts.push('Read-only — this tab shows the rendered Markdown of the file.')
    } else if (batch.state.active) {
      parts.push(
        'Read-only — a batch is running and its tasks edit these files. Saving is refused until the lot stops.',
      )
    } else if (isEditable.value) {
      parts.push(
        'Editable — the WorkBench is idle. Unsaved edits are kept in the tab, not written to disk yet.',
      )
    } else {
      parts.push(
        'Read-only — a WorkBench reply is running and may be editing these files. Saving is refused until it ends.',
      )
    }

    // The badge is where a *pane-wide* state gets explained, so the freshness
    // answer about the buffer on screen rides here too: the chip's marker says
    // which tab, this says what it means for the pane.
    const active = files.state.open.find((item) => item.key === files.state.active)
    if (active?.kind === 'file') {
      // A dirty buffer needs its unsaved state spelled out here: this is the
      // one pane-wide indicator of it that survives a tab switch, the dot on
      // the chip being easy to lose in a full strip. The wording follows the
      // mode, because a locked pane holding unwritten work is exactly the case
      // where saying "Ctrl+S" would be a lie.
      if (active.dirty) {
        parts.push(
          isEditable.value
            ? 'This buffer has changes that are not on disk yet (Ctrl+S writes them).'
            : 'This buffer has changes that are not on disk, and cannot be written until the pane is editable.',
        )
      }
      if (active.deleted) {
        parts.push('This file is gone from disk; the buffer is all that is left of it.')
      } else if (active.stale) {
        parts.push('This file changed on disk while it is open here.')
      }
      // The last word on "why did my Ctrl+S do nothing?": the refusal this path
      // is under, quoted from the notice the write raised.
      const refusal = notices.noticeForPath(active.path)
      if (refusal) parts.push(refusal.text)
    }
    return parts.join(' ')
  })

  return {
    editorMode,
    paneReadonly,
    isEditable,
    isDiffTab,
    isPreviewTab,
    isViewTab,
    badgeLabel,
    badgeTitle,
  }
}
