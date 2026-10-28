import { beforeEach, describe, expect, it } from 'vitest'

import { useBatch } from '../src/composables/useBatch'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorMode } from '../src/composables/useEditorMode'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useChat } from '../src/composables/useChat'

// Layer 1 (jsdom): the single rule behind the Editor panes — the mode the file
// tree's header *badge* states and the code pane enforces on Monaco. It lives
// in a composable precisely because those are two components, so the rule and
// the wording of the badge are asserted here rather than in either component.

const files = useEditorFiles()
const mode = useEditorMode()
const batch = useBatch()
const notices = useEditorNotices()
const chat = useChat()

function resetState() {
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  chat.state.phase = 'idle'
  batch.state.active = false
  notices.clear()
}

function open(kind) {
  const key = kind === 'diff' ? 'diff:a.py' : 'a.py'
  files.state.open.push({ kind, key, path: 'a.py' })
  files.state.active = key
}

describe('useEditorMode: one rule, shown and enforced', () => {
  beforeEach(resetState)

  it('is read_write with nothing open and an idle WorkBench', () => {
    expect(mode.editorMode.value).toBe('read_write')
    expect(mode.isEditable.value).toBe(true)
    expect(mode.paneReadonly.value).toBe(false)
    expect(mode.badgeLabel.value).toBe('read-write')
  })

  it('goes read-only for the whole duration of a streamed reply', () => {
    open('file')
    for (const phase of ['waiting', 'reasoning', 'answering']) {
      chat.state.phase = phase
      expect(mode.editorMode.value).toBe('read_only')
      expect(mode.paneReadonly.value).toBe(true)
      expect(mode.badgeLabel.value).toBe('read-only')
    }
    chat.state.phase = 'idle'
    expect(mode.paneReadonly.value).toBe(false)
  })

  it('is read-only while the active tab is a diff, idle chat or not', () => {
    open('diff')
    expect(mode.editorMode.value).toBe('read_only')
    expect(mode.badgeTitle.value).toContain('this tab is a diff')
    // The neighbouring file buffer of the same path is editable again: the
    // rule follows the *active* tab, not the paths that are open.
    files.state.open.push({ kind: 'file', key: 'a.py', path: 'a.py' })
    files.state.active = 'a.py'
    expect(mode.editorMode.value).toBe('read_write')
  })

  // The badge is where a pane-wide state is explained, so the freshness answer
  // about the buffer on screen rides in its title — the chip marks *which* tab,
  // the badge says what is true of the pane.
  it('explains a buffer the disk moved under', () => {
    open('file')
    files.state.open[0].stale = true
    expect(mode.badgeTitle.value).toContain('changed on disk')
    // The mode itself does not change: staleness is not a lock on the surface.
    expect(mode.editorMode.value).toBe('read_write')
  })

  it('explains a buffer whose file is gone', () => {
    open('file')
    files.state.open[0].deleted = true
    expect(mode.badgeTitle.value).toContain('gone from disk')
  })

  // D7: batch mode owns the conversation between two task turns with
  // `state.active` while `phase` drops back to 'idle'. The server refuses the
  // write in that gap whatever the badge says, so the pane must not *invite* it.
  it('is read-only while a batch runs, in the idle gap between its turns', () => {
    open('file')
    chat.state.phase = 'idle'
    batch.state.active = true
    expect(mode.editorMode.value).toBe('read_only')
    expect(mode.paneReadonly.value).toBe(true)
    expect(mode.badgeTitle.value).toContain('a batch is running')
    expect(mode.badgeTitle.value).toContain('Saving is refused until the lot stops')

    // The lot ending gives the pane back — the existing idle case, unchanged.
    batch.state.active = false
    expect(mode.editorMode.value).toBe('read_write')
  })

  // The badge is the only pane-wide indicator that survives a tab switch, so a
  // dirty buffer gets its unsaved state named — in wording the mode can honour.
  it('mentions unsaved work, and only promises Ctrl+S when the pane is editable', () => {
    open('file')
    expect(mode.badgeTitle.value).not.toContain('not on disk')
    files.state.open[0].dirty = true
    expect(mode.badgeTitle.value).toContain('not on disk yet (Ctrl+S writes them)')

    // Locked pane, same buffer: the same news without the promise.
    chat.state.phase = 'answering'
    expect(mode.badgeTitle.value).toContain('cannot be written until the pane is editable')
    expect(mode.badgeTitle.value).not.toContain('Ctrl+S')
  })

  // "Why did my Ctrl+S do nothing?" — the refusal this path is under gets the
  // last word in the title, quoted from the notice the write raised.
  it('quotes the refusal this path is under', () => {
    open('file')
    files.state.open[0].dirty = true
    notices.notice(
      'error',
      'Could not save a.py: Cannot save while a chat reply is streaming',
      'a.py',
    )
    expect(mode.badgeTitle.value).toContain('Could not save a.py: Cannot save')
    // A refusal about another path is not this badge's business.
    notices.notice('error', 'Could not save other.py: refused', 'other.py')
    expect(mode.badgeTitle.value).not.toContain('other.py')
  })

  // A reply streaming both explains the lock and names the consequence: the
  // write route refuses it, which is the answer a user reaching for Ctrl+S needs.
  it('says saving is refused while a reply streams', () => {
    open('file')
    chat.state.phase = 'answering'
    expect(mode.badgeTitle.value).toContain('Saving is refused until it ends')
  })
})
