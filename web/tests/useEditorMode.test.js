import { beforeEach, describe, expect, it } from 'vitest'

import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorMode } from '../src/composables/useEditorMode'
import { useChat } from '../src/composables/useChat'

// Layer 1 (jsdom): the single rule behind the Editor panes — the mode the file
// tree's header *badge* states and the code pane enforces on Monaco. It lives
// in a composable precisely because those are two components, so the rule and
// the wording of the badge are asserted here rather than in either component.

const files = useEditorFiles()
const mode = useEditorMode()
const chat = useChat()

function resetState() {
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  chat.state.phase = 'idle'
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
})
