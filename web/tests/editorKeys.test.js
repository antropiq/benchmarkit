import { describe, expect, it, vi } from 'vitest'

import { applyEditorChord, editorChord } from '../src/lib/editor-keys'

// The Editor pane's three chords, tested as the decision rather than as a
// mounted component: which key means what, and when the pane may consume the
// event at all. The listener in EditorArea only supplies the effects.

const ON_EDITOR = { activeTab: 'editor', active: 'src/a.py' }

function key(over) {
  // Defaults are the shape of the app's own chord: plain Ctrl + the key. Each
  // test then turns one modifier off or on, so a wrong modifier rule shows up as
  // one failed expectation rather than as a null that the calling assertion
  // cannot tell apart from "the function refused".
  return { key: 's', ctrlKey: true, metaKey: false, shiftKey: false, altKey: false, ...over }
}

describe('editorChord', () => {
  it('maps Ctrl+S and Cmd+S to saving the active buffer', () => {
    expect(editorChord(key({ key: 's' }), ON_EDITOR).action).toBe('save')
    expect(editorChord(key({ key: 'S', metaKey: true }), ON_EDITOR).action).toBe('save')
  })

  it('maps Ctrl+Shift+S to saving every dirty buffer', () => {
    expect(editorChord(key({ key: 's', shiftKey: true }), ON_EDITOR).action).toBe('save-all')
  })

  it('maps Ctrl+W to closing the active tab', () => {
    expect(editorChord(key({ key: 'w' }), ON_EDITOR).action).toBe('close')
    expect(editorChord(key({ key: 'w', metaKey: true }), ON_EDITOR).action).toBe('close')
  })

  it('leaves every other screen alone', () => {
    const elsewhere = { activeTab: 'workbench', active: 'src/a.py' }
    for (const k of ['s', 'w']) expect(editorChord(key({ key: k }), elsewhere)).toBeNull()
  })

  it('needs a modifier, and an active tab to act on', () => {
    expect(editorChord(key({ key: 's', ctrlKey: false }), ON_EDITOR)).toBeNull()
    expect(editorChord(key({ key: 's', altKey: true }), ON_EDITOR)).toBeNull()
    for (const k of ['s', 'w']) {
      expect(editorChord(key({ key: k, ctrlKey: true }), { ...ON_EDITOR, active: null })).toBeNull()
    }
  })

  it('is not the whole keyboard: Ctrl+Shift+W and plain letters stay the browser\'s', () => {
    expect(editorChord(key({ key: 'w', shiftKey: true }), ON_EDITOR)).toBeNull()
    expect(editorChord(key({ key: 'p', ctrlKey: true }), ON_EDITOR)).toBeNull()
  })
})

describe('applyEditorChord', () => {
  it('consumes and runs the handler for a chord the pane owns', () => {
    const save = vi.fn()
    const event = key({ key: 's', ctrlKey: true })
    event.preventDefault = vi.fn()
    expect(applyEditorChord(event, ON_EDITOR, { save })).toBe(true)
    expect(save).toHaveBeenCalledTimes(1)
    // Without the preventDefault the browser's own Save-page dialog opens on
    // top of the app's answer.
    expect(event.preventDefault).toHaveBeenCalledTimes(1)
  })

  it('consumes nothing for a chord it does not own', () => {
    const event = key({ key: 's', ctrlKey: true })
    event.preventDefault = vi.fn()
    const handlers = { save: vi.fn(), 'save-all': vi.fn(), close: vi.fn() }
    expect(applyEditorChord(event, { activeTab: 'benchmark', active: 'a' }, handlers)).toBe(false)
    expect(event.preventDefault).not.toHaveBeenCalled()
    for (const handler of Object.values(handlers)) expect(handler).not.toHaveBeenCalled()
  })
})
