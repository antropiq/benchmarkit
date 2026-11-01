import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

import EditorMoveDialog from '../src/components/EditorMoveDialog.vue'
import { useBatch } from '../src/composables/useBatch'
import { useChat } from '../src/composables/useChat'
import { useEditorMove } from '../src/composables/useEditorMove'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// ---------------------------------------------------------------------------
// The move dialog, mounted for real.
//
// It decides nothing, so this suite is about *what is on screen* rather than
// about the flow: the title's count, the subject list rendered verbatim, the
// prefill's exact text, and which state disables the Move button. The flow
// itself (the request, the reconciliation) is `useEditorMove.test.js`, and the
// route's rules are `tests/test_editorfs_write.py` — nothing here re-litigates
// either.
//
// The composable is driven, never stubbed (the shape `searchDialog.test.js`
// uses): `startMove` raises `movePrompt` exactly as mov-05's menu item will, and
// the DOM is read back. That is what makes the narrowing case testable at all —
// the dialog has to show the list the *tree* settled on, not re-derive one.
//
// jsdom has no `showModal`/`close` on `<dialog>`, so both are patched on
// `Element.prototype` the way `useEditorWriteMatrix.test.js:385` does: they only
// have to keep `el.open` truthful, which is all the component's watch reads.
// ---------------------------------------------------------------------------

const mover = useEditorMove()
const tree = useWorkspaceTree()
const chat = useChat()
const batch = useBatch()
const notices = useEditorNotices()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

let host = null

function mountDialog() {
  host = document.createElement('div')
  document.body.appendChild(host)
  return mount(EditorMoveDialog, { attachTo: host })
}

async function openBatch(paths, parent) {
  mover.startMove({ paths, parent })
  await flushPromises()
}

function field(w) {
  return w.find('.editor-move-destination')
}

function moveButton(w) {
  return w.find('.editor-move-confirm')
}

beforeEach(() => {
  Element.prototype.showModal = function showModal() {
    this.open = true
  }
  Element.prototype.close = function close() {
    this.open = false
  }
  mover.state.movePrompt = null
  tree.state.nodes = {}
  tree.state.marked = []
  tree.state.markAnchor = null
  chat.state.phase = 'idle'
  batch.state.active = false
  notices.state.notices.length = 0
  vi.restoreAllMocks()
})

afterEach(() => {
  host?.remove()
  host = null
  document.body.querySelectorAll('dialog').forEach((el) => el.remove())
})

describe('the title and the subject list', () => {
  // The count is the batch's, in the honest singular at N=1.
  it('titles a batch of three `Move 3 files` and lists all three paths', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md', 'notes/b.md', 'notes/c.md'], 'notes')

    expect(w.find('h2').text()).toBe('Move 3 files')
    expect(moveButton(w).text()).toBe('Move 3 files')
    const lines = w.findAll('.editor-move-list p')
    expect(lines.map((node) => node.text())).toEqual([
      'notes/a.md',
      'notes/b.md',
      'notes/c.md',
    ])
    // The delete confirm's shape: a path is a caption in mono, and it is text —
    // no element in the line came from the path itself.
    expect(lines[0].classes()).toContain('caption')
    expect(lines[0].classes()).toContain('mono')
    w.unmount()
  })

  it('titles a batch of one `Move 1 file`', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')

    expect(w.find('h2').text()).toBe('Move 1 file')
    expect(w.findAll('.editor-move-list p')).toHaveLength(1)
    w.unmount()
  })

  // D4/R1: a cross-parent selection narrows to the clicked row, and the dialog
  // is where that becomes visible — one line, and a singular title.
  it('shows the narrowed subject, not the marks the batch came from', async () => {
    tree.state.marked = ['notes/a.md', 'other/b.md']
    const w = mountDialog()
    await openBatch(['other/b.md'], 'other')

    expect(w.find('h2').text()).toBe('Move 1 file')
    expect(w.findAll('.editor-move-list p').map((node) => node.text())).toEqual([
      'other/b.md',
    ])
    w.unmount()
  })

  // A path is a string from the user's own tree, not markup: bound as text, and
  // a name that *looks* like a tag stays a name.
  it('renders a path that looks like markup as text', async () => {
    const w = mountDialog()
    await openBatch(['notes/a <b>.md'], 'notes')

    const line = w.find('.editor-move-list p')
    expect(line.text()).toBe('notes/a <b>.md')
    expect(line.element.innerHTML).not.toContain('<b')
    w.unmount()
  })
})

describe('the destination field', () => {
  // D7/D8: the prefill is the sources' parent, spelled the way the tree spells
  // it — not a basename, not the absolute working location.
  it('holds the shared parent verbatim, including when that is the root', async () => {
    const w = mountDialog()
    await openBatch(['nested/deep/a.md', 'nested/deep/b.md'], 'nested/deep')

    expect(field(w).element.value).toBe('nested/deep')
    expect(field(w).element.value).not.toBe('deep')
    expect(field(w).element.value).not.toContain('/mnt/')
    w.unmount()
  })

  it('is one `<input>`, and the hint under it is R3 in one sentence', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')

    expect(field(w).element.tagName).toBe('INPUT')
    expect(w.find('textarea').exists()).toBe(false)
    expect(w.find('.editor-move-hint').text()).toBe('The folder must already exist.')
    w.unmount()
  })

  // The prefill's own consequence (D7): the no-op case opens explained, and the
  // edit is what enables the button.
  it('disables Move while the field equals the sources\' parent, and names why', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md', 'notes/b.md'], 'notes')

    expect(moveButton(w).element.disabled).toBe(true)
    expect(w.find('.editor-move-error').text()).toMatch(/already are/)

    await field(w).setValue('notes/archive')
    expect(moveButton(w).element.disabled).toBe(false)
    expect(w.find('.editor-move-error').exists()).toBe(false)
    w.unmount()
  })

  // The shape rules are the composable's; the dialog only shows their sentence.
  it('disables Move for a destination the field answers itself', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')

    for (const typed of ['../outside', '/etc', '']) {
      await field(w).setValue(typed)
      expect(moveButton(w).element.disabled).toBe(true)
      expect(w.find('.editor-move-error').text()).not.toBe('')
    }
    w.unmount()
  })

  it('writes what is typed into the prompt, so a later submit reads the field', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')
    await field(w).setValue('notes/archive')

    expect(mover.state.movePrompt.value).toBe('notes/archive')
    expect(mover.state.movePrompt.parent).toBe('notes')
    w.unmount()
  })

  // D17: the badge's answer is the only client-side claim about writability —
  // no `ro` pre-check, and nothing else.
  it('disables Move while the pane is read-only, and leaves Cancel free', async () => {
    chat.state.phase = 'answering'
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')
    await field(w).setValue('notes/archive')

    expect(moveButton(w).element.disabled).toBe(true)
    expect(moveButton(w).attributes('title')).toMatch(/read-only/)
    expect(w.findAll('footer button')[0].element.disabled).toBe(false)

    chat.state.phase = 'idle'
    await flushPromises()
    expect(moveButton(w).element.disabled).toBe(false)
    w.unmount()
  })
})

describe('opening and closing follow the prompt', () => {
  // No `v-if` on the element: unmounting it would drop the field's caret and a
  // half-typed value, which is what a refusal is about to be edited into.
  it('is in the DOM before anything raises it, and opens closed', () => {
    const w = mountDialog()

    expect(mover.state.movePrompt).toBeNull()
    expect(w.find('dialog').exists()).toBe(true)
    expect(w.find('dialog').element.open).toBe(false)
    w.unmount()
  })

  it('shows the modal when the prompt is raised, with the field focused', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md', 'notes/b.md'], 'notes')

    expect(w.find('dialog').element.open).toBe(true)
    expect(document.activeElement).toBe(field(w).element)
    // The prefill arrives *selected*: editing it is the common case.
    expect(field(w).element.selectionStart).toBe(0)
    expect(field(w).element.selectionEnd).toBe('notes'.length)
    w.unmount()
  })

  it('closes when the prompt is cleared, and Cancel clears it', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')
    expect(w.find('dialog').element.open).toBe(true)

    mover.cancelMove()
    await flushPromises()
    expect(w.find('dialog').element.open).toBe(false)

    await openBatch(['notes/a.md'], 'notes')
    w.findAll('footer button')[0].trigger('click')
    await flushPromises()
    expect(mover.state.movePrompt).toBeNull()
    expect(w.find('dialog').element.open).toBe(false)
    w.unmount()
  })

  // Esc and a click outside are the same answer as Cancel — the destructive one
  // is never the keyboard default.
  it('a backdrop click answers Cancel', async () => {
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')

    w.find('dialog').trigger('click')
    await flushPromises()
    expect(mover.state.movePrompt).toBeNull()
    expect(w.find('dialog').element.open).toBe(false)
    w.unmount()
  })
})

describe('submitting', () => {
  // §5.1 step 6: a refusal is an edit, not a restart.
  it('a refusal keeps the dialog open with the typed value and the server sentence', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ detail: 'Parent folder does not exist: notes/archiv' }, false, 409),
    )
    const w = mountDialog()
    await openBatch(['notes/a.md', 'notes/b.md'], 'notes')
    await field(w).setValue('notes/archiv')

    await moveButton(w).trigger('click')
    await flushPromises()

    expect(w.find('dialog').element.open).toBe(true)
    expect(field(w).element.value).toBe('notes/archiv')
    expect(w.find('.editor-move-error').text()).toBe(
      'Parent folder does not exist: notes/archiv',
    )
    // The value is a shape the field accepts, so Move is clickable again — the
    // refusal was the server's answer about that folder, and editing it is what
    // a user does with the sentence under the field.
    expect(moveButton(w).element.disabled).toBe(false)
    expect(mover.state.movePrompt.busy).toBe(false)
    expect(mover.state.movePrompt.error).toBe('Parent folder does not exist: notes/archiv')
    w.unmount()
  })

  it('Enter in the field submits, and a disabled value sends no request at all', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ detail: 'no' }, false, 409),
    )
    const w = mountDialog()
    await openBatch(['notes/a.md'], 'notes')

    await field(w).trigger('keydown.enter')
    expect(fetchMock.mock.calls).toHaveLength(0)

    await field(w).setValue('notes/archive')
    await field(w).trigger('keydown.enter')
    expect(fetchMock.mock.calls).toHaveLength(1)
    w.unmount()
  })

  // While the request is in flight the dialog is inert *and* stays up: the
  // answer belongs under the field that typed it.
  it('in flight: both buttons inert, and Esc does not dismiss', async () => {
    let release
    const gate = new Promise((resolve) => {
      release = resolve
    })
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => gate)
    const w = mountDialog()
    await openBatch(['notes/a.md', 'notes/b.md'], 'notes')
    await field(w).setValue('notes/archive')

    moveButton(w).trigger('click')
    await flushPromises()
    expect(mover.state.movePrompt.busy).toBe(true)
    expect(w.findAll('footer button').map((node) => node.element.disabled)).toEqual([
      true,
      true,
    ])

    w.find('dialog').element.dispatchEvent(new Event('close'))
    await flushPromises()
    expect(w.find('dialog').element.open).toBe(true)
    expect(mover.state.movePrompt).not.toBeNull()

    release(jsonResponse({ moved: [], to_parent: 'notes/archive' }))
    await flushPromises()
    w.unmount()
  })
})

// The few rules that are about *what the file may contain* and cannot be seen
// on a DOM node (the approach `searchDialog.test.js` takes for the same reason).
const componentSource = readFileSync(
  join(resolve(), 'src', 'components', 'EditorMoveDialog.vue'),
  'utf8',
)

describe('what the component is not allowed to know', () => {
  it('never fetches, never renders raw HTML, never imports the tree or the file layer', () => {
    expect(componentSource).not.toMatch(/\bfetch\(/)
    expect(componentSource).not.toMatch(/v-html/)
    expect(componentSource).not.toMatch(/lib\/markdown/)
    expect(componentSource).not.toMatch(/useWorkspaceTree/)
    expect(componentSource).not.toMatch(/useEditorFiles/)
    // The cap is the server's 400, shown like any other refusal.
    expect(componentSource).not.toMatch(/MAX_MOVE_BATCH/)
  })
})
