import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import AgentModesDialog from '../src/components/AgentModesDialog.vue'
import { personaByteCount, useAgentModes } from '../src/composables/useAgentModes'

// ---------------------------------------------------------------------------
// The agent-modes dialog, mounted for real (eds-06).
//
// The component decides nothing — every rule it renders is a function from
// `useAgentModes` and is judged in `agentModes.test.js`. So this suite is about
// *what is on screen*: one tab per entry the **file** declares (in the file's
// order, D2), the finding sitting next to the field it names, which affordance
// disables Save, and what a click does to the two routes. Nothing here
// re-litigates a rule; it checks that the dialog renders the rules it is given.
//
// The fixtures mirror the real `GET /api/agent-modes/document` payload shape
// (`src/core/agent_modes.read_agent_modes_document`), findings included, so a
// finding shown here is the loader's own sentence and not a paraphrase.
//
// jsdom has no `showModal`/`close` on `<dialog>`, so both are patched on
// `Element.prototype` the way `editorMoveDialog.test.js` does: they only have to
// keep `el.open` truthful, which is all the component's watch reads.
// ---------------------------------------------------------------------------

const EDITOR_CAPS = {
  maxModes: 16,
  maxIdChars: 32,
  idPattern: '^[a-z][A-Za-z0-9_-]*$',
  maxLabelChars: 40,
  maxPersonaBytes: 16384,
  maxPersonaLines: 16,
}

const EDITOR_DEFAULTS = {
  version: 1,
  readme: ['This file lists the workspace agent modes.'],
  default: 'code',
  agents: {
    chat: { label: 'Chat', mcptools: false, lsptools: false, systemMessage: ['You never call a tool.'] },
    code: { label: 'Code', mcptools: true, lsptools: true, systemMessage: ['You edit files.'] },
    architect: { label: 'Architect', mcptools: true, lsptools: false, systemMessage: ['You design.'] },
  },
}

function entry(key, fields = {}, extra = {}) {
  return { key, hasForm: true, fields, findings: [], ...extra }
}

function documentBody({
  entries = [
    entry('chat', EDITOR_DEFAULTS.agents.chat),
    entry('code', EDITOR_DEFAULTS.agents.code),
    entry('architect', EDITOR_DEFAULTS.agents.architect),
  ],
  defaultId = 'code',
  findings = [],
  caps = EDITOR_CAPS,
  defaults = EDITOR_DEFAULTS,
  source = {},
} = {}) {
  return {
    entries,
    default: defaultId,
    findings,
    source: {
      path: '/work/.nojae/agent-modes.json',
      exists: true,
      loaded: true,
      mtime: '1700000000000000000',
      ...source,
    },
    caps,
    defaults,
  }
}

function absentFileBody() {
  return documentBody({
    entries: [],
    defaultId: null,
    source: { exists: false, loaded: false, mtime: null },
  })
}

// What `POST /api/agent-modes/save` answers: the new verdict in the dropdown's
// own shape plus the new mtime token.
function saveResponse({ defaultId = 'code', mtime = '1700000009000000000' } = {}) {
  return {
    saved: true,
    created: false,
    backedUp: true,
    modes: [
      { id: 'chat', label: 'Chat', mcptools: false, lsptools: false, editable: true },
      { id: 'code', label: 'Code', mcptools: true, lsptools: true, editable: true },
      { id: 'architect', label: 'Architect', mcptools: true, lsptools: false, editable: true },
    ],
    default: defaultId,
    findings: [],
    source: {
      path: '/work/.nojae/agent-modes.json',
      exists: true,
      loaded: true,
      mtime,
    },
  }
}

// The editor pair only: `calls` records them in order, so "Create wrote nothing"
// and "Save posted exactly one document" are asserted rather than inferred.
function stubEditorRoutes({ document = documentBody(), save = saveResponse(), calls = [] } = {}) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
    const key = String(url)
    if (key === '/api/agent-modes/document') {
      calls.push({ key, init })
      return jsonResponse(document)
    }
    if (key === '/api/agent-modes/save') {
      calls.push({ key, init })
      if (typeof save === 'object' && save.__fail) {
        return jsonResponse({ detail: save.detail }, false, save.status)
      }
      return jsonResponse(save)
    }
    throw new Error(`unexpected fetch: ${key}`)
  })
}

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

let host = null

// Mount with the card already open (the pencil's `open` ref is true by the time
// the component exists), with the routes stubbed **first**: opening *is* the
// document read, so every test calls `stubEditorRoutes` *before* this — the
// fetch the component makes on opening has to hit the stub, not the network.
async function openDialog(props = {}) {
  host = document.createElement('div')
  document.body.appendChild(host)
  const wrapper = mount(AgentModesDialog, {
    attachTo: host,
    props: { open: true, ...props },
  })
  await flushPromises()
  return wrapper
}

function resetEditor() {
  const editor = useAgentModes().editor
  editor.open = false
  editor.loading = false
  editor.saving = false
  editor.error = null
  editor.conflict = null
  editor.invalidFindings = []
}

// The nav's *tabs* only — Common then the file's entries in order. The Create
// button lives in the same nav but is not a tab, so the selector is the tab
// hook the component puts on each one, not "every button in the nav".
function tabs(w) {
  return w.findAll('.agent-modes-tab').map((node) => node.text())
}

// A nav tab by position: 0 is Common, then the file's entries in the file's
// order.
function tabButton(w, index) {
  return w.findAll('.agent-modes-tab')[index]
}

function section(w, key) {
  return w.find(`section[data-tab="${key}"]`)
}

function saveButton(w) {
  return w.find('.agent-modes-save')
}

function createButton(w) {
  return w.find('.agent-modes-create-button')
}

function personaField(w, key) {
  return section(w, key).find('textarea')
}

function labelField(w, key) {
  return section(w, key).find('input[type="text"]')
}

function fieldFindings(w, key) {
  return section(w, key).findAll('.agent-modes-field-finding').map((node) => node.text())
}

function buttonByText(w, text) {
  return w.findAll('button').filter((node) => node.text() === text)[0]
}

beforeEach(() => {
  Element.prototype.showModal = function showModal() {
    this.open = true
  }
  Element.prototype.close = function close() {
    this.open = false
  }
  resetEditor()
  vi.restoreAllMocks()
})

afterEach(() => {
  host?.remove()
  host = null
  document.body.querySelectorAll('dialog').forEach((el) => el.remove())
})

describe('the tabs (D2, D5)', () => {
  it('opens with Common plus one tab per entry the file declares, in the file order', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    expect(tabs(w)).toEqual(['Common', 'Chat', 'Code', 'Architect'])
    // The order is the *file's*, not an alphabetical or built-in one: a payload
    // in another order gets its own tab order.
    w.unmount()
  })

  it('follows the file order for a file that spells its modes differently', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [
          entry('architect', EDITOR_DEFAULTS.agents.architect),
          entry('reviewer', { label: 'Reviewer', mcptools: false, lsptools: false }),
          entry('chat', EDITOR_DEFAULTS.agents.chat),
        ],
        defaultId: 'reviewer',
      }),
    })
    const w = await openDialog()

    expect(tabs(w)).toEqual(['Common', 'Architect', 'Reviewer', 'Chat'])
    // Nothing was added: the built-in fallback and the app-owned mode are absent.
    expect(section(w, 'code').exists()).toBe(false)
    w.unmount()
  })

  it('shows no app-owned mode anywhere — not a tab, not a default option', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    const text = w.text()
    expect(text).not.toContain('Todo')
    expect(text).not.toContain('builder')
    const options = section(w, 'common').findAll('option').map((node) => node.text())
    expect(options).toEqual(['Chat', 'Code', 'Architect'])
    w.unmount()
  })

  it('opens on the first mode the file declares, so a broken entry is on screen', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [
          entry('broken', {}, {
            hasForm: false,
            findings: [{ kind: 'error', message: 'agents.broken: is a string, not an object with label, mcptools, lsptools, systemMessage' }],
          }),
        ],
        defaultId: null,
      }),
    })
    const w = await openDialog()

    // `v-show` hides a section by *removing* it from the layout; the tab that is
    // open is the one that still has its fields.
    expect(section(w, 'broken').find('.agent-modes-delete-button').exists()).toBe(true)
    expect(fieldFindings(w, 'broken')[0]).toContain('agents.broken: is a string')
    w.unmount()
  })

  it('opens the tab of the entry whose finding blocks Save, not the first tab', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [
          entry('chat', EDITOR_DEFAULTS.agents.chat),
          entry('architect', { label: 'Architect', mcptools: true }, {
            findings: [{ kind: 'error', message: 'agents.architect.mcptool: unknown key (expected label, mcptools, lsptools, systemMessage)' }],
          }),
        ],
      }),
    })
    const w = await openDialog()

    // D3: the finding opens *that* entry's tab, so the fix is on screen and not
    // one click away — even though `chat` is the file's first entry.
    expect(section(w, 'architect').find('.agent-modes-delete-button').exists()).toBe(true)
    expect(tabButton(w, 2).classes()).toContain('active')
    w.unmount()
  })
})

describe('the mode tab (D5)', () => {
  it('holds Label, the two switches and the System prompt — and nothing else', async () => {
    stubEditorRoutes()
    const w = await openDialog()
    await tabButton(w, 2).trigger('click') // Code

    const code = section(w, 'code')
    expect(code.find('input[type="text"]').exists()).toBe(true)
    expect(code.findAll('input[role="switch"]')).toHaveLength(2)
    expect(code.find('textarea').exists()).toBe(true)
    // The switches carry the file's own values (the built-in Code mode has both
    // hubs on; Architect is the one that narrows LSP off).
    const switches = code.findAll('input[role="switch"]')
    expect(switches[0].element.checked).toBe(true) // mcptools
    expect(switches[1].element.checked).toBe(true) // lsptools
    expect(code.find('textarea').element.value).toBe('You edit files.')
    w.unmount()
  })

  it('says what an empty persona means', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [entry('chat', { ...EDITOR_DEFAULTS.agents.chat, systemMessage: null })],
        defaultId: 'chat',
      }),
    })
    const w = await openDialog()

    expect(section(w, 'chat').find('textarea').element.value).toBe('')
    expect(section(w, 'chat').text()).toContain('use the configured system prompt')
    w.unmount()
  })

  it('labels the two switches as narrowing, in the dropdown tooltip sentence', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    const hint = w.find('.agent-modes-gating-hint')
    expect(hint.exists()).toBe(true)
    expect(hint.text()).toContain('narrow')
    expect(hint.text()).toContain('off in Settings stays off')
    w.unmount()
  })

  it('an existing mode id is read-only and says why; a new one gets an editable field', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    expect(section(w, 'code').find('.agent-modes-id-readonly').text()).toContain('cannot be')
    // The id is a caption in that paragraph, not a field: the only text input in
    // an existing mode's tab is its Label.
    expect(section(w, 'code').findAll('input[type="text"]')).toHaveLength(1)
    expect(section(w, 'code').find('.agent-modes-id-readonly').text()).toContain('code')

    await w.find('.agent-modes-create input').setValue('Reviewer')
    await createButton(w).trigger('click')
    const created = section(w, 'reviewer')
    expect(created.findAll('input[type="text"]').map((node) => node.element.value)).toEqual([
      'Reviewer',
      'reviewer',
    ])
    w.unmount()
  })

  it('a Label edit changes the tab caption immediately', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    await labelField(w, 'chat').setValue('Chatting')
    expect(tabs(w)).toEqual(['Common', 'Chatting', 'Code', 'Architect'])
    expect(section(w, 'chat').find('h3').text()).toBe('Chatting')

    // An empty label falls back to the id, which is the loader's own rule.
    await labelField(w, 'chat').setValue('')
    expect(tabs(w)).toEqual(['Common', 'chat', 'Code', 'Architect'])
    w.unmount()
  })
})

describe('Create agent (D5)', () => {
  it('appends a tab, focuses it, and writes nothing', async () => {
    const calls = []
    stubEditorRoutes({ calls })
    const w = await openDialog()

    await w.find('.agent-modes-create input').setValue('Docs Review (v2)!')
    await createButton(w).trigger('click')

    expect(tabs(w)).toEqual(['Common', 'Chat', 'Code', 'Architect', 'Docs Review (v2)!'])
    expect(section(w, 'docs-review-v2').exists()).toBe(true)
    // Focused: the caret is inside the new mode's own section.
    expect(document.activeElement?.closest(`section[data-tab="docs-review-v2"]`)).not.toBeNull()
    // Nothing was written: only the document read happened, no save call.
    expect(calls.map((call) => call.key)).toEqual(['/api/agent-modes/document'])
    w.unmount()
  })

  it('is inert at the mode cap the payload states', async () => {
    const entries = Array.from({ length: EDITOR_CAPS.maxModes }, (_unused, i) =>
      entry(`mode${i}`, { label: `Mode ${i}`, mcptools: false, lsptools: false }),
    )
    stubEditorRoutes({ document: documentBody({ entries, defaultId: 'mode0' }) })
    const w = await openDialog()

    expect(createButton(w).element.disabled).toBe(true)
    w.unmount()
  })

  it('a new mode whose id collides is named by the payload rule, not by a guess', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    await w.find('.agent-modes-create input').setValue('Code')
    await createButton(w).trigger('click')

    // The slug gave up its collision: the tab is `code-2`, and its id field says so.
    expect(tabs(w)).toContain('Code')
    const created = section(w, 'code-2')
    expect(created.exists()).toBe(true)
    expect(created.findAll('input[type="text"]')[1].element.value).toBe('code-2')
    w.unmount()
  })
})

describe('findings (D3)', () => {
  it('an entry finding opens its tab, marks the field it names, and disables Save', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [
          entry('chat', EDITOR_DEFAULTS.agents.chat),
          entry('code', { label: 'Code', mcptools: true }, {
            findings: [
              { kind: 'error', message: 'agents.code.mcptool: unknown key (expected label, mcptools, lsptools, systemMessage)' },
              { kind: 'error', message: 'agents.code.label: is a number, not a string' },
            ],
          }),
        ],
      }),
    })
    const w = await openDialog()

    // The tab is marked and Save is disabled before anything is clicked.
    expect(saveButton(w).element.disabled).toBe(true)
    const brokenTab = w.findAll('.agent-modes-nav button')[2]
    expect(brokenTab.classes()).toContain('agent-modes-tab--broken')

    await brokenTab.trigger('click')
    const shown = fieldFindings(w, 'code')
    // The label finding sits with the Label field; the unknown key has no field
    // to sit on, so it is listed as one the form cannot fix.
    expect(shown).toEqual([
      'agents.code.label: is a number, not a string',
      'agents.code.mcptool: unknown key (expected label, mcptools, lsptools, systemMessage)',
    ])
    expect(w.find('.agent-modes-blocking-note').text()).toContain('fixed or removed')
    w.unmount()
  })

  it('fixing the entry enables Save', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [entry('architect', EDITOR_DEFAULTS.agents.architect)],
        defaultId: 'architect',
      }),
    })
    const w = await openDialog()
    expect(saveButton(w).element.disabled).toBe(false)

    // The cap is enforced *in the form*: typing past it blocks Save, editing back
    // under it releases it — the server never gets the chance to drop the entry.
    await personaField(w, 'architect').setValue('x'.repeat(EDITOR_CAPS.maxPersonaBytes + 1))
    expect(saveButton(w).element.disabled).toBe(true)
    expect(fieldFindings(w, 'architect')[0]).toContain('agents.architect.systemMessage')

    await personaField(w, 'architect').setValue('You design.')
    expect(saveButton(w).element.disabled).toBe(false)
    expect(fieldFindings(w, 'architect')).toEqual([])
    w.unmount()
  })

  it('deleting the entry enables Save', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [
          entry('code', EDITOR_DEFAULTS.agents.code),
          entry('broken', {}, {
            hasForm: false,
            findings: [{ kind: 'error', message: 'agents.broken: is a string, not an object with label, mcptools, lsptools, systemMessage' }],
          }),
        ],
      }),
    })
    const w = await openDialog()
    expect(saveButton(w).element.disabled).toBe(true)

    await tabButton(w, 2).trigger('click')
    await section(w, 'broken').find('.agent-modes-delete-button').trigger('click')
    expect(section(w, 'broken').find('.agent-modes-delete-confirm').exists()).toBe(true)
    await section(w, 'broken').find('.agent-modes-delete-yes').trigger('click')

    expect(tabs(w)).toEqual(['Common', 'Code'])
    expect(saveButton(w).element.disabled).toBe(false)
    w.unmount()
  })

  it('a file-level finding is listed, disables Save, and offers no defaults draft', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [],
        defaultId: null,
        findings: [{ kind: 'error', message: 'The agent-modes file /work/.nojae/agent-modes.json is not valid JSON: boom (line 1, column 2)' }],
        source: { exists: true, loaded: false },
      }),
    })
    const w = await openDialog()

    expect(w.find('.agent-modes-file-findings').text()).toContain('not valid JSON')
    expect(saveButton(w).element.disabled).toBe(true)
    // The one affordance this case must *not* have: turning a syntax error into
    // the loss of every mode the user wrote.
    expect(w.findAll('button').map((node) => node.text()).join(' ')).not.toMatch(/default/i)
    expect(w.find('.agent-modes-creates-file').exists()).toBe(false)
    // The path is named, because that is where the fix lives.
    expect(w.text()).toContain('/work/.nojae/agent-modes.json')
    w.unmount()
  })

  it('an unrepresentable entry gets a tab with the finding and a Delete only', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [
          entry('code', EDITOR_DEFAULTS.agents.code),
          entry('broken', {}, {
            hasForm: false,
            findings: [{ kind: 'error', message: 'agents.broken: is a string, not an object with label, mcptools, lsptools, systemMessage' }],
          }),
        ],
      }),
    })
    const w = await openDialog()

    const tab = section(w, 'broken')
    expect(tab.exists()).toBe(true)
    expect(tab.find('input').exists()).toBe(false)
    expect(tab.find('textarea').exists()).toBe(false)
    expect(tab.find('.agent-modes-delete-button').exists()).toBe(true)
    expect(tab.text()).toContain('no fields to edit')
    // The caption is the raw key — what the file actually says.
    expect(tabs(w)).toEqual(['Common', 'Code', 'broken'])
    w.unmount()
  })

  it('a note about a hand-written reserved id is listed, has no tab, and does not hold Save', async () => {
    // The loader's own sentence for this case, built from the id read out of the
    // module that owns it (naming it here would make *this* file a hit in the
    // reserved-id scan, and the point is that the dialog renders the server's
    // data — id included — without ever holding the id itself).
    const python = readFileSync(resolve('../src/core/agent_modes.py'), 'utf8')
    const reserved = /RESERVED_MODE_IDS[^=]*=\s*\(([^)]*)\)/.exec(python)[1]
    const id = [...reserved.matchAll(/"([^"]+)"/g)][0][1]
    stubEditorRoutes({
      document: documentBody({
        findings: [
          { kind: 'note', message: `agents.${id}: "${id}" is defined by benchmarkit and cannot be set here; this entry is ignored` },
        ],
      }),
    })
    const w = await openDialog()

    expect(w.find('.agent-modes-file-notes').text()).toContain('is defined by benchmarkit')
    expect(saveButton(w).element.disabled).toBe(false)
    // A note names the pointer as server data; the dialog shows it and adds no tab.
    expect(tabs(w)).toEqual(['Common', 'Chat', 'Code', 'Architect'])
    expect(w.text()).toContain(id)
    w.unmount()
  })
})

describe('an absent file (D3)', () => {
  it('offers the built-in document as a draft and says saving creates the file', async () => {
    stubEditorRoutes({ document: absentFileBody() })
    const w = await openDialog()

    expect(tabs(w)).toEqual(['Common', 'Chat', 'Code', 'Architect'])
    expect(section(w, 'chat').find('textarea').element.value).toBe('You never call a tool.')
    expect(w.find('.agent-modes-creates-file').text()).toContain('saving will create it')
    // The button says what it does.
    expect(saveButton(w).text()).toBe('Create file')
    expect(saveButton(w).element.disabled).toBe(false)
    w.unmount()
  })
})

describe('the persona counters', () => {
  it('counts bytes against the payload cap, and shows the remaining line budget', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [entry('chat', { ...EDITOR_DEFAULTS.agents.chat, systemMessage: '日本語' })],
        defaultId: 'chat',
      }),
    })
    const w = await openDialog()

    const count = section(w, 'chat').find('.agent-modes-persona-count')
    expect(count.text()).toBe(`${personaByteCount('日本語')} / ${EDITOR_CAPS.maxPersonaBytes} bytes`)
    expect(count.text()).toContain('9') // bytes, not the 3 characters

    const lines = section(w, 'chat').find('.agent-modes-persona-lines')
    expect(lines.text()).toContain('remaining line cap: 15')
    w.unmount()
  })

  it('the line budget warns but never disables Save', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [entry('chat', { ...EDITOR_DEFAULTS.agents.chat, systemMessage: null })],
        defaultId: 'chat',
      }),
    })
    const w = await openDialog()

    const text = Array.from({ length: 20 }, (_unused, i) => `line ${i}`).join('\n')
    await personaField(w, 'chat').setValue(text)

    expect(section(w, 'chat').find('.agent-modes-persona-warning').text()).toContain('16-line')
    expect(saveButton(w).element.disabled).toBe(false) // soft: a style budget, not a file rule
    w.unmount()
  })
})

describe('reset this persona', () => {
  it('appears for an entry the app has a built-in persona for, and replaces the field', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [entry('code', { ...EDITOR_DEFAULTS.agents.code, systemMessage: 'Something I wrote myself.' })],
        defaultId: 'code',
      }),
    })
    const w = await openDialog()

    const reset = section(w, 'code').find('.agent-modes-reset')
    expect(reset.exists()).toBe(true)
    await personaField(w, 'code').setValue('Something I wrote myself.')
    await reset.trigger('click')
    expect(personaField(w, 'code').element.value).toBe('You edit files.')
    // It is an edit, not a save: the file is untouched until Save.
    w.unmount()
  })

  it('does not appear for a user-created mode', async () => {
    stubEditorRoutes({
      document: documentBody({
        entries: [entry('reviewer', { label: 'Reviewer', mcptools: false, lsptools: false, systemMessage: 'My own instructions.' })],
        defaultId: 'reviewer',
      }),
    })
    const w = await openDialog()

    expect(section(w, 'reviewer').find('.agent-modes-reset').exists()).toBe(false)
    w.unmount()
  })
})

describe('delete (resolved open question 1)', () => {
  it('is refused for the mode that is currently the default', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    const code = section(w, 'code') // 'code' is the file's default
    await tabButton(w, 2).trigger('click')
    expect(code.find('.agent-modes-delete-button').element.disabled).toBe(true)
    expect(code.find('.agent-modes-delete-confirm').exists()).toBe(false)
    expect(code.text()).toContain('pick another default')
    w.unmount()
  })

  it('needs a confirm for another mode, and the confirm is what removes the tab', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    await tabButton(w, 1).trigger('click') // Chat
    await section(w, 'chat').find('.agent-modes-delete-button').trigger('click')
    // The confirm is asked for; nothing has been removed yet.
    expect(tabs(w)).toEqual(['Common', 'Chat', 'Code', 'Architect'])

    await section(w, 'chat').find('.agent-modes-delete-yes').trigger('click')
    expect(tabs(w)).toEqual(['Common', 'Code', 'Architect'])
    w.unmount()
  })

  it('"Keep it" leaves the entry alone', async () => {
    stubEditorRoutes()
    const w = await openDialog()

    await tabButton(w, 1).trigger('click')
    await section(w, 'chat').find('.agent-modes-delete-button').trigger('click')
    await buttonByText(section(w, 'chat'), 'Keep it').trigger('click')

    expect(tabs(w)).toEqual(['Common', 'Chat', 'Code', 'Architect'])
    expect(section(w, 'chat').find('.agent-modes-delete-confirm').exists()).toBe(false)
    w.unmount()
  })
})

describe('save and cancel (D6, D7)', () => {
  it('Save posts the whole document with the token, and the payload is what the form showed', async () => {
    const calls = []
    stubEditorRoutes({ calls, save: saveResponse({ defaultId: 'reviewer' }) })
    const w = await openDialog()

    await labelField(w, 'chat').setValue('Chatting')
    await personaField(w, 'chat').setValue('You never call a tool.\nNever guess.')
    const switches = section(w, 'code').findAll('input[role="switch"]')
    await switches[0].setValue(false)
    await switches[1].setValue(true)
    await w.find('.agent-modes-create input').setValue('Reviewer')
    await createButton(w).trigger('click')
    await section(w, 'common').find('select').setValue('reviewer')

    await saveButton(w).trigger('click')
    await flushPromises()

    expect(calls.map((call) => call.key)).toEqual(['/api/agent-modes/document', '/api/agent-modes/save'])
    const body = JSON.parse(calls[1].init.body)
    expect(body.mtime).toBe('1700000000000000000') // the token the read returned
    // D6's shape: no version, no readme, agents in the dialog's order.
    expect('version' in body.document).toBe(false)
    expect('readme' in body.document).toBe(false)
    expect(Object.keys(body.document.agents)).toEqual(['chat', 'code', 'architect', 'reviewer'])
    expect(body.document.default).toBe('reviewer')
    // What the fields showed is what the file gets, persona text verbatim.
    expect(body.document.agents.chat.label).toBe('Chatting')
    expect(body.document.agents.chat.systemMessage).toBe('You never call a tool.\nNever guess.')
    expect(body.document.agents.code.mcptools).toBe(false)
    expect(body.document.agents.code.lsptools).toBe(true)
    // A successful save closes the card.
    expect(w.emitted('close')).toBeTruthy()
    w.unmount()
  })

  it('Cancel after edits makes no save call and writes nothing', async () => {
    const calls = []
    stubEditorRoutes({ calls })
    const w = await openDialog()

    await labelField(w, 'chat').setValue('Chatting')
    await personaField(w, 'chat').setValue('A persona nobody saved.')
    expect(w.find('.agent-modes-dirty-note').text()).toContain('writes nothing')

    await buttonByText(w, 'Cancel').trigger('click')

    expect(calls.map((call) => call.key)).toEqual(['/api/agent-modes/document'])
    expect(w.emitted('close')).toBeTruthy()
    w.unmount()
  })

  it('a 409 keeps the dialog open, names the conflict, and keeps the draft', async () => {
    stubEditorRoutes({
      save: {
        __fail: true,
        status: 409,
        detail: {
          reason: 'stale',
          path: '/work/.nojae/agent-modes.json',
          diskMtime: '1700000005000000000',
          documentMtime: '1700000000000000000',
        },
      },
    })
    const w = await openDialog()

    await labelField(w, 'chat').setValue('Chatting')
    await saveButton(w).trigger('click')
    await flushPromises()

    expect(w.find('.agent-modes-conflict').text()).toContain('changed on disk')
    expect(w.text()).toContain('1700000005000000000') // both numbers are named
    expect(w.text()).toContain('1700000000000000000')
    // The card is still open and the draft is intact: nothing merged, nothing
    // overwritten, no reload-and-re-apply.
    expect(w.find('dialog').element.open).toBe(true)
    expect(labelField(w, 'chat').element.value).toBe('Chatting')
    // The draft is intact in every field the user touched — including the caption
    // that follows it, which is how "your edits are still here" is visible.
    expect(tabs(w)).toEqual(['Common', 'Chatting', 'Code', 'Architect'])
    w.unmount()
  })

  it('a server 422 is shown rather than swallowed', async () => {
    stubEditorRoutes({
      save: {
        __fail: true,
        status: 422,
        detail: { reason: 'invalid', findings: [{ kind: 'error', message: 'agents.x: unknown key' }] },
      },
    })
    const w = await openDialog()

    await saveButton(w).trigger('click')
    await flushPromises()

    expect(w.find('.agent-modes-server-findings').text()).toContain('agents.x: unknown key')
    expect(w.find('dialog').element.open).toBe(true)
    w.unmount()
  })
})

describe('what this component is not allowed to be', () => {
  const source = () => readFileSync(resolve('src/components/AgentModesDialog.vue'), 'utf8')

  it('names no reserved mode id, no cap and no route of its own', () => {
    const text = source()
    // The reserved id is read out of the Python module rather than hardcoded
    // here, so this check cannot be satisfied by a different string.
    const python = readFileSync(resolve('../src/core/agent_modes.py'), 'utf8')
    const declaration = /RESERVED_MODE_IDS[^=]*=\s*\(([^)]*)\)/.exec(python)
    expect(declaration, 'agent_modes.RESERVED_MODE_IDS is not readable').toBeTruthy()
    for (const match of declaration[1].matchAll(/"([^"]+)"/g)) {
      expect(text, `the component names the reserved mode id ${match[1]}`).not.toContain(match[1])
    }
    // No cap is spelled in the client: every number arrives in the payload.
    expect(text).not.toContain('16384')
    expect(text).not.toContain('maxModes = 16')
    // The dialog reaches the file through the composable's two calls only. It
    // spells **no route at all**: the write-marker scan
    // (`tests/test_write_surface_isolation.py`) allows the save route's spelling
    // in the composable alone, so a component naming it would be an offender —
    // and no file-tree route is reachable from here either.
    expect(text).not.toContain('/api/')
    expect(text).not.toContain('/api/fs')
  })

  it('renders field text only — no raw v-html anywhere', () => {
    expect(source()).not.toContain('v-html')
  })

  it('sizes the article, never the dialog element', () => {
    // The card family: the component carries no inline sizing of <dialog>.
    const dialogTag = /<dialog[\s\S]*?>/.exec(source())
    expect(dialogTag[0]).not.toMatch(/style=/)
  })

  it('the scan over web/src covers it (the tree walk reaches components)', () => {
    // The reserved-id scan (`agentModes.test.js`) walks `web/src` in full, so a
    // new component is inside it without being added to anything. What could
    // silently leave this file out is that scan's extension filter, so *that* is
    // what is asserted here — the scan's own list, read from the file that owns
    // it rather than copied (a second copy of a guard is how a guard drifts).
    const scan = readFileSync(resolve('tests/agentModes.test.js'), 'utf8')
    const extensions = /SCANNED_EXTENSIONS = \[([^\]]*)\]/.exec(scan)
    expect(extensions, 'the reserved-id scan is not readable').toBeTruthy()
    expect(extensions[1]).toContain("'.vue'")
    expect(scan).toContain("walk(join(resolve(), 'src'))")
  })
})
