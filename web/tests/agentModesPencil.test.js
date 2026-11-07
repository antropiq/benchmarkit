import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import WorkBenchInput from '../src/components/WorkBenchInput.vue'
import { loadAgentModes, setAgentMode, useAgentModes } from '../src/composables/useAgentModes'
import { useBatch } from '../src/composables/useBatch'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useChat } from '../src/composables/useChat'

// ---------------------------------------------------------------------------
// The pencil beside the mode dropdown (eds-07), over a mounted composer.
//
// The card it opens is judged in `agentModesDialog.test.js` and every rule
// behind it in `agentModes.test.js`. What lives here is the *affordance*: that
// the button sits in the mode column next to the select, which states of the
// app make it inert (D10), that it is reachable when the dropdown's list failed
// to load, and that opening the card reads the file and writes nothing.
//
// The composer is mounted for real because enablement is its own composition of
// `useChat`'s phase and `useBatch`'s flag — a rule no composable function owns.
// ---------------------------------------------------------------------------

const agentModes = useAgentModes().state
const chat = useChat().state
const batch = useBatch().state
const benchmark = useBenchmark().state

const DOCUMENT_BODY = {
  entries: [
    { key: 'code', hasForm: true, fields: { label: 'Code' }, findings: [] },
  ],
  default: 'code',
  findings: [],
  source: {
    path: '/work/.nojae/agent-modes.json',
    exists: true,
    loaded: true,
    mtime: '1700000000000000000',
  },
  caps: {
    maxModes: 16,
    maxIdChars: 32,
    idPattern: '^[a-z][A-Za-z0-9_-]*$',
    maxLabelChars: 40,
    maxPersonaBytes: 16384,
    maxPersonaLines: 16,
  },
  defaults: { version: 1, default: 'code', agents: {} },
}

function modesBody() {
  return {
    modes: [
      { id: 'chat', label: 'Chat', mcptools: false, lsptools: false, editable: true },
      { id: 'code', label: 'Code', mcptools: true, lsptools: true, editable: true },
      // The app-owned mode, recognised only by its flag (never by an id this
      // file may not contain) — the one state that locks the *select*, which
      // the pencil must not inherit.
      { id: 'reserved-mode-under-test', label: 'Reserved', mcptools: true, lsptools: false, editable: false },
    ],
    default: 'code',
    source: { path: '/work/.nojae/agent-modes.json', exists: true, loaded: true },
    findings: [],
  }
}

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// `calls` records every route the composer touches, in order. `list: null` makes
// the dropdown's own fetch fail — the case the pencil must survive.
// (`documentBody`, not `document`: the global is what the assertions read.)
function stubRoutes({ list = modesBody(), documentBody = DOCUMENT_BODY, calls = [] } = {}) {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
    const key = String(url)
    calls.push({ key, method: init?.method ?? 'GET' })
    if (key === '/api/agent-modes') {
      return list === null ? jsonResponse({}, false, 500) : jsonResponse(list)
    }
    if (key === '/api/agent-modes/document') return jsonResponse(documentBody)
    if (key === '/api/agent-modes/save') {
      throw new Error('the pencil must not save on its own')
    }
    throw new Error(`unexpected fetch: ${key}`)
  })
  return calls
}

function resetAll() {
  chat.phase = 'idle'
  chat.messages = []
  chat.error = null
  chat.pausePending = false
  chat.pendingImage = null
  chat.mcp = { connectors: [], enabled: {}, loaded: false, locked: false, restarting: false }
  chat.lsp = { servers: [], enabled: {}, loaded: false, locked: false, restarting: false }
  batch.active = false
  batch.status = 'idle'
  benchmark.status = 'idle'
  benchmark.activeTab = 'chat'
  agentModes.modes = []
  agentModes.current = null
  agentModes.default = null
  agentModes.loaded = false
  agentModes.locked = false
  agentModes.findings = []
  agentModes.source = null
  agentModes.notice = null
  const editor = useAgentModes().editor
  editor.open = false
  editor.loading = false
  editor.saving = false
  editor.error = null
  editor.conflict = null
  editor.invalidFindings = []
}

let host = null

async function mountComposer() {
  host = document.createElement('div')
  document.body.appendChild(host)
  const wrapper = mount(WorkBenchInput, { attachTo: host })
  await flushPromises()
  return wrapper
}

function pencil(w) {
  return w.find('.chat-mode-edit')
}

beforeEach(() => {
  // jsdom has no <dialog> methods; the card's watch only reads `el.open`.
  Element.prototype.showModal = function showModal() {
    this.open = true
  }
  Element.prototype.close = function close() {
    this.open = false
  }
  resetAll()
})

afterEach(() => {
  host?.remove()
  host = null
  document.body.querySelectorAll('dialog').forEach((el) => el.remove())
  vi.restoreAllMocks()
})

describe('the pencil beside the mode dropdown', () => {
  it('renders in the mode column, immediately after the select', async () => {
    stubRoutes()
    await loadAgentModes()
    const w = await mountComposer()

    const select = w.find('.chat-mode-select')
    const button = pencil(w)
    expect(select.exists()).toBe(true)
    expect(button.exists()).toBe(true)
    // The pencil glyph the app already draws (the "Edit and resend" chip).
    expect(button.find('svg').exists()).toBe(true)
    // Same row, and the pencil is the select's next sibling: the layout promise
    // the e2e spec measures is "the pencil joined the column, nothing moved".
    const row = w.find('.chat-mode-row')
    expect(row.exists()).toBe(true)
    expect(row.element.children[0]).toBe(select.element)
    expect(row.element.children[1]).toBe(button.element)
    // Named, so the affordance says what it opens.
    expect(button.attributes('aria-label')).toContain('modes')
    expect(button.attributes('title')).toContain('modes')
    // …but not with the *select's* name: a query for "Agent mode" must keep
    // answering one element, or the pinned layout spec would start matching the
    // pencil too.
    expect(select.attributes('aria-label')).toBe('Agent mode')
    for (const name of [button.attributes('aria-label'), button.attributes('title')]) {
      expect(name.toLowerCase()).not.toContain('agent mode')
    }
  })

  it('opens the dialog, which reads the file and writes nothing', async () => {
    const calls = stubRoutes()
    await loadAgentModes()
    const w = await mountComposer()

    expect(document.querySelector('dialog.agent-modes-dialog')?.open ?? false).toBe(false)
    await pencil(w).trigger('click')
    await flushPromises()

    expect(document.querySelector('dialog.agent-modes-dialog')?.open).toBe(true)
    // Opening *is* the document read; no save happens without pressing Save.
    expect(calls.map((call) => call.key)).toEqual([
      '/api/agent-modes',
      '/api/agent-modes/document',
    ])
    expect(calls.some((call) => call.method === 'POST')).toBe(false)
  })

  it('is inert while a reply streams, and live otherwise', async () => {
    stubRoutes()
    await loadAgentModes()
    const w = await mountComposer()
    expect(pencil(w).attributes('disabled')).toBeUndefined()

    chat.phase = 'answering'
    await flushPromises()
    expect(pencil(w).attributes('disabled')).toBeDefined()
    // A click in that window opens nothing: the card's Save would be refused.
    await pencil(w).trigger('click')
    await flushPromises()
    expect(document.querySelector('dialog.agent-modes-dialog')?.open ?? false).toBe(false)

    chat.phase = 'idle'
    await flushPromises()
    expect(pencil(w).attributes('disabled')).toBeUndefined()
  })

  it('is inert while a lot runs, and live after it', async () => {
    stubRoutes()
    await loadAgentModes()
    const w = await mountComposer()

    batch.active = true
    await flushPromises()
    expect(pencil(w).attributes('disabled')).toBeDefined()
    // The select obeys the same rule: one live and the other not is the drift
    // this task is told to avoid.
    expect(w.find('.chat-mode-select').attributes('disabled')).toBeDefined()

    batch.active = false
    await flushPromises()
    expect(pencil(w).attributes('disabled')).toBeUndefined()
    expect(w.find('.chat-mode-select').attributes('disabled')).toBeUndefined()
  })

  it('stays reachable when the dropdown list failed to load', async () => {
    const calls = stubRoutes({ list: null })
    await loadAgentModes()
    expect(agentModes.loaded).toBe(false)
    const w = await mountComposer()

    // No select (a control that cannot be used is worse than none), but the
    // editor is there: a broken modes file is exactly the one worth opening.
    expect(w.find('.chat-mode-select').exists()).toBe(false)
    expect(pencil(w).exists()).toBe(true)
    expect(pencil(w).attributes('disabled')).toBeUndefined()

    await pencil(w).trigger('click')
    await flushPromises()
    expect(document.querySelector('dialog.agent-modes-dialog')?.open).toBe(true)
    expect(calls.some((call) => call.key === '/api/agent-modes/document')).toBe(true)
  })

  it('a locked conversation still lets the file be edited', async () => {
    // The pencil's window is *streaming* and *lot*, not the select's whole
    // predicate: the D7 lock is about this conversation's persona, and the
    // workspace's modes file is not the conversation's. D9's
    // `resetDecisionAfterSave` exists precisely because a save can land while
    // the selection is locked — it leaves the selection alone and says so.
    stubRoutes()
    await loadAgentModes()
    const w = await mountComposer()

    const reserved = agentModes.modes.find((mode) => mode.editable === false)
    expect(reserved, 'the fixture needs the non-editable mode').toBeTruthy()
    // The lock is a function of the thread (the probe useChat injects), so the
    // thread itself is what moves here — no hand-set flag.
    chat.messages = [{ role: 'user', raw: 'one turn', html: '', images: [] }]
    setAgentMode(reserved.id)
    await flushPromises()
    expect(agentModes.locked).toBe(true)

    // The select is inert (the lock refuses a switch); the pencil is not.
    expect(w.find('.chat-mode-select').attributes('disabled')).toBeDefined()
    expect(pencil(w).attributes('disabled')).toBeUndefined()

    await pencil(w).trigger('click')
    await flushPromises()
    expect(document.querySelector('dialog.agent-modes-dialog')?.open).toBe(true)
  })

  it('the notice line still renders under the column, pencil included', async () => {
    // D9's quiet line — a skipped reset arrives through the same notice slot,
    // so the column must render for a notice with no list behind it.
    stubRoutes({ list: null })
    agentModes.notice =
      'The agent-modes file was saved — the selection did not move: a running lot picks its mode at launch.'
    const w = await mountComposer()

    const line = w.find('.chat-mode-findings')
    expect(line.exists()).toBe(true)
    expect(line.text()).toContain('did not move')
    expect(pencil(w).exists()).toBe(true)
  })
})
