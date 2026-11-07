import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

import {
  addModeToDraft,
  agentModePayloadId,
  agentModesLoaded,
  applySessionAgentMode,
  blockingFindings,
  createAgentModesDraft,
  createDisabledFor,
  currentMode,
  deleteDisabledFor,
  deleteModeFromDraft,
  draftIsDirty,
  draftWillCreateFile,
  fetchAgentModesDocument,
  findingsLine,
  idEditableFor,
  idValidityProblem,
  loadAgentModes,
  modeBlocksHub,
  modeCountProblem,
  modeHint,
  modeOptionDisabled,
  modeSelectDisabled,
  modeSelectTitle,
  personaByteCount,
  personaLineBudget,
  personaOverByteCap,
  resetDecisionAfterSave,
  saveAgentModesDocument,
  saveDisabledFor,
  setAgentMode,
  slugForDraft,
  slugForModeId,
  syncAgentModeLock,
  useAgentModes,
} from '../src/composables/useAgentModes'
import WorkBenchMcpPanel from '../src/components/WorkBenchMcpPanel.vue'
import {
  SESSION_FILE_VERSION,
  conversationJson,
  parseSessionFile,
  useChat,
} from '../src/composables/useChat'
import { useBatch } from '../src/composables/useBatch'
import { useBenchmark } from '../src/composables/useBenchmark'

// Layer 1 (jsdom): the mode dropdown's state machine. What is asserted here is
// the composable's decisions — the default it seeds, the payload key it adds,
// the D7 lock, the hub gate and the session envelope — because every one of
// them is a function in `useAgentModes.js` and no `.vue` re-derives it. The
// stubbed payloads mirror what `GET /api/agent-modes` answers (am-06); the
// fetch harness is the one batchRetries.test.js uses for the chat routes.

const agentModes = useAgentModes().state
const chat = useChat().state
const batch = useBatch().state
const benchmark = useBenchmark().state

// The app-owned mode is recognised only by `editable: false` (D9), so this file
// calls it "the reserved mode" and never names it — the same rule the client
// code is held to (no such id anywhere in `web/src`).
const RESERVED_ID = 'reserved-mode-under-test'

function modesBody({ defaultId = 'code', findings = [], path } = {}) {
  return {
    modes: [
      { id: 'chat', label: 'Chat', mcptools: false, lsptools: false, editable: true },
      { id: 'code', label: 'Code', mcptools: true, lsptools: true, editable: true },
      { id: 'architect', label: 'Architect', mcptools: true, lsptools: false, editable: true },
      { id: RESERVED_ID, label: 'Reserved', mcptools: true, lsptools: false, editable: false },
    ],
    default: defaultId,
    source: {
      path: path ?? '/work/.nojae/agent-modes.json',
      exists: true,
      loaded: true,
      version: 1,
    },
    findings,
  }
}

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

function chatDoneFrame() {
  return 'event: done\ndata: {"tokens":12,"took":1.5,"ttft":0.4,"decode":1.1}\n\n'
}

// A body useChat's reader loop accepts: one chunk holding the frame, then the
// stream ends (jsdom has no ReadableStream, so this is the minimal shape).
function sseBody(text) {
  const bytes = new TextEncoder().encode(text)
  let delivered = false
  return {
    getReader() {
      return {
        async read() {
          if (delivered) return { done: true, value: undefined }
          delivered = true
          return { done: false, value: bytes }
        },
      }
    },
  }
}

// The routes a conversation touches. `modes: null` makes the list fetch fail,
// and every posted /api/chat body lands in `sent`.
function stubRoutes({ modes = modesBody(), sent = [] } = {}) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
    const key = String(url)
    if (key.startsWith('/api/agent-modes')) {
      return modes === null ? jsonResponse({}, false, 500) : jsonResponse(modes)
    }
    if (key.startsWith('/api/chat')) {
      sent.push(JSON.parse(init.body))
      return { ok: true, body: sseBody(chatDoneFrame()) }
    }
    throw new Error(`unexpected fetch: ${key}`)
  })
}

function resetAll() {
  chat.phase = 'idle'
  chat.messages = []
  chat.error = null
  chat.pausePending = false
  chat.session.startedAt = 0
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
}

async function typeAndSend(text) {
  const { setDraft, send } = useChat()
  setDraft(text)
  await send()
}

describe('loadAgentModes: seeding and degrading', () => {
  beforeEach(resetAll)

  it('seeds current from the server default', async () => {
    stubRoutes()
    await loadAgentModes()
    expect(agentModes.loaded).toBe(true)
    expect(agentModes.current).toBe('code')
    expect(agentModes.default).toBe('code')
    expect(currentMode().label).toBe('Code')
    expect(agentModePayloadId()).toBe('code')
  })

  it('a failed refresh hides nothing and changes nothing', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('architect')
    stubRoutes({ modes: null })
    await loadAgentModes()
    // `loaded` is false, so send() omits the key and the server's own default
    // applies — what a client predating the feature gets. The selection is not
    // silently downgraded, and the previous list survives for the label.
    expect(agentModes.loaded).toBe(false)
    expect(agentModesLoaded()).toBe(false)
    expect(agentModes.current).toBe('architect')
    expect(agentModePayloadId()).toBeNull()
    expect(modeHint('architect')).toBe('Architect — MCP available')
  })

  it('a workspace switch re-seeds current from the new default', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('architect')
    stubRoutes({ modes: modesBody({ defaultId: 'chat' }) })
    await loadAgentModes({ reseed: true })
    // An id from the old root's file may not exist in the new one; the
    // alternative to re-seeding would be a 422 on every send.
    expect(agentModes.current).toBe('chat')
    expect(agentModes.default).toBe('chat')
  })

  it('a plain refresh keeps a choice the list still offers', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('chat')
    await loadAgentModes()
    expect(agentModes.current).toBe('chat')
  })

  it('findings become one quiet line naming the file', async () => {
    stubRoutes({
      modes: modesBody({
        findings: [
          { kind: 'error', message: 'agents.x: unknown key' },
          { kind: 'note', message: 'agents.y: trimmed label' },
        ],
      }),
    })
    await loadAgentModes()
    expect(findingsLine()).toBe(
      '2 problems in the workspace agent-modes file — see /work/.nojae/agent-modes.json',
    )
  })
})

describe('the payload of a real send', () => {
  beforeEach(resetAll)

  it('carries agentMode with the selected id', async () => {
    const sent = []
    stubRoutes({ sent })
    await loadAgentModes()
    setAgentMode('chat')
    await typeAndSend('hello')
    expect(sent).toHaveLength(1)
    expect(sent[0].agentMode).toBe('chat')
    // The hub allowlists keep their own omission rule (their snapshots never
    // loaded here), so the mode key did not drag them along.
    expect('mcpEnabled' in sent[0]).toBe(false)
  })

  it('sends no agentMode key at all when the list failed to load', async () => {
    const sent = []
    stubRoutes({ sent, modes: null })
    await loadAgentModes()
    await typeAndSend('hello')
    expect(sent).toHaveLength(1)
    expect('agentMode' in sent[0]).toBe(false)
    // The composer behaves exactly as it did before the feature: the request is
    // otherwise identical.
    expect(sent[0].messages).toEqual([{ role: 'user', content: 'hello' }])
  })
})

describe('the D7 lock', () => {
  beforeEach(resetAll)

  it('every editable mode is switchable at any time', async () => {
    stubRoutes()
    await loadAgentModes()
    syncAgentModeLock(true) // the thread has a turn
    for (const id of ['chat', 'code', 'architect']) {
      setAgentMode(id)
      expect(agentModes.current).toBe(id)
      expect(agentModes.locked).toBe(false)
    }
  })

  it('the reserved mode is enterable, then not leaveable once the thread turned', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode(RESERVED_ID)
    expect(agentModes.current).toBe(RESERVED_ID)
    expect(agentModes.locked).toBe(false) // still no turn on this thread
    syncAgentModeLock(true)
    expect(agentModes.locked).toBe(true)
    setAgentMode('code')
    expect(agentModes.current).toBe(RESERVED_ID)
    // The other options are disabled; the current one stays selectable, so the
    // select keeps reading correctly.
    expect(modeOptionDisabled('code')).toBe(true)
    expect(modeOptionDisabled(RESERVED_ID)).toBe(false)
  })

  it('entering it on a thread that already turned locks at once', async () => {
    stubRoutes()
    await loadAgentModes()
    // The turn count comes from useChat (the injected probe), so the thread
    // itself is what this test moves — no hand-set flag.
    chat.messages = [{ role: 'user', raw: 'one turn', html: '', images: [] }]
    setAgentMode(RESERVED_ID)
    expect(agentModes.current).toBe(RESERVED_ID)
    expect(agentModes.locked).toBe(true)
  })

  it('unknown ids are refused, so the select can never invent a mode', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('nope')
    expect(agentModes.current).toBe('code')
  })

  it('a reset unlocks: a cleared thread has no turns', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode(RESERVED_ID)
    syncAgentModeLock(true)
    expect(agentModes.locked).toBe(true)
    useChat().reset({ refetchHubs: false })
    expect(agentModes.locked).toBe(false)
    setAgentMode('code')
    expect(agentModes.current).toBe('code')
  })

  it('a real send locks it — useChat owns the turn count, not this test', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode(RESERVED_ID)
    expect(agentModes.locked).toBe(false)
    await typeAndSend('draft the lot')
    expect(chat.messages.length).toBeGreaterThan(0)
    expect(agentModes.locked).toBe(true)
    setAgentMode('chat')
    expect(agentModes.current).toBe(RESERVED_ID)
  })
})

describe('the hub gate (D4)', () => {
  beforeEach(resetAll)

  it('each hub is off exactly for the modes whose flag is false', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('chat')
    expect(modeBlocksHub('mcp')).toBe(true)
    expect(modeBlocksHub('lsp')).toBe(true)
    setAgentMode('architect')
    expect(modeBlocksHub('mcp')).toBe(false)
    expect(modeBlocksHub('lsp')).toBe(true)
    setAgentMode('code')
    expect(modeBlocksHub('mcp')).toBe(false)
    expect(modeBlocksHub('lsp')).toBe(false)
  })

  it('no gate at all when the list never loaded', async () => {
    stubRoutes({ modes: null })
    await loadAgentModes()
    expect(currentMode()).toBeNull()
    expect(modeBlocksHub('mcp')).toBe(false)
    expect(modeBlocksHub('lsp')).toBe(false)
  })

  it('a stale list gates nothing either — the payload carries no mode', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('chat')
    expect(modeBlocksHub('mcp')).toBe(true)
    // The refresh failed: `send()` omits `agentMode`, so the *server's* default
    // answers the next turn and this client no longer knows what it would gate.
    stubRoutes({ modes: null })
    await loadAgentModes()
    expect(agentModes.current).toBe('chat') // still shown, never downgraded
    expect(modeBlocksHub('mcp')).toBe(false)
    expect(modeBlocksHub('lsp')).toBe(false)
  })

  it('the MCP panel renders its switches inert with a reason, not hidden', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('chat')
    seedMcpSnapshot()
    const wrapper = mount(WorkBenchMcpPanel)
    expect(wrapper.find('.chat-mcp-panel').exists()).toBe(true)
    expect(wrapper.get('input[role="switch"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('.chat-hub-gate').text()).toContain('Chat mode')
    wrapper.unmount()
  })

  it('the same panel is live for a mode that opens the hub', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('code')
    seedMcpSnapshot()
    const wrapper = mount(WorkBenchMcpPanel)
    expect(wrapper.find('.chat-hub-gate').exists()).toBe(false)
    expect(wrapper.get('input[role="switch"]').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })
})

function seedMcpSnapshot() {
  chat.mcp.loaded = true
  chat.mcp.connectors = [{ id: 'fedit', label: 'File editor', status: 'running' }]
  chat.mcp.enabled = { fedit: true }
}

describe('the mode hint', () => {
  beforeEach(resetAll)

  it('names the hubs the mode can reach — the UI sentence about gating', async () => {
    stubRoutes()
    await loadAgentModes()
    expect(modeHint('code')).toBe('Code — MCP and language servers available')
    expect(modeHint('chat')).toBe('Chat — no tools')
    expect(modeHint('architect')).toBe('Architect — MCP available')
  })

  it('the select explains the D7 lock and the batch pin, in that order', async () => {
    stubRoutes()
    await loadAgentModes()
    expect(modeSelectTitle()).toBe('Code — MCP and language servers available')
    batch.active = true
    expect(modeSelectTitle({ batchActive: true })).toContain(
      'a running batch picks its own mode',
    )
    batch.active = false
    setAgentMode(RESERVED_ID)
    chat.messages = [{ role: 'user', raw: 'one turn', html: '', images: [] }]
    syncAgentModeLock(true)
    expect(modeSelectTitle({ batchActive: false })).toContain('fixed once it has a turn')
  })
})

describe('the session envelope', () => {
  beforeEach(resetAll)

  it('saves the mode and reads it back', async () => {
    stubRoutes()
    await loadAgentModes()
    setAgentMode('architect')
    chat.messages = [
      { role: 'user', raw: 'plan it', html: '', images: [] },
      { role: 'assistant', raw: 'here is the spec', html: '', images: [] },
    ]
    const doc = conversationJson(chat.messages)
    expect(doc.version).toBe(SESSION_FILE_VERSION)
    expect(doc.agentMode).toBe('architect')
    const parsed = parseSessionFile(JSON.stringify(doc))
    expect(parsed.agentMode).toBe('architect')
    applySessionAgentMode(parsed.agentMode)
    expect(agentModes.current).toBe('architect')
    expect(agentModes.notice).toBeNull()
  })

  it('a v3 file parses with no agentMode and keeps the workspace default', async () => {
    stubRoutes()
    await loadAgentModes()
    const v3 = {
      format: 'benchmarkit-chat-session',
      version: 3,
      messages: [{ role: 'user', content: 'older conversation' }],
    }
    const parsed = parseSessionFile(JSON.stringify(v3))
    expect(parsed.agentMode).toBeNull()
    applySessionAgentMode(parsed.agentMode)
    expect(agentModes.current).toBe('code')
    expect(agentModes.notice).toBeNull()
  })

  it('a mode unknown to this workspace falls back to the default, out loud', async () => {
    stubRoutes()
    await loadAgentModes()
    applySessionAgentMode('reviewer')
    expect(agentModes.current).toBe('code')
    expect(agentModes.notice).toContain('"reviewer"')
  })

  it('a saved mode with no list to judge it against is still said out loud', async () => {
    stubRoutes({ modes: null })
    await loadAgentModes()
    applySessionAgentMode('code')
    expect(agentModes.current).toBeNull() // nothing was ever seeded
    expect(agentModes.notice).toContain('could not be read')
    // And the payload keeps omitting the key, so the server decides.
    expect(agentModePayloadId()).toBeNull()
  })

  it('a restored thread in the reserved mode restores the lock', async () => {
    stubRoutes()
    await loadAgentModes()
    chat.messages = [{ role: 'user', raw: 'x', html: '', images: [] }]
    applySessionAgentMode(RESERVED_ID)
    expect(agentModes.current).toBe(RESERVED_ID)
    expect(agentModes.locked).toBe(true)
  })

  it('a save with no loaded list carries no agentMode key', () => {
    chat.messages = [{ role: 'user', raw: 'x', html: '', images: [] }]
    const doc = conversationJson(chat.messages)
    expect('agentMode' in doc).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// The modes editor (eds-05): the dialog's rules over a draft document.
//
// These are Layer-1 tests on purpose: every one judges a *draft* — the editable
// copy of what `GET /api/agent-modes/document` answers — through the plain
// functions in useAgentModes.js, because that is exactly what the dialog
// component (eds-06) will render instead of re-deriving. The fixtures mirror
// the real payload shape from `src/core/agent_modes.read_agent_modes_document`.
// ---------------------------------------------------------------------------

// The caps as the document payload publishes them. No test here hardcodes a
// rule *number* except through this object, and one case patches it to prove
// the verdict follows the payload rather than a literal.
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

function editorEntry(key, fields = {}, extra = {}) {
  return { key, hasForm: true, fields, findings: [], ...extra }
}

function documentBody({
  entries = [
    editorEntry('chat', EDITOR_DEFAULTS.agents.chat),
    editorEntry('code', EDITOR_DEFAULTS.agents.code),
    editorEntry('architect', EDITOR_DEFAULTS.agents.architect),
  ],
  defaultId = 'code',
  findings = [],
  caps = EDITOR_CAPS,
  defaults = EDITOR_DEFAULTS,
  source = {},
  readme,
} = {}) {
  const body = {
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
  if (readme !== undefined) body.readme = readme
  return body
}

// What `POST /api/agent-modes/save` answers: the new verdict in the dropdown's
// own shape plus the new mtime token.
function saveResponse({ defaultId = 'code', modes, mtime = '1700000009000000000' } = {}) {
  const base = modesBody({ defaultId })
  return {
    saved: true,
    created: false,
    backedUp: true,
    modes: modes ?? base.modes,
    default: defaultId,
    findings: [],
    source: { ...base.source, mtime },
  }
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

// A fetch stub for the whole agent-modes surface: the dropdown's reader (so a
// test can load the list first) and the two editor routes. Only the **editor**
// pair lands in `calls`, in order, so a test can assert "no second round-trip"
// instead of inferring it.
function stubEditorRoutes({ document = documentBody(), save = saveResponse(), calls = [] } = {}) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
    const key = String(url)
    if (key === '/api/agent-modes') {
      return jsonResponse(modesBody())
    }
    if (key === '/api/agent-modes/document') {
      calls.push({ key, init })
      return document === null ? jsonResponse({}, false, 500) : jsonResponse(document)
    }
    if (key === '/api/agent-modes/save') {
      calls.push({ key, init })
      if (save === null) return jsonResponse({}, false, 500)
      if (typeof save === 'object' && save.__fail) {
        return jsonResponse({ detail: save.detail }, false, save.status)
      }
      return jsonResponse(save)
    }
    throw new Error(`unexpected fetch: ${key}`)
  })
}

describe('the draft document', () => {
  beforeEach(() => {
    resetAll()
    resetEditor()
  })

  it('holds the file entries in the file order, persona verbatim', () => {
    const draft = createAgentModesDraft(
      documentBody({
        entries: [
          editorEntry('architect', { label: 'Architect', mcptools: true, systemMessage: ['a\nb'] }),
          editorEntry('chat', { label: 'Chat', systemMessage: null }),
        ],
        defaultId: 'architect',
      }),
    )
    expect(draft.entries.map((entry) => entry.key)).toEqual(['architect', 'chat'])
    // A list persona joins the way the loader joins it, so the round-trip is
    // byte-identical; a null one is "use the configured prompt" — empty text.
    expect(draft.entries[0].personaText).toBe('a\nb')
    expect(draft.entries[1].personaText).toBe('')
    expect(draft.entries[1].personaNull).toBe(true)
    expect(draft.default).toBe('architect')
    expect(draft.mtime).toBe('1700000000000000000')
    expect(draftIsDirty(draft)).toBe(false)
  })

  it('an absent file drafts from the built-in document and says saving creates it', () => {
    const draft = createAgentModesDraft(
      documentBody({ entries: [], defaultId: null, source: { exists: false, loaded: false, mtime: null } }),
    )
    expect(draft.entries.map((entry) => entry.key)).toEqual(['chat', 'code', 'architect'])
    expect(draft.default).toBe('code')
    expect(draft.mtime).toBeNull()
    expect(draftWillCreateFile(draft)).toBe(true)
  })

  it('a file-level finding is carried, and the draft is not silently emptied', () => {
    const draft = createAgentModesDraft(
      documentBody({
        entries: [],
        defaultId: null,
        findings: [{ kind: 'error', message: 'The agent-modes file /work/x is not valid JSON: boom (line 1, column 2)' }],
        source: { exists: true, loaded: false },
      }),
    )
    expect(draft.entries).toEqual([])
    expect(draft.findings).toHaveLength(1)
    expect(saveDisabledFor(draft)).toBe(true)
  })
})

describe('slugForModeId', () => {
  it('from a plain label', () => {
    expect(slugForModeId('Reviewer', [], EDITOR_CAPS)).toBe('reviewer')
  })

  it('from a label with punctuation', () => {
    expect(slugForModeId('Docs Review (v2)!', [], EDITOR_CAPS)).toBe('docs-review-v2')
    expect(slugForModeId('  Spaced   Out  ', [], EDITOR_CAPS)).toBe('spaced-out')
  })

  it('colliding with an existing id gains a suffix', () => {
    expect(slugForModeId('Reviewer', ['reviewer'], EDITOR_CAPS)).toBe('reviewer-2')
    expect(slugForModeId('Reviewer', ['reviewer', 'reviewer-2'], EDITOR_CAPS)).toBe('reviewer-3')
  })

  it('stays within the payload cap, suffix included', () => {
    const long = 'x'.repeat(80)
    const slug = slugForModeId(long, [], EDITOR_CAPS)
    expect(slug.length).toBeLessThanOrEqual(EDITOR_CAPS.maxIdChars)
    // A collision at the cap cannot push the id past it: the base gives up its
    // tail for the suffix.
    const taken = [slug]
    const next = slugForModeId(long, taken, EDITOR_CAPS)
    expect(next.length).toBeLessThanOrEqual(EDITOR_CAPS.maxIdChars)
    expect(taken).not.toContain(next)
  })

  it('follows a patched cap, not a literal', () => {
    const caps = { ...EDITOR_CAPS, maxIdChars: 8 }
    expect(slugForModeId('Reviewer Of Docs', [], caps)).toHaveLength(8)
  })

  it('produces an id the payload pattern accepts, even from a label that cannot start one', () => {
    const caps = EDITOR_CAPS
    for (const label of ['2 Fast', 'Ünicode Näme', '', '---']) {
      const slug = slugForModeId(label, [], caps)
      expect(slug, `label ${JSON.stringify(label)}`).toMatch(new RegExp(caps.idPattern))
    }
  })

  it('slugForDraft judges against the ids the draft already holds', () => {
    const draft = createAgentModesDraft(documentBody())
    expect(slugForDraft(draft, 'Code')).toBe('code-2')
  })
})

describe('the id field (D5)', () => {
  beforeEach(() => {
    resetAll()
    resetEditor()
  })

  it('an existing mode id is read-only while a new one is editable', () => {
    const draft = createAgentModesDraft(documentBody())
    expect(idEditableFor(draft.entries[0])).toBe(false)
    const created = addModeToDraft(draft, 'Reviewer')
    expect(idEditableFor(created)).toBe(true)
  })

  it('a new id is judged against the pattern, the cap and the other ids', () => {
    const draft = createAgentModesDraft(documentBody())
    const created = addModeToDraft(draft, 'Reviewer')
    expect(idValidityProblem(draft, created)).toBeNull()
    created.key = 'Reviewer' // uppercase: the pattern wants a lowercase start
    expect(idValidityProblem(draft, created)).toContain('must match')
    created.key = 'x'.repeat(33)
    expect(idValidityProblem(draft, created)).toContain('at most 32')
    created.key = 'code'
    expect(idValidityProblem(draft, created)).toContain('already used')
    created.key = 'reviewer'
    expect(idValidityProblem(draft, created)).toBeNull()
  })

  it('an existing entry is never judged by the id rule — its field is read-only', () => {
    const draft = createAgentModesDraft(documentBody())
    expect(idValidityProblem(draft, draft.entries[1])).toBeNull()
  })
})

describe('the persona counters', () => {
  beforeEach(() => {
    resetAll()
    resetEditor()
  })

  it('counts UTF-8 bytes, not characters', () => {
    expect(personaByteCount('ascii')).toBe(5)
    expect(personaByteCount('héllo wörld')).toBe(13) // two 2-byte letters
    expect(personaByteCount('日本語')).toBe(9) // three 3-byte letters
    expect(personaByteCount('')).toBe(0)
  })

  it('over the byte cap is invalid, and the verdict follows the payload', () => {
    const draft = createAgentModesDraft(documentBody())
    const text = 'a'.repeat(16385)
    expect(personaOverByteCap(draft, text)).toBe(true)
    draft.entries[0].personaText = text
    expect(saveDisabledFor(draft)).toBe(true)
    expect(blockingFindings(draft)[0].message).toContain('agents.chat.systemMessage')

    // Patch the cap in the payload: the same text is fine under a bigger one,
    // so the rule is the payload's number and not a literal in this file.
    const roomy = createAgentModesDraft(
      documentBody({ caps: { ...EDITOR_CAPS, maxPersonaBytes: 32768 } }),
    )
    roomy.entries[0].personaText = text
    expect(personaOverByteCap(roomy, text)).toBe(false)
    expect(saveDisabledFor(roomy)).toBe(false)
  })

  it('the line budget warns and does not block Save', () => {
    const draft = createAgentModesDraft(documentBody())
    const text = Array.from({ length: 20 }, (_unused, i) => `line ${i}`).join('\n')
    const budget = personaLineBudget(draft, text)
    expect(budget.budget).toBe(16)
    expect(budget.lines).toBe(20)
    expect(budget.over).toBe(true)
    expect(budget.remaining).toBe(0)
    expect(budget.warning).toContain('16-line')
    draft.entries[0].personaText = text
    expect(saveDisabledFor(draft)).toBe(false) // soft: it warns, it never blocks
  })

  it('within the budget there is nothing to warn about', () => {
    const draft = createAgentModesDraft(documentBody())
    const budget = personaLineBudget(draft, 'one\nthree')
    expect(budget.over).toBe(false)
    expect(budget.warning).toBeNull()
    expect(budget.remaining).toBe(14)
  })
})

describe('blockingFindings (D3)', () => {
  beforeEach(() => {
    resetAll()
    resetEditor()
  })

  it('an entry error blocks Save', () => {
    const draft = createAgentModesDraft(
      documentBody({
        entries: [
          editorEntry('code', EDITOR_DEFAULTS.agents.code, {
            findings: [{ kind: 'error', message: 'agents.code.mcptool: unknown key (expected label, mcptools, lsptools, systemMessage)' }],
          }),
        ],
        defaultId: 'code',
      }),
    )
    expect(blockingFindings(draft).map((finding) => finding.message)).toEqual([
      'agents.code.mcptool: unknown key (expected label, mcptools, lsptools, systemMessage)',
    ])
    expect(saveDisabledFor(draft)).toBe(true)
  })

  it('a file-level error blocks Save', () => {
    const draft = createAgentModesDraft(
      documentBody({
        findings: [{ kind: 'error', message: 'The agent-modes file /work/x "version" is 2; this benchmarkit understands version 1 only' }],
      }),
    )
    expect(blockingFindings(draft)).toHaveLength(1)
    expect(saveDisabledFor(draft)).toBe(true)
  })

  it('a note does not block Save', () => {
    const draft = createAgentModesDraft(
      documentBody({
        entries: [
          editorEntry('code', EDITOR_DEFAULTS.agents.code),
          editorEntry('x', EDITOR_DEFAULTS.agents.chat, {
            findings: [{ kind: 'note', message: 'agents.x: "x" is defined by benchmarkit and cannot be set here; this entry is ignored' }],
          }),
        ],
      }),
    )
    expect(blockingFindings(draft)).toEqual([])
    expect(saveDisabledFor(draft)).toBe(false)
  })

  it('an entry with no form blocks until it is deleted', () => {
    const draft = createAgentModesDraft(
      documentBody({
        entries: [
          editorEntry('code', EDITOR_DEFAULTS.agents.code),
          editorEntry('broken', {}, { hasForm: false, fields: {}, findings: [{ kind: 'error', message: 'agents.broken: is a string, not an object with label, mcptools, lsptools, systemMessage' }] }),
        ],
      }),
    )
    expect(saveDisabledFor(draft)).toBe(true)
    expect(deleteModeFromDraft(draft, 'broken')).toBe(true)
    expect(saveDisabledFor(draft)).toBe(false)
  })

  it('a finding the form can only produce validly for is resolved by the draft', () => {
    const draft = createAgentModesDraft(
      documentBody({
        entries: [
          editorEntry('code', { label: 'Code', mcptools: true }, {
            findings: [
              { kind: 'error', message: 'agents.code.label: is a number, not a string' },
              { kind: 'error', message: 'agents.code.mcptools: is a string, not a boolean: write "mcptools": true or "mcptools": false' },
            ],
          }),
        ],
      }),
    )
    // The draft holds a string and a boolean for both — the shapes the finding
    // named no longer exist, so the entry is fixed.
    expect(blockingFindings(draft)).toEqual([])
  })

  it('a persona over the cap blocks even when the file said it was fine', () => {
    const draft = createAgentModesDraft(documentBody())
    draft.entries[1].personaText = 'b'.repeat(EDITOR_CAPS.maxPersonaBytes + 1)
    expect(saveDisabledFor(draft)).toBe(true)
    draft.entries[1].personaText = 'short'
    expect(saveDisabledFor(draft)).toBe(false)
  })

  it('a refused file default is answered by the Common tab, not by Save', () => {
    const draft = createAgentModesDraft(
      documentBody({
        defaultId: 'gone', // the file's own value, naming nothing it declares
        findings: [{ kind: 'error', message: 'The agent-modes file /work/x "default" is "gone", which is not one of the modes it declares' }],
      }),
    )
    // The refused value is never adopted as a choice; the draft starts on a
    // default that *is* declared, so the finding is already answered and does
    // not hold up the Save the user came here to make.
    expect(draft.default).toBe('code')
    expect(saveDisabledFor(draft)).toBe(false)
  })

  it('a default naming no declared mode blocks Save', () => {
    const draft = createAgentModesDraft(documentBody())
    draft.default = 'gone'
    expect(saveDisabledFor(draft)).toBe(true)
    expect(blockingFindings(draft)[0].message).toContain('must name one of the modes')
  })

  it('the mode cap blocks past the cap, and Create is inert at it', () => {
    const entries = Array.from({ length: 16 }, (_unused, i) =>
      editorEntry(`mode${i}`, { label: `Mode ${i}`, mcptools: false, lsptools: false }),
    )
    const atCap = createAgentModesDraft(documentBody({ entries, defaultId: 'mode0' }))
    expect(modeCountProblem(atCap)).toBeNull()
    expect(saveDisabledFor(atCap)).toBe(false)
    expect(createDisabledFor(atCap)).toBe(true) // the cap is felt at the button, not at Save

    entries.push(editorEntry('mode16', { label: 'Mode 16' }))
    const over = createAgentModesDraft(
      documentBody({
        entries,
        defaultId: 'mode0',
        findings: [{ kind: 'error', message: 'agents: declares more than the 16 mode(s) this file may hold' }],
      }),
    )
    expect(modeCountProblem(over)).toContain('17 modes')
    expect(saveDisabledFor(over)).toBe(true)
  })

  it('a label over the label cap is not a form error — the server trims it', () => {
    const draft = createAgentModesDraft(
      documentBody({
        entries: [editorEntry('code', { ...EDITOR_DEFAULTS.agents.code, label: 'L'.repeat(60) })],
      }),
    )
    expect(blockingFindings(draft)).toEqual([])
    expect(saveDisabledFor(draft)).toBe(false)
  })

  it('a draft with no entries at all is not blocked by the default rule', () => {
    const draft = createAgentModesDraft(
      documentBody({ entries: [], defaultId: null, source: { exists: true, loaded: true } }),
    )
    expect(draft.default).toBe('code') // the built-in document's own default
    expect(saveDisabledFor(draft)).toBe(false)
  })
})

describe('dirty tracking', () => {
  beforeEach(() => {
    resetAll()
    resetEditor()
  })

  it('an untouched draft is clean, a touched one is dirty', () => {
    const draft = createAgentModesDraft(documentBody())
    expect(draftIsDirty(draft)).toBe(false)
    draft.entries[0].label = 'Chatting'
    expect(draftIsDirty(draft)).toBe(true)
  })

  it('Create makes the draft dirty without writing anything', () => {
    stubEditorRoutes()
    const draft = createAgentModesDraft(documentBody())
    addModeToDraft(draft, 'Reviewer')
    expect(draftIsDirty(draft)).toBe(true)
    expect(draft.entries).toHaveLength(4)
    expect(fetch.mock.calls).toHaveLength(0) // nothing is written until Save
  })

  it('Delete is refused for the mode that is the default', () => {
    const draft = createAgentModesDraft(documentBody())
    expect(deleteDisabledFor(draft, 'code')).toBe(true)
    expect(deleteModeFromDraft(draft, 'code')).toBe(false)
    expect(deleteDisabledFor(draft, 'chat')).toBe(false)
    expect(deleteModeFromDraft(draft, 'chat')).toBe(true)
  })
})

describe('the two calls', () => {
  beforeEach(() => {
    resetAll()
    resetEditor()
  })

  it('the document read answers a draft and touches nothing else', async () => {
    stubEditorRoutes()
    await loadAgentModes() // the dropdown's own list
    const before = { ...agentModes }
    const draft = await fetchAgentModesDocument()
    expect(draft.entries.map((entry) => entry.key)).toEqual(['chat', 'code', 'architect'])
    expect(agentModes.modes).toEqual(before.modes)
    expect(agentModes.current).toBe(before.current)
  })

  it('the save body is D6 shape: no version, no readme, entries in draft order', async () => {
    const calls = []
    stubEditorRoutes({ save: saveResponse(), calls })
    const draft = await fetchAgentModesDocument()
    addModeToDraft(draft, 'Reviewer')
    draft.default = 'reviewer'
    await saveAgentModesDocument(draft)
    expect(calls.map((call) => call.key)).toEqual([
      '/api/agent-modes/document',
      '/api/agent-modes/save',
    ])
    const body = JSON.parse(calls[1].init.body)
    expect('version' in body.document).toBe(false)
    expect('readme' in body.document).toBe(false)
    expect(Object.keys(body.document.agents)).toEqual(['chat', 'code', 'architect', 'reviewer'])
    expect(body.document.default).toBe('reviewer')
    expect(body.mtime).toBe('1700000000000000000') // the token the read returned
    // A new mode with no persona says nothing about the persona at all, and an
    // existing one keeps its text — as the textarea holds it (a string): the
    // writer is what chooses the JSON form (D6), not the client.
    expect('systemMessage' in body.document.agents.reviewer).toBe(false)
    expect(body.document.agents.chat.systemMessage).toBe('You never call a tool.')
    expect(body.document.agents.architect.lsptools).toBe(false)
  })

  it('a failed document read says so and leaves the dropdown exactly as it was', async () => {
    stubEditorRoutes({ document: null })
    await loadAgentModes() // the dropdown's reader still works
    stubEditorRoutes({ document: null })
    const draft = await fetchAgentModesDocument()
    expect(draft).toBeNull()
    expect(useAgentModes().editor.error).toContain('500')
    expect(agentModes.loaded).toBe(true)
    expect(agentModes.current).toBe('code')
  })

  it('a save reloads the dropdown from its own response — no second round-trip', async () => {
    const calls = []
    const modes = [
      { id: 'chat', label: 'Chat', mcptools: false, lsptools: false, editable: true },
      { id: 'code', label: 'Code', mcptools: true, lsptools: true, editable: true },
      { id: 'reviewer', label: 'Reviewer', mcptools: true, lsptools: false, editable: true },
      { id: RESERVED_ID, label: 'Reserved', mcptools: true, lsptools: false, editable: false },
    ]
    stubEditorRoutes({ save: saveResponse({ defaultId: 'reviewer', modes }), calls })
    await loadAgentModes()
    const fetchesBefore = fetch.mock.calls.length
    const draft = await fetchAgentModesDocument()
    addModeToDraft(draft, 'Reviewer')
    draft.default = 'reviewer'
    const ok = await saveAgentModesDocument(draft)
    expect(ok).toBe(true)
    // Only the two editor calls happened after the dropdown's own load: the
    // list was not re-fetched.
    expect(fetch.mock.calls.length).toBe(fetchesBefore + 2)
    expect(calls.map((call) => call.key)).toEqual([
      '/api/agent-modes/document',
      '/api/agent-modes/save',
    ])
    expect(agentModes.modes.map((mode) => mode.id)).toEqual(['chat', 'code', 'reviewer', RESERVED_ID])
    expect(agentModes.current).toBe('reviewer') // D9: the selection moves to the new default
    expect(agentModes.default).toBe('reviewer')
  })

  it('a saved draft is clean and carries the new token', async () => {
    stubEditorRoutes({ save: saveResponse({ mtime: '1700000009000000000' }) })
    const draft = await fetchAgentModesDocument()
    draft.entries[0].label = 'Chatting'
    expect(draftIsDirty(draft)).toBe(true)
    expect(await saveAgentModesDocument(draft)).toBe(true)
    expect(draftIsDirty(draft)).toBe(false)
    expect(draft.mtime).toBe('1700000009000000000')
  })

  it('a 409 leaves the dialog open with the conflict named and the draft intact', async () => {
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
    await loadAgentModes()
    const editor = useAgentModes().editor
    editor.open = true
    const draft = await fetchAgentModesDocument()
    draft.entries[0].label = 'Chatting'
    expect(await saveAgentModesDocument(draft)).toBe(false)
    expect(editor.open).toBe(true) // stays open, D7
    expect(editor.conflict).toEqual({
      path: '/work/.nojae/agent-modes.json',
      diskMtime: '1700000005000000000',
      documentMtime: '1700000000000000000',
    })
    expect(editor.error).toBeNull()
    // Nothing was merged, nothing was overwritten, and the user's text survives.
    expect(draft.entries[0].label).toBe('Chatting')
    expect(draftIsDirty(draft)).toBe(true)
    expect(draft.mtime).toBe('1700000000000000000')
    expect(agentModes.current).toBe('code') // the dropdown was not touched
  })

  it('a saved file whose reload of the verdict failed is still said to be saved', async () => {
    // A 200 whose body carries no modes: the bytes landed, only the dropdown's
    // reload *of them* failed. The previous list stands and `loaded` goes false
    // — the same non-destructive degradation `loadAgentModes` already does.
    stubEditorRoutes({ save: { saved: true, created: false, modes: [], default: 'code', findings: [], source: {} } })
    await loadAgentModes()
    const draft = await fetchAgentModesDocument()
    expect(await saveAgentModesDocument(draft)).toBe(true)
    expect(agentModes.loaded).toBe(false)
    expect(agentModes.current).toBe('code') // never silently downgraded
    expect(agentModePayloadId()).toBeNull() // so send() omits the key
    expect(useAgentModes().editor.error).toBeNull()
  })

  it('a server 422 shows its findings rather than swallowing them', async () => {
    stubEditorRoutes({
      save: {
        __fail: true,
        status: 422,
        detail: { reason: 'invalid', findings: [{ kind: 'error', message: 'agents.x: unknown key' }] },
      },
    })
    const draft = await fetchAgentModesDocument()
    expect(await saveAgentModesDocument(draft)).toBe(false)
    expect(useAgentModes().editor.invalidFindings).toEqual([
      { kind: 'error', message: 'agents.x: unknown key' },
    ])
  })
})

describe('the D9 reset decision', () => {
  beforeEach(() => {
    resetAll()
    resetEditor()
  })

  it('unlocked: the selection moves to the newly set default', async () => {
    stubEditorRoutes({ save: saveResponse({ defaultId: 'architect' }) })
    await loadAgentModes()
    setAgentMode('chat')
    const draft = await fetchAgentModesDocument()
    expect(resetDecisionAfterSave()).toEqual({ move: true, reason: null })
    expect(await saveAgentModesDocument(draft)).toBe(true)
    expect(agentModes.current).toBe('architect')
    expect(agentModes.notice).toBeNull()
  })

  it('locked: it does not move, and the quiet line says why', async () => {
    stubEditorRoutes({ save: saveResponse({ defaultId: 'code' }) })
    await loadAgentModes()
    chat.messages = [{ role: 'user', raw: 'one turn', html: '', images: [] }]
    setAgentMode(RESERVED_ID)
    expect(agentModes.locked).toBe(true)
    const decision = resetDecisionAfterSave()
    expect(decision.move).toBe(false)
    expect(decision.reason).toContain('fixed once it has a turn')
    const draft = await fetchAgentModesDocument()
    expect(await saveAgentModesDocument(draft)).toBe(true)
    expect(agentModes.current).toBe(RESERVED_ID) // the lock survives the save
    expect(agentModes.locked).toBe(true)
    expect(agentModes.notice).toContain('did not move')
    expect(agentModes.notice).toContain('fixed once it has a turn')
  })

  it('a locked selection the save deleted falls back out loud, not onto nothing', async () => {
    // The saved document no longer declares the mode the lock was holding, so
    // there is nothing to keep: pointing `current` at a mode this workspace no
    // longer has would be a 422 on the next turn.
    const modes = [
      { id: 'chat', label: 'Chat', mcptools: false, lsptools: false, editable: true },
      { id: 'code', label: 'Code', mcptools: true, lsptools: true, editable: true },
    ]
    stubEditorRoutes({ save: saveResponse({ defaultId: 'code', modes }) })
    await loadAgentModes()
    chat.messages = [{ role: 'user', raw: 'one turn', html: '', images: [] }]
    setAgentMode(RESERVED_ID)
    expect(agentModes.locked).toBe(true)
    const draft = await fetchAgentModesDocument()
    deleteModeFromDraft(draft, 'architect') // the response above is what this wrote
    expect(await saveAgentModesDocument(draft)).toBe(true)
    expect(agentModes.current).toBe('code')
    expect(agentModes.locked).toBe(false) // the mode that owned the contract is gone
    expect(agentModes.notice).toContain(`"${RESERVED_ID}"`)
    expect(agentModes.notice).toContain('using code')
  })

  it('a running lot: its mode is pinned at launch and a save cannot move it', async () => {
    stubEditorRoutes({ save: saveResponse({ defaultId: 'architect' }) })
    await loadAgentModes()
    batch.active = true
    const decision = resetDecisionAfterSave({ batchActive: true })
    expect(decision.move).toBe(false)
    expect(decision.reason).toContain('running lot')
    const draft = await fetchAgentModesDocument()
    expect(await saveAgentModesDocument(draft, { batchActive: true })).toBe(true)
    expect(agentModes.current).toBe('code')
    expect(agentModes.notice).toContain('running lot')
  })

  it('a deleted-while-selected mode lands on the default', async () => {
    // The save response publishes the new verdict the way `GET /api/agent-modes`
    // does — the app-owned mode last, appended by the server — and `architect`
    // is gone from it, because the save wrote the draft that deleted it.
    const modes = [
      { id: 'chat', label: 'Chat', mcptools: false, lsptools: false, editable: true },
      { id: 'code', label: 'Code', mcptools: true, lsptools: true, editable: true },
      { id: RESERVED_ID, label: 'Reserved', mcptools: true, lsptools: false, editable: false },
    ]
    stubEditorRoutes({ save: saveResponse({ defaultId: 'code', modes }) })
    await loadAgentModes()
    setAgentMode('architect')
    const draft = await fetchAgentModesDocument()
    deleteModeFromDraft(draft, 'architect') // the selected one
    expect(await saveAgentModesDocument(draft)).toBe(true)
    expect(agentModes.current).toBe('code')
    expect(agentModes.modes.map((mode) => mode.id)).toEqual(['chat', 'code', RESERVED_ID])
  })
})

// The app-owned mode's id must never reach the client: the dropdown recognises
// that mode by `editable: false` alone (D9), so an id hardcoded in a component
// would be a second source of truth — and one that outlives a rename. The honest
// check is a scan, and it reads the id out of the Python module that owns it
// rather than restating it here (restating it would make this file a hit).
const SCANNED_EXTENSIONS = ['.js', '.vue', '.css', '.json']

function walk(dir) {
  const found = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = `${dir}/${entry.name}`
    if (entry.isDirectory()) found.push(...walk(path))
    else if (SCANNED_EXTENSIONS.some((ext) => entry.name.endsWith(ext))) found.push(path)
  }
  return found
}

describe('no reserved-mode id in the client', () => {
  it('web/src never contains an id agent_modes marks as reserved', () => {
    // Vitest runs from `web/`, and the jsdom environment leaves
    // `import.meta.url` non-file, so the roots are taken from the working
    // directory the way the other tests take their route paths.
    const source = readFileSync(
      resolve('..', 'src/core/agent_modes.py'),
      'utf8',
    )
    const declaration = /RESERVED_MODE_IDS[^=]*=\s*\(([^)]*)\)/.exec(source)
    expect(declaration, 'agent_modes.RESERVED_MODE_IDS is not readable').toBeTruthy()
    const ids = [...declaration[1].matchAll(/"([^"]+)"/g)].map((match) => match[1])
    expect(ids.length, 'no reserved id declared').toBeGreaterThan(0)

    const files = walk(join(resolve(), 'src'))
    expect(files.length, 'the scan found nothing to scan').toBeGreaterThan(0)
    for (const file of files) {
      const text = readFileSync(file, 'utf8')
      for (const id of ids) {
        expect(text, `${file} mentions the reserved mode id ${id}`).not.toContain(id)
      }
    }
  })
})
