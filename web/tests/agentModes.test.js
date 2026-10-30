import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

import {
  agentModePayloadId,
  agentModesLoaded,
  applySessionAgentMode,
  currentMode,
  findingsLine,
  loadAgentModes,
  modeBlocksHub,
  modeHint,
  modeOptionDisabled,
  modeSelectDisabled,
  modeSelectTitle,
  setAgentMode,
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
