import { flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  PROMPT_HELPER_DEFAULT_COLOR,
  PROMPT_HELPER_SLOT_COUNT,
  usePromptHelpers,
} from '../src/composables/usePromptHelpers'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// Layer 1 (jsdom): the bullet row's own store. What is pinned here is D4's UI
// claim — a workspace whose prompt-helpers file cannot be used costs a red
// note, not a broken row: the client's state is nine slots *before* any fetch,
// and a failed load leaves it that way. The stubbed payloads mirror what
// `GET /api/prompt-helpers` answers (mbl-03).

const { state, loadPromptHelpers } = usePromptHelpers()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

function slotsBody(first = 'a') {
  return {
    slots: Array.from({ length: PROMPT_HELPER_SLOT_COUNT }, (_, i) => ({
      color: i === 0 ? '#2ec27e' : PROMPT_HELPER_DEFAULT_COLOR,
      content: i === 0 ? first : '',
    })),
  }
}

function resetState() {
  state.slots = Array.from({ length: PROMPT_HELPER_SLOT_COUNT }, () => ({
    color: PROMPT_HELPER_DEFAULT_COLOR,
    content: '',
  }))
  state.loaded = false
  state.saving = false
  state.error = null
}

describe('loadPromptHelpers', () => {
  beforeEach(() => {
    resetState()
  })

  it('keeps its own nine slots until the first fetch answers', () => {
    expect(state.slots).toHaveLength(PROMPT_HELPER_SLOT_COUNT)
    expect(state.slots.every((slot) => slot.content === '')).toBe(true)
    expect(state.slots[0].color).toBe(PROMPT_HELPER_DEFAULT_COLOR)
  })

  it('replaces the slots with the workspace file contents', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse(slotsBody()))
    await loadPromptHelpers()
    expect(state.slots[0].content).toBe('a')
    expect(state.slots).toHaveLength(9)
    expect(state.loaded).toBe(true)
    expect(state.error).toBeNull()
  })

  it('a 422 naming the bad file leaves nine usable slots and one error', async () => {
    // mbl-03 refuses to pad or truncate: the sentence arrives as a string
    // `detail`, which describeError passes through verbatim.
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ detail: 'the file holds 8 entries' }, false, 422),
    )
    await loadPromptHelpers()
    expect(state.slots).toHaveLength(PROMPT_HELPER_SLOT_COUNT)
    expect(state.slots.every((slot) => slot.content === '')).toBe(true)
    expect(state.error).toBe('the file holds 8 entries')
  })

  it('an unreachable API is an error too, not an empty row', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => Promise.reject(new Error('down')))
    await loadPromptHelpers()
    expect(state.slots).toHaveLength(PROMPT_HELPER_SLOT_COUNT)
    expect(state.error).toContain('Could not reach the API')
  })
})

// mbl-05 / spec D8: the row is workspace state, so a working-location switch
// reloads it. This is the regression the task exists for — nothing else on the
// client pins it, because the bullet row only fetches when the chat tab mounts
// and `App.vue` keeps every view mounted, so a switch used to leave the old
// workspace's prompts on screen.
describe('a working-location switch reloads the bullets', () => {
  const { setPathLocation } = useBenchmark()
  const benchmark = useBenchmark().state

  function slotsNamed(name) {
    return {
      slots: Array.from({ length: PROMPT_HELPER_SLOT_COUNT }, (_, i) => ({
        color: PROMPT_HELPER_DEFAULT_COLOR,
        content: i === 0 ? name : '',
      })),
    }
  }

  beforeEach(() => {
    vi.restoreAllMocks()
    resetState()
    benchmark.pathLocation = '/srv/new'
    benchmark.pathLocationCurrent = '/srv/old'
    benchmark.pathLocationSaving = false
    benchmark.pathLocationError = null
    const tree = useWorkspaceTree()
    tree.state.nodes = {}
    tree.state.expanded = { '': true }
  })

  it('shows the new root slots without leaving the chat tab', async () => {
    // The same two GETs a user gets: the row's own first load (the old root's
    // file), then the reload the switch triggers. Only the *answer* changes, so
    // what is under test is that the second request is made at all.
    let slotOne = 'old workspace'
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation((url, options) => {
      const key = String(url)
      if (key === '/api/path-location' && options?.method === 'POST') {
        return jsonResponse({
          key: 'BENCHMARKIT_PATH_LOCATION',
          path: '/srv/new',
          default: '/srv/app',
          mcp: { installed: true, connectors: [], externalsError: null },
          lsp: { installed: true, servers: [], serversError: null },
          policy: { state: 'created', path: '/srv/new/.nojae/read-write-policy.json', error: null },
          agentModes: { state: 'created', path: '/srv/new/.nojae/agent-modes.json', error: null },
          promptHelpers: { state: 'created', path: '/srv/new/.nojae/prompt-helpers.json', error: null },
        })
      }
      if (key === '/api/prompt-helpers') return jsonResponse(slotsNamed(slotOne))
      if (key.startsWith('/api/fs/')) {
        return jsonResponse({ path: '', workspace: '/srv/new', entries: [], truncated: false })
      }
      return jsonResponse({})
    })

    await loadPromptHelpers()
    expect(state.slots[0].content).toBe('old workspace')

    slotOne = 'new workspace'
    await setPathLocation()
    // The switch does not await the reload (it matches the MCP/LSP reloads
    // beside it), so let that fetch settle before reading the slots.
    await flushPromises()

    expect(state.slots[0].content).toBe('new workspace')
    expect(benchmark.pathLocationCurrent).toBe('/srv/new')
    expect(benchmark.pathLocationError).toBeNull()
    expect(state.slots).toHaveLength(PROMPT_HELPER_SLOT_COUNT)
    expect(state.error).toBeNull()
    const helperFetches = fetchSpy.mock.calls.filter(
      ([url]) => String(url) === '/api/prompt-helpers',
    )
    expect(helperFetches).toHaveLength(2)
  })
})
