import { beforeEach, describe, expect, it, vi } from 'vitest'

import { describeAgentModes, describePolicy } from '../src/lib/path-location-policy'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// pol-05 and am-03, client half: what the working-location payload's `policy`
// and `agentModes` blocks become in the dialog.
//
// The rule this pins is the *tone*, not the text: a switch that could not leave
// a file behind **succeeded**, so it must not be reported as
// `pathLocationError` (the dialog renders that as a refusal and the location
// really did move). It goes to a separate note. The agent-modes block answers
// with the same four states on the same two endpoints, so it is carried by the
// same two functions under the same rule. And GET never seeds — the
// server's read half reports `absent` without writing into a browsed project —
// so the client has to carry that state rather than invent one.

const { state, loadPathLocation, setPathLocation } = useBenchmark()

function jsonResponse(body) {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) })
}

const treePayload = { path: '', workspace: '/srv/proj', entries: [], truncated: false }

function stub({ postPolicy, getPolicy, postModes, getModes }) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((url, options) => {
    const key = String(url)
    if (key === '/api/path-location' && options?.method === 'POST') {
      return jsonResponse({
        key: 'BENCHMARKIT_PATH_LOCATION',
        path: '/srv/proj',
        default: '/srv/app',
        mcp: { installed: true, connectors: [], externalsError: null },
        lsp: { installed: true, servers: [], serversError: null },
        policy: postPolicy,
        agentModes: postModes,
      })
    }
    if (key.startsWith('/api/path-location')) {
      return jsonResponse({
        key: 'BENCHMARKIT_PATH_LOCATION',
        path: '/srv/proj',
        default: '/srv/app',
        policy: getPolicy,
        agentModes: getModes,
      })
    }
    if (key.startsWith('/api/fs/')) return jsonResponse(treePayload)
    return jsonResponse({})
  })
}

function reset() {
  state.pathLocation = '/srv/proj'
  state.pathLocationCurrent = '/srv/old'
  state.pathLocationError = null
  state.pathLocationPolicy = null
  state.pathLocationPolicyNote = null
  state.pathLocationAgentModes = null
  state.pathLocationAgentModesNote = null
  state.pathLocationSaving = false
  const tree = useWorkspaceTree()
  tree.state.nodes = {}
  tree.state.expanded = { '': true }
}

describe('a switch carries the seed result', () => {
  beforeEach(() => {
    reset()
    vi.restoreAllMocks()
  })

  it('says nothing alarming when the policy file was created', async () => {
    stub({ postPolicy: { state: 'created', path: '/srv/proj/.nojae/read-write-policy.json', error: null } })
    await setPathLocation()

    expect(state.pathLocationPolicy.state).toBe('created')
    expect(state.pathLocationPolicyNote).toBeNull()
    expect(state.pathLocationError).toBeNull()
  })

  it('reports a failed seed as a note, never as a failed switch', async () => {
    stub({
      postPolicy: {
        state: 'failed',
        path: '/srv/proj/.nojae/read-write-policy.json',
        error: 'Read-only file system',
      },
    })
    await setPathLocation()

    // The location moved and the tree re-rooted; only the note says the file
    // could not be written.
    expect(state.pathLocationCurrent).toBe('/srv/proj')
    expect(state.pathLocationError).toBeNull()
    expect(state.pathLocationPolicyNote).toContain('could not create its policy file')
    expect(state.pathLocationPolicyNote).toContain('Read-only file system')
    expect(state.pathLocationPolicyNote).toContain('/srv/proj/.nojae/read-write-policy.json')
  })

  it('clears a stale note on the next attempt', async () => {
    stub({ postPolicy: { state: 'failed', path: '/x', error: 'nope' } })
    await setPathLocation()
    expect(state.pathLocationPolicyNote).not.toBeNull()

    state.pathLocationCurrent = '/srv/old'
    stub({ postPolicy: { state: 'created', path: '/x', error: null } })
    await setPathLocation()
    expect(state.pathLocationPolicyNote).toBeNull()
  })

  it('keeps whatever the server reported on a read', async () => {
    stub({ getPolicy: { state: 'absent', path: '/srv/proj/.nojae/read-write-policy.json', error: null } })
    await loadPathLocation()

    expect(state.pathLocationPolicy.state).toBe('absent')
    // `absent` is not an error and not a note — the dialog words it plainly.
    expect(state.pathLocationPolicyNote).toBeNull()
    expect(state.pathLocationError).toBeNull()
  })

  it('survives a payload with no policy block at all', async () => {
    stub({ getPolicy: undefined })
    await loadPathLocation()
    expect(state.pathLocationPolicy).toBeNull()
    expect(state.pathLocationPolicyNote).toBeNull()
    // am-03's block is optional for the same reason: an older server, or a
    // response that carries one half of the pair, must not become an error.
    expect(state.pathLocationAgentModes).toBeNull()
    expect(state.pathLocationAgentModesNote).toBeNull()
  })
})

// am-03, client half: the second file of the reserved folder, carried by the
// same two endpoints in the same shape.
describe('a switch carries the agent-modes seed too', () => {
  beforeEach(() => {
    reset()
    vi.restoreAllMocks()
  })

  it('stays quiet when both files landed', async () => {
    stub({
      postPolicy: { state: 'created', path: '/srv/proj/.nojae/read-write-policy.json', error: null },
      postModes: { state: 'created', path: '/srv/proj/.nojae/agent-modes.json', error: null },
    })
    await setPathLocation()

    expect(state.pathLocationAgentModes.state).toBe('created')
    expect(state.pathLocationAgentModesNote).toBeNull()
    expect(state.pathLocationPolicyNote).toBeNull()
    expect(state.pathLocationError).toBeNull()
  })

  it('notes a modes-only failure without touching the policy line', async () => {
    stub({
      postPolicy: { state: 'created', path: '/srv/proj/.nojae/read-write-policy.json', error: null },
      postModes: {
        state: 'failed',
        path: '/srv/proj/.nojae/agent-modes.json',
        error: 'Read-only file system',
      },
    })
    await setPathLocation()

    expect(state.pathLocationCurrent).toBe('/srv/proj')
    expect(state.pathLocationError).toBeNull()
    expect(state.pathLocationPolicyNote).toBeNull()
    expect(state.pathLocationAgentModesNote).toContain('agent-modes file')
    expect(state.pathLocationAgentModesNote).toContain('Read-only file system')
    expect(state.pathLocationAgentModesNote).toContain('/srv/proj/.nojae/agent-modes.json')
  })

  it('keeps the two notes apart, so each names its own file', async () => {
    stub({
      postPolicy: { state: 'failed', path: '/x/read-write-policy.json', error: 'nope' },
      postModes: { state: 'failed', path: '/x/agent-modes.json', error: 'nah' },
    })
    await setPathLocation()

    expect(state.pathLocationPolicyNote).toContain('policy file')
    expect(state.pathLocationAgentModesNote).toContain('agent-modes file')
    // Distinct fields: the dialog can join them, but neither sentence swallows
    // the other.
    expect(state.pathLocationPolicyNote).not.toBe(state.pathLocationAgentModesNote)
  })

  it('carries an absent modes block on a read, without alarm', async () => {
    stub({
      getPolicy: { state: 'existing', path: '/x/read-write-policy.json', error: null },
      getModes: { state: 'absent', path: '/srv/proj/.nojae/agent-modes.json', error: null },
    })
    await loadPathLocation()

    expect(state.pathLocationAgentModes.state).toBe('absent')
    expect(state.pathLocationAgentModesNote).toBeNull()
    expect(state.pathLocationError).toBeNull()
  })

  it('describes each modes state with a tone', () => {
    expect(describeAgentModes(null)).toBeNull()
    expect(describeAgentModes({ state: 'nonsense' })).toBeNull()
    for (const kind of ['created', 'existing', 'absent']) {
      const described = describeAgentModes({
        state: kind,
        path: '/srv/proj/.nojae/agent-modes.json',
        error: null,
      })
      expect(described.tone).toBe('plain')
      expect(described.text).toContain(kind === 'absent' ? 'agent-modes file' : '/srv/proj/.nojae/agent-modes.json')
    }
    const failed = describeAgentModes({ state: 'failed', path: '/p', error: 'boom' })
    expect(failed.tone).toBe('warn')
    expect(failed.text).toContain('boom')
    // The modes wording never claims the *switch* failed, and never borrows the
    // policy's sentence about an unfiltered tree.
    expect(describePolicy(state.pathLocationPolicy)).not.toEqual(failed)
  })
})
