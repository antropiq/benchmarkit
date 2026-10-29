import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBenchmark } from '../src/composables/useBenchmark'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// pol-05, client half: what the working-location payload's `policy` block
// becomes in the dialog.
//
// The rule this pins is the *tone*, not the text: a switch that could not leave
// a policy file behind **succeeded**, so it must not be reported as
// `pathLocationError` (the dialog renders that as a refusal and the location
// really did move). It goes to a separate note. And GET never seeds — the
// server's read half reports `absent` without writing into a browsed project —
// so the client has to carry that state rather than invent one.

const { state, loadPathLocation, setPathLocation } = useBenchmark()

function jsonResponse(body) {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) })
}

const treePayload = { path: '', workspace: '/srv/proj', entries: [], truncated: false }

function stub({ postPolicy, getPolicy }) {
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
      })
    }
    if (key.startsWith('/api/path-location')) {
      return jsonResponse({
        key: 'BENCHMARKIT_PATH_LOCATION',
        path: '/srv/proj',
        default: '/srv/app',
        policy: getPolicy,
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
  })
})
