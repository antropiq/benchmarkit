import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import AppBar from '../src/components/AppBar.vue'
import {
  POLICY_STATES,
  PROMPT_HELPER_STATES,
  describeAgentModes,
  describePolicy,
  describePromptHelpers,
} from '../src/lib/path-location-policy'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useChat } from '../src/composables/useChat'
import { useBatch } from '../src/composables/useBatch'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// pol-05 and am-03, client half: what the working-location payload's `policy`
// and `agentModes` blocks become on screen. mbl-05 adds the third block,
// `promptHelpers`, which shares the four states and the tone rule but answers
// two of its own from the one-time lift of the old app-level file.
//
// The surface that renders these is the app bar's working-location field, which
// is what the switch is made from; the settings dialog has no part in it.
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

function stub({
  postPolicy,
  getPolicy,
  postModes,
  getModes,
  postHelpers,
  getHelpers,
}) {
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
        promptHelpers: postHelpers,
      })
    }
    if (key.startsWith('/api/path-location')) {
      return jsonResponse({
        key: 'BENCHMARKIT_PATH_LOCATION',
        path: '/srv/proj',
        default: '/srv/app',
        policy: getPolicy,
        agentModes: getModes,
        promptHelpers: getHelpers,
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
  state.pathLocationPromptHelpers = null
  state.pathLocationPromptHelpersNote = null
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
    for (const kind of ['created', 'absent']) {
      const described = describeAgentModes({
        state: kind,
        path: '/srv/proj/.nojae/agent-modes.json',
        error: null,
      })
      expect(described.tone).toBe('plain')
      expect(described.text).toContain(kind === 'absent' ? 'agent-modes file' : '/srv/proj/.nojae/agent-modes.json')
    }
    // The ordinary case says nothing: a project that already had its file
    // gains nothing from being told the path of it.
    expect(
      describeAgentModes({ state: 'existing', path: '/srv/proj/.nojae/agent-modes.json', error: null }),
    ).toBeNull()
    const failed = describeAgentModes({ state: 'failed', path: '/p', error: 'boom' })
    expect(failed.tone).toBe('warn')
    expect(failed.text).toContain('boom')
    // The modes wording never claims the *switch* failed, and never borrows the
    // policy's sentence about an unfiltered tree.
    expect(describePolicy(state.pathLocationPolicy)).not.toEqual(failed)
  })
})

// mbl-05, client half: the third reserved-folder file on the same two
// endpoints. The tone rule is the same, plus D6's two extra states — a lift
// that happened is good news (plain), a lift that could not happen is a fact
// the user has to act on (warn).
const HELPERS_PATH = '/srv/proj/.nojae/prompt-helpers.json'

describe('a switch carries the prompt-helpers seed too', () => {
  beforeEach(() => {
    reset()
    vi.restoreAllMocks()
  })

  it.each(['created', 'absent'])(
    'leaves the %s report as a plain line, never a note',
    async (kind) => {
      stub({
        postPolicy: { state: 'created', path: '/srv/proj/.nojae/read-write-policy.json', error: null },
        postHelpers: { state: kind, path: HELPERS_PATH, error: null },
      })
      await setPathLocation()

      expect(state.pathLocationPromptHelpers.state).toBe(kind)
      expect(state.pathLocationPromptHelpersNote).toBeNull()
      expect(state.pathLocationError).toBeNull()
      const described = describePromptHelpers(state.pathLocationPromptHelpers)
      expect(described.tone).toBe('plain')
      expect(described.text).toContain(kind === 'absent' ? 'prompt-helper file' : HELPERS_PATH)
    },
  )

  it('says nothing at all about an existing helpers file', async () => {
    stub({
      postHelpers: { state: 'existing', path: HELPERS_PATH, error: null },
    })
    await setPathLocation()

    // The block is still carried (the bullets belong to that workspace), but
    // its report has no sentence, so the bar has nothing to show.
    expect(state.pathLocationPromptHelpers.state).toBe('existing')
    expect(state.pathLocationPromptHelpersNote).toBeNull()
    expect(describePromptHelpers(state.pathLocationPromptHelpers)).toBeNull()
  })

  it('notes a failed seed, never a failed switch', async () => {
    stub({
      postPolicy: { state: 'created', path: '/srv/proj/.nojae/read-write-policy.json', error: null },
      postHelpers: { state: 'failed', path: HELPERS_PATH, error: 'Read-only file system' },
    })
    await setPathLocation()

    expect(state.pathLocationCurrent).toBe('/srv/proj')
    expect(state.pathLocationError).toBeNull()
    expect(state.pathLocationPromptHelpersNote).toContain('prompt-helper file')
    expect(state.pathLocationPromptHelpersNote).toContain('Read-only file system')
    expect(state.pathLocationPromptHelpersNote).toContain(HELPERS_PATH)
  })

  it('keeps a lift that happened off the warnings', async () => {
    stub({
      postHelpers: { state: 'migrated', path: HELPERS_PATH, error: null },
    })
    await setPathLocation()

    // Good news: the older app-wide prompts are now this project's, and nothing
    // needs acting on — so no warn line and no red switch.
    expect(state.pathLocationPromptHelpersNote).toBeNull()
    expect(state.pathLocationError).toBeNull()
    const described = describePromptHelpers(state.pathLocationPromptHelpers)
    expect(described.tone).toBe('plain')
    expect(described.text).toContain(HELPERS_PATH)
    // It says where they came from, which is the part a reader wants to know.
    expect(described.text.toLowerCase()).toContain('moved')
  })

  it('warns about a lift that was skipped, naming the count and the old path', async () => {
    const legacy = '/srv/app/resources/memory_registration.json'
    stub({
      postHelpers: {
        state: 'migration-skipped',
        path: HELPERS_PATH,
        error: `the legacy file ${legacy} holds 7 entries, expected exactly 9; the new file keeps the default slots`,
      },
    })
    await setPathLocation()

    expect(state.pathLocationPromptHelpersNote).toContain('7 entries')
    expect(state.pathLocationPromptHelpersNote).toContain(legacy)
    expect(state.pathLocationPromptHelpersNote).toContain(HELPERS_PATH)
    expect(state.pathLocationError).toBeNull()
  })

  it('is one warn line, not one per file', async () => {
    // The three seeds fail independently; a failed policy beside an existing
    // helpers file must still be a single line about the policy.
    stub({
      postPolicy: { state: 'failed', path: '/x/read-write-policy.json', error: 'nope' },
      postHelpers: { state: 'existing', path: HELPERS_PATH, error: null },
    })
    await setPathLocation()

    expect(state.pathLocationPolicyNote).not.toBeNull()
    expect(state.pathLocationPromptHelpersNote).toBeNull()
  })

  it('carries an absent helpers block on a read, without alarm', async () => {
    stub({
      getPolicy: { state: 'existing', path: '/x/read-write-policy.json', error: null },
      getHelpers: { state: 'absent', path: HELPERS_PATH, error: null },
    })
    await loadPathLocation()

    expect(state.pathLocationPromptHelpers.state).toBe('absent')
    expect(state.pathLocationPromptHelpersNote).toBeNull()
    expect(state.pathLocationError).toBeNull()
  })

  it('describes every helpers state with a tone, and nothing else', () => {
    expect(describePromptHelpers(null)).toBeNull()
    expect(describePromptHelpers({ state: 'nonsense' })).toBeNull()
    for (const kind of PROMPT_HELPER_STATES) {
      const described = describePromptHelpers({
        state: kind,
        path: HELPERS_PATH,
        error: kind === 'migration-skipped' ? 'holds 7 entries' : null,
      })
      // `existing` is the state with nothing to tell (see the header), so it is
      // the one that answers no sentence at all; the rest carry a tone.
      if (kind === 'existing') {
        expect(described).toBeNull()
        continue
      }
      expect(described).not.toBeNull()
      expect(described.text).not.toBe('')
      // D6: the warn tone belongs to the two states the user must act on.
      expect(described.tone).toBe(
        kind === 'failed' || kind === 'migration-skipped' ? 'warn' : 'plain',
      )
    }
    // And the shared four are all in the helpers list, so the loop above covers
    // what the other two files answer too.
    for (const kind of POLICY_STATES) {
      expect(PROMPT_HELPER_STATES).toContain(kind)
    }
  })

  it('shows a seed warning under the app bar field, not a refusal', async () => {
    // Read through the mounted bar: the point is that the one surface holding
    // the field is also the one that answers the switch, and that a seed that
    // did not land is the warn line — the red one belongs to a refusal alone.
    state.pathLocationPromptHelpers = {
      state: 'migration-skipped',
      path: HELPERS_PATH,
      error: 'the legacy file /srv/app/resources/memory_registration.json holds 7 entries',
    }
    state.pathLocationPromptHelpersNote =
      describePromptHelpers(state.pathLocationPromptHelpers).text
    state.pathLocationError = null

    const wrapper = mount(AppBar)
    const notes = wrapper.findAll('.appbar-location-note')
    expect(notes).toHaveLength(1)
    expect(notes[0].classes()).not.toContain('appbar-location-note--error')
    expect(notes[0].text()).toContain('7 entries')
    wrapper.unmount()
  })

  it('shows a refusal as the error line, ahead of any seed note', async () => {
    state.pathLocationError = 'Working location does not exist: /nope'
    state.pathLocationPolicyNote = 'Working location is set, but ...'

    const wrapper = mount(AppBar)
    const notes = wrapper.findAll('.appbar-location-note')
    expect(notes).toHaveLength(1)
    expect(notes[0].classes()).toContain('appbar-location-note--error')
    expect(notes[0].text()).toContain('/nope')
    wrapper.unmount()
  })

  it('leaves the line out when the switch had nothing to say', async () => {
    const wrapper = mount(AppBar)
    expect(wrapper.findAll('.appbar-location-note')).toHaveLength(0)
    wrapper.unmount()
  })
})

// The Set button is the one gesture that moves the root, so it is inert
// whenever a session is live: the route answers 409 to a switch during a run or
// a reply, and a batch would find its lot folder re-rooted mid-task.
describe('Set is inert while a session runs', () => {
  beforeEach(() => {
    reset()
    vi.restoreAllMocks()
  })

  function setButton(wrapper) {
    return wrapper.find('.appbar-location-set')
  }

  it('is enabled when nothing runs and the field differs', async () => {
    const wrapper = mount(AppBar)
    const button = setButton(wrapper)
    expect(button.attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it.each([
    ['a benchmark run', () => (state.status = 'running')],
    ['a chat reply', () => (useChat().state.phase = 'answering')],
    ['a batch', () => (useBatch().state.active = true)],
  ])('is inert during %s, and says why', async (_label, arm) => {
    arm()
    const wrapper = mount(AppBar)
    const button = setButton(wrapper)
    expect(button.attributes('disabled')).toBeDefined()
    expect(button.attributes('title')).toContain('while a session is running')
    wrapper.unmount()
    state.status = 'idle'
    useChat().state.phase = 'idle'
    useBatch().state.active = false
  })

  it('refuses Enter in the field while a session runs', async () => {
    const fetchSpy = stub({ postPolicy: { state: 'created', path: '/x', error: null } })
    state.status = 'running'
    const wrapper = mount(AppBar)
    await wrapper.find('.appbar-location-input').trigger('keydown.enter')
    const calls = fetchSpy.mock.calls.map(([url]) => String(url))
    expect(calls.some((url) => url === '/api/path-location')).toBe(false)
    state.status = 'idle'
    wrapper.unmount()
  })

  it('applies the field on Enter when nothing runs', async () => {
    const fetchSpy = stub({ postPolicy: { state: 'created', path: '/x', error: null } })
    const wrapper = mount(AppBar)
    await wrapper.find('.appbar-location-input').trigger('keydown.enter')
    const posts = fetchSpy.mock.calls.filter(
      ([url, options]) => String(url) === '/api/path-location' && options?.method === 'POST',
    )
    expect(posts).toHaveLength(1)
    wrapper.unmount()
  })
})
