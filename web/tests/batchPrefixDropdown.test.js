import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

import BatchModeDialog from '../src/components/BatchModeDialog.vue'
import { NO_BATCH_ITEMS_MESSAGE, useBatch } from '../src/composables/useBatch'
import { useChat } from '../src/composables/useChat'

// The lot is *chosen*, not typed: the dialog offers one option per prefix that
// has a task in the lot folder, so a lot that cannot launch is never offered.
// The prefix list is therefore the server's answer (GET /api/batch/prefixes),
// and the dialog keeps no prefix knowledge of its own.

const batch = useBatch().state
const chat = useChat().state

function prefixEntry(prefix, tasks, hasLot = false) {
  return { prefix, tasks, hasLot }
}

function listingBody({ prefix = 'doc-', tasks = [], lot = null, lotNote = null } = {}) {
  return {
    dir: 'todos',
    workspace: '/srv/proj',
    prefix,
    taskCount: Math.max(0, ...tasks.map((item) => item.number)),
    doneFolder: true,
    lot,
    lotNote,
    tasks,
    skipped: [],
    truncated: false,
  }
}

function taskEntry(number, name) {
  return {
    number,
    name,
    size: 40,
    taskLocation: `todos/${name}`,
    heading: null,
  }
}

// `prefixes` is the /api/batch/prefixes answer (null: the fetch fails);
// `listing` is the /api/batch/tasks answer. Both are recorded so a test can say
// *which* folder was listed and with which prefix.
function stubRoutes({ prefixes = [], listing = null, prefixCalls = [], taskCalls = [] }) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
    const key = String(url)
    if (key.startsWith('/api/batch/prefixes')) {
      prefixCalls.push(key)
      if (prefixes === null) {
        return {
          ok: false,
          status: 404,
          json: async () => ({ detail: 'Batch folder does not exist: todos' }),
        }
      }
      return { ok: true, json: async () => ({ dir: 'todos', prefixes }) }
    }
    if (key.startsWith('/api/batch/tasks')) {
      taskCalls.push(key)
      return { ok: true, json: async () => listing ?? listingBody() }
    }
    throw new Error(`unexpected fetch: ${key}`)
  })
}

function reset() {
  vi.restoreAllMocks()
  chat.phase = 'idle'
  chat.messages = []
  chat.error = null
  batch.open = false
  batch.prefix = ''
  batch.dir = 'todos'
  batch.prefixes = []
  batch.prefixesLoading = false
  batch.listing = null
  batch.loading = false
  batch.error = null
  batch.active = false
  batch.status = 'idle'
}

describe('the prefix dropdown', () => {
  beforeEach(reset)

  it('offers one option per prefix the folder holds', async () => {
    stubRoutes({
      prefixes: [prefixEntry('doc-', 3, true), prefixEntry('oth-', 1)],
    })
    const { refreshLot } = useBatch()
    await refreshLot()

    const wrapper = mount(BatchModeDialog)
    const options = wrapper.findAll('select option')
    // The first is the placeholder; the lots come after it.
    expect(options).toHaveLength(3)
    expect(options[1].text()).toContain('doc-')
    expect(options[1].text()).toContain('3 tasks')
    expect(options[2].text()).toContain('oth-')
    expect(options[2].text()).toContain('1 task')
    wrapper.unmount()
  })

  it('lists the prefixes of the folder the field names', async () => {
    const prefixCalls = []
    stubRoutes({ prefixes: [], prefixCalls })
    batch.dir = 'work/todos'
    const { refreshLot } = useBatch()
    await refreshLot()

    expect(prefixCalls).toHaveLength(1)
    expect(prefixCalls[0]).toContain('dir=work%2Ftodos')
  })

  it('a folder that cannot be listed is an error, not an empty list', async () => {
    stubRoutes({ prefixes: null })
    const { refreshLot } = useBatch()
    await refreshLot()

    expect(batch.prefixes).toEqual([])
    expect(batch.error).toContain('Batch folder does not exist')
    // The "no batch items" sentence would be a lie about a folder the app could
    // not open, so it stays hidden while the error line speaks.
    const wrapper = mount(BatchModeDialog)
    expect(wrapper.text()).not.toContain('There is no existing batch items')
    expect(wrapper.find('.alert-error').text()).toContain('does not exist')
    wrapper.unmount()
  })

  it('a prefix whose lot is gone is dropped from the selection', async () => {
    batch.prefix = 'ed-'
    batch.listing = listingBody({ tasks: [taskEntry(1, 'ed-01-a.md')] })
    stubRoutes({ prefixes: [prefixEntry('doc-', 2)] })
    const { refreshLot } = useBatch()
    await refreshLot()

    // The stale choice and the listing built from it are both gone: Launch would
    // otherwise run a lot the folder no longer holds.
    expect(batch.prefix).toBe('')
    expect(batch.listing).toBeNull()
  })

  it('a prefix that survived the re-fetch keeps its listing', async () => {
    batch.prefix = 'doc-'
    const taskCalls = []
    stubRoutes({ prefixes: [prefixEntry('doc-', 2)], taskCalls })
    const { refreshLot } = useBatch()
    await refreshLot()

    expect(batch.prefix).toBe('doc-')
    // The count is re-read too: a task moved into done/ must not be launchable
    // a second time off a stale listing.
    expect(taskCalls).toHaveLength(1)
  })

  it('picking an option lists that lot', async () => {
    const taskCalls = []
    stubRoutes({
      prefixes: [prefixEntry('doc-', 2), prefixEntry('oth-', 1)],
      listing: listingBody({ prefix: 'oth-', tasks: [taskEntry(1, 'oth-01-c.md')] }),
      taskCalls,
    })
    const { refreshLot } = useBatch()
    await refreshLot()
    const wrapper = mount(BatchModeDialog)
    await wrapper.find('select').setValue('oth-')
    await vi.waitFor(() => expect(taskCalls).toHaveLength(1))

    expect(batch.prefix).toBe('oth-')
    expect(taskCalls[0]).toContain('prefix=oth-')
    expect(wrapper.text()).toContain('Found tasks 1')
    wrapper.unmount()
  })

  it('says what to do when the folder holds no batch item', async () => {
    stubRoutes({ prefixes: [] })
    const { refreshLot } = useBatch()
    await refreshLot()

    const wrapper = mount(BatchModeDialog)
    expect(wrapper.text()).toContain(NO_BATCH_ITEMS_MESSAGE)
    expect(wrapper.text()).toContain('Todo builder agent')
    // Nothing to choose and nothing to launch.
    expect(wrapper.find('select').exists()).toBe(false)
    expect(wrapper.find('button[type="submit"]').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })
})

describe('the dialog no longer asks', () => {
  beforeEach(reset)

  it('has no big-picture checkbox — a 00 file is always included', async () => {
    stubRoutes({ prefixes: [prefixEntry('doc-', 2, true)] })
    const { refreshLot } = useBatch()
    await refreshLot()
    batch.listing = listingBody({
      tasks: [taskEntry(1, 'doc-01-a.md')],
      lot: { name: 'doc-00-lot.md', path: 'todos/doc-00-lot.md', size: 20, heading: null, text: 'the big picture' },
    })

    const wrapper = mount(BatchModeDialog)
    expect(wrapper.find('input[type="checkbox"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('Include the big picture')
    // The lot is still reported, as a fact rather than a choice.
    expect(wrapper.text()).toContain('big picture: doc-00-lot.md')
    wrapper.unmount()
  })

  it('has no explanatory paragraph about the retry budget', async () => {
    stubRoutes({ prefixes: [prefixEntry('doc-', 2)] })
    const { refreshLot } = useBatch()
    await refreshLot()

    const wrapper = mount(BatchModeDialog)
    expect(wrapper.text()).not.toContain('one task per chat turn')
    expect(wrapper.text()).not.toContain('relaunched')
    // What the form needs is still there: the folder field and the count line.
    expect(wrapper.find('input[type="text"]').exists()).toBe(true)
    expect(wrapper.find('#batch-count').exists()).toBe(true)
    wrapper.unmount()
  })
})
