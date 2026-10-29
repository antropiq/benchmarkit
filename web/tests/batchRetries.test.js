import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  MAX_TASK_RETRIES,
  RETRY_NOTE_TEMPLATE,
  TASK_INSTRUCTION_TEMPLATE,
  nextBatchAction,
  taskCountedForProgress,
  taskInstruction,
  taskInstructionForAttempt,
  useBatch,
} from '../src/composables/useBatch'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useChat } from '../src/composables/useChat'

// Layer 1 (jsdom): the retry rule of batch mode. A task is finished when its
// file has left the lot folder — the move into `done/` is the agent's own
// acknowledgement, and the re-listing (the only file surface batch mode has,
// B13) is how the loop reads it. What is asserted here is the decision table
// and the loop that acts on it, against a stubbed /api/chat and /api/batch/tasks:
// the outcome wording and the retry note are e2e/visual concerns.

const batch = useBatch().state
const { launch, stopBatch } = useBatch()
const chat = useChat().state
const benchmark = useBenchmark().state

// ---------------------------------------------------------------------------
// The pure rules first: the decision table, and the message a retry carries.

describe('the retry note', () => {
  it('the first attempt is the plain instruction, byte for byte', () => {
    expect(taskInstructionForAttempt('todos/ed-01-a.md', 0)).toBe(
      taskInstruction('todos/ed-01-a.md'),
    )
    expect(taskInstructionForAttempt('todos/ed-01-a.md')).toBe(
      TASK_INSTRUCTION_TEMPLATE.replace('<task_location>', 'todos/ed-01-a.md'),
    )
  })

  it('a relaunch appends the note naming the retry number', () => {
    // The wording is the architect's and the model acts on it, so it is pinned
    // verbatim — the emphasis is what tells it the *move* is the missing step.
    expect(taskInstructionForAttempt('todos/ed-01-a.md', 1)).toBe(
      "You've been given task: todos/ed-01-a.md to proceed. Do the coding " +
        'tasks needed to implement the feature. Once done move it into the ' +
        'todos/done folder.\n\n' +
        'Important: The task as been relaunch because *it has not been moved* ' +
        'to the **todos/done** folder (retry number *1*)',
    )
  })

  it('the note carries the retry number of the attempt being made', () => {
    for (const attempt of [1, 2, MAX_TASK_RETRIES]) {
      expect(taskInstructionForAttempt('todos/ed-02-b.md', attempt)).toContain(
        `(retry number *${attempt}*)`,
      )
    }
    expect(RETRY_NOTE_TEMPLATE).toContain('<retry_number>')
  })
})

describe('taskCountedForProgress', () => {
  it('a turn that ran to its end fills the bar', () => {
    for (const outcome of ['done', 'truncated', 'capped']) {
      expect(taskCountedForProgress({ outcome, moved: true })).toBe(true)
    }
  })

  it('a turn whose task file is still in the lot does not', () => {
    expect(taskCountedForProgress({ outcome: 'done', moved: false })).toBe(false)
  })

  it('an inconclusive check still counts: the lot was not told it failed', () => {
    expect(taskCountedForProgress({ outcome: 'done', moved: null })).toBe(true)
  })

  it('a stopped, failed or never-started turn never counts', () => {
    for (const outcome of ['stopped', 'error', 'not-started']) {
      expect(taskCountedForProgress({ outcome, moved: true })).toBe(false)
    }
  })
})

describe('nextBatchAction', () => {
  it('a task file still in the lot is relaunched, with a rising retry number', () => {
    for (let attempt = 0; attempt < MAX_TASK_RETRIES; attempt += 1) {
      expect(
        nextBatchAction({ outcome: 'done', moved: false, attempt, active: true }),
      ).toBe('retry')
    }
  })

  it('once the retry budget is spent the whole lot is aborted', () => {
    expect(
      nextBatchAction({
        outcome: 'done',
        moved: false,
        attempt: MAX_TASK_RETRIES,
        active: true,
      }),
    ).toBe('abort')
  })

  it('a moved task, and an unjudgeable one, let the lot move on', () => {
    for (const moved of [true, null]) {
      expect(
        nextBatchAction({ outcome: 'done', moved, attempt: 0, active: true }),
      ).toBe('next')
    }
  })

  it('Stop outranks a pending retry', () => {
    expect(
      nextBatchAction({
        outcome: 'done',
        moved: false,
        attempt: 0,
        active: false,
      }),
    ).toBe('stopped')
    expect(
      nextBatchAction({
        outcome: 'stopped',
        moved: null,
        attempt: 2,
        active: true,
      }),
    ).toBe('stopped')
  })

  it('a failed turn ends the lot before the completion check can matter', () => {
    for (const outcome of ['error', 'not-started']) {
      expect(
        nextBatchAction({ outcome, moved: false, attempt: 0, active: true }),
      ).toBe('failed')
    }
  })
})

// ---------------------------------------------------------------------------
// The loop, against stubbed routes. `/api/chat` answers every turn with a
// normal `done` frame; `/api/batch/tasks` reports what is *still* in the lot
// folder, which is the only evidence the retry rule reads.

function chatDoneFrame() {
  return 'event: done\ndata: {"tokens":12,"took":1.5,"ttft":0.4,"decode":1.1}\n\n'
}

// A body useChat's reader loop accepts: one chunk holding the frame, then the
// stream ends. jsdom has no ReadableStream, so the reader is the minimal shape
// the loop actually calls.
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

function task(number, name) {
  return { number, name }
}

function listingOf(stillInLot) {
  return {
    dir: 'todos',
    workspace: '/srv/proj',
    prefix: 'ed-',
    taskCount: Math.max(0, ...stillInLot.map((item) => item.number)),
    doneFolder: true,
    lot: null,
    lotNote: null,
    tasks: stillInLot.map((item) => ({
      number: item.number,
      name: item.name,
      size: 40,
      taskLocation: `todos/${item.name}`,
      heading: null,
    })),
    skipped: [],
    truncated: false,
  }
}

// The lot folder as the loop sees it: a set of task files, from which a turn
// removes at most one — the file whose agent moved it into `done/`.
// `movedAfter[number]` is how many turns of that task it takes (Infinity: the
// agent never moves it, the case the retry exists for).
function stubRoutes({
  sent,
  tasks,
  movedAfter = {},
  failListing = false,
  duringTurn = null,
}) {
  const stillInLot = tasks.slice()
  const turnsOf = new Map()
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
    const key = String(url)
    if (key.startsWith('/api/chat')) {
      const instruction = JSON.parse(init.body).messages[0].content
      sent.push(instruction)
      const task = tasks.find((item) =>
        instruction.includes(`todos/${item.name}`),
      )
      if (task) {
        const turns = (turnsOf.get(task.number) ?? 0) + 1
        turnsOf.set(task.number, turns)
        if (turns >= (movedAfter[task.number] ?? 1)) {
          const index = stillInLot.findIndex((item) => item === task)
          if (index !== -1) stillInLot.splice(index, 1)
        }
      }
      if (duringTurn) duringTurn()
      return { ok: true, body: sseBody(chatDoneFrame()) }
    }
    if (key.startsWith('/api/batch/tasks')) {
      if (failListing) {
        return {
          ok: false,
          status: 404,
          json: async () => ({ detail: 'Batch folder does not exist: todos' }),
        }
      }
      const payload = listingOf(stillInLot)
      return { ok: true, json: async () => payload }
    }
    throw new Error(`unexpected fetch: ${key}`)
  })
}

function seedLot(tasks) {
  batch.listing = listingOf(tasks)
}

function reset() {
  chat.phase = 'idle'
  chat.messages = []
  chat.error = null
  chat.pausePending = false
  chat.session.startedAt = 0
  chat.mcp.loaded = false
  chat.mcp.locked = false
  chat.lsp.loaded = false
  chat.lsp.locked = false
  benchmark.status = 'idle'
  benchmark.activeTab = 'chat'
  batch.active = false
  batch.status = 'idle'
  batch.total = 0
  batch.index = 0
  batch.current = null
  batch.outcomes = []
  batch.error = null
  batch.listing = null
  batch.loading = false
  batch.prefix = 'ed-'
  batch.dir = 'todos'
  batch.includeLot = true
}

describe('the loop relaunches a task whose file never left the lot', () => {
  beforeEach(reset)

  // The path every lot took before retries existed stays exactly as it was: a
  // task whose file left the folder is never asked about again.
  it('moves on when the task file left the lot folder', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    stubRoutes({ sent, tasks }) // every turn moves its own task file
    seedLot(tasks)

    await launch()

    expect(sent).toEqual([
      taskInstruction('todos/ed-01-a.md'),
      taskInstruction('todos/ed-02-b.md'),
    ])
    expect(batch.status).toBe('finished')
    expect(batch.index).toBe(2)
    expect(batch.outcomes.map((item) => item.moved)).toEqual([true, true])
    expect(chat.error).toBeNull()
  })

  it('relaunches the task once, then takes the next one', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    // Task 1's agent forgets the move on its first turn and does it on the retry.
    stubRoutes({ sent, tasks, movedAfter: { 1: 2 } })
    seedLot(tasks)

    await launch()

    expect(sent).toEqual([
      taskInstruction('todos/ed-01-a.md'),
      taskInstructionForAttempt('todos/ed-01-a.md', 1),
      taskInstruction('todos/ed-02-b.md'),
    ])
    expect(batch.status).toBe('finished')
    expect(batch.index).toBe(2)
    // One record per *turn*: the relaunch is what the bar's title reports.
    expect(batch.outcomes).toEqual([
      { number: 1, attempt: 0, outcome: 'done', moved: false },
      { number: 1, attempt: 1, outcome: 'done', moved: true },
      { number: 2, attempt: 0, outcome: 'done', moved: true },
    ])
  })

  it('does not let the bar claim a task it is about to redo', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, tasks, movedAfter: { 1: 2 } })
    seedLot(tasks)

    await launch()

    expect(batch.outcomes.map((item) => item.moved)).toEqual([false, true])
    expect(batch.outcomes[0].outcome).toBe('done')
    // The unfinished turn left the bar where it was; the retry filled it.
    expect(batch.index).toBe(1)
  })

  // The budget is what stops the loop, not the model's good intentions: n runs
  // from 1 to MAX_TASK_RETRIES, then the remaining tasks are never started.
  it('aborts the lot once the retry budget is spent', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    // Task 1's file never leaves the folder: context overflow, a dead tool, or
    // an agent that simply will not do the move.
    stubRoutes({ sent, tasks, movedAfter: { 1: Infinity } })
    seedLot(tasks)

    await launch()

    // The first try plus MAX_TASK_RETRIES relaunches, all about task 1.
    expect(sent).toHaveLength(MAX_TASK_RETRIES + 1)
    for (let attempt = 1; attempt <= MAX_TASK_RETRIES; attempt += 1) {
      expect(sent[attempt]).toBe(
        taskInstructionForAttempt('todos/ed-01-a.md', attempt),
      )
    }
    // Task 2 was never started: the lot is one job.
    expect(sent.every((text) => text.includes('todos/ed-01-a.md'))).toBe(true)
    expect(batch.status).toBe('failed')
    expect(batch.outcomes).toHaveLength(MAX_TASK_RETRIES + 1)
    expect(batch.outcomes.every((item) => item.moved === false)).toBe(true)
    expect(chat.error).toContain('todos/ed-01-a.md')
    expect(chat.error).toContain(String(MAX_TASK_RETRIES))
    // An aborted lot leaves the app launchable again, like any failed one.
    expect(batch.active).toBe(false)
    expect(batch.current).toBeNull()
  })

  // Context overflow (the reply cut short) and tool-iteration overflow both
  // used to let the lot move on (B8/E4): the file staying in the folder is what
  // now says the task is not finished, whatever shape the turn ended in.
  it('relaunches a turn that was cut short and left the file in the lot', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
      const key = String(url)
      if (key.startsWith('/api/chat')) {
        sent.push(JSON.parse(init.body).messages[0].content)
        // The honest server-cut signal: an `error` frame carrying `truncated`.
        return {
          ok: true,
          body: sseBody(
            'event: token\ndata: {"text":"half an answer"}\n\n' +
              'event: error\ndata: {"message":"stream cut short","truncated":true}\n\n',
          ),
        }
      }
      const payload = listingOf(tasks) // nothing ever moves
      return { ok: true, json: async () => payload }
    })
    seedLot(tasks)

    await launch()

    expect(sent).toHaveLength(MAX_TASK_RETRIES + 1)
    expect(batch.outcomes.map((item) => item.outcome)).toEqual(
      Array(MAX_TASK_RETRIES + 1).fill('truncated'),
    )
    expect(batch.outcomes.every((item) => item.moved === false)).toBe(true)
    expect(batch.status).toBe('failed')
  })

  // A listing that cannot answer is not evidence of an unfinished task: it must
  // not burn turns relaunching work that was already done.
  it('a listing that fails does not trigger a retry', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    stubRoutes({ sent, tasks, failListing: true })
    seedLot(tasks)

    await launch()

    expect(sent).toHaveLength(2)
    expect(batch.outcomes.map((item) => item.moved)).toEqual([null, null])
    expect(batch.status).toBe('finished')
    expect(batch.index).toBe(2)
    expect(chat.error).toBeNull()
  })

  // Stop during the turn that left the file in place: relaunching it would be
  // restarting work the user just interrupted.
  it('Stop wins over a pending retry', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    stubRoutes({
      sent,
      tasks,
      movedAfter: { 1: Infinity },
      duringTurn: () => stopBatch(),
    })
    seedLot(tasks)

    await launch()

    expect(sent).toHaveLength(1)
    expect(batch.outcomes).toEqual([
      { number: 1, attempt: 0, outcome: 'stopped', moved: null },
    ])
    expect(batch.status).toBe('stopped')
    expect(batch.index).toBe(0)
  })
})
