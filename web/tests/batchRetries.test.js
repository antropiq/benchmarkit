import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  LOT_TASK_MODE_ID,
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
import { loadAgentModes, setAgentMode, useAgentModes } from '../src/composables/useAgentModes'

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
// `modes` is what `GET /api/agent-modes` answers (default: the seeded four);
// `modes: null` makes that fetch fail. `payloads` collects whole /api/chat bodies.
function stubRoutes({
  sent,
  tasks,
  movedAfter = {},
  failListing = false,
  duringTurn = null,
  payloads = null,
  // Absent (undefined) answers the seeded four; `null` makes the fetch fail.
  modes,
}) {
  const stillInLot = tasks.slice()
  const turnsOf = new Map()
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
    const key = String(url)
    if (key.startsWith('/api/agent-modes')) {
      if (modes === null) {
        return { ok: false, status: 500, json: async () => ({}) }
      }
      return { ok: true, json: async () => modes ?? seededModesBody() }
    }
    if (key.startsWith('/api/chat')) {
      const body = JSON.parse(init.body)
      const instruction = body.messages[0].content
      sent.push(instruction)
      if (payloads) payloads.push(body)
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

// A modes payload in the shape `GET /api/agent-modes` answers (am-06). The
// app-owned mode is spelled with a stand-in id here too: this file may not name
// it either, for the same reason `web/src` may not (the scan in
// agentModes.test.js reads the id out of the Python module).
function seededModesBody({ codeMcpTools = true, withoutCodeMode = false } = {}) {
  const modes = [
    { id: 'chat', label: 'Chat', mcptools: false, lsptools: false, editable: true },
    {
      id: LOT_TASK_MODE_ID,
      label: 'Code',
      mcptools: codeMcpTools,
      lsptools: true,
      editable: true,
    },
    { id: 'architect', label: 'Architect', mcptools: true, lsptools: false, editable: true },
    { id: 'reserved-mode-under-test', label: 'Reserved', mcptools: true, lsptools: false, editable: false },
  ]
  return {
    modes: withoutCodeMode
      ? modes.filter((mode) => mode.id !== LOT_TASK_MODE_ID)
      : modes,
    default: 'chat',
    source: {
      path: '/work/.nojae/agent-modes.json',
      exists: true,
      loaded: true,
      version: 1,
    },
    findings: [],
  }
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
  batch.pinnedMode = null
  batch.modeNote = null
  agentModes.modes = []
  agentModes.current = null
  agentModes.default = null
  agentModes.loaded = false
  agentModes.locked = false
  agentModes.findings = []
  agentModes.notice = null
}

const agentModes = useAgentModes().state

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

// ---------------------------------------------------------------------------
// D13 — the runner pins its own mode. A lot is authored in the app-owned,
// contract-owning mode and *executed* by the runner, so its turns run under the
// workspace's `code` mode regardless of what the composer's dropdown shows — and
// the dropdown cannot change anything mid-lot.

describe('the lot pins its mode', () => {
  beforeEach(reset)

  // Every task request carries the pinned id, and only that one — the loop
  // reads `state.pinnedMode` per turn rather than the composer's selection.
  it('launch resolves the mode once and every task turn carries it', async () => {
    const sent = []
    const payloads = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    stubRoutes({ sent, payloads, tasks })
    await loadAgentModes()
    // The user was drafting the lot in another mode before opening Batch.
    setAgentMode('architect')
    seedLot(tasks)

    await launch()

    expect(batch.pinnedMode).toBe(LOT_TASK_MODE_ID)
    expect(batch.modeNote).toBeNull()
    expect(payloads).toHaveLength(2)
    for (const payload of payloads) {
      expect(payload.agentMode).toBe(LOT_TASK_MODE_ID)
    }
    // The composer's own selection is untouched by the pin: the select shows
    // what the *human* picked, and is inert while the lot runs.
    expect(agentModes.current).toBe('architect')
  })

  // The reason the pin is state and not an argument: a dropdown change between
  // two turns must not reach the next one.
  it('a dropdown change mid-run cannot reach the next task', async () => {
    const sent = []
    const payloads = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    stubRoutes({
      sent,
      payloads,
      tasks,
      // Task 1's turn ends; the human switches mode before task 2 is sent.
      duringTurn: () => setAgentMode('chat'),
    })
    await loadAgentModes()
    seedLot(tasks)

    await launch()

    expect(payloads).toHaveLength(2)
    expect(payloads[0].agentMode).toBe(LOT_TASK_MODE_ID)
    expect(payloads[1].agentMode).toBe(LOT_TASK_MODE_ID)
    expect(agentModes.current).toBe('chat')
  })

  // A workspace that renamed its coding mode is not refused: that would make
  // modes able to break a lot that works today. It sends no `agentMode` at all
  // (the server's default answers, the pre-feature shape) and says so once.
  it('with no code mode the launch proceeds unpinned, with one line', async () => {
    const sent = []
    const payloads = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, payloads, tasks, modes: seededModesBody({ withoutCodeMode: true }) })
    await loadAgentModes()
    seedLot(tasks)

    await launch()

    expect(batch.pinnedMode).toBeNull()
    expect(batch.modeNote).toContain(`no "${LOT_TASK_MODE_ID}" mode`)
    // Names the workspace's own default, so the line says what really answers.
    expect(batch.modeNote).toContain('"chat"')
    expect(payloads).toHaveLength(1)
    expect('agentMode' in payloads[0]).toBe(false)
    expect(batch.status).toBe('finished')
  })

  // Same laxness when the list could not be read at all: nothing to pin, and a
  // line that does not claim to know this workspace's default.
  it('an unreadable modes list pins nothing and launches anyway', async () => {
    const sent = []
    const payloads = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, payloads, tasks, modes: null })
    await loadAgentModes()
    seedLot(tasks)

    await launch()

    expect(batch.pinnedMode).toBeNull()
    expect(batch.modeNote).toContain('could not be read')
    expect('agentMode' in payloads[0]).toBe(false)
  })

  // The §1 clause: the mode gate joins the *existing* connector list rather
  // than becoming a second guard, so one message explains one dead lot. Here the
  // connectors are all ticked, so the mode is the only thing wrong.
  it('a code mode with no MCP tools is reported in the launch guard', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, tasks, modes: seededModesBody({ codeMcpTools: false }) })
    await loadAgentModes()
    chat.mcp.loaded = true
    chat.mcp.enabled = { fedit: true, console: true }
    seedLot(tasks)

    await launch()

    expect(chat.error).toContain(LOT_TASK_MODE_ID)
    expect(chat.error).toContain('MCP tools')
    // Nothing ran, and no pin was kept for a lot that never launched.
    expect(sent).toEqual([])
    expect(batch.active).toBe(false)
    expect(batch.pinnedMode).toBeNull()
  })

  // ...and it lands in the same sentence as missing connectors — one list, one
  // reason, one place the user learns why Launch did nothing.
  it('a missing connector and a tools-off mode are one message', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, tasks, modes: seededModesBody({ codeMcpTools: false }) })
    await loadAgentModes()
    chat.mcp.loaded = true
    chat.mcp.enabled = { fedit: true, console: false }
    seedLot(tasks)

    await launch()

    expect(chat.error).toContain('console')
    expect(chat.error).toContain(LOT_TASK_MODE_ID)
    // One sentence, not two refusals: the connector list and the mode clause
    // are joined by one "and", and both fixes are in the same tail.
    expect(chat.error.match(/Batch mode needs/g)).toHaveLength(1)
    expect(chat.error.indexOf(' and ') < chat.error.indexOf(' — ')).toBe(true)
    expect(chat.error.split('; ')).toHaveLength(2) // the two fixes, one tail
    expect(sent).toEqual([])
    expect(chat.error).toContain('Settings → MCP')
    expect(chat.error).toContain('dropdown')
  })

  // A pinned mode whose hub is open costs nothing: the guard stays quiet about
  // modes when there is nothing wrong with them.
  it('a tools-on code mode passes the guard that the connectors alone pass', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, tasks })
    await loadAgentModes()
    chat.mcp.loaded = true
    chat.mcp.enabled = { fedit: true, console: true }
    seedLot(tasks)

    await launch()

    expect(chat.error).toBeNull()
    expect(sent).toHaveLength(1)
  })

  // The pin belongs to one run: Restart (B15) forgets it, so the next lot
  // re-resolves it against whatever the modes file says by then.
  it('clearing a run drops the pin', async () => {
    const sent = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, tasks, movedAfter: { 1: Infinity } })
    await loadAgentModes()
    seedLot(tasks)

    await launch()
    expect(batch.pinnedMode).toBe(LOT_TASK_MODE_ID)

    useBatch().clearBatchRun()
    expect(batch.pinnedMode).toBeNull()
    expect(batch.modeNote).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// The big picture is a fact about the lot, not an option: there is no checkbox
// any more, so a listing that found a 00 file sends it on every task turn, and
// one that found none sends nothing.

describe('the big picture rides automatically', () => {
  beforeEach(reset)

  it('a lot with a 00 file sends its text on every task turn', async () => {
    const sent = []
    const payloads = []
    const tasks = [task(1, 'ed-01-a.md'), task(2, 'ed-02-b.md')]
    stubRoutes({ sent, payloads, tasks })
    await loadAgentModes()
    seedLot(tasks)
    batch.listing.lot = {
      name: 'ed-00-lot.md',
      path: 'todos/ed-00-lot.md',
      size: 15,
      heading: null,
      text: 'the lot big picture',
    }

    await launch()

    expect(payloads).toHaveLength(2)
    for (const payload of payloads) {
      expect(payload.batchContext).toBe('the lot big picture')
    }
  })

  it('a lot without one sends no batchContext at all', async () => {
    const sent = []
    const payloads = []
    const tasks = [task(1, 'ed-01-a.md')]
    stubRoutes({ sent, payloads, tasks })
    await loadAgentModes()
    seedLot(tasks) // listingOf() carries lot: null

    await launch()

    expect(payloads).toHaveLength(1)
    expect('batchContext' in payloads[0]).toBe(false)
  })
})
