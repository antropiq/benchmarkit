import { computed, reactive, watch } from 'vue'
import { registerChatBusyGuard, useChat } from './useChat'
import { registerBatchGuard, useBenchmark } from './useBenchmark'

// ---------------------------------------------------------------------------
// Batch mode: feed a numbered lot of markdown task files to WorkBench, one
// task per chat turn, then stop (at the end of the lot, or on the first turn
// that ends in error).
//
// This composable owns *only* the batch: the conversation, the stream and the
// measurements stay in useChat, and every task turn goes through
// useChat.sendBatchTurn(). The loop is deliberately client-side (B1) — which is
// also why a browser reload discards a batch (E8): there is no server-side "a
// batch is running" to reconnect to.
//
// Two invariants the rest of the design rests on:
//   * the app never reads a task body (B13). The instruction names the file's
//     path and the model opens it with its own connectors, so the listing
//     endpoint stays the only file surface batch mode has;
//   * the LLM context is reset per task (B6): sendBatchTurn posts one message,
//     and the screen is wiped between tasks (useChat's reset()) so a task is
//     shown on a clean conversation too — chat stream, reasoning/tool panel,
//     ctx badge and stopwatch included. Continuity between tasks comes from
//     disk (git, todos/done/) plus the system prompt — which is why the lot's
//     big picture rides on every request as `batchContext`.
// ---------------------------------------------------------------------------

// The per-task user message, verbatim (the architect's wording, including the
// `commited` spelling — not to be silently rephrased). `<task_location>` is the
// root-relative path returned by the listing, which both connectors and the
// agent-instructions block resolve against the project root.
export const TASK_INSTRUCTION_TEMPLATE =
  "You've been given task: <task_location> to proceed. Do the coding tasks " +
  'needed to implement the feature. Once done stage your modified file and ' +
  'create a concise commit message. When the task is commited move it into the ' +
  'todos/done folder.'

// Client-side mirror of the backend PREFIX_PATTERN (core/batch.py): the same
// single check that keeps the query out of glob-injection territory, so an
// illegal prefix is reported without a round-trip.
export const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/

// The connectors the instruction depends on: reading and editing the task's
// files, and staging/committing/moving them. With either off, every task would
// fail the same way, so Launch refuses rather than burning the whole lot to
// prove the point. These are the connector ids declared in externals.json.
export const REQUIRED_MCP_CONNECTORS = [
  { id: 'fedit', what: 'read and edit the task files' },
  { id: 'console', what: 'run git add / git commit / mv into todos/done' },
]

export function taskInstruction(taskLocation) {
  return TASK_INSTRUCTION_TEMPLATE.replace('<task_location>', taskLocation)
}

const state = reactive({
  // --- dialog state ---
  open: false,
  prefix: '', // 'ed-'
  dir: 'todos', // relative to the working location
  includeLot: true, // send the 00 body as `batchContext` on every task (B7)
  listing: null, // payload of GET /api/batch/tasks
  loading: false, // the blur-time fetch
  error: null, // fetch / validation message inside the dialog
  // --- run state ---
  active: false, // batch_in_progress — the one flag the UI locks on (B10/B11)
  total: 0, // taskCount (the highest task number, not a file count)
  index: 0, // highest task number whose turn ran to its end (a gap in the
            // numbering still fills the bar honestly; a stopped or failed turn
            // does not advance it — Stop on task 3 leaves the bar at 2/3)
  current: null, // { number, name, taskLocation } of the streaming turn
  // One entry per task started: 'done' | 'capped' | 'truncated' | 'stopped' |
  // 'not-started' | 'error'. These are *turn* outcomes, not verdicts: with B13
  // the app never sees the task body, so it cannot know whether the model
  // staged, committed or moved anything — the bar's title says so.
  outcomes: [],
  status: 'idle', // idle | running | stopped | failed | finished
})

// A batch owns the conversation even in the fraction of a second between two
// task turns, when useChat.state.phase is back to 'idle'. Registering the
// predicate here lets useChat's hub guards (B11) see it without useChat
// importing this module: the dependency direction stays acyclic
// (useBatch -> useChat -> nothing new).
registerChatBusyGuard(() => state.active)
// ... and the same for the tab/settings guards owned by useBenchmark.
registerBatchGuard(() => state.active)

// Which turn outcomes count as a completed task for the bar. `stopped` and
// `error`/`not-started` do not: the bar's number is the count of tasks whose
// turn ran to its end, so a Stop on task 3 leaves it reading 2/3.
const TURN_COMPLETED = new Set(['done', 'truncated', 'capped'])

function describeError(body, fallback) {
  const detail = body?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      const field = Array.isArray(item?.loc)
        ? item.loc.filter((part) => part !== 'query' && part !== 'body').join('.')
        : ''
      return field ? `${field}: ${item?.msg ?? 'invalid'}` : item?.msg ?? 'invalid'
    })
    if (messages.length) return messages.join(' · ')
  }
  return fallback
}

export function useBatch() {
  // `reset: resetChat` — `reset` is already a local name inside closeDialog's
  // options, and the between-task wipe below must not read like a dialog reset.
  const { state: chat, sendBatchTurn, stop, reset: resetChat } = useChat()
  const { state: benchmark, setTab } = useBenchmark()

  // A refusal about the conversation lands on the WorkBench error line: the
  // dialog is closed by then, so the message has to be visible where the user
  // already is.
  function fail(message) {
    chat.error = message
  }

  // Fetch the lot listing for the current prefix/dir. Fires on blur, on Enter,
  // and once more right before Launch (the listing could be stale by then).
  async function validate() {
    const prefix = state.prefix.trim()
    state.error = null
    if (!prefix) {
      state.listing = null
      return false
    }
    if (!PREFIX_PATTERN.test(prefix)) {
      state.error =
        "Prefix must start with a letter or digit, then letters, digits, '_' " +
        'or "-" — no path or glob characters.'
      state.listing = null
      return false
    }
    state.loading = true
    try {
      const query = new URLSearchParams({ prefix, dir: state.dir.trim() })
      const res = await fetch(`/api/batch/tasks?${query}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.error = describeError(
          body,
          `Could not list the task lot (${res.status}).`,
        )
        state.listing = null
        return false
      }
      state.listing = body
      return true
    } catch (exc) {
      state.error = `Could not reach the API: ${exc}`
      state.listing = null
      return false
    } finally {
      state.loading = false
    }
  }

  // Launch needs WorkBench at its initial state (B14): a task lot on top of a
  // human thread is meaningless, and the transcript is never auto-purged — so
  // refusing beats silently wiping somebody's conversation.
  function ensureFreshChat() {
    if (chat.phase !== 'idle') {
      fail('Cannot start a batch while a reply is streaming.')
      return false
    }
    if (chat.messages.length > 0) {
      fail('Restart the conversation first — a batch starts on an empty thread.')
      return false
    }
    return true
  }

  // The batch posts the Chat tab's current mcpEnabled allowlist, so a connector
  // switched off here stays off for every task. Checked against the
  // conversation-scoped filters, not the Settings-activated set. When the
  // snapshot could not load the request carries no filter at all (the server
  // keeps every activated connector), so there is nothing to judge.
  function ensureToolsAvailable() {
    if (!chat.mcp.loaded) return true
    const missing = REQUIRED_MCP_CONNECTORS.filter(
      (connector) => chat.mcp.enabled[connector.id] !== true,
    )
    if (!missing.length) return true
    const names = missing.map((connector) => `${connector.id} (${connector.what})`)
    fail(
      `Batch mode needs the MCP connectors ${names.join(' and ')} — switch them ` +
        'on (Settings → MCP to activate one, then the WorkBench MCP panel for ' +
        'this conversation) and launch again. They cannot change mid-lot.',
    )
    return false
  }

  // One-shot wait on the signal every component already trusts: phase returns
  // to 'idle' from finish(), from the pause_ack case, or from the non-OK path —
  // i.e. after every terminal frame of a reply.
  function waitUntilIdle() {
    return new Promise((resolve) => {
      if (chat.phase === 'idle') {
        resolve()
        return
      }
      const stopWatching = watch(
        () => chat.phase,
        (phase) => {
          if (phase !== 'idle') return
          stopWatching()
          resolve()
        },
      )
    })
  }

  // One task turn: hand the instruction to useChat, wait for the terminal
  // frame, then read the outcome off the message the placeholder became and off
  // the chat error line. No new event, no new frame, nothing in core/.
  async function runOneTurn(task) {
    const lotText =
      state.includeLot && state.listing?.lot?.text ? state.listing.lot.text : null
    // E14: a turn that never fires would hang the loop on a phase change that
    // never happens, so sendBatchTurn's boolean decides immediately.
    if (
      !sendBatchTurn(taskInstruction(task.taskLocation), { batchContext: lotText })
    ) {
      return 'not-started'
    }
    await waitUntilIdle()

    // Stop wins over everything else: the partial turn went the way an
    // interactive Stop takes it, and the reason we are back at idle is the user.
    if (!state.active) return 'stopped'
    const message = chat.messages[chat.messages.length - 1]
    // E4: a server-cut stream counts as done and the lot moves on — but it has
    // to be read before `chat.error`, because the truncation frame sets both.
    if (message?.truncated) return 'truncated'
    // E16 / C2: the reply hit the shared tool-iteration cap and still ended as
    // a normal `done`. Both numbers come from the server-reported frames; the
    // client never re-derives the limit, and never advises about the value.
    if (message && message.mcpMax > 0 && message.mcpIteration >= message.mcpMax) {
      return 'capped'
    }
    // E3: an errored turn aborts the batch — the lot is one job, and 8 identical
    // failures teach nothing while each costs a round-trip.
    if (chat.error) return 'error'
    return 'done'
  }

  // Between two tasks: wipe the visible conversation exactly as the Restart
  // button does. The LLM context was already per-task (sendBatchTurn posts one
  // message), but the browser thread kept task N's bubbles — so task N+1 opened
  // with its bubbles and its reasoning blocks already on screen, and the ctx
  // badge still read task N's occupancy (`contextUsage` walks the messages
  // backwards and a fresh placeholder carries contextTokens 0) until task
  // N+1's first `usage` frame landed, which is what made the context look like
  // it carried over. Only called when a next turn really follows: a stopped or
  // failed lot keeps its residue, like an interactive Stop (B9).
  function startNextTaskOnCleanSlate() {
    // refetchHubs: false — the wipe must not unlock and re-seed the
    // conversation-scoped MCP/LSP toggles: the whole lot runs under one
    // identical tool set (§2.2), and a connector activated mid-lot must not
    // join task N+1's tools.
    resetChat({ refetchHubs: false })
  }

  async function launch() {
    if (state.active || !canLaunch.value) return // one batch at a time
    if (!ensureFreshChat()) return
    if (!ensureToolsAvailable()) return

    const tasks = state.listing.tasks
    state.total = state.listing.taskCount
    state.index = 0
    state.outcomes = []
    state.status = 'running'
    state.active = true // batch_in_progress — the flag the UI locks on
    try {
      for (const task of tasks) {
        if (!state.active) return // Stop won between the turns (E2)
        state.current = task
        const outcome = await runOneTurn(task)
        state.outcomes.push(outcome)
        // The bar counts finished *turns*: a Stop or a failure leaves the bar at
        // the last task that really ran, which is what stays honest on screen
        // (and `current` is what names the task that was interrupted).
        if (TURN_COMPLETED.has(outcome)) state.index = task.number
        if (outcome === 'error' || outcome === 'not-started') {
          state.status = 'failed'
          return
        }
        // The lot moves on: hand the next task a clean screen. The last task
        // of the lot is not wiped — its transcript is the lot's outcome, and
        // the user can still save it or read what the last turn did.
        if (state.active && task !== tasks[tasks.length - 1]) {
          startNextTaskOnCleanSlate()
        }
      }
      if (state.status === 'running') state.status = 'finished'
    } finally {
      state.active = false
      state.current = null
    }
  }

  // The dialog's Launch, in this exact order: re-validate, switch to the
  // WorkBench tab *before* the lock-down flag exists (E11 — the setTab guard
  // added for a running batch would otherwise block the launch's own tab
  // switch), close the dialog, then run.
  async function requestLaunch() {
    if (state.active || state.loading) return
    if (!PREFIX_PATTERN.test(state.prefix.trim())) {
      await validate() // shows the message in the dialog, launches nothing
      return
    }
    await validate()
    if (benchmark.status === 'running') {
      state.error =
        'A benchmark run is in progress — stop it before launching a batch.'
      return
    }
    if (!canLaunch.value) return
    setTab('chat')
    state.open = false
    await launch()
  }

  // Stop kills the whole lot (B9); the transcript is kept, exactly like an
  // interactive Stop, and there is no resume. The loop's inter-task guard reads
  // `state.active`, so the flag is cleared here rather than in the loop.
  function stopBatch() {
    if (!state.active) return
    state.active = false
    state.status = 'stopped'
    stop() // useChat.stop(): aborts the in-flight turn
  }

  // Restart (B15) means Stop + wipe: the one way back to a launchable state
  // after a stopped batch. Pair this with useChat's reset() — otherwise the
  // loop is left stranded on a wiped conversation, the same class of bug the
  // pause lot had to fix for pausePending.
  function clearBatchRun() {
    stopBatch()
    state.status = 'idle'
    state.total = 0
    state.index = 0
    state.current = null
    state.outcomes = []
  }

  // Cancel / Esc / backdrop click: the form is dropped, nothing is launched.
  function closeDialog({ reset = false } = {}) {
    state.open = false
    if (!reset) return
    state.prefix = ''
    state.dir = 'todos'
    state.includeLot = true
    state.listing = null
    state.loading = false
    state.error = null
  }

  function openDialog() {
    if (state.active || benchmark.status === 'running') return
    state.error = null
    state.open = true
    // A lot that just ran leaves its prefix in the form (Launch closes without
    // resetting it), so reopening re-lists: the tasks whose files the model
    // already moved into `done/` must be gone from the count, and a stale
    // listing would let them be launched a second time.
    if (state.prefix.trim()) validate()
  }

  const canLaunch = computed(() => {
    if (state.active || state.loading) return false
    if (!PREFIX_PATTERN.test(state.prefix.trim())) return false
    const listing = state.listing
    return !!listing && listing.taskCount > 0 && listing.tasks.length > 0
  })

  // The dialog's count line: how many task files are actually in the lot —
  // `tasks.length`, not `taskCount`. `taskCount` is the *highest* task number
  // (the bar's scale, so a numbering gap still fills it honestly), which would
  // read as "N left to do" when earlier numbers already moved into done/.
  // '—' before any listing.
  const countLabel = computed(() =>
    state.listing ? `Found tasks ${state.listing.tasks.length}` : '—',
  )

  const skippedLabel = computed(() => {
    const skipped = state.listing?.skipped ?? []
    if (!skipped.length) return ''
    const plural = skipped.length === 1 ? 'file skipped' : 'files skipped'
    const reasons = [...new Set(skipped.map((item) => item.reason))]
    return `${skipped.length} ${plural} (${reasons.join('; ')})`
  })

  const lotLabel = computed(() => {
    const lot = state.listing?.lot
    return lot ? `big picture: ${lot.name}` : 'no 00 file'
  })

  return {
    state,
    canLaunch,
    countLabel,
    skippedLabel,
    lotLabel,
    validate,
    openDialog,
    closeDialog,
    requestLaunch,
    launch,
    stopBatch,
    clearBatchRun,
  }
}
