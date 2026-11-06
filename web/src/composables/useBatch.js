import { computed, reactive, watch } from 'vue'
import { registerChatBusyGuard, useChat } from './useChat'
import { registerBatchGuard, useBenchmark } from './useBenchmark'
import { agentModesLoaded, modeById, useAgentModes } from './useAgentModes'

// ---------------------------------------------------------------------------
// Batch mode: feed a numbered lot of markdown task files to WorkBench, one
// task per chat turn, then stop — at the end of the lot, on the first turn that
// ends in error, or when a task stays unfinished for too many attempts.
//
// This composable owns *only* the batch: the conversation, the stream and the
// measurements stay in useChat, and every task turn goes through
// useChat.sendBatchTurn(). The loop is deliberately client-side (B1) — which is
// also why a browser reload discards a batch (E8): there is no server-side "a
// batch is running" to reconnect to.
//
// Three invariants the rest of the design rests on:
//   * the app never reads a task body (B13). The instruction names the file's
//     path and the model opens it with its own connectors, so the listing
//     endpoint stays the only file surface batch mode has;
//   * a task is only *finished* when its file left the lot folder. The move into
//     `<dir>/done` is the agent's own acknowledgement of completeness and, with
//     B13, the re-listing is the only way the app can read it. So after every
//     turn the loop checks it: a task whose file is still in the lot is
//     relaunched with a retry note appended to its instruction instead of the
//     lot moving on — up to MAX_TASK_RETRIES times, after which the lot aborts.
//   * the LLM context is reset per task (B6): sendBatchTurn posts one message,
//     and the screen is wiped between tasks (useChat's reset()) so a task is
//     shown on a clean conversation too — chat stream, reasoning/tool panel,
//     ctx badge and stopwatch included. Continuity between tasks comes from
//     disk (git, todos/done/) plus the system prompt — which is why the lot's
//     big picture rides on every request as `batchContext`.
// ---------------------------------------------------------------------------

// The per-task user message, verbatim. `<task_location>` is the root-relative
// path returned by the listing, which both connectors and the agent-instructions
// block resolve against the project root. It asks for no commit: a batch leaves
// its edits in the working tree, for the human to review and commit.
export const TASK_INSTRUCTION_TEMPLATE =
  "You've been given task: <task_location> to proceed. Do the coding tasks " +
  'needed to implement the feature. Once done move it into the todos/done ' +
  'folder.'

// How many times an unfinished task may be relaunched before the lot is given
// up on (the spec's "n is 1 to 3"). Fixed rather than configurable: an agent
// that forgot the move needs one nudge, and every attempt costs a full turn of
// the whole lot.
export const MAX_TASK_RETRIES = 3

// Appended to the instruction of a relaunched task, so the model is told *why*
// it is being asked again — the file not having left the lot folder is the only
// evidence the app has that the task is unfinished (the usual causes are context
// overflow, a connectivity error, a tool-loop cap, or simply forgetting the
// move). The wording and the markdown emphasis are the architect's, and render
// in the user bubble like the instruction itself.
export const RETRY_NOTE_TEMPLATE =
  'Important: The task as been relaunch because *it has not been moved* to ' +
  'the **todos/done** folder (retry number *<retry_number>*)'

// Client-side mirror of the backend PREFIX_PATTERN (core/batch.py): the same
// single check that keeps the query out of glob-injection territory, so an
// illegal prefix is reported without a round-trip.
export const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/

// The connectors the instruction depends on: reading and editing the task's
// files, and moving them into done/. With either off, every task would fail the
// same way, so Launch refuses rather than burning the whole lot to prove the
// point. These are the connector ids declared in the workspace's
// .nojae/mcp-settings.json.
export const REQUIRED_MCP_CONNECTORS = [
  { id: 'fedit', what: 'read and edit the task files' },
  { id: 'console', what: 'run mv into todos/done' },
]

// The mode a lot runs its tasks in (D13). `code` is the *name* of the contract
// — a lot executes coding tasks, so its turns must not inherit whatever the
// composer happens to show, and they must not be at the mercy of a dropdown the
// user moved while task 4 was streaming. The id is resolved against *this*
// workspace's modes rather than being sent blind: a workspace whose author
// renamed the coding mode has no `code`, and would silently get a different
// agent for its tasks.
export const LOT_TASK_MODE_ID = 'code'

export function taskInstruction(taskLocation) {
  return TASK_INSTRUCTION_TEMPLATE.replace('<task_location>', taskLocation)
}

// The instruction of attempt N of a task: the same template plus the retry note.
// Attempt 0 — the first try — gets the plain instruction, so a lot whose tasks
// behave is byte-identical to what it sent before retries existed.
export function taskInstructionForAttempt(taskLocation, attempt = 0) {
  const base = taskInstruction(taskLocation)
  if (attempt <= 0) return base
  return `${base}\n\n${RETRY_NOTE_TEMPLATE.replace(
    '<retry_number>',
    String(attempt),
  )}`
}

// Shown in place of the dropdown when the lot folder holds no task lot, so the
// dialog says what to do next instead of presenting an empty select.
export const NO_BATCH_ITEMS_MESSAGE =
  'There is no existing batch items available. Create batch tasks with the ' +
  'Todo builder agent.'

const state = reactive({
  // --- dialog state ---
  open: false,
  prefix: '', // the prefix picked in the dropdown ('ed-')
  dir: 'todos', // relative to the working location
  // The dropdown's options, from GET /api/batch/prefixes: one entry per prefix
  // that has at least one task in `dir` ({ prefix, tasks, hasLot }). Names
  // only -- batch mode reads no file body (B13), and a prefix is offered exactly
  // when listing it would give a non-empty lot.
  prefixes: [],
  prefixesLoading: false,
  listing: null, // payload of GET /api/batch/tasks
  loading: false, // the lot listing fetch
  error: null, // fetch / validation message inside the dialog
  // --- run state ---
  active: false, // batch_in_progress — the one flag the UI locks on (B10/B11)
  total: 0, // taskCount (the highest task number, not a file count)
  index: 0, // highest task number whose turn ran to its end (a gap in the
            // numbering still fills the bar honestly; a stopped or failed turn
            // does not advance it — Stop on task 3 leaves the bar at 2/3)
  current: null, // { number, name, taskLocation } of the streaming turn
  // One entry per *turn* started — a relaunched task has one entry per attempt:
  // { number, attempt, outcome, moved }. `outcome` is how the turn ended
  // ('done' | 'capped' | 'truncated' | 'stopped' | 'not-started' | 'error'),
  // never a verdict on the task itself; `moved` is the completion check's
  // verdict for that turn — true (the file left the lot folder), false (it is
  // still there, so the task is relaunched) or null (the check could not run).
  outcomes: [],
  status: 'idle', // idle | running | stopped | failed | finished
  // D13: the agent mode this lot's task turns run in, resolved once at launch
  // (`LOT_TASK_MODE_ID` against this workspace's modes) and read per turn, so a
  // dropdown change during task 7 cannot reach task 8. null = unresolved (the
  // workspace has no `code` mode, or the list never loaded): the task turns ask
  // the server for its default, and `modeNote` says so.
  pinnedMode: null,
  // One quiet line in the batch status area explaining the pin: either that the
  // workspace has no `code` mode, or — when the pinned mode closes a hub — what
  // that means for the tasks. null while nothing needs saying.
  modeNote: null,
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

// Whether a finished turn moves the bar on. A turn that ended with the task
// file still in the lot folder does not: that task is about to be done again,
// so counting it would read as progress the lot has not made. `moved === null`
// (the check could not run) does count — see taskMovedOutOfLot.
export function taskCountedForProgress({ outcome, moved }) {
  return TURN_COMPLETED.has(outcome) && moved !== false
}

// What the loop does after one attempt at a task. A table rather than inline
// branches so the retry rule is one testable function (the shape
// lib/inference-state.js uses) and the loop reads as its actions:
//   'failed'  — the turn itself failed: the lot is one job, so it is over (B8);
//   'stopped' — the user stopped it (B9), which outranks a pending retry: a
//               relaunch would restart work they just interrupted;
//   'abort'   — the file is still in the lot and the retry budget is spent, so
//               the *remaining* tasks are not started either;
//   'retry'   — relaunch this same task, one retry note further;
//   'next'    — nothing says unfinished: move on.
export function nextBatchAction({ outcome, moved, attempt, active }) {
  if (outcome === 'error' || outcome === 'not-started') return 'failed'
  if (!active || outcome === 'stopped') return 'stopped'
  if (moved === false) return attempt >= MAX_TASK_RETRIES ? 'abort' : 'retry'
  return 'next'
}

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
  const { state: modes } = useAgentModes()

  // A refusal about the conversation lands on the WorkBench error line: the
  // dialog is closed by then, so the message has to be visible where the user
  // already is.
  function fail(message) {
    chat.error = message
  }

  // The dropdown's options: every prefix that has at least one task in `dir`.
  // Fires when the dialog opens and whenever `dir` changes — the folder is the
  // only thing the option list depends on. A folder that cannot be listed is
  // reported on the dialog's error line (the same honesty as the listing): an
  // empty dropdown next to a silent failure would read as "no tasks", which is
  // a different and much less actionable statement.
  async function fetchPrefixes() {
    state.error = null
    state.prefixesLoading = true
    try {
      const query = new URLSearchParams({ dir: state.dir.trim() })
      const res = await fetch(`/api/batch/prefixes?${query}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.prefixes = []
        state.error = describeError(
          body,
          `Could not list the task prefixes (${res.status}).`,
        )
        return false
      }
      state.prefixes = Array.isArray(body?.prefixes) ? body.prefixes : []
      // A prefix that no longer has a lot (its files moved into done/, or the
      // folder changed) must not stay selected: Launch would have nothing to
      // run, and the dropdown would show a value that is not among its options.
      if (!state.prefixes.some((item) => item.prefix === state.prefix)) {
        state.prefix = ''
        state.listing = null
      }
      return true
    } catch (exc) {
      state.prefixes = []
      state.error = `Could not reach the API: ${exc}`
      return false
    } finally {
      state.prefixesLoading = false
    }
  }

  // Fetch the lot listing for the current prefix/dir. Fires when a prefix is
  // picked, when `dir` changes, and once more right before Launch (the listing
  // could be stale by then).
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
  //
  // The lot's pinned mode belongs to the *same* check (D13): a mode whose
  // `mcptools` is false makes those connectors unreachable for every task,
  // which is the identical failure — one list, one reason, one place the user
  // learns why Launch is disabled. Nothing is pinned here; this only judges.
  function ensureToolsAvailable() {
    const needs = []
    const fixes = []
    if (chat.mcp.loaded) {
      const missing = REQUIRED_MCP_CONNECTORS.filter(
        (connector) => chat.mcp.enabled[connector.id] !== true,
      )
      if (missing.length) {
        const names = missing.map(
          (connector) => `${connector.id} (${connector.what})`,
        )
        needs.push(`the MCP connectors ${names.join(' and ')}`)
        fixes.push(
          'activate them (Settings → MCP), then tick them in the WorkBench ' +
            'MCP panel for this conversation',
        )
      }
    }
    const mode = lotMode()
    if (mode && mode.mcptools === false) {
      needs.push(
        `a \`${LOT_TASK_MODE_ID}\` mode with MCP tools — this one has none, ` +
          'so a task could not move its own file',
      )
      fixes.push('switch the WorkBench dropdown to a mode that has tools')
    }
    if (!needs.length) return true
    fail(
      `Batch mode needs ${needs.join(' and ')} — ${fixes.join('; ')} and ` +
        'launch again. Neither can change mid-lot.',
    )
    return false
  }

  // The workspace's mode a lot runs its tasks in, or null when there is none to
  // pin — the modes list could not be read, or this workspace renamed the coding
  // mode. Never a guess: the id is the app's contract (`LOT_TASK_MODE_ID`), the
  // list belongs to the workspace.
  function lotMode() {
    return agentModesLoaded() ? modeById(LOT_TASK_MODE_ID) : null
  }

  // D13, step 2: hold that verdict for the whole run. It is read per turn —
  // never copied into the composer's selection — so a dropdown change during
  // task 7 cannot reach task 8. An unresolved pin is *not* a refusal: the tasks
  // then carry no `agentMode` and the server's default answers, which is what a
  // lot ran under before modes existed, plus one line saying so.
  function pinLotMode() {
    const mode = lotMode()
    state.pinnedMode = mode ? LOT_TASK_MODE_ID : null
    if (mode) {
      state.modeNote = null
      return
    }
    // The two ways there is nothing to pin are worth telling apart: a workspace
    // that renamed the coding mode runs on *its* default, and one whose list
    // could not be read runs on whatever the server decides.
    state.modeNote = agentModesLoaded()
      ? `no "${LOT_TASK_MODE_ID}" mode in this workspace; tasks run in ` +
        `"${modes.default ?? LOT_TASK_MODE_ID}"`
      : 'the agent-mode list could not be read; tasks run in the server default'
  }

  // A second, quiet listing of the same lot — the completion check's only
  // source. Deliberately *not* `validate()`: the dialog is closed while a lot
  // runs, so a failed fetch must not write into `state.error`/`state.listing`
  // (which would leave the launch-time listing inconsistent with the loop that
  // is still iterating it). `openDialog()` re-lists on its own when the user
  // looks again.
  // Returns the payload, or null when the folder could not be listed.
  async function relistLot() {
    try {
      const query = new URLSearchParams({
        prefix: state.prefix.trim(),
        dir: state.dir.trim(),
      })
      const res = await fetch(`/api/batch/tasks?${query}`)
      if (!res.ok) return null
      return await res.json()
    } catch {
      return null
    }
  }

  // Did the task file leave the lot folder? That move into `<dir>/done` is the
  // agent's own acknowledgement of completeness, and — because batch mode reads
  // no task body (B13) and has no stat route — the re-listing above is the only
  // way the app can see it.
  //
  // Returns `false` when the fresh listing still carries the task's
  // `taskLocation` (unfinished: relaunch it), `true` when it is gone, and
  // `null` when the check could not run at all (listing failed, or the lot
  // folder itself disappeared). `null` must not be folded into `false`:
  // retrying a finished task on an outage costs a full turn, and a lot whose
  // folder vanished cannot be judged by anything this endpoint can say.
  //
  // "Gone from the listing" is the honest limit of what is checkable: a file
  // renamed inside the lot folder to a name the prefix no longer matches also
  // reads as moved, and the app has no route to tell the two apart.
  async function taskMovedOutOfLot(task) {
    const fresh = await relistLot()
    if (!fresh || !Array.isArray(fresh.tasks)) return null
    return !fresh.tasks.some((item) => item.taskLocation === task.taskLocation)
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
  //
  // `attempt` is how many times this task has already been relaunched: it only
  // selects the message text (the retry note), never a different flow.
  //
  // The turn carries the lot's pinned mode (D13), read fresh each turn and
  // never the composer's selection: every task of one lot runs under one set of
  // rules whatever the dropdown is showing while they stream. A null pin (no
  // `code` mode in this workspace) sends no `agentMode` at all, so the server's
  // default answers — see `pinLotMode`.
  async function runOneTurn(task, attempt = 0) {
    // The lot's big picture rides on every task turn whenever the listing found
    // one: there is no opt-out — a 00 file is the lot's context, and the per-task
    // context reset (B6) is exactly why it must not be left to a checkbox.
    const lotText = state.listing?.lot?.text ? state.listing.lot.text : null
    // E14: a turn that never fires would hang the loop on a phase change that
    // never happens, so sendBatchTurn's boolean decides immediately.
    if (
      !sendBatchTurn(taskInstructionForAttempt(task.taskLocation, attempt), {
        batchContext: lotText,
        agentMode: state.pinnedMode,
      })
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
    // Resolved once per run, after the refusals (a lot that never launched owns
    // no pin) and read per turn (see `runOneTurn`) — so a dropdown change
    // during task 7 cannot reach task 8. The D7 lock can never conflict with
    // it: a lot starts on an empty thread and every task turn begins on one,
    // and a thread with no turns holds no lock — so the mode the lot was
    // *authored* in was already released before the first task turn. A lot
    // never runs as that mode; it runs in this one, which is also why the
    // composer's select is inert for the whole run (am-07's rendering rule).
    pinLotMode()

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
        let attempt = 0
        // One task, retried until its file leaves the lot folder (or until the
        // retry budget is spent). `attempt` is also the retry number quoted in
        // the instruction of the next try.
        for (;;) {
          const outcome = await runOneTurn(task, attempt)
          const record = { number: task.number, attempt, outcome, moved: null }
          // The completion check only speaks about a turn that reached its end;
          // a Stop, an error or a turn that never started is judged by its own
          // rule, not by the listing.
          if (TURN_COMPLETED.has(outcome)) {
            record.moved = await taskMovedOutOfLot(task)
          }
          state.outcomes.push(record)
          if (taskCountedForProgress(record)) state.index = task.number

          const action = nextBatchAction({
            outcome,
            moved: record.moved,
            attempt,
            active: state.active, // Stop between frames wins over a pending retry
          })
          if (action === 'retry') {
            attempt += 1
            // A retry is a fresh turn on a fresh context, exactly like the step
            // to the next task: the model gets a clean conversation carrying one
            // more piece of information — the note that the move never happened.
            startNextTaskOnCleanSlate()
            continue
          }
          if (action === 'next') break
          if (action === 'abort') {
            // The WorkBench error line is where a lot explains itself: the
            // dialog is closed by then, and the progress bar's `outcomes`
            // already name the task and its attempts.
            fail(
              `Batch aborted: ${task.taskLocation} was still in the lot folder ` +
                `after ${MAX_TASK_RETRIES} retries — its file never reached the ` +
                'done folder, so the remaining tasks were not started.',
            )
          }
          state.status = action === 'stopped' ? 'stopped' : 'failed'
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

  // The lot folder changed (or the dialog opened): the option list is the first
  // thing to re-read, because it is what the selected prefix belongs to. Only a
  // prefix that survived the re-fetch is worth listing — `fetchPrefixes` clears
  // the selection when it did not, and listing an empty one would put a folder
  // error on the dialog for a folder the user just typed.
  async function refreshLot() {
    await fetchPrefixes()
    if (state.prefix.trim()) await validate()
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
    // The mode pin belongs to one run: a lot about to be re-launched resolves it
    // again (`pinLotMode` in `launch`), and a cleared run has no line to keep.
    state.pinnedMode = null
    state.modeNote = null
  }

  // Cancel / Esc / backdrop click: the form is dropped, nothing is launched.
  function closeDialog({ reset = false } = {}) {
    state.open = false
    if (!reset) return
    state.prefix = ''
    state.dir = 'todos'
    state.prefixes = []
    state.prefixesLoading = false
    state.listing = null
    state.loading = false
    state.error = null
  }

  function openDialog() {
    if (state.active || benchmark.status === 'running') return
    state.error = null
    state.open = true
    // A lot that just ran left its tasks in `done/`, so both the option list and
    // the count are re-read: a prefix whose lot is empty now must disappear from
    // the dropdown, which is what keeps a finished task from being launched a
    // second time.
    refreshLot()
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

  // The empty case: the folder was read and holds no task lot at all. Not while
  // it is still being listed, and not when the read itself failed — that already
  // has its own line, and "no batch items" would be a lie about a folder the app
  // could not open.
  const noBatchItems = computed(
    () =>
      !state.prefixesLoading &&
      state.prefixes.length === 0 &&
      !state.error,
  )

  return {
    state,
    canLaunch,
    countLabel,
    skippedLabel,
    lotLabel,
    noBatchItems,
    refreshLot,
    validate,
    openDialog,
    closeDialog,
    requestLaunch,
    launch,
    stopBatch,
    clearBatchRun,
  }
}
