# Batch mode — run a `todos/` task lot through WorkBench

A third way to drive the model: instead of one question typed by hand, the app
feeds a **numbered lot of markdown task files** to WorkBench, one task per chat
turn, and streams the result exactly like an interactive conversation. The user
launches it from a dialog (prefix → number of tasks), watches a progress bar,
and the **only** controls still enabled while it runs are **Stop** (and
**Restart**, which means Stop + wipe — B15).

Why the lot is cut into tasks at all, stated up front because it drives the
whole design: these are tasks that **do not fit in one context window**. So the
LLM context is **reset after every task** — task *N+1* is sent a one-message
conversation (its instruction), not the transcript of tasks 1…*N*. What carries
over between tasks is what is on disk (the code, git, `todos/done/`) plus the
fixed system prompt (configured prompt + `AGENTS.md` + the lot's big picture).
The browser transcript, by contrast, is *not* wiped: it keeps accumulating so
the user can still follow the whole lot (§2.4).

The lot is the one this very folder uses: `todos/ed-00-big-picture.md` (the
big picture) followed by `ed-01…ed-NN` (the tasks). That convention is already
in use and already understood by the model — this feature automates the loop a
human runs today by hand, one task message at a time.

## Verdict: feasible — no new execution engine

One read-only listing endpoint, one optional `ChatIn` field, and one extra
system-prompt block: that is the entire backend. The heavy lifting is already
built and already single-owner:

| Need | Already there |
|---|---|
| One streamed model turn, with the MCP/LSP tool loop | `core/chat.py:127 stream_chat()`, driven by `POST /api/chat` (`routes.py:1167`) |
| A per-request message list, no server-side conversation | `useChat.js:893` rebuilds `messages` on every send, so a *fresh* one-message list is one expression, not a new mode |
| A system message the server owns and re-prepends per request | `core/chat.py:188-207` — configured `sysprompt`, then `TOOL_NOTICE`, then `AGENTS.md` (`config.py:306/322`) |
| A hard stop that works mid-stream **and** mid-tool-call | `controller.abort()` + the SSE generator's `finally: cancel_event.set()` (`routes.py:1248-1252`) |
| Live chat stream, reasoning stream, tool badges, tok/s, ctx, stopwatch | `WorkBenchView.vue`, `WorkBenchToolPanel.vue`, `WorkBenchSidePanel.vue`, `ChatStopwatch.vue` |
| A place to hang a progress bar under the stopwatch | `WorkBenchSidePanel.vue:132`, right after `<ChatStopwatch />` |
| A reusable meter | `GaugeBar.vue` (or the `<progress>` of `.progress-row`, `theme.css:787`) |
| An app-bar pill already driven by chat state | `AppBar.vue:22-36 pillLabel` / `pillClass` reading `useChat.state.phase` |
| The model reading and editing task files itself | `fedit-mcp` (`read_file`, rooted at `--root-dir`) and `console-mcp` — `git add`, `git commit` and `mv todos/x.md todos/done/` are all **allowed** (verified with `check_command_allowed`, the same parse `execute_command` applies) |
| A `<dialog>` pattern to copy | `RegistrationMemoryDlg.vue` (watch + `showModal()` + backdrop + native close) |

So batch mode is **only** an orchestrator: a list of tasks, and a loop that
writes the task *instruction* into the composer state and presses Send. The
task **body is never fetched by the app**: the message carries the file's path
(`<task_location>`) and the model opens the file with its own tools. That is
what removes any file-read endpoint from this lot — and any dependency on the
Editor lot (`todos/ed-01…ed-10`), which is not implemented yet. **No part of
batch mode may call `GET /api/fs/*`** (B13). Every part of the chat contract
(streaming, tools, measurements, truncation handling) keeps applying untouched,
which is what makes the run observable without new events.

Four things do not exist yet and must be added:

1. an endpoint that turns a file-name prefix into an ordered task list;
2. a frontend loop that owns "one turn, then the next" (`useBatch.js`);
3. a modal input mode that shuts down every user surface except Stop;
4. a per-request system-prompt extension (the lot's big picture) and a fresh
   message list per task — §1.2 and §2.4.

## Decisions taken (all confirmed with the architect — none open)

| # | Decision |
|---|---|
| B1 | The orchestrator is **frontend-only** (client loop over `POST /api/chat`), not a backend job with its own SSE bus. Reason: the transcript, its measurements and its streams already live client-side. A backend batch would re-own all of that to buy nothing; the Stop button already works at this level — `useChat.stop()` aborts the fetch and the route's `finally` closes the upstream stream. Accepted cost: no reload survival (E8). |
| B2 | The lot lives in `<BENCHMARKIT_PATH_LOCATION>/todos` — the same working location the hubs, the tree and `AGENTS.md` use (`config.py:281 path_location()`). Overridable by a field in the dialog, default `todos`. |
| B3 | A task is a **`.md` file whose name starts with `<prefix>`**, the `<prefix>` matching `^[A-Za-z0-9][A-Za-z0-9_-]*$` (the spec's `ed-`). The app never sends the file's body: the user message names its **path**, and the model reads the file itself (B13). |
| B4 | The `NN` number is read from the characters that follow the prefix (`ed-01-…` → 1). Sort is numeric ascending; the `NN` need not be contiguous. A matching file with **no** number after the prefix (`ed-foo.md`) is not a task and lands in the endpoint's `skipped` list. |
| B5 | `NN == 00` is the lot description, **never a task** — but the spec's "Found tasks 8" is the highest task number, not a file count (the spec's own example: `ed-00` + `ed-01…ed-08` = 8 tasks). `taskCount = max(NN)`; gaps, the `00` file and non-numeric siblings are reported in a second caption line. |
| B6 | **One chat turn per task, and the LLM context is reset between tasks.** Task *N+1* is posted as a one-message conversation (`messages: [instruction]`); the server re-prepends its own system message (`core/chat.py:188`), so nothing leaks from task *N*. Continuity comes from disk — the committed code, git history, `todos/done/` — and from the system prompt, not from the transcript. This is the point of the feature: the lot cannot be done in one context window. |
| B7 | The `00` big-picture file rides in the **system prompt**, never in the visible thread: a new optional `ChatIn.batchContext` field, appended server-side next to `agent_instructions` (`core/chat.py:201-207`). It is invisible to the user and identical on every task, which is exactly what a per-task context reset needs. |
| B8 | A turn that ends in `error` **aborts the batch** (the batch is one job, not 8 independent ones). A turn that ends `truncated` counts as done and moves on. Both are recorded per task in the bar's title. |
| B9 | **Stop kills the whole batch**, not just the current task. The transcript is kept, exactly like an interactive Stop. There is no resume: restarting the lot is a new Launch, on an empty thread (B14). |
| B10 | The pill shows `Batch` (`AppBar.vue:25`), class `status-batch`, and the progress bar appears **only** while `state.batch.active`. |
| B11 | During a batch, the **app bar Settings button, both tabs and the batch button are disabled**; a running batch cannot be reconfigured, and switching tabs is refused while `state.batch.active`. |
| B12 | Batch mode does not touch the benchmark tab, `RunState`, `results.json` or the verifier: it is WorkBench, not a scored run. |
| B13 | **No dependency on the Editor lot.** Batch mode reads no file through the app. It must not call `GET /api/fs/*` (specified in `todos/ed-01`/`todos/ed-03`, not implemented), and it defines no file-read route of its own either. The only endpoint it needs is the *listing* (§1.1), whose payload is names, sizes and headings — enough to count and to label, not enough to feed the model. |
| B14 | Launch requires WorkBench to be at its **initial state**: `messages.length === 0` and `phase === 'idle'`. Otherwise it is refused with a message ("Restart the conversation first"). No silent purge: a Launch that wipes a human's thread is worse than one that refuses. |
| B15 | **Restart stays enabled** during a batch and behaves as Stop + wipe (`WorkBenchInput.vue:77` already means `stop()+reset()`). This is the one deviation from "only Stop is active", agreed: with Restart disabled, a stopped batch would leave its residue on screen with no way back to a launchable state. |
| B16 | **`MAX_TOOL_ITERATION` sizing is the user's responsibility**, not batch mode's. This deployment runs it at **250** (`.env`); batch mode neither reads the value to judge it, nor warns about it, nor sets a batch-specific cap. It only *reports* a turn that hit it (E16). |
| B17 | **No batch log service in v1** (agreed, deferred to its own lot): batch mode writes nothing. See "Explicitly out of scope". |

---

## Phase 1 — backend: the task list endpoint

### 1.1 `GET /api/batch/tasks?prefix=<p>&dir=<rel>`

Lives inside `build_router()` (`routes.py:623`), thin, delegating to a core
function — the "core owns logic, API owns transport" rule. Put the discovery in
`src/core/batch.py` (a sibling of `core/job.py`; no new package is justified
for one listing). This endpoint is the **whole** backend surface batch mode
needs from disk (§1.2 and B13).

```jsonc
// GET /api/batch/tasks?prefix=ed-
{
  "dir": "todos",                          // normalised, relative to the workspace
  "workspace": "/mnt/bigfoot/dev/benchmarkit",
  "prefix": "ed-",
  "taskCount": 8,                          // max NN among the tasks → the label
  "doneFolder": true,                      // todos/done/ exists (§2.2 item 4)
  "lot": {
    "name": "ed-00-big-picture.md", "path": "todos/ed-00-big-picture.md",
    "size": 13604,
    "text": "…the whole markdown body of the 00 file…"   // null with no 00 file
  },
  "tasks": [
    { "number": 1, "name": "ed-01-workspace-navigator.md",
      "size": 7849, "taskLocation": "todos/ed-01-workspace-navigator.md" },
    …
  ],
  "skipped": [                              // never counted, never run
    { "name": "ed-07-notes.md", "reason": "no task number after the prefix" },
    { "name": "ed-99-something.md", "reason": "number 99 > MAX_BATCH_TASKS" }
  ],
  "truncated": false                        // more than MAX_BATCH_TASKS entries
}
```

`lot.text` is here because the client must be able to *send* the big picture
(§1.4) and B13 forbids a file-read route: the listing, which already opened
the folder, hands over the one body it needs. It is capped (see the rules) and
it is the only file **content** any batch route ever returns. Tasks carry
`taskLocation` — the exact string interpolated into the user message (§1.3) —
so the frontend never builds a path from parts. It is also the field to name in the bar's `aria-label`.

Rules, in the order they are applied (each one unit-testable on its own):

* `prefix` is validated against `^[A-Za-z0-9][A-Za-z0-9_-]*$` → **422**
  otherwise, with a message naming the allowed shape. This single check is what
  keeps the query out of glob-injection territory (`*`, `?`, `[`, `]`, `/`,
  NUL are all refused).
* `dir` defaults to `todos`; it is resolved with three locks, implemented in
  `core/batch.py` itself (batch mode must not import the not-yet-existing
  editor code — B13): no absolute path and no `..` component; a `realpath`
  prefix check against `path_location()`; then
  `mcp_connectors.path_policy.deny_reason(resolved, for_write=False) is None`
  (that module exists today and is the shared list the two connectors use —
  reusing it is what keeps this surface bounded like every other one, per
  `AGENTS.md`). Without the policy call, `?dir=../.ssh` would make batch mode
  a GUI for reading a secret tree.
* `dir` missing → **404** `Batch folder does not exist: todos`; `dir` is a file
  → **422**; policy refusal → **422** naming the rule.
* Only `*.md` files whose name starts with the prefix are candidates,
  case-sensitive, non-recursive: one `os.scandir` of `dir`, no descent into
  `todos/done/` or any subfolder (D2 — flat only, so a file cannot be in two
  lots at once and a completed task that was `mv`'d away is simply gone from
  the next listing). A file that does not match the prefix is not mentioned at
  all: it is outside the lot by definition, not a skipped task.
* `NN` = the longest digit run right after the prefix, `int()`-normalised
  (`ed-01` → 1). `NN == 0` → the lot file. `NN >= 1` → a task.
* `heading` (optional, dialog-only — named so it cannot be confused with the
  bar's HTML `title`) is the first `# ` heading of the file, read server-side
  and stopped at the first heading. Dropping it costs nothing to the run: the
  number is the authority, and the bar's `aria-label` can name the file alone.
* `lot.text` is read whole, capped at `batch.constants.MAX_BATCH_LOT_BYTES`
  (module attribute, patchable in tests). Over the cap → `lot.text: null` plus
  a `lotNote` ("big picture too large to inject: 812 KB") rather than a 422:
  the lot is still runnable, just without the extra system-prompt context, and
  the dialog says so.
* `taskCount = max(number)`. Report, don't guess: `skipped` explains every file
  that matched the prefix but is not a runnable task, so `Found tasks 8` next
  to an explanation line never reads as "the app lost a file".
* Cap at `batch.constants.MAX_BATCH_TASKS` (module attribute, patchable in
  tests, same convention as `lsphub.constants`) → `truncated: true`, and the
  entries past the cap go to `skipped` rather than silently vanishing.
* Empty result: **200** with `taskCount: 0` and empty lists — the dialog shows
  `Found tasks 0` and disables Launch. A 404 for "no match" would conflate
  "folder missing" and "nothing matches", which are different user messages.

### 1.2 No task-content endpoint — the model reads its own task file

The app **never** fetches a task body. The user message carries the path
(`§1.3`) and the model opens the file with `fedit-mcp`'s `read_file`, which is
rooted at `--root-dir` = `BENCHMARKIT_PATH_LOCATION`
(`fedit_mcp.py:871` resolves a relative path against `ROOT_DIR`), i.e. the same
folder the listing scanned. Consequences, all of them load-bearing:

* **zero dependency on the Editor lot** (B13): nothing in batch mode may call
  `GET /api/fs/*`, and batch mode defines no reader of its own. `GET
  /api/batch/tasks` is the only route it adds for files.
* The task is read **at the model's own pace**, possibly in several windowed
  reads (`read_file` returns 400 lines at a time) — a 40 KB task file does not
  blow up the payload, it costs context the way the model asks for it.
* Batch mode requires `fedit-mcp` (and `console-mcp` for the staging/commit/
  move steps) to be **activated**; see the Launch-time check in §2.2.
* A task file that vanishes mid-lot is no longer an HTTP 404 the app can see —
  the model reports it in its answer. E5 keeps the outcome honest without
  pretending the app detected it.

### 1.3 The per-task user message (verbatim)

One template, no variant, built by the frontend from `taskLocation`:

```text
You've been given task: <task_location> to proceed. Do the coding tasks needed
to implement the feature. Once done stage your modified file and create a
concise commit message. When the task is commited move it into the todos/done
folder.
```

with `<task_location>` replaced by `todos/ed-03-tree-selection-load.md` (the
path as returned by the listing — relative to the project root, which
`agent_instructions_block` already pins for the model: `config.py:322` emits
"The current project root is …", and both connectors resolve relative paths
against it). Store the template as a single exported constant
(`web/src/composables/useBatch.js`) so it is one line to review and one place
to change. Two notes for the implementation: the wording is the architect's
and is **not** to be silently rephrased (including `commited`/`staged`), and
`todos/done` in the last sentence follows the same root-relative rule — the
`mv` is allowed (`check_command_allowed("mv todos/x.md todos/done/")` →
ALLOWED) but `todos/done/` must exist for the `mv` to land; the listing's
`skipped` output already proves whether the lot is flat, and §2.2 has the dialog
*report* a missing `todos/done/` rather than leave the model to guess — batch
mode never creates a folder of its own (C1).

### 1.4 `ChatIn.batchContext` — the big picture, in the system prompt

The `00` file must reach the model on **every** task without ever appearing in
the visible thread (the architect's requirement, and the only sane shape once
the context is reset per task: a context turn would have to be re-typed into
each fresh conversation). Server side, three edits:

* `ChatIn` (`routes.py:266`): `batchContext: Optional[str] = None`, plus a
  length cap in the existing `@field_validator` style (reuse
  `MAX_BATCH_LOT_BYTES`, so the cap the listing enforced and the cap the chat
  accepts are the same number).
* `post_chat` (`routes.py:1213-1220`): pass it through to `stream_chat` as a
  new `batch_instructions` keyword, **alongside** `agent_instructions`, never
  merged into `sysprompt` — that separation is what keeps the `TOOL_NOTICE`
  word check judging only the user's own prompt (`core/chat.py:182-186`).
* `core/chat.py` (`:201-207`): append it as a third system-message block, in
  the order *sysprompt → TOOL_NOTICE → AGENTS.md → big picture*. The big
  picture goes last so the project instructions stay the closest thing to the
  conversation, and the block is framed like `agent_instructions_block` does —
  e.g. `"The following is the big-picture description of the task lot being
  worked through, one task at a time: …"` — so the model reads it as context,
  not as an instruction to do everything now.

Cost, stated so nobody is surprised: the big picture is re-sent once per task
(one `POST /api/chat` per task, the tool loop lives inside that one request),
not once per tool round. `ed-00-big-picture.md` here is ~14 KB ≈ 3-4k tokens —
worth it, since it is the only cross-task memory besides git.

Not validated as trusted input beyond the size cap: this app is single-user
and the same person already owns `LLM_SYSPROMPT` through the settings dialog,
so a caller-chosen system block is not a new privilege. Say that in the
docstring rather than pretending the field is untrusted.

### 1.5 The missing `POST /run` guard (server-enforced exclusion)

`_CHAT_TASKS` (`routes.py:294`) is consulted by the two hub restarts and
`POST /path-location` (`:854`, `:995`, `:1079`) but **not** by `POST /run`
(`:1281`), while `POST /chat` does refuse a run in progress (`:1180-1184`).
The asymmetry lets a benchmark start under a streaming chat turn — two
concurrent streams fighting over the single `_CHAT_HANDLE`. Add to
`start_run()`:

```python
if _CHAT_TASKS:
    raise HTTPException(
        status_code=409,
        detail="Cannot start a run while a chat reply is streaming",
    )
```

One line (plus its comment), mirroring `:1079`. Batch mode makes the race
likely (a batch turn streams for minutes, and the Settings/tabs surfaces the
UI would have locked are one reload away), so the exclusion must be enforced
by the server, not by a disabled button. Test it in `tests/test_batch.py`.

### 1.6 Tests — `tests/test_batch.py`

`tmp_path` workspace, `monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", …)`
(the pattern `tests/test_api.py:666-720` already uses), fixture-built lot:

1. `ed-00` + `ed-01..ed-08` → `taskCount == 8`, 8 tasks in numeric order,
   `lot.path` = the `00` file, `lot.text` its body, each `taskLocation`
   `todos/ed-NN-…md` (exactly what §1.3 interpolates);
2. gap (`ed-01`, `ed-05`) → `taskCount == 5`, 2 tasks, no phantom entries;
3. `ed-1x`, `ed` (no number) → `skipped` with the right reason, none counted;
   `opt-*` and `ed-03.txt` → absent from the payload entirely;
4. `?prefix=ed-;rm -rf` / `?prefix=*` / `?prefix=../` → 422;
5. `?dir=../.ssh` → 422 (policy), `?dir=nope` → 404, `?dir=/etc` → 422;
6. two-digit vs three-digit numbers sort numerically (`ed-10` after `ed-09`);
7. cap: `MAX_BATCH_TASKS + 5` files → `truncated: true`, exactly the cap;
8. no `00` file → `lot: null`, `taskCount` still right;
9. `lot.text` over `MAX_BATCH_LOT_BYTES` → `lot.text: null` + `lotNote`, still
   200 and still runnable;
10. a subfolder (`todos/done/ed-04-x.md`) is never listed, whatever its name;
11. `POST /api/chat` with `batchContext` → the outbound `messages[0]` carries
    the big-picture block **after** `AGENTS.md` (assert on the payload handed to
    `stream_chat`, patched as the existing chat tests do), and without
    `batchContext` the system message is byte-identical to today's;
12. `batchContext` over the cap → 422;
13. `POST /run` while a chat reply is streaming → 409 (§1.5).

---

## Phase 2 — `web/src/composables/useBatch.js` (the loop)

A new singleton composable, mirroring the shape of `useMemory.js` (a module-level
`reactive` state + a `useBatch()` returning actions). It owns *only* the batch;
the conversation, the stream and the measurement stay in `useChat`.

```js
const state = reactive({
  // dialog state
  open: false,
  prefix: '',            // "ed-"
  dir: 'todos',          // relative to the working location
  includeLot: true,      // send the 00 body as `batchContext` on every task (B7)
  listing: null,         // payload of GET /api/batch/tasks
  loading: false,        // the blur-time fetch
  error: null,           // fetch / validation message inside the dialog
  // run state
  active: false,         // batch_in_progress — the flag the UI locks on
  total: 0,              // taskCount
  index: 0,              // tasks finished (or in flight, for the bar label)
  current: null,         // { number, name, taskLocation } of the streaming turn
  outcomes: [],          // per task: 'done' | 'capped' | 'truncated' | 'stopped' | 'not-started' | 'error'
  status: 'idle',        // idle | running | stopped | failed | finished
})
```

### 2.1 The loop

```js
async function launch() {
  if (state.active || !canLaunch.value) return       // one batch at a time
  if (!ensureFreshChat()) { fail('Restart the conversation first'); return }   // B14
  if (!ensureToolsAvailable()) { fail('…console-mcp and fedit-mcp must be on…'); return }
  const tasks = state.listing.tasks
  state.active = true                                // batch_in_progress
  state.status = 'running'
  for (const task of tasks) {
    if (!state.active) return                        // Stop won between tasks (E2)
    state.current = task
    const instruction = taskInstruction(task.taskLocation)   // §1.3 template
    const outcome = await runOneTurn(instruction)   // resolves on the terminal frame
    // runOneTurn resolves 'not-started' if sendBatchTurn refused (E14)
    state.outcomes.push(outcome)
    state.index = task.number
    if (outcome === 'error') { state.status = 'failed'; return }   // B8 + E14
  }
  state.status = state.status === 'running' ? 'finished' : state.status
  state.active = false
  state.current = null
}
```

Two notes on what is *not* in this loop: there is no `injectContext` step (the
big picture rides on every request instead — §2.3) and there is no
`fetchTaskBody` (B13: the app never reads the task file).

`runOneTurn(text)` is the one place that talks to `useChat`, and it must not
re-implement any of it:

* it calls `setDraft(text)` then `sendBatchTurn(text)` (§2.4) — which pushes the
  user bubble and the assistant placeholder, opens the `fetch('/api/chat')` and
  runs `finish()` in its `finally`, exactly as an interactive send does;
* it waits by **watching `useChat.state.phase` back to `'idle'`** (a one-shot
  `watch`/promise, ~10 lines), which is the same signal every component already
  trusts — `phase` only returns to `idle` from `finish()` (`:842`), the
  `pause_ack` case (`:704`), or the non-OK path (`:966`), i.e. after every
  terminal frame;
* it reads the turn's outcome off the message the placeholder became:
  `state.messages.at(-1).truncated`, and `useChat.state.error` for a hard
  failure (B8). No new event, no new frame, nothing in `core/`.

Stop is then free: `useChat.stop()` (`:999`) aborts the fetch, `finish()` sets
`phase = 'idle'`, `runOneTurn` resolves with `'stopped'`, and `launch()`'s next
guard sees `state.active === false` — set by the wrapper below, not by the loop.

**What the app cannot know.** With B13 the app never sees the task body, so it
cannot verify the three things the instruction asks for (files staged, commit
created, file moved to `todos/done/`). `outcomes` therefore records how the
*turn* ended, not whether the task succeeded — say so in the bar's `title`
("task 3: turn finished" ≠ "task 3 done"). Reality surfaces where it belongs:
re-opening the dialog re-lists the folder, and a task that was moved away is no
longer counted. A later iteration could read `git log` through the console
connector; v1 does not.

### 2.2 Launch preconditions and Stop wiring

`launch()` checks, in order, and refuses with a specific message (the dialog
keeps these reachable from its own error line):

1. `phase === 'idle'` and no batch already active;
2. **the thread is empty** — `state.messages.length === 0` (B14). A task lot on
   top of a human thread is meaningless, and with B6 the transcript is never
   auto-purged;
3. **the tools the instruction needs are on.** The batch posts the Chat tab's
   current `mcpEnabled` allowlist (see below), so if `fedit-mcp` (reading and
   editing files) or `console-mcp` (`git add`, `git commit`, `mv … todos/done/`)
   is off, every task would fail the same way. Refuse with a message naming the
   connectors and pointing at the WorkBench panels, rather than burning 8 turns
   to prove the point. Checked against `useChat.state.mcp.enabled` — the
   conversation-scoped filters, not the Settings-activated set.
4. `todos/done/` exists. The instruction's last sentence depends on it; the
   listing already knows (it scanned the folder: a `done` **directory** entry is
   visible to the same `scandir`, so it can be reported as `doneFolder: true`).
   A missing `done/` is a warning in the dialog (C1, closed): batch mode creates
   no folder of its own and keeps exactly one route. The user must not discover
   the missing folder from a failed `mv` on task 1.

Because Launch requires an empty thread, the MCP/LSP toggles are unlocked at
that moment (`useChat.js:159-181` seeds them all to *on*), and they freeze with
the first accepted send (`:973-974`). Consequence worth stating in the dialog
hint: **choose the connectors before Launch**; they cannot change mid-lot, which
is also what makes all 8 tasks run under one identical tool set.

**Stop wiring.** Do **not** fork a second stop button. `WorkBenchInput.vue:411-423` already
renders Stop while `streaming`; batch mode must route through it:

```js
// useBatch.js
function stopBatch() {
  state.active = false            // the loop's inter-task guard
  state.status = 'stopped'
  stop()                          // useChat.stop(): abort the in-flight turn
}
```

and `WorkBenchInput.vue`'s Stop `@click` becomes
`@click="batch.active ? stopBatch() : stop()"`. `restart()` (`:77`) likewise
clears the batch (`stopBatch()` then `reset()`), so the Restart icon never
strands a loop on a wiped conversation — this is the same class of bug the pause
lot had to fix for `pausePending` (`todos/done/inference-pause.md:430`).

### 2.3 The big picture, on every request

`runOneTurn` passes `state.listing.lot.text` (when `includeLot`) as
`batchContext` in the request body — server side, §1.4 turns it into a
system-prompt block. Nothing else: no context turn, no extra bubble, no extra
assistant reply, and the same text is attached to **every** task, which is what
makes it survive the per-task context reset of §2.4.

`includeLot` is a dialog checkbox, default on. Unchecking it is the cheap way to
run a long lot under a small context window (the big picture is the single
biggest prompt cost per task); the dialog states the cost in bytes next to the
checkbox (`lot.size` from the listing) so the choice is informed.

### 2.4 Context reset per task (the heart of the feature)

`useChat.send()` builds the payload as `state.messages.map(messagePayload)`
(`:893`) — the whole accumulated thread. Batch mode must **not** send that, or
task 8 would arrive with seven codebases of tool results in front of it, which
is exactly the situation the lot exists to avoid.

So `useChat` gains one exported variant, `sendBatchTurn(text)`, sharing the body
of `send()` (draft handling, the two bubbles, `phase`, the controller, `finish()`)
but building the request from the current turn alone:

```js
// messages: just this turn — the server prepends its own system message
const history = [{ role: 'user', content: text }]
```

while the **displayed** thread keeps accumulating (the user still watches the
whole lot, and the session-save button can still export it). Two invariants to
enforce in review: `state.messages` is never used to build a batch payload, and
`core/chat.py:188` stays the only place a system message is created — the client
never sends one (`ChatMessage.role` is `Literal["user","assistant"]`,
`routes.py:237`, and that must stay true).

Refactor note: factor the ~60 lines of `send()` into a private helper both
entry points call, rather than copying it — the pause/truncation/measure logic
in `finish()` has already been the source of bugs once
(`todos/done/inference-pause.md:430`) and must not be forked.

---

## Phase 3 — `BatchModeDialog.vue`

A `<dialog>` cloned from `RegistrationMemoryDlg.vue` (the smallest of the two:
`watch` on `open` + `showModal()` + backdrop click + `@close` sync), mounted
once from `App.vue` next to `<SettingsDialog />` so it survives tab switches.

| File | Role |
|---|---|
| `web/src/components/BatchModeDialog.vue` | the form: prefix, dir, big-picture tick, the count label, Cancel / Launch |
| `web/src/composables/useBatch.js` | already owns `open/prefix/listing/error` (Phase 2) — the dialog is a thin view |
| `web/src/components/AppBar.vue` | the new icon button, **immediately left** of the Settings button (`:69-82`) |

### 3.1 Fields and the blur check

```html
<label>Task file prefix
  <input v-model="prefix" @blur="validate" @keyup.enter="validate" />
</label>
<p class="caption" role="status" aria-live="polite">{{ countLabel }}</p>
<p v-if="listing?.skipped?.length" class="caption">{{ skippedLabel }}</p>
<label class="checkbox"><input type="checkbox" v-model="includeLot" />
  Include the big picture ({{ lotSize }})</label>
<p v-if="listing?.lotNote" class="caption">{{ listing.lotNote }}</p>
<p v-if="doneFolderMissing" class="caption">{{ doneFolderHint }}</p>
<p v-if="error" class="alert-error" role="alert">{{ error }}</p>
```

* `validate()` fires on `blur` (the spec's "lorsque le champ perd le focus"),
  also on `Enter`, and fetches
  `/api/batch/tasks?prefix=<enc(prefix)>&dir=<enc(dir)>`.
* Client-side pre-check on the same regex as the server (mirror rule, as
  `useChat.js:57` mirrors the backend image limits): an illegal prefix shows the
  message without a round-trip.
* `countLabel` is exactly the spec's wording: `Found tasks 8`. With `listing`
  null it reads `—`; with `taskCount === 0`, `Found tasks 0` and Launch is
  disabled.
* `skippedLabel` explains the rest, e.g. `2 files skipped (no task number)`, and
  `lot` presence is echoed (`big picture: ed-00-big-picture.md` / `no 00 file`),
  so the number can be sanity-checked by eye before Launch.
* The `dir` field defaults to `todos`, is disabled while `listing` loads, and
  the dialog shows the absolute path it resolved to (`listing.workspace` +
  `/` + `listing.dir`) — the same honesty the app bar applies to the working
  location (`AppBar.vue:15-18`).
* `includeLot` (§2.3) and the `done/` hint (§2.2 item 4) are the only extra
  controls beyond the spec's list: both state a precondition the instruction in
  §1.3 depends on, which is cheaper than letting 8 tasks fail on it.

### 3.2 Buttons

* **Cancel** — clears `prefix`, `dir`, `listing`, `error`, then closes
  (`emit('close')` → `state.open = false`). Esc and a backdrop click follow the
  same path as `RegistrationMemoryDlg` (`onNativeClose`), and Esc must **not**
  launch anything.
* **Launch** — disabled while `loading`, while `taskCount === 0`, while a prefix
  is syntactically invalid, and while `state.active` (one batch at a time). On
  click, in this exact order: re-validate (`listing` could be stale after the
  blur), **switch to the WorkBench tab first** (`useBenchmark.setTab('chat')`,
  `useBenchmark.js:408` — refused while a benchmark runs, which is the right
  behaviour; it must precede `state.active = true`, otherwise Phase 4's own
  "setTab refuses while a batch runs" rule blocks the launch's own tab switch),
  close the dialog, then `launch()`, whose precondition failures (§2.2) surface
  on the WorkBench error line — the dialog is closed by then, so the message has
  to be visible where the user already is.
* The dialog's button row reuses `.settings-actions` (the class
  `RegistrationMemoryDlg.vue:99` uses), so no new CSS is needed there.

### 3.3 App-bar button

A third `icon-btn` in `.appbar-actions` (`AppBar.vue:65`), placed **before** the
Settings button, same `:disabled="running"` gate plus `:disabled="batch.active"`
(B11), with an inline SVG (the file's convention: no icon font) — e.g. a stack
of three bars — and `aria-label`/`title` "Batch mode".

---

## Phase 4 — WorkBench lock-down and the progress bar

### 4.1 The one flag the UI reads

`useBatch.state.active` *is* the spec's `batch_in_progress`. Do not add a second
flag in `useChat`: components that need it import `useBatch()` — and they already
import `useChat()` and `useBenchmark()`, so the dependency direction stays acyclic
(`useBatch` → `useChat`, `useChat` → nothing new; `useBenchmark` already imports
`useChat`, so nothing may import `useBenchmark` from `useChat`).

Surfaces to disable — every one of them is already keyed on `streaming` or
`state.phase`, so each is a one-clause change to an existing expression:

| Surface | Where | Change |
|---|---|---|
| textarea (composer) | `WorkBenchInput.vue:426` | add `:disabled="batch.active"` + a batch-aware `placeholder`/`aria-label`. (Today the textarea is *never* disabled — `todos/done/inference-pause.md:417` — so this is genuinely new.) |
| Send / attach / copy / save / load / restart / edit-and-resend | `WorkBenchInput.vue:270-410`, `WorkBenchView.vue:84` | `canSend` (`:58`), `canCopy`, `canSave`, `pickSessionFile`, `pickImage`, `editLast` already refuse while streaming; each also refuses on `batch.active`. **Restart is the exception: it stays enabled and acts as Stop + wipe** (B15), the only way back to a launchable state after a stopped batch. |
| MCP switches + hub restart | `WorkBenchMcpPanel.vue:20,37` | `locked.value = state.mcp.locked || state.phase !== 'idle' || batch.active`; and `setChatMcp`/`restartHub` already refuse on `phase !== 'idle'` (`useChat.js:184,218,246`) — add the batch flag there too, so the rule lives in the composable and not only in the view. (They are frozen by the first send anyway, `:973-974`; the explicit flag covers the between-tasks window.) |
| LSP switches + hub restart | `WorkBenchLspPanel.vue:23,40` | same pair |
| Memory bullets (click appends to the draft) | `MemoryBullets.vue:48` | early-return when `batch.active` (and a `:disabled`/dim state + tooltip "unavailable while a batch runs"). Ctrl+click on the registration dialog is refused the same way. |
| Memory registration dialog | `WorkBenchInput.vue:38` (`openBulletDialog`) | guard, and it cannot be reached anyway once the bullets are inert |
| App-bar Settings + tabs | `AppBar.vue:72`, `TabNav.vue:13,25`, `useBenchmark.js:408 setTab` | disable while `batch.active`; `setTab` also *refuses* (like it does for `status === 'running'`) so a reload-free keyboard path cannot escape WorkBench mid-batch |
| Batch button | `AppBar.vue` (new) | disabled while `batch.active` |
| Pause (⏸) | `WorkBenchInput.vue:380` | **hidden** while `batch.active` (`v-if="streaming && !batch.active"`): pausing one turn of a lot and leaving the loop to continue onto the next task would contradict "Stop is the only control", and a user comment is meaningless between automated tasks |
| Splitter, scroll, lightbox, theme toggle | `useSplitter.js`, `useAutoScroll.js`, `ThemeToggle.vue` | untouched — view-only, no interaction with the run |

Still live and readable (the spec's "contrôles sans interaction"): the chat
stream, the reasoning/tool timeline and its iteration gauge, the tok/s and ctx
badges, the stopwatch, the error line. All are pure `useChat` render state, so
they need **no** change — which is the payoff of B1.

### 4.2 The batch indicator (`BatchProgressBar.vue`)

Rendered in `WorkBenchSidePanel.vue` immediately after `<ChatStopwatch />`
(`:132`), i.e. exactly "below the elapsed-time control", and only when
`batch.active` (or `status` is `stopped`/`failed`, so the outcome of the batch
stays visible after it ends — see 4.3):

```
ed- · todos                 ← caption: prefix · dir
▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░   ← the meter
3/8 task                    ← mono caption
```

* Structure: the `<progress>` + caption pair already styled by `.progress-row`
  (`theme.css:787-818`, used by the benchmark tab) is the closest match to the
  spec's three-line sketch and needs no new CSS for the bar itself; wrap it in a
  `.batch-progress` block whose caption line reads
  `${prefix} · ${dir}` above and `${index}/${total} task` below. Reusing
  `GaugeBar.vue` is the alternative, but its green→red gradient encodes a
  *level*, which a task counter has no opinion about; the neutral `<progress>`
  is the honest choice.
* Numbers: `value = state.index` (highest completed task number, so a gap in the
  numbering still fills the bar honestly), `max = state.total`. `aria-label`
  carries `ed- batch — task 3 of 8: ed-03-tree-selection-load.md`, so the
  progress is readable without the bar.
* Width: the side panel column is pinned by `.chat-stopwatch`'s `min-width:
  12rem` (`theme.css:1248`), so the bar inherits a sane width for free.
* While `phase === 'idle'` *inside* an active batch (the fraction of a second
  between tasks), the bar keeps its last value — never flickers to 0.

### 4.3 The app-bar pill

`AppBar.vue:25` gains one branch ahead of the chat branch:

```js
const pillLabel = computed(() => {
  if (!running.value && batch.active) return 'Batch'
  …existing chat branch…
})
const pillClass = computed(() => {
  if (!running.value && batch.active) return 'status-batch'
  …
})
```

plus `.status-batch` beside `.status-thinking` / `.status-reasoning`
(`theme.css:255-264`) — violet dot pulsing via the existing `bi-pulse`. Keep the
existing precedence rule (a running benchmark wins) and state it in the comment:
`state.status === 'running'` and `batch.active` are mutually exclusive anyway
(`POST /api/chat` is refused with 409 while a run is in progress,
`routes.py:1180`), so the pill can never disagree with the server.

Status colouring after the end: `stopped` → reuse the `status-cancelled`
warning tone, `failed` → `status-failed`, `finished` → `status-finished`, each
already defined (`theme.css:238-252`), for ~2 s after the batch ends before the
pill falls back to `idle`.

---

## Phase 5 — sequencing and dependencies

| Phase | Depends on | Can ship alone |
|---|---|---|
| 1 — `core/batch.py` + `GET /api/batch/tasks`, `ChatIn.batchContext`, the `POST /run` guard, tests | nothing. The gate is batch mode's own, and `mcp_connectors/path_policy.py` is imported the same plain project-root way the API already imports `mcphub` (`AGENTS.md`: the connectors live at the root) | yes — all of Phase 1 is `TestClient`-testable |
| 2 — `useBatch.js` loop + `useChat.sendBatchTurn` | 1 | no (needs the listing), but a `curl` of it in the console is enough to drive the loop by hand |
| 3 — dialog + app-bar button | 1, 2 | no |
| 4 — lock-down + bar + pill | 2 (`batch.active` must exist) | partially (the bar alone is harmless) |

**No dependency on the Editor lot (`todos/ed-01…ed-10`) at any phase** — B13 is
what makes this lot independently deliverable, and it is the first thing to
re-check if any implementation step starts reaching for `GET /api/fs/*`.

Suggested order: **1 → 2 → 4 (bar + pill only) → 3 → 4 (lock-down)**. The
lock-down last, because it is the phase that touches the most existing
components and the one a manual checklist has to prove. `useChat.sendBatchTurn`
(§2.4) belongs to Phase 2 and is the one change to shared chat code: land it
with a test that an *interactive* send still posts the full history.

## Edge cases and races (each needs a decision or a test)

| # | Case | Handling |
|---|---|---|
| E1 | Stop pressed mid-turn | `stopBatch()` → `useChat.stop()` → abort → `finish()` → `phase = 'idle'` → `runOneTurn` resolves `'stopped'` → the loop's `if (!state.active) return` guard ends it. The partial turn stays in the thread (unchanged Stop semantics, `useChat.js:986-994`). |
| E2 | Stop pressed *between* turns (the `phase === 'idle'` window) | Same guard at the top of the next iteration. No turn is in flight, so nothing to abort. |
| E3 | A turn ends `error` (LLM unreachable, 409, malformed payload) | `state.error` is set by `applyEvent`/`send()`; the loop aborts with `status: 'failed'`, the bar keeps its value, the pill turns the `failed` tone. Do **not** continue: 8 identical failures teach nothing and each costs a round-trip. |
| E4 | A turn ends `truncated` (server cut the stream) | Counted as done, next task proceeds (`truncated` already renders its own caption, `WorkBenchView.vue:122`). Flagged in `outcomes` so the bar's title can say "task 5 was cut short". |
| E5 | A task file vanishes or is renamed mid-lot | The app cannot see it (B13): the model reports "file not found" in its answer and the turn ends as a normal `done`. The bar stays honest by naming the outcome as a *turn* outcome, not a verdict (see §2.1). Re-listing the dialog is how a human notices. |
| E6 | Context pressure **within** one task | B6 makes this the only window pressure that exists: each task starts from sysprompt + `AGENTS.md` + the big picture + its own tool results. The existing ctx badge and `contextWindow` warning (`WorkBenchSidePanel.vue:64-111`) now read per task, which is more useful than a cumulative figure would have been. No auto-truncation (D5): one runaway task is stopped by hand, the other seven are unaffected because they never shared its context. |
| E7 | A benchmark run is started while a batch runs | Impossible from this UI (Settings disabled, Start on the other tab which is locked) **and** now refused server-side by the `POST /run` guard added in §1.5. `POST /api/chat` already 409s while `state.status === 'running'` (`routes.py:1180`), so both directions are closed. |
| E7b | The working location changes mid-batch | Already covered today: `POST /path-location` refuses while any chat reply streams (`routes.py:1079`), and the Settings dialog is locked by B11. No new code. |
| E8 | Browser reload / tab close mid-batch | The SSE generator's `finally` sets `cancel_event` (`routes.py:1250`), so the in-flight turn dies server-side; the loop is client-only, so the batch is simply gone. Nothing to reconcile on reload: `useBatch.state` starts `idle` and `GET /api/config`'s `runStatus` is unaffected (B12). This is the main *cost* of B1, stated plainly: a batch does not survive a page reload, and there is no server-side "a batch is running" to reconnect to. |
| E9 | Launch with a non-empty conversation | Refused up front (B14 / §2.2 item 2). With B6 the check is not cosmetic: the transcript is never purged, so nothing else would ever clear it. |
| E10 | Launch while a chat reply is already streaming | `ensureFreshChat()` refuses (`phase !== 'idle'`). The dialog's button is disabled anyway by then. |
| E11 | `setTab` vs Launch | `setTab('chat')` must run **before** `state.active = true`, otherwise the lock-down rule added in Phase 4 ("setTab refuses while `batch.active`") blocks the launch's own tab switch. Order it in `launch()`, and let the refusal target *leaving* the chat tab (or exempt an already-targeted `chat`). |
| E12 | The MCP/LSP hubs are restarted (or their declaration files edited) mid-batch | The toggles are frozen, but the hub restart buttons are disabled (Phase 4) and `POST /api/mcp/restart` is already refused while `_CHAT_TASKS` is non-empty (`routes.py:854`, `:995`). A hand-edited `externals.json` therefore takes effect on the *next* batch, not mid-lot. Say so in the dialog hint. |
| E13 | Prefix matching the lot file only (`ed-00` and nothing else) | `taskCount: 0` → Launch disabled, caption `Found tasks 0 — only the big-picture file matches`. |
| E14 | A turn that never fires | The pre-B13 version of this bug was an empty task body tripping `send()`'s empty-draft early return (`useChat.js:874`) and hanging the loop on a `phase` change that never happens. With §1.3 the instruction is a fixed non-empty template, so the empty-draft path is unreachable — but the class of bug is not: `send()` also returns early on `phase !== 'idle'`. **Make `sendBatchTurn` return a boolean** ("the turn was started"); `runOneTurn` resolves `'not-started'` immediately instead of awaiting. The loop treats it as `error` (the state machine is confused; stop the batch). |
| E15 | Very large task file | No app-side cap applies (B13): the model reads it in `read_file` windows. The only cost is context inside that one task, visible on the ctx badge. |
| E16 | A task outgrows the tool-iteration cap | `max_tool_iteration()` (`config.py:583`, the `MAX_TOOL_ITERATION` `.env` knob) caps *one* reply's whole tool loop. Its code default is 25, but this deployment's `.env` sets **250**, and **sizing it is the user's responsibility, not a batch concern**: a task that exhausts the cap means the value should be raised in Settings, and that has to happen **before** Launch, since Settings is locked during a batch (B11). Batch mode must **not** read `.env`, judge whether a value is "too low", warn about it, or introduce a batch-specific cap. What it does is *report*: `core/chat.py:527-533` ends such a reply with `(maximum of N tool iterations reached — stopping)`, which is a **`done`**, not an error — so without reporting, the loop would simply move on with task 5 half-done. `runOneTurn` flags the outcome `'capped'` when the assistant placeholder carries `mcpIteration >= mcpMax`, and the bar's title says "task 5: hit the tool-iteration cap". Both numbers are read from the **server-reported** frames (`useChat.js:726-727`, set on the placeholder at `:913-914`); the client never re-derives the limit. |
| E17 | A task leaves the tree dirty (crash, Stop, cap) | The next task starts from that state — that is the design (disk is the only carrier, B6). Not repaired by the app: no automatic `git stash`/revert. The instruction's commit step is what makes each task a checkpoint; a human reviewing `git status` after a stopped batch sees exactly where it stopped. |
| E18 | The browser transcript keeps growing across tasks (display only) | No LLM cost (§2.4), but 8 coding tasks of tool badges is a long DOM. Existing auto-scroll and per-message collapse already handle it; session-save still exports the whole thread. Only check: do not "optimise" by clearing `state.messages` between tasks — the bar's history, the stopwatch and the save button all read it. |

## Questions the architect answered (all closed)

| # | Question | Answer |
|---|---|---|
| D1 | Frontend loop or backend runner with its own bus? | **Frontend** (B1). Accepted cost: no reload survival (E8). |
| D2 | Recursive lot folders or flat? | **Flat only** — one `scandir`, `todos/done/` is never listed. |
| D3 | Read task bodies through the Editor's `GET /api/fs/file`, or a temporary `GET /api/batch/file`? | **Neither** — the feature is not implemented, so batch mode takes **no** dependency: the model reads its own task file (§1.2, B13). |
| D4 | How does the `00` big picture reach the model? | **System prompt** (`ChatIn.batchContext`, §1.4), never a visible client-side turn. |
| D5 | Auto-truncate or restart the context when the window fills? | **No** — the per-task reset (B6) is the mechanism; within a task, report and let the user Stop. |
| D6 | Stop = stop the lot, or skip to the next task? | **Stop the whole lot** (B9). |
| D7 | Restart during a batch? | **Enabled, as Stop + wipe** (B15) — the agreed deviation from "only Stop". |
| D8 | Write `results.json` / a batch report? | **No** (B12); session-save of the thread is enough. |
| D9 | Per-task durations/tokens on the bar? | Out of scope for v1 — the data is already on each assistant message. |
| D10 | Add the missing `_CHAT_TASKS` guard to `POST /run`? | **Yes** (§1.5) — server-enforced, not just UI-enforced. |
| D11 | Launch on a non-empty thread: refuse or purge? | **Refuse** (B14): WorkBench must be at its initial state. |

## Two small choices, now closed

| # | Choice | Decision |
|---|---|---|
| C1 | Missing `todos/done/`: second route (`POST /api/batch/ensure-done`) or a dialog warning telling the user to create it? | **The warning.** Batch mode keeps exactly one route, so its path gate stays in one place; the listing already reports `doneFolder`, and the dialog shows the hint (§2.2 item 4, §3.1). The user must not learn about it from a failed `mv` on task 1. |
| C2 | Detect the tool-iteration cap client-side (`mcpIteration >= mcpMax`, E16) or only document the knob? | **Detect and report it** — the two numbers are already on the message. This is reporting only: sizing the knob stays the user's job, and batch mode adds no cap, no default and no advice about the value (E16). |

## Explicitly out of scope (agreed, not to be designed here)

**A batch log service.** Batch mode writes no file: no report, no journal, no
per-task log — `results.json` belongs to benchmark runs and is untouched (B12),
and the thread can still be exported with the existing session-save. A dedicated
server-side log for batches is a separate, later lot; nothing in this
specification should pre-shape it, and nothing here may open a log file. The one
consequence worth noting for that future lot: with B1 the loop is client-side, so
today's only durable record of a batch is the transcript the user chooses to save
— which is also why the bar's `outcomes` stay honest per-turn outcomes (§2.1) and
why a reload discards everything (E8). If a log service ever exists, that is the
place where "what did task 5 actually do" gets answered; it is not batch mode's
job in v1.

## Size estimate

| Phase | Files | Roughly |
|---|---|---|
| 1 — listing + gate + `batchContext` + `POST /run` guard + tests | `src/core/batch.py`, `src/api/routes.py`, `src/core/chat.py`, `tests/test_batch.py` | ~150 lines Python + ~140 lines tests, 13 tests |
| 2 — loop composable + `sendBatchTurn` | `web/src/composables/useBatch.js`, `web/src/composables/useChat.js` | ~120 lines new JS + a `send()` refactor |
| 3 — dialog + app-bar button | `BatchModeDialog.vue`, `AppBar.vue`, `App.vue` | ~140 lines |
| 4 — lock-down + bar + pill + css | `useBatch.js`, `useChat.js` (guards), `WorkBenchInput.vue`, `WorkBenchMcpPanel.vue`, `WorkBenchLspPanel.vue`, `MemoryBullets.vue`, `WorkBenchSidePanel.vue`, `BatchProgressBar.vue`, `AppBar.vue`, `TabNav.vue`, `useBenchmark.js`, `theme.css` | ~150 lines, spread thin |

No frontend test runner exists, so Phases 2-4 are proven by the checklist below
(the convention of `todos/done/inference-pause.md:491` and `todos/ed-01:127`).

## Manual verification checklist

Use a throwaway lot first (`bt-00…bt-03` in a scratch folder, with `bt-02`
asking for a one-line change to a file): a real `ed-` lot rewrites the repo,
which is a poor way to test a UI.

1. `prefix: bt-` → on blur, `Found tasks 3`; the caption names
   `bt-00-…` as the big picture, and `todos/`'s own `opt-orphan-hub-children.md`
   is nowhere in the payload (it does not match the prefix — §1.1).
2. A lot of 10 tasks (`ed-`) still reads `Found tasks 10` with `ed-00`
   excluded, and `todos/done/*.md` is never listed.
3. Launch → WorkBench tab opens, the pill reads `Batch`, the bar appears under
   the stopwatch reading `1/3 task` → `2/3 task` as turns complete; the chat
   and reasoning/tool streams scroll throughout.
4. **Context reset**: with the llama.cpp log (or `LITELLM_DEBUG=1`) open, task 2
   must arrive with one user message, not task 1's transcript — while the
   browser thread still shows both. This is the single most important line of
   this checklist: it is what the feature exists to do.
5. The system prompt of *every* task carries the `bt-00` text after `AGENTS.md`
   (same log); unticking "include the big picture" makes it disappear while the
   thread looks otherwise identical.
6. The instruction as sent matches §1.3 verbatim, `<task_location>` being the
   root-relative path from the listing; the model then reads that file itself
   (visible as a `read_file` tool badge).
7. End-to-end on `bt-02`: the model edits, stages, commits, and the task file is
   in `todos/done/`. Re-opening the dialog drops the count by one.
8. During a turn: textarea, memory bullets, MCP/LSP switches, copy/save/load/
   attach/edit-and-resend, Settings, both tabs and the batch button are all
   inert — try each with the keyboard, not only the mouse. Stop and Restart are
   the only live controls.
9. Stop mid-tool-call → the badge leaves `running` (existing `finish()`), the bar
   freezes at `2/3 task`, the pill leaves `Batch`, no task 3 turn starts. Then
   Restart → clean empty thread, and Launch works again (B15).
10. Launch on a thread that already has messages → refused, nothing runs (B14).
11. Kill the LLM endpoint mid-lot → turn errors, batch `failed`, no further
    turns (E3).
12. Deliberately lower `MAX_TOOL_ITERATION` in Settings **before** Launch (e.g.
    2), run one task → the reply carries the "maximum of 2 tool iterations
    reached" note, the batch continues, and the bar's title flags the capped
    task (E16 / C2). Nothing else about the cap: no warning, no suggested value.

13. `?prefix=..%2F`, `?dir=..%2F.ssh` → 422 naming the rule; `?dir=nope` → 404.
14. Reload the page mid-batch → the app comes back idle, no orphan generation on
    the server (check the llama.cpp log).
15. Start a benchmark from a second tab (or `curl POST /api/run`) while a batch
    turn streams → 409 (§1.5).
16. `AGENTS.md` note: the one line to add is that the `todos/` lot is *listed*
    by the app through the shared deny-list, and *read/written* only through the
    connectors — batch mode itself never touches file contents.

## Files touched (summary)

```
src/core/batch.py                       NEW  listing + gate + numbering + lot text
src/api/routes.py                       +GET /api/batch/tasks, ChatIn.batchContext,
                                        +_CHAT_TASKS guard on POST /run
src/core/chat.py                        +batch_instructions system block
tests/test_batch.py                     NEW  the 13 cases of §1.6
web/src/composables/useBatch.js         NEW  dialog state + the loop + template
web/src/composables/useChat.js          send() split, +sendBatchTurn, batch guards
web/src/components/BatchModeDialog.vue  NEW  prefix / dir / count / Cancel / Launch
web/src/components/BatchProgressBar.vue NEW  bar under the stopwatch
web/src/components/AppBar.vue           +batch button, +Batch pill branch
web/src/components/TabNav.vue           disabled while a batch runs
web/src/components/WorkBenchSidePanel.vue  +<BatchProgressBar />
web/src/components/WorkBenchInput.vue   Stop routing, textarea + icons lock-down
web/src/components/WorkBench{Mcp,Lsp}Panel.vue  switches locked while a batch runs
web/src/components/MemoryBullets.vue    inert while a batch runs
web/src/composables/useBenchmark.js     setTab guard
web/src/styles/theme.css                .status-batch, .batch-progress
```

`core/llm.py`, `core/job.py`, `RunState`, `results.json`, the verifier and the
two hubs are untouched, and `core/chat.py` gains only the one system block of
§1.4. That is the whole point of B1/B13: this lot is UI work, one read-only
endpoint and one protocol field — not a new execution engine.
