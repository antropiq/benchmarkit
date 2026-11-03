# lspfinal-02 — put a language server's process *tree* in a job, and share one teardown

`src/core/proctree_jobs.py` is finished and the terminal already uses it. The LSP
hub does not: `_close_session` still branches on `os.name == "posix"` and, on
Windows, terminates **the wrapper only** — leaving `tsserver` and Volar's workers
alive, holding a project index no later hub will know about. This task makes the
hub's children be born inside a job and deletes the branch.

Depends on: nothing in this lot (independent of 01). **Blocks: lspfinal-04** — the
platform half of the reach seam may only be dropped once a language server's tree
is genuinely bounded here.
Spec: `windows-support-target-B-connectors.md` **B-D5** (decision + "Landed
(csl-02 …) … the hubs' job assignment … is **not** wired yet, so the LSP hub's
Windows refusal stands unchanged and that lift is its own lot" — that sentence is
this task) and **B-D6**'s second bullet. §7 "What B must not break" is binding.

## The two facts that shape this task

**(a) A job is a spawn-time boundary.** `proctree_jobs` records it, measured:
"assign the parent, then let it spawn a grandchild, and the grandchild is a member
nobody assigned … assigning a parent after its child exists covers the parent only.
A caller that opens its job at the spawn has no such gap; one that assigns late
cannot claim tree coverage."

**(b) The LSP hub does not spawn the child — pygls does.** `_start_session` calls
`client.start_io(...)`, and pygls' `start_io` calls `asyncio.create_subprocess_exec`
itself (`pygls/client.py:85-96`). There is **no spawn hook** on the client. So (a)
and (b) collide, and the collision is the whole task. Read the next section before
writing any code.

## The part to decide, not guess: how late is the assignment?

`assign_pid_to_job` needs the pid, and the pid exists only after `start_io`
returns — one event-loop turn after the child was created. That window is where
`tsserver` could already have been spawned **by the server**, outside the job.

- **Measure it on this machine before you design.** Start a real
  `typescript-language-server` (or `vue-language-server`), assign its pid to a job
  immediately after `start_io` returns, then read `ProcessJob.members()`. Does
  `tsserver` appear? Write the answer in the commit message and let it choose the
  implementation. Do not assume either outcome.
- If late assignment is **not** sufficient, do not fake it with a process walk.
  The honest options are (i) pass the job into the spawn — which means the hub
  creating the subprocess and handing pygls its transport, i.e. a `start_io` twin
  that keeps the hub's existing `client._server` contract — or (ii) a documented,
  tested statement of what the boundary does and does not cover. Choose (i) only
  after measuring; it is more code and it must keep the stdio pipes that the
  protocol writes to (the current comment above `start_io` warns that creating a
  subprocess the protocol has no transport for breaks everything).
- `start_new_session=True` is **accepted and ignored** on Windows (verified —
  `create_subprocess_exec` succeeds and runs). Keep passing it: it is what makes
  the POSIX group teardown work, and a platform-specific argument would be a
  second shape for one call site. Say so in the comment so nobody "removes the
  dead flag".
- No breakaway flag: a hub child must never detach from its job. If the measured
  answer is that a Node server *does* detach, that is a finding to report, not a
  flag to add.

## Files

| file | change |
|---|---|
| `src/core/lsphub/manager.py` | one `ProcessJob` per language-server child, opened **before** `start_io` and assigned at the spawn site; `_close_session` calls the shared teardown with no `os.name` branch; the job handle is closed on every path that replaces or drops a child |
| `src/core/lsphub/session.py` | the job lives on `ServerSession` (the state object owns it — check the module docstring's claim that these are "pure state machines with no … subprocess": update the docstring if the job lands here) |
| `src/core/lsphub/constants.py` | the comment at `:85-95` that explains why the refusal stands on the *tree* half becomes false when the wiring lands — rewrite it in the same task, do not leave it lying |
| `tests/test_lsp.py` | teardown tests (below) |
| `tests/fake_langserver.py` | it must spawn a grandchild, so the tree claim is actually tested |

## What must not change

§7 is explicit, and it is the reason this is a separate task:

- the POSIX teardown **verbatim**: SIGTERM before force, the bounded grace
  period, and the leader reaped *before* the tree is probed (why `proctree`
  forbids `getpgid()`);
- the escalation is reached **through `proctree`'s two entry points**, not by
  calling `proctree_jobs` primitives directly from the hub — the terminal already
  calls `ProcessJob` methods directly because its driver seam needs that, and a
  hub copying that shape is how the two drift. `proctree.kill_process_tree(pid, …)`
  already dispatches to `kill_job_tree` when `JOBS_SUPPORTED`; that is the seam;
- the graceful step stays the hub's own LSP `shutdown`/`exit` **before** any
  teardown is asked — `kill_job_tree` documents that inside the teardown the first
  step terminates the leader alone, and the grace window is the tree's chance to
  leave with it;
- **the handle must not leak.** A ctypes handle is a plain integer; nothing closes
  it for you. A hub restart replaces a child, so it must close the *old* job
  rather than drop the reference — kill-on-close is the last guarantee and it only
  fires when the last handle goes.

## Tests

- **Tree, not wrapper.** With the fake server spawning a grandchild, assert after
  `_close_session` that **no member of the job survives** — not "the process
  object has a return code". On POSIX this must keep passing exactly as today.
- `windows_behaviour`-marked (the marker exists in `conftest.py`; the pattern to
  copy is `test_proctree.py:684`). It must **fail** rather than skip on Windows if
  the wiring is missing — a test that passes because a capability was never asked
  for is the shape §6 warns about.
- **The branch is gone.** A test that reads the hub's teardown path and asserts it
  asks the shared entry point on both platforms (no `os.name` in `_close_session`).
- **A failed assignment is visible.** `ProcessJob.assign` answers `None` where jobs
  do not exist so a spawn site can be written once; but on Windows a `False`
  means the tree is unbounded. Assert that case reaches the server's status or the
  log — `kill_job_tree` logs "pid %s was never assigned to a job", and a hub that
  ignores a `False` is the spawn site that "forgot the call" that message exists
  to find.
- **Restart closes the old job.** Assert the handle is closed when a server is
  stopped and re-started.

## Acceptance

- `_close_session` contains **no `os.name == "posix"` branch** and no second copy
  of the escalation.
- On Linux: `tests/test_lsp.py` and `tests/test_proctree.py` green, **the same
  skip count as before** (baselines in the big picture; §6 counts a new skip as a
  defect in this task).
- On Windows: the grandchild test runs and passes, and the fake server's whole tree
  is gone after a stop — assert `ProcessJob.members()` is empty, which is the only
  Windows-side evidence a unit test can produce. The Task-Manager check is
  `lspfinal-06`, not here.
- The comment in `lsphub/constants.py` no longer claims the tree half is missing.
- Do **not** touch `lsp_reach_is_refused()`, `local_reach_is_bounded()` or the
  refusal wording — that is `lspfinal-04`, and it must land *after* this so the
  lift is honest. After this task the three declared servers are **still refused**,
  now solely because of the guardrail/platform predicate, which 01 and 04 handle.
- Move this file to `todos/done/` when the above is true.

## Results (Windows runner: CPython 3.13.13 x64, pygls 2.1.1, Volta-installed
`typescript-language-server` 6.0.3 / typescript 5.9.3, 2026-11-03)

**The measurement, and what it decided.** With a *real* `typescript-language-server`
started through `client.start_io(NODE, cli.mjs, "--stdio")` and the leader pid
assigned to a job at the first statement after `start_io` returned:

```
assign right after start_io -> True
members right after assign: (9352,)
members after didOpen + 1..6s: (9352, 8252, 10372, 4948, 9660, 10200)   # 6 members
node pids NOT in job: [3416, 4672, 1940, 11036]                          # pre-existing, unrelated
members after teardown: None
node pids still alive after teardown: [3416, 4672, 1940, 11036]          # none of the job's
```

The same shape through the hub with the real server (`start_all` → 5 s →
`stop_all`): `leaders=[7592]`, 6 members at 5 s, `job.handle = None` afterwards
and **no surviving member**. The counter-case was re-measured too: a grandchild
already created *before* the assignment is **not** migrated (`members=(9056,)` —
leader only), the fact `test_proctree.py` already pins.

So **late assignment is sufficient** for these children, because a Node server
creates its own children only once its runtime is up — far later than the one
event-loop turn `start_io` costs. Option (i) (a `start_io` twin) was therefore
**not** taken; the ordering is asserted by a test, and the window and what it
does not cover are stated in the comment at the spawn site. No Node server was
seen to break away from its parent's job, so no breakaway flag exists.

**What landed.** `_start_session` opens one `ProcessJob(f"lsp-<id>")` before
`start_io` and `_contain_tree` assigns the child with `leader=True` as the first
statement after it returns — the session is registered before that call, so the
handle has an owner from the moment the child exists. `_close_session` has no
`os.name` branch, no `.terminate()`/`.kill()` fallback and no second escalation:
one `proctree.kill_process_tree(pid, self._reap_leader(process), STOP_GRACE_SECONDS)`
on both platforms, then `_release_job(session)` in a `finally`. The handle is
released on every path that replaces or drops a child: the two spawn-failure
paths, the crash watcher (`_watch_exit`), `_close_session`, and `restart()`'s
`_sessions.clear()`. `ServerSession` gained the `job` attribute and its module
docstring's "no subprocess" claim now names that exception. `start_new_session`
is still passed on both platforms, with the comment saying it is ignored on
Windows rather than dead.

**A failed assignment is visible.** `assign` answering `False` logs a warning and
appends to the server's own log ("pid N is not contained in a job, so its
descendants are not bounded") without failing the spawn; `None` (the POSIX
answer) is silent. Both asserted.

**Suites on this runner.** `tests/test_lsp.py` → **107 passed, 5 skipped**
(97 before this task — 88 plus lspfinal-01's 9 — plus this task's 10; the 5
skips are the same named pre-existing ones, none added).
`tests/test_proctree.py` + `tests/test_path_policy.py` → **65 passed, 4 skipped** —
exactly what they were before this task (this one adds no test there), with the
same 4 named pre-existing skips.
`test_mcp.py`, `test_api.py`, `test_chat.py`, `test_batch.py`, `test_terminal*.py`,
the `test_editorfs*` files and `test_write_surface_isolation.py` are green.

**The wiring is proven load-bearing, not decorative**: commenting out the
`_contain_tree` call makes the Windows-only tree test **fail** (`9720 in ()` on
`job.leaders`, and the teardown logs "pid … was never assigned to a job") rather
than skip — 4 of the new tests fail, none of them vacuously.

**Untouched, on purpose:** `lsp_reach_is_refused()`, `local_reach_is_bounded()`
and `LOCAL_REACH_REFUSAL`. Verified still refusing here: `refused=True`,
`bounded seam=False`, `guardrails=True` — so `lspfinal-04` has exactly one half
left to lift, and it can now do so honestly.
