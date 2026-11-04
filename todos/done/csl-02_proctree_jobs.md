# csl-02 — `proctree_jobs.py`: the Job Object half of the shared teardown

## Deliver this

`src/core/proctree.py` becomes the **one teardown the whole app asks for**, on
either platform, and its Windows implementation is a **Job Object with
kill-on-close** — roadmap §5's decided mechanism, not a `killpg` shim (B-D5).

Requirements, in the order that matters:

* **The callers do not change.** `proctree.kill_process_tree(pgid, reap_leader,
  grace)` and `proctree.kill_session_tree(sid, reap_leader, grace)` keep their
  names, their signatures and their semantics. Callers (`mcphub/manager.py`,
  `lsphub/manager.py`, `termhub/session.py`) keep calling exactly what they call
  today. On Windows they land in the job half; on POSIX nothing changes.
* **The same escalation shape**: graceful first, bounded wait to the grace
  deadline, then force — *"the same escalation shape … and the same injected reap
  step so the two stay honest to the same contract"* (roadmap §5). The reap step
  stays injected and stays bounded by the deadline it is handed.
* The job is created **by the caller's side of the seam**, not inside the kill:
  a process becomes a member when it is spawned, and the kill terminates *the
  job*, which is what makes *"kill process trees, not wrappers"* true for
  grandchildren. So `proctree_jobs.py` exposes the whole lifecycle — create the
  job, assign a process to it, then terminate-and-drain it — and each hub/session
  owns one job per child/session. Which of today's spawn sites gain the
  assignment is a decision **inside this task**, and it must be recorded: the
  minimum this lot needs is the terminal session's tree (csl-07 is the second
  caller); the hub spawn sites may be wired here or left for the later hub lot,
  but the seam must be complete enough that they only need the call.
* A **liveness probe** the tests can use cross-platform: given a pid, is it gone?
  (The tests need this because `os.kill(pid, 0)` is *not* a probe on Windows —
  CPython maps a plain signal number to `TerminateProcess`.)
* `ctypes` only — **no new dependency**. Roadmap §8 already scoped B-D5 as
  *"none (ctypes/Win32 for jobs …)"*.

## Files it touches

| file | change |
|---|---|
| `src/core/proctree_jobs.py` | **new** — the job lifecycle, the terminate-and-drain escalation, the liveness probe |
| `src/core/proctree.py` | edit — the two entry points dispatch to the POSIX half or the job half; docstring states both mechanisms and the shared contract |
| `tests/test_proctree.py` | edit — the existing escalation tests stay green; add the job-half tests below |
| `pyproject.toml` | **no change** (ctypes is stdlib) |

## Rules it must not break (quoted)

* AGENTS.md: *"**Teardown kills process *trees*, not wrappers**, so children a
  hub spawned of its own never outlive the app that started them."*
* AGENTS.md: *"Hubs are long-lived singletons. A restart rebuilds a hub's
  internals but never replaces the object the app holds, because worker threads
  keep the reference across a request."* A job handle is such a reference: the
  module must not turn a hub's child into a different object, and a hub restart
  must not leak a job handle.
* `proctree.py`'s own contract, verbatim: the leader is **reaped before the tree
  is probed**, because *"a tree only disappears once every member is dead **and**
  reaped"*; and the session scope exists because *"an interactive shell runs with
  job control: every job it starts gets a process group of its own … A terminal
  teardown that stopped there would let a backgrounded `python -m http.server`
  outlive the tab"*. The job half must satisfy the same promise for the same
  ghost.
* AGENTS.md: *"Core owns logic, the API owns transport … nothing under `core` may
  import the API layer."*
* AGENTS.md: *"Tests stay hermetic"* — the module's tunables (poll interval,
  grace fallback) are module attributes patched through the module, and a test
  spawns its own throw-away processes, never a real hub or a real server.
* **Do not** touch `mcphub`'s or `lsphub`'s Windows refusal
  (`local_reach_is_bounded()`): lifting the LSP hub's refusal is **out of scope
  for this lot** (big picture). If a hub's tests start passing on Windows after
  this task, report it; do not enable anything.
* Do not delete or weaken the POSIX tests. Windows differences are asserted with
  the `windows_behaviour` marker plus seam-forced tests on Linux (roadmap §9
  rule 2), never by making a POSIX assertion platform-agnostic.

## Depends on

Nothing. Independent of csl-01 and csl-03. csl-07 is this task's first real
consumer.

## How to tell it is done

1. `pytest tests/test_proctree.py` green on **both** runners, with **no POSIX
   test weakened or skipped on Linux**: the existing graceful→force, reap-before-
   probe and grace-deadline assertions keep their exact shape.
2. New tests, each with the reason a skip would name:
   * **Windows**: spawn `cmd /c "start …"` or a child that itself spawns a
     grandchild, put it in a job, terminate the job, and assert **the grandchild
     is gone too** — the wrapper-kill case AGENTS.md names, observed rather than
     assumed.
   * **Windows**: the grace deadline is honoured — a child that traps/ignores the
     graceful step is force-killed within the patched grace, and the whole
     teardown returns inside it.
   * **Linux**: the same entry points still produce the same `/proc`-based
     verdict (the existing session-member assertions).
   * **Both**: the liveness probe agrees with an observation of a process the
     test itself spawned and with the same pid after it exited — no `os.kill(pid, 0)`
     anywhere in the suite for liveness.
3. `pytest tests/test_mcp.py tests/test_write_surface_isolation.py` green, and
   `git diff` shows **no change** to `mcphub/discovery.py` or `lsphub/discovery.py`
   (the refusals are untouched).
4. Append a *Results* note to this file recording, on the runner you have: which
   Win32 calls the job half actually uses, whether assignment succeeded for a
   process the driver did not spawn directly (csl-07 depends on that answer), and
   anything the job refuses to contain (a process already in another job).

## Results (Windows runner: Windows 11 10.0.26300, CPython 3.13.13 x64, 2026-11-02)

Every claim below was measured with a throw-away probe or asserted by a test in
`tests/test_proctree.py`; nothing here is inferred from the API docs.

**The Win32 calls the job half actually uses** (all `ctypes`, none optional):
`CreateJobObjectW`, `SetInformationJobObject`, `QueryInformationJobObject`,
`AssignProcessToJobObject`, `TerminateJobObject`, `OpenProcess`,
`GetExitCodeProcess`, `TerminateProcess`, `CloseHandle`. Two information classes
matter, and both were found by trying rather than by the documented number:

* `JOBOBJECT_EXTENDED_LIMIT_INFORMATION` = **9** is the one that carries
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Classes 1/2/4/6/7 were rejected with
  `ERROR_BAD_LENGTH` (24) and set nothing — including 7, which is what the
  public header calls the basic-limit class. Only 9 produced a job whose last
  handle closing actually killed a live member.
* `JOBOBJECT_BASIC_PROCESS_ID_LIST` = **3**. Class 2 answers 24 (it is the
  basic-*limit* class on this Windows), so a member query written against the
  doc's number silently "finds no members" — the worst possible failure, a
  teardown that reports a tree drained.

The structures are declared from `ctypes` primitives (`c_void_p` handles,
`c_size_t` for the `ULONG_PTR` pid entries) with `restype`/`argtypes` bound once;
without that, ctypes returns handles as 32-bit ints and every handle is wrong on
x64.

**Assignment of a process the driver did not spawn: yes, works** — this is the
answer csl-07 depends on. `AssignProcessToJobObject` after the spawn takes any
pid, including a process this one did not create (asserted by
`test_a_process_the_caller_did_not_spawn_is_assignable_by_pid`, which puts a
foreign child in the job, lists it as a member and kills it through the job). A
helper cannot be assigned *before* it exists, so the driver must hand its
children's pids over right after starting each.

**The boundary that binds every caller: containment is decided at a process's
creation.** Measured by gating a child so its own child is born on command:

* assign the parent, *then* let it spawn → the descendant is a job member nobody
  assigned, and terminating the job kills it;
* let it spawn, *then* assign the parent → the job lists only the parent and the
  earlier descendant **survives** the job's termination.

Pinned by `test_terminating_the_job_takes_the_grandchild_too` and, as the
negative case, `test_a_descendant_born_before_the_assignment_is_not_migrated`.
Consequence for csl-07: the session must open its job and assign the shell
**before the shell can run a command**, and must assign the driver's console
helper as soon as the driver reports it. A late assignment is not "slightly
weaker", it is a different guarantee.

**What the job refuses to contain: nothing.** A process already in a job is
accepted by a second `AssignProcessToJobObject` (no error), and it is then
listed by **both** jobs; terminating either one kills it. So there is no
"already jobbed" answer for the seam to report, and the discipline is the app's:
one child, one job — two jobs over one tree means two owners of its death.
Asserted by `test_a_process_already_in_a_job_is_contained_by_both`. (The app
itself is *already inside a job* on this machine — `IsProcessInJob(self, NULL)`
answers true — which is why nothing may assign the app's own pid.)

**Other measurements that shaped the code:**

* `TerminateJobObject` is asynchronous: it returns in microseconds while members
  are still listed, so the escalation needs a bounded drain of its own. The
  graceful phase therefore stops at `grace × (1 - JOB_FORCE_DRAIN_SHARE)` and
  the force step drains the rest, so the teardown returns inside the caller's
  grace *whether or not* it escalated. Observed wall time for the
  stubborn-tree test with a 0.4 s grace: ~0.4–0.5 s.
* The job handle is waitable (`WaitForSingleObject` signals after a terminate),
  but it did **not** signal for a job whose members exited on their own, and a
  fresh empty job is unsignalled — so waiting on the handle cannot answer "is
  the tree empty". The member query is the only reliable probe, which is what
  `terminate_and_drain` uses.
* On `ERROR_MORE_DATA` (234) the reply fills exactly
  `MaximumNumberOfProcessIds` entries while `NumberOfProcessIds` reports the
  larger total, so the count is clamped by what was written; a crowded tree is
  reported as a non-empty prefix and the probe keeps waiting.
* A job holding no process answers the member query **successfully** with
  `NumberOfProcessIds = 0` — there is no `ERROR_NOT_SUPPORTED` (50) to handle,
  so the empty tree is a real empty reply and `job_member_pids` returns `()`.
  Only a genuine failure returns `None` ("could not be asked"), which the probe
  treats as "still not drained" rather than as an empty tree.
* Liveness: `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` +
  `GetExitCodeProcess` (259 = still running). No test in the suite may use
  `os.kill(pid, 0)`: verified on this machine that it *terminates* the process
  (CPython maps a plain signal number to `TerminateProcess`), and
  `test_no_test_in_the_suite_uses_a_zero_signal_as_a_probe` guards the whole
  `tests/` tree by parsing the call, not matching text.

**Decisions taken inside this task, recorded.**

* Which spawn sites gained the assignment: **none yet.** The minimum the lot
  needs is the terminal's tree (csl-07, the first real consumer), and the hub
  calls are deliberately left alone: `lsphub/manager.py:704` keeps its own
  `os.name == "posix"` guard and its terminate/kill fallback, and
  `mcphub`/`lsphub` are out of scope for this lot. The seam is complete for
  them — open a `ProcessJob` before the spawn, `assign(pid, leader=True)` after
  it, `close()` when a restart replaces the child — so the hub half needs only
  those calls plus lifting its own guard.
* The callers are unchanged: `kill_process_tree` / `kill_session_tree` keep
  their names, signatures and semantics, and dispatch to the job half *before*
  treating the pid as a group id (a `killpg` on that pid would signal the app's
  own group).
* `ProcessJob` has no `__del__`: closing a handle is a kill, so an implicit
  finaliser would let a GC pause choose when a tree dies. `close()` is explicit
  and idempotent, and the registry entry is dropped with it so a recycled pid
  cannot point at a closed job.

**Test evidence on this runner.** `tests/test_proctree.py` did **not** exist on
this checkout (no such file in `git log --all`), so the "existing escalation
tests stay green" premise had nothing to keep: the file is new, and it carries
both halves — the POSIX group/session assertions (marked `posix_behaviour`, with
a real tree whose descendant ignores the graceful signal, and a session member
that leads its own group) and the Windows job assertions
(`windows_behaviour`, real processes), plus the seam-forced dispatch and
escalation tests that run everywhere.

```
pytest tests/test_proctree.py                → 23 passed, 4 skipped
  SKIPPED posix_behaviour ×3, /proc probe ×1   (the only skips; all named)
pytest tests/test_mcp.py tests/test_write_surface_isolation.py  → 95 passed, 1 skipped
pytest tests/test_console_gbash.py tests/test_console_policy.py → 69 passed, 6 skipped
cd web && npm test                            → 472 passed (35 files)
```

The POSIX assertions therefore did **not** run on this Windows runner — they are
the marked ones and must be confirmed green on the Linux runner (they exercise
unchanged code: `proctree.py`'s group/session bodies are byte-identical apart
from the two added dispatch lines).

**Pre-existing failures on this runner, not caused by this task** (reported, not
hidden; each is unreachable from `proctree`):

* `tests/test_lsp.py` — 36 failures, every one carrying the LSP reach-refusal
  sentence (`its reach cannot be bounded on this filesystem yet (see target B)`)
  from `mcphub/process.local_reach_is_bounded()`, which is still
  `os.name == "posix"` and which this task left exactly as it is. Lifting it is
  a separate lot. Note these failures show the job half is *not* what the hub
  asks for yet — as intended.
* `tests/test_terminal.py::TestGuards/TestRegistry` — hang here. Verified the
  cause is unrelated to `proctree`: `api/routes.py:1756` still refuses the
  WebSocket on non-POSIX (`os.name != "posix"`), sends its error frame and then
  Starlette's `TestClient` wedges on that accept-then-close sequence;
  `termhub/session.py:386` keeps its own `os.name == "posix"` guard around the
  teardown call, so the shared entry point is never reached from the terminal on
  Windows before csl-04…csl-07. (csl-08 is the task that re-gates this file.)
* `tests/test_api.py` — 2 failures from `NameError: name 'helpers_block' is not
  defined` at `api/routes.py:1417`, in a file this task never touched (the
  checkout is mid-merge: `AGENTS.md` is unmerged and B1's changes are staged).

**The Linux promise, as stated.** Nothing in the POSIX path changed behaviour:
`proctree.py`'s group and session bodies, signals, probes and docstring rules are
untouched except for the two capability-guarded dispatch lines, and
`proctree_jobs.py` binds no calls and creates no job on a non-Windows runner
(`JOBS_SUPPORTED` False, `_api` None), so `ProcessJob` is inert and a caller
that asks gets `None`.
