# csl-07 — One job for the whole terminal session (shell + console helper + their children)

## Deliver this

The terminal becomes the **second caller of the shared teardown on Windows**
(C-D5), so a session is a process tree there too:

> *"after B-D5, the terminal asks the same shared teardown, and the job it creates
> must contain **both** the shell and the `OpenConsole.exe`/conhost process
> `pywinpty` starts for it. That second process is the one an implementer will
> forget, and it is precisely the 'kill process trees, not wrappers' case AGENTS.md
> names."*

Concretely:

* `termhub/session.py`'s teardown calls **`proctree`'s existing entry point** (as
  amended by csl-02) on **both** platforms — no `os.name` guard left around the
  call, no second kill path, no Windows-only variant of `teardown()`.
* On Windows the session owns one job: created before the shell spawns, containing
  the shell, **the console helper the driver started**, and therefore everything
  bash backgrounds or forks. On POSIX the job half is inert and the session scope
  stays exactly as it is today.
* `_finish_pump()`'s reasoning stays written down and stays true on Windows too:
  the reader is drained **before** the driver's end is closed, and a reader that
  survives the kill is cancelled rather than left reading a closed handle
  (*"leaking one descriptor beats reading it as if it were the next session's"*).
* The teardown stays shielded against the cancellation that triggered it — the
  existing comment is the reason, and C must not weaken it while adding a
  platform.
* `MAX_SESSIONS`, `STOP_GRACE_SECONDS` and `_reap_leader()` keep their roles: the
  app exit pays *"at most `STOP_GRACE_SECONDS` per stubborn shell, not per
  process"* (`termhub/manager.py`'s `stop_all` docstring) on Windows as well.
* Ctrl-C stays an input event carried by the driver (C-D5: *"Ctrl-C is a separate
  mechanism, not a signal … `session.py`'s write stays a plain byte writer"*);
  `^Z`, job stop/resume are **not promised** on Windows.

## Files it touches

| file | change |
|---|---|
| `src/core/termhub/session.py` | edit — the teardown guard goes; the session owns one job and hands it to the shared teardown |
| `src/core/termhub/pty_conpty.py` | edit — exposes every process it started, in spawn order, so the job can contain them |
| `src/core/termhub/pty_posix.py` | untouched (or a naming adapter only) |
| `src/core/proctree.py` / `src/core/proctree_jobs.py` | edit only if the session's call needs a capability csl-02 did not provide — record any such addition |
| `src/core/termhub/manager.py` | edit only if `stop_all` needs the job released on app shutdown |
| `tests/test_terminal.py` + the Windows driver test file | the teardown tests below |

## Rules it must not break (quoted)

* AGENTS.md: *"**Teardown kills process *trees*, not wrappers**, so children a hub
  spawned of its own never outlive the app that started them."*
* `proctree.py`'s session-scope rationale, verbatim: *"An interactive shell runs
  with job control: every job it starts gets a process group of its own, all
  inside the shell's session, so a `killpg` on the shell's group reaches the shell
  and nothing else. A terminal teardown that stopped there would let a
  backgrounded `python -m http.server` outlive the tab — the exact ghost this
  module exists to prevent."* On Windows the job is what reaches them instead.
* C-D5's Ctrl-C rule: *"the seam must own the difference rather than have
  `write()` special-case byte 3."*
* C §7: *"the rule that a terminal session is a process tree, never a wrapper"*,
  and the four-session cap, the close codes and `termhub/policy.py`'s origin rule
  are untouched by this task.
* C-D3: `STOP_GRACE_SECONDS` — *"it is the same number and the same rule as both
  hubs'"*, and C must keep that sentence true.
* **Linux must not change at all**: same kill, same order (kill → drain reader →
  release the terminal end), same reaped exit code, same grace. If the POSIX path
  needs a rewrite to accommodate the job, the rewrite is wrong — add the job to
  the Windows half.
* Do not touch `mcphub`/`lsphub` (out of scope for this lot: the hub-side job
  wiring and the LSP refusal are a separate lot).
* AGENTS.md: *"Tests stay hermetic"* — patch the grace through
  `termhub.constants` (the file's own convention) and spawn throw-away children,
  never a real language server or connector.

## Depends on

csl-02 (`proctree_jobs.py` + the liveness probe) and csl-06 (the driver and its
helper processes). Both *Results* notes are inputs to this task: read them first.

## How to tell it is done

1. **Linux**: `pytest tests/test_terminal.py` — identical results to before this
   task, zero skips, and `test_disconnect_tears_down_the_whole_session_tree`
   unchanged in what it asserts. `git diff` shows no change to the POSIX kill
   order.
2. **Windows** (`windows_behaviour` + the driver installed): a new test starts a
   **backgrounded** child from the shell (bash job control — `sleep 40 &`, or the
   `$!` pid it prints), drops the socket, and asserts **both** the shell pid and
   the child pid are gone via csl-02's liveness probe — the exact ghost
   `proctree.py` names, on the platform where `killpg` cannot reach it.
3. **Windows**: a second test asserts the **console helper** the driver started is
   gone after `teardown()` (and after a shell that exits on its own). If the
   driver's helper cannot be enumerated, that is a failure of this task, not a
   note.
4. **Windows**: `test_kill_frame_closes_session_and_kills_pid` and the cap test
   pass against the job-based teardown, unmodified in what they assert; the
   teardown still returns inside the patched grace.
5. No `os.name` check remains in `termhub/session.py`'s teardown path.
6. `pytest tests/test_proctree.py tests/test_write_surface_isolation.py` green;
   `cd web && npm test` green; `git diff --stat web/` empty.
7. *Results* note: the process list the job actually contained (names and how
   each was obtained), the observed teardown duration, whether a
   `setsid`-equivalent escape exists on Windows (the documented POSIX opt-out),
   and what happens to a child that outlived the grace.

## Results (Windows runner: Windows 11 10.0.26300, CPython 3.13.13 x64, pywinpty 3.1.2 ConPTY, Git at `C:\Program Files\Git`, 2026-11-02)

**What landed.** `session.py` now owns one `proctree_jobs.ProcessJob`
(`terminal-<session id>`), opened in `spawn()` **before** the driver's pair is
opened and the shell starts; `_contain_spawned()` assigns the shell with
`leader=True` and every other pid the driver reports through its existing
`started_pids()` as a plain member; `teardown()` asks
`proctree.kill_session_tree(pid, self._reap_leader(), STOP_GRACE_SECONDS)`
with no platform guard (the `os.name` test and its `import os` are gone from
the file), then drains the reader, closes the driver's end, and closes the
job (idempotent — the job path of the entry point already closed it; the
session's own close covers the no-pid and failed-assignment paths, and
`spawn()`'s failure path releases it too, so no handle is ever left armed).
POSIX order is byte-for-byte what it was: kill → drain reader → release the
terminal end; the job object on POSIX is inert (no handle, every `assign`
answers `None`), asserted seam-forced on this runner. `pty_posix.py`,
`manager.py`, `proctree.py` and `proctree_jobs.py` needed **no** change — no
capability csl-02 did not provide was required.

**The process list the job actually contained** (measured through a real
session, `sleep 40 &` + a backgrounded `python`): four members —
`bash.exe` (the shell; pid from pywinpty's `PTY.pid`, handed over by
`ConPty.started_pids()` and assigned `leader=True`), `OpenConsole.exe` (the
console helper; pid from the driver's before/after Toolhelp32 child snapshot,
name-matched, assigned as a plain member), and **two further members that are
not children of the app at all** (bash's backgrounded `sleep` and `python` —
they appear in nobody's child list except the job's own, obtained via
`QueryInformationJobObject` class 3 through `job.members()`). That last pair
is the whole point: they joined the job by inheritance at creation and died
with its termination, assigned by nobody.

**Teardown duration observed** (grace patched to 1.0 s through
`termhub.constants`): a live shell + helper + backgrounded child, torn down
through the run loop — **0.20–0.23 s**; a direct `teardown()` — 0.204 s. The
job's graceful step (TerminateProcess on the leader) kills bash outright, so
the force step rarely fires; the promise "at most STOP_GRACE_SECONDS per
stubborn shell" is asserted by a test with a `grace + 1` bound (the extra
second is `_finish_pump`'s drain, the same ceiling `stop_all` allows).

**The `$!` premise disagrees with an observation — recorded.** Git Bash reports
**MSYS-namespace pids** from `$!`/`$$` (measured: bash's Windows pid 8408, its
own `$$` printed 833; a `CHILD-$!` pid was not recognisable by
`OpenProcess`/`GetExitCodeProcess`), so the new tree test cannot name the
backgrounded child from the shell's printout. The observation wins: the ghost
is taken from **`job.members()`** — the OS's own authoritative list of the
tree — and the test first *waits* for a member outside the driver's reported
set, so it asserts only about a child that demonstrably existed and was in
the tree. csl-08's gate should know: on Git Bash, `$!` is not a Windows pid.

**Is there a `setsid`-equivalent escape?** None found on this install.
`setsid.exe` is not in Git's `usr\bin` or `ucrt64\bin` (checked directly);
bash job-control (`&`, `disown`) does not detach a process from a Windows
job — detachment needs the spawn-time `CREATE_BREAKAWAY_FROM_JOB` flag (or a
job that lacks `JOB_OBJECT_LIMIT_SILENT_BREAKAWAY` handling), which the shell
itself cannot pass and its inherited children never get. So the documented
POSIX opt-out has no observed Windows twin for this shell; `_finish_pump`'s
timeout → cancel-the-reader path (leak one descriptor rather than read the
next session's) stays as the guard for a holder that somehow survives, and
the teardown's drain comment now says "the far-end answer" rather than
EIO-only, true on both.

**A child that outlived the grace:** measured with the grace deliberately cut
to 0.05 s — shorter than the force step's drain — **nothing survived**: the
escalation returned in 0.08 s and shell, helper and backgrounded child were
all gone by the liveness probe immediately after, and still gone 1 s later.
That is `terminate_job` (asynchronous but fast) plus the final `close()` —
kill-on-close re-arms over anything the terminate call had not finished
taking. If a member were *still* listed at the deadline, `terminate_and_drain`
logs "job … still holds N member(s)" — the leak stays reported, per csl-02.

**One bounded race, stated rather than smoothed:** assignment necessarily
happens *after* `pty.spawn()` returns (raw ConPTY creation is the rejected
alternative), so a process bash created within the millisecond between its
own creation and its assignment — only reachable from `.bashrc` init, since
no keystroke can reach the shell before `run()` starts after `spawn()`
returns — would be outside the job. Measured: a backgrounded child started at
the first prompt is always inside (the wait-for-member test would otherwise
fail). csl-02's rule ("assign before the shell can run a command") holds in
the only sense the driver allows: before any *command* can be typed.

**Tests.** `tests/test_terminal_conpty.py` grew two groups:
`TestTheSessionOwnsOneJob` (seam-forced, runs on **either** runner — planted
console + recording `ProcessJob`: job named for the session, shell leader /
helper plain member, helper-less driver assigns only the shell, failed spawn
releases the handle, teardown calls the shared entry point with the leader
pid and the grace and closes the job, idempotent; plus the inert-POSIX half
forced the other way) and five real-console `windows_behaviour` cases in
`TestRealPseudoConsole`: disconnect takes shell + helper + backgrounded child
and leaves `job.handle is None`; a **self-exiting** shell still takes the
helper (csl-06 measured the helper outlives a self-exited shell while the
console is referenced — the job is what ends it); a `kill` frame ends the run
loop and the pid inside the patched grace; the teardown returns inside the
grace; and a driver that named no helper still contains the shell (the
containment promise survives the naming gap, with the driver's warning).

**Evidence on this runner.**
```
pytest tests/test_terminal_conpty.py                        → 52 passed (was 42)
pytest tests/test_terminal.py                               → 11 passed, 5 skipped
   (the 5 skips are the pre-existing `no shell at /bin/sh` probe skips —
   csl-08's gate retargets them; nothing new skipped)
scratch run with SHELL pointed at Git Bash:
pytest tests/test_terminal.py                               → 16 passed
   (TestSession unmodified: the kill-frame test, the cap test and
   test_disconnect_tears_down_the_whole_session_tree all pass against the
   job-based teardown; on Windows the tree test's `/proc` helper is the
   vacuous half — the real Windows tree assertion is the new one above)
pytest tests/test_proctree.py tests/test_write_surface_isolation.py → 43 passed, 4 skipped (all named, pre-existing posix marks)
pytest tests/test_mcp.py tests/test_console_gbash.py tests/test_console_policy.py → 144 passed, 7 skipped (pre-existing)
cd web && npm test                                          → 472 passed
git diff --stat web/                                        → empty
```

**Pre-existing on this runner, not caused by this task** (unchanged from
csl-02's record): `tests/test_api.py` — 2 failures in the path-location
route (`NameError: helpers_block` at `api/routes.py`, mid-merge checkout, and
the deny-list fold test on this volume). Nothing in this task's diff reaches
that route; the terminal WebSocket route is not touched.

**Linux promise.** The POSIX teardown order, signals, grace and reaped exit
code are untouched code — `kill_session_tree`'s POSIX half is what runs there
(the seam `JOBS_SUPPORTED` answers False), the session's `ProcessJob` is
inert, and the diff to the teardown body is: the `os.name == "posix"` guard
removed, the same call, and one inert `close()` after the fd release. The
inertness is asserted seam-forced on this runner, so the Linux runner's
`test_terminal.py` results are the check to compare against: they must be
identical, zero skips.
