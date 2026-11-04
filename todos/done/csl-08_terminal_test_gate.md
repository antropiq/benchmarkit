# csl-08 — The test gate: the same terminal assertions on both platforms

## Deliver this

`tests/test_terminal.py` stops being a POSIX-shaped suite that skips on Windows
and becomes **the suite for whichever driver the platform provides** — same
assertions, same names, platform-specific only where a fact genuinely is
platform-specific (C §6: *"spawn/pump/write/resize/teardown run against whichever
driver the platform provides, and the same assertions hold"*).

Land:

1. **A shell probe that answers the selected shell.** Today
   `tests/conftest.py`'s `SHELL = $SHELL or "/bin/sh"` makes `shell_probe` answer
   "no shell" on Windows, which is why the whole `TestSession` class skips there.
   The probe must ask the **same question `termhub/shells.py` answers** (csl-05),
   so a Windows machine with Git gets a real session under test and a Linux box
   keeps its existing answer. One probe, consulted by the marker and the tests —
   conftest's own rule: *"a per-file copy of a probe is how a guard drifts from
   what it guards."*
2. **A liveness probe used instead of `/proc`** where liveness matters:
   `pid_gone()` today reads `/proc/<pid>/stat`; `termhub_session_members()` today
   calls `proctree.session_members`. Both go through csl-02's cross-platform
   liveness answer (or a documented per-platform equivalent), and **no test may
   probe liveness with `os.kill(pid, 0)`, which terminates a process on Windows**.
3. **Platform-free tree assertions**: the tree test finds its child through the
   shell itself (the shell prints the child's pid — POSIX `$!` works in bash and
   in `/bin/sh`) instead of through procfs, so the same test shape runs on both.
   The procfs-specific assertion stays, marked as the POSIX fact it is.
4. **The resize assertion split honestly**: the server-side round-trip
   (`resize` frame → the size the session reports back) is asserted everywhere;
   the *"shell saw it"* variant (`stty size`) is asserted where the shell can
   answer it, and skipped with a sentence naming the invariant where it cannot
   (csl-06's *Results* note tells you whether that is the case on Windows).
5. **The C §6 unit list, complete on both platforms**: first output arrives at the
   right width (size before spawn), a `resize` round-trips, a self-exiting shell
   produces an `exit` frame with a code, a teardown ends the pump inside
   `STOP_GRACE_SECONDS` with no leaked task, the 5th WebSocket is refused with the
   capacity code, and the **absent-driver case** is forced by hiding the driver
   (so it is tested on Linux too).
6. **C §6's Windows-only set**, marked `windows_behaviour`: C-D3's two
   end-of-stream tests, C-D6's non-ASCII round-trip, and csl-07's shell-child +
   console-helper teardown tests (they may already exist from csl-06/csl-07 — this
   task makes sure they exist, are marked, and skip with a sentence naming the
   invariant).

## Files it touches

| file | change |
|---|---|
| `tests/conftest.py` | edit — the shell probe asks the seam's answer; a probe for "a driver is available"; the `windows_behaviour`/`posix_behaviour` markers stay as they are |
| `tests/test_terminal.py` | edit — guards, `pid_gone`, the tree test, the resize test, the new refusal test's home |
| `tests/test_terminal_conpty.py` (from csl-06) / `tests/test_terminal_shells.py` (from csl-05) | edit — consolidate the marking and the skip sentences; move nothing that belongs to the platform-neutral file |
| `src/core/termhub/*` | **only** where a test needs a seam it cannot otherwise reach (e.g. a way to force "no driver"). No behaviour change: this task adds no feature |
| `src/core/proctree_jobs.py` | read-only consumer (its liveness probe) |

## Rules it must not break (quoted)

* roadmap §9 rule 2: *"the difference sits behind one named seam, and the *other*
  platform's tests keep running unchanged"* — and a Windows-only fact is asserted
  through the seam on Linux wherever the seam can be forced.
* roadmap §9 rule 3: *"A refusal is a tested behaviour, not a fallback. Every
  'unsupported' path … gets a test that asserts the sentence and the status/close
  code."* The driver-absent refusal is therefore tested **on Linux**.
* C §6's Linux cadence, verbatim: *"The POSIX terminal tests must not gain a
  single skip: if moving code into `pty_posix.py` costs a test its reach, the move
  is wrong."* The same sentence governs this task: no Linux test may end up
  skipped, renamed-away or weakened.
* AGENTS.md: *"Tests stay hermetic: … a backend test's workspace is a throw-away
  `tmp_path` pointed at by `BENCHMARKIT_PATH_LOCATION`, never this checkout"*, and
  a test patches a seam's own constants rather than a path it computed.
* conftest's own convention: a skipped test *"is *skipped with a sentence naming
  the invariant it would have checked*, never weakened"*, and every platform guard
  consults one shared `_Probe`.
* `tests/test_write_surface_isolation.py` asserts that *"no connector, tool or chat
  path reaches a write route, and the terminal is never a tool"* — no test in this
  task may drive the terminal through a hub, a tool or a chat turn.
* Do not add a Windows-only **behaviour** to make a test pass: if an assertion
  cannot hold on Windows, that is a finding to report to csl-06/csl-07, not a
  marker to sprinkle.

## Depends on

csl-04 (the driver seam and its refusal), csl-05 (`shells.py`, which the shell
probe must consult), csl-06 and csl-07 (the driver + job, whose *Results* notes
decide which assertions are platform-neutral).

## How to tell it is done

1. **Linux**: `pytest tests/test_terminal.py -rs` → every test passes and **no skip
   line is printed**; the test count is at least what it was before this task.
2. **Linux**: `pytest tests -q -rs` → the suite's **new** skips are zero compared
   with before this task; pre-existing skips (tool probes, symlink probe) are
   unchanged. Paste both counts in your *Results* note.
3. **Windows** (with the `windows` extra installed and Git present): `pytest
   tests/test_terminal.py` **passes with real sessions running** — the process
   tests are not all skipped. If Git or the driver is missing on that box, the
   skip sentences must name which capability is absent.
4. There is exactly **one** shell-capability probe in the suite and both the
   `needs_shell` guard and the shell-selection tests consult it.
5. `grep -n "os.kill(.*, *0)" tests/` finds **no** liveness probe; `grep -n
   "/proc/" tests/test_terminal.py` finds only a procfs-specific, explicitly
   marked POSIX test.
6. The driver-absent refusal test runs on Linux, asserts the sentence **and**
   4501, and leaves the seam restored for later tests (a following terminal test
   still spawns a shell).
7. `pytest tests/test_proctree.py tests/test_api.py tests/test_write_surface_isolation.py`
   green; `cd web && npm test` green; `git diff --stat web/` empty.
8. *Results* note: the pass/skip counts on each runner, which assertions had to be
   marked platform-specific (and the sentence each skip prints), and any assertion
   from C §6's list that could **not** be made to hold — that is the finding the
   docs task must record.

## Out of scope here

Playwright e2e for the terminal pane is **deferred by this lot** (big picture):
it needs a Windows runner with a real shell, and the lot's Linux-safe scope cannot
assert it. Do not add a spec that would skip on both runners; csl-09 records the
deferral instead.

## Results (append as you measure)

Run on this Windows runner (Git at `C:\Program Files\Git`, `pywinpty` 3.1.2 from
the `windows` extra) — the runner csl-06/csl-07 left green. The Linux half is
**not re-runnable from this machine**; what was proven instead is stated below.

**1. The one shell probe (criterion 4).** `tests/conftest.py`'s `shell_probe`
now asks `termhub.shells.select_shell()` with no arguments — the seam's own
answer (csl-05's `SELECTION_PLATFORM` note) — and keeps the program it names for
the tests (`selected_shell()`). The old `$SHELL or /bin/sh` + `shutil.which`
probe is gone, and `SHELL` as a conftest global is gone with it (grep: no test
reads it). `driver_probe` likewise asks `termhub.drivers.driver_available()`,
and `windows_driver_probe` asks `pty_conpty.available()` — the
`test_terminal_conpty.py` gate was consolidated onto it, so the file no longer
owns a private `_Probe`. A guard-test in `test_terminal_shells.py`
(`TestTheSuitesShellProbe`) asserts the probe and the seam never disagree, so
"one probe consulted by the marker and the tests" is pinned, not hoped.

**2. The guards are markers, not fixed skipif reasons.** `needs_shell`,
`needs_driver`, `needs_stty`, `needs_windows_driver` joined
`_GUARD_MARKERS` (the `windows_behaviour`/`posix_behaviour`/`requires_symlink`
markers are untouched). They are markers rather than `skipif`s for one reason:
a skipif reason is written at import time and cannot carry what the probe
*learned*. `_Probe` grew an optional `detail` (and `clear_verdict()`), and a
failed shell probe quotes the seam's own refusal sentence. Measured here with
`TERMINAL_SHELL=C:\nope\bash.exe` (a scratch run, not committed): 25 session
tests skip with `needs_shell: … names no program here (the configured shell is
not usable on Windows: TERMINAL_SHELL='C:\nope\bash.exe' is not an existing
executable file. …)`, and forcing both driver probes to answer None produces
`needs_driver: no pseudo-terminal driver answers available (the kernel PTY, or
the 'windows' extra's ConPTY), so no session can spawn at all`. So criterion 3
of the *task* ("the skip sentences must name which capability is absent") is
satisfied for both the no-Git and no-driver cases.

**3. Liveness (criterion 5).** `pid_gone()` is now
`proctree_jobs.process_is_gone` — csl-02's cross-platform probe, whose POSIX
half is the same `/proc/<pid>/stat` state read with the same zombie rule (the
test's behaviour is unchanged there; the old inline body is verbatim that
function's POSIX branch). `grep "os.kill(.*, *0)" tests/` finds only doc-comment
warnings about it, no call. `termhub_session_members` (the procfs helper) stays
but only the one marked POSIX test uses it.

**4. The tree assertion: the `$!` premise disagrees with an observation, again.**
Re-measured here: Git Bash's `$!` is an MSYS pid (1951, 1777, 1782 in three
runs) while the child's Windows pid was 10620/4340-ish and shows in `ps -aW`'s
WINPID column — csl-07's finding reproduced. The platform-free shape that *does*
hold on both: the shell prints its child's pid, translated by the shell itself —
`$!` verbatim where it is the OS pid (Linux), `ps -aW`'s WINPID column where
`uname` says MINGW/MSYS/CYGWIN. The test then *pre-checks* the printed pid is a
live process before asserting about it, so a failed translation fails loudly
instead of going vacuous. `test_disconnect_tears_down_the_whole_session_tree`
runs and passes on **both** shapes here (Git Bash over ConPTY). The procfs
session-scan assertion kept its own explicitly marked POSIX test
(`test_the_disconnected_shell_leads_no_live_session_member`,
`@pytest.mark.posix_behaviour` with the sentence naming the job twin).
`grep "/proc/" tests/test_terminal.py` → zero hits (the marked test delegates to
`proctree.session_members`, no literal path in the test file).

**5. The resize split (criterion 4 of the task).** `test_resize_updates_the_window_size`
(the *shell saw it* half, `stty size`) is guarded by `needs_stty`, whose probe
asks the child's own search path through `shells.child_environment` — measured:
`shutil.which("stty")` in the test process is None on this box, the pane
child's PATH carries it, so a PATH-of-the-test-process probe would have skipped
a passing capability. The server-side round-trip is its own test,
`test_the_size_the_session_opens_with_round_trips_a_resize`: the size the driver
reports *before any input* (C-D4's size-before-spawn) and after a `resize`
frame, against the platform's real driver on either runner. Kept honest about
csl-06's hazard: the resize and the command are separate frames with the
shell's first output drained between them and a short pause — with them in one
tick, csl-06 measured keystrokes garbled by the agent's `ESC[6n` query.

**6. C §6's unit list, complete.** New here: the **default-cap** fifth-WebSocket
refusal (`test_the_default_cap_refuses_the_fifth_websocket`, 4429, nothing
spawned — C §6 names the 5th), the **grace promise**
(`test_a_teardown_ends_the_pump_inside_the_grace_with_no_leaked_task`: real
driver + real shell + backgrounded child, disconnect, the run loop ends inside
`STOP_GRACE_SECONDS` + the reader drain, and no task outlives it), and the
**driver-absent case forced through the probes** (not just the
`DRIVER_ABSENT_FORCED` flag): `hidden_driver_probes` patches the two probes
`selected_driver()` asks, asserts the same error frame + 4501, and its tail
restores them and re-answers the probe — the following `TestRegistry`/
`TestSession` tests then spawn real shells, which is the second proof the seam
came back (criterion 6). `TestGuards`' two session-opening guards now carry
`needs_shell`/`needs_driver`: on a driverless Windows the route refuses *after*
accepting, and an assertion about an opened session was never a skip — those
five tests were the "pre-existing Windows hang" csl-04 recorded; here they pass
or skip honestly.

**7. Counts.**
```
pytest tests/test_terminal.py -q -rs   (this Windows runner, driver+Git present)
  before csl-08: 11 passed,  5 skipped   (all five "no shell at /bin/sh")
  after  csl-08: 20 passed,  1 skipped   (the one skip is the marked POSIX
                                          procfs test; 16 -> 21 tests)
pytest tests/test_terminal.py tests/test_terminal_conpty.py
     tests/test_terminal_shells.py      → 90 passed, 1 skipped
pytest tests/test_proctree.py tests/test_write_surface_isolation.py
                                         → 43 passed, 4 skipped (all pre-existing
                                         posix marks, sentences unchanged)
pytest tests/test_api.py               → 63 passed, 2 failed, 1 skipped —
   identical to csl-04/05/07's record: the `NameError: helpers_block` at
   `routes.py:1418` (mid-merge checkout) and this volume's NTFS fold. Not
   terminal, not this diff.
pytest tests (one call, ~2 min):  2248 collected → 2104 passed, 95 failed,
   49 skipped. The 95 failures are all in editorfs*/git/search/policy-seed/
   lsp/prompt-helpers/api — Windows path-spelling, CRLF, symlink-privilege and
   the LSP hub's Windows refusal (out of scope: "leave the refusal exactly as
   it is") — plus the routes.py NameError; the same set csl-02/04/05/07 each
   recorded around their own diffs. Zero terminal-file failures, and no test I
   touched fails. Across the three terminal files the skips went 5 -> 1 (the
   five "no shell at /bin/sh" skips are now one marked-POSIX skip): this task
   removed skips rather than adding any, and no new skip category appeared in
   the 49 the whole suite prints here.
cd web && npm test                     → 472 passed; git diff --stat web/ empty
```
No behaviour of `src/core/termhub/*` or `proctree_jobs` was changed — the diff
is `tests/conftest.py`, `tests/test_terminal.py`, `tests/test_terminal_shells.py`
(2 guard-agreement tests + comment), `tests/test_terminal_conpty.py` (marking
consolidation + comment) only.

**8. What could not be made to hold platform-neutrally (for csl-09 to record).**
* The bare `$!`-pid tree assertion (item 4): Git Bash's `$!` is an MSYS pid; the
  platform-free form is "the shell prints the child's OS pid", with each shell
  translating its own namespace. Not a defect of the seam — a fact of the shell.
* The procfs *session-scope* assertion: no Windows equivalent of "who shares
  this session id" exists; the job's member list is the equivalent and is
  already asserted (marked `windows_behaviour`) in `test_terminal_conpty.py`.
* Everything else on C §6's list holds on both shapes, including the shell-side
  `stty size` resize on Git Bash over ConPTY.
* **Linux promise:** not re-runnable here. The Linux path is unchanged code
  reached by the same names — `_select_posix` untouched, `pid_gone`'s `/proc`
  read verbatim the shared probe's POSIX branch, `needs_stty` answering yes on
  any box with `/bin/sh` + coreutils, the marked POSIX test running there. The
  expected Linux result is 21 passed, **zero** skips for this file; if a Linux
  runner prints any skip line for `tests/test_terminal.py`, that is this task's
  bug to fix, per C §6's cadence.
