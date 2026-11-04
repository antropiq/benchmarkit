# Target C — the Editor's terminal on Windows (ConPTY)

**Status**: **implemented** by the csl lot (csl-01 … csl-08, 2026-11-02) — the
terminal runs Git Bash over ConPTY on Windows, and `tests/test_terminal.py`
holds the same assertions on both platforms. Two things stay open on purpose:
§4's Playwright spec is **deferred** (it needs a Windows runner with a real
shell), and §6's **manual Windows pass has not been performed by any task in
the lot** — it is a human step, restated with its pass criteria in §6.
**Depends on**: target A (path semantics, the test gate) and **B-D5 specifically**
(the process-tree teardown a session needs), landed as
`src/core/proctree_jobs.py`.
**Read first**: roadmap §5 and §6. **Code referenced**: `ef21ea3` + the A, B1 and
csl changes; the csl lot's own measured findings are in its task files,
`todos/done/csl-01_gitbash_probe_module.md` through
`todos/done/csl-08_terminal_test_gate.md`.

## 1. What C is for

C makes the Editor tab's terminal run a real interactive shell on Windows, with the same guarantees the POSIX path already makes: a session is a *process tree*, it is capped, it is resized by the browser, and it dies with the app.

The AGENTS.md of the planning day described what C replaced (the row has since
been rewritten by csl-09, per §7):

> `core/termhub/` — Editor's terminal: one PTY shell per WebSocket, session registry on `app.state.terminal`

"PTY" is the POSIX-only word in that sentence, and `termhub` is the only place in the roadmap where the gap is *already handled well*: the module imports `pty`/`fcntl`/`termios` behind an `ImportError → None` guard (`session.py:43-53`) purely so it imports on Windows, `spawn()` raises `TerminalUnsupported` (`:137-140`), the route answers an `error` frame then `CLOSE_CODE_UNSUPPORTED` = 4501 (`api/routes.py:1655-1664`, and the close-code block in `termhub/constants.py`), and `useTerminal.js` renders the sentence. **C's job is to remove that refusal, not to change its shape.**

*Where those pieces are now*: the import guard and the PTY machinery live in
`termhub/pty_posix.py` (csl-03), the raise is `spawn()` asking
`termhub/drivers.py` (csl-04), the route's guard is
`term_drivers.driver_available()`, and `CLOSE_CODE_UNSUPPORTED`,
`NO_DRIVER_MESSAGE` and the sentence's rendering in `useTerminal.js` are
unchanged. The line numbers above are the planning day's.

## 2. Decisions taken

### C-D1 — ConPTY, behind one seam, with `pywinpty` as the Windows driver

Windows' pseudo-console (ConPTY, `CreatePseudoConsole`) is the only mechanism that gives a console program a real terminal: it owns the console buffer, translates Win32 console API calls into VT sequences, and reflows output on resize — which is what `vim`, `less`, `top` (named in `termhub/constants.py:38`) need.

**Decision**: keep `termhub/session.py` as the single session class and the single frame contract, and put the pseudo-terminal behind one seam:

```
termhub/pty_posix.py   the today-behaviour: openpty, TIOCSWINSZ, TIOCGWINSZ, the EIO-ends-the-read rule
termhub/pty_conpty.py  (new) the Windows driver, same small surface
termhub/session.py     (edit) asks the seam; keeps spawn/pump/write/resize/teardown/run/status as now
```

The seam is deliberately tiny — open, read, write, set_size, close — because the valuable part of `session.py` is not the PTY, it is everything around it: the `PTY_READ_BYTES` framing rule (`pump`, `:216-257`), the incremental UTF-8 decoder that keeps a multi-byte character whole across chunks (`:112-114`), the publish-failure-means-client-left path (`:246-249`), the session cap (`MAX_SESSIONS = 4`), the close codes, and the origin policy in `termhub/policy.py` (no platform dependency, unchanged by C).

**Driver**: `pywinpty` (the PTY layer Jupyter and VS Code's terminals sit on) instead of raw `CreatePseudoConsole` through `ctypes`. It already handles the conhost/`OpenConsole.exe` dance, which is where the subtle bugs live.

| alternative | why rejected |
|---|---|
| Raw ConPTY via `ctypes` | keeps the dependency count at zero, but re-implements a console-host handshake in a module whose failure mode is a hung terminal. B-D5 chose `ctypes` for job objects because that surface is four functions; this one is not |
| Plain pipes (`Popen(stdin=PIPE, stdout=PIPE)`) on `cmd.exe` | no terminal at all: no `isatty`, no resize, no line discipline, no full-screen app. It would make the tab a log viewer, and `constants.py:38`'s `TERM_VALUE` would be a lie |
| Bridge into WSL (`wsl.exe`) so the existing POSIX code runs | needs WSL installed, and — decisively — the shell's filesystem would no longer be the working location. AGENTS.md: *"One setting defines the working location that every file-touching part of the app roots itself at"* and *"Every path the app holds is relative to it"*; a terminal browsing a WSL mount while `fedit` edits the Windows tree splits the one working location the whole app is built around |

`pywinpty` goes in a **new optional extra**, not in `[mcp]`:

```toml
windows = ["pywinpty>=2.0; sys_platform == 'win32'"]
```

so a Linux install never resolves it (roadmap §9: the platform difference sits behind one named seam). C is the **only** target in the roadmap that adds a dependency, and this is the place it is stated.

### C-D2 — Which shell, and what "interactive" means there

`shell_command()` (`session.py:82-92`) reads `$SHELL`, requires `os.path.isabs` and `os.access(..., os.X_OK)`, and falls back to `constants.FALLBACK_SHELL = "/bin/sh"`. On Windows there is no `$SHELL`, the fallback does not exist, and the interactive flag is the bash-specific `-i` (`:148` — `[self.shell, "-i"]`).

**Decision**: the seam also owns shell selection, so the fallback lives with the platform. Landed as `termhub/shells.py` (csl-05); the POSIX half is `shell_command()`'s old body moved verbatim (`$SHELL` → `FALLBACK_SHELL`, `-i`).

**Amended by the operator (2026-11-02): the Windows shell is Git Bash.** The
original order was `TERMINAL_SHELL` → `%COMSPEC%` (`cmd.exe`) → refuse, and this
section explicitly refused to auto-pick a shell. The amendment removes the
`cmd.exe` step: the order that landed is

1. `TERMINAL_SHELL` when it is **drive-qualified** and executable — a
   POSIX-shaped value (`/bin/bash`, or the `/c/…` mount form a Git Bash window
   carries in `SHELL`) is refused **by name**, not spawned and not fallen
   through: `os.path.isabs("/bin/bash")` answering `False` on Windows was luck,
   not a rule, and a silent fallback would make the one knob dishonest.
   *Measured correction:* `path_norm.is_qualified` alone is **not** the drive
   rule — a leading separator counts as qualified (that is how it closes the
   `\etc\passwd` hole), so `is_qualified("/bin/bash")` is `True`. The landed
   check asks two questions of the spec — `is_qualified` (refuses a bare or
   relative name) **and** a non-empty `ntpath.splitdrive` root (refuses
   `/bin/bash` and every MSYS-mount spelling, accepts `C:\…`, `C:/…` and a real
   UNC). It asks them of the raw text, because the canonical form folds a UNC
   root's leading separators away.
2. **Git Bash**, found through the probe the console connector already proved
   (G-D4): `mcp_connectors/gitbash.py` — one install probe, two callers — with
   its `git_child_env` handing the child the same search order.
3. else refuse, with a sentence naming everything that was looked for.

*Measured correction on why the `PATH` hand-off matters.* The premise handed
down was that a Git Bash child given only the ambient Windows `PATH` answers
"command not found" for `stty`/`tty`/`which`. Run against the real
`C:\Program Files\Git\bin\bash.exe`, that is **not** what this install does:
`command -v` resolved `stty`, `tput`, `which`, `tty`, `vim` and `less` in every
case tried — our env, the ambient `PATH`, a `PATH` of System32 only, and no
`PATH` key at all — because `usr\bin\bash.exe` rebuilds its own `PATH` from the
install's POSIX root before it execs anything. `git_child_env` stays (it is the
documented, asserted order G-D4 proved, and it is what a driver that cannot rely
on bash's self-configuration needs), but two consequences are real: a missing
`PATH` entry is **not** the cause of a future "command not found" here, and
csl-08's `stty` probe must ask the *child's* search path rather than the test
process's (on this box `shutil.which("stty")` in the test process is `None`
while the pane's child has it — a probe of the wrong `PATH` would skip a passing
capability).

`%COMSPEC%`/`cmd.exe` is *not consulted at all* — not as a fallback, not as a
candidate — and `pwsh`/`powershell` are still never auto-selected (the original
rejection stands: the app does not decide a user's shell). Why Git Bash wins:
the app's console story on Windows is already Git's toolset (B1 landed), and a
`cmd.exe` pane would give one working location two different tool vocabularies.
The amendment buys no escape from C-D1/C-D3/C-D5: a pipe pair would give bash no
tty, so the pseudo-console, the EOF rule and the job teardown all still land.

The interactive flag belongs to the shell spec, not to `session.py` (which used
to hard-code the bash-specific `-i`): Git Bash takes `-i` for the reason
`constants.py:44-46` gives — interactive-non-login reads `~/.bashrc` without
re-running the profile's environment setup once per tab.

`TERMINAL_SHELL` is **process-environment only**, never a managed `.env` key —
the same reasoning `BENCHMARKIT_PATH_LOCATION` and `BENCHMARKIT_GIT_BASH_DIR`
carry (a settings save must not erase it), so the managed template's key set is
not widened for a platform detail.

* `TERM`/`TERM_VALUE`: set for the child as today, beside `COLUMNS`/`LINES`. ConPTY does not need `TERM` to work, but a program that reads it should still see `xterm-256color` rather than nothing — and Git's `usr\share\terminfo` ships a real database that resolves it.

This keeps the app's one knob honest: the shell is the user's choice where the
user has expressed one, and the platform's proven toolset where it has not.

### C-D3 — The pump's end-of-stream rule differs, and must be re-stated

`pump()` (`:216-257`) blocks in `os.read(master_fd, PTY_READ_BYTES)` on a thread and relies on a specific fact: *"it is unblocked by the teardown, because once the shell and its children are gone the master answers `EIO`"* (`:221-223`), with the matching comment at `:238` (*"EIO once every holder of the slave is gone"*). The whole shutdown sequence — `_finish_pump` (`:259`), `teardown` (`:362`), and `spawn`'s careful closing of the parent's copy of the slave (`:126-129`: *"while any process holds the slave open, a read of the master never reports the shell's death"*) — is built on that.

A ConPTY pipe ends with **EOF (an empty read)**, not `EIO`, and there is no slave fd to close. **Decision**: the seam exposes one predicate — *"is the far end gone?"* — and `pump()` keeps its shape while the reason moves into the driver. Two tests pin the behaviour on both platforms:

1. shell exits on its own (`exit`, Ctrl-D) → an `exit` frame with the code;
2. teardown requested → the pump stops within the grace period, no leaked task.

`STOP_GRACE_SECONDS = 5.0` (`constants.py:24`) is unchanged: it is *"the same number and the same rule as both hubs'"*, and C must keep that sentence true.

**As landed (csl-03, csl-06).** The predicate is the `FarEndGone` exception
(`termhub/pty_posix.py`) — an `OSError` subclass on purpose, so `pump()` keeps
its single `except OSError: break` and the POSIX outcome is byte-identical to
before the move. The ConPTY driver answers the same question three ways,
measured on the real console: `iseof()` or a dead child, and a firm answer from
a released pair — because a blocking read of a *silent live* console waits for
input indefinitely, the driver never enters one; it polls the non-blocking read
every `READ_POLL_INTERVAL_SECONDS` (20 ms), and a teardown's `close_master()`
(`cancel_io()` then release) ended a blocked read in 2–17 ms. The exit code is a
separate answer (`get_exitstatus()`; a child gone with no status is
`EXIT_CODE_UNKNOWN`, never an invented 0), so an EOF is not mistaken for proof
the shell exited cleanly — AGENTS.md's *"a stream cut by the server looks like a
finished answer"* applies here too. A real pipe failure while the shell lives
raises `ConPtyError`, which is an `OSError` that is *not* `FarEndGone`, so it
keeps the pump's unexpected-error path. Both C-D3 tests pass on the real
pseudo-console (`tests/test_terminal_conpty.py`); the same two assertions on the
platform's own driver live in `tests/test_terminal.py` under the `needs_shell` /
`needs_driver` guards (csl-08), so neither platform runs a suit the other cannot.

### C-D4 — Resize has a different delivery mechanism; the contract does not change

POSIX: `_set_window_size` (`:175-180`) `TIOCSWINSZ`s the master *before* spawn so the first prompt wraps correctly, `resize()` (`:189-194`) re-issues it and the kernel sends `SIGWINCH`, `terminal_size()` (`:196-214`) reads it back. The client side is already platform-neutral: xterm.js + `FitAddon`, a debounced 50 ms `pushResize` (`useTerminal.js:216-236`), and the guard for a hidden host yielding 0 rows (`:225-229`).

ConPTY: the size is passed when the pseudo-console is created (so "set before spawn" maps exactly), and `ResizePseudoConsole` reflows the buffer. There is no `SIGWINCH`; a program learns about the new size by the console reporting it.

**Decision**: the seam's `set_size`/`get_size` cover both; `session.py`'s three call sites do not change, and the client needs **no change at all** — ConPTY emits VT, which is what xterm.js already consumes. State the one honest difference in the driver's docstring rather than papering over it: a POSIX program notifies itself on `SIGWINCH`, a Windows console program is reflowed by the console, and a program that reads its size once at startup behaves differently on the two. That is a fact about the platforms, not a defect to hide.

**As landed, with the shell-visible answer measured (csl-06).** The
server-side round-trip is asserted on both runners
(`test_the_size_the_session_opens_with_round_trips_a_resize`, against the
platform's real driver, before any input — the size-before-spawn case). The
question this section left open — does
the *child* see the resize — answers **yes for this shell**: `stty size` printed
the new size after a resize (observed `41 132`, `30 100`), because bash
re-reads, while the `COLUMNS`/`LINES` pair set at spawn keeps its old number —
exactly the stated platform difference. One hazard found, and the test gate is
honest about it rather than smoothing it: on a resize the ConPTY agent emits a
device-attribute query (`ESC[6n`) and eats keystrokes until the terminal answers
it (xterm.js does answer; a raw test client does not). Measured: input written
in the *same tick* as a resize was garbled 4 times in 5, and the suite's
shell-saw-the-resize test keeps the resize frame and the command frame separate
with the shell's first output drained between them — which is the shape a real
pane produces. The `stty` half is guarded by `needs_stty`, whose probe asks the
*child's* search path (`shells.child_environment`), because the test process's
own `PATH` does not carry `stty` on this box while the pane's child does.

### C-D5 — Teardown is B-D5's job object, extended to the console host

`teardown()` (`:362-397`) calls `proctree.kill_session_tree` guarded by `os.name == "posix"` (`:386`), and the module docstring explains why session-scope matters: a job-controlling shell puts every job in its own group, so a `killpg` would reach the shell alone and a backgrounded `python -m http.server` would outlive the tab (`proctree.py:18-27`).

**Decision**: after B-D5, the terminal asks the *same* shared teardown, and the job it creates must contain **both** the shell and the `OpenConsole.exe`/conhost process `pywinpty` starts for it. That second process is the one an implementer will forget, and it is precisely the *"kill process trees, not wrappers"* case AGENTS.md names.

Ctrl-C is a separate mechanism, not a signal: a PTY's line discipline turns `^C` into `SIGINT`; ConPTY turns the input event into a `CTRL_C_EVENT` for the console group. Same keystroke, same effect, and the seam must own the difference rather than have `write()` special-case byte 3 — `session.py:182-187` stays a plain byte writer. `^Z` and job stop/resume are **not** promised on Windows.

**As landed (csl-02, csl-07).** The session opens its one `ProcessJob` in
`spawn()` **before** the console is created, the shell is assigned
`leader=True`, and every other pid the driver names through `started_pids()` is
a plain member — pywinpty does not hand over the helper's pid, so
`ConPty.spawn_shell()` snapshots this process's children around the console
creation and claims the new ones whose exe name is in `CONSOLE_HELPER_NAMES`
(`OpenConsole.exe` on the ConPTY backend, `winpty-agent.exe` on the legacy one).
`teardown()` calls `proctree.kill_session_tree` with **no platform guard** (the
`os.name == "posix"` branch is gone from `session.py`): on Windows the entry
point lands in the job half, on POSIX in the group half, one call site.
Measured on the real console: a disconnect took shell + `OpenConsole.exe` + a
backgrounded child (0.20–0.23 s with a 1 s grace); the backgrounded child was a
job member *nobody assigned* — it joined by inheritance at creation, which is
the property B-D5 chose a job for. Two honesty findings: a **self-exiting**
shell leaves the helper alive while the app references the console, so the job,
not the dependency's release, is what ends it (asserted); and no
`setsid`-equivalent escape was found on this install — `&`/`disown` do not
detach from a Windows job, which needs the spawn-time breakaway flag the shell
cannot pass. Ctrl-C through `write()` with nothing special-cased aborted a
half-typed line and interrupted a foreground `sleep`, as specified.

### C-D6 — Encoding: verify, do not assume

The pump decodes incrementally as UTF-8 with `errors="replace"` (`:114`, `:244`). ConPTY's internal buffer is UTF-16 and the driver hands the app bytes, but a `cmd.exe` child writing under a legacy OEM code page can produce output that is not UTF-8 — and a Chinese or accented path in a directory listing is the realistic case. **Decision**: a test that echoes a non-ASCII string and reads it back identically, run on Windows. If the driver cannot guarantee UTF-8, the session sets the console to UTF-8 for the child (the `chcp 65001`-equivalent API call) rather than letting mojibake reach xterm. This is listed as an *unknown to resolve on the Windows machine*, not a settled design, because it depends on the driver's behaviour — exactly the kind of claim this roadmap has so far refused to make from the sofa.

**The answer, measured on the Windows machine (csl-06): no extra step is
needed.** The first option — the driver guarantees UTF-8 for this shell — is
what held. `echo café中文 ☃` came back byte-identical through the real
pseudo-console (accents, CJK, snowman, also Greek `π`), and `printf '\303\251' |
wc -c` under the session answered 2: Git Bash's locale is `C.UTF-8` and ConPTY
carries UTF-16 internally with the agent framing its pipe as UTF-8. The driver
converts symmetrically at the boundary (`utf-8` / `surrogateescape`), and the
byte cap is applied to the converted bytes, so the session's per-chunk framing
and its incremental decoder both held (a multi-byte character cut by the cap was
still joined). No `chcp`/code-page call was added — the OEM-code-page worry was
a `cmd.exe` worry, and `cmd.exe` is not this pane's shell. A non-UTF-8 byte a
child does write is repaired by the console itself (a lone `\xff` arrives as
U+FFFD), which matches the session's `errors="replace"` rule. The round-trip is
pinned by `test_a_non_ascii_echo_comes_back_as_the_same_bytes`
(`tests/test_terminal_conpty.py`, `windows_behaviour`).

## 3. What it builds on

* B-D5's `proctree_jobs.py`. C is the second caller of that seam; a terminal session with no job is a leak, and B is what makes "the same teardown" mean something on both platforms.
* `termhub/constants.py` as the patch-point convention (*"a test that exercises a cap or a grace period patches it here"*) — C adds no new module-level constants beyond the driver's own file.
* The refusal path as the fallback that stays: if `pywinpty` is absent on Windows, or the pseudo-console cannot be created, `TerminalUnsupported` and close code 4501 still happen, with a sentence naming the missing dependency. A's test for that refusal (roadmap §9 rule 3) is not deleted by C — it becomes the test for "driver not installed".
* `useTerminal.js`'s existing rendering of an `error` frame, and the whole xterm.js stack: **no frontend change in C.** That is the payoff of the frame contract (`input`/`resize`/`kill` out, `output`/`exit` in, `useTerminal.js:19`).

## 4. Shape and files of the pieces

As planned, plus one extraction the amended C-D2 forced (the pane and the
console connector must not own two copies of "where is Git"), and one seam file
the plan left implicit (the selection question in front of the two drivers).
Job assignment stayed in `session.py`, not in the driver, so the same
containment call runs on both platforms.

| piece | landed as | what it holds |
|---|---|---|
| `src/core/termhub/pty_posix.py` | **new** (csl-03) | the code moved verbatim out of `session.py`: `openpty`, the slave-close rule, `TIOCSWINSZ`/`TIOCGWINSZ`, and the `EIO` → `FarEndGone` translation (the seam's "is the far end gone?" answer) |
| `src/core/termhub/pty_conpty.py` | **new** (csl-06) | the `pywinpty`/ConPTY driver: console created **with** its size, byte-level read/write cut to `PTY_READ_BYTES`, resize, the EOF → `FarEndGone` answer, `spawn_shell` (the pair starts the child; there is no slave), and `started_pids()` naming the shell **and** the console helper for the session's job |
| `src/core/termhub/drivers.py` | **new** (csl-04) | the seam's selection question — `selected_driver()` asks each driver's own `available()` probe, never `os.name` — plus the `TerminalPty`/`TerminalDriver`/`TerminalChild` Protocols and `pair_starts_the_child()` |
| `src/core/termhub/shells.py` | **new** (csl-05) | shell selection per platform (C-D2 as amended: `TERMINAL_SHELL` drive-qualified → Git Bash → refuse) and the child's environment (`TERM`, `COLUMNS`/`LINES`, Git's `PATH` order) |
| `mcp_connectors/gitbash.py` | **new** (csl-01) | the one Git-install probe — discovery, search paths, the shell program, the child env — shared by the console connector and `termhub.shells`; stdlib-only, policy-free |
| `src/core/proctree_jobs.py` | **new** (csl-02, B-D5) | the Job Object half of the teardown behind `proctree`'s two entry points, plus `process_is_gone` — the cross-platform liveness probe (a liveness probe must **never** be `os.kill(pid, 0)`: CPython maps a plain signal to `TerminateProcess` on Windows) |
| `src/core/termhub/session.py` | edit | keeps its structure and every frame rule; asks the seam at spawn and raises `TerminalUnsupported` with the no-driver or no-shell sentence; opens the one `ProcessJob` at spawn; `teardown()` calls the shared entry point with no platform guard |
| `src/api/routes.py` | edit | the guard becomes `term_drivers.driver_available()` rather than `os.name != "posix"`, keeping `CLOSE_CODE_UNSUPPORTED` (4501) and the one `NO_DRIVER_MESSAGE` sentence |
| `pyproject.toml` | edit | the `windows` extra: `pywinpty>=2.0; sys_platform == 'win32'` |
| `tests/conftest.py`, `tests/test_terminal*.py` | edit / **new** | csl-08's gate: the `needs_shell` / `needs_driver` / `needs_stty` / `needs_windows_driver` guard *markers* (they carry what the probe learned in the skip sentence), one shell probe that asks `shells.select_shell()` itself, and the new driver/shell/job suites |
| `web/e2e/` | **deferred** | the terminal's first e2e spec was **not** written: it needs a Windows runner with a real shell to assert anything, so it cannot be a Linux-safe task in this lot. The gap is the pane-level half of §6's manual pass, not a unit gap. |

## 5. Out of scope for C

* Job control stop/resume: `^Z`, `fg`/`bg`, `SIGTSTP`. There is no Windows equivalent of the signal, so nothing beyond what bash and the console already do is claimed or tested here (the shell's own `&` works, and the job teardown is what catches what it starts).
* More than one shell flavour, shell-detection UX, a shell picker in the Editor UI. (The amended C-D2 settled *which* shell, so the picker question never arose.)
* Scrollback parity, mouse reporting, OSC 52 clipboard. The pane is already xterm.js, so whatever the browser does, it does — C promises no more.
* Any claim about `TERM`-database behaviour for programs that assume extensions a real terminal emulator provides.
* The rest of B (connectors, command sets, guardrails): independent of C.

## 6. Verification

**The suite as it runs now.** Three files, and the platform's own driver is
what they exercise — one shape, not a POSIX suite and a Windows suite:

| file | what it proves |
|---|---|
| `tests/test_terminal.py` | the contract, against whichever driver and shell the platform answers: spawn/pump/write/resize/teardown, the frame sequence, the cap, the guards, the refusal |
| `tests/test_terminal_shells.py` | `shells.py`: the per-platform selection order, the drive-qualified rule, both refusal sentences, the child env |
| `tests/test_terminal_conpty.py` | the Windows driver: the seam half (planted `winpty`, runs anywhere) and the real-pseudo-console half (`windows_behaviour`) |

**Unit, both platforms** (`tests/test_terminal.py`, re-gated by csl-08):

* spawn/pump/write/resize/teardown run against whichever driver the platform provides, and the same assertions hold: the first output arrives at the right width (C-D4's "size before spawn"), `terminal_size()` round-trips a `resize`, a self-exiting shell produces an `exit` frame with a code, a teardown ends the pump inside `STOP_GRACE_SECONDS`, and the session cap refuses the 5th WebSocket with 4429.
* the absent-driver case: `TerminalUnsupported` + 4501 + the naming sentence — forced by hiding **the drivers' own probes**, so it is tested on Linux too, and the same tests that hide them restore them afterwards, which is the second proof the seam came back.
* the guards are **markers, not fixed skip reasons** (`needs_shell`, `needs_driver`, `needs_stty`, `needs_windows_driver`): each asks the seam itself (the shell probe calls `shells.select_shell()`; `pid_gone` is `proctree_jobs.process_is_gone`, never `os.kill(pid, 0)`), and a skip **quotes what the probe learned** — the seam's own refusal sentence — so a skipped test says which capability is absent instead of hiding the reason.

**Windows-only**: C-D3's two end-of-stream tests; C-D6's non-ASCII round-trip; and the teardown test that a shell's *child* and the console helper are both gone after `teardown()` — the point of C-D5, and the reason it cannot be asserted on Linux. The tree assertion has one measured difference from the plan: **Git Bash's `$!` is an MSYS pid, not a Windows pid**, so the test has the *shell* print its child's OS pid (each shell translates its own namespace; on Git Bash via `ps -aW`), pre-checks that pid is live, and asserts about it; the job's own `members()` list is the authoritative Windows twin, asserted in `test_terminal_conpty.py`. The procfs session-scan keeps its own `posix_behaviour` test, with the sentence naming its job-object twin.

**Linux regression, every session** (the cadence B already set): `pytest` green with **zero** skips, `cd web && npm test` green, `cd web && npm run test:e2e` where `.env.test` reaches an endpoint. The POSIX terminal tests must not gain a single skip: if moving code into `pty_posix.py` costs a test its reach, the move is wrong. **`tests/test_terminal.py` must show zero skips on Linux** — the one skip it has on Windows is the marked-POSIX session scan, which runs there instead. The lot itself ran on a Windows machine, so that Linux line is a **still-open check**, not a measured one: 21 pass with no skips is what csl-08 predicts, and if a Linux runner prints any skip for that file, that is csl-08's bug to fix. The lot's Windows record for the same file was 20 passed / 1 skipped, and it removed skips rather than adding any (five "no shell at /bin/sh" skips became that one).


**Manual, on Windows — not performed by any task in this lot.** It is the human
step that separates "it spawns" from "it is a terminal", and no automated task
here can run it (it needs a desktop browser and Task Manager). Each line states
what proves it passed:

1. `vim` a file, resize the pane, quit — pass: no artefacts, text reflowed to the new width, prompt intact.
2. A full-screen program (`less`, `vim`) after a resize **mid-draw** — pass: the repaint fills the new grid with no stale rows at the bottom.
3. A backgrounded child: `sleep 40 &` (or `python -m http.server`), close the tab, then Task Manager — pass: **no `bash.exe`, no `OpenConsole.exe`, no child** of either. This is the ghost `proctree` was written for; the job tests assert it headlessly, this checks it in the real app.
4. Ctrl-C a running command; Ctrl-L clears; arrow-key history — pass: the interrupt returns the prompt, the screen clears, history walks.
5. Resize the browser window and re-show the tab — pass: no 0-row state (the `useTerminal.js` hidden-host guard); the pane refits and the shell's next prompt wraps at the visible width.
6. Kill the app from the server console — pass: no shell, helper or backgrounded child survives (kill-on-close, observed rather than trusted).

If any line fails, it is a finding for `termhub/`, not something to write off as
a platform quirk: the headless tests already claim the same behaviour.

## 7. What C must not break

The POSIX terminal, exactly: the frame contract (`input`/`resize`/`kill` ↔ `output`/`exit`), the per-`os.read`-chunk framing that keeps a `cat` of a big file from becoming one giant message, the incremental decoder that never splits a multi-byte character, the four-session cap, the three close codes and what each mirrors (*"4000-4999 … 403 policy, 429 capacity, 501 platform"* — C makes 4501 mean "no driver", which is still the 501 case), the origin policy in `termhub/policy.py`, and the rule that a terminal session is a process tree, never a wrapper.

**AGENTS.md** (done by csl-09): `core/termhub/`'s row now reads as *"one shell
per WebSocket, on the platform's pseudo-terminal (a PTY, or ConPTY)"* and names
which shell (`TERMINAL_SHELL`, else Git Bash, on Windows);
`core/proctree.py` and `core/proctree_jobs.py` share one row naming both
mechanisms behind the seam; and the process-tree bullet covers the terminal as
the second caller. The terminal stays *"a human surface (`core/termhub/`, one
WebSocket per session), never an agent tool"* — C did not change who may open a
shell, only whether a Windows user can open one. The install line and the two
process-environment variables went to the README, which AGENTS.md's own rules
require, and the file stayed at its pre-C length.
