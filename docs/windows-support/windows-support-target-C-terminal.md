# Target C — the Editor's terminal on Windows (ConPTY)

**Status**: planned, not started. **Depends on**: target A (path semantics, the test gate) and **B-D5 specifically** (the process-tree teardown a session needs). It can start as soon as B-D5 lands — the rest of B is independent of C.
**Read first**: roadmap §5 and §6. **Code referenced**: `ef21ea3` + A/B changes.

## 1. What C is for

C makes the Editor tab's terminal run a real interactive shell on Windows, with the same guarantees the POSIX path already makes: a session is a *process tree*, it is capped, it is resized by the browser, and it dies with the app.

AGENTS.md describes what is being replaced:

> `core/termhub/` — Editor's terminal: one PTY shell per WebSocket, session registry on `app.state.terminal`

"PTY" is the POSIX-only word in that sentence, and `termhub` is the only place in the roadmap where the gap is *already handled well*: the module imports `pty`/`fcntl`/`termios` behind an `ImportError → None` guard (`session.py:43-53`) purely so it imports on Windows, `spawn()` raises `TerminalUnsupported` (`:137-140`), the route answers an `error` frame then `CLOSE_CODE_UNSUPPORTED` = 4501 (`api/routes.py:1655-1664`, and the close-code block in `termhub/constants.py`), and `useTerminal.js` renders the sentence. **C's job is to remove that refusal, not to change its shape.**

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
| Bridge into WSL (`wsl.exe`) so the existing POSIX code runs | needs WSL installed, and — decisively — the shell's filesystem would no longer be `BENCHMARKIT_PATH_LOCATION`. AGENTS.md: *"The connectors answer level B for their own root"* and *"every path the app holds is relative to that location"*; a terminal browsing a WSL mount while `fedit` edits the Windows tree splits the one working location the whole app is built around |

`pywinpty` goes in a **new optional extra**, not in `[mcp]`:

```toml
windows = ["pywinpty>=2.0; sys_platform == 'win32'"]
```

so a Linux install never resolves it (roadmap §9: the platform difference sits behind one named seam). C is the **only** target in the roadmap that adds a dependency, and this is the place it is stated.

### C-D2 — Which shell, and what "interactive" means there

`shell_command()` (`session.py:82-92`) reads `$SHELL`, requires `os.path.isabs` and `os.access(..., os.X_OK)`, and falls back to `constants.FALLBACK_SHELL = "/bin/sh"`. On Windows there is no `$SHELL`, the fallback does not exist, and the interactive flag is the bash-specific `-i` (`:148` — `[self.shell, "-i"]`).

**Decision**: the seam also owns shell selection, so the fallback lives with the platform:

* Windows order: a configured `TERMINAL_SHELL` if set and executable → `%COMSPEC%` (`cmd.exe`) → refuse with the existing `TerminalUnsupported` sentence naming what was looked for. `pwsh`/`powershell` are *not* auto-selected: the app should not decide a user's shell, and PowerShell's startup cost breaks the "prompt appears immediately" expectation.
* No `-i`: `cmd.exe` is interactive by default when it has a console, and ConPTY gives it one. `constants.py:44-46`'s comment (interactive-non-login so `~/.bashrc` is read but not `~/.profile`) stays true as the *reason*; the flag itself is a POSIX detail.
* `TERM`/`TERM_VALUE`: set for the child as today. ConPTY does not need `TERM` to work, but a program that reads it should still see `xterm-256color` rather than nothing.

This keeps the app's one knob honest: the shell is the user's choice, expressed the same way on both platforms.

### C-D3 — The pump's end-of-stream rule differs, and must be re-stated

`pump()` (`:216-257`) blocks in `os.read(master_fd, PTY_READ_BYTES)` on a thread and relies on a specific fact: *"it is unblocked by the teardown, because once the shell and its children are gone the master answers `EIO`"* (`:221-223`), with the matching comment at `:238` (*"EIO once every holder of the slave is gone"*). The whole shutdown sequence — `_finish_pump` (`:259`), `teardown` (`:362`), and `spawn`'s careful closing of the parent's copy of the slave (`:126-129`: *"while any process holds the slave open, a read of the master never reports the shell's death"*) — is built on that.

A ConPTY pipe ends with **EOF (an empty read)**, not `EIO`, and there is no slave fd to close. **Decision**: the seam exposes one predicate — *"is the far end gone?"* — and `pump()` keeps its shape while the reason moves into the driver. Two tests pin the behaviour on both platforms:

1. shell exits on its own (`exit`, Ctrl-D) → an `exit` frame with the code;
2. teardown requested → the pump stops within the grace period, no leaked task.

`STOP_GRACE_SECONDS = 5.0` (`constants.py:24`) is unchanged: it is *"the same number and the same rule as both hubs'"*, and C must keep that sentence true.

### C-D4 — Resize has a different delivery mechanism; the contract does not change

POSIX: `_set_window_size` (`:175-180`) `TIOCSWINSZ`s the master *before* spawn so the first prompt wraps correctly, `resize()` (`:189-194`) re-issues it and the kernel sends `SIGWINCH`, `terminal_size()` (`:196-214`) reads it back. The client side is already platform-neutral: xterm.js + `FitAddon`, a debounced 50 ms `pushResize` (`useTerminal.js:216-236`), and the guard for a hidden host yielding 0 rows (`:225-229`).

ConPTY: the size is passed when the pseudo-console is created (so "set before spawn" maps exactly), and `ResizePseudoConsole` reflows the buffer. There is no `SIGWINCH`; a program learns about the new size by the console reporting it.

**Decision**: the seam's `set_size`/`get_size` cover both; `session.py`'s three call sites do not change, and the client needs **no change at all** — ConPTY emits VT, which is what xterm.js already consumes. State the one honest difference in the driver's docstring rather than papering over it: a POSIX program notifies itself on `SIGWINCH`, a Windows console program is reflowed by the console, and a program that reads its size once at startup behaves differently on the two. That is a fact about the platforms, not a defect to hide.

### C-D5 — Teardown is B-D5's job object, extended to the console host

`teardown()` (`:362-397`) calls `proctree.kill_session_tree` guarded by `os.name == "posix"` (`:386`), and the module docstring explains why session-scope matters: a job-controlling shell puts every job in its own group, so a `killpg` would reach the shell alone and a backgrounded `python -m http.server` would outlive the tab (`proctree.py:18-27`).

**Decision**: after B-D5, the terminal asks the *same* shared teardown, and the job it creates must contain **both** the shell and the `OpenConsole.exe`/conhost process `pywinpty` starts for it. That second process is the one an implementer will forget, and it is precisely the *"kill process trees, not wrappers"* case AGENTS.md names.

Ctrl-C is a separate mechanism, not a signal: a PTY's line discipline turns `^C` into `SIGINT`; ConPTY turns the input event into a `CTRL_C_EVENT` for the console group. Same keystroke, same effect, and the seam must own the difference rather than have `write()` special-case byte 3 — `session.py:182-187` stays a plain byte writer. `^Z` and job stop/resume are **not** promised on Windows.

### C-D6 — Encoding: verify, do not assume

The pump decodes incrementally as UTF-8 with `errors="replace"` (`:114`, `:244`). ConPTY's internal buffer is UTF-16 and the driver hands the app bytes, but a `cmd.exe` child writing under a legacy OEM code page can produce output that is not UTF-8 — and a Chinese or accented path in a directory listing is the realistic case. **Decision**: a test that echoes a non-ASCII string and reads it back identically, run on Windows. If the driver cannot guarantee UTF-8, the session sets the console to UTF-8 for the child (the `chcp 65001`-equivalent API call) rather than letting mojibake reach xterm. This is listed as an *unknown to resolve on the Windows machine*, not a settled design, because it depends on the driver's behaviour — exactly the kind of claim this roadmap has so far refused to make from the sofa.

## 3. What it builds on

* B-D5's `proctree_jobs.py`. C is the second caller of that seam; a terminal session with no job is a leak, and B is what makes "the same teardown" mean something on both platforms.
* `termhub/constants.py` as the patch-point convention (*"a test that exercises a cap or a grace period patches it here"*) — C adds no new module-level constants beyond the driver's own file.
* The refusal path as the fallback that stays: if `pywinpty` is absent on Windows, or the pseudo-console cannot be created, `TerminalUnsupported` and close code 4501 still happen, with a sentence naming the missing dependency. A's test for that refusal (roadmap §9 rule 3) is not deleted by C — it becomes the test for "driver not installed".
* `useTerminal.js`'s existing rendering of an `error` frame, and the whole xterm.js stack: **no frontend change in C.** That is the payoff of the frame contract (`input`/`resize`/`kill` out, `output`/`exit` in, `useTerminal.js:19`).

## 4. Shape and files of the pieces

| piece | change | what it holds |
|---|---|---|
| `src/core/termhub/pty_posix.py` | **new** | the code moved out of `session.py`: `openpty`, the slave-close rule, `TIOCSWINSZ`/`TIOCGWINSZ`, the `EIO` predicate |
| `src/core/termhub/pty_conpty.py` | **new** | the `pywinpty` driver: create with size, read/write, resize, the EOF predicate, job assignment of shell **and** conhost, shell selection (C-D2) |
| `src/core/termhub/session.py` | edit | keeps its structure; the `os.name != "posix"` guard at `:137` becomes "is a driver available?"; the `pty`/`fcntl`/`termios` import at `:43-53` moves into `pty_posix.py` |
| `src/api/routes.py` | edit | the guard at `:1655-1664` becomes "no driver available" rather than `os.name != "posix"`, keeping `CLOSE_CODE_UNSUPPORTED` |
| `pyproject.toml` | edit | the `windows` extra with the `sys_platform == 'win32'` marker |
| `web/e2e/` | maybe | one spec that types a command in the terminal and asserts its output appears — the first e2e of the terminal; skipped with a reason where no shell is available (the `needs_shell` pattern, `tests/test_terminal.py:27-29`) |

## 5. Out of scope for C

* Job control: `^Z`, `fg`/`bg`, `SIGTSTP`. `cmd.exe` has no job-control model to expose, and inventing one in the pane would be a feature, not support.
* More than one shell flavour, shell-detection UX, a shell picker in the Editor UI.
* Scrollback parity, mouse reporting, OSC 52 clipboard. The pane is already xterm.js, so whatever the browser does, it does — C promises no more.
* Any claim about `TERM`-database behaviour for programs that assume extensions a real terminal emulator provides.
* The rest of B (connectors, command sets, guardrails): independent of C.

## 6. Verification

**Unit, both platforms** (`tests/test_terminal.py` — it already has the seam-aware `needs_shell` marker at `:27-29`, which generalises the same way A's markers do):

* spawn/pump/write/resize/teardown run against whichever driver the platform provides, and the same assertions hold: the first output arrives at the right width (C-D4's "size before spawn"), `terminal_size()` round-trips a `resize`, a self-exiting shell produces an `exit` frame with a code, a teardown ends the pump inside `STOP_GRACE_SECONDS`, and the session cap refuses the 5th WebSocket with 4429.
* the absent-driver case: `TerminalUnsupported` + 4501 + the naming sentence — forced by hiding the driver module, so it is tested on Linux too.

**Windows-only**: C-D3's two end-of-stream tests; C-D6's non-ASCII round-trip; and the teardown test that a shell's *child* and the conhost process are both gone after `teardown()` — the point of C-D5, and the reason it cannot be asserted on Linux.

**Linux regression, every session** (the cadence B already set): `pytest` green with **zero** skips, `cd web && npm test` green, `cd web && npm run test:e2e` where `.env.test` reaches an endpoint. The POSIX terminal tests must not gain a single skip: if moving code into `pty_posix.py` costs a test its reach, the move is wrong.

**Manual, on Windows** — the list that separates "it spawns" from "it is a terminal":

1. `vim` a file, resize the pane, quit — no artefacts, correct width.
2. A full-screen program (`less`, `vim`) paints correctly after a resize mid-draw.
3. A backgrounded child: start `python -m http.server`, close the tab, check Task Manager for the conhost *and* the python. This is the ghost `proctree.py:20-27` was written for.
4. Ctrl-C a running command; Ctrl-L clears; arrow-key history works.
5. Resize the browser window and confirm no 0-row state after re-showing the tab (the `useTerminal.js:225-229` guard).
6. Kill the app from the server console and confirm no shell survives.

## 7. What C must not break

The POSIX terminal, exactly: the frame contract (`input`/`resize`/`kill` ↔ `output`/`exit`), the per-`os.read`-chunk framing that keeps a `cat` of a big file from becoming one giant message, the incremental decoder that never splits a multi-byte character, the four-session cap, the three close codes and what each mirrors (*"4000-4999 … 403 policy, 429 capacity, 501 platform"* — C makes 4501 mean "no driver", which is still the 501 case), the origin policy in `termhub/policy.py`, and the rule that a terminal session is a process tree, never a wrapper.

**AGENTS.md**: `core/termhub/`'s row becomes *"one shell per WebSocket, on the platform's pseudo-terminal (a PTY, or ConPTY)"*, and the process-tree bullet already rewritten by B now covers the terminal as its second caller. The terminal stays *"a human surface (`core/termhub/`, one WebSocket per session), never an agent tool"* — C does not change who may open a shell, only whether a Windows user can open one.
