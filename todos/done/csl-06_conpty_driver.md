# csl-06 — `termhub/pty_conpty.py`: the Windows pseudo-console driver

## Deliver this

The second implementation of the seam csl-03/csl-04 established: a driver that
gives the shell a **real Windows pseudo-console**, so the Git Bash csl-05 selects
has a terminal — prompt, line editing, colours, full-screen programs, resize.

The driver implements **the surface csl-03's *Results* note names**, nothing
wider: open-with-size, spawn the shell on it, read bytes, write bytes, set size,
get size, close, and **"is the far end gone?"**. Requirements:

* **Size is set when the pseudo-console is created** — C-D4: *"the size is passed
  when the pseudo-console is created (so 'set before spawn' maps exactly)"*.
  `resize()` re-issues it and `terminal_size()` reads the current size back, so
  the existing round-trip test can hold on Windows.
* **Read answers bytes**, in the driver's own chunk size but never larger than
  `constants.PTY_READ_BYTES`: the session's per-chunk framing and its incremental
  UTF-8 decoder stay the only place bytes become text (C §7).
* **The end-of-stream rule is re-stated, not inherited** — C-D3: *"A ConPTY pipe
  ends with **EOF (an empty read)**, not `EIO`, and there is no slave fd to
  close. … the seam exposes one predicate — 'is the far end gone?' — and `pump()`
  keeps its shape while the reason moves into the driver."* So the driver answers
  EOF as "gone", and a real error stays an error the session can distinguish.
  `STOP_GRACE_SECONDS` is *"unchanged: it is the same number and the same rule as
  both hubs'"* — do not add a Windows-only grace value.
* **Encoding is verified, not assumed** (C-D6). Run the non-ASCII echo round-trip
  on Windows. If the driver cannot hand back the same bytes, set the child's
  console/`PYTHONIOENCODING`-equivalent to UTF-8 rather than letting mojibake
  reach xterm — and record which of the two you needed. *"This is listed as an
  unknown to resolve on the Windows machine, not a settled design."*
* The driver spawns the shell with the spec and environment csl-05 produced, and
  leaves the teardown to the session + csl-02's job seam: **it must not implement
  its own kill** (csl-07 wires the job).
* Ctrl-C is the input event, not a byte the code special-cases: C-D5 — *"the seam
  must own the difference rather than have `write()` special-case byte 3"*.
  `write()` stays a plain byte writer. `^Z`/job stop/resume are **not promised**.
* The one honest platform difference is written in the driver's docstring rather
  than papered over: *"a POSIX program notifies itself on `SIGWINCH`, a Windows
  console program is reflowed by the console, and a program that reads its size
  once at startup behaves differently on the two."*

## Files it touches

| file | change |
|---|---|
| `src/core/termhub/pty_conpty.py` | **new** — the driver, its dependency import, the EOF predicate, the docstring stating the platform difference |
| `src/core/termhub/drivers.py` | edit — this driver becomes the answer the seam can select when its dependency is importable |
| `src/core/termhub/session.py` | edit only where the seam's surface needs an adapter; `spawn`/`pump`/`resize`/`teardown` keep their structure |
| `pyproject.toml` | already carries the extra from csl-04; no new dependency |
| `tests/test_terminal.py` + `tests/test_terminal_conpty.py` (new) | the Windows-only tests below, plus Linux-runnable seam tests |
| `web/` | **no change** — *"ConPTY emits VT, which is what xterm.js already consumes"* |

## Rules it must not break (quoted)

* C-D1's rejected alternatives stay rejected — this task implements the accepted
  one and nothing else:
  * *"Raw ConPTY via `ctypes` … re-implements a console-host handshake in a module
    whose failure mode is a hung terminal"*;
  * *"Plain pipes … no terminal at all: no `isatty`, no resize, no line discipline,
    no full-screen app. It would make the tab a log viewer"*;
  * *"Bridge into WSL … the shell's filesystem would no longer be
    `BENCHMARKIT_PATH_LOCATION`"* — the shell runs the Windows filesystem at the
    working location, always;
  * Git's bundled `winpty.dll`/`winpty-agent.exe` and `mintty.exe`/`git-bash.exe`
    are out of scope (big picture).
* The dependency comes from the **`windows` extra csl-04 added** and stays
  optional: with it absent, csl-04's refusal still happens, with its sentence.
  *"so a Linux install never resolves it (roadmap §9: the platform difference sits
  behind one named seam)."*
* The module **must import cleanly on Linux** and must not be selected there:
  the seam answers POSIX. A driver that cannot be imported on the Linux runner
  breaks the suite, and a driver imported but selected on Linux breaks the
  terminal that works today.
* C §7's must-not-break list applies unchanged: the frame contract, the
  per-chunk framing, the incremental decoder, the four-session cap, the three
  close codes, `termhub/policy.py`'s origin rule, and *"a terminal session is a
  process tree, never a wrapper"*.
* The pseudo-console helper process the driver starts is **spawned, tracked and
  reported** (its pid/pid-like handle must be reachable by csl-07's job). *"That
  second process is the one an implementer will forget."*
* AGENTS.md: *"nothing under `core` may import the API layer"*; blocking calls
  run in a worker thread the way `pump` already does.
* AGENTS.md: *"Tests stay hermetic"* — the driver's tunables live as patchable
  module attributes; no test depends on a specific Git install path.

## Depends on

csl-03 (the seam surface), csl-04 (the selection seam and the extra), csl-05 (the
shell spec and child env). csl-07 depends on this task's helper-process answer.

## How to tell it is done

1. **Linux, unchanged**: `pytest tests/test_terminal.py` — same tests, same
   results, **zero skips**, no new skip reason; `src/core/termhub/pty_conpty.py`
   imports without error on Linux (assert this in a test) and the seam does **not**
   select it there; with the driver's dependency hidden, the seam's answer is
   "none" and the 4501 refusal test still passes.
2. **Windows** (marked `windows_behaviour`, plus the driver actually installed):
   * `test_input_produces_output_and_shell_stays_open`,
     `test_exit_frame_reports_the_code`, `test_kill_frame_closes_session_and_kills_pid`
     and `test_session_cap_refuses_the_next_upgrade` pass against this driver,
     unmodified in what they assert;
   * the **two C-D3 tests** exist: (a) the shell exits on its own → an `exit`
     frame carrying the code; (b) a teardown requested → the pump stops within the
     grace period and no task/reader is left running;
   * the **resize round-trip**: a `resize` frame changes the size the driver
     reports back. Whether the *shell* also sees it is recorded honestly (see the
     hazard below), not asserted as parity;
   * the **C-D6 non-ASCII round-trip**: echo a string with accents/CJK and read it
     back identical.
3. `cd web && npm test` green; `git diff --stat web/` empty; `pytest
   tests/test_write_surface_isolation.py` green.
4. *Results* note (this task is the lot's main Windows measurement — be specific):
   the driver version used; the chunk size and whether the framing rule held; the
   exact EOF/error behaviour observed; the helper process' name and how its pid
   was obtained (csl-07 needs it); the encoding answer and whether an extra step
   was needed; whether `stty size`/`vim`/`less` behaved under the pseudo-console;
   anything that had to be done differently for a **bash** child than for a native
   console program.

## Hazards to resolve rather than guess

* **A pseudo-console helper that outlives the shell.** The driver must be able to
  name every process it started. If the dependency will not hand over a pid, say
  so in *Results* and make the session own the helper's lifetime explicitly — do
  not leave it to garbage collection.
* **Read blocking forever on a Windows pipe.** The POSIX pump relies on the
  teardown unblocking the read. Verify on Windows that a teardown actually
  unblocks it; if it does not, that is a finding for csl-07 (the job), not
  something to paper over with a timeout in `pump`.
* **A truncated stream that looks finished.** AGENTS.md's general rule (*"A stream
  cut by the server looks like a finished answer"*) applies here: an EOF is not
  proof the shell exited cleanly. Keep the exit code honest, and if the driver can
  report the child's status separately, report it.

## Results (landed 2026-11-02, on the Windows runner)

**Driver version.** `pywinpty 3.1.2` (the `winpty` package as installed from the
`windows` extra), **ConPTY backend** — the default: constructing `winpty.PTY`
with no `backend` selects ConPTY, and the helper process it starts is
`OpenConsole.exe` (`.venv\...\winpty\OpenConsole.exe`, child of the app process).
The older `winpty` backend (backend=1) was probed once for comparison and its
helper is `winpty-agent.exe`; the driver is not used with that backend, but
`CONSOLE_HELPER_NAMES` names both so the capture works if the default ever
changes.

**Chunk size and the framing rule.** The agent's `read(blocking=False)` answers
whole *strings* of its own choosing: measured max 16 384 characters per call
(a 252 000-byte `cat` arrives as 48 reads, alternating 16 384/16 383 chars).
The driver converts to UTF-8 bytes and cuts to `min(max_bytes, PTY_READ_BYTES)`
itself, buffering the remainder, so the session's framing rule held: no frame
ever carried more than 4 096 bytes' worth of a chunk, and a multi-byte character
cut by the cap was still joined by the session's incremental decoder.

**EOF/error behaviour observed.** `iseof()` flips a moment *after* the empty
read on an `exit` (measured: EOF seen within ~1.4 s of a tree kill; on a clean
`exit 3`, after 5 reads and 1 empty read); so the predicate is `iseof() or not
isalive()` plus the firm answer from a released console. A blocking read on a
**silent live** console does not time out at the default 30 s — it waits for
input — so the driver never enters pywinpty's blocking read at all: it polls the
non-blocking read every `READ_POLL_INTERVAL_SECONDS` (20 ms). A teardown's
`close_master()` (which calls `cancel_io()` then releases the native object)
ended a blocked read in **2–17 ms** (six trials). A real pipe failure while the
shell is alive raises `ConPtyError` (an `OSError` that is *not* `FarEndGone`) —
the session keeps the pump's single `except OSError` path, and the exit code
comes from `get_exitstatus()` separately: `exit 7` reported 7, a killed tree
reported 1, `exit 9` arrived within 0.4 s. A child gone with no status answers
`EXIT_CODE_UNKNOWN` (−1), never an invented 0.

**The helper process.** Name: `OpenConsole.exe` (ConPTY backend). Its pid is
**not** handed over by pywinpty — `PTY` exposes only the shell's pid — so
`ConPty.spawn_shell()` snapshots this process's children *before* creating the
console and again after, and claims the newly-appeared ones whose exe name is in
`CONSOLE_HELPER_NAMES` (name-matched so a hub's child born in the same instant
is not claimed, and a second session's helper — already in the pre-snapshot —
is not stolen). The helper is born with the **console creation**, before
`spawn()`, and lives as long as the PTY object is referenced. `started_pids()`
returns shell-then-helper; the `windows_behaviour` test
`test_the_console_helper_is_named_as_a_live_process_of_this_app` proves both
pids are real and live. Measured: while the app holds the PTY, a *self-exited*
shell leaves the helper alive; releasing the PTY (what `close_master` does) took
helper, shell **and** a backgrounded `sleep 40 &` child with it (they are in the
console's own process group — the driver does **not** kill: `close_master` makes
no kill call, and csl-07's job is the promise). One caution for csl-07: that
coverage is the dependency's behaviour, not this driver's — two probes released
the PTY and disagreed about what survived it (one: shell and helper alive 1.5 s
after a plain reference-drop with no `cancel_io()`; the other, the same drop:
both gone within 2 s). The release is not a dependable kill either way, which is
exactly why the job is the guarantee; the driver reports every pid it started so
the job can hold them.

**Encoding (C-D6).** The first option: **no extra step was needed**. The
round-trip `echo café中文 ☃` came back byte-identical through the real
pseudo-console (accents, CJK, snowman; also Greek `π`), and `printf '\303\251' |
wc -c` under the session answered 2 — Git Bash's locale is `C.UTF-8` and ConPTY
carries UTF-16 internally with the agent framing its pipe as UTF-8. The test
`test_a_non_ascii_echo_comes_back_as_the_same_bytes` pins it. The driver converts
symmetrically at the boundary (`utf-8`/`surrogateescape` both ways). No
`chcp`/code-page call was added. (A non-UTF-8 byte a child writes is repaired by
the console itself — a lone `\xff` arrives as U+FFFD — which the session's
`errors="replace"` decoder matches.)

**`stty size` / `vim` / `less` under the pseudo-console.** `tty` answers
`/dev/cons0` (isatty true). `stty size` prints the live size after a resize
(observed `41 132` and `30 100`) — so this shell **does** see the resize the way
the server set it. `vim` and `less` both entered the alternate screen
(`ESC[?1049h`), painted, and took their input (`q` quit `less`, `:q` quit `vim`).
Ctrl-C as byte 3 through `write()` with nothing special-cased: at the prompt it
aborts the half-typed line (the `^C` is echoed and the line is discarded), and
after a foreground `sleep 30` it brought the prompt back with the next command
accepted — the interrupt works through the console's input event.

**One bash-specific hazard found (finding for the pane/csl-08, not a defect of
the seam).** On a resize the ConPTY agent emits a device-attribute query
(`ESC[6n`) and **eats keystrokes until the terminal answers it** (observed:
`stty size` arriving as `tty size`). Three measurements, no mechanism claimed
beyond them: input written in the **same tick** as the resize was garbled 4
times in 5 through the direct session API; answering the query once it has
appeared (writing `ESC[1;1R` back, what xterm.js does for its pane) made the
next input clean, while answering before it appeared did not; and the suite's
`test_resize_updates_the_window_size` — resize and input as two separate
WebSocket frames, no responder in the test client — passed three consecutive
runs on this machine. A finding for csl-08's gate to keep honest about, not a
seam defect. A program
reading its size once at startup (e.g. `COLUMNS`/`LINES` set at spawn) still
keeps its old number — C-D4's stated platform difference, confirmed: bash's
`stty size` re-reads, the env pair does not.

**Test evidence (this runner, driver installed).**
* `pytest tests/test_terminal_conpty.py` — 42 passed: the seam half (planted
  `winpty`, import-without-dependency by reload with the import blocked, POSIX
  selection never displaced) **and** the real-console half
  (`windows_behaviour` + the driver probe): C-D3 case 1 (`exit 7` → exit frame
  code 7), Ctrl-D → exit frame code 0, C-D3 case 2 (teardown ends the pump well
  inside the 5 s grace — the blocked read returned 2–17 ms after `close_master`
  and the task resolved), the resize round-trip through
  `session.terminal_size()`, the helper naming, C-D6's round-trip.
* `pytest tests/test_terminal.py` — 11 passed / 5 skipped; the 5 skips are the
  **pre-existing** `no shell at /bin/sh` probe skips (the conftest gate csl-08
  retargets). With `SHELL` pointed at a Git Bash so the old probe answers — run
  scratch, deleted — all five of `TestSession` pass unmodified against this
  driver: `test_input_produces_output_and_shell_stays_open`,
  `test_exit_frame_reports_the_code`, `test_resize_updates_the_window_size`,
  `test_kill_frame_closes_session_and_kills_pid`,
  `test_disconnect_tears_down_the_whole_session_tree` (three consecutive clean
  runs).
* `pytest tests/test_terminal.py tests/test_terminal_conpty.py
  tests/test_terminal_shells.py tests/test_write_surface_isolation.py` — 89
  passed, only the 5 pre-existing skips.
* `cd web && npm test` — 472 passed; `git diff --stat web/` empty.

**No session adapter was needed** beyond what csl-04 pre-arranged:
`session.spawn()`'s `pair_starts_the_child` branch, `close_slave()` as the named
no-op and `drivers.TerminalChild` (`pid`/`wait`) are exactly the surface the
driver implements; `session.py` is unchanged by this task.
