# Target C — the Editor's terminal on Windows, run over Git Bash

Make the Editor tab's terminal open a **real interactive shell on Windows** with
every guarantee the POSIX path already makes — a session is a *process tree*, it
is capped, the browser resizes it, and it dies with the app — and make the
absence of any of that **a sentence, not a degraded pane**.

The authority for this lot is `docs/windows-support/windows-support-target-C-terminal.md`
(C-D1 … C-D6, the rejected alternatives, §7 "What C must not break"). Read it
before your task: every task file below cites the decision numbers it implements.
Wider context: `docs/windows-support/windows-support-road_to_target.md` (§5 process
trees, §6 the terminal as the honesty model, §9 the three rules) and
`docs/windows-support/windows-support-target-B1-console-gbash.md` (G-D3, G-D4 — the
Git toolset the console connector already runs on this machine). This file is the
map and the cross-task contract; it does not restate those docs.

## The one amendment this lot makes to the C plan

C-D2 said: Windows shell order is `$TERMINAL_SHELL` → `%COMSPEC%` (`cmd.exe`) →
refuse, and explicitly refused to auto-pick a shell. **That is amended by the
operator (2026-11-02): the shell is Git Bash.**

* Windows order: `TERMINAL_SHELL` if set and drive-qualified and executable →
  **Git Bash**, found through the discovery probe the console connector already
  proved (G-D4) → **refuse with a sentence naming what was looked for.**
  `cmd.exe`/`%COMSPEC%` is **not** a fallback, and `pwsh`/`powershell` are still
  never auto-selected.
* Why: the app's own console story on Windows is already Git's toolset (B1
  landed). A `cmd.exe` pane would give one working location two different tool
  vocabularies, and every existing terminal test (`echo`, `exit 7`, `sleep 40 &`,
  `stty size`) transfers to a bash session instead of forking into a
  cmd-specific suite — roadmap §9 rule 2's preferred shape.
* What the amendment does **not** buy: it does not remove the pseudo-terminal,
  the driver dependency, the EOF-instead-of-`EIO` rule, or the job-object
  teardown. A pipe pair would give bash no tty — no prompt, no line editing, no
  full-screen program — which is exactly the "log viewer" C-D1 rejects.

`TERMINAL_SHELL` is **process-environment only**, never a managed `.env` key —
the same reasoning `BENCHMARKIT_GIT_BASH_DIR` and `BENCHMARKIT_PATH_LOCATION`
carry: a settings save must not erase it, and the managed template's key set is
not widened for a platform detail.

## Verified on this machine (2026-11-02, Git at `C:\Program Files\Git`)

Measured, not assumed. If your task's premise disagrees with an observation, the
observation wins and you record it.

* `bin\bash.exe` and `usr\bin\bash.exe`/`sh.exe` exist. `usr\bin` carries
  `stty.exe`, `tty.exe`, `tput.exe`, `which.exe`, `uname.exe`, `clear.exe`,
  `vim.exe`, `less.exe`, `sleep.exe`, `env.exe` — the tools the existing terminal
  tests and C-D6's non-ASCII echo and C's manual list (`vim`, `less`) depend on.
* `usr\share\terminfo` ships a real terminfo database including `xterm-256color`
  (and an `xterm.js` entry), so `TERM_VALUE` is resolvable by the child instead of
  being a hopeful string.
* The Git tool half lives in `ucrt64\bin` on this install; `mingw64\bin` does not
  exist here. A search path that names a missing directory is harmless, but no
  task may *assume* that directory exists.
* `os.path.isabs("/bin/bash")` is **False** on Windows, and
  `os.access("/bin/bash", os.X_OK)` is False: a POSIX-shaped shell spec inherited
  from a Git Bash launch is already rejected today. That is luck, not a rule —
  Windows shell acceptance must be stated as *drive-qualified* (via
  `path_norm.is_qualified`) and pinned in a test, so it stops being an accident.
* `SHELL` is unset in this environment; `COMSPEC` points at
  `C:\WINDOWS\system32\cmd.exe`. Today `tests/conftest.py`'s
  `SHELL = $SHELL or "/bin/sh"` therefore makes `shell_probe` answer "no shell",
  which is why `TestSession` skips on Windows. csl-08 changes what that probe
  asks.
* `pywinpty` is **not installed** here, and `winpty.dll` / `winpty-agent.exe` in
  Git's `usr\bin` are Git's own pre-ConPTY shim, versioned by the user's Git.
* The console connector **refuses a command named `bash` by design** (G-D3's
  no-shell sentence). The terminal spawning bash does not change that: the
  console tool runs no shell, the pane is a human surface.

Two facts this lot inherits from the earlier B1 session and must **re-verify
before relying on** (record the answer in the task file that measured it):

1. A Git Bash child given only the ambient Windows `PATH` answers "command not
   found" for `stty`/`tty`/`which`; with Git's own directories prepended it
   resolves them. So the driver hands the child the same search order G-D4 proved.
2. Whether the child sees a **resize** the way a POSIX shell does (via
   `SIGWINCH`) is a driver-behaviour question, not a design promise. C-D4's rule
   covers it: assert the server-side round-trip, and state the shell-visible
   difference honestly.

## The seam (the cross-task contract)

One small driver surface, two implementations, and a session class that keeps
every platform-neutral guarantee it has today (C-D1, C-D3, C-D4):

| name | lands in | shape |
|---|---|---|
| `mcp_connectors/gitbash.py` | csl-01 | the Git-install probe shared by the console connector and the terminal: `discover_git_install(candidates, env_override)`, `git_search_paths(root)`, `git_shell_program(root)`, `git_child_env(base, root)`. Stdlib-only, parameterised — no hidden global reads (G-D2's discipline) |
| `src/core/proctree_jobs.py` | csl-02 | Job Object teardown behind `proctree`'s **existing two entry points**, same escalation (graceful → bounded wait → force), same injected reap step; plus a liveness probe the tests may use |
| `src/core/termhub/pty_posix.py` | csl-03 | today's PTY moved out verbatim: `openpty`, the slave-close rule, `TIOCSWINSZ`/`TIOCGWINSZ`, and the `EIO` → "far end gone" translation |
| `src/core/termhub/drivers.py` | csl-04 | the named seam: *is a driver available?* — a capability answer, never `os.name`; the refusal sentence for "no driver" |
| `src/core/termhub/shells.py` | csl-05 | shell selection per platform, with the interactive flag belonging to each shell spec |
| `src/core/termhub/pty_conpty.py` | csl-06 | the Windows driver: pseudo-console opened **with** a size, byte-level read/write, resize, the EOF → "far end gone" answer |
| the job covers the session | csl-07 | the shell **and** the console helper the driver spawned are members of the one job |
| the test gate for C | csl-08 | markers, liveness, `$!`-based tree assertions, the driver-absent refusal |

The driver answers exactly: open-with-size, spawn, read, write, set-size,
get-size, close, and **"is the far end gone?"**. `read` answers *bytes* — never
already-decoded text — because `session.py`'s incremental UTF-8 decoder and the
per-chunk `PTY_READ_BYTES` framing must stay the single source of truth for both
platforms. `pump()` keeps its shape; only the *reason* the stream ends moves into
the driver.

**Behaviour-preservation rule for csl-03** (this lot's Linux promise): today an
`OSError` out of the read ends the pump and then reaps and emits the `exit`
frame. Whatever the seam looks like, that outcome stays identical on POSIX —
`EIO` becomes the driver's "far end gone" answer, an unexpected error keeps
today's path, and no Linux-visible frame sequence changes. If a refactor would
change it, the refactor is wrong.

## Task order

| # | task | doc ref | depends on |
|---|---|---|---|
| csl-01 | Extract the Git-install probe into `mcp_connectors/gitbash.py`; the console connector delegates to it | G-D4, C-D2 (amended) | — |
| csl-02 | `proctree_jobs.py`: Job Object teardown behind `proctree`'s entry points | roadmap §5, C-D5 | — |
| csl-03 | `pty_posix.py`: the PTY moves out of `session.py`, behaviour-identically | C-D1, C-D3 | — |
| csl-04 | The driver seam + the honest refusal + the `windows` extra in `pyproject.toml` | C-D1, C-D3's fallback, roadmap §9 rule 3 | csl-03 |
| csl-05 | `shells.py`: Windows shell selection (Git Bash, no `cmd.exe`) and the child's environment | C-D2 as amended | csl-01, csl-04 |
| csl-06 | `pty_conpty.py`: the pseudo-console driver | C-D1, C-D3, C-D4, C-D6 | csl-03, csl-04, csl-05 |
| csl-07 | One job for the whole session: shell, console helper, and everything they spawn | C-D5 | csl-02, csl-06 |
| csl-08 | The test gate: probes that answer *this* session, `$!` tree assertions, the driver-absent refusal | §6, roadmap §9 | csl-04 … csl-07 |
| csl-09 | Docs: the AGENTS.md clause, the README Windows lines, the C/B1/roadmap status + pointers, the manual checklist | §6 "manual", §7 AGENTS.md | csl-01 … csl-08 |

01, 02 and 03 are independent and may run in any order relative to each other.
**09 is last**: it documents sentences that only exist once 01-08 land, and
AGENTS.md is injected into every WorkBench turn — wrong there is worse than missing.

## Out of scope for this lot (do not add it)

* **Lifting the LSP hub's Windows refusal** (`local_reach_is_bounded()`), which
  B1 deferred until `proctree_jobs.py` landed. csl-02 lands that module, but the
  hub half is **a separate lot with its own real-language-server verification**.
  Leave the refusal and its tests exactly as they are.
* Playwright e2e for the terminal pane (C §4's "maybe" row): **deferred** — it
  would need a Windows runner for a real shell, so it cannot be a Linux-safe
  task here. Record it as deferred in csl-09.
* `cmd.exe` as a shell, a shell picker in the UI, more than one shell flavour.
* Job control promises: `^Z`, `fg`/`bg`, `SIGTSTP` parity (C §5). Ctrl-C/Ctrl-L/
  history come from bash; nothing extra is claimed.
* Scrollback parity, mouse reporting, OSC 52 clipboard, `TERM`-database
  extensions (C §5).
* A managed `.env` key for the shell, a `.env.example` change, a settings-panel
  setting, or **any frontend change** — the frame contract is why C needs none.
* Git's own `winpty.dll`/`winpty-agent.exe`, `mintty.exe`/`git-bash.exe`, raw
  ConPTY through `ctypes`, WSL, plain pipes: all rejected (C-D1's table).
* A Windows CI runner or installer.

## Project rules that bind these tasks (from `AGENTS.md`)

* **`core/termhub/` — the Editor's terminal, one shell session per WebSocket.**
  And: the terminal is *"a human surface (`core/termhub/`, one WebSocket per
  session), never an agent tool"*. C changes *whether a Windows user can open a
  shell*, never who may.
* **Tools are read-only by design**; *"no connector, tool or chat path reaches a
  write route, and the terminal is never a tool"* —
  `tests/test_write_surface_isolation.py` asserts it. Nothing in this lot may
  make a shell reachable from chat, a hub or a tool, and **the console
  connector's refusal of a command named `bash` stays exactly as written**.
* **Teardown kills process *trees*, not wrappers**, so children a hub spawned of
  its own never outlive the app that started them. csl-02 and csl-07 are that
  sentence becoming true on Windows.
* **Core owns logic, the API owns transport**: nothing under `core` may import
  the API layer; blocking work runs in a worker thread.
* **Tests stay hermetic**: a test patches a seam's own constants rather than a
  path it computed, and a backend test's workspace is a throw-away `tmp_path`
  pointed at by `BENCHMARKIT_PATH_LOCATION`, never this checkout. Git discovery,
  the driver seam and the shell order are all patchable module attributes.
* **`.env` is regenerated from a fixed template on every save** — *"a new or
  renamed key means changing the template, the renderer and the committed example
  together"*. This lot deliberately adds **no** managed key.
* **Every path the app holds is relative to the working location**, so the
  terminal's `cwd` stays `path_location()` and the server takes no path from the
  client.
* **No lint step.** `pytest` for Python, `cd web && npm test` for Vitest,
  `cd web && npm run test:e2e` for Playwright. Run tests with the project venv
  (see README for this machine's invocation).
* Read a file with the read tool before editing it, dry-run the edit, and keep
  the text after the line number verbatim — including indentation.

## The Linux promise (the reason this lot is ordered as it is)

The POSIX terminal works today and is the reference. Every task keeps it working
at its own finish line, and the cadence C §6 sets is the acceptance baseline:

* `pytest` green with **zero marked skips on Linux** for the touched files — if
  moving code into `pty_posix.py` costs a test its reach, the move is wrong.
  > DO NOT RUN ALL TEST SUITE. RUN ONLY TESTS TOUCHING THE AREA OF THE FEATURE.
* `cd web && npm test` green (no frontend change in this lot, so this is a
  regression check, not a feature check).
* On a Windows runner the same files are green with only the **pre-existing**
  skips (the tool probes and the symlink probe). A *new* skip on Windows is a
  finding to report, not a thing to hide behind.
* Windows-only assertions use the existing `windows_behaviour` marker; a Linux
  runner must prove the Windows half **by forcing the seam**, never by pretending
  to be Windows (roadmap §9 rule 2).
* One trap to name out loud: **`os.kill(pid, 0)` on Windows terminates the
  process** (CPython maps a plain signal number to `TerminateProcess`). No task
  may use it as a liveness probe. Liveness comes from csl-02's platform half.

## Definition of done for the lot

1. The terminal spawns **Git Bash** on Windows, in the working location, over a
   pseudo-console, and the pane behaves as a terminal: prompt, echo, exit code,
   resize, Ctrl-C.
2. `tests/test_terminal.py`'s assertions run on **both** platforms against
   whichever driver the platform provides, with the same expectations — the suite
   did not fork into a POSIX-only shape.
3. A session is a process tree on Windows too: a backgrounded child and the
   console helper are both gone after a teardown, asserted, not promised (C-D5).
4. With no driver installed, Windows answers an `error` frame naming the missing
   dependency and close code **4501** — the same refusal shape, and it is tested
   on Linux by hiding the driver.
5. Encoding is **verified, not assumed** (C-D6): a non-ASCII round-trip on
   Windows, and if the driver cannot guarantee UTF-8 the session fixes the
   child's encoding rather than shipping mojibake.
6. `AGENTS.md` names the terminal's second mechanism and `proctree`'s second
   implementation; the README says how to install the extra; the C, B1 and
   roadmap docs carry their new status and pointers.
7. The manual Windows checklist (C §6's six items) is written down as a
   human-only step and **not** attempted by any task in this lot.
