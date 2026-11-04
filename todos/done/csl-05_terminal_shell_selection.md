# csl-05 — `termhub/shells.py`: which shell, and what the child's environment says

## Deliver this

Shell selection becomes a platform-owned decision in one new module,
`src/core/termhub/shells.py`, implementing **C-D2 as amended by this lot**:

* **POSIX**: today's behaviour verbatim — `constants.SHELL_KEY` (`$SHELL`) if it
  names an absolute, executable program, else `constants.FALLBACK_SHELL`
  (`/bin/sh`), with the interactive flag today's spawn uses. *(That is the
  behaviour Linux must keep; do not "improve" it.)*
* **Windows**:
  1. `TERMINAL_SHELL` from the process environment — accepted only if it is
     **drive-qualified** (via `mcp_connectors/path_norm.is_qualified`) and the
     file is executable;
  2. else **Git Bash**, through the probe csl-01 landed
     (`mcp_connectors/gitbash.discover_git_install` + `git_shell_program`);
  3. else **refuse** — `TerminalUnsupported` with a sentence naming what was
     looked for (`TERMINAL_SHELL`, the Git install candidates,
     `BENCHMARKIT_GIT_BASH_DIR`), in the win-07 / G-D1 refusal shape.
* `%COMSPEC%`/`cmd.exe` is **not** consulted as a shell, and `pwsh`/`powershell`
  are never auto-selected. This is the operator's decision for this lot and it
  **overrides C-D2's cmd-first wording** — record that in the module's docstring
  and in your *Results* note, and do not silently re-add a cmd fallback.
* The child's environment is part of this module's answer: the shell gets `TERM`
  (`constants.TERM_VALUE`), `COLUMNS`/`LINES` as today, **and** — for the Git
  Bash case — `PATH` with Git's directories prepended in Windows form
  (`gitbash.git_child_env`), because a bash started with only the ambient Windows
  `PATH` cannot find `stty`, `tput` or `which` (csl-01 measured this; read its
  *Results* note before wiring it).
* Each shell spec carries its own interactive flag, so `session.py` stops
  hard-coding the bash-specific one. POSIX keeps `-i` and the reason
  (*"interactive-non-login so `~/.bashrc` is read but not `~/.profile`"*); the
  Windows spec says what bash needs there and why.

## Files it touches

| file | change |
|---|---|
| `src/core/termhub/shells.py` | **new** — the selection order, the qualification rule, the child env, the refusal sentence, the interactive flag per shell |
| `src/core/termhub/session.py` | edit — `shell_command()` moves out; `spawn()` asks the seam for the shell spec and the env |
| `src/core/termhub/constants.py` | the existing keys stay; add the Windows shell env-var name (`TERMINAL_SHELL`) as a constant. **No managed `.env` key** |
| `src/core/termhub/manager.py` | untouched |
| `tests/test_terminal.py` (or a new `tests/test_terminal_shells.py`) | **new tests** for the selection order, forced through the seam |
| `mcp_connectors/gitbash.py` | read-only here (it is csl-01's); use it, do not fork it |

## Rules it must not break (quoted)

* C-D2's shape stays: *"the seam also owns shell selection, so the fallback lives
  with the platform"*, and *"This keeps the app's one knob honest: the shell is
  the user's choice, expressed the same way on both platforms."* The amended
  order is the operator's; the *honesty* requirement is C-D2's and unchanged.
* C-D2's rejection stands: `pwsh`/`powershell` are *"not auto-selected: the app
  should not decide a user's shell"*, and `cmd.exe` is now refused for the
  bigger reason — one working location, one tool vocabulary (this lot's
  amendment).
* AGENTS.md / G-D3: the console connector's refusal of a command named `bash`
  (*"this connector runs no shell; Git Bash supplies the tools, not an
  interpreter"*) **stays untouched**. The pane is a human surface; the tool is
  not. `tests/test_console_gbash.py`'s shell-refusal tests must stay green and
  unedited.
* AGENTS.md: *"One setting defines the working location … Every path the app
  holds is relative to it."* `cwd` remains `path_location()` and the server still
  takes no path from the client.
* AGENTS.md: *"`.env` is regenerated from a fixed template on every save — fixed
  key set and order … A new or renamed key means changing the template, the
  renderer and the committed example together."* This task deliberately does
  **not** add a managed key: `TERMINAL_SHELL` and `BENCHMARKIT_GIT_BASH_DIR` are
  process-environment only, the same reasoning as `BENCHMARKIT_PATH_LOCATION`
  (*"a settings save can neither erase it nor resurrect a stale value"*).
* Windows path acceptance: a POSIX-shaped spec (`/bin/bash`, `/c/…`) is **not** a
  Windows shell. Today `os.path.isabs("/bin/bash")` happens to answer False on
  Windows; make that an explicit rule through `path_norm.is_qualified` and pin it
  in a test, so a change of check cannot ever let a MSYS-mount path be spawned as
  a Windows child.
* AGENTS.md: *"The connectors answer the workspace level for their own root … the
  app and the connector processes share one parser, one matcher and one refusal
  sentence."* Import the shared probe (`mcp_connectors/gitbash.py`); `src` may
  import `mcp_connectors` (the existing precedent: `editorfs/fscase.py` re-exports
  `mcp_connectors/fscase.py`). Never copy its logic.
* AGENTS.md: *"Tests stay hermetic"* — every candidate list, the override name and
  the driver/shell seam are patchable module attributes; a test plants a fake Git
  tree in `tmp_path` rather than depending on the machine's Git.

## Depends on

csl-01 (the shared probe) and csl-04 (the seam the selection plugs into).

## How to tell it is done

1. `pytest tests/test_terminal.py` on Linux: **unchanged results, zero skips** —
   `open` still reports the same `cwd` and a POSIX shell, and every existing
   assertion about them passes.
2. New tests, all forcing the seam so they run **on Linux**:
   * with a planted fake Git tree (the `usr\bin\ls.exe` verification), the Windows
     selection answers its `bin\bash.exe`, and the refusal never fires;
   * `TERMINAL_SHELL` wins when it is set and qualified+executable; a
     POSIX-shaped `/bin/bash` spec is **refused by name**, not spawned;
   * with no `TERMINAL_SHELL` and no discoverable Git (candidates patched),
     selection raises `TerminalUnsupported` and the message names what was looked
     for — the sentence is asserted, per roadmap §9 rule 3;
   * `%COMSPEC%` is never consulted: a test sets it to something absurd and the
     answer is the refusal, not that program;
   * the child env for the Git case has Git's directories first on `PATH`,
     preserves the base entries, and keeps `TERM`/`COLUMNS`/`LINES`.
3. `pytest tests/test_console_gbash.py tests/test_console_policy.py` green with
   **no edit** to either file (the console connector's bash refusal is untouched).
4. `pytest tests/test_write_surface_isolation.py tests/test_api.py` green;
   `cd web && npm test` green; `git diff --stat web/` empty.
5. No managed key appeared: `git diff` shows no change to `src/config.py`'s
   managed key set, `_render_env()` or `.env.example`.
6. *Results* note: the exact refusal sentence, the exact env keys the Git case
   sets, the interactive flag chosen for bash on Windows, and — measured on the
   runner — whether the selected bash sees `stty`/`tput` with that env (csl-06 and
   csl-08 depend on this answer rather than on assumption).

## A hazard to handle explicitly

When the app itself is launched *from* a Git Bash window, `SHELL` in the
environment holds a POSIX-shaped path (`/bin/bash`) and `$PATH` holds MSYS-form
entries. The Windows branch must not read that as a usable shell spec or forward a
`PATH` full of `/usr/bin`-style entries to a Windows child and then wonder why
nothing resolves. Test that shape: `SHELL=/bin/bash` set in a Windows-shaped
selection run must still land on Git Bash (or the refusal), never on `/bin/bash`.

## Results (append as you measure)

Landed 2026-11-02, on the Windows runner (Git at `C:\Program Files\Git`, Python
3.13, `nt`/`win32`).

**What landed.** `src/core/termhub/shells.py` (new): `select_shell(platform,
environ)` and `child_environment(shell, cols, rows, environ)`, the refusal
sentence, and `SELECTION_PLATFORM` / `GIT_INSTALL_CANDIDATES` /
`GIT_ENV_OVERRIDE` / `_program_is_runnable` as patch-points. `session.py`:
`shell_command()` deleted, `spawn()` calls `shells.select_shell()` and passes
`list(shell)` plus `shells.child_environment(shell, cols, rows)`; the
bash-specific `-i` and the inline env dict are gone. `constants.py`: added
`TERMINAL_SHELL_KEY = "TERMINAL_SHELL"` (comment: process-environment only, never
a managed key). `termhub/__init__.py`: `shell_command` removed from the
re-exports, the module list names `termhub.shells`. New tests:
`tests/test_terminal_shells.py` (16).

**The amendment, recorded where it is implemented.** The module's docstring
quotes C-D2's original *"`TERMINAL_SHELL` → `%COMSPEC%` (`cmd.exe`) → refuse"*
and states that the operator's amendment of this lot removed the `cmd.exe` step.
`COMSPEC` is read nowhere in `termhub/` — the only other hit for that name in
`src/` is `config.py`'s reserved-environment-name list, which is what stops a
managed `.env` key ever shadowing it. `pwsh`/`powershell` are not auto-selected
and the refusal says so. The rest of C-D2 — the seam owning shell selection, and
`TERM` set for the child — is unchanged.

**The refusal sentences, verbatim.** No shell found (439 chars on this machine,
the candidates interpolated):

> the terminal found no shell to run: TERMINAL_SHELL is not set, `git` was not on
> PATH, none of the Git install candidates (C:\Program Files\Git, …) carries
> usr\bin\ls.exe, and BENCHMARKIT_GIT_BASH_DIR names no usable install. This pane
> runs Git Bash; install Git for Windows, point BENCHMARKIT_GIT_BASH_DIR at its
> install root, or set TERMINAL_SHELL to a drive-qualified shell path. cmd.exe and
> pwsh/powershell are not started here.

A `TERMINAL_SHELL` that is present but unusable refuses **by name** rather than
falling through, because a silent fallback would make the one knob dishonest:

> the configured shell is not usable on Windows: TERMINAL_SHELL='/bin/bash' names
> no Windows drive or share, so it is a POSIX-shaped path — the kind a Git Bash
> window carries, not a program this app can start as a Windows child (use a
> drive-qualified one such as C:\Program Files\Git\bin\bash.exe). Set it to a full
> Windows path (…), or leave it unset to use Git Bash.

Both raise `TerminalUnsupported`, so `spawn()`'s existing route handling
(`error` frame + 4501) renders them with no change to `routes.py` — tested by
`test_a_shell_refusal_reaches_the_session_as_the_same_refusal`.

**The qualification rule, measured first.** The task asked for "drive-qualified
via `path_norm.is_qualified`"; on this runner
`is_qualified("/bin/bash")` and `is_qualified("/c/Program Files/Git/…")` are both
**True** — a leading separator is qualified, because that is how the module
closes the `\etc\passwd` hole. So `is_qualified` alone cannot be the drive rule.
`_windows_shell_program` therefore asks two questions, both needed:
`is_qualified` (refuses a bare `bash`/relative name) **and**
`ntpath.splitdrive(spec)[0]` non-empty (refuses `/bin/bash`, `/c/…`, and any
MSYS-mount spelling; accepts `C:\…`, `C:/…` and a real UNC root). The drive check
is asked of the raw text, not `canonical_rel`, because the canonical form folds a
UNC root's leading separators away. Pinned by
`test_a_posix_shaped_terminal_shell_is_refused_by_name_not_spawned`.

**Env keys the Git case sets.** Exactly three added — `TERM` (from
`TERM_VALUE`), `COLUMNS`, `LINES` — and `PATH` rewritten with Git's directories
prepended in Windows form via `gitbash.git_child_env`; every other key is the
inherited process environment verbatim (today's documented trade-off, unchanged).
Which program gets that treatment is answered by the shared toolset marker on the
**selected program** (`_git_root_of`), not by `os.name`, so a `TERMINAL_SHELL`
outside a verified Git install keeps the ambient `PATH` untouched and the Linux
runner can prove the whole PATH assertion against a planted tree.

**The interactive flag chosen for bash on Windows: `-i`**, and it rides in the
spec, not in `session.py`. Reason in the docstring: interactive-non-login reads
`~/.bashrc` (prompt, aliases, colours) without re-running the profile's
environment setup once per tab.

**Measured: does the selected bash see `stty`/`tput`/`which` with that env? Yes —
and the finding matters for csl-06/csl-08.** Run against the real
`C:\Program Files\Git\bin\bash.exe` with `--norc --noprofile`, `command -v`
resolved `stty`, `tput`, `which`, `tty`, `vim`, `less` to `/usr/bin/…` in every
case tried: with our env, with the ambient Windows `PATH`, with `PATH` pointing at
System32 only, and with **no `PATH` key at all**. The reason is that
`usr\bin\bash.exe` rebuilds its own `PATH` from the install's POSIX root before it
executes anything (with no `PATH` it reports
`/ucrt64/bin:/usr/bin:/d/bin`). So on this install the tools resolve **whether or
not** the driver hands over the env — the inherited MSYS-form entries are not a
problem either. `git_child_env` stays because it is the *documented, asserted*
order (the one G-D4 proved for the connector's own resolver) and because the
tools resolve identically for a child started by a driver that cannot rely on
bash's self-configuration; but csl-06 must not treat a missing `PATH` entry as the
cause of a "command not found", and csl-08's `stty size` assertion is safe on this
machine for the first reason, not the second.

**Criterion 1 (the POSIX behaviour stays verbatim).** Not re-runnable from this
Windows machine: this runner has no POSIX driver (`pty_posix.available()` is
False) and conftest's `SHELL=/bin/sh` probe answers "no shell", so
`TestSession`'s 5 tests keep their **pre-existing** skips — unchanged by this
task. What was proven instead: `_select_posix` is the old `shell_command()` body
moved word for word (same `SHELL_KEY`, same `os.path.isabs` +
`os.access(..., X_OK)`, same `FALLBACK_SHELL`, same `-i`), the POSIX half is
asserted by 3 tests that run here, and `pytest --collect-only tests/test_terminal.py`
still shows its **16** tests, unchanged in name and count (the 12 pre-existing +
the 4 seam tests csl-04 added) — I edited no line of that file.

**What ran green here.** `pytest tests/test_terminal_shells.py` 16 passed, zero
skips. `tests/test_console_gbash.py tests/test_console_policy.py` 69 passed /
6 pre-existing skips, and `git diff` shows **no edit** to either file.
`tests/test_write_surface_isolation.py` 20 passed. `tests/test_proctree.py` and
`tests/test_terminal.py::TestDriverSeam` green. `cd web && npm test` 472 passed;
`git diff --stat web/` empty. `git diff` shows no change to `src/config.py`,
`_render_env()` or `.env.example` (grep `TERMINAL_SHELL` in `.env.example`:
zero hits). `test_api.py`: 63 passed / 2 failed, both **pre-existing** and
outside this diff, already recorded by csl-04 — the `NameError: helpers_block` at
`routes.py:1418` (the mid-merge checkout) and the NTFS fold fact of this machine.

**One note for csl-08.** The seam's platform answer is now
`shells.SELECTION_PLATFORM` (patchable, seeded from `os.name` at import) — the
shell probe can ask `shells.select_shell()` with no arguments and get this
runner's real answer, which is what the conftest gate needs.
`tests/conftest.py` was deliberately not touched: re-gating `shell_probe` is
csl-08's job, not this task's.
