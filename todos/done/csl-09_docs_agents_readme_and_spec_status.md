# csl-09 — Docs: the AGENTS.md clause, the README, the spec status, the manual pass

## Deliver this

Write down what actually landed, in the four places that are read for different
reasons, and record the findings the earlier tasks measured. **This task changes
no code and no test.**

1. **`AGENTS.md`** (injected into every WorkBench turn — accuracy beats volume):
   * `core/termhub/`'s row becomes what C §7 asks: *"one shell per WebSocket, on
     the platform's pseudo-terminal (a PTY, or ConPTY)"*, and — because this lot
     settled the Windows shell — name **which shell** without describing an
     implementation a routine change would move.
   * `core/proctree.py`'s row states it is the teardown with **two
     implementations** (group/session signalling, and a job object) behind one
     seam, and that the hubs and the terminal share it.
   * The process-tree bullet already says *"Teardown kills process *trees*, not
     wrappers"* — extend it to cover the terminal as its second caller rather than
     adding a new bullet beside it.
   * The terminal stays *"a human surface (`core/termhub/`, one WebSocket per
     session), never an agent tool"* — C §7's closing sentence: *"C does not change
     who may open a shell, only whether a Windows user can open one."*
   * Follow AGENTS.md's own rules for the file: *"It states **what the app is and
     how its pieces fit**, not how to build it"*, *"no **code details that
     move**"*, *"One idea per bullet"*, and *"Cut or shrink a paragraph when the
     code it described changes instead of appending a newer one beside it."* Edit
     the existing rows; do not append a Windows section.
2. **`README.md`**: the Windows line(s) — what works there now, the one optional
   extra and how to install it (matching the README's existing extras style, with
   no ports or machine-specific commands), the process-environment variables that
   affect the Windows terminal (`TERMINAL_SHELL`, `BENCHMARKIT_GIT_BASH_DIR`), and
   the refusal a user sees when neither a driver nor a Git install is present.
3. **The spec docs** (`docs/windows-support/`):
   * `windows-support-target-C-terminal.md`: status → implemented by this lot;
     **C-D2 amended** (Git Bash chosen, `cmd.exe` not a fallback, `TERMINAL_SHELL`
     process-env only) written into C-D2 itself rather than beside it; §4's file
     table and §6's verification lists corrected to what landed; the C-D6 encoding
     answer recorded as an answer; **C §4's Playwright row marked deferred** and
     the manual pass marked as the human step it is.
   * `windows-support-target-B1-console-gbash.md`: its §5 "out of scope … The
     Editor terminal (target C)" and its §9 open *manual pass* line gain a pointer
     to C's doc and to the shared `mcp_connectors/gitbash.py`; the statement that
     the console **refuses** a shell stays exactly as written (unchanged fact) and
     must not be softened.
   * `windows-support-road_to_target.md`: §8's C row and §10's bullet listing
     *"Git Bash / MSYS as a runtime"* as out of scope for the whole roadmap must be
     **corrected**, not annotated beside — this lot deliberately runs the terminal's
     shell out of Git for Windows (as the console connector already does), while
     the console tool still executes no shell. Say which of the two the bullet
     meant. §5's Job Object decision gets a pointer to `proctree_jobs.py`.
4. **The manual Windows pass**, written as a checklist a human runs — C §6's six
   items (vim + resize, a full-screen program painting after a resize mid-draw, a
   backgrounded child checked in Task Manager together with the console helper,
   Ctrl-C / Ctrl-L / arrow history, the 0-row guard after re-showing the tab, and
   killing the app from the server console leaving no shell) — each with what
   proves it passed. State plainly that **no task in this lot performed it**.

## Files it touches

| file | change |
|---|---|
| `AGENTS.md` | edit — the `termhub` and `proctree` rows, the process-tree bullet, and (only if a sentence is genuinely needed) one line in "How a request flows" about the terminal's refusal |
| `README.md` | edit — the Windows paragraph, the extras install line, the two process-env variables |
| `docs/windows-support/windows-support-target-C-terminal.md` | edit — status, C-D2 amendment, §4 table, §6 verification, C-D6 answer, deferred e2e |
| `docs/windows-support/windows-support-target-B1-console-gbash.md` | edit — pointer(s) only; no re-deciding G-D3/G-D4 |
| `docs/windows-support/windows-support-road_to_target.md` | edit — §5, §8, §10 corrections |
| `docs/windows-support/windows-support-target-A-no-crash.md`, `windows-support-target-B-connectors.md` | a pointer line only if one of their decisions is now implemented elsewhere; otherwise untouched |
| `src/`, `mcp_connectors/`, `tests/`, `web/` | **no change** |

## Rules it must not break (quoted)

* AGENTS.md: *"This file is written **for agents** … and it is injected into every
  WorkBench turn, so every line is paid for on every message"*, *"Keep it **short
  and current**"*, *"No **machine-specific setup**: no virtualenv manager,
  package-manager or browser-install commands, no ports or paths tied to one
  machine"* — the install line belongs in the README, not here.
* AGENTS.md: *"A **big-picture view only**: what each part is and which invariants
  hold. Feature-by-feature detail belongs in the README, route-level detail in the
  code and its doc comments."* Do not document the driver's API in AGENTS.md.
* AGENTS.md: *"A test that exists to enforce an invariant is worth naming as a
  pointer — the invariant stays the point."*
* AGENTS.md: *"This file feeds the chat: when the working location carries it, it
  is injected into every chat turn and re-read per request. The model acts on it,
  so accuracy matters more than volume."* Every sentence you add must be true of
  the code as it now is — verified against the diff of csl-01…csl-08, not against
  the spec's intentions.
* AGENTS.md: *"Adding a connector is never a module under `src/`"* — nothing in the
  docs may describe the terminal as a connector or a tool.
* Docs-set rule from the roadmap §11: *"every claim names a file and a line that
  exists"* (line numbers are pointers, not the thing) — after this lot, a claim
  must name a file/function that exists **now**. Re-check each one you cite.
* AGENTS.md stays **one idea per bullet, and no line so long it stops being read**.

## Depends on

csl-01 … csl-08. Read each of their *Results* notes first: this task is where the
lot's measured findings become project statements, and where an unmet acceptance
becomes a written gap instead of a silent one.

## How to tell it is done

1. `AGENTS.md` no longer describes the terminal as POSIX-only and no longer
   describes `proctree` as a single mechanism; the word count did not grow by more
   than a few lines (it is a replacement of two rows plus one bullet, not a new
   section).
2. Every file/function name cited in the four docs **exists** in the tree (check
   each with the read/grep tools; a stale name is a defect in this task).
3. `README.md` states the Windows extra install in the same style as the existing
   extras and names the two process-env variables, and no port, path or
   package-manager command appears in `AGENTS.md`.
4. The C doc's C-D2 section now reads as the amended decision (Git Bash; no
   `cmd.exe` fallback; refusal naming the gap; `TERMINAL_SHELL` process-env only),
   and the doc still records what was **rejected**, so a future reader knows the
   alternative was considered.
5. The deferred Playwright spec and the unperformed manual Windows pass are both
   written down as open items with the reason each is open.
6. Every *Results* finding from csl-01…csl-08 that contradicts a spec sentence has
   a corrected sentence somewhere — listed in this task's own *Results* note as
   "finding → where it landed".
7. `pytest tests/test_write_surface_isolation.py` still green (nothing else
   changed), and `git diff --stat src/ mcp_connectors/ tests/ web/` is **empty**.

## Results (what changed, and which findings landed where)

Docs-only task. No file under `src/`, `mcp_connectors/`, `tests/` or `web/` was
**edited by this task**. `git diff --stat -- web/` is empty. Every code/test path
in `git diff` was already present when this task began — the pre-task
`git status` already listed `src/api/routes.py`, `src/core/proctree.py`,
`src/core/termhub/{__init__,constants,session}.py`, `tests/conftest.py`,
`tests/test_terminal.py` (unstaged) and the B1-stage connector/hub files
(staged), plus the untracked `gitbash.py`, `proctree_jobs.py`, `drivers.py`,
`pty_conpty.py`, `pty_posix.py`, `shells.py` and the three new terminal test
files — all csl-01…csl-08 / B1 work, none introduced here. The only files this
task wrote are `AGENTS.md`, `README.md`, the five `docs/windows-support/*.md`
docs, and this task file.

Measured before/after (this Windows runner, project venv):

* `pytest tests/test_write_surface_isolation.py` → 20 passed.
* `pytest tests/test_terminal.py` → 20 passed, 1 skipped (the one marked-POSIX
  procfs session-scan) — matches csl-08's recorded count, so the docs task broke
  nothing.
* `pytest tests/test_write_surface_isolation.py tests/test_terminal_shells.py` →
  38 passed.
* `cd web && npm test` → 472 passed (regression check; no frontend change).

**AGENTS.md.** The file arrived with **three unresolved merge-conflict blocks**
(`<<<<<<< Updated upstream` / `=======` / `>>>>>>> Stashed changes`) left by the
earlier session; resolving them was a precondition of editing the rows at all.
The bullet-list form (the stash side, this file's current prose style) was kept;
the table side (the old commit's shape) was dropped; the stash side's facts were
corrected where HEAD's newer work made them stale (the reserved-folder row and
bullet now name the app-owned prompt-helper store the app *does* write, which
the stash side's "changed only outside the app" denied). Then C's own edits:

* `core/termhub/` — *"one shell per WebSocket, on the platform's
  pseudo-terminal (a PTY, or ConPTY); `$SHELL` on POSIX, and on Windows
  `TERMINAL_SHELL` or, by default, Git Bash."*
* `core/proctree.py` and `core/proctree_jobs.py` — one row: *"the process-tree
  teardown the hubs and the terminal share: one seam, group signalling or a job
  object."*
* The process-tree bullet extended in place (*"on either platform and by either
  mechanism … a hub or the terminal spawned of their own"*), no new bullet beside
  it.
* No "How a request flows" line added: the terminal's refusal sentence is fully
  carried by the termhub row, and AGENTS.md's own rule is to name the invariant,
  not the mechanism. No Windows section, no install command, no port/path, no
  driver API. The "human surface … never an agent tool" sentence was already
  there and stands.
* Result: **250 lines**. The clean pre-C content (the stash side of the
  conflict, before its markers) was 245; the conflicted working file was 315.
  So C's change is a replacement of two rows plus one bullet plus one stale
  fact corrected, at **+5 lines** — each of the four places gained one line
  because it now carries the second mechanism or the second platform. No new
  section, no append beside an old paragraph.
* Staged the resolution (`git add AGENTS.md`) so the file leaves the unmerged
  state its content had been written into; nothing was committed.

**README.md.** A Windows paragraph under *Running the app*, and a new
**#### The Editor terminal (POSIX and Windows)** subsection beside the MCP
connectors one, in the same extras style: `pip install -e ".[windows]"`, the
no-driver refusal (error line + close 4501), the no-Git refusal, the two
process-env variables (`TERMINAL_SHELL` — drive-qualified, POSIX-shaped values
refused by name; `BENCHMARKIT_GIT_BASH_DIR` — the install root), the
never-`cmd.exe`/PowerShell rule, and the boundary that the pane is human-only
while the console connector still refuses a shell.

**windows-support-target-C-terminal.md.** Status → implemented by the csl lot,
naming its two open items. C-D2 rewritten *as the amended decision* (the Git
Bash order, `cmd.exe` never consulted, `TERMINAL_SHELL` process-env only) while
keeping the rejected alternatives visible. §4's file table now lists what landed
(`drivers.py`, `shells.py`, `gitbash.py`, `proctree_jobs.py`, the three test
files) and marks the Playwright row **deferred** with the reason. §6 corrected to
the landed gate (the three files, the guard markers quoting the probe's verdict,
the `$!`-is-an-MSYS-pid finding, the Linux run recorded as still open). C-D3,
C-D4, C-D5 and C-D6 each gained an "as landed / measured" paragraph. §7's
AGENTS.md sentence marked done. §5's `cmd.exe` job-control wording corrected
(the pane runs bash; the out-of-scope claim is the stop/resume signal, not
`cmd.exe`).

**windows-support-target-B1-console-gbash.md.** Pointers only: §5's
"Editor terminal (target C)" line now says landed and points at the shared
`mcp_connectors/gitbash.py` extraction; the PowerShell/cmd refusal line states
unchanged-but-now-pointed (G-D3 stands, the console runs no shell); §9's open
manual-pass line gained "still not performed" + the shared-discovery note + the
pointer to C §6's checklist. No G-D decision re-opened.

**windows-support-road_to_target.md.** §5's Job Object decision gained its
"landed as `src/core/proctree_jobs.py` (ctypes, no package)" note. §6 gained the
"both halves landed, refusal is now a capability probe" note. §8 gained a
"How each landed" paragraph *correcting* the table (B landed for the MCP hub
only; the LSP refusal stands; C landed with C-D2 amended; `pywinpty` is the one
new dependency, jobs settled as ctypes). §10's
"Git Bash / MSYS as a runtime" bullet was **corrected, not annotated**: it meant
the app's own runtime, not Git's binaries — the bullets now say which of the two,
naming the console toolset and the terminal shell as shipped users of the one
install probe.

**windows-support-target-A-no-crash.md / target-B-connectors.md.** One line each:
A's A-D5 terminal row marked superseded by C (same 4501 shape, now a probe);
B's B-D5 code block marked landed as `proctree_jobs.py` with the hub-side
assignment still unwired and the LSP refusal standing.

**Findings → where they landed** (each from a csl task's *Results*):

| finding (measured) | where the sentence now is |
|---|---|
| `path_norm.is_qualified("/bin/bash")` is **True** — the spec's "via is_qualified" premise cannot be the drive rule alone | C doc, C-D2 (the two-question rule: `is_qualified` **and** a non-empty `splitdrive` root) |
| a Git Bash child resolves `stty`/`tput`/`which` even with only the ambient (or no) `PATH` — the lot's inherited premise is wrong for this install; `git_child_env` stays for the documented order, and the `stty` probe must ask the child's PATH | C doc, C-D2 (measured correction) + C-D4's `needs_stty` note |
| Git Bash's `$!` / `$$` are **MSYS pids**, not Windows pids — the `$!`-based tree assertion is unusable as planned | C doc, §6 Windows-only paragraph (the shell prints its child's OS pid; `job.members()` is the authoritative twin) |
| the ConPTY agent emits `ESC[6n` on a resize and eats keystrokes until answered — same-tick input garbles | C doc, C-D4 (and the suite keeps resize and input as separate frames) |
| a self-exited shell leaves `OpenConsole.exe` alive while the PTY is referenced — the dependency's release is not a dependable kill | C doc, C-D5 (the job is the guarantee; asserted) |
| no `setsid`-equivalent escape found (`&`/`disown` do not leave a job) | C doc, C-D5 |
| encoding: UTF-8 round-trip holds with **no** `chcp` step (Git's `C.UTF-8` + ConPTY's UTF-16/UTF-8 agent) | C doc, C-D6 recorded as the answered first option |
| teardown measured 0.20–0.23 s; nothing survived even a 0.05 s grace | C doc, C-D5 + §6 item 3 |
| Linux cadence not re-runnable from the Windows runner — 21 pass / 0 skips is csl-08's *prediction*, not a measurement | C doc, §6 (still-open check, named as such) |
| the LSP hub's Windows refusal stands untouched though `proctree_jobs.py` landed | roadmap §5/§8 and target-B B-D5 notes (further lot, own verification) |

**Also fixed while editing AGENTS.md (not C's work, but the file was unreadable
without it):** the three merge-conflict blocks. The pre-existing
`helpers_block` NameError in `api/routes.py` (recorded by csl-02/03/04/05/07/08
as a checkout artefact of that same unresolved merge) is **not** touched by this
task — it is a code defect for whoever completes that merge, and this task's
rules forbid `src/` edits.

**Open items this task wrote down rather than closed:** the manual Windows
checklist (six lines, each with its pass criterion, in C §6) and the deferred
terminal e2e spec (C §4 row) — both need a Windows desktop/browser session and
cannot be performed by a task in this lot.
