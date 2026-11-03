# Target B1 — the console connector over Git Bash (proposal)

**Status**: decisions settled (operator sign-off 2026-11-01, §8); implementation tracked in §9.
**Replaces (for the console connector)**: `windows-support-target-B-connectors.md`
B-D3 and B-D4 — the native-Windows command set and the in-place edit of
`console_mcp.py`. B-D1, B-D2, B-D5 and B-D7 stay as written; B1 *depends* on
B-D1/B-D2 landing.
**Depends on**: Target A (done — win-01 … win-07).
**Read first**: target-B doc §1–§2, win-00's three roadmap §9 rules.

## 1. What B1 is for

Give the WorkBench agent a working console on Windows **without touching the
app's no-shell invariant**: the command is still parsed into one argv and
exec'd directly, `check_command_allowed` is still the same parse, and the
policy is still path-scoped with the workspace policy honoured inside the root.

The idea: instead of teaching the connector a Windows command vocabulary it
does not have (`findstr`, `certutil`, … while `grep`, `sed`, `awk` do not
exist), run it over **Git for Windows' toolset**. "Git Bash" here means the
*binaries Git Bash ships* — `ls.exe`, `grep.exe`, `find.exe` … — not bash as
an interpreter. `bash.exe` is exactly the shell the connector refuses (it also
lives in `System32` as the WSL launcher, so refusing it is not theoretical).

### Probe evidence (2026-11-01, this machine, Git at `C:\Program Files\Git`)

- `usr\bin` carries almost the whole `ALLOWED_COMMANDS` vocabulary of
  `console_mcp.py:153-193` as real PE executables: `ls cat grep sed awk find
  head tail wc sort uniq cut tr tac nl column comm paste less which file stat
  basename dirname realpath readlink diff cmp seq printf numfmt mkdir touch cp
  mv ln mktemp md5sum sha1sum sha256sum tar gzip unzip ps df du hostname uname
  nproc id groups who env printenv sleep date echo dir openssl chmod chown rm
  rmdir shred truncate kill`. These are directly `Popen`-able with
  `shell=False`.
- Absent everywhere: `man bc zip wget top vmstat free uptime lscpu lsb_release
  w tracepath ss ip dig jq sqlite3 rg ag fd make lsof tree more.exe`
  (`more.com` exists in `System32`). `curl ping nslookup ipconfig where fc
  certutil attrib systeminfo tasklist findstr tracert netstat arp` exist only
  in `System32`. `python node npm pip npx` resolve from the ambient PATH, not
  from Git.
- Name collisions resolved by search order: `find sort tar hostname` exist in
  both trees; Git first means GNU semantics.
- **Path forms verified**: `ls.exe`/`cat.exe`/`grep.exe` accept a Windows-form
  argument with backslashes (`D:\dev\benchmarkit\docs`), with forward slashes,
  the MSYS mount form (`/d/dev/benchmarkit/docs`), and a relative argument
  against a Windows `cwd`. So the connector keeps **one path model** — the
  Windows `Path` machinery Target A built (`path_norm`, the win-06 containment)
  — and hands argv through unchanged.

### Why this is the easier route

The Target B plan's hardest console pieces shrink: B-D4's "Windows command set"
is mostly *the existing list, resolved against Git's bin*, and the whole POSIX
refusal vocabulary (`rm`, `chmod`, `sudo`…) stays meaningful because the tools
that implement them exist. What survives unchanged: B-D3 (the tokenizer eats
backslashes — `shlex.split(r'cat C:\Users\vgr\a.txt')` →
`['cat', 'C:Usersvgra.txt']`, `console_mcp.py:309`/`:334`), and the guardrail
values (B-D1/B-D2) without which the win-07 hub refusal
(`mcphub/process.py:33-48`, `mcphub/constants.py:49-51`) keeps every `local`
connector off Windows. **B1 must not ship before B-D1/B-D2 land** — target B
§1's rule ("B never starts where the guardrails match nothing") binds here too.

## 2. Decisions (numbered G-D; each with the alternative it rejects)

### G-D1 — one connector file: `console_mcp.py` selects its toolset at startup (amended, operator decision 2026-11-01)

The Git-toolset mode lands *inside* `console_mcp.py`, not as a second connector
file. The mode is not a platform branch inside the policy — it is a startup
capability probe: the connector discovers Git's toolset; found → the gbash
semantics (resolver, command set, backslash rule, mount-form translation);
absent on Windows → the connector refuses to start with a sentence naming the
gap; POSIX → today's behaviour, unchanged. The grammar, the per-statement
policy and the output honesty stay one copy in one file; only four things are
mode-selected — the command sets, the resolver, the backslash flag, the
mount-form translation — and they sit in one startup-selected named block.

Why this is the easier route (the justification the operator asked for):

- **No new declaration entry.** The existing `console_mcp` entry works on both
  machines. A second entry is a second source of truth that must be
  machine-correct — enabling the wrong console on the wrong box is exactly the
  failure the declaration-file rule guards.
- **Anti-drift is structural.** One module, one core. `console_shared.py`
  existed only to stop two files forking one policy (G-D2); with one file there
  is nothing to fork, so the drift the fedit-vs-console story warns about
  cannot reappear here.
- **The Linux gate stays literal.** `test_console_policy.py` exercises the same
  file, unchanged, zero skips.
- **The Windows half is one named seam** (discovery + resolver + sets +
  tokenizer flag), roadmap §9 rule 2's shape: tests force it by patching the
  seam, never by pretending to be another OS.

*Rejected — a separate `console_gbash_mcp.py` (this section's original
decision).* Two connector modules carrying one policy is exactly the drift the
project keeps warning about; the shared-core extraction it forces is machinery
to undo a fork this option creates for no gain, plus a per-machine declaration
entry. The underscore-name reasoning survives as a rule for any *future*
connector file: a hyphen breaks `python -m mcp_connectors.<name>` and the
tests' module import (`tests/test_console_policy.py:17,36`).

*Rejected — a native-Windows `console-win_mcp.py`.* Smaller vocabulary
(no `grep`/`sed`/`awk`), cmd-builtin refusals, same tokenizer fix, and the
security lists must be re-derived anyway. Kept as a later fallback if Git is
absent; v1 refuses with a sentence naming that gap instead of inventing it.

### G-D2 — anti-drift: the single file *is* the shared core (superseded by the amended G-D1)

The extraction of `console_shared.py` is dropped: with one connector module
there is no fork to prevent. The discipline it encoded still governs the new
code — every mode-selected helper takes the root and the fold **as
parameters** (no hidden global reads inside the shared grammar), and the mode
itself is one startup-selected attribute. The invariant that survives, restated
in §7: a policy rule changes in `console_mcp.py` / `path_policy.py` once, and
both modes inherit it.

### G-D3 — no shell, and the shells are blocked by name

`bash sh cmd powershell pwsh cscript wscript` join `ALWAYS_BLOCKED_COMMANDS`
(they are never path work). The refusal sentence names the design: *this
connector runs no shell; Git Bash supplies the tools, not an interpreter.*
`cmd.exe /c` and `bash -c` are the same refusal. PowerShell-style tools
(`Select-String`) stay out exactly as B-D4 rejected them.

### G-D4 — Git-root discovery and command resolution

At startup the connector resolves its tool search path:

1. `shutil.which("git")` → walk up to the install root (…`cmd\git.exe` →
   install dir) and verify `usr\bin\ls.exe`;
2. else fixed candidates: `C:\Program Files\Git`,
   `%LOCALAPPDATA%\Programs\Git`, `%USERPROFILE%\scoop\apps\git\current`,
   `C:\ProgramData\chocolatey\opt\git\current`;
3. else a connector-local env override (candidate name
   `BENCHMARKIT_GIT_BASH_DIR`) — deliberately **not** a managed `.env` key, the
   same reasoning as `BENCHMARKIT_PATH_LOCATION` (a settings save must not
   erase it, and A-D6's reserved-name rule never applies to it);
4. else refuse to start with a sentence (the win-07 honesty shape, on stderr,
   non-zero exit; the hub reports the failed connection).

Resolution order for every command: `Git\usr\bin` → `Git\bin` →
`Git\mingw64\bin` → ambient PATH (so `python`/`npm`/`curl`/`ping` resolve).
`shutil.which` honours `PATHEXT` (B-D6's rule — `more.com` resolves,
`.cmd` shims resolve), and **never** falls back to `shell=True`. An unresolved
command is a refusal naming what was searched — the `FileNotFoundError` at
`console_mcp.py:1068` folded into `check_command_allowed`'s verdict, so the
oracle and the execution agree.

The discovery candidates and the search path are module attributes so tests
can patch them (the hermetic-tests convention). A startup self-check resolves
every entry of `ALLOWED_COMMANDS` once and logs the split — the list is
"selected by which platform's binaries the connector can actually resolve"
(B-D4's principle), not a hand-verified guess.

### G-D5 — the tokenizer keeps backslashes (B-D3, load-bearing here)

The statement grammar is untouched — same operators, same refusal sentences,
the C-1…C-6 tests pass unchanged. Only the argument layer changes, behind a
named flag on the shared parser (`literal_backslash`, default `False` so
Linux stays exact — roadmap §9 rule 2): an unquoted `\` is a literal character
in argv; quoting keeps spaces (`cat "C:\Users\vgr\a file.txt"`). `~` stays
raw in argv (MSYS expands it itself to the user profile) while the *policy*
still expands it via `os.path.expanduser` — so `cat ~/.ssh/id_rsa` is refused
by the component rule on both platforms.

### G-D6 — one path model: Windows paths, with the mount form translated for the verdict

`--root-dir` stays `${BENCHMARKIT_PATH_LOCATION}` (a Windows path), exactly as
console_mcp's entry passes it. Policy questions (`is_within_root`, the
workspace policy, the deny-lists) run on the Windows `Path` machinery. A
leading `/c/…`-style mount form is translated to `C:\…` **for the verdict
only**; argv reaches the tool unchanged (proven to accept both forms).
Drive-relative specs (`C:x`) are refused by name — `path_norm.is_qualified`
already calls that shape a hole; the connector names it. UNC paths are outside
the root and live under the allow/block lists like any other outside path.
Symlink/junction honesty is unchanged: a token is judged both as written and
as its resolved target (`protected_path_error` already does this).

### G-D7 — command sets (from the probe, refined by the startup self-check)

- **Allowed (resolve in Git's trees)**: the existing POSIX list, minus the
  names the probe shows nowhere (`man bc zip top vmstat free uptime lscpu
  lsb_release w tracepath ss ip dig jq sqlite3 rg ag fd make lsof tree wget`),
  plus `fc` (System32 `fc.exe` is a real file-compare tool — it leaves the
  builtin list on this connector, where it is not a builtin).
- **Allowed (resolve from ambient PATH)**: `python python3 node npm npx pip
  git curl ping nslookup ipconfig where certutil systeminfo tasklist findstr
  tracert netstat more`. `certutil -hashfile` fills the `md5sum/sha256sum`
  role if Git's coreutils are ever absent.
- **Blocked everywhere** (added to `ALWAYS_BLOCKED_COMMANDS`): `bash sh cmd
  powershell pwsh cscript wscript del erase rd format takeown icacls cacls
  net sc reg schtasks runas cipher diskpart mountvol subst fsutil bcdedit
  vssadmin wmic msiexec regsvr32`. The existing POSIX always-blocked set stays
  (Git ships `chmod chown rm rmdir shred truncate kill` — the in-root bypass
  judges them exactly as it does on Linux: writes inside the root are the
  operator's own workspace, `rm` there is allowed by design, same as today,
  with the Editor's blessed delete as the app-side path).
- **Builtin refusals**: the existing `SHELL_BUILTIN_COMMANDS` minus `fc`,
  plus `cls copy move ren md ver vol start call if for goto` — names with no
  `.exe` under `shell=False` (`type` among them), refused with the builtin
  hint rather than `FileNotFoundError`. `dir`/`echo` resolve to real coreutils
  binaries here, so they behave like Linux `dir`: an `ls`-shadow, allowed
  inside the root, list-checked outside it. No new confusion, same as today.

### G-D8 — guardrails first (this is B-D1 + B-D2, the hard prerequisite)

Before the connector can be *enabled* on Windows, the lists must speak there
(win-07's refusal is what this section replaces):

- `path_policy`'s prefix lists gain Windows forms **expanded at read time**
  from the environment (`SystemRoot`, `ProgramFiles`, `ProgramW6432`,
  `ProgramFiles(x86)`, `ProgramData`, `TEMP`/`TMP` for the scratch allowance)
  beside the existing POSIX entries — one list, both forms, per B-D1's rejected
  alternatives. `DENY_COMPONENTS`/`DENY_FILENAMES` already work cross-platform
  (they read `Path.parts` and a lowercased name) and stay as they are.
- `prefix_hit` becomes **component-wise** (via `path_norm.path_parts`, with an
  explicit `fold` argument) so `C:\Windows.old` is not under `C:\Windows`.
  The case fold for connectors needs a probe answer for the volume walked —
  the probe lives in `mcp_connectors/` (stdlib-only; `src` may import
  `mcp_connectors`, as `mcphub/process.py:22-28` already does, so `editorfs`
  and the connectors can share one probe).
- Scratch keeps B-D2's ordering: `%TEMP%` is allowed **only when** the
  component/name deny half says nothing — a path traversing from temp into a
  protected store is refused, and the as-written/resolved pair is checked as
  `protected_path_error` already does.
- The hub seam: `mcp_reach_is_refused()`
  (`mcphub/discovery.py:33`, `manager.py:81-95/137-139/243-246`) consults
  `guardrails_speak_here()` (already shape-based, `process.py:51-79`) for
  **`local` MCP connectors**; `local_reach_is_bounded()`'s `os.name` branch
  stays for the **LSP hub**, whose grandchildren need B-D5's job objects. The
  honest asymmetry is stated in the seam's docstring: an MCP console's children
  are short-lived direct children it kills on timeout (plus the hub's wrapper
  kill); a language server's tree is not, so the LSP refusal stands until
  `proctree_jobs.py` lands. win-07's forced-refusal tests
  (`tests/test_mcp.py:420-421` and friends) keep asserting the refusal — they
  force it by patching the guardrail half, which is the half that changes.

### G-D9 — what is unchanged, and must stay unchanged

Timeouts and their partial output, output caps, decode honesty
(`backslashreplace` + the note), the per-stream cap, `2>&1` acceptance, the
pol-08 workspace-policy refusals (same file, same matcher, same sentences as
the Editor pane), `resolve_work_dir`'s three questions, the root bypass and its
documented asymmetries. All of it stays one copy in `console_mcp.py`; B1 adds
no new policy power anywhere — that sentence must stay true after the mode
lands. The one narrowing B1 does make on both platforms: the scratch allowance
no longer outranks the component/name half of the deny-list (a secret named
inside `/tmp` or `%TEMP%` is refused), per B-D2's ordering decision.

## 3. Shape and files

| piece | change | holds |
|---|---|---|
| `mcp_connectors/path_policy.py` | edit | Windows prefix values (env-expanded), component-wise `prefix_hit` with `fold` |
| `mcp_connectors/` case probe | new (small) | the fold answer the connectors may ask (shared with `editorfs`) |
| `src/core/mcphub/process.py`, `constants.py`, `discovery.py` | edit | the seam lift for `local` MCP connectors; LSP half unchanged |
| `mcp_connectors/console_shared.py` | dropped (G-D2) | one file is the core; nothing to extract |
| `mcp_connectors/console_mcp.py` | edit | the toolset mode: discovery, search path, resolver, backslash flag, Windows sets; the grammar/policy/output core stays one copy |
| `mcp_connectors/externals.json` | **no entry needed** | the existing `console_mcp` entry runs both modes — the declaration file stays machine-neutral |
| `tests/test_console_policy.py` | unchanged | the Linux regression gate |
| `tests/test_console_gbash.py` | **new** | the behaviour tests below |

No externals change: the existing `console_mcp` entry (`"command": ["python",
"mcp_connectors/console_mcp.py", "--root-dir",
"${BENCHMARKIT_PATH_LOCATION}"]`) selects its toolset at startup, so one
machine-neutral declaration runs the POSIX mode on Linux and the Git-toolset
mode on Windows.

## 4. Tests (roadmap §9 rule 3: every refusal is a tested behaviour)

**Both platforms** (through the one core): the existing
`test_console_policy.py` green with zero skips — the mode selection changes no
POSIX verdict; `cat C:\Users\vgr\a.txt` survives the tokenizer verbatim under
the gbash flag and stays mangled under the Linux flag (the flag is asserted in
both directions on one runner).

**Windows-only** (target B's `windows_behaviour` marker):

- resolution order: `find`/`sort`/`tar` resolve to Git's GNU binaries, not
  `System32`; a `.cmd`/`.com` shim resolves via `PATHEXT`;
- an unresolved command is refused naming the searched directories, never
  `FileNotFoundError`;
- `bash`, `sh`, `cmd`, `powershell` refused by name with the no-shell sentence
  (asserted even though `System32\bash.exe` exists);
- `cd`, `export`, `source`, `type` refused with the builtin hint; `fc` allowed
  as the real tool;
- `ls C:\Windows` judged by the Windows guardrail values (component-wise,
  case-folded — `c:\windows` and `C:\WINDOWS` one answer); `C:\Windows.old`
  not under `C:\Windows`;
- `cat ~/.ssh/id_rsa` refused through Git's `cat.exe` exactly as through coreutils;
- `%TEMP%\x` allowed; `%TEMP%\.ssh\id_rsa` refused (scratch does not outrank
  the component rule);
- `/c/Windows` translated for the verdict, argv unchanged; `C:x` refused by
  name; a junction in temp whose target is a store is refused;
- pol-08 inside the root: a `hidden` folder is refused as argument *and* as
  `cwd`; an `ro` folder refuses `touch`/`mv`/`cp` but stays browsable — same
  sentences as the Editor pane;
- no Git install → the connector exits non-zero naming the gap (discovery
  candidates patched, per the hermetic rule).

**Manual, on Windows**: one WorkBench turn with `console` + `fedit`
enabled — `ls`, `grep`, a refused `cat .env`-outside-root, a refused pipe —
then quit the app and confirm in Task Manager that no connector process
survived (the wrapper-kill claim B1 makes must be observed, not asserted).

## 5. Out of scope for B1

- Job Objects / `proctree_jobs.py` (B-D5) — separate task; B1 keeps the LSP
  refusal and documents the wrapper-only teardown for console children. That
  module has since landed (csl-02, the target C lot), and the LSP hub's refusal
  was **left exactly as it is** — lifting it is a further lot with its own
  real-language-server verification.
- The LSP hub on Windows, `restricted-paths.txt` rework (B-D1's LSP half),
  fedit's NTFS read-only-attribute behaviour (B-D7) — unchanged targets.
- The native-Windows `console-win_mcp.py` fallback (G-D1's rejected option,
  kept as a named possibility).
- PowerShell or cmd as interpreters — refused, permanently. That refusal is
  unchanged by target C landing: the Editor's terminal pane runs a shell for a
  **human** in the browser, and the console connector still runs none — G-D3's
  shells-blocked-by-name rule and `NO_SHELL_SENTENCE` stand as written, and
  `tests/test_console_policy.py` keeps asserting them.
- The Editor terminal (target C). **Landed**: `windows-support-target-C-terminal.md`
  (status and §6 give its measured results). That pane runs **Git Bash** — which
  is why G-D4's discovery moved out of `console_mcp.py` into
  `mcp_connectors/gitbash.py` (csl-01), so the connector and the pane answer
  "where is Git, and what search path does a child get" from one probe. The
  extraction changed no console verdict: `console_mcp.py` delegates to it and
  `tests/test_console_gbash.py` pins the delegation.

## 6. Task order (todos/ files when the lot starts)

| # | task | lands |
|---|---|---|
| gb-01 | B-D1 + B-D2: guardrail values, component-wise matching, the connector-side fold probe, the MCP seam lift + its tests | §2 G-D8 |
| gb-02 | the Windows path model inside `console_mcp.py`: backslash flag, mount-form translation for the verdict, drive-relative refusal, scratch/name ordering, fold threading | G-D5/G-D6, B-D2 |
| gb-03 | the toolset mode in `console_mcp.py`: Git discovery, search path, resolver folded into the verdict, command sets, no-Git refusal | G-D3–G-D7 |
| gb-04 | tests (`windows_behaviour` marker), the AGENTS.md clause, the target-B pointer | §4 |

gb-02 and gb-03 may overlap; gb-01 gates everything (it is the safety half),
and gb-04's manual pass closes the lot the way win-09 closed target A.

## 7. What B1 must not break

- *"`check_command_allowed` is the same parse"* — the resolver's verdict is
  folded into it, so the oracle never promises a command `execute_command`
  cannot run.
- *"each connector's reach is bounded by a deny-list"* — on Windows that
  sentence is only allowed to become true after gb-01; until then the hub
  refusal stays.
- The Linux connector's behaviour, verbatim: every refusal sentence, the root
  bypass, the pol-08 split, the output honesty, the 721-line policy test file.
- The two modes must not drift: one file, one core — a policy rule changes in
  `console_mcp.py` / `path_policy.py` once, and both modes inherit it.

## 8. Operator decisions (settled 2026-11-01)

1. **G-D1** — settled: the in-place startup mode inside `console_mcp.py`
   (operator decision; the section above carries the justification). The
   target-B doc's B-D4 carries a pointer here rather than two plans.
2. **Naming** — moot: no new connector file. The underscore rule stands for
   any future connector script.
3. **G-D8's scope** — settled: the seam lift applies to **all** `local` MCP
   connectors, so `fedit` is un-refused on Windows too (its policy machinery
   is already A-correct; its B-D7 NTFS bit stays open as its own task).
4. **G-D7's set** — settled: signed off as specified (`fc` allowed, `dir`/
   `echo` as coreutils, `certutil`/`findstr`/`tasklist` allowed); the startup
   self-check logs which names actually resolve.
5. **Target-B pointer** — done in gb-04: B-D4 gains a pointer to this doc.

## 9. Implementation TODO

Run tests with the project venv: `.\.venv\Scripts\python.exe -m pytest …`.

- [x] **gb-01a** — guardrail values + component-wise matching + fold probe +
      scratch/name ordering + MCP seam lift (code landed, see "State" below)
- [x] **gb-01b** — Windows test gaps closed (decisions below; `test_fscase`,
      `test_mcp::TestConnectorEnvironment` green on Windows)
- [x] **gb-02** — the Windows path model in `console_mcp.py`: backslash flag,
      mount-form translation for the verdict, drive-relative refusal, fold
      threading (G-D5/G-D6)
- [x] **gb-03** — the toolset mode in `console_mcp.py`: Git discovery, search
      path, resolver folded into the verdict, command sets, no-Git refusal
      (G-D3–G-D7)
- [x] **gb-04** — `tests/test_console_gbash.py` (17 tests: 11 runner-neutral
      through the seam, 6 `windows_behaviour`), the `windows_behaviour` marker
      in `tests/conftest.py`, the AGENTS.md clause; the full perimeter green
      on Windows with zero Linux skips
- [ ] **manual pass** (closes the lot the way win-09 closed target A): one
      WorkBench turn with `console` + `fedit` enabled — `ls`, `grep`, a refused
      out-of-root secret read, a refused pipe — then quit the app and confirm
      in Task Manager that no connector process survived. **Still not
      performed.** The Git install that turn selects is the one
      `mcp_connectors/gitbash.py` finds, which is now also what the Editor's
      terminal runs its shell from, so a pass here proves the discovery for both
      surfaces at once. The terminal pane's own manual checklist — six lines,
      and equally unperformed — is in
      `windows-support-target-C-terminal.md` §6.

### State at the last session end (2026-11-01, second session)

Branch `feat/windows-support`; nothing committed yet in this lot. Spec amended
(G-D1 single-file decision, G-D2 dropped, target-B B-D4 pointer added).

**gb-02 landed** (the design decision recorded below was followed exactly):

*Tokenizer* — `unquote(token, literal_backslash=False)`: the mangling point was
`shlex.split` inside `unquote` only (`shlex.shlex` with `punctuation_chars`
already preserves backslashes in the token), so the grammar is untouched and
`unquote_literal` strips quoting verbatim. `parse_shell_statements`,
`validate_command` and `execute_command` thread the flag; the default derives
from `root_is_drive_form()`.

*Path model from the root's shape* — `path_model_backslash()`,
`path_model_fold()` (drive-shaped root → `fscase.case_folded(ROOT_DIR)`;
otherwise the `CASE_FOLDED` fallback), `verdict_path_text()` (mount form via
new `path_policy.mount_form_to_drive`, drive roots only), and
`drive_relative_spec()` (`path_norm.is_qualified` + `ntpath.splitdrive` — a
text test; `Path.is_absolute` would answer differently per platform).
`resolve_as_path`/`absolute_as_written` run the translated text; argv is
untouched. The three known `test_console_policy` tokenizer failures closed
with no test edits, and `test_fedit_policy` stayed green (as predicted).

**gb-03 landed:**

*The named block* — `Toolset` (allowed sets, builtin set, resolves?) with
`POSIX_TOOLSET` / `GBASH_TOOLSET`; `TOOLSET` is the startup-selected module
attribute (tests patch it — the Linux gate runs `POSIX_TOOLSET` unchanged).
`GBASH_ALLOWED_COMMANDS` = the POSIX list minus `GBASH_ABSENT_COMMANDS` (the
probe's nowhere-names) plus `GBASH_AMBIENT_COMMANDS` (System32/PATH names,
`fc` among them); `GBASH_BUILTIN_COMMANDS` = POSIX builtins minus `fc` plus
`cls copy move ren md ver vol start call if for goto type`. Shells
(`bash sh cmd powershell pwsh cscript wscript`) joined `ALWAYS_BLOCKED_COMMANDS`
on both platforms with `NO_SHELL_SENTENCE` naming the design; the Windows
power/admin names joined it too (G-D7).

*Discovery/resolver* — `discover_git_install()` (which-git walk-up verified by
`usr\bin\ls.exe` → candidates → `BENCHMARKIT_GIT_BASH_DIR` → None),
`git_search_paths()`, `resolve_command()` (`shutil.which` honouring PATHEXT,
Git trees before ambient PATH, path-spelled argv[0] asked directly, never a
shell), `select_toolset()` (startup only; POSIX not probed; Windows absent →
`no_toolset_refusal()` → main prints to stderr and exits 1), `toolset_split()`
(the startup self-check logged in the banner). The resolver refusal sits at the
two points a statement passes the name checks (in-root bypass, end of the
lists) so list sentences keep their own words; `execute_command` execs the
resolved absolute path in gbash mode.

*Verified on this machine (Git at `C:\Program Files\Git`)*: startup selects
gbash, 83 allowed commands resolve, `find` → Git's `find.EXE`, `more` →
System32's `more.COM`, `bash` resolves but is refused by name; unresolved at
startup here: `pytest`, `time` (logged, not hidden).

**gb-04 landed:** `tests/conftest.py` gained the `windows_behaviour` marker
(`windows_probe`, mirroring `posix_behaviour`); `tests/test_console_gbash.py`
asserts the flag both directions on one runner, the mount form translated for
the verdict with argv unchanged, the drive-relative refusal (and its absence
under a POSIX-shaped root), the shell refusals naming the design, the gbash
sets vs the POSIX ones, resolution order + the resolver refusal naming the
searched dirs (never `FileNotFoundError`), the discovery questions + the
refusal sentence, and the Windows half (guardrail spellings one verdict,
`System32\config` refused, `Windows.old` not under `Windows`, PATHEXT shim,
Git-GNU resolution, `%TEMP%` scratch vs a store named inside it, no-Git
startup refusal). The AGENTS.md clause (toolset mode + root-shape path model)
is in place. `path_policy.py`'s module docstring became raw (`\W` warning from
the gb-01 wording).

**Perimeter green on Windows** (537 passed, 16 skipped — the skips are the
existing tool probes and the new `windows_behaviour` skips on Linux):
`test_console_policy` (its three known failures closed by gb-02, zero test
edits), `test_console_gbash`, `test_fedit_policy`, `test_path_policy`,
`test_path_norm`, `test_fscase`, `test_mcp`, `test_write_surface_isolation`.
The full-suite run needs more than the tool timeout here — run it per file.

**Known Windows gaps status:** items 1 and 2 of the list below are **closed by
gb-02** (verified: `test_console_policy` 42 passed / 6 skipped on Windows,
`test_fedit_policy` green); items 3 and 4 were already closed by gb-01b.

**gb-01a landed (detail):**

*`mcp_connectors/path_policy.py`* — Windows values join the three prefix lists
at read time from the environment (`_env_absolute`, `_dedupe`,
`scratch_prefix_candidates`, `system_deny_prefix_candidates`,
`system_read_deny_prefix_candidates`; env names in `_SYSTEM_DENY_ENV` /
`_SCRATCH_ENV`; the read list gains `SystemRoot\System32\config`).
`prefix_hit` is now component-wise via `_rooted_parts` (root + whole
components; UNC share kept whole) with an explicit `fold` kwarg; `under_prefix`
and `glob_hit` take `fold`; `deny_reason` split into `prefix_deny_reason` +
`name_component_deny_reason` (both `fold`), composed location-then-identity.

*`mcp_connectors/console_mcp.py`* — `CASE_FOLDED` module attribute (default
`False`, read only through the explicit `fold` argument);
`protected_path_error` reordered: identity half → scratch allowance → location
half (B-D2's ordering); `_trusted_prefix_hit` threads `fold`. (The backslash
flag, mount-form translation, drive-relative refusal and the mode itself
landed in gb-02/gb-03 above; `path_model_fold()` now derives the fold from the
root's shape and `CASE_FOLDED` is its POSIX/unrooted fallback.)

*Fold probe* — moved to new `mcp_connectors/fscase.py` (shared: the app's
`editorfs/fscase.py` is now a re-export of it, same cache object);
`tests/test_fscase.py` retargeted to patch `mcp_connectors.fscase` internals
(`probe_shared._probe`, `probe_shared.os`) plus a new "one probe" delegation
test.

*MCP seam lift* — `mcphub/discovery.mcp_reach_is_refused()` consults only
`guardrails_speak_here()` (connector lists); `local_reach_is_bounded()` stays
the LSP hub's half (`lsphub/discovery.lsp_reach_is_refused` unchanged);
docstrings updated in `mcphub/process.py`, `mcphub/constants.py`,
`lsphub/constants.py` to state the asymmetry.

*Tests added* — `test_path_policy.py`: `TestPrefixHitComponentWise` (drive
prefixes, `Windows.old`, fold both ways, backslash spellings, UNC, POSIX old
verdicts kept) and `TestWindowsValuesArriveFromTheEnvironment` (candidate
helpers over a fake env, dedupe, hive dir, module lists speak the runner's
own location form); `TestListsAreMeaningful` rewritten to name the POSIX-only
list explicitly (its zero-match assertions were POSIX-default assumptions);
module-default test widened to allow drive entries. `test_mcp.py`:
`test_the_seam_is_bounded_here_so_nothing_is_refused` rewritten (guardrail
half only), `test_the_platform_half_of_the_seam_is_posix_here`
(`posix_behaviour`), `test_the_mcp_refusal_asks_only_the_guardrail_half`,
`test_the_lsp_hub_still_asks_both_halves`.

**gb-01b landed (decisions):** the rewrite option, not markers — the failing
assertions were *runner-shape* assumptions, not platform facts.
`test_fscase.py`: the probe's expectation is now "whatever this volume
answers" — a new agreement test (`test_the_probe_agrees_with_the_volume_it_
walks`) compares the probe against a direct observation of the planted names;
the cache tests force their answers through the `_probe` seam or the cache
(`test_two_roots_are_answered_separately`, `test_mode_for_hands_the_verdict_
the_roots_answer` both directions on one runner); the settings-switch test
pins the *fresh re-probe* after invalidation instead of a `False` answer;
`test_the_verdict_without_a_folded_filesystem_is_unchanged` became
`test_the_exact_spelling_matches_whatever_the_volume_answers`. No cache/
invalidation test was weakened. `test_mcp.py::TestConnectorEnvironment`:
platform-neutral assertions (relative values relative on both platforms,
`Path(...).is_absolute()`, `os.pathsep`-split `PYTHONPATH`).

**Known Windows gaps (all PRE-EXISTING — verified by stashing the gb-01
changes and re-running; none are regressions):**

1. `test_console_policy.py` 3 failures — `test_a_traversal_out_of_a_secret_store_is_still_refused`,
   `test_the_creating_names_are_allowed_only_as_path_scoped_workspace_work`,
   `test_the_policy_speaks_inside_the_root_and_the_deny_list_outside_it`:
   the shlex tokenizer eats backslashes (`cat C:\Users\vgr\a.txt` →
   `C:Usersvgr…`), so out-of-root Windows tokens read as in-root and bypass —
   exactly B-D3/G-D5, which **gb-02 fixes**; these three tests then pass on
   Windows without touching the file.
2. `test_fedit_policy.py` — same class, expect the same fix from gb-02.
   (Verified at gb-01b time: this file is currently **green on Windows** —
   fedit normalises its path arguments, so the tokenizer class does not reach
   it; keep an eye on it when gb-02 lands.)
3. ~~`test_mcp.py::TestConnectorEnvironment` 2 failures~~ — **closed by
   gb-01b** (platform-neutral assertions chosen over the `posix_behaviour`
   marker: the invariant is cross-platform, only the spellings were POSIX).
4. ~~`test_fscase.py` 7 failures~~ — **closed by gb-01b** (the volume-answer
   expectation + cache-forced branches, per roadmap §9 rule 2; the cache/
   invalidation tests were kept, not weakened).

**Design decision taken at gb-02 (recorded here for the next session):** the
*path model* half (backslash flag, mount-form translation, drive-relative
refusal) is derived from the **shape of `ROOT_DIR`** (`path_policy.root_form`
answers `"drive"`) — a probe of the root, not an OS name, so on Windows test
roots the three tokenizer failures close with no test edits and on Linux roots
the verdicts stay exact; tests force either direction by patching `ROOT_DIR`
or passing the explicit flag. The *toolset* half (Git discovery, resolver,
command sets) is selected only at connector **startup** (`main()`), so
`test_console_policy.py` keeps running the POSIX command sets on a Windows
runner (its `rg`/`make`/`tree` expectations stay true) and the gbash half is
exercised through `tests/test_console_gbash.py` by patching the seam.

**Next session's first move:** gb-01b (decide + apply the marker/rewrite
choices above so `test_path_policy`, `test_fscase`, `test_mcp` are green on
Windows), then gb-02 (backslash flag in `console_mcp.py`: `parse_shell_statements`
must keep `\` verbatim — `shlex.split` at `console_mcp.py:309`/`:334` is the
mangling point; mount-form `/c/…` → `C:\…` **for the verdict only**, argv
unchanged; drive-relative `C:x` refused by name via `path_norm.is_qualified`).
Then gb-03 (discovery candidates as patchable module attrs per the hermetic
convention, `shutil.which` search order `Git\usr\bin → Git\bin →
Git\mingw64\bin → PATH`, resolver verdict folded into `check_command_allowed`,
no-Git refusal naming the gap) and gb-04 (`windows_behaviour` marker in
`tests/conftest.py` beside the existing two, `tests/test_console_gbash.py`,
AGENTS.md clause). Manual pass (Task Manager check) closes the lot.
