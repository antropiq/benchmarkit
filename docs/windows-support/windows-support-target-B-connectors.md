# Target B — connectors, language servers and process trees on Windows

**Status**: planned, not started. **Depends on**: all of target A
(`windows-support-target-A-no-crash.md`) — non-negotiable, see §1.
**Blocks**: C (its `proctree` half). **Read first**: roadmap §3 and §5.
**Code referenced**: `ef21ea3` + A's changes.

## 1. What B is for

B makes the local MCP connectors (`console`, `fedit`), the Playwright connector
and the language servers actually work on Windows — **with their reach bounded on
Windows terms** — and replaces the POSIX process-tree teardown with a mechanism
that has a Windows equivalent.

B is unsafe without A, and this is the one hard ordering rule in the roadmap.
Today on Windows the connector deny-lists match nothing (roadmap §3): a `local`
connector's path policy is a no-op. A turns that into a refusal; B turns it into a
real bound. **B never starts on a tree where A's refusal is still in place but the
matcher is not fixed**, because the first thing B enables is a tool with filesystem
reach.

AGENTS.md's invariant B must make true again:

> each connector's reach is bounded by a deny-list —
> `lsp_connectors/restricted-paths.txt` for paths,
> `mcp_connectors/restricted-fqdn.txt` for hosts

and this one:

> every child leads its own process group and is signalled by group, so the
> children a language server spawns of its own never outlive the app that started
> them

B changes the *mechanism* that sentence describes (a job, not a group) without
letting the guarantee weaken.

## 2. Decisions taken

### B-D1 — Guardrail lists gain a Windows form, and the format stops assuming `/`

`mcp_connectors/path_policy.py:53-73` and `lsp_connectors/restricted-paths.txt`
are the two lists, and both are POSIX text. `restricted-paths.txt` is worse than
POSIX-shaped: it is *machine*-shaped — `/home/vgr/.ssh`, `/home/vgr/.aws`,
`/home/vgr/.config/gcloud` (lines 12-14 of that file).

**Decision**: keep one file and one list format, and make its entries
*pattern-capable over both forms* by resolving them the same way the app resolves
a path — through A's `path_norm` — rather than adding a second file per platform.
Concretely:

- `DENY_PREFIXES` / `DENY_PREFIXES_READ` gain Windows system prefixes
  (`C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`, and the
  `%ProgramData%` tree) expanded **at read time** from the environment, reusing
  the hub's existing `${NAME:-fallback}` expansion (AGENTS.md: *"The declaration
  files accept `${NAME}` / `${NAME:-fallback}`, expanded at read time by each
  hub's `discovery` module"*). `restricted-paths.txt` already takes one prefix per
  line with `#` comments, and its entries pass through the same expansion.
- The `/home/<user>/.ssh`-style entries become `${HOME}/.ssh` and
  `${USERPROFILE}/.ssh`, so the file is machine-neutral and both platforms are
  expressed once. This also removes a stale personal path from a committed file.
- Matching uses A's `path_norm.path_parts` + the A-D3 case fold, so
  `c:\windows\system32` and `C:\WINDOWS\System32` are the same prefix, and a
  prefix is compared **component-wise, not by string prefix**. The current
  `prefix_hit` (`mcp_connectors/path_policy.py:143`,
  `text.startswith(prefix.rstrip("/") + "/")`) is a text-prefix test: correct for
  the `/`-normalised form it is fed, but handed raw Windows text it would treat
  `C:\Windows.old` as being under `C:\Windows`. Component-wise comparison removes
  that whole class.

*Rejected — a `restricted-paths.windows.txt` beside the existing file.* Two lists
means two sources of truth, and AGENTS.md is explicit that the declaration files
are *the only source of truth for what runs*. A user editing one would not know
which one the running hub read.

*Rejected — leave them POSIX and rely on `DENY_COMPONENTS`/`DENY_FILENAMES`.*
Those (`path_policy.py:79,84,98`) do work on Windows because they inspect
`Path.parts` and a lowercased name — but they name *names*, not *locations*. They
cannot express "not the OS directory", which is half the point of the write
deny-list.

### B-D2 — The scratch allowance needs a Windows answer, and it must stay narrow

`ALLOW_PREFIXES = "/tmp,/var/tmp"` (`path_policy.py:53-55`) is the "throwaway work
skips the policy" allowance, and `console_mcp.py:517` checks it the same way. On
Windows nothing matches it, so today the *allowance* fails closed while the
*deny-list* fails open — the two halves of one rule moving in opposite directions,
which is why the behaviour looks random rather than broken.

**Decision**: derive the scratch prefixes from the OS (`%TEMP%`/`%TMP%`, falling
back to a directory the app owns under the user profile) — and **keep them
narrower than the user profile**. The Windows temp directory is
`C:\Users\<user>\AppData\Local\Temp`, i.e. *inside* the profile, so a naive "allow
the temp dir" grants a subtree of a tree the protected-component list guards. The
allow-check therefore keeps its current ordering relative to the component and
name checks (`deny_reason` at `path_policy.py:161` tests prefixes, then
components, then names), and the Windows temp path must not be allowed ahead of
them.

**Tests that decide this rather than assume it**: a write under `%TEMP%` is
allowed; a path traversing from temp into a protected store is refused; a path
*named* inside temp but which `realpath`-es elsewhere (a junction) is refused —
the same rule AGENTS.md states for the Editor's reads, applied to the connectors.

### B-D3 — `console_mcp`'s tokenizer must stop eating backslashes

The sharpest bug in B, because it corrupts input *before* any policy decision.
Verified:

```
shlex.split(r'cat C:\Users\vgr\a.txt')  →  ['cat', 'C:Usersvgra.txt']
```

`shlex` treats `\` as an escape. `console_mcp` builds argv with it twice
(`mcp_connectors/console_mcp.py:309` — `shlex.shlex(..., punctuation_chars=...)` —
and `:334` — `shlex.split(token)`), so every Windows path argument arrives mangled.
The consequences are worse than a failed command:

1. The **path-scope decision** is made on the mangled text. This connector's model
   is deliberately *not* a whitelist: *"the root is the operator's own workspace,
   so a command whose path arguments all resolve inside it runs without a list
   check"* (`console_mcp.py:102-107`). `C:Usersvgra.txt` is not absolute, so it
   resolves *relative to the root* and reads as "inside the root" — an
   out-of-root path can be judged in-root.
2. `Popen(argv, shell=False)` (`:1061-1067`) then receives a path that is not the
   one the user wrote.

**Decision**: a Windows-aware split for the *argument* layer, keeping the existing
statement/punctuation grammar untouched. Quoted arguments keep their literal text;
an unquoted `\` is a literal character, not an escape. The pipe/chain/redirect
detection (`_PUNCTUATION_CHARS`, `:214-219`) is unaffected — it works on unquoted
metacharacters, which is exactly where `\` matters least.

The connector's deliberate no-shell rule stays (AGENTS.md: *"the `console`
connector execs one argv with no shell (pipes, chains, redirections and `$VAR`
expansion are refused, `check_command_allowed` is the same parse)"*). B must not
"support Windows" by routing through `cmd.exe /c`: that would delete the grammar
the policy is built on and make `check_command_allowed` a different parse than
`execute_command` — the two are the same verdict today, and that equivalence is
what makes the checker useful.

### B-D4 — A command set that exists, and a resolver that says why when it does not

`ALLOWED_COMMANDS` (`console_mcp.py:153-179`) is a POSIX vocabulary: `ls`, `grep`,
`sed`, `awk`, `find`, `basename`, `dirname`, `readlink`, `ln`, `md5sum`, `sha1sum`,
`vmstat`, `lscpu`, `lsb_release`, `ss`, `dig`, `tracepath`, `env`. Most have no
`.exe` counterpart on Windows; `dir` (`:155`) *looks* Windows-native but is a
`cmd.exe` builtin with no `dir.exe`, so under `shell=False` it raises
`FileNotFoundError` → "Command not found" (`:1068-1069`).

**Decision**: two named, disjoint sets — the existing one and a Windows one —
selected by which platform's binaries the connector can actually resolve, and
`check_command_allowed` reports the same verdict `execute_command` would give (the
`FileNotFoundError` equivalent folded into the answer, not swallowed).
Windows vocabulary to add: `findstr`, `where`, `type`, `more`, `sort`, `fc`,
`certutil` (checksums), `attrib`, `systeminfo`, `tasklist`, `ipconfig`, `nslookup`,
`ping`, `curl`, `tar`. PowerShell-style tools (`Select-String`) are **out**: they
would have to be launched as `pwsh -Command`, and a shell is not allowed.
The Windows `BLOCKED_COMMANDS` mirror (`:113-134`): `del`, `erase`, `rd`, `rmdir`,
`format`, `takeown`, `icacls`, `cacls`, `net`, `sc`, `reg`, `schtasks`, `shutdown`,
`runas`, `cipher`.

`dir` leaves `ALLOWED_COMMANDS` on **both** platforms: it is either a builtin
(Windows) or a shadow of `ls` (coreutils), and keeping it in the list creates the
confusion the comment at `:108-109` already warns about (*"entries in both are
effectively blocked and only create confusion"*).

`ln`/`readlink` stay POSIX-only; on Windows the connector refuses them by name
rather than substituting `mklink`, which is a `cmd` builtin and therefore refused
by the no-shell rule.

### B-D5 — Process trees: Job Objects, with the same escalation contract

`src/core/proctree.py` is the shared teardown (167 lines) and is entirely POSIX:
`os.killpg` (`:59,79,92,140`), `signal.SIGKILL` (`:92,167`, which does not exist on
Windows), and `session_members` reading `/proc/<pid>/stat` (`:110,120`). The reason
it cannot be shimmed: `start_new_session=True` is **silently ignored** by CPython on
Windows, so there is no group to signal, and `lsphub/manager.py:654`'s
`os.name == "posix"` guard already falls back to terminate/kill of the **wrapper
only** (`:658-669`) — leaving tsserver and Volar workers orphaned, which is exactly
what the comment above it (`:644-652`) says must not happen. `mcphub/manager.py:313-315`
assumes closing the transport stack is the whole teardown, which on Windows means no
tree kill at all.

**Decision**: a Windows implementation using a **Job Object with
`JOB_OBJECT_LIMIT_KILL_ON_CLOSE`**, one job per hub child (and per terminal session
in C), assigned at spawn before the child can spawn anything. Why a job and not a
pid walk: job membership is maintained by the OS, so a grandchild the language
server starts is already inside the boundary — the exact property the AGENTS.md
invariant asks for, and one an enumeration can only approximate.

The module keeps its contract, not its syscalls: the same entry-point signatures,
the same "graceful → bounded wait → force" escalation, the same injected reap step
(`ReapLeader`), so the callers in `lsphub/manager.py` and `termhub/session.py` do
not fork their own logic.

```
proctree.py        the contract + the POSIX implementation (as today)
proctree_jobs.py   (new) the Windows implementation, same two entry points,
                   selected once at import
```

`kill_session_tree` has no Windows analogue and must not be faked: for a hub child,
the job boundary *is* the session boundary, so the Windows implementation maps
session teardown onto the job and documents that the "job-controlling shell" case
belongs to the terminal (C) rather than being emulated here.

**Dependency decision (open — settle it on the Windows machine before coding)**:
`ctypes` against `kernel32`'s job API, or `pywin32`.

| | ctypes | pywin32 |
|---|---|---|
| extra dependency | none | one, in the `[mcp]`/`[lsp]` extras |
| surface to hold | `CreateJobObjectW`, `AssignProcessToJobObject`, `SetInformationJobObject`, `OpenProcess` | already bound |
| failure mode | wrong struct layout, silent | import error, loud |

Recommendation: **ctypes**, isolated to one small module with the structs written
out and a test that the job really kills a two-level tree — because adding a
runtime dependency to `core` (which the hubs are part of, not an extra) is the more
expensive choice.

### B-D6 — Spawning: what the code must not assume

- The hub event loops (`mcphub/manager.py:84`, `lsphub/manager.py:121`) create a
  `new_event_loop()` on a worker thread; on Windows that is the proactor loop,
  which does support subprocess pipes. **No action** (roadmap §7) — but a test
  asserting an LSP stdio session works on that loop protects an assumption that is
  currently invisible.
- Two places a real subprocess is born: `lsphub/manager.py:274` (with
  `start_new_session=True`) and the MCP `StdioServerParameters` at
  `mcphub/manager.py:148-157`. Both need the job assignment at the same moment.
- A declared command needs `.cmd`/`.bat` awareness: `pyright-langserver` is a `.cmd`
  shim under npm/volta, and under `shell=False` a bare name with no extension does
  not resolve to `.cmd`. **Decision**: resolve through `shutil.which` at spawn
  (which honours `PATHEXT`), and refuse with a sentence naming what was searched
  when resolution fails — never fall back to `shell=True`.

### B-D7 — `fedit_mcp`'s mode handling on NTFS

`fedit_mcp.py:1176-1186` preserves `st_mode & 0o7777` through `os.chmod`, which is a
near-no-op on NTFS; `editorfs/files.py:404` has the same shape. The *backup*
behaviour AGENTS.md promises (*"a save backs up what it replaces, a delete moves the
file to a dated batch"*) is rename-based and works.

**Decision**: B keeps the `chmod` calls (correct and load-bearing on POSIX) and adds
nothing for NTFS attributes, **except** that a file whose read-only attribute is set
must not be silently overwritten: the atomic replace surfaces the OS refusal as the
connector's own refusal sentence. The app has no way to represent that attribute in
its policy model, so this is the one Windows file behaviour B makes visible rather
than invisible.

## 3. What it builds on

- A's `path_norm` + `fscase` (the component splitter, the fold, the containment
  helper). B adds no path semantics of its own.
- The shared matcher/refusal architecture (pol-08): `policy_glob` in
  `mcp_connectors/`, reached by both stdio connector processes and the app, so the
  Windows prefix rules are written once.
- The `${NAME:-fallback}` expansion in `mcphub/validation.py`, run *before*
  validation and keeping an unresolved name verbatim with a note — which is how
  B-D1's env-derived prefixes arrive without putting machine paths in a committed
  file.
- `proctree`'s existing injected-reap design, which is what lets a second
  implementation share the escalation instead of duplicating it.
- `proctree.py:100-102`'s documented exception (*"A job that `setsid`s out … leaves
  the session and so opts out by design"*) — B states the Windows analogue honestly:
  a process that escapes the job escapes the teardown, and under a job that is an
  explicit act rather than an accident.

## 4. Shape and files of the pieces

| piece | change | what it holds |
|---|---|---|
| `mcp_connectors/path_policy.py` | edit | Windows prefixes; component-wise prefix matching; `lists_are_meaningful` becomes a real check instead of a warning |
| `lsp_connectors/restricted-paths.txt` | edit | both forms via `${HOME}`/`${USERPROFILE}` plus the Windows system prefixes; drops the `/home/vgr/*` lines |
| `mcp_connectors/console_mcp.py` | edit | the argument splitter (B-D3), the Windows command set (B-D4), `shutil.which` resolution before `Popen` |
| `src/core/proctree_jobs.py` | **new** | the job-object implementation of the same two entry points |
| `src/core/proctree.py` | edit | selects the implementation at import; contract docstring rewritten to state the *guarantee*, not `killpg` |
| `src/core/lsphub/manager.py` | edit | drops the `os.name == "posix"` branch at `:654` in favour of the shared call; job assignment at spawn |
| `src/core/mcphub/manager.py` | edit | job assignment for a `local` child; `:313-315`'s comment replaced by the real teardown |
| `src/core/lsphub/discovery.py` | edit | `restricted_lsp_roots` compares component-wise; A-D4's refusal lifted |
| `src/config.py`, `.env.example` | only if needed | if a new prefix needs a managed key, check it against the `SETTING_KEYS`/`_render_env`/template rule *and* A-D6's reserved-name rule before adding one |

## 5. Out of scope for B

- The Editor terminal (C). A `termhub` session on Windows still refuses after B —
  B only supplies the teardown C will need.
- Signals beyond graceful/force. There is no Windows equivalent of SIGTERM to a
  group and B does not invent one: a process that ignores the closing of its job is
  killed, not persuaded.
- `st_mode`/ACL preservation, and junction or symlink *creation*. Junctions appear
  in tests at most; B adds no junction feature to the Editor.
- PowerShell as a command interpreter (rejected in B-D3 and B-D4: it is a shell).
- Transport differences inside the MCP and LSP layers: both hubs use stdio, so
  there is nothing to do. Stated so nobody "fixes" it.

## 6. Verification

B is developed **on Windows**, so its tests are the first that can only pass there.
Roadmap §9's rule stays: the platform difference sits behind one named seam, and the
Linux suite keeps running unchanged.

- **A's gate is already in place** (its two markers); B's Windows-only tests add a
  third marker, `windows_behaviour`, and A's conventions keep the Linux skip count
  at zero.
- New tests, Windows: the job kills a two-level tree (a wrapper that spawns a child
  that spawns a grandchild — the tsserver shape), while the Linux equivalent test
  still passes unchanged; a Windows path argument survives the tokenizer
  (`cat C:\Users\vgr\a.txt` → argv carries the path verbatim); the path-scope
  decision is made on un-mangled text; `%TEMP%` is allowed and a protected store
  under the profile is refused; `c:\windows\x` is refused case-insensitively; `dir`
  is refused by name with a sentence, not by `FileNotFoundError`; a `.cmd` shim
  resolves via `PATHEXT`.
- New tests, both platforms: `restricted-paths.txt` expands `${HOME}`/
  `${USERPROFILE}` and still refuses an absolute system root as a language-server
  `workspaceRoot`; `lists_are_meaningful` is true on both platforms — the test that
  says A's note is no longer needed.
- **Regression check on Linux, every session** (the cadence the user asked for):
  `pytest` green with zero skips; `cd web && npm test` green;
  `cd web && npm run test:e2e` green wherever `.env.test` reaches an endpoint. A new
  skip on Linux is a defect in B, not a fact about Windows.
- Manual, on Windows: a benchmark run, one WorkBench turn with `fedit` and `console`
  both enabled, one language-server check, then quit the app and confirm in Task
  Manager that no `node`, `pyright` or connector process survived. That last check
  is the whole point of B-D5 and no unit test on Linux can make it.

## 7. What B must not break

The POSIX teardown behaviour, verbatim: SIGTERM before force, the bounded grace
period, and the leader reaped *before* the tree is probed (the reason
`proctree.py:11-14` says `getpgid()` could not be used). And the rule that the hubs
and the terminal share **one** teardown — a second copy is how they drift.

Two sentences B must still be able to say truthfully afterwards: *"`check_command_allowed`
is the same parse"* (so the checker is a real oracle), and *"each connector's reach
is bounded by a deny-list"* (so a list that matches nothing is a defect forever, not
a platform quirk).

**AGENTS.md**: the process-tree bullet moves from *"every child leads its own process
group and is signalled by group"* to the platform-neutral form — *every child is
born inside a boundary the OS maintains, and teardown closes that boundary, so no
descendant outlives the app*. The tool-read-only and deny-list bullets gain "on both
platforms' path forms". Nothing else moves; stay under ~300 lines.
