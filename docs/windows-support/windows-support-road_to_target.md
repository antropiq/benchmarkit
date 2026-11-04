# Windows support — the road to target

**Status**: investigation complete, conclusions reached. Three plans follow this
document, one per target: `windows-support-target-A-no-crash.md`,
`windows-support-target-B-connectors.md`, `windows-support-target-C-terminal.md`.
**Code referenced**: `ef21ea3`. Line numbers are from that commit — they are
pointers to the thing, not the thing itself.

## 1. What this is actually about

The first Windows symptom was the `TEMP` key in `.env` colliding with the OS
variable (fixed in `6a397d6` → `ef21ea3`). That one is done, and it was the
visible tip of a wider fact: **benchmarkit is not a Linux app with a few POSIX
paths in it — several of its security invariants are implemented in terms of
POSIX text shapes.** The separators are the easy part. The hard part is that a
rule which stops matching produces no error.

The investigation found four independent classes of problem:

| class | what POSIX assumption it encodes | how it fails on Windows |
|---|---|---|
| **Path text** | a path is `/`-joined, normalised with `os.path.normpath` | rules stop matching — **silently permissive** |
| **Guardrails** | a protected path starts with `/etc`, `/usr`, `/home/<user>/.ssh` | the deny-lists match nothing — **silently permissive** |
| **Process trees** | `start_new_session=True` + `killpg` + `/proc` | descendants are orphaned and leak; `signal.SIGKILL` does not exist |
| **Terminal** | `pty`/`fcntl`/`termios` | (already honest — refuses with a named reason) |

**Conclusion: this is a security-correctness fix that happens to also enable
Windows, not a port.** Target A is not "make it not crash"; it is "make the
rules the app already promises keep holding on a case-insensitive, `\`-separated
filesystem, and refuse loudly wherever they cannot". A is the only target that
closes a hole. B and C add capability.

## 2. The finding that sets the priority

`editorfs.tree._rel_of` (`src/core/editorfs/tree.py:53`) builds the
workspace-relative form every rule is written against, `/`-joined. The frontend
agrees (`useWorkspaceTree.js:88`, `:535`, `useEditorRename.js:50` all `split('/')`)
and so does `.nojae/read-write-policy.json`, whose patterns are `/`-joined by
construction. The invariant is: **one textual form, `/`-joined, everywhere.**

That form is broken at exactly one place — the inbound lock:

```python
# src/core/editorfs/policy.py:198
normalised = os.path.normpath(text)   # → 'src\config.py' on Windows
```

From there everything downstream splits on `/` and gets *one* component. I ran
the real matcher (`mcp_connectors/policy_glob.py`) against that text to be sure
of the direction:

```
segments('src\config.py')                → ['src\config.py']      # 'dist/**' cannot match
segments('.nojae\read-write-policy.json')→ ['.nojae\…']           # reserved_component_hit → None
```

The consequences, in the order they matter:

1. **The reserved folder stops being reserved.** `policy_glob.reserved_component_hit`
   (`mcp_connectors/policy_glob.py:160`) and `_path_segments` (`:179`) split on
   `/`. AGENTS.md: *"`<working location>/.nojae/` is **reserved by code**: hidden
   from the tree, the read and git status"*. On Windows `path=.nojae\x` reaches
   the read gate, matches nothing, falls back to the default verdict, and **reads
   succeed**. The safety net that would have caught it — `_path_segments`
   returning `None` for an un-normalised path, whose docstring says answering
   "no match" is what keeps a skipped lock from being *reported* as a hiding rule
   — does not fire, because `src\config.py` has no empty or `.` component.
2. **The strict write gate degrades to the leaf-only gate.** `resolve_write_target`
   (`src/core/editorfs/policy.py:282`) walks ancestors with
   `normalised.split("/")[:-1]`, which is `[]`. So
   *"adding to it, emptying it, and moving something in or out of it are all
   refused"* (pol-00 D3) stops being true for every folder rule.
3. **Level A and B multi-component rules stop matching** — `**/.env**`, `logs/**`,
   `dist/**`, and every `ro` folder rule. Reads of hidden files succeed (breaking
   the "an observation of a hidden path is indistinguishable from absence"
   contract), writes into `ro` folders succeed.
4. **Create lands at the wrong depth**: `normalised.rpartition("/")`
   (`policy.py:370`) yields an empty parent, so a create names the workspace root.
5. **`os.sep` prefixes are fine, but not for the reason one would hope.**
   `policy.py:216`, `policy.py:172`, `lsphub/discovery.py:156`, `batch.py:228` and
   `routes.py:1298` build containment prefixes with `+ os.sep`, and both sides are
   `realpath`-resolved, so they agree on separator. They fail only on **case** and
   on drive/UNC form (§4). `path_policy.prefix_hit`
   (`mcp_connectors/path_policy.py:143`) hardcodes `+ "/"` against
   `Path(...).as_posix()` — which *is* consistent, and still POSIX-shaped (§3).

**The one good news in this class**: the failure that *is* loud is also real.
`mutate._rel_in_workspace` (`src/core/editorfs/mutate.py:459`) computes its prefix
as `workspace_root().rstrip("/") + "/"` — on `C:\…` that never matches
`resolved.as_posix()`, so `_subtree_refusal` answers "cannot be settled" and
**every folder delete is refused**. Loud, correct-by-accident, and it will look
like a bug report rather than a security issue. `search.py:113`
(`rel.rfind("/")`) makes the `.gitignore` honouring from `b55dd20` stop working.

**Fix direction (decided):** keep producing the `/`-joined form, and stop the
inbound lock from translating it. `normalise_rel` must canonicalise *to* `/`
(accept `\` from a Windows caller, emit `/` always), so every consumer — the
matcher, the ancestor walk, the payloads echoed to the browser, the frontend's
`split('/')` — keeps working unchanged. Fixing the producers instead would change
the wire format, the policy file format and the frontend together, for no gain.

## 3. The guardrails become no-ops

Three lists decide what a connector may reach, and all three are written in POSIX
text:

- `lsp_connectors/restricted-paths.txt` — `/etc`, `/root`, `/proc`, `/sys`,
  `/home/vgr/.ssh`, `/home/vgr/.aws`, `/home/vgr/.config/gcloud`. Note this file
  is also *machine-specific* (`/home/vgr`), a second reason it needs a Windows form.
- `mcp_connectors/path_policy.py:53-73` — `DENY_PREFIXES` (`/etc,/usr,/bin,…`),
  `DENY_PREFIXES_READ`, and `ALLOW_PREFIXES` (`/tmp,/var/tmp`) — the scratch-space
  allowance. On Windows the write deny-lists match nothing **and** the scratch
  allowance matches nothing, so behaviour is determined only by `DENY_COMPONENTS`,
  `DENY_FILENAMES` and the suffix lists, which do work because they look at
  `Path.parts` and the lowercased name.
- `console_mcp`'s `ALLOW_PREFIXES` check (`mcp_connectors/console_mcp.py:517`).

AGENTS.md: *"each connector's reach is bounded by a deny-list"*. On Windows that
sentence is false, and **nothing says so** — a tool call on a protected Windows
path is refused only if its name happens to contain `.ssh`, `shadow`, `.pem`.
This is the second silently-permissive class, and the reason B cannot be skipped
if connectors are to be enabled on Windows at all.

Two related decisions:
- **Component and name lists become case-folded on case-insensitive filesystems**
  (`DENY_COMPONENTS`, `DENY_COMPONENTS_SECRET`, `DENY_FILENAMES`,
  `DENY_FILE_SUFFIXES`) — `C:\Users\VGR\.SSH\id_rsa` must be refused.
- **A guardrail list that matches nothing is itself an error, not a default.** If
  the app cannot state its own reach on the platform it is running on, it must say
  so at boot and in the hub's notes. Silent zero-match is the failure shape this
  whole document exists to prevent.

## 4. Case: the one question that needed a decision

The matcher is deliberately case-*exact*:

> Comparison is exact — no case-folding, the temptation that would make `Dist`
> and `dist` one folder. — `mcp_connectors/policy_glob.py:226`

On Linux that is right. On NTFS, `Dist` and `dist` **are** one folder. So with
`dist/**` marked `hidden`, the tree lists `dist` (hidden: pruned), yet
`GET /api/fs/file?path=Dist/app.js` resolves to that same file and no rule
matches it — same for `.nojae` vs `.NOJAE` in the reserved check, and same for a
connector's `path` argument.

**Decision (recommended, accepted): make comparison case-insensitive and
drive-aware where the filesystem is case-insensitive — decided by a probe, not by
`sys.platform`.**

| alternative | why rejected |
|---|---|
| Rules are case-exact everywhere; document a Windows caveat | leaves a documented hole in `hidden`, in `ro` and in the reserved-folder reservation. Not acceptable for the reserved folder |
| Fold case unconditionally | breaks the Linux case the docstring names: `Dist` and `dist` genuinely are different folders there, and a rule would over-reach |
| `os.path.normcase` | folds *only* on Windows by platform — which is the same as the second row for a case-insensitive mount on Linux, and cannot be tested on the CI box |

The probe: compare `stat` of a created name against its folded variant in the
workspace, once per workspace-root switch, cached with the root (AGENTS.md already
requires that a location switch invalidates everything derived from it). Then
**fold only inside comparisons** — never in the stored, displayed, or wire form of
a path. Rules keep their exact text; matching gains a `casefold` on both sides.
CI on Linux keeps proving the logic by forcing both outcomes in tests.

## 5. Process trees: not a shim, a different mechanism

`src/core/proctree.py` (167 lines) is the teardown shared by both hubs and the
terminal, and it is entirely POSIX: `os.killpg` (`:59`, `:79`, `:92`, `:140`),
`signal.SIGKILL` (`:92`, `:167` — does not exist on Windows), and `session_members`
scanning `/proc/<pid>/stat` (`:110`, `:120`).

The subtlety that makes this a design item rather than an `if`:
`start_new_session=True` is **silently ignored** on Windows — CPython's Windows
`_execute_child` takes `unused_start_new_session`. So the invariant

> every child leads its own process group and is signalled by group, so the
> children a language server spawns of its own never outlive the app that started
> them

has no group to signal. `lsphub/manager.py:654` already guards with
`os.name == "posix"` and falls back to `process.terminate()` → `process.kill()`
(`:658-669`) — which kills **only the wrapper**. For
`typescript-language-server` (which spawns tsserver) and Volar (workers), that is
exactly the leak the comment above it describes: an orphaned indexer holding a
project index "no later hub will ever know about". `mcphub/manager.py:313-315`
goes further and assumes closing the transport stack is sufficient teardown,
which on Windows means no tree kill at all.

**Decision: the Windows equivalent is a Job Object with kill-on-close, not a
`killpg` shim.** A job is a different mechanism with a different lifetime model
(the boundary is the handle, not a signal), so `proctree` gains a second
implementation selected by capability, with the *same* escalation shape
(graceful, bounded wait, force) and the same injected reap step so the two stay
honest to the same contract. This belongs to B because it is only needed once
connectors and language servers run there.

**Landed (csl-02) as `src/core/proctree_jobs.py`** — `ctypes` only, no new
dependency — behind `proctree`'s two existing entry points, plus its
cross-platform liveness probe `process_is_gone`. The terminal is its second
caller (target C); the LSP hub still refuses on Windows until a further lot
lifts that guard.

## 6. The terminal is the model, not a problem

`termhub/session.py` imports `pty`/`fcntl`/`termios` under an
`ImportError → None` guard (`:47-53`) purely so the module *imports* on Windows,
raises `TerminalUnsupported` at `:137`, and the route answers `error` + close code
`CLOSE_CODE_UNSUPPORTED` (`api/routes.py:1655-1664`) — which `useTerminal.js:169`
renders as a sentence. That is the pattern every other subsystem should follow
when it cannot work: **refuse in words that name the reason.** A's job for the
terminal is only to keep it that way. C is the work to make it actually run
(ConPTY), and it is the only target that needs a new dependency.

**Both halves landed, and the sentence is still the model** (target C, the csl
lot): the platform-name guard is gone from `routes.py` and from `session.py`, and
what refuses is a **capability probe** — no available pseudo-console driver, or
no usable shell — each answering the same shape (`error` frame, close code 4501,
one sentence naming what was looked for), forced by tests on both platforms.
`useTerminal.js` renders it unchanged: C needed no frontend change.

## 7. What is already correct — do not let a "Windows pass" churn it

Verified, not assumed:

- `.env` round-trip: `_encode_value` (`src/config.py:620`) quotes values
  containing `\` and double-escapes them, so
  `BENCHMARKIT_PATH_LOCATION=C:\Users\vgr\dev\benchmarkit` survives
  `dotenv_values`. The `TEMP`→`TEMPERATURE` rename fixed the *name* collision;
  the value handling was already sound.
- **No silent CRLF damage**: reads return bytes decoded explicitly
  (`files.py:148`), `_atomic_replace` writes `"wb"`, and `fedit_mcp` *detects and
  preserves* the terminator (`detect_terminator` `:200`, `split_text` `:232`,
  `join_text` `:241`) and writes with no newline translation (`:1197`).
- `gitstatus._to_workspace_rel` (`src/core/editorfs/gitstatus.py:123`) already
  does `path.replace(os.sep, "/")` — it is the one place that got the direction
  right, and it is the template for the fix in §2.
- `fsync_dir` (`files.py:366-380`) swallows its `OSError` already, so opening a
  directory is not a Windows crash.
- The hubs run their own `asyncio.new_event_loop()` on a worker thread
  (`mcphub/manager.py:84`, `lsphub/manager.py:121`); on Windows that is the proactor
  loop, which supports subprocess pipes. `uvicorn[standard]` correctly excludes
  `uvloop` on win32. **Checked, no action.**

One thing to *record as a rule* rather than fix once: `load_dotenv_once`
(`src/config.py:257`) calls `load_dotenv` with no check that a key name does not
collide with an OS variable, and `SETTING_KEYS` (`:35`) has no such guard either.
`TEMP` was the first casualty; `TOP_P`, `MIN_P`, `REASONING` are safe by luck.
A's fix is the invariant, not the memory.

## 8. The three targets

Each target is a **promise the app can make out loud**, not a percentage of
compatibility. The line between them is drawn by capability boundaries that
already exist in the code (the terminal's refusal is the template), so nothing in
a lower target pretends to be a higher one.

| | **A — no-crash** | **B — connectors** | **C — terminal** |
|---|---|---|---|
| **The promise** | Benchmark, WorkBench chat and the Editor's file surface work correctly on Windows; everything that cannot work says so in words | + local MCP connectors, the file/console connectors and language servers actually work, with their reach bounded on Windows terms | + the Editor's terminal runs a real shell |
| **Closes** | §2 (path text, case, the reserved folder), §3's *honesty* half (a zero-match guardrail is reported, and connectors default off), §4, §7's rule | §3 in full, §5 (Job Objects), `console_mcp`'s tokenizer and command set | ConPTY: `pty`/`fcntl`/`termios`, `TIOCSWINSZ`, the sizing/pump model |
| **New dependency** | none | none (ctypes/Win32 for jobs, or `pywin32` — decided in B) | one (`pywinpty`-class) |
| **Size** | small and surgical: one normalisation point, one case decision, a refusal path, the test gate | medium: a redesign of `proctree` and `console_mcp`'s command model | medium, isolated to `termhub/` |
| **Blocks the other** | B and C both sit on A — a connector whose path policy does not match is unsafe to enable | C sits on B's `proctree` work (a terminal session is a process tree) | — |

**How each landed** (a correction to the table, not an annotation beside it):
**A** landed as the win lot. **B** landed only for the MCP hub — target B1, the
console connector over Git's toolset — so B's promise is **not** fully kept:
language servers still refuse on Windows, and lifting that is a further lot with
its own verification. **C** landed as the csl lot, with **C-D2 amended**: the
terminal's shell is Git Bash over ConPTY, not `cmd.exe`, and the `windows` extra
(`pywinpty`) is the one new dependency the C column predicted. B's open
dependency question settled as `ctypes` against `kernel32` — no package for the
job half.

**Order, and why.** A first, always: it is the only target that closes a hole,
and B is *unsafe* without it — enabling a connector whose path policy silently
matches nothing means giving an agent an unbounded surface. Then B, then C, and
C can start as soon as B's `proctree` half is done, since that is its only
dependency on B.

After A, B and C are developed **on a Windows machine** (agreed), which is the
only way to test job objects, ConPTY and NTFS semantics for real. The
counterweight is that the platform-specific code must stay importable and
testable on Linux, so the Linux suite never goes blind — see §9.

## 9. The rule that keeps Linux from regressing

A and B change the code that decides what every Editor surface may touch, so the
existing Linux behaviour is the thing most at risk. Three rules, stated once here
and referenced by each plan:

1. **The canonical path form is `/`-joined workspace-relative text on every
   platform.** Nothing in the wire format, the policy file format or the frontend
   changes for Windows; only the *inbound* lock gains the ability to accept `\`.
   A Windows-only test cannot break a Linux path, because there is no
   Windows-only path form in the system.
2. **Platform differences are selected by capability probes, not by
   `sys.platform`, wherever a probe is possible** (case-sensitivity: §4; the
   guardrail zero-match check: §3). Where it is not possible (job objects,
   ConPTY), the difference sits behind one named seam, and the *other* platform's
   tests keep running unchanged.
3. **A refusal is a tested behaviour, not a fallback.** Every "unsupported" path
   added by A gets a test that asserts the sentence and the status/close code, so
   B and C cannot turn a refusal into a crash without failing something on Linux.

The gate itself (which tests must pass on Windows, which are `skipif`-guarded and
why, Linux staying green) is specified in the A plan, because A is where the test
suite becomes runnable on Windows at all: 8 test files create symlinks (24 call
sites — `test_editorfs_write.py` 9, `test_fedit_policy.py` 4,
`test_editorfs_search.py` 3, `test_editorfs.py` 2, `test_editorfs_policy_file.py` 2,
`test_batch.py` 2, `test_api.py` 1, `test_editorfs_builtin_policy.py` 1), and
`test_console_policy.py` / `test_fedit_policy.py` assert POSIX-only facts
(`/etc/shadow` at `test_fedit_policy.py:50`, `chmod`/`st_mode` at `:166-176`).
Today `pytest` on Windows is not a usable signal at all — which is precisely the
risk in developing B and C there.

## 10. Out of scope for the whole roadmap

- **Anything POSIX-specific beyond the terminal.** Signals other than
  TERM/KILL, `st_mode` permission bits as a security control, fifos/sockets as
  Editor buffers (`files.py:136` already refuses non-regular files), `ln`/`readlink`
  in the console command set. Windows has no equivalent the app would use; the
  behaviour degrades to a refusal, not to a guess.
- **Windows-only conveniences**: drive letters as a concept the UI exposes,
  UNC-path UX, "Open with benchmarkit" shell integration, a Windows installer.
- **Serving the app under IIS, or any non-`uvicorn` host.**
- **A Windows CI runner.** The plans assume a Linux runner plus a developer's
  Windows box, with the A plan's marker conventions making that safe. Adding a
  Windows job is an option, not a requirement.
- **Windows ARM and Server Core.** The target is Windows 10/11 desktop with
  CPython and Node, as AGENTS.md already scopes the app to desktop browsers.
- **"Git Bash / MSYS as a runtime" — corrected: this excluded the app itself
  running on MSYS, not Git's binaries being used.** The target is still native
  Windows CPython and Node: nothing about the app is hosted under an emulation
  layer, and its path model stays the Windows one. What the two shipped lots do
  use Git for Windows for is (a) the `console` connector's **tool binaries**
  (target B1 — `ls.exe`, `grep.exe`, … `Popen`'d directly, never an interpreter)
  and (b) the Editor terminal's **shell** (target C's amendment to C-D2, Git Bash
  over ConPTY). Both come from the one install probe, `mcp_connectors/gitbash.py`.
  The console tool still executes no shell. Launching the app *from* a Git Bash
  window is ordinary too: `termhub/shells.py` refuses a POSIX-shaped
  `TERMINAL_SHELL` by name rather than spawning it, and hands the child Git's
  directories prepended to `PATH` in Windows form.

## 11. How this document set is verified

Not by a test. These are plans, and their checks are:

- every claim names a file and a line that exists at `ef21ea3` (the four claims
  about behaviour were confirmed by executing the code — §2's matcher run, the
  `ntpath.isabs`/`normpath` semantics, the `shlex` backslash mangling, and the
  `_encode_value` round-trip);
- the A/B/C boundaries match capability boundaries already present in the code
  (§6), so no target requires a refusal to be redefined later;
- each plan is independently executable, and each states what it must not break.
