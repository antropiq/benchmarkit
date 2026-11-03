# LSP finalise — make the declared language servers start on Windows

Three language servers are declared in `lsp_connectors/lsp_servers.json` (`pyright`,
typescript, vue), all `enable: true`, and **none of them starts**. This is not a bug in
the declaration file: the hub refuses them on purpose, in words, because it cannot
bound their reach on this filesystem. The lot finishes target B's LSP half so that
refusal stops being true and can be lifted honestly.

The authority for this lot is **`docs/windows-support/windows-support-target-B-connectors.md`**
(decisions B-D1 … B-D7 with their rejected alternatives, §6 verification, §7 "What B
must not break"). Target B1 (`windows-support-target-B1-console-gbash.md`) already
landed much of it for the **connector** side — see "What is already done" below, and
do not re-do it. The wider context is `windows-support-road_to_target.md` §3
(guardrails), §5 (B), §9 (the three rules). This file is the map and the cross-task
contract; it does not restate those docs.

## The refusal, stated exactly (verified on this machine)

`lsphub.discovery.lsp_reach_is_refused()` is the one predicate, and it is a
conjunction of two halves — both currently `False` here:

| half | source | why it is False on Windows |
|---|---|---|
| guardrails speak this path form | `mcphub.process.guardrails_speak_here(restricted_lsp_roots() or None)` | `restricted-paths.txt` holds only `/etc`-shaped entries; the working location is `D:\dev\benchmarkit`; `path_policy.lists_are_meaningful` compares *root forms*, so nothing speaks a drive path |
| the platform can bound a local child at all | `mcphub.process.local_reach_is_bounded()` → `os.name == "posix"` | the job-object teardown is not wired into this hub yet |

The refusal is checked at three gates — `_index`, `_start_one` (the gate every
attempt passes) and `start_all` — and each leaves the server at
`status: "error"`, `error: LOCAL_REACH_REFUSAL` ("its reach cannot be bounded on
this filesystem yet (see target B)"). The string is shared with `mcphub.constants`
and a test asserts the two hubs refuse in one voice. `_start_session` is never
reached, so no child is spawned. **All of this is correct behaviour today.**

## Why the refusal is right, and what makes it liftable

A guardrail list that matches nothing produces **no error** — the verdict falls back
to allowing. That is the failure class the whole Windows roadmap exists to remove
(target A's framing). So the two halves must become true **before** the predicate
is relaxed, in this order:

1. `restricted-paths.txt` must bound a *Windows* tree (`lspfinal-01`);
2. the server's process *tree* — tsserver, Volar workers — must be inside a
   boundary the OS maintains (`lspfinal-02`);
3. only then may the predicate drop its platform half (`lspfinal-04`).

**The hard ordering rule for this lot: never lift the seam before task 01 has
landed.** Verified consequence of doing it in the other order — the current entries
do not merely fail to match, `os.path.realpath` resolves them into *non-existent
drive-rooted folders*:

```
denied set: ['D:\\etc', 'D:\\root', 'D:\\proc', 'D:\\sys',
             'D:\\home\\vgr\\.ssh', 'D:\\home\\vgr\\.aws', 'D:\\home\\vgr\\.config\\gcloud']
C:/Windows               refused? False
C:/Users/vgr/.ssh        refused? False
C:/Windows/System32/config refused? False
```

A `workspaceRoot: "C:/Users/vgr/.ssh"` declaration would then start a language
server indexing a secret tree — silently, because the deny-list "was consulted".

## The blast radius of `restricted-paths.txt` (read this before touching it)

That one file is **not** only the LSP hub's input. `restricted_lsp_roots()` is read
by four consumers, all of which `realpath` each line:

- `lsphub.discovery.load_lsp_servers` — the `workspaceRoot` deny check;
- `lsphub.discovery.lsp_reach_is_refused` — the guardrail half of the seam;
- `editorfs.policy._check_workspace_root` — refuses to *browse* a location the app
  would have rejected;
- `POST /path-location` in `api/routes.py` — the same question, one rule.

So rewriting the list changes the Editor's verdicts and the settings route too.
That is intended and good (the guardrails become real on Windows), but task 01
owns all four consumers, and its acceptance includes them.

## What is already done — do not redo it

Target B1 landed the **connector** half of B; target C landed the terminal's job
wiring. Verified present on this branch:

- **B-D1 (connector lists)** — `path_policy` already derives Windows prefixes from
  the environment (`_SYSTEM_DENY_ENV`, `_env_absolute`, G-D8) and `prefix_hit` is
  already **component-wise** with an explicit `fold`. `DENY_PREFIXES` on this
  machine carries `C:\WINDOWS`, `C:\Program Files`, `C:\ProgramData`.
- **B-D2 (scratch allowance)** — `_SCRATCH_ENV` / `scratch_prefix_candidates`, and
  `ALLOW_PREFIXES` already includes this machine's `%TEMP%`.
- **B-D3 (the shlex backslash bug)** — done in `console_mcp`; a Windows path
  argument survives the split.
- **B-D4 (the command set)** — done as the **gbash toolset** (`Toolset`,
  `select_toolset`, `no_toolset_refusal`), which superseded B-D4's in-place lists.
- **B-D5 (the job object itself)** — `src/core/proctree_jobs.py` exists and is
  complete: `ProcessJob`, `assign`, `members`, `close`, `kill_job_tree`,
  `terminate_and_drain`, `jobs_available`. `proctree.py` already *dispatches* to it
  when `JOBS_SUPPORTED`. What is missing is the **LSP hub's** use of it — see 02.
- **The MCP hub's seam** — already lifted (`mcp_reach_is_refused()` consults the
  guardrail half only). `test_mcp.py::test_the_mcp_refusal_asks_only_the_guardrail_half`
  pins that, and `test_the_lsp_hub_still_asks_both_halves` pins the *asymmetry* this
  lot removes.

**Out of scope for this lot** (state it in the report rather than fixing it):
B-D6's `shutil.which` rule for **`console_mcp`**, and the MCP hub's own job
assignment — B1 lifted the MCP refusal on the argument that a connector's children
are its own short-lived direct children, so a connector can still spawn a grandchild
that outlives the app on Windows. That is a real gap; it is not what blocks the LSP
hub, and folding it in here would make the lot unreviewable.

## The tasks, and what each one can be judged on

| file | task | one-line verdict |
|---|---|---|
| `lspfinal-01_restricted_paths_both_forms.md` | B-D1's LSP half: the list gains Windows forms and `${…}` expansion at read time | `C:/Windows` and `${USERPROFILE}/.ssh` are refused as a `workspaceRoot`; Linux verdicts unchanged |
| `lspfinal-02_job_containment_at_spawn.md` | B-D5's LSP half: one job per language-server child, and one shared teardown call | no `os.name == "posix"` branch left in `_close_session`; a declared server's descendants are inside the job |
| `lspfinal-03_cmd_shim_resolution.md` | B-D6's LSP half: resolve the declared command through `PATHEXT`, refuse by name with the search stated | an extensionless Volta shim spawns; a missing command names what was searched |
| `lspfinal-04_lift_the_reach_seam.md` | drop the platform half from `lsp_reach_is_refused()` and update the tests that pin the asymmetry | on Windows, the three declared servers are no longer refused |
| `lspfinal-05_declarations_and_examples.md` | the committed example and this workspace's declarations in Windows form | `load_lsp_servers` reports no note and no error for the example file |
| `lspfinal-06_windows_sweep_and_docs.md` | the §6 manual check, the Linux regression run, the AGENTS.md clause | servers answer real LSP tools on Windows; nothing survives app exit |

## The cross-task contract

1. **The Linux verdict never moves.** Every task's acceptance includes the suite
   running on Linux with **zero new skips**; `tests/test_lsp.py` currently reports
   `88 passed, 5 skipped` on this machine and `test_proctree.py + test_path_policy.py`
   `64 passed, 4 skipped` — a new skip is a defect, per §6.
2. **Platform differences are asked of a capability probe, not `sys.platform`,**
   wherever a probe exists (roadmap §9 rule 2). `jobs_available()` and
   `fscase.case_folded()` are that probe; `guardrails_speak_here` is deliberately a
   *shape* question, so a deployment can fix it by writing values instead of the
   code fixing itself. Keep it that way — task 01 must pass with **no change to
   `lists_are_meaningful`**.
3. **One teardown, one parser, one sentence.** The hubs and the terminal keep
   sharing `proctree`/`proctree_jobs` and `path_policy`; a second copy is how they
   drift. `LOCAL_REACH_REFUSAL` stays one string shared by both hubs.
4. **No new dependency.** B-D5's decision is `ctypes`, not `pywin32`; `core` does
   not gain a runtime dependency.
5. **A refusal must stay a refusal.** Where a capability is genuinely absent, the
   server is left at `error` with a sentence naming what was looked for — never a
   partial start, never `shell=True`, never a silent fallback.
6. When a task is done, **move its file to `todos/done/`** — that move is how the
   batch loop sees completion, and it is the agent's own acknowledgement.

## Running this lot in batch mode

The batch lot prefix is **`lspfinal-` with the trailing dash**: the task number must
follow the prefix *immediately* (`core/batch.py::parse_task_number`), so typing
`lspfinal` puts all seven files in `skipped` with "no task number after the prefix".
Typing `lspfinal-` finds the big picture (`NN == 0`, injected into the system
prompt) and tasks 1…6 in order.
