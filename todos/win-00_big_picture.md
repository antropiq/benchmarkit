# Windows support, Target A — "correct everywhere, honest about the rest"

Make benchmarkit **correct on Windows for the surfaces it can already support** —
Benchmark, WorkBench chat, the verifier, and the Editor's file surface (tree,
read, search, git status, diff, create/rename/delete/backup) — and make every
surface it **cannot** support say so in words instead of running degraded.

The authority for this lot is **`docs/windows-support/windows-support-target-A-no-crash.md`**
(374 lines: decisions A-D1 … A-D8 with their rejected alternatives, the file and
line pointers, the behaviour tests of §6, and "What A must not break" in §7).
Read it before your task — every task file below cites the decision numbers it
implements. The wider context is `docs/windows-support/windows-support-road_to_target.md`
(§2 path text, §3 guardrails, §4 case, §9 the three rules). This file is the map
and the cross-task contract; it does not restate those docs.

## The framing that makes this a security lot, not a port

Several of the app's security invariants are implemented as **POSIX text
shapes**. The separators are the easy part. The dangerous part is that a rule
which stops matching produces **no error**: the verdict falls back to `rw`. So
on Windows, today:

- the reserved `.nojae` folder **stops being reserved** — a read of `.nojae\x`
  succeeds (roadmap §2.1);
- the strict write gate degrades to leaf-only — an `ro` folder stops protecting
  what is **inside** it (§2.2);
- level A and level B multi-component rules (`**/.env**`, `logs/**`, `dist/**`)
  stop matching at all (§2.3);
- the connector deny-lists (`/etc`, `/usr`, `/home/vgr/.ssh`) match **nothing**
  and nothing says so (§3).

Three of the four failure classes are therefore **silently permissive**. A is
the only target that closes a hole; B (connectors) and C (terminal) add
capability, and both sit on A — a connector whose path policy matches nothing
is unsafe to enable.

## The shape of the fix, in one paragraph

The canonical path form stays **`/`-joined workspace-relative text on every
platform**. Translation happens at exactly one place — the inbound lock
`editorfs.policy.normalise_rel` — which starts accepting `\` and always emits
`/`. Nothing downstream changes its mind about the wire format, the policy file
format, or the frontend's `split('/')`. Then the four places that consumed the
form as if it could contain `\` go through one shared component splitter; the
containment compares go through one helper that folds case **only when the
filesystem folds it** (a probe, not `sys.platform`); and the shapes that slip
past `os.path.isabs` on Windows (`/etc/x`, `//srv/share`, `C:x`) are refused by
name. Guardrails A cannot have on Windows (the local connectors, the language
servers) are refused at start, with a sentence naming target B — the shape the
terminal already uses (`termhub/session.py:137` → an `error` frame +
`CLOSE_CODE_UNSUPPORTED`).

## The three roadmap §9 rules — every task is bound by these

1. **The canonical path form is `/`-joined workspace-relative text on every
   platform.** Only the *inbound* lock gains the ability to accept `\`. A
   Windows-shaped input never reaches a payload, the tree, `results.json` or the
   browser as `\`.
2. **Platform differences are selected by capability probes, not by
   `sys.platform`, wherever a probe is possible** (case-sensitivity, the
   guardrail zero-match check). Where a probe is impossible, the difference sits
   behind one named seam and the *other* platform's tests keep running.
3. **A refusal is a tested behaviour, not a fallback.** Every "unsupported" path
   this lot adds gets a test asserting the sentence *and* the status — so B and
   C cannot turn a refusal into a crash without failing something on Linux.

## What is shared between tasks (the contract)

| name | lands in | used by |
|---|---|---|
| `mcp_connectors/path_norm.py` — `canonical_rel(text)`, `is_qualified(text)`, `path_parts(text)` | win-02 | win-03, win-04 |
| `editorfs.policy._is_within(parent, child)` — one containment test, fold-aware | win-06 (its first version is exact: same verdict as today) | win-06's five call sites; win-05 only *threads* `fold`, win-06 *applies* it |
| `src/core/editorfs/fscase.py` — `case_folded(workspace_root) -> bool`, cached per root | win-05 | win-05's verdict path |
| `policy_glob` gains a keyword `fold: bool` (default `False`) on `_name_matches`, `reserved_component_hit`, `pattern_matches` | win-05 | win-05, win-07 |
| `path_policy.lists_are_meaningful(text) -> bool` — "does any of these prefixes speak this path's form?" | win-07 | win-07's hub check |
| `@pytest.mark.requires_symlink` / `@pytest.mark.posix_behaviour` in `tests/conftest.py` | win-01 | every task that adds or touches a test |

`path_norm` sits in `mcp_connectors/`, **not** under `src`, for the same reason
`policy_glob` and `policy_shared` do: the two stdio connectors run as scripts
launched by absolute path and cannot `import src`. Keep it stdlib-only and free
of any `sys.platform` branch — it is pure text.

**Fold threading, decided (not in the doc):** the app-side verdict path is the
only thing that passes a real `fold`. The connector call sites
(`mcp_connectors/policy_file.workspace_policy_reason`, `fedit_mcp`,
`console_mcp`) keep the default `fold=False` — Linux-exact, and honest there,
because A refuses those connectors on Windows anyway; B supplies real values.
`fold` is **always an explicit argument**, never read from a global, so both
outcomes stay testable on the Linux runner (rule 2).

## Task order

| # | task | doc ref | depends on |
|---|---|---|---|
| win-01 | The pytest platform gate: two markers, 24 symlink sites, POSIX-only fixtures | §6 "the gate first" | — |
| win-02 | `mcp_connectors/path_norm.py`: three pure functions + unit tests | §4 piece 1 | — |
| win-03 | `normalise_rel`: accept `\`, emit `/`, refuse the three Windows shapes | A-D1, A-D2.1 | win-01, win-02 |
| win-04 | The four `/`-splitting consumers: ancestor walk, create parent, search prune, batch lot dir | A-D1 table | win-03 |
| win-05 | Case: the `fscase` probe + `fold` threaded through the matcher and the verdict | A-D3 | win-01, win-03 |
| win-06 | Containment: one `_is_within` at all five compare sites; folder delete starts working | A-D2.2, A-D5 | win-05 |
| win-07 | Hub honesty: `lists_are_meaningful`, and `local`/LSP entries refused on Windows naming target B | A-D4 | win-01, win-06 |
| win-08 | The honesty chores: reserved env names, the two template notes, Playwright OS checks | A-D6, A-D7, §6 frontend | win-01, win-03 |
| win-09 | Docs: the README Windows section, the one AGENTS.md clause | A-D8, §7 | win-01 … win-08 |

01 and 02 can run in parallel; the rest needs 01 (tests) and 02/03 (code).
**09 is last**: it documents sentences that only exist once 01-08 land, and
AGENTS.md is injected into every WorkBench prompt — wrong there is worse than
missing.

## Decisions already taken (do not re-open them)

All eight are in the A doc with their rejected alternatives. The two easiest to
undo by accident:

- **A-D1**: the canonical form is `/`-joined. Rejected: normalise to the OS
  separator — it would change the wire format, `results.json`, the policy file
  format and the frontend in one go.
- **A-D3**: fold case where the *filesystem* folds it, decided by a probe.
  Rejected: `os.path.normcase` (keys off the platform, wrong for a
  case-insensitive mount on Linux, untestable on the Linux runner) and folding
  unconditionally (breaks `Dist` vs `dist` on Linux, which the
  `policy_glob.py:226` docstring names out loud — that docstring is updated by
  win-05, not defended).

## Out of scope for this lot (do not add it)

- Anything that makes a connector, a language server or the terminal **work** on
  Windows — that is B and C. A's deliverable for them is a refusal with a name.
- Job Objects, `proctree`'s second implementation, `console_mcp`'s tokenizer and
  command set (B). ConPTY (C). Windows guardrail *values* (B) — A only reports
  that its POSIX ones speak nothing.
- Preserving NTFS attributes/ACLs through the atomic replace, or mapping
  `st_mode` to the read-only attribute. A's only promise here is that the
  **Linux** mode round-trip (`st_mode & 0o7777`) keeps working.
- Case-insensitive *git* path matching: `gitstatus.py:123`
  (`path.replace(os.sep, "/")`) is already right and the fold must not reach it.
- `.nojae` **seeding** semantics: `os.open(..., O_CREAT | O_EXCL)`
  (`agent_modes.py:887`, `api/server.py:35-58`) is exclusive-create on both
  platforms and needs no change. A's reserved-folder work is the *matcher*.
- **The manual Windows pass** (A doc §6, last paragraph) is out of scope: it is
  run by hand on a Windows box after the lot. Do not write or perform a step
  that boots the app on Windows — the runner is on Linux.
- A Windows CI runner, a Windows installer, drive-letter or UNC UX, and
  `os.path.normcase` anywhere.

## Project rules that bind these tasks (from `AGENTS.md`)

- **`mcphub/` and `lsphub/` mirror each other** — `constants`, `discovery`,
  `manager`, plus formatting of results. *"Keep that symmetry when touching
  either one."* win-07's refusal lands in both, in the same shape.
- **Path rules come from two levels, one verdict.** *"The strictest match wins,
  so B tightens but never loosens A. A hidden path answers 404 to a read and 422
  naming the rule to a write; `ro` answers 422 to a write only."* A changes what
  the locks **mean**, never their order: the locks run before `mode_for`, and a
  verdict can only narrow what the locks already allowed.
- **`mode_for` is the one place the verdict is computed** (pol-06). Do not add a
  second predicate: every change lands inside the matcher or the locks.
- **The declaration files are the only source of truth for what runs**, and an
  unresolved `${NAME}` is *"kept verbatim with a note, never blanked"* — that is
  the shape win-07's refusal reuses. Secrets (`headers`, `env`,
  `initializationOptions`) never leave the server.
- **`.env` is regenerated from a fixed template on every save**: *"a new or
  renamed key means editing `SETTING_KEYS`, `_render_env()` and `.env.example`
  together."* A-D6 joins that rule: a key must also be free of OS-variable
  collisions on every supported platform.
- **Every path the app holds is relative to the working location**, *"so a switch
  invalidates them all at once"*. win-05's probe cache is invalidated by that
  same switch.
- **Tools are read-only by design**; each connector's reach is bounded by a
  deny-list (`lsp_connectors/restricted-paths.txt`,
  `mcp_connectors/restricted-fqdn.txt`). win-07 is what keeps that sentence true
  on a platform where the lists match nothing.
- **A `/api/fs` write route is reachable from the SPA only** —
  `tests/test_write_surface_isolation.py` asserts it by walking `src/core` for
  write markers and `*.py` by glob. Keep new module and function names clear of
  that vocabulary so the guard does not fire on them.
- **Python style**: explicit type annotations, `typing` imports, no bare
  containers.
- **Tests stay hermetic**: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1` (set by
  `tests/conftest.py`); patchable file locations live on `mcphub.constants` /
  `lsphub.constants` and must be patched through those modules; a backend test's
  workspace is a throw-away `tmp_path` pointed at by `BENCHMARKIT_PATH_LOCATION`,
  never this checkout.
- No lint step. `pytest` for Python, `cd web && npm test` for Vitest,
  `cd web && npm run test:e2e` for Playwright, `cd web && npm run build` for
  `web/dist`.

## Definition of done for the lot (A doc §6, both platforms)

- `pytest` green with **no failures** on Linux and **zero** marked skips (Linux
  has symlinks and is POSIX — a skip on Linux means a marker landed on the wrong
  test).
- `cd web && npm test` green; `cd web && npm run test:e2e` either runs or says
  plainly why it did not (win-08).
- The eight behaviour tests of §6 (restated inside the task files) exist and
  pass **on Linux**, by forcing `fold=True`/`False` and by feeding `\`-shaped
  input.
- Linux behaviour is otherwise exactly what it was: every existing verdict in
  every existing policy file, the 404-vs-422 asymmetry (`policy.py:133`
  `hidden_error`), the all-or-nothing subtree delete, the `/`-joined payload
  form, and the mode round-trip. The only user-visible Linux change is that
  `a\b` now means `a/b` rather than a filename containing a backslash — and no
  POSIX tool in this app produces such a name. **Plus one this lot derives
  rather than inherits** (see win-02's *"the residue"* note): a path whose first
  component is a single letter plus a colon (`a:b`) is refused as a drive spec.
  Refusing `C:x` — which A-D2.1 requires — cannot be done without also refusing
  `a:b`. It must be pinned in a test and named in the commit note, not slipped
  in.
