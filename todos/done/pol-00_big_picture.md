# Workspace file policy — big picture

Replace the Editor tab's **hardcoded** path rules with a **policy made of two
levels**: one built into the app and not configurable, one declared per workspace
in `<workspace>/.nojae/read-write-policy.json`. Three outcomes per path —
`hidden`, `ro`, `rw` — applied to the tree, the file read, the write routes, the
diff and git status, all from one verdict function.

Why it exists: today the rules live in `mcp_connectors/path_policy.py`, which
refuses anything whose name starts with `.env`. That is right for *this* repo,
whose `.env` the app regenerates on every settings save — and wrong for the same
tab pointed at another project, where `.env` is an ordinary file the user edits
by hand. The rule was never about `.env`; it was about **who owns the file**.

## The two levels

| | level A — built in | level B — the workspace file |
|---|---|---|
| where | code (`editorfs/builtin_policy.py`) | `<workspace>/.nojae/read-write-policy.json` |
| when it applies | **only when `workspace_root() == realpath(PROJECT_ROOT)`** — benchmarkit editing itself | always |
| configurable | no | yes, edited outside the app |
| owns | files the app itself writes or regenerates | everything a project calls noise or off-limits |

Level A is conditional, not global: point the working location at
`/home/vgr/anotherproject` and level A is dormant, so that project's `.env` is a
normal file. That condition is one comparison, and
`editorfs.policy.workspace_root()` already does the resolving.

**The mixed state is the normal state.** The default working location *is* the
install dir (`config.setup_path_location`), so the app boots already editing
itself and level A is live from the first request.

### Precedence: restrictions are a union, never an override

Strictness order `hidden` > `ro` > `rw`. A path is governed by the strictest
match from either level; a level-B `{"path": ".env**", "mode": "rw"}` cannot
un-hide benchmarkit's own `.env`, and does not need to be refused to say so — it
is reported as a **note** ("already hidden by benchmarkit's rule for this
workspace"). The reason is not secrecy but ownership: `config._render_env()`
rebuilds that file from `SETTING_KEYS` on every save, so hand-editing it in a
pane is a fight the user loses silently.

Consequence worth naming: a broken level-B file can only ever loosen level B.
Level A keeps holding.

## The three modes

| mode | tree | read / stat / diff | create / save / rename / delete |
|---|---|---|---|
| `hidden` | pruned, never walked | **404** | **422** naming the rule |
| `ro` | listed | normal | **422** naming the rule |
| `rw` | listed | normal | normal (the default) |

**404 to look, 422 to touch.** A hidden path must be indistinguishable from
absence for anything that *observes* it (a read that answers 422 proves the file
is there). A mutation must never answer "absent", because 404 is the green light
for `O_CREAT | O_EXCL` — so mutation refusals name the rule. The rules are not a
secret: the policy file is the user's own, and pol-07 publishes both levels in
the UI.

`ro` on a folder covers its **whole subtree** and means *nothing under it
changes, including its own listing*: create inside, save, delete, rename-in and
**rename-out** are all refused (a move out rewrites the folder's directory
entries). `mutate._subtree_refusal` (`mutate.py:375`) already walks a subtree
all-or-nothing and names the offending descendant; it becomes the executor of
this rule rather than a new mechanism.

## Matching

One matcher (pol-02), path-aware: `*` stays inside one name, `**` crosses
directories. `fnmatch` (what `path_policy.glob_hit` uses today, line 132) is
**not** usable — its `*` eats `/`, so `web/*` would silently cover `web/a/b/c`.

Patterns are matched against the **normalised workspace-relative** path, never
the absolute one. Rules are evaluated in file order and the **last match wins**,
which is what lets a later rule carve an exception out of an earlier folder rule
(the `.gitignore` / `iptables` model):

```jsonc
{"path": "**/node_modules", "mode": "ro"},
{"path": "**/node_modules/**/README.md", "mode": "rw"}
```

A `hidden` folder is **pruned, not walked**, so nothing below it is reachable —
not even by carve-out. To make most of a folder invisible but the folder itself
browsable, use `ro` on the folder and `hidden` on its children.

## `.nojae` is reserved, by code

`POLICY_DIRNAME = ".nojae"` and the folder are **not policy-configurable**: a
rule about the thing that holds the rules is a contradiction. Concretely it is
absent from the tree, `stat`, read, write, rename, create, git status and the
diff — checked on the **component list before any `stat`**, the trick
`resolve_parent_in_workspace` already uses for a leaf that is not on disk
(`policy.py:269-276`), so "does it exist?" never leaks.

It is edited **only outside the app**, by hand. It is **out of source control**
(`.gitignore` gains `.nojae/`), so it is machine-local per checkout — like
`.env` and the two hub declaration files, which are gitignored for the same
class of reason.

The file reloads when it changes. Cache key is `(workspace_root, mtime_ns,
size)` — the root must be in the key, or a workspace switch would run the new
root under the old root's rules.

## Seeding: a workspace gets its file the moment it is targeted

`ensure_policy_file(root)` writes a default `.nojae/read-write-policy.json`
containing the common-sense set (`.git` hidden, `**/__pycache__` hidden,
`**/node_modules` ro, …). Two entry points, because those are the two moments a
workspace becomes active:

1. `config.setup_path_location()` (`config.py:268`) — boot, whichever branch
   chose the value;
2. `POST /api/path-location` (`routes.py:1156`), **after** the two hub restarts,
   so a seeding hiccup can never trigger the env-rollback at `routes.py:1229`.

Three hard rules:

* **Exclusive create** (`open(..., "x")`) — one syscall, so a user's edited file
  is never clobbered and two concurrent requests cannot both write.
* **A failure never fails the switch** — a read-only mount or a foreign-owned
  root returns success plus a note in the payload (`"policy": {"state":
  "failed", "error": …}`), and the Editor banner says so. Refusing to open
  someone's project because a file could not be left behind is worse than an
  unfiltered tree.
* **Never called from `path_location()`** — that is a per-request reader, and a
  reader that writes is how a stray file appears in a project.

Seeding is why the built-in `IGNORED_DIR_NAMES` (`constants.py:41`) **goes away**:
if both mechanisms applied, a user could never un-hide `node_modules`, which
contradicts the point of a config file. Names move into the seeded file; only
the caps (`MAX_TREE_ENTRIES_PER_DIR`) and the `truncated` flag stay in code.

## Validation: loud, and split by who can fix it

| finding | how it lands | why |
|---|---|---|
| bad JSON, unknown top-level key, unknown `mode` spelling | **exception**; the file is ignored | the user can fix it |
| pattern that can never match (its literal prefix does not exist) | exception + closest-sibling suggestion | the user can fix it |
| pattern naming `.nojae` | exception | reserved |
| pattern shadowed by a stricter level-A rule | **note** | the user cannot fix it — an exception would be a wall |

"Exception" means: level B contributes nothing, reads stay allowed, and **writes
are refused (422)** while the file is broken — the policy that was supposed to
protect a folder demonstrably did not load, so pretending it did not exist and
letting writes through is the one option that is not acceptable. The Editor shows
the file path and the message verbatim (pol-07).

## What does *not* move

The policy change replaces *rules*, not *invariants*. Each is asserted today:

1. **The three locks stay** (`policy.py`): textual (`normalise_rel` rejects
   absolute, `~`, NUL, any `..` that survives `normpath`), the `realpath`
   containment check (a symlink inside the workspace pointing at `/etc`), and
   `_check_workspace_root()` against `lsp_connectors/restricted-paths.txt`.
   Without them a policy-driven tree is a filesystem browser.
2. **Every write stays browser-only.** No MCP connector, LSP tool or
   `core/chat.py` path may reach a `/api/fs` write route —
   `tests/test_write_surface_isolation.py` asserts this, and it keeps asserting
   it after pol-06.
3. **Nothing is destroyed silently.** Backup-before-save, delete-by-move into
   `logs/trash/<…>`, the `mtime` conflict guard (409 with both mtimes), the
   chat-stream gate, temp-write + `os.replace`, no partial write of a `truncated`
   or `binary` buffer.
4. **Read caps stay** (`MAX_EDITOR_BYTES`, `MAX_TREE_ENTRIES_PER_DIR`,
   `MAX_STATUS_ENTRIES`) with `truncated` reported rather than entries dropped.
5. **The connectors keep their own floor.** `mcp_connectors/path_policy.py`
   stays and keeps its hardcoded secret list — it guards a different trust
   surface (an agent putting `id_rsa` into a context window vs a human looking at
   their own file in a pane). pol-06 stops `editorfs` calling its rule list and
   pol-08 layers the workspace policy *on top* for agents.

   What stays shared is the **machinery**, not the rules: the matcher and the
   parser live in `mcp_connectors/` (stdlib-only, importable by both the app and
   the two stdio connector processes). The new rationale goes in
   `path_policy.py`'s docstring, or a contributor will "restore" the shared rules.

### The asymmetry pol-08 has to know about

Measured in the code, not assumed: both local connectors **skip** that floor for
paths inside their `--root-dir`, which `externals.json` sets to
`${BENCHMARKIT_PATH_LOCATION}` — `fedit_mcp.py:895` (`if _within_root(real) and
not _system_prefix_hit(absolute): return absolute`) and `console_mcp.py:505-507`.
So *today* an agent in this session can edit the app's own `.env` through `fedit`,
while the Editor pane cannot read it. The pane is the stricter of the two.

That inverts what pol-08 is for: applying the workspace policy to the connectors
**narrows** an agent's reach inside the root (an `ro` folder becomes uneditable
by the agent too) instead of adding a new floor. The name-based floor stays
meaningful outside the root — where the connectors can still wander.

The pane-and-agent disagreement is also why the two rule sources must not be
merged into one: the same list cannot both keep the app's `.env` away from a
browser and stay out of the way of a connector working the tree the operator
chose.

## The task lot

`pol-00` is this file. Tasks are sequential; 01–05 change no visible behaviour,
so nothing is exposed on the day they land.

| # | File | Delivers | Visible? |
|---|---|---|---|
| 1 | `pol-01_verdict_primitive.md` | one `mode_for(rel) -> Verdict` replacing `listable`/`writable`/`refusal`; `POLICY_DIRNAME` | no |
| 2 | `pol-02_glob_matcher.md` | path-aware `*`/`**` matcher, subtree + last-match-wins | no |
| 3 | `pol-03_builtin_app_policy.md` | level A, conditional on root == app root | yes: a foreign `.env` becomes an ordinary file |
| 4 | `pol-04_policy_file_loader.md` | parse + validate + cache the `.nojae` file; the note/exception table | no |
| 5 | `pol-05_seed_on_switch.md` | `ensure_policy_file` at the two entry points; `.gitignore` | yes: a new workspace arrives with a tree already filtered |
| 6 | `pol-06_wire_surfaces.md` | tree, read/stat, mutate, git status all answer from `mode_for`; `IGNORED_DIR_NAMES` deleted; the `path_policy` import dropped from `editorfs` | yes, the big one |
| 7 | `pol-07_policy_panel.md` | `GET /api/fs/policy`, the footer/banner wording, unknown-mode errors in the UI | yes |
| 8 | `pol-08_connectors_policy.md` | agents honour the workspace's `ro`, keeping their own secret floor | yes, for agents |

## Decisions taken for this lot

| # | Question | Decision |
|---|---|---|
| D1 | Levels | Two: level A in code, live only when the workspace **is** the app; level B in `<root>/.nojae/read-write-policy.json`, always live. |
| D2 | Precedence | Union, strictest wins. Level B can tighten, never loosen, level A. |
| D3 | `ro` on a folder | Applies to the subtree and to the folder's own listing: create, save, delete, rename-in **and rename-out** refused. |
| D4 | Broken policy file | Exception with a named cause; level B ignored, reads stay open, **writes refused** until fixed. |
| D5 | `.nojae` | Reserved by code. Hidden everywhere, addressable nowhere in `/api/fs`, edited only outside the app, gitignored. |
| D6 | `.git` | A level-B rule (`.git: hidden`), present in every seeded file — not a code rule. |
| D7 | `IGNORED_DIR_NAMES` | Deleted; its names move into the seeded file, so the file is the only visibility mechanism. |
| D8 | Hidden answers | 404 for anything that observes (read/stat/diff), 422 naming the rule for anything that mutates (incl. create). |
| D9 | Seeding | Exclusive create, failure is a note not an error, called from boot and the location switch only. |
| D10 | Connectors | Keep the hardcoded floor; the workspace policy is layered on top (pol-08). |
| D11 | File naming | `pol-` prefix, `_` between number and slug — the separator `core/batch.py` requires the digits immediately after. |
| D12 | Where the shared machinery lives | `mcp_connectors/policy_shared.py` (vocabulary), `policy_glob.py` (matcher), `policy_file.py` (parser + validator). Stdlib-only, no `src` import: the two stdio connector processes must reach them, and the wheel packages only `src`. The app wraps them with its cache and its `WorkspacePathError` conversion. |

## Explicitly out of scope

Per-user or per-session policies, a UI that edits the policy file (it is edited
outside the app by decision D5), inheritance from a parent directory's `.nojae`,
regex patterns, `deny`/`allow`-style ACLs, making hidden paths searchable,
watching the policy file with a filesystem event (mtime-on-read is enough).

## Files this lot touches

```
mcp_connectors/policy_shared.py     NEW: modes, ranks, levels, reserved names
mcp_connectors/policy_glob.py       NEW: the matcher (pol-02)
mcp_connectors/policy_file.py       NEW: parse + validate (pol-04), used by pol-08
mcp_connectors/path_policy.py       docstring: why the connector floor is separate
mcp_connectors/fedit_mcp.py         policy check + an honest edit_policy() (pol-08)
mcp_connectors/console_mcp.py       same, in protected_path_error (pol-08)

src/core/editorfs/policy_modes.py   NEW: Verdict + mode_for (pol-01)
src/core/editorfs/policy_file.py    NEW: cache + WorkspacePathError wrapper (pol-04)
src/core/editorfs/policy_seed.py    NEW: DEFAULT_POLICY_JSON, ensure_policy_file (pol-05)
src/core/editorfs/builtin_policy.py NEW: level A + is_self_workspace (pol-03)
src/core/editorfs/policy.py         the rules swap; listable/writable/refusal go
src/core/editorfs/constants.py      POLICY_* aliases in; IGNORED_DIR_NAMES out
src/core/editorfs/{tree,files,mutate,gitstatus}.py   answer from mode_for (pol-06)
src/config.py                       boot seeding (pol-05)
src/api/routes.py                   /api/fs/policy + the policy block (pol-05, 07)
web/src/composables/useEditorPolicy.js  NEW (pol-07)
web/src/composables/useBenchmark.js     the policy note after a switch (pol-05)
web/src/components/{WorkspaceNavigator,SettingsDialog,EditorTreeDialogs}.vue
web/src/styles/theme.css                the footer and the panel
.gitignore                              .nojae/
AGENTS.md                               the two levels, ~8 lines
```

## Definition of done for the lot

* Benchmarkit-as-workspace keeps today's protection: `.env*` unreachable from the
  tree and the write routes, and the settings dialog stays its only editor.
* A foreign workspace shows and edits its own `.env`, and arrives with a seeded
  policy file it can read as a worked example of the syntax.
* Every refusal names the rule, the level it came from and the mode.
* No `/api/fs` write route is reachable from a connector or an LSP tool
  (`tests/test_write_surface_isolation.py` unchanged and green).
* `pytest`, `cd web && npm test` and `cd web && npm run test:e2e` pass; AGENTS.md
  describes the two levels in under ~10 lines and stays under ~300.
