# Seeding the two built-in MCP connectors, and renaming the declaration file

**Status**: spec written, implementation not started.
**Surface**: the MCP hub's declaration file only — one new seeder module, one
constant rename, one merge in `discovery.load_externals`, plus the docs and
ignore files that name the old path. No new route, no new frontend state, no
change to `McpManager`.

## 1. The need

`console` and `fedit` are not optional connectors: they are the two stdio servers
shipped in `mcp_connectors/`, they are the reason WorkBench can touch the
workspace at all, and the batch loop refuses to launch a lot without them
(`useBatch.js`'s `REQUIRED_MCP_CONNECTORS`). Yet today they exist only because
each checkout hand-writes them into the gitignored declaration file — with the
machine's interpreter path spelled out literally, so a fresh clone has no file
editor and no shell until someone copies an entry in.

Two changes, one gesture:

1. the declaration file is renamed `externals.json` → **`mcp-settings.json`**
   (with `.gitignore` adapted);
2. the app **seeds** that file with the `console` and `fedit` entries the first
   time it is needed, so a fresh checkout boots with the two connectors declared
   and every other connector stays a hand-written entry.

Seeding was chosen over synthesising the entries in code (the alternative that
bypasses the file entirely) because the file is what the `enable` toggle writes
to. A built-in that only exists in memory cannot be switched off durably:
`set_externals_enabled` only touches entries already present in the file, and
answers "no file, nothing to do" for a missing one — so a user who turned
`console` off would find it back on the next boot, and the Settings card and the
live tool list would disagree. A seeded entry is a real entry: same toggle, same
restart, same hot edit, same panel.

## 2. Decisions taken

### 2.1 One file per installation, not one per workspace

The seeded file lives at `constants.MCP_SETTINGS_PATH` — inside the **install**
directory, anchored on `PROJECT_ROOT`. It is *not* the `.nojae` pattern:
`.nojae/read-write-policy.json` and `agent-modes.json` are per-workspace because
their content describes the workspace, while this file describes the app's own
hubs and holds its API keys. So:

* switching the working location does **not** create a second declaration file
  in someone else's project — the same file is re-read, and only the
  `${BENCHMARKIT_PATH_LOCATION}` inside the two `command` argvs changes;
* the file is gitignored, may hold secrets, and is edited by hand or by the
  Settings toggle — never by a connector. (The `fedit` connector cannot reach it
  anyway: `mcp_connectors` is in `path_policy.DENY_COMPONENTS`.)

It follows that the file has **no committed template** (unlike `.env.example` and
`lsp_connectors/lsp_servers.example.json`): it is written by the seeding code,
which is also true of `.nojae/`.

### 2.2 Seed-then-merge, so a failed seed is not a lost feature

The two halves are one guarantee, and each alone is not enough:

| half | what it buys |
|---|---|
| the seed writes the entries into the file | they are toggleable, editable, restartable — real entries |
| `load_externals` merges the same entries when the file has none of them | the app still has a file editor and a shell when the seed could not happen (read-only install dir, unpacked wheel, permissions) |

Merge, not "file missing → write a default": the built-ins are laid under the
file's entries and the **file's entry wins for the same id**, so a user who
re-declares `console` with extra flags (a `--debug`, another `--root-dir`) keeps
theirs and the app never edits a file to contradict it. Id order stays the
file's `sorted(data)` order, with an injected built-in sorting into its place.

The seed is **exclusive create** (`O_WRONLY | O_CREAT | O_EXCL`), one level,
never `makedirs` — byte-for-byte the contract of `editorfs/policy_seed.py`,
including the reasons: a hand-edited file is never clobbered, two concurrent
seeds have one winner, a `FileExistsError` is the ordinary answer, and a
half-written file (write failed after the create) is removed again rather than
left to poison every later scan. The whole thing answers
`{"state": "created" | "existing" | "failed", "path": str, "error": Optional[str]}`
and raises nothing, and it runs *before* `app = FastAPI(...)` in `create_app()` —
so the same `mcp_connectors/` folder is a precondition of the lifespan's first
scan, and a seed that cannot happen must not take the boot down.

### 2.3 The seeded document carries its own explanation

JSON has no comments, and this file is a user's first look at the hub's
configuration, so the seeded bytes explain themselves from inside the file — the
same device as the policy seed's `readme` key.

**Trap, verified**: the MCP loader has no comment convention. `lsphub.discovery`
skips a key starting with `_` (which is why `lsp_servers.example.json` carries a
`"_note"` key), but `mcphub.discovery` does **not**: an entry whose value is not
an object is skipped *with a per-entry note*. So a `"_note": "…"` string in
`mcp-settings.json` becomes a permanent line in `externalsError`, which the
Settings dialog and the MCP panel render as an alert. Either put the explanation
in each entry's `description`, or teach the MCP loader lsphub's `_` skip (one
line, in the same place, and it makes the two hubs behave alike — the symmetry
AGENTS.md asks for). Recommended: adopt the `_` skip; the seeded text is then a
`"_note"` like the LSP template's.

The two seeded entries:

```json
{
  "console": {
    "type": "local",
    "enable": true,
    "description": "Console command (built-in; one argv, no shell)",
    "command": [
      "<sys.executable>",
      "<this install dir>/mcp_connectors/console_mcp.py",
      "--root-dir", "${BENCHMARKIT_PATH_LOCATION}",
      "--transport", "stdio"
    ]
  },
  "fedit": { "same": "shape, fedit_mcp.py" }
}
```

Three details that are load-bearing:

* **`sys.executable`, not a remembered path.** That is what makes a fresh clone
  work on a different machine, and it is the whole point of seeding — today's
  file hardcodes a `~/.pyenv/versions/benchmarkit/bin/python` and a
  `~/dev/benchmarkit`.
* **The `${BENCHMARKIT_PATH_LOCATION}` placeholder is written and read as text,
  never resolved at seed time.** Expansion belongs to `load_externals`, which
  runs per scan — that is what makes a working-location switch relaunch both
  connectors with the new `--root-dir` (the switch works by re-reading the file).
  A value resolved once by the seeder freezes the old root into the argv and
  breaks that path silently.
* **An entry is seeded only if its script exists.** `pyproject.toml` ships
  `packages = ["src"]`, so an installed wheel has no `mcp_connectors/`; seeding
  an entry that points at a missing file buys a boot that spends
  `EXTERNAL_CONNECT_RETRIES` (60 × 0.5 s) retrying nothing and then reports
  `error`. Absent script → no entry, and one note in the log.

### 2.4 Renaming the file is not renaming the code's vocabulary

Only the *file name* changes. Deliberately not renamed, because each is a wire
or API surface with readers outside the MCP package:

| name | why it stays |
|---|---|
| `externalsError` | response field read by `useBenchmark.js`, `SettingsDialog.vue`, `web/tests/pathLocationPolicy.test.js` and the `05-workbench-focus` e2e stub — renaming it is a wire change with no user-visible gain |
| `load_externals` / `set_externals_enabled` | re-exported through `mcphub.__all__`, imported by `routes.py` and `server.py`, and named in `tests/test_mcp.py` throughout |
| `mcpEnabled`, the panel's `enable` semantics | unchanged behaviour, so unchanged names |

The constant **does** change, because it would otherwise name a file that no
longer exists: `constants.EXTERNALS_PATH` → `constants.MCP_SETTINGS_PATH`, no
alias left behind (the repo keeps one name per thing). It is a `TYPE_CHECKING`
re-export and an `__all__` member of `mcphub/__init__.py`, and 8 test sites
patch it through `mcphub.constants` (`test_api.py`, `test_lsp.py`, `test_mcp.py`,
`test_fscase.py`, `test_agent_modes_seed.py`, `test_editorfs_policy_seed.py`);
all of those edits are mechanical.

### 2.5 The legacy file is reported, never migrated

An upgrade leaves the user's real configuration in a file nothing reads any
more — the quiet worst case is a `git pull` after which every remote connector
(the Context7 key, the browser) vanishes with no explanation. The seeder
therefore checks, while it is seeding: if `externals.json` exists and
`mcp-settings.json` did not, it emits one validation note in the same channel as
every other problem (`externalsError`, so the Settings alert shows it):

> `externals.json` is no longer read — rename it to `mcp-settings.json`.

The app does **not** rename or copy it: the file holds API keys, moving a
secrets file inside a user's checkout without being asked is not this app's
business, and one hand-run `mv` is cheaper than a migration path that has to
stay right forever. (`BENCHMARKIT_PATH_LOCATION` is the only other place a
machine-local value survives a rename, and it is process state.)

### 2.6 How the suite stays hermetic

`tests/conftest.py` already keeps every test from spawning connectors
(autouse `BENCHMARKIT_DISABLE_MCP=1`) and the hub/API fixtures redirect the
declaration path into a `tmp_path`. The first two rules keep that property, and
they decide how much test churn this costs; 3. is the one moment neither covers.

1. **Synthesis (not the seed) applies only when `path is None`** — a caller that
   passes an explicit `path` to `load_externals` gets exactly what that file
   declares. The 17 direct `load_externals`/`set_externals_enabled` calls in
   `tests/test_mcp.py` all pass a tmp path, so that whole class is untouched.
2. A second patchable seam, `constants.SYNTHESIZE_BUILTIN_CONNECTORS = True`,
   set to `False` by an autouse `conftest` fixture next to the two
   `BENCHMARKIT_DISABLE_*` ones. The handful of tests that assert about the app's
   own file (below) set it back to `True` for their own case.

Without rule 2, the API-level assertions that name "only what the file declares"
would start seeing two more connectors —
`test_get_without_declaration_file_lists_nothing`,
`test_get_lists_declared_connectors_with_enable_as_active`,
`test_restart_rereads_the_declaration_file`,
`test_chat_uses_the_enabled_declaration_entries`, the win-07 note-count tests (a
seeded `local` entry that wants to run joins their refusal notes), and the
lifespan's `load_externals()[0]`. With rule 2 they keep their meaning: they
describe *declared* connectors, which is what they were written to pin.

3. **The seed may fire before any fixture runs, and that is acceptable only
   because of 2.2.** `api/server.py` ends with a module-level `app =
   create_app()`, so importing that module (which the suite does) reaches the
   seed at *collection* time — before a `monkeypatch` of the path could apply.
   An autouse pin like conftest's `_no_trash_of_its_own` cannot cover that
   moment. Two consequences, and they are the reason the merge half is not
   optional:

   * creating the gitignored file at import is the *install* behaviour, not a
     leak — it is what running `benchmarkit` once would do, and the file holds no
     secret the machine did not already have;
   * a test whose seed **fails** (or which patches the path to a folder with no
     file) must still see a working hub. That is what the merge in
     `load_externals` guarantees, and it is why 2.2 is two halves and not one.

## 3. What it builds on

* `core/editorfs/policy_seed.py` (pol-05) — the seeder's whole contract: exclusive
  create, one-level `mkdir`, never-overwrite, failure-is-a-note, the
  `created/existing/failed/absent` dict, and the module-attribute path lookup
  that makes it patchable. The new module is its sibling, not a variant.
* `core/agent_modes.py` `ensure_agent_modes_file` (am-03) — the second instance
  of the same contract, and the reason the two `.nojae` files and this one must
  not be described as one mechanism in `AGENTS.md` (different home per file,
  see 2.1).
* `mcphub/validation.expand_env_list` — already expands `${NAME}` for every
  `command` argv, so a seeded entry needs no new expansion path.
* `mcphub/discovery.set_externals_enabled` — already preserves unknown keys and
  entry order on write-back, so the `_note` key (2.3) and a user's re-declared
  entry both survive a Settings toggle untouched.

## 4. Shape and files of the new pieces

### `src/core/mcphub/seed.py` (new)

```python
def builtin_connector_entries() -> Dict[str, Dict[str, Any]]:
    """The app-owned entries, resolved against ``constants`` per call."""

def default_declaration_text() -> str:
    """The bytes the seed writes, rendered from the literal above."""

def ensure_declaration_file() -> Dict[str, Any]:
    """Exclusive-create ``mcp-settings.json``; never raises, never overwrites."""
```

Module attributes from `constants` are read **inside** each call
(`MCP_SETTINGS_PATH`, `MCP_CONNECTORS_DIR`) — the documented convention of that
module, and the only reason the tests can redirect the hub at a temp folder.

`builtin_connector_entries()` is the single source for both halves (the seed's
bytes and `load_externals`' merge), which is what keeps the written file and the
read one from agreeing only by accident — the same argument that makes
`policy_seed` render from a Python literal instead of a blob.

### Edits

| file | change |
|---|---|
| `src/core/mcphub/constants.py` | `EXTERNALS_PATH` → `MCP_SETTINGS_PATH`, value `mcp_connectors/mcp-settings.json`; new `SYNTHESIZE_BUILTIN_CONNECTORS = True`; comment updated (the file is now seeded, not only hand-written) |
| `src/core/mcphub/discovery.py` | merge `seed.builtin_connector_entries()` under the file's entries when `path is None` and the seam is on; skip a `_`-prefixed key (2.3); the legacy-file note (2.5); docstrings |
| `src/core/mcphub/__init__.py` | re-export the new constant and `ensure_declaration_file`; `TYPE_CHECKING` block and `__all__` |
| `src/api/server.py` | one `_seed_mcp_declaration()` wrapper (log-only, same shape as `_seed_workspace_policy`) called beside the other two seeds |
| `src/config.py` | the generated `.env` comment naming `mcp_connectors/externals.json` (line ~608) — this text is re-written into every `.env` on save, so it is a code edit, not a doc edit |
| `.gitignore`, `.kilocodeignore` | `mcp_connectors/externals.json` → `mcp_connectors/mcp-settings.json` (the ignore file also carries the comment about why: it holds keys) |
| `AGENTS.md`, `README.md` | see §7 |
| `web/src/components/SettingsDialog.vue`, `WorkBenchMcpPanel.vue`, `composables/useChat.js`, `useBenchmark.js` | only the *strings and comments* naming `externals.json` (the Settings caption, the restart button's title); no code, no state |
| tests | `EXTERNALS_PATH` → `MCP_SETTINGS_PATH` at the 8 patch sites; new `tests/test_mcp_declaration_seed.py` |

Not touched: `manager.py` (it consumes already-normalised dicts), every route,
any frontend *logic* (only the four caption strings above change), the LSP hub,
the `.env` key set, and the `vite build` step.

## 5. Migration, by hand, in one line each

* existing checkout: `mv mcp_connectors/externals.json mcp_connectors/mcp-settings.json`
  (gitignored, so this is a filesystem move, not a `git mv`); then delete the two
  `console`/`fedit` entries, whose hardcoded interpreter path is now the app's
  problem and whose `${BENCHMARKIT_PATH_LOCATION}` the seed writes correctly;
* a contributor who leaves the old file in place gets one note in
  `externalsError` rather than an app with no connectors and no explanation;
* a fresh clone: nothing — the boot seed writes the file with the two entries and
  the user adds their remotes to it.

## 6. Verification

### Backend (`tests/test_mcp_declaration_seed.py`, new)

Hermetic, `tmp_path`, no network; the seam of 2.6 set per test.

| test | asserts |
|---|---|
| seed creates and it is valid JSON | a write into a pre-existing `MCP_CONNECTORS_DIR` stand-in; both ids present, `enable: true`, argv starts with `sys.executable` |
| argv keeps the placeholder as text | `"${BENCHMARKIT_PATH_LOCATION}"` is in the bytes, and after `load_externals` it is the *environment's* value |
| second call is `existing`, bytes unchanged | a hand-edited file is never clobbered |
| the race is closed by the flag, not a check | two calls in a row have one winner (the policy-seed idiom) |
| unwritable dir answers `failed`, raises nothing | and the app boots anyway |
| missing `MCP_CONNECTORS_DIR` answers `failed` | and creates no chain |
| script absent → no entry | the wheel case of 2.3: no entry, one note, and `load_externals` synthesises nothing for it either |
| merge: file wins for the same id | a user's own `console` entry with extra args survives, in the file's own order |
| merge when the file is absent | the two connectors exist with `kind == "local"` — the degraded install still has its editor |
| merge is skipped for an explicit `path` | a tmp-path call sees only that file (rule 1 of 2.6) |
| `_note` key is not an entry | no validation note, not listed as a connector (2.3) |
| legacy file | `externals.json` present, `mcp-settings.json` absent → one note naming the new file; the old bytes are unmodified |
| the toggle round-trips a built-in | `POST /api/mcp` without `console` flips the flag **in the file**, and a re-read still lists it as disabled |

The last one is the regression the seeding choice exists to prevent, so it is the
test that matters most: it must fail loudly under a synthesise-only design.

### Frontend

Nothing new. `web/tests/pathLocationPolicy.test.js` stubs the hub payload, and
`05-workbench-focus.spec.js` routes its own connector list, so neither sees the
change. `04-agent-modes.spec.js` counts `.chat-mcp-panel` switches from the
payload it just read, so it follows whatever the file declares.

### e2e

`web/playwright.config.js` keeps `BENCHMARKIT_DISABLE_MCP: '1'`, so nothing
spawns, and the boot seed lands in the install dir's (gitignored)
`mcp_connectors/`. No spec asserts a connector count, so no spec changes — and
this is the case worth an explicit run of `npm run test:e2e`, because it is the
one place the seed runs against the *real* folder.

### Not tested, and why

* Real connector startup (the seeded argv against the real server): it needs the
  `[mcp]` extra and a live stdio child, and `tests/test_write_surface_isolation.py`
  already pins the connectors' tool listings from the modules directly.
* An upgrade's hand-migration: it is one `mv`, and the note is what is tested.

## 7. Note for `AGENTS.md` when this lands

This changes a stated invariant, so the doc moves with the code — three edits, no
new paragraphs:

* the module table's `core/mcphub/` row and the data-file table's row name the
  new file, and say it is seeded by code with no template (as `.nojae/` does);
* *"The declaration files are the only source of truth for what runs"* gains its
  honest exception in one clause: the app seeds its own two connectors into the
  file, and it is still the only thing read — nothing runs that is not in it;
* *"Adding a connector is never a module in `src/` … Add the file **and** the
  entry"* stays, and gains that `console`/`fedit` are seeded, so a new entry is
  an addition to an existing document.

`README.md`'s MCP section ("Connectors are declared — never coded") is the
user-facing half of the same sentence: it names the new file and says the two
built-ins arrive by themselves.

## 8. Out of scope

* Seeding the LSP hub (`lsp_servers.json`): its servers are per-machine by
  nature (`--tsdk`, `VOLTA_BIN`, a `workspaceRoot`), and no LSP feature is
  refused for their absence the way batch mode is refused without `fedit`.
* Making a built-in non-toggleable, or a side-store for built-in flags — 2.1 is
  the decision that removes the need for either.
* Auto-migrating `externals.json` (2.5), and any change to how secrets are kept
  in the file.
* A committed `mcp-settings.example.json`: the seed is the template, and a second
  copy of the entry text is exactly the drift 2.3's single source avoids.
