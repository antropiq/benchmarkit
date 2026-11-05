# MCP and LSP declarations become workspace configuration

**Status**: spec written, implementation not started.

**Supersedes**: `docs/seeding-builtin-mcp-spec.md`, which is **deleted** by this
change. That spec never reached a todo iteration, and it decided the opposite of
what is asked here — its section 2.1 is titled "One file per installation, not one
per workspace", and this spec reverses it.
Nothing from it survives, including its fallback of synthesising the built-in
connectors inside `load_externals` when the declaration file lacks them.

**Surface**: two declaration files move into the workspace's reserved folder and
are re-named; two install-level files become committed templates; the hub path
constants stop being import-time values; both seeders join the two existing
seeding moments; the captions and restart titles that name the old files name the
workspace ones instead. No new tab, no new
route, no change to either loader's validation rules beyond one skip.

## 1. The need

The two hub declaration files are the last pieces of configuration that belong to
the **installation** while describing the **workspace**:

* Point the working location at another project and the connectors and language
  servers are still benchmarkit's own — the ones whose `command` argv and
  `workspaceRoot` were written for this checkout.
* They are hand-maintained per checkout (`mcp_connectors/externals.json` holds
  this machine's interpreter path and a Context7 key; `lsp_connectors/lsp_servers.json`
  holds this machine's pyright/Vue entries), and a fresh workspace starts with
  neither file, so the agent tabs are simply empty there.

The other workspace-scoped configuration already lives in `<working
location>/.nojae/` and arrives by one rule: **seeded by exclusive create at the two
moments a folder becomes the working location, never overwritten** —
`read-write-policy.json`, `agent-modes.json`, `prompt-helpers.json`. The two
declarations join that family, with one asymmetry that is the whole design:

| file | seeded as | why |
|---|---|---|
| `.nojae/mcp-settings.json` | the real file, with the two connectors benchmarkit cannot work without (`console`, `fedit`) | Their paths are knowable at seed time (this install's interpreter and scripts), so the app can write something that works. |
| `.nojae/lsp-settings.example.json` | the **example**, never `.nojae/lsp-settings.json` | No server choice can be guessed for an unknown workspace. The user copies/renames it and points it at the right tools. |

## 2. Decisions

### D1 — one folder, two levels, and the templates stay where they are

`<root>/.nojae/` is the workspace level; `mcp_connectors/` and `lsp_connectors/`
keep the install-level *templates*, and their names are fixed by this spec:

| today | becomes | level | git |
|---|---|---|---|
| `mcp_connectors/externals.json` (read) | `.nojae/mcp-settings.json` (read + toggled) | workspace | ignored (already, via `.nojae/`) |
| — | `mcp_connectors/mcp-settings.default.json` | install | **tracked** |
| `lsp_connectors/lsp_servers.json` (read) | `.nojae/lsp-settings.json` (read + toggled) | workspace | ignored |
| `lsp_connectors/lsp_servers.example.json` | `lsp_connectors/lsp-settings.example.json` | install | tracked (already) |

Rejected: putting the templates in `.nojae` too. A hidden folder cannot be read
through the app's own file surface, so the example the user is told to copy would
be unreachable from the Editor — and the templates are app source, not workspace
state.

Rejected: keeping the old names. `externals.json` never says who it belongs to,
and the two files no longer sit beside each other; `mcp-settings` / `lsp-settings`
name the *setting* rather than its old location, and the `.default` / `.example`
suffixes say which side of the copy step each template is on.

### D2 — each seeder makes the folder it writes into; the seeds are order-free

Every existing seeder already creates `.nojae` itself and swallows the
`FileExistsError` (`policy_seed._ensure:191`, `agent_modes._ensure_agent_modes:890`,
`prompt_helpers._ensure_prompt_helpers:274`) — one level, never `makedirs`, because
the root exists by construction. So no seeder depends on another having run, which
is exactly what D4 needs: the MCP seed moves ahead of the policy seed and still
writes. The two new seeds copy that same three-line block and nothing else, and no
existing line changes.

The consequence to keep in mind: whichever seeder meets a root it cannot write
reports `failed` for *its own* file, and the others answer theirs independently. A
root whose folder cannot be created therefore reports up to five notes rather than
one — which is the honest payload (five files did not get written) and matches what
the three seeders do to each other today.

### D3 — the hub paths stop being import-time constants

`mcphub.constants.EXTERNALS_PATH` and `lsphub.constants.LSP_SERVERS_PATH` are
`Path`s computed at import, so they cannot name a folder chosen later. Each hub
gains a per-call resolver, in the shape `agent_modes.agent_modes_path()` already
uses — built per call out of module attributes so a patched name or a switched
location moves it:

* `MCP_SETTINGS_FILENAME = "mcp-settings.json"` (module attribute, patchable);
* `def mcp_settings_path() -> Path` = `Path(path_location()) / POLICY_DIRNAME / MCP_SETTINGS_FILENAME`;
* `MCP_SETTINGS_PATH: Optional[Path] = None` — **a pin, not the default.** When it
  is not `None` it wins over the computed path; when it is `None` the resolver is
  asked. `MCP_SETTINGS_PATH` is the new name; `EXTERNALS_PATH` is not kept as an
  alias (a name that says "externals" would outlive the file it names).

The LSP hub mirrors all three (`LSP_SETTINGS_FILENAME`, `lsp_settings_path()`,
`LSP_SERVERS_PATH: Optional[Path] = None` → `LSP_SETTINGS_PATH`). The file name
each loader prints in its notes comes from the resolved `Path.name`, so a note
quotes the file it actually read.

Why keep a pin at all: the repo's convention is "module attribute, so a test moves
the lookup", and ~20 test call sites do exactly that today with no per-test root
(they set `BENCHMARKIT_PATH_LOCATION` only when they assert about a root). Keeping
one patchable override is a 3-line change in 5 files instead of a rewrite of 20
fixtures; `TRASH_DIR: Optional[Path] = None` in `editorfs.constants` is the
existing precedent for a name that means "override", and its `None` default makes
"the workspace's own file is the default" the visible fact.

Rejected: an env var read at call time. It would be a second source of truth next
to `BENCHMARKIT_PATH_LOCATION`, and a path a test sets in the process environment
leaks across tests in a way `monkeypatch.setattr` does not.

The reach-capability seam is untouched: both hubs keep importing
`src.config.path_location` the way `mcphub/process.py:30` already does, so no new
edge enters the dependency graph (`config` imports no `core` module except
`core.models`).

### D4 — seeding happens *before* the hub restarts

`POST /api/path-location` restarts both hubs (routes.py:1382-1383) and seeds the
`.nojae` files afterwards (1414-1435), deliberately, so a seeding hiccup cannot be
mistaken for a hub failure and trip the environment rollback. With the declaration
file inside `.nojae` that order is now wrong: the restart would re-scan a workspace
whose declaration file has not been seeded yet, and the new workspace would come up
with **zero** connectors and zero servers until the next restart.

So on the switch: set the env value → **seed both hub files** → restart the MCP hub →
restart the LSP hub → seed the three existing files → build the payload. Only the
MCP seed's position is load-bearing: the LSP restart reads `.nojae/lsp-settings.json`
which nothing seeds (D7), so its seed could sit anywhere — it goes beside the MCP one
because "both declarations exist before either hub is rebuilt" is one rule to
remember instead of two orders. The rollback guard stays honest because each seeder
answers a dict and raises nothing (the same argument that lets today's seeds run
outside the `try`), and each is wrapped the way `server._seed_*` wraps the boot ones.

The `existing` answer is the ordinary one and writes nothing, so it cannot disturb
the restart either way. Both new blocks are added to the GET payload as read-only
`*_file_state()` answers, beside `policy` / `agentModes` / `promptHelpers` — a GET
must never leave a file behind.

At boot (`create_app()`), the two seeds go beside the three existing ones
(`server.py:158-175`), which already run before the lifespan's first scan, so no
order change is needed there.

### D5 — the seeded MCP file is rendered from the committed template

`mcp_connectors/mcp-settings.default.json` is the **byte source** the seeder reads,
with three placeholder tokens it substitutes once, at seed time:

```json
{
  "_note": "benchmarkit created this file when this folder became its working location. Edit it outside benchmarkit and press Restart in Settings → MCP: each entry's enable flag is the only switch, and this file is the only source of which connectors exist. Add a connector with a \"remote\" url or a \"local\" command, exactly like the two below. An underscore key is a comment, never a connector.",
  "_paths_note": "The paths below were written for this install: the Python that runs benchmarkit and the two connector scripts in its mcp_connectors folder. They are absolute because a connector child process does not inherit benchmarkit's working directory. ${BENCHMARKIT_PATH_LOCATION} is left as written and expanded on every read to the active working location, so this file stays correct after a workspace switch.",
  "console": {
    "type": "local",
    "enable": true,
    "description": "Console command",
    "command": ["{{PYTHON}}", "{{CONSOLE_SCRIPT}}", "--root-dir", "${BENCHMARKIT_PATH_LOCATION}", "--transport", "stdio"]
  },
  "fedit": {
    "type": "local",
    "enable": true,
    "description": "File editor",
    "command": ["{{PYTHON}}", "{{FEDIT_SCRIPT}}", "--root-dir", "${BENCHMARKIT_PATH_LOCATION}", "--transport", "stdio"]
  }
}
```

* `{{PYTHON}}` ← `sys.executable`; `{{CONSOLE_SCRIPT}}` / `{{FEDIT_SCRIPT}}` ←
  `MCP_CONNECTORS_DIR / "console_mcp.py"` and `"fedit_mcp.py"`. Each token is a
  **whole** path substituted into one argv slot — never a directory joined to a file
  name with `/` in the template, which on Windows would spell a path with both
  separators and read as a hand-written one.
  This is what makes "benchmarkit installed at `/home/someone/apps/benchmarkit`"
  come out right without a hand-written path anywhere in the file.
* `${BENCHMARKIT_PATH_LOCATION}` is written **as literal text**, because each
  loader's `expand_env` resolves it on every read — the mechanism that keeps the
  seeded argv pointing at the active workspace across switches.
* The placeholder spelling is `{{NAME}}`, never `${NAME}`: the latter is the
  loaders' own expansion syntax, and an unresolved `${NAME}` in a declaration is
  reported as a permanent validation note (and, in a `local` `command`, produces an
  unusable argv). `{{…}}` is inert to every loader, so a template read *without*
  substitution silently yields an invalid path — which is exactly why §7 pins the
  substitution instead of trusting the bytes.
* Both `local` entries keep `enable: true`: today's `externals.json` ships them
  enabled, and batch mode needs both.
* `context7` and `chrome-devtools` are removed from the template (optional
  third-party connectors, one of them carrying a key). They come back by hand,
  which is the same act as declaring any new connector.

Rejected: generating the seeded JSON from a Python literal in a new module, the way
`policy_seed.DEFAULT_POLICY_RULES` and `agent_modes_defaults` do. Here that would
put the same declaration in two places — the file an operator reads to learn the
format, and the literal the app actually writes — and they can drift. A committed
template file also gives the seeder a legitimate reason to read an install-level
path (`mcp_connectors/mcp-settings.default.json`) instead of inventing one.

Rejected: writing `${…}` references for the install paths so nothing is
substituted. `sys.executable` is not in the environment, and inventing a
benchmarkit-specific name for it would be a third expansion mechanism beside
`${NAME}` and `${NAME:-fallback}` — the loaders' expansion contract is a project
invariant.

### D6 — an underscore key is a comment in both hubs

`lsphub/discovery.py:230` skips a key starting with `_`; the MCP loader has no such
skip, so a commented declaration there renders as a per-entry alert. The seeded
file needs comments (D5 carries two), and the fix that keeps the two hubs symmetric
(AGENTS.md) is a `connector_id.startswith("_")` skip in
`mcphub/discovery.load_externals`, with the same one-line comment. `_note` is then
a comment in both files, and `set_externals_enabled` keeps it untouched exactly as
`set_lsp_enabled` does — both re-write the parsed document and preserve unknown
keys.

### D7 — the LSP side writes only the example

The seeder copies `lsp_connectors/lsp-settings.example.json` byte-for-byte to
`<root>/.nojae/lsp-settings.example.json`, and `.nojae/lsp-settings.json` is
created by nobody but the user. A workspace with no such file is a **normal state,
not an error**: `load_lsp_servers` already answers `([], [])` for a missing file, so
the Settings panel says "copy `.nojae/lsp-settings.example.json` to
`.nojae/lsp-settings.json` and adapt it" — which is the sentence that panel and the
WorkBench LSP panel already try to say.

The copy is byte-for-byte, not a re-render: the file is documentation-first (its
three `_…_note` keys explain `${NAME}` expansion and the Windows spelling of the
same three entries) and its `/absolute/path/to/…` placeholders are deliberately
placeholders — `test_lsp.py:625-656` pins that the committed example loads with
*only* placeholder notes, precisely so nobody edits a real machine path into the
template. A user who copies without editing therefore gets per-entry
`workspaceRoot` notes in `lspServersError`, which is honest (the root is wrong) and
self-explaining (the note names the entry and the key). A re-rendered, pre-filled
`workspaceRoot` would look configured while its `command` is still a guess, which
is worse.

Rejected: seeding `.nojae/lsp-settings.json` itself. It would arrive with
placeholder paths no loader can accept — noise in place of a config — and it would
make the app the author of a declaration it cannot know.

Rejected: seeding a trimmed file with `enable: false` entries. The example's three
entries are *already* `enable: false`; the noise comes from `workspaceRoot`, which
is refused whatever the flag, so trimming buys nothing the byte-for-byte copy does
not already cost.

### D8 — no legacy migration in code

`mcp_connectors/externals.json` and `lsp_connectors/lsp_servers.json` stop being
read, on every machine, with no note pointing at them. Their content is moved
**once, by hand, for this workspace** (the commands in §6), because:

* writing `.nojae/lsp-settings.json` from app code is what D7 rules out, and an
  asymmetric lift (MCP only) would leave a code path that is only correct for an
  install that happens to own that file's secrets;
* the app cannot tell whether the legacy file it found is *this* user's declaration
  or a stranger's left over from another checkout;
* the two legacy ignore entries stay in `.gitignore`, so a leftover file can never
  be committed by accident on any machine.

Rejected: `prompt_helpers._lift_legacy_helpers`' pattern (read the legacy file on
the exclusive-create branch only). It worked there because the legacy file is the
app's *own store*, with a known shape and a `migrated` state to report; here it is
user-authored config holding secrets whose owner the app cannot identify.

### D9 — the toggle keeps writing the declaration file, inside `.nojae`

`set_externals_enabled` / `set_lsp_enabled` are unchanged apart from the path they
resolve, so the Settings toggles keep writing `enable` back to the one file that is
the source of truth (AGENTS.md). This is a deliberate exception to the reserved
folder's "edited only outside the app" rule, and it is not new:
`prompt-helpers.json` already made it. The Editor's write surface is **not** that
exception — `.nojae` is held back by the unconditional built-in rule in
`builtin_policy.builtin_rules`, which no level-B rule can unlock, so the
declarations stay unreachable from the file pane and from every connector. No
level-A rule is added or removed (`mcp_connectors` is already a fedit write-deny
part, and nothing here touches it).

## 3. What exists today (verified against the code)

| piece | where | what it does |
|---|---|---|
| MCP path | `mcphub/constants.py:24` | `EXTERNALS_PATH = MCP_CONNECTORS_DIR / "externals.json"` — import-time. |
| LSP path | `lsphub/constants.py:24` | `LSP_SERVERS_PATH = LSP_CONNECTORS_DIR / "lsp_servers.json"` — import-time. |
| readers | `mcphub/discovery.py:72,182`, `lsphub/discovery.py:207,389` | `root = path or constants.X` — the read and the `enable` write-back, one seam each. |
| who asks | `routes.py:651,987,1032,1085`, `server.py:190,201` | active-id lists, the payloads, the hub restarts, the lifespan's first scan. |
| boot seeds | `server.py:158-175` | `_seed_workspace_policy` → `_seed_agent_modes` → `_seed_prompt_helpers` → `_sweep_trash`, all before the lifespan. |
| switch | `routes.py:1379-1445` | env set → 2 restarts → 3 seeds → payload; `except BaseException` rolls the env back. |
| read-only states | `routes.py:1259-1293` | `_policy_block` / `_agent_modes_block` / `_prompt_helpers_block`, each a `*_file_state(path_location())`. |
| seeders to mirror | `policy_seed.py`, `agent_modes.py:801-960`, `prompt_helpers.py:251-347` | each one `os.mkdir`s the folder, exclusive-creates the file, answers `_state_dict`, reports a folder at the path as `failed`, and discards a half-written file. |
| path pattern to copy | `agent_modes.py:588-600` | `_active_root()` = `realpath(path_location())`; `agent_modes_path()` built per call out of module attributes. |
| `_` comments | `lsphub/discovery.py:230` | the skip the MCP loader does not have. |
| strings naming the old paths | `SettingsDialog.vue:451,467,517,534-535`, `WorkBenchMcpPanel.vue:53,63`, `WorkBenchLspPanel.vue:51,61`, `useChat.js:282`, `useBenchmark.js:62,70,72` | captions and comments — there is no path logic on the client. |
| `.env` template text | `config.py:478-483` | re-written into every `.env` on save, so it is a code edit, not a doc edit. |
| ignores | `.gitignore:22-26`, `.kilocodeignore:18` | both name the legacy paths; `.nojae/` is already ignored at `.gitignore:29`. |
| tests that pin the constants | `test_mcp.py:802`, `test_lsp.py:2102,2499`, `test_api.py:682,690`, `test_agent_modes_seed.py:318,324`, `test_editorfs_policy_seed.py:291,297`, `test_fscase.py:283,290`, `test_prompt_helpers_api.py:74,80` | all `monkeypatch.setattr(…, "EXTERNALS_PATH"/"LSP_SERVERS_PATH", …)` — D3's pin keeps them meaningful under the new names. |
| tests that read the templates | `test_lsp.py:637,673`, `test_write_surface_isolation.py:216` | name `lsp_connectors/lsp_servers.example.json`; they follow the rename. |
| legacy example file | `lsp_connectors/lsp_servers.example.json` | three `_…_note` keys + `pyright`/`typescript`/`vue`, all `enable: false`. |
| legacy live file on this machine | `mcp_connectors/externals.json` | `context7` (with a key), `chrome-devtools`, `console`, `fedit` (absolute pyenv-python + repo paths). |
| the two shipped connectors | `mcp_connectors/console_mcp.py`, `mcp_connectors/fedit_mcp.py` | what the seeded entries must point at; `test_write_surface_isolation.py:188-205` already treats them as the connector set. |

## 4. Shape of the new pieces

| file | change |
|---|---|
| `src/core/mcphub/constants.py` | `EXTERNALS_PATH` → `MCP_SETTINGS_PATH: Optional[Path] = None`; new `MCP_SETTINGS_FILENAME = "mcp-settings.json"` and `MCP_SETTINGS_TEMPLATE = MCP_CONNECTORS_DIR / "mcp-settings.default.json"` (both patchable); `def mcp_settings_path() -> Path` — the pin wins when set, else `path_location() / POLICY_DIRNAME / MCP_SETTINGS_FILENAME`; the three template placeholder tokens (`{{PYTHON}}`, `{{CONSOLE_SCRIPT}}`, `{{FEDIT_SCRIPT}}`) and what each resolves to; module docstring rewritten — the declaration is a *workspace* file, seeded, and the install-level file beside it is a template. |
| `src/core/mcphub/seed.py` | **new**, the hub's own module so the mirror stays whole (AGENTS.md's `constants` / `discovery` / `manager` trio gains `seed`). `mcp_settings_file_path(root)`, `default_mcp_settings_text()` (reads the template and substitutes the three tokens — the one place that touches it), `ensure_mcp_settings_file(root)`, `mcp_settings_file_state(root)` and the private helpers, following `agent_modes.py:801-960`'s structure (module-level `_state_dict` / `_is_regular_file` / `_discard_seeded_file`, and the folder-making block D2 describes, `_ensure` never raising). No cache to invalidate: the hub re-scans the file on every load, so unlike the policy and the modes there is nothing to forget. |
| `src/core/mcphub/discovery.py` | both `root = path or constants.mcp_settings_path()`; the D6 `_`-key skip; docstrings name `.nojae/mcp-settings.json`. |
| `src/core/mcphub/__init__.py` | re-exports follow the rename, through the existing PEP-562 `__getattr__` (`MCP_SETTINGS_PATH`, `MCP_SETTINGS_FILENAME`, `ensure_mcp_settings_file`). |
| `src/core/lsphub/constants.py` | mirror: `LSP_SETTINGS_PATH: Optional[Path] = None`, `LSP_SETTINGS_FILENAME = "lsp-settings.json"`, `LSP_SETTINGS_EXAMPLE = LSP_CONNECTORS_DIR / "lsp-settings.example.json"`, `def lsp_settings_path() -> Path`. |
| `src/core/lsphub/seed.py` | **new**, the mirror: `lsp_example_file_path(root)`, `ensure_lsp_example_file(root)` (byte-for-byte copy of the template), `lsp_settings_file_state(root)` (read-only over `.nojae/lsp-settings.json`, whose `absent` sentence names the example to copy), same private helpers, same four states. |
| `src/core/lsphub/discovery.py` | both `root = path or constants.lsp_settings_path()`; docstrings name the new path. |
| `src/core/lsphub/__init__.py` | re-export renames, as the MCP package. |
| `src/api/routes.py` | `_mcp_settings_block()` + `_lsp_settings_block()` beside the three existing read-only blocks; the POST runs the two new seeds **before** their restarts (D4) and answers `"mcpSettings"` / `"lspSettings"`; the comment that justifies today's order is rewritten to state the new one and why. Docstrings naming the old files (`:1021-1023`, `:1093`, `:1299-1301`) name the workspace paths. |
| `src/api/server.py` | `_seed_mcp_settings()` + `_seed_lsp_example()`, each `try/except` → one `_log.warning` note and never a boot failure, called after `_seed_prompt_helpers()` and before `_sweep_trash()`; the lifespan comments at `:187` and `:198` name the new files. |
| `mcp_connectors/mcp-settings.default.json` | **new, committed**: D5's two connector entries and two comment keys, placeholders and all. |
| `mcp_connectors/externals.json` | **deleted** (its `console`/`fedit` shape is D5's source; its content is hand-moved per D8). |
| `lsp_connectors/lsp-settings.example.json` | **rename** of `lsp_servers.example.json`; content unchanged except its own first `_note`, which currently says "Copy to lsp_servers.json" and must read "copy this file to `.nojae/lsp-settings.json`". |
| `src/config.py` | the `.env` template's MCP/LSP comment block (`:478-483`) names `.nojae/mcp-settings.json` and `.nojae/lsp-settings.json`. |
| `web/src/lib/path-location-policy.js` | two more describers over the same `POLICY_STATES`, differing only in the `absent` sentence — MCP `absent` = "no connectors declared, the agent tabs are empty"; LSP `absent` = "copy the `.example` file beside it to get language-server tools". The header comment that enumerates the folder's files stops enumerating. |
| `web/src/composables/useBenchmark.js` | `pathLocationMcpSettings(+Note)` / `pathLocationLspSettings(+Note)`, read from the payload on GET and POST and cleared per attempt — the pattern at `:818-860` and `:869-917`. |
| `web/src/components/SettingsDialog.vue` | the two empty-state captions (`:467`, `:534-535`) name the new paths (and the LSP one names `.nojae/lsp-settings.example.json` as the thing to copy); the working-dir section renders the two new notes as it renders the other three. |
| `web/src/components/WorkBenchMcpPanel.vue`, `WorkBenchLspPanel.vue` | the restart titles (`:63` / `:61`) name the workspace file. |
| `web/src/composables/useChat.js:282`, `useBenchmark.js:62,70,72` | comments only. |
| `.gitignore` | the two legacy lines **stay**, each with a comment saying the file is legacy and no longer read (a leftover on any machine must remain untracked); nothing is added for the new templates — they live in committed folders and no rule covers them. |
| `.kilocodeignore` | `mcp_connectors/externals.json` → the two legacy lines, same reasoning. |
| `docs/seeding-builtin-mcp-spec.md` | **deleted**. |
| `README.md` | the declaration table (`:176-177`) and the sentence under it about gitignored secrets (`:179-180`); the two section headings + copy steps (`:182`, `:273-276`); the sentence about the shipped connectors (`:265-270`) becomes "`console` and `fedit` are seeded into a new workspace's file; anything else is added by hand". |
| `AGENTS.md` | see §6. |

## 5. Out of scope

* **Automatic migration** of the two legacy files (D8), on any machine.
* **Seeding `.nojae/lsp-settings.json`** (D7) — the app never authors a
  language-server declaration.
* **Shipping third-party connectors** (`context7`, `chrome-devtools`) in any
  template, and any handling of their keys.
* **A new expansion mechanism** for install paths (D5) — `${NAME}` and
  `${NAME:-fallback}` stay the whole contract.
* **Editing the declarations from the Editor** — `.nojae` stays hidden and
  unwritable there, as it is for the policy and the modes.
* **Per-workspace hub state** beyond the file: no cache, no history, no
  import/export, nothing copied forward on a switch.
* **A `version` key** in either declaration file. The loaders ignore unknown keys,
  a bad shape is already a per-entry note, and neither file has a second format to
  tell apart — a version would be a rule with nothing to apply to.
* **The LSP `restricted-paths.txt` deny-list**, which stays install-level on
  purpose: it guards against secrets, so it cannot live in a per-project file (the
  same reasoning AGENTS.md gives for keeping the connector deny-lists separate).

## 6. Moving this workspace's own declarations (manual, once)

The app reads neither legacy file once this lands, so this checkout's connectors and
servers are moved by hand — **after** the code lands, because the first boot of the
new code is what creates `.nojae/mcp-settings.json`:

```bash
# 1. boot the app once against this workspace, then stop it.
# 2. add this machine's two optional connectors (context7, chrome-devtools)
#    by hand to .nojae/mcp-settings.json — context7's header carries a key,
#    and .nojae is gitignored, so the key does not become source.
# 3. the language servers:
cp .nojae/lsp-settings.example.json .nojae/lsp-settings.json
# then edit it: enable pyright / typescript / vue, point their workspaceRoot
# and --tsdk at this checkout, and delete the _…_note keys.
# 4. drop the files nothing reads any more:
rm mcp_connectors/externals.json lsp_connectors/lsp_servers.json
```

`.nojae` is hidden from the Editor's tree and from every connector, so steps 2-3 are
done in a normal editor (or the Editor's terminal tab) — which is what the reserved
folder already means for the policy and the modes file.

**AGENTS.md**, three edits, each stating an invariant rather than a list:

* the `core` row of the code layout — the hubs' mirror sentence gains `seed`
  alongside `constants` and `discovery`, phrased as "leave the workspace its
  declaration file the first time it is opened";
* the data-files table — the two declaration rows move into the reserved-folder
  row's wording instead of naming install paths. The row stops enumerating the
  folder's files by name (it holds the policy, the modes, the helper store and the
  two hub declarations; a count is exactly the detail AGENTS.md says to avoid) and
  instead says what changed for the declarations: they describe *this* workspace;
* the "The declaration files are the only source of truth for what runs" bullet —
  one added clause: they are the **workspace's** declarations in the reserved
  folder, seeded from a committed template when a folder becomes the working
  location, and a template names neither a machine nor a secret.

## 7. How it is verified

New backend file `tests/test_hub_settings_seed.py` — the two seeds are one feature
with one asymmetry, so one file, mirroring `tests/test_agent_modes_seed.py`. Each
test pins `BENCHMARKIT_PATH_LOCATION` to its own `tmp_path` the way
`test_agent_modes_seed.py:45` does.

| test | asserts |
|---|---|
| exclusive create, never clobber | a hand-written `.nojae/mcp-settings.json` survives `ensure_mcp_settings_file` byte-for-byte and answers `existing`; a **folder** at that path answers `failed`, not `created` |
| **the seed substitutes** | the created file's argv holds `sys.executable` and the two `MCP_CONNECTORS_DIR` script paths, contains no `{{` and no `}}`, keeps `${BENCHMARKIT_PATH_LOCATION}` as literal text, and parses through `load_externals` with `errors == []` and ids `["console", "fedit"]` |
| the template is not a working file | the committed `mcp-settings.default.json` still holds the `{{…}}` tokens (so it can never be mistaken for a declaration) and declares exactly the two connectors — the test that keeps `context7`/`chrome-devtools` out |
| `_` keys are comments | `load_externals` skips `_note` / `_paths_note` and reports nothing about them (D6); `set_externals_enabled` writes `enable` back with both keys still in place and in order |
| a reader never writes | `mcp_settings_file_state` / `lsp_settings_file_state` on an empty root answer `absent` and leave **no `.nojae` behind**; the LSP `absent` sentence names `.nojae/lsp-settings.example.json` |
| the LSP seed copies bytes | `.nojae/lsp-settings.example.json` is byte-identical to `lsp_connectors/lsp-settings.example.json`; **`.nojae/lsp-settings.json` does not exist** afterwards; the three `_…_note` keys and the three server ids survive the copy |
| a missing template is a note | template patched to a non-existent path → `state: "failed"` naming the template, and no half-written file at the destination |
| **the workspace owns the file** | two roots, two files: each seeded `--root-dir` expands to its own root, and switching `BENCHMARKIT_PATH_LOCATION` moves `mcp_settings_path()` / `lsp_settings_path()` with it (D3's resolver) |
| the pin still pins | `MCP_SETTINGS_PATH` / `LSP_SETTINGS_PATH` patched → the loader and the `enable` write-back use it whatever `path_location()` says; left at `None` they use the workspace's file |
| the existing pins keep their meaning | the `test_mcp` / `test_lsp` / `test_api` / `test_agent_modes_seed` / `test_editorfs_policy_seed` / `test_fscase` / `test_prompt_helpers_api` patch sites still assert what they asserted — including the one that the app's own `mcp_connectors/` declaration is never written |
| `GET` never seeds | both blocks answer `absent` (then `existing`) across two `GET /api/path-location` calls on an unseeded root; `POST` answers `created` for both and returns them as `mcpSettings` / `lspSettings` |

The order invariants, as tests rather than sentences:

| test | asserts |
|---|---|
| **switch order** | `POST /api/path-location` onto a fresh root leaves the hub holding the `console`/`fedit` connectors — proof the restart read the seeded file, not a scan of an unseeded folder. This is the test that fails if the seed moves back after `_restart_mcp_hub`. LSP needs no configured server to pass, so it asserts the LSP restart met a *missing* `.nojae/lsp-settings.json` without an error, and that the example landed |
| a failing seed is a note | `ensure_*` patched to answer `failed` → the response still carries the new `path`, both hub payloads and the two notes; the environment is **not** rolled back and the status is not 4xx |
| boot order | `create_app()` on an empty root (hubs off, per the autouse conftest switches) leaves `.nojae/mcp-settings.json` and `.nojae/lsp-settings.example.json` behind and `.nojae/lsp-settings.json` absent |

Frontend: one Vitest case per new describer in `web/tests/pathLocationPolicy.test.js`
(all four states, the differing `absent` wording, `failed` → `tone: "warn"`);
`web/tests/useWorkspaceSwitch.test.js` gains the two note fields beside the three it
already asserts. One e2e addition in `web/e2e/03-workspace-switch.spec.js`: switch
to a fresh scratch root and assert the MCP list is `console` + `fedit` **with no
validation error**, and that the section names `<root>/.nojae/mcp-settings.json` —
the user-visible point of the whole change.

Manual, on this machine, after §6: Settings → MCP lists `console`, `fedit`,
`context7`, `chrome-devtools` with no validation note and no unresolved-`${…}`
alert; Settings → LSP lists pyright / typescript / vue and its toggles write
`.nojae/lsp-settings.json`; the Editor's tree shows no `.nojae` entry; a WorkBench
turn can call `execute_command` and `apply_edit`; switching to another project gives
the same two connectors with their `--root-dir` at the new root; `git status` shows
nothing from `.nojae` and no deleted `externals.json`.
