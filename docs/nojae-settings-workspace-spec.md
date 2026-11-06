# Workspace-scoped LLM settings — `.nojae/nojae-settings.json`

Status: spec, complete — no open questions. Not implemented. Scope: move the
model-connection and sampling settings out of the global `.env` into a
per-working-location file, so each workspace targets its own server, model,
prompt and sampling parameters.

## 1. What this is for

Today one `.env` holds everything, so switching from a 27B llama.cpp box to a
hosted 8B model means editing the same file that also holds the verifier
credentials — and the two tabs (WorkBench, Benchmark) always talk to whatever
the last save left there. The install directory's `.env` is also the *only*
place these values can live, so a workspace cannot travel with its own model
choice the way `.nojae/mcp-settings.json` and `.nojae/lsp-settings.json`
already travel with their own connectors and servers.

After this change:

- **`.nojae/nojae-settings.json`** (working location, reserved folder) holds
  everything a model connection needs — server, key, model id, system prompt,
  sampling, vision flag, LSP activation, tool-loop cap. One file per workspace,
  edited from the Settings dialog or by hand.
- **`.env`** keeps only what is scoped to *this benchmarkit install*: the
  verifier agent, the question set, and the two technical keys.

### The three groups, and where each ends up

| Group | Keys | Home after this change |
|---|---|---|
| 1 — model connection, read by WorkBench **and** Benchmark | `SERVER_URL`, `SERVER_URL_API_KEY`, `ENDPOINT`, `SUPPORTS_VISION`, `LLM_SYSPROMPT`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`, `LSP_ENABLED`, `MAX_TOOL_ITERATION` | `.nojae/nojae-settings.json` |
| 2 — Benchmark only | `VERIFIER_AGENT_SYSPROMPT`, `VERIFIER_AGENT_SERVERURL`, `VERIFIER_AGENT_SERVERURL_API_KEY`, `VERIFIER_AGENT_ENDPOINT`, `QUESTION_SET` | `.env` (unchanged) |
| 3 — technical, not editable from the UI | `LLM_DEBUG`, `BENCHMARKIT_PYTHON_ROOT` | `.env` (unchanged) |

`LLM_DEBUG` is in group 3 by intent, but the dialog edits it today — it leaves
the dialog with this change (§7).

## 2. The file

Seeded copy of the committed template `nojae-settings.default.json`, at
`<working location>/.nojae/nojae-settings.json`.

```json
{
  "serverUrl": "Put here the url of your OpenAI-compatible server (do not omit /v1 at the end)",
  "serverApiKey": "Put here the api key if your server needs one (leave blank otherwise)",
  "endpoint": "Put here the model id to send requests to, exactly as your server names it",
  "supportsVision": false,
  "sysprompt": "You are a helpful assistant.",
  "temperature": 0.2,
  "topP": 0.95,
  "topK": 20,
  "minP": 0,
  "repeatPenalty": 1.0,
  "presencePenalty": 0,
  "reasoning": "off",
  "lspEnabled": [],
  "maxToolIteration": 300
}
```

### Key map

| `.env` key (before) | JSON member | JSON type | Member absent means | Today's default / rule |
|---|---|---|---|---|
| `SERVER_URL` | `serverUrl` | string | no endpoint | `""` (`EndpointsConfig`); written by `REQUIRED_SETTING_KEYS` |
| `SERVER_URL_API_KEY` | `serverApiKey` | string | no `Authorization` header | `""` |
| `ENDPOINT` | `endpoint` | string | no model | `None`; `_strip_endpoint` blanks it; dialog fills it from `/models` |
| `SUPPORTS_VISION` | `supportsVision` | boolean | `false` | `False`; `env_flag` truthy set `1/true/yes/on` |
| `LLM_SYSPROMPT` | `sysprompt` | string | app default | `DEFAULT_LLM_SYSPROMPT` |
| `TEMPERATURE` | `temperature` | number | omitted from the request body | `None`, `exclude_none` in `SamplingParams.to_kwargs()` |
| `TOP_P` | `topP` | number | omitted | `None` |
| `TOP_K` | `topK` | integer | omitted | `None` |
| `MIN_P` | `minP` | number | omitted | `None` |
| `REPEAT_PENALTY` | `repeatPenalty` | number | omitted | `None` |
| `PRESENCE_PENALTY` | `presencePenalty` | number | omitted | `None` |
| `REASONING` | `reasoning` | string enum | omitted | `Literal["off","low","medium","high","xhigh"]`, lower-cased |
| `LSP_ENABLED` | `lspEnabled` | array of strings | `[]` → no server starts | `lsp_enabled_ids()` split/dedup/order |
| `MAX_TOOL_ITERATION` | `maxToolIteration` | integer | 25 | `DEFAULT_MAX_TOOL_ITERATION = 25`, `max(1, int(...))` |

`maxToolIteration` is the one member whose template value is a real value
(`300`) rather than an instruction: a seeded workspace is expected to run agent
turns out of the box, and 300 against the code default of 25 is the difference
between a batch task finishing and the lot aborting. `lspEnabled`'s template
value is `[]` because an array cannot hold prose without it becoming a bogus
id — `[]` is also the honest "nothing enabled yet", the state the LSP section
of the README already describes for a workspace with no declaration.

## 3. Decisions

**D1 — the file is a store the app writes, seeded declaration-style.** The
Settings dialog keeps writing all of group 1, so a dialog save must land in
this file. Rejected: hand-edit only (the `/models` dropdown, the sampling
sliders and the LSP toggles would stop persisting — a real loss of control);
and rejected: keeping group 1 in `.env` and layering a workspace override on
top, which leaves two sources of truth for one number and no way to tell which
one answered.

**D2 — no `.env` fallback for group 1.** The file is the single source:
`SERVER_URL` and friends are no longer read from `os.environ` at all. Rejected:
"file wins, `.env` as fallback" — a leftover `.env` from before the change
would silently answer for whichever member the file left out, which is exactly
the ambiguity this change removes. Consequence: group-1 lines left in an
existing `.env` are dead bytes — read by nothing, and the workspace starts from
the seeded template (§9).

**D3 — app-owned-store exception, not a policy change.** No path-policy edit is
needed: `HIDDEN` on `.nojae` binds the Editor's `/api/fs/*` surface and the
connectors, never an app-internal writer — `core/prompt_helpers.py` already
writes `.nojae/prompt-helpers.json` with a plain `write_text`, and the five
`_seed_*` calls in `api/server.py` create files there directly. What changes is
AGENTS.md's sentence *"changed only outside the app — except the app's own
prompt-helper store"*, which names the folder's two app-owned stores:
`prompt-helpers.json` and `nojae-settings.json`.

**D4 — read at boot and on location switch; a dialog save is live, a hand-edit
needs a reboot.** Same contract `.env` already has (`load_dotenv_once()` runs
once, `save_settings()` refreshes `os.environ` in place, so a hand-edit of
`.env` is invisible until restart). Rejected: re-reading per request
declaration-style — it forces a lost-update rule for a hand-edit made while
the dialog is open, and the reward (hot hand-edits) duplicates the dialog. The
dialog says so in one line (§5, Frontend) so the reboot is expected, not
discovered. `lspEnabled` from a **dialog** save reconciles live servers in the
same request exactly as `post_lsp()` does today; a hand-edit of it waits for
the reboot like the rest.

**D5 — `.nojae` stays invisible to the built-in Editor.** Because `HIDDEN` is
absolute, the file is not in the tree, not readable through `/api/fs/read` and
not in git status — hand-editing means *outside* the app, the same as
`mcp-settings.json` and `agent-modes.json`. Stated explicitly so nobody
specifies a tree entry for it.

**D6 — native JSON types, and therefore a second builder.**
`EndpointsConfig.from_env()` cannot take them: its local `get()` strips only
`str`, and `supports_vision` is computed from a string membership test, so
native `false` / `0.2` / `["pyright"]` would read as falsy or raise. A new
builder takes the JSON mapping and the `.env` mapping separately (§5).
Rejected: writing the file as strings to keep one builder — it preserves the
dotenv wart (`"0.2"`, `"1"`, comma lists) for no gain, and `SamplingParams`
already coerces numbers.

**D7 — member names are the approved ones; the dialog keeps its field names.**
The file uses `serverApiKey` (fixing the `SERVER_URL_API_KEY` typo) and
`temperature`. The dialog's payload fields stay `apiKey` and `temp`, so the
port is one new mapping table rather than a rewrite of
`web/src/components/SettingsDialog.vue`. Rejected: renaming the API fields in
the same step — touches the dialog, `useBenchmark.js` and their tests for no
functional gain. Rename later, separately, if the mismatch earns it.

**D8 — a value that is still the seed text is reported, not dialed.** The
seeded file holds instruction sentences in `serverUrl`/`endpoint`, so a
freshly seeded workspace is valid JSON and unusable. The template's instruction
values are matched on read and answered as "not configured" with a message
naming the file and its path, instead of a transport error quoting a sentence.
Rejected: seeding a `.template.json` the app never reads, the `lsphub` way —
here the app *does* own the file, so a copy-me step for a file the dialog can
already write would be a chore, not a guard.

**D9 — a malformed file is a named state, never a silent default.** Follows
`prompt_helpers._state_dict()`'s `ok` / `absent` / `failed`: broken JSON or an
unknown member is reported in `GET /api/settings` (so the dialog can show the
parse error the user just typed) and a chat or benchmark request refuses with
that same sentence. Rejected: falling back to defaults — a typo in
`"temperature"` silently benchmarking is the worst outcome this feature could
ship, and `EndpointsConfig`'s `extra="ignore"` would swallow the typo silently,
so the reader checks the member set itself.

**D10 — the seed goes before the hub restarts on a location switch.**
`post_path_location()` seeds `mcp-settings.json` *before* the restarts for
exactly this reason (`routes.py:1424`), and the other seeds follow the restarts
because a seeding hiccup must not look like a hub failure. `lspEnabled` is read
during `_restart_lsp_hub()` (`_active_server_ids()` → `lsp_enabled_ids()`), so
seeding after would leave the new workspace on zero language servers until the
next restart. At boot it goes after `_seed_workspace_policy()` (that one
creates the folder) and before the lifespan's first LSP scan.

## 4. What it builds on

- `src/core/prompt_helpers.py` — the folder's other app-owned store: the shape
  to copy (`prompt_helpers_file_path(root)`, `prompt_helpers_path()` reading
  `path_location()` per call, the `ok`/`absent`/`failed` state dict, and the
  name-patchability loop over `("PROMPT_HELPERS_FILENAME", "POLICY_DIRNAME", ...)`).
- `src/core/mcphub/seed.py` + `mcp_connectors/mcp-settings.default.json` —
  exclusive-create seeding of a committed default at the active location, and
  the committed-default-lives-outside-`.nojae` rule (`.gitignore` carries
  `.nojae/`, so the template cannot live there).
- `src/core/models.py` — `SamplingParams` + `EndpointsConfig`, whose validation
  already rejects a bad `reasoning` value and coerces types.
- `src/api/routes.py` — `SETTINGS_ENV_KEYS` / `VERIFIER_ENV_KEYS` as the
  existing dialog-field↔key tables, `post_lsp()` as the live-reconcile path for
  `lspEnabled`, and the `_seed_*` state blocks in `GET /api/settings` as the
  pattern for the seed note.
- `src/api/server.py` — the boot seed sequence and the two-gate LSP start
  (`enable` in `lsp-settings.json` **and** `lspEnabled`).
- AGENTS.md's trio rule (`SETTING_KEYS`, `_render_env()`, `.env.example` move
  together) and its `_RESERVED_ENV_NAMES` lesson: nothing here adds an
  environment key, so no new OS-name collision is possible.

## 5. Shape of the new pieces

### New module — `src/core/workspace_settings.py`

One module beside `prompt_helpers.py`, same conventions, owning the file
end-to-end:

- `WORKSPACE_SETTINGS_FILENAME = "nojae-settings.json"`,
  `WORKSPACE_SETTINGS_TEMPLATE = <install root>/nojae-settings.default.json`
  (both patchable, in the module's name-patchability loop like
  `prompt_helpers`').
- `workspace_settings_path()` — `path_location()` + `POLICY_DIRNAME` +
  filename, resolved per call.
- `ensure_workspace_settings_file(root) -> dict` — exclusive-create copy of the
  template, answers a dict, raises nothing (so a hiccup is a note, never a hub
  or boot failure).
- `load_workspace_settings()` — cached read; `ok` / `absent` / `failed` state,
  the seed-text check of D8, and the member-set check of D9.
- `save_workspace_settings(values)` — writes only the keys it is given,
  preserving unknown-but-valid ones, then invalidates the cache.
- `invalidate_workspace_settings()` — called by the save path and by
  `post_path_location()`.
- `workspace_setting(name)` — the typed accessor group 1's readers use.

`lsp_enabled_ids()` and `max_tool_iteration()` stay where they live
(`src/config.py`) and change only their source: `os.environ` →
`workspace_setting(...)`. Keeping them put keeps `chat.py`, `server.py` and
`routes.py` importing what they already import.

### New template — `nojae-settings.default.json` (committed, install root)

The JSON of §2. At the root rather than beside the reader because its sibling
is `.env.example`: it is the committed template of a settings file, not a
connector (the `mcp_connectors/` and `lsp_connectors/` precedents are for
connector declarations).

### `EndpointsConfig` — one new builder

`EndpointsConfig.from_workspace(workspace: Mapping[str, Any], env: Mapping[str, str])`
in `src/core/models.py`: group 1 from the JSON mapping (native types pass
straight to pydantic), the `VERIFIER_AGENT_*` block from the env mapping.
`from_env()` loses its group-1 reads — its doc comment currently *lists* those
keys, so it changes with them. Both callers move:
`config.load_endpoints_config()` and `routes._load_config()`.

### `src/config.py` — the `.env` shrink

- `SETTING_KEYS` keeps: the four `VERIFIER_AGENT_*` keys, `QUESTION_SET`,
  `LLM_DEBUG`, `BENCHMARKIT_PYTHON_ROOT`. Group 1 leaves it.
- `REQUIRED_SETTING_KEYS` becomes empty (it named `SERVER_URL` and `ENDPOINT`),
  so `.env` gains no always-written key.
- `_render_env()` loses the group-1 lines and their comments, gains a
  `.nojae/nojae-settings.json` pointer comment where the `LSP_ENABLED` /
  `MAX_TOOL_ITERATION` comments sat — same job the existing MCP comment does.
- `.env.example` changes in the same commit (AGENTS.md trio rule).
- `load_settings()` / `save_settings()` keep their `.env` job only.
- `normalize_env_flag()` loses its `SUPPORTS_VISION` caller; `LLM_DEBUG` keeps
  one (§7).
- The `LEGACY_LLM_DEBUG_KEY` / `MAX_MCP_ITERATION` fallbacks: `MAX_MCP_ITERATION`
  stops being honoured (it was a `.env`-hand-write escape hatch for a file that
  no longer holds the key); an absent `maxToolIteration` falls back to
  `DEFAULT_MAX_TOOL_ITERATION` as today.

### `src/api/routes.py` — the dialog splits across two writers

- `WORKSPACE_SETTING_KEYS: Dict[str, str]` — dialog field → JSON member
  (`serverUrl→serverUrl`, `apiKey→serverApiKey`, `temp→temperature`,
  `supportsVision→supportsVision`, …), the sibling of the existing
  `SETTINGS_ENV_KEYS` / `VERIFIER_ENV_KEYS` tables.
- `GET /api/settings` answers from both stores and grows the workspace state
  block (`seeded` / `ok` / `failed` + message) beside the existing seed blocks.
- `POST /api/settings` routes group 1 to `save_workspace_settings()` and group
  2/3 to `save_settings()`. The existing `EndpointsConfig` validation gate runs
  before either write, on the merged result, so a bad `reasoning` still cannot
  reach disk. The mid-run 409 guard is unchanged.
- `_settings_env_values()` **must stop emitting `LLM_DEBUG`** (§7).
- `_settings_json()` drops `llmDebug`, adds `lspEnabled` and
  `maxToolIteration` (the LSP tab already round-trips both through
  `POST /api/lsp`, which now writes the JSON array instead of the comma string).
- `post_lsp()` writes `lspEnabled` + `maxToolIteration` through
  `save_workspace_settings()` and keeps its live `start_server`/`stop_server`
  reconcile and `_restart_lsp_hub()` call unchanged.
- `post_path_location()` seeds before the restarts (D10) and invalidates the
  cache, so the new workspace's file — not the previous one's — answers.

### Frontend

- `SettingsDialog.vue` — group-1 fields unchanged in behaviour (their source is
  now the workspace file), `LLM_DEBUG` field removed, the LSP tab's toggles
  unchanged. One line naming the file it edits, with the workspace path — the
  same dialog now means a different server per workspace — and one line stating
  that a hand-edit needs a server restart. The failed / still-seed-text state of
  D8 and D9 shows where the MCP `externalsError` note shows today.
- `useBenchmark.js` — the `GET`/`POST /api/settings` calls at its two existing
  call sites, unchanged in shape.

## 6. Documentation

- **README §2 table** — a fourth row: `.nojae/nojae-settings.json` (seeded from
  `nojae-settings.default.json`) | the model connection, sampling and LSP
  activation for this workspace. The `.env` row loses "Endpoint, model,
  sampling, vision flag".
- **"all three are hot" sentence** — it becomes untrue for the new file, which
  follows `.env`'s reboot rule (D4). The paragraph must say which files are hot
  (the two declaration files) and which are not (`.env`,
  `nojae-settings.json`).
- **README LSP section** (`### The LSP setup file`) — gains the `lspEnabled`
  documentation, next to the two-gate sentence it must carry: a server runs
  only when its `enable` flag in `lsp-settings.json` is true **and** its id is
  in `lspEnabled`. Real ids there: `pyright`, `typescript`, `vue` — the
  `npm install -g` line in that section already names the servers, so the
  reader learns which exist and which to enable in one place.
- **README `### LLM configuration — .env`** — reflects the smaller `.env`.
- **AGENTS.md** — the D3 exception; `.env`'s row in the Data-files table (it
  says "Endpoint, key, prompt, sampling, set, verifier"); a row for the new
  file; the "Configuration … is edited from the UI and takes effect hot"
  sentence gains its exception; the two-gate prose follows the key's move.

## 7. `LLM_DEBUG` leaves the dialog

`SettingsIn.llmDebug`, its entry in `SETTINGS_ENV_KEYS`, the `normalize_env_flag`
line in `_settings_env_values()` and the dialog field are removed. The key
stays a managed `.env` key, hand-edited like `BENCHMARKIT_PYTHON_ROOT`.

**The trap this creates:** `_settings_env_values()` currently writes
`values["LLM_DEBUG"] = normalize_env_flag(...)`, i.e. it *always* emits the
key — and `post_settings()`'s `env_values = {**load_settings(), **env_values}`
lets that emitted `""` win over the stored value. Remove the dialog field
without removing that line and every settings save comments `LLM_DEBUG` out and
pops it from `os.environ` — the debug switch would switch itself off. Removing
the line is part of the same commit; `save_settings()`'s `os.environ` refresh
and its `apply_llm_debug()` call then simply re-apply the value the file still
has. `apply_llm_debug()` keeps its boot call (`server.py:200`).

## 8. Out of scope

- Groups 2 and 3 staying in `.env` — including making the verifier workspace-
  scoped. This is the first file of its kind; the verifier can follow it.
- Renaming the API payload fields (`temp`, `apiKey`) to match the file (D7).
- Making a hand-edit hot, or any lost-update guard for concurrent writers
  (D4). `.env` behaves the same way and nobody has asked for more.
- A path-policy entry, tree entry or Editor tab for the new file (D3, D5).
- Per-workspace `QUESTION_SET`, themes, chat layout or agent modes.
- Anything about how `.nojae/mcp-settings.json` or `lsp-settings.json` behave,
  beyond the `lspEnabled` source change and the README's two-gate sentence.

## 9. No migration — decided

The app is not in use by anyone else yet, so there is no installed base to
carry over. The only existing deployment is the author's own, and the group-1
values in its `.env` will be typed into the Settings dialog once, by hand.

**D11 — the seed is always the template; nothing reads `.env`'s group-1 keys.**
No one-time move, no "seed from the leftover values" branch, no
upgrade-detection code, no README migration section. The seed path has exactly
one input (`nojae-settings.default.json`) and one behaviour, which keeps
"the file is a copy of the template, then yours" a sentence that is always true.
Rejected: a compatibility branch that seeds from `.env` when the file is absent
— it costs a second source for the file's contents and a code path that is
already dead for every future user, in exchange for saving one person one
afternoon of retyping.

Consequences to hold, precisely because nothing migrates:

- A group-1 key still set in `.env` is read by **nothing** (D2). It is not a
  fallback, not a warning, not an override — just bytes `_render_env()` no
  longer renders and `from_env()` no longer reads.
- Deleting those lines from `.env` is a manual act, done once by hand; the app
  neither does it nor notices it. `save_settings()` regenerating `.env` from the
  shrunken `SETTING_KEYS` template drops them anyway on the first save, per
  AGENTS.md ("hand-written extras lost"), so the dead bytes are transient even
  if nobody edits the file.
- The first run after this change is therefore a *new* run: the dialog answers
  with the template's instruction text, D8 reports "not configured", and the
  user fills the dialog as if from scratch. That is the intended first-run
  experience, not a regression to smooth over.

## 10. How it is verified

New tests, in the hermetic style the folder's other seeds already use
(`monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))`):

- **`tests/test_workspace_settings_seed.py`** — a new location gets the file
  with the template's bytes; an existing file is never overwritten; the seed is
  a note, never a boot or switch failure; **the seed lands before
  `_restart_lsp_hub()`**, so a switch to a workspace declaring `pyright` (with
  `enable: true`) has it running immediately (D10); a seeded file whose
  `serverUrl` is still the instruction text reports "not configured" naming the
  path (D8); malformed JSON reports `failed` with the parse error, and a chat /
  benchmark request refuses on it (D9).
- **`tests/test_api.py`** — `POST /api/settings` writes group 1 to the
  workspace file and group 2 to `.env`, and neither write touches the other
  file; the `.env` template no longer renders a group-1 key (parity between
  `SETTING_KEYS`, `_render_env()` and `.env.example`); `REQUIRED_SETTING_KEYS`
  is empty; `GET` then reflects a second workspace's values after a switch, not
  the first one's.
- **`tests/test_api.py` — the `LLM_DEBUG` regression**: an existing
  `LLM_DEBUG=1` survives a settings save that never mentions it, and the payload
  has no `llmDebug` field. Replaces `test_save_normalizes_llm_debug_and_applies_it`.
- **`tests/test_config.py`** — `load_endpoints_config()` reads group 1 from the
  file and the verifier block from the env; a group-1 value present only in
  `os.environ` is *not* picked up (D2); an absent `temperature` omits the key
  from `to_kwargs()`; `supportsVision: true` is `True` (D6); `lspEnabled`
  deduplicates and preserves order.
- **`tests/test_lsp.py`** — `post_lsp` persists `lspEnabled` as a JSON array and
  still reconciles live servers in the same request; a hand-edit of `lspEnabled`
  does not change the running hub until a restart (D4); `maxToolIteration`
  reaches the tool loop from the file.
- **`tests/test_env_isolation.py` — the leak this change creates.** Group 1
  leaving `SETTING_KEYS` removes the protection of the six
  `for key in config_module.SETTING_KEYS` wipe loops
  (`test_api.py:308`, `test_lsp.py:2091` + `:2475`, `test_mcp.py:792`,
  `test_mcp_settings_seed.py:217`, `test_lsp_settings_seed.py:204`). Without a
  new guard, **a developer's real `.nojae/nojae-settings.json` — live
  `serverUrl` and `serverApiKey` — reaches every test**. The test asserts that
  no test can read group-1 values from a location it did not create: the fixture
  neutralises the workspace file the way it already neutralises `.env`, and
  `tests/conftest.py` (which today deletes `LSP_ENABLED` at `:409-417`) points
  the same assertion at the new file's own constant.
- **Existing suites that must still pass unchanged**: `test_prompt_helpers*`,
  `test_agent_modes*`, `test_mcp_settings_seed`, `test_lsp_settings_seed`,
  `test_editorfs*`, `test_write_surface_isolation` (the new writer is
  app-internal and reachable from no connector, tool or chat path),
  `test_llmkit_contract` (sampling still goes out verbatim).

Manual check, desktop browser: open workspace A, set a model in the dialog,
chat; switch to workspace B in the app bar; the dialog shows B's own file
(path in the header) and B's server; a second server starts and B's enabled LSP
is live without a reboot; edit A's file by hand, restart, the new value answers.
