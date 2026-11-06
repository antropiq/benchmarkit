# wss — the model connection becomes workspace-scoped

Today one `.env` holds the model connection, so both tabs always talk to whatever
the last save left there and a workspace cannot travel with its own model the way
`.nojae/mcp-settings.json` already travels with its connectors. This lot moves the
14 group-1 keys out of `.env` into **`<working location>/.nojae/nojae-settings.json`**,
seeded from a committed template, written by the Settings dialog and read at boot.

The authority is **`docs/nojae-settings-workspace-spec.md`** (427 lines: the three
groups and where each ends up, the file and its key map, decisions D1 … D11 with
their rejected alternatives, §5's shape of every new piece, §7's `LLM_DEBUG` trap,
§9's no-migration decision, §10's test list). Read it before your task — every
task file below cites the decision numbers it implements. This file is the map and
the cross-task contract; it does not restate that doc.

## What changes for a user (so no task quietly changes more)

1. **One file per workspace** holds server, key, model id, prompt, sampling,
   vision, `lspEnabled` and `maxToolIteration`. The dialog writes it; a hand-edit
   needs a reboot (D4 — exactly `.env`'s existing contract).
2. **`.env` shrinks to the install's own scope**: the four `VERIFIER_AGENT_*`
   keys, `QUESTION_SET`, `LLM_DEBUG`, `BENCHMARKIT_PYTHON_ROOT`.
3. **A group-1 key left in `.env` is read by nothing** (D2). Not a fallback, not a
   warning — dead bytes. Nothing migrates (D11).
4. **A freshly seeded workspace is valid JSON and unusable** (D8): the template
   holds instruction sentences, which the app reports as "not configured" naming
   the path, never as a transport error quoting a sentence.
5. **A malformed file is a named state** (D9): reported in `GET /api/settings`,
   and a chat or benchmark request refuses with that same sentence. It never falls
   back to defaults.
6. **`LLM_DEBUG` leaves the dialog** (§7) and stays a hand-edited managed key.

## Decisions already taken (do not re-open them)

All eleven are in the spec with their rejected alternatives. The ones most likely
to be undone by accident:

- **D2 / D11**: no `.env` fallback and no migration branch. The seed path has
  exactly one input (`nojae-settings.default.json`) and one behaviour. A "seed
  from the leftover `.env` values when the file is absent" branch is rejected.
- **D4**: read at boot and on location switch, cache invalidated on a dialog save.
  Do **not** re-read per request, and do not add a lost-update guard.
- **D6**: native JSON types, therefore a **second builder**
  (`EndpointsConfig.from_workspace`). Do not keep one builder by writing the file
  as strings.
- **D7**: the file's member names are the approved ones (`serverApiKey`,
  `temperature`); the dialog's payload fields stay `apiKey` / `temp`. The bridge is
  one mapping table, not a rewrite of the dialog.
- **D3 / D5**: no path-policy entry, no tree entry, no Editor tab. `.nojae` stays
  invisible to the Editor and the connectors; an app-internal writer was always
  allowed there (the `prompt_helpers.py` precedent).
- **D9**: the reader checks the member set itself — `EndpointsConfig`'s
  `extra="ignore"` would swallow a typo silently.
- **D10**: the seed lands **before** `_restart_lsp_hub()`, because `lspEnabled` is
  read during that restart.

## Shared contract between tasks

Names are the lot's contract: they are chosen once in ws-01 and every later task
uses them verbatim, so no task has to re-derive them.

| name | lands in | used by |
|---|---|---|
| `src/core/workspace_settings.py` — `WORKSPACE_SETTINGS_FILENAME`, `WORKSPACE_SETTINGS_TEMPLATE`, `POLICY_DIRNAME` (imported), `WORKSPACE_SETTINGS_KEYS` (member → JSON type), `WorkspaceSettingsFileError`, `workspace_settings_file_path(root)`, `workspace_settings_path()`, `ensure_workspace_settings_file(root) -> dict`, `load_workspace_settings() -> dict`, `workspace_settings_state() -> dict`, `save_workspace_settings(values: Mapping[str, Any]) -> None`, `invalidate_workspace_settings()`, `workspace_setting(name) -> Any`, module-level `_self_check()` | ws-01 | ws-02, ws-03, ws-04, ws-06 |
| `nojae-settings.default.json` at the **install root** (committed; sibling of `.env.example`) — the §2 JSON verbatim | ws-01 | ws-02, ws-04, ws-06, ws-07 |
| the state vocabulary — `{state, path, error}` with `state` in `ok`/`absent`/`failed` (read), `created`/`existing`/`failed` (seed), plus the D8 `configured` flag | ws-01 | ws-02, ws-04, ws-05, ws-06 |
| `EndpointsConfig.from_workspace(workspace: Mapping[str, Any], env: Mapping[str, str])` in `src/core/models.py`; `from_env()` keeps group 2/3 only | ws-03 | ws-04 |
| the D9 refusal sentence, one builder shared by `GET /api/settings`, chat and benchmark | ws-03 | ws-04, ws-05, ws-06 |
| `WORKSPACE_SETTING_KEYS: Dict[str, str]` (dialog field → JSON member) beside `SETTINGS_ENV_KEYS` / `VERIFIER_ENV_KEYS` in `src/api/routes.py` | ws-04 | ws-05, ws-06 |
| `save_workspace_settings()` takes **JSON member names** as its keys; the dialog-field → member translation lives only in `routes.py` (D7) | ws-01 (writer), ws-04 (caller) | ws-06 |
| cache invalidation is called by the save path **and** by `post_path_location()` — nowhere else (D4) | ws-01 (hook); callers: ws-02 (the switch), ws-04 (the save) | ws-06 |
| `workspace_settings` test fixture / helper `neutralise_workspace_settings(monkeypatch, tmp_path)` in `tests/conftest.py` — the `.env` neutralisation's twin | ws-03 | ws-04, ws-06 |

`config.lsp_enabled_ids()` and `config.max_tool_iteration()` **stay in
`src/config.py`** and change only their source (D-spec §5): keeping them put means
`chat.py`, `server.py` and `routes.py` import what they already import.

Tests for the new module patch `BENCHMARKIT_PATH_LOCATION` per test
(`monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))`, the way
`tests/test_agent_modes.py` does it) and move the lookup through the module's **own
attributes** (`workspace_settings.WORKSPACE_SETTINGS_FILENAME`,
`workspace_settings.WORKSPACE_SETTINGS_TEMPLATE`) — never by writing a literal
`.nojae/nojae-settings.json` and never by touching this checkout.

## Task order

| # | task | spec ref | depends on |
|---|---|---|---|
| ws-01 | `src/core/workspace_settings.py` + `nojae-settings.default.json` + unit tests. **No callers.** | §2, D6, D8, D9 | — |
| ws-02 | Seeding at the two moments: `_seed_workspace_settings()` at boot + the switch seed **before the hub restarts** (D10) | D8, D10, D11 | ws-01 |
| ws-03 | **Reads cutover** + hermeticity: `EndpointsConfig.from_workspace`, `load_endpoints_config`, `lsp_enabled_ids`/`max_tool_iteration` source, `MAX_MCP_ITERATION` dropped, `post_lsp`'s persistence follows the source, `conftest.py` neutralises the workspace file, new `tests/test_env_isolation.py` | D2, D6, D9, §10 | ws-01, ws-02 |
| ws-04 | **Writes cutover**: `GET`/`POST /api/settings` split, `WORKSPACE_SETTING_KEYS`, `LLM_DEBUG` out of the dialog API (§7), and the `.env` trio shrinks (`SETTING_KEYS`, `REQUIRED_SETTING_KEYS`, `_render_env()`, `.env.example`) | D7, §5, §7, §9 | ws-03 |
| ws-05 | Frontend: `SettingsDialog.vue` (the two sentences, the `llmDebug` field, the failed note), `useBenchmark.js` | D7, D8, D9 | ws-04 |
| ws-06 | Verification pass: the full suite, the §10 "unchanged" sweep, static scans, the D10 end-to-end + D9 refusal cases, one e2e, the human smoke checklist | §10 | ws-01 … ws-05 |
| ws-07 | Docs last: README (§2 table, the hot sentence, the LSP section, the `.env` section), AGENTS.md, the spec's status line | §6 | ws-01 … ws-06 |

**ws-03 and ws-04 are deliberately split at the read/write seam**, so each stays one
session. Between them the app **boots, chats and benchmarks against
`.nojae/nojae-settings.json`** (the reads are cut over), while a settings save still
writes group 1 into `.env` — so the dialog appears inert for those fields and a save
does not lose anything (the `.env` trio is still whole). That interim is accepted and
must be short: **ws-04 lands immediately after ws-03**, and nothing else is scheduled
between them. Do not reorder — ws-04 shrinks the `.env` template, which is only safe
once the reads no longer need those keys, and ws-03's guard test is only meaningful
once the reads come from the file.

**ws-07 is last**: it asserts sentences that only exist once the rest has landed,
and it edits `AGENTS.md`, which is injected into every WorkBench turn — wrong there
is worse than missing.

## Out of scope for this lot (do not add it)

- Making the **verifier** workspace-scoped, or groups 2/3 anything other than
  `.env`'s (spec §8).
- Renaming the API payload fields (`temp`, `apiKey`) to match the file (D7).
- Making a hand-edit hot, or any lost-update guard for concurrent writers (D4).
- A path-policy entry, tree entry or Editor tab for the new file (D3, D5).
- Per-workspace `QUESTION_SET`, themes, chat layout or agent modes.
- Any change to how `.nojae/mcp-settings.json` or `lsp-settings.json` behave beyond
  the `lspEnabled` source change and the README's two-gate sentence.
- **Deleting the group-1 lines from anyone's `.env`, by code or by hand**, and any
  migration, upgrade-detection or "seed from leftover values" branch (D11).
- **Performing the manual check** (spec §10, last paragraph: two workspaces, a switch,
  a live LSP, a hand-edit + restart). ws-06 *writes it down* as a human-only checklist;
  no task runs it. Do not write or perform a step that boots the app against this
  checkout's live `.env`/`.nojae` state, and do not make it a test.
- **Deciding the fate of `normalize_env_flag()`** is ws-04's, not ws-03's: it lives in
  `config.py` but its only callers are the two `routes.py` lines this lot removes.
  ws-03 leaves the function untouched; ws-04 `grep`s for the last caller and removes
  the function only when it has none.

## Project rules that bind these tasks (from `AGENTS.md`)

- **Python style**: explicit type annotations, `typing` imports, no bare containers.
- **Tests stay hermetic**: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1`; a backend test's
  workspace is a throw-away `tmp_path` pointed at by `BENCHMARKIT_PATH_LOCATION`,
  never this checkout; a test patches a hub's declared file location through that
  module's own constants rather than a path it computed.
- **The new module is under `src/core`**, so `tests/test_write_surface_isolation.py`
  applies to it: keep the markers `PUT /api/fs`, `POST /api/fs`, `DELETE /api/fs`
  out of it, and reach no write route from it. It is app-internal and reachable
  from no connector, tool or chat path.
- **Nothing under `core` may import the API layer**; `src/config.py` sits at the
  bottom of the dependency graph, so a module that reads and seeds `.nojae` lives
  beside `prompt_helpers.py`, not in `config.py`.
- **`.nojae` is seeded by exclusive create and never clobbered**; a seeding failure
  is a reported note, never a boot or switch failure.
- **`BENCHMARKIT_PATH_LOCATION` is deliberately not a managed `.env` key**; do not
  add it to `SETTING_KEYS`.
- **The `.env` trio moves together** (`SETTING_KEYS`, `_render_env()`,
  `.env.example`) — AGENTS.md's rule. Nothing here **adds** an environment key, so
  no new OS-name collision is possible (`_RESERVED_ENV_NAMES`).
- **One LLM layer**: sampling reaches `core/llmkit` verbatim, so no task here
  whitelists or rewrites sampler names.
- No lint step. `pytest` for Python, `cd web && npm test` for Vitest,
  `cd web && npm run build` for `web/dist`.

## Context budget (why the split is where it is)

The largest files a task must read are `src/api/routes.py` (~2.3k lines, but only
the ~250-line settings/LSP/path-location regions), `tests/test_lsp.py` (~2.9k lines,
only the `post_lsp` region), `web/src/components/SettingsDialog.vue` (~480 lines)
and `tests/test_api.py` (~1.3k lines, the settings class). **No task reads the whole
repo**; each file below lists exactly which files and which regions to read. If your
task turns out to need more than that, stop and say so rather than broadening.

