# Settings dialog (.env editor) — implementation plan

## Goal
Add a settings modal to the Vue frontend, backed by two new FastAPI endpoints, that reads and writes the project-root `.env` (target LLM + sampling + verifier config). A gear button in the app bar (between stop and theme toggle) opens it; it is disabled while a run is active.

## Resolved decisions
- **Save effect**: `POST /api/settings` writes `.env` AND updates `os.environ` for the managed keys, so the next run picks up changes without restart. (`load_endpoints_config()` already re-reads `os.environ` on every request.)
- **Empty optional fields**: omitted from `.env` and removed from `os.environ`. Required fields (`SERVER_URL`, `ENDPOINT`) always written.
- **REASONING select options**: `None` (empty → omit key), `low`, `medium`, `high`, `xhigh`. `None` sends `""`; an existing `off` value loads as `None` (per spec, `off` is not offered in the UI).
- **Button enablement**: disabled only while `state.status === 'running'`; backend also guards with 409 during a run (single-run app, other tabs).

## Backend

### 1. `src/config.py` — env read/write helpers
- Add `SETTINGS_KEYS` mapping of the managed env keys: `SERVER_URL`, `ENDPOINT`, `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`, `VERIFIER_AGENT_SYSPROMPT`, `VERIFIER_AGENT_SERVERURL`, `VERIFIER_AGENT_ENDPOINT`.
- `load_settings() -> Dict[str, str]`: read effective values from `os.environ` (it already reflects the startup `.env` and later saves); missing → `""`.
- `save_settings(values: Dict[str, str]) -> None`:
  - Regenerate `.env` from a fixed template mirroring the current `.env.example` structure (same comments/sections). Required keys always written; optional keys with empty values written as commented-out lines (`#TEMP=`). `VERIFIER_AGENT_SYSPROMPT` written with `dotenv`-safe quoting/escaping (use `python-dotenv`'s writer or `shlex`-style quoting consistent with the existing `.env`, which wraps the prompt in `"` with `\"` escapes). Simplest robust approach: write via `dotenv.set_key`/manual quoting using `dotenv.main.quote_value`.
  - Update `os.environ`: `os.environ[k] = v` for non-empty; `os.environ.pop(k, None)` for empty.
  - Write atomically (write temp + `replace`) reading `ENV_PATH` from module global so tests can monkeypatch it.

### 2. `src/api/routes.py` — endpoints
- Pydantic request model `SettingsIn` (in `routes.py` or `core/models.py`) with camelCase string fields: `serverUrl`, `endpoint`, `temp`, `topP`, `topK`, `minP`, `repeatPenalty`, `presencePenalty`, `reasoning`, and nested `verifier: {serverUrl, endpoint, sysprompt}`. All optional strings defaulting to `""` (raw strings — server is the validator via `EndpointsConfig`).
- `GET /api/settings` → `Dict[str, Any]` from `load_settings()` shaped into the same camelCase/nested JSON. 200 always (empty strings when unset).
- `POST /api/settings` (200):
  1. `if state.status == "running": raise HTTPException(409, "Cannot change settings while a benchmark run is in progress")`.
  2. Convert payload → env-key dict (`""` stays `""`).
  3. Validate: run `EndpointsConfig.from_env()` against a merged view (`{**os.environ, **new_values}` with empty optionals removed) — reuse the pattern of `_load_config()`; on `ValidationError` raise 422 with `json.loads(error.json())` (same shape as existing config errors).
  4. `save_settings(...)`; return the saved payload.
- Note in docstring: keys that existed as *real* process env vars before startup keep precedence in the live process until restart (file is still updated) — accepted, per chosen option.

### 3. `.env.example`
- Uncomment `VERIFIER_AGENT_SYSPROMPT` and set it to the exact default value currently found in `.env` (the "You are a response expert analyser…" prompt), same quoting style as `.env`.

## Frontend

### 4. `web/src/composables/useBenchmark.js`
- Extend singleton state: `settingsOpen: false`, `settings: null`, `settingsLoading: false`, `settingsSaving: false`, `settingsError: null`.
- `openSettings()`: `GET /api/settings` → `state.settings`, `settingsOpen = true`; errors → `settingsError`.
- `saveSettings()`: `POST /api/settings` with `JSON.stringify(state.settings)`; on 422/409 surface `detail` (stringify arrays) in `settingsError`; on success close dialog and call `loadConfig()` so app bar model name / sampling summary / `verifierConfigured` warning refresh.
- `closeSettings()`: close without saving (re-fetch on next open).

### 5. `web/src/components/SettingsDialog.vue` (new)
- Pico-native `<dialog>` (`:open` via `v-if="state.settingsOpen"` or `dialog.showModal()` on mount + `@close`). Click on backdrop closes (cancel). Escape works natively.
- Two `<fieldset>` sections:
  - **Benchmark LLM**: Server URL (`input type=url`), Model (text), Temperature / Top-P / Min-P / Repeat penalty / Presence penalty (`input type=number step=any`), Top-K (`type=number step=1`), Reasoning (`<select>`: `""`=None, low, medium, high, xhigh).
  - **Verifier**: Server URL, Model, System prompt (`<textarea rows=5>`).
- Footer buttons: **Cancel** (secondary, closes) and **Save** (primary, disabled + "Saving…" while `settingsSaving`). Show `settingsError` inline; server 422 per-field errors: display flattened messages (not required to map to inputs).
- Bind directly to `state.settings` fields; on open the composable fills it from the server.

### 6. `web/src/components/AppBar.vue`
- Add gear-icon button between `RunControls` and `ThemeToggle`: class `run-btn run-btn-settings` (reuse app-bar button styling), `aria-label="Settings"`, `:disabled="running"`, `:title="running ? 'Available when idle' : 'Settings'"`, `@click="openSettings"` (from `useBenchmark`, already imported for `state`).

### 7. `web/src/App.vue`
- Mount `<SettingsDialog />` after `<AppBar />`; no prop plumbing (dialog reads the composable singleton).

### 8. `web/src/styles/theme.css`
- Minimal dialog styles only if Pico defaults are insufficient: `dialog::backdrop` dim, section spacing, footer button row (Pico already styles `dialog`, `fieldset`, buttons). Reuse existing CSS variables.

## Task order
1. `.env.example` sysprompt default.
2. `src/config.py` helpers (`load_settings`, `save_settings`).
3. `src/api/routes.py` endpoints + `SettingsIn` model.
4. Backend tests (below), run `pytest`.
5. Composable, `SettingsDialog.vue`, AppBar button, App mount, styles.
6. `cd web && npm run build` to verify the SPA compiles.

## Validation
- `pytest`:
  - `GET /api/settings` returns env values (monkeypatch `os.environ` / `routes` module functions following the existing `client` fixture pattern; monkeypatch `config.ENV_PATH` to `tmp_path/.env` for save tests).
  - `POST /api/settings` writes `.env` (file exists, contains `SERVER_URL=…`, sysprompt round-trips through `dotenv_values`), updates `os.environ`, and removing an optional value pops it from `os.environ` and leaves it commented in the file.
  - `POST /api/settings` with `TEMP=abc` → 422; while `state.status == "running"` → 409.
  - Round-trip: save then `GET /api/config` reflects new model.
- Manual: open app, gear disabled while running; save new model → app bar sub-label updates; empty optional field → key commented out in `.env`.

## Edge cases / caveats
- `ENDPOINTS` (multi-model) is not edited by the dialog; the untouched `#ENDPOINTS=` comment line is preserved in the template. If only `ENDPOINTS` is set, Model field shows empty and saving Model would set `ENDPOINT` (which then takes precedence — matches existing precedence rule).
- Multi-tab race: backend 409 guard covers saves attempted during a run started elsewhere.
- `.env` written by save is regenerated from the template (custom user comments in `.env` are not preserved) — acceptable; document in route docstring.
- No secret concerns: values already live in `.env`; endpoints are same-origin, no auth in this app (consistent with existing API).
