# ws-03 — The reads cutover, and the hermeticity guard it forces

Make the app **read** group 1 from `.nojae/nojae-settings.json` and from nothing
else (D2), and close the leak that very move opens: with group 1 out of
`SETTING_KEYS`, the suite's six wipe loops stop protecting a test from the
developer's live `serverUrl` and `serverApiKey`.

This is the task where the behaviour changes. It is also the widest one — but its
width is reads and test fixtures, not new logic, and its file list is closed.

Depends on: ws-01 (the module), ws-02 (the file exists wherever the app boots).
Spec: **D2** (no `.env` fallback), **D6** (native types ⇒ a second builder), **D9**
(a malformed file is a named state that refuses), **§5** (the `EndpointsConfig`,
`src/config.py` and `New module` rows), **§10** (`test_config.py`, `test_env_isolation.py`,
and the six wipe-loop line numbers).

## Files

| file | change |
|---|---|
| `src/core/models.py` | **new** `EndpointsConfig.from_workspace(workspace, env)`; `from_env()` loses its group-1 reads and its doc comment's list of those keys changes with them |
| `src/config.py` | `load_endpoints_config()` builds from the two stores; `lsp_enabled_ids()` and `max_tool_iteration()` change **source only** (`os.environ` → `workspace_setting(...)`); `MAX_MCP_ITERATION` stops being honoured. `normalize_env_flag()` and `LLM_DEBUG` are untouched here — their callers are in `routes.py` and belong to ws-04 |
| `tests/conftest.py` | a `neutralise_workspace_settings(monkeypatch, tmp_path)` helper (or the autouse fixture of the same job): points the workspace file at a throw-away location so **no test can read a group-1 value from a location it did not create** — the twin of what the file already does for `.env` and for `LSP_ENABLED` at `:403-418` |
| `tests/test_env_isolation.py` | **new** (§10's "the leak this change creates") |
| `src/api/routes.py` | **`post_lsp()` only**: `lspEnabled` + `maxToolIteration` are persisted through `save_workspace_settings()` (JSON array, not the comma string) instead of `merged["LSP_ENABLED"] = ",".join(...)`, and its docstring's sentence naming `LSP_ENABLED` in `.env` changes with it. The live `start_server`/`stop_server` reconcile and the `set_lsp_enabled()` write-back to `lsp-settings.json` are **unchanged**. This write belongs with the read: leaving it in `.env` would make the LSP toggles inert the moment the source moves. |
| `tests/test_lsp.py` | the two fixtures with the `SETTING_KEYS` wipe loop (`:2091`, `:2475`) gain the workspace-file neutralisation, and the `post_lsp` assertions that read `LSP_ENABLED` out of `.env` are re-homed onto the file (§10: "`post_lsp` persists `lspEnabled` as a JSON array and still reconciles live servers in the same request"). Read only those regions — the file is ~2.9k lines. |
| `tests/test_config.py` | §10's `test_config.py` list, replacing the cases that asserted group-1 keys read from the env |
| test files whose fixtures set group-1 keys through `os.environ` / the `.env` fixture | switch those to the shared helper (or to writing the workspace file). `grep -n "SERVER_URL\|ENDPOINT\|LLM_SYSPROMPT\|TEMPERATURE" tests/*.py` is the list; do not edit their **assertions** |

**Not this task:** `SETTING_KEYS`, `REQUIRED_SETTING_KEYS`, `_render_env()`,
`.env.example`, `GET`/`POST /api/settings`, `LLM_DEBUG`, the frontend. Those are ws-04
— the dialog's writes and the `.env` trio move together, which is the trio rule.

## 1. The second builder (D6)

```python
@classmethod
def from_workspace(
    cls, workspace: Mapping[str, Any], env: Mapping[str, str]
) -> "EndpointsConfig": ...
```

Group 1 from the **JSON mapping** — native `True` / `0.2` / `["pyright"]` pass
straight to pydantic, which already validates `reasoning` and coerces numbers; the
`VERIFIER_AGENT_*` block from the **env mapping**. `from_env()` keeps group 2/3 and
loses the fourteen group-1 reads; its doc comment currently *lists* those keys, so it
is edited in the same breath. Do not make `from_env()` call `from_workspace()` with a
stringly-typed shim — that re-introduces the wart D6 rejects.

Apply the §2 "member absent means" column exactly: an absent `temperature` stays
`None` so `SamplingParams.to_kwargs()` omits it (`exclude_none`), absent
`supportsVision` is `False`, absent `sysprompt` is `DEFAULT_LLM_SYSPROMPT`'s value
(read the real name in the code), absent `maxToolIteration` is
`DEFAULT_MAX_TOOL_ITERATION`.

## 2. The config functions keep their names

`lsp_enabled_ids()` and `max_tool_iteration()` stay in `src/config.py` and keep their
signatures, so `chat.py`, `server.py` and `routes.py` import what they already import
(spec §5, and the big picture's contract). Only their source changes:
`os.environ.get("LSP_ENABLED")` → `workspace_setting("lspEnabled")`, same
split/dedup/order semantics; `os.environ.get("MAX_TOOL_ITERATION")` →
`workspace_setting("maxToolIteration")` with `max(1, int(...))` and the same
`DEFAULT_MAX_TOOL_ITERATION` fallback. `MAX_MCP_ITERATION` **stops being honoured**
(§5: it was a `.env`-hand-write escape hatch for a file that no longer holds the key)
— delete the fallback and its `or os.environ.get("MAX_MCP_ITERATION", "")` line.
`LEGACY_LLM_DEBUG_KEY` **stays** (it is the `LLM_DEBUG` fallback, group 3, untouched
by this lot) — but the sentence in its comment block (`config.py:116-120`) cites
`MAX_MCP_ITERATION` as the precedent it follows, so that **cross-reference is
reworded** in the same breath (keep the sentence's point, drop the dead name).
`tests/conftest.py`'s `_no_legacy_tool_cap` fixture (`:403-418`) also names
`MAX_MCP_ITERATION` and `LSP_ENABLED` in its docstring and its `delenv` calls:
`MAX_MCP_ITERATION`'s `delenv` goes with the fallback, the `LSP_ENABLED` one stays or
goes with a sentence saying the key is no longer read — do not leave a fixture
clearing a name nothing reads without saying why.

A **local import** of `workspace_settings` inside these two functions is the right
shape: `src/config.py` sits at the bottom of the dependency graph and
`workspace_settings` imports `path_location` *from* it — `prompt_helpers.py`'s and
`policy_seed`'s callers show the same cycle-avoidance reasoning; write the one-line
comment that says so.

## 3. The refusal (D9)

`load_endpoints_config()` is where the two stores meet, and it already returns
`(config, validation_error)`. Route the file's state through the **same** channel:
a `failed` state, or an `absent` file, or the D8 "still the seed text" condition,
becomes a `None` config plus a sentence naming the file and its path — so the two
`_load_config()` call sites in `routes.py` (`:838`) and the chat/benchmark paths that
use them refuse with that sentence rather than dialing a sentence as a URL or guessing
defaults. Build that sentence in **one** place (this module or `config.load_endpoints_config`)
and reuse it in ws-04's `GET /api/settings`; note the pointer in a comment so ws-04
does not write a second one.

Do not touch `apply_llm_debug()`, `load_settings()` or `save_settings()`'s `.env` job.

## 4. The guard test (this is why the task is not "just a refactor")

Group 1 leaving `SETTING_KEYS` removes the protection of the six
`for key in config_module.SETTING_KEYS` wipe loops (`test_api.py:308`, `test_lsp.py:2091`
and `:2475`, `test_mcp.py:792`, `test_mcp_settings_seed.py:217`,
`test_lsp_settings_seed.py:204`). Without a replacement, a developer's real
`.nojae/nojae-settings.json` — live server URL and real API key — reaches every test.

`tests/test_env_isolation.py` asserts the invariant instead of trusting the fixture:

1. **the guard is pointed at the new file's own constant**, not a re-derived path:
   with the location neutralised, `workspace_settings.load_workspace_settings()`
   answers empty / the neutralised root, even though `WORKSPACE_SETTINGS_TEMPLATE`'s
   install directory (this checkout) has a real `.nojae/nojae-settings.json`. Read
   `tests/conftest.py:403-418` first and mirror how *it* names the key it clears.
2. **a test cannot read a group-1 value from a location it did not create**: with the
   fixture in place, `load_endpoints_config()` reports not-configured and its
   `server_url` is empty; after the test writes its own file at its own `tmp_path`,
   the values answer.
3. **`config.lsp_enabled_ids()` / `max_tool_iteration()` are neutral too**: with no
   file of the test's own, they answer `[]` and the code default — never whatever this
   machine has enabled.
4. **the old wipe loops still hold for group 2/3**: `SETTING_KEYS` (unchanged in this
   task) is still wiped by the existing loops — a sentence in the docstring pointing
   at ws-04, which shrinks that tuple and must keep this test honest.

## Tests (`tests/test_config.py`, per §10)

- `load_endpoints_config()` reads group 1 from the file and the verifier block from
  the env (one test, both halves asserted).
- a group-1 value present **only** in `os.environ` is **not** picked up (D2) — set
  `SERVER_URL` in the env, no file, and assert the config has no server url.
- an absent `temperature` omits the key from `to_kwargs()`.
- `"supportsVision": true` is `True` (D6) — and, in the same test, the string
  `"true"` is *not* accepted as a boolean by the reader (that is a `failed` state,
  ws-01's test; here assert the native path only).
- `lspEnabled` deduplicates and preserves order through `config.lsp_enabled_ids()`.
- `max_tool_iteration()`: the file's value wins; absent → `DEFAULT_MAX_TOOL_ITERATION`;
  `0` or negative → `1`; **`MAX_MCP_ITERATION` in the env is ignored** (the fallback is
  gone).
- the `EndpointsConfig.from_workspace` unit cases belong here (or in
  `tests/test_models.py` if that file already owns `from_env`'s cases — read it; keep
  the pair together, do not split the two builders' tests across files).

## Acceptance

- `pytest -q` **fully green** — this is the task where the suite must be moved onto the
  new source, so a red test is a fixture left behind, not an accepted interim.
- `cd web && npm test` unchanged green; `cd web && npm run build` succeeds.
- `pytest tests/test_llmkit_contract.py tests/test_write_surface_isolation.py
  tests/test_prompt_helpers.py tests/test_agent_modes.py -q` green **and unedited**
  (§10's "must still pass unchanged" list — `git diff --stat` shows you touched none
  of them).
- `grep -n "os.environ" src/config.py` shows no group-1 **read** left in
  `load_endpoints_config()` / `lsp_enabled_ids()` / `max_tool_iteration()`;
  `grep -n "MAX_MCP_ITERATION" src` is empty. (`SETTING_KEYS` and `_render_env()` still
  *name* the group-1 keys — that is ws-04's half of the trio, and the interim is what
  keeps a save from losing them.)
- The settings routes and the frontend are untouched (`git diff` proves it): the dialog
  still round-trips group 1 through `.env`, so it looks inert for a live model call.
  That is the accepted interim; **ws-04 lands immediately after this.**
- Move this file to `todos/done/` when the above is true.
