# llmh-07 — The debug switch becomes `LLM_DEBUG`; the cost table is deleted

**Spec:** Phase 4 (steps 17, 18, 20, 21, 22), D6, D7, D8, §4.4 rows `src/config.py`
and `save_settings`.
**Depends on:** llmh-03 (the `benchmarkit.llm` logger name and the client's
debug hook), llmh-05, llmh-06 (both paths must be on `llmkit` before the switch
stops meaning "litellm's loggers").
**Leaves the app working:** the settings dialog's checkbox still round-trips (on
the **old** field name until llmh-08 — read the compatibility note).

## Deliver this

**`src/config.py`**, four coordinated changes:

1. **Rename the flag pair (D6)**: `LITELLM_DEBUG_KEY` → `LLM_DEBUG_KEY =
   "LLM_DEBUG"`; `litellm_debug_enabled()` → `llm_debug_enabled()`;
   `apply_litellm_debug()` → `apply_llm_debug()`. `SETTING_KEYS` entry
   `LITELLM_DEBUG` → `LLM_DEBUG` (keep its position in the tuple).
   **Fallback:** `llm_debug_enabled()` reads `LLM_DEBUG` and falls back to the
   legacy `LITELLM_DEBUG` when the new key is absent — the precedent the project
   already follows with `MAX_TOOL_ITERATION` / `MAX_MCP_ITERATION`. The new key
   wins when both are set. Keep the string `"LITELLM_DEBUG"` in exactly one
   clearly-commented place: the fallback. Check the OS-reserved-names block
   above `SETTING_KEYS`: the rule is that a managed key must be free of OS
   variables on every supported platform — `LLM_DEBUG` is, but confirm rather
   than assume, and say so in the commit message.
2. **`apply_llm_debug()` implemented against our own logger**: raise
   `logging.getLogger(constants.DEBUG_LOGGER_NAME)` (i.e. `benchmarkit.llm`) to
   `DEBUG` when enabled, reset it to `NOTSET`/`WARNING` (whatever today's off
   path effectively leaves) when not. No `import litellm` anywhere in the
   function; the whole "private litellm API / `_turn_on_debug` /
   `suppress_debug_info`" apparatus and `LITELLM_LOGGER_NAMES` go. Keep the
   return-bool contract (callers and tests use it) and keep the "applied at
   server start and on every settings save" behaviour.
   Then wire the **payload half of D8**: point `llmkit.client`'s debug hook
   (llmh-03 left it as a module-level callable defaulting to off) at
   `llm_debug_enabled` — from the API/composition side or from `config`,
   whichever keeps the dependency arrow pointing the right way
   (`core` must not import `api`; `config` is below both, so wiring it in
   `apply_llm_debug` is acceptable; wiring it in `api/server.py` is too).
3. **Delete the cost-registration block (D7)**: `ZERO_COST_MODEL_INFO`,
   `LITELLM_MODEL_PREFIX`, `register_unmapped_models`,
   `register_configured_models` and every mention — with **no replacement**.
4. **`_render_env`**: the key becomes `LLM_DEBUG` and its five-line comment
   block is rewritten (it currently explains litellm's three loggers and the
   `_turn_on_debug` hint). The new comment says what the flag does now: logs
   each request's payload to the server console at DEBUG, so a failed call
   shows the real upstream status and body; note that the *status and body*
   reach the UI's error line with the flag **off** (D8) and that what the flag
   adds is the payload. `save_settings` keeps calling the (renamed) apply on
   save, and its `register_configured_models()` line and comment go.

**`src/api/server.py`** (step 18): one boot call instead of two — the
`register_configured_models` import and call and their explanatory comment go;
the rename applies to the remaining call and to the comment above it ("read
from the environment, so it is applied at boot"; the "tests build an app without
touching the loggers of the whole process" reason stays true and stays written).

**Env files (step 20)**: `#LITELLM_DEBUG=1` → `#LLM_DEBUG=1` **with its
comment**, in `.env.example`, `.env.test` and `.env.test.example`. `.env` itself
is user data — do not touch it; note in the commit message that a hand-written
`LITELLM_DEBUG=1` keeps working through the fallback until the dialog is saved.

**Tests (steps 21, 22)**:

* `tests/test_config.py`: `TestLiteLLMDebugFlag` → the renamed pair; **new**
  cases: the legacy key alone still enables it, the new key wins when both are
  set, and neither set means off. `apply_llm_debug`'s on/off assertions move from
  litellm's loggers to `benchmarkit.llm`'s level.
  Delete `TestUnmappedModelCostRegistration` entirely (it is the only reader of
  `completion_cost` — spec D7).
* `tests/test_api.py`: `test_save_normalizes_litellm_debug_and_applies_it` →
  `…_llm_debug_…`, posting the **new** field name and asserting the new env key.
  **Compatibility note:** the settings model's field is still `litellmDebug`
  until llmh-08. So in this task, `save_settings` / `_render_env` / the env-key
  mapping use `LLM_DEBUG`, and the API's `SETTINGS_ENV_KEYS` mapping entry is
  updated in the same commit (`"litellmDebug": "LLM_DEBUG"` — the *field* name
  renames in llmh-08). Do not change `SettingsDialog.vue` here; a test asserting
  the wire field name `litellmDebug` stays valid.
* **new** (step 22): in `tests/test_llmkit_client.py`, an HTTP failure's `str()`
  names the status and the upstream message with the debug flag **off** — the
  message is what the UI's error line shows. (llmh-03 asserts the same fact; if
  it already covers it, add the explicit "debug off" framing rather than a
  second test.)

## Files it touches

`src/config.py`, `src/api/server.py`, `.env.example`, `.env.test`,
`.env.test.example`, `tests/test_config.py`, `tests/test_api.py`, and the
debug-hook wiring line (either `src/config.py` or `src/api/server.py`).

## Rules it must not break (quoted)

* AGENTS.md: *".env is regenerated from a fixed template on every save… A new or
  renamed key means changing the template, the renderer and the committed example
  together."* — `SETTING_KEYS`, `_render_env`, `.env.example`,
  `.env.test.example`, `.env.test` all move in this one task, or the next save
  silently drops the key.
* *"A test run takes its endpoint from the test env file, never from a
  developer's."* — `.env.test` changes **only** its debug line. Its
  `SERVER_URL`/verifier hosts stay exactly as they are (do not "fix" them to
  localhost).
* The debug output must never contain an API key.

## Acceptance

```
pytest tests/test_config.py tests/test_api.py tests/test_llmkit_client.py -q
```
green. Then the settings round-trip check by hand (spec §6.3.8, cheap): save the
dialog and confirm `.env` comes back with `#LLM_DEBUG=` commented out and no
other managed key lost.

## Done — what landed

**`src/config.py`.** `LLM_DEBUG_KEY = "LLM_DEBUG"` (same position in
`SETTING_KEYS`), `LEGACY_LLM_DEBUG_KEY = "LITELLM_DEBUG"` as the single place the
old name is written, `llm_debug_enabled()` and `apply_llm_debug()`. The predicate
picks *which* name answers (the new key while it holds something) and then applies
the existing `env_flag` truthiness to it, so an explicit `LLM_DEBUG=0` means off
even with the legacy key on. `apply_llm_debug` raises `benchmarkit.llm` to DEBUG,
puts the off path back to `NOTSET` (what the off path always effectively left:
WARNING inherited from the root), and re-points `llmkit.client.debug_enabled` at
itself — wired here, in the layer below both, with a lazy import so a config
import still costs no transport.

**One thing the task's wording did not anticipate: a level alone logs nothing.**
uvicorn's `LOGGING_CONFIG` attaches handlers to `uvicorn.*` only and leaves the
root logger bare, so a raised `benchmarkit.llm` has nowhere to write and the
switch would silently print nothing. litellm never had that problem — it hangs
its own handler on its loggers at import time, which is why `_turn_on_debug()`
only had to set levels — so removing its apparatus removed the handler with it.
The switch therefore owns a handler of its own: `_DebugStreamHandler`, attached
with `propagate` off (so a deployment that *does* configure a root logger is not
printed twice) and removed by the off path, which restores `propagate` with it.
It is identified **by class**, not remembered on the logger, so "exactly one of
ours" holds by construction: an attach clears every handler of that class it
finds — including one an earlier boot attached — instead of stacking one per
settings save, and a test can count its own without tripping over pytest's
capture handlers. A fresh handler per call is also what binds the console as it
is now (a replaced `sys.stderr` included). Verified end to end: with
`LLM_DEBUG=1`, two `apply_llm_debug()` calls leave one handler, `_log_request`
emits the POST line and the JSON payload, and neither the `Authorization` header
nor the key appears.

**`save_settings` also clears the legacy key** when the new one comes back empty.
The regenerated file no longer holds it, so an `os.environ` still carrying the
value the boot loaded would have kept the switch running through a save that
unticked it — the fallback now lasts exactly as long as the hand-written `.env` it
exists for. Pinned by `test_save_unticked_debug_clears_the_legacy_key_too`.

**Cost table (D7)** — `ZERO_COST_MODEL_INFO`, `LITELLM_MODEL_PREFIX`,
`register_unmapped_models`, `register_configured_models`, `LITELLM_LOGGER_NAMES`
and both call sites (`server.py` boot, `save_settings`) are gone with no
replacement; `Iterable` left the import list with them.

**`tests/conftest.py` gained `_no_debug_switch_leak`** (autouse, beside
`_no_legacy_tool_cap`). Two leaks made the switch machine-dependent: `create_app()`
loads the developer's `.env`, and `save_settings` writes the key into `os.environ`,
which `monkeypatch` never undoes — after the settings tests ran, `test_llmkit_client`'s
"the hook defaults to off" assertion was reading the previous test's environment.
The fixture clears both keys and hands the logger (level, handlers, propagate) and
the client hook to the next test as it found them, dropping the switch's own
handler from that snapshot — it is the switch's state to rebuild, and restoring
it alone would leave a logger bound to a gone stream with `propagate` off, i.e.
one that silently writes nowhere.

**`SETTINGS_ENV_KEYS`** now maps `"litellmDebug"` → `LLM_DEBUG` (field name kept
for llmh-08, as instructed); `_settings_env_values` normalises `LLM_DEBUG`.

### Notes for the commit message

* `LLM_DEBUG` was checked against `_RESERVED_ENV_NAMES` rather than assumed: it is
  not an OS variable on any supported platform (no `LLM_*` name is), so
  `load_dotenv_once`'s "existing environment wins" precedence cannot make it read
  something the app never wrote.
* `.env` is untouched user data: a hand-written `LITELLM_DEBUG=1` keeps working
  through the fallback until the dialog is saved, at which point the save drops
  both the key and its environment copy.
* `.env.test` changed **only** its debug line; its `SERVER_URL`/verifier hosts are
  exactly as they were, and no request left this machine.

### Verification

`pytest tests/test_config.py tests/test_api.py tests/test_llmkit_client.py -q` →
**134 passed**. The wider touched set (`test_llmkit_contract test_llmkit_sse
test_write_surface_isolation test_llm test_chat test_bench test_agent_modes
test_models test_batch`) → **558 passed**; the other app-booting modules
(`test_mcp test_lsp test_console_gbash test_prompt_helpers_api test_models
test_utils test_events test_batch test_agent_modes test_agent_modes_seed`) →
**493 passed, 9 skipped**, because this task touched `tests/conftest.py`. No
full suite (that is llmh-10).

The §6.3.8 round-trip was run through `save_settings` on a **temporary copy** of
the real `.env` rather than by clicking the dialog, so the user's file stayed
untouched: the key comes back as `#LLM_DEBUG=`, no managed key is missing, no
key that had a value became commented out, and the legacy name is gone from the
rendered file. `test_save_normalizes_llm_debug_and_applies_it` covers the same
round-trip through the route. The live part of §6.3.8 — ticking the box and
watching payload lines reach a real server console — is llmh-10's human-only
checklist item; the plumbing was proven by hand here instead: with
`LLM_DEBUG=1` set, `apply_llm_debug()` returned `True` and raised the logger,
and `_log_request` wrote the POST line plus the JSON body to stderr with the key
absent; the off path left level 0, no handlers and `propagate: True`.
