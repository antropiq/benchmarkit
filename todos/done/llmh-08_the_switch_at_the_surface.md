# llmh-08 — The switch at the surface: `routes.py`, the dialog, `models_url`

**Spec:** Phase 4 (steps 19, 20's frontend share), §4.4 rows `src/api/routes.py`
and `web/src/components/SettingsDialog.vue`, D6, D9.
**Depends on:** llmh-07 (`LLM_DEBUG` is the env key; `apply_llm_debug` exists).
**Leaves the app working:** backend field, dialog binding and vitest fixtures
rename in the same task, so the settings screen never halves.

## Deliver this

**`src/api/routes.py`**:

* the settings model field `litellmDebug` → `llmDebug` (its pydantic-style
  model, line ~101) and `SETTINGS_ENV_KEYS["llmDebug"] = "LLM_DEBUG"` (llmh-07
  already moved the env-key side of that mapping); the
  `_settings_env_values` normalisation line (`normalize_env_flag` of the debug
  value) follows the new env key name;
* delete the local `_models_url` (function + docstring) and import
  `models_url` from `src.core.llmkit.client` instead — the D9 rule: one URL
  rule, imported, not duplicated (its two call sites keep working unchanged if
  you bind the import to the same local name or rename at the call sites; either
  is fine, the point is one definition);
* the docstrings that still say "sync litellm in a worker thread" /
  "litellm's one-line summary" now say "the sync LLM client" — grep `litellm`
  in this file and land on zero hits.

**`web/src/components/SettingsDialog.vue`**:

* `state.settings.litellmDebug` → `state.settings.llmDebug` (both the `:checked`
  and the `@change` binding);
* label: **"Log LLM request internals (debug)"**;
* caption reworded — it currently promises "LiteLLM's own DEBUG output". New
  text, same spirit and same length: what it does now is write each LLM
  request's payload and the client's debug output to the server console, so a
  failed request shows the real upstream error; still notes it floods the
  console and should be switched back off once diagnosed.

**`web/tests/`** vitest fixtures: `litellmDebug: ''` → `llmDebug: ''` in
`pathLocationPolicy.test.js` (two sites) and `settingsSyspromptHelp.test.js`
(one site), or wherever the settings fixture object is built. If a test asserts
the checkbox's label text, update the expectation to the new label.

**`web/src/composables/useChat.js`**: one stale comment mentioning litellm
(line ~434, about the classic message form) — reword to "the backend and the
LLM client keep the classic form". Comment-only; no logic.

## Files it touches

| file | change |
|---|---|
| `src/api/routes.py` | field rename, `SETTINGS_ENV_KEYS` key, `_models_url` → import from `llmkit`, docstrings |
| `web/src/components/SettingsDialog.vue` | binding + label + caption |
| `web/tests/pathLocationPolicy.test.js`, `web/tests/settingsSyspromptHelp.test.js` | fixture field rename |
| `web/src/composables/useChat.js` | one comment |

## Rules it must not break

* AGENTS.md, the `.env` template rule again: the settings save path must still
  emit **the full fixed key set** — this task renames the *wire field*, not a
  key (llmh-07 moved the key). A settings save with everything else empty must
  lose nothing.
* The dialog's checkbox must survive a save→load round-trip ticked
  (`"1"` normalisation lives in `_settings_env_values`; don't move it).
* The app's own SSE layer is out of scope — routes' streaming code is not
  touched beyond comments/docstrings.
* `grep -rn "litellmDebug" web/src src` after this task: empty.

## Acceptance

```
pytest tests/test_api.py tests/test_config.py -q
cd web && npm test
```
both green. Manual round-trip (§6.3.8): open Settings, tick the box, save,
confirm `.env` holds `LLM_DEBUG=1` and no managed key vanished; reload the
dialog and the box is still ticked.

## Note for the reader

If a `tests/test_api.py` assertion on the field name `litellmDebug` still
exists after llmh-07 (it was allowed to), it renames here — that is the
hand-off point, not a regression. `web/dist/**` and `web/node_modules/**` are
build artifacts; never edit or grep-trust them — the live sources are
`web/src` and `web/tests`.

## Done — what landed

* **`src/api/routes.py`** — `SettingsIn.llmDebug`, `SETTINGS_ENV_KEYS["llmDebug"]
  = "LLM_DEBUG"`, and the local `_models_url` (function + docstring) deleted in
  favour of `from src.core.llmkit.client import models_url` (D9: one URL rule).
  Both call sites now call the imported `models_url`; inside `get_models` the
  local that used to shadow the function name is just `url`, which also keeps
  the 502 detail sentence identical. The four remaining litellm sentences
  (`ChatMessage` docstring, `payload()`, `post_chat` docstring, the
  client-gone comment) now name the LLM client — `grep -rn "litellm" src/api`
  is empty. `_settings_env_values` was not touched: the `LLM_DEBUG`
  normalisation line already read the new env key after llmh-07.
* **`web/src/components/SettingsDialog.vue`** — the checkbox binds
  `state.settings.llmDebug`, label is "Log LLM request internals (debug)",
  caption reworded to what the switch does now (request payload + client debug
  output to the server console, real upstream error, floods the console).
* **`web/src/composables/useBenchmark.js` needed no change**: the dialog posts
  the settings object it got from `GET /api/settings`, so the renamed field
  travels by itself in both directions — nothing hand-lists the keys.
* **`web/tests/`** — `llmDebug: ''` in the three settings fixtures
  (`pathLocationPolicy.test.js` ×2, `settingsSyspromptHelp.test.js`). No vitest
  test asserted the checkbox's label text, so no expectation moved.
* **`web/src/composables/useChat.js`** — the one comment now reads "the backend
  and the LLM client keep the classic form". Comment-only.
* **`tests/test_api.py`** — the hand-off the task file predicted: the two
  `self._payload(litellmDebug=…)` posts and the response-field assertion
  renamed, the "llmh-08 renames it" note deleted, and the save test gained the
  round-trip read (`GET /api/settings` after the save answers `llmDebug == "1"`)
  so the ticked-box-survives-a-reload half of §6.3.8 is asserted rather than
  clicked.

## Verification

`pytest tests/test_api.py tests/test_config.py -q` → **94 passed**;
`cd web && npm test` → **494 passed (36 files)**. The neighbouring touched set
(`test_llmkit_contract test_llmkit_sse test_llmkit_client
test_write_surface_isolation test_llm test_chat test_bench test_agent_modes`) →
**393 passed**, and `test_write_surface_isolation` on its own → **24 passed**
(it imports `models_url` from `llmkit.client` and asserts the URL builders
cannot produce a path under `/api/`, which the new import must not disturb).
No full suite — that is llmh-10. Pyright on `routes.py`: no diagnostics.

The live half of §6.3.8 (tick the box, watch payload lines reach a real server
console) stays on llmh-10's human-only checklist; the round-trip through the
route is covered by the test above. No request left this machine.

