# mbl-03 — Backend cutover: the routes serve the workspace file, and `config.py` loses the block

Move the app onto the new store: rename the two routes and their request model,
switch the payload key to `slots`, map the new error to 422, add the third
reserved-folder block to both working-location endpoints, seed on the switch —
and **delete** everything the feature leaves behind in `src/config.py`. This is
the task that changes the app's behaviour; it is also the one after which the
SPA's bullet row shows nine empty slots until mbl-04 lands. That interim state is
accepted (big picture, "Task order").

Depends on: mbl-01 (the module), mbl-02 (the seeding half, `prompt_helpers_file_state`,
the boot seeder).
Spec: **D3** (config keeps nothing; the question-set skip is deleted), **D4**
(count errors become 422), **D5** (GET reports `absent` without writing), **D8**
(route names, payload key), §4's `routes.py` and `server.py` rows, §7's rows
*a reader never writes*, *bad version / bad JSON*, *the count is an error*, *bad
colour coerced*, *per-workspace*, *switch payload*.

## Files

| file | change |
|---|---|
| `src/api/routes.py` | imports: drop the five `MEMORY_*` + `load_memory_registrations` + `save_memory_registration` from `src.config` (`:22-25`, `:31`, `:39`), import `src.core.prompt_helpers`. `MemorySlotIn` → `PromptHelperSlotIn` (`:127-155`). `GET /api/memory` / `POST /api/memory` (`:904-918`) → `GET`/`POST /api/prompt-helpers` answering `{"slots": […]}`. `_prompt_helpers_block()` beside `_agent_modes_block()` (`:1215-1225`); `"promptHelpers"` on `GET /api/path-location` (`:1236-1239`) and on the `POST` return (`:1346-1354`) with `prompt_helpers.ensure_prompt_helpers_file(expanded)` called beside the other two seeds (`:1338-1339`). |
| `src/config.py` | **delete** `MEMORY_*` (`:21-28`), `memory_path` … `save_memory_registration` (`:387-458`), and the name skip in `list_question_sets()` (`:360-361`). |
| `tests/test_memory.py` | **delete** (its contents become `tests/test_prompt_helpers_api.py`, below) |
| `tests/test_prompt_helpers_api.py` | **new** — the HTTP half, carrying over every still-valid case from `test_memory.py` |
| `tests/test_api.py` | the `TestPathLocationEndpoint` payload rows (`:667-705`): a third block on GET and POST |
| `tests/test_prompt_helpers_seed.py` | mbl-02's static guard: widen the caller set to `{src/api/routes.py, src/api/server.py}` |

No frontend file changes here — mbl-04 owns the SPA.

## 1. The routes

```
GET  /api/prompt-helpers  ->  {"slots": [ {color, content} x9 ]}
POST /api/prompt-helpers  <-  {index, color, content}   ->  {"slots": […]}
```

- `PromptHelperSlotIn`: same three fields and the same validators as today
  (`index` `ge=0, le=prompt_helpers.SLOT_COUNT - 1`; hex colour via
  `prompt_helpers.COLOR_RE`; the `CONTENT_MAX_CHARS` cap), now sourced from the
  new module. Keep the docstring's point that there is nothing to create or
  delete — reworded for the new file.
- Both routes catch `prompt_helpers.PromptHelpersFileError` and re-raise as
  `HTTPException(422, detail=str(exc))` — D4: *"it does not become a 500"*. A bad
  `version`, bad JSON and a wrong count all arrive through that one exception,
  so one `try` per route is enough; the sentence the module built (naming the
  path, and the count) is what the user sees.
- Keep the existing comment about **not** gating this POST while a run streams
  (`routes.py:912-915`): that decision is unchanged.
- Update the docstrings — "self-heals the file to 9 defaults" is now false. The
  GET's docstring should say the true thing: a missing file answers the defaults
  and writes nothing; a malformed file is a 422.

## 2. The working-location endpoints

`_prompt_helpers_block() -> Dict[str, Any]` next to `_agent_modes_block()`, one
line returning `prompt_helpers.prompt_helpers_file_state(path_location())`, with
a docstring in the same shape as its two neighbours: read-only, because the
dialog opens far more often than a workspace is switched.

- `GET /api/path-location`: add `"promptHelpers": _prompt_helpers_block()` to the
  payload (`:1236-1239`).
- `POST /api/path-location`: `helpers_block = prompt_helpers.ensure_prompt_helpers_file(expanded)`
  right after `modes_block` (`:1339`), and `"promptHelpers": helpers_block` in the
  returned dict (`:1346-1354`). It stays **after** both hub restarts for the
  reason already written at `:1319-1322` (a seeding hiccup must not be mistaken
  for a hub failure and roll the environment back). There is **no cache
  invalidation** to add — say so in the comment beside the other two seeds, where
  `agent_modes.invalidate_agent_modes_cache()` sits, so the absence reads as
  deliberate.
- A seed failure is a note in the payload, never a 4xx (D5, and the invariant in
  the big picture's project rules).

## 3. `config.py` loses the feature entirely (D3)

Delete, do not leave shims or re-exports:

- `MEMORY_REGISTRATION_FILENAME`, `MEMORY_SLOT_COUNT`, `MEMORY_DEFAULT_COLOR`,
  `MEMORY_CONTENT_MAX_CHARS`, `MEMORY_COLOR_RE` and the comment above them
  (`:21-28`);
- `memory_path`, `default_memory_registrations`, `_normalize_memory_slot`,
  `_write_memory_registrations`, `load_memory_registrations`,
  `save_memory_registration` (`:387-458`);
- the skip at `:360-361` in `list_question_sets()` — after this move there is no
  non-question-set file in `resources/` to skip. Also fix the docstring sentence
  that mentions it, if any.

`grep -rn "MEMORY_\|memory" src/config.py` must be empty. `resources/` itself is
untouched: **do not delete `resources/memory_registration.json`** (mbl-02's rule;
the file is the user's to remove).

## Tests

Delete `tests/test_memory.py` and re-home its cases into
`tests/test_prompt_helpers_api.py`, with two changes of substance:

- the fixture is `monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))`
  (per test, like `tests/test_agent_modes.py:127-134`), **not**
  `monkeypatch.setattr(config, "RESOURCES_DIR", …)`;
- the three tests that assert the reader repairs the file
  (`test_creates_defaults_when_file_missing`'s file assertion,
  `test_corrupt_file_is_normalized`, `test_short_array_is_padded`,
  `test_long_array_is_truncated_and_bad_color_coerced`) are **replaced**, not
  renamed. §7's table is the replacement list.

Required cases (§7, HTTP half):

1. `GET /api/prompt-helpers` on an empty root → 200, nine defaults, and **no
   `.nojae` folder on disk**;
2. bad JSON / `version: 2` / missing `version` → **422** whose `detail` names the
   path, and the file is byte-identical afterwards;
3. eight entries, and ten entries → **422** naming the count, file untouched, and
   the response is not a 500;
4. nine entries with `"color": "bogus"` → 200 with `DEFAULT_COLOR`, and the file
   **is** rewritten with the coercion;
5. `POST` one slot → 200, `{"slots": […]}` with that slot, nine entries on disk,
   and the document on disk uses the `helpers` key with `version: 1`;
6. the still-valid validation cases from `test_memory.py` (index out of range,
   bad colour in the request, oversized content) keep their assertions;
7. **per-workspace**: seed root A with a filled slot via `POST`, switch
   `BENCHMARKIT_PATH_LOCATION` to root B, `GET` answers root B's nine defaults
   and root A's file is not read;
8. **switch payload**: `POST /api/path-location` reports `promptHelpers.state`
   (`"created"`, or `"migrated"` when a legacy file is present — patch
   `prompt_helpers.LEGACY_PROMPT_HELPERS_PATH` for that case), and a failing seed
   (chmod the root read-only) is a note with `state: "failed"`, **not** a 4xx —
   the response is still 200;
9. the old names are gone: `GET /api/memory` and `POST /api/memory` answer 404;
10. the question-set guard: with an isolated `RESOURCES_DIR` (patch
    `config.RESOURCES_DIR` — this is the one place that fixture style is still
    right), every `*.json` that carries a `questions` key is listed, and a file
    named `memory_registration.json` **that does carry a `questions` key** is now
    listed (proving the name-based skip is gone). Hermetic: do not assert
    anything about this checkout's real `resources/` contents.

`tests/test_api.py::TestPathLocationEndpoint`: extend the two existing payload
tests to assert `set(body["promptHelpers"]) == {"state", "path", "error"}` and
that the path ends with `POLICY_DIRNAME + os.sep + PROMPT_HELPERS_FILENAME`, and
that the POST's `promptHelpers["state"]` is `"created"` for a fresh `tmp_path`
root — same shape as the `agentModes` assertions beside them.

## Acceptance

- `pytest -q` fully green. `tests/test_memory.py` no longer exists;
  `pytest tests/ -q --collect-only` shows `test_prompt_helpers.py`,
  `test_prompt_helpers_seed.py`, `test_prompt_helpers_api.py`.
- `grep -rn "MEMORY_\|memory_path\|/api/memory\|MemorySlotIn" src tests` is
  empty (allow `mcp_connectors`/`lsp_connectors` hits only if they are about the
  in-memory event bus — there should be none).
- `grep -n "prompt-helpers" src/api/routes.py` shows both routes and both
  payload blocks.
- Manual check on a scratch root, not this checkout: start the app with
  `BENCHMARKIT_PATH_LOCATION` pointed at an empty temp dir,
  `curl localhost:8000/api/prompt-helpers` → nine empty slots and
  `ls <temp>` shows **no** `.nojae` folder; `curl localhost:8000/api/memory` → 404.
- The frontend is expected to be broken for the bullet row only (404 ⇒ nine empty
  bullets); everything else works. Do not "fix" it by keeping an
  `/api/memory` alias — the rename is the point (D8).
- Move this file to `todos/done/` when the above is true.
