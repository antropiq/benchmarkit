# ws-04 — The writes cutover: the dialog splits across two writers, `.env` shrinks

Move the Settings dialog's writes onto the two stores — group 1 to
`save_workspace_settings()`, groups 2/3 to `save_settings()` — take `LLM_DEBUG` out
of the dialog API (with the trap that hides there), and shrink the `.env` trio in the
same commit. After this task the feature is functionally complete on the backend:
what the dialog saves is what the app reads.

Depends on: ws-03 (the reads already answer from the file — this task's writes must
land on the store the reads read). **Lands immediately after ws-03.**
Spec: **§5** (`routes.py` row — the six bullets), **D7** (one new mapping table, the
dialog's field names survive), **§7** (`LLM_DEBUG` leaves the dialog — read this
section twice, it names the trap), **§9** (the dead bytes), and §10's `test_api.py`
rows. The trio rule is in `AGENTS.md` (`SETTING_KEYS`, `_render_env()` and
`.env.example` move together).

## Files

| file | change |
|---|---|
| `src/api/routes.py` | the whole settings surface, §1–§4 below |
| `src/config.py` | `SETTING_KEYS` shrink, `REQUIRED_SETTING_KEYS` empty, `_render_env()` rewrite, §5 below |
| `.env.example` | loses the group-1 keys and their comments, gains the pointer comment (same commit — trio rule) |
| `tests/test_api.py` | §10's `test_api.py` rows, read **only** `TestSettingsEndpoint` (`:303`–) and the seed-note blocks beside it |

No `models.py`, no `server.py`, no `workspace_settings.py`, no frontend (ws-05).

## 1. The two tables and the split

- **New** `WORKSPACE_SETTING_KEYS: Dict[str, str]` — dialog field → JSON member
  (`serverUrl→serverUrl`, `apiKey→serverApiKey`, `temp→temperature`,
  `supportsVision→supportsVision`, `endpoint`, `sysprompt`, `topP→topP`, `topK`,
  `minP`, `repeatPenalty`, `presencePenalty`, `reasoning`) — the sibling of the
  existing `SETTINGS_ENV_KEYS` / `VERIFIER_ENV_KEYS` tables (`routes.py:532-552`),
  written beside them with the same comment. `SETTINGS_ENV_KEYS` **loses exactly the
  fields that move** plus `llmDebug` (§3): what remains of it is at most the
  question-set plumbing — if the table ends up empty, delete it and its uses rather
  than keeping a table of nothing.
- `GET /api/settings` (`:879`) answers from **both stores**: group-1 fields read
  through the new table from `load_workspace_settings()` (native JSON values —
  the dialog's `supportsVision === '1'` comparison is ws-05's problem, and note the
  payload's types in the docstring so ws-05 reads them), groups 2/3 from
  `load_settings()` as today. It **grows the workspace state block** beside the
  existing seed-note blocks: `{"workspaceSettings": workspace_settings.workspace_settings_state()}`
  — reusing ws-03's single refusal sentence for the message, no second sentence.
- `POST /api/settings` (`:883`): the mid-run 409 guard is **unchanged**. Translate
  the payload through the two tables; run the existing `EndpointsConfig` validation
  gate **before either write**, on the **merged result** (the gate at `:896-905`
  today merges `os.environ` with the submitted env values — after this task it
  validates `EndpointsConfig.from_workspace(merged_workspace_values, merged_env_values)`,
  so a bad `reasoning` still cannot reach either disk). Then write: group 1 through
  `save_workspace_settings()` (member names, native types — `normalize_env_flag`
  has no caller for `supportsVision` any more, the checkbox becomes a real boolean),
  groups 2/3 through `save_settings()` with the same merge-forward of keys the dialog
  does not own. The `os.environ` merge dance around the gate stays honest for the
  env half only.

**The type seam to watch:** `SettingsFieldsMixin` *stringifies incoming scalars*
(`routes.py:79-88`: `null` becomes `""` = "unset"; numbers become strings; a
**boolean passes through untouched**). So the POST handler receives the numeric
group-1 fields as **strings** and must coerce them back to the file's native JSON
types — through `workspace_settings.WORKSPACE_SETTINGS_KEYS`' type map
(number / integer / boolean / string) — with `""` meaning *omit the member* (the §2
"member absent means" contract) — before calling `save_workspace_settings()`. The
file must never hold `"0.2"` or `"1"`: that is the dotenv wart D6 exists to end.
Whichever way the coercion and the omit rule end up shaped, the `test_api.py`
round-trip below (POST → GET answers native, and a cleared numeric field leaves its
member **absent** in the file rather than `null` or `""`) is the referee.

## 2. `_settings_json()` after the split

Drops `llmDebug`; the group-1 entries come from the JSON mapping with native values
(`supportsVision: true/false`, `temperature: 0.2` or `null`, numbers not strings).
It gains nothing else — the LSP tab already round-trips `lspEnabled` /
`maxIterations` through `GET`/`POST /api/lsp`, which ws-03 made read/write the JSON
file; do not duplicate them into the settings payload.

## 3. `LLM_DEBUG` leaves the dialog (§7 — the trap)

Remove, all in this commit: `SettingsIn.llmDebug`; its `SETTINGS_ENV_KEYS` entry;
**the `values["LLM_DEBUG"] = normalize_env_flag(...)` line in
`_settings_env_values()`** (`routes.py:628`); nothing else touches it.

The trap, in the spec's words: that line *always* emits the key, and
`env_values = {**load_settings(), **env_values}` lets the emitted `""` win over the
stored value — remove the dialog field without removing the line and every settings
save comments `LLM_DEBUG` out and pops it from `os.environ`; the debug switch would
switch itself off. `save_settings()`'s `os.environ` refresh and its
`apply_llm_debug()` call then simply re-apply the value the file still has;
`apply_llm_debug()` keeps its boot call (`server.py:200`). The §10 regression test
proves this exact scenario — write that test **first**, watch it fail with the line
in place, then remove the line.

The key stays a managed `.env` key (`SETTING_KEYS` keeps it), hand-edited like
`BENCHMARKIT_PYTHON_ROOT`. `normalize_env_flag()` keeps its `LLM_DEBUG`-shaped
semantics for whoever still calls it — if the `supportsVision` and `llmDebug` lines
were its only callers, the function itself and its unit tests go too; `grep` decides,
do not guess.

## 4. `post_path_location()`

ws-02 already placed the seed + invalidation there. This task owns nothing in that
route; if you find yourself editing it, you are duplicating ws-02 — stop.

## 5. The `.env` shrink (`src/config.py` + `.env.example`)

- `SETTING_KEYS` keeps exactly: the four `VERIFIER_AGENT_*` keys, `QUESTION_SET`,
  `LLM_DEBUG`, `BENCHMARKIT_PYTHON_ROOT` (spec §5). Group 1 leaves the tuple.
- `REQUIRED_SETTING_KEYS` becomes **empty** (it named `SERVER_URL` and `ENDPOINT`) —
  keep the constant and its comment (adjusted: "no key is always-written") rather
  than deleting it, `save_settings` consumes it.
- `_render_env()` loses the group-1 lines **and their comments**, and gains a pointer
  comment naming `.nojae/nojae-settings.json` where the `LSP_ENABLED` /
  `MAX_TOOL_ITERATION` comments sat — read the existing MCP pointer comment; it does
  this same job for the other moved store, copy its voice.
- `.env.example` loses the same keys and comments and gains the same pointer, in the
  same commit.
- `load_settings()` / `save_settings()` keep their `.env` job only — no behaviour
  change in them.

The **trio-parity test** already in `tests/test_api.py`'s style (the §10 row: "the
`.env` template no longer renders a group-1 key (parity between `SETTING_KEYS`,
`_render_env()` and `.env.example`)") — if a parity test already exists, extend it;
if not, add the three-way check: every `SETTING_KEYS` member is rendered by
`_render_env()` and present in `.env.example`, and no group-1 key appears in any of
the three.

## Tests (`tests/test_api.py`, §10)

- `POST /api/settings` writes group 1 to the **workspace file** and group 2 to
  **`.env`**, and neither write touches the other file (assert on both files' bytes).
- the `.env` trio parity, and `REQUIRED_SETTING_KEYS` is empty.
- `GET` then reflects a **second workspace's** values after a switch, not the first
  one's (the D4 cache-invalidation proof at API level; two `tmp_path` roots).
- the **`LLM_DEBUG` regression**: an existing `LLM_DEBUG=1` survives a settings save
  that never mentions it, and the payload has no `llmDebug` field. **Replaces
  `test_save_normalizes_llm_debug_and_applies_it`** — delete that test, do not keep
  both.
- a POST carrying a bad `reasoning` is a 422 and **neither file** was written (the
  validation gate before both writes).
- the fixture `TestSettingsEndpoint.isolated_settings_env` (`:303-310`) keeps its
  `SETTING_KEYS` wipe (now smaller) **and** gains ws-03's workspace-file
  neutralisation — a settings test must never read this checkout's file either.

## Acceptance

- `pytest -q` fully green, including `tests/test_workspace_settings*.py`,
  `tests/test_env_isolation.py`, `tests/test_config.py`, `tests/test_api.py`,
  `tests/test_lsp.py`.
- `grep -n "SERVER_URL\|TEMPERATURE\|LSP_ENABLED\|MAX_TOOL_ITERATION" src/config.py`
  → no group-1 key in `SETTING_KEYS`, `_render_env()` or the comments (pointer
  sentences naming the JSON file are fine); `grep -n "llmDebug" src/api/routes.py` →
  empty; `grep -c "SUPPORTS_VISION" .env.example src/config.py` → 0.
- The interim from ws-03 is closed: a `POST /api/settings` followed by a model call
  uses what was just saved (the existing settings tests already prove the read-your-
  write if they assert it; otherwise the switch test above carries it).
- Frontend untouched (`git diff web/` empty). The dialog still sends `llmDebug` in
  its payload, and that is **safe**: `SettingsIn` inherits
  `SettingsFieldsMixin.model_config = ConfigDict(extra="ignore")`
  (`routes.py:70-77`), so the dropped field is ignored, not a 422 — one fewer test
  in the interim between this task and ws-05. Add a comment on the `SettingsIn`
  class docstring only if its "(raw strings)" sentence stops being true for the
  group-1 fields after §2 (it may not: the *request* stays raw strings; only the
  GET's response goes native).
- Move this file to `todos/done/` when the above is true.
