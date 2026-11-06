# ws-01 — `src/core/workspace_settings.py` and its template, with no callers yet

Create the new core module that owns `.nojae/nojae-settings.json` end to end —
constants, path, seed, cached read, write, typed accessor, the member-set check —
plus the committed template it seeds from, plus unit tests. **Nothing imports it
yet**: `config.py` keeps working exactly as today and the app's behaviour does not
change one bit. ws-03 (the reads) and ws-04 (the writes) do the cutover.

Depends on: nothing (ws-01 is first).
Spec: `docs/nojae-settings-workspace-spec.md` **§2** (the file and its key map —
the authoritative member list, types and "member absent means" column), **§5**
("New module" and "New template" — the required names), **D6** (native JSON types),
**D8** (seed text reported, not dialed), **D9** (malformed file is a named state).
Read those sections plus the §2 JSON block verbatim before writing.

## Files

| file | change |
|---|---|
| `src/core/workspace_settings.py` | **new** |
| `nojae-settings.default.json` | **new**, at the install root (sibling of `.env.example`) — the §2 JSON verbatim |
| `tests/test_workspace_settings.py` | **new** — unit tests for path, seed, read, write, the seed-text and member-set checks. Does **not** test the boot seed (ws-02) or the routes (ws-04). |

Nothing else. No `src/config.py` edit, no route edit, no frontend edit, no `.gitignore`
edit (`.nojae/` is already ignored; `nojae-settings.default.json` is committed).

## 1. Model it on the folder's other app-owned store

Read `src/core/prompt_helpers.py` first — the spec names it (§4) as the shape to
copy: the module header and type-annotation style, the `try: from mcp_connectors
import policy_shared` block with the `sys.path` fallback that yields
`POLICY_DIRNAME` (never re-declare the literal `".nojae"`), `prompt_helpers_file_path(root)`
vs `prompt_helpers_path()` (explicit root vs `path_location()` per call),
`_state_dict(state, path, error)`, and the `_self_check()` called at import.

Required names (exact spellings — later tasks and the shared contract use them):

| name | meaning |
|---|---|
| `WORKSPACE_SETTINGS_FILENAME = "nojae-settings.json"` | module attribute so a test can move the lookup (the repo's `mcphub.constants` convention) |
| `WORKSPACE_SETTINGS_TEMPLATE: Path` | `<install root>/nojae-settings.default.json`, built from `PROJECT_ROOT`; a module attribute for the same reason |
| `POLICY_DIRNAME` | **imported**, not re-declared — as in `prompt_helpers.py` |
| `WORKSPACE_SETTINGS_KEYS: Dict[str, type]` | the approved member set → its JSON type, read off the §2 key map. The member-set check of D9 compares against its keys, and it is what `save_workspace_settings` validates against. |
| `class WorkspaceSettingsFileError(Exception)` | carries the sentence naming the path (and, for a bad member, the member) |
| `workspace_settings_file_path(root: str) -> str` | `os.path.join(root, POLICY_DIRNAME, WORKSPACE_SETTINGS_FILENAME)` |
| `workspace_settings_path() -> str` | the active workspace's file, from `path_location()`, resolved **per call** |
| `ensure_workspace_settings_file(root: str) -> Dict[str, Any]` | exclusive-create copy of the template; answers a state dict; **raises nothing** |
| `load_workspace_settings() -> Dict[str, Any]` | cached read; see §3 |
| `save_workspace_settings(values: Mapping[str, Any]) -> None` | see §4 |
| `invalidate_workspace_settings() -> None` | drops the cache; called by the save path and by `post_path_location()` |
| `workspace_setting(name: str) -> Any` | the typed accessor group 1's readers use — `config.lsp_enabled_ids()` / `max_tool_iteration()` in ws-03; answers the member or its documented "absent" default |
| `workspace_settings_state() -> Dict[str, Any]` | read-only `{state, path, error}` for the dialog (ws-04's `GET /api/settings` reads it); no write |
| `_self_check() -> None` | see §6 |

The **state vocabulary** is `prompt_helpers._state_dict`'s: `ok` / `absent` /
`failed`, plus the seed's `created` / `existing` for `ensure_*`, and the D8
"still the seed text" condition which ws-04 renders as "not configured" — carry it
as a field on the state dict (e.g. `"configured": bool`) rather than inventing a
new state name, so the dialog has one shape to switch on.

## 2. The seed (exclusive create, one input)

`ensure_workspace_settings_file(root)` copies the template's **bytes** to
`<root>/.nojae/nojae-settings.json` with an exclusive create (the `mcphub/seed.py`
and `prompt_helpers.ensure_prompt_helpers_file` precedent): an existing file is
**never** overwritten, `mkdir(parents=True)` for the folder, and every failure is
answered as `{"state": "failed", ...}` — the function **raises nothing**, so a
seeding hiccup is a note and never a boot or hub failure (the invariant the big
picture names). One input only (D11): read the template, write it. No `.env` branch,
no leftover-values branch, no migration — that whole topic is decided (§9).

## 3. The read side (D8 + D9) — this is the part that must not be papered over

`load_workspace_settings()`:

- **absent** → state `absent`; answer `{}` (or an empty mapping) and write nothing.
  A reader never writes.
- **not valid JSON** → state `failed`, `error` = the parse error's own sentence
  naming the path. Never a silent default (D9).
- **a member outside `WORKSPACE_SETTINGS_KEYS`** → state `failed` naming that
  member. `EndpointsConfig`'s `extra="ignore"` would swallow a typo silently, so
  this reader checks the member set itself (D9).
- **a member of the wrong JSON type** (per the §2 map: `string` / `boolean` /
  `number` / `integer` / `array of strings`) → `failed` naming the member and what
  it expected. `null` is not a value a member of this file may hold.
- **`serverUrl` or `endpoint` still equals the template's instruction sentence**
  (D8) → the file parses and is `ok`, but reports **not configured**: match the
  template's exact strings and set the flag so the caller answers "not configured"
  naming the file and its path instead of making a transport call with a sentence.
  Keep the comparison against the strings read from the template constant, not a
  re-typed copy.
- **cache**: cache the parsed mapping keyed on the resolved path, like
  `agent_modes` caches per root — group 1 is read on every chat turn.
  `invalidate_workspace_settings()` is the only way it changes; a dialog save and a
  workspace switch call it (D4). Nothing re-reads per request.

## 4. The write side (D7 keeps the names separate)

`save_workspace_settings(values)` takes **JSON member names** (this module knows
nothing about dialog fields — the field→member table lives in `routes.py`, ws-05):

- reads the current file (so it merges), writes **only the members it is given**,
  preserving valid-but-unmentioned ones, then calls `invalidate_workspace_settings()`.
- rejects an unknown member (raise `WorkspaceSettingsFileError`, naming it) — the
  same check as the read side, one rule, so a caller cannot introduce by save the
  very typo the reader would refuse.
- writes with the folder's own style (`json.dumps(payload, indent=2,
  ensure_ascii=False) + "\n"`), so the bytes this writes are bytes the loader reads.

## 5. Native types, no coercion cleverness (D6)

Store and return native JSON values (`True`, `0.2`, `["pyright"]`). Do not stringify
"for safety" — the whole point of D6 is that the file holds native types and a
**second builder** (`EndpointsConfig.from_workspace`, ws-04) feeds them straight to
pydantic. `workspace_setting(name)` returns the member as stored; a caller wanting a
default passes its own, matching the §2 "member absent means" column.

## 6. `_self_check()`

Mirror `prompt_helpers._self_check()` / `agent_modes._self_check()` (read one).
Assert: `WORKSPACE_SETTINGS_FILENAME` and `POLICY_DIRNAME` are non-empty strings;
`WORKSPACE_SETTINGS_TEMPLATE` exists and parses as JSON; **its member set equals
`WORKSPACE_SETTINGS_KEYS` exactly** (the template and the check must never drift);
its `maxToolIteration` is an `int` and its `lspEnabled` is a `list` (the two
members whose template value is a real value, spec §2's note). Name the checked
attributes in the failure text. Call it at import.

## Tests (`tests/test_workspace_settings.py`)

Fixture: `monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))` per test
(the `test_agent_modes.py` shape). Build paths from `workspace_settings.POLICY_DIRNAME`
/ `WORKSPACE_SETTINGS_FILENAME` / `WORKSPACE_SETTINGS_TEMPLATE`, never literals, so
a rename cannot silently stop testing the real path. Never touch this checkout.

Cover at least:

1. `ensure_*` on an empty root → `state: "created"`, the file's bytes **equal the
   template's bytes**; a second call → `"existing"` and the file untouched (write a
   marker into it first, then call again and re-read).
2. `ensure_*` with the template patched to a nonexistent path → `"failed"` and
   **no exception**; with the root unwritable (chmod) → `"failed"`, no exception.
3. `load_*` absent → empty mapping, state `absent`, and **no `.nojae` folder created**.
4. `load_*` malformed JSON → state `failed`, error names the path, file byte-identical
   after the read.
5. `load_*` with an unknown member (`"temperture": 0.2`) → `failed` naming that
   member (D9); with `"temperature": "0.2"` (string) → `failed` naming the type;
   with `"temperature": null` → `failed`.
6. `load_*` on a freshly seeded file → parses `ok` but **not configured**, and the
   message names the file's path (D8). Then set `serverUrl`/`endpoint` to real
   values → configured.
7. `save_workspace_settings({"temperature": 0.7})` on a file holding a real
   `serverUrl` → `temperature` updated, `serverUrl` **preserved**, cache invalidated
   (a following `load_*`/`workspace_setting` answers the new value without a reload
   of the module).
8. `save_workspace_settings({"lspEnabled": ["pyright", "typescript"]})` → the file
   holds a JSON array, and `workspace_setting("lspEnabled")` returns that list
   (native type, D6).
9. `save_workspace_settings({"temperture": 0.2})` → raises, file untouched.
10. `ensure_*` writes a document that `load_*` accepts with state `ok` (the
    seeder/loader pairing test the folder's other modules carry).

## Acceptance

- `pytest tests/test_workspace_settings.py -q` green; full `pytest -q` **unchanged**
  (no existing test edited, nothing cut over yet).
- `grep -rn "workspace_settings" src` matches **only** the new module — no caller.
  If `src/config.py`, `src/api/routes.py` or `src/api/server.py` changed, this task
  overreached.
- `nojae-settings.default.json` parses, is at the install root, and its member set
  equals `WORKSPACE_SETTINGS_KEYS` (the self-check proves it at import).
- The new module contains none of the three write-surface markers
  (`PUT /api/fs`, `POST /api/fs`, `DELETE /api/fs`) and reaches no write route.
- Move this file to `todos/done/` when the above is true.
