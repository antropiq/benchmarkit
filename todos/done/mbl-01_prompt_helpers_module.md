# mbl-01 — `src/core/prompt_helpers.py`: the module, with no callers yet

Create the new core module that owns the prompt-helper slots end to end —
constants, path, read, write, defaults, the error type and the numeric
self-check — plus its unit tests. **Nothing imports it yet**: `config.py` keeps
working exactly as today and the app's behaviour does not change one bit. mbl-03
does the cutover and deletes the old `config.py` block.

Depends on: nothing (mbl-01 is first).
Spec: `docs/prompt-helpers-workspace-scope.md` **D2** (name and location from one
place), **D3** (the logic leaves `config.py`; the mirrored function names),
**D4** (exactly nine slots, wrong count raises, field coercion stays). §4's first
row of "New / changed files" is the module's inventory.

## Files

| file | change |
|---|---|
| `src/core/prompt_helpers.py` | **new** |
| `tests/test_prompt_helpers.py` | **new** — unit tests for the loader, the writer, the normaliser and `_self_check`. Does **not** test seeding or migration (mbl-02) or the routes (mbl-03). |

Nothing else. No `src/config.py` edit, no route edit, no frontend edit.

## 1. Module shape

Model it on `src/core/agent_modes.py`, which the spec names as the pattern (D3:
*"loader + seeder in one file, because the seeder must write exactly the bytes
the loader reads"*). Read `agent_modes.py:40-75` before writing: the module
header, the type-annotation style, and the way constants are declared and
explained are what a reviewer will compare this to.

Required names (exact spellings — later tasks and the shared contract use them):

| name | meaning |
|---|---|
| `PROMPT_HELPERS_FILENAME = "prompt-helpers.json"` | module attribute so a test can move the lookup (D2) |
| `POLICY_DIRNAME` | **imported**, not re-declared — see §2 |
| `PROMPT_HELPERS_VERSION = 1` | a mismatch is an error, not a guess (§4 "The document") |
| `SLOT_COUNT = 9` | layout contract (D4) |
| `DEFAULT_COLOR = "#808080"`, `CONTENT_MAX_CHARS = 32000`, `COLOR_RE` | same values as today's `MEMORY_DEFAULT_COLOR` / `MEMORY_CONTENT_MAX_CHARS` / `MEMORY_COLOR_RE` (`config.py:23-28`) |
| `class PromptHelpersFileError(Exception)` | carries the message naming path and count |
| `prompt_helpers_path() -> Path`/`str` | the active workspace's file, from `config.path_location()` |
| `prompt_helpers_file_path(root: str) -> str` | the same for an explicit root (mirrors `agent_modes.agent_modes_file_path`) |
| `default_prompt_helpers() -> List[Dict[str, str]]` | nine empty gray slots — byte-for-byte what `config.default_memory_registrations()` returns today (`config.py:392-397`) |
| `default_prompt_helpers_text() -> str` | the seeded document, `json.dumps` of a Python literal (`{"version": 1, "helpers": nine defaults}`) |
| `load_prompt_helpers() -> List[Dict[str, str]]` | read side, see §3 |
| `save_prompt_helper(index: int, color: str, content: str) -> List[Dict[str, str]]` | replaces one element, writes all nine, returns them |
| `_normalize_slot(item: Any) -> Dict[str, str]` | today's `config._normalize_memory_slot` (`config.py:400-415`) moved verbatim in behaviour |
| `_self_check() -> None` | see §5 |

Leave `ensure_prompt_helpers_file`, `prompt_helpers_file_state`,
`LEGACY_PROMPT_HELPERS_PATH` and the migration out — **mbl-02 adds them**. Do not
stub them here; a half-built seeder invites mbl-02 to contradict D6.

## 2. The folder name comes from one place

Do **not** write the literal `".nojae"` in this module. Import
`POLICY_DIRNAME` from `mcp_connectors/policy_shared` with the `sys.path` fallback
block, copying `agent_modes.py:48-63` — that module lives outside `src` for the
connectors' sake, so the import needs the same guard, and `agent_modes.py:49`
already says why. `agent_modes.py:63` re-exports it as
`POLICY_DIRNAME: str = shared.POLICY_DIRNAME`; do the same.

The path is built with `os.path.join(root, POLICY_DIRNAME, PROMPT_HELPERS_FILENAME)`
against `config.path_location()` — the same construction as
`agent_modes.py:599` and `:823`. `RESOURCES_DIR` plays no part: this is
workspace-relative.

## 3. The read side (D4 + D5) — this is the part that changes behaviour

`load_prompt_helpers()`:

- **file absent** → return the nine defaults and write **nothing**. Not "create
  the file", not "warn". A reader that writes is exactly what D5 forbids
  (`policy_seed.py:22-25`, `agent_modes.py:805-807` — read those comments, the
  reasoning is already written down).
- **unreadable / not JSON / `version` missing or ≠ `PROMPT_HELPERS_VERSION` /
  `helpers` not a list / list length ≠ `SLOT_COUNT`** →
  `raise PromptHelpersFileError(...)` with a sentence naming the path and, for
  the count case, the count found. No pad, no truncate, no rewrite: the bytes on
  disk must be identical after the call.
- **nine entries, a malformed field** → return the normalized list and **rewrite
  the file** (D4's last paragraph: coercing a colour in memory only would let the
  next save re-commit the bad value). Reuse today's `_write_memory_registrations`
  write style (`config.py:418-423`: `json.dumps(payload, indent=2,
  ensure_ascii=False) + "\n"`), but the payload key is `"helpers"` and the
  document carries `"version"`.

Note the payload key change: on disk it is `{"version": 1, "helpers": […]}` —
the file key is `helpers` (§4), while the **route** payload key is `slots`
(`{"slots": […]}`, D8). Those are two different keys and mbl-03 owns the second.

`save_prompt_helper` cannot emit a wrong length: it calls `load_prompt_helpers()`
(which raised otherwise) and replaces one element (D4's last paragraph). Say that
in a comment so nobody adds a guard that can never fire.

## 4. No cache

`agent_modes` caches per root because it is consulted on every chat turn; this
file is read on a tab switch and on a switch. D5: *"Nine small strings do not
earn a cache and its invalidation rules."* Do not add one, and put that sentence
(in your own words) in the module docstring so the next reader does not "improve"
it in.

## 5. `_self_check()`

Mirror `agent_modes._self_check()` (`agent_modes.py:942-959`) — read it first.
It asserts the numeric/format invariants (`SLOT_COUNT` matches the row the UI
draws, `DEFAULT_COLOR` matches `COLOR_RE`, `PROMPT_HELPERS_FILENAME` and
`POLICY_DIRNAME` are non-empty strings) and, the way `agent_modes.py:960` does,
names the attributes it checked in the failure text. Call it at import time the
same way `agent_modes` does.

## Tests (`tests/test_prompt_helpers.py`)

Fixture: a throw-away `tmp_path` set as the active location with
`monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))`, shaped like
`tests/test_agent_modes.py:127-134`. Never `config.RESOURCES_DIR`, never this
checkout. Write files at `root/.nojae/prompt-helpers.json` using
`prompt_helpers.POLICY_DIRNAME` / `PROMPT_HELPERS_FILENAME` (not literals) so a
rename cannot silently stop testing the real path.

Cover at least:

1. absent file → nine defaults returned **and** `not (tmp_path / ".nojae").exists()`;
2. 8 entries and 10 entries → `PromptHelpersFileError` naming the count, and the
   file's bytes unchanged (read them before and after);
3. bad JSON, a missing `version`, `version: 2`, `helpers` not a list → the same
   error naming the path;
4. a bad colour (`"bogus"`) → `DEFAULT_COLOR`, and the file **is** rewritten with
   the coercion; a non-string `content` → `""`;
5. `default_prompt_helpers_text()` round-trips: write it at the path, then
   `load_prompt_helpers()` returns `default_prompt_helpers()` with no error
   (the `policy_seed`/`agent_modes` pairing test the spec asks for in §7);
6. `save_prompt_helper(3, "#0000ff", "x")` → slot 3 set, length still 9, and the
   file on disk holds `version` + `helpers` with that slot;
7. `_self_check()` passes at import.

## Acceptance

- `pytest tests/test_prompt_helpers.py -q` green; full `pytest -q` **unchanged**
  (no existing test edited, because nothing was cut over yet).
- `cd web && npm test` unchanged green; `cd web && npm run build` succeeds.
- `grep -rn "prompt_helpers" src` matches only the new module (plus, if you
  wrote it, nothing else) — **no caller**. If `src/config.py` or
  `src/api/routes.py` changed, this task overreached.
- The new module contains none of the three write-surface markers
  (`PUT /api/fs`, `POST /api/fs`, `DELETE /api/fs`).
- Move this file to `todos/done/` when the above is true.
