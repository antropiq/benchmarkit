# mbl-02 — seeding the workspace file, and the one-shot lift from the legacy path

Add the seeding half to the module mbl-01 created — `ensure_prompt_helpers_file`,
`prompt_helpers_file_state`, `default_prompt_helpers_text`,
`LEGACY_PROMPT_HELPERS_PATH` and the D6 migration — and wire the boot entry
point into `api/server.py`. **The routes are not touched here**: `GET
/api/memory` still serves the old app-level file, so the app's visible behaviour
is unchanged. mbl-03 switches the readers over.

Depends on: mbl-01 (the module, its loader, its constants).
Spec: **D5** (seeding follows the pair, at the same two moments; a reader never
writes; no cache) and **D6** (the lift: order, states, the not-migrated case,
leaving the legacy file alone). §7's rows *exclusive create*, *the lift*, *a
legacy file that will not lift*, *no lift over an existing file* are this task's
test list.

## Files

| file | change |
|---|---|
| `src/core/prompt_helpers.py` | gains `ensure_prompt_helpers_file`, `prompt_helpers_file_state`, `_ensure_prompt_helpers` (the inner), `_state_dict`, `LEGACY_PROMPT_HELPERS_PATH`, and the migration; `default_prompt_helpers_text()` landed in mbl-01 — extend it if it needs the legacy document helper |
| `tests/test_prompt_helpers_seed.py` | **new** — the seeding and migration tests |
| `src/api/server.py` | `_seed_prompt_helpers()` beside the two existing seeders, and its call in `create_app()` after `_seed_agent_modes()` (`:144`) |
| `tests/test_prompt_helpers.py` | mbl-01's file: only if a shared fixture must be factored out. No behaviour edits. |
| `tests/test_agent_modes_seed.py` | **must** edit `:325-351`: that test asserts boot leaves *exactly two* files in `.nojae`. Boot now leaves three. Add `prompt_helpers.PROMPT_HELPERS_FILENAME` to both the `folder.iterdir()` list and its docstring ("the two files" → three), or the suite goes red on your change. |

No route changes, no `config.py` changes, no frontend changes.

## 1. `ensure_prompt_helpers_file(root)` — the shape is already written

Copy the shape of `agent_modes.ensure_agent_modes_file` / `_ensure_agent_modes`
(`agent_modes.py:856-916`) — read them in full first. The invariants it carries
are the ones D5 repeats:

- one `try`/`except Exception` at the top returning
  `{"state": "failed", "path": …, "error": "…"}`: **a seed is a note, never a
  raise**. A workspace with no helpers file chats with nine empty bullets.
- the folder is a **one-level** `os.mkdir(folder, mode=0o755)` tolerating
  `FileExistsError`; the file is one `os.open(path, O_WRONLY | O_CREAT | O_EXCL,
  0o644)` — the exclusivity is the whole race story, see
  `tests/test_editorfs_policy_seed.py` (*"the race is closed by the flag, not by
  a check"*). No `isfile()` guard before the open.
- `FileExistsError` + a regular file → `"existing"`, writing nothing; a **folder**
  standing at the path → `"failed"` with a sentence naming that, exactly as
  `agent_modes.py:889-895`.
- the bytes are written from `default_prompt_helpers_text()` — the same text the
  loader is asserted to read cleanly (D5: the seeder's bytes and the loader's
  expectations must not agree by luck; `policy_seed.py:9-15` is the argument).
- a write failure discards the half-written file and answers `"failed"`
  (`_discard_seeded_file`, `agent_modes.py:907`).
- **no cache to invalidate** — say so in a comment where `agent_modes.py:910-915`
  puts its invalidation, so the difference is deliberate and visible (D5).

## 2. `prompt_helpers_file_state(root)` — the read-only mirror

Mirror `agent_modes.agent_modes_file_state` (`agent_modes.py:919-940`): `stat`
only; `absent` / `existing` / `failed`; never a write. `GET /api/path-location`
will call it in mbl-03. Keep the docstring's reason ("the settings dialog opens
far more often than a workspace is switched") — it is the sentence that stops a
later reader from calling the seeding version instead.

## 3. The lift (D6) — read this twice, the order is the design

Module attribute, so a test can move it (the `mcphub.constants` convention):

```python
LEGACY_PROMPT_HELPERS_PATH: Path = <install dir> / "resources" / "memory_registration.json"
```

`ensure_prompt_helpers_file(root)` gains one branch, **only on the branch where
the exclusive create succeeded**:

- new file already existed → `"existing"`, and the legacy path is **never
  opened**. No clobbering, no second copy.
- create succeeded, no legacy file → `"created"` (today's behaviour).
- create succeeded, legacy file exists and parses, its `"registration"` list
  holds **exactly nine** entries → rewrite the new file with those slots, each
  through `_normalize_slot`, and answer `"migrated"`.
- create succeeded, legacy list holds any other count → the new file keeps the
  nine defaults, answer `"migration-skipped"` naming the count found and the
  legacy path. D6: silent repair during migration is the one place an overwrite
  would stay invisible forever, because nothing reads the old path again.
- a legacy file that is not readable JSON, or has no `registration` list → treat
  it as "no legacy file": `"created"`. There is no state for a file that was
  never a helper store.

Hard rules:

- the legacy file is **left in place** — no `os.remove`, no `os.replace`, no
  `.nojae/trash`. Not a side effect of a rename (D6's rejected alternative
  explains the risk being avoided: destroying the only copy of somebody's
  prompts if the new write fails).
- `load_prompt_helpers()` (mbl-01) must **not** grow any legacy fallback. One
  lookup, one truth; the lift happens only at a write moment.
- the legacy document's key is `"registration"` (the old file's shape,
  `config.py:420`); the new document's key is `"helpers"`. Do not confuse them —
  the migration reads the old key and writes the new one.

## 4. The boot entry point

`_seed_prompt_helpers() -> Dict[str, Any]` in `api/server.py`, a sibling of
`_seed_workspace_policy` (`:35-55`) and `_seed_agent_modes` (`:58-79`) — same
contract: call the ensure, guard the unexpected, `logging.warning` on a failure,
return the dict. Call it in `create_app()` right after `_seed_agent_modes()`
(`:144`), before `_sweep_trash()`, with a comment in the style of the one at
`:137-143`: same reserved folder, after the seed that creates it, failure is a
note and not a boot failure. Do **not** call it from `config.py` or from
`path_location()` (D5: *"never called from `config.path_location()`"*).

`POST /api/path-location` gains its call in **mbl-03** (that task owns the
payload), so after this task boot seeds and the switch does not yet. Do not fix
that here; note it in the commit.

## Tests (`tests/test_prompt_helpers_seed.py`)

Model the file on `tests/test_agent_modes_seed.py` (read its docstring: it is
the same set of invariants for a different file, including "a reader never
writes"). Same fixture style: `tmp_path` as `BENCHMARKIT_PATH_LOCATION`, paths
built from the module attributes, the legacy path patched through
`monkeypatch.setattr(prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", …)` — never
the real `resources/memory_registration.json`.

Cover at least, one test per §7 row:

1. `created` on an empty root: file exists, holds `version: 1` and nine defaults,
   `0o644`, one `.nojae` level;
2. a pre-existing hand-edited file is byte-identical after a second call
   (`"existing"`, and check size **and** mtime like
   `test_editorfs_policy_seed.py:75-95` does);
3. a folder standing at the file path → `"failed"`, not `"created"`;
4. `prompt_helpers_file_state` on an empty root answers `"absent"` and leaves **no
   `.nojae` folder**; after `ensure_*` it answers `"existing"`;
5. **the lift**: legacy file with 5 filled slots padded to nine → `"migrated"`,
   the five prompts in the new file in the same order, the legacy file still
   present and byte-identical, and a second `ensure_*` answers `"existing"` and
   changes nothing;
6. **a legacy file that will not lift**: seven entries → the new file holds the
   nine defaults, `"migration-skipped"` naming `7` and the legacy path, legacy
   bytes untouched;
7. **no lift over an existing file**: both paths present, new file pre-written
   with a sentinel → byte-for-byte the sentinel, `"existing"`;
8. a legacy file of junk, or of nine entries whose colours are malformed →
   `"created"` / normalised slots respectively;
9. read-only root (a chmod-guarded folder, as `test_agent_modes_seed.py:222`
   does) → `"failed"` with an error string, and the caller does not raise;
10. the **static guard**, mirroring `test_agent_modes_seed.py:515-533`: walk
    `src/**/*.py` and assert the only callers of `ensure_prompt_helpers_file(` are
    `src/api/server.py` (this task) and, once mbl-03 lands, `src/api/routes.py`.
    Write it as a set that this task asserts equals `{"src/api/server.py"}` —
    mbl-03 widens it. Also assert `"ensure_prompt_helpers_file" not in
    src/config.py`.
11. boot seeds it: `create_app()` against a fresh `tmp_path` root leaves
    `prompt-helpers.json` in `.nojae` — this is the same test mbl-02 edits in
    `test_agent_modes_seed.py:325-351`; do not duplicate it in the new file.

## Acceptance

- `pytest tests/test_prompt_helpers_seed.py tests/test_prompt_helpers.py -q`
  green; full `pytest -q` green with **no existing test edited** (the app's
  visible behaviour is unchanged — `/api/memory` still reads the old file).
- Booting the app does not create a `.nojae` folder in a scratch root that has
  none… except that it does now, by design: boot seeds. What must not happen is
  `GET` seeding — that guard is test 4.
- `grep -n "_seed_prompt_helpers" src/api/server.py` shows the wrapper and its
  call; `grep -n "ensure_prompt_helpers_file" src/config.py` is empty.
- `resources/memory_registration.json` is untouched: `git status` shows no
  deletion, no move, no trash entry.
- Move this file to `todos/done/` when the above is true.
