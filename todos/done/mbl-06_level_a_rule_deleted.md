# mbl-06 — Level A: the `resources/memory_registration.json` rule is deleted, and the new file is hidden everywhere

Remove the app's own `ro` rule for the old path from the built-in policy, and
make the tests say the new truth: the file now lives in `.nojae`, which the
policy answers **`hidden`, unconditionally, in every workspace**. The rule is
deleted, not moved — there is no level-A pattern for the new path.

Depends on: mbl-01 (so the new path is a known constant) and mbl-03 (the old path
is no longer read by anything, so removing its rule strands no behaviour).
Spec: **D7** in full, plus §4's `builtin_policy.py` row and §7's *level A* row.

## Files

| file | change |
|---|---|
| `src/core/editorfs/builtin_policy.py` | **delete** the rule and its comment (`:86-91`) from `BUILTIN_RULES` |
| `tests/test_editorfs_builtin_policy.py` | the four sites that name the old path: `:129-134`, `:152`, `:239`, `:407-408`; plus new assertions that `.nojae/prompt-helpers.json` is `hidden` in a **foreign** workspace |

Nothing else. No change to `policy_modes`, `policy_glob`, `policy_shared`,
`builtin_rules()`'s reservation branch (`:147-162`), or the seeded level-B file.

## 1. Why nothing replaces the rule (read D7 before editing)

`builtin_rules()` already returns a `hidden` `Rule` for any path whose component
chain hits `POLICY_DIRNAME` (`:147-162`), **before** the `is_self_workspace()`
check at `:164-165`. That is the reservation, and it is unconditional: it applies
in a foreign workspace too. So the new file is:

- invisible in the tree, absent from search results, unreadable through
  `/api/fs/file` (**404**, because a hidden path is indistinguishable from
  absence), and a write through `/api/fs/file` is refused with **422 naming the
  reserved folder**;
- reachable by the app itself, which does not go through the Editor surface —
  `prompt_helpers.py` reads and writes the file in-process (D1: the app stays the
  file's only writer, and `POST /api/prompt-helpers` is the SPA's own path, not an
  `/api/fs` route).

The **behaviour change to state in the commit note**: the old file was browsable
in the tree and refused writes; the new one is not browsable at all. Less
reachable, not more.

Rejected by D7, so do not do it: adding a level-A pattern for
`.nojae/prompt-helpers.json`. A pattern naming `.nojae` is rejected by the policy
loader, so such a rule would be un-writeable anyway.

## 2. The test edits (§7's *level A* row)

In `tests/test_editorfs_builtin_policy.py`:

1. `:129-134` — the parametrised `(rel, pattern)` pair for
   `resources/memory_registration.json` **leaves the list**. Everything else in
   that list (`**/.env**`, `logs/**`, `web/dist/**`) stays.
2. `:145-155` — the `(rel, mode)` pair asserting `READ_ONLY` for that path
   **leaves the list**.
3. `:238-241` (`test_the_app_owned_names_are_ordinary_too`, foreign workspace) —
   drop `resources/memory_registration.json` from the loop, and **add the new
   assertion here**: in a **foreign** workspace,
   `.nojae/prompt-helpers.json` is `hidden` at `LEVEL_BUILTIN` with the reserved
   reason, while `resources/memory_registration.json` in that same foreign
   workspace is ordinary `READ_WRITE` (it is just a stale file there now). Use
   `prompt_helpers.POLICY_DIRNAME` and `prompt_helpers.PROMPT_HELPERS_FILENAME`
   to build the path, not a literal — a rename must not silently stop testing the
   real path.
4. `:407-408` — the parametrised list for
   `test_an_app_owned_build_or_data_file_is_readable_and_locked_from_writes`
   keeps `web/dist/index.html` and **loses** the resources path (that test is
   about `ro`: readable in the tree, locked from writes — which no longer
   describes our file).
5. New: an HTTP-level assertion on the **hidden** asymmetry for the new path, in
   the app workspace: `GET /api/fs/file?path=.nojae/prompt-helpers.json` → **404**,
   `PUT /api/fs/file` on the same path → **422** whose detail quotes the reserved
   folder rule and says benchmarkit's own rules did it, and the name does not
   appear in `GET /api/fs/tree?path=` at the root level. The file already has
   exactly this shape for `.env` and the reserved folder — find that test and
   extend its parameter list rather than writing a new one, unless the existing
   list is about a different verdict.

Also check for any other test that counts `BUILTIN_RULES` entries or asserts an
exact rule list (`grep -n "BUILTIN_RULES" tests`); if one asserts a length or a
verbatim tuple list, it must be updated — a rule was removed, and that is the
kind of count that lives in tests, not in AGENTS.md.

## 3. What must not change (the guard for this task)

- `.nojae` reservation semantics: unchanged. This task **adds no predicate**;
  `policy_modes.mode_for` stays the only place a verdict is computed.
- The 404-vs-422 asymmetry for hidden paths, the level-A/level-B ordering, the
  "strictest match wins" rule, and level A's dormancy in a foreign workspace (the
  test at `:242-246`, *"a copy of the repo is not the app"*, must still pass).
- The other three level-A rules and their reasons, verbatim.
- `resources/` question-set discovery: its guard test lives in mbl-03 (the task
  that deletes the name skip). Do not duplicate it here.

## Acceptance

- `pytest tests/test_editorfs_builtin_policy.py -q` green; full `pytest -q` green.
- `grep -rn "memory_registration" src tests web` is empty.
- `grep -n "resources/" src/core/editorfs/builtin_policy.py` shows no rule for it
  (and `logs/**` / `web/dist/**` / `**/.env**` still there, in the same order).
- A live check against **this checkout as the workspace** is allowed here (it is
  read-only): open the Editor, confirm the tree shows no `.nojae` entry at the
  root, and that `resources/memory_registration.json` — if present on this
  machine — is now an ordinary readable file (it is no longer locked, and no
  longer read by the app). Do **not** delete or edit it.
- Move this file to `todos/done/` when the above is true.
