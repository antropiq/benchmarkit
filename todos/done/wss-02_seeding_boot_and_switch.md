# ws-02 — Seeding at the two moments: boot and the working-location switch

Wire the file ws-01 built into the app's seeding sequence: a `_seed_workspace_settings()`
at boot and the matching `ensure_workspace_settings_file(expanded)` + cache
invalidation on `POST /api/path-location` — **before** the hub restarts (D10),
which is the one ordering rule in this lot that a later task cannot repair.

Depends on: ws-01 (the module and the committed template).
Spec: **D10** (the seed goes before the hub restarts, and why), **D8** (the seeded
file is reported "not configured"), **D11** (the seed's only input is the template),
**§5** (`routes.py` row: "`post_path_location()` seeds before the restarts (D10) and
invalidates the cache"), **§10** (`tests/test_workspace_settings_seed.py`).

## Files

| file | change |
|---|---|
| `src/api/server.py` | a `_seed_workspace_settings()` beside the other five `_seed_*` functions, called at boot in the order §"Where it goes" states |
| `src/api/routes.py` | inside `post_path_location`: `workspace_settings.ensure_workspace_settings_file(expanded)` **before** the two hub restarts, `invalidate_workspace_settings()`, and the block added to the returned payload |
| `tests/test_workspace_settings_seed.py` | **new** |

No `src/config.py` edit, no `models.py` edit, no frontend edit. The settings routes
are untouched — ws-04 owns them.

## Where it goes (this is the whole task, read it twice)

*Boot* (`create_app()`, after `load_dotenv_once()` / `apply_llm_debug()` /
`setup_path_location()`): call it **after `_seed_workspace_policy()`** — that one
creates the reserved folder — and **before the lifespan's first LSP scan**. Read the
comments above `_seed_agent_modes()` and `_seed_mcp_settings()`
(`server.py:221-252`) and write yours in the same voice: same reserved folder,
failure is a note in the log, never a boot failure, and *why* the ordering is what
it is (the MCP seed's comment is the nearest precedent — it too must land before the
lifespan scans what it seeds).

*Switch* (`post_path_location`): put the call where the MCP seed already is
(`ensure_mcp_settings_file(expanded)`, immediately before the `try:` that restarts
the hubs) and say in the comment that it follows the same exception to the "all seeds
after the restarts" rule for the same reason: `lspEnabled` is read **during**
`_restart_lsp_hub()` (`_active_server_ids()` → `config.lsp_enabled_ids()`), so
seeding after would leave the new workspace on **zero** language servers until the
next restart (D10). It is safe to break the rule because `ensure_workspace_settings_file`
answers a dict and raises nothing, so a hiccup cannot be mistaken for a hub failure
and cannot trip the rollback. Also call `invalidate_workspace_settings()` there, so
the **new** workspace's file answers afterwards, not the previous one's (D4) — the
comment beside `agent_modes.invalidate_agent_modes_cache()` is where that already
lives.

Add the block to the POST's returned payload (`"workspaceSettings": {...}`) with the
same `{state, path, error}` shape as its `mcpSettings` / `agentModes` /
`promptHelpers` neighbours, and to `GET /api/path-location`'s payload too, so the app
bar can render a seed note the same way it renders the other three. A seed failure is
a **note**, never a 4xx: the response is still 200.

## Tests (`tests/test_workspace_settings_seed.py`)

Hermetic style, as the folder's other seed tests do it
(`tests/test_prompt_helpers_seed.py`, `tests/test_lsp_settings_seed.py` are the two
to read — ~220 lines each, not the whole suite):
`monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))`, paths built from
`workspace_settings.POLICY_DIRNAME` / `WORKSPACE_SETTINGS_FILENAME`.

Cover:

1. **a new location gets the file with the template's bytes** — boot an app (or call
   the seed function directly, whichever the two files above do) with
   `BENCHMARKIT_PATH_LOCATION` at an empty `tmp_path`, then assert the bytes equal
   `WORKSPACE_SETTINGS_TEMPLATE`'s bytes.
2. **an existing file is never overwritten** — write a marker into it first, boot,
   the bytes are still the marker.
3. **the seed is a note, never a failure** — patch
   `workspace_settings.WORKSPACE_SETTINGS_TEMPLATE` to a nonexistent path (or chmod
   the root read-only) and assert the app still boots / `POST /api/path-location`
   still answers **200** with `workspaceSettings.state == "failed"` and a non-empty
   `error`.
4. **the seed lands before `_restart_lsp_hub()`** (D10) — do **not** spin up a real
   language server. Instead patch the restart seam the switch calls (the module-level
   helper `routes` reaches, or `request.app.state.lsp.restart`) with a recorder that
   asserts `.nojae/nojae-settings.json` **already exists** at the moment it is called,
   and that the file it sees is the *new* root's. That is what "has `pyright` running
   immediately" reduces to: the value was on disk before the hub read it. Say in a
   comment that the end-to-end form of this is the lot's manual check.
5. **a seeded file whose `serverUrl` is still the instruction text reports
   "not configured" naming the path** (D8) — through `workspace_settings_state()`
   (or `load_workspace_settings()`'s configured flag), asserting the message names
   the file's path. Do not assert about a transport call.

Do **not** write the "a chat / benchmark request refuses on a malformed file" case
(D9) here: nothing reads the file yet, so it could only pass by faking the very thing
ws-03's read cutover builds. That case is written in **ws-06** (the verification pass
adds it to this file, once the chat and benchmark paths really consult the file).

## Acceptance

- `pytest tests/test_workspace_settings_seed.py tests/test_workspace_settings.py -q`
  green.
- Full `pytest -q` **unchanged**: every existing test still passes, because the file
  is seeded but read by nobody. That is the point of the split — say so in the commit.
- `grep -n "workspace_settings" src/api/server.py src/api/routes.py` shows the boot
  seed, the switch seed + invalidation, and both payload blocks — and nothing else.
- A switch to a fresh `tmp_path` in a test leaves a `.nojae/nojae-settings.json`
  **before** the restart recorder fires, and `GET /api/settings` is **not** changed
  by this task (`git diff` proves it).
- Move this file to `todos/done/` when the above is true.
