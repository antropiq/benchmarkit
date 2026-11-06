# ws-06 — Verification pass: the sweep, the static scans, one e2e, the human list

Nothing here changes behaviour. It proves the §10 lists, fixes what a failure sends
you back to, and writes down the one check no task may automate.

Depends on: ws-01 … ws-05 — the whole implementation. Run it only when those are in
`todos/done/`.
Spec: **§10 in full** (the suites that must pass unchanged, the D10 end-to-end case,
the manual check) and **D10**/**D4**/**D8**/**D9** as the invariants the scans probe.

## 1. The full suite, in one place

```
pytest
cd web && npm test
cd web && npm run build
```

Green. A failure names its owning task: 01 = the module's read/write rules, 02 = the
seed and its ordering, 03 = the reads and the hermeticity guard, 04 = the writes and
the `.env` trio, 05 = the dialog. Fix it in the spirit of that task's file, re-run the
affected files, then the whole again. Do not weaken an assertion to make a sweep pass:
if a §10 "must pass unchanged" suite fails, the lot broke an invariant and the fix is
on the source side, never in that suite.

## 2. The §10 "unchanged" list, run and recorded

`pytest tests/test_prompt_helpers.py tests/test_prompt_helpers_api.py
tests/test_prompt_helpers_seed.py tests/test_agent_modes.py tests/test_agent_modes_seed.py
tests/test_mcp_settings_seed.py tests/test_lsp_settings_seed.py
tests/test_editorfs.py tests/test_editorfs_write.py tests/test_editorfs_policy_seed.py
tests/test_editorfs_builtin_policy.py tests/test_editorfs_policy_file.py
tests/test_editorfs_policy_modes.py tests/test_editorfs_git.py
tests/test_editorfs_search.py tests/test_write_surface_isolation.py
tests/test_llmkit_contract.py -q`

All green **and byte-identical** — `git log --oneline` / `git diff` against the lot's
base must show no edit in that list. `test_write_surface_isolation` is the one that
matters most here (the new writer is app-internal and reachable from no connector,
tool or chat path) and `test_llmkit_contract` proves sampling still goes out verbatim
through the new builder's output. Record the counts in the done-file.

## 3. The static scans (run each, record the output)

* `grep -rn "MAX_MCP_ITERATION" src tests` → empty (the fallback is gone, §5).
* `grep -rn "llmDebug\|LLM_DEBUG" web/src` → empty; `grep -n "llmDebug" src/api/routes.py` → empty;
  `grep -n "LLM_DEBUG" src/config.py src/api/server.py` → still there (group 3 keeps it).
* `grep -rn "SERVER_URL\|LLM_SYSPROMPT\|SUPPORTS_VISION\|TEMPERATURE\|LSP_ENABLED\|MAX_TOOL_ITERATION" src/config.py .env.example`
  → only the pointer sentences naming `.nojae/nojae-settings.json`, no key.
* `grep -rn "from_env(" src` → the group-2/3 caller(s) only; `from_workspace(` used by
  `load_endpoints_config()` and `routes._load_config()` — the spec §5 "both callers move"
  check.
* `grep -rn "nojae-settings.json" src README.md AGENTS.md` → the constant, the docs, and
  no hand-built path anywhere else (one builder of the path, `workspace_settings_file_path`).

If a scan is not clean, that is a task's residue: fix it where it lives and say which.

## 4. The two cases the earlier tasks could not write

1. **The D10 end-to-end shape** (spec §10's first bullet, the half ws-02 stated as a
   recorder): in `tests/test_workspace_settings_seed.py`, a switch to a workspace whose
   `lsp-settings.json` declares `pyright` with `enable: true` **and** whose
   `nojae-settings.json` lists `pyright` in `lspEnabled`, with the LSP hub replaced by a
   fake manager that records `start_server` calls — assert the fake was asked to start
   `pyright` on the switch, with **no restart in between**. `BENCHMARKIT_DISABLE_LSP=1`
   is what the suite already uses to keep real servers out; the fake stands in for the
   hub the same way the existing LSP tests' fakes do — read `tests/test_lsp.py`'s fake
   manager (`grep -n "class .*Manager\|start_server" tests/test_lsp.py` finds it) and
   reuse the pattern rather than inventing one.
2. **A chat and a benchmark request refuse on a malformed file** (D9, spec §10's last
   seed bullet, deferred by ws-02): write invalid JSON into the workspace file, then
   assert `POST /api/chat` (or the seam the chat route uses to load its config) and the
   benchmark-start route answer with the server's sentence naming the path — not a 200
   with a guessed model, not a transport error quoting a template sentence.

## 5. One e2e for the headline behaviour

`web/e2e/03-workspace-switch.spec.js` already owns the scratch-root machinery
(`create()`, `setRoot()`, `removeScratch()`, the boot-root restore). Add **one** test
beside the existing ones, reusing them as-is, no LLM call:

1. open Settings at the boot root, read the file path the dialog names (§ ws-05) and
   the `serverUrl` field;
2. `setRoot(page, <scratch>)`, reopen Settings — the path names the **scratch** root's
   `.nojae/nojae-settings.json` and the server field is that workspace's own (a scratch
   root is freshly seeded, so it is the template's instruction text and the
   not-configured note — assert the note too);
3. `setRoot(page, bootRoot)` back, assert the original values came back.

That is the manual check's first half made permanent, and the regression the lot would
otherwise leave unpinned. `cd web && npm run test:e2e` needs an endpoint in `.env.test`;
if this machine has none, say so in the done-file rather than skipping silently. Leave
`removeScratch()` exactly as it is — the seeded `.nojae` subtree is why it uses `node:fs`.

## 6. The human-only step, written down

Create **`docs/nojae-settings-workspace-smoke-checklist.md`** with spec §10's last
paragraph as its body — the five observations in the spec's own order (workspace A:
set a model, chat; switch to B in the app bar; the dialog shows B's path and B's server;
B's enabled LSP is live without a reboot; hand-edit A's file, restart, the new value
answers), each with its expected observation. Head it with the rule that binds it:
**run against this machine's own servers; a scratch workspace, never a project that
matters** — and a line saying the hand-edit step is the only one that needs a restart,
by design (D4). **No task in this lot performs the list**; like the llmh lot's
checklist, it exists so the human step is not invented later.

## Files it touches

| file | change |
|---|---|
| `tests/test_workspace_settings_seed.py` | §4's two cases |
| `web/e2e/03-workspace-switch.spec.js` | §5's one test |
| `docs/nojae-settings-workspace-smoke-checklist.md` | **new** |
| whatever a failure sends you back to | named in the done-note, with its owning task |

## Acceptance

- The §1 command trio and the §2 list green; the §3 scans recorded with their output.
- `git diff --stat` for the §2 list is empty.
- The done-note records: the counts, each scan's verdict, any residue found and where it
  was fixed, and the fact that the smoke checklist was **written, not run**.
- Move this file to `todos/done/` when the above is true.
