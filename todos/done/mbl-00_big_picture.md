# mbl — the WorkBench prompt helpers move into the workspace

The nine coloured bullets above the WorkBench textarea hold frequently-reused
prompts. Today they come from `resources/memory_registration.json`, an **app**
file: point the working location at another project and the bullets still offer
benchmarkit's own habits. This lot makes them **workspace-scoped** —
`<working location>/.nojae/prompt-helpers.json` — and renames the feature,
because nothing about it is "memory" in the model's sense.

The authority for this lot is **`docs/prompt-helpers-workspace-scope.md`**
(366 lines: decisions D1 … D8 with their rejected alternatives, a table of what
exists today with line pointers, §4's file list, §6's exact AGENTS.md wording,
and §7's behaviour-test table). Read it before your task — every task file below
cites the decision numbers it implements. This file is the map and the
cross-task contract; it does not restate that doc.

## The five behaviour changes (this is not a path rename)

1. **A reader never writes.** Today the first `GET` of a session *materialises*
   the file (D5, `config.py:444-449`). New: an absent file answers the nine
   defaults and leaves **no `.nojae` folder behind**. The app now points at
   projects someone only means to browse.
2. **A wrong slot count is an error, never repaired** (D4). 8 or 10 entries ⇒
   `PromptHelpersFileError` ⇒ **422** naming the path and the count, file
   untouched. The two tests that assert today's silent pad/truncate are
   replaced, not kept.
3. **The file becomes invisible in the Editor** (D7). It moves into `.nojae`,
   which `builtin_rules()` already answers `hidden` unconditionally in every
   workspace. The level-A `ro` rule is **deleted**, not moved: the old file was
   browsable and refused writes, the new one is not browsable at all.
4. **The bullets reload on a workspace switch** (D8). `setPathLocation()`
   reloads MCP, LSP and the agent modes but not the bullets, so today the
   previous workspace's prompts stay on screen until the user leaves the chat
   tab and returns.
5. **Names change all the way to the wire**: `/api/memory` →
   `/api/prompt-helpers`, payload key `registration` → `slots`,
   `useMemory.js` → `usePromptHelpers.js`, `MEMORY_*` → the new module's
   constants.

## What does NOT change

- **The app stays the file's only writer** (D1). It joins `.nojae` but not that
  folder's "edited only outside the app" rule: `POST /api/prompt-helpers` keeps
  writing it, because the bullet picker *is* the feature. What stays reserved is
  the folder, not the editing rights — which is why §6 reword AGENTS.md.
- Nine slots, the bullet row's shape, its markup and its CSS: no layout change,
  no tenth helper, no growable list (out of scope, §5).
- **Field coercion stays self-healing** (D4 last paragraph): a bad colour falls
  back to the default gray and *does* rewrite the file. That is the one
  remaining write on a read, and it is deliberate.
- Document `version: 1`, key `helpers`, nine empty gray slots as the seed
  (D4/§4). No real example prompts in the seed — shipping benchmarkit's habits
  into a foreign workspace is the bug this lot removes.
- `resources/` is untouched apart from losing one file, and **the legacy file is
  left in place** (D6): not deleted, not trashed, not `os.replace`d. Nothing
  reads that path again; the user removes it by hand.

## Decisions already taken (do not re-open them)

All eight are in the spec with their rejected alternatives. The four most likely
to be undone by accident:

- **D3**: the logic leaves `config.py` entirely and becomes
  `src/core/prompt_helpers.py`. Rejected: leaving it in `config.py` with a local
  import — `config.py` sits at the bottom of the dependency graph and a module
  that reads and seeds `.nojae` cannot live there without a cycle
  (`server.py:129-135` says this out loud).
- **D4**: a wrong count raises. Rejected: keeping the repair, and rejected:
  hard-failing the composer (the UI survives a 422 because the client's own
  state is nine slots before any fetch).
- **D6**: the migration is a **one-shot lift at the seeding moment only** —
  exclusive create decides *whether* this workspace is seeded; the read side
  never looks at the legacy path. A legacy list that does not hold exactly nine
  entries is **not** migrated (`"migration-skipped"` naming the count).
- **D8**: both route names change with the file name. `POST /api/prompt-helpers`
  stays browser-only exactly as today — the write-surface guard covers `/api/fs`
  markers only, so this rename neither breaks it nor needs it extended.

## Shared contract between tasks

| name | lands in | used by |
|---|---|---|
| `src/core/prompt_helpers.py` — `PROMPT_HELPERS_FILENAME`, `PROMPT_HELPERS_VERSION`, `SLOT_COUNT`, `DEFAULT_COLOR`, `CONTENT_MAX_CHARS`, `COLOR_RE`, `LEGACY_PROMPT_HELPERS_PATH`, `prompt_helpers_path()`, `prompt_helpers_file_path(root)`, `load_prompt_helpers()`, `save_prompt_helper()`, `default_prompt_helpers()`, `default_prompt_helpers_text()`, `_normalize_slot()`, `PromptHelpersFileError`, `ensure_prompt_helpers_file()`, `prompt_helpers_file_state()`, `_self_check()` | mbl-01 | mbl-02, mbl-03 |
| `.nojae` folder name is **not** re-declared: import `POLICY_DIRNAME` from `mcp_connectors/policy_shared` with the `sys.path` fallback, exactly as `core/agent_modes.py:48-63` does | mbl-01 | every task |
| the `{"slots": [...]}` payload and the `/api/prompt-helpers` route pair | mbl-03 | mbl-04, mbl-07 |
| `promptHelpers` block (`{state, path, error}`) on `GET`/`POST /api/path-location` | mbl-03 | mbl-05, mbl-07 |
| `describePromptHelpers` in `web/src/lib/path-location-policy.js` — six states: `created`/`existing`/`absent`/`failed` + `migrated`/`migration-skipped`; `warn` tone only for the last two | mbl-05 | mbl-07 |
| `usePromptHelpers()` exposing `loadPromptHelpers()` | mbl-04 | mbl-05 |

Tests for the new module patch `BENCHMARKIT_PATH_LOCATION` per test (the way
`tests/test_agent_modes.py:127-134` does it), **not** `config.RESOURCES_DIR`.
`prompt_helpers.PROMPT_HELPERS_FILENAME` and
`prompt_helpers.LEGACY_PROMPT_HELPERS_PATH` are module attributes precisely so a
test can move the lookup — the repo's `mcphub.constants` convention (D2).

## Task order

| # | task | spec ref | depends on |
|---|---|---|---|
| mbl-01 | `src/core/prompt_helpers.py`: the module and its unit tests, no callers | D2, D3, D4 | — |
| mbl-02 | Seeding and the one-shot lift: `ensure_*`, `*_file_state`, `default_*_text`, `LEGACY_…`, `_seed_prompt_helpers()` in `server.py` | D5, D6 | mbl-01 |
| mbl-03 | Backend cutover: routes renamed, `{"slots": …}`, 422 mapping, `promptHelpers` block, **delete the `config.py` block** | D3, D4, D8 | mbl-01, mbl-02 |
| mbl-04 | SPA cutover: `useMemory.js` → `usePromptHelpers.js`, the four component imports | D8 | mbl-03 |
| mbl-05 | The switch reload: `describePromptHelpers`, `setPathLocation()`, the third settings note | D6, D8 | mbl-03, mbl-04 |
| mbl-06 | Level A: delete the `ro` rule, update the policy tests, question-set guard | D7, D3 | mbl-01, mbl-03 |
| mbl-07 | e2e (extend `03-workspace-switch.spec.js`), the repo-wide name sweep, AGENTS.md | §6, §7 | mbl-01 … mbl-06 |

**mbl-03 and mbl-04 are deliberately split at the wire boundary.** Between
them, the app boots and every other tab works, but the bullet row fetches a
404 and shows nine empty slots (the composable's pre-fetch fallback). That
interim state is accepted, so each task stays one session. mbl-04 must land
immediately after mbl-03.

**mbl-07 is last**: it asserts sentences that only exist once the rest has
landed, and it edits AGENTS.md, which is injected into every WorkBench prompt —
wrong there is worse than missing.

## Out of scope for this lot (do not add it)

- Per-mode helpers; a tenth or fewer helper; editing `.nojae` from the Editor;
  sharing/importing/exporting helpers; anything else in `resources/` (§5).
- **Deleting `resources/memory_registration.json`** from any machine, by code or
  by hand, and any `.nojae/trash` involvement (D6).
- **`.gitignore`**: line 29 (`resources/memory_registration.json`) is handled by
  the human out of band. **No task touches `.gitignore`.** The line stays
  harmless — `.nojae/` is already ignored at line 27.
- **The README subsection** describing the nine bullets beside "Agent modes":
  the spec calls it not required (§4, README row). Skip it; mbl-07 edits only
  the one paragraph that names the folder's contents.
- **The manual pass** (spec §7, last paragraph — boot against this machine's
  five real slots, confirm they survived and no `.nojae` entry shows in the
  tree). It is run by hand after the lot, on the developer's machine. Do not
  write or perform a step that boots the app against this checkout's live
  `.env`/`resources` state, and do not make it a test: the runner must not
  mutate live user data.
- Any CSS, markup or layout change to the bullet row, and any rename of the
  words "bullet" in user-visible text (§3 D3: "the UI keeps saying *bullet*").

## Project rules that bind these tasks (from `AGENTS.md`)

- **Python style**: explicit type annotations, `typing` imports, no bare
  containers.
- **Tests stay hermetic**: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1`; patchable
  file locations live on the module's own attributes (`editorfs.constants`,
  `mcphub.constants`, `agent_modes.AGENT_MODES_FILENAME`) and must be patched
  through those modules; a backend test's workspace is a throw-away `tmp_path`
  pointed at by `BENCHMARKIT_PATH_LOCATION`, never this checkout.
- **`BENCHMARKIT_PATH_LOCATION` is deliberately not a managed `.env` key**, so a
  settings save can neither erase it nor resurrect a stale value. Do not add it
  to `SETTING_KEYS`.
- **Every path the app holds is relative to the working location**, "so a switch
  invalidates them all at once" — mbl-05 is the bullet half of that sentence.
- **`.nojae` is reserved by code**: hidden from the tree, the read and git
  status; its files are seeded by exclusive create and never clobbered, and a
  seeding failure is a note, never a refused switch (mbl-02 must obey).
- **Path rules come from two levels, one verdict**: the strictest match wins, B
  tightens but never loosens A. A hidden path answers 404 to a read and 422
  naming the rule to a write. mbl-06 deletes an A rule; it adds no verdict and
  no second predicate — `mode_for` stays the only place a verdict is computed.
- **A `/api/fs` write route is reachable from the SPA only** —
  `tests/test_write_surface_isolation.py` scans `src/core`, `mcp_connectors`,
  `lsp_connectors` and `useChat.js` for the markers `PUT /api/fs`,
  `POST /api/fs`, `DELETE /api/fs`. `src/core/prompt_helpers.py` is a new file
  under `src/core`, so keep those three strings out of it (a docstring naming
  the route as `POST /api/prompt-helpers` is fine).
- No lint step. `pytest` for Python, `cd web && npm test` for Vitest,
  `cd web && npm run test:e2e` for Playwright, `cd web && npm run build` for
  `web/dist`.
