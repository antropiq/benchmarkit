# ws-05 — Frontend: the dialog says which file it edits, and survives its two states

Make `SettingsDialog.vue` honest about the file behind the fields (D4's reboot line,
D8/D9's failed state), remove the `LLM_DEBUG` checkbox, and adapt the group-1 fields
to the native JSON types `GET /api/settings` now answers. Small in lines, and every
line is user-visible — do exactly these, nothing else.

Depends on: ws-04 (the GET's native types, the state block, `llmDebug` gone from the
API). **Lands immediately after ws-04** — until it does, the dialog's checkbox writes
an ignored field and its inputs stringify on save, which is inert, not broken.
Spec: **§5 "Frontend"** (both bullets), **D7** (the dialog keeps its field names —
do **not** rename `temp`/`apiKey` in the payload), **D8/D9** (where the note shows),
**D4** (the reboot line's exact job).

## Files

| file | change |
|---|---|
| `web/src/components/SettingsDialog.vue` | §1–§3 below |
| `web/src/composables/useBenchmark.js` | the settings state carries the workspace state block through (`state.workspaceSettings = body.workspaceSettings \|\| null`); the two `GET`/`POST /api/settings` call sites stay unchanged in shape |
| `web/tests/` (Vitest) / `web/e2e/` (Playwright) | only what actually exists: if a SettingsDialog/useBenchmark spec already exists, extend it; if none does, do not create one for CSS-level sentences — §4 decides per case |

## 1. The two sentences (spec §5, Frontend, first bullet)

- One line **naming the file it edits, with the workspace path** — the same dialog
  now means a different server per workspace, so the user must see *which* workspace
  this dialog is saving. The path comes from the `workspaceSettings.path` block on
  `GET /api/settings` (ws-04), never composed from a client-side guess. Put it where
  the dialog names what it edits (the header or the Connection tab's top, wherever
  the MCP/LSP tabs already name their file — read the dialog first and follow it).
- One line stating that **a hand-edit needs a server restart** (D4 — the dialog is
  live, a hand-edit is not; "so the reboot is expected, not discovered").

## 2. The failed / still-seed-text state (D8 + D9)

When `workspaceSettings.state` is `"failed"`, or `ok` but not configured (the D8
flag), show the note where the MCP `externalsError` note shows today — read
`useBenchmark.js:750`/`:793`/`:919` (`state.mcpExternalsError = body.externalsError …`)
and find the markup in the dialog that renders `mcpExternalsError`; the new note
takes the same slot and the same tone. The message is the server's sentence (it
already names the file and path); the client shows it, adds nothing, translates
nothing.

## 3. The field-level changes

- **Remove the `LLM_DEBUG` checkbox** (the input bound to `state.settings.llmDebug`,
  `SettingsDialog.vue:199-200` and its surrounding field markup). The key is
  hand-edited now; nothing in the dialog may mention it.
- **Native types in, native types out**: the GET now answers `supportsVision` as a
  **boolean** and the sampling fields as **numbers or null**, not `"1"`/`"0.2"`
  strings. The checkbox binding (`supportsVision === '1'` → `:checked` /
  `@change` writing `'1'`/`''`) becomes boolean `v-model` (a boolean survives the
  server's `_stringify_scalars` validator untouched, `routes.py:84-85`). The numeric
  inputs keep working through Vue's number coercion; a field the user cleared sends
  `null` — the mixin turns `null` into `""` and ws-04's translation omits the member
  from the file (the §2 "member absent means" contract), so the dialog must not send
  the literal string `"0.2"`-style content it no longer has, nor invent a sentinel.
  The two tasks meet exactly at this rule, and ws-04's `test_api.py` round-trip
  (POST with a cleared field → the member is absent in the file, GET answers `null`)
  is the referee.
- The LSP tab's toggles and `maxIterations`: **unchanged** (they already round-trip
  through `/api/lsp`, which ws-03 moved to the JSON file).
- The **payload field names stay** `apiKey`, `temp`, … (D7). If you find yourself
  renaming one, stop — that rename is out of scope (§8).

## 4. Tests

- Vitest (`cd web && npm test`): extend existing specs only where they already touch
  the settings payload or the dialog's notes. A new unit test earns its place **only**
  for the empty-field → `null` payload rule (§3's third bullet), if the payload
  builder is a testable function; prose lines are not tested in JS.
- The Playwright e2e for the settings dialog (if one exists — check
  `web/e2e`/`web/test` for a settings spec): its `llmDebug` interactions, if any,
  are removed with the field. **Do not write new e2e coverage** for this lot; the
  §10 manual check covers the dialog end to end.

## Acceptance

- `cd web && npm test` green; `cd web && npm run build` succeeds.
- `grep -n "llmDebug\|LLM_DEBUG" web/src` → **empty** (state, markup, comments).
- `git diff --stat web/` shows `SettingsDialog.vue` + `useBenchmark.js` (+ a test
  file if §4's rule earned one) and nothing else.
- Manual spot-check on a scratch root only (never this checkout's live state):
  boot with `BENCHMARKIT_PATH_LOCATION` at an empty temp dir → the dialog opens,
  names the file and its path, shows the not-configured note, and a filled + saved
  form replaces the note with nothing. Delete the scratch dir afterwards.
- Move this file to `todos/done/` when the above is true.
