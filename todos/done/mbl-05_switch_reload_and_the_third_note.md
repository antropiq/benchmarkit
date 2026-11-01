# mbl-05 — The switch reload: new workspace, new bullets, and a third settings note

Close the last behaviour gap: a working-location switch now reloads the bullet
row, and the settings dialog reports the third reserved-folder file the way it
already reports the other two. Without this task the switch leaves the previous
workspace's prompts on screen until the user leaves the chat tab and comes back.

Depends on: mbl-03 (the `promptHelpers` block on both endpoints), mbl-04
(`usePromptHelpers()` / `loadPromptHelpers()` exist to call).
Spec: **D8** (the added call in `setPathLocation()`, the two note fields) and
**D6** (the two extra states `migrated` / `migration-skipped`, and the describer
pattern in `lib/path-location-policy.js`). §7's *switch payload* row and the
`describePromptHelpers` Vitest case.

## Files

| file | change |
|---|---|
| `web/src/lib/path-location-policy.js` | third describer `describePromptHelpers(block)`, six states |
| `web/src/composables/useBenchmark.js` | `pathLocationPromptHelpers` + `pathLocationPromptHelpersNote` state (`:95-96` neighbourhood), their resets (`:505-506`, `:849-850`), the GET read (`:829-833` neighbourhood), the POST read (`:874-886` neighbourhood) **plus** `loadPromptHelpers()` inside `setPathLocation()` (`:843-935`) |
| `web/src/components/SettingsDialog.vue` | the third note paragraph + its sentence computed for the plain (non-warn) case — mirror the two `agentModes` sites (`:4`, `:53`, `:62`, `:232-237`) |
| `web/tests/pathLocationPolicy.test.js` | the describer cases + the two new state fields in `reset()`/`stub()` |
| `web/tests/promptHelpers.test.js` | mbl-04's file: the reload-on-switch case |

No backend change. No CSS.

## 1. `describePromptHelpers` (D6)

Same file, same shape as `describePolicy` and `describeAgentModes`: takes the
`{state, path, error}` block, returns `{tone, text}` or `null` for a
missing/unknown state. Six states — the four shared ones plus two of its own:

| state | tone | what the sentence must say |
|---|---|---|
| `created` | plain | the workspace got its own prompt-helper file at `path`; the bullets above the WorkBench textarea are now this project's |
| `existing` | plain | this project has its own prompt-helper file: `path` |
| `absent` | plain | this project has no prompt-helper file yet, so the bullets are empty until one is seeded or saved |
| `failed` | **warn** | the *`Working location is set, but …`* preamble again — a complete sentence, no dependence on the note above it (the file at `:48-57` explains why: the two files fail independently) |
| `migrated` | plain | benchmarkit **moved this machine's older bullet set into** `path` — the five (or nine) prompts that used to be app-wide are now this workspace's. Say where they came from, because a user who sees their own prompts appear in a new project wants to know why |
| `migration-skipped` | **warn** | the older file at `path`/`error`'s named legacy path could not be moved (its entry count is wrong), so this workspace starts with empty bullets and the old file was left where it is |

`migrated` is plain and `migration-skipped` is warn because the first is good
news and the second is a fact the user has to act on (fix the old file, or
re-do the switch after the seed ran). D6's own rule: *"`warn` tone only for
`failed` and `migration-skipped`."*

Update the file's header comment: it says *"the two files in the workspace's
reserved folder"*; there are now three, and they no longer share the
"edited outside the app" claim — this one is the app's own store (D1). That
sentence is the doc of this module, so fix it here rather than in mbl-07.

## 2. `setPathLocation()` (D8)

After the `agentModes` handling (`:878-886`), add the block read and the note:

- `state.pathLocationPromptHelpers = body.promptHelpers || null`;
- `described = describePromptHelpers(body.promptHelpers)`;
- `state.pathLocationPromptHelpersNote = described && described.tone === 'warn' ? described.text : null`
  — the same one-line rule as its two neighbours, and **not** into
  `pathLocationError` (the location moved; a red line there would claim it did
  not).
- Then the reload itself: `await loadPromptHelpers()` (or the un-awaited call, to
  match how the neighbouring MCP/LSP reloads are written — read `:894-924` and
  match them). Place it beside the other reloads, not inside the note block, and
  comment it as D8 states the reason: the row is the only workspace state that
  used to survive a switch by accident.
- Reset both new fields wherever `pathLocationPolicyNote` /
  `pathLocationAgentModesNote` are reset (`:505-506` and `:849-850`), and read the
  block on the **GET** path too (`:829-833`) so a dialog opened without a switch
  still reports the file's state (`absent` is a normal answer).

Import `describePromptHelpers` alongside the other two at `:3`, and
`usePromptHelpers` from the composable. `useBenchmark.js` importing a second
composable is fine — it already reaches for the tree and the hubs; do not thread
the call through a component.

## 3. `SettingsDialog.vue`

Three edits, each a copy of the `agentModes` equivalent:

- import `describePromptHelpers`;
- a computed `promptHelpersSentence` beside `agentModesSentence` (`:53`);
- the `v-if="state.pathLocationPromptHelpers && !state.pathLocationPromptHelpersNote"`
  paragraph after the agent-modes one (`:232-237`), reusing the existing
  `path-location-policy-note` class (no new CSS);
- check `:62`'s `[state.pathLocationPolicyNote, state.pathLocationAgentModesNote]`
  aggregation — add the third note where it belongs so `seedFailureNote` renders
  all warnings.

## Tests

`web/tests/pathLocationPolicy.test.js` (read `:27-67` first — `stub()` builds the
POST and GET payloads, and it needs `promptHelpers:` in both; `reset()` needs the
two new fields):

1. `created` / `existing` / `absent` → no warn note, and the plain sentence
   renders (the pattern at `:151-152`, `:197-198`);
2. `failed` → `pathLocationPromptHelpersNote` names the prompt-helper file, the
   path and the error, and `pathLocationError` stays `null` (`:171-173`);
3. `migrated` → no warn note (it is good news) and the sentence mentions the move;
4. `migration-skipped` → a warn note naming the count and the legacy path;
5. the three notes are mutually independent: a `failed` policy with an
   `existing` helpers block produces one warn line, not two (`:187` is the
   sibling assertion);
6. `describePromptHelpers(null)` and `describePromptHelpers({state:'nonsense'})`
   → `null`, and every state in `POLICY_STATES` plus the two new ones returns a
   non-empty `text` with a `tone` (the loop at `:206`).

`web/tests/promptHelpers.test.js` — a case that the switch reloads: stub
`POST /api/path-location` (with a `promptHelpers` block), a first
`GET /api/prompt-helpers` answering slot 0 = `"old workspace"`, then after
`setPathLocation()` a second `GET` answering slot 0 = `"new workspace"`, and
assert `state.slots[0].content === 'new workspace'`. That is the regression this
task exists for; without it, nothing on the client side pins D8.

## Acceptance

- `cd web && npm test` green, including both new/extended spec files.
- `cd web && npm run build` succeeds.
- `grep -n "promptHelpers" web/src/composables/useBenchmark.js web/src/components/SettingsDialog.vue web/src/lib/path-location-policy.js` shows the state fields, the note paragraph and the describer.
- Live check against two scratch roots (not this checkout): set bullet 1 in root
  A, switch to root B from the settings dialog, **stay on the WorkBench tab**, and
  see bullet 1 empty without leaving the tab; switch back and see it again. The
  dialog shows a `created`/`existing` sentence for the helper file beside the
  other two.
- `git diff --stat` touches no file under `src/`.
- Move this file to `todos/done/` when the above is true.
