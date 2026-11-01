# mbl-04 — SPA cutover: `useMemory.js` becomes `usePromptHelpers.js`

Rename the composable and point it at the renamed route and payload key, so the
bullet row works against the workspace store again. **Do this immediately after
mbl-03**: between those two tasks the row shows nine empty bullets (its own
pre-fetch fallback against a 404), which is the accepted interim state.

Depends on: mbl-03 (the route and the `slots` payload must exist).
Spec: **D8** (the composable rename, the state key, the mirrored constants) and
§4's two frontend rows. D3's last paragraph: *"Not renamed: the payload key stays
a list of slots, and the UI keeps saying 'bullet'."*

## Files

| file | change |
|---|---|
| `web/src/composables/useMemory.js` | **renamed** → `web/src/composables/usePromptHelpers.js` (git mv, so history follows) |
| `web/src/components/MemoryBullets.vue` | import + `state.registration` → `state.slots` + constant names |
| `web/src/components/RegistrationMemoryDlg.vue` | same |
| `web/src/components/WorkBenchView.vue` | `useMemory`/`loadMemory` → `usePromptHelpers`/`loadPromptHelpers` (`:8`, `:20`, `:36`) |
| `web/src/components/WorkBenchInput.vue` | component import names only (`:4-5`) — the tags at `:467` / `:563` change **only if** you rename the components (see §3: you probably should not) |

No CSS, no markup structure, no new behaviour (the switch reload is mbl-05).

## 1. The composable

- `export const MEMORY_SLOT_COUNT` → `PROMPT_HELPER_SLOT_COUNT`,
  `MEMORY_DEFAULT_COLOR` → `PROMPT_HELPER_DEFAULT_COLOR`,
  `MEMORY_CONTENT_MAX_CHARS` → `PROMPT_HELPER_CONTENT_MAX_CHARS`, with the
  "must mirror the backend" comment now pointing at
  `src/core/prompt_helpers.py` (it pointed at `src/config.py`, which no longer
  holds them — that stale pointer is exactly what this task fixes).
  These three JS constants are a **hand mirror** of the Python ones; keep them in
  sync by value (`9`, `#808080`, `32000`) and keep the comment that says so.
- `export function useMemory()` → `usePromptHelpers()`; `loadMemory()` →
  `loadPromptHelpers()`; `saveMemory()` → `savePromptHelper()`.
- state key `registration` → `slots`. The initial value stays
  `Array.from({ length: PROMPT_HELPER_SLOT_COUNT }, emptySlot)` — that
  nine-before-the-first-fetch is what keeps the row usable when the load fails
  (D4's UI argument), so do not "simplify" it to `[]`.
- fetch URLs `/api/memory` → `/api/prompt-helpers`; response reads
  `body.registration` → `body.slots`; the POST body shape (`index`, `color`,
  `content`) is unchanged.
- Error handling is unchanged in behaviour: a failed load sets `state.error`, and
  the row still renders nine slots. Verify that after the rename by reading
  `describeError`'s use — the 422 from mbl-03 arrives as a `detail` **string**,
  which the existing code already handles.

## 2. The four consumers

Update imports and the `state.registration` reads. In `MemoryBullets.vue` the
per-index fallback at `:16` (`?? { color: …, content: '' }`) must stay — it is
the other half of "a failed load costs a red note, not a broken row".

`WorkBenchView.vue:26-39`: the comment naming "the memory bullets" and the
`loadMemory()` call become the new names; the lazy-load-on-tab-selection
behaviour itself is unchanged (mbl-05 adds the *second* load, on a switch).

## 3. What you must NOT rename

- **CSS class names**: `.memory-bullets`, `.memory-bullet`, `.memory-dialog`,
  `.memory-color` are referenced by `web/src/styles/theme.css` (`:1608-1740`, and
  the specificity note at `:1677-1684`) and by
  `web/e2e/04-agent-modes.spec.js:180`. Renaming them means touching CSS and an
  unrelated e2e spec for no functional gain — out of scope for this lot. Keep the
  selectors; leave the CSS comment that explains the specificity trap alone.
- **User-visible words**: the UI says "bullet". Not "helper", not "prompt
  helper". The spec is explicit (D3).
- The **component file names** (`MemoryBullets.vue`, `RegistrationMemoryDlg.vue`)
  may stay as they are — this task's budget is the composable and the wire. If
  you rename them, then `WorkBenchInput.vue`'s imports and tags change too, the
  three cross-referencing comments in `BatchModeDialog.vue:8`,
  `EditorTreeDialogs.vue:10` and `SearchDialog.vue:7` name
  `RegistrationMemoryDlg.vue`, and those comments must follow. **Prefer not to**;
  if you do, that sweep is part of this task's acceptance, not mbl-07's.
- Comments in `useChat.js:96,104` and `useSplitter.js:7` that describe the
  feature as "memory bullets": reword to "prompt helper bullets" **only** where
  you are already in the file; do not open those files for a comment.

## Tests

No Vitest spec asserts the payload key or the bullet count today
(§7: *"Frontend: no Playwright spec asserts the bullet count or the payload
key"*). So:

- run `cd web && npm test` — if a spec breaks, it referenced a renamed symbol;
  fix the reference, not the assertion;
- **add one small Vitest case** in a new `web/tests/promptHelpers.test.js`:
  stub `fetch` for `GET /api/prompt-helpers` returning
  `{slots: [{color:'#2ec27e',content:'a'}, …nine…]}`, call
  `loadPromptHelpers()`, and assert `state.slots[0].content === 'a'` and
  `state.slots.length === 9`. Then a second case where the fetch answers
  `{detail: 'the file holds 8 entries'}` with `ok: false`, and assert the state
  keeps nine slots and `state.error` carries the sentence — that pins D4's "the
  UI keeps working" claim, which until now had no test.

## Acceptance

- `cd web && npm test` green, including the new spec; `cd web && npm run build`
  succeeds.
- `grep -rn "useMemory\|loadMemory\|memory.registration\|/api/memory" web/src`
  is empty.
- `grep -rn "state.slots" web/src` shows the composable and both components.
- `.memory-bullets` still appears in `theme.css` and in `MemoryBullets.vue`
  (proving the CSS was left alone).
- Live check against a scratch root (not this checkout): with mbl-03's server
  running, open WorkBench, edit bullet 3 in the dialog, reload the page, and see
  slot 3 back — and confirm `<scratch>/.nojae/prompt-helpers.json` holds it.
- Move this file to `todos/done/` when the above is true.
