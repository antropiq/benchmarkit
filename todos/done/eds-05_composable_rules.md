# eds-05 — `useAgentModes.js`: the dialog's rules as plain functions

**Spec:** D3 (the form rules), D5 (slug, new-vs-existing id), D7 (the conflict
state), D9 (the reset decision), §4 "Files touched" row 4.
**Depends on:** eds-03 (the two routes exist to be fetched and posted).
**Files it touches:** `web/src/composables/useAgentModes.js`,
`web/tests/agentModes.test.js`. No `.vue` file — the dialog itself is eds-06.

The file's own header already promises this task: *"Every decision lives in this
file as a plain function, so a Settings editor for modes can reuse the rules
instead of re-deriving them in a component."* Deliver on it: the rules are plain
exported functions that a hypothetical document can be judged by, so they stay
testable without a mounted app.

## What to deliver

**The two calls.** A document fetch (`GET /api/agent-modes/document`) and a whole-
document save (`POST /api/agent-modes/save`) that carries the `mtime` token the
read returned. The save response feeds the dropdown's singleton through the
existing `acceptPayload` path, so the list reloads **without a second
round-trip** (D9). Failure of either leaves the dropdown exactly as it was — the
same non-destructive rule `loadAgentModes` already follows.

**The form rules, as functions over a draft document:**

* `slugForModeId(label, takenIds)` — lowercase, then letters/digits/`_`/`-`,
  within the id length cap, made unique against the ids already in the document.
  The caps come from the payload, never a literal here.
* id validity against the payload's pattern and length cap; the id field is
  editable **only while the mode is new** — an existing mode's id is read-only
  (renaming an id is a delete plus an add and silently orphans every saved
  session that names it).
* persona byte count (UTF-8) against the payload's byte cap, and persona line
  count against the payload's line budget — the byte cap is **hard** (invalid,
  blocks Save), the line budget is **soft** (a warning, never blocks).
* the mode-count cap: the form stops being valid at the cap rather than the save
  being refused by the server.
* a label over the label cap is **not** a form error — the server trims it, as it
  already does.
* `blockingFindings(draft)` — which findings hold Save: every `error` kind,
  whether about an entry or the file as a whole. A `note` (a file that named a
  reserved id by hand) never holds Save.
* `saveDisabledFor(...)` — the one answer the dialog renders, so the rule is not
  re-derived in the component.
* dirty tracking: whether the draft differs from what was read (Cancel needs no
  confirm when it does not; Save needs nothing when it does not).
* the D9 reset decision: after a save, does the selection move to the new default?
  It moves — unless the current mode is **locked** (`state.locked`, the existing
  `agentModeLockedFor` rule) or a lot is running (the existing `modeSelectDisabled`
  / `pinnedMode` rule). When it is skipped, the quiet line under the select says
  why, reusing the notice slot the composer already renders.
* the conflict state: a 409 leaves the dialog open with the conflict named
  (reload-and-re-apply is out of scope for this lot — no merge, no overwrite).

## Rules this task must not break

* **`web/src` contains no reserved mode id.** The scan in
  `web/tests/agentModes.test.js` walks the whole tree and reads the id out of the
  Python module; the new code must pass it, not be exempted from it. The dialog
  learns "this entry is app-owned" only from server data.
* Do not hardcode a cap in the client: every number comes from the document
  payload's `caps`.
* The existing dropdown rules stay exactly as they are — `modeBlocksHub`,
  `modeHint`, `modeSelectTitle`, `modeSelectDisabled`, `setAgentMode`,
  `applySessionAgentMode`, the D7 lock. The composer's behaviour does not change
  in this task.
* `useAgentModes` must not import `useChat` (the cycle is documented in the file);
  the turn-count probe and the batch flag stay injected.
* The batch contract (`LOT_TASK_MODE_ID` in `useBatch.js`) is untouched: a lot
  pins its mode at launch and a save cannot move it.

## Tests to add (`web/tests/agentModes.test.js`)

* slug: from a plain label, from a label with punctuation, colliding with an
  existing id, over the length cap.
* the id of an existing mode is not editable while a new one's is (whatever shape
  that rule takes, it is a function's answer, not a template's).
* byte counter: ASCII and multi-byte text count bytes, not characters; over the
  cap is invalid; patching the cap in the payload changes the verdict.
* line budget: over it warns and does **not** block Save.
* `blockingFindings`: an entry error blocks, a file-level error blocks, a note
  does not.
* the mode cap blocks at the cap.
* a label over the label cap is not an error.
* dirty tracking: an untouched draft is clean, a touched one is dirty.
* the D9 reset decision: unlocked → moves to the new default; locked → does not
  move and says why; a running lot → does not move.
* a deleted-while-selected mode lands on the default.
* a save response reloads the singleton's list without a second fetch (assert the
  fetch count).
* a 409 leaves the dialog state open with the conflict named and the draft intact.

## How to tell it is done

`cd web && npm test` green (this file's suite plus the reserved-id scan), and the
reserved-id scan still passes with the new code in the tree. Move this file to
`todos/done/`.
