# eds-07 — The pencil button, the mount, the card styles

**Spec:** D10, §4 "Files touched" rows 6 and 7.
**Depends on:** eds-06 (the component to mount).
**Files it touches:** `web/src/components/WorkBenchInput.vue`,
`web/src/styles/theme.css`, and the frontend test that covers the composer's mode
column (`web/tests/agentModes.test.js` or a new `web/tests/agentModesPencil.test.js`).

## What to deliver

* A pencil button in `.chat-mode-col`, **immediately after the `<select>`**, using
  the pencil glyph the app already draws (the "Edit and resend" chip in
  `WorkBenchView.vue`) — same icon, not a new one. It gets an `aria-label` and a
  `title` naming what it opens.
* It opens `AgentModesDialog`, mounted in the composer the way the other dialogs
  are mounted where they are used.
* Enablement follows the dropdown's window exactly: **inert while a chat reply
  streams** (the same window in which the Editor's writes are refused — opening an
  editor whose Save will be refused is worse than not opening it) and **inert
  while a lot runs** (a lot runs on the modes the file had when it launched).
  **Available when the dropdown's list failed to load** — the dialog reads the
  file, not the dropdown's verdict, and a broken file is precisely when it is most
  useful. Note that today the whole `.chat-mode-col` renders only when
  `agentModes.loaded || agentModes.notice`, so check that the pencil is reachable
  in the failed-fetch case the existing e2e spec pins ("a failing
  `/api/agent-modes` leaves the composer as it was") — if the column stays
  unrendered there, the pencil cannot be available, and the render condition is
  part of what this task settles.
* Styles: `.agent-modes-dialog > article` in the existing card family (the
  Settings/batch/memory pattern — size the `> article`, never the `<dialog>`
  element), and the nav reuses the settings two-column rules rather than
  duplicating them. Fixed card size so switching tabs never resizes the dialog.
* The D9 quiet line: when a save's reset is skipped (locked mode, running lot),
  the line under the select says why, using the notice slot the column already
  renders.

## Rules this task must not break

* The composer's existing layout promise, which an e2e spec pins: the mode select
  sits in the **left** action group, keeps the icon-button height, stays left of
  the memory bullets, and the bullets stay centered. The pencil joins the same
  column without moving anything.
* The dropdown's own behaviour is untouched: `modeSelectDisabled`,
  `modeOptionDisabled`, `modeSelectTitle`, `setAgentMode` and the two quiet lines
  (a session's missing mode, the file's findings) keep working exactly as they do
  now. Leaving a locked mode is still refused.
* The select stays inert while a lot runs, and the pencil joins that same rule —
  do not make one live and the other not.
* No new CSS that duplicates the settings nav rules; extend the existing family.
* No reserved mode id anywhere in `web/src` (the scan covers `.css` too).

## Tests to add

* The pencil renders beside the select and opens the dialog (a mounted-composer
  test with the document route stubbed).
* It is disabled while a reply streams and while a lot runs, and enabled in both
  cases otherwise.
* It is reachable when the dropdown's list failed to load.
* Opening it makes no save call.
* The layout assertion the e2e spec already makes still holds (run the e2e mode
  spec, or at minimum keep the select's height/order assertions true).

## How to tell it is done

`cd web && npm test` green and `cd web && npm run build` clean. Move this file to
`todos/done/`.
