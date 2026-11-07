# eds-08 — The e2e spec: open, create, save, reload, and the disk agrees

**Spec:** §4 "Files touched" last row, acceptance items 1, 3, 4 and 9.
**Depends on:** eds-07 (the button exists to click).
**Files it touches:** `web/e2e/04-agent-modes.spec.js` (a new `test.describe`
block in the existing file). No production code — if the spec cannot pass, that is
a bug to report, not to design around.

## What to deliver

A spec on a **scratch root**, following this file's existing shape exactly: it
remembers the boot root, creates the scratch through `/api/fs/create`, moves the
working location with `POST /api/path-location` (which seeds the modes document as
it goes), reads the payload from there, and restores the boot root in
`afterEach` — removing the scratch with node's `fs`, never `/api/fs/file`, because
the reserved folder inside it is not deletable through the Editor.

What it measures:

1. The pencil sits beside the mode dropdown and opens the dialog; the dialog shows
   one tab per mode the **file** declares, in the file's order, plus Common; the
   app-owned mode appears nowhere in it.
2. **Create agent** adds a tab, focuses it, and writes nothing — Cancel leaves the
   file byte-identical (read the bytes before and after).
3. Save writes the document, and the dropdown reloads: the new mode is selectable,
   the selection is the newly set default, and the file on disk holds what the
   form showed — including a persona that round-trips through the loader with no
   findings (assert `GET /api/agent-modes` answers `findings: []` afterwards).
The pencil being inert while a reply streams or a lot runs is **not** measured
here: reaching either state needs a model, and the suite makes no LLM call. That
rule belongs to eds-07's component tests.

Every expected label and mode id is read from the endpoints, never written here,
the way the existing specs do it. **The spec must not name the app-owned mode's
id** — it finds the non-editable entry by its flag, and it asserts the dialog has
no tab for it by counting tabs against the file's declared entries.

## Rules this task must not break

* The suite's hermeticity: the server under test runs with the hubs disabled and
  reads `.nojae/nojae-test-settings.json`, never a developer's settings. **No LLM
  call is made** — everything here is a file read, a modal and a save.
* A spec that fails mid-way must not leave the working location pointing at a
  scratch folder: keep the existing `switched` flag + `afterEach` restore.
* Nothing in `web/e2e/**` names the reserved mode id either, if you want the rule
  to hold for the suite as well as for `web/src` — the existing file's header
  already commits to that ("Nothing here names the app-owned mode").
* Assertions are on app state the composables own and on the file on disk, never
  on a sleep.

## How to tell it is done

`cd web && npm run build && npm run test:e2e` green (the whole e2e suite, not just
this file — the working-location switch is shared state). Move this file to
`todos/done/`.
