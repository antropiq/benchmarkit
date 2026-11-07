# eds-06 — `AgentModesDialog.vue`: the modal

**Spec:** D3, D4, D5, D7, the ASCII layout in §4, §4 "Files touched" row 5, and
the resolved open questions (Delete, the line-budget label, the reset button).
**Depends on:** eds-05 (every rule this component renders is a function from the
composable — the component decides nothing).
**Files it touches:** `web/src/components/AgentModesDialog.vue` (new),
`web/tests/agentModesDialog.test.js` (new).

## What to deliver

A `<dialog>` + `<article>` card in the existing family, opened the way
`SettingsDialog` / `RegistrationMemoryDlg` / `BatchModeDialog` open theirs (watch
+ `showModal()` + close), reusing the Settings dialog's two-column shape: a
vertical nav beside the editor column, fixed card size so switching never resizes
it.

* **Common tab** — the default mode (a dropdown over the editable modes the file
  declares; it cannot select the app-owned mode) and **Create agent**.
* **one tab per mode the file declares**, in the file's order, captioned with its
  label, the caption following the Label field live.
* **Create agent** appends a tab immediately and focuses it; nothing is written —
  the new mode exists only in the dialog's state until Save. Its id is the slug
  from the label, shown as an editable field only while the mode is new.
* **Each mode tab** holds exactly what the file holds: Label (free text, empty
  falls back to the id), MCP tools and LSP tools as on/off switches, and the
  System prompt textarea with a live byte counter against the cap **and** a small
  label under the textarea showing the remaining line budget. Empty persona means
  "use the configured system prompt" and says so.
* The two switches are labelled as **narrowing**, not granting: repeat the one
  sentence the dropdown's tooltip gives, so the user learns the rule where they
  set it. A hub off in Settings stays off whatever this says.
* **Reset this persona** — shown only for an entry the app has a built-in persona
  for (the payload's `defaults`), replacing the textarea's content with the
  built-in one. It is a form edit, not a save: it still needs Save, and Cancel
  discards it.
* **Delete** — per tab, with a confirm, refused for the mode that is currently
  the default (a default must name a declared mode).
* **Findings.** An entry finding opens that entry's tab, marks the offending
  field with the loader's own sentence, and disables Save until the entry is
  fixed or removed. A file-level finding (not JSON, not an object, an unknown
  `version`, `agents` not an object) is listed in the dialog and disables Save
  with one line naming the path — and there is **no** "start from benchmarkit's
  defaults" in that case. An entry with no fields to put in a form (value not an
  object, id outside the pattern) still gets a tab: captioned with the raw key,
  holding the finding and a Delete only. A note about a hand-written reserved id
  is listed, gets no tab, and does **not** disable Save.
* **An absent file**: the dialog offers the built-in document as a draft and says
  plainly that saving will create the file.
* **Save** posts the whole document with the `mtime` token. A 409 conflict keeps
  the dialog open with the conflict named and the draft intact — no merge, no
  overwrite, no reload-and-re-apply (out of scope for this lot).
* **Cancel** writes nothing: a file that existed is byte-identical, a file that
  did not exist still does not.

## Rules this task must not break

* **No reserved mode id in `web/src`** — the existing scan walks every `.js`,
  `.vue`, `.css` and `.json` under `web/src`, so this component must not name it.
  The app-owned mode is simply absent from everything this dialog renders; the
  server's findings may still *carry* the id as data.
* Every rule is imported from `useAgentModes` — no re-derived slug, cap, counter
  or Save-disabled predicate in the template.
* All rendered HTML goes through the app's markdown renderer if any markdown is
  rendered at all; this dialog renders plain field text, so it must not reach for
  raw `v-html`.
* The dialog is a browser surface only; it names the two purpose-built routes and
  never an `/api/fs` route.
* The composer's dropdown behaviour is unchanged: this task does not mount
  anything (eds-07 does) and does not touch `WorkBenchInput.vue`.
* Pico card family: do not size the `<dialog>` element itself, only the
  `> article` (the styles themselves are eds-07's).

## Tests to add (`web/tests/agentModesDialog.test.js`)

* Opens with one tab per entry the **file** declares, in the file's order, plus
  Common; the app-owned mode appears nowhere.
* A broken entry gets a tab and its finding, and Save is disabled; fixing or
  deleting the entry enables it.
* A file-level finding lists and disables Save; no defaults-draft affordance is
  offered.
* An absent file offers the built-in draft and says saving creates the file.
* An unrepresentable entry (value not an object) shows a tab with the finding and
  a Delete only.
* A note about a reserved id is listed, has no tab, and Save stays enabled.
* Label edit changes the tab caption immediately.
* Create adds a tab, focuses it, and no fetch to the save route happens.
* Cancel after edits makes no save call.
* Save posts the whole document with the token; the payload round-trips what the
  form showed.
* A 409 keeps the dialog open, names the conflict, keeps the draft.
* Delete is refused for the current default; confirmed for another.
* Reset appears for a built-in entry and not for a user-created one.
* The byte counter counts bytes; the line-budget label appears and never disables
  Save.
* The reserved-id scan (`web/tests/agentModes.test.js`) still passes with this
  component in the tree.

## How to tell it is done

`cd web && npm test` green. Move this file to `todos/done/`.
