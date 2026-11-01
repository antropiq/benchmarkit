# mbl-07 — One e2e for the switch, the repo-wide name sweep, and the docs

Last task. Nothing in it changes behaviour: it proves the switch reloads the
bullets end to end, then removes every remaining trace of the old name from the
repo, and updates the two documents that still describe the old shape.

Depends on: mbl-01 … mbl-06 — the whole lot. Run it only when the other six are
in `todos/done/`.
Spec: §7 (the e2e row, the AGENTS.md table row) and **§6 in full** (the AGENTS.md
edits are quoted there verbatim — copy them, do not paraphrase).

## Files

| file | change |
|---|---|
| `web/e2e/03-workspace-switch.spec.js` | **extend** — one new test: switch the working location, assert the bullet row changes. Do not create a new spec file. |
| `AGENTS.md` | the three edits of §6: delete the `:135` row, rewrite the `:139` row, add the `core/prompt_helpers.py` row to the `:68` module table |
| `README.md` | one edit only: the `.nojae/` paragraph (`:178-185`) now says that folder also holds the workspace's prompt helpers, which the app writes |
| whatever the sweep finds | renames and stale comments only — see §3's list of what is *not* residue |

Do **not** touch `.gitignore`. Its `resources/memory_registration.json` line is
removed by hand, outside this lot.

## 1. The e2e (extend the existing module)

`03-workspace-switch.spec.js` already owns the machinery this test needs: the
`e2e-ws-root` scratch folder, `create()` / `currentRoot()` / `removeScratch()` /
`setRoot()`, and the `bootRoot` remember-and-restore so a failing test cannot
leave the suite pointing at a scratch root. Add one test beside the three that
are there, reusing them as-is. No LLM call.

What it asserts:

1. Read the row from the **app**, not the API: open WorkBench, read the nine
   `.memory-bullet` buttons, and note the boot root's own values.
2. Give the scratch root different helpers — `POST /api/prompt-helpers` for one
   slot with a recognisable string. Either before the switch (the file already
   exists if the folder was a working location earlier) or right after it, then
   re-read; whichever order the fixture makes simplest.
3. `setRoot(page, <scratch>)`, then assert the bullet row shows the scratch
   root's slot and **not** the boot root's. This is D8's whole point; it is the
   regression this lot would otherwise leave unpinned.
4. `setRoot(page, bootRoot)` back — the afterEach restores too, but assert the
   original values came back.

The spec's own comment says the scratch root is removed with `node:fs` because a
switch-in seeds `.nojae` there and the Editor will not delete that subtree. The
new test makes that sentence truer (one more seeded file), so leave the
`removeScratch()` calls exactly as they are and do not "improve" the cleanup.

One caution the spec is already written around: `04-agent-modes.spec.js:180`
measures `.memory-bullets`' box. That class name is unchanged by this lot (mbl-04
was import-renames only, no markup, no CSS), so that spec must keep passing
without any edit. If it fails, something in this lot renamed markup it should not
have — fix that, not the spec.

`cd web && npm run test:e2e` is the check. It needs a real endpoint in
`.env.test`; if the machine has none, say so in the done-note rather than
skipping silently — this is the only task whose acceptance is a Playwright run.

## 2. AGENTS.md — §6, exactly

Three edits. §6 quotes the replacement text for the `.nojae/` row; use it
verbatim.

- The `resources/memory_registration.json` row of the Data files table:
  **delete the row.**
- The `<working location>/.nojae/` row: replace it with §6's blockquote — the one
  rule becomes a stated **rule split** (policy + modes: seeded, then edited only
  outside the app; prompt helpers: seeded the same way and written by the app
  itself). Do not write "three files": a count is the kind of detail AGENTS.md's
  own editing rules say to avoid.
- The module table: one row for `core/prompt_helpers.py` — the workspace's nine
  prompt-helper slots: read, save, seed, and the one-time lift from the legacy
  app-level file.

Stay under the ~300 line ceiling (currently ~300): the deleted row pays for the
added one, so the net is roughly flat. If it grows past the ceiling, shrink
another paragraph rather than dropping a row.

AGENTS.md describes the app as it is now: no "previously", no "moved from
`resources/`", no migration story. The migration lives in the code and in
`docs/prompt-helpers-workspace-scope.md`, not here.

## 3. The sweep

Grep the repo for `memory` / `MEMORY` (case-insensitive) and clear real residue.
Most hits are legitimate, and renaming them would be damage:

| leave alone | why |
|---|---|
| `src/api/events.py:1`, `src/core/editorfs/constants.py:71`, `src/core/bench.py` | "in-memory" / process memory, not the feature |
| `web/src/composables/useEditorFiles.js:21`, `useSplitter.js:7` | in-memory buffer; layout prose |
| `web/src/styles/theme.css` in full (`.memory-bullets`, `.memory-bullet`, `.memory-dialog`, `.memory-color`) | mbl-04 changed no markup and no CSS; the class names are stable and `04-agent-modes.spec.js:180` measures one of them |
| `web/src/components/BatchModeDialog.vue:8`, `SearchDialog.vue:7`, `EditorTreeDialogs.vue:10` | they cite `RegistrationMemoryDlg.vue` as the `<dialog>` pattern; the component keeps that name |
| `web/src/composables/useChat.js:96,104` | comments about inserting "a memory bullet" into the draft — reword to "prompt helper" only if the sentence reads oddly; prose, not identifiers |
| `resources/*.json` | question sets, matched by accident |

**Real residue — fix**:

- any Python/JS identifier, route string, or dict key still spelling
  `memory`/`MEMORY` **for this feature** (`memory_registration`, `MemorySlotIn`,
  `loadMemory`, `MEMORY_FILENAME`, `/api/memory`, …);
- comments or docstrings describing the **old** behaviour — above all any sentence
  claiming the app repairs a wrong slot count (D4: it answers 422 now), or that
  the file lives in `resources/`;
- `README.md:178-185` (the one required README edit, §2).

`grep -rin "memory" src web/src tests web/e2e docs resources` still prints lines
when you are done — that is correct. The end state: every remaining hit is about
process/RAM memory, a deliberate CSS or component name this lot kept, or prose
that no longer misdescribes the feature.

Do not add a README subsection describing the nine bullets. The spec calls it
optional and it was declined.

## 4. Rules

- **The legacy file is not touched**: no delete, no move, no trash, no edit of
  `resources/memory_registration.json`, and no `.gitignore` edit. The lift leaves
  it in place (D6); the user cleans it up by hand.
- No behaviour change in this task. If the sweep seems to need one, stop and
  report it instead of making it.
- Quote AGENTS.md's rules to yourself before editing it: under ~300 lines, one
  idea per bullet, no counts, no history.

## 5. Done when

- `cd web && npm run test:e2e` passes, including the new bullet-switch test, and
  `04-agent-modes.spec.js` passes unedited.
- `pytest` passes, unchanged from mbl-06 (this task edits no Python logic).
- AGENTS.md carries the three §6 edits, states the rule split rather than a file
  count, and `grep -in "memory" AGENTS.md README.md` prints nothing that
  describes the nine prompts under the old name.
- `git status` shows no change to `.gitignore` and none to
  `resources/memory_registration.json`.
- Move this file to `todos/done/` when the above is true.
