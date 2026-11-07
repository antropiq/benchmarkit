# eds-10 — Docs last: `AGENTS.md`, `README.md`, the spec's status line

**Spec:** §6 in full ("AGENTS.md and README.md, exactly"), plus the Data-files
table's count that D1 makes false.
**Depends on:** eds-09 — the whole lot, every file in `todos/done/`.
**Code changes:** none. This task only writes sentences.

## Why it is last

`AGENTS.md` is injected into **every WorkBench turn** (its own rule: *"The model
acts on it, so accuracy matters more than volume"*). A sentence written before the
code settled is a lie the model acts on. Write from the code as it is now — read
`src/core/agent_modes.py`, `src/core/agent_modes_write.py`, the two routes,
`useAgentModes.js`, `AgentModesDialog.vue` — not from the spec as it was hoped. If
a done-file and the code disagree, the code wins; say so.

## Deliver this

**`AGENTS.md` — three edits, and one of them is a rule that changes.**

1. The reserved-folder bullet under "Data files" ("The reserved folder is seeded,
   then read-only to the app… except the two stores it writes…"). The modes file
   joins the store list, and the sentence must **stop counting**. The spec gives
   the wording to aim at: declarations are created only when absent and changed
   only outside the app, while stores — the prompt helpers, this workspace's model
   connection and the agent-mode personas — are written by the app through their
   own routes and never through a path. Keep the rest of the bullet (hidden from
   the tree, reads and git status; a failed seed is a note; no write destroys
   bytes).
2. The agent-mode bullet under "How a request flows": it currently ends *"one mode
   is app-owned and not editable from that file"*. That gains that the app-owned
   mode is absent from the editor too, and that the file is edited from the
   composer. Keep the existing invariants — a mode only ever narrows what settings
   and the conversation already allow; AGENTS.md is injected whatever the mode.
3. The Data-files table's **"Reserved workspace folder"** row, which names "the
   two stores the app writes — prompt helpers and the model connection". D1 makes
   that count false. State the store list **without counting it** — that is the
   rule this file sets for itself. Check the same paragraph's *"each has a
   committed template"* and *"a reserved-folder file has no template beside it"*
   sentences still read true for the modes file rather than rewriting them blind.

Also check, and fix only if now false: the "Shape of the app" / "Code layout"
lines that describe the modes file as hand-edited, and the Key-patterns sentence
about the Editor's write surface being the browser's alone (the modes save is a
second browser-only write surface that is not an `/api/fs` route — if that
sentence reads as though only `/api/fs` writes exist, tighten it).

**`README.md` — two places.**

* The WorkBench paragraph currently ends *"The editable modes live in
  `<working location>/.nojae/agent-modes.json`, seeded for you and edited outside
  the app."* That becomes the pencil button beside the dropdown and what it can
  and cannot change: fields per mode (label, the two hub switches, the persona),
  create and delete, the default, the caps, and the fact that the app-owned mode
  is not in it. The file stays editable outside the app too — say so, because the
  spec keeps that true.
* The `.nojae` paragraph near the working-location section gains the modes file to
  the list of files the app writes. Check the surrounding "which files are hot /
  which need a restart" prose still reads true, and the data-file tables at the
  top of that section if they claim the modes file is hand-edited-only.

**`docs/agent-modes-editor-spec.md` — the status line only.** Replace *"Status:
draft spec, open questions at the end. Implementation not started."* with the
house style the other docs use (see the top of
`docs/nojae-settings-workspace-spec.md`): **landed by lot `eds-`**, one line
pointing at where each part's evidence is — the task files in `todos/done/eds-*.md`,
the module names, the guard tests. Annotate, do not revise: the decisions and the
open-question answers stay as written, because they are the record of why the code
is shaped this way.

## Rules this task must not break

* `AGENTS.md`'s own rules, quoted: *"It states what the app is and how its pieces
  fit, not how to build it."* *"No code details that move: no line numbers, quoted
  code, exhaustive route or member lists, counts or version numbers."* *"Keep it
  short and current — Cut or shrink a paragraph when the code it described changes
  instead of appending a newer one beside it."* One idea per bullet. A test that
  enforces an invariant is worth one pointer (the write-surface suite is the one
  worth naming here).
* No route lists, no field lists, no decision numbers in `AGENTS.md`. Describe the
  app as it is now — no "recently became writable by the app" framing; the history
  belongs in the spec's status line.
* Do not document the modes file as reachable from the Editor tab: it is not, and
  the reservation was not lifted.
* Do not touch the code, the tests, `.env.example` or any template. If a doc and a
  template disagree, that is a bug to report, not to edit away.
* `todos/**` is untouched except moving this task's own file into `done/`.

## Acceptance

* `grep -n "two stores" AGENTS.md` answers nothing; the store list is stated
  without a count.
* `grep -n "edited outside the app" README.md AGENTS.md` — every remaining hit is
  true of the file it describes (the modes file is still editable outside the app;
  the declarations still are).
* `AGENTS.md` read top-to-bottom describes the app as it now is, and its length is
  roughly flat — the edits replace, they do not append.
* `pytest tests/test_chat.py tests/test_config.py -q` green (AGENTS.md is injected
  into every chat turn and those modules assert the injection's shape), and
  `pytest -q` green — no code was edited by accident.
* `git diff --stat` for this task is `AGENTS.md`, `README.md`, the spec's status
  line, and the moved task file. Nothing else.
* Move this file to `todos/done/` when the above is true.

## After this task

The lot is complete.
