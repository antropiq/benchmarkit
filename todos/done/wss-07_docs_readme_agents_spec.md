# user notes

This task was delayed and is now ready to be accomplish. 
For more information read the docs/docs/nojae-test-settings-spec.md that as been adressed to fix the gap.

# ws-07 — Docs last: the README, `AGENTS.md`, the spec's status

**Spec:** §6 in full (README §2 table, the "all three are hot" sentence, the LSP
section, the `.env` section, and the AGENTS.md list).
**Depends on:** ws-01 … ws-06 — every one of them in `todos/done/`.
**Code changes:** none. This task only writes sentences.

## Why it is last

`AGENTS.md` is injected into **every WorkBench turn** (its own rule: *"The model acts
on it, so accuracy matters more than volume"*). A sentence written before the code
settled is a lie the model acts on. Write from the code as it is now — read
`src/core/workspace_settings.py`, `src/config.py`, the settings routes — not from the
spec as it was hoped. If a done-file and the code disagree, the code wins; say so.

## Deliver this

1. **README §2 ("Configuring the app")** —
   * The table gains a fourth row: `.nojae/nojae-settings.json` (seeded from
     `nojae-settings.default.json`) | the model connection, sampling and LSP
     activation for this workspace. The `.env` row loses "Endpoint, model, sampling,
     vision flag" and keeps the verifier, the question set and
     `BENCHMARKIT_PYTHON_ROOT` — read the row as it stands and edit it, don't replace
     it wholesale.
   * **"Three files hold everything, and all three are hot"** is now false. The
     paragraph must say **which** files are hot (the two declaration files) and
     **which are not** (`.env` and `.nojae/nojae-settings.json` — a dialog save is
     live, a hand-edit needs a restart). That is D4; state the rule, do not cite the
     decision number. The same fix applies to the other hot-sounding sentence further
     down (`Configuration changes take effect hot: no restart, ever.` — grep
     `take effect hot`; leave a truthful sentence at that place too).
   * The seeding paragraph: the new file joins the seeded set (seeded from its
     committed template, never overwritten). Keep the paragraph's existing structure;
     it is already written for two seeds and one hand-written file, and nothing about
     that distinction changes.
2. **README "### The LSP setup file"** — gains the `lspEnabled` documentation next to
   the two-gate sentence it must already carry: a language server runs only when its
   `enable` flag in `lsp-settings.json` is true **and** its id is in `lspEnabled` (in
   `nojae-settings.json`). Real ids: `pyright`, `typescript`, `vue` — the section's
   existing `npm install -g` line already names the servers, so the reader learns
   which exist and which to enable in one place. Check that section's
   "nothing enabled yet" prose still reads true for `lspEnabled: []` (the template's
   value) — it should need at most a pointer.
3. **README "### LLM configuration — `.env`"** — reflects the smaller `.env`: what is
   left (the verifier block, `QUESTION_SET`, the two technical keys), where what left
   went, and the `LLM_DEBUG` line's new status (a managed key, hand-edited — it left
   the dialog; the README's key block already carries `#LLM_DEBUG=1` in a comment,
   adjust its wording only if it claims the dialog edits it).
   Do **not** document `.env.test` as the endpoint's source: the e2e suite's model
   connection is `.nojae/nojae-test-settings.json` at the working location, read only
   under `BENCHMARKIT_TEST_SETTINGS` (`docs/nojae-test-settings-spec.md`), and the
   README's "Setting up tests for benchmarkit" section already says so — one pointer
   from the `.env` section is all this needs.
4. **`AGENTS.md`** — four small edits, in the file's own voice, honouring its own
   rules (no line numbers, no counts, cut or shrink rather than append; net length
   roughly flat):
   * the reserved-folder exception sentence — *"changed only outside the app — except
     the app's own prompt-helper store"* — names the folder's **two** app-owned
     stores now: the prompt-helper store and this settings file (that is D3's whole
     AGENTS.md consequence);
   * the Data-files table: `.env`'s row says "Endpoint, key, prompt, sampling, set,
     verifier" — that becomes the install-scoped truth (verifier, set, the two
     technical keys); one new row for the new file (it is seeded, app-written, holds
     secrets, has a committed template at the install root);
   * the *"Configuration — `.env` and the two hub declaration files — is edited from
     the UI and takes effect hot"* sentence gains its exception (the new file joins
     the UI, but follows `.env`'s reboot rule);
   * the two-gate prose (`enable` **and** `lspEnabled`) follows the key's move — same
     gate, new file; and the Data-files line "the reserved folder's files have no
     template — the app seeds them" stays true (the new file's template is at the
     install root, like `.env`'s; check the sentence rather than rewriting it).
5. **`docs/nojae-settings-workspace-spec.md`** — a status line at the top in the house
   style of the other docs (see `docs/in-house-llm-client-spec.md` after lot llmh, or
   a `docs/windows-support/*` file): **landed by lot `wss` (todos/wss-…)**, one line
   on where each §'s landing evidence is (the task files in `todos/done/`), and that
   the §10 manual check lives in `docs/nojae-settings-workspace-smoke-checklist.md`
   as the human step. Do **not** rewrite the spec — annotate, don't revise; its
   "Not implemented" sentence in the header is the one line that must change.

## Files it touches

| file | change |
|---|---|
| `README.md` | §1–§3 above |
| `AGENTS.md` | §4's four edits |
| `docs/nojae-settings-workspace-spec.md` | status line only |

## Rules it must not break

* `AGENTS.md`'s own rules, quoted: *"It states what the app is and how its pieces fit,
  not how to build it"*; *"No code details that move"*; *"Keep it short and current…
  Cut or shrink a paragraph when the code it described changes instead of appending a
  newer one beside it."* One idea per bullet. If a test enforces an invariant, name
  the test as a pointer (`tests/test_env_isolation.py` is the lot's guard — it is
  worth one such pointer).
* Do not restate the spec in `AGENTS.md`: no decision numbers, no member lists, no
  key tables.
* Do not touch `todos/**` (except moving this task's own file to `done/`), the code,
  `.env.example` or `nojae-settings.default.json` (docs don't move templates — if a
  doc and a template disagree, the disagreement is a bug to report, not to edit away).
* No "recently moved from `.env`" framing in `AGENTS.md` — describe the app as it is
  now; the move is history and belongs in the spec's status line.

## Acceptance

* `grep -rn -i "hot" README.md` — every remaining "hot" sentence is true for the file
  it describes; `grep -n "nojae-settings" README.md AGENTS.md` shows the new rows.
* `AGENTS.md` read top-to-bottom still describes the app as it now is; no sentence in
  it is longer or vaguer than it was; its length is within a line or two of before.
* `pytest tests/test_config.py tests/test_api.py -q` still green (paranoia check that
  no file was edited by accident), and `pytest tests/test_chat.py -q` green —
  `AGENTS.md` is injected into every chat turn and those modules assert the
  injection's shape.
* `git diff --stat` for this task is `README.md`, `AGENTS.md`, the spec's status line,
  and the moved task file. Nothing else.
* Move this file to `todos/done/` when the above is true.

## After this task

The lot is complete. A human runs `docs/nojae-settings-workspace-smoke-checklist.md`
(ws-06's file); that is the only step left, and it is deliberately not a task.
