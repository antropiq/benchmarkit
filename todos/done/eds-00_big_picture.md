# Agent modes editor — big picture

**Spec:** `docs/agent-modes-editor-spec.md` (read it; this map does not replace
it). Everything below is that spec's D1–D11 plus the answers written under its
open questions — those answers are decisions, not options.

Give the composer's mode dropdown a way in: one pencil button beside the
`<select>` opens one modal that edits `<working location>/.nojae/agent-modes.json`
the way the Settings dialog edits the model connection — fields instead of JSON,
the caps enforced on the way in, and the file's own problems named where they can
be fixed. The modes file stops being hand-edited-only and becomes a third **store
the app writes**; the reservation of `.nojae` is not lifted.

No new tab. No change to the dropdown's behaviour. No change to what a mode *does*
to a chat turn, to the benchmark, or to the loader's leniency (a broken file still
chats — that is the modes lot's D11 and it is untouched).

## The shape

| piece | role | task |
|---|---|---|
| `src/core/agent_modes.py` | gains the **document read**: the file's own entries verbatim, in file order, plus its `mtime`, findings, caps and the built-in defaults. Read side and leniency unchanged. | eds-01 |
| `src/core/agent_modes_write.py` | **new.** The whole-document writer: D6's shape, D7's `mtime` guard, D8's backup, exclusive create, and the server-side half of D3 (refuse what the loader would drop). | eds-02 |
| `src/api/routes.py` | two new browser-only routes on their own path pair: `GET /api/agent-modes/document`, `POST /api/agent-modes/save`. `GET /api/agent-modes` is untouched. | eds-03 |
| `tests/test_write_surface_isolation.py`, `tests/test_editorfs.py` | the existing agent-modes guard is **rewritten** (its prefix match would otherwise fail the day the save route lands), and the new pair gets its own pinned verb set and its own reach checks. | eds-03, eds-04 |
| `web/src/composables/useAgentModes.js` | the dialog's rules as plain functions (document fetch/save, slug, dirty, counters, the D9 reset decision), reusing what is already there. | eds-05 |
| `web/src/components/AgentModesDialog.vue` | **new.** The modal. | eds-06 |
| `web/src/components/WorkBenchInput.vue`, `web/src/styles/theme.css` | the pencil in `.chat-mode-col`, the mount, the card styles. | eds-07 |
| `web/e2e/04-agent-modes.spec.js` | a new spec on a scratch root: open, create, save, reload, and the file on disk says so. | eds-08 |

## The two reads must stay two (D2)

`GET /api/agent-modes` answers *what can I select right now* — valid entries plus
the built-in fallback plus the app-owned mode last, **no persona**. The editor's
read answers *what does my file say* — the file's own entries, in the file's own
order, nothing added, persona verbatim, broken entries **shown** with their
finding, plus the `mtime` the save compares. Merging them loses data: the reader
drops an invalid entry as a unit, so a dialog fed by it would show three modes for
a file that holds four and a save would write back three. **A dialog that silently
destroys an entry the user never touched is the failure mode this lot exists to
prevent.**

The document read is **not cached** — it is the writer's baseline, and a cached
`mtime` would defeat the conflict guard.

## Decisions, in the words a task must honour

| # | decision |
|---|---|
| D1 | The modes file becomes a store the app writes — **the browser only, through one route, never through a path parameter**. The reader path keeps exactly one verb and zero arguments; the editor pair lives on its own paths. The save is refused (409) while a chat reply streams, and is **not** gated during a benchmark run — a run never consults a mode, so gating it would be a rule without a race behind it. |
| D2 | Two reads, never merged (above). |
| D3 | The dialog refuses to save what it cannot faithfully write. Entry findings open the tab, mark the field and **disable Save** until fixed or deleted. File-level findings (not JSON, not an object, a `version` this app does not understand, `agents` not an object) are listed and disable Save, naming the path — and there is **no** "start from benchmarkit's defaults" for that case. An **absent** file is the one case where the built-in document is a legitimate draft, and the dialog says saving will create the file. An entry with no fields to put in a form (value not an object, id outside the pattern) still gets a tab: raw key, the finding, and Delete only. |
| D4 | The app-owned mode is absent from the dialog entirely, and the save route **refuses a document that declares a reserved id**; the client never learns the id. A file that names one by hand keeps today's shape: a **note** naming the pointer, **no tab**, and it does **not** hold up Save — the save is what cleans it up. The finding sentence contains the id as server data; that is correct and must not be "fixed" by filtering findings. |
| D5 | One tab per mode plus a Common tab; tab order **is** dropdown order is file order, so there is no drag-to-reorder (out of scope). Create appends a tab and focuses it, writing nothing; its id is a slug of the label, editable **only while the mode is new** — an existing id is read-only, because renaming an id is a delete plus an add and silently orphans every saved session that names it. |
| D6 | One save writes the whole document. `version` comes from the app's constant, never the form. `readme` is preserved verbatim if the file had one; a file with none gets none; a newly created file gets the built-in one. `agents` in the dialog's order. The persona goes to disk as a **list only when no line is blank**, and as a string otherwise — otherwise the app writes a file its own reader refuses. Persona text is never stripped or rewritten. |
| D7 | A save whose `mtime` token no longer matches the disk answers **409** and the dialog **stays open** with the conflict named. Nothing is merged, nothing is overwritten, and reload-and-re-apply is out of scope for this lot. |
| D8 | A modes save keeps the bytes it replaced: the same pre-save backup the Editor's saves use (the app's own backup folder, not inside the workspace). |
| D9 | The save response carries the new verdict, so the dropdown reloads without a second round-trip; the selection moves to the newly set default, and a deleted-while-selected mode lands on the default too. **Skipped when the current mode is locked** (a thread that entered the app-owned mode keeps it) and for a running lot (its mode is pinned at launch); the quiet line under the select says why. |
| D10 | The pencil sits in `.chat-mode-col` immediately after the `<select>`, using the pencil glyph the app already draws. It is a `<dialog>` + `<article>` card opened the way `SettingsDialog` opens its own. Inert while a reply streams and while a lot runs; **available when the dropdown's list failed to load** — a broken file is precisely when it is most useful. |
| D11 | Out of scope: drag reordering, editing or viewing the app-owned mode, raw-JSON editing, per-mode sampling/helpers/allowlists, copy/import/export, a 17th mode, undo, anything about the benchmark tab. |

Resolved open questions, binding:

* **Delete** exists: per-tab, with a confirm, refused for the mode that is
  currently the default (a default must name a declared mode).
* A newer `version` needs no special button — refusing the save is enough.
* The persona field shows the hard byte counter **and** a small label with the
  remaining line budget. The budget is soft: it warns, it never blocks a save.
* A **reset to benchmarkit's default persona** button, shown only for an entry
  the app has a built-in persona for.

## The caps (one home each, in `agent_modes.py`)

16 modes; id ≤ 32 chars matching the id pattern; label ≤ 40 (trimmed on the
server, never a form error — it is presentation); persona ≤ 16 KiB of UTF-8
(over the cap drops the entry). The soft persona line budget the form shows is
**16** and gets its own constant in eds-01, because that number already lives in
a prose test and must not be spelled twice. The client reads all of them from the
document payload — **no cap is hardcoded in `web/src`**.

## Invariants every task must keep

* `GET /api/agent-modes` keeps one verb and zero arguments. The new pair is
  pinned to `get` / `post` and is reachable from the browser only: no connector
  tool, no LSP spec, no chat path and no `llmkit` URL builder can name the save
  route. AGENTS.md: *"Tools are read-only by design… `tests/test_write_surface_isolation.py`
  asserts that instead of trusting this sentence: no connector, tool or chat path
  reaches a write route."*
* The reserved folder stays hidden from the tree, from `/api/fs/*`, from search
  and from git status. The modes file is reachable only through its own routes,
  never through a path parameter.
* Nothing under `src/core` imports the API layer, and `agent_modes.py` still
  imports no `editorfs` (its loader test forbids it) — which is why the write half
  is its own module.
* The loader's never-raises, never-blocks contract is untouched: the dialog is a
  **writer** and may be strict; the reader stays forgiving.
* `editorfs` exports no new write spelling (`test_editorfs_exposes_the_guarded_write_and_nothing_beyond_it`
  refuses one). The modes writer reuses the public backup helper and does its own
  temp-file + `os.replace`.
* `web/src` contains no reserved mode id — the scan in `web/tests/agentModes.test.js`
  already walks the whole tree, so the new component must pass it rather than be
  added to it.
* Tests stay hermetic: no reachable model, hubs disabled, no test reads a real
  model connection.

## How to run this lot

Launch Batch mode with **prefix `eds-`** (hyphen included), lot folder `todos`.
The hyphen is part of the prefix: `core/batch.py`'s `parse_task_number` wants the
digit run immediately after it. A task is finished when its file has moved into
`todos/done/`.

## Task order

| # | task | depends on |
|---|---|---|
| eds-01 | The document read in `agent_modes.py` (+ caps constant) and its tests | — |
| eds-02 | `agent_modes_write.py`: the whole-document writer, guard, backup | eds-01 |
| eds-03 | The two routes, and the rewritten agent-modes verb guards | eds-01, eds-02 |
| eds-04 | The reach guards: marker scan, connectors, LSP, model client | eds-03 |
| eds-05 | `useAgentModes.js`: the dialog's rules as plain functions | eds-03 |
| eds-06 | `AgentModesDialog.vue` and its component tests | eds-05 |
| eds-07 | The pencil button, the mount, the card styles | eds-06 |
| eds-08 | The e2e spec on a scratch root | eds-07 |
| eds-09 | Verification pass across both suites | eds-01 … eds-08 |
| eds-10 | Docs last: `AGENTS.md`, `README.md`, the spec's status line | eds-09 |

Each task ships its own tests (pytest for the core and API halves, Vitest for the
frontend ones) and leaves its suites green. eds-09 is the sweep for what an
individual task cannot see; eds-10 writes the sentences from the code as it is
then, not from this map.
