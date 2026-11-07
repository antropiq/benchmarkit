# Agent modes: editing `.nojae/agent-modes.json` from the composer

**Status**: **landed** by lot `eds-` (`todos/done/eds-01_document_read.md` …
`todos/done/eds-10_docs_last.md`) — D2's document read is
`agent_modes.read_agent_modes_document()` plus the caps constants it publishes
(eds-01, `tests/test_agent_modes.py`); D3/D6/D7/D8's writer is
`src/core/agent_modes_write.py` with the backup callable injected by its route
(eds-02, `tests/test_agent_modes_write.py`); §5's two routes are
`routes.get_agent_modes_document` / `routes.post_agent_modes_save`, with the
rewritten verb guards in
`tests/test_write_surface_isolation.py::TestTheAgentModesRoutesStayPinnedToTheirPaths`
(eds-03) and the tool, connector, LSP, shell and model-client reach guards in the
same file (eds-04, whose `TestTheGuardsActuallyFail` poisons each one); the
dialog's rules are the plain draft functions in
`web/src/composables/useAgentModes.js` (eds-05, `web/tests/agentModes.test.js`);
§5's modal is `web/src/components/AgentModesDialog.vue` (eds-06,
`web/tests/agentModesDialog.test.js`); the pencil and the card styles are in
`WorkBenchInput.vue`'s `.chat-mode-col` and `styles/theme.css` (eds-07,
`web/tests/agentModesPencil.test.js`); §5's acceptance list is
`web/e2e/04-agent-modes.spec.js` on a scratch root (eds-08) and the cross-task
sweep is eds-09. §6's sentences are in the README and `AGENTS.md` (eds-10). This
document is the lot's authority and its historical record — it was annotated, and
is not to be revised: its present-tense descriptions of the modes file as
hand-edited-only describe the state the lot started from.

**Surface**: one new modal (`AgentModesDialog`) reached by one new pencil button
beside the composer's mode dropdown; one new **browser-only** read route and one
new **browser-only** write route; the modes file stops being hand-edited-only and
becomes a third store the app writes. No new tab, no change to the dropdown's
behaviour, no change to what a mode *does* to a chat turn.

## 1. The need

The composer's dropdown offers the workspace's agent modes — **Chat**, **Code**,
**Architect**, and the app-owned **Todo builder**. They come from
`<working location>/.nojae/agent-modes.json`, which the app seeds once and never
writes again: from then on it is edited **by hand, outside the app**.

That rule was right when the file had no UI at all, and it has three costs now
that the feature is used:

* **The vocabulary is invisible.** The file's keys are matched exactly and an
  unknown key is an error, not a suggestion. A user who writes `"mcptool": true`
  instead of `"mcptools": true` loses the whole entry and has to read a finding
  naming a JSON pointer to find out why. The app already knows the vocabulary —
  it validates against it — and shows none of it.
* **The persona is the hard part and it is the least supported.** A mode's
  `systemMessage` is instructions to the model, in the second person, inside a
  line budget, re-sent on every turn. Writing that well inside JSON — as a
  string with `\n`, or as a list of non-blank lines — is a formatting exercise
  with no feedback until the server reports it.
* **There is no way in.** The dropdown is right there, one click from the thing
  it configures, and the only path to changing it is to locate a hidden folder in
  an external editor. The app deliberately hides `.nojae` from the tree, so even
  the Editor tab cannot reach it.

The need is one modal that edits the file the way the Settings dialog edits the
model connection: fields instead of JSON, the caps enforced on the way in, and
the file's own problems named where they can be fixed.

## 2. What exists today (verified against the code)

| piece | where | what it does |
|---|---|---|
| the file | `<working location>/.nojae/agent-modes.json` | `version`, `readme`, `default`, `agents` (id → `{label, mcptools, lsptools, systemMessage}`). Seeded by exclusive create at the two moments a folder becomes the working location; never written by the app afterwards. |
| read + validate | `src/core/agent_modes.py` | The whole read side: `resolve_modes()` answers `{default, modes, findings, source}`. A broken file never raises and never blocks — the built-in defaults answer and `findings` name the JSON pointer (D11 of the modes lot). Cached per workspace on `(mtime_ns, size)`. |
| the words | `src/core/agent_modes_defaults.py` | The seeded document, the three personas, the app-owned mode, and the readme lines. The seeded bytes and the broken-file fallback are the same object. |
| the reader route | `GET /api/agent-modes` (`src/api/routes.py`) | Publishes `modes` with exactly `id`/`label`/`mcptools`/`lsptools`/`editable`, plus `default`, `source`, `findings`. **No `systemMessage`, no `instructions`, no raw bytes, no `?path=`.** |
| the client | `web/src/composables/useAgentModes.js` | One fetch fills the singleton: the dropdown, the hub panels' gray-out and the `agentMode` key of every chat payload all read the same verdict. All the rules are plain functions here, and the file's header says a Settings editor for modes should reuse them rather than re-derive them. |
| the control | `web/src/components/WorkBenchInput.vue` (`.chat-mode-col`) | The `<select>`, plus the two quiet lines it can carry (a session's missing mode, the file's findings). Inert while a lot runs; leaving a locked mode is refused. |
| the caps | `agent_modes.py` constants | `MAX_MODES = 16`, id ≤ 32 chars matching `^[a-z][A-Za-z0-9_-]*$`, label ≤ 40 (trimmed, never refused), persona ≤ 16 KiB of UTF-8 (**over the cap drops the entry**). |
| the invariant | `tests/test_write_surface_isolation.py` | `TestTheAgentModesReaderStaysReadOnly` asserts the verbs registered on `/agent-modes` are `["get"]` and the handler takes no argument — *"a verb on /agent-modes would put the file that decides whether tools exist behind a route"*. |
| the browser-only allowlist | `tests/test_editorfs.py` (a verb map scoped to `/api/fs`), `tests/test_editorfs_write.py` | Mutating routes are an explicit allowlist; a new one appears there on purpose. Note the map's scope: a route outside `/api/fs` is **not** covered by it, so it needs its own pinned verb set. |
| the streaming gate | `_refuse_if_chat_streaming()` (`routes.py`) | 409 while a chat reply is in flight — the one place that rule is written, used by every Editor write. |
| the batch contract | `web/src/composables/useBatch.js` (`LOT_TASK_MODE_ID`) | A lot pins a `code` mode at launch; a workspace without one falls back to the server default and says so. |

So the pieces the dialog needs mostly exist. What is missing is a **write** path,
a **read** path that carries the persona (the dropdown's reader deliberately does
not), and the modal itself.

## 3. Decisions

### D1 — the modes file becomes a store the app writes, and the reservation moves one level up

`.nojae` already holds two kinds of file, and this moves one file from the first
group into the second:

Today:

* **declarations** — the read/write policy, the hub declarations, the agent modes:
  seeded, then owned by the user;
* **stores** — the prompt helpers, this workspace's model connection: seeded, then
  written by the app, because the UI *is* the feature.

`agent-modes.json` moves from the first group to the second. The reservation is
**not** lifted: the folder stays hidden from the tree, from `/api/fs/*`, from
search and from git status, and the file is still reachable only through its own
purpose-built routes — never through a path parameter. What changes is who may
write it: the browser, through one route, and nothing else.

This is the decision that breaks an existing assertion, and the assertion is worth
quoting because the reason behind it survives the change:

> a verb on `/agent-modes` would put the switch that decides whether tools exist
> behind a route, which no marker scan would see because that route is not an
> `/api/fs` route.

The concern is a **tool** reaching the switch, not the browser. So:

* `GET /api/agent-modes` keeps **exactly one verb and zero arguments**. The
  dropdown's reader is untouched, publishes no persona, and stays the app's
  cheapest read.
* The editor's read and write live on their **own paths** (`/agent-modes/document`
  and `/agent-modes/save`).
* **The existing guard is rewritten, not left alone — and this has to be said
  plainly.** It walks the route registrations with `path.startswith("/agent-modes")`,
  so both new paths fall inside it and its `== ["get"]` assertion fails the day the
  save route is added. Pretending otherwise would be a spec that cannot be
  implemented. What replaces it keeps the same teeth and states the narrower truth:
  *the reader path has one verb and no argument*, and *the editor pair exists, is
  pinned to `get`/`post`, and is reachable from the browser only*. The poisoned
  case that class feeds itself is updated to poison the new pair, so the rewritten
  guard is still a guard that can fail.
* The isolation suite gains the sibling check the pair needs: no connector tool,
  no LSP spec, no chat path and no `llmkit` URL builder can name the save route.
  (`tests/test_editorfs.py`'s verb map is scoped to `/api/fs` and will **not**
  cover these paths, so the pair gets its own pinned verb set rather than being
  assumed into an existing allowlist.)
* The save is refused **while a chat reply streams** (409, the shared gate), like
  every other browser write: a reply is the only other writer of the workspace.
  It is **not** gated during a benchmark run — a run never consults a mode
  (`test_the_benchmark_never_consults_a_mode`), so gating it would be a rule
  without a race behind it.

Rejected: editing the file through the Editor tab by un-hiding `.nojae`. That
opens the whole folder — the policy file, the hub declarations with their secrets,
the trash batches — to reach one file, and it would leave the vocabulary problem
exactly where it is.

### D2 — the dialog reads the file's own entries, not the dropdown's list

The two reads answer different questions and must not be merged:

| | `GET /api/agent-modes` (exists) | the editor's read (new) |
|---|---|---|
| answers | *what can I select right now?* | *what does my file say?* |
| entries | valid ones, plus the built-in fallback, plus the app-owned mode last | the file's own entries, in the file's own order, nothing added |
| persona | never published | published, verbatim |
| a broken entry | dropped, with a finding | **shown**, with its finding, and it blocks the save (D3) |

Merging them would be the tempting simplification and it would lose data. The
reader drops an invalid entry as a unit; a dialog fed by it would show three
modes for a file that holds four, and a save would write back three. **A dialog
that silently destroys an entry the user never touched is the failure mode this
decision exists to prevent.**

The editor's read also carries the file's `mtime` — the token D7 compares.

### D3 — the dialog refuses to save what it cannot faithfully write

The file's rules stay the file's rules; the dialog's job is to make them visible
before the save, not to paper over them afterwards:

* **Findings about an entry** (unknown key, a tool flag that is not a boolean, a
  persona over the cap): the entry's tab is opened with the finding shown on the
  offending field, and **Save is disabled** until the entry is fixed or removed.
  The finding is the loader's own sentence, naming the pointer — the dialog just
  puts it next to the field it talks about.
* **Findings about the file as a whole** (not JSON, not an object, a `version`
  this app does not understand, `agents` not an object): the dialog opens, lists
  them, and **Save is disabled** with one line naming the path. It does *not*
  offer "start from benchmarkit's defaults" — that would turn a syntax error into
  the loss of every mode the user wrote.
* **The file is absent** (a workspace whose seed failed, or one that predates the
  feature): the dialog offers the built-in document as a starting point and says
  plainly that saving will create the file. There is nothing there to lose, so
  this is the one case where the defaults are a legitimate draft.
* **How an unrepresentable entry is shown.** An entry whose *value* is not an
  object, or whose id does not match the pattern, has no fields to put in a form.
  It still gets a tab — captioned with the raw key, holding the finding and a
  **Delete** only — so the user can see what the file says and get rid of it. An
  entry that is an object with a bad field is a normal tab with the finding on the
  field. Either way the entry is *visible*, which is what keeps a save from
  quietly dropping it (D2).
* **The caps** (16 modes, 32-char id, 16 KiB persona) are enforced *in the form*:
  the field stops being valid and the counter shows the limit, rather than the
  save being refused by the server. A label over 40 characters stays a
  server-side trim, not a form error — it is presentation, and the loader already
  treats it that way.

The loader's leniency (D11 of the modes lot: a broken file never blocks a chat
turn) is untouched. The dialog is a **writer** and a writer may be strict; the
reader stays forgiving, so a file that is mid-repair still chats.

### D4 — the app-owned mode is absent from the dialog, and cannot be named by it

**Todo builder** is not shown: not as a tab, not in the default dropdown, not in
any list. It is rendered by the app rather than parsed, it carries generated
blocks, and it is the mode the batch runner's contract depends on — a workspace
that could edit its way out of it could not launch a lot it can start.

The rule is not "hide a tab": the save route **refuses a document that declares a
reserved id**, and the client never learns the id. That is the existing rule
(`web/src` must not contain a reserved mode id — asserted in
`web/tests/agentModes.test.js`), and the dialog must not be the thing that breaks
it. So the editor's read answers the file's declared entries only. The one edge
case is a file that *does* name a reserved id by hand: today the loader answers a
**note** and ignores the entry, and the dialog keeps that shape — the entry is
reported as a finding naming the pointer, and it gets **no tab**, because there is
nothing there to edit. Saving drops it, which is the fix the note asks for. It is
a note rather than an error, so unlike D3's blocking findings it does not hold up
the Save button: the user did not lose anything by writing it, and the save is
what cleans it up.

Consequence for the default dropdown: it lists the editable modes the file
declares. It cannot select the app-owned mode, and today the loader refuses such a
`default` anyway.

### D5 — one tab per mode, plus a Common tab, and the tab order is the dropdown order

The dialog reuses the Settings dialog's two-column shape (a vertical nav beside
the editor column, fixed card size so switching never resizes it):

* **Common** — the default mode (a dropdown over the editable modes) and **Create
  agent**.
* **one tab per mode in the file**, in the file's order, captioned with its label.

Tab order *is* dropdown order — the loader returns file order and the dropdown
renders that, and the modes lot's tests assert it rather than trust it. So the
dialog needs no drag-to-reorder to be honest about order, and adding one is out of
scope (D11).

**Create agent** appends a tab immediately and focuses it. Nothing is written: the
new mode exists only in the dialog's state until Save. Its id is generated from
the label as a slug (lowercase, then letters/digits/`_`/`-`, within 32 chars, made
unique against the ids already in the document) and is shown as an editable field
**only while the mode is new**. An existing mode's id is read-only: renaming an id
is a delete plus an add, and it silently orphans every saved session that names
it.

Each mode tab holds exactly what the file holds:

| field | file key | form rule |
|---|---|---|
| Label | `label` | free text; the tab caption follows it live; empty falls back to the id, as the loader already does |
| MCP tools | `mcptools` | on/off switch |
| LSP tools | `lsptools` | on/off switch |
| System prompt | `systemMessage` | textarea; live byte counter against the cap; empty means "use the configured system prompt" and says so |

The two switches are labelled as **narrowing**, not granting: a hub that is off in
Settings stays off whatever this says, and the panel's gray-out already explains
that. The tab repeats the one sentence the dropdown's tooltip gives, so the user
learns the rule where they set it.

### D6 — the save is whole-document, and the file keeps its own shape

One save writes the whole document. There is no per-field save, because the file's
rules are cross-entry: the `default` has to name a declared mode, the count has a
cap, and an entry is validated as a unit.

The writer keeps what the dialog did not touch:

* `version` — written from the app's own constant, never from the form.
* `readme` — **preserved verbatim** if the file had one. It is prose for the human
  who edits the file by hand, and the dialog is not its author. A file with no
  `readme` gets none; a newly created file gets the built-in one.
* `agents` — the dialog's entries, in the dialog's order.
* `default` — the Common tab's choice.

Persona form on disk: the loader accepts a string or a list of **non-blank**
lines, and a list holding a blank line makes the entry invalid. So the writer
emits the **list form only when no line is blank**, and the string form otherwise
— otherwise the app would write a file its own reader refuses, which is the exact
bug the round-trip tests exist to catch. The text itself is never stripped or
rewritten (the loader's rule, and the dialog's).

A save that creates the file (D3's absent case) uses **exclusive create**: a file
that appeared between the read and the save is a conflict (D7), not something to
overwrite.

### D7 — a save that would land on someone else's edit is refused

The file is still editable outside the app, and the Editor's terminal can reach
it, so a dialog that has been open for ten minutes is a stale view. The save
carries the `mtime` the read returned; a disk `mtime` that has moved answers **409**
and the dialog **stays open** with the conflict named — same guard, same shape as
the Editor's save. Nothing is merged and nothing is overwritten.

### D8 — a modes save keeps the bytes it replaced

The app's rule is that no write destroys bytes: an Editor save backs up what it
replaces, a delete moves the file into a dated batch. A modes save is a write of a
file the user owns, so it takes the same pre-save backup the Editor's saves use
(the app's own backup folder, not inside the workspace). Without it, one bad Save
on a document that had four hand-written modes is four lost personas.

**One constraint on where the write half lives.** `tests/test_agent_modes.py`'s
`test_the_loader_imports_no_transport_no_model_and_no_hub` forbids
`agent_modes.py` from importing `editorfs` — and `editorfs.constants.BACKUP_DIR`
is where that backup folder is named. So the writer either reaches the backup
without importing that module, or the **write half lives in its own module**
(`agent_modes_write.py` or similar) beside the loader. The second is the cleaner
answer and it does not break the pairing the seeder relies on: the seeder stays
where it is, because the seeded bytes and the loader must remain one object. This
is an implementation choice, but the spec records the constraint so it is not
discovered as a failing test mid-task.

### D9 — after a save, the dropdown reloads and returns to the default, unless it is locked

The save response is the new document's verdict, so the client reloads the mode
list without a second round-trip. Then:

* the active selection moves to the **newly set default**;
* a newly created mode is present in the dropdown, in the file's order;
* a mode that was **deleted** while selected also lands on the default.

**Exception (proposed).** The modes lot's own D7 (not this spec's) locks the
selection: a thread that entered the app-owned mode keeps it once it has a turn
(`agentModeLockedFor` in `useAgentModes.js`). An unconditional reset would break
that lock from a settings surface, so the reset is skipped when the current mode
is locked, and the quiet line under the select says why. Same for a running lot:
its mode is pinned at launch and a save cannot move it — the dropdown is already
inert for it, and the line under the select already says so.

### D10 — the pencil button lives beside the dropdown, and follows its enablement

The button sits in `.chat-mode-col`, immediately after the `<select>`, using the
pencil glyph the app already draws (the "Edit and resend" chip). It is a
`<dialog>` + `<article>` card in the existing family, opened the way
`SettingsDialog` / `RegistrationMemoryDlg` / `BatchModeDialog` open theirs.

It is **inert while a chat reply streams** (the same window in which the Editor's
writes are refused — opening an editor whose Save will be refused is worse than
not opening it), and **inert while a lot runs** (a lot is running on the modes the
file had when it launched; editing them mid-lot changes what the next task of it
would get on a later launch, which is a decision for an idle moment). It is
available when the list failed to load — the dialog reads the file, not the
dropdown's verdict, and a broken file is precisely when it is most useful.

### D11 — out of scope

* **Reordering modes by drag.** Tab order is file order; a reorder gesture is a
  layout of its own and nothing in the request needs it.
* **Editing or viewing the app-owned mode.** D4.
* **Editing the raw JSON.** No "advanced" tab, no source view. The dialog is the
  vocabulary; the file stays editable outside the app for anyone who prefers it.
* **Per-mode prompt helpers, per-mode sampling, per-mode hub allowlists.** A mode
  is a persona and two hub switches, which is what the file says.
* **Copying a mode, importing/exporting, sharing modes across workspaces.**
* **A 17th mode.** The cap is the feature's edge and the form says so.
* **Undo.** D8's backup is the recovery path; a restore is one copy, as elsewhere.
* **Anything about the benchmark tab.** A run has `LLM_SYSPROMPT` and that is
  enough.

## 4. Shape of the feature

### The dialog

```
┌─ Agent modes ─────────────────────────────────────────────────────────┐
│ This workspace's modes are saved to .nojae/agent-modes.json. A mode   │
│ replaces the persona sent to the model and can switch a tool hub off; │
│ it can only narrow what Settings already allows.                      │
│                                                                       │
│ ┌──────────────┐  ┌─ Chat ─────────────────────────────────────────┐  │
│ │ Common       │  │ Label        [Chat                        ]    │  │
│ │ Chat    ◀    │  │ MCP tools    [on ]  LSP tools  [off]           │  │
│ │ Code         │  │ System prompt                                 │  │
│ │ Architect    │  │ [ You never call a tool…                   ]   │  │
│ │              │  │                                              │  │
│ │ Default:     │  │ 412 / 16384 bytes                            │  │
│ │ [Code    ▼]  │  │                                              │  │
│ │              │  │                                              │  │
│ │[+ Create]    │  │                                              │  │
│ └──────────────┘  └──────────────────────────────────────────────┘  │
│ ⚠ agents.architect.mcptool: unknown key (expected label, mcptools, …)│
│                                          [ Cancel ]  [ Save ]        │
└─────────────────────────────────────────────────────────────────────┘
```

The findings line is the loader's own sentence, and the tab holding the offending
entry is marked. Save is disabled while any finding of D3's blocking kind stands.

### Files touched

| file | change |
|---|---|
| `src/core/agent_modes.py` | gains the **document read** (the file's own entries verbatim + `mtime`) and the server-side half of D3 — a body naming a reserved id, or carrying an entry the loader would drop, is refused rather than written. The loader's read side and its leniency are untouched. `test_the_loader_writes_nothing` pins writes to a named function allowlist, so whatever writes joins that list deliberately. |
| `src/core/agent_modes_write.py` (or the write half of the module above) | **new.** The document writer: D6's shape, D8's backup, D7's guard. Kept separate if the backup forces an `editorfs` import (D8's constraint). |
| `src/api/routes.py` | two new routes on their own path pair, both browser-only; the save takes a body, is gated by `_refuse_if_chat_streaming`, and maps the core's errors to 409 (conflict, streaming) / 422 (invalid document). `GET /api/agent-modes` unchanged. |
| `web/src/composables/useAgentModes.js` | the dialog's rules as plain functions (slug generation, dirty tracking, the D9 reset decision), reusing what is already here rather than re-deriving it in a component. |
| `web/src/components/AgentModesDialog.vue` | **new.** |
| `web/src/components/WorkBenchInput.vue` | the pencil button in `.chat-mode-col`, and the dialog mount. |
| `web/src/styles/theme.css` | `.agent-modes-dialog > article` in the existing card family; the nav reuses the settings two-column rules. |
| `tests/test_agent_modes*.py` | the writer: round-trip, exclusive create, conflict, backup, reserved-id refusal, blank-line persona form. |
| `tests/test_write_surface_isolation.py` | `TestTheAgentModesReaderStaysReadOnly` is **rewritten** per D1 (its prefix match would otherwise fail on the new pair), and the poisoned case is updated to poison the new pair instead. Plus the browser-only check over the save route. |
| `tests/test_editorfs.py` | its verb map is scoped to `/api/fs` and does **not** cover the new pair, so the pair gets its own pinned verb set (`document` → `get`, `save` → `post`) beside it. |
| `web/tests/agentModes.test.js` | the reserved-id scan extended over the new component; slug, dirty and reset rules. |
| `web/e2e/04-agent-modes.spec.js` | a new spec on a scratch root: open, create, save, reload, assert the dropdown gained the mode and the file on disk says so. |
| `AGENTS.md` | see §6. |
| `README.md` | see §6. |

## 5. Acceptance

1. The pencil button sits beside the mode dropdown and opens the dialog; the
   dialog shows one tab per mode the **file** declares, in the file's order, plus
   Common. The app-owned mode appears nowhere in it.
2. Changing a Label changes that tab's caption immediately.
3. **Create agent** adds a tab, focuses it, and writes nothing. Cancel leaves the
   file byte-identical.
4. Save writes the document, and the dropdown reloads: the new mode is selectable,
   the selection is the newly set default, and the file on disk holds what the
   form showed — including a persona that round-trips through the loader with no
   findings.
5. A file with a broken entry shows that entry and its finding, and cannot be
   saved until it is fixed or removed. No entry is ever dropped silently.
6. A file that is not valid JSON is listed and not overwritten; Save is disabled.
7. A save against a file that changed on disk since the read is refused, the
   dialog stays open, and the bytes on disk are the ones the user put there.
8. The previous content is recoverable from the app's backup folder after any save.
9. No connector tool, LSP spec, chat path or model-client URL builder can reach
   the save route; the save is refused while a reply streams and the pencil is
   inert while a lot runs.
10. `web/src` contains no reserved mode id.

## 6. AGENTS.md and README.md, exactly

**AGENTS.md** — three edits, and one of them is a rule that changes:

* The reserved-folder bullet ("The reserved folder is seeded, then read-only to
  the app … except the two stores it writes") — the store list gains the modes
  file, and the sentence should stop counting:
  > The reserved folder is seeded, then read-only to the app: its **declarations**
  > are created only when absent and changed only outside the app, while its
  > **stores** — the prompt helpers, this workspace's model connection and the
  > agent-mode personas — are written by the app through their own routes and
  > never through a path. The folder stays hidden from the tree, from reads and
  > from git status; a failed seed is a note, never a refusal. No write destroys
  > bytes: a save backs up what it replaces, a delete moves the file into a dated
  > batch.
* The agent-mode bullet under "How a request flows" — "one mode is app-owned and
  not editable from that file" gains that the app-owned mode is absent from the
  editor too, and that the file is edited from the composer.
* The Data files table's **"Reserved workspace folder"** row — it names "the two
  stores the app writes — prompt helpers and the model connection", and D1 makes
  that count false. Better than a new number: state the store list without
  counting it, which is the rule this file sets for itself.

**README.md** — the WorkBench paragraph currently ends *"The editable modes live
in `<working location>/.nojae/agent-modes.json`, seeded for you and edited
outside the app."* That becomes the pencil button beside the dropdown and what it
can and cannot change. The `.nojae` paragraph near the working-location section
gains the modes file to the list of files the app writes.

## 7. Open questions

1. **Deleting a mode.** Not in your draft, but the draft leans on it: D3's
   unrepresentable entry has **no form fix** — Delete is the only way out of it —
   and with a 16-mode cap, a mis-created agent would otherwise be removable only
   by hand-editing JSON, which is the thing this feature removes. Proposed: a
   per-tab Delete with a confirm, refused for the mode that is currently the
   default (a default must name a declared mode). **Confirm or drop?** If it is
   dropped, D3's "unrepresentable entry" bullet needs a different answer.

   **User**: Accepted.
   
2. **A file whose `version` is newer than this app understands.** D3 refuses the
   save. The alternative is to refuse to *open* the dialog at all. Refusing the
   save keeps the findings readable; is that enough, or should the button say why
   it is inert?

   **User**: Not needed since the version is fixed and unlikely to change.

3. **Where the persona's line budget is stated.** The modes lot's prose rule is
   ~15 lines for an editable persona; the hard cap is 16 KiB. The byte counter is
   the hard one. Should the form also warn softly at the prose budget, or stay
   silent and leave style to the user?

   **User**: If possible, display a "remaining line cap: xxx" within a small label under the textarea.

4. **A prompt-helper-style "reset this mode to benchmarkit's default persona"?**
   Cheap, and it makes the seeded personas discoverable — but it invites a user to
   overwrite a persona they wrote. Proposed: out of scope.

   **User**: That's a good idea for embedded existing agents (chat, code, architect).
   So yes add a reset button if the agent as a default value in the app.

5. **Does the Common tab need anything else?** The draft keeps it to the default
   and Create. The file's `readme` is prose the app preserves but never shows;
   showing it is a one-line change and arguably the best place for it.

   **User**: Not needed.

6. **The 409 conflict's recovery path.** The draft keeps the dialog open and names
   the conflict. Should it offer "reload the file and re-apply my changes", or is
   copy-out by hand acceptable for a file this small?

   **User**: Out of scope for this lot.
