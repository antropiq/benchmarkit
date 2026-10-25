# Parked enhancements (not scheduled)

Everything deliberately left out of ed-01 … ed-09, written down so the
decisions are not lost. Nothing here is required by the brief; each item is
independently shippable once the earlier steps are in place, and each has a reason it
was deferred.

## A. File mutations — moved out of this file, into ed-10

The full standard-editor behaviour (create, edit, delete) is a **future
version**, and it is specified properly in `ed-10-editor-write-layer.md` rather
than parked here as a sketch: decision D4 puts the whole write layer outside
this lot, so the write endpoint, the conflict rule, the backups and the
mutation endpoints are designed as one piece.

What ed-09 already prepared for it: the open-file records carry `mtime`, and the
tree rows carry a directory `mtime`, which is what the conflict precondition and
the tree invalidation after a rename/delete will compare.

## B. LSP diagnostics inside the editor

The LSP hub already returns diagnostics (`lsp_*_diagnostics` tools,
`lsphub/features.py`, `DIAGNOSTICS_WAIT_SECONDS`), so the data exists; what is
missing is a bridge from the hub to a CodeMirror `linter()` source.

Shape worth keeping: one `POST /api/lsp/diagnostics` taking `{ path, text }` —
i.e. **the buffer's text, not the file on disk**, otherwise the editor reports
errors for code the user has already fixed. `@codemirror/lint` renders it
(that is the package ed-04 excluded, and this is where it enters), positions
convert from LSP UTF-16 0-based to CM6 character offsets.

Two cautions: a language server session must be *started* for the file's
language (a cold pyright takes seconds, so diagnostics appear 3–10 s after a
keystroke — debounce ~800 ms and show a "checking…" marker rather than nothing),
and the hub's read-only invariant stays: `didOpen`/`didChange` from a *buffer*
is information flow into the server, never a write to disk. The
"an activated language server is never invisible" rule applies — if there is no
live session, the gutter says so instead of showing a clean bill of health.

Also worth noting for whoever picks this up: `lsp_vue_*` checks templates only
(Volar delegates `<script>` semantics to tsserver, AGENTS.md gotcha), so a Vue
file's gutter would look clean while its script has type errors. That asymmetry
must be visible in the UI or it becomes a false assurance.

## C. Session/tab persistence

Open tabs and the expanded tree do not survive a page reload (noted in ed-03).
A `localStorage` snapshot (`benchmarkit:editorTabs`: paths + active) would
restore *which* files, re-fetching content on boot — cheap, ~30 lines.
Deliberately **not** storing the text: with a read-only pane (D4) there is no
unsaved text to lose, and the same reasoning already keeps
`BENCHMARKIT_PATH_LOCATION` out of persisted state ("a machine move can never be
broken by a remembered path"). Once ed-10 makes the pane editable, this item has
to be re-decided together with the dirty-state rules rather than kept as is.

## D. Quick-open (`Ctrl+P`)

Needs a workspace-wide file list. Options: an unbounded recursive walk (bad on a
large checkout), a server-cached index (invalidation problem), or reuse the
existing LSP `workspace/symbol` (needs a running server, and returns symbols
rather than files). No cheap correct answer, so it waits behind A and B.

## E. Diff view on save

`fedit_mcp` already produces unified diffs and SEARCH/REPLACE previews; the same
diff computed server-side for the future `PUT /api/fs/file` could render in
`@codemirror/merge` before the write, giving the human the same "see what you
are about to change" step the agent gets. Nice, not necessary; the backup +
revert path designed in ed-10 is what makes a bad save recoverable — neither
exists yet in this lot.

## F. Terminal niceties

* multiple terminal tabs in the strip (the session registry in ed-06 is already
  keyed by id, so this is mostly UI);
* `serialize()` addon → "Save transcript", mirroring the chat session save in
  `todos/done/session-saving.md`;
* copy-on-select, search addon, unicode11 addon.

Skip unicode11 unless a real need appears: it adds a WASM payload to a tab that
is already carrying mermaid.

## G. Watcher / auto-refresh of the tree

ed-09 already keeps the *open* files and the *expanded* branches fresh while a
chat reply runs, using a bounded `stat` probe. A file watcher is the wider
version of that — every change anywhere in the workspace, at any time, including
changes made by the terminal or by another process.

The tree outside those two sets is still refreshed by hand. A file watcher
(`watchfiles`, already installed transitively under `uvicorn[standard]`) pushing
invalidation over SSE would make external edits visible. Deferred because a
recursive watcher on an arbitrary workspace is a real resource cost (inotify
descriptors on `node_modules`-sized trees) and because ed-09's probe already
covers the case that actually hurts — the model editing a file the human is
looking at.

## H. Editor ↔ WorkBench handoff

The tempting feature: "ask the model about this selection", sending the selected
lines plus the file path into the chat. It respects the read-only rule (it is
the user pasting context, not a tool reading disk) but it touches `useChat`
state and the conversation-lock rules, so it deserves its own note rather than
tacking onto step 4. Symmetrically, "insert the model's snippet at the cursor"
belongs here, not in the chat composable.

## Sequencing advice

| Want | Do |
|---|---|
| The editor to feel like a real IDE soonest | B (diagnostics) — the single biggest perceived-quality jump after syntax colouring |
| The safest next capability | C (tab persistence) — 30 lines, no new guard surface |
| The most asked-for file-management feature | A → see ed-10; budget the guard suite, not just the endpoints |

## What must not be relaxed when any of these is built

* every path-taking endpoint keeps `editorfs.policy` as its only gate, with
  `for_write=True` for anything that mutates (D3);
* no editor or terminal capability is ever exposed to the MCP/LSP hubs —
  read-only-by-design is about the *agent's* reach, and this tab is the human's
  (in this lot the editor cannot write at all, per D4 — ed-10 keeps that
  separation when it adds the write path);
* any new hub-like long-lived process (more terminal sessions, a watcher) joins
  the process-*tree* teardown rule, and the "an orphan must be nameable"
  visibility rule;
* the Benchmark tab's critical path stays free of editor/terminal chunks (the
  ed-05 acceptance criterion, re-checked after any dependency is added).
