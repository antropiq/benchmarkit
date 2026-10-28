# ed-09 — Prove it, and describe it: the guard suite, e2e, and the docs

The last task is not a cleanup pass. Its job is the two guarantees the whole lot
rests on and that no earlier task can honestly self-certify:

1. **nothing but the browser can write**, asserted in code rather than promised
   in prose;
2. the feature is described **as it now is** in the documents the model reads.

Depends on: ed-01 … ed-08 (it tests their totality).

## 1. The read-only-tools guard (the important one)

AGENTS.md's "Tools are read-only by design" survives this lot only if something
asserts it. Add to `tests/test_editorfs_write.py` (or a small
`tests/test_write_surface_isolation.py`, your call — keep the name honest):

* **No write route is a tool.** Walk the MCP hub's and LSP hub's advertised tool
  lists (with `BENCHMARKIT_DISABLE_MCP`/`_LSP` the hubs start disconnected, so
  build the lists from a patched fixture or the declaration templates) and assert
  no tool name or description mentions a `/api/fs` write route. Cheap, but it
  fails the day someone wires a connector to the new routes.
* **No `core/` code path calls them.** A source-scanning test: the strings
  `PUT /api/fs`, `POST /api/fs`, `DELETE /api/fs` must not appear anywhere under
  `src/core/`, `mcp_connectors/`, `lsp_connectors/` or
  `web/src/composables/useChat.js`. They may appear only in `src/api/routes.py`
  (the definition) and the Editor composables (the caller). Same
  "assert the invariant, not the intent" style as the file-icons sync test.
* **The console connector's verdicts are unchanged** by this lot: extend
  `tests/test_console_policy.py` with the commands that would emulate the new
  routes (`rm`, `mv`, `touch`, `mkdir`, `cp`) and assert their current verdicts
  stay as they are. The connector is not given anything here, and a test says so.
* **The terminal stays a human surface**: keep/extend the existing termhub
  assertions that no hub exposes it, so the AGENTS.md sentence about it stays
  checked rather than remembered.

## 2. The cross-cutting write matrix

Each earlier task tested its own route. Add the properties that only make sense
stated once for all of them, so a fifth write route added next month has a table
to slot into:

| Property | Applies to | how |
|---|---|---|
| refuses an escaped path (422) | save, create, rename(src), rename(dst), delete | one parametrised test over the routes |
| refuses the write-flavour deny-list (422, naming the entry) | the same five | parametrised |
| refuses while a chat reply streams (409) | all four routes | set `_CHAT_HANDLE` as the pause tests do |
| never silently overwrites | create, rename, delete | compare the content after the call |
| answers the post-mutation `mtime`/`size` | save, create, rename | re-`stat` and compare |
| leaves no temp file behind on failure | save | directory listing before/after |
| keeps the read routes' answers identical | all | `tests/test_editorfs.py` untouched |

One row deserves its own test: a workspace pointed at the **app repo** browses
`logs/`, and this lot adds `logs/editor_backups` and `logs/trash` to it. That is
acceptable and honest (`logs/` was already there and is gitignored) — say so in
the docstring, and assert that a trashed copy of a policy-refused file is still
refused by the read gate, so the trash cannot become a side door.

## 3. Playwright e2e — the loop a human performs

`web/e2e/02-editor-write.spec.js`, following the conventions of
`01-app-boot.spec.js`: derive expectations from the API rather than hardcoding,
and use the app's own controls to reach a known state. It runs against the real
FastAPI app on its own port with `.env.test` — **no LLM call is needed**, so this
is the cheapest end-to-end proof in the repo.

Work in a scratch folder under the test workspace (created through the API,
removed at the end of the spec):

1. open the Editor tab, create a folder and a file inside it;
2. type in the pane → the dirty dot appears; `Ctrl+S` → the dot disappears;
3. reload the page → the file still holds the text, verified through
   `GET /api/fs/file` rather than the DOM;
4. rename the file from the menu → the chip's label follows, the old path 404s;
5. delete it → it is gone from the tree and a notice names the trash folder
   (assert it contains the path, not a timestamp);
6. the refusals: create `.env`, rename into `mcp_connectors/` — and assert the
   **server's wording** reaches the DOM. A connector rule a human cannot read is
   a rule that gets worked around;
7. the read-only badge's state and its `title` (driving a real streaming reply
   needs an endpoint and is out of scope here — assert from what the app exposes).

If typing into Monaco proves unreliable in the harness, keep the DOM assertions on
the dot and the notice and drive the content through the save route — but state
that limitation in a comment in the spec instead of silently weakening the spec to
"the button exists".

## 4. Frontend test inventory to close out

ed-02 … ed-08 each named their own Vitest cases. Verify the set is complete and
non-overlapping, and fill the combinations no single task owns:

* create → type → save → rename → save again (the token survives a rename);
* delete a folder containing an open dirty buffer (the confirm warns once);
* rename a folder containing an open buffer (repointed, still dirty, saves to the
  new path);
* a probe reload landing while a save is in flight (the `saving` guard wins);
* the tree's keyboard chords while a dialog is open (nothing happens).

## 5. Documentation

### AGENTS.md — the file the model acts on

Three edits, no additions of new sections, stay under the ~300-line rule:

* **the tab description** (`AGENTS.md:18-21`): the Editor's file surface is no
  longer read-only — it has a bounded write surface (save, create, delete to
  trash, rename) that is browser-only. The terminal sentence stays: the shell is a
  human surface, never an agent tool.
* **the `core/editorfs/` row** of the module table (`:68`): replace "Read-only —
  no write route exists" with one line saying the write half reuses the same gate,
  naming the trash and the backup convention.
* **the "Tools are read-only by design" bullet** (`:176`): it stays true of tools;
  add half a line that the browser's write routes are gated by the same policy and
  refused while a reply streams, so the tool/human-surface distinction is explicit
  where a contributor will bump into it.

If one new invariant earns a bullet, it is this: *a `/api/fs` write route is
reachable from the SPA only, and the guard suite asserts it.* One sentence — it is
the rule with the most ways to rot.

No task numbers, no dates, no history of the previous attempt in AGENTS.md.

### README.md

The README lists the commands and the screens; the Editor tab is not described
there. Add it where the other screens are: one sentence on what it does, one on
where a deleted file goes, so a user who deleted something can find it. Nothing
else.

### Code level

* Every docstring written under time pressure in ed-01…ed-08 gets one honest
  read: no task number, no "for now", no "in a later version".
* The two new `logs/` artefacts (`editor_backups`, `trash`) get one line each
  wherever runtime artefacts are documented — a docstring is fine, a stray
  `logs/README` is not.

## 6. Acceptance

* `pytest`, `cd web && npm test`, `cd web && npm run test:e2e` all green.
* The guard tests genuinely fail when they should: verify locally by exposing a
  write route to a fixture tool and by calling one from `src/core/`, then revert.
* `grep -rn "ed-[0-9]" src web/src tests AGENTS.md README.md` → empty except the
  synthetic `tests/test_batch.py` fixtures.
* `grep -rn "no write route" src AGENTS.md` → empty.
* `wc -l AGENTS.md` still under ~300 lines.
* A stranger can read AGENTS.md and the docstrings of `core/editorfs/` and know
  exactly what the app allows, without asking a chat window.
