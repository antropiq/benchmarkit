# Editor write features — big picture

Give the Editor tab the four everyday file operations it does not have: **save,
create, delete, rename** — for files *and* folders. Deletes go to a trash
folder, never straight to `unlink`.

The read half of this feature already shipped and works: the tree, the bounded
file read, the `stat` probe, git markers, diff tabs, the Monaco pane, the PTY
terminal. What is missing is the write half, and the two decisions that made the
read half safe (one path policy per surface; nothing here writes) are the two
things this lot has to change — carefully, and only for the browser.

The previous attempt at this tab used CodeMirror and was replaced by Monaco. The
tracked tree no longer mentions it, but the tree is still covered in comments and
docstrings that cite the **task documents of that attempt** — those files are
gone, so the citations point at nothing. `ed-01` §4 sweeps them, and that sweep is
deliberately separable from the write work: it touches no behaviour and can be
executed (or reviewed) on its own, first. `logs/` is gitignored and holds old
backup copies from that era; they are not part of the source tree and no task
touches them.

## What already exists, and what this lot adds

| Piece | Today | Added here |
|---|---|---|
| `core/editorfs/policy.py` | `resolve_in_workspace(rel, for_write=)` and `writable()` exist, **unused** | the write flavour becomes live |
| `GET /api/fs/file` | returns `mtime`, and `truncated`/`binary` | its `mtime` becomes the save guard's token |
| `GET /api/fs/stat` | a working `mtime`/`size`/`exists` probe | polled for open buffers (freshness) |
| `useEditorFiles.js` | `text`, `baseline`, `dirty`, `mtime`, `stale`, `reloadFile()` | save, conflict state, rename/delete hooks |
| `useEditorMode.js` | editable while chat is idle, read-only while it streams | the same rule enforced **server-side** on writes |
| `MonacoEditor.vue` | `content-change`; `readOnly` flipped hot | `Ctrl+S`, save-all, model/EOL handling |
| `WorkspaceNavigator.vue` | context menu for **file rows only** | menu for folder + root rows; New/Rename/Delete |
| `useTerminal.js:366` | a `beforeunload` confirm, commented out | re-enabled on the dirty-buffer condition |
| any write route | none | `PUT /api/fs/file`, `POST /api/fs/create`, `POST /api/fs/rename`, `DELETE /api/fs/file` |

## Non-negotiable rules (each is asserted by a test)

1. **Writes are browser-only.** A route under `/api/fs` that mutates disk is
   reachable from the SPA and from nothing else: no MCP connector, no LSP tool,
   no `core/chat.py` code path may call it. AGENTS.md currently says "Tools are
   read-only by design"; that stays true, and ed-09 asserts it rather than
   trusting the prose.
2. **One gate.** Every write goes through `resolve_in_workspace(rel,
   for_write=True)` — the three locks plus the write flavour of
   `mcp_connectors/path_policy.deny_reason` (the stricter component list:
   `.git`, `.ssh`, `.pyenv`, `mcp_connectors`, plus the secret stores). No second
   list of protected paths, ever.
3. **No lost work.** A save compares the buffer's `mtime` (taken at read time)
   against the disk `mtime` and answers **409** carrying both. A dirty buffer is
   never silently overwritten by a reload, and never silently discarded on close
   or page unload.
4. **No partial writes.** A save is write-to-temp + `os.replace`, and the
   directory entry is fsync'd. A `truncated` or `binary` buffer is refused —
   never write back a file we only read half of.
5. **Delete is reversible.** `logs/trash/<sha12>_<stamp>/<relative path>` keeps
   the tree layout, is gitignored (same argument as `logs/fedit_backups`), and
   is swept after a retention window. A denied path is never moved: the subtree
   is checked first and **all-or-nothing**.
6. **The chat-stream gate holds for writes too.** A route refusing a write while
   a chat reply streams reuses `_CHAT_HANDLE` (`api/routes.py:341`), the same
   process-wide slot `/api/chat/pause` uses. The client-side rule in
   `useEditorMode` is the *display* of this, not its enforcement. Batch mode owns
   the conversation between turns with `phase === 'idle'`, so the client-side
   rule also checks `useBatch.state.active` (D7).
7. **Read behaviour does not move.** Every rule of the existing read surface
   (caps, binary answer, omitted denied siblings, the git queries) keeps its
   test; a task that breaks one is wrong, not the test.
8. **Frontend invariants hold**: run/stream state in composables, styles in
   `styles/theme.css`, the ARIA quality of the tree/tab strip, and the lazy
   Editor chunk stays lazy.
9. **Python style**: explicit annotations; patchable limits live as module
   attributes on `editorfs.constants` and tests patch them there.

## The task lot

`ed-00` is this file. `ed-01 … ed-09` are the tasks; they are strictly
sequential, each one leaving the app in a working, testable state.

| # | File | Delivers | Depends on |
|---|---|---|---|
| 1 | `ed-01-write-core-save.md` | `PUT /api/fs/file`: write gate, atomic write, mtime guard (409), chat-stream gate, backup; + the stale-comment/docstring cleanup | — |
| 2 | `ed-02-freshness-conflicts.md` | `/api/fs/stat` polling, `stale`/`conflict`, reload-or-overwrite UI | 1 |
| 3 | `ed-03-save-ux.md` | `Ctrl+S`, save/save-all, dirty dot, confirm-on-close, `beforeunload` | 1, 2 |
| 4 | `ed-04-create.md` | `POST /api/fs/create` + New file / New folder in the tree | 1 |
| 5 | `ed-05-delete.md` | `DELETE /api/fs/file` → trash, confirm dialog, subtree rule | 1, 4 (dialogs) |
| 6 | `ed-06-rename-move.md` | `POST /api/fs/rename`, both ends gated, open tabs re-pointed | 1 |
| 7 | `ed-07-navigator-ux.md` | folder/root menus, `refreshBranch`, keyboard chords, inline name input | 4, 5, 6 |
| 8 | `ed-08-write-feedback.md` | where a 4xx answer lands in the UI, and how it is dismissible | 3, 5, 6, 7 |
| 9 | `ed-09-tests-docs.md` | the guard suite (incl. "no tool can write"), Vitest, Playwright e2e, AGENTS.md rewrite | 1–8 |

Two tasks are worth calling out beyond the four headline features:

* **ed-02 is not optional.** Writing without knowing disk moved underneath the
  buffer is how this feature would destroy work: the WorkBench and the terminal
  are live writers of the same files. The probe route already exists, so the
  cost is a client loop and a conflict dialog.
* **ed-04 (create) comes before ed-05 (delete)** only so the dialogs and the
  tree's refresh plumbing exist once, for the friendlier operation first. The
  trash itself is ed-05's.

## Decisions taken for this lot

| # | Question | Decision |
|---|---|---|
| D1 | Scope of the write surface | Save, create, delete, rename — files **and folders**. |
| D2 | Delete semantics | Move to `logs/trash/<sha12>_<stamp>/…`, keeping the relative layout. Sweep after `TRASH_RETENTION_DAYS`; no restore UI in this lot (a manual move back is enough, and the folder is named in the confirmation). |
| D3 | Save conflict protocol | Optimistic concurrency on `mtime` + 409. No edit tokens, no CRDT, no server-side diff3. |
| D4 | Chat-stream gate | Kept, and extended to every write route. Client-side it also covers batch's idle gaps. |
| D5 | Line endings | What the editor says is what lands. Monaco's model `EOL` is set from the file; the server never normalises. |
| D6 | Undo / per-buffer view state | Out of scope. The pane keeps its single shared Monaco model and the documented `setValue` rule (`MonacoEditor.vue`); per-buffer undo history and scroll state stay as they are. |
| D7 | Batch-mode gap | The client rule becomes `phase === 'idle' && !useBatch.state.active`; the server gate stays authoritative. |
| D8 | Autosave | Not this lot. A dirty buffer is saved by an explicit action only. |
| D9 | Where refusals land | `useEditorFiles.state.message` stays the one-line answer for a *refused open*; a **write** failure gets a dismissible notice (ed-08) that does not disappear on the next click. |
| D10 | Task files | This lot is named `ed-` with `ed-00_big-picture.md`; the `_` between number and slug is one of the separators `core/batch.py` accepts, so the lot lists correctly. |

## Explicitly out of scope

Quick-open (`Ctrl+P`), search-in-files, command palette, LSP diagnostics in the
gutter, a file watcher, save-as / revert-to-saved, autosave, per-buffer view
state or undo history, copy/paste of files, drag-and-drop move or upload, tab
reordering, split editor views, a trash-restore UI, multi-window editing.

## Files this lot touches

```
src/core/editorfs/__init__.py   module docstring: the write half exists
src/core/editorfs/constants.py  WRITE caps, trash paths, retention
src/core/editorfs/policy.py     the write flavour goes live (docstrings)
src/core/editorfs/files.py      write_file, create/rename/delete, trash
src/api/routes.py               4 new routes; the docstrings of the 5 old ones
web/src/composables/useEditorFiles.js   save, conflict, mutation hooks
web/src/composables/useEditorMode.js    batch-aware editability
web/src/composables/useWorkspaceTree.js refreshBranch, selection semantics
web/src/composables/useTerminal.js      the beforeunload confirm
web/src/components/MonacoEditor.vue     keybindings, EOL
web/src/components/EditorArea.vue       save affordances, close confirm
web/src/components/WorkspaceNavigator.vue  menus, name input
web/src/components/EditorTreeDialogs.vue   NEW: rename/delete/conflict dialogs
web/src/styles/theme.css                the dialogs and the notice
tests/test_editorfs_write.py      NEW: the guard suite
web/tests/useEditorWrite.test.js  NEW: the composable logic
web/e2e/02-editor-write.spec.js   NEW: the user-visible loop
AGENTS.md                         three paragraphs, not one more
```

## How a write flows (the shape every task keeps)

```
browser action
   → composable (state first, so the UI never lies about what it did)
   → /api/fs/<route>
       1. resolve_in_workspace(rel, for_write=True)   ← three locks + deny-list
       2. _CHAT_HANDLE is set → 409  (a reply may be editing these files)
       3. route-specific precondition (mtime / exists / subtree)
       4. backup or trash, then the mutation (atomic where it can be)
       5. payload answers the new mtime/size, so the caller re-baselines
   → composable updates the record (mtime, baseline, dirty) and the tree
```

The core functions stay FastAPI-free (the "core owns logic, API owns transport"
rule): they raise `WorkspacePathError(status, detail)` and the route only turns
that into an `HTTPException`, exactly as the five read routes do.

## Testing strategy

* **Unit (pytest, `tests/test_editorfs_write.py`)** — the guards, one test per
  rule: traversal, deny-list write flavour, stale mtime → 409 with both mtimes,
  chat-stream 409 on all four routes, create-over-existing, rename across the
  policy boundary, recursive delete all-or-nothing, CRLF round-trip, the size
  cap, a symlink inside the workspace, trash retention.
* **The read-only-tools guard** — assert that no MCP/LSP tool list and no
  `core/chat.py` path reaches a write route. AGENTS.md's promise survives only
  while something asserts it.
* **Vitest (`web/tests`)** — the save/conflict/mutation logic of the
  composables, with `fetch` stubbed: the pure parts (conflict decision,
  re-baselining, path re-pointing) need no browser, matching the existing
  `useEditorFilesDiff.test.js` style.
* **Playwright e2e (`web/e2e`)** — create → type → save → reload → delete →
  trash, against the real app on its own port with `.env.test`, and asserting
  the read-only badge and the 409 path. No LLM call is needed for any of it.
* Hermetic by the repo's rule: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1`, patch
  caps through `editorfs.constants`, never through the app's own `todos/`.

## Definition of done for the lot

* The four operations work for files and folders, from the tree and the keyboard.
* Every refusal names its rule, on the wire and in the UI.
* No test of the read surface changed its expectation, and no connector or LSP
  tool gained any write ability.
* `pytest`, `cd web && npm test` and `cd web && npm run test:e2e` pass; AGENTS.md
  describes the app as it now is, in under ~300 lines.
