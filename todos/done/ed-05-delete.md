# ed-05 — Delete: to the trash, never to `unlink`

The third write operation: **Delete**, for a file *or* a folder (D1), and it moves
the tree into a trash folder instead of removing it (D2). One rule outranks the
convenience: a denied path is never moved, so the whole subtree is checked before
anything happens.

Depends on: ed-01 (write gate, chat gate), ed-04 (the tree's `refreshBranch`, the
dialogs file, the menu that is now parent-aware).

## 1. `DELETE /api/fs/file?path=<rel>`

(A `DELETE` with a query parameter, matching how the other `/api/fs` routes name
their path. No request body.)

Core: `editorfs.trash_path(rel)` in `core/editorfs/mutate.py`.

Steps, in this order — the order is the test list:

1. Gate the path with the write flavour. 422 naming the rule for `.git/…`,
   `mcp_connectors/…`, `.env`, `.ssh`, an escape, a NUL.
2. Resolve it (`resolve_existing_in_workspace`) → 404 when absent. Absent is not
   "already deleted, fine": the caller's view is wrong and needs to be told.
3. **Subtree check, all-or-nothing.** Walk the tree *without following symlinks*
   (`os.walk(followlinks=False)`) and check every descendant with
   `policy.writable()`. If **any** descendant is denied → 422 naming the first
   offending descendant, and **nothing at all** happens. Deleting `src/` because
   its name is innocent, while it contains `.env`, is exactly the accident this
   rule exists to prevent. The workspace-root check (`_check_workspace_root`)
   applies to deleting `""` — which is refused outright (400, "the working
   location itself cannot be deleted").
4. Symlink at the top level: `lstat` says symlink → refuse (400, "a symlink is
   not a file to delete" — the target may live outside the workspace and the
   policy cannot vouch for it). Removing a link is a human call for the terminal.
5. Chat-stream gate.
6. Move to trash (§2), then answer:
   `{"path", "dir", "trashedTo": "logs/trash/<batch>", "entries": N}` — `N` being
   the descendant count, so the confirmation the UI already showed matches what
   happened.

## 2. The trash

| Constant (in `editorfs.constants`) | Default | Notes |
|---|---|---|
| `TRASH_DIR` | `<app repo>/logs/trash` | app repo, not the workspace — same argument as ed-01's `BACKUP_DIR`: a workspace of `$HOME` must not gain a copy of itself |
| `TRASH_RETENTION_DAYS` | 30 | swept on the same reasoning as `fedit`'s `BACKUP_PRUNE_DAYS` (`fedit_mcp.py:1027`) |

Shape:

```
logs/trash/<sha12-of-resolved-path>_<%Y%m%d-%H%M%S.%f>/
    <the path exactly as it was, relative to the workspace>
```

Keeping the relative layout inside the batch directory is what makes a restore a
plain `mv` — with a bare `f.py` you cannot tell `src/f.py` from `web/f.py`, and a
name collision inside one batch would be a data loss. The digest keys on the
resolved path (the `fedit_mcp.py:976` reasoning: two spellings of one mount are
one file); the stamp makes batches unique.

* `shutil.move` on the top-level entry (it preserves the subtree), not a copy +
  `rmtree`. On a cross-device workspace that is slower and correct — correctness
  wins; note it in the docstring.
* Sweep: on each trashing, prune batches older than `TRASH_RETENTION_DAYS`.
  Reuse the *shape* of `prune_old_backups` (`fedit_mcp.py:1027`) including its
  rule that a failed delete never fails the operation — but do not import it
  across the connector boundary: the connector is a separate process and a
  `src/` → `mcp_connectors/` import is the direction this repo avoids (only
  `path_policy` is shared, and it is imported lazily, `editorfs/policy.py:53`).
* `logs/` is already gitignored, so the trash never appears in `git status` — the
  same reason `fedit_mcp.py:123` puts backups there.
* **No restore UI in this lot.** The confirmation names the trash folder and says
  the retention in days, so a mistaken delete is recoverable by the user with two
  commands. Restoring is a later feature (and it needs its own conflict rules).

## 3. The confirmation

In `EditorTreeDialogs.vue`, the shape `RegistrationMemoryDlg.vue` establishes:

* title `Delete <name>`; for a folder, a scope line: "this folder and everything
  inside it". If a listing for that folder is already cached
  (`useWorkspaceTree.state.nodes[path].entries.length`), it may add "(at least N
  items shown)" — labelled as a lower bound, because the tree is lazy and
  ignored/denied entries are not in it. Never walk the tree from the browser to
  count.
* The full relative path in a `<code>` so the user can see what is about to go.
* Buttons: **Cancel** (default, focused) and **Delete**. Enter runs the focused
  button, so focus decides the default — this is the one place the keyboard gets
  the destructive order wrong if you forget it.
* A second line: "moved to `logs/trash/…`, kept 30 days".

Open the dialog from the context menu (`Delete` item, and `Delete` for folder
rows too) and from the `Delete` key chord in `useWorkspaceTree.handleKey`.

## 4. What the client does with the result

`useEditorFiles.js` gains `trashPath(path)` (one function: the server decides from
the entry whether it was a file or a folder); `useWorkspaceTree` gets nothing new
beyond `refreshBranch`. The `entries` count in the answer comes from the server's
own walk — the browser never estimates it.

* every open record whose `path === path` **or** whose path starts with
  `path + '/'` is closed — silently, because the buffer no longer has a future
  (its file is in the trash). Mark them `deleted: true` for one frame if a
  transition reads better, but do not leave a tab pointing at a trashed path:
  `refreshBranch` will drop the row and `reveal()` would fail.
* a **dirty** buffer under the deleted path is the one case worth a second
  question: "this buffer has unsaved changes that will be lost" appended to the
  same dialog, not a separate one (two dialogs for one action is a bad afternoon).
* `refreshBranch(parentOf(path))` + `useGitStatus().load({ force: true })`.
* The notice (`state.notice`, ed-08 gives it its final shape) confirms:
  "Deleted `src/x.py` — kept in `logs/trash/…` for 30 days".

## 5. Tests

`tests/test_editorfs_write.py`:

1. delete a file → it is gone from the workspace, present under
   `logs/trash/<batch>/<rel>`, answer names the batch;
2. delete a folder → the whole subtree is under the batch, layout intact;
3. a folder containing `.env` (or `mcp_connectors/x`) → 422 naming the descendant,
   and the folder and its innocent files are **still there**;
4. deleting `""` → 400;
5. deleting a symlink → 400, link untouched;
6. deleting a denied path directly (`.git/config`) → 422;
7. deleting a missing path → 404;
8. a second delete of the same path → 404 (its first copy is already trashed);
9. two deletes → two batch directories, no collision;
10. two same-named files from different parents, deleted separately → both survive
    under their own batch (`<b1>/a/f.py`, `<b2>/b/f.py`) — no collision;
11. retention: an old batch is swept, a young one is not (patch
    `TRASH_RETENTION_DAYS`, set mtimes);
12. delete while a chat reply streams → 409, nothing moved;
13. the read routes and `tests/test_editorfs.py` unchanged.

Vitest: closing the right records (exact path and `path/` prefix), the dirty
buffer asking once, `refreshBranch` + git reload, a 422 leaving every record
alone.

## 6. Acceptance

* Right-click a file → Delete → Cancel does nothing; Delete moves it to
  `logs/trash/<batch>/…`, the row disappears, a notice names the trash path.
* Same for a folder with nested content; the layout under the batch matches.
* Deleting a folder that contains `.env` is refused and names the descendant; the
  folder survives.
* A dirty open buffer under the deleted path is warned about in the same dialog.
* `cd web && npm test`, `pytest` green.
