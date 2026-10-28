# ed-04 — Create: a file or a folder, from the tree

Add the second write operation: **New file** and **New folder**, for files *and*
folders (D1). This task also lands the two pieces of tree plumbing every later
mutation needs — a per-branch refresh and an inline name input — so do them once,
here.

Depends on: ed-01 (the write gate, the chat-stream gate, the constants), ed-02 and
ed-03 (the dialogs file, the notice pattern, the save path a new empty buffer
will use).

## 1. `POST /api/fs/create`

```jsonc
POST /api/fs/create
{ "path": "src/new/thing.py",   // the path to create, parent must exist
  "dir": false }                // true → mkdir instead of touch
```

Core: `editorfs.create_path(rel, *, is_dir: bool)` in a new
`src/core/editorfs/mutate.py` (keep `files.py` about reading; the write half
deserves its own module, sibling to `tree.py`/`files.py`).

Rules:

1. `resolve_parent_in_workspace(path)` (ed-01) → the parent must exist and be a
   directory; if it does not, **409** `"Parent folder does not exist: …"` rather
   than creating a chain (a typo in a dialog becomes a stray tree otherwise).
   Creating intermediate folders is the user's choice in a *later* call, one
   `POST` per level, each one gated — never a `mkdir -p` side effect.
2. The whole ancestor chain plus the leaf name go through the write flavour of
   the policy. `mcp_connectors/x.py`, `.git/config`, `.env`, `id_rsa` → 422
   naming the entry.
3. **Never overwrite.** The path exists (file, folder, or a dangling symlink) →
   **409** with `{"reason": "exists", "path": …}`. A create that silently
   truncates an existing file is the worst possible bug in this task.
4. Chat-stream gate (same helper as ed-01).
5. `dir: true` → `os.mkdir`; `dir: false` → create an **empty** file with
   `os.open(..., O_CREAT | O_EXCL | O_WRONLY)`. `O_EXCL` is the race guard: two
   tabs, one winner, the loser sees the 409.
6. Answer `{"path", "dir", "size", "mtime"}` — the same shape a save answers, so
   the client's re-baselining code has one shape to know.

`IGNORED_DIR_NAMES` is not a write rule: creating a folder called `node_modules`
is legal and simply invisible afterwards (the read surface hides it). Say so in
the docstring, because it looks like a bug from the browser.

## 2. The name input, in the tree

An inline input on the tree rather than a modal form — it is where the new row
will appear, and the parent is whatever the user pointed at.

* `useWorkspaceTree.js` gains `creating: { parent, isDir }` and
  `startCreate(parent, isDir)` / `cancelCreate()`. The navigator renders one
  input row inside that parent's listing (`.tree-row.creating`), pre-focused,
  `Enter` submits, `Escape` cancels, and the row it belongs to is expanded first
  (`expand(parent)` then `load(parent)` if not loaded — `reveal()` already knows
  how to make a branch exist).
* Validation before the request (fast feedback, and the server re-checks
  everything): no `/`, no `\0`, not `.` or `..`, not empty, not a name already
  present in that listing, and — the one that surprises people — the name may
  *look* legal and still be refused by the policy (`.env`, `id_rsa.pem`). That
  one arrives as a 422 and must be shown, not pre-empted by a client-side copy of
  the deny-list. **No second list of protected names in the frontend.**
* On success: `refreshBranch(parent)`, then `reveal(newPath)` and — for a file —
  `openFile(newPath)` so the empty buffer is a tab immediately, ready for
  `Ctrl+S`. For a folder: expand it, and select its row.

## 3. `refreshBranch(path)` in the tree

Today the only refresh is the whole-tree one (`useWorkspaceTree.refresh()`,
`:144`), which re-lists every opened folder. A mutation touches **one** listing:

```
refreshBranch(path)   // path is a directory ("" for the root)
  if (state.nodes[path]) load(path, { force: true })   // only if it was loaded
```

Keep `refresh()` for the button and the working-location change; they answer
different questions. After a mutation, also re-run `useGitStatus().load({force})`
(a new untracked file should be marked like any other untracked file).

## 4. Menus and chords

The context menu currently refuses folder and root rows
(`WorkspaceNavigator.openMenu` returns when `row.dataset.dir === 'true'`,
`:155-157`). That guard goes away here, and the menu becomes parent-aware:

| Row | Items added in this task |
|---|---|
| folder | New file…, New folder…, Copy relative path |
| root (`""`) | New file…, New folder… (the root row is synthetic; `entryFor('')` already returns it) |
| file | (unchanged for now — ed-05/ed-06 add Rename/Delete here) |

The menu's `menuCan*` computeds must therefore key on the row's `dir` flag, not
on "always a file" as the current comment asserts (`:150`). Update that comment.

Chords, handled by `useWorkspaceTree.handleKey` (which already owns the tree's
keyboard model, `:316`): `n` → New file in the focused folder (or in the focused
file's parent), `N` (shift) → New folder. Keep the model's existing shape: one
`if/else if` chain, `preventDefault()` only for handled keys, and
`Enter`/`Space` keep meaning select/open.

Note that `Delete` and `F2` arrive in ed-05/ed-06 — leave the chain shaped so
they are one line each to add.

## 5. Tests

`tests/test_editorfs_write.py`, extended:

1. create a file → an empty file exists, answer carries the new `mtime`;
2. create a folder → it is a directory;
3. over an existing file / folder / dangling symlink → 409, nothing changed;
4. missing parent → 409 naming it, no chain created;
5. `mcp_connectors/new.py`, `.git/x`, `.env`, `sub/.ssh/y` → 422 naming the rule;
6. a payload whose leaf carries a `/` (a path, not a name) → the server treats it
   as a nested path, so it answers 409 "parent folder does not exist"; the
   client-side validator is what stops a name with a `/` reaching it at all;
7. create while a chat reply streams → 409;
8. a name `IGNORED_DIR_NAMES` hides → the create succeeds and the tree still
   hides it (assert both halves, that is the documented behaviour);
9. the read routes are unaffected (`tests/test_editorfs.py` untouched).

Vitest, in `web/tests/` (a new `useWorkspaceTreeCreate.test.js` or the write
suite, your call — keep the naming honest):

1. `startCreate` then submit issues one `POST` with the joined path;
2. a name already in the listing is refused client-side, no request;
3. a name containing `/` is refused client-side;
4. a 422 from the server surfaces the server's text (no local rule invented);
5. success calls `refreshBranch(parent)` and, for a file, `openFile(newPath)`;
6. `refreshBranch` on an unloaded folder issues no request.

## 6. Acceptance

* Right-click the root → New folder → type a name → the folder appears, expanded.
* Right-click a folder → New file → type → an empty tab opens, dirty-free; type
  into it, `Ctrl+S` saves it (ed-03's path, with a create's fresh `mtime`).
* `n` / `N` do the same from the keyboard, on the focused row.
* Trying to create `.env` or anything under `mcp_connectors/` is refused, with
  the connector's own wording visible.
* Trying to create over an existing file changes nothing and says so.
* `cd web && npm test`, `pytest` green.
