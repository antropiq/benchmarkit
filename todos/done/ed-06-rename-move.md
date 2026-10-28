# ed-06 — Rename and move: both ends of the path are gated

The fourth write operation: **Rename** (and therefore **move**, same route, since
a rename with a different parent *is* a move) — for files and folders (D1).

This is the operation with the most ways to get the policy wrong, because two
user-supplied paths describe one filesystem action. Everything else in this task
is what happens to the app's own view of the moved thing.

Depends on: ed-01 (gate, chat gate, helper names), ed-04 (`refreshBranch`, the
inline name input), ed-05 (the notice pattern).

## 1. `POST /api/fs/rename`

```jsonc
POST /api/fs/rename
{ "from": "src/old_name.py",
  "to":   "src/sub/new_name.py" }   // full workspace-relative target, not a name
```

Taking the **full target path** (not just a new name) is what makes move work,
and it means the target's parent needs the same treatment as everything else.

Core: `editorfs.rename_path(frm, to)` in `core/editorfs/mutate.py`.

Steps, in order — and the order is the test list:

1. Gate `from` (write flavour) → 422; resolve it, 404 when missing; a symlink
   (lstat) → 400 (the ed-05 argument: the target may leave the workspace).
2. Gate `to` **and** `to`'s whole ancestor chain (write flavour) → 422. This is
   the rule that catches `to = ".git/hooks/pre-commit"` and
   `to = "mcp_connectors/policy_patch.py"` while `from` is innocent.
3. The target's parent must exist → 409 `"Parent folder does not exist"` (no
   implicit `mkdir`, the ed-04 rule).
4. `from` and `to` normalise to the same path → 400 `"Source and target are the
   same path"`. (Case-only renames on a case-sensitive filesystem are legal and
   must work: `Readme.md` → `README.md` is not the same normalised path here.)
5. **A folder cannot be moved into itself.** `to` starts with `from + '/'` → 409
   naming the rule (`os.rename` would fail with EINVAL anyway, but the answer must
   say why). The near-miss cases are legal and must be tested as legal: `src` →
   `src2`, and `src/sub` → `src/sub2` (a sibling whose name starts with the same
   string) — the prefix check needs the trailing `/` precisely so those pass.
6. Target exists (file, folder, dangling symlink) → 409
   `{"reason": "exists", "path": to}`. Never `os.rename`'s silent overwrite.
7. Refuse to rename the workspace root (`from == ""`) → 400.
8. Chat-stream gate.
9. `os.rename(from, to)` — same filesystem, so atomic. If it raises `EXDEV`
   (a bind mount inside the workspace), refuse with 409 naming it: **do not**
   fall back to copy+delete, which is not atomic and would lose the mode/xattrs
   that make the file what it is.
10. Answer `{"from", "to", "dir", "size", "mtime"}`.

No backup for a rename: the file's content did not change, and the trash + the
git status already describe the move. Say that in the docstring so it does not
read as an oversight — the save path backs up, this one deliberately does not.

## 2. The app's own view of the moved path

The interesting half of this task. Five things hold a path:

| Holder | What to do |
|---|---|
| open file records (`useEditorFiles.state.open`) | `repointPath(from, to)`: for every record whose `path === from` **or** starts with `from + '/'`, set `path`, `key` (a file's `key` **is** its path), `name`, and re-derive the diff tab's `key` (`diff:` + new path). `mtime` is unchanged for the file itself — the rename does not touch its content — so no record becomes stale. |
| `state.active` | it is a *key*, so update it in the same pass, or the pane loses the buffer it is showing. |
| the tree | `dropBranch(from)` + `refreshBranch(parentOf(from))` + `refreshBranch(parentOf(to))` (ed-07 formalises `dropBranch`; here implement the minimal version: delete the cache/expanded keys under the old prefix). Then `reveal(to)`. |
| git status | `useGitStatus().load({ force: true })` — a rename is a git event. |
| the WorkBench's open context | nothing: the chat keeps no file state. A prompt that named the old path stays as written; that is correct, it is history. |

Also: renaming the file of the **active** tab must not change what is on screen —
same buffer, same text, new label. Assert the `key`/`active` pair, because that is
the detail that breaks silently.

## 3. Rename in the UI

* Inline input in the tree (ed-04's widget), prefilled with the current name,
  extension left unselected. Typing a `/` in it is how a **move** is expressed —
  and the input must then validate the *target path* (no `..`, no absolute) the
  same way the tree validates any path, rather than the create rule ("no `/`").
  So the validator is per-mode: `create` forbids `/`, `rename` allows a
  workspace-relative path and still forbids `..`.
* Menu item `Rename…` on file and folder rows; chord `F2` (added in ed-07's
  keyboard table — if ed-07 has not run yet, add the one line here and let ed-07
  fold it into its table).
* A 409/422 answer puts the server's text **inside the row** (`.tree-row-error`)
  and keeps the input open with its value, so the user can fix the name rather
  than start over.

## 4. Tests

`tests/test_editorfs_write.py`:

1. rename a file in place → the bytes and `mtime` are unchanged, the name moved;
2. move a file to another existing folder → same;
3. rename a folder (with children) → the whole tree moved;
4. `to` under `.git/…`, `mcp_connectors/…`, or a denied ancestor → 422, nothing
   moved (the both-ends rule, asserted from both directions: innocent `from`,
   denied `to`);
5. target exists → 409, both paths as they were;
6. missing target parent → 409 naming it;
7. `from === to` → 400; a case-only rename → succeeds;
8. folder into itself → 409 naming the rule; a legal sibling move with a similar
   name → succeeds;
9. `from = ""` → 400; a symlink → 400;
10. rename while a chat reply streams → 409, nothing moved;
11. no backup is written for a rename (the deliberate rule);
12. the read routes and `tests/test_editorfs.py` unchanged.

Vitest (`web/tests/useEditorWrite.test.js` or a `useEditorRepoint.test.js`):

1. `repointPath` updates `path`, `key`, `name` of the file record;
2. the diff tab of the same file gets the new `diff:` key and stays the same tab
   kind;
3. a record under a renamed **folder** (`src/x.py` when `src` → `web`) is updated;
4. `state.active` follows the key;
5. `mtime`/`dirty`/`text` are untouched by a rename;
6. the tree calls are made (drop + two refreshes + reveal), and a failure leaves
   every record pointing at the old path.

## 5. Acceptance

* Right-click a file → Rename… → new name → the row moves, the tab's label
  changes, the buffer is untouched and still dirty if it was.
* Same for a folder: its children stay expanded under the new name.
* Renaming `x.py` to `.env`, or into `.git/…`, is refused with the connector's
  wording, and the file is where it was.
* Renaming `todos/` to `todos/x/y` when `todos/x` does not exist → refused, and
  the message names the missing parent.
* A dirty buffer of the renamed file stays dirty and saves to the new path
  (that is the real end-to-end proof: the token, the path and the buffer all
  agree).
* `cd web && npm test`, `pytest` green.
