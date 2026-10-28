# ed-01 — The write core, and the first write: save

Open the write half of `core/editorfs/` and deliver exactly one user-visible
capability: **saving a file from the Editor pane reaches disk**. Everything the
other tasks need — the write flavour of the policy gate, the mtime guard, the
chat-stream refusal, backups — is established here, because a second task
inventing its own gate is how a "one policy per surface" invariant dies.

Depends on: nothing (it is task 1). Everything else depends on it.

Also done here: the documentation/comment cleanup (§4). It is a docstring change,
it belongs to the task that makes the docstring false-to-true, and doing it in a
later task leaves the codebase lying for eight steps.

## 1. The write gate

`resolve_in_workspace(rel, for_write=True)` already exists and is never called
(`src/core/editorfs/policy.py:123`). Use it. Three extra rules the read gate does
not need:

| Helper (new, in `policy.py`) | Rule |
|---|---|
| `resolve_parent_in_workspace(rel)` | resolves the **parent** of `rel` and requires it to be an existing directory, then returns `(normalised_rel, parent_path, name)`. A create targets a name that does not exist yet, so `resolve_in_workspace` on the leaf cannot be the check. |
| `resolve_existing_in_workspace(rel, *, for_write=True)` | the gate plus a `404` when the path is absent — the shape save/rename/delete all want, and it keeps the 404 wording in one place. |
| ancestor chain | for a mutation, the whole chain from the workspace root down to the leaf is checked with `writable()` — a rename *into* `.git/x` is refused by the leaf check, but a write whose parent is denied and whose leaf name is innocent must be refused too. |

Write-flavour refusals quote the same `deny_reason(...)` text the connectors
quote, with the verb already handled by that module (`"… which connectors never
write to"`). Do not re-word the deny-list.

## 2. `PUT /api/fs/file` — the save route

Request body (add a Pydantic model next to the other `api/routes.py` input
models, so a malformed body is a 422 from FastAPI, not a 500 from the core):

```jsonc
PUT /api/fs/file
{ "path": "src/config.py",
  "text": "…the whole buffer…",
  "ifUnmodifiedSince": 1761371340.0 }   // the mtime GET /api/fs/file reported
```

Rules, each one mirrored by a test in §5:

1. **Gate first.** `resolve_existing_in_workspace(path, for_write=True)` → 422
   naming the rule for an escape/absolute/NUL, or a deny-listed path.
2. **Chat-stream gate.** If the module-level `_CHAT_HANDLE`
   (`api/routes.py:326-341`) is set → **409** `"Cannot save while a chat reply
   is streaming"`. This is the same slot `POST /api/chat/pause` uses and the same
   pattern the MCP/LSP restart and path-location routes already use
   (`routes.py:885`, `1026`, `1110`). Extract the check into one small helper
   (e.g. `_refuse_if_chat_streaming()`) and use it in the new routes; do not
   copy the four-line body a fifth time.
3. **mtime guard.** `stat(resolved).st_mtime` differs from `ifUnmodifiedSince`
   → **409** whose `detail` is a JSON object (not a string) so the client can act
   on it without parsing prose:
   `{"reason": "stale", "path": …, "diskMtime": …, "bufferMtime": …}`.
   Absent `ifUnmodifiedSince` means "no expectation" and skips the guard — but
   the client always sends it, and the route logs a save without a token (it is
   the shape of a client bug, not a feature).
   Tolerance: compare with a small epsilon (the filesystem's own resolution),
   not `==`; use a named constant `MTIME_GUARD_EPSILON = 0.01`.
4. **A partial buffer never goes back.** If the record was read `truncated` or
   `binary` the client must not save — and the server cannot know that, so the
   client refuses first **and** the server has its own check: if the bytes
   currently on disk exceed `MAX_EDITOR_BYTES`, refuse with 413 (same message
   shape as the read route). Writing over a file we never fully read is data
   loss, whatever the buffer says.
5. **Size cap on the way in.** The payload's UTF-8 length is checked against
   `MAX_EDITOR_BYTES` before anything is written → 413. A 400 MB text body from a
   confused browser tab is refused the same way the read route refuses it.
6. **Backup before mutation.** `backup_for_write(resolved)` (see §3).
7. **Atomic write.** `tempfile.NamedTemporaryFile(dir=parent, delete=False)` in
   the **same directory** (a cross-device temp would make `os.replace` a copy),
   write the UTF-8 bytes, `flush` + `os.fsync`, `os.replace`, then `os.fsync` the
   parent directory fd so the rename itself is durable. Chmod the temp to the
   existing file's mode before the replace — a save must not turn a `0755`
   script into `0600`.
8. **Answer.** `{"path", "size", "mtime"}` with the **new** values, re-`stat`ed
   after the replace. The client re-baselines on this, so the next save's token
   is the server's truth and not a guess.
9. **Encoding.** The text is written as UTF-8 with **no** normalisation: `\r\n`
   in, `\r\n` out (D5). The read route already promises exact characters
   (`files.py:77`); the write route keeps that promise symmetric. Do not touch
   the BOM either — if the buffer holds one, it was read from disk.

Line endings worth a test rather than a comment: Monaco emits exactly what its
model holds, so a file that starts as CRLF stays CRLF through a save. That is a
round-trip assertion, not a conversion feature.

## 3. Backups, and the constants that own them

Add to `src/core/editorfs/constants.py` (module attributes, patchable — the
existing convention, see the docstring at `constants.py:3`):

| Constant | Default | Purpose |
|---|---|---|
| `BACKUP_DIR` | `<app repo>/logs/editor_backups` | where a pre-save copy lands. **App-repo**, not workspace — the workspace may be `$HOME`, and a backup tree inside it would be a copy of the user's home. |
| `BACKUP_KEEP` | 10 | copies kept per file, older ones pruned |
| `BACKUP_PRUNE_DAYS` | 30 | age after which any backup is swept |
| `MTIME_GUARD_EPSILON` | 0.01 | the mtime comparison tolerance (seconds) |

Reuse `fedit`'s proven naming rather than inventing one
(`mcp_connectors/fedit_mcp.py:976-1025`): `<sha12-of-resolved-path>_<safe-name>_<%Y%m%d-%H%M%S.%f>.orig`,
keyed on the resolved path so the two spellings of one mount share a chain,
`shutil.copy2` followed by an `os.utime` reset (a `copy2` inherits the source
mtime, which would birth an old file's backup past the age limit — the comment at
`fedit_mcp.py:1008` explains exactly this trap; the new copy needs the same
comment or it will be "fixed" later).

Two differences from fedit, both deliberate:

* **Not the same folder.** `logs/fedit_backups/` is that connector's chain and a
  human's `revert_edit` reaches it by name; mixing app writes into it makes
  `list_edits` answer about files the connector never touched.
* `logs/` is gitignored already (`.gitignore`), so nothing new to declare, and
  the same argument applies that `fedit_mcp.py:123` makes: backups must not
  pollute a working tree or show up in `git status`.

## 4. Documentation and comment cleanup (do it here, not later)

The read-only decision (old "D4") is written into the code, and this task makes
those sentences false. Rewrite them in the same commit:

| Where | Today says | Must say |
|---|---|---|
| `core/editorfs/__init__.py:11` | "ed-10, a future version, adds the guarded write — see decision D4" | the write half is here: `write_file`, and the operations added by later tasks |
| `core/editorfs/__init__.py:24` | "**nothing here writes**" | nothing here writes *except* the guarded write surface, which is browser-only |
| `core/editorfs/files.py:10` | "The write half belongs to a future version (ed-10)" | drop the version reference; name the write entry points |
| `core/editorfs/policy.py:26-27` | "this lot has no write route at all (decision D4), so nothing calls it" | the write flavour is live; `writable()` gates writes |
| `core/editorfs/policy.py:159-161` | "Nothing in this lot writes … keeps the read-only stance explicit" | same: no task-number references, describe the rule |
| `api/routes.py:1172-1173` | "there is no write route under `/api/fs` in this lot" | the read routes are read-only; the write routes are the ones added here, browser-only |
| `api/routes.py:1191-1194` | "the write guard's token for the write layer of ed-10" | `mtime` is the token the save guard compares; no task numbers |
| `AGENTS.md:18-21` | "the *file* surface is read-only (no write route…)" | one bounded write surface for the browser; tools still read-only. **Final wording is ed-09's job** — here, only delete the sentence that is now false. |

**No task-number references anywhere in `src/`, `web/src/` or `tests/`.** The
previous lot's files were deleted, so every "(ed-06's contract)", "(ed-03
consumes this)" or "ed-09's live reload" comment left in the tree points at a
document that no longer exists — it is the second thing this cleanup fixes, and
it is bigger than the write half:

```
web/src/composables/useWorkspaceTree.js  4 comments
web/src/composables/useEditorFiles.js    5 comments
web/src/composables/useTerminal.js       4 comments
web/src/composables/useBenchmark.js      1 comment
web/src/components/WorkspaceNavigator.vue 2 comments
web/src/components/EditorView.vue        2 comments
web/src/components/EditorTerminal.vue    3 comments
web/src/App.vue                          2 comments
web/src/styles/theme.css                 4 comments
src/core/proctree.py                     1 comment
src/core/editorfs/{__init__,files,policy,tree}.py  8 comments
src/core/termhub/{__init__,session,constants,policy}.py  6 comments
src/api/routes.py                        4 comments
tests/test_terminal.py, tests/test_editorfs.py       2 comments
```

Replace each with what the rule *is*, not which document states it — e.g.
"ed-09's live reload compares this" → "the freshness probe compares this". Grep
for `ed-[0-9]` in those directories must come back empty when this task ends —
with one exemption: `tests/test_batch.py`, whose `ed-` names are synthetic lot
fixtures for the batch parser, not references to a document.

Two comments are *kept* on purpose, because the rule they describe still holds:
`core/termhub/__init__.py:14` ("tools are read-only by design… while a real
shell exists") and `MonacoEditor.vue:35` ("read-only by default"). The second one
changes its reason in ed-03 (a pane that *can* save is a different default), not
its safety.

## 5. Tests for this task

New file `tests/test_editorfs_write.py`, hermetic (patch the workspace through
`config.path_location`, patch caps through `editorfs.constants`;
`BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1`). One test per rule:

1. save writes the bytes and answers the new `mtime`/`size`;
2. CRLF round-trip: read → save → bytes identical;
3. traversal / absolute / NUL → 422 naming the rule;
4. deny-list write flavour: `mcp_connectors/…`, `.git/…`, `.env*`, `id_rsa` →
   422 naming the entry; the same path is *readable* and that stays true;
5. parent denied, leaf innocent → refused (the ancestor-chain rule);
6. stale `ifUnmodifiedSince` → 409, detail carries both mtimes, disk untouched;
7. fresh token → succeeds; a save after a successful save re-baselines and works
   (the token round-trip);
8. chat streaming → 409 on the save route (set the handle the way the pause tests
   do);
9. over-cap payload → 413, disk untouched;
10. over-cap file on disk → 413 even with a fresh token;
11. file mode preserved (`0755` stays `0755`);
12. a backup exists after the save, and a second save keeps two;
13. a symlink inside the workspace pointing outside → 422 (the realpath lock);
14. `PUT` to a directory → 400; to a missing path → 404;
15. the five read routes still answer exactly as `tests/test_editorfs.py` says
    (that file must not be edited by this task — if it needs editing, the change
    is wrong).

## 6. Acceptance

* `PUT /api/fs/file` works against a real workspace; `curl` shows the 409.
* `pytest` green, `tests/test_editorfs.py` unchanged.
* `grep -rn "ed-[0-9]" src web/src tests` → empty.
* `grep -rni codemirror src web/src tests` → empty (it already is in the tracked
  tree; the check exists so the cleanup does not regress).
* The app behaves exactly as before for a human: no UI change lands in this task
  except that a pane which was already editable can now actually save, via the
  composable hook ed-03 wires to a key. If you add no keybinding here, say so in
  the commit message — do not leave a half-wired shortcut.
