# mov-01 — `move_paths_batch` and `POST /api/fs/move-batch`

Build the whole backend half of the batch move: one core function that moves N files
into one existing folder **or refuses without touching disk**, the cap it is bounded
by, the route in front of it, and — the part most likely to be missed — the four test
lists that a fifth write route must be registered in. When this task is done the
endpoint is live and testable with `curl`; there is still no UI for it.

Depends on: **nothing** (this is the lot's first backend task).
Read `docs/tree-move-batch-spec.md` §2.5 in full (the body shape, the seven validation
steps, the response contract, and the two guards), plus §2.1's argument for one request
and §7's backend list. Decisions D11, D12, D13 and *Traps* 1, 2, 4 of
`todos/mov-00_big_picture.md` come from this task.

## Files

| file | change |
|---|---|
| `src/core/editorfs/constants.py` | `MAX_MOVE_BATCH = 200`, beside the other caps, with a comment in their voice (why a cap, what it bounds) |
| `src/core/editorfs/mutate.py` | `move_paths_batch`, directly after `rename_path`, sharing its helpers |
| `src/core/editorfs/__init__.py` | re-export `move_paths_batch` exactly the way `rename_path` is re-exported |
| `src/api/routes.py` | `FileMoveBatchIn` beside `FileRenameIn`, and `@router.post("/fs/move-batch")` beside `post_fs_rename` |
| `tests/test_editorfs_write.py` | a `TestMoveBatch` class right after `TestRename`; the two `writers` scans; `call_write` + `WRITE_KINDS`; one chat-stream-gate test |
| `tests/test_editorfs.py` | the `/api/fs` verb map in `TestSurfaceShape` gains `"/api/fs/move-batch": ["post"]` |
| `tests/test_write_surface_isolation.py` | `writers` tuple + `ALLOWED_HOMES` |

Seven files, and the reason it is not fewer: a new write route has to be registered in
four separate guards by design. Do not split the guard edits into a later task — they
*are* the review of this route, and a task that shipped the route without them would
leave the suite red.

## What to deliver

### The cap

`MAX_MOVE_BATCH = 200` in `editorfs/constants.py`. Read it **qualified at call time**
(`constants.MAX_MOVE_BATCH`), the way `tree.py` reads `MAX_TREE_ENTRIES_PER_DIR` — a
test patches the *name*, and `from .constants import MAX_MOVE_BATCH` would freeze a
copy the patch cannot reach.

### `move_paths_batch(paths: List[str], to_parent: str) -> Dict[str, Any]`

It is a **reuse of `rename_path`'s rules, not a second rename**: the same helpers, in
the same order, no new refusal sentence anywhere. Use `resolve_write_target`,
`resolve_parent_in_workspace`, `missing_path_error`, `_exists_answer`,
`_parent_missing_error`, `fsync_dir` — everything the rename calls. If this function
invents a message the rename already has, that is a bug: the rule must stay one
sentence everywhere.

The seven steps, in this order (the order **is** the test list; §2.5):

1. `paths` empty → **400**. Over `MAX_MOVE_BATCH` entries → **400** naming the cap.
2. Normalise `to_parent` (`normalise_rel`). The client refuses the no-op case; the
   server has no reason to refuse it — it falls out of step 6 as "occupied".
3. Every source: gated through itself **and its whole ancestor chain**, then must
   exist (**404** for the first absent one), must be a **file** and not a folder
   (**400** naming the path — R4 is enforced server-side so the route cannot be talked
   into moving a tree), must not be a top-level **symlink** (400, the rename's
   argument verbatim).
4. Join each target server-side — `to_parent + "/" + basename(source)` — and gate it
   through its whole ancestor chain. **The client never sends a joined target**
   (D11): add no field for one.
5. `to_parent` must already be a directory → **409** `_parent_missing_error`, reusing
   the existing sentence verbatim (R3).
6. No target occupied (`os.path.lexists`) → **409** `_exists_answer` with the
   `{"reason": "exists", "path", "dir"}` object. And no two sources resolving to the
   same target name → refuse rather than let the second overwrite the first. Because
   R1's same-parent rule makes that unreachable from the UI, this check is a guard
   against a hand-built body, not a feature (D13) — keep it three lines and no
   collision table.
7. **Only now** the renames, in the given order. The loop stops at the first
   `OSError` and the payload says what landed (D12) — never continue silently past a
   failure. `EXDEV` and `ENOENT` map to the existing 409 sentences. No backup is
   written: a move changes no bytes, which is `rename_path`'s argument restated.

Response, verbatim contract (mov-03's reconciliation loops over exactly this):

```json
{"moved": [{"from": "notes/a.md", "to": "notes/archive/a.md", "mtime": 1712345.0}],
 "to_parent": "notes/archive"}
```

`mtime` is the disk's own, per moved file — the same non-invented number the
`test_each_route_answers_what_a_restat_finds` property insists on. `to_parent` is the
normalised value, so the client re-lists the folder the server actually used.

A source and its target normalising to the same path is **not** special-cased: the
target exists (it is the source), so step 6 answers 409. §7 records this so nobody
"fixes" it by skipping.

### The route

```python
class FileMoveBatchIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    paths: List[str]
    to_parent: str
```

Docstring in `FileCreateIn`/`FileRenameIn`'s voice: why the sources are full paths and
the parent is a separate field (D11), and that `""` means the working location.

`@router.post("/fs/move-batch")`, placed immediately after `post_fs_rename`, inside the
guarded-write block. Its body is **transport only**:
`_refuse_if_chat_streaming("move files")`, then the core call, then the two `except`
clauses the rename has (`WorkspacePathExists` → `detail_object`, `WorkspacePathError` →
`detail`). If the handler grows past that, a rule has leaked into `routes.py`.
Docstring: one paragraph naming where the rules live (`move_paths_batch`), one sentence
on the statuses, one sentence saying that the transaction is the *ordering* — validate
all, apply none until all pass — and that the apply loop is not atomic across paths and
says so, and the closing "Browser-only, like the other write routes" sentence this file
carries on every mutation.

### The four registrations (Traps 1, 2, 4)

- `tests/test_editorfs.py` `TestSurfaceShape`: add `"/api/fs/move-batch": ["post"]`.
  That allowlist *is* the review of "this route is a write" — its docstring says a new
  mutation has to appear here on purpose.
- `tests/test_write_surface_isolation.py`: add `"move_paths_batch"` to the `writers`
  tuple in `test_the_write_core_is_called_from_the_routes_and_nowhere_else`, and add
  `Path("web/src/composables/useEditorMove.js")  # move batch` to `ALLOWED_HOMES`.
  The second is deliberate foresight: that module arrives in mov-03 and its `fetch`
  will name the route, so the marker scan would flag it otherwise. Add the entry with a
  comment naming the task that fills it in — do **not** create the file here.
- `tests/test_editorfs_write.py::TestWritesStayBrowserOnly`: the same `writers` tuple
  is duplicated in that file's own copy of the scan. Add the name there too, or the
  suite goes red on a green guard.
- `tests/test_editorfs_write.py`'s matrix: add a branch for each end in `call_write`
  and one kind each in `WRITE_KINDS` — a source-side (`paths: [path]`, an innocent
  `to_parent`) and a destination-side (`paths: ["README.md"]`, `to_parent: path`). That
  is what makes the escape and locked-path properties travel to the new route "written
  once here". Do **not** add a batch row to
  `test_no_route_silently_replaces_or_reduces_what_is_already_there` (Trap 4): it would
  hit the same-path case the spec refuses to exempt.

### Tests — `TestMoveBatch`, beside `TestRename`, mirroring its list

Each of these is one test; the second clause of each is the R2 proof (§7):

- two files land, both readable at the new path, **no backup written**;
- one occupied target → 409 with the `{reason: "exists"}` object **and both files still
  at their original paths** — the test that proves the ordering;
- one `ro` source, one `hidden` source, one `ro` destination, one `hidden` destination →
  refused in the policy's own words, nothing moved;
- destination missing → 409 naming it; destination is a file → the same 409 sentence
  (R3 says "must already be a directory");
- a folder in `paths` → 400 and the folder survives whole;
- a symlink in `paths` → 400;
- an escape on either field (`..`, absolute, `~`, NUL) → 422;
- empty `paths` → 400; over-cap → 400 naming `MAX_MOVE_BATCH`, reached by
  `monkeypatch.setattr(constants, "MAX_MOVE_BATCH", 2)`;
- a source whose name is duplicated in `paths` and two sources that would land on one
  target name → refused, nothing moved (the step-6 guard, hand-built body);
- a chat reply streaming → 409 carrying `while a chat reply is streaming`, in the
  `TestChatStreamGate` style (arm/clear the handle, then assert every source still on
  disk);
- source parent == destination → 409 occupied, **not** a 200 no-op;
- malformed bodies (missing `paths`, `paths` not a list, `to_parent` absent) → 422.

Hermetic in the project's sense: the workspace is the suite's throw-away `tmp_path`,
never this checkout; `BENCHMARKIT_DISABLE_MCP`/`_LSP` come from `tests/conftest.py`.

## Rules this task must not break

- *Core owns logic, API owns transport*: `move_paths_batch` takes plain values, returns
  a dict, imports no FastAPI. `src/core` must keep importing no `src.api` — the guard
  in `test_write_surface_isolation.py` fails otherwise.
- *A `/api/fs` write route is reachable from the SPA only*: no connector, hub, LSP tool
  or chat module may name `POST /api/fs/move-batch` or call the new core function. The
  marker scan and the `writers` scan are the enforcement — that is why they are in this
  task's file list and why the scan's `callers` assertion must still be exactly
  `routes.py`.
- *Every write route is refused while a chat reply streams.*
- *Path rules come from two levels, one verdict*: ask the same verdict `rename_path`
  asks, through the same helpers. No new deny-list, no new refusal wording, no copy of
  `restricted-paths.txt` or the workspace policy.
- *Python style*: explicit annotations and `typing` imports (`List[str]`,
  `Dict[str, Any]`), no bare containers.
- The file's own ordering rule: a hidden path answers 404 to a read and 422 naming the
  rule to a **write**. Every entry of a batch is a write, so no batch entry may answer
  the 404-for-hidden form.

## Done when

```bash
pytest tests/test_editorfs_write.py -q          # TestMoveBatch green, everything else green
pytest tests/test_editorfs.py -q                # the verb map, widened on purpose
pytest tests/test_write_surface_isolation.py -q # writers + ALLOWED_HOMES updated
pytest -q                                       # the whole suite green
```

And: `curl -s -X POST /api/fs/move-batch -d '{"paths":["a.md","b.md"],"to_parent":"docs"}' -H 'Content-Type: application/json'`
against a dev server returns 200 with the `moved` array, and a second identical call
returns the 409 `exists` object with both files where they were.

## Out of scope here

anything in `web/` except the one `ALLOWED_HOMES` line; moving folders; copy; undo; an
implicit `mkdir`; a collision table beyond step 6's three lines; SSE; a batch of
batches. The frontend does not exist yet — do not add a menu item, a dialog or a
composable to "make it usable".
