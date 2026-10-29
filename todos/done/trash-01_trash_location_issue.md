# trash-01 — The trash belongs to the workspace, not to `logs/`

Two defects in one behaviour, reported from the Editor: with the working location
pointed at another project, deleting `test.txt` lands the copy in
`benchmarkit/logs/trash/396ac179e554_20261029-104031.653354`.

| # | Defect | Cause |
|---|---|---|
| 1 | The trash sits under `logs/`, a folder whose name promises *text logs* | `TRASH_DIR = PROJECT_ROOT / "logs" / "trash"` (`editorfs.constants:95`), chosen before `.nojae/` existed |
| 2 | One trash for every workspace: deleting in `~/dev/tetris-game` files bytes inside the benchmarkit checkout | the same constant is a *module-level* path, resolved once at import and never re-read against the active workspace |

Both have the same fix: the trash moves into the workspace's own reserved folder,
`<workspace>/.nojae/trash`, resolved **per call** from `workspace_root()`.

```
/home/vgr/dev/benchmarkit   ->  /home/vgr/dev/benchmarkit/.nojae/trash/<batch>/test.txt
/home/vgr/dev/tetris-game   ->  /home/vgr/dev/tetris-game/.nojae/trash/<batch>/test.txt
```

No behaviour of the delete changes: it is still a *move* (never `unlink`), still
one dated batch per delete, still keyed on the digest of the resolved path, still
keeping the workspace-relative layout inside the batch so a restore is a plain
`mv`, still swept after `TRASH_RETENTION_DAYS`.

Depends on: nothing — pol-05 (`.nojae` seeding) and ed-05 (the delete route) are
both done. This task edits both.

## 1. Why `.nojae/trash` answers the old objection

The comment at `constants.py:90-94` rejects a trash inside the workspace for two
reasons: a working location of `$HOME` would gain "a second copy of the user's
home", and the artefacts would show up in that project's `git status`. Both are
answered by the folder this task moves the trash into:

* the move is a **rename inside one tree** — the trash holds exactly the paths the
  user deleted, nothing more; it never duplicates what is still on disk. A
  workspace of `$HOME` gains `~/.nojae/trash/<batch>/Documents/x`, which is a
  copy of one deleted file, kept 30 days, and it costs no extra I/O;
* `.nojae` is **reserved by code** (pol-00 D5): hidden from the tree, the read
  route and the write route, in *every* workspace, whatever the policy file says,
  and a level-B rule naming it is rejected by pol-04's validation. So the trash
  cannot become a browsable second copy — the invariant is stronger than the one
  it replaces, where hiding it took a level-A rule (`logs/**`, builtin_policy:81)
  that was only live in benchmarkit's own checkout.

Per-workspace is also the *correct* semantics: a batch is recoverable by `mv` into
the tree it came from, and a foreign project's deleted files have no business
being written inside the app's install directory (which can be read-only, on
another filesystem, or shared between users).

Wrinkle to accept and document, not to fix: a **foreign** workspace has no
`.nojae/` in its `.gitignore` (that entry lives in benchmarkit's own repo), so the
user's `git status` in a terminal will show `.nojae/` as untracked. The app itself
stays clean — `gitstatus` re-filters every reported path through the verdict, and
`.nojae` is hidden, so no marker is ever drawn for it (`gitstatus.py:19-23`).
Do **not** write to a foreign repo's `.git/info/exclude` or `.gitignore`: the app
has no mandate to edit a repository it was pointed at. Say so in the README.

## 2. The constant becomes a resolver

Keep the patchability convention (`lsphub.constants` style: an attribute exists so
a test can move the lookup) while making the value depend on the *active*
workspace. Do not put the resolver in `constants.py` — that module must not import
`.policy` (the package already imports `constants` from `policy`, and a cycle
through a partially-initialised module is not worth one helper). `mutate.py`
already imports both sides.

```python
# editorfs/constants.py  (replaces the TRASH_DIR literal)
TRASH_SUBDIR: str = "trash"

# None = derive from the active workspace: `<root>/.nojae/trash`, per call.
# A test (or a deployment) that pins one absolute folder sets this instead;
# it is read at call time, never captured at import.
TRASH_DIR: Optional[Path] = None

# Where the batches the *old* code wrote still live. Swept, never written.
LEGACY_TRASH_DIR = PROJECT_ROOT / "logs" / "trash"
```

```python
# editorfs/mutate.py
def trash_dir() -> Path:
    """The active workspace's trash — `.nojae/trash` under the resolved root."""
    override = constants.TRASH_DIR
    if override is not None:
        return Path(override)
    return Path(workspace_root()) / constants.POLICY_DIRNAME / constants.TRASH_SUBDIR
```

Every consumer calls `trash_dir()` per operation: `trash_batches()`,
`sweep_trash()`, and the move in `trash_path()`. That is what makes defect 2 go
away — a settings switch changes `workspace_root()`, and the very next delete
files into the new root.

The reserved name comes from `constants.POLICY_DIRNAME` (mirrored from
`policy_shared`), so a test that moves the reserved folder moves the trash with
it — no second literal to drift.

## 3. The gate must NOT be applied to the destination

The destination now lives under a *hidden* path, so a well-meaning refactor that
runs it through `resolve_write_target` would refuse every delete with a 422
naming the reserved-folder rule. State that in the `trash_path` docstring:

* the **source** path is gated (chain + leaf + the all-or-nothing subtree walk) —
  unchanged;
* the **destination** is built with plain `mkdir`/`shutil.move`, exactly as now,
  because it is app-chosen space inside the reserved folder and nothing may
  address it through a route anyway;
* the sweep walks the folder with `os`/`shutil` calls for the same reason.

Nothing can delete `.nojae` or anything in it (the reserved hit answers first,
before any `stat`), so a batch can never be trashed into itself, and no folder
under the root contains the trash path — the `shutil.move` "cannot move a
directory into itself" hazard does not arise.

## 4. `trashedTo` becomes workspace-relative

`_batch_label()` (mutate.py:453) rebases on `workspace_root()` instead of
`PROJECT_ROOT`, falling back to the absolute path when the pin moved the trash
outside the workspace. So the payload names `.nojae/trash/<digest>_<stamp>`.

That is the honest label for the new shape: every other Editor payload is
workspace-relative, and the e2e witness (`existsOnDisk`) joins the **working
location**, not the install dir. Say in the docstring that the label is a folder
the pane deliberately cannot open, so the confirmation must phrase it as a path to
`cd` to in a terminal, not as a browsable location.

## 5. The legacy batches

`logs/trash/` keeps nothing writable but must not rot. Cheapest honest answer:
`sweep_trash()` prunes **both** folders — it already treats "a batch that cannot
be deleted is not worth failing a delete over", and age is the only signal for a
path that no longer exists. It never *moves* old batches (that would be a surprise
rewrite of bytes the user is not looking at), it just lets them expire.

Optional and cheap: call `sweep_trash()` once in `api/server.py`'s `lifespan`
after `ensure_policy_file`, so the legacy folder is pruned even when no further
delete ever happens. Keep it a best-effort try/except like the seed note.

## 6. Level A keeps `logs/**`, with a truer reason

The rule stays (backups still live in `logs/editor_backups`, and the fedit
connector in `logs/fedit_backups`), but its reason string
(`builtin_policy.py:81`) says "the app's own backups **and trash**" — drop the
"and trash". Same for the explanatory comment above it (lines 75-80), which
describes the trash as living under `logs/`.

Worth checking while there: the rule is now *only* about backups, so a foreign
workspace's `logs/` folder is untouched (as before) — no change in behaviour,
which the existing tests assert.

## 7. Front-end

* `web/src/composables/useEditorFiles.js`: add `TRASH_SUBDIR`/a
  `TRASH_FOLDER` named export mirroring the server's `.nojae/trash`, with the same
  "asked before the request exists" comment as `TRASH_RETENTION_DAYS`. The route's
  `trashedTo` stays the word of record for the exact batch.
* `web/src/components/EditorTreeDialogs.vue:404`: `kept in <code>logs/trash/</code>`
  → the `.nojae/trash/` label, plus one clause saying that folder is reserved and
  not shown in the tree (the restore happens in a terminal). Fix the comment at
  line 364 ("a batch under the app's trash folder").
* `web/src/components/WorkspaceNavigator.vue:194` comment: unchanged wording is
  fine, but check it does not name `logs/`.
* No new payload plumbing: the dialog needs a name for a promise it makes before
  the request, and `/api/policy` already publishes `reserved: [POLICY_DIRNAME]`
  if a later task wants the real value instead of the mirror.

## 8. Tests

Python (`tests/`):

| File | Change |
|---|---|
| `test_editorfs_write.py` | the `trash` fixture keeps patching `constants.TRASH_DIR` (the pin still works — that is the point of keeping the attribute). Add: default resolution is `<root>/.nojae/trash`; a second workspace with `BENCHMARKIT_PATH_LOCATION` switched gets its **own** trash and cannot see the first root's batches; `trashedTo` is `.nojae/trash/<batch>`; a read/tree/delete of any `.nojae/...` path stays refused (404/404/422) so the new location is not a side door — extend `TestTrashIsNotASideDoor`, whose docstring argues the old placement, to the reserved-folder shape; rewrite its `repo_like` fixture to `.nojae/trash`. |
| `test_editorfs_builtin_policy.py` | the `logs/**` reason string, wherever the tuple is asserted (lines ~127, ~384). |
| `test_editorfs.py` | `logs/trash/x` fixtures still test `logs/**` visibility for backups; keep, but rename the comment so it no longer claims the trash lives there. |

Vitest (`web/tests`): `useEditorWriteMatrix.test.js` mocks `trashedTo`
as `logs/trash/abc-…` (lines 135, 362) and `useWorkspaceTreeCommands.test.js:386`
— switch the fixtures to `.nojae/trash/…` and assert the notice quotes the
route's answer, which is what these tests exist to prove.

Playwright (`web/e2e/02-editor-write.spec.js`): `TRASH_BATCH` becomes
`.nojae/trash` (line 24). The witness is already `existsOnDisk`, which joins the
working location — that keeps working, and it is now the *only* way to see the
batch, since the reserved folder is not listable. Update the header comment
(lines 19-22): `logs/**` is hidden for **backups** now, and the scratch used to
live there for a reason that is now half-gone. `03-workspace-switch.spec.js` keeps
its `logs` assertion and may add a `.nojae` one. Add one case: delete in a
switched-in workspace, assert the batch exists under **that** root.

## 9. Docs

* `README.md:33` — "moves it into a dated batch under `logs/trash/`" →
  `.nojae/trash/` of the **current** working location; add the sentence about the
  untracked `.nojae/` in a foreign repo's `git status`.
* `src/api/routes.py` `delete_fs_file` docstring (line 1389) and
  `src/core/editorfs/__init__.py` (lines 35-36) — the destination sentence.
* `AGENTS.md` — the Data-files row for `.nojae/`: it holds the policy **and** the
  trash; and the bullet about the delete route if it names the folder. Keep the
  file under its size budget: this is a rename of a location, not a new section.

## 10. Order of work

1. `constants.py` + `mutate.trash_dir()` + the three call sites (`trash_batches`,
   `sweep_trash`, `trash_path`), `_batch_label` rebase, legacy sweep.
2. Reason-string edits in `builtin_policy.py`.
3. Python tests (the default-resolution and per-workspace cases are the proof of
   defect 2 — write them before touching the front end).
4. Front-end label + dialog copy, Vitest fixtures.
5. e2e, README, `AGENTS.md`.

Non-goals: no restore UI (the batch is a folder and `mv` is the restore, as the
existing confirmation already promises); no change to `BACKUP_DIR` (the pre-save
chain stays in `logs/editor_backups`, where the `fedit`/`list_edits` symmetry
lives); no change to retention or batch naming.

