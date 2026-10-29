# pol-06 — Every surface answers from `mode_for`

The swap. Tree, read, stat, diff, git status and the four write routes stop
consulting the connector deny-list and consult the policy instead; the old
predicates go away. This is the task where the visible behaviour actually changes,
so its tests are its argument.

Depends on: pol-01 … pol-05 (both levels are populated, and every workspace that
the tests use either has a seeded file or deliberately has none).

## 1. The gate keeps its locks, changes its rules

`policy.py` keeps, untouched: `normalise_rel`, `_check_workspace_root`, the
`realpath` containment check in `_resolve`, and the shape of
`resolve_write_target` / `resolve_parent_in_workspace`. Those are the three locks
plus the root deny-list, and pol-00 lists why they never move.

Replaced: the `refusal()` call inside `_resolve` (line 161) and inside
`resolve_parent_in_workspace` (line 274) becomes a `mode_for()` call. Deleted:
`listable`, `writable` and `refusal`. The **rule-list** import is gone — no
surface in `editorfs` calls `mcp_connectors.path_policy.deny_reason` any more.

`_path_policy()`'s `sys.path` shim (`policy.py:61-75`) **stays, renamed** to
something honest like `_policy_modules()`, because pol-02's matcher and pol-04's
parser also live in `mcp_connectors/` and need the same reach. Its docstring
changes from "the shared connector deny-list" to "the shared, stdlib-only policy
modules — the matcher and the parser, not a rule list".

```python
def _resolve(normalised: str, *, for_write: bool) -> Path:
    workspace = _check_workspace_root()
    candidate = os.path.realpath(os.path.join(workspace, normalised))
    if candidate != workspace and not candidate.startswith(workspace + os.sep):
        raise WorkspacePathError(422, f"Path is outside the working location: …")
    verdict = mode_for(normalised)          # ← the rules, from two levels
    if verdict.mode == HIDDEN:
        raise WorkspacePathError(404 if not for_write else 422,
                                 …)
    if for_write and verdict.mode == READ_ONLY:
        raise WorkspacePathError(422, refusal_message(verdict, for_write=True))
    return Path(candidate)
```

**`mode_for` takes the normalised *relative* path, not the resolved absolute
one.** That is the whole reason pol-01 chose relative matching: rules are written
against the workspace the user sees, and a `$HOME` workspace must not turn
`web/**` into an accidental absolute match.

The one thing the old code did that the new code must not lose: the leaf-name
check **before any `stat`** in `resolve_parent_in_workspace` (`policy.py:269-276`).
It stays, now asking `mode_for(parent_rel + "/" + name)` — a create whose name is
hidden must be a 422, never the "parent does not exist" 409 that would confirm
the name is free.

## 2. `hidden` means different numbers on the two kinds of surface (D8)

| surface | old answer for `.env` | new answer |
|---|---|---|
| `GET /api/fs/tree` | sibling omitted | omitted (same) |
| `GET /api/fs/file` | 422 naming the rule | **404 `File does not exist`** |
| `GET /api/fs/stat` | 422 | `{"exists": false, …}` — the route's own absence answer |
| `GET /api/fs/diff` | 422 | 404 |
| `PUT /api/fs/file`, `POST /api/fs/create`, `POST /api/fs/rename`, `DELETE /api/fs/file` | 422 | **422 naming the rule** (never 404 — 404 invites a create) |
| git status | path filtered out | filtered out (same) |

Rationale, in the docstrings, in one sentence each: an *observation* of a hidden
path must be indistinguishable from absence; a *mutation* must say why, because
"not there" is the green light for `O_CREAT | O_EXCL`. The old code used 422 for
both because the deny-list was a secret from the browser; a policy the user wrote
(and pol-07 shows them) is not, so the asymmetry is now safe and useful.

`tests/test_editorfs.py:224` (`test_denied_path_is_a_refusal_not_exists_false`)
is therefore **expected to change in this task** — it asserts the old premise.
Rewrite it as `test_hidden_path_is_absent_not_a_refusal`, asserting
`exists: false`, and keep a sibling test asserting that a `ro` path still answers
normally to stat (it is not hidden, and it is readable). Every other read test
in that file stays byte-identical.

## 3. Tree: prune, and stop double-filtering

`tree.list_dir` (`tree.py:85-89`) drops the `IGNORED_DIR_NAMES` test and asks the
policy once:

```python
if mode_for(rel_of(child)).mode == HIDDEN:
    continue
```

`_has_children` (`tree.py:44-61`) uses the same single filter — the docstring's
promise is that `children: false` never precedes a non-empty listing, and that
promise only holds while the two apply identical rules.

The `rel` handed to `mode_for` is the **normalised relative** path the entry
already carries in its payload (`tree.py:100` builds it), not the absolute one.

`hidden` on a directory prunes it without a second `scandir` — no walk, no
carve-out below it (pol-00). Add that to the module docstring, because it is the
semantics a policy author will otherwise get wrong.

## 4. Writes: `ro` blocks the subtree and the folder's own listing (D3)

Already-shape, new question:

* `save` / `write_file` — `resolve_existing_in_workspace(rel)` already walks the
  ancestor chain (`resolve_write_target`, `policy.py:199-207`); with the chain
  now asking `mode_for` at **each** level, a save under an `ro` folder is refused
  by the ancestor, and the message names that ancestor. This is the behaviour
  `policy.py:191-197` already describes for protected parents; only the source of
  the rule changed.
* `create` — parent chain plus the leaf, both `rw`-required (the
  `policy.py:269-276` check above).
* `rename` — both ends go through the write gate (`mutate.py:230` and `:246`), so
  moving *out of* an `ro` folder is refused at the source end and moving *into*
  one at the destination end. **No new code**: the two ends were already gated for
  exactly this reason; say so in the rename docstring, since D3's "rename-out is
  refused" reads like a new rule.
* `delete` — `_subtree_refusal` (`mutate.py:375-410`) keeps its walk and its
  all-or-nothing answer, and its per-candidate `refusal(target, for_write=True)`
  becomes `mode_for(rel_of(target))`. The message keeps naming the **descendant**
  (`f"{deny} — inside {root.name}"`), which is the part that makes the refusal
  actionable.

Add to `mutate`'s module docstring: an `ro` folder cannot be deleted, emptied,
renamed or added to — the rule is about the subtree *and* the folder's own
directory entries.

## 5. Broken policy file: writes off, everything else on

At the top of each write route's core (or once, in `resolve_write_target` — one
place, so a route cannot forget it):

```python
blocked = policy_blocking_error()   # pol-04
if blocked is not None:
    raise blocked
```

Reads and the tree stay available so the user can diagnose; the message from
pol-04 is the whole answer (file path + the finding verbatim).

## 6. `IGNORED_DIR_NAMES` is deleted

`constants.py:41-51` goes away with its comment. Two tests patch it today and must
be rewritten to write a policy file instead:

* `tests/test_editorfs.py:100` (`test_ignored_set_is_patchable_through_constants`)
  → `test_policy_rules_are_read_from_the_workspace_file`: seed a policy with
  `{"path": "src", "mode": "hidden"}`, assert `src` is absent from the root
  listing, and that a second workspace without the file lists it;
* `tests/test_editorfs_git.py:186` → the same, against `_displayable`.

`gitstatus._displayable` (`gitstatus.py:112-121`) becomes one policy call:

```python
return mode_for(rel).mode != HIDDEN
```

and the comment about "the tree's two omission rules" becomes one rule. Note the
consequence worth a line in the docstring: a `ro` folder's dirty files **do** show
in git status now (they are visible in the tree), where `__pycache__`-shaped
noise does not — which is the correct pairing, since the marker and the row
appear or disappear together.

## 7. `mutate.py` docstring and the create note

`mutate.py:117-121` currently explains that `IGNORED_DIR_NAMES` is not a write
rule. Replace with the true statement of the new model: **`hidden` is a write
rule** (creating a name a policy hides is refused — D8), **`ro` is a write rule**,
and neither is a *read* rule except `hidden`. The old note existed because the
constant and the deny-list disagreed; under one policy there is nothing to
disagree about.

## 8. `editorfs/__init__.py` and the exports

Drop `listable` / `writable` / `refusal` from the package exports and the module
docstring (`__init__.py:7-8`, `:61-67`); export `mode_for`, `Verdict`, `MODES`,
`policy_blocking_error`. `tests/test_editorfs.py:453`
(`test_editorfs_exposes_the_guarded_write_and_nothing_beyond_it`) is the test that
tells you the list changed — update its expectation deliberately, not by
relaxing the assertion.

## 9. Tests

The full set of behaviour changes, each one a test:

**In `tests/test_editorfs.py` (read surface)**
1. foreign workspace, `.env` present → it **is** listed and readable (the fix, end
   to end);
2. self workspace, `.env` → absent from the listing, 404 on read, `exists: false`
   on stat;
3. `ro` folder: listed, readable, and `children: true` when it has children;
4. `hidden` folder: pruned — and a path *below* it is unreachable even though no
   rule mentions it (the subtree rule at the route level);
5. `.nojae` absent, at any depth, in both workspace kinds;
6. the rewritten stat test of §2.

**In `tests/test_editorfs_write.py`**
7. `ro` folder: save → 422 naming the rule; create inside → 422; delete → 422
   naming the descendant; rename-out → 422 at the source; rename-in → 422 at the
   destination;
8. carve-out: `**/node_modules` `ro` + `**/node_modules/**/README.md` `rw` → the
   README saves and its sibling `index.js` does not;
9. hidden create: `PUT`/`POST` on a hidden name → 422, not 409 "parent missing"
   and not 404;
10. level A in the self workspace: `.env` and `logs/…` writes refused with the
    app's own wording ("benchmarkit's own rules"), while the same paths in a
    foreign workspace are accepted;
11. a corrupt policy file → every write route 422 with pol-04's message, while
    `GET /api/fs/file` on an ordinary path still answers 200;
12. **unchanged and still green**: the mtime guard, the chat-stream gate, the
    backup-before-save, the trash move, the size caps, the CRLF round-trip, and
    all of `tests/test_write_surface_isolation.py` (no connector or LSP tool
    gained a write route — this task must not touch it).

**Frontend (`web/tests`)** — only where a status code changed: a hidden file's
failed open now arrives as 404 rather than 422, so any assertion on that text
moves. No new component behaviour in this task (that is pol-07).

## 10. Acceptance

* `pytest` green. `test_write_surface_isolation.py` untouched.
* Two workspaces, side by side, by hand: benchmarkit-as-workspace still hides
  `.env`, `logs/`, `.nojae`; a foreign project shows its `.env`, browses
  `node_modules`, and refuses to edit either.
* `grep -rn "path_policy" src/` returns **only** `core/batch.py:235` (the batch
  lot listing keeps its own guard — it is not an Editor surface, and changing it
  is out of scope here).
  (`policy_glob` / `policy_file` imports from `mcp_connectors/` are expected and
  wanted — that is the shared matcher; only the deny-list is gone.)
* `grep -rn "IGNORED_DIR_NAMES" src/ web/ tests/` returns nothing.

## Notes for whoever executes it

* Do the four write routes and the read surfaces in **one** commit. Half-swapped,
  the tree and the write gate disagree, and that is the one state this lot must
  never ship.
* If a test of the read surface fails on a *cap*, a binary answer or a traversal —
  pol-06 broke a lock, and the test is right.
