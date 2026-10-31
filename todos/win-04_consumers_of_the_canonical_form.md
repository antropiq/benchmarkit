# win-04 — The four consumers: ancestor walk, create parent, ignore prune, batch lot folder

After win-03 the canonical form is `/`-joined everywhere — but four places still
consume it *as if* it could contain `\`, and three of them are silently wrong on
Windows. Replace each hand-rolled split with `path_norm.path_parts`, so the
ancestor walk and the matcher agree **by construction** (A §4: *"they should
anyway"*). One of the tests this task adds is **the test that fails today** and
is the proof A-D1 landed.

Depends on: win-03 (and through it win-02, win-01).
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` A-D1's table
(all four rows) and §6 behaviour tests 2 and 6.

## Files

| file | change |
|---|---|
| `src/core/editorfs/policy.py` | `resolve_write_target:282` ancestor walk → `path_parts`; `resolve_parent_in_workspace:370` parent split → `path_parts` |
| `src/core/editorfs/search.py` | `_ignored_hit:113`'s `rfind("/")` cut loop → `path_parts` (or reuse the matcher's cut loop — see §3) |
| `src/core/batch.py` | `resolve_lot_dir:198` (the `os.path.normpath` at `:216`) → `path_norm.canonical_rel` |
| `tests/test_editorfs_write.py` | §6 test 2 (the failing one) + a create-depth case |
| `tests/test_editorfs_search.py` | §6 test 6 |
| `tests/test_batch.py` | a `\`-shaped lot folder names the same folder |

## 1. `resolve_write_target` — the ancestor walk (`policy.py:282`)

Today: `for component in normalised.split("/")[:-1]` builds each prefix. On
Windows with the old `normpath` output that list is empty, so the strict gate
degrades to the leaf and *"adding to it, emptying it, and moving something in or
out of it are all refused"* (pol-00 D3) stops being true for every folder rule.
After win-03 the input is already `/`-joined, so this specific walk no longer
breaks — **but it must still go through `path_parts`**: the rule is that the walk
and the matcher split the same way, not that they happen to.

Keep the chain shape exactly as it is — `(normalised_rel, resolved_path)` pairs,
ancestors first, leaf last, each prefix `/`-joined — because
`tests/test_editorfs_write.py:528` (`test_the_chain_helper_reports_each_level`)
pins `["src", "src/core", "src/core/job.py"]` verbatim.

**§6 test 2 (the important one):** with a policy rule `{"path": "secrets", "mode":
"ro"}`, `resolve_write_target("secrets\\nested\\k.txt")` refuses, and the refusal
names the `secrets` rule (the ancestor, not the leaf). Write this test **before**
the code change and note in the commit that it fails today — A §6 says this is
the proof A-D1 landed. It passes on Linux once win-03 is in, which is exactly
what makes it a useful regression guard on both platforms.

## 2. `resolve_parent_in_workspace` — a create's parent (`policy.py:370`)

Today: `parent_rel, _, name = normalised.rpartition("/")`. On a `\`-shaped
input the parent is empty, so *"a create names the workspace root"* (roadmap
§2.4). Use `path_parts`: the parent is everything but the last component, joined
with `/`; the leaf name is the last component. The existing behaviour of
re-normalising the parent (`normalise_rel(parent_rel)`) and of resolving the leaf
name through the same verdict on the **relative** form stays exactly as the
comment describes — do not restructure that function, only its split.

Add one test: `create_path("notes\\sub\\x.md", kind="file")` refuses with a
message about the **missing parent** (`notes/sub`), not one about the root —
compare `test_editorfs_write.py:641` (`test_a_missing_parent_is_409_naming_it_and_creates_no_chain`)
for the existing wording, and assert no chain of folders was created.

## 3. `_ignored_hit` — the `.gitignore` prune (`search.py:113`)

Today: a `while cut > 0` loop over `rel.rfind("/", 0, cut)`, checking each
ancestor against the set of git-ignored names. A `\`-shaped `rel` (before win-03)
found no cut and the prune from `b55dd20` stopped firing. Two acceptable shapes:

- walk `path_parts(rel)` prefixes, or
- reuse the matcher's own cut idea (`policy_glob._match_with_subtree` tests
  whole-component prefixes).

**One implementation, not two** (pol-06's rule): if you keep the loop, make it
derive its cuts from `path_parts` so there is no second notion of "component" in
the codebase. Note that the set `names` comes from git and is `/`-joined by
`gitstatus._to_workspace_rel` (`gitstatus.py:123`) — that side is already right
and must not be touched (A §5).

**§6 test 6**: `_ignored_hit("build/output/x.js", {"build"})` is `True` — and, at
the search level, a scope inside an ignored folder answers an **empty scan, not a
404** (the reason is in `search.py:163-167`: an ignored folder is a row the tree
shows). Keep that distinction in the new test's assertions.

## 4. `batch.resolve_lot_dir` (`batch.py:198`, normalising at `:216`)

A-D1's last line names `batch.py:216`: `os.path.normpath(raw)` on the lot folder
name → `path_norm.canonical_rel(raw)`,
so a Windows user typing `todos\drafts` gets the same folder as `todos/drafts`
and the `taskLocation` strings handed to the model stay `/`-joined. Keep the
surrounding refusals (`isabs`, `~`, surviving `..`, "must be a named folder") and
their messages; only the normalisation changes, and the `..`-in-`Path(text).parts`
check at `:213` must be evaluated on the **normalised** text (today it reads the
raw one — verify what that means for `a\..\..\b` and make it consistent with
`normalise_rel`, since A-D1's property is that a surviving `..` is what gets
refused).

`batch.py` also keeps its own containment compare at `:228`
(`resolved.startswith(str(workspace) + os.sep)`) — **win-06's job, not yours**.
Do not refactor it here; just leave a clean seam.

Batch's other invariant is untouched: the lot is *listed* through the shared
deny-list and **never reads a task body** (AGENTS.md: *"Batch mode never touches
file contents"*).

## 5. What must not change

- Every Linux verdict, status code and refusal message that is not about the
  three shapes above. Existing tests in `test_editorfs_write.py` (the `ro` folder
  family at `:505`, `:2078`), `test_editorfs_search.py`, `test_batch.py` stay
  green untouched.
- The chain payload shape (§1) and the 409-vs-422 create dialog difference
  (`batch.py`'s docstring on "create the folder" vs a 404).
- `search.py`'s D2 rule: **one verdict per path via `mode_for`, and no second
  answer anywhere**; the policy prune and the ignore prune stay two separate
  checks and neither counts what it pruned.
- Frontend untouched: the tree's `split('/')` and `results.json`'s form depend on
  the canonical form staying `/`-joined.

## Acceptance

- `pytest -q` green **with these new tests present**: §6 test 2 (ro ancestor
  refused from `\`-shaped input), §6 test 6 (`_ignored_hit` prunes an ignored
  ancestor), the create-depth case, and the batch folder equivalence.
- `grep -n 'split("/")\|rpartition("/")\|rfind("/")\|normpath'` across
  `src/core/editorfs/policy.py`, `search.py`, `src/core/batch.py` shows no
  hand-rolled component split left on a **relative** path (an absolute-path
  `os.sep` containment compare is win-06's and may remain).
- `git diff --stat` touches exactly the six files in the table (plus test files).
  No `policy_glob.py`, no `mutate.py`, no frontend.
- The app still works end to end on Linux: create a nested file in a scratch
  workspace through `/api/fs/create` and the tree shows it at the right depth.
- Move this file to `todos/done/` when the above is true.
