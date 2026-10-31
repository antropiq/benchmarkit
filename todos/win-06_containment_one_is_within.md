# win-06 — Containment: one `_is_within` at all five compare sites, and folder delete starts working

Five places decide "is this path inside that path?" by string-prefixing an
`os.sep`-joined absolute path. Roadmap §2.5: they *are* separator-consistent
(both sides are `realpath`-resolved) — **they fail on case**, and on drive/UNC
form. This task replaces all of them with one helper that applies the fold from
win-05, and in doing so un-breaks **folder delete**, which today is refused by
accident on Windows.

Depends on: win-05 (`fscase.case_folded`, the `fold` convention). win-03/win-04
already landed the text side.
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` A-D2.2 and
A-D2.3, A-D5's *folder delete* row, §4 (the `policy.py` and `mutate.py` pieces),
§6 behaviour test 5.

## Files

| file | change |
|---|---|
| `src/core/editorfs/policy.py` | new `_is_within(parent, child, *, fold: bool) -> bool`; used at `:172` (`_check_workspace_root`) and `:216` (`_resolve`) |
| `src/core/editorfs/mutate.py` | `_rel_in_workspace:459` stops hand-building a prefix; asks `_is_within` and returns the remainder |
| `src/core/lsphub/discovery.py` | `:156`'s `real_root.startswith(denied_entry + os.sep)` → the shared rule |
| `src/core/batch.py` | `:228`'s `resolved.startswith(str(workspace) + os.sep)` → the shared rule |
| `src/api/routes.py` | `:1298` (the `POST /path-location` deny-list compare) → the shared rule |
| tests: `test_editorfs_write.py`, `test_editorfs.py`, `test_batch.py`, `test_lsp.py` | §6 test 5 + one case per migrated site |

## 1. The helper

```python
def _is_within(parent: str, child: str, *, fold: bool) -> bool: ...
```

Both arguments are **absolute, already-resolved** (`realpath`) text. True when
`child == parent` or `child` sits below it. The rule it replaces is
`child == parent or child.startswith(parent + os.sep)` — keep that meaning, and:

- compare with `casefold()` on both sides when `fold` is `True`;
- **never** fold when `fold` is `False` (the Linux default everywhere, so every
  existing verdict is unchanged);
- do not use `os.path.commonpath`, `Path.relative_to` or `Path.is_relative_to`:
  they are case- and separator-exact and raise on a mismatch — A §4 names
  `Path.relative_to` explicitly as unusable here;
- do not use `os.path.normcase` (A-D3's rejected option).

Where it lives: `editorfs/policy.py` is the natural home (A §4 says so), and the
three callers outside `editorfs` (`lsphub/discovery.py`, `batch.py`,
`api/routes.py`) import it from there — the same direction `batch.py` already
uses for the editorfs write gates. **`lsphub/discovery.py` importing from
`editorfs` is a new edge, so check it does not create a cycle** (`editorfs.policy`
already lazily imports `lsphub.discovery.restricted_lsp_roots` at `:167`). If it
would, the right home is the same `src`-free side as `path_norm` — but decide by
reading the imports, and say which you chose and why in the commit note.

## 2. The six rows, five rules

| site | today | after |
|---|---|---|
| `policy.py:172` `_check_workspace_root` | `root == entry or root.startswith(entry + os.sep)` | `_is_within(entry, root, fold=…)` |
| `policy.py:216` `_resolve` | `candidate != workspace and not candidate.startswith(workspace + os.sep)` | inverted `_is_within` |
| `mutate.py:459` `_rel_in_workspace` | `workspace_root().rstrip("/") + "/"` prefix against `resolved.as_posix()` | `_is_within` + return the remainder |
| `lsphub/discovery.py:156` | `real_root == denied_entry or real_root.startswith(denied_entry + os.sep)` | `_is_within` |
| `batch.py:228` | `resolved != str(workspace) and not resolved.startswith(str(workspace) + os.sep)` | inverted `_is_within` |
| `routes.py:1298` | same shape as `:172`, from the settings switch | `_is_within` |

`policy.py:172` and `routes.py:1298` are the same deny-list question asked from
two entry points — the comment at `policy.py:158-164` explains why the tree
inherits the guard the switch already applies. After this task **no site builds
a `+ os.sep` prefix by hand**.

Note what `mutate._rel_in_workspace` does today: it compares `as_posix()` text
against a `rstrip("/") + "/"` prefix, and on `C:\…` that prefix never matches,
so it returns `None`, `_subtree_refusal` refuses with the "cannot be settled"
wording, and **every folder delete is refused** (roadmap §2's "one good news":
loud, correct by accident). Keep `None` as a real answer — it means "this path
sits where no policy of this workspace reaches", and the caller's refusal depends
on it. Return the remainder with `/` separators, the frame every other Editor
payload uses.

## 3. Where `fold` comes from

Each site asks `fscase.case_folded(...)` and passes it as an explicit keyword —
the helper never reads a global inside itself (same convention as win-05):

- `policy._resolve` and `mutate`: the **workspace** root.
- `_check_workspace_root`, `routes.py:1298`, `lsphub/discovery.py`: the root
  being compared against the deny-list is not the workspace (the denied entry may
  be `/home/vgr/.ssh`, a different filesystem). The honest simple choice is to
  fold those compares only when the probe for the **workspace** says the
  filesystem is case-insensitive — one answer per running configuration, no
  per-entry probing. Record that choice in the helper's docstring so B can
  revisit it.

## 4. Tests (§6 behaviour test 5, and one per site)

- **5:** with a policy rule marking a folder `ro` (or `hidden`), **folder delete
  works** when nothing in the subtree is locked, and `_subtree_refusal` still
  names the offending **descendant** when something is. The all-or-nothing rule
  and the message shape are pinned today — A §7 names *the all-or-nothing subtree
  delete* as something A must not break.
- `_is_within` unit pairs, both folds: `C:\WS\a` inside `C:\ws` when
  `fold=True`, not when `fold=False`; `/WS/a` never inside `/ws` on Linux;
  `/a/bc` is **not** inside `/a/b` (the trailing-separator trap the old code got
  right — keep it passing).
- A symlinked descendant whose target is spelled with a different case still
  resolves inside the workspace when folded. That test creates a symlink →
  `@pytest.mark.requires_symlink` (win-01) with its reason string.
- One regression test per migrated site: the batch lot folder outside the
  workspace still 422s; `POST /path-location` still refuses a deny-listed root
  (`test_editorfs.py:287` covers the tree side); `lsphub`'s
  `workspaceRoot is on the deny-list` note still fires (`test_lsp.py:165`).

## Rules this task must not break

- Locks run **before** `mode_for` and are not configurable
  (`policy_modes.py` module docstring). `_is_within` is a lock.
- The 404-vs-422 asymmetry for hidden paths (`policy.py:133 hidden_error`).
- The all-or-nothing subtree delete, and the trash semantics (a delete *moves*
  bytes to a dated batch in `.nojae/trash`; it never unlinks).
- `expanduser` stays where it already runs — `routes.py:1291`, for the settings
  field — and **nowhere else** (A-D2.3: *"No connector may expand it"*).
- `gitstatus.py:123` untouched; git output keeps its own form (A §5).
- Linux verdicts are identical: with `fold=False` the helper is exactly the old
  prefix test.

## Acceptance

- `pytest -q` green with §6 test 5 present.
- No hand-built containment prefix left: searching those five files for
  `os.sep` inside a `startswith` call and for `rstrip("/") + "/"` returns
  nothing.
- Folder delete in a scratch workspace still moves the bytes to
  `.nojae/trash/<batch>` and the response names the batch workspace-relatively
  (unchanged behaviour on Linux).
- `git diff --stat` lists the five source files above plus tests — no frontend,
  no `proctree.py`, no `console_mcp`/`fedit_mcp`.
- Move this file to `todos/done/` when the above is true.
