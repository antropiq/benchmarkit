# win-05 — Case: a probe per workspace root, and one explicit `fold` through the matcher

On NTFS, `Dist` and `dist` **are** one folder. The matcher is deliberately
case-exact (`policy_glob.py:226`'s docstring says so, and on Linux it is right),
so on Windows a rule `dist/**` marked `hidden` leaves `GET /api/fs/file?path=Dist/app.js`
matching **nothing** — and `.NOJAE\x` walks past the reserved-folder check. This
task makes comparison fold **where the filesystem folds**, decided by a probe
rather than by `sys.platform` (roadmap §4, A-D3), and updates the docstring that
is the reason nobody fixed it yet.

Depends on: win-01 (markers), win-03 (`\`-shaped input is already canonical by
the time the matcher sees it).
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` A-D3 and §4
(the `policy_glob` and `fscase` pieces); §6 behaviour test 3 and behaviour test
7's matcher half.

## Files

| file | change |
|---|---|
| `src/core/editorfs/fscase.py` | **new** — `case_folded(workspace_root) -> bool`, cached per root |
| `mcp_connectors/policy_glob.py` | keyword `fold: bool = False` on `_name_matches`, `reserved_component_hit`, `pattern_matches`; `_path_segments` → `path_norm.path_parts`; the `:226` docstring corrected |
| `src/core/editorfs/policy_modes.py` | `mode_for` / `list_builtin_rules` / `list_workspace_rules` / `reserved_hit` pass the root's fold |
| `src/core/editorfs/policy_file.py`, `src/core/editorfs/builtin_policy.py` | accept and forward `fold` (their rule-matching calls) |
| `src/core/editorfs/policy_seed.py` and/or `src/api/routes.py` (the location switch) | invalidate the probe cache with the policy cache |
| `tests/test_fscase.py` | **new** |
| `tests/test_policy_glob.py`, `tests/test_editorfs_policy_modes.py`, `tests/test_editorfs.py` | §6 tests 3 and 7 |

## 1. The probe (`editorfs/fscase.py`)

`case_folded(workspace_root: str) -> bool`: create a temp name inside the root
with a cased letter, `stat` its folded variant, compare identity (`st_ino` +
`st_dev`, or `os.path.samefile`), delete both. Answer cached keyed on the root —
**the same cache shape and the same invalidation as `policy_file._CACHE`**
(`policy_file.py:82-101` explains why the key is the root and why a switch must
forget it; mirror that reasoning in the docstring, including a bound like
`_CACHE_MAX_ROOTS`).

Rules for this file:

- It **touches disk**, which is exactly why it lives on the app side and the
  matcher stays pure (A §4).
- A root that cannot be written to (a read-only mount, a workspace whose policy
  marks everything `ro`) must not raise: answer `False` — *fold nothing you
  cannot prove*. A wrong `False` is over-strict, which is the safe direction for
  a security matcher. Say that in the docstring.
- Clean up the probe files even when the answer came back folded, and never
  leave anything behind in the user's workspace (the Editor's own rule: a delete
  moves bytes to the trash — but these are *this function's* bytes, created and
  removed in one call, so `os.unlink` of names it made itself is right; do not
  route them through `mutate`).
- The cache is invalidated at the **same two moments** the policy cache is:
  the seeder's invalidation (`policy_seed.py:228`) and the settings switch
  (`routes.py:~1345`, next to `agent_modes.invalidate_agent_modes_cache()`).
  AGENTS.md: *"Every path the app holds is relative to that location, so a switch
  invalidates them all at once."* Add an `invalidate_case_cache()` and call it
  where `invalidate_policy_cache()` is already called.
- **No `sys.platform` branch** in this module: the probe *is* the answer
  (roadmap §9 rule 2). A test can also monkeypatch the cached answer, which is
  what makes both outcomes reachable on Linux.

## 2. `fold` is an explicit parameter, everywhere

`policy_glob` gains `fold: bool` as a **keyword with default `False`** on
`pattern_matches`, `reserved_component_hit`, and the private `_name_matches` /
`_match_segments` / `_match_with_subtree` chain. Default `False` means
every existing call site — including both stdio connectors' — keeps today's
exact semantics without being touched.

**Decided for this lot** (big picture, "Fold threading"): the app-side verdict
path is the only caller that passes a real `fold`. The connector call sites
(`mcp_connectors/policy_file.workspace_policy_reason`, `fedit_mcp`,
`console_mcp`) keep the default — Linux-exact, and honest there, because win-07
refuses those connectors on Windows anyway; B supplies real values. Do **not**
read a global, a module attribute or the probe from inside `policy_glob`: the
whole point of the parameter is that a test can force either outcome on Linux.

Folding is **comparison only** (A-D3: *"never in the stored/displayed/wire form
of a path"*): `casefold()` both sides at the comparison, never the pattern, the
path, the verdict's `pattern` field, or a refusal message. A `hidden` rule for
`dist/**` still *prints* `dist/**`.

`_path_segments` now delegates to `path_norm.path_parts` (win-02's splitter)
instead of `rel.split("/")`, keeping its contract exactly: **`None` when the
input is not normalised** (`""`, `"."`, `".."` among the components) — that `None`
is the tripwire that keeps a skipped lock from being reported as a hiding rule,
and its docstring is worth re-reading before you change the body.

## 3. The docstring at `policy_glob.py:226`

*"Comparison is exact — no case-folding, the temptation that would make `Dist`
and `dist` one folder"* becomes the truth: **exact where the filesystem is
exact, folded where it is not; the caller decides by passing `fold`**. A-D3 names
this sentence as *"the reason nobody fixed this yet"*, so correcting it is part
of the deliverable, not a comment tidy-up. Also update the module docstring's
*"Case-sensitive on POSIX: `Dist` and `dist` are two directories"* line
(`policy_glob.py:29-30`) to the same shape.

## 4. Threading it through the verdict

`mode_for` is still **the one place the verdict is computed** (pol-06 — do not
add a second predicate). It reads the fold itself, from
`fscase.case_folded(workspace_root())`, and passes it down to
`list_builtin_rules` / `list_workspace_rules`, which pass it to the matcher. Same
for `policy_modes.reserved_hit` → `policy_glob.reserved_component_hit`. Keep the
signatures source-compatible (`fold` keyword, defaulted) so the connector side and
the panel (`rule_lists`, pol-07) do not have to change shape.

`builtin_policy.is_self_workspace()` compares two `realpath`s for **equality** —
that is an absolute-path compare and belongs to win-06's fold work, not here. Do
not touch it.

## 5. Tests (§6 tests 3 and 7, both outcomes on Linux)

- **7 (matcher half):** one rule `dist/**` marked `hidden` blocks `Dist/app.js`
  when `fold=True` and does not when `fold=False`; the same pair for
  `path_policy.component_hit` with `.ssh` is **win-07's half** — leave it.
- **3:** `reserved_component_hit(".NOJAE\\x", ".nojae", fold=True)` hits. Then
  the whole chain through the real gate: with the probe forced to `fold=True`,
  a read of `.nojae\x` answers **404** and a write answers **422 naming the
  rule** (the 404-vs-422 asymmetry from `policy.py:133 hidden_error` must be
  preserved — A §7 names it).
- The probe itself (`tests/test_fscase.py`): on the Linux runner, a
  case-sensitive tmp dir answers `False`; force the folded answer through a
  monkeypatched probe rather than trying to find a case-insensitive mount. Also
  assert the cache: two calls, one create (count through a patched `os.stat` or
  a counter), and that `invalidate_case_cache()` makes it re-probe.
- The switch invalidates it: a test that changes `BENCHMARKIT_PATH_LOCATION` and
  asserts a fresh probe (mirror how `tests/test_editorfs_policy_file.py` pins the
  policy cache across roots).

## Acceptance

- `pytest -q` green, with the §6 test-3 chain and the §6 test-7 pair present.
- `grep -n "sys.platform\|os.name\|normcase" src/core/editorfs/fscase.py mcp_connectors/policy_glob.py`
  is empty.
- `grep -n "casefold" mcp_connectors/policy_glob.py` shows folding **only** at
  comparison points (no `casefold` on a returned pattern, verdict field or
  message).
- `tests/test_policy_glob.py:259-260` (`pattern_matches("D*", "Dist")` True,
  `("D*", "dist")` False) still passes **as written** — that is the exact-match
  default holding. Add the folded pair next to it, do not edit it.
- Both `policy_glob.py:226` and the module docstring at `:29-30` now state the
  fold rule honestly.
- No change to `mutate.py`, `path_policy.py`, `discovery.py` (win-06/win-07), and
  no connector call site passes `fold=True`.
- Move this file to `todos/done/` when the above is true.
