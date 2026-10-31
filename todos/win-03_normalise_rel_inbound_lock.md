# win-03 — `normalise_rel`: accept `\`, emit `/`, and refuse what `os.path.isabs` misses

Wire `path_norm` into the **one inbound lock** (`src/core/editorfs/policy.py:180
normalise_rel`). This is the keystone of A-D1: after this task, the string every
downstream consumer sees — the matcher, the ancestor walk, every `path` payload
echoed to the browser, `results.json` — is `/`-joined on every platform, while a
Windows user typing `src\config.py` gets the same file as a Linux user typing
`src/config.py`. It also closes A-D2.1: the three path shapes that slip past
`os.path.isabs` on Windows are refused **by name**, not caught later by
containment luck.

Depends on: win-01 (markers), win-02 (`path_norm`).
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` A-D1, A-D2.1
and A-D2.3; §6 behaviour tests 1 and 4. Also read `policy_modes.py`'s module
docstring on locks-vs-verdicts — this task changes a lock, not a verdict.

## Files

| file | change |
|---|---|
| `src/core/editorfs/policy.py` | `normalise_rel` calls `path_norm.canonical_rel` + `is_qualified`; `os.path.normpath` gone |
| `tests/test_editorfs.py` (`TestPathLocks`, `:208-240`) | add the §6 test-1 and test-4 cases |
| `tests/test_policy_glob.py` | optional: one case pinning that a `\`-shaped path reaches the matcher only after normalisation |

## 1. The new lock, in order

The order is load-bearing (locks before verdicts; the checks before the
collapse):

1. NUL check — unchanged.
2. **Qualified check first**: `path_norm.is_qualified(text)` → refuse, naming
   the shape. This subsumes today's `os.path.isabs(text)` (keep `isabs` or drop
   it as you prefer, but `is_qualified` must cover it: a leading `/` or `\`, a
   single-letter drive spec, a UNC root).
3. `~` check — unchanged, and still **before** any expansion (A-D2.3:
   *"`~` is checked before expansion, which is right — keep it"*).
4. `normalised = path_norm.canonical_rel(text)`.
5. `""`/`"."` → `""` — unchanged.
6. The surviving-`..` check — unchanged in meaning
   (`normalised == ".."` or `".." in Path(normalised).parts`). It still works
   because `canonical_rel` keeps a `..` that would leave the start. Do **not**
   add `os.path.isabs(normalised)` back as a belt-and-braces second absolute
   check; step 2 owns that, and a duplicate is how the two drift.

`editorfs.normalise_rel` stays exported under that name (the `TestSurfaceShape`
gate list at `tests/test_editorfs.py:714` pins it).

## 2. Refusal messages name the shape (A-D2.1: *"not by containment-by-luck"*)

Each refused shape gets its own sentence, and the sentence must name **what was
wrong**, not where it happened to land:

- `/etc/x` → an absolute POSIX path, e.g. *"an absolute path — the workspace is
  the root of a relative path"*;
- `\\srv\share` / `//srv/share` → a UNC share root;
- `C:x` → *"a drive-relative path (it names a drive, not the workspace)"*;
- `C:\x` → an absolute Windows path.

**Keep the phrase "relative to the workspace" in every one of them** — the
existing test `tests/test_editorfs.py:215`
(`test_absolute_path_detail_names_the_rule`) asserts exactly that substring, and
the §6 parametrisation in `TestPathLocks` (`:210`) already covers `/etc`. Update
that parametrised list with the three new shapes rather than renaming the test's
contract.

## 3. What this deliberately changes on Linux

`a\b` used to be a *filename containing a backslash* (`os.path.normpath` leaves
it alone on POSIX); from now on it means `a/b`. A §7 calls this out as the only
user-visible Linux change and notes that no POSIX tool in this app produces such
a name. Two things to check rather than assume:

- a test (or fixture) somewhere creates a file whose real name contains a `\` —
  if one exists, its assertion must move to a quoted-name form that survives, and
  you must say so in the commit note;
- **a second widening, from win-02's *"residue"* note**: a path whose first
  component is one letter plus a colon (`a:b`) is now refused as a drive spec,
  even though it is a legal POSIX filename. Add that case to the lock's tests
  with its own refusal message, and list it in the commit note beside the `a\b`
  one — §7 only promises the first, so the second must be visible, not silent;
- `gitstatus` is **not** touched here: `gitstatus.py:123`
  (`path.replace(os.sep, "/")`) already goes the right direction and must keep
  doing its own job (A §5).

## 4. Tests (§6 behaviour tests 1 and 4)

All runnable on Linux, no marker needed:

1. `normalise_rel("src\\config.py") == "src/config.py"`; likewise `a\\..\\b` →
   `a/b`… **careful**: `a\..\b` collapses to `b` (the `..` does not survive), so
   pin `b`; and `a\\/\\/b` → `a/b`; and a leading `\` is refused (test 1's last
   case is the *refusal*, not a translation).
2. The three §6 test-4 shapes (`/etc/x`, `//srv/share`, `C:x`) each raise
   `WorkspacePathError` 422 with a message naming the shape — and, to prove the
   "not by luck" half, assert the **message**, never "outside the working
   location".
3. Route-level: `GET /api/fs/tree?path=src\..\src` answers with
   `payload["path"] == "src"` (compare the existing
   `test_traversal_inside_the_workspace_stays_legal` at `:228`).
4. A regression guard for the §2.4 "create lands at the wrong depth" symptom is
   **not** this task's — win-04 owns `resolve_parent_in_workspace`. Do not fix it
   here; it will look tempting because it is the same file.

## Rules this task must not break

- Locks before verdicts, and a verdict can only narrow what the locks allowed
  (`policy_modes.py` module docstring, quoted in the big picture).
- The 404-vs-422 asymmetry for hidden paths (`policy.py:133 hidden_error`) is
  untouched: this task is a lock, and `hidden` is a verdict.
- `/`-joined payloads: nothing may now emit a `\` in a `path` field. The
  frontend's `split('/')` (`useWorkspaceTree.js:88`, `:535`,
  `useEditorRename.js:50`) stays correct — **do not touch the frontend**.
- Every existing rule in every existing policy file keeps its Linux verdict;
  `pytest -q` before your new tests are added must be green with only the
  expected message-text diffs.
- Python style: explicit annotations, `typing` imports.

## Import mechanism

`policy.py` reaches `mcp_connectors` the way `editorfs/constants.py` and
`core/batch.py` already do (the `sys.path` fallback, resolved at call/import time
so a test's patch is seen). Reuse `editorfs.constants` — it already mirrors
`policy_glob` and `policy_shared` for exactly this reason; **adding
`path_norm` to that mirror is the right move** and keeps the one pattern rather
than inventing a third import path.

## Acceptance

- `pytest -q` green, including the new §6 test-1 and test-4 cases.
- `grep -n "normpath" src/core/editorfs/policy.py` is **empty**. (`batch.py`
  still has its own — win-04's job.)
- `tests/test_editorfs.py::TestPathLocks` passes with the three new shapes added
  to its parametrisation, and no assertion in that file was weakened.
- The `path` field of a tree/read/create response for a `\`-shaped request is
  `/`-joined (check one response body by hand with a `TestClient` print, or
  assert it in a new test).
- No frontend file touched; no `search.py`, `mutate.py`, `batch.py` or
  `policy_glob.py` touched (those are win-04/win-05).
- Move this file to `todos/done/` when the above is true.
