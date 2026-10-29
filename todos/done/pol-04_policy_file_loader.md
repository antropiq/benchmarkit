# pol-04 — Level B: read, validate and cache `.nojae/read-write-policy.json`

Fill the second half of `mode_for` from the workspace's own file, with pol-00's
validation split (exception vs note) implemented here, where it is testable
without any HTTP. Still wired to no surface — pol-06 does the wiring.

Depends on: pol-01 (vocabulary), pol-02 (matcher, `pattern_error`,
`literal_prefix`), pol-03 (level A, needed to compute *shadowed* notes).

## 1. The file

`<workspace_root>/<POLICY_DIRNAME>/<POLICY_FILENAME>` — both names from
`editorfs/constants.py` (pol-01), so a test patches the location through the
module like every other cap in this repo.

```jsonc
{
  "version": 1,
  "readme": ["Rules are matched in order; the last match wins, so a later rule",
             "carves an exception out of an earlier one.",
             "path is relative to this workspace root; * stays inside one name,",
             "** crosses directories. A folder rule covers its whole subtree.",
             "mode: hidden = not shown and not reachable; ro = browsable and",
             "readable, nothing under it changes; rw = the default."],
  "policies": [
    {"path": ".git",            "mode": "hidden"},
    {"path": "**/__pycache__",  "mode": "hidden", "reason": "bytecode"},
    {"path": "**/node_modules", "mode": "ro",     "reason": "browse docs, edit nothing"}
  ]
}
```

`readme` is **accepted and ignored by the matcher** — JSON has no comments, and
the seeded file is the user's first look at the syntax (pol-05), so the syntax
must be able to explain itself. `reason` is optional and lands in
`Verdict.reason` and in the refusal sentence.

## 2. Shape the loader accepts

Top-level keys: `version` (int, equal to `POLICY_VERSION`), `readme` (string or
list of strings), `policies` (list). **Any other top-level key is an error** —
the alternative is `"policyies": [...]` silently meaning "no rules", the single
worst failure mode of a config file.

Per rule: `path` (string, required), `mode` (one of `MODES`, required,
case-sensitive), `reason` (optional string). Anything else is an error naming the
rule's index and the offending field.

```python
@dataclass(frozen=True)
class Rule:
    mode: str
    pattern: str
    reason: Optional[str]
    index: int
    level: str          # LEVEL_BUILTIN | LEVEL_WORKSPACE


@dataclass(frozen=True)
class PolicyFinding:
    kind: str           # "error" | "note"
    index: Optional[int]
    message: str        # names the file and the rule; ready for the UI
```

### Where the parser lives: shared with the connectors

Parse **and** validate go in **`mcp_connectors/policy_file.py`** — stdlib-only,
alongside `path_policy.py` and `policy_glob.py`, and importing its vocabulary from
`policy_shared.py` (pol-01 §1: the vocabulary module is **also** outside `src`,
since a stdio connector cannot `import src.core`). One schema and one set of
validation messages, two callers — the same argument that created
`path_policy.py`. Duplicating the parser for pol-08 would leave two schema
definitions able to drift, which is the "second list" failure this lot removes
elsewhere.

```python
# mcp_connectors/policy_file.py — no `src` imports
def load_policy_text(text: str) -> Tuple[List[Rule], List[PolicyFinding]]: ...
```

The src-side wrapper (`src/core/editorfs/policy_file.py`) owns the three things a
connector cannot do: the workspace-root cache (§4), the shadowing notes against
level A (pol-03 — connectors do not apply level A), and converting an error
finding into a `WorkspacePathError`. State that split in **both** docstrings, or
the next reader will "consolidate" the pair and break pol-08.

## 3. The validation table, implemented

| case | kind | message must name |
|---|---|---|
| file absent | nothing — level B contributes no rules; the normal state of a project that never got one | — |
| unreadable (`EACCES`, a directory where the file should be) | error | the path and the OS error |
| malformed JSON | error | line/column from `json.JSONDecodeError` |
| `version` missing or unknown | error | what was found, what is supported |
| unknown top-level key | error | the key |
| `policies` not a list; a rule not an object; `path`/`mode` missing or the wrong type | error | the rule index and the field |
| `mode` not in `MODES` | error | the value and the three valid ones (a spelling fix, free) |
| `pattern_error(pattern)` non-None | error | the reason from pol-02 |
| pattern naming `POLICY_DIRNAME` | error | that `.nojae` is reserved and not addressable |
| `literal_prefix(pattern)` does not exist in the workspace | error | the pattern plus a closest-sibling suggestion (`web/node_module` → "did you mean `web/node_modules`?") |
| rule identical to, or fully shadowed by, an **earlier** rule of the same file | note | both indices |
| rule shadowed by a stricter **level-A** rule | note | that benchmarkit's own rule for this workspace wins and editing the file cannot change it |
| `policies: []` | nothing | — (a project that wants everything visible) |

The unreachable-prefix check must **not** walk the tree: `os.path.exists` on the
prefix is the whole check, and a pattern whose prefix is `""` (a leading `**`) is
never reported unreachable. Wrap it in `try/except OSError` — a broken symlink in
the prefix is not the user's typo.

For the suggestion: `difflib.get_close_matches` over the siblings of the prefix's
parent directory, capped at 3. Cheap, and it is the difference between an error
the user acts on and one they ignore.

## 4. The cache, and why the root is in the key

```python
def workspace_rules() -> tuple[Rule, ...]: ...
def workspace_findings() -> tuple[PolicyFinding, ...]: ...
```

Cached in a module-level dict keyed by **`workspace_root()`**, each entry
carrying `(st_mtime_ns, st_size)` of the policy file. A hit recomputes nothing; a
miss re-reads. Three reasons the root is in the key, each one a test:

1. `BENCHMARKIT_PATH_LOCATION` is hot-swapped by `POST /api/path-location`
   (`routes.py:1219`) — keyed on `(mtime, size)` alone, the new root would run
   under the old root's rules;
2. two roots can hold files of the same size, and a same-size edit is entirely
   possible (one pattern renamed to an equal-length one);
3. the test suite patches the location per test — a stale cache makes those tests
   order-dependent.

Expose `invalidate_policy_cache()` for tests. And note in the docstring that the
loader is the **one** reader of `.nojae`: the app owns that read, which is what
makes pol-03's reservation and pol-05's seeding both legal. Without that sentence
the next contributor sees a hole where there is a design.

## 5. Failing closed on writes, open on reads

D4. When the file exists **and** carries an error finding:

* reads and the tree: level B contributes nothing → visible unless level A hides
  it;
* writes: **refused**, 422, message = the finding verbatim plus the file path —
  "the policy meant to protect `web/node_modules` did not load, so nothing under
  it is editable until it does".

Expose that as a third accessor so pol-06 can ask it directly:

```python
def policy_blocking_error() -> Optional[WorkspacePathError]:
    """A 422 when the workspace policy failed to load, else None."""
```

Checked by the *write* gate only. Reads stay available so the user can see what
is wrong — via the terminal or an external editor, since `.nojae` itself is not
addressable in the pane.

## 6. Tests

New `tests/test_editorfs_policy_file.py`, all against a `tmp_path` workspace with
its `.nojae/read-write-policy.json` written by the test:

1. absent file → no rules, no findings, `policy_blocking_error() is None`;
2. the three example rules resolve as written (`.git/HEAD` hidden,
   `a/b/__pycache__/c.pyc` hidden, `web/node_modules/react/x.js` ro);
3. last-match-wins across two rules on the same path, **in both directions**;
4. each malformed case → an error naming the file and the rule index; unknown
   `mode` names the three valid values;
5. `{"path": ".nojae/x", "mode": "rw"}` → an error, and the rule does not apply
   (assert `.nojae/x` is still hidden — the reservation survives its own file);
6. `{"path": "web/node_module", "mode": "ro"}` where `web/node_modules/` exists →
   an error whose message contains `node_modules`;
7. a rule shadowed by level A (`.env**` in a **self** workspace) → a note, not an
   error, and the effective mode is still `hidden`;
8. cache: edit the file to the same size with a bumped mtime (`os.utime`) → the
   new rules apply; switch the root to a second `tmp_path` → the second file's
   rules apply, not the first's (the regression test for §4);
9. `policy_blocking_error()` non-None for the malformed cases, while the tree
   still lists (the two halves of D4, asserted together).

## 7. Acceptance

* `pytest` green; `tests/test_editorfs.py` / `test_editorfs_write.py` untouched.
* A deliberately corrupt policy file in a scratch workspace yields: tree
  unaffected, `mode_for` = level A's answer, `policy_blocking_error()` naming the
  file and the line.
* No surface in `tree.py` / `files.py` / `mutate.py` / `gitstatus.py` mentions
  `workspace_rules` yet.

## Notes left by whoever executed it

Three things the §3 table left open, resolved in the code — read these before
"restoring" the row as written.

* **The unreachable-prefix row means *typo*, not *absent*.** A missing literal
  prefix is an **error** only when a near-miss sibling exists
  (`web/node_module` next to `web/node_modules`), because that is the only case
  with a suggestion to show — and §3 asks for one. A prefix with no near name
  (`.hg`, `.svn`, `**/build`) contributes **no finding**: pol-05's seed names
  those in *every* workspace, so taking absence as an error would ignore the seed,
  refuse writes everywhere the app goes, and fail pol-05's own "the seed round-
  trips through the loader" test. A rule written for a folder that may appear
  later is a normal thing to write in a policy file.
* **The check reports the first *missing component*, not the whole prefix** — for
  `danglin/x` it names `danglin` and suggests from the root, instead of looking
  for siblings inside a directory that does not exist. It stops at the first
  symlink (`lstat`, not `exists`/`lexists`: `exists` calls a dangling link absent,
  `lexists` swallows the `EACCES` worth distinguishing), so a workspace pointing a
  folder at another disk is not read through and not called a typo.
* **Level-A shadowing needs a self workspace to test.** Both sides of
  `is_self_workspace` are read at call time, so
  `monkeypatch.setattr(editorfs.constants, "PROJECT_ROOT", tmp_path)` alongside
  the location turns a temp folder into "the app" — which is how a test can write
  a policy file for a live level A instead of editing the real checkout.

The `readme` key is shape-checked only (string, or list of strings) and never
reaches a rule. A `not a directory` on the way to the file is an error naming the
OS error, since `.nojae` being a plain file is a mistake the user can fix; a file
that simply vanished mid-read is the absent case.
