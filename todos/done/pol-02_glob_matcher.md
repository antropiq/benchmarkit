# pol-02 — The path-aware matcher

One function that answers *"does this pattern match this path?"*, with the
semantics every later task depends on. Unit-tested in isolation, wired to
nothing: like pol-01, this task changes no behaviour and exists so pol-04's
validation and pol-06's wiring can both be short.

Depends on: pol-01 (the mode vocabulary).

## 1. Why `fnmatch` cannot be used

`mcp_connectors/path_policy.glob_hit` matches with `fnmatch.fnmatchcase`
(`path_policy.py:135`). `fnmatch`'s `*` matches **any characters including `/`**,
so `web/*` would match `web/src/api/routes.py` as well as `web/index.html`, and
`*` alone would match everything. For a single-level deny-list entry that is
harmless; for a user-authored policy where `*` reads as "the names in this
folder", it is a silent over-match that hides or locks whole trees.

Use `pathlib.PurePosixPath.match`-like semantics, written out explicitly rather
than delegated: the stdlib's `match` right-anchors a pattern with no `**` in it,
so it compares a *suffix* instead of a path — `PurePosixPath("a/b/c").match("b/c")`
being True is exactly the surprise a policy author must not get.

## 2. The signature

```python
def pattern_matches(pattern: str, rel: str) -> bool:
    """True when `pattern` matches the normalised workspace-relative `rel`.

    Both sides are POSIX-joined relative paths: no leading "./", no trailing
    "/". The workspace root ("") is matched only by a pattern that means it,
    never by accident.
    """


def literal_prefix(pattern: str) -> str:
    """The leading run of literal components of `pattern`.

    "web/node_module/**" -> "web/node_module"; "**/__pycache__" -> "".
    pol-04 calls a rule **unreachable** when that prefix does not exist in the
    workspace, which is how a typo becomes a message instead of a rule that
    quietly never fires.
    """


def pattern_error(pattern: str) -> Optional[str]:
    """Why `pattern` can never match a workspace-relative path, or None."""
```

All three live in **`mcp_connectors/policy_glob.py`** — not under `src/`. The
reason is a hard constraint, not taste: pol-08 has the two stdio connector
processes ask the same question, and those scripts cannot import `src.core`
reliably (they are launched by absolute path, run as scripts with their own
directory on `sys.path`, and the wheel packages only `src`; pulling the whole app
package — FastAPI, litellm — into a 400-line connector is a dependency nobody
asked for). `mcp_connectors/` is the one location **both** sides already reach:
the connectors import it as their own package (`console_mcp.py:47-50` does exactly
this dance for `path_policy`), and `editorfs` reaches it through the `sys.path`
insert that `policy._path_policy()` (`policy.py:61-75`) already performs.

So the file is stdlib-only, no `src` import, no `os` calls — and pol-06 keeps a
narrow version of that import shim for the matcher. What pol-06 deletes is the
*rule list* import, not the ability to share a pure function:

```python
# editorfs/policy.py, after pol-06
from mcp_connectors.policy_glob import pattern_matches   # a matcher, not a rule
```

The distinction is worth a comment on the import line, because "why does the
Editor import from the connector folder" is the first question a reader asks.

## 3. Semantics to implement, exactly

| pattern | matches | does not match |
|---|---|---|
| `.env` | `.env` | `.env.local`, `sub/.env` |
| `web/dist` | `web/dist` and, as a folder rule, its subtree | `web/Dist`, `webx/dist` |
| `.env*` | `.env`, `.env.test`, `.env.local` | `sub/.env` |
| `*/x` | `a/x`, `b/x` | `x`, `a/b/x` |
| `**/x` | `x`, `a/x`, `a/b/x` | `xa` |
| `**/__pycache__/*` | `a/__pycache__/b.pyc` | `a/__pycache__` itself |
| `a/**/b` | `a/b`, `a/x/b`, `a/x/y/b` | `a`, `x/b` |
| `web/**` | `web` itself and everything under it | `webby` |

Plus the two policy-shaped rules, implemented *here* so the matcher and the
folder model stay one idea:

* **A rule naming a directory covers its subtree**: the pattern is tested as a
  whole-component prefix of the path too, so `web/node_modules` matches
  `web/node_modules/react/index.js` without writing `web/node_modules/**`. This
  is what makes `{"path": ".git", "mode": "hidden"}` hide everything inside.
* **Component boundary, always**: `web/node` must never match
  `web/node_modules/x`. A wildcard ends on a `/` or on the end of a name, never
  mid-name — unless the pattern says so (`.env*`, `node*`).
* Case-sensitive on POSIX. Do not copy `fnmatch`'s case-folding temptation:
  `Dist` and `dist` are two directories.

Implementation shape: split on `/`, compare the two segment lists with `*` →
`[^/]*`, `?` → `[^/]`, and `**` → "zero or more segments". A ~40-line
recursive matcher over the segment lists is easier to read (and to test) than a
generated regex. Either is fine — the tests below are the contract.

## 4. Guards the matcher enforces about the pattern itself

`pattern_matches` never raises on a weird pattern; `pattern_error` exposes the
reasons so pol-04 can turn them into validation messages. Reject: an absolute
pattern (`/etc`), a leading `~`, any `..` component, an empty component
(`a//b`), a trailing `/`, a NUL, and a bare `**` as the whole pattern — that one
means "the entire workspace" and is almost certainly a typo for `**/*`; pol-04
reports it as a note rather than applying it.

## 5. Tests

New `tests/test_policy_glob.py` — pure; no app, no fixtures except where a test
checks `literal_prefix` against a real `tmp_path` listing:

1. every row of the table above, parametrised;
2. `web/node` does **not** match `web/node_modules/x` (component boundary);
3. a folder rule matches its own path **and** any depth below it;
4. `**/x` matches at depth 0, 1 and 3 (leading `**` is optional-segments);
5. `*` does not cross `/`: `*/x` misses `a/b/x`;
6. `literal_prefix`: `web/node_module/**` → `web/node_module`; `**/x` → `""`;
   `a/*/b` → `a`; `.env*` → `""` (a wildcard already in the first segment);
7. `pattern_error` answers for each rejected shape and None for each valid one;
8. the root path `""` is matched only by the patterns that mean it;
9. case sensitivity pinned (`Dist` vs `dist`), so a future "helpful" case-fold
   cannot slip in;
10. `**/node_modules/**/README.md` matches both `node_modules/x/README.md` and
    `a/b/node_modules/README.md` — the carve-out shape from pol-00's example.

No existing test changes in this task.

## 6. Acceptance

* `pytest tests/test_policy_glob.py` green; full `pytest` green and untouched.
* `pattern_error("**/x")` is None and `pattern_error("/etc")` names the reason —
  pol-04's messages read from this function only.
* Nothing else in `src/core/editorfs` imports `policy_glob` yet (at most
  `policy_modes`, whose rule sources are still empty).

## Notes for whoever executes it

* Keep the matcher **pure**: no workspace root, no `os` calls. pol-08's connector
  processes must be able to import it and apply it to a path the caller made
  workspace-relative.
* Put the folder/subtree rule in the module docstring with one example, in the
  same words pol-00 uses — the matcher and the policy must not describe the
  model differently.

## Notes from executing it

Nothing here contradicts the task; these are the three edges the wording left
open, decided in the module docstring so pol-04, pol-06 and pol-07 inherit one
answer.

* **The root is matched by no pattern at all.** §2 says ``""`` is "matched only
  by a pattern that means it", and §4 rejects the only pattern that would mean
  it (bare ``**``), so the two rules together are "by none" — written that way,
  and the subtree walk starts at one component for exactly that reason. pol-04
  can therefore report a bare ``**`` as a note without it also having matched
  the workspace folder itself.
* **The subtree rule follows a wildcard that matched a folder.** ``logs/*``
  matches ``logs/artifacts``, so it covers ``logs/artifacts/run.txt`` too: the
  matcher cannot know which names are folders, and D3 makes *any* folder rule
  own its whole subtree. There is no spelling for "one level of a folder I have
  not named" — and pol-06 only ever asks about a path it has already resolved,
  so a rule never has to. ``*/x`` (a wildcard with a literal after it) stays
  bounded to one directory; the subtree walk only ever adds components *after*
  the whole pattern is consumed.
* **``pattern_matches`` is total.** A rejected pattern matches nothing instead
  of raising, so a caller that skips validation cannot get an exception in the
  middle of a tree walk — and an un-normalised ``rel`` (``"./a"``, ``"a/"``)
  matches nothing either, so a missed lock reads as an inert rule rather than a
  firing one.

No character classes are supported (``[old]`` is a literal directory name): the
task lists ``*``, ``?`` and ``**`` and pol-00 rules regex out, so an author gets
exactly what they typed. ``pattern_error`` answers are sentences beginning with
"it …" so pol-04 can quote them after the rule text without re-spelling it.

