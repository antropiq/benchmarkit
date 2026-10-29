# pol-01 — One verdict function, nothing wired to it yet

Introduce the single question every surface will eventually ask — *what does the
policy say about this path?* — as one function returning one value. Add it
**beside** the three predicates it will replace (`listable`, `writable`,
`refusal`) and change **no call site**. This task is invisible on purpose: it
buys the later wiring the ability to be a one-line-per-surface change.

Depends on: nothing. Pol-02, 03 and 04 all feed this function.

## 1. The vocabulary

**Two modules, on purpose.**

`mcp_connectors/policy_shared.py` — stdlib-only, **no `src` imports** — holds the
names everything else reads, because pol-08 needs the two stdio connector
processes to answer the same question and they cannot `import src.core` (they run
as scripts launched by absolute path; the wheel packages only `src`). This is the
same reason `path_policy.py` is outside `src` rather than inside it.

`src/core/editorfs/policy_modes.py` holds the *decision* — the verdict and the
rule sources. Kept separate from `policy.py` so the locks and the rules stay
separable: the locks never change in this lot, and a reader of `policy.py` should
see that immediately.

```python
# mcp_connectors/policy_shared.py
HIDDEN = "hidden"
READ_ONLY = "ro"
READ_WRITE = "rw"
MODES: frozenset[str] = frozenset({HIDDEN, READ_ONLY, READ_WRITE})

# Strictness only: a match from any level wins over a looser one (D2).
MODE_RANK: dict[str, int] = {READ_WRITE: 0, READ_ONLY: 1, HIDDEN: 2}

LEVEL_BUILTIN = "app"        # benchmarkit's own rules, this workspace only
LEVEL_WORKSPACE = "workspace"  # <root>/.nojae/read-write-policy.json

POLICY_DIRNAME = ".nojae"    # reserved by code, D5 — patchable module attr
```

`POLICY_FILENAME = "read-write-policy.json"` and `POLICY_VERSION = 1` join them.
`POLICY_DIRNAME` is a **patchable module attribute**, mirrored on
`editorfs/constants.py` as an alias the app's tests patch (the
`mcphub.constants` / `lsphub.constants` convention, and what
`tests/test_editorfs.py:103` does for `IGNORED_DIR_NAMES` today) — so the
connector processes keep the immutable default while a test can move the app's
lookup.

## 2. The verdict

```python
@dataclass(frozen=True)
class Rule:                      # also in policy_shared.py — both levels and the
    mode: str                    # connectors need the same shape
    pattern: str
    reason: Optional[str]
    index: int
    level: str


@dataclass(frozen=True)
class Verdict:
    mode: str                 # one of MODES
    level: Optional[str]      # which level matched; None for the default
    pattern: Optional[str]    # the rule that matched, verbatim from the file
    reason: Optional[str]     # optional text from the rule, echoed in the message


def mode_for(rel: str) -> Verdict:
    """The policy's answer for a *normalised workspace-relative* path.

    `rel` (not an absolute path): rules are written relative to the workspace,
    so matching on the absolute form would make the same file match different
    rules in different checkouts, and a workspace of `$HOME` would make `~`-like
    patterns meaningless.

    Never raises. Absence, unreadability and escapes are the caller's business —
    the three locks in `policy.py` answer those *before* this is consulted.
    """
```

Shape of the implementation in this task (pol-03 and pol-04 fill both halves):

```python
def mode_for(rel: str) -> Verdict:
    candidates = list_builtin_rules(rel) + list_workspace_rules(rel)   # both empty for now
    if not candidates:
        return Verdict(READ_WRITE, None, None, None)
    # Union, strictest wins (D2); ties go to the later rule, so a carve-out
    # beats the folder rule it carves out of.
    winner = max(candidates, key=lambda c: (MODE_RANK[c.mode], c.index))
    return Verdict(winner.mode, winner.level, winner.pattern, winner.reason)
```

Leave `list_builtin_rules` / `list_workspace_rules` returning `[]` with a
docstring naming the task that fills them. Do **not** stub them by copying the
current deny-list into this module — the point of pol-03/04 is that the rules
arrive from their own two sources, and a copy here becomes a third list, the one
thing this lot exists to eliminate.

## 3. A reserved-name helper, because every surface needs it

```python
def reserved_hit(rel: str) -> Optional[str]:
    """The component of `rel` equal to POLICY_DIRNAME, or None.

    Checked on the **component list, before any stat** — an absent path still has
    a name, and the answer for `.nojae/whatever` must not depend on whether the
    folder exists. `policy.resolve_parent_in_workspace` (`policy.py:269-276`)
    already does exactly this for a create whose leaf is not on disk; this
    generalises it to every surface and every depth.
    """
```

`reserved_hit` is **not** a rule and not policy-configurable: a pattern that
names `.nojae` is a validation error in pol-04, and no rule can un-reserve it.
It is a separate question from `mode_for` because the answer must survive a
broken policy file (a JSON typo must not open the folder that holds the JSON).

## 4. The wording

Add, next to `policy.refusal` (which stays as it is until pol-06 retires it):

```python
def refusal_message(verdict: Verdict, *, for_write: bool) -> str:
    """One sentence naming the level, the mode and the pattern.

    `hidden` and `ro` read differently to a human: "never shown" is about the
    tree, "read-only" is about the mutation they just tried. Say which.
    """
```

Examples the tests pin:

```
Hidden by the workspace policy: ".git" (hidden) — it is not shown and not reachable
Refused by the workspace policy: "web/node_modules" (read-only) — nothing under it may change
Refused by benchmarkit's own rules for this workspace: ".env**" (hidden)
```

The last one matters: when level A is what stopped them, the message must say the
app did it and *not* suggest the user edit their policy file to allow it (they
cannot, and pol-07's panel would show the note that says so).

## 5. Tests

New `tests/test_editorfs_policy_modes.py` — pure, no HTTP, no app:

1. `mode_for` on an unmatched path → `rw`, `level is None`;
2. tie-break: two rules, `ro` then `rw` on the same path → `rw` wins on index,
   `rw` then `ro` → `ro`; strictness beats index when ranks differ (so an earlier
   `hidden` beats a later `rw`);
3. `MODE_RANK` orders `rw < ro < hidden`;
4. `reserved_hit` on `.nojae`, `sub/.nojae/x`, `.nojae/…/deep` → the component;
   on `.nojaeer`, `nojae`, `x.nojae` → None (a component equality, not a prefix);
5. `reserved_hit` answers for a path that does not exist (assert with a tmp dir
   that has no `.nojae`);
6. `refusal_message` names the pattern verbatim, the mode and the level.

Existing suites must stay green **untouched**: `tests/test_editorfs.py`,
`tests/test_editorfs_write.py`, `tests/test_write_surface_isolation.py`. If one
of them has to change, pol-01 wired something it should not have.

## 6. Acceptance

* `pytest` green, with the new file collected and the old ones byte-identical.
* The app behaves exactly as before: same tree, same refusals, same wording on
  the wire (nothing reads the new function yet).
* `grep -rn "mode_for" src/` shows the definition plus its two empty rule
  sources, and **no** caller.

## Notes for whoever executes it

* `@dataclass(frozen=True)`, explicit annotations, `from typing import Optional`
  — the repo's Python style.
* The verdict carries the *rule text* rather than a summary because pol-07
  renders the rule list in the UI from the same object: one shape, one source.
* Do not export `mode_for` from `editorfs/__init__.py` yet. The package's public
  list is asserted by `tests/test_editorfs.py:453`
  (`test_editorfs_exposes_the_guarded_write_and_nothing_beyond_it`) — widen it in
  pol-06, where the exports actually change.

## Deviation applied when executing it

The `max(candidates, key=(MODE_RANK[mode], index))` one-liner in §2 contradicts
both its own test list and pol-00, and is not what shipped:

* test 2 asks `ro` then `rw` on one path → `rw`, which a rank-first sort answers
  `ro`;
* pol-00's matching section makes **last match wins** the mechanism for carving
  an exception out of a folder rule (`**/node_modules` → `ro`, then
  `**/node_modules/**/README.md` → `rw`) — under that sort the idiom is dead code
  that pol-04 would then report as a shadowed rule.

What shipped (`policy_modes._winner`) keeps both orders, separated by what they
govern: `hidden` is absolute (pol-00: a hidden folder is pruned, "not even by
carve-out"), each level contributes only its own last match, and the strictest
**level answer** wins between them — which is D2's actual point, a level-B `rw`
never loosening a level-A rule. On equal strictness across levels the level-A
rule is the one quoted, so a refusal names the rule the user cannot edit rather
than their own line.

Consequence for pol-04: the "shadowed by an **earlier** rule of the same file"
note only fires when that earlier rule is `hidden`, since a later rule otherwise
replaces it by design.
