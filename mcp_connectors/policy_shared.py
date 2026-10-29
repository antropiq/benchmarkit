"""The vocabulary every path-policy level is written in — nothing else.

A policy answer is three things: a *mode* for the path, the *level* the rule
came from, and the *rule text* itself so the answer can be shown to whoever
asked. This module holds those names and the three shapes (``Rule``,
``Verdict``, ``PolicyFinding``), and no rule list, no matcher, no parser and
no I/O.

Why it lives here rather than under ``src``: the two stdio connectors
(``fedit_mcp``, ``console_mcp``) answer the same question as the Editor tab
(pol-08) and they run as scripts launched by absolute path, so they cannot
``import src.core`` — the wheel packages only ``src``. It is the same reason
:mod:`mcp_connectors.path_policy` sits outside the package, and the same rule
applies to the matcher (pol-02, ``policy_glob``) and the parser (pol-04,
``policy_file``): stdlib-only, never importing from ``src``.

Two levels feed one verdict:

* ``LEVEL_BUILTIN`` — benchmarkit's own rules, live **only** when the working
  location *is* this installation (the app editing itself). Conditional on
  purpose: point the location at another project and that project's ``.env``
  is an ordinary file, because the rule was never about the name — it was about
  who owns the file.
* ``LEVEL_WORKSPACE`` — the project's own rules, read from
  ``<root>/.nojae/read-write-policy.json``. Always live, and the only level a
  foreign workspace has.

Strictness order is ``hidden`` > ``ro`` > ``rw``, and the levels are a **union**:
the strictest match anywhere wins, so a workspace rule can tighten what the app's
rules say but can never loosen them.

The merge and the sentence are here too, and that is not a convenience: pol-08
has the two stdio connectors answer the same question as the Editor pane, and the
acceptance condition is that both describe one rule *in the same words*. Two
implementations of ``_winner`` or of the refusal text is how a pane and an agent
start disagreeing about the same folder — so :func:`verdict_for`,
:func:`refusal_message` and :func:`reserved_reason` exist once, and each surface
wraps them in its own error type.
"""

from dataclasses import dataclass
from typing import Optional

# -- the three outcomes ---------------------------------------------------------

HIDDEN = "hidden"
READ_ONLY = "ro"
READ_WRITE = "rw"
MODES: frozenset[str] = frozenset({HIDDEN, READ_ONLY, READ_WRITE})

# Strictness only: a match from any level wins over a looser one (D2). This is
# not a precedence *between* levels — the levels are equals, their answers are
# not.
MODE_RANK: dict[str, int] = {READ_WRITE: 0, READ_ONLY: 1, HIDDEN: 2}

# How a mode is spelled to a human, in a sentence about one path.
MODE_LABELS: dict[str, str] = {
    READ_WRITE: "read-write",
    READ_ONLY: "read-only",
    HIDDEN: "hidden",
}

# -- who wrote the rule ---------------------------------------------------------

LEVEL_BUILTIN = "app"          # benchmarkit's own rules, this workspace only
LEVEL_WORKSPACE = "workspace"  # <root>/.nojae/read-write-policy.json
LEVELS: frozenset[str] = frozenset({LEVEL_BUILTIN, LEVEL_WORKSPACE})

# The level is named in every refusal because the two are fixed differently: a
# workspace rule the user edits by hand, an app rule they cannot touch (pol-07
# shows it as a note rather than an error for exactly that reason).
LEVEL_LABELS: dict[str, str] = {
    LEVEL_BUILTIN: "benchmarkit's own rules for this workspace",
    LEVEL_WORKSPACE: "the workspace policy",
}

# -- the reserved folder, reserved by code (D5) ---------------------------------

# A rule about the folder that holds the rules is a contradiction, so this name
# is not policy-configurable: it is absent from the tree, ``stat``, the read,
# the write routes, rename, create, git status and the diff, whatever the file
# says. Patchable as a module attribute (the ``mcphub.constants`` convention) so
# an app-side test can move the lookup; the connector processes keep this
# default.
POLICY_DIRNAME = ".nojae"

POLICY_FILENAME = "read-write-policy.json"

# The ``version`` key a seeded file carries, and the only one pol-04 accepts.
POLICY_VERSION = 1


@dataclass(frozen=True)
class Rule:
    """One rule as its level holds it, in enough shape to be *shown*.

    ``pattern`` is the rule text verbatim and ``index`` its position in that
    level's list: the pair pol-07 renders in the UI and the pair every refusal
    quotes, so the sentence a user reads and the line they would edit are the
    same object. ``index`` also breaks ties — rules are evaluated in file order
    and the later one wins, which is what lets a carve-out beat the folder rule
    it carves out of.
    """

    mode: str
    pattern: str
    reason: Optional[str]
    index: int
    level: str


@dataclass(frozen=True)
class Verdict:
    """The policy's answer for one path: its mode, and who said so."""

    mode: str                # one of MODES
    level: Optional[str]     # which level matched; None for the default
    pattern: Optional[str]   # the rule that matched, verbatim from the file
    reason: Optional[str]    # optional text from the rule, echoed in the message


# -- the merge: several matching rules, one answer ------------------------------


def verdict_for(rules: list["Rule"]) -> "Verdict":
    """The one rule that decides, out of everything that matched a path.

    Takes **matching** rules (the caller matched them with ``policy_glob``) and
    answers with the same three steps the Editor pane documents, because a
    connector that merged them differently would give an agent a different answer
    for the folder the user is looking at in the pane:

    1. **``hidden`` is absolute.** If anything matched as hidden, that is the
       answer, whatever its position or level. A hidden folder is *pruned, not
       walked*, so nothing below it is reachable "not even by carve-out".
    2. **Inside one level, the last match wins** — what lets a later line carve
       an exception out of the folder rule above it. Each level therefore
       contributes only its own last match, so a level's internal order cannot
       leak across the boundary.
    3. **Between levels, the strictest answer wins** (D2).

    On equal strictness the **built-in level is the one quoted**, because a
    refusal has to name the rule the reader *cannot* edit rather than their own
    line. Empty input is the default answer, not an error: an unmatched path is
    read-write.
    """
    if not rules:
        return Verdict(READ_WRITE, None, None, None)
    hidden = [rule for rule in rules if rule.mode == HIDDEN]
    winner = _strictest(_last_match_per_level(hidden or rules))
    return Verdict(winner.mode, winner.level, winner.pattern, winner.reason)


def _last_match_per_level(rules: list["Rule"]) -> list["Rule"]:
    """The last matching rule of each level — by ``index``, its position in file.

    Ordered on the index rather than the list order, so a rule source that hands
    back its rules out of order still merges the way the file reads.
    """
    per_level: dict[str, "Rule"] = {}
    for rule in rules:
        current = per_level.get(rule.level)
        if current is None or rule.index > current.index:
            per_level[rule.level] = rule
    return list(per_level.values())


def _strictest(rules: list["Rule"]) -> "Rule":
    """The strictest of ``rules``; on a tie, the built-in level's rule."""
    return max(
        rules,
        key=lambda rule: (
            MODE_RANK[rule.mode],
            rule.level == LEVEL_BUILTIN,
        ),
    )


# -- the sentence every refusal is made of --------------------------------------


def reserved_reason(dirname: str = POLICY_DIRNAME) -> str:
    """Why the reserved folder is hidden, spelled with the name in use.

    One sentence, because the pane, the write gate and (pol-08) the connectors
    all refuse the same folder and must not paraphrase it three ways.
    """
    return (
        f"the {dirname} folder holds the workspace policy and "
        "is reserved by benchmarkit"
    )


def refusal_message(verdict: "Verdict", *, for_write: bool) -> str:
    """One sentence naming the level, the mode and the rule that matched.

    ``hidden`` and ``ro`` read differently to a human, so the verb follows what
    was attempted: a hidden path that was *looked at* is "not shown and not
    reachable", a path a mutation tried is "refused" and "nothing under it may
    change". Naming the **level** is not decoration either — a workspace rule is
    something the user can go edit by hand, an app rule is not, and a message
    that implies otherwise sends them to a file that cannot help them.

    The rule is quoted verbatim (``pattern``), because that is the exact text the
    policy panel lists and the line the user would change. A ``reason`` from the
    rule replaces the stock clause when it is there — the author's own words about
    their own folder say more than the mode's definition does.

    Every refusal in this project is this function's output: the pane wraps it in
    a 404/422, the connectors end it with their own pointer to ``edit_policy()``.
    """
    level_label = LEVEL_LABELS.get(verdict.level or "", LEVEL_LABELS[LEVEL_WORKSPACE])
    mode_label = MODE_LABELS.get(verdict.mode, verdict.mode)
    pattern = f'"{verdict.pattern or ""}"'

    if verdict.mode == HIDDEN:
        verb = "Refused by" if for_write else "Hidden by"
        # A hidden path needs no explanation once a *mutation* is what failed:
        # the mode says it, and appending "not shown" to a refusal would talk
        # about the tree instead of the save.
        tail = None if for_write else "it is not shown and not reachable"
    else:
        verb = "Refused by"
        tail = "nothing under it may change"

    message = f"{verb} {level_label}: {pattern} ({mode_label})"
    reason = (verdict.reason or "").strip()
    if reason:
        tail = reason
    if tail:
        message = f"{message} — {tail}"
    return message


# -- what the loader has to say about the file it read --------------------------

# ``error`` and ``note`` are split by *who can fix it*, which is the only
# distinction that changes what the app does with the finding:
#
# * an ``error`` means level B contributed nothing, and the app refuses writes
#   until the file is fixed (pol-00 D4) — the policy meant to protect a folder
#   demonstrably did not load, so answering as if there were no rules is the one
#   option that is not acceptable;
# * a ``note`` is about a rule that loaded fine but cannot take effect (a
#   stricter app-level rule already decided this path). The user cannot fix it
#   and does not need to, so refusing writes over it would be a wall — the
#   finding is only ever *shown* to them.
KIND_ERROR = "error"
KIND_NOTE = "note"
FINDING_KINDS: frozenset[str] = frozenset({KIND_ERROR, KIND_NOTE})


@dataclass(frozen=True)
class PolicyFinding:
    """One thing the loader noticed about the workspace's policy file.

    ``message`` is written to be read verbatim in the Editor: it names the file,
    and for a per-rule finding the index and field it came from. ``index`` is
    ``None`` for anything about the file as a whole.
    """

    kind: str                  # KIND_ERROR | KIND_NOTE
    index: Optional[int]       # which rule, when the finding is about one
    message: str
