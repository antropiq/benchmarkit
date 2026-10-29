"""The one question every Editor surface will ask: what does the policy say?

:func:`mode_for` answers it for a normalised workspace-relative path, and it is
the single place the answer is computed — the tree, the read, the write routes,
the diff and git status all end up asking this one function (pol-06) so they
cannot disagree the way two predicates and a refusal-string did.

The three predicates it replaced are gone, which is the point: one question, one
answer, and no surface free to answer it slightly differently. What stays ahead
of it are the **locks** — the textual check, the ``realpath`` containment check
and the workspace-root deny-list in :mod:`editorfs.policy` — which run *before*
this is consulted and are not configurable, and which are why ``mode_for`` never
raises and never sees an absolute path. A verdict can only ever narrow what the
locks already allowed.

The vocabulary (modes, ranks, levels, the reserved name, ``Rule``, ``Verdict``)
lives in ``mcp_connectors/policy_shared.py`` — not here — because the two stdio
connectors answer the same question (pol-08) and cannot import ``src``. This
module holds the decision and the two rule sources; :mod:`editorfs.policy` holds
the locks. Keeping them apart is what lets a reader of ``policy.py`` see that the
locks do not change in this lot.
"""

from typing import Any, Dict, List, Optional, Tuple

from . import builtin_policy, constants, policy_file
from .policy import WorkspacePathError

from .constants import policy_glob
from .constants import policy_shared as shared

# The two rule sources, each written in its own place: level A (the app's rules
# about itself) in :mod:`editorfs.builtin_policy`, level B (the workspace's
# ``.nojae/read-write-policy.json``) in pol-04's loader. Neither is *stubbed* with
# a copy of today's deny-list — a third list is the one thing this lot exists to
# remove, so the rules arrive from their own sources or not at all.


def list_builtin_rules(rel: str) -> List[shared.Rule]:
    """Rules benchmarkit's own code contributes.

    Not written here: they live in :mod:`editorfs.builtin_policy`, where the
    condition on the workspace *being* this installation is stated once, so a
    foreign project's ``.env`` stays an ordinary file. The reserved folder is
    answered there too, and unconditionally.
    """
    return builtin_policy.builtin_rules(rel)


def list_workspace_rules(rel: str) -> List[shared.Rule]:
    """Rules from ``<workspace>/.nojae/read-write-policy.json``, as pol-04 read it.

    The loader owns the file: it parses, validates and caches it, and answers the
    two questions this one cannot — whether a pattern is malformed, and whether
    the whole file loaded. A file that did **not** load contributes no rules here,
    so ``mode_for`` keeps answering (level A's verdict, plus the default) and
    never raises: the broken file is a *write* problem, and
    :func:`editorfs.policy_file.policy_blocking_error` is what the write gate asks.

    Absence is the other normal state — a workspace that never got a policy file,
    or whose seed failed, has level B empty and level A dormant unless it *is*
    benchmarkit.
    """
    return policy_file.matching_rules(rel)


def rule_lists() -> Tuple[List[shared.Rule], List[shared.Rule]]:
    """``(level A, level B)`` — every rule the resolver *holds*, matched or not.

    pol-07's panel asks this rather than reading the two sources itself, so what
    it shows and what :func:`mode_for` merges come from the same two calls: a
    rule that answers for a path is a rule the panel lists, and a rule the panel
    lists is one the resolver holds. That is the whole point of pol-01's one
    verdict, expressed on the way out.

    Level A is empty for a foreign workspace (:func:`builtin_policy.builtin_rule_list`
    applies the pol-03 condition), which is why the panel shows the *condition*
    beside it rather than an empty table. Neither list holds the reserved folder:
    it is not a rule (see :func:`reserved_hit`), and the panel gives it its own
    line.
    """
    return builtin_policy.builtin_rule_list(), list(policy_file.workspace_rules())


def blocking_write_error() -> Optional[WorkspacePathError]:
    """The 422 the write gate would raise, or ``None`` — asked, never raised.

    The panel needs the same answer the gate gives (``blockingWrites`` in
    pol-07's payload) without being the gate: a route that *renders* the policy
    must not itself be refused by it, or a broken file would hide its own
    diagnosis behind a 422.
    """
    return policy_file.policy_blocking_error()


def policy_snapshot() -> Dict[str, Any]:
    """The whole policy for the active workspace, as one JSON-ready dict.

    What ``GET /api/fs/policy`` answers (pol-07): the panel a user reads has to
    be the policy that is *running*, so every field here comes from the same
    objects :func:`mode_for` consults — the two rule lists from
    :func:`rule_lists`, the findings from the loader that produced them, the
    write block from :func:`blocking_write_error`. A panel assembled from a
    second description of the rules is a panel that can be wrong while the tree
    is right, which is the one thing this task is for.

    Shape notes:

    * ``builtin`` and ``rules`` are **separate keys** — the levels differ in who
      may change them, and one merged list would hide that. Neither is called
      ``workspace``: the root path owns that word already (every ``/api/fs``
      payload carries it).
    * ``builtin`` is empty for a foreign workspace, and ``isSelfWorkspace`` is
      what says so. Showing the rules of an app that is not the workspace would
      imply an editable list; the UI renders the *condition*.
    * The **raw text of the file is not here**. The rules are what matter, and
      returning the file would make this route a read of a reserved path — the
      loader is the one reader pol-04 documents, and a payload is not a pane.
    * ``reserved`` names the folder that is hidden by code, not by a rule, and
      no rule can unlock it (D5). It travels as its own key so the panel can
      show it without the user reading it as one of their lines.
    * The file-level ``mtime``/``version``/``exists`` come from
      :func:`editorfs.policy_file.policy_file_info`, and ``version`` is the one
      the file was *understood* at — ``None`` when it did not parse, which is
      what the findings then explain.
    """
    builtin_rules, workspace_rules = rule_lists()
    blocked = blocking_write_error()
    return {
        "workspaceRoot": policy_file.workspace_root(),
        "isSelfWorkspace": builtin_policy.is_self_workspace(),
        "file": policy_file.policy_file_info(),
        "reserved": [constants.POLICY_DIRNAME],
        "reservedReason": builtin_policy.reserved_reason(),
        "builtin": [_rule_dict(rule) for rule in builtin_rules],
        "rules": [_rule_dict(rule) for rule in workspace_rules],
        "findings": [_finding_dict(finding)
                     for finding in policy_file.workspace_findings()],
        "blockingWrites": blocked is not None,
        # The gate's own sentence, so the footer and the 422 a save gets are one
        # string rather than two paraphrases of the same failure.
        "blockingMessage": blocked.detail if blocked is not None else None,
    }


def _rule_dict(rule: shared.Rule) -> Dict[str, Any]:
    """One ``Rule`` as the panel renders it: the pattern verbatim, plus who wrote it.

    ``index`` is the rule's position **in its own level's list** — the number a
    finding quotes, so a note about ``policies[3]`` and the row the user reads
    are the same line.
    """
    return {
        "path": rule.pattern,
        "mode": rule.mode,
        "reason": rule.reason,
        "index": rule.index,
        "level": rule.level,
    }


def _finding_dict(finding: shared.PolicyFinding) -> Dict[str, Any]:
    return {"kind": finding.kind, "index": finding.index, "message": finding.message}


def mode_for(rel: str) -> shared.Verdict:
    """The policy's answer for a *normalised workspace-relative* path.

    ``rel`` (never an absolute path): rules are written relative to the
    workspace, so matching on the absolute form would make the same file match
    different rules in different checkouts, and a workspace of ``$HOME`` would
    make ``~``-like patterns meaningless.

    Never raises. Absence, unreadability and escapes are the caller's business —
    the three locks in :mod:`editorfs.policy` answer those *before* this is
    consulted, and :func:`reserved_hit` answers the reserved folder separately so
    a broken policy file cannot open it.
    """
    candidates = list_builtin_rules(rel) + list_workspace_rules(rel)
    return shared.verdict_for(candidates)


def reserved_hit(rel: str) -> Optional[str]:
    """The component of ``rel`` equal to the reserved folder name, else None.

    Checked on the **component list, before any stat** — an absent path still has
    a name, so the answer for ``.nojae/whatever`` must not depend on whether the
    folder exists (the same trick ``resolve_parent_in_workspace`` uses for a
    create whose leaf is not on disk, generalised to every surface and depth).

    This is deliberately *not* a rule and not policy-configurable: a pattern that
    names the reserved folder is a validation error (pol-04) and no rule can
    un-reserve it. It is a separate question from :func:`mode_for` because its
    answer has to survive a broken policy file — a JSON typo must not open the
    folder that holds the JSON.
    """
    # The one implementation is in ``policy_glob``, where pol-08's stdio
    # connectors reach it too. The name is read through ``constants`` (the
    # mirrored alias) so an app-side test can move it without touching the
    # connector processes' default.
    return policy_glob.reserved_component_hit(rel, constants.POLICY_DIRNAME)


def refusal_message(verdict: shared.Verdict, *, for_write: bool) -> str:
    """One sentence naming the level, the mode and the rule that matched.

    Thin re-export of :func:`policy_shared.refusal_message`, and it stays thin on
    purpose: pol-08's acceptance is that the Editor's panel, a save's 422 and an
    agent's refusal describe the same rule **in the same words**, so the sentence
    exists once. The docstring there holds the reasoning (why the verb follows
    what was attempted, why the level is named, why a ``reason`` replaces the
    stock clause).
    """
    return shared.refusal_message(verdict, for_write=for_write)
