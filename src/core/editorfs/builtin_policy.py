"""Level A: the rules benchmarkit has about **itself** (pol-03).

These are the files *this app* writes, regenerates, or keeps for its own use —
the half of :func:`editorfs.policy_modes.mode_for` that no workspace file can
configure. They are conditional on purpose:

*the rule that stops a pane editing ``.env`` is not "``.env`` is dangerous", it
is "this app rebuilds that file from ``SETTING_KEYS`` on every settings save"
(``config._render_env()``). That second sentence is false for every other
project, so the rule must be false for them too — point the working location at
another checkout and its ``.env`` is an ordinary file, visible in the tree and
editable.*

That is what :func:`is_self_workspace` decides, and it is the only condition in
this file. If a rule here ever needs to apply to a foreign workspace, it is the
wrong rule for this file: it belongs in the seeded policy (pol-05), where the
user can read and change it.

The reserved folder is the one exception to the condition, and it is not a
coincidence: ``.nojae`` holds the rules, and the app owns that relationship in
every workspace (D5).

Rules are written in pol-02's pattern language and matched against the
**normalised workspace-relative** path — the same form the level-B file uses, so
one matcher answers both. Order is part of the semantics (:func:`mode_for` reads
the index for its last-match-wins step), which is why this is a tuple of tuples
and not a dict.
"""

import os
from typing import List, Tuple

from . import constants
from .constants import policy_glob
from .constants import policy_shared as shared
from .policy import workspace_root


def is_self_workspace() -> bool:
    """True when the active workspace IS the benchmarkit install directory.

    One comparison, and :func:`editorfs.policy.workspace_root` already does the
    resolving: ``realpath(path_location()) == realpath(PROJECT_ROOT)``. Everything
    level A owns hangs on this, so it gets one function and one docstring rather
    than an inline ``==`` at each rule.

    The comparison is **exact**, not a prefix: the app's own parent directory, or
    a copy of the repo somewhere else, is not the app — the file that copy
    regenerates is not the one this process reads.
    """
    # ``constants.PROJECT_ROOT`` (the install dir as this package sees it) rather
    # than a second import of ``src.config``, and read at call time so a test can
    # move either side of the comparison.
    return workspace_root() == os.path.realpath(str(constants.PROJECT_ROOT))


# ``(pattern, mode, reason)`` in evaluation order. Each reason is echoed verbatim
# by :func:`editorfs.policy_modes.refusal_message`, because "benchmarkit hides
# this because the settings dialog regenerates it" is a different conversation
# from "your policy hides it" — and the first is a rule the reader cannot edit.
BUILTIN_RULES: Tuple[Tuple[str, str, str], ...] = (
    # The ``**/`` is the point: what this rule replaces is a *name* check — the
    # connector floor's ``DENY_FILE_PREFIXES = (".env",)``, which reaches
    # ``.env``, ``.env.test`` and ``sub/.env.local`` at any depth. Level A has to
    # reach the same set or the swap would open ``.env`` files it never meant to.
    # A bare ``.env**`` would cover only the root one, because pol-02's ``**``
    # never crosses a component *backwards*. It is ``hidden`` rather than ``ro``
    # for the same reason as the floor's name check: the settings dialog is that
    # file's only editor, and the pane must not become the second one.
    (
        "**/.env**",
        shared.HIDDEN,
        "benchmarkit regenerates it from the settings template on every save",
    ),
    # ``BACKUP_DIR`` (``constants.py``) lives under ``logs/``, beside the
    # ``fedit`` connector's chain — outside the workspace *by path*, but inside
    # the tree whenever the workspace is the repo, so the Editor could browse
    # the app's own pre-save copies. A workspace pointed at another project has
    # no reason to hide a folder called ``logs``, which is exactly why this rule
    # is level A and not the seeded file. The trash is not in here: it lives in
    # each workspace's reserved ``.nojae/trash``, hidden by D5's reservation.
    ("logs/**", shared.HIDDEN, "the app's own backups"),
    # Build output: worth looking at, never worth hand-editing, and
    # ``npm run build`` overwrites whatever is there.
    ("web/dist/**", shared.READ_ONLY, "vite build output"),
)


def reserved_reason() -> str:
    """Why the reserved folder is hidden, spelled with the name in use.

    Built per call rather than at import: the folder name is a ``constants``
    attribute precisely so a test can move it, and a reason naming a folder the
    rules no longer reserve would be a sentence about a file that is not special.
    """
    return (
        f"the {constants.POLICY_DIRNAME} folder holds the workspace policy and "
        "is reserved by benchmarkit"
    )


def builtin_rule_list() -> List[shared.Rule]:
    """Every level-A rule as a ``Rule``, in evaluation order — **matched or not**.

    What the resolver filters, unfiltered: :func:`builtin_rules` is this list
    narrowed by the matcher, so a panel that shows *this* list shows the exact
    rules the resolver holds — the same ``Rule`` objects, the same indices, no
    second copy of the table that could drift (the point pol-01 made about one
    source).

    The reserved folder is deliberately **not** in here: it is level A's one
    unconditional rule, and the policy panel treats it as its own line rather
    than as one of the workspace-conditional rules.

    Empty when the workspace is not this installation — which is the condition
    pol-03 exists for, and what the panel renders as a caption rather than as an
    empty rule list that would look like a rule the user could have deleted.
    """
    if not is_self_workspace():
        return []
    return [
        shared.Rule(mode=mode, pattern=pattern, reason=reason, index=index,
                    level=shared.LEVEL_BUILTIN)
        for index, (pattern, mode, reason) in enumerate(BUILTIN_RULES)
    ]


def builtin_rules(rel: str) -> List[shared.Rule]:
    """Every level-A rule matching the normalised workspace-relative ``rel``.

    Never raises and never touches the disk: the caller resolved the path before
    asking (the three locks in :mod:`editorfs.policy`), and a rule here is a
    statement about a *name*, not about what is on disk under it.

    The reserved folder is the one rule that applies in **every** workspace,
    self or foreign, whether or not a policy file exists or loaded — and no
    level-B rule can override it (D5): a pattern naming ``.nojae`` is rejected
    by pol-04's validation, and even a matching one could not beat ``hidden``,
    which :func:`policy_shared.verdict_for` treats as absolute.
    """
    if policy_glob.reserved_component_hit(rel, constants.POLICY_DIRNAME) is not None:
        # Quoted as the reserved name itself, so the refusal names the folder the
        # user has to edit the file in — not a pattern they never wrote. It is a
        # **built-in** verdict, so the sentence says benchmarkit reserved it and
        # not "your policy did" — and pol-08's connectors build the same Rule
        # with the same level, which is what makes their refusal match this one.
        return [
            shared.Rule(
                mode=shared.HIDDEN,
                pattern=constants.POLICY_DIRNAME,
                reason=reserved_reason(),
                # After the listed rules, so it never displaces one in a quote.
                index=len(BUILTIN_RULES),
                level=shared.LEVEL_BUILTIN,
            )
        ]

    if not is_self_workspace():
        return []

    return [
        shared.Rule(
            mode=mode,
            pattern=pattern,
            reason=reason,
            index=index,
            level=shared.LEVEL_BUILTIN,
        )
        for index, (pattern, mode, reason) in enumerate(BUILTIN_RULES)
        if policy_glob.pattern_matches(pattern, rel)
    ]
