"""Level B: the workspace's own rules, read from ``.nojae/read-write-policy.json``.

The app-side half of pol-04. The **schema and every validation message** live in
``mcp_connectors/policy_file.py`` — stdlib-only, because pol-08 gives the two
stdio connectors the same file to read and they cannot import ``src``. What lives
here is the three things a connector cannot do, and the split is stated in both
docstrings on purpose, because the obvious "consolidation" would break pol-08:

* the **app's cache**: one read per file change rather than one per rule-matching
  call, keyed on the workspace root (:func:`_CACHE`, §4 below). pol-08 gave the
  connector processes their own cache in the shared module — a different process
  with a different lifetime, not a second copy of this one's answers, and the
  two are never consulted by the same caller;
* the **notes about level A** (:mod:`editorfs.builtin_policy`): a workspace rule
  that a stricter app rule already overrides is reported as a note, not an error,
  because the user cannot fix it and an error would be a wall;
* the app's failure type: :func:`policy_blocking_error` turns an *error* finding
  into a :class:`~editorfs.policy.WorkspacePathError` the routes raise as a 422.

This loader is the **one** reader of ``.nojae`` *in this process*. That is a
design fact, not an accident: pol-03 reserves the folder in code and pol-05 seeds
the file, and both are only legal because exactly one place in the app opens it —
so "could something else read the rules and disagree?" has one answer *per
surface*. pol-08 put a second reader in each connector process (it must: a
connector is a separate process, and may outlive or precede the app), and both
readers go through the same parser, the same cache key and the same verdict, so a
disagreement would need a bug in one of the three things they share. The reserved
folder stays unreachable even when the file inside it is broken, because
:func:`editorfs.policy_modes.reserved_hit` answers from the component list before
any rule is consulted — and the connectors answer it from
:func:`policy_glob.reserved_component_hit` before theirs.

Failing is asymmetric on purpose (pol-00 D4): a file that does not load
contributes **no rules**, so reads and the tree stay open and the user can look
at what is wrong, while every **write** is refused until it does, because the
policy that was supposed to protect a folder demonstrably did not load.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from . import builtin_policy, constants
from .constants import policy_glob
from .constants import policy_shared as shared
from .policy import WorkspacePathError, workspace_root


def policy_path() -> str:
    """Where this workspace's policy file lives (absolute; it may not exist).

    Built per call out of ``constants``: both names are module attributes
    precisely so a test can move the lookup — the ``mcphub.constants`` /
    ``lsphub.constants`` convention this repo uses for every cap.
    """
    return os.path.join(
        workspace_root(), constants.POLICY_DIRNAME, constants.POLICY_FILENAME
    )


# -- the cache -------------------------------------------------------------------


class _Entry:
    """One workspace's parsed file, plus the stamp it was read at.

    ``stamp`` is ``(st_mtime_ns, st_size)`` of the policy file, or ``None`` when
    the file is absent — a state worth caching too, so a workspace without a
    policy file pays one ``stat`` per request rather than one per surface.
    ``names`` is the ``(dirname, filename)`` the entry was read under, so a patch
    of the reserved name invalidates the entry instead of answering from it.
    """

    __slots__ = ("stamp", "names", "rules", "findings")

    def __init__(
        self,
        stamp: Optional[Tuple[int, int]],
        names: Tuple[str, str],
        rules: Tuple[shared.Rule, ...],
        findings: Tuple[shared.PolicyFinding, ...],
    ) -> None:
        self.stamp = stamp
        self.names = names
        self.rules = rules
        self.findings = findings


# Keyed on the **workspace root**, and that is the whole point of the key:
#
# 1. ``BENCHMARKIT_PATH_LOCATION`` is hot-swapped by ``POST /api/path-location``,
#    so a cache keyed on ``(mtime, size)`` alone would run the newly opened
#    project under the previous project's rules;
# 2. two roots routinely hold same-sized config files, and a same-size edit is
#    entirely possible (a pattern renamed to an equal-length one), so a stamp
#    cannot identify a file across roots;
# 3. the test suite patches the location per test; a root-less cache would make
#    those tests order-dependent.
_CACHE: Dict[str, _Entry] = {}

# How many workspaces are remembered. A process switches working location rarely
# (one hot swap per settings dialog at most), so this is a bound against a long
# uptime and a curious operator, not a working set: past it the cache is dropped
# whole, which costs a re-read of one small JSON file and never an answer.
_CACHE_MAX_ROOTS = 32


def _file_stamp(path: str) -> Optional[Tuple[int, int]]:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _current() -> _Entry:
    """This workspace's entry, re-read only when its file changed."""
    root = workspace_root()
    names = (constants.POLICY_DIRNAME, constants.POLICY_FILENAME)
    path = os.path.join(root, *names)
    stamp = _file_stamp(path)

    entry = _CACHE.get(root)
    if entry is not None and entry.stamp == stamp and entry.names == names:
        return entry

    rules, findings = _parser().load_policy_file(
        path,
        workspace_root=root,
        policy_dirname=constants.POLICY_DIRNAME,
    )
    entry = _Entry(stamp, names, tuple(rules), tuple(findings) + tuple(_level_a_notes(rules)))
    if len(_CACHE) >= _CACHE_MAX_ROOTS:
        _CACHE.clear()
    _CACHE[root] = entry
    return entry


def _parser():
    """The shared parser, reached the connectors' own way.

    ``mcp_connectors`` sits at the project root, outside the ``src`` package, so
    it is importable as a package only when that root is on ``sys.path``; the
    fallback is the same one :mod:`editorfs.constants` uses when it pulls in the
    matcher and the vocabulary, and the one ``core.batch`` uses for its own guard.
    Resolved per call, so a test's patch of the module (the cache counts reads
    through it) is seen here too.
    """
    try:
        from mcp_connectors import policy_file
    except ImportError:  # pragma: no cover - depends on how the app is started
        import sys

        if str(constants.PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(constants.PROJECT_ROOT))
        from mcp_connectors import policy_file  # type: ignore[no-redef]
    return policy_file


def invalidate_policy_cache() -> None:
    """Forget every cached workspace. For tests and for a workspace switch."""
    _CACHE.clear()


# -- the accessors ---------------------------------------------------------------


def workspace_rules() -> Tuple[shared.Rule, ...]:
    """Every rule the file contributes — matched or not, in file order.

    Empty when the file is absent, and empty when it carries an error: a broken
    policy contributes nothing, which is what :func:`policy_blocking_error` turns
    into a refusal so an unfiltered tree is not mistaken for an unprotected one.
    """
    return _current().rules


def workspace_findings() -> Tuple[shared.PolicyFinding, ...]:
    """What the loader noticed about the file — errors and notes together.

    pol-07 renders these verbatim: every message already names the file, and for
    a per-rule finding the index of the line the user is looking at.
    """
    return _current().findings


def policy_file_info() -> Dict[str, Any]:
    """What the file *is*: ``{"path", "exists", "version", "mtime"}``.

    pol-07's panel shows this above the rule list, because a rule the user
    cannot trace to a file on disk is a rule they cannot go and edit — so the
    absolute path is the payload's most useful field, and it is always filled,
    including when the file is absent (that is the path pol-05 would have seeded
    it at, which is what the empty-state caption names).

    ``version`` is the version this file was **understood** at, not a value read
    back out of it: the parser validates the key and reports a mismatch as an
    error finding that quotes the number the file actually carries, so a number
    here would either repeat a finding the panel already prints or, worse, claim
    a file parsed when it did not. ``None`` therefore means "no understood
    version" — absent file or broken file — and the findings say which.

    One ``stat`` of its own rather than the cache entry's stamp: the stamp is
    ``(mtime_ns, size)`` and exists to detect *change*, while this answers "when
    was it last modified" for a human, and re-deriving one from the other would
    make the cache's invalidation key a display format.
    """
    path = policy_path()
    try:
        info = os.stat(path) if os.path.isfile(path) else None
    except OSError:
        info = None
    loaded = info is not None and not any(
        finding.kind == shared.KIND_ERROR for finding in _current().findings
    )
    return {
        "path": path,
        "exists": info is not None,
        "version": constants.POLICY_VERSION if loaded else None,
        "mtime": info.st_mtime if info else None,
    }


def matching_rules(rel: str) -> List[shared.Rule]:
    """The level-B rules matching a normalised workspace-relative ``rel``.

    :func:`editorfs.policy_modes.list_workspace_rules` is its only caller: which
    rules match is decided where the file is read, so the verdict in
    ``policy_modes`` stays the pure merge of two rule lists it was in pol-01 and
    never has to know that a cache exists.
    """
    return [
        rule
        for rule in workspace_rules()
        if policy_glob.pattern_matches(rule.pattern, rel)
    ]


def policy_blocking_error() -> Optional[WorkspacePathError]:
    """A 422 when the workspace policy failed to load, else ``None``.

    Asked by the **write** gate only (pol-00 D4). Reads stay available so the user
    can see what is wrong — and since ``.nojae`` is reserved, they reach the file
    through the terminal or an external editor, not the pane. Each error finding's
    message is quoted verbatim with its path, because "the policy meant to protect
    ``web/node_modules`` did not load" is only actionable together with the file
    that has to be fixed.
    """
    errors = [
        finding
        for finding in _current().findings
        if finding.kind == shared.KIND_ERROR
    ]
    if not errors:
        return None
    detail = " ".join(finding.message for finding in errors)
    return WorkspacePathError(
        422,
        "The workspace policy did not load, so nothing in this workspace can be "
        f"changed until it does: {detail}",
    )


# -- shadowing by level A --------------------------------------------------------


def _level_a_notes(rules: List[shared.Rule]) -> List[shared.PolicyFinding]:
    """Notes for a workspace rule that benchmarkit's own rule already overrides.

    Level A is live only while the workspace *is* this installation
    (:func:`builtin_policy.is_self_workspace`) and it is not configurable, so a
    user rule that loses to it deserves one explanatory line rather than an
    error: refusing writes over a file the user can fix is right, refusing them
    over a rule they cannot change is a wall (pol-00's validation table).

    The test is a **witness path**, not a proof about pattern sets: one concrete
    path per rule (:func:`_witness_path`), fed to level A. A full containment test
    over this pattern language is a decision problem this module will not pretend
    to solve, and a wrong note about a rule that *does* fire would teach the user
    to ignore the panel.
    """
    notes: List[shared.PolicyFinding] = []
    for rule in rules:
        witness = _witness_path(rule.pattern)
        if not witness:
            continue
        for app_rule in builtin_policy.builtin_rules(witness):
            # Strictly stricter, per pol-00's table: a workspace rule that
            # *agrees* with the app's answer needs no explanation — the two are
            # simply saying the same thing, and a note about each of them would
            # be noise in the panel.
            if shared.MODE_RANK[app_rule.mode] <= shared.MODE_RANK[rule.mode]:
                continue
            notes.append(shared.PolicyFinding(
                kind=shared.KIND_NOTE,
                index=rule.index,
                message=(
                    f"policies[{rule.index}] ({rule.pattern!r}, {rule.mode}) "
                    f"cannot loosen benchmarkit's own rule for "
                    f"{witness!r} ({app_rule.pattern!r}, {app_rule.mode}): the "
                    "app's rules for this workspace are built in and no edit to "
                    "this file can change them; the strictest answer wins"
                ),
            ))
            break
    return notes


def _witness_path(pattern: str) -> str:
    """One concrete path ``pattern`` could be about, or ``""`` when none follows.

    ``**`` absorbs zero directories, so it drops out; ``*`` matches an empty run
    inside a name, so it does too; ``?`` insists on a character, which becomes
    ``x``. So ``.env**`` -> ``.env``, ``web/**`` -> ``web``,
    ``**/node_modules/**/README.md`` -> ``node_modules/README.md`` — in each case a
    path the rule really matches, which is all a witness has to be.
    """
    components: List[str] = []
    for component in pattern.split("/"):
        if component == policy_glob.CROSS_COMPONENT:
            continue
        stripped = component.replace("*", "").replace("?", "x")
        if not stripped:
            return ""
        components.append(stripped)
    return "/".join(components)
