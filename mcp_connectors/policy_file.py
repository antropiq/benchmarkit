#!/usr/bin/env python3
"""Parse, validate and *answer with* ``<workspace>/.nojae/read-write-policy.json``.

One schema, one set of validation messages, one verdict, **two kinds of caller**:

* the Editor tab, through ``src/core/editorfs/policy_file.py`` (which wraps this
  with the app's error type and the level-A shadow notes);
* the two stdio connectors — pol-08 — through :func:`workspace_policy_reason`
  below, which reads the same file and answers the same question so an agent and
  a human cannot be told different things about one folder.

Duplicating the parser for pol-08 would have left two schema definitions able to
drift, which is the "second list" failure this lot removes elsewhere. It is the
same argument that put :mod:`mcp_connectors.path_policy` here, and the module
obeys the same packaging rule: stdlib only, never importing from ``src`` (the
wheel ships ``src`` alone and a connector runs as a bare script path).

Two loaders, then, deliberately **not** one:

* :func:`load_policy_text` / :func:`load_policy_file` — pure, given a path and
  the text: what the app's cached loader calls;
* :func:`load_workspace_policy` / :func:`workspace_policy_reason` — the
  connector-side pair, which know a *root* and keep a per-process cache keyed on
  ``(root, mtime_ns, size)`` (pol-04 §4's key, for its §4 reasons). A connector
  cannot share the app's cache: it is a separate process, and it may be pointed
  at a workspace the app never started.

What stays on the app side, because a connector has nowhere to put it:

* the notes about rules shadowed by **level A**, the app's own rules about
  itself, which the connectors do not apply (see
  :func:`workspace_policy_reason` for why);
* turning an ``error`` finding into the app's ``WorkspacePathError``.

The vocabulary the file is written in (``Rule``, ``PolicyFinding``, ``MODES``,
``POLICY_VERSION``, ``POLICY_DIRNAME``) lives in :mod:`mcp_connectors.policy_shared`;
the pattern language is :mod:`mcp_connectors.policy_glob`, whose
:func:`~policy_glob.pattern_error` and :func:`~policy_glob.literal_prefix` are
the two questions this module asks per rule.

The rule that decides how errors are handled: **the last match wins**, so the
file is read as written and a later line may carve an exception out of an earlier
one. The consequence for a parser is that it must never "helpfully" reorder,
deduplicate or quietly drop a rule. A rule it cannot make sense of is *reported*,
and a file with any error in it contributes no rules at all — the caller turns
that into a refusal rather than a shrug (pol-00 D4), because a policy that was
supposed to protect a folder demonstrably did not load.
"""

import difflib
import json
import os
import stat as stat_module
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # imported as part of the package (tests, `python -m mcp_connectors...`)
    from . import policy_glob, policy_shared as shared
except ImportError:  # run as a plain script path, as externals.json does
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import policy_glob  # type: ignore[no-redef]
    import policy_shared as shared  # type: ignore[no-redef]

# Top-level keys the file may carry. Anything else is an error rather than a
# warning: the alternative is `"policyies": [...]` silently meaning "no rules",
# the single worst failure mode of a config file.
TOP_LEVEL_KEYS: frozenset[str] = frozenset({"version", "readme", "policies"})

# Per-rule keys. ``reason`` is optional and reaches ``Verdict.reason`` and the
# refusal sentence, which is why it is a documented field and not a comment.
RULE_KEYS: frozenset[str] = frozenset({"path", "mode", "reason"})

# How many "did you mean" names one message may offer, and how many sibling
# names are worth scanning to find them. Both caps exist because the check runs
# while loading a config file: it may list one directory, never a tree.
MAX_SUGGESTIONS = 3
MAX_SIBLING_SCAN = 500

# How far apart two names may be and still be one typing mistake (see
# :func:`_sibling_suggestion`). The veto is what keeps the seeded file loadable.
MAX_TYPO_EDITS = 2


def load_policy_text(
    text: str,
    *,
    location: str,
    workspace_root: Optional[str] = None,
    policy_dirname: Optional[str] = None,
) -> Tuple[List[shared.Rule], List[shared.PolicyFinding]]:
    """Parse one policy file's ``text`` into ``(rules, findings)``.

    ``location`` is the path quoted in every message. This module never builds
    it — it does not know where the file lives, and pol-08's connectors have a
    different one from the app's.

    ``workspace_root`` is the absolute root the rules are written relative to,
    used for exactly one disk question: does a pattern's literal prefix exist
    here (and if not, what nearby name was probably meant). With ``None`` that
    check is skipped entirely: a caller that does not say where the workspace is
    gets no opinion about what is in it.

    ``policy_dirname`` is the reserved folder a rule may not name. It defaults to
    :data:`policy_shared.POLICY_DIRNAME`, and the app passes its own
    ``editorfs.constants.POLICY_DIRNAME`` instead — that name is a module
    attribute there precisely so a test can move the lookup, and a loader that
    checked a different name than the caller looked the file up under would let a
    rule address the folder that holds the rules.

    The two kinds of finding are pol-00's split, and the contract that goes with
    them:

    * any ``KIND_ERROR`` → ``rules`` comes back **empty**. A half-validated file
      is not a policy, and pretending it is one lets writes through a folder the
      broken rules were written to protect;
    * ``KIND_NOTE`` → the rules stand. A note explains a rule that loaded
      correctly but cannot take effect, which the user is told about and cannot
      fix.

    Never raises: a file that cannot be understood has to be *shown* to the user
    with the reason, not answered with a traceback.
    """
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        return (
            [],
            [_file_error(
                location,
                f"is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
            )],
        )

    if not isinstance(document, dict):
        return (
            [],
            [_file_error(
                location,
                f"holds {_kind_of(document)}, not an object with "
                f"{_quoted(sorted(TOP_LEVEL_KEYS))}",
            )],
        )

    findings: List[shared.PolicyFinding] = [
        _file_error(
            location,
            f"has an unknown key {key!r}: this file understands "
            f"{_quoted(sorted(TOP_LEVEL_KEYS))}",
        )
        for key in sorted(document)
        if key not in TOP_LEVEL_KEYS
    ]
    findings.extend(_version_findings(document.get("version"), location))
    findings.extend(_readme_findings(document.get("readme"), location))

    rules, rule_findings = _policy_findings(
        document.get("policies"), location, workspace_root, policy_dirname
    )
    findings.extend(rule_findings)

    if any(finding.kind == shared.KIND_ERROR for finding in findings):
        return ([], findings)
    return (rules, findings)


def load_policy_file(
    path: str,
    *,
    workspace_root: Optional[str] = None,
    policy_dirname: Optional[str] = None,
) -> Tuple[List[shared.Rule], List[shared.PolicyFinding]]:
    """Read one policy file and hand its text to :func:`load_policy_text`.

    The two ways a read fails are findings, not exceptions, and they are not the
    same finding: an **absent** file is nothing at all — the normal state of a
    project that never got one, since seeding is pol-05's job and a workspace may
    refuse it — while an **unreadable** one (permissions, a directory where the
    file should be) is an error naming the path and the OS error, because that
    the user can fix.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return ([], [])
    except IsADirectoryError as exc:
        return ([], [_file_error(path, f"is a folder, not a file: {exc.strerror or exc}")])
    # ``.nojae`` itself is a plain file, so nothing can be read from under it: a
    # mistake the user can fix, and the same kind of finding as the row above.
    except NotADirectoryError as exc:
        return ([], [_file_error(path, f"cannot be reached: {exc.strerror or exc}")])
    except OSError as exc:
        return ([], [_file_error(path, f"cannot be read: {exc.strerror or exc}")])
    return load_policy_text(
        text,
        location=path,
        workspace_root=workspace_root,
        policy_dirname=policy_dirname,
    )


# -- the one question a connector asks (pol-08) ---------------------------------


def workspace_policy_path(root: str, policy_dirname: Optional[str] = None) -> str:
    """Where the policy file lives under ``root`` (it may not exist)."""
    return os.path.join(root, policy_dirname or shared.POLICY_DIRNAME,
                        shared.POLICY_FILENAME)


def workspace_policy_reason(
    rel: str,
    *,
    root: Optional[str],
    for_write: bool,
) -> Optional[str]:
    """The workspace policy's refusal for a workspace-relative ``rel``, or None.

    This is what makes a folder the user marked ``ro`` uneditable by an agent
    too, asked **in the connector's own process**: a connector may be pointed at
    a workspace by an operator who never started the app, so there is no IPC
    here and no ``src`` import -- that is why the parser, the matcher and the
    vocabulary all sit in this directory (``policy_shared``'s docstring holds the
    packaging rule).

    Four things this deliberately does **not** do, each for a stated reason:

    * **level A is not consulted.** The app's own rules (``**/.env**``,
      ``logs/**``) exist because the *pane* fights the settings dialog over a
      generated file (``config._render_env()`` rebuilds it on every save). That
      is a property of the pane, not of an agent's edit, and applying it here
      would put the connector's root outside the app and then invent rules about
      a project it has no relationship with. A reader who expects the connector
      to skip what the pane enforces is right: it does, and the reserved folder
      below is the only thing it enforces unconditionally.
    * **a path outside the root answers None.** The workspace's rules are written
      relative to the workspace; outside it the existing connector deny-list
      already spoke (and it speaks by *name*, which is what keeps ``id_rsa`` out
      of a context window wherever it lives). The two floors have different
      domains and neither is loosened by the other.
    * **no writing.** A connector never seeds the file (pol-05 seeds it from the
      app, at the two moments a folder becomes active) -- a reader that leaves a
      folder behind in someone's project is the mistake that rule exists to
      prevent.
    * **no IPC.** It reads the same file the pane reads, per-process cached.

    The verdict is the same merge the pane uses (:func:`policy_shared.verdict_for`
    -- one implementation, so an agent and a human cannot be told different
    things about one folder), and a refusal is the pane's sentence
    (:func:`policy_shared.refusal_message`), naming the level, the mode and the
    pattern verbatim so the model can *ask differently* instead of giving up.

    Failure is asymmetric exactly as it is in the app (pol-00 D4): a file that
    does not parse contributes no rules, so **reads stay open** (the user has to
    be able to look at what is wrong) and **writes are refused** naming the
    cause -- the policy that was supposed to protect a folder demonstrably did
    not load, and letting an agent rewrite inside it is the one answer that is
    not acceptable.

    The reserved folder is refused **before the file is even opened**, for both
    directions: an agent editing the rules that constrain it is the same
    accident ``DENY_COMPONENTS`` includes ``mcp_connectors`` to prevent
    (``path_policy.py``), and that must survive a corrupt file -- which is why
    the check is on the component list, not on a rule.
    """
    if not root:
        return None
    dirname = shared.POLICY_DIRNAME
    hit = policy_glob.reserved_component_hit(rel, dirname)
    if hit is not None:
        # ``LEVEL_BUILTIN`` because the reservation is code's rule, not a line in
        # the user's file: this is the one level-A fact a connector *does* apply
        # (it is not about who owns a generated file, it is about who owns the
        # rules). The pane answers the same path with the same level, so its 404
        # and this refusal are the same sentence.
        verdict = shared.Verdict(shared.HIDDEN, shared.LEVEL_BUILTIN, hit,
                                 shared.reserved_reason(hit))
        return shared.refusal_message(verdict, for_write=for_write)

    rules, findings = load_workspace_policy(root, policy_dirname=dirname)
    errors = [f for f in findings if f.kind == shared.KIND_ERROR]
    if errors and for_write:
        return (
            f"The workspace policy file did not load, so nothing in {root} can "
            f"be changed until it does: "
            + " ".join(finding.message for finding in errors)
        )
    if not rules:
        # No rules to match: an absent or empty file (reads and writes alike),
        # or a broken one on a *read*, which stays open by design.
        return None

    matched = [rule for rule in rules if policy_glob.pattern_matches(rule.pattern, rel)]
    verdict = shared.verdict_for(matched)
    if verdict.mode == shared.HIDDEN:
        return shared.refusal_message(verdict, for_write=for_write)
    if for_write and verdict.mode == shared.READ_ONLY:
        return shared.refusal_message(verdict, for_write=True)
    return None


# -- the per-process cache ------------------------------------------------------
#
# Same three keys pol-04's app-side cache uses -- ``(root, mtime_ns, size)`` --
# and for the same three reasons: the file can change while this process lives
# (the user edits it by hand between tool calls), two roots routinely hold
# same-sized config files, and a same-size edit (a pattern renamed to an
# equal-length one) is entirely possible, so a stamp cannot identify a file
# across roots.
#
# ``ROOT_DIR`` is frozen per process, so in the ordinary case there is exactly
# one entry -- a hot workspace switch relaunches both connectors with the new
# ``--root-dir`` (``routes.py`` restarts both hubs), so the new value is read at
# startup and there is nothing stale to invalidate. The bound exists for the
# tests and for a process whose root is patched, never to keep a dead root's
# rules alive.


class _WorkspaceEntry:
    __slots__ = ("stamp", "rules", "findings")

    def __init__(
        self,
        stamp: Optional[Tuple[int, int]],
        rules: Tuple[shared.Rule, ...],
        findings: Tuple[shared.PolicyFinding, ...],
    ) -> None:
        self.stamp = stamp
        self.rules = rules
        self.findings = findings


_WORKSPACE_CACHE: Dict[str, _WorkspaceEntry] = {}
_WORKSPACE_CACHE_MAX = 32


def load_workspace_policy(
    root: str,
    *,
    policy_dirname: Optional[str] = None,
) -> Tuple[List[shared.Rule], List[shared.PolicyFinding]]:
    """This workspace's ``(rules, findings)``, re-read only when the file changed."""
    dirname = policy_dirname or shared.POLICY_DIRNAME
    path = workspace_policy_path(root, dirname)
    stamp = _file_stamp(path)

    entry = _WORKSPACE_CACHE.get(root)
    if entry is not None and entry.stamp == stamp:
        return (list(entry.rules), list(entry.findings))

    rules, findings = load_policy_file(
        path, workspace_root=root, policy_dirname=dirname
    )
    entry = _WorkspaceEntry(stamp, tuple(rules), tuple(findings))
    if len(_WORKSPACE_CACHE) >= _WORKSPACE_CACHE_MAX:
        _WORKSPACE_CACHE.clear()
    _WORKSPACE_CACHE[root] = entry
    return (list(entry.rules), list(entry.findings))


def invalidate_workspace_policy_cache() -> None:
    """Forget every cached workspace. For tests, and for a process whose root moves."""
    _WORKSPACE_CACHE.clear()


def _file_stamp(path: str) -> Optional[Tuple[int, int]]:
    """``(st_mtime_ns, st_size)`` of the policy file, or None when it is absent.

    Absence is cached too -- a workspace without a policy file (its seed failed,
    or it predates the feature) should not pay a ``stat`` and a read on every
    path token of every command.
    """
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


# -- the shape checks ------------------------------------------------------------


def _file_error(location: str, problem: str) -> shared.PolicyFinding:
    """A finding about the file as a whole, naming the file."""
    return shared.PolicyFinding(
        kind=shared.KIND_ERROR, index=None,
        message=f"The workspace policy file {location} {problem}.",
    )


def _rule_error(location: str, index: int, problem: str) -> shared.PolicyFinding:
    """A finding about one rule, naming both the file and the rule's index.

    Self-contained on purpose: a message that only said ``policies[2]`` would
    depend on every caller remembering to prepend the path, and pol-08's
    connectors quote findings straight into a tool result.
    """
    return shared.PolicyFinding(
        kind=shared.KIND_ERROR, index=index, message=f"{location}: {problem}"
    )


def _version_findings(version: Any, location: str) -> List[shared.PolicyFinding]:
    if version is None:
        return [_file_error(
            location, f"has no \"version\" key (expected {shared.POLICY_VERSION})"
        )]
    # ``bool`` is an ``int`` in Python, and ``true`` is not a version.
    if isinstance(version, bool) or not isinstance(version, int):
        return [_file_error(
            location,
            f"\"version\" is {_kind_of(version)}, not a number "
            f"(expected {shared.POLICY_VERSION})",
        )]
    if version != shared.POLICY_VERSION:
        return [_file_error(
            location,
            f"\"version\" is {version}; this benchmarkit understands version "
            f"{shared.POLICY_VERSION} only",
        )]
    return []


def _readme_findings(readme: Any, location: str) -> List[shared.PolicyFinding]:
    """``readme`` is documentation, accepted and ignored by the matcher.

    JSON has no comments and the seeded file is a user's first look at the syntax
    (pol-05), so the syntax has to be able to explain itself from inside the file.
    Only its *shape* is checked here — a ``readme`` that cannot be shown as text
    is a sign the file was hand-mangled, which is worth one line, but the key
    itself never reaches a rule.
    """
    if isinstance(readme, (str, type(None))):
        return []
    if isinstance(readme, list) and all(isinstance(line, str) for line in readme):
        return []
    return [_file_error(
        location,
        "\"readme\" is neither a string nor a list of strings, so it cannot be "
        "shown as documentation",
    )]


def _policy_findings(
    policies: Any,
    location: str,
    workspace_root: Optional[str],
    policy_dirname: Optional[str],
) -> Tuple[List[shared.Rule], List[shared.PolicyFinding]]:
    """The ``policies`` list, one rule at a time, in **file order**.

    Order is not cosmetic: :func:`editorfs.policy_modes.mode_for` reads each
    rule's ``index`` for its last-match-wins step, so the index given here *is*
    the entry's position in the file — kept for refused entries too, so the
    numbers quoted in the messages point at the list the user is reading.
    """
    if policies is None:
        return ([], [_file_error(location, "has no \"policies\" list")])
    if not isinstance(policies, list):
        return ([], [_file_error(
            location, f"\"policies\" is {_kind_of(policies)}, not a list"
        )])

    rules: List[shared.Rule] = []
    findings: List[shared.PolicyFinding] = []

    for index, entry in enumerate(policies):
        if not isinstance(entry, dict):
            findings.append(_rule_error(
                location, index,
                f"policies[{index}] is {_kind_of(entry)}, not an object with "
                f"{_quoted(sorted(RULE_KEYS))}",
            ))
            continue
        rule, own = _one_rule(entry, index, location, workspace_root, policy_dirname)
        findings.extend(own)
        if rule is not None:
            rules.append(rule)

    findings.extend(_shadowed_notes(rules))
    return (rules, findings)


def _one_rule(
    entry: Dict[str, Any],
    index: int,
    location: str,
    workspace_root: Optional[str],
    policy_dirname: Optional[str],
) -> Tuple[Optional[shared.Rule], List[shared.PolicyFinding]]:
    """Validate one rule, and build it if every check passed.

    The checks run in the order a reader fixes them: first the shape (a field of
    the wrong type), then the pattern's own legality, then the reserved folder,
    then whether the pattern could reach anything on this disk. Each message
    names ``policies[index]`` and the offending value verbatim.
    """
    findings: List[shared.PolicyFinding] = [
        _rule_error(
            location, index,
            f"policies[{index}] has an unknown field {key!r}: a rule is "
            f"{_quoted(sorted(RULE_KEYS))}",
        )
        for key in sorted(entry)
        if key not in RULE_KEYS
    ]

    pattern = entry.get("path")
    if not isinstance(pattern, str) or not pattern.strip():
        findings.append(_rule_error(
            location, index,
            f"policies[{index}] has no usable \"path\": "
            + ("the key is missing" if "path" not in entry
               else f"{_kind_of(pattern)}, not a non-empty string"),
        ))
        return (None, findings)

    # ``mode`` is checked with an explicit ``str`` test rather than ``in MODES``:
    # the membership test on a frozenset raises on a list or dict, and a
    # validation module may not raise about a typo.
    mode = entry.get("mode")
    if not isinstance(mode, str) or mode not in shared.MODES:
        findings.append(_rule_error(
            location, index,
            f"policies[{index}] mode {mode!r} is not one of "
            f"{_quoted(sorted(shared.MODES))} — hidden (not shown, not reachable), "
            "ro (readable, nothing under it changes), rw (the default)",
        ))
        return (None, findings)

    reason = entry.get("reason")
    if reason is not None and not isinstance(reason, str):
        findings.append(_rule_error(
            location, index,
            f"policies[{index}] \"reason\" is {_kind_of(reason)}, not a string",
        ))
        return (None, findings)

    broken = policy_glob.pattern_error(pattern)
    if broken is not None:
        findings.append(_rule_error(
            location, index,
            f"policies[{index}] path {pattern!r} can never match a path in this "
            f"workspace: {broken}",
        ))
        return (None, findings)

    reserved = policy_dirname or shared.POLICY_DIRNAME
    if policy_glob.reserved_component_hit(pattern, reserved) is not None:
        findings.append(_rule_error(
            location, index,
            f"policies[{index}] path {pattern!r} names "
            f"{reserved!r}, the folder this file lives in, which "
            "benchmarkit reserves: it is hidden in every workspace and a rule "
            "cannot address it, hide it further, or unlock it",
        ))
        return (None, findings)

    unreachable = _unreachable_message(location, pattern, index, workspace_root)
    if unreachable is not None:
        findings.append(unreachable)
        return (None, findings)

    return (
        shared.Rule(
            mode=mode, pattern=pattern, reason=reason,
            index=index, level=shared.LEVEL_WORKSPACE,
        ),
        findings,
    )


# -- can the rule reach anything here? ------------------------------------------


def _unreachable_message(
    location: str,
    pattern: str,
    index: int,
    workspace_root: Optional[str],
) -> Optional[shared.PolicyFinding]:
    """A rule whose literal prefix is missing is a typo *only if a name close to it
    is here*; otherwise it is a folder that does not exist yet.

    A typo in a policy file is otherwise invisible: the rule never fires, the
    folder stays open, and nothing the user does afterwards explains why. So the
    prefix is checked with one ``lstat`` per component — never a tree walk — and an
    empty prefix (the pattern starts with a wildcard, ``**/node_modules``,
    ``.env*``) means there is nothing literal to check, which is never a failure.

    What a missing prefix *means* depends on whether a near-miss sibling exists,
    and the two answers are different kinds:

    * ``web/node_module`` next to a real ``web/node_modules`` — a typo: **error**,
      naming the sibling. This is pol-04's row, and the case a user acts on.
    * ``.hg`` in a checkout with no Mercurial folder — a rule about a folder that
      is simply not here, and an ordinary one: pol-05's seeded file names ``.hg``,
      ``.svn`` and the cache directories in *every* workspace, so calling absence
      an error would fail writes everywhere the app goes and break pol-05's own
      "the seed round-trips through the loader" test. **No finding** — a rule that
      matches nothing today may match something tomorrow, and a policy file is
      allowed to be written for a folder that appears later.

    An ``OSError`` in the lookup (an unreadable directory half-way up) also answers
    ``None``: a filesystem condition the author did not type is not a reason to
    refuse to load their file.
    """
    if workspace_root is None:
        return None
    prefix = policy_glob.literal_prefix(pattern)
    if not prefix:
        return None
    missing = _first_missing_component(workspace_root, prefix)
    if missing is None:
        return None
    parent_rel, name = missing

    suggestion = _sibling_suggestion(workspace_root, parent_rel, name)
    if not suggestion:
        return None

    # ``/``-joined: a suggestion is a **policy path**, and the policy file's
    # form is ``/``-joined on every platform (roadmap §9 rule 1). ``os.path.join``
    # would hand a Windows user a ``web\src`` they cannot paste into their own
    # file as a rule that matches.
    names = ", ".join(
        f"`{parent_rel + '/' + near if parent_rel else near}`"
        for near in suggestion
    )
    return _rule_error(
        location,
        index,
        f"policies[{index}] path {pattern!r} can never match: "
        f"{'this workspace has no' if not parent_rel else f'`{parent_rel}` has no'} "
        f"{name!r} — did you mean {names}?",
    )


def _first_missing_component(
    workspace_root: str, prefix: str
) -> Optional[Tuple[str, str]]:
    """``(parent_rel, name)`` of the first component of ``prefix`` that is not here.

    ``None`` when the prefix is whole, and also when the answer cannot be known —
    the two are deliberately the same, because this feeds a *typo* accusation.

    One ``lstat`` per component, never a walk, and ``lstat`` rather than
    ``os.path.exists`` or ``lexists``: ``exists`` answers *False* for a symlink
    whose target is missing (a dangling link is a name the author can see in the
    folder, and the matcher judges names, not what they resolve to), and
    ``lexists`` swallows the ``EACCES`` this has to tell apart from a missing
    name. The walk stops at the first link, because following it could leave the
    workspace and a link to a directory elsewhere may hold exactly the folder the
    rule means.

    Reporting the *first* missing component — not the whole prefix — is what makes
    the suggestion useful: for ``danglin/x`` the author's mistake is the name
    ``danglin`` in the root, and its siblings are the candidates. ``ENOTDIR`` (a
    component above is a plain file) counts as missing here too, since nothing can
    be called that. Any other ``OSError`` — an unreadable directory half-way up —
    answers ``None``: no claim, no error finding.
    """
    components = prefix.split("/")
    current = os.path.abspath(workspace_root)
    for depth, component in enumerate(components):
        current = os.path.join(current, component)
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return ("/".join(components[:depth]), component)
        except NotADirectoryError:
            return ("/".join(components[:depth]), component)
        except OSError:
            return None
        if stat_module.S_ISLNK(info.st_mode):
            return None
    return None


def _sibling_suggestion(
    workspace_root: str, parent_rel: str, name: str
) -> List[str]:
    """Names in ``parent_rel`` that are plausibly what ``name`` was meant to be.

    Two tests, and the second is the one that keeps the feature usable.

    **Similarity**: ``difflib.get_close_matches`` over at most
    ``MAX_SIBLING_SCAN`` entries of that **one** directory — the listing this
    check is allowed to do, never a walk — capped at ``MAX_SUGGESTIONS``. The cap
    and the scan limit exist for the same reason: it is the difference between an
    error a user acts on and one they ignore, and it must not cost a second of
    loading a config file. A listing that fails contributes no suggestion: the
    message about the missing name stands on its own.

    **Distance**: at most ``MAX_TYPO_EDITS`` character edits apart
    (:func:`_is_a_typo`). A ratio alone cannot tell a mistyped name from an
    unrelated one once names get short — ``.svn`` and ``.venv`` score 0.67 — and
    this is not an academic case: pol-05's seeded file writes ``.svn``, ``.hg``
    and the cache directories into **every** workspace it touches, so a project
    with a ``.venv`` and no Subversion checkout would have its whole policy
    rejected as a typo the author never typed, refusing every write in it (D4).
    The claim is "you mistyped this", and a typo is a short edit distance, not a
    family resemblance: ``node_module``/``node_modules`` and ``danglin``/
    ``dangling`` are one edit, ``scr``/``src`` is two, and two tool directories
    that diverge past that are two different directories.

    Both are needed: the ratio rejects a long name with a short common run, the
    distance rejects a short name that happens to share most of its letters.
    """
    parent = os.path.join(workspace_root, parent_rel) if parent_rel else workspace_root
    try:
        with os.scandir(parent) as entries:
            # ``islice`` so a 100k-entry folder costs 500 iterations, not one walk.
            names = [entry.name for entry in islice(entries, MAX_SIBLING_SCAN)]
    except OSError:
        return []
    return [
        near
        for near in difflib.get_close_matches(name, names, n=MAX_SUGGESTIONS)
        if _is_a_typo(name, near)
    ]


def _is_a_typo(missing: str, near: str) -> bool:
    """Are the two names within ``MAX_TYPO_EDITS`` single-character edits?

    Plain Levenshtein, computed on two rows, over names of the size a directory
    entry has — this runs on at most ``MAX_SUGGESTIONS`` pairs per rule, so its
    cost is not worth a cleverer algorithm.
    """
    previous = list(range(len(near) + 1))
    for i, ch in enumerate(missing, start=1):
        current = [i]
        for j, other in enumerate(near, start=1):
            current.append(min(
                previous[j] + 1,          # delete
                current[j - 1] + 1,       # insert
                previous[j - 1] + (ch != other),  # substitute
            ))
        previous = current
    return previous[len(near)] <= MAX_TYPO_EDITS


# -- shadowing inside the file ---------------------------------------------------


def _shadowed_notes(rules: List[shared.Rule]) -> List[shared.PolicyFinding]:
    """Notes for a rule that another rule of the same file already swallows.

    Two claims are made, and only these two — a wrong note about a rule that
    *does* fire would train the user to ignore the panel:

    * an **identical** pair (same pattern, same mode): the later one is the one
      that answers, so the earlier line does nothing;
    * a rule written *under* an earlier ``hidden`` one (``.git/HEAD`` after
      ``.git``): :func:`policy_shared.verdict_for` treats ``hidden`` as absolute,
      so a carve-out of a hidden folder is dead on arrival — pol-00's "not even
      by carve-out", said while the file is still being read.

    The containment test is one match, and it is sound because of the subtree
    rule: if the earlier ``hidden`` pattern matches the **literal prefix** of the
    later pattern, then it matches every path that later pattern can reach
    (pattern matching already handles the wildcards, so ``**/__pycache__`` counts
    as well as ``.git``). A later pattern with no literal prefix at all —
    ``**/README.md`` — is never claimed: its reach cannot be summarised by one
    path, and no note beats a wrong note.
    """
    notes: List[shared.PolicyFinding] = []
    # The index is the rule's position in *the file*, which is not its position in
    # this list once a refused entry has been skipped — so the pattern travels too.
    hidden_rules: List[Tuple[int, str]] = [
        (rule.index, rule.pattern) for rule in rules if rule.mode == shared.HIDDEN
    ]
    seen: Dict[Tuple[str, str], int] = {}

    for rule in rules:
        key = (rule.pattern, rule.mode)
        earlier = seen.get(key)
        if earlier is not None:
            notes.append(_note(
                rule.index,
                f"policies[{rule.index}] repeats policies[{earlier}] "
                f"({rule.pattern!r}, {rule.mode}) — of two identical rules the "
                "later one is the answer, so this line changes nothing",
            ))
        own_prefix = policy_glob.literal_prefix(rule.pattern)
        if own_prefix:
            for hidden_index, hidden_pattern in hidden_rules:
                if hidden_index >= rule.index:
                    continue
                # The identical pair is already reported above, as a repeat.
                if hidden_pattern == rule.pattern:
                    continue
                if not policy_glob.pattern_matches(hidden_pattern, own_prefix):
                    continue
                notes.append(_note(
                    rule.index,
                    f"policies[{rule.index}] ({rule.pattern!r}) sits inside the "
                    f"hidden rule at policies[{hidden_index}] ({hidden_pattern!r}) "
                    "— a hidden folder is pruned, so no rule below it can bring "
                    "anything back",
                ))
                break
        seen[key] = rule.index
    return notes


def _note(index: int, message: str) -> shared.PolicyFinding:
    return shared.PolicyFinding(kind=shared.KIND_NOTE, index=index, message=message)


# -- wording helpers -------------------------------------------------------------


def _kind_of(value: Any) -> str:
    """What to call a JSON value in a message — the JSON word, not the Python one."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__  # pragma: no cover - json never produces this


def _quoted(names: Sequence[str]) -> str:
    return ", ".join(f"\"{name}\"" for name in names)
