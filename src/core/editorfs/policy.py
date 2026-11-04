"""The one gate every Editor-tab path goes through.

``resolve_in_workspace`` is the whole file surface of the Editor tab: turn a
caller-supplied root-relative path into a real path that is provably inside
the active workspace and provably not refused by the workspace's policy. Every
surface that touches disk — the tree, the file read, the stat probe, the write
endpoints — calls it, so none of them can disagree about what is readable or
writable: one verdict per surface, computed by
:func:`editorfs.policy_modes.mode_for` from the two policy levels.

Three independent locks, because each alone is bypassable in a way the others
are not:

* the textual check rejects the obvious attack (absolute path, ``..``);
* the ``realpath`` prefix check rejects a symlink *inside* the workspace that
  points at ``/etc``;
* the workspace's own rules reject ``~/.ssh``-shaped names that a workspace of
  ``$HOME`` would otherwise expose through the first two locks.

The workspace root itself is additionally checked against the LSP
workspace-root deny-list (``lsp_connectors/restricted-paths.txt``), the same
guard ``POST /api/path-location`` applies when the location is set — so the
tree inherits it for free and cannot browse a tree a language server is
forbidden to index.

A mutation checks the whole ancestor **chain**, not just the leaf: a write
whose parent is read-only while its own name is innocent is refused too, which
is what stops a save landing inside a locked folder through a path the leaf
check alone would wave through (:func:`resolve_write_target`). The refusal text
names the level, the mode and the rule that matched, so what the browser reads
is what the policy panel shows.

The **answers differ by what was attempted** (pol-00 D8), and that asymmetry is
the whole shape of this module's error codes:

* *looking* at a hidden path answers **404** — an observation must be
  indistinguishable from absence, or a refusal becomes a way to learn that the
  file is there;
* *changing* anything hidden or read-only answers **422 naming the rule** —
  "not there" is the green light for ``O_CREAT | O_EXCL``, so a mutation must
  never report absence for a path that exists.

The rules and the **locks** are different things and live apart: the locks above
are not configurable and did not change in pol-06; the rules come from
:mod:`editorfs.policy_modes` (level A: the app's rules about itself, live only
while the workspace *is* this installation; level B:
``<workspace>/.nojae/read-write-policy.json``). Everything this module resolves
is a *real path known to be inside the workspace* — which is what lets that
matcher work on workspace-relative text, and what keeps a policy file from
becoming a way to browse outside the working location.

One of those rules is not policy-configurable, and it is checked here on the
component list, so no ``stat`` is involved: the reserved ``.nojae`` folder (D5)
— a folder nobody may address, hide further or unlock, whatever the file inside
it says. :mod:`editorfs.builtin_policy` answers it unconditionally, so the
reservation survives a workspace whose policy file is broken or absent;
:mod:`editorfs.policy_seed` leaves that folder behind as soon as a folder is
targeted, and a config folder that then showed up in the tree or in
``git status`` is the leak the reservation exists to prevent.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from src.config import path_location

from . import fscase
from .constants import path_norm, policy_shared as shared


class WorkspacePathError(Exception):
    """A path that must be refused, carrying its HTTP status.

    Like :class:`core.batch.BatchListingError`: the status is part of the
    core's contract (422 "you asked outside the workspace" versus 404 "that is
    not there" are two different messages in the browser), and ``routes`` only
    re-raises it as an ``HTTPException``.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _verdict(normalised: str) -> shared.Verdict:
    """The policy's answer for a normalised workspace-relative path.

    Reached lazily, and only for this: :mod:`editorfs.policy_modes` imports the
    two rule sources, and one of them (:mod:`editorfs.builtin_policy`) asks
    *this* module for :func:`workspace_root`, so a module-level import of the
    verdict would be a cycle.

    The path it is handed is **relative**, never the resolved absolute one — the
    whole reason pol-01 chose relative matching. Rules are written against the
    workspace the user sees, so a workspace of ``$HOME`` must not turn ``web/**``
    into an accidental absolute match.
    """
    from .policy_modes import mode_for

    return mode_for(normalised)


def policy_blocking_error() -> Optional[WorkspacePathError]:
    """A 422 when this workspace's policy file failed to load, else ``None``.

    Asked by the write gate alone (pol-00 D4), and public because that is the
    one question a surface outside the gate may also need to answer honestly.
    Reached lazily for the same reason as :func:`_verdict`:
    :mod:`editorfs.policy_file` imports this module's :class:`WorkspacePathError`.
    The message is pol-04's whole answer — the file path and the finding
    verbatim — because the only useful thing a refusal can tell the user is which
    file to fix and what is wrong in it. Reads and the tree never ask it: the
    file has to stay reachable so it can be fixed.
    """
    from .policy_file import policy_blocking_error as findings_block_writes

    return findings_block_writes()


def _refusal(verdict: shared.Verdict, *, for_write: bool) -> str:
    """One sentence naming the level, the mode and the rule that matched.

    Written by :func:`editorfs.policy_modes.refusal_message`, not here, so the
    pane, the connectors and the policy panel describe a refusal in the same
    words.
    """
    from .policy_modes import refusal_message

    return refusal_message(verdict, for_write=for_write)


def hidden_error(normalised: str, *, for_write: bool) -> WorkspacePathError:
    """The refusal for a hidden path: **404** to look, **422** to touch (D8).

    An *observation* of a hidden path has to be indistinguishable from absence —
    a refusal that names the rule would confirm the file is there, which is the
    leak a hidden name exists to prevent. A *mutation* is the opposite case:
    "not there" is the green light for ``O_CREAT | O_EXCL``, so it has to say
    which rule refused it instead.

    The old gate answered 422 both ways because the connector deny-list was a
    secret kept from the browser; a policy the user wrote themselves (and pol-07
    shows them) is not a secret, which is what makes the asymmetry both safe and
    useful.
    """
    verdict = _verdict(normalised)
    if for_write:
        return WorkspacePathError(422, _refusal(verdict, for_write=True))
    return missing_path_error(normalised)


def workspace_root() -> str:
    """The active workspace: ``path_location()``, fully resolved."""
    return os.path.realpath(path_location())


def _is_within(parent: str, child: str, *, fold: bool) -> bool:
    r"""Is ``child`` the same path as ``parent`` or below it? The one rule (A-D2.2).

    Both arguments are **absolute, already-resolved** text (``os.path.realpath``
    or ``Path.resolve()``) — this answers where a path landed, never whether a
    caller may ask for it; the textual lock in :func:`normalise_rel` and the
    verdict in :func:`editorfs.policy_modes.mode_for` stay the other two.

    It replaces the prefix test each of five sites hand-built — the child
    equal to the parent, or beginning with the parent plus one separator
    (roadmap §2.5). Those were separator-consistent, both sides came from
    ``realpath``, but they were exact-text, so on a volume that folds case one
    spelling of one folder stopped matching. Comparing **component lists**
    through the shared splitter (``path_norm.path_parts``) keeps the meaning and
    drops the two traps: a ``\``-spelled Windows path compares correctly on any
    runner, and ``/a/bc`` is not inside ``/a/b`` because the components differ,
    not because a trailing separator is missing.

    ``fold`` is always explicit (win-05's convention, roadmap §9 rule 2): the
    answer comes from :func:`editorfs.fscase.case_folded` at the call site, which
    keeps both outcomes testable on the Linux runner. With ``fold=False`` this is
    exactly the old prefix test, so every existing Linux verdict is unchanged.
    Folding cannot merge two components — no character casefolds to a separator —
    so the component counts still line up.

    A parent with **no components to compare** contains nothing: ``""`` is a
    caller bug, and refusing is the safe direction for a lock (see
    :func:`_absolute_parts`, which is what leaves a component list empty).

    **Which root's fold answers a deny-list compare.** The sites that compare a
    root against ``lsp_connectors/restricted-paths.txt`` (:func:`_check_workspace_root`,
    ``api/routes.py``, ``lsphub/discovery.py``) ask about a pair that may sit on
    different volumes — a denied credential store under one profile home and a
    workspace on another volume.

    The honest simple choice, taken here and to be revisited by target B: fold
    such a compare when the probe of the **workspace** says its filesystem folds
    case — the workspace the app works in, or the one being switched *to* at the
    settings dialog — so there is one answer per running configuration and no
    probing a folder the app has just decided it may not open.
    """
    parent_parts = _absolute_parts(parent)
    child_parts = _absolute_parts(child)
    if not parent_parts or len(child_parts) < len(parent_parts):
        return False
    if fold:
        parent_parts = tuple(part.casefold() for part in parent_parts)
        child_parts = tuple(part.casefold() for part in child_parts)
    return child_parts[: len(parent_parts)] == parent_parts


def _absolute_parts(text: str) -> Tuple[str, ...]:
    """``text`` as the components a containment compare is allowed to see.

    The shared splitter (win-02/win-04) answers what a path *text* is, so it is
    used here too — and for an absolute path its leading separator survives as an
    empty leading component. Dropping the empties keeps the compare about
    **names**, and keeps the guard above honest: a text with no component of its
    own — ``""``, or the bare ``/`` the old prefix test matched nothing against
    either — contributes no list to match, so it contains nothing rather than
    everything. Two real paths cannot be merged by it: what remains still has to
    match in order and in count.
    """
    return tuple(part for part in path_norm.path_parts(text) if part)


def _check_workspace_root() -> str:
    """The workspace root, refused when it sits on the LSP root deny-list.

    ``POST /path-location`` already refuses such a value, but the environment
    can also be set by hand before boot — the tree must not become a way to
    browse a location the app itself would have rejected.
    """
    # Imported lazily: the editorfs surface must stay usable when the optional
    # [lsp] extra (and so parts of lsphub) are not installed.
    from src.core.lsphub.discovery import restricted_lsp_roots

    root = workspace_root()
    denied_entries = restricted_lsp_roots()
    # One fold answer for the whole compare, from the *workspace's* volume
    # (see :func:`_is_within`); the denied entries themselves are never probed.
    # Asked only when the list has lines, so a hub without a deny-list costs no
    # disk question.
    fold = fscase.case_folded(root) if denied_entries else False
    for denied in denied_entries:
        entry = os.path.realpath(denied)
        if _is_within(entry, root, fold=fold):
            raise WorkspacePathError(
                422,
                f"Working location is on the deny-list: {denied}",
            )
    return root


def _qualified_refusal(text: str, rel: Optional[str]) -> WorkspacePathError:
    """The 422 for a path text that **cannot** be workspace-relative, by shape.

    A-D2.1: the sentence names what is wrong with the text — an absolute POSIX
    path, a UNC share root, a drive-relative name, an absolute Windows path —
    never where the path happened to land. "outside the working location" is the
    containment lock's sentence, and using it here would let the reader conclude
    that a different workspace would take the path, when the shape never can.

    Every sentence keeps the phrase **"relative to the workspace"**: the shape
    has to be refused whatever the working location is, and that is the one
    clause the browser and the tests both match on.
    """
    workspace = path_location()
    backslash = "\\"
    # The leading-separator shapes are read off the caller's own text, so a UNC
    # root keeps its name (``canonical_rel`` collapses ``//srv/share`` to a
    # single-slash path). The drive letter is read off the **canonical** form:
    # ``..`` cancellation can create the leading spec (``a\..\C:x`` -> ``C:x``),
    # and the shape to name is the one a later ``join`` would have received.
    canonical = path_norm.canonical_rel(text)
    if text[:2] in ("//", backslash * 2):
        # ``\\srv\share`` / ``//srv/share`` — a share root, whose first name is
        # a server rather than a component of this workspace.
        reason = "it is a UNC share root"
    elif text[:1] in ("/", backslash):
        # ``/etc/x``: on Windows ``os.path.isabs`` calls this *relative*, so it
        # used to reach ``os.path.join`` and escape through it.
        reason = "it is an absolute path, rooted at the machine rather than here"
    elif canonical[2:3] in ("/", backslash):
        # ``C:\x`` — absolute on a drive the workspace is not on.
        reason = f"it names a drive ({canonical[:2]}), not the workspace"
    else:
        # ``C:x`` names that drive's *current* directory, so
        # ``ntpath.join(workspace, "C:x")`` answers a path on another drive.
        # ``a:b`` is the residue of the same rule (win-02): a legal POSIX name,
        # and no rule refuses ``C:x`` without refusing it too, so the sentence
        # names the shape rather than the letter.
        reason = (
            f"it names a drive ({canonical[:2]}), not the workspace "
            "(a drive-relative path)"
        )
    return WorkspacePathError(
        422,
        f"Path must be relative to the workspace: {rel!r} — {reason} "
        f"(workspace {workspace})",
    )


def normalise_rel(rel: Optional[str]) -> str:
    """The textual lock: the canonical root-relative path, or a 422.

    The one inbound lock (A-D1): everything downstream — the matcher, the
    ancestor walk, every ``path`` payload echoed to the browser, ``results.json``
    — sees ``/``-joined workspace-relative text **on every platform**. A backslash
    is accepted here and never emitted, so Windows ``src\\config.py`` and Linux
    ``src/config.py`` name one file, and a path string still means the same thing
    in a report from either box.

    Returns "" for the workspace root itself. ``path_norm.canonical_rel`` collapses
    ``a/../b`` *before* the check, so a traversal that stays inside the workspace
    stays legal and only a ``..`` that survives normalisation (one that would
    leave it) is refused — as are a NUL byte, a qualified path of any shape (see
    :func:`_qualified_refusal`) and a leading ``~``, which the connectors never
    expand and which is therefore refused **before** any expansion could run
    (A-D2.3).

    The order of those checks is load-bearing: a shape is refused before it is
    canonicalised, because ``canonical_rel`` deliberately keeps a leading
    separator rather than translating the shape away. ``os.path.isabs`` is not
    consulted at all — it is OS-relative and misses ``/etc/x``, ``//srv/share``
    and ``C:x`` on Windows (roadmap §2.1), and a second absolute check here would
    be a second definition of the same thing, free to drift from the first.
    """
    text = (rel or "").strip()
    if "\0" in text:
        raise WorkspacePathError(422, "Path must not contain a NUL byte")
    if path_norm.is_qualified(text):
        raise _qualified_refusal(text, rel)
    if text.startswith("~"):
        raise WorkspacePathError(
            422,
            f"Path must be relative to the workspace: {rel} — a leading '~' is "
            f"a home directory the workspace does not own, and it is not expanded "
            f"(workspace {path_location()})",
        )
    normalised = path_norm.canonical_rel(text)
    if normalised in ("", "."):
        return ""
    if normalised == ".." or ".." in Path(normalised).parts:
        raise WorkspacePathError(422, f"Path must not contain '..': {rel}")
    return normalised


def _resolve(normalised: str, *, for_write: bool) -> Path:
    """The two remaining locks, then the verdict, for one relative path.

    The order is not arbitrary: the locks answer *where* the path is (outside
    the workspace there is no policy to consult) and the verdict answers *what
    the policy says* about it. An ``ro`` path is a refusal only for a write —
    reading it is the whole point of marking it read-only.
    """
    workspace = _check_workspace_root()
    candidate = os.path.realpath(os.path.join(workspace, normalised))
    if not _is_within(workspace, candidate, fold=fscase.case_folded(workspace)):
        raise WorkspacePathError(
            422,
            f"Path is outside the working location: {normalised} "
            f"(workspace {workspace})",
        )
    verdict = _verdict(normalised)
    if verdict.mode == shared.HIDDEN:
        raise hidden_error(normalised, for_write=for_write)
    if for_write and verdict.mode == shared.READ_ONLY:
        raise WorkspacePathError(422, _refusal(verdict, for_write=True))
    return Path(candidate)


def resolve_in_workspace(
    rel: Optional[str], *, for_write: bool = False
) -> Tuple[str, Path]:
    """Resolve ``rel`` to ``(normalised_rel, real_absolute_path)``.

    Raises :class:`WorkspacePathError` — 422 when a lock or an ``ro``/``hidden``
    write refuses the path, **404** when a read asks for a hidden one. The
    returned path is always the ``realpath``, so every later ``stat``/``open``
    follows exactly the links the policy followed.

    Read callers leave ``for_write`` at its default. The write routes check the
    leaf through :func:`resolve_write_target`, so this function stays the read
    gate and the write gate has one entry point.
    """
    normalised = normalise_rel(rel)
    return (normalised or "", _resolve(normalised, for_write=for_write))


def resolve_write_target(rel: Optional[str]) -> Tuple[Tuple[str, Path], ...]:
    """The strict gate: every path in the chain, each needing ``rw``.

    A create, a rename or a delete changes the **directory entries** of every
    folder above the name it touches, so each of those folders must be
    read-write in its own right: this returns the resolutions of each ancestor
    of ``rel`` followed by the resolution of ``rel`` itself — a tuple of
    ``(normalised_rel, path)`` pairs, the last one the leaf — and refuses the
    first level whose verdict is ``hidden`` or ``ro``. That is what makes an
    ``ro`` folder mean "nothing under it changes, and not its own listing
    either" (pol-00 D3): adding to it, emptying it, and moving something in or
    out of it are all refused. A refusal from an ancestor is reported as *that*
    path, so the message names the folder the write has to move away from.

    Saving the **content** of a file that already exists is not a change to any
    listing, so it asks the leaf alone — see
    :func:`resolve_save_target`, which is what makes a later rule able to carve
    one file back to ``rw`` under a folder rule.

    This is the **one** entry point of the write surface, which is why the check
    for a policy file that failed to load lives here rather than in a route: a
    route cannot forget it (pol-00 D4). A broken level B contributes no rules,
    so the tree and the read stay open for the user to go look at the file — but
    writes are refused until it loads, because the rules that were supposed to
    protect a folder demonstrably did not run.
    """
    blocked = policy_blocking_error()
    if blocked is not None:
        raise blocked
    normalised = normalise_rel(rel)
    resolved = _resolve(normalised, for_write=True)
    # ``path_norm.path_parts`` rather than a split of this module's own: the
    # ancestor walk and the matcher must agree about what a component is *by
    # construction* (win-04 / A-D1). The payload shape is unchanged — one pair
    # per level, ancestors first, each prefix ``/``-joined.
    chain: Tuple[Tuple[str, Path], ...] = ()
    components = path_norm.path_parts(normalised)
    for depth in range(1, len(components)):
        prefix = "/".join(components[:depth])
        chain += ((prefix, _resolve(prefix, for_write=True)),)
    return chain + ((normalised, resolved),)


def missing_path_error(rel: Optional[str]) -> WorkspacePathError:
    """The one 404 wording for a path a mutation expected to find.

    A save of a file the model deleted, a rename whose source is gone: the
    client needs to tell "not there" from "refused", so the sentence is written
    once here rather than per route.
    """
    return WorkspacePathError(404, f"Path does not exist: {rel or ''}")


def resolve_save_target(rel: Optional[str]) -> Tuple[str, Path]:
    """The gate for a **content** write: the leaf's verdict, plus a 404.

    Replacing the bytes of a file that is already there changes no directory
    listing, so only the leaf needs ``rw``. That is the difference from
    :func:`resolve_write_target`, and it is what makes a carve-out work: with
    ``**/node_modules`` ``ro`` and ``**/node_modules/**/README.md`` ``rw``, the
    README saves and its sibling does not — because the subtree rule makes the
    sibling ``ro`` *in its own right*, not merely a child of something locked.

    Everything else about a save is unchanged: a ``hidden`` leaf is 422 (a
    mutation never answers "absent"), the broken-policy-file block (D4) is
    checked here too so neither gate can be bypassed, and the leaf has to exist
    — a save is not how a file is created.
    """
    blocked = policy_blocking_error()
    if blocked is not None:
        raise blocked
    normalised = normalise_rel(rel)
    resolved = _resolve(normalised, for_write=True)
    if not resolved.exists():
        raise missing_path_error(normalised or rel or "")
    return (normalised, resolved)


def resolve_existing_in_workspace(
    rel: Optional[str], *, for_write: bool = True
) -> Tuple[str, Path]:
    """The gate plus a 404 when the path is absent — the shape a mutation wants.

    A write to a path that is not there is ``404`` naming it (a save of a file
    the model deleted, a rename whose source is gone), and that wording lives
    here so the routes cannot each invent their own.

    With ``for_write`` this is the *strict* write gate — the whole chain, each
    level needing ``rw`` — which is what a rename and a delete need, because
    both rewrite the directory entries above the name they touch. Saving a
    buffer's content is the exception and calls :func:`resolve_save_target`
    instead. The leaf is then required to exist; a create is the one mutation
    that must not, and it uses :func:`resolve_parent_in_workspace`.
    """
    if for_write:
        normalised, resolved = resolve_write_target(rel)[-1]
    else:
        normalised, resolved = resolve_in_workspace(rel)
    if not resolved.exists():
        raise missing_path_error(normalised or rel or "")
    return (normalised, resolved)


def resolve_parent_in_workspace(rel: Optional[str]) -> Tuple[str, Path, str]:
    """``(normalised_rel, parent_path, name)`` for a name that does not exist yet.

    A create targets a name that is not on disk, so resolving the leaf cannot
    be the check: this resolves its **parent** with the write gate and requires
    that to be an existing directory. An empty parent (a name at the workspace
    root) resolves to the workspace root itself. The parent goes through
    :func:`resolve_write_target`, so its own ancestor chain is gated too — a
    leaf name may be innocent while a folder above it is protected.

    An absent parent is **409** naming it, not a 404: the name the user typed is
    the thing that may be wrong (a typo in a dialog), and "the name is fine, the
    folder around it is not there" is a state to report rather than a missing
    resource. And nothing is created on the way to that answer — never a
    ``mkdir -p`` side effect, because intermediate folders are a choice the
    client makes deliberately, one gated request per level.
    """
    normalised = normalise_rel(rel)
    if not normalised:
        raise WorkspacePathError(
            400, "A new file or folder needs a name inside the workspace"
        )
    # One splitter again (win-04 / A-D1): on a ``\``-shaped input the right-hand
    # split found no separator and handed back an empty parent, so a create
    # *named* the workspace root (roadmap §2.4). The parent is everything but the
    # last component, re-joined with ``/`` and re-normalised as before.
    components = path_norm.path_parts(normalised)
    name = components[-1]
    parent_rel = normalise_rel("/".join(components[:-1]))
    _, parent = resolve_write_target(parent_rel)[-1]
    # The leaf name itself, asked of the same verdict — on the **relative** path,
    # since that is the form the rules are written in. It cannot go through the
    # realpath locks: it is not on disk, and an existing dangling symlink's
    # ``realpath`` would look like an escape it is not. Asking before any
    # ``stat`` is what keeps a hidden name a 422 that names the rule, rather
    # than the 409 "parent does not exist" that would confirm the name is free.
    leaf = _verdict(normalised)
    if leaf.mode == shared.HIDDEN:
        raise hidden_error(normalised, for_write=True)
    if leaf.mode == shared.READ_ONLY:
        # A name the policy marks read-only may not be made, either: ``ro`` says
        # nothing at that path changes, and a create is a change.
        raise WorkspacePathError(422, _refusal(leaf, for_write=True))
    if not parent.is_dir():
        if parent.exists():
            raise WorkspacePathError(
                400, f"Parent path is a file, not a folder: {parent_rel}"
            )
        raise WorkspacePathError(
            409, f"Parent folder does not exist: {parent_rel or '.'}"
        )
    return (normalised, parent, name)

