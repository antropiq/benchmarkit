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

from .constants import policy_shared as shared


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
    for denied in restricted_lsp_roots():
        entry = os.path.realpath(denied)
        if root == entry or root.startswith(entry + os.sep):
            raise WorkspacePathError(
                422,
                f"Working location is on the deny-list: {denied}",
            )
    return root


def normalise_rel(rel: Optional[str]) -> str:
    """The textual lock: a clean root-relative path, or a 422.

    Returns "" for the workspace root itself. ``os.path.normpath`` collapses
    ``a/../b`` *before* the check, so a traversal that stays inside the
    workspace stays legal and only a ``..`` that survives normalisation (one
    that would leave it) is refused — as are a NUL byte, an absolute path and
    a leading ``~`` (which the connectors never expand).
    """
    text = (rel or "").strip()
    if "\0" in text:
        raise WorkspacePathError(422, "Path must not contain a NUL byte")
    if os.path.isabs(text) or text.startswith("~"):
        raise WorkspacePathError(
            422,
            f"Path must be relative to the workspace: {rel} "
            f"(workspace {path_location()})",
        )
    normalised = os.path.normpath(text)
    if normalised in ("", "."):
        return ""
    if normalised == ".." or os.path.isabs(normalised) or ".." in Path(normalised).parts:
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
    if candidate != workspace and not candidate.startswith(workspace + os.sep):
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
    chain: Tuple[Tuple[str, Path], ...] = ()
    if normalised:
        prefix = ""
        for component in normalised.split("/")[:-1]:
            prefix = f"{prefix}/{component}" if prefix else component
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
    parent_rel, _, name = normalised.rpartition("/")
    parent_rel = normalise_rel(parent_rel)
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

