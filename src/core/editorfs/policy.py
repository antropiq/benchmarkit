"""The one gate every Editor-tab path goes through.

``resolve_in_workspace`` is the whole file surface of the Editor tab: turn a
caller-supplied root-relative path into a real path that is provably inside
the active workspace and provably not refused by the connector policy. Every
surface that touches disk — the tree, the file read, the stat probe, the write
endpoints — calls it, so none of them can disagree about what is readable or
writable: one policy per surface, reusing
:mod:`mcp_connectors.path_policy` rather than keeping a second list of
protected paths.

Three independent locks, because each alone is bypassable in a way the others
are not:

* the textual check rejects the obvious attack (absolute path, ``..``);
* the ``realpath`` prefix check rejects a symlink *inside* the workspace that
  points at ``/etc``;
* the deny-list rejects ``~/.ssh``-shaped names that a workspace of ``$HOME``
  would otherwise expose through the first two locks.

The workspace root itself is additionally checked against the LSP
workspace-root deny-list (``lsp_connectors/restricted-paths.txt``), the same
guard ``POST /api/path-location`` applies when the location is set — so the
tree inherits it for free and cannot browse a tree a language server is
forbidden to index.

``writable`` is the *write* flavour of the same list, and it gates every write:
the stricter component set (``.git``, ``.ssh``, ``.pyenv``, ``mcp_connectors``
and the secret stores), every protected prefix rather than only the credential
ones, and the same secret-filename rules. A mutation checks the whole ancestor
chain, not just the leaf — a write whose parent is protected and whose own name
is innocent is refused too, which is what stops a save landing inside ``.git``
through a path the leaf check alone would wave through. The refusal text is the
one ``path_policy.deny_reason`` already produces for the connectors, so the
browser and a connector are told the same rule in the same words.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from src.config import PROJECT_ROOT, path_location


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


def _path_policy():
    """The shared connector deny-list, imported the connectors' own way.

    ``mcp_connectors`` lives at the project root, outside the ``src`` package,
    so it is importable as a package only when that root is on ``sys.path``;
    the fallback mirrors what ``core.batch`` does. Resolved at call time, so a
    patch of the module (the tests') is seen here too.
    """
    try:
        from mcp_connectors import path_policy
    except ImportError:  # pragma: no cover - depends on how the app is started
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from mcp_connectors import path_policy  # type: ignore[no-redef]
    return path_policy


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


def refusal(resolved: Path, *, for_write: bool) -> Optional[str]:
    """The deny-list's own sentence about ``resolved``, or None.

    The wording comes from :func:`mcp_connectors.path_policy.deny_reason` and
    is not re-phrased here: a browser and a connector are told the same rule in
    the same words, which is what makes one list auditable.

    Public because :mod:`editorfs.mutate` needs it for a create's **leaf name**:
    a name that is not on disk yet cannot go through the realpath locks (there
    is nothing to resolve, and a dangling symlink's ``realpath`` would look like
    an escape), but the list still applies to it.
    """
    deny = _path_policy().deny_reason(resolved, for_write=for_write)
    if deny is None:
        return None
    verb = "write to" if for_write else "read from"
    return f"Path is refused by the connector path policy: {deny} " \
        f"(connectors never {verb} it)"


def _resolve(normalised: str, *, for_write: bool) -> Path:
    """The two remaining locks for an already-normalised relative path."""
    workspace = _check_workspace_root()
    candidate = os.path.realpath(os.path.join(workspace, normalised))
    if candidate != workspace and not candidate.startswith(workspace + os.sep):
        raise WorkspacePathError(
            422,
            f"Path is outside the working location: {normalised} "
            f"(workspace {workspace})",
        )
    deny = refusal(Path(candidate), for_write=for_write)
    if deny is not None:
        raise WorkspacePathError(422, deny)
    return Path(candidate)


def resolve_in_workspace(
    rel: Optional[str], *, for_write: bool = False
) -> Tuple[str, Path]:
    """Resolve ``rel`` to ``(normalised_rel, real_absolute_path)``.

    Raises :class:`WorkspacePathError` (422) when any of the three locks
    refuses the path. The returned path is always the ``realpath``, so every
    later ``stat``/``open`` follows exactly the links the policy followed.

    Read callers leave ``for_write`` at its default. The write routes check the
    leaf with the write flavour through :func:`resolve_write_target`, so this
    function stays the read gate and the write flavour has one entry point.
    """
    normalised = normalise_rel(rel)
    return (normalised or "", _resolve(normalised, for_write=for_write))


def resolve_write_target(rel: Optional[str]) -> Tuple[Tuple[str, Path], ...]:
    """The whole gate for a mutation: every path in the chain, write flavour.

    Returns the resolutions of each ancestor of ``rel`` followed by the
    resolution of ``rel`` itself — a tuple of ``(normalised_rel, path)`` pairs,
    the last one being the leaf.

    Checking the chain, not only the leaf, is what refuses a write whose
    *parent* is protected while its own name is innocent. The component rules
    already look at every component of the leaf's path, so what the chain adds
    is the per-name rules at each level: a folder named ``.env.local`` holding
    ``notes.txt`` has an innocent leaf name and a protected parent, and only a
    parent-level check sees that. A denied ancestor is reported as *that* path,
    so the message names the level the write has to move away from.
    """
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


def resolve_existing_in_workspace(
    rel: Optional[str], *, for_write: bool = True
) -> Tuple[str, Path]:
    """The gate plus a 404 when the path is absent — the shape a mutation wants.

    A write to a path that is not there is ``404`` naming it (a save of a file
    the model deleted, a rename whose source is gone), and that wording lives
    here so the routes cannot each invent their own.

    With ``for_write`` this is the *whole* write gate: the chain of ancestors
    and the leaf, all with the write flavour — which is what save, rename and
    delete need and why they call this rather than the read gate. The leaf is
    then required to exist; a create is the one mutation that must not, and it
    uses :func:`resolve_parent_in_workspace` instead.
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
    # The leaf name itself, against the same list. It cannot go through the
    # realpath locks — it is not on disk, and an existing dangling symlink's
    # realpath would look like an escape it is not — so the deny-list is the
    # check that counts here. Doing it before any ``stat`` also keeps a denied
    # name a refusal rather than a "does it exist?" answer.
    deny = refusal(parent / name, for_write=True)
    if deny is not None:
        raise WorkspacePathError(422, deny)
    if not parent.is_dir():
        if parent.exists():
            raise WorkspacePathError(
                400, f"Parent path is a file, not a folder: {parent_rel}"
            )
        raise WorkspacePathError(
            409, f"Parent folder does not exist: {parent_rel or '.'}"
        )
    return (normalised, parent, name)


def listable(path: Path) -> bool:
    """True when ``path`` may appear in a listing (the read flavour)."""
    return _path_policy().deny_reason(path, for_write=False) is None


def writable(path: Path) -> bool:
    """True when ``path`` would survive the *write* flavour of the policy.

    The predicate behind every write the Editor tab makes: the stricter
    component set (``.git``, ``.ssh``, ``.pyenv``, ``mcp_connectors`` plus the
    secret stores) and every protected prefix, not only the credential ones.
    It answers about one path; :func:`resolve_write_target` is what applies it
    to a whole chain.
    """
    return _path_policy().deny_reason(path, for_write=True) is None

