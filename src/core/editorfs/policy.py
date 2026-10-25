"""The one gate every Editor-tab path goes through.

``resolve_in_workspace`` is the whole file surface of the Editor tab: turn a
caller-supplied root-relative path into a real path that is provably inside
the active workspace and provably not refused by the connector policy. Both
the tree and (ed-03) the read endpoint call it, so the two can never disagree
about what is readable — decision D3 of the Editor lot: one policy per
surface that touches disk, reusing :mod:`mcp_connectors.path_policy` rather
than keeping a second list of protected paths.

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

``writable`` exists so the policy reads as the two operations it will one day
gate: this lot has no write route at all (decision D4), so nothing calls it.
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


def resolve_in_workspace(
    rel: Optional[str], *, for_write: bool = False
) -> Tuple[str, Path]:
    """Resolve ``rel`` to ``(normalised_rel, real_absolute_path)``.

    Raises :class:`WorkspacePathError` (422) when any of the three locks
    refuses the path. The returned path is always the ``realpath``, so every
    later ``stat``/``open`` follows exactly the links the policy followed.
    """
    normalised = normalise_rel(rel)
    workspace = _check_workspace_root()
    candidate = os.path.realpath(os.path.join(workspace, normalised))
    if candidate != workspace and not candidate.startswith(workspace + os.sep):
        raise WorkspacePathError(
            422,
            f"Path is outside the working location: {rel} (workspace {workspace})",
        )
    deny = _path_policy().deny_reason(Path(candidate), for_write=for_write)
    if deny is not None:
        verb = "write to" if for_write else "read from"
        raise WorkspacePathError(
            422,
            f"Path is refused by the connector path policy: {deny} "
            f"(connectors never {verb} it)",
        )
    return (normalised or "", Path(candidate))


def listable(path: Path) -> bool:
    """True when ``path`` may appear in a listing (the read flavour)."""
    return _path_policy().deny_reason(path, for_write=False) is None


def writable(path: Path) -> bool:
    """True when ``path`` would survive the *write* flavour of the policy.

    Nothing in this lot writes (decision D4 — there is no write route at all);
    the helper exists so ed-10's guarded write reuses this gate rather than
    re-deriving the rules, which keeps the read-only stance explicit in code.
    """
    return _path_policy().deny_reason(path, for_write=True) is None

