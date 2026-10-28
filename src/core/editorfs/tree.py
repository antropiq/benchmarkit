"""One directory level of the workspace, as the navigator needs it.

``list_dir`` is the core behind ``GET /api/fs/tree``. Deliberately *not*
recursive: the workspace may be a huge checkout, so the first paint costs one
``scandir`` and every branch below it is fetched on first expand.

What the payload promises (each rule mirrored by a test):

* directories first, then files, each group alphabetical and case-insensitive;
* **both kinds carry ``mtime``** — for a directory that is the mtime of the
  directory itself, i.e. when its immediate children changed, which is the
  cheap "did a file appear, disappear or get renamed here?" predicate for an
  already-expanded branch. (A directory's mtime does *not* move when a
  descendant is edited, which is why the freshness probe stats open files
  directly instead of inferring from the tree.);
* ``children`` on a directory means "expanding it will show something":
  non-empty *and* not everything in it ignored — decided with one ``any()``,
  never a second full listing;
* denied siblings are **omitted**, not marked. The navigator's footer explains
  that some files are hidden by the connector policy, so the UI never has to
  render a lock icon (or leak that a secret file exists);
* a symlinked directory is listed but not followed, so the tree can never walk
  out of the workspace one hop at a time.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import constants
from .policy import (
    WorkspacePathError,
    listable,
    resolve_in_workspace,
    workspace_root,
)


def _name_key(entry: Dict[str, Any]) -> Tuple[str, str]:
    """Sort key: case-insensitive, ties broken by the exact name."""
    return (entry["name"].lower(), entry["name"])


def _has_children(path: Path) -> bool:
    """True when expanding ``path`` would show at least one entry.

    One ``any()`` over the raw ``scandir`` — the same two filters the listing
    applies (ignored name, deny-list), so ``children: false`` never promises an
    empty branch that then lists something.
    """
    try:
        with os.scandir(path) as scan:
            return any(
                entry.name not in constants.IGNORED_DIR_NAMES
                and listable(Path(entry.path))
                for entry in scan
            )
    except OSError:
        # Unreadable directory: report it as a leaf rather than failing the
        # parent listing (the branch will answer 404/403 on its own request).
        return False


def list_dir(rel: Optional[str] = None) -> Dict[str, Any]:
    """The tree payload for one level of the workspace.

    ``rel`` is workspace-relative ("" is the root). Raises
    :class:`~src.core.editorfs.policy.WorkspacePathError` with 422 when the
    path escapes the workspace or the deny-list refuses it, 404 when it is not
    there, and 400 when the caller asked for a file — three different things
    for the navigator to say.
    """
    normalised, resolved = resolve_in_workspace(rel)
    if resolved.is_file():
        raise WorkspacePathError(400, f"Path is a file, not a directory: {rel}")
    if not resolved.is_dir():
        raise WorkspacePathError(404, f"Directory does not exist: {normalised or '.'}")

    dirs: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    truncated = False
    try:
        with os.scandir(resolved) as scan:
            for scan_entry in scan:
                if scan_entry.name in constants.IGNORED_DIR_NAMES:
                    continue
                child = Path(scan_entry.path)
                if not listable(child):
                    continue
                try:
                    stat = scan_entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                # ``is_dir(follow_symlinks=False)``: a symlink is never a
                # directory here, so the tree cannot be walked out of the
                # workspace through it.
                is_dir = scan_entry.is_dir(follow_symlinks=False)
                entry: Dict[str, Any] = {
                    "name": scan_entry.name,
                    "path": f"{normalised}/{scan_entry.name}" if normalised else scan_entry.name,
                    "dir": is_dir,
                    "mtime": stat.st_mtime,
                }
                if is_dir:
                    entry["children"] = _has_children(child)
                    dirs.append(entry)
                else:
                    entry["size"] = stat.st_size
                    files.append(entry)
    except OSError as exc:
        raise WorkspacePathError(404, f"Directory cannot be read: {exc}") from exc

    dirs.sort(key=_name_key)
    files.sort(key=_name_key)
    entries = dirs + files
    cap = constants.MAX_TREE_ENTRIES_PER_DIR
    if len(entries) > cap:
        entries = entries[:cap]
        truncated = True

    return {
        "path": normalised,
        "workspace": workspace_root(),
        "entries": entries,
        "truncated": truncated,
    }
