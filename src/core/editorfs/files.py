"""The file-level half of the Editor tab's surface: ``stat_paths``.

Only the cheap probe lives here so far. The read half belongs to ed-03
(``GET /api/fs/file``) and the write half to a future version (ed-10) — this
lot has no write route at all (decision D4), so nothing here opens a file for
writing, and ``policy.writable`` is the only write-shaped thing in the package.

``stat_paths`` exists for ed-09's live reload: the client knows exactly which
files are open, so it asks for *those* paths' ``mtime``/``size`` instead of
polling a whole tree or running a file watcher. Exact paths only (each one goes
through the same gate as the tree), no glob, no recursion.
"""

import os
from typing import Any, Dict, Iterable, List

from .policy import WorkspacePathError, resolve_in_workspace


def stat_paths(paths: Iterable[str]) -> List[Dict[str, Any]]:
    """``mtime``/``size``/``exists`` for each exact workspace-relative path.

    A path that is simply absent answers ``exists: false`` — that is the answer
    ed-09 needs (the model deleted an open buffer's file), not an error. A path
    the policy refuses still raises :class:`WorkspacePathError` (422), so the
    probe can never be used to *stat* a secret file into existence: the answer
    for a denied path is a refusal, never ``exists: false``, which would leak
    that it is there.
    """
    results: List[Dict[str, Any]] = []
    for raw in paths:
        normalised, resolved = resolve_in_workspace(raw)
        try:
            stat = os.stat(resolved)
        except OSError:
            results.append(
                {"path": normalised, "exists": False, "mtime": None, "size": None,
                 "dir": False}
            )
            continue
        results.append(
            {
                "path": normalised,
                "exists": True,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "dir": os.path.isdir(resolved),
            }
        )
    return results
