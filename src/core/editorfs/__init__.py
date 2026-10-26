"""File surface of the Editor tab: a bounded, read-only view of the workspace.

Core-only (no FastAPI, no browser concerns), a sibling of ``mcphub/`` and
``lsphub/`` whose layout it mirrors — one concern per file:

* :mod:`editorfs.constants` entry cap, size cap, the ignored directory names
* :mod:`editorfs.policy` the one gate: ``resolve_in_workspace`` + the two
  ``listable``/``writable`` predicates, reusing ``mcp_connectors/path_policy``
* :mod:`editorfs.tree` one directory level -> entry dicts (``GET /api/fs/tree``)
* :mod:`editorfs.files` ``read_file`` (``GET /api/fs/file``) and ``stat_paths``
  (the live-reload probe); ed-10, a future version, adds the guarded write —
  see decision D4)
* :mod:`editorfs.gitstatus` the read-only git queries: ``git_status``
  (``GET /api/fs/status``, which rows to mark) and ``file_diff``
  (``GET /api/fs/diff``, the two sides of one file). They answer *about* the
  working location and never change it — no stage, no commit, no checkout —
  and every path they report goes through the same policy as the tree.

Two invariants the whole Editor lot rests on:

* every path is **relative to ``BENCHMARKIT_PATH_LOCATION``** — the same
  working location both hubs point at, so one value moves the tree, the hubs
  and (ed-06) the terminal's ``cwd`` together;
* **nothing here writes.** The connector deny-list still applies, because the
  tab must not become a GUI way to read ``~/.ssh`` or overwrite the policy
  files the connectors refuse.
"""

from typing import TYPE_CHECKING, Any

from . import constants
from .files import read_file, stat_paths
from .gitstatus import file_diff, git_status
from .policy import (
    WorkspacePathError,
    listable,
    normalise_rel,
    resolve_in_workspace,
    writable,
    workspace_root,
)
from .tree import list_dir

_MISSING = object()


def __getattr__(name: str) -> Any:  # PEP 562
    """Resolve a constant from :mod:`editorfs.constants` at lookup time.

    Same rule as the two hubs: the caps and the ignored-names set are module
    attributes precisely so a test can patch them, so they must not be bound
    into this module's namespace (a copy would go stale under the patch).
    """
    value = getattr(constants, name, _MISSING)
    if value is not _MISSING:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from .constants import (
        BINARY_SNIFF_BYTES,
        GIT_COMMAND_TIMEOUT,
        IGNORED_DIR_NAMES,
        MAX_EDITOR_BYTES,
        MAX_STATUS_ENTRIES,
        MAX_TREE_ENTRIES_PER_DIR,
        PROJECT_ROOT,
    )
