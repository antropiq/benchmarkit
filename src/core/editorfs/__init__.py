"""File surface of the Editor tab: a bounded view of the workspace.

Core-only (no FastAPI, no browser concerns), a sibling of ``mcphub/`` and
``lsphub/`` whose layout it mirrors — one concern per file:

* :mod:`editorfs.constants` entry cap, size cap, the ignored directory names
* :mod:`editorfs.policy` the one gate: ``resolve_in_workspace`` + the two
  ``listable``/``writable`` predicates, reusing ``mcp_connectors/path_policy``
* :mod:`editorfs.tree` one directory level -> entry dicts (``GET /api/fs/tree``)
* :mod:`editorfs.files` ``read_file`` (``GET /api/fs/file``), ``stat_paths``
  (the freshness probe) and ``write_file`` (the save route) — the guarded write:
  the write flavour of the gate, the ``mtime`` guard, a pre-save backup and an
  atomic replace
* :mod:`editorfs.mutate` the operations on *paths* rather than contents:
  ``create_path`` (the create route) makes one empty file or one folder, never
  over an existing name; ``rename_path`` (the rename route) moves one path to a
  new one — both ends and both parent chains gated, never over an occupied
  target; and ``trash_path`` (the delete route) moves one file or one whole
  folder into a dated batch under ``constants.TRASH_DIR`` — never ``unlink``ed,
  and all-or-nothing on the subtree's write policy. The parent chain and the
  leaf name both go through the write flavour of the same gate.
* :mod:`editorfs.gitstatus` the read-only git queries: ``git_status``
  (``GET /api/fs/status``, which rows to mark) and ``file_diff``
  (``GET /api/fs/diff``, the two sides of one file). They answer *about* the
  working location and never change it — no stage, no commit, no checkout —
  and every path they report goes through the same policy as the tree.

Two invariants the whole Editor surface rests on:

* every path is **relative to ``BENCHMARKIT_PATH_LOCATION``** — the same
  working location both hubs point at, so one value moves the tree, the hubs
  and the terminal's ``cwd`` together;
* **every writer here is guarded, and browser-only.** No connector, no LSP tool
  and no chat code path reaches a write route. The connector deny-list still
  applies to both directions, because the tab must not become a GUI way to read
  ``~/.ssh`` or overwrite the policy files the connectors refuse.
"""

from typing import TYPE_CHECKING, Any

from . import constants
from .files import (
    WorkspaceSaveConflict,
    backup_for_write,
    read_file,
    stat_paths,
    write_file,
)
from .gitstatus import file_diff, git_status
from .mutate import (
    WorkspacePathExists,
    create_path,
    rename_path,
    sweep_trash,
    trash_batches,
    trash_digest,
    trash_path,
)
from .policy import (
    WorkspacePathError,
    listable,
    normalise_rel,
    resolve_existing_in_workspace,
    resolve_in_workspace,
    resolve_parent_in_workspace,
    resolve_write_target,
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
