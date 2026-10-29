"""File surface of the Editor tab: a bounded view of the workspace.

Core-only (no FastAPI, no browser concerns), a sibling of ``mcphub/`` and
``lsphub/`` whose layout it mirrors — one concern per file:

* :mod:`editorfs.constants` the entry caps and the size caps (visibility is not
  a constant any more: it is a rule in the workspace's policy file)
* :mod:`editorfs.policy_modes` the one verdict: ``mode_for`` answering what the
  policy says about a workspace-relative path, from two levels
* :mod:`editorfs.policy` the one gate: the three locks, then that verdict —
  ``resolve_in_workspace`` for a read, ``resolve_write_target`` for a change to
  a folder's listing, ``resolve_save_target`` for a change to a file's bytes.
  Every surface below asks it, so none of them can disagree about what a path
  may be listed as, read as, or changed into.
* :mod:`editorfs.builtin_policy` level A of that verdict: the rules about files
  *this app* writes or regenerates, live only while the workspace **is** the
  install directory (``is_self_workspace``), plus the reserved ``.nojae`` folder,
  which is reserved in every workspace
* :mod:`editorfs.policy_file` level B of that verdict: the one reader of
  ``<workspace>/.nojae/read-write-policy.json`` — it parses, validates and caches
  the file (the schema itself lives in ``mcp_connectors/policy_file.py``, shared
  with the connectors) and reports findings as errors or notes
* :mod:`editorfs.policy_seed` the *other* direction on that same file: it writes
  a default one when a folder becomes the working location (boot, and the
  settings switch), never over an existing one, and reports a failure as a note
* :mod:`editorfs.tree` one directory level -> entry dicts (``GET /api/fs/tree``)
* :mod:`editorfs.files` ``read_file`` (``GET /api/fs/file``), ``stat_paths``
  (the freshness probe) and ``write_file`` (the save route) — the guarded write:
  the write gate on the leaf, the ``mtime`` guard, a pre-save backup and an
  atomic replace
* :mod:`editorfs.mutate` the operations on *paths* rather than contents:
  ``create_path`` (the create route) makes one empty file or one folder, never
  over an existing name; ``rename_path`` (the rename route) moves one path to a
  new one — both ends and both parent chains gated, never over an occupied
  target; and ``trash_path`` (the delete route) moves one file or one whole
  folder into a dated batch under ``constants.TRASH_DIR`` — never ``unlink``ed,
  and all-or-nothing on the subtree's verdict. The parent chain and the leaf
  name both go through the same write gate, which is one policy answer per
  path and nothing else.
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
  guards its own surface (an agent putting a private key into a context window
  is a different question from a human looking at their own file in a pane), and
  the workspace's policy is what guards this one.
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
    normalise_rel,
    policy_blocking_error,
    resolve_existing_in_workspace,
    resolve_save_target,
    resolve_in_workspace,
    resolve_parent_in_workspace,
    resolve_write_target,
    workspace_root,
)
from .policy_modes import mode_for, refusal_message
from .tree import list_dir

# The verdict's vocabulary, from the shared module the connectors use too. Bound
# here rather than reached through ``constants`` by every caller: they are names
# of this surface, and the definitions live outside ``src`` so the two stdio
# connectors answer the same question with the same words.
MODES = constants.policy_shared.MODES
Verdict = constants.policy_shared.Verdict

_MISSING = object()


def __getattr__(name: str) -> Any:  # PEP 562
    """Resolve a constant from :mod:`editorfs.constants` at lookup time.

    Same rule as the two hubs: the caps are module attributes precisely so a test
    can patch them, so they must not be bound into this module's namespace (a
    copy would go stale under the patch).
    """
    value = getattr(constants, name, _MISSING)
    if value is not _MISSING:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from .constants import (
        BINARY_SNIFF_BYTES,
        GIT_COMMAND_TIMEOUT,
        MAX_EDITOR_BYTES,
        MAX_STATUS_ENTRIES,
        MAX_TREE_ENTRIES_PER_DIR,
        PROJECT_ROOT,
    )
