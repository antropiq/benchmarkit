"""Static configuration of the Editor tab's file surface.

Module attributes on purpose (the ``lsphub.constants`` convention): a test
that exercises a cap patches it here — ``editorfs.constants.MAX_TREE_ENTRIES_PER_DIR``
— instead of reaching into the function that reads it.
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The shared policy modules live outside ``src`` so the two stdio connectors can
# read them too (pol-08), so they are imported the connectors' own way — the same
# ``sys.path`` fallback ``core.batch`` uses, because the project root is not
# always on the path. They hold the vocabulary and the machinery, never a rule
# list: that is what ``.nojae/read-write-policy.json`` and ``builtin_policy`` are.
# ``policy_glob`` (pol-02) is the matcher both policy levels are evaluated with,
# and level A (pol-03) is written in that pattern language, so it is mirrored
# here beside the vocabulary for the same reason: patchability.
try:
    from mcp_connectors import policy_glob, policy_shared
except ImportError:  # pragma: no cover - depends on how the app is started
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from mcp_connectors import policy_glob, policy_shared  # type: ignore[no-redef]

# Entries one directory listing may return. A huge checkout stays responsive
# because the tree is one level per request, and the cap bounds even that one
# level (a folder with 100k files would otherwise serialise all of them). The
# payload says ``truncated: true`` rather than dropping entries silently.
MAX_TREE_ENTRIES_PER_DIR = 2000

# Largest file the editor may load. Same number and same argument as
# ``lsphub.constants.MAX_DOCUMENT_BYTES``: streaming a giant file into a
# browser tab is as pointless as streaming it into the context window.
MAX_EDITOR_BYTES = 1_000_000

# Leading bytes scanned for a NUL when deciding whether a file is binary.
BINARY_SNIFF_BYTES = 8192

# How long one read-only git query may take (``editorfs.gitstatus``). The
# status of a huge checkout is still a directory scan, and a stale
# ``.git/index.lock`` must not be able to hang an Editor request.
GIT_COMMAND_TIMEOUT = 10.0

# Changed paths one ``/api/fs/status`` payload may carry — the sibling rule of
# ``MAX_TREE_ENTRIES_PER_DIR``: a work tree with 200k loose files answers a
# bounded list plus ``truncated: true`` instead of a huge response.
MAX_STATUS_ENTRIES = 5000

# -- the path-policy vocabulary, mirrored from ``mcp_connectors/policy_shared`` --
#
# Those names live outside ``src`` so the two stdio connectors can read them
# too (pol-08), and a test patches *this* copy — the ``mcphub.constants`` /
# ``lsphub.constants`` convention: a module attribute exists precisely so a test
# can move the lookup without touching the connector processes' default.
POLICY_DIRNAME: str = policy_shared.POLICY_DIRNAME
POLICY_FILENAME: str = policy_shared.POLICY_FILENAME
POLICY_VERSION: int = policy_shared.POLICY_VERSION

# -- the guarded write surface -------------------------------------------------

# Where a pre-save copy of a file lands. Deliberately in the *app* repo, not in
# the workspace: the working location may be ``$HOME``, so a backup tree inside
# it would be a second copy of the user's home, and it would show up in the
# ``git status`` of whatever is being edited. ``logs/`` is gitignored already,
# for the same reason ``fedit_mcp`` keeps its backups out of the tree.
# Separate from ``logs/fedit_backups/`` on purpose too: that one is the fedit
# connector's chain, and its ``list_edits``/``revert_edit`` must keep answering
# only about files that connector actually touched.
BACKUP_DIR = PROJECT_ROOT / "logs" / "editor_backups"

# Pre-save copies kept per file; older ones are pruned after each backup.
BACKUP_KEEP = 10

# Any backup older than this is swept regardless of its file — a renamed or
# deleted file leaves copies no per-file keep can ever see. 0 disables the sweep.
BACKUP_PRUNE_DAYS = 30

# Slack allowed when comparing a buffer's ``mtime`` against the disk one. The
# value round-trips through a JSON float, and filesystems tick at anything from
# nanoseconds to whole seconds, so an exact ``==`` would report a conflict for a
# save that only just happened.
MTIME_GUARD_EPSILON = 0.01

# -- delete: the trash, never ``unlink`` --------------------------------------

# Where a deleted path is *moved* to, one dated batch per delete: the active
# workspace's own reserved folder, ``<workspace>/.nojae/trash``, resolved **per
# call** by :func:`editorfs.mutate.trash_dir` — so a settings switch moves the
# trash with the workspace instead of filing another project's deleted bytes
# inside the install directory.
#
# A trash inside the workspace is not the second copy of ``$HOME`` the old
# comment feared: the move is a *rename*, so the folder holds exactly what the
# user deleted and nothing that is still on disk. And it is not browsable —
# ``.nojae`` is reserved by code in every workspace (pol-00 D5), hidden from the
# tree, the read and the write route whatever the policy file says. That is a
# stronger guarantee than the level-A ``logs/**`` rule this replaces, which was
# only live while the workspace was benchmarkit's own checkout.
TRASH_SUBDIR: str = "trash"

# ``None`` = derive from the active workspace (``<root>/.nojae/trash``). A test
# — or a deployment — that pins one absolute folder sets this instead; it is
# read at call time, never captured at import, which is what keeps the pin
# patchable in the ``lsphub.constants`` style.
TRASH_DIR: Optional[Path] = None

# Where the batches the *old* code wrote still live. Swept by ``sweep_trash``,
# never written to again.
LEGACY_TRASH_DIR = PROJECT_ROOT / "logs" / "trash"

# How long a trashed batch stays recoverable, swept on the same reasoning as
# ``BACKUP_PRUNE_DAYS``. It is what lets the confirmation name a number out
# loud ("kept 30 days") instead of promising a restore it never delivers.
# 0 disables the sweep.
TRASH_RETENTION_DAYS = 30

