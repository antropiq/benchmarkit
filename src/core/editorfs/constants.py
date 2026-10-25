"""Static configuration of the Editor tab's file surface.

Module attributes on purpose (the ``lsphub.constants`` convention): a test
that exercises a cap patches it here — ``editorfs.constants.MAX_TREE_ENTRIES_PER_DIR``
— instead of reaching into the function that reads it.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

# Directory names never offered by the tree. A productivity default, not a
# security rule — the deny-list in ``mcp_connectors/path_policy.py`` is the
# security rule (``.git`` is denied there too, so it could never appear).
# Kept here as a module attribute so a deployment can widen or narrow it
# without touching the component.
IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "dist",
        ".pytest_cache",
        ".coverage",
    }
)

