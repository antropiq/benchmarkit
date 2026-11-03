"""The app-side name for the shared case probe (win-05, gb-01).

The probe itself lives in :mod:`mcp_connectors.fscase` — the same question the
two stdio connectors ask for the root they walk (target B1 / G-D8), and a
connector process cannot import ``src``. This module keeps the app's import
path and the app's invalidation points (the seeder's write, the settings
switch) working unchanged, and re-exports the shared cache so a test can force
either answer on one runner (roadmap §9 rule 2).

The answer is *fold only inside comparisons*, never in the stored, displayed or
wire form of a path — the full contract, the probe mechanics and the cache
reasoning are the shared module's docstring.
"""

from mcp_connectors import fscase as _shared
from mcp_connectors.fscase import (  # noqa: F401 — the one cache, both spellings
    _CACHE,
    _CACHE_MAX_ROOTS,
    _PROBE_PREFIX,
    case_folded,
    invalidate_case_cache,
)

__all__ = [
    "case_folded",
    "invalidate_case_cache",
    "_CACHE",
    "_CACHE_MAX_ROOTS",
    "_PROBE_PREFIX",
]
