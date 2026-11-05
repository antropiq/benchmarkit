"""MCP connector discovery, tool formatting, and lifecycle management.

Core-only: no FastAPI or browser concerns. Loads the declarative connector
entries of the workspace's ``.nojae/mcp-settings.json`` (kilo.json-style
``remote``/``local`` entries with an ``enable`` flag) — that file is the one
and only source of connectors, seeded from the committed template
``mcp_connectors/mcp-settings.default.json`` when a folder becomes the
working location — connects one persistent MCP client per enabled connector
for the whole lifetime of the app, on a dedicated asyncio thread, so
synchronous callers (chat worker threads) can bridge onto it with
:meth:`McpManager.call_tool_sync`.

Layout — the module members live one concern per file, mirroring
``core/lsphub``:

* :mod:`mcphub.constants` paths (including the workspace declaration
  resolver), retry/timeout policy
* :mod:`mcphub.validation` shared ``headers``/``env`` and command shapes
* :mod:`mcphub.discovery` reading/writing the workspace declaration file
* :mod:`mcphub.seed` leaving a seeded declaration behind when a folder
  becomes the working location
* :mod:`mcphub.formatting` MCP tool/result objects -> OpenAI/plain text
* :mod:`mcphub.process` the subprocess environment helper (shared with lsphub)
* :mod:`mcphub.manager` ``McpManager``: lifecycle, hot toggles, sync bridge

The heavy ``mcp`` package is imported lazily so discovery and formatting
keep working when the optional ``[mcp]`` extra is not installed.
"""

from typing import TYPE_CHECKING, Any

from . import constants
from .discovery import (
    load_externals,
    mcp_installed,
    mcp_reach_is_refused,
    set_externals_enabled,
)
from .formatting import extract_tool_result_text, tools_to_openai_format
from .manager import McpManager
from .process import connector_environment
from .seed import ensure_mcp_settings_file, mcp_settings_file_state
from .validation import argv_from_command, str_map

# Make the constants re-exported through ``__getattr__`` visible to type
# checkers (they appear in ``__all__`` and are readable as ``mcphub.X``).
# These imports are never executed at runtime, so the names are still
# resolved dynamically by ``__getattr__`` and stay patchable on
# ``mcphub.constants``.
if TYPE_CHECKING:
    from .constants import (
        CONNECT_INTERVAL,
        CONNECT_RETRIES,
        CONNECTORS_MISSING_HINT,
        EXTERNAL_CONNECT_RETRIES,
        MCP_CONNECTORS_DIR,
        MCP_SETTINGS_FILENAME,
        MCP_SETTINGS_PATH,
        MCP_SETTINGS_TEMPLATE,
        PROJECT_ROOT,
        STOP_GRACE_SECONDS,
        mcp_settings_path,
    )

_MISSING = object()


def __getattr__(name: str) -> Any:  # PEP 562
    """Resolve a constant from :mod:`mcphub.constants` at lookup time.

    Deliberately not a plain ``from .constants import …``: the connector
    directory and the workspace declaration location are module attributes
    precisely so they can be redirected (the tests do), and the code reads
    them through this module. A copy bound here would go stale under a patch
    and make the redirect silently stop working.
    """
    value = getattr(constants, name, _MISSING)
    if value is not _MISSING:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CONNECT_INTERVAL",
    "CONNECT_RETRIES",
    "CONNECTORS_MISSING_HINT",
    "EXTERNAL_CONNECT_RETRIES",
    "MCP_CONNECTORS_DIR",
    "MCP_SETTINGS_FILENAME",
    "MCP_SETTINGS_PATH",
    "MCP_SETTINGS_TEMPLATE",
    "McpManager",
    "PROJECT_ROOT",
    "STOP_GRACE_SECONDS",
    "argv_from_command",
    "connector_environment",
    "ensure_mcp_settings_file",
    "extract_tool_result_text",
    "load_externals",
    "mcp_installed",
    "mcp_reach_is_refused",
    "mcp_settings_file_state",
    "mcp_settings_path",
    "set_externals_enabled",
    "str_map",
    "tools_to_openai_format",
]
