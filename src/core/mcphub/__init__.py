"""MCP connector discovery, tool formatting, and lifecycle management.

Core-only: no FastAPI or browser concerns. Discovers the standalone MCP
connector servers in ``mcp_connectors/`` (one ``*_mcp.py`` module each),
loads declarative external connectors from
``mcp_connectors/externals.json`` (kilo.json-style ``remote``/``local``
entries with an ``enable`` flag), spawns one subprocess per connector for
the whole lifetime of the app,
and keeps a persistent MCP client per connector on a dedicated asyncio
thread so synchronous callers (chat worker threads) can bridge onto it
with :meth:`McpManager.call_tool_sync`.

Layout — the module members live one concern per file, mirroring
``core/lsphub``:

* :mod:`mcphub.constants` paths, port range, retry/timeout policy
* :mod:`mcphub.ports` the auto port allocation (preferred-port hash + probe)
* :mod:`mcphub.validation` shared ``headers``/``env`` and command shapes
* :mod:`mcphub.discovery` scanning ``*_mcp.py`` built-ins, reading/writing
  ``externals.json``
* :mod:`mcphub.formatting` MCP tool/result objects -> OpenAI/plain text
* :mod:`mcphub.process` the subprocess environment helper (shared with lsphub)
* :mod:`mcphub.manager` ``McpManager``: lifecycle, hot toggles, sync bridge

The heavy ``mcp`` package is imported lazily so discovery and formatting
keep working when the optional ``[mcp]`` extra is not installed.
"""

from typing import Any

from . import constants
from .discovery import (
    all_connectors,
    discover_connectors,
    load_externals,
    mcp_installed,
    set_externals_enabled,
)
from .formatting import extract_tool_result_text, tools_to_openai_format
from .manager import McpManager
from .ports import allocate_port, port_is_free, preferred_port
from .process import connector_environment
from .validation import argv_from_command, str_map

_MISSING = object()


def __getattr__(name: str) -> Any:  # PEP 562
    """Resolve a constant from :mod:`mcphub.constants` at lookup time.

    Deliberately not a plain ``from .constants import …``: the connector
    directory and ``externals.json`` location are module attributes precisely
    so they can be redirected (the tests do), and the code reads them
    through this module. A copy bound here would go stale under a patch and
    make the redirect silently stop working.
    """
    value = getattr(constants, name, _MISSING)
    if value is not _MISSING:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AUTO_PORT_BASE",
    "AUTO_PORT_SPAN",
    "CONNECT_INTERVAL",
    "CONNECT_RETRIES",
    "CONNECTORS_MISSING_HINT",
    "EXTERNALS_PATH",
    "EXTERNAL_CONNECT_RETRIES",
    "MCP_CONNECTORS_DIR",
    "MCP_LOGS_DIR",
    "McpManager",
    "PROJECT_ROOT",
    "STOP_GRACE_SECONDS",
    "allocate_port",
    "all_connectors",
    "argv_from_command",
    "connector_environment",
    "discover_connectors",
    "extract_tool_result_text",
    "load_externals",
    "mcp_installed",
    "port_is_free",
    "preferred_port",
    "set_externals_enabled",
    "str_map",
    "tools_to_openai_format",
]
