"""Language-server discovery, LSP sessions, and tool synthesis.

Core-only: no FastAPI or browser concerns. Reads the declarative language
server definitions from ``lsp_connectors/lsp_servers.json`` (one entry per
server id, ``enable`` as the on/off source of truth), spawns one subprocess
per activated server, performs the LSP handshake (``initialize`` ->
``initialized``), keeps the session alive for the app's lifetime, and turns
the server's advertised capabilities into OpenAI-format tool definitions so
the Chat tab's tool loop can call them.

Like the ``mcphub`` package (which this package deliberately mirrors), all LSP I/O
runs on a private asyncio thread so synchronous callers (chat worker
threads) can bridge onto it with :meth:`LspManager.call_tool_sync`. The
language servers themselves are only ever spoken to over stdio JSON-RPC by
this hub: the LLM calls ``lsp_<id>_<action>`` with plain JSON arguments,
never a command line.

Layout — the module members live one concern per file:

* :mod:`lsphub.constants` paths, timeouts, the Volar relay method names
* :mod:`lsphub.discovery`` reading/validating/writing ``lsp_servers.json``
* :mod:`lsphub.tools` the action catalogue and the tool-name scheme
* :mod:`lsphub.session` ``DiagnosticsInbox``, ``DocumentStore``, ``ServerSession``
* :mod:`lsphub.features` the client-side feature handlers (incl. the relay)
* :mod:`lsphub.actions` document sync and the per-action LSP requests
* :mod:`lsphub.rendering` LSP structures -> compact, line-numbered text
* :mod:`lsphub.tsserver` the tsserver relay wire shapes and project answer
* :mod:`lsphub.manager` ``LspManager``: lifecycle, policy, bridge, reaper

The heavy ``pygls``/``lsprotocol`` packages are imported lazily so
discovery and the settings panel keep working when the optional ``[lsp]``
extra is not installed.
"""

from typing import Any

from .actions import fetch_diagnostics, position, run_action, sync_document, workspace_symbols
from . import constants

_MISSING = object()


def __getattr__(name: str) -> Any:  # PEP 562
    """Resolve a constant from :mod:`lsphub.constants` at lookup time.

    Deliberately not a plain ``from .constants import …``: the declaration
    file and the deny-list are module attributes precisely so they can be
    redirected (the tests do), and the code reads them through this module.
    A copy bound here would go stale under a patch and make the redirect
    silently stop working.
    """
    value = getattr(constants, name, _MISSING)
    if value is not _MISSING:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .discovery import (
    all_servers,
    load_lsp_servers,
    lsp_installed,
    restricted_lsp_roots,
    set_lsp_enabled,
)
from .features import client_capabilities, register_features
from .manager import LspManager
from .rendering import (
    apply_whole_line_edits,
    language_id_for_suffix,
    location_line,
    relative,
    render_completion,
    render_diagnostics,
    render_document_symbols,
    render_formatting,
    render_hover,
    render_locations,
    render_signature_help,
    render_symbols,
    severity_name,
    symbol_kind_name,
    truncate,
    uri_to_path,
)
from .session import DiagnosticsInbox, DocumentStore, ServerSession
from .tsserver import (
    find_project_config,
    parse_tsserver_request,
    tsserver_project_info,
    tsserver_response_params,
)
from .tools import (
    TOOL_SPECS,
    action_of_tool_name,
    tool_name_for,
    tools_for_server,
)

__all__ = [
    "DEFAULT_IDLE_SHUTDOWN_SECONDS",
    "DEFAULT_MAX_DIAGNOSTICS",
    "DIAGNOSTICS_WAIT_SECONDS",
    "DiagnosticsInbox",
    "DocumentStore",
    "INIT_TIMEOUT",
    "LSP_CONNECTORS_DIR",
    "LSP_LOGS_DIR",
    "LSP_MISSING_HINT",
    "LSP_RESTRICTED_PATHS",
    "LSP_SERVERS_PATH",
    "LspManager",
    "MAX_DOCUMENT_BYTES",
    "MAX_RESULT_CHARS",
    "PROJECT_CONFIG_NAMES",
    "PROJECT_ROOT",
    "REAPER_INTERVAL_SECONDS",
    "REQUEST_TIMEOUT",
    "STOP_GRACE_SECONDS",
    "ServerSession",
    "TOOL_SPECS",
    "TSSERVER_COMMAND_PREFIX",
    "TSSERVER_PROJECT_INFO",
    "TSSERVER_REQUEST",
    "TSSERVER_RESPONSE",
    "action_of_tool_name",
    "all_servers",
    "apply_whole_line_edits",
    "client_capabilities",
    "fetch_diagnostics",
    "find_project_config",
    "language_id_for_suffix",
    "load_lsp_servers",
    "location_line",
    "lsp_installed",
    "parse_tsserver_request",
    "position",
    "register_features",
    "relative",
    "render_completion",
    "render_diagnostics",
    "render_document_symbols",
    "render_formatting",
    "render_hover",
    "render_locations",
    "render_signature_help",
    "render_symbols",
    "restricted_lsp_roots",
    "run_action",
    "set_lsp_enabled",
    "severity_name",
    "sync_document",
    "symbol_kind_name",
    "tool_name_for",
    "tools_for_server",
    "truncate",
    "tsserver_project_info",
    "tsserver_response_params",
    "uri_to_path",
    "workspace_symbols",
]
