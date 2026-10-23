"""Static configuration of the LSP hub: paths, timeouts, wire constants.

The gitignored declaration file and the workspace deny-list live here as
*module attributes* on purpose: the code that reads them resolves them
through this module, so pointing the whole hub at another
``lsp_servers.json`` (the tests do exactly this) is one patch of
``lsphub.constants.LSP_SERVERS_PATH`` instead of a hunt for every importer.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LSP_CONNECTORS_DIR = PROJECT_ROOT / "lsp_connectors"
LSP_LOGS_DIR = PROJECT_ROOT / "logs" / "lsp"

# Declarative language servers, keyed by id (mirrors externals.json). The
# file may carry secrets in ``env`` and is therefore gitignored; its ``env``
# values are never returned by the API.
LSP_SERVERS_PATH = LSP_CONNECTORS_DIR / "lsp_servers.json"

# Deny-list of workspace roots a server may not be pointed at (same idea as
# mcp_connectors/restricted-fqdn.txt). One absolute prefix per line.
LSP_RESTRICTED_PATHS = LSP_CONNECTORS_DIR / "restricted-paths.txt"

# Per-request and handshake timeouts. Language servers are chatty and slow
# on cold start (tsserver indexes the whole project), so ``initialize`` gets
# far more room than an ordinary request.
REQUEST_TIMEOUT = 20.0
INIT_TIMEOUT = 60.0
# Grace given to a language server between shutdown/exit and SIGTERM/SIGKILL.
STOP_GRACE_SECONDS = 5.0
# How long ``lsp_*_diagnostics`` waits for a publishDiagnostics notification
# after opening/changing a document (servers push; there is no pull request
# unless the server advertises the 3.17 diagnosticProvider).
DIAGNOSTICS_WAIT_SECONDS = 5.0
# Documents bigger than this are refused: a language server is not a way to
# stream a giant file into the context window.
MAX_DOCUMENT_BYTES = 1_000_000
# Default per-server cap on reported diagnostics.
DEFAULT_MAX_DIAGNOSTICS = 40
# Default idle time after which a server is torn down and respawned on the
# next tool call (0 disables the reaper).
DEFAULT_IDLE_SHUTDOWN_SECONDS = 300
# How often the idle reaper wakes up on the hub thread.
REAPER_INTERVAL_SECONDS = 30.0
# Result text cap, mirroring extract_tool_result_text's truncation idea.
MAX_RESULT_CHARS = 12_000

# Volar (``vue-language-server``) delegates every TypeScript semantic to a
# tsserver it expects its *client* to own, and asks for it over the private
# ``tsserver/request`` notification. Its ``getLanguageService()`` awaits the
# answer, so an unanswered relay hangs the session forever: every later
# request for that file queues behind the dangling promise and times out.
# Answering at least ``projectInfo`` is therefore what makes ``vue`` usable.
TSSERVER_REQUEST = "tsserver/request"
TSSERVER_RESPONSE = "tsserver/response"
# Volar prefixes every relayed command with ``_vue:``.
TSSERVER_COMMAND_PREFIX = "_vue:"
TSSERVER_PROJECT_INFO = "_vue:projectInfo"
# Config files a projectInfo answer may point at. With none found the hub
# answers ``null`` and Volar falls back to its inferred ("simple") project,
# which still resolves Vue semantics for a standalone workspace.
PROJECT_CONFIG_NAMES = ("tsconfig.json", "jsconfig.json")

LSP_MISSING_HINT = (
    'The "pygls" package is not installed — run: pip install -e ".[lsp]"'
)
