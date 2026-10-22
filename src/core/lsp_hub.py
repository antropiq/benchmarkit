"""Language-server discovery, LSP sessions, and tool synthesis.

Core-only: no FastAPI or browser concerns. Reads the declarative language
server definitions from ``lsp_connectors/lsp_servers.json`` (one entry per
server id, ``enable`` as the on/off source of truth), spawns one subprocess
per activated server, performs the LSP handshake (``initialize`` ->
``initialized``), keeps the session alive for the app's lifetime, and turns
the server's advertised capabilities into OpenAI-format tool definitions so
the Chat tab's tool loop can call them.

Like ``mcp_hub.py`` (which this module deliberately mirrors), all LSP I/O
runs on a private asyncio thread so synchronous callers (chat worker
threads) can bridge onto it with :meth:`LspManager.call_tool_sync`. The
language servers themselves are only ever spoken to over stdio JSON-RPC by
this hub: the LLM calls ``lsp_<id>_<action>`` with plain JSON arguments,
never a command line.

The heavy ``pygls``/``lsprotocol`` packages are imported lazily so
discovery and the settings panel keep working when the optional ``[lsp]``
extra is not installed.
"""

import asyncio
import json
import logging
import os
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.mcp_hub import _argv_from_command, _str_map, connector_environment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

logger = logging.getLogger("benchmarkit.lsp")


def lsp_installed() -> bool:
    """True when the optional LSP deps (the ``[lsp]`` extra) are importable."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec("pygls") is not None
            and importlib.util.find_spec("lsprotocol") is not None
        )
    except (ImportError, ValueError):
        return False


def restricted_lsp_roots(path: Optional[Path] = None) -> List[str]:
    """Workspace-root prefixes refused by the hub, from restricted-paths.txt."""
    root = path or LSP_RESTRICTED_PATHS
    try:
        lines = root.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def load_lsp_servers(
    path: Optional[Path] = None, reserved_ids: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read ``lsp_servers.json`` into language-server dicts.

    A missing file yields ``([], [])``. Unreadable/invalid JSON yields one
    file-level error and no servers. A single malformed entry is skipped with
    a per-entry note; unknown keys are ignored. Entries keep their ``enable``
    flag; the caller decides what to do with disabled ones. Never raises.
    """
    root = path or LSP_SERVERS_PATH
    servers: List[Dict[str, Any]] = []
    errors: List[str] = []
    reserved = {str(entry) for entry in reserved_ids or ()}
    if not root.is_file():
        return servers, errors
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"{root.name} is not readable JSON: {exc}"]
    if not isinstance(data, dict):
        return [], [f"{root.name} must contain an object of servers keyed by id"]
    denied = [os.path.realpath(entry) for entry in restricted_lsp_roots()]
    for server_id in sorted(data):
        prefix = f"{root.name}: '{server_id}'"
        if server_id in reserved:
            errors.append(f"{prefix}: id is reserved — skipped")
            continue
        if server_id.startswith("_"):
            # Convention: an underscore key is a comment, not a server.
            continue
        entry = data[server_id]
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: entry must be a JSON object — skipped")
            continue
        command = _argv_from_command(entry.get("command"), entry.get("args"))
        if command is None:
            errors.append(f"{prefix}: an LSP server needs a command — skipped")
            continue
        env = _str_map(entry.get("env"))
        if env is None:
            errors.append(f"{prefix}: env must be an object of string values — skipped")
            continue
        extensions = entry.get("extensions")
        if extensions is None:
            extensions = []
        if not isinstance(extensions, list) or not all(
            isinstance(item, str) and item.startswith(".") for item in extensions
        ):
            errors.append(
                f"{prefix}: extensions must be a list of '.ext' strings — skipped"
            )
            continue
        workspace_root = entry.get("workspaceRoot")
        if not isinstance(workspace_root, str) or not os.path.isabs(workspace_root):
            errors.append(f"{prefix}: workspaceRoot must be an absolute path — skipped")
            continue
        real_root = os.path.realpath(workspace_root)
        if not os.path.isdir(real_root):
            errors.append(f"{prefix}: workspaceRoot does not exist — skipped")
            continue
        if any(
            real_root == denied_entry or real_root.startswith(denied_entry + os.sep)
            for denied_entry in denied
        ):
            errors.append(f"{prefix}: workspaceRoot is on the deny-list — skipped")
            continue
        initialization_options = entry.get("initializationOptions")
        if initialization_options is not None and not isinstance(
            initialization_options, dict
        ):
            errors.append(
                f"{prefix}: initializationOptions must be an object — skipped"
            )
            continue
        # Answers handed back to the server's own workspace/configuration
        # requests, keyed by LSP section ('html.customData',
        # 'vue.suggest.propNameCasing', …). Anything not listed is answered
        # null, i.e. "not set", which is what the LSP spec prescribes and
        # what makes a server's own ?? fallback land on its default.
        settings = entry.get("settings")
        if settings is not None and not isinstance(settings, dict):
            errors.append(f"{prefix}: settings must be an object — skipped")
            continue
        tools = entry.get("tools")
        if tools is not None and (
            not isinstance(tools, list)
            or not all(isinstance(item, str) and item for item in tools)
        ):
            errors.append(f"{prefix}: tools must be a list of names — skipped")
            continue
        max_diagnostics = entry.get("maxDiagnostics", DEFAULT_MAX_DIAGNOSTICS)
        if not isinstance(max_diagnostics, int) or max_diagnostics < 1:
            errors.append(
                f"{prefix}: maxDiagnostics must be a positive integer — skipped"
            )
            continue
        idle_shutdown = entry.get("idleShutdownSeconds", DEFAULT_IDLE_SHUTDOWN_SECONDS)
        if not isinstance(idle_shutdown, int) or idle_shutdown < 0:
            errors.append(
                f"{prefix}: idleShutdownSeconds must be an integer (0 disables) — skipped"
            )
            continue
        label = entry.get("label")
        language_id = entry.get("languageId")
        servers.append(
            {
                "id": server_id,
                "label": label if isinstance(label, str) and label else server_id,
                "description": (
                    entry["description"]
                    if isinstance(entry.get("description"), str)
                    else ""
                ),
                "command": command,
                "workspaceRoot": workspace_root,
                "extensions": sorted({item.lower() for item in extensions}),
                "languageId": language_id if isinstance(language_id, str) else "",
                "enable": bool(entry.get("enable", False)),
                "lazy": bool(entry.get("lazy", False)),
                "initializationOptions": initialization_options or {},
                "tools": list(tools) if tools is not None else None,
                "env": env or {},
                "settings": settings or {},
                "maxDiagnostics": max_diagnostics,
                "idleShutdownSeconds": idle_shutdown,
            }
        )
    return servers, errors


def all_servers(path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Every declared language server plus the file's validation errors."""
    return load_lsp_servers(path)


def set_lsp_enabled(enabled: Dict[str, bool], path: Optional[Path] = None) -> List[str]:
    """Write ``enable`` flags back to ``lsp_servers.json``, keeping everything else.

    Only entries already present in the file are touched; other keys, entry
    order, and unknown ids are preserved. A missing file is a no-op; a file
    that cannot be parsed is left untouched and reported as an error list.
    """
    root = path or LSP_SERVERS_PATH
    if not root.is_file():
        return []
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Could not update {root.name}: {exc}"]
    if not isinstance(data, dict):
        return [f"Could not update {root.name}: top level is not an object"]
    for server_id, value in enabled.items():
        entry = data.get(server_id)
        if isinstance(entry, dict):
            entry["enable"] = bool(value)
    try:
        root.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return [f"Could not write {root.name}: {exc}"]
    return []


# ---------------------------------------------------------------------------
# Tool catalogue: the LSP requests the hub can expose as LLM tools
# ---------------------------------------------------------------------------

#: ``path``/``line``/``character`` keep the LSP convention (zero-based,
#: character = UTF-16 code unit) and the descriptions say so, because models
#: otherwise invent one-based offsets and blame the server. Text *output*
#: carries 1-based line/column for readability — the asymmetry is stated too.
_PATH_PROPERTY = {
    "type": "string",
    "description": "File path, relative to the server's workspace root.",
}
_POSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "path": _PATH_PROPERTY,
        "line": {
            "type": "integer",
            "minimum": 0,
            "description": "Zero-based line, as reported by lsp_*_diagnostics.",
        },
        "character": {
            "type": "integer",
            "minimum": 0,
            "description": "Zero-based character (UTF-16 code unit) on that line.",
        },
    },
    "required": ["path", "line", "character"],
}
_PATH_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"path": _PATH_PROPERTY},
    "required": ["path"],
}

#: action -> LSP capability attribute, request method, argument schema, and
#: the description the model reads. A capability the server does not
#: advertise never becomes a tool.
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "diagnostics": {
        "capability": None,  # push notification (or 3.17 pull), always offered
        "method": "textDocument/publishDiagnostics",
        "schema": _PATH_ONLY_SCHEMA,
        "description": (
            "Analyse one file and return its diagnostics as "
            "'path:line:col severity: message (code)' lines. Positions in the "
            "output are 1-based for readability; feed positions back to the "
            "other tools as 0-based line/character."
        ),
    },
    "hover": {
        "capability": "hover_provider",
        "method": "textDocument/hover",
        "schema": _POSITION_SCHEMA,
        "description": (
            "Hover information (inferred type / docstring) at a position. "
            "line/character are 0-based."
        ),
    },
    "definition": {
        "capability": "definition_provider",
        "method": "textDocument/definition",
        "schema": _POSITION_SCHEMA,
        "description": "Where the symbol at a position is defined (0-based position).",
    },
    "references": {
        "capability": "references_provider",
        "method": "textDocument/references",
        "schema": {
            "type": "object",
            "properties": {
                "path": _PATH_PROPERTY,
                "line": {"type": "integer", "minimum": 0},
                "character": {"type": "integer", "minimum": 0},
                "includeDeclaration": {"type": "boolean", "default": True},
            },
            "required": ["path", "line", "character"],
        },
        "description": "All references to the symbol at a position (0-based position).",
    },
    "document_symbols": {
        "capability": "document_symbol_provider",
        "method": "textDocument/documentSymbol",
        "schema": _PATH_ONLY_SCHEMA,
        "description": "Outline of a file: symbol names with kind and 1-based line.",
    },
    "workspace_symbols": {
        "capability": "workspace_symbol_provider",
        "method": "workspace/symbol",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Symbol name to search for."}
            },
            "required": ["query"],
        },
        "description": "Search the whole workspace for symbols matching a query.",
    },
    "completion": {
        "capability": "completion_provider",
        "method": "textDocument/completion",
        "schema": {
            "type": "object",
            "properties": {
                "path": _PATH_PROPERTY,
                "line": {"type": "integer", "minimum": 0},
                "character": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
            "required": ["path", "line", "character"],
        },
        "description": "Completions at a position (0-based), capped by limit.",
    },
    "signature_help": {
        "capability": "signature_help_provider",
        "method": "textDocument/signatureHelp",
        "schema": _POSITION_SCHEMA,
        "description": "Call signature(s) at a position (0-based).",
    },
    "formatting": {
        "capability": "document_formatting_provider",
        "method": "textDocument/formatting",
        "schema": _PATH_ONLY_SCHEMA,
        "description": (
            "Proposed whole-file formatting returned as a unified diff. Never "
            "writes: apply it with the file-edit tools so the change goes "
            "through the backup chain."
        ),
    },
}


def _capability_present(capabilities: Any, capability: Optional[str]) -> bool:
    """Whether a server advertises one capability (absent/False/None = no)."""
    if capability is None:
        return True
    value = getattr(capabilities, capability, None)
    return value is not None and value is not False


def tool_name_for(server_id: str, action: str) -> str:
    """Namespaced tool name — unique across hubs, attributable to a server."""
    return f"lsp_{server_id}_{action}"


def action_of_tool_name(server_id: str, tool: str) -> str:
    """The action behind ``lsp_<serverId>_<action>`` ("" when not a match)."""
    prefix = f"lsp_{server_id}_"
    return tool[len(prefix) :] if tool.startswith(prefix) else ""


def tools_for_server(
    server: Dict[str, Any], capabilities: Any = None
) -> List[Dict[str, Any]]:
    """OpenAI function-calling defs for one server, intersected with its caps.

    ``capabilities`` is the server's ``initialize`` ``ServerCapabilities``,
    or ``None`` before the handshake (then only the declared ``tools`` list
    is consulted). Names are ``lsp_<serverId>_<action>``.
    """
    declared = server.get("tools")
    names = (
        list(TOOL_SPECS)
        if declared is None
        else [action for action in declared if action in TOOL_SPECS]
    )
    definitions: List[Dict[str, Any]] = []
    for action in names:
        spec = TOOL_SPECS[action]
        if capabilities is not None and not _capability_present(
            capabilities, spec["capability"]
        ):
            continue
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name_for(server["id"], action),
                    "description": f"{spec['description']} (LSP server: {server['id']})",
                    "parameters": spec["schema"],
                },
            }
        )
    return definitions


# ---------------------------------------------------------------------------
# Result rendering: LSP structures -> compact, line-numbered text
# ---------------------------------------------------------------------------


def parse_tsserver_request(params: Any) -> Optional[Tuple[int, str, Any]]:
    """Split one ``tsserver/request`` payload into ``(id, command, args)``.

    Volar sends ``sendNotification('tsserver/request', [id, command, args])``.
    vscode-jsonrpc treats a single *array* argument as positional parameters,
    so it reaches the wire doubly wrapped — ``params: [[id, command, args]]``
    — and its own handler destructures that outer array (``([id, res]) =>``).
    Accepting the wrapped and the flat form (plus an object form) keeps the
    hub working across language-server versions instead of betting on one.
    Returns ``None`` when the payload is unrecognisable, so the caller can
    log it rather than crash the session.
    """
    if isinstance(params, list) and len(params) == 1 and isinstance(params[0], list):
        params = params[0]  # unwrap the positional-argument wrapper
    if isinstance(params, dict):
        request_id = params.get("id")
        command = params.get("command")
        args = params.get("args")
    elif isinstance(params, list) and len(params) >= 2:
        request_id = params[0]
        command = params[1]
        args = params[2] if len(params) > 2 else None
    else:
        return None
    if not isinstance(request_id, int) or not isinstance(command, str):
        return None
    return request_id, command, args


def tsserver_response_params(request_id: int, result: Any) -> List[Any]:
    """The payload Volar's ``tsserver/response`` handler expects.

    Its handler is ``connection.onNotification('tsserver/response',
    ([id, res]) => …)``: destructuring ``params`` yields ``id`` and the
    answer, so the pair has to be one array *inside* the params array —
    the same single-array-argument wrapping the request side uses. Sending
    it flat makes the handler destructure an int and silently never resolve.
    """
    return [[request_id, result]]


def find_project_config(root: Path) -> Optional[str]:
    """Absolute path of the tsconfig/jsconfig a projectInfo answer should use.

    Volar builds its language service from the ``configFileName`` in this
    answer, so pointing it at the workspace's own config is what makes the
    resolved include/exclude match the real project. ``tsconfig`` wins over
    ``jsconfig``, matching how TypeScript itself prefers it.
    """
    for name in PROJECT_CONFIG_NAMES:
        candidate = root / name
        if candidate.is_file():
            return str(candidate)
    return None


def tsserver_project_info(file_name: str, config_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """A ``projectInfo`` answer for one file, or ``None`` for no project.

    ``projectKind: "external"`` says "this file belongs to that config file",
    which is the only kind tsserver accepts alongside a ``configFileName``.
    Answering ``None`` is a real answer, not a shrug: Volar then creates its
    inferred project, giving Vue semantics without TypeScript project context.
    """
    if config_file is None:
        return None
    return {
        "configFileName": config_file,
        "configFileExists": True,
        "isExternalConfig": True,
        "projectKind": "external",
        "projectFileName": config_file,
        "languageServiceEnabled": True,
    }


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _uri_to_path(uri: str) -> str:
    from pygls import uris as lsp_uris

    return lsp_uris.to_fs_path(uri) or uri


def _relative(root: Path, uri: str) -> str:
    """Workspace-relative path for a file: URI (falls back to the raw path)."""
    path = Path(_uri_to_path(uri))
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _severity_name(severity: Any) -> str:
    return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(
        int(severity) if severity is not None else 1, "error"
    )


def _render_diagnostics(
    root: Path, diagnostics: List[Any], limit: int, default_uri: str = ""
) -> str:
    """``path:line:col severity: message (code)`` lines.

    ``publishDiagnostics`` carries the URI on the *notification*, not on each
    diagnostic, so the owning document's URI is passed in as ``default_uri``.
    """
    if not diagnostics:
        return "No diagnostics."
    lines: List[str] = []
    for diagnostic in diagnostics[:limit]:
        start = diagnostic.range.start
        code = getattr(diagnostic, "code", None)
        source = getattr(diagnostic, "source", None)
        suffix = ""
        if code is not None:
            suffix = f" ({source}{code})" if source else f" ({code})"
        message = " ".join(str(diagnostic.message).split())
        lines.append(
            f"{_relative(root, getattr(diagnostic, 'uri', '') or default_uri)}"
            f":{start.line + 1}:{start.character + 1} "
            f"{_severity_name(getattr(diagnostic, 'severity', None))}: "
            f"{message}{suffix}"
        )
    hidden = len(diagnostics) - len(lines)
    if hidden > 0:
        lines.append(f"[{hidden} more diagnostics suppressed]")
    return "\n".join(lines)


def _location_line(root: Path, location: Any) -> str:
    # A Location carries uri/range; a LocationLink (definition/declaration
    # responses) carry target_uri/target_range instead.
    uri = getattr(location, "uri", None) or getattr(location, "target_uri", None)
    range_ = getattr(location, "range", None) or getattr(location, "target_range", None)
    if uri is None or range_ is None:
        return f"<unparsed location: {type(location).__name__}>"
    start = range_.start
    return f"{_relative(root, uri)}:{start.line + 1}:{start.character + 1}"


def _render_locations(root: Path, locations: Any, limit: int = 40) -> str:
    if not locations:
        return "No results."
    # Some servers (pyright) come back through cattrs as a tuple rather than
    # a list; treat any list/tuple as a collection of locations.
    if not isinstance(locations, (list, tuple)):
        locations = [locations]
    lines = [_location_line(root, location) for location in locations[:limit]]
    if len(locations) > limit:
        lines.append(f"[{len(locations) - limit} more results suppressed]")
    return "\n".join(lines)


def _symbol_kind_name(kind: Any) -> str:
    from lsprotocol import types as lsp

    try:
        return lsp.SymbolKind(int(kind)).name.lower()
    except (TypeError, ValueError):
        return "symbol"


def _render_document_symbols(symbols: Any, limit: int = 200) -> str:
    if not symbols:
        return "No symbols."
    lines: List[str] = []

    def walk(items: Any, depth: int) -> None:
        for symbol in items:
            if len(lines) >= limit:
                return
            name = getattr(symbol, "name", "?")
            kind = _symbol_kind_name(getattr(symbol, "kind", None))
            location = getattr(symbol, "location", None)
            range_ = getattr(symbol, "range", None)
            if location is not None:
                line = location.range.start.line + 1
            elif range_ is not None:
                line = range_.start.line + 1
            else:
                line = 0
            lines.append(f"{'  ' * depth}{kind} {name} (line {line})")
            children = getattr(symbol, "children", None)
            if children:
                walk(children, depth + 1)

    walk(symbols, 0)
    if len(lines) >= limit:
        lines.append("[more symbols suppressed]")
    return "\n".join(lines)


def _render_hover(hover: Any) -> str:
    if hover is None or getattr(hover, "contents", None) is None:
        return "No hover information."
    from lsprotocol import types as lsp

    contents = hover.contents
    parts: List[str] = []
    if isinstance(contents, str):
        parts.append(contents)
    elif isinstance(contents, lsp.MarkupContent):
        parts.append(contents.value)
    elif isinstance(contents, lsp.MarkedStringWithLanguage):
        value = contents.value
        language = contents.language
        parts.append(f"```{language}\n{value}\n```" if language else value)
    else:
        for item in contents or []:
            if isinstance(item, str):
                parts.append(item)
            else:
                value = getattr(item, "value", "")
                language = getattr(item, "language", "")
                parts.append(f"```{language}\n{value}\n```" if language else value)
    text = "\n".join(part for part in parts if part)
    return text or "No hover information."


def _render_completion(result: Any, limit: int) -> str:
    from lsprotocol import types as lsp

    items = result.items if isinstance(result, lsp.CompletionList) else result
    if not items:
        return "No completions."
    lines: List[str] = []
    for item in items[:limit]:
        label = getattr(item, "label", "?")
        detail = getattr(item, "detail", "") or ""
        lines.append(f"{label} — {detail}" if detail else str(label))
    if len(items) > limit:
        lines.append(f"[{len(items) - limit} more completions suppressed]")
    return "\n".join(lines)


def _render_signature_help(help_: Any) -> str:
    if help_ is None or not getattr(help_, "signatures", None):
        return "No signature help."
    lines: List[str] = []
    for signature in help_.signatures:
        label = str(getattr(signature, "label", ""))
        documentation = getattr(signature, "documentation", None)
        if documentation is not None:
            value = documentation.value if hasattr(documentation, "value") else documentation
            text = " ".join(str(value).split())
            if text:
                label += f"\n  {text}"
        lines.append(label)
    return "\n".join(lines) or "No signature help."


def _render_symbols(root: Path, symbols: Any, limit: int = 40) -> str:
    if not symbols:
        return "No symbols."
    lines: List[str] = []
    for symbol in symbols[:limit]:
        name = getattr(symbol, "name", "?")
        kind = _symbol_kind_name(getattr(symbol, "kind", None))
        location = getattr(symbol, "location", None)
        where = _location_line(root, location) if location is not None else "?"
        lines.append(f"{where} {kind} {name}")
    if len(symbols) > limit:
        lines.append(f"[{len(symbols) - limit} more symbols suppressed]")
    return "\n".join(lines)


def _apply_whole_line_edits(text: str, edits: List[Any]) -> str:
    """Apply TextEdits to a copy of ``text`` (for the formatting diff preview).

    Applied bottom-up so the offsets of the not-yet-applied edits stay valid.
    Never touches the file on disk.
    """
    lines = text.splitlines(keepends=True)
    ordered = sorted(
        edits,
        key=lambda edit: (edit.range.start.line, edit.range.start.character),
        reverse=True,
    )
    for edit in ordered:
        start, end = edit.range.start, edit.range.end
        replacement = edit.new_text if isinstance(edit.new_text, str) else str(edit.new_text)
        if start.line >= len(lines):
            lines.append("\n")
        end_line = min(end.line, len(lines) - 1)
        prefix = "".join(lines[: start.line]) + (
            lines[start.line][: start.character] if start.line < len(lines) else ""
        )
        suffix = (lines[end_line][end.character :] if end_line < len(lines) else "") + "".join(
            lines[end_line + 1 :]
        )
        text = prefix + replacement + suffix
        lines = text.splitlines(keepends=True)
    return text


def _render_formatting(relative_path: str, text: str, edits: Any) -> str:
    """Formatting edits rendered as a unified diff (never applied to disk)."""
    import difflib

    if not edits:
        return "Already formatted (no edits proposed)."
    usable = [
        edit
        for edit in edits
        if getattr(edit, "range", None) is not None and edit.new_text is not None
    ]
    if not usable:
        return "Already formatted (no edits proposed)."
    proposed = _apply_whole_line_edits(text, usable)
    diff = difflib.unified_diff(
        text.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile=relative_path,
        tofile=relative_path,
        n=3,
    )
    rendered = "".join(diff)
    if not rendered:
        return "Already formatted (no visible change)."
    return rendered if rendered.endswith("\n") else rendered + "\n"


class DiagnosticsInbox:
    """Latest ``publishDiagnostics`` payload per URI, plus a wait-for-push step.

    LSP diagnostics arrive as a server->client *notification*, so a tool call
    cannot simply await a response: it opens/changes the document and then
    waits for the next publish for that URI (or a deadline).
    """

    def __init__(self) -> None:
        self._latest: Dict[str, Tuple[Optional[int], List[Any]]] = {}
        self._events: Dict[str, asyncio.Event] = {}

    def publish(self, uri: str, version: Optional[int], diagnostics: List[Any]) -> None:
        self._latest[uri] = (version, list(diagnostics))
        self._events.setdefault(uri, asyncio.Event()).set()

    def peek(self, uri: str) -> Optional[Tuple[Optional[int], List[Any]]]:
        return self._latest.get(uri)

    def clear(self, uri: Optional[str] = None) -> None:
        if uri is None:
            self._latest.clear()
            self._events.clear()
            return
        self._latest.pop(uri, None)
        self._events.pop(uri, None)

    def forget_session(self) -> None:
        """Drop everything (server went away) so a new session starts clean."""
        self.clear()

    async def wait(self, uri: str, version: Optional[int], timeout: float) -> Optional[List[Any]]:
        """Wait for the publish of ``version`` (any publish when the server
        does not report versions), returning its diagnostics — or ``None``
        when nothing fresh arrived before the deadline."""
        entry = self._latest.get(uri)
        if entry is not None and (
            version is None or entry[0] is None or entry[0] >= version
        ):
            return entry[1]
        # A fresh event: only a publish that arrives *now* counts, never a
        # stale one from an earlier document version.
        event = asyncio.Event()
        self._events[uri] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        entry = self._latest.get(uri)
        return entry[1] if entry else None


class DocumentStore:
    """The documents currently ``didOpen``ed on one server, with versions.

    Every tool call naming a file first opens it (or ``didChange``s it with a
    full-text change when already open), so the server analyses the *current*
    content rather than a possibly stale on-disk snapshot. The version
    counter increments monotonically.
    """

    def __init__(self) -> None:
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._last_used: Dict[str, float] = {}

    @staticmethod
    def _now() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return 0.0

    def version(self, uri: str) -> Optional[int]:
        entry = self._docs.get(uri)
        return entry["version"] if entry else None

    def touch(self, uri: str) -> None:
        self._last_used[uri] = self._now()

    def next_version(self, uri: str) -> int:
        entry = self._docs.get(uri)
        return (entry["version"] + 1) if entry else 1

    def add(self, uri: str, version: int, text: str, language_id: str) -> None:
        self._docs[uri] = {"version": version, "text": text, "languageId": language_id}
        self.touch(uri)

    def update(self, uri: str, version: int, text: str) -> None:
        entry = self._docs.get(uri)
        if entry is None:
            self.add(uri, version, text, "")
            return
        entry["version"] = version
        entry["text"] = text
        self.touch(uri)

    def text(self, uri: str) -> Optional[str]:
        entry = self._docs.get(uri)
        return entry["text"] if entry else None

    def remove(self, uri: str) -> Optional[Dict[str, Any]]:
        self._last_used.pop(uri, None)
        return self._docs.pop(uri, None)

    def uris(self) -> List[str]:
        return list(self._docs)

    def idle_uris(self, idle_seconds: float) -> List[str]:
        """Documents untouched for at least ``idle_seconds`` (loop clock)."""
        now = self._now()
        return [uri for uri, used in self._last_used.items() if now - used >= idle_seconds]


class _ServerSession:
    """One live language server: process, pygls client, document store, inbox."""

    def __init__(self, info: Dict[str, Any], client: Any, process: Any, log_path: Path) -> None:
        self.info = info
        self.client = client
        self.process = process
        self.log_path = log_path
        self.capabilities: Any = None
        self.documents = DocumentStore()
        self.inbox = DiagnosticsInbox()
        self.pull_diagnostics = False
        self.stderr_task: Optional[asyncio.Task] = None
        self.exit_task: Optional[asyncio.Task] = None
        #: Volar's tsserver relay answers, counted per command, so the status
        #: panel can show that a Vue session is actually being served rather
        #: than merely started (the failure mode here is silent, not loud).
        self.ts_relay_counts: Dict[str, int] = {}


class LspManager:
    """Spawn language servers and bridge sync callers onto their sessions.

    Lifecycle: :meth:`start_all` on app startup (servers declared
    ``"lazy": true`` spawn on their first tool call instead), :meth:`stop_all`
    on shutdown. All LSP I/O runs on a private asyncio thread named
    ``lsp-hub`` so chat worker threads stay synchronous.
    """

    def __init__(
        self,
        servers: List[Dict[str, Any]],
        python_exe: Optional[str] = None,
        project_root: Path = PROJECT_ROOT,
        logs_dir: Optional[Path] = None,
    ) -> None:
        self._servers: Dict[str, Dict[str, Any]] = {
            server["id"]: {
                **server,
                "status": "stopped",
                "error": None,
                "last_used": 0.0,
            }
            for server in servers
        }
        self._python_exe = python_exe
        self._project_root = project_root
        self._logs_dir = logs_dir or (project_root / "logs" / "lsp")
        self._sessions: Dict[str, _ServerSession] = {}
        self._defs: Dict[str, List[Dict[str, Any]]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False
        self._installed = lsp_installed()
        self._reaper: Optional[asyncio.TimerHandle] = None

    @property
    def installed(self) -> bool:
        return self._installed

    # -- startup / shutdown --------------------------------------------------

    async def start_all(self) -> None:
        """Spawn every activated server. Never raises.

        Servers declared ``"lazy": true`` stay stopped until their first tool
        call (tsserver's cold start is multi-second and eats RAM); the others
        come up with the app.
        """
        await self._start_all_matching(None)

    async def start_enabled(
        self, server_ids: List[str]
    ) -> None:
        """Spawn only the servers whose id is in ``server_ids``. Never raises.

        The API passes the intersection of ``LSP_ENABLED`` and each entry's
        ``enable`` flag: a prepared ``lsp_servers.json`` can stay entirely
        dormant until the user asks for it.
        """
        await self._start_all_matching(set(server_ids))

    async def _start_all_matching(self, only_ids: Optional[Set[str]]) -> None:
        if self._started or self._stopped:
            return
        self._started = True
        if not self._installed:
            for info in self._servers.values():
                if info["enable"]:
                    info["status"] = "error"
                    info["error"] = LSP_MISSING_HINT
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="lsp-hub", daemon=True)
        self._thread.start()
        futures = [
            asyncio.run_coroutine_threadsafe(self._start_one(server_id), self._loop)
            for server_id, info in self._servers.items()
            if info["enable"]
            and not info["lazy"]
            and (only_ids is None or server_id in only_ids)
        ]
        await asyncio.gather(*(asyncio.wrap_future(future) for future in futures))
        self._schedule_reaper()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _log_tail(self, log_path: Path, chars: int = 600) -> str:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        return text[-chars:]

    def _append_log(self, log_path: Path, message: str) -> None:
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(str(message).rstrip("\n") + "\n")
        except OSError:
            pass

    async def _pump_stderr(self, client: Any, log_path: Path) -> None:
        """Copy the server's stderr pipe into its per-server log file.

        pygls keeps the child's stderr on a pipe (so nothing deadlocks if the
        server is chatty); draining it is what puts tsserver's own words in
        the log, where ``_log_tail`` can surface them on an initialize error.
        """
        stream = getattr(client._server, "stderr", None)  # noqa: SLF001
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            self._append_log(log_path, line.decode("utf-8", errors="replace"))

    async def _watch_exit(self, server_id: str, process: Any) -> None:
        """Surface a server that dies mid-conversation as a status (crash
        path): the exit code plus its log tail land in ``info['error']``, so
        the settings panel shows what happened instead of a stale 'running'.
        Cancelled by ``_close_session`` for the deliberate shutdown paths."""
        try:
            await process.wait()
        except Exception:  # noqa: BLE001 — nothing to report for a dead pipe
            return
        session = self._sessions.get(server_id)
        if session is None or session.process is not process:
            return  # deliberately closed and (maybe) already respawned
        info = self._servers.get(server_id)
        if info is None or info["status"] != "running":
            return
        info["status"] = "error"
        tail = self._log_tail(session.log_path)
        info["error"] = (
            f"language server exited with code {process.returncode}"
            + (f": {tail}" if tail else "")
        )
        self._sessions.pop(server_id, None)
        self._defs.pop(server_id, None)

    def _environment(self, info: Dict[str, Any]) -> Dict[str, str]:
        """Spawn environment: ``connector_environment`` pins a valid TMPDIR
        (the app's ``TEMP`` sampling variable would otherwise break Node),
        then the entry's own ``env`` overrides (volta shims need a PATH that
        resolves ``node``)."""
        env = connector_environment(self._project_root)
        env.update(info.get("env") or {})
        return env

    def _register_features(self, client: Any, info: Dict[str, Any], log_path: Path) -> None:
        """Answer the server->client messages the hub must not crash on.

        pygls 2.x calls a feature handler with a single ``params`` argument.
        tsserver and Volar are the noisiest here: an unanswerable
        ``workspace/configuration`` or ``client/registerCapability`` is what
        makes or breaks the TS/Vue servers.
        """

        @client.feature("textDocument/publishDiagnostics")
        def on_diagnostics(params: Any) -> None:
            # Resolved through the manager so a restart re-binds a fresh inbox.
            session = self._sessions.get(info["id"])
            if session is None:
                return
            session.inbox.publish(
                params.uri, getattr(params, "version", None), list(params.diagnostics)
            )

        @client.feature("workspace/configuration")
        def on_configuration(params: Any) -> List[Any]:
            """One answer per requested section: ``null`` means 'not set'.

            The LSP spec's ``ConfigurationResponse`` is a per-item value that
            is ``null`` when nothing is configured. Answering ``{}`` instead
            (as this handler used to) is not a no-op: a server doing the
            idiomatic ``getConfiguration('x') ?? []`` keeps the ``{}`` because
            it is not nullish, and then iterates it — which is exactly how
            Volar dies on ``html.customData`` with ``TypeError: paths is not
            iterable``, taking the whole process down on the first Vue
            completion request. ``null`` makes every such fallback land on
            its own default.
            """
            answers: List[Any] = []
            for item in getattr(params, "items", None) or []:
                section = getattr(item, "section", None) or (
                    item.get("section") if isinstance(item, dict) else None
                )
                if section in info.get("settings", {}):
                    answers.append(info["settings"][section])
                else:
                    answers.append(None)
            return answers

        @client.feature("client/registerCapability")
        def on_register_capability(params: Any) -> None:
            return None

        @client.feature("client/unregisterCapability")
        def on_unregister_capability(params: Any) -> None:
            return None

        @client.feature("window/workDoneProgress/create")
        def on_progress_create(params: Any) -> None:
            return None

        def on_log_message(params: Any) -> None:
            self._append_log(log_path, getattr(params, "message", params))

        client.feature("window/logMessage")(on_log_message)
        client.feature("$/logTrace")(on_log_message)
        # Vendor extensions we deliberately ignore (pygls logs an
        # "unknown method" warning for each otherwise).
        for ignored in (
            "$/progress",
            "$/setTrace",
            "$/log",
            "$/typescriptVersion",
            "language/status",
            "language/action",
            "textDocument/publishWarning",
            "textDocument/publishLink",
            "$/publishDiagnostics",
        ):
            with suppress(Exception):
                client.feature(ignored)(lambda params: None)

        # Volar's tsserver relay. It owns no tsserver: it asks *us* for the
        # answers a VS Code extension host would get from its tsserver child
        # process, and awaits them inside ``getLanguageService()``. Refusing
        # to answer is worse than answering wrongly — the awaited promise
        # never settles, so every request touching that file times out.
        #
        # The hub does not run a tsserver (that would double the RAM bill and
        # duplicate the ``typescript`` server). It answers the one command
        # that gates everything else, ``_vue:projectInfo``, with the
        # workspace's own tsconfig/jsconfig; Volar then builds its language
        # service itself and serves documentSymbol/hover/completion/diagnostics
        # from Vue's own semantic model. Commands it still asks for are
        # component-detail helpers (props/slots/directives/meta) — answered
        # ``null`` so their awaited promise settles and Vue degrades those
        # features instead of hanging the whole session.
        @client.feature(TSSERVER_REQUEST)
        def on_tsserver_request(params: Any) -> None:
            session = self._sessions.get(info["id"])
            if session is None:
                return
            parsed = parse_tsserver_request(params)
            if parsed is None:
                self._append_log(
                    log_path, f"[hub] unrecognised {TSSERVER_REQUEST} params: {params!r}"
                )
                return
            request_id, command, args = parsed
            session.ts_relay_counts[command] = session.ts_relay_counts.get(command, 0) + 1
            if command == TSSERVER_PROJECT_INFO:
                file_name = ""
                if isinstance(args, dict) and isinstance(args.get("file"), str):
                    file_name = args["file"]
                config_file = self._project_config_for(session, file_name)
                result: Any = tsserver_project_info(file_name, config_file)
                if config_file is None:
                    # A null projectInfo is Volar's "no project found" signal:
                    # it falls back to an inferred project that still gives
                    # Vue semantics for a config-less workspace.
                    self._append_log(
                        log_path,
                        f"[hub] {TSSERVER_PROJECT_INFO}: no "
                        + "/".join(PROJECT_CONFIG_NAMES)
                        + f" under {info['workspaceRoot']} — answering null "
                        "(Volar will use its inferred project)",
                    )
            else:
                result = None
            if command != TSSERVER_PROJECT_INFO and session.ts_relay_counts[command] == 1:
                # Log each degraded command once. Without this, a caller who
                # gets "No hover information." out of lsp_vue_hover has no
                # way to learn that the answer came from the hub rather than
                # from a real tsserver — the relay's limitation is otherwise
                # invisible.
                self._append_log(
                    log_path,
                    f"[hub] {command}: answered null (the hub runs no tsserver, so "
                    f"{command.removeprefix('_vue:')} is unavailable; the "
                    "'typescript' server covers plain .js/.ts files)",
                )
            # pygls 2.x exposes no client-side send_notification: the raw
            # JSON-RPC layer is ``client.protocol.notify``. The method is not
            # in lsprotocol's METHOD_TO_TYPES, so it goes out verbatim as a
            # generic notification rather than being restructured.
            try:
                client.protocol.notify(
                    TSSERVER_RESPONSE, tsserver_response_params(request_id, result)
                )
            except Exception as exc:  # noqa: BLE001 — a dead pipe must not kill the handler
                self._append_log(log_path, f"[hub] {TSSERVER_RESPONSE} failed: {exc}")

    async def _start_one(self, server_id: str) -> None:
        """Bring one server up, handshake included. Never raises."""
        info = self._servers.get(server_id)
        if info is None or info["status"] in ("running", "starting"):
            return
        try:
            await self._start_session(server_id, info)
        except Exception as exc:  # noqa: BLE001 — startup errors land in status
            info["status"] = "error"
            info["error"] = str(exc) or exc.__class__.__name__
            await self._close_session(server_id)

    async def _start_session(self, server_id: str, info: Dict[str, Any]) -> None:
        from lsprotocol import types as lsp
        from pygls import uris as lsp_uris
        from pygls.lsp.client import LanguageClient

        info["status"] = "starting"
        info["error"] = None
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._logs_dir / f"{server_id}.log"
        argv = list(info["command"])

        client = LanguageClient("benchmarkit", "1.0")
        self._register_features(client, info, log_path)
        client.report_server_error = lambda error, source: self._append_log(
            log_path, f"[hub] {getattr(source, '__name__', source)}: {error}"
        )
        # pygls spawns the child itself and owns its stdio pipes — the hub
        # must not create its own subprocess, or the protocol has no
        # transport to write to. stderr is captured and pumped to the log.
        try:
            await client.start_io(
                argv[0],
                *argv[1:],
                cwd=info["workspaceRoot"],
                env=self._environment(info),
            )
        except OSError as exc:
            info["status"] = "error"
            info["error"] = f"Could not spawn language server: {exc}"
            return
        process = client._server  # noqa: SLF001 — pygls keeps it private
        if process is None:
            info["status"] = "error"
            info["error"] = "Language server started but exposes no process handle"
            with suppress(Exception):
                await client.stop()
            return
        session = _ServerSession(info, client, process, log_path)
        session.stderr_task = asyncio.ensure_future(
            self._pump_stderr(client, log_path)
        )
        self._sessions[server_id] = session
        session.exit_task = asyncio.ensure_future(
            self._watch_exit(server_id, process)
        )

        root_uri = lsp_uris.from_fs_path(info["workspaceRoot"]) or "file://" + info["workspaceRoot"]
        try:
            result = await asyncio.wait_for(
                client.initialize_async(
                    lsp.InitializeParams(
                        client_info=lsp.ClientInfo(name="benchmarkit", version="1.0"),
                        process_id=process.pid,
                        root_uri=root_uri,
                        root_path=info["workspaceRoot"],
                        capabilities=self._client_capabilities(),
                        initialization_options=info.get("initializationOptions") or None,
                    )
                ),
                timeout=INIT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            info["status"] = "error"
            info["error"] = f"{server_id} did not answer initialize within {INIT_TIMEOUT:.0f}s"
            await self._close_session(server_id)
            return
        except Exception as exc:  # noqa: BLE001 — the message IS the diagnosis
            # An initialize failure must surface the server's own words:
            # "Could not find a valid TypeScript installation" is a config
            # problem the user can only fix if it is reported verbatim.
            info["status"] = "error"
            info["error"] = str(exc) or exc.__class__.__name__
            tail = self._log_tail(log_path)
            if tail:
                info["error"] += f": {tail}"
            await self._close_session(server_id)
            return

        session.capabilities = result.capabilities
        session.pull_diagnostics = result.capabilities.diagnostic_provider is not None
        client.initialized(lsp.InitializedParams())
        self._defs[server_id] = tools_for_server(info, result.capabilities)
        info["last_used"] = asyncio.get_event_loop().time()
        info["status"] = "running"
        info["error"] = None
        # Every session gets a log line of its own. Without it a quiet server
        # (Volar writes nothing until it is asked something) leaves no
        # logs/lsp/<id>.log at all, so ``GET /api/lsp/log`` answers "" exactly
        # when the user needs it most, and there is no record of which config
        # the tsserver relay will answer with.
        config_file = find_project_config(Path(info["workspaceRoot"]).resolve())
        self._append_log(
            log_path,
            f"[hub] {server_id} running (pid {process.pid}, root {info['workspaceRoot']}, "
            f"{len(self._defs[server_id])} tools, "
            + (f"project config {config_file}" if config_file else "no tsconfig/jsconfig")
            + ")",
        )

    def _client_capabilities(self) -> Any:
        """What the hub declares it supports. Deliberately explicit: Pyright
        only publishes diagnostics when the client advertises
        ``publishDiagnostics``, and only asks ``workspace/configuration`` when
        the client declares ``configuration``."""
        from lsprotocol import types as lsp

        return lsp.ClientCapabilities(
            workspace=lsp.WorkspaceClientCapabilities(
                apply_edit=False,
                configuration=True,
                workspace_folders=True,
                did_change_configuration=lsp.DidChangeConfigurationClientCapabilities(
                    dynamic_registration=True
                ),
            ),
            text_document=lsp.TextDocumentClientCapabilities(
                publish_diagnostics=lsp.PublishDiagnosticsClientCapabilities(
                    related_information=True,
                    version_support=True,
                ),
                hover=lsp.HoverClientCapabilities(
                    content_format=[lsp.MarkupKind.Markdown, lsp.MarkupKind.PlainText]
                ),
                # The hub renders both shapes (Location and LocationLink),
                # so it can accept links. Declaring False made Volar refuse
                # to answer definition requests for Vue components and log
                # "textDocument.definition.linkSupport is not supported by
                # the client but could be used by the server".
                definition=lsp.DefinitionClientCapabilities(
                    link_support=True,
                ),
                references=lsp.ReferenceClientCapabilities(),
                document_symbol=lsp.DocumentSymbolClientCapabilities(
                    hierarchical_document_symbol_support=True
                ),
                completion=lsp.CompletionClientCapabilities(),
                signature_help=lsp.SignatureHelpClientCapabilities(),
                formatting=lsp.DocumentFormattingClientCapabilities(),
                diagnostic=lsp.DiagnosticClientCapabilities(),
            ),
            window=lsp.WindowClientCapabilities(work_done_progress=True),
        )

    # -- hot toggles ---------------------------------------------------------

    def start_server(self, server_id: str) -> None:
        """Start (or restart) one server from a synchronous caller. Never raises."""
        info = self._servers.get(server_id)
        if info is None or self._loop is None or self._stopped:
            return
        info["status"] = "starting"
        info["error"] = None
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._restart_one(server_id), self._loop)

    def stop_server(self, server_id: str) -> None:
        """Tear one server down, waiting briefly so callers can re-read status."""
        if server_id not in self._servers or self._loop is None or self._stopped:
            return
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._stop_one(server_id), self._loop).result(
                timeout=2 * STOP_GRACE_SECONDS + 5
            )

    async def _restart_one(self, server_id: str) -> None:
        await self._close_session(server_id)
        await self._start_one(server_id)

    async def _stop_one(self, server_id: str) -> None:
        await self._close_session(server_id)
        info = self._servers[server_id]
        info["status"] = "stopped"
        info["error"] = None

    def server_status(self, server_id: str) -> Optional[str]:
        """Live status of one known server, or ``None`` when unknown."""
        info = self._servers.get(server_id)
        return info["status"] if info is not None else None

    def log_tail(self, server_id: str, chars: int = 4000) -> str:
        """Tail of one server's log file ("" when there is none)."""
        if server_id not in self._servers:
            return ""
        return self._log_tail(self._logs_dir / f"{server_id}.log", chars)

    # -- tool catalogue ------------------------------------------------------

    def active_tool_defs(
        self, server_ids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """OpenAI-format tools of the given running servers + tool -> server map.

        A tool name is attributed to the first server offering it in
        ``server_ids`` order (names are namespaced, so they never collide).

        A lazy server that has not spawned yet is still represented, using
        the tools its config declares (its real capabilities are only known
        after the handshake its first call triggers).
        """
        tools: List[Dict[str, Any]] = []
        tool_to_server: Dict[str, str] = {}
        for server_id in server_ids:
            info = self._servers.get(server_id)
            if info is None:
                continue
            definitions = self._defs.get(server_id)
            if definitions is None:
                if not (info["enable"] and info["lazy"] and info["status"] == "stopped"):
                    continue
                definitions = tools_for_server(info, None)
            for definition in definitions:
                name = definition["function"]["name"]
                if name in tool_to_server:
                    continue
                tool_to_server[name] = server_id
                tools.append(definition)
        return tools, tool_to_server

    # -- the sync bridge -----------------------------------------------------

    def call_tool_sync(
        self,
        server_id: str,
        tool: str,
        arguments: Dict[str, Any],
        timeout: float = REQUEST_TIMEOUT + DIAGNOSTICS_WAIT_SECONDS + 5.0,
    ) -> str:
        """Execute one LSP-backed tool call from a synchronous thread."""
        if self._loop is None:
            raise RuntimeError("LspManager is not started")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(server_id, tool, arguments), self._loop
        )
        return future.result(timeout=timeout)

    async def _call_tool(self, server_id: str, tool: str, arguments: Dict[str, Any]) -> str:
        info = self._servers.get(server_id)
        if info is None:
            return f"Error: language server '{server_id}' is not declared"
        if info["status"] != "running":
            if not info["enable"]:
                return f"Error: language server '{server_id}' is disabled in Settings"
            if info["status"] == "stopped":
                # Lazy server: bring it up on its first tool call.
                await self._start_one(server_id)
            if info["status"] != "running":
                return (
                    f"Error: language server '{server_id}' is {info['status']}: "
                    f"{info['error'] or 'no details'}"
                )
        session = self._sessions.get(server_id)
        if session is None:
            return f"Error: language server '{server_id}' has no live session"
        action = action_of_tool_name(server_id, tool)
        if action not in TOOL_SPECS:
            return f"Error: unknown LSP tool '{tool}'"
        info["last_used"] = asyncio.get_event_loop().time()
        try:
            return await self._run_action(session, action, arguments or {})
        except asyncio.TimeoutError:
            return f"Error: {server_id} did not answer {action} within {REQUEST_TIMEOUT:.0f}s"
        except Exception as exc:  # noqa: BLE001 — feed the failure to the model
            return f"Error: {server_id} {action} failed: {exc}"

    # -- path policy ---------------------------------------------------------

    def _resolve_path(
        self, session: _ServerSession, raw_path: Any
    ) -> Tuple[Optional[Path], Optional[str]]:
        """Policy-checked absolute path for a tool argument.

        Confines reads to the server's declared workspace root, with the
        expected extension, size-capped. This is what stops
        "read ~/.ssh/id_rsa through the Python server" style tricks and makes
        the LSP hub useless as a generic file reader.
        """
        info = session.info
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None, "path is required"
        root = Path(info["workspaceRoot"]).resolve()
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            return None, f"path is not usable ({raw_path}): {exc}"
        if resolved != root and root not in resolved.parents:
            return None, f"path must live inside the workspace root {root}"
        if not resolved.is_file():
            return None, self._missing_path_hint(root, raw_path, resolved)
        suffix = resolved.suffix.lower()
        extensions = info.get("extensions") or []
        if extensions and suffix not in extensions:
            return None, (
                f"{suffix} is not handled by {info['id']} (handles: {', '.join(extensions)})"
            )
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            return None, f"could not stat {resolved}: {exc}"
        if size > MAX_DOCUMENT_BYTES:
            return None, f"file exceeds the {MAX_DOCUMENT_BYTES} byte cap"
        return resolved, None

    @staticmethod
    def _missing_path_hint(root: Path, raw_path: str, resolved: Path) -> str:
        """``not a file: …`` plus, when the cause is obvious, the fix.

        ``path`` is documented as workspace-relative, but a model that has
        just read the root out of the server's own config tends to send it
        absolute (``web/src/App.vue`` against root ``…/web``), which resolves
        to a doubled ``…/web/web/src/App.vue``. Saying only ``not a file``
        makes the model retry the same path; naming the path that does exist
        costs one line and saves a round trip.
        """
        message = f"not a file: {resolved}"
        if not isinstance(raw_path, str):
            return message
        # Candidates: the literal argument re-rooted at the workspace root,
        # and the same with the root's own name stripped from the front.
        candidates: List[Path] = []
        relative = Path(raw_path)
        candidates.append(root / relative.name if relative.is_absolute() else root / relative)
        parts = list(relative.parts)
        if parts and parts[0] == root.name:
            candidates.append(root.joinpath(*parts[1:]))
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return (
                        f"{message} — path is relative to the workspace root "
                        f"{root}, try: {candidate.relative_to(root)}"
                    )
            except OSError:
                continue
        return message

    # -- tsserver relay support ---------------------------------------------

    def _project_config_for(self, session: _ServerSession, file_name: str) -> Optional[str]:
        """The tsconfig/jsconfig governing one relayed file, if any.

        TypeScript resolves a project by walking up from the file until it
        finds a config, so the relay walks the same way — but stops at the
        server's workspace root, which is the boundary the hub promises never
        to read past. A Vue component three levels down still gets the
        workspace's own config, and a workspace with no config at all gets
        ``None`` (Volar's inferred project) instead of some unrelated config
        from a parent directory.
        """
        root = Path(session.info["workspaceRoot"]).resolve()
        start: Optional[Path] = None
        if file_name:
            if file_name.startswith("file:"):
                # Volar sends ``uri.fsPath`` (a plain path), but a file URI
                # reaching here must not be mangled into a relative segment.
                file_name = _uri_to_path(file_name) or file_name
            candidate = Path(file_name).resolve() if os.path.isabs(file_name) else root / file_name
            try:
                start = candidate.parent
            except OSError:
                start = None
        current = start if start is not None else root
        while current == root or root in current.parents:
            found = find_project_config(current)
            if found is not None:
                return found
            if current == root:
                break
            current = current.parent
        return None

    # -- document sync -------------------------------------------------------

    async def _sync_document(self, session: _ServerSession, path: Path) -> str:
        """didOpen/didChange the file with its current content, return its URI."""
        from lsprotocol import types as lsp
        from pygls import uris as lsp_uris

        uri = lsp_uris.from_fs_path(str(path)) or "file://" + str(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"could not read {path}: {exc}") from exc
        store = session.documents
        language_id = session.info.get("languageId") or language_id_for_suffix(path.suffix)
        version = store.next_version(uri)
        if store.version(uri) is None:
            store.add(uri, version, text, language_id)
            session.client.text_document_did_open(
                lsp.DidOpenTextDocumentParams(
                    text_document=lsp.TextDocumentItem(
                        uri=uri,
                        language_id=language_id or "plaintext",
                        version=version,
                        text=text,
                    )
                )
            )
        else:
            store.update(uri, version, text)
            session.client.text_document_did_change(
                lsp.DidChangeTextDocumentParams(
                    text_document=lsp.VersionedTextDocumentIdentifier(
                        uri=uri, version=version
                    ),
                    content_changes=[
                        lsp.TextDocumentContentChangeWholeDocument(text=text)
                    ],
                )
            )
        return uri

    @staticmethod
    def _position(arguments: Dict[str, Any]) -> Any:
        from lsprotocol import types as lsp

        line = arguments.get("line")
        character = arguments.get("character")
        if not isinstance(line, int) or not isinstance(character, int) or line < 0:
            raise RuntimeError(
                "line and character must be non-negative integers (0-based, as "
                "reported by lsp_*_diagnostics)"
            )
        return lsp.Position(line=line, character=max(int(character), 0))

    # -- the actions themselves ----------------------------------------------

    async def _run_action(
        self, session: _ServerSession, action: str, arguments: Dict[str, Any]
    ) -> str:
        from lsprotocol import types as lsp

        info = session.info
        root = Path(info["workspaceRoot"]).resolve()

        if action == "workspace_symbols":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return "Error: query is required"
            result = await asyncio.wait_for(
                session.client.workspace_symbol_async(lsp.WorkspaceSymbolParams(query=query)),
                timeout=REQUEST_TIMEOUT,
            )
            symbols = result if isinstance(result, list) else getattr(result, "items", None)
            return _truncate(_render_symbols(root, symbols or []))

        path, error = self._resolve_path(session, arguments.get("path"))
        if error is not None or path is None:
            return f"Error: {error}"
        uri = await self._sync_document(session, path)
        document = lsp.TextDocumentIdentifier(uri=uri)

        if action == "diagnostics":
            return await self._diagnostics(session, root, uri)
        if action == "hover":
            result = await asyncio.wait_for(
                session.client.text_document_hover_async(
                    lsp.HoverParams(text_document=document, position=self._position(arguments))
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return _truncate(_render_hover(result))
        if action == "definition":
            result = await asyncio.wait_for(
                session.client.text_document_definition_async(
                    lsp.DefinitionParams(text_document=document, position=self._position(arguments))
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return _truncate(_render_locations(root, result))
        if action == "references":
            result = await asyncio.wait_for(
                session.client.text_document_references_async(
                    lsp.ReferenceParams(
                        text_document=document,
                        position=self._position(arguments),
                        context=lsp.ReferenceContext(
                            include_declaration=bool(arguments.get("includeDeclaration", True))
                        ),
                    )
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return _truncate(_render_locations(root, result))
        if action == "completion":
            limit = arguments.get("limit") or 20
            result = await asyncio.wait_for(
                session.client.text_document_completion_async(
                    lsp.CompletionParams(text_document=document, position=self._position(arguments))
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return _truncate(_render_completion(result, int(limit)))
        if action == "document_symbols":
            result = await asyncio.wait_for(
                session.client.text_document_document_symbol_async(
                    lsp.DocumentSymbolParams(text_document=document)
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return _truncate(_render_document_symbols(result))
        if action == "signature_help":
            result = await asyncio.wait_for(
                session.client.text_document_signature_help_async(
                    lsp.SignatureHelpParams(text_document=document, position=self._position(arguments))
                ),
                timeout=REQUEST_TIMEOUT,
            )
            return _truncate(_render_signature_help(result))
        if action == "formatting":
            result = await asyncio.wait_for(
                session.client.text_document_formatting_async(
                    lsp.DocumentFormattingParams(
                        text_document=document,
                        options=lsp.FormattingOptions(tab_size=4, insert_spaces=True),
                    )
                ),
                timeout=REQUEST_TIMEOUT,
            )
            text = session.documents.text(uri) or path.read_text(
                encoding="utf-8", errors="replace"
            )
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                relative = str(path)
            return _truncate(_render_formatting(relative, text, result or []))
        return f"Error: unsupported LSP action '{action}'"

    async def _diagnostics(self, session: _ServerSession, root: Path, uri: str) -> str:
        """Pull diagnostics when the server advertises 3.17 pull support, else
        wait for the publishDiagnostics push on the inbox."""
        from lsprotocol import types as lsp

        limit = session.info.get("maxDiagnostics") or DEFAULT_MAX_DIAGNOSTICS
        if session.pull_diagnostics:
            report = await asyncio.wait_for(
                session.client.text_document_diagnostic_async(
                    lsp.DocumentDiagnosticParams(
                        text_document=lsp.TextDocumentIdentifier(uri=uri)
                    )
                ),
                timeout=REQUEST_TIMEOUT,
            )
            items = list(getattr(report, "items", None) or [])
            related = getattr(report, "related_documents", None) or {}
            for related_report in related.values():
                items.extend(getattr(related_report, "items", None) or [])
            return _truncate(_render_diagnostics(root, items, limit, uri))
        version = session.documents.version(uri)
        diagnostics = await session.inbox.wait(uri, version, DIAGNOSTICS_WAIT_SECONDS)
        if diagnostics is None:
            # Nothing ever arrived for this document: an empty answer would be
            # a lie, the server may simply still be indexing.
            return (
                "No diagnostics published by the server within "
                f"{DIAGNOSTICS_WAIT_SECONDS:.0f}s — the file may be clean, or "
                "the server is still indexing; call this tool again if in doubt."
            )
        return _truncate(_render_diagnostics(root, diagnostics, limit, uri))

    # -- teardown -------------------------------------------------------------

    async def _close_session(self, server_id: str) -> None:
        """shutdown/exit, then SIGTERM and SIGKILL after a grace period."""
        session = self._sessions.pop(server_id, None)
        self._defs.pop(server_id, None)
        if session is None:
            return
        if session.stderr_task is not None:
            with suppress(Exception):
                session.stderr_task.cancel()
        if session.exit_task is not None:
            with suppress(Exception):
                session.exit_task.cancel()
        session.inbox.forget_session()
        client = session.client
        if client is not None and not client.stopped:
            with suppress(Exception):
                await asyncio.wait_for(
                    client.shutdown_async(None), timeout=STOP_GRACE_SECONDS
                )
            with suppress(Exception):
                client.exit(None)
            with suppress(Exception):
                await asyncio.wait_for(client.stop(), timeout=STOP_GRACE_SECONDS)
        process = session.process
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=STOP_GRACE_SECONDS)
            except (asyncio.TimeoutError, TimeoutError):
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(Exception):
                    await process.wait()

    # -- idle reaper ----------------------------------------------------------

    def _schedule_reaper(self) -> None:
        if self._loop is None or self._stopped:
            return
        self._reaper = self._loop.call_later(REAPER_INTERVAL_SECONDS, self._reap)

    def _reap(self) -> None:
        """Tear down servers untouched for ``idleShutdownSeconds`` (0 = never)."""
        if self._loop is None or self._stopped:
            return
        now = self._loop.time()
        for server_id, info in self._servers.items():
            idle = info.get("idleShutdownSeconds") or 0
            if info["status"] != "running" or idle <= 0:
                continue
            if now - info.get("last_used", 0.0) < idle:
                continue
            with suppress(Exception):
                asyncio.ensure_future(self._reap_one(server_id), loop=self._loop)
        self._schedule_reaper()

    async def _reap_one(self, server_id: str) -> None:
        info = self._servers[server_id]
        await self._close_session(server_id)
        info["status"] = "stopped"
        info["error"] = None

    async def _stop_on_loop(self) -> None:
        if self._reaper is not None:
            with suppress(Exception):
                self._reaper.cancel()
            self._reaper = None
        for server_id in list(self._sessions):
            await self._close_session(server_id)
            self._servers[server_id]["status"] = "stopped"

    async def stop_all(self) -> None:
        """Close every session and terminate every server process. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self._loop is None:
            return
        with suppress(Exception):
            await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(self._stop_on_loop(), self._loop)
                ),
                timeout=(len(self._servers) + 1) * (2 * STOP_GRACE_SECONDS + 5),
            )
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=STOP_GRACE_SECONDS)
        self._loop = None
        self._thread = None

    # -- status ----------------------------------------------------------------

    def status(self) -> List[Dict[str, Any]]:
        """Server definitions merged with live status, in declaration order.

        ``env`` values are deliberately excluded — they may hold secrets and
        never leave the server.
        """
        return [
            {
                **{
                    key: info.get(key)
                    for key in (
                        "id",
                        "label",
                        "description",
                        "command",
                        "workspaceRoot",
                        "extensions",
                        "languageId",
                        "enable",
                        "lazy",
                        "tools",
                        "maxDiagnostics",
                        "idleShutdownSeconds",
                    )
                },
                "status": info["status"],
                "error": info["error"],
                "toolCount": len(self._defs.get(info["id"], [])),
            }
            for info in self._servers.values()
        ]


def language_id_for_suffix(suffix: str) -> str:
    """LSP language id for a file suffix ("" when unknown)."""
    return {
        ".py": "python",
        ".pyi": "python",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascriptreact",
        ".json": "json",
        ".vue": "vue",
    }.get(suffix.lower(), "")
