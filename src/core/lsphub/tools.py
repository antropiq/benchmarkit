"""Tool catalogue: the LSP requests the hub can expose as LLM tools.

Kept apart from the manager so the catalogue (and the wording the model
reads) can be reviewed on its own, and so ``tools_for_server`` can be unit
tested without a live server.
"""

from typing import Any, Dict, List, Optional

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
