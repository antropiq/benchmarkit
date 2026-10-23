"""Discovery and validation of the declared language servers.

Reads ``lsp_connectors/lsp_servers.json`` into the dicts the rest of the
package consumes. The two file paths are looked up through
``lsphub.constants`` *inside* each function call, so redirecting the hub at
another declaration file is one attribute patch, not an import dance.
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.mcphub.validation import argv_from_command, str_map

from . import constants


def lsp_installed() -> bool:
    """True when the optional LSP deps (the ``[lsp]`` extra) are importable."""
    try:
        return (
            importlib.util.find_spec("pygls") is not None
            and importlib.util.find_spec("lsprotocol") is not None
        )
    except (ImportError, ValueError):
        return False


def restricted_lsp_roots(path: Optional[Path] = None) -> List[str]:
    """Workspace-root prefixes refused by the hub, from restricted-paths.txt."""
    root = path or constants.LSP_RESTRICTED_PATHS
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
    root = path or constants.LSP_SERVERS_PATH
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
        command = argv_from_command(entry.get("command"), entry.get("args"))
        if command is None:
            errors.append(f"{prefix}: an LSP server needs a command — skipped")
            continue
        env = str_map(entry.get("env"))
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
        max_diagnostics = entry.get("maxDiagnostics", constants.DEFAULT_MAX_DIAGNOSTICS)
        if not isinstance(max_diagnostics, int) or max_diagnostics < 1:
            errors.append(
                f"{prefix}: maxDiagnostics must be a positive integer — skipped"
            )
            continue
        idle_shutdown = entry.get(
            "idleShutdownSeconds", constants.DEFAULT_IDLE_SHUTDOWN_SECONDS
        )
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


def all_servers(
    path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Every declared language server plus the file's validation errors."""
    return load_lsp_servers(path)


def set_lsp_enabled(
    enabled: Dict[str, bool], path: Optional[Path] = None
) -> List[str]:
    """Write ``enable`` flags back to ``lsp_servers.json``, keeping everything else.

    Only entries already present in the file are touched; other keys, entry
    order, and unknown ids are preserved. A missing file is a no-op; a file
    that cannot be parsed is left untouched and reported as an error list.
    """
    root = path or constants.LSP_SERVERS_PATH
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
