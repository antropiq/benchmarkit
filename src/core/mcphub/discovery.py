"""Discovery and validation of the built-in and declared connectors.

Scans ``mcp_connectors/`` for ``*_mcp.py`` built-ins (docstring read via
``ast``, never imported) and reads ``mcp_connectors/externals.json`` into
the dicts the rest of the package consumes. The two file paths are looked
up through ``mcphub.constants`` *inside* each function call, so redirecting
the hub at another connector folder is one attribute patch, not an import
dance. The heavy ``mcp`` package is only checked for, never imported here,
so discovery keeps working without the optional ``[mcp]`` extra.
"""

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import constants
from .ports import allocate_port
from .validation import argv_from_command, str_map


def mcp_installed() -> bool:
    """True when the optional ``mcp`` dependency (the ``[mcp]`` extra) is present."""
    return importlib.util.find_spec("mcp") is not None


def _description_from_docstring(docstring: str) -> str:
    """First docstring paragraph, whitespace-collapsed, for the settings panel."""
    paragraph = docstring.strip().split("\n\n", 1)[0]
    return " ".join(paragraph.split())


def discover_connectors(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan ``mcp_connectors/`` for ``*_mcp.py`` connector modules.

    Reads each module's docstring (via ``ast``, never imports it) for the
    panel description. Returns ``{id, label, description, port, endpoint}``
    dicts sorted by id. ``port``/``endpoint`` are hub-assigned, not
    connector-declared: a stable preferred port is derived from the
    connector id and probed for availability (:func:`ports.allocate_port`),
    so adding a connector needs no registry edit. ``port`` is ``0`` only
    when the whole auto range is exhausted.
    """
    root = directory or constants.MCP_CONNECTORS_DIR
    connectors: List[Dict[str, Any]] = []
    if not root.is_dir():
        return connectors
    claimed: Set[int] = set()
    for path in sorted(root.glob("*_mcp.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        docstring = ast.get_docstring(tree) or ""
        connector_id = path.stem
        port = allocate_port(connector_id, claimed) or 0
        endpoint = f"http://127.0.0.1:{port}/mcp" if port else ""
        connectors.append(
            {
                "id": connector_id,
                "label": connector_id.removesuffix("_mcp").replace("_", " ").title(),
                "description": _description_from_docstring(docstring),
                "port": port,
                "endpoint": endpoint,
            }
        )
    return connectors


def load_externals(
    path: Optional[Path] = None, reserved_ids: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read ``mcp_connectors/externals.json`` into external connector dicts.

    A missing file yields ``([], [])``. Unreadable/invalid JSON yields one
    file-level error and no connectors. A single malformed entry is skipped
    with a per-entry note; ``reserved_ids`` (built-in ids) rejects colliding
    entries. Never raises. Entries keep their ``enable`` flag; the caller
    decides what to do with disabled ones.
    """
    root = path or constants.EXTERNALS_PATH
    reserved = set(reserved_ids or ())
    connectors: List[Dict[str, Any]] = []
    errors: List[str] = []
    if not root.is_file():
        return connectors, errors
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"{root.name} is not readable JSON: {exc}"]
    if not isinstance(data, dict):
        return [], [f"{root.name} must contain an object of connectors keyed by id"]
    for connector_id in sorted(data):
        prefix = f"{root.name}: '{connector_id}'"
        if connector_id in reserved:
            errors.append(f"{prefix}: id collides with a built-in connector — skipped")
            continue
        entry = data[connector_id]
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: entry must be a JSON object — skipped")
            continue
        kind = entry.get("type")
        kind = kind.lower() if isinstance(kind, str) else ""
        if kind not in ("local", "remote"):
            errors.append(f"{prefix}: type must be \"local\" or \"remote\" — skipped")
            continue
        headers = str_map(entry.get("headers"))
        env = str_map(entry.get("env"))
        if headers is None or env is None:
            errors.append(
                f"{prefix}: headers/env must be objects of string values — skipped"
            )
            continue
        endpoint = ""
        command: Optional[List[str]] = None
        if kind == "remote":
            url = entry.get("url")
            if not isinstance(url, str) or not url.lower().startswith(
                ("http://", "https://")
            ):
                errors.append(
                    f"{prefix}: remote connector needs an http(s) url — skipped"
                )
                continue
            endpoint = url
        else:
            command = argv_from_command(entry.get("command"), entry.get("args"))
            if command is None:
                errors.append(
                    f"{prefix}: local connector needs a command — skipped"
                )
                continue
        description = entry.get("description")
        connectors.append(
            {
                "id": connector_id,
                "label": connector_id,
                "description": description if isinstance(description, str) else "",
                "port": 0,
                "endpoint": endpoint,
                "command": command,
                "external": True,
                "kind": kind,
                "enable": bool(entry.get("enable", False)),
                "headers": headers or {},
                "env": env or {},
            }
        )
    return connectors, errors


def all_connectors() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Built-in ``*_mcp.py`` connectors plus the declared external ones.

    Returns the merged list (built-ins first, discovery order preserved) and
    the external file's validation errors.
    """
    builtins = [
        {**connector, "external": False, "kind": "builtin"}
        for connector in discover_connectors()
    ]
    externals, errors = load_externals(
        reserved_ids={connector["id"] for connector in builtins}
    )
    return builtins + externals, errors


def set_externals_enabled(
    enabled: Dict[str, bool], path: Optional[Path] = None
) -> List[str]:
    """Write ``enable`` flags back to ``externals.json``, keeping everything else.

    Only entries already present in the file are touched; other keys, entry
    order, and unknown ids are preserved. A missing file is a no-op. A file
    that cannot be parsed is left untouched (never rebuilt from partial
    knowledge) and reported back as an error list.
    """
    root = path or constants.EXTERNALS_PATH
    if not root.is_file():
        return []
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Could not update {root.name}: {exc}"]
    if not isinstance(data, dict):
        return [f"Could not update {root.name}: top level is not an object"]
    for connector_id, value in enabled.items():
        entry = data.get(connector_id)
        if isinstance(entry, dict):
            entry["enable"] = bool(value)
    try:
        root.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return [f"Could not write {root.name}: {exc}"]
    return []
