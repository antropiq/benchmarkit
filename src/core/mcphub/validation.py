"""Shape validation of declarative connector entries.

The two validators below interpret the JSON shapes shared by
``mcp_connectors/externals.json`` and ``lsp_connectors/lsp_servers.json``
(a ``headers``/``env`` mapping of scalars, a kilo-style command), so the
two hubs import them from here rather than duplicating the rules.
"""

import shlex
from typing import Any, Dict, List, Optional


def str_map(value: Any) -> Optional[Dict[str, str]]:
    """Validate a ``headers``/``env`` entry: string keys, scalar values.

    Returns the coerced mapping, ``{}`` for an unset value, or ``None`` when
    the shape is invalid.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        return None
    coerced: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (str, int, float, bool)):
            return None
        coerced[key] = str(item)
    return coerced


def argv_from_command(command: Any, args: Any) -> Optional[List[str]]:
    """Normalize a local connector command into an argv list.

    Accepts a kilo-style argv array or a plain command string (shell-quoting
    rules) with an optional ``args`` list appended. ``None`` when invalid.
    """
    argv: List[str]
    if isinstance(command, str):
        try:
            argv = shlex.split(command)
        except ValueError:
            return None
    elif isinstance(command, list) and command and all(
        isinstance(part, str) and part for part in command
    ):
        argv = list(command)
    else:
        return None
    if args is not None:
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            return None
        argv.extend(args)
    return argv or None
