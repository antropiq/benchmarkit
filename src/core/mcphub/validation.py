"""Shape validation of declarative connector entries.

The validators below interpret the JSON shapes shared by
``mcp_connectors/externals.json`` and ``lsp_connectors/lsp_servers.json``
(a ``headers``/``env`` mapping of scalars, a kilo-style command), so the
two hubs import them from here rather than duplicating the rules.

Environment references are expanded here as well (``expand_env``), by the
two *discovery* modules right after parsing — that way a per-machine path
or an API key can live in the environment instead of in these gitignored
declarations, and the same file works on every checkout.
"""

import os
import re
import shlex
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ``${NAME}``, plus the shell's ``${NAME:-fallback}`` escape hatch. Only a
# braced reference is special, so a bare ``$`` stays literal and npm version
# pins (``pkg@$latest``) or regexes in a ``command`` are never touched.
ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> Tuple[str, List[str]]:
    """Substitute ``${NAME}`` / ``${NAME:-fallback}`` references in a string.

    Reads :data:`os.environ`, which already carries the ``.env`` loaded at
    startup (``config.load_dotenv_once``), so ``${HOME}`` and a project key
    like ``${CONTEXT7_API_KEY}`` are equally available. Returns the expanded
    text plus the names that resolved to nothing: an unresolved reference is
    left written as-is rather than blanked out, so a typo stays visible in
    the entry's validation note instead of quietly becoming an empty path or
    an empty API key.
    """
    missing: List[str] = []

    def replace(match: "re.Match[str]") -> str:
        name, fallback = match.group(1), match.group(2)
        found = os.environ.get(name)
        if found is not None:
            return found
        if fallback is not None:
            return fallback
        missing.append(name)
        return match.group(0)

    return ENV_REF_RE.sub(replace, value), missing


def expand_env_map(value: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """Expand every value of a ``headers``/``env`` mapping (keys stay literal)."""
    expanded: Dict[str, str] = {}
    missing: List[str] = []
    for key, item in value.items():
        expanded[key], names = expand_env(item)
        missing.extend(names)
    return expanded, missing


def expand_env_list(values: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Expand an argv list one element at a time.

    Per-element on purpose: unlike a shell, an expansion here never re-splits
    into several arguments, so an expanded value containing spaces stays one
    argument.
    """
    expanded: List[str] = []
    missing: List[str] = []
    for item in values:
        text, names = expand_env(item)
        expanded.append(text)
        missing.extend(names)
    return expanded, missing


def expand_env_json(value: Any) -> Tuple[Any, List[str]]:
    """Expand string references anywhere inside a decoded JSON structure.

    Used for the free-form ``initializationOptions``/``settings`` objects,
    which carry server-specific paths (a ``--tsdk``, a ``tsserver.path``) just
    like ``command`` does. Dict keys are never expanded, and non-strings
    (numbers, booleans, ``null``) are returned untouched.
    """
    missing: List[str] = []
    if isinstance(value, str):
        return expand_env(value)
    if isinstance(value, list):
        items: List[Any] = []
        for item in value:
            expanded, names = expand_env_json(item)
            items.append(expanded)
            missing.extend(names)
        return items, missing
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            expanded, names = expand_env_json(item)
            result[key] = expanded
            missing.extend(names)
        return result, missing
    return value, missing


def unresolved_env_note(prefix: str, missing: Sequence[str]) -> Optional[str]:
    """One validation note for the environment names an entry referenced."""
    if not missing:
        return None
    names = ", ".join(f"${{{name}}}" for name in dict.fromkeys(missing))
    return f"{prefix}: {names} is not set in the environment — left as written"


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
