"""Volar's private tsserver relay: the wire shapes and the project answer.

``vue-language-server`` owns no tsserver; it asks its client for TypeScript
answers over ``tsserver/request`` and awaits them inside
``getLanguageService()``. The hub answers (see ``features.py``), and this
module pins the two things that make answering correct: the doubly nested
array shape vscode-jsonrpc puts on the wire, and the projectInfo payload that
tells Volar which tsconfig governs a file.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import PROJECT_CONFIG_NAMES


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
