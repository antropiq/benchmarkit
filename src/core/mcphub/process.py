"""Subprocess plumbing shared by the hub's children (and the LSP hub's).

``connector_environment`` builds the environment every connector /
language-server subprocess is launched with; it is deliberately free of
MCP-specific knowledge so ``lsphub`` can reuse it verbatim.

The reach-capability function below answers one question for **both** hubs
(win-07 / A-D4): do the guardrails that would confine a local child process's
path reach speak this filesystem at all? It lives beside the environment helper
rather than once per hub so the answer cannot drift between the MCP hub and the
LSP hub; each hub's ``constants`` module re-exports it, which is where a test
forces the other answer (the ``mcphub.constants`` / ``lsphub.constants``
patchability convention).
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

try:  # imported as part of the package (tests, ``python -m mcp_connectors...``)
    from mcp_connectors import path_policy
except ImportError:  # pragma: no cover - depends on how the app is started
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from mcp_connectors import path_policy  # type: ignore[no-redef]

from src.config import path_location


def guardrails_speak_here(prefixes: Optional[Sequence[str]] = None) -> bool:
    """True when the guardrail prefixes could speak the working location's form.

    A prefix matches as *text*, so a list of ``/etc``-shaped entries covers
    nothing once the working location is a ``C:`` tree — every rule stops
    matching and says nothing about it (roadmap §3). The answer comes from the
    shape of the entries against the shape of the path, not from the platform's
    name, so a deployment that supplies matching prefixes is believed; A added
    the question, target B1 wrote the connector lists (``path_policy`` gains
    its Windows forms from the environment at read time) and B-D1 wrote the LSP
    list (``restricted-paths.txt``) the same way, so a deployment that supplies
    matching prefixes is believed on either hub.

    ``prefixes`` is the caller's own list (the LSP hub's workspace-root
    deny-list); ``None`` asks the connector deny/allow prefixes. An **empty**
    list answers True: carrying no prefixes is a declaration, not a list that
    has stopped matching, and a hub whose list is absent has never been bounded
    by one either — refusing on that reading would change Linux.
    """
    entries: Sequence[str] = (
        tuple(path_policy.ALLOW_PREFIXES)
        + tuple(path_policy.DENY_PREFIXES)
        + tuple(path_policy.DENY_PREFIXES_READ)
        if prefixes is None
        else tuple(prefixes)
    )
    if not entries:
        return True
    # ``resolve()``, never ``expanduser()`` (A-D2.3): the working location is a
    # real folder by the time any hub reads it.
    resolved = Path(path_location()).resolve().as_posix()
    return path_policy.lists_are_meaningful(resolved, prefixes=entries)


def connector_environment(project_root: Path, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for a connector subprocess.

    Adds the project root to ``PYTHONPATH`` so ``mcp_connectors`` imports,
    and pins a valid ``TMPDIR``: Node's ``os.tmpdir()`` (the playwright
    driver) falls back to ``TMPDIR``/``TMP``/``TEMP``, so a relative value
    in any of them — e.g. the Windows-style ``TEMP`` carried by an ambient
    environment — would otherwise be taken as a relative temp directory
    and break browser launches.
    """
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    temp_dir = tempfile.gettempdir()
    for key in ("TMPDIR", "TMP", "TEMP"):
        value = env.get(key, "")
        if value and not Path(value).is_absolute():
            env.pop(key, None)
    env["TMPDIR"] = temp_dir
    return env
