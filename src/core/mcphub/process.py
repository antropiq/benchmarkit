"""Subprocess plumbing shared by the hub's children (and the LSP hub's).

``connector_environment`` builds the environment every connector /
language-server subprocess is launched with; it is deliberately free of
MCP-specific knowledge so ``lsphub`` can reuse it verbatim.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional


def connector_environment(project_root: Path, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for a connector subprocess.

    Adds the project root to ``PYTHONPATH`` so ``mcp_connectors`` imports,
    and pins a valid ``TMPDIR``: benchmarkit's sampling parameter ``TEMP``
    lands in the process environment from ``.env``, and Node's
    ``os.tmpdir()`` (the playwright driver) falls back to ``TMPDIR``/``TMP``/
    ``TEMP`` — a sampling value like ``TEMP=1`` would otherwise be taken as
    a relative temp directory and break browser launches.
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
