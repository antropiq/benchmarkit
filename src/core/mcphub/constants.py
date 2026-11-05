"""Static configuration of the MCP hub: paths, port range, retry policy.

The connector directory and the *workspace's* declaration file are
*module attributes* on purpose (same idea as ``lsphub.constants``): the
code that reads them resolves them through this module at call time, so
pointing the hub at another folder (the tests do exactly this) is one
attribute patch instead of an import dance.

The declaration itself moved into the workspace: it describes *this*
folder's connectors, so ``mcp_settings_path()`` is built per call out of
the active working location — a patched filename or a switched location
moves the lookup (the ``agent_modes.agent_modes_path()`` precedent). The
install-level file beside the connector scripts
(``mcp_connectors/mcp-settings.default.json``) is no longer read by the
hub: it is the committed *template* the seed copies when a folder becomes
the working location.
"""

import sys
from pathlib import Path
from typing import Optional

# The one reach-capability seam, re-exported through *this* module so both hubs
# answer the question with the same function and a test forces it the same way
# it redirects any other hub constant (win-07 / A-D4).
from .process import guardrails_speak_here  # noqa: F401 — bound as a hub attribute
from src.config import path_location

# The reserved folder's name is not re-declared here: it comes from
# ``mcp_connectors/policy_shared``, with the same ``sys.path`` fallback
# ``core.prompt_helpers`` uses, because that module lives outside ``src`` so
# the stdio connectors can read it too. Two constants for one folder name is
# how the folder stops being one folder.
try:
    from mcp_connectors import policy_shared as shared
except ImportError:  # pragma: no cover - depends on how the app is started
    _root = str(Path(__file__).resolve().parents[3])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from mcp_connectors import policy_shared as shared  # type: ignore[no-redef]

POLICY_DIRNAME: str = shared.POLICY_DIRNAME

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MCP_CONNECTORS_DIR = PROJECT_ROOT / "mcp_connectors"

# The workspace's connector declaration file (kilo.json-style, keyed by id),
# inside the reserved folder at the working location: the one and only source
# of MCP connectors. It may hold API keys in ``headers``/``env`` and is
# therefore gitignored (via ``.nojae/``); its contents are never returned by
# the API. The ``enable`` toggles write back into it — the folder's "edited
# only outside the app" rule bends for it the way it already does for the
# prompt-helper store.
MCP_SETTINGS_FILENAME = "mcp-settings.json"

# The committed install-level template the seed copies, byte for byte, when a
# folder becomes the working location. Its ``${NAME}`` references
# (``BENCHMARKIT_PYTHON_ROOT``, ``PROJECT_ROOT``, ``BENCHMARKIT_PATH_LOCATION``)
# are *not* substituted at seed time: every one of them resolves on every read
# (``validation.expand_env``), which is what keeps one seeded file correct on
# every machine and across every workspace switch.
MCP_SETTINGS_TEMPLATE = MCP_CONNECTORS_DIR / "mcp-settings.default.json"

# **A pin, not the default.** When not ``None`` it wins over the computed path
# below — the ``editorfs.constants.TRASH_DIR`` precedent: one patchable
# override keeps the tests' one-attribute redirect without re-introducing an
# import-time constant. Left at ``None``, the workspace's own file answers.
MCP_SETTINGS_PATH: Optional[Path] = None


def mcp_settings_path() -> Path:
    """The active workspace's ``.nojae/mcp-settings.json`` (may not exist).

    Built per call out of module attributes, so the pin above, a patched
    filename, or a switched working location each move the lookup the same
    way. A read of an absent file is ``([], [])`` — an unseeded workspace
    simply has no connectors until its declaration is written.
    """
    if MCP_SETTINGS_PATH is not None:
        return MCP_SETTINGS_PATH
    return Path(path_location()) / POLICY_DIRNAME / MCP_SETTINGS_FILENAME

# Connection retry policy for a freshly launched connector session.
CONNECT_RETRIES = 20
# Local externals launch through a package runner (``npx``), whose cold start
# (download + Node boot) can far exceed the built-in connectors' window.
EXTERNAL_CONNECT_RETRIES = 60
CONNECT_INTERVAL = 0.5
# Grace given to a connector process between SIGTERM and SIGKILL.
STOP_GRACE_SECONDS = 5.0

CONNECTORS_MISSING_HINT = (
    'The "mcp" package is not installed — run: pip install -e ".[mcp]"'
)

# win-07 / A-D4, as lifted by target B1 (G-D8): a `local` (stdio) connector's
# reach is bounded by the deny-prefixes of ``mcp_connectors/path_policy``,
# which now speak both platforms' path forms. Where they speak *nothing* — a
# working location whose form no prefix can name — the bound is a claim the app
# must not make, and the entry is refused instead: same channel as an
# unresolved ``${NAME}``, with a sentence naming the gap and pointing at the
# capability that closes it. Asserted identical to ``lsphub.constants``' copy:
# the two hubs refuse in one wording, because since lspfinal-04 they refuse on
# one and the same fact — an inert guardrail list — and on nothing else.
#
# lspfinal-06 dropped the ``(see target B)`` pointer: target B is finished for
# the hub that used to need it, so it named a closed plan rather than the thing
# to fix. What is left is the sentence's own diagnosis — a guardrail list with
# no entry in this filesystem's path form — which is what an operator acts on
# (the list is a file they edit, not a roadmap).
LOCAL_REACH_REFUSAL = (
    "its reach cannot be bounded on this filesystem: no guardrail entry speaks "
    "this working location's path form"
)
