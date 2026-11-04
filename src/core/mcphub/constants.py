"""Static configuration of the MCP hub: paths, port range, retry policy.

The connector directory and the gitignored declaration file live here as
*module attributes* on purpose (same idea as ``lsphub.constants``): the code
that reads them resolves them through this module at call time, so pointing
the whole hub at another ``mcp_connectors/`` folder (the tests do exactly
this) is one attribute patch instead of an import dance.
"""

from pathlib import Path

# The one reach-capability seam, re-exported through *this* module so both hubs
# answer the question with the same function and a test forces it the same way
# it redirects any other hub constant (win-07 / A-D4).
from .process import guardrails_speak_here  # noqa: F401 — bound as a hub attribute

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MCP_CONNECTORS_DIR = PROJECT_ROOT / "mcp_connectors"

# The declarative connector declaration file (kilo.json-style, keyed by id):
# the one and only source of MCP connectors. It may hold API keys in
# ``headers``/``env`` and is therefore gitignored; its contents are never
# returned by the API.
EXTERNALS_PATH = MCP_CONNECTORS_DIR / "externals.json"

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
