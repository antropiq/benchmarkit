"""Static configuration of the MCP hub: paths, port range, retry policy.

The connector directory and the gitignored declaration file live here as
*module attributes* on purpose (same idea as ``lsphub.constants``): the code
that reads them resolves them through this module at call time, so pointing
the whole hub at another ``mcp_connectors/`` folder (the tests do exactly
this) is one attribute patch instead of an import dance.
"""

from pathlib import Path

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
