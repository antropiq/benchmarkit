#!/usr/bin/env bash
# Start the console MCP server rooted at this script's location.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

PYTHON="$HOME/.pyenv/versions/benchmarkit/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

cd "$DIR"
exec "$PYTHON" -m mcp_connectors.console_mcp --root-dir "$DIR" "$@"
