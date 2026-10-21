"""Standalone MCP connector servers bundled with benchmarkit.

Contract for a built-in connector (``*_mcp.py``):
    It MUST accept a ``--port`` flag (and ``--host``, default ``127.0.0.1``)
    and bind exactly what it is given. The benchmarkit MCP hub is the single
    port authority: it allocates a port per connector at discovery time
    (deterministically from the id, probed for availability) and always
    launches the module with ``--port``/``--host``. A module must not assume
    a port of its own — the internal ``DEFAULT_PORT`` / argparse default is
    only a fallback for running the server by hand.

Modules:
    console_mcp    — Console command execution MCP server
    playwright_mcp — Browser automation MCP server
    fedit_mcp      — File edit (diff / SEARCH-REPLACE) MCP server

Pair ``console_mcp`` with ``fedit_mcp``: the console connector runs shell
commands, while every file read/edit (with the path policy, backups and
revert) goes through ``fedit_mcp`` — the single file-content authority.

Run either as a standalone server, e.g.:
    python -m mcp_connectors.console_mcp --port 9999
    python -m mcp_connectors.playwright_mcp --port 8934
"""
