"""Standalone MCP connector servers bundled with agentic-tool.

Modules:
    console_mcp    — Console command execution MCP server (default port 9999)
    playwright_mcp — Browser automation MCP server (default port 8934)

Run either as a standalone server, e.g.:
    python -m mcp_connectors.console_mcp --port 9999
    python -m mcp_connectors.playwright_mcp --port 8934
"""
