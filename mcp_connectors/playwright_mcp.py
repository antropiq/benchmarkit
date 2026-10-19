#!/usr/bin/env python3
"""
Playwright MCP Server
Exposes browser automation tools via MCP (Model Context Protocol)
on http://localhost:8934/mcp

Usage:
    python -m mcp_connectors.playwright_mcp               # Serve on 127.0.0.1:8934
    python -m mcp_connectors.playwright_mcp --port 9000   # Custom port
"""

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mcp.server import Server
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------------------------------------------------------
# Restricted FQDN list (restricted-fqdn.txt next to this module)
# ---------------------------------------------------------------------------

RESTRICTED_FQDN_FILE = Path(__file__).resolve().parent / "restricted-fqdn.txt"


def load_restricted_fqdns() -> list[str]:
    """Read restricted FQDNs from the file (one per line, # comments allowed).

    The file is re-read on every call so edits are picked up without restart.
    A missing or unreadable file simply means no restrictions.
    """
    try:
        lines = RESTRICTED_FQDN_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        entry = line.strip().lower().lstrip(".")
        if entry and not entry.startswith("#"):
            entries.append(entry)
    return entries


def is_restricted_url(url: str) -> bool:
    """True when the URL's host is, or is a subdomain of, a restricted FQDN."""
    restricted = load_restricted_fqdns()
    if not restricted:
        return False
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    return any(host == entry or host.endswith("." + entry) for entry in restricted)


async def block_restricted_routes(route):
    """Playwright route handler: drop requests to restricted FQDNs silently."""
    try:
        if is_restricted_url(route.request.url):
            await route.abort()
        else:
            await route.continue_()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Browser state (singleton – one browser per server session)
# ---------------------------------------------------------------------------
_browser = None
_page = None


async def get_browser():
    """Lazy-initialise a Playwright browser instance."""
    global _browser
    if _browser is None:
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        # FIX: Added --no-sandbox and other flags to prevent hanging in systemd/headless env
        _browser = await pw.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
    return _browser


async def get_page():
    """Return the current page, or open a blank one if none exists."""
    global _page, _browser
    if _page is None or _page.is_closed():
        browser = await get_browser()
        _page = await browser.new_page()
        await _page.route("**/*", block_restricted_routes)
        # Set a shorter default timeout to prevent long hangs
        # (sync in playwright >= 1.63, coroutine in older versions)
        timeout_result = _page.set_default_timeout(10000)
        if hasattr(timeout_result, "__await__"):
            await timeout_result
    return _page


async def close_browser():
    global _browser, _page
    if _page and not _page.is_closed():
        await _page.close()
        _page = None
    if _browser:
        await _browser.close()
        _browser = None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="browser_navigate",
        description="Navigate to a URL in the browser.",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to."},
            },
            "required": ["url"],
        },
    ),
    Tool(
        name="browser_screenshot",
        description="Take a screenshot of the current page.",
        inputSchema={
            "type": "object",
            "properties": {
                "full_page": {
                    "type": "boolean",
                    "description": "Whether to screenshot the full scrollable page.",
                    "default": False,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="browser_click",
        description="Click an element on the page by its CSS selector.",
        inputSchema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to click.",
                },
            },
            "required": ["selector"],
        },
    ),
    Tool(
        name="browser_type",
        description="Type text into an input element.",
        inputSchema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input element.",
                },
                "text": {
                    "type": "string",
                    "description": "The text to type.",
                },
                "submit": {
                    "type": "boolean",
                    "description": "Whether to press Enter after typing.",
                    "default": False,
                },
            },
            "required": ["selector", "text"],
        },
    ),
    Tool(
        name="browser_get_text",
        description="Get the text content of an element or the whole page.",
        inputSchema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": (
                        "CSS selector. If omitted, returns the full page text."
                    ),
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="browser_evaluate",
        description="Execute JavaScript in the page context and return the result.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
    ),
    Tool(
        name="browser_select_option",
        description="Select an option in a <select> element.",
        inputSchema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the <select> element.",
                },
                "value": {
                    "type": "string",
                    "description": "The value of the option to select.",
                },
            },
            "required": ["selector", "value"],
        },
    ),
    Tool(
        name="browser_press_key",
        description="Press a keyboard key.",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key name (e.g. 'Enter', 'Tab', 'Escape').",
                },
            },
            "required": ["key"],
        },
    ),
    Tool(
        name="browser_scroll_down",
        description="Scroll the page down by one viewport.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="browser_scroll_up",
        description="Scroll the page up by one viewport.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="browser_go_back",
        description="Navigate back in browser history.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="browser_go_forward",
        description="Navigate forward in browser history.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="browser_close",
        description="Close the browser and end the session.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


async def list_tools() -> list[Tool]:
    return TOOLS


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch a tool call."""
    try:
        if name == "browser_navigate" and is_restricted_url(arguments.get("url", "")):
            # Restricted FQDN: return an empty response, not an error.
            return [TextContent(type="text", text="")]

        page = await get_page()

        if name == "browser_navigate":
            url = arguments["url"]
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            title = await page.title()
            return [TextContent(type="text", text=f"Navigated to {url} (title: {title})")]

        elif name == "browser_screenshot":
            full_page = arguments.get("full_page", False)
            screenshot = await page.screenshot(full_page=full_page, type="png")
            return [TextContent(
                type="text",
                text=f"Screenshot captured ({len(screenshot)} bytes).",
            )]

        elif name == "browser_click":
            selector = arguments["selector"]
            await page.click(selector, timeout=10000)
            return [TextContent(type="text", text=f"Clicked '{selector}'.")]

        elif name == "browser_type":
            selector = arguments["selector"]
            text = arguments["text"]
            submit = arguments.get("submit", False)
            await page.fill(selector, text, timeout=10000)
            if submit:
                await page.keyboard.press("Enter")
            return [TextContent(type="text", text=f"Typed '{text}' into '{selector}'.")]

        elif name == "browser_get_text":
            selector = arguments.get("selector")
            if selector:
                text = await page.inner_text(selector)
            else:
                text = await page.inner_text("body")
            return [TextContent(type="text", text=text)]

        elif name == "browser_evaluate":
            expression = arguments["expression"]
            result = await page.evaluate(expression)
            return [TextContent(type="text", text=str(result))]

        elif name == "browser_select_option":
            selector = arguments["selector"]
            value = arguments["value"]
            await page.select_option(selector, value, timeout=10000)
            return [TextContent(type="text", text=f"Selected '{value}' in '{selector}'.")]

        elif name == "browser_press_key":
            key = arguments["key"]
            await page.keyboard.press(key)
            return [TextContent(type="text", text=f"Pressed key '{key}'.")]

        elif name == "browser_scroll_down":
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            return [TextContent(type="text", text="Scrolled down.")]

        elif name == "browser_scroll_up":
            await page.evaluate("window.scrollBy(0, -window.innerHeight)")
            return [TextContent(type="text", text="Scrolled up.")]

        elif name == "browser_go_back":
            await page.go_back()
            return [TextContent(type="text", text="Navigated back.")]

        elif name == "browser_go_forward":
            await page.go_forward()
            return [TextContent(type="text", text="Navigated forward.")]

        elif name == "browser_close":
            await close_browser()
            return [TextContent(type="text", text="Browser closed.")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        # Return error details to n8n instead of hanging
        return [TextContent(type="text", text=f"Error executing tool: {str(e)}")]


async def on_list_tools(ctx, params) -> ListToolsResult:
    return ListToolsResult(tools=await list_tools())


async def on_call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
    content = await call_tool(params.name, params.arguments or {})
    return CallToolResult(content=content)


server = Server(
    "playwright-mcp",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)


# ---------------------------------------------------------------------------
# HTTP transport (Streamable HTTP, served at /mcp)
# ---------------------------------------------------------------------------

# DNS rebinding protection is left off to preserve the original server's
# behavior of accepting any Host/Origin.
starlette_app = server.streamable_http_app(
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

starlette_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    parser = argparse.ArgumentParser(
        description="Playwright MCP Server - Browser automation via MCP protocol"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8934,
        help="Port to listen on (default: 8934)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    print(f"Starting Playwright MCP Server on http://{args.host}:{args.port}")
    print(f"  Streamable HTTP   : http://{args.host}:{args.port}/mcp")
    uvicorn.run(starlette_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
