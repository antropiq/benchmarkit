"""MCP connector discovery, tool formatting, and lifecycle management.

Core-only: no FastAPI or browser concerns. Discovers the standalone MCP
connector servers in ``mcp_connectors/`` (one ``*_mcp.py`` module each),
spawns one subprocess per connector for the whole lifetime of the app,
and keeps a persistent MCP client per connector on a dedicated asyncio
thread so synchronous callers (chat worker threads) can bridge onto it
with :meth:`McpManager.call_tool_sync`.

The heavy ``mcp`` package is imported lazily so discovery and formatting
keep working when the optional ``[mcp]`` extra is not installed.
"""

import ast
import asyncio
import importlib.util
import os
import sys
import tempfile
import threading
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_CONNECTORS_DIR = PROJECT_ROOT / "mcp_connectors"
MCP_LOGS_DIR = PROJECT_ROOT / "logs" / "mcp"

# Connector id -> default listen port, kept in sync with each connector
# module's ``--port`` argparse default.
CONNECTOR_PORTS: Dict[str, int] = {
    "console_mcp": 9999,
    "playwright_mcp": 8934,
}

# Connection retry policy for a freshly spawned connector process.
CONNECT_RETRIES = 20
CONNECT_INTERVAL = 0.5
# Grace given to a connector process between SIGTERM and SIGKILL.
STOP_GRACE_SECONDS = 5.0

CONNECTORS_MISSING_HINT = (
    'The "mcp" package is not installed — run: pip install -e ".[mcp]"'
)


def mcp_installed() -> bool:
    """True when the optional ``mcp`` dependency (the ``[mcp]`` extra) is present."""
    return importlib.util.find_spec("mcp") is not None


def _description_from_docstring(docstring: str) -> str:
    """First docstring paragraph, whitespace-collapsed, for the settings panel."""
    paragraph = docstring.strip().split("\n\n", 1)[0]
    return " ".join(paragraph.split())


def discover_connectors(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan ``mcp_connectors/`` for ``*_mcp.py`` connector modules.

    Reads each module's docstring (via ``ast``, never imports it) for the
    panel description. Returns ``{id, label, description, port, endpoint}``
    dicts sorted by id.
    """
    root = directory or MCP_CONNECTORS_DIR
    connectors: List[Dict[str, Any]] = []
    if not root.is_dir():
        return connectors
    for path in sorted(root.glob("*_mcp.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        docstring = ast.get_docstring(tree) or ""
        connector_id = path.stem
        port = CONNECTOR_PORTS.get(connector_id, 0)
        endpoint = f"http://127.0.0.1:{port}/mcp" if port else ""
        connectors.append(
            {
                "id": connector_id,
                "label": connector_id.removesuffix("_mcp").replace("_", " ").title(),
                "description": _description_from_docstring(docstring),
                "port": port,
                "endpoint": endpoint,
            }
        )
    return connectors


def tools_to_openai_format(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert MCP tool objects into OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def extract_tool_result_text(result: Any, max_length: int = 120_000) -> str:
    """Extract plain text from an MCP CallToolResult, truncating if needed."""
    if getattr(result, "is_error", False):
        error_texts = [
            item.text if hasattr(item, "text") else item.get("text", "")
            for item in result.content
            if hasattr(item, "text") or (isinstance(item, dict) and "text" in item)
        ]
        return "\n".join(error_texts) or "[Tool execution failed — no details.]"

    text_parts: List[str] = []
    non_text_types: List[str] = []
    for item in result.content:
        if hasattr(item, "text"):
            text_parts.append(item.text)
        elif isinstance(item, dict) and "text" in item:
            text_parts.append(item["text"])
        else:
            content_type = (
                getattr(item, "type", None)
                or (item.get("type") if isinstance(item, dict) else None)
                or type(item).__name__
            )
            non_text_types.append(str(content_type))

    if text_parts:
        text = "\n".join(text_parts)
        if non_text_types:
            summary = ", ".join(sorted(set(non_text_types)))
            text += f"\n[Note: tool also returned non-text content of types: {summary}]"
    elif non_text_types:
        summary = ", ".join(sorted(set(non_text_types)))
        text = f"[Tool returned no text output. Content types received: {summary}]"
    else:
        text = "[Tool returned empty response.]"

    if len(text) > max_length:
        text = text[:max_length] + "\n... [truncated]"
    return text


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


class McpManager:
    """Spawn connector subprocesses and bridge sync callers onto their clients.

    Lifecycle: :meth:`start_all` on app startup, :meth:`stop_all` on shutdown.
    All MCP client I/O runs on a private asyncio thread so chat worker threads
    can call tools without owning an event loop.
    """

    def __init__(
        self,
        connectors: List[Dict[str, Any]],
        python_exe: Optional[str] = None,
        project_root: Path = PROJECT_ROOT,
        logs_dir: Optional[Path] = None,
    ) -> None:
        self._connectors: Dict[str, Dict[str, Any]] = {
            connector["id"]: {**connector, "status": "stopped", "error": None}
            for connector in connectors
        }
        self._python_exe = python_exe or sys.executable
        self._project_root = project_root
        self._logs_dir = logs_dir or (project_root / "logs" / "mcp")
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._stacks: Dict[str, AsyncExitStack] = {}
        self._clients: Dict[str, Any] = {}
        self._tools: Dict[str, List[Any]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False
        self._installed = mcp_installed()

    @property
    def installed(self) -> bool:
        return self._installed

    async def start_all(self) -> None:
        """Spawn every connector and connect a client to each. Never raises."""
        if self._started or self._stopped:
            return
        self._started = True
        if not self._installed:
            for info in self._connectors.values():
                info["status"] = "stopped"
                info["error"] = CONNECTORS_MISSING_HINT
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-hub", daemon=True)
        self._thread.start()
        futures = [
            asyncio.run_coroutine_threadsafe(self._start_one(cid), self._loop)
            for cid in self._connectors
        ]
        await asyncio.gather(*(asyncio.wrap_future(f) for f in futures))

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _log_tail(self, log_path: Path, chars: int = 600) -> str:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        return text[-chars:]

    async def _start_one(self, connector_id: str) -> None:
        info = self._connectors[connector_id]
        if not info["port"]:
            info["status"] = "error"
            info["error"] = (
                f"Unknown default port for {connector_id} — "
                "add it to CONNECTOR_PORTS in src/core/mcp_hub.py"
            )
            return
        info["status"] = "starting"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._logs_dir / f"{connector_id}.log"
        env = connector_environment(self._project_root)
        last_error: Optional[str] = None
        try:
            with log_path.open("ab") as log_file:
                process = await asyncio.create_subprocess_exec(
                    self._python_exe,
                    "-m",
                    f"mcp_connectors.{connector_id}",
                    cwd=str(self._project_root),
                    env=env,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                )
        except OSError as exc:
            info["status"] = "error"
            info["error"] = f"Could not spawn connector: {exc}"
            return
        self._processes[connector_id] = process

        from mcp import Client  # lazy: importable only with the [mcp] extra

        for _ in range(CONNECT_RETRIES):
            if process.returncode is not None:
                tail = self._log_tail(log_path)
                last_error = f"Connector process exited ({process.returncode})"
                if tail:
                    last_error += f": {tail}"
                break
            stack = AsyncExitStack()
            try:
                client = Client(info["endpoint"])
                await stack.enter_async_context(client)
                listing = await client.list_tools()
                self._stacks[connector_id] = stack
                self._clients[connector_id] = client
                self._tools[connector_id] = list(listing.tools)
                info["status"] = "running"
                info["error"] = None
                return
            except Exception as exc:  # noqa: BLE001 — retry until the server is up
                last_error = str(exc) or exc.__class__.__name__
                with suppress(Exception):
                    await asyncio.wait_for(stack.aclose(), timeout=STOP_GRACE_SECONDS)
                await asyncio.sleep(CONNECT_INTERVAL)
        await self._close_connector(connector_id)
        info["status"] = "error"
        info["error"] = last_error or "Connector did not become ready"

    def active_tool_defs(
        self, connector_ids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """OpenAI-format tools of the given running connectors + tool -> connector map.

        A tool name offered by several connectors is attributed to the first
        one in ``connector_ids`` order.
        """
        tools: List[Any] = []
        tool_to_server: Dict[str, str] = {}
        for connector_id in connector_ids:
            info = self._connectors.get(connector_id)
            if info is None or info["status"] != "running":
                continue
            for tool in self._tools.get(connector_id, []):
                if tool.name in tool_to_server:
                    continue
                tool_to_server[tool.name] = connector_id
                tools.append(tool)
        return tools_to_openai_format(tools), tool_to_server

    def call_tool_sync(
        self, connector_id: str, name: str, arguments: Dict[str, Any], timeout: float = 120.0
    ) -> str:
        """Execute a tool call from a synchronous thread, returning its text result."""
        if self._loop is None:
            raise RuntimeError("McpManager is not started")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(connector_id, name, arguments), self._loop
        )
        return future.result(timeout=timeout)

    async def _call_tool(
        self, connector_id: str, name: str, arguments: Dict[str, Any]
    ) -> str:
        client = self._clients.get(connector_id)
        if client is None:
            return f"Error: connector '{connector_id}' is not connected"
        result = await client.call_tool(name, arguments)
        return extract_tool_result_text(result)

    def status(self) -> List[Dict[str, Any]]:
        """Connector definitions merged with live status, in discovery order."""
        return [
            {
                **{key: info[key] for key in ("id", "label", "description", "port", "endpoint")},
                "status": info["status"],
                "error": info["error"],
                "toolCount": len(self._tools.get(info["id"], [])),
            }
            for info in self._connectors.values()
        ]

    async def _close_connector(self, connector_id: str) -> None:
        stack = self._stacks.pop(connector_id, None)
        self._clients.pop(connector_id, None)
        self._tools.pop(connector_id, None)
        if stack is not None:
            with suppress(Exception):
                await asyncio.wait_for(stack.aclose(), timeout=STOP_GRACE_SECONDS)
        process = self._processes.pop(connector_id, None)
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=STOP_GRACE_SECONDS)
            except (asyncio.TimeoutError, TimeoutError):
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(Exception):
                    await process.wait()

    async def _stop_on_loop(self) -> None:
        for connector_id in list(self._connectors):
            await self._close_connector(connector_id)
            self._connectors[connector_id]["status"] = "stopped"

    async def stop_all(self) -> None:
        """Close every client session and terminate every connector process. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self._loop is None:
            return
        with suppress(Exception):
            await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(self._stop_on_loop(), self._loop)
                ),
                timeout=(CONNECT_RETRIES + 10) * STOP_GRACE_SECONDS,
            )
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=STOP_GRACE_SECONDS)
        self._loop = None
        self._thread = None
