"""``McpManager``: spawn connector subprocesses and bridge sync callers onto them.

The hub's own asyncio thread (``mcp-hub``) owns every client session, so
the synchronous chat worker threads can call ``call_tool_sync`` without
ever running an event loop themselves. This module keeps only the
orchestration — lifecycle, hot toggles, the tool/status payloads and the
sync bridge. What it delegates: discovery (``discovery.py``), port
allocation (``ports.py``), tool/result rendering (``formatting.py``) and
the subprocess environment (``process.py``). Timeouts and the retry policy
are read through :mod:`mcphub.constants` at call time.

The heavy ``mcp`` package stays lazily imported throughout, so the manager
can be constructed (and its bookkeeping tested) without the ``[mcp]`` extra.
"""

import asyncio
import sys
import threading
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import constants
from .discovery import mcp_installed
from .formatting import extract_tool_result_text, tools_to_openai_format
from .process import connector_environment


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
        project_root: Path = constants.PROJECT_ROOT,
        logs_dir: Optional[Path] = None,
    ) -> None:
        self._connectors: Dict[str, Dict[str, Any]] = {
            connector["id"]: {
                **connector,
                "kind": connector.get("kind", "builtin"),
                "external": connector.get("external", False),
                "status": "stopped",
                "error": None,
            }
            for connector in connectors
        }
        self._python_exe = python_exe or sys.executable
        self._project_root = project_root
        self._logs_dir = logs_dir or (project_root / "logs" / "mcp")
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._stacks: Dict[str, AsyncExitStack] = {}
        self._clients: Dict[str, Any] = {}
        self._http_clients: Dict[str, Any] = {}
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
        """Spawn every connector and connect a client to each. Never raises.

        Disabled externals (``enable: false`` in ``externals.json``) stay
        ``stopped`` until :meth:`start_connector`.
        """
        if self._started or self._stopped:
            return
        self._started = True
        if not self._installed:
            for info in self._connectors.values():
                info["status"] = "stopped"
                info["error"] = constants.CONNECTORS_MISSING_HINT
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-hub", daemon=True)
        self._thread.start()
        futures = [
            asyncio.run_coroutine_threadsafe(self._start_one(cid), self._loop)
            for cid, info in self._connectors.items()
            if not (info["external"] and not info.get("enable"))
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
        """Bring one connector up, dispatching on its kind. Never raises."""
        info = self._connectors[connector_id]
        try:
            if info["kind"] in ("local", "remote"):
                await self._start_external(connector_id, info)
            else:
                await self._start_builtin(connector_id, info)
        except Exception as exc:  # noqa: BLE001 — startup errors land in status
            info["status"] = "error"
            info["error"] = str(exc) or exc.__class__.__name__

    async def _start_builtin(self, connector_id: str, info: Dict[str, Any]) -> None:
        if not info["port"]:
            info["status"] = "error"
            info["error"] = (
                f"No free port in range {constants.AUTO_PORT_BASE}-"
                f"{constants.AUTO_PORT_BASE + constants.AUTO_PORT_SPAN - 1} for {connector_id}"
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
                    "--host=127.0.0.1",
                    f"--port={info['port']}",
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

        for _ in range(constants.CONNECT_RETRIES):
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
                    await asyncio.wait_for(stack.aclose(), timeout=constants.STOP_GRACE_SECONDS)
                await asyncio.sleep(constants.CONNECT_INTERVAL)
        await self._close_connector(connector_id)
        info["status"] = "error"
        info["error"] = last_error or "Connector did not become ready"

    def _make_http_client(self, headers: Dict[str, str]) -> Any:
        """HTTP client for remote externals, carrying their custom headers."""
        try:
            from mcp.shared._httpx_utils import create_mcp_http_client

            return create_mcp_http_client(headers=headers or None)
        except Exception:  # noqa: BLE001 — private helper, fall back to raw httpx
            import httpx2

            kwargs: Dict[str, Any] = {
                "timeout": httpx2.Timeout(30.0, read=300.0),
            }
            if headers:
                kwargs["headers"] = headers
            return httpx2.AsyncClient(**kwargs)

    async def _start_external(self, connector_id: str, info: Dict[str, Any]) -> None:
        """Connect an external connector: Streamable HTTP (remote) or stdio (local)."""
        info["status"] = "starting"
        from mcp import Client  # lazy: importable only with the [mcp] extra

        retries = (
            constants.EXTERNAL_CONNECT_RETRIES
            if info["kind"] == "local"
            else constants.CONNECT_RETRIES
        )
        last_error: Optional[str] = None
        for _ in range(retries):
            stack = AsyncExitStack()
            http_client = None
            try:
                if info["kind"] == "remote":
                    from mcp.client.streamable_http import streamable_http_client

                    http_client = self._make_http_client(info.get("headers") or {})
                    transport = streamable_http_client(
                        info["endpoint"], http_client=http_client
                    )
                    client = Client(transport)
                else:
                    from mcp import StdioServerParameters

                    argv = info.get("command") or []
                    if not argv:
                        raise ValueError("local connector has no command")
                    env = {
                        **connector_environment(self._project_root),
                        **(info.get("env") or {}),
                    }
                    params = StdioServerParameters(
                        command=argv[0],
                        args=list(argv[1:]),
                        env=env,
                        cwd=str(self._project_root),
                    )
                    client = Client(params)
                await stack.enter_async_context(client)
                listing = await client.list_tools()
                self._stacks[connector_id] = stack
                self._clients[connector_id] = client
                if http_client is not None:
                    self._http_clients[connector_id] = http_client
                self._tools[connector_id] = list(listing.tools)
                info["status"] = "running"
                info["error"] = None
                return
            except Exception as exc:  # noqa: BLE001 — retry until the server is up
                last_error = str(exc) or exc.__class__.__name__
                with suppress(Exception):
                    await asyncio.wait_for(
                        stack.aclose(), timeout=constants.STOP_GRACE_SECONDS
                    )
                # A caller-supplied http client is not owned by the transport.
                if (
                    http_client is not None
                    and self._http_clients.get(connector_id) is not http_client
                ):
                    with suppress(Exception):
                        await http_client.aclose()
                await asyncio.sleep(constants.CONNECT_INTERVAL)
        await self._close_connector(connector_id)
        info["status"] = "error"
        info["error"] = last_error or "Connector did not become ready"

    def start_connector(self, connector_id: str) -> None:
        """Start (or restart) one connector from a synchronous caller. Never raises.

        Fire-and-forget: the status flips to ``starting`` right away and the
        connection attempt runs on the hub thread.
        """
        info = self._connectors.get(connector_id)
        if info is None or self._loop is None or self._stopped:
            return
        info["status"] = "starting"
        info["error"] = None
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(
                self._restart_one(connector_id), self._loop
            )

    def stop_connector(self, connector_id: str) -> None:
        """Tear one connector down, waiting briefly so callers can re-read status."""
        if connector_id not in self._connectors or self._loop is None or self._stopped:
            return
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(
                self._stop_one(connector_id), self._loop
            ).result(timeout=2 * constants.STOP_GRACE_SECONDS + 5)

    async def _restart_one(self, connector_id: str) -> None:
        await self._close_connector(connector_id)
        await self._start_one(connector_id)

    async def _stop_one(self, connector_id: str) -> None:
        await self._close_connector(connector_id)
        info = self._connectors[connector_id]
        info["status"] = "stopped"
        info["error"] = None

    def connector_status(self, connector_id: str) -> Optional[str]:
        """Live status of one known connector, or ``None`` when unknown."""
        info = self._connectors.get(connector_id)
        return info["status"] if info is not None else None

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
        """Connector definitions merged with live status, in discovery order.

        ``headers``/``env`` of external connectors are deliberately excluded.
        """
        return [
            {
                **{
                    key: info.get(key)
                    for key in (
                        "id",
                        "label",
                        "description",
                        "port",
                        "endpoint",
                        "command",
                        "external",
                        "kind",
                    )
                },
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
        http_client = self._http_clients.pop(connector_id, None)
        if stack is not None:
            with suppress(Exception):
                await asyncio.wait_for(
                    stack.aclose(), timeout=constants.STOP_GRACE_SECONDS
                )
        if http_client is not None:
            with suppress(Exception):
                await asyncio.wait_for(
                    http_client.aclose(), timeout=constants.STOP_GRACE_SECONDS
                )
        process = self._processes.pop(connector_id, None)
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=constants.STOP_GRACE_SECONDS
                )
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
                timeout=(constants.CONNECT_RETRIES + 10) * constants.STOP_GRACE_SECONDS,
            )
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=constants.STOP_GRACE_SECONDS)
        self._loop = None
        self._thread = None
