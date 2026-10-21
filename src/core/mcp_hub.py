"""MCP connector discovery, tool formatting, and lifecycle management.

Core-only: no FastAPI or browser concerns. Discovers the standalone MCP
connector servers in ``mcp_connectors/`` (one ``*_mcp.py`` module each),
loads declarative external connectors from
``mcp_connectors/externals.json`` (kilo.json-style ``remote``/``local``
entries with an ``enable`` flag), spawns one subprocess per connector for
the whole lifetime of the app,
and keeps a persistent MCP client per connector on a dedicated asyncio
thread so synchronous callers (chat worker threads) can bridge onto it
with :meth:`McpManager.call_tool_sync`.

The heavy ``mcp`` package is imported lazily so discovery and formatting
keep working when the optional ``[mcp]`` extra is not installed.
"""

import ast
import asyncio
import hashlib
import importlib.util
import json
import os
import shlex
import socket
import sys
import tempfile
import threading
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_CONNECTORS_DIR = PROJECT_ROOT / "mcp_connectors"
MCP_LOGS_DIR = PROJECT_ROOT / "logs" / "mcp"

# Declarative third-party connectors (kilo.json-style, keyed by id). The file
# may hold API keys in ``headers``/``env`` and is therefore gitignored; its
# contents are never returned by the API.
EXTERNALS_PATH = MCP_CONNECTORS_DIR / "externals.json"

# Auto-allocated listen ports for built-in connectors. The hub derives a
# stable preferred port per connector id inside this range, probes it for
# availability, and passes the result to the child process as ``--port``.
# A connector module must accept ``--port`` and never assume a port of its
# own; its internal default is only a fallback for manual runs.
AUTO_PORT_BASE = 8940
AUTO_PORT_SPAN = 60

# Connection retry policy for a freshly spawned connector process.
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


def mcp_installed() -> bool:
    """True when the optional ``mcp`` dependency (the ``[mcp]`` extra) is present."""
    return importlib.util.find_spec("mcp") is not None


def _description_from_docstring(docstring: str) -> str:
    """First docstring paragraph, whitespace-collapsed, for the settings panel."""
    paragraph = docstring.strip().split("\n\n", 1)[0]
    return " ".join(paragraph.split())


def _preferred_port(connector_id: str) -> int:
    """Deterministic preferred port for a connector id inside the auto range.

    Hashes with blake2b, never ``hash()``: Python randomises
    ``PYTHONHASHSEED``, so ``hash()`` would reshuffle the ports at every
    app start.
    """
    digest = hashlib.blake2b(connector_id.encode("utf-8"), digest_size=4).hexdigest()
    return AUTO_PORT_BASE + int(digest, 16) % AUTO_PORT_SPAN


def _port_is_free(port: int, claimed: Set[int]) -> bool:
    """True when the port is unclaimed by this hub run and bindable locally."""
    if port in claimed:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _allocate_port(connector_id: str, claimed: Set[int]) -> Optional[int]:
    """Probe forward from the preferred port (wrapping) until one is free.

    Returns the port after adding it to ``claimed`` so connectors of this
    run never collide with each other, or ``None`` when the whole range is
    occupied (foreign processes included).
    """
    preferred = _preferred_port(connector_id)
    for offset in range(AUTO_PORT_SPAN):
        port = AUTO_PORT_BASE + (preferred - AUTO_PORT_BASE + offset) % AUTO_PORT_SPAN
        if _port_is_free(port, claimed):
            claimed.add(port)
            return port
    return None


def discover_connectors(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan ``mcp_connectors/`` for ``*_mcp.py`` connector modules.

    Reads each module's docstring (via ``ast``, never imports it) for the
    panel description. Returns ``{id, label, description, port, endpoint}``
    dicts sorted by id. ``port``/``endpoint`` are hub-assigned, not
    connector-declared: a stable preferred port is derived from the
    connector id and probed for availability (:func:`_allocate_port`), so
    adding a connector needs no registry edit. ``port`` is ``0`` only when
    the whole auto range is exhausted.
    """
    root = directory or MCP_CONNECTORS_DIR
    connectors: List[Dict[str, Any]] = []
    if not root.is_dir():
        return connectors
    claimed: Set[int] = set()
    for path in sorted(root.glob("*_mcp.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        docstring = ast.get_docstring(tree) or ""
        connector_id = path.stem
        port = _allocate_port(connector_id, claimed) or 0
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


def _str_map(value: Any) -> Optional[Dict[str, str]]:
    """Validate a ``headers``/``env`` entry: string keys, scalar values.

    Returns the coerced mapping, ``{}`` for an unset value, or ``None`` when
    the shape is invalid.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        return None
    coerced: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (str, int, float, bool)):
            return None
        coerced[key] = str(item)
    return coerced


def _argv_from_command(command: Any, args: Any) -> Optional[List[str]]:
    """Normalize a local connector command into an argv list.

    Accepts a kilo-style argv array or a plain command string (shell-quoting
    rules) with an optional ``args`` list appended. ``None`` when invalid.
    """
    argv: List[str]
    if isinstance(command, str):
        try:
            argv = shlex.split(command)
        except ValueError:
            return None
    elif isinstance(command, list) and command and all(
        isinstance(part, str) and part for part in command
    ):
        argv = list(command)
    else:
        return None
    if args is not None:
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            return None
        argv.extend(args)
    return argv or None


def load_externals(
    path: Optional[Path] = None, reserved_ids: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read ``mcp_connectors/externals.json`` into external connector dicts.

    A missing file yields ``([], [])``. Unreadable/invalid JSON yields one
    file-level error and no connectors. A single malformed entry is skipped
    with a per-entry note; ``reserved_ids`` (built-in ids) rejects colliding
    entries. Never raises. Entries keep their ``enable`` flag; the caller
    decides what to do with disabled ones.
    """
    root = path or EXTERNALS_PATH
    reserved = set(reserved_ids or ())
    connectors: List[Dict[str, Any]] = []
    errors: List[str] = []
    if not root.is_file():
        return connectors, errors
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"{root.name} is not readable JSON: {exc}"]
    if not isinstance(data, dict):
        return [], [f"{root.name} must contain an object of connectors keyed by id"]
    for connector_id in sorted(data):
        prefix = f"{root.name}: '{connector_id}'"
        if connector_id in reserved:
            errors.append(f"{prefix}: id collides with a built-in connector — skipped")
            continue
        entry = data[connector_id]
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: entry must be a JSON object — skipped")
            continue
        kind = entry.get("type")
        kind = kind.lower() if isinstance(kind, str) else ""
        if kind not in ("local", "remote"):
            errors.append(f"{prefix}: type must be \"local\" or \"remote\" — skipped")
            continue
        headers = _str_map(entry.get("headers"))
        env = _str_map(entry.get("env"))
        if headers is None or env is None:
            errors.append(
                f"{prefix}: headers/env must be objects of string values — skipped"
            )
            continue
        endpoint = ""
        command: Optional[List[str]] = None
        if kind == "remote":
            url = entry.get("url")
            if not isinstance(url, str) or not url.lower().startswith(
                ("http://", "https://")
            ):
                errors.append(
                    f"{prefix}: remote connector needs an http(s) url — skipped"
                )
                continue
            endpoint = url
        else:
            command = _argv_from_command(entry.get("command"), entry.get("args"))
            if command is None:
                errors.append(
                    f"{prefix}: local connector needs a command — skipped"
                )
                continue
        description = entry.get("description")
        connectors.append(
            {
                "id": connector_id,
                "label": connector_id,
                "description": description if isinstance(description, str) else "",
                "port": 0,
                "endpoint": endpoint,
                "command": command,
                "external": True,
                "kind": kind,
                "enable": bool(entry.get("enable", False)),
                "headers": headers or {},
                "env": env or {},
            }
        )
    return connectors, errors


def all_connectors() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Built-in ``*_mcp.py`` connectors plus the declared external ones.

    Returns the merged list (built-ins first, discovery order preserved) and
    the external file's validation errors.
    """
    builtins = [
        {**connector, "external": False, "kind": "builtin"}
        for connector in discover_connectors()
    ]
    externals, errors = load_externals(
        reserved_ids={connector["id"] for connector in builtins}
    )
    return builtins + externals, errors


def set_externals_enabled(enabled: Dict[str, bool], path: Optional[Path] = None) -> List[str]:
    """Write ``enable`` flags back to ``externals.json``, keeping everything else.

    Only entries already present in the file are touched; other keys, entry
    order, and unknown ids are preserved. A missing file is a no-op. A file
    that cannot be parsed is left untouched (never rebuilt from partial
    knowledge) and reported back as an error list.
    """
    root = path or EXTERNALS_PATH
    if not root.is_file():
        return []
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Could not update {root.name}: {exc}"]
    if not isinstance(data, dict):
        return [f"Could not update {root.name}: top level is not an object"]
    for connector_id, value in enabled.items():
        entry = data.get(connector_id)
        if isinstance(entry, dict):
            entry["enable"] = bool(value)
    try:
        root.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return [f"Could not write {root.name}: {exc}"]
    return []


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
                info["error"] = CONNECTORS_MISSING_HINT
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
                f"No free port in range {AUTO_PORT_BASE}-"
                f"{AUTO_PORT_BASE + AUTO_PORT_SPAN - 1} for {connector_id}"
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
            EXTERNAL_CONNECT_RETRIES if info["kind"] == "local" else CONNECT_RETRIES
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
                    await asyncio.wait_for(stack.aclose(), timeout=STOP_GRACE_SECONDS)
                # A caller-supplied http client is not owned by the transport.
                if (
                    http_client is not None
                    and self._http_clients.get(connector_id) is not http_client
                ):
                    with suppress(Exception):
                        await http_client.aclose()
                await asyncio.sleep(CONNECT_INTERVAL)
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
            ).result(timeout=2 * STOP_GRACE_SECONDS + 5)

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
                await asyncio.wait_for(stack.aclose(), timeout=STOP_GRACE_SECONDS)
        if http_client is not None:
            with suppress(Exception):
                await asyncio.wait_for(
                    http_client.aclose(), timeout=STOP_GRACE_SECONDS
                )
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
