"""``LspManager``: spawn language servers and bridge sync callers onto them.

The hub's own asyncio thread (``lsp-hub``) owns every session, so the
synchronous chat worker threads can call ``call_tool_sync`` without ever
touching the protocol themselves. This module keeps only the orchestration —
lifecycle, hot toggles, the path policy, document sync, the sync bridge, the
idle reaper and the status payload. What it delegates: discovery
(``discovery.py``), the tool catalogue (``tools.py``), the client-side
handlers (``features.py``), the per-session state (``session.py``), the
per-action requests (``actions.py``), rendering (``rendering.py``) and the
tsserver relay shapes (``tsserver.py``).

The heavy ``pygls``/``lsprotocol`` packages stay lazily imported throughout,
so discovery and the settings panel keep working without the ``[lsp]`` extra.
"""

import asyncio
import logging
import os
import signal
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.mcphub import connector_environment

from . import actions, constants
from .discovery import lsp_installed
from .features import client_capabilities, register_features
from .session import ServerSession
from .tsserver import find_project_config
from .tools import TOOL_SPECS, action_of_tool_name, tools_for_server

logger = logging.getLogger("benchmarkit.lsp")

# How often a process-group teardown probes for surviving members between its
# SIGTERM and its SIGKILL (same idea as the MCP hub's own tree killer).
_GROUP_POLL_INTERVAL_SECONDS = 0.01


class LspManager:
    """Spawn language servers and bridge sync callers onto their sessions.

    Lifecycle: :meth:`start_all` on app startup (servers declared
    ``"lazy": true`` spawn on their first tool call instead), :meth:`stop_all`
    on shutdown. All LSP I/O runs on a private asyncio thread named
    ``lsp-hub`` so chat worker threads stay synchronous.
    """

    def __init__(
        self,
        servers: List[Dict[str, Any]],
        python_exe: Optional[str] = None,
        project_root: Path = constants.PROJECT_ROOT,
        logs_dir: Optional[Path] = None,
    ) -> None:
        self._servers: Dict[str, Dict[str, Any]] = self._index(servers)
        self._python_exe = python_exe
        self._project_root = project_root
        self._logs_dir = logs_dir or (project_root / "logs" / "lsp")
        self._sessions: Dict[str, ServerSession] = {}
        self._defs: Dict[str, List[Dict[str, Any]]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False
        self._installed = lsp_installed()
        #: ids whose start attempt is currently running on the hub thread
        self._starting: Set[str] = set()
        #: stops requested while a start attempt was still in flight
        self._stop_requested: Set[str] = set()
        self._reaper: Optional[asyncio.TimerHandle] = None

    @staticmethod
    def _index(servers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Definition list -> id-keyed live records (also used by :meth:`restart`)."""
        return {
            server["id"]: {
                **server,
                "status": "stopped",
                "error": None,
                "last_used": 0.0,
                "startingSince": 0.0,
            }
            for server in servers
        }

    @property
    def installed(self) -> bool:
        return self._installed

    # -- startup / shutdown --------------------------------------------------

    async def start_all(self) -> None:
        """Spawn every activated server. Never raises.

        Servers declared ``"lazy": true`` stay stopped until their first tool
        call (tsserver's cold start is multi-second and eats RAM); the others
        come up with the app.
        """
        await self._start_all_matching(None)

    async def start_enabled(
        self, server_ids: List[str]
    ) -> None:
        """Spawn only the servers whose id is in ``server_ids``. Never raises.

        The API passes the intersection of ``LSP_ENABLED`` and each entry's
        ``enable`` flag: a prepared ``lsp_servers.json`` can stay entirely
        dormant until the user asks for it.
        """
        await self._start_all_matching(set(server_ids))

    async def _start_all_matching(self, only_ids: Optional[Set[str]]) -> None:
        if self._started or self._stopped:
            return
        self._started = True
        if not self._installed:
            for info in self._servers.values():
                if info["enable"]:
                    info["status"] = "error"
                    info["error"] = constants.LSP_MISSING_HINT
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="lsp-hub", daemon=True)
        self._thread.start()
        futures = [
            asyncio.run_coroutine_threadsafe(self._start_one(server_id), self._loop)
            for server_id, info in self._servers.items()
            if info["enable"]
            and not info["lazy"]
            and (only_ids is None or server_id in only_ids)
        ]
        await asyncio.gather(*(asyncio.wrap_future(future) for future in futures))
        self._schedule_reaper()

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

    def _append_log(self, log_path: Path, message: str) -> None:
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(str(message).rstrip("\n") + "\n")
        except OSError:
            pass

    async def _pump_stderr(self, client: Any, log_path: Path) -> None:
        """Copy the server's stderr pipe into its per-server log file.

        pygls keeps the child's stderr on a pipe (so nothing deadlocks if the
        server is chatty); draining it is what puts tsserver's own words in
        the log, where ``_log_tail`` can surface them on an initialize error.
        """
        stream = getattr(client._server, "stderr", None)  # noqa: SLF001
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            self._append_log(log_path, line.decode("utf-8", errors="replace"))

    async def _watch_exit(self, server_id: str, process: Any) -> None:
        """Surface a server that dies mid-conversation as a status (crash
        path): the exit code plus its log tail land in ``info['error']``, so
        the settings panel shows what happened instead of a stale 'running'.
        Cancelled by ``_close_session`` for the deliberate shutdown paths."""
        try:
            await process.wait()
        except Exception:  # noqa: BLE001 — nothing to report for a dead pipe
            return
        session = self._sessions.get(server_id)
        if session is None or session.process is not process:
            return  # deliberately closed and (maybe) already respawned
        info = self._servers.get(server_id)
        if info is None or info["status"] != "running":
            return
        info["status"] = "error"
        tail = self._log_tail(session.log_path)
        info["error"] = (
            f"language server exited with code {process.returncode}"
            + (f": {tail}" if tail else "")
        )
        self._sessions.pop(server_id, None)
        self._defs.pop(server_id, None)

    def _environment(self, info: Dict[str, Any]) -> Dict[str, str]:
        """Spawn environment: ``connector_environment`` pins a valid TMPDIR
        (the app's ``TEMP`` sampling variable would otherwise break Node),
        then the entry's own ``env`` overrides (volta shims need a PATH that
        resolves ``node``)."""
        env = connector_environment(self._project_root)
        env.update(info.get("env") or {})
        return env

    async def _start_one(self, server_id: str) -> None:
        """Bring one server up, handshake included. Never raises.

        The guard is deliberately *not* on ``status``. The synchronous entry
        point (:meth:`start_server`) publishes ``starting`` *before* dispatching
        this coroutine onto the hub thread, so a status-based guard rejects the
        very start that put the server there: it would sit in ``starting``
        forever — no process, no tools, and no status the settings toggle can
        retry. "Already up" is a live session; "an attempt is already running"
        is membership in :attr:`_starting`.
        """
        info = self._servers.get(server_id)
        if info is None or server_id in self._sessions or server_id in self._starting:
            return
        self._starting.add(server_id)
        self._stop_requested.discard(server_id)
        try:
            try:
                await self._start_session(server_id, info)
            except Exception as exc:  # noqa: BLE001 — startup errors land in status
                info["status"] = "error"
                info["error"] = str(exc) or exc.__class__.__name__
                await self._close_session(server_id)
        finally:
            self._starting.discard(server_id)
            if server_id in self._stop_requested:
                # A stop landed while this attempt was in flight (the user
                # disabled the server mid-handshake): honour it whatever the
                # attempt did, so the teardown it triggered cannot be undone by
                # a handshake that happened to succeed, nor a deliberate stop
                # reported as a spawn failure.
                self._stop_requested.discard(server_id)
                await self._close_session(server_id)
                info["status"] = "stopped"
                info["error"] = None

    async def _start_session(self, server_id: str, info: Dict[str, Any]) -> None:
        from lsprotocol import types as lsp
        from pygls import uris as lsp_uris
        from pygls.lsp.client import LanguageClient

        info["status"] = "starting"
        info["error"] = None
        info["startingSince"] = asyncio.get_event_loop().time()
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._logs_dir / f"{server_id}.log"
        argv = list(info["command"])

        client = LanguageClient("benchmarkit", "1.0")
        register_features(
            client,
            info,
            log_path,
            self._sessions,
            self._append_log,
            self._project_config_for,
        )
        client.report_server_error = lambda error, source: self._append_log(
            log_path, f"[hub] {getattr(source, '__name__', source)}: {error}"
        )
        # pygls spawns the child itself and owns its stdio pipes — the hub
        # must not create its own subprocess, or the protocol has no
        # transport to write to. stderr is captured and pumped to the log.
        try:
            await client.start_io(
                argv[0],
                *argv[1:],
                cwd=info["workspaceRoot"],
                env=self._environment(info),
                # pygls forwards these to create_subprocess_exec. Leading its
                # own process group is what lets the teardown signal the whole
                # tree rather than only the wrapper (see _kill_process_tree).
                start_new_session=True,
            )
        except OSError as exc:
            info["status"] = "error"
            info["error"] = f"Could not spawn language server: {exc}"
            return
        process = client._server  # noqa: SLF001 — pygls keeps it private
        if process is None:
            info["status"] = "error"
            info["error"] = "Language server started but exposes no process handle"
            with suppress(Exception):
                await client.stop()
            return
        session = ServerSession(info, client, process, log_path)
        session.stderr_task = asyncio.ensure_future(
            self._pump_stderr(client, log_path)
        )
        self._sessions[server_id] = session
        session.exit_task = asyncio.ensure_future(
            self._watch_exit(server_id, process)
        )

        root_uri = lsp_uris.from_fs_path(info["workspaceRoot"]) or "file://" + info["workspaceRoot"]
        try:
            result = await asyncio.wait_for(
                client.initialize_async(
                    lsp.InitializeParams(
                        client_info=lsp.ClientInfo(name="benchmarkit", version="1.0"),
                        process_id=process.pid,
                        root_uri=root_uri,
                        root_path=info["workspaceRoot"],
                        capabilities=client_capabilities(),
                        initialization_options=info.get("initializationOptions") or None,
                    )
                ),
                timeout=constants.INIT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            info["status"] = "error"
            info["error"] = (
                f"{server_id} did not answer initialize within {constants.INIT_TIMEOUT:.0f}s"
            )
            await self._close_session(server_id)
            return
        except Exception as exc:  # noqa: BLE001 — the message IS the diagnosis
            # An initialize failure must surface the server's own words:
            # "Could not find a valid TypeScript installation" is a config
            # problem the user can only fix if it is reported verbatim.
            info["status"] = "error"
            info["error"] = str(exc) or exc.__class__.__name__
            tail = self._log_tail(log_path)
            if tail:
                info["error"] += f": {tail}"
            await self._close_session(server_id)
            return

        session.capabilities = result.capabilities
        session.pull_diagnostics = result.capabilities.diagnostic_provider is not None
        client.initialized(lsp.InitializedParams())
        self._defs[server_id] = tools_for_server(info, result.capabilities)
        info["last_used"] = asyncio.get_event_loop().time()
        info["status"] = "running"
        info["error"] = None
        # Every session gets a log line of its own. Without it a quiet server
        # (Volar writes nothing until it is asked something) leaves no
        # logs/lsp/<id>.log at all, so ``GET /api/lsp/log`` answers "" exactly
        # when the user needs it most, and there is no record of which config
        # the tsserver relay will answer with.
        config_file = find_project_config(Path(info["workspaceRoot"]).resolve())
        self._append_log(
            log_path,
            f"[hub] {server_id} running (pid {process.pid}, root {info['workspaceRoot']}, "
            f"{len(self._defs[server_id])} tools, "
            + (f"project config {config_file}" if config_file else "no tsconfig/jsconfig")
            + ")",
        )

    # -- hot toggles ---------------------------------------------------------

    def start_server(self, server_id: str) -> None:
        """Start (or restart) one server from a synchronous caller. Never raises.

        Idempotent: an attempt already in flight on the hub thread is left
        alone, so a second click cannot tear down a handshake the first one
        started. A ``starting`` with nothing in flight behind it (a start that
        died before recording anything) is exactly what this call can repair.
        """
        info = self._servers.get(server_id)
        if info is None or self._loop is None or self._stopped:
            return
        if server_id in self._starting:
            return
        info["status"] = "starting"
        info["startingSince"] = self._loop.time()
        info["error"] = None
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._restart_one(server_id), self._loop)

    def stop_server(self, server_id: str) -> None:
        """Tear one server down, waiting briefly so callers can re-read status."""
        if server_id not in self._servers or self._loop is None or self._stopped:
            return
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._stop_one(server_id), self._loop).result(
                timeout=2 * constants.STOP_GRACE_SECONDS + 5
            )

    async def _restart_one(self, server_id: str) -> None:
        await self._close_session(server_id)
        await self._start_one(server_id)

    async def _stop_one(self, server_id: str) -> None:
        self._stop_requested.add(server_id)
        await self._close_session(server_id)
        info = self._servers[server_id]
        info["status"] = "stopped"
        info["error"] = None

    def server_status(self, server_id: str) -> Optional[str]:
        """Live status of one known server, or ``None`` when unknown."""
        info = self._servers.get(server_id)
        return info["status"] if info is not None else None

    def log_tail(self, server_id: str, chars: int = 4000) -> str:
        """Tail of one server's log file ("" when there is none)."""
        if server_id not in self._servers:
            return ""
        return self._log_tail(self._logs_dir / f"{server_id}.log", chars)

    # -- tool catalogue ------------------------------------------------------

    def active_tool_defs(
        self, server_ids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """OpenAI-format tools of the given running servers + tool -> server map.

        A tool name is attributed to the first server offering it in
        ``server_ids`` order (names are namespaced, so they never collide).

        A server that is asked for but has no live session yet is still
        represented, using the tools its config declares (its real
        capabilities are only known after the handshake, which its first call
        triggers). This covers both ways a hub ends up without one: a *lazy*
        server that has not been called yet, and a server the idle reaper tore
        down mid-conversation — a reaped ``pyright`` would otherwise vanish from
        the next request's tool list with nothing to bring it back. Including
        ``starting`` matters too: tsserver's cold start is multi-second, and a
        server just switched on must not be absent from the very next request.
        A server that failed (``error``) is deliberately not published.
        """
        tools: List[Dict[str, Any]] = []
        tool_to_server: Dict[str, str] = {}
        for server_id in server_ids:
            info = self._servers.get(server_id)
            if info is None:
                continue
            definitions = self._defs.get(server_id)
            if definitions is None:
                if not (
                    info["enable"]
                    and info["status"] in ("stopped", "starting")
                ):
                    continue
                definitions = tools_for_server(info, None)
            for definition in definitions:
                name = definition["function"]["name"]
                if name in tool_to_server:
                    continue
                tool_to_server[name] = server_id
                tools.append(definition)
        return tools, tool_to_server

    # -- the sync bridge -----------------------------------------------------

    def call_tool_sync(
        self,
        server_id: str,
        tool: str,
        arguments: Dict[str, Any],
        timeout: float = constants.REQUEST_TIMEOUT
        + constants.DIAGNOSTICS_WAIT_SECONDS
        + 5.0,
    ) -> str:
        """Execute one LSP-backed tool call from a synchronous thread."""
        if self._loop is None:
            raise RuntimeError("LspManager is not started")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(server_id, tool, arguments), self._loop
        )
        return future.result(timeout=timeout)

    async def _call_tool(self, server_id: str, tool: str, arguments: Dict[str, Any]) -> str:
        info = self._servers.get(server_id)
        if info is None:
            return f"Error: language server '{server_id}' is not declared"
        if info["status"] != "running":
            if not info["enable"]:
                return f"Error: language server '{server_id}' is disabled in Settings"
            if info["status"] in ("stopped", "starting"):
                # Bring it up on its first tool call — the lazy path, and the
                # way back for a server the idle reaper stopped. ``starting`` is
                # accepted too, so a start attempt that never completed (no
                # session behind the status) is retried here instead of staying
                # unreachable; an attempt still in flight is a no-op.
                await self._start_one(server_id)
            if info["status"] != "running":
                return (
                    f"Error: language server '{server_id}' is {info['status']}: "
                    f"{info['error'] or 'no details'}"
                )
        session = self._sessions.get(server_id)
        if session is None:
            return f"Error: language server '{server_id}' has no live session"
        action = action_of_tool_name(server_id, tool)
        if action not in TOOL_SPECS:
            return f"Error: unknown LSP tool '{tool}'"
        info["last_used"] = asyncio.get_event_loop().time()
        try:
            if action == "workspace_symbols":
                return await actions.workspace_symbols(session, arguments or {})
            path, error = self._resolve_path(session, (arguments or {}).get("path"))
            if error is not None or path is None:
                return f"Error: {error}"
            return await actions.run_action(session, action, arguments or {}, path)
        except asyncio.TimeoutError:
            return (
                f"Error: {server_id} did not answer {action} within "
                f"{constants.REQUEST_TIMEOUT:.0f}s"
            )
        except Exception as exc:  # noqa: BLE001 — feed the failure to the model
            return f"Error: {server_id} {action} failed: {exc}"

    # -- path policy ---------------------------------------------------------

    def _resolve_path(
        self, session: ServerSession, raw_path: Any
    ) -> Tuple[Optional[Path], Optional[str]]:
        """Policy-checked absolute path for a tool argument.

        Confines reads to the server's declared workspace root, with the
        expected extension, size-capped. This is what stops
        "read ~/.ssh/id_rsa through the Python server" style tricks and makes
        the LSP hub useless as a generic file reader.
        """
        info = session.info
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None, "path is required"
        root = Path(info["workspaceRoot"]).resolve()
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            return None, f"path is not usable ({raw_path}): {exc}"
        if resolved != root and root not in resolved.parents:
            return None, f"path must live inside the workspace root {root}"
        if not resolved.is_file():
            return None, self._missing_path_hint(root, raw_path, resolved)
        suffix = resolved.suffix.lower()
        extensions = info.get("extensions") or []
        if extensions and suffix not in extensions:
            return None, (
                f"{suffix} is not handled by {info['id']} (handles: {', '.join(extensions)})"
            )
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            return None, f"could not stat {resolved}: {exc}"
        if size > constants.MAX_DOCUMENT_BYTES:
            return None, f"file exceeds the {constants.MAX_DOCUMENT_BYTES} byte cap"
        return resolved, None

    @staticmethod
    def _missing_path_hint(root: Path, raw_path: str, resolved: Path) -> str:
        """``not a file: …`` plus, when the cause is obvious, the fix.

        ``path`` is documented as workspace-relative, but a model that has
        just read the root out of the server's own config tends to send it
        absolute (``web/src/App.vue`` against root ``…/web``), which resolves
        to a doubled ``…/web/web/src/App.vue``. Saying only ``not a file``
        makes the model retry the same path; naming the path that does exist
        costs one line and saves a round trip.
        """
        message = f"not a file: {resolved}"
        if not isinstance(raw_path, str):
            return message
        # Candidates: the literal argument re-rooted at the workspace root,
        # and the same with the root's own name stripped from the front.
        candidates: List[Path] = []
        relative = Path(raw_path)
        candidates.append(root / relative.name if relative.is_absolute() else root / relative)
        parts = list(relative.parts)
        if parts and parts[0] == root.name:
            candidates.append(root.joinpath(*parts[1:]))
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return (
                        f"{message} — path is relative to the workspace root "
                        f"{root}, try: {candidate.relative_to(root)}"
                    )
            except OSError:
                continue
        return message

    # -- tsserver relay support ---------------------------------------------

    def _project_config_for(self, session: ServerSession, file_name: str) -> Optional[str]:
        """The tsconfig/jsconfig governing one relayed file, if any.

        TypeScript resolves a project by walking up from the file until it
        finds a config, so the relay walks the same way — but stops at the
        server's workspace root, which is the boundary the hub promises never
        to read past. A Vue component three levels down still gets the
        workspace's own config, and a workspace with no config at all gets
        ``None`` (Volar's inferred project) instead of some unrelated config
        from a parent directory.
        """
        from .rendering import uri_to_path

        root = Path(session.info["workspaceRoot"]).resolve()
        start: Optional[Path] = None
        if file_name:
            if file_name.startswith("file:"):
                # Volar sends ``uri.fsPath`` (a plain path), but a file URI
                # reaching here must not be mangled into a relative segment.
                file_name = uri_to_path(file_name) or file_name
            candidate = Path(file_name).resolve() if os.path.isabs(file_name) else root / file_name
            try:
                start = candidate.parent
            except OSError:
                start = None
        current = start if start is not None else root
        while current == root or root in current.parents:
            found = find_project_config(current)
            if found is not None:
                return found
            if current == root:
                break
            current = current.parent
        return None

    # -- teardown -------------------------------------------------------------

    async def _close_session(self, server_id: str) -> None:
        """shutdown/exit, then SIGTERM the whole process group, then SIGKILL."""
        session = self._sessions.pop(server_id, None)
        self._defs.pop(server_id, None)
        if session is None:
            return
        if session.stderr_task is not None:
            with suppress(Exception):
                session.stderr_task.cancel()
        if session.exit_task is not None:
            with suppress(Exception):
                session.exit_task.cancel()
        session.inbox.forget_session()
        client = session.client
        if client is not None and not client.stopped:
            with suppress(Exception):
                await asyncio.wait_for(
                    client.shutdown_async(None), timeout=constants.STOP_GRACE_SECONDS
                )
            with suppress(Exception):
                client.exit(None)
            with suppress(Exception):
                await asyncio.wait_for(client.stop(), timeout=constants.STOP_GRACE_SECONDS)
        process = session.process
        if process is not None:
            # Signalling only the wrapper is not enough: typescript-language-server
            # spawns tsserver, and Volar spawns workers. A SIGTERM aimed at the
            # wrapper leaves those alive, reparented to init, holding a project
            # index no later hub will ever know about. The server leads its own
            # process group (``start_new_session``), so ``killpg`` reaches every
            # descendant — the same teardown the MCP hub applies. This runs even
            # when the graceful shutdown/exit already stopped the leader: what
            # has to be empty is the *group*, and the leader's pid names it
            # whether or not the leader is still running.
            pid = getattr(process, "pid", None)
            if os.name == "posix" and isinstance(pid, int):
                await self._kill_process_tree(process, pid)
            elif process.returncode is None:
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

    @staticmethod
    def _group_alive(pgid: int) -> bool:
        """Probe the group with signal 0 — only ESRCH proves it is gone.

        EPERM is ambiguous (a foreign-euid member), so it counts as alive and
        keeps waiting rather than pretending the group drained.
        """
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True

    async def _kill_process_tree(self, process: Any, pgid: int) -> None:
        """SIGTERM the group, reap the leader, SIGKILL whatever outlives it.

        The leader's pid *is* the pgid, which is why no ``getpgid()`` is used —
        and none could work here: it fails once the leader has been reaped, even
        while other members are still running. The leader is reaped *before* the
        group is probed, because a group only disappears once every member is
        dead **and** reaped, so an unreaped leader would keep the probe — and so
        the whole grace period — alive.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + constants.STOP_GRACE_SECONDS
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return  # the whole group is already gone
        except PermissionError:
            logger.warning("No permission to signal process group %d", pgid)
        with suppress(Exception):
            await asyncio.wait_for(
                process.wait(), timeout=max(deadline - loop.time(), 0.01)
            )
        while loop.time() < deadline:
            if not self._group_alive(pgid):
                return
            await asyncio.sleep(_GROUP_POLL_INTERVAL_SECONDS)
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)

    # -- idle reaper ----------------------------------------------------------

    def _schedule_reaper(self) -> None:
        if self._loop is None or self._stopped:
            return
        self._reaper = self._loop.call_later(constants.REAPER_INTERVAL_SECONDS, self._reap)

    def _reap(self) -> None:
        """Tear down idle servers, and expire a ``starting`` that will never
        finish."""
        if self._loop is None or self._stopped:
            return
        now = self._loop.time()
        for server_id, info in self._servers.items():
            if info["status"] == "starting":
                # A ``starting`` that outlived the whole start budget is stuck,
                # not slow: it is the one status no other path clears, so an
                # attempt that died before recording anything would otherwise
                # be permanent. The budget (handshake + teardown grace + slack)
                # is above anything a *successful* attempt can take, so a
                # healthy cold start is never interrupted — and if an attempt
                # later does succeed, it simply overwrites this with
                # ``running``. Naming it as ``error`` makes the panel say so and
                # lets the settings toggle retry.
                elapsed = now - info.get("startingSince", now)
                if elapsed >= constants.STARTING_STUCK_SECONDS:
                    info["status"] = "error"
                    info["error"] = (
                        f"{server_id} has been starting for {elapsed:.0f}s "
                        f"without completing its handshake"
                    )
                    self._append_log(
                        self._logs_dir / f"{server_id}.log", f"[hub] {info['error']}"
                    )
            idle = info.get("idleShutdownSeconds") or 0
            if info["status"] != "running" or idle <= 0:
                continue
            if now - info.get("last_used", 0.0) < idle:
                continue
            with suppress(Exception):
                asyncio.ensure_future(self._reap_one(server_id), loop=self._loop)
        self._schedule_reaper()

    async def _reap_one(self, server_id: str) -> None:
        info = self._servers[server_id]
        await self._close_session(server_id)
        info["status"] = "stopped"
        info["error"] = None

    async def _stop_on_loop(self) -> None:
        if self._reaper is not None:
            with suppress(Exception):
                self._reaper.cancel()
            self._reaper = None
        for server_id in list(self._sessions):
            await self._close_session(server_id)
            self._servers[server_id]["status"] = "stopped"

    async def stop_all(self) -> None:
        """Close every session and terminate every server process. Idempotent."""
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
                timeout=(len(self._servers) + 1)
                * (2 * constants.STOP_GRACE_SECONDS + 5),
            )
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=constants.STOP_GRACE_SECONDS)
        self._loop = None
        self._thread = None

    async def restart(
        self,
        servers: List[Dict[str, Any]],
        enabled_ids: Optional[List[str]] = None,
    ) -> None:
        """Re-read ``lsp_servers.json``: shut every server down and respawn.

        Unlike :meth:`stop_all`, the manager stays usable afterwards: a fresh
        ``lsp-hub`` thread (and reaper) is created and *this same object* keeps
        serving the chat worker threads that already hold a reference to it, so
        a restart never invalidates an in-flight conversation. ``servers`` is
        the freshly validated file scan, so a hand-edited declaration — a new
        server entry, a changed ``command``, another ``workspaceRoot`` — takes
        effect without restarting the web app.

        ``enabled_ids`` re-applies the two-gate rule (entry ``enable`` flag and
        ``LSP_ENABLED``); ``None`` keeps the startup behaviour of spawning every
        entry whose flag is on. Never raises: a server that fails to come back
        up lands in its own ``status``/``error``, exactly as on startup.
        """
        await self.stop_all()
        self._servers = self._index(servers)
        self._sessions.clear()
        self._defs.clear()
        self._starting.clear()
        self._stop_requested.clear()
        self._loop = None
        self._thread = None
        self._reaper = None
        self._started = False
        self._stopped = False
        self._installed = lsp_installed()
        if enabled_ids is None:
            await self.start_all()
        else:
            await self.start_enabled(enabled_ids)

    # -- status ----------------------------------------------------------------

    def status(self) -> List[Dict[str, Any]]:
        """Server definitions merged with live status, in declaration order.

        ``env`` values are deliberately excluded — they may hold secrets and
        never leave the server.
        """
        return [
            {
                **{
                    key: info.get(key)
                    for key in (
                        "id",
                        "label",
                        "description",
                        "command",
                        "workspaceRoot",
                        "extensions",
                        "languageId",
                        "enable",
                        "lazy",
                        "tools",
                        "maxDiagnostics",
                        "idleShutdownSeconds",
                    )
                },
                "status": info["status"],
                "error": info["error"],
                "toolCount": len(self._defs.get(info["id"], [])),
            }
            for info in self._servers.values()
        ]
