"""The per-server session state one language server owns while it is up.

Split out of the manager because these three are pure state machines with no
FastAPI and no asyncio loop of their own — they are the parts of the hub that
can be reasoned about (and tested) on their own.

One thing here is not a state machine: a :class:`ServerSession` *holds* the
process-tree boundary of the child it belongs to (``proctree_jobs.ProcessJob``,
B-D5). It spawns nothing, signals nothing and waits for nothing — the manager
does all of that — but the boundary belongs to the child, so the object that
owns the child for as long as it lives owns the handle too, and releasing it
is what a replaced child must never leave behind.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DiagnosticsInbox:
    """Latest ``publishDiagnostics`` payload per URI, plus a wait-for-push step.

    LSP diagnostics arrive as a server->client *notification*, so a tool call
    cannot simply await a response: it opens/changes the document and then
    waits for the next publish for that URI (or a deadline).
    """

    def __init__(self) -> None:
        self._latest: Dict[str, Tuple[Optional[int], List[Any]]] = {}
        self._events: Dict[str, asyncio.Event] = {}

    def publish(self, uri: str, version: Optional[int], diagnostics: List[Any]) -> None:
        self._latest[uri] = (version, list(diagnostics))
        self._events.setdefault(uri, asyncio.Event()).set()

    def peek(self, uri: str) -> Optional[Tuple[Optional[int], List[Any]]]:
        return self._latest.get(uri)

    def clear(self, uri: Optional[str] = None) -> None:
        if uri is None:
            self._latest.clear()
            self._events.clear()
            return
        self._latest.pop(uri, None)
        self._events.pop(uri, None)

    def forget_session(self) -> None:
        """Drop everything (server went away) so a new session starts clean."""
        self.clear()

    async def wait(self, uri: str, version: Optional[int], timeout: float) -> Optional[List[Any]]:
        """Wait for the publish of ``version`` (any publish when the server
        does not report versions), returning its diagnostics — or ``None``
        when nothing fresh arrived before the deadline."""
        entry = self._latest.get(uri)
        if entry is not None and (
            version is None or entry[0] is None or entry[0] >= version
        ):
            return entry[1]
        # A fresh event: only a publish that arrives *now* counts, never a
        # stale one from an earlier document version.
        event = asyncio.Event()
        self._events[uri] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        entry = self._latest.get(uri)
        return entry[1] if entry else None


class DocumentStore:
    """The documents currently ``didOpen``ed on one server, with versions.

    Every tool call naming a file first opens it (or ``didChange``s it with a
    full-text change when already open), so the server analyses the *current*
    content rather than a possibly stale on-disk snapshot. The version
    counter increments monotonically.
    """

    def __init__(self) -> None:
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._last_used: Dict[str, float] = {}

    @staticmethod
    def _now() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return 0.0

    def version(self, uri: str) -> Optional[int]:
        entry = self._docs.get(uri)
        return entry["version"] if entry else None

    def touch(self, uri: str) -> None:
        self._last_used[uri] = self._now()

    def next_version(self, uri: str) -> int:
        entry = self._docs.get(uri)
        return (entry["version"] + 1) if entry else 1

    def add(self, uri: str, version: int, text: str, language_id: str) -> None:
        self._docs[uri] = {"version": version, "text": text, "languageId": language_id}
        self.touch(uri)

    def update(self, uri: str, version: int, text: str) -> None:
        entry = self._docs.get(uri)
        if entry is None:
            self.add(uri, version, text, "")
            return
        entry["version"] = version
        entry["text"] = text
        self.touch(uri)

    def text(self, uri: str) -> Optional[str]:
        entry = self._docs.get(uri)
        return entry["text"] if entry else None

    def remove(self, uri: str) -> Optional[Dict[str, Any]]:
        self._last_used.pop(uri, None)
        return self._docs.pop(uri, None)

    def uris(self) -> List[str]:
        return list(self._docs)

    def idle_uris(self, idle_seconds: float) -> List[str]:
        """Documents untouched for at least ``idle_seconds`` (loop clock)."""
        now = self._now()
        return [uri for uri, used in self._last_used.items() if now - used >= idle_seconds]


class ServerSession:
    """One live language server: process, pygls client, tree boundary, stores."""

    def __init__(
        self,
        info: Dict[str, Any],
        client: Any,
        process: Any,
        log_path: Path,
        job: Optional[Any] = None,
    ) -> None:
        self.info = info
        self.client = client
        self.process = process
        self.log_path = log_path
        #: The job containing this child's whole process tree on the platform
        #: that has no process groups; inert where the group is the boundary.
        #: Opened by the manager before the spawn, released with the child.
        self.job: Optional[Any] = job
        self.capabilities: Any = None
        self.documents = DocumentStore()
        self.inbox = DiagnosticsInbox()
        self.pull_diagnostics = False
        self.stderr_task: Optional[asyncio.Task] = None
        self.exit_task: Optional[asyncio.Task] = None
        #: Volar's tsserver relay answers, counted per command, so the status
        #: panel can show that a Vue session is actually being served rather
        #: than merely started (the failure mode here is silent, not loud).
        self.ts_relay_counts: Dict[str, int] = {}
