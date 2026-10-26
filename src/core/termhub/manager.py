"""The live set of terminal sessions: capacity, listing, shutdown.

``TerminalHub`` is the ``app.state.terminal`` singleton, mirroring what the two
hubs are for their children: a long-lived registry the API reads (``GET
/api/terminal``) and the ``lifespan`` shutdown stops. It holds no loop of its
own — unlike the hubs, a terminal session is driven by the WebSocket coroutine
that opened it, on the main event loop — so this is deliberately a thin owner:
the cap decision, the id→session map, and ``stop_all``.

The same visibility rule the hubs follow (AGENTS.md): *an orphan must be
nameable*. ``list_sessions`` reports pid/cwd/started/bytes so a shell that
somehow outlived its socket can be seen; ``stop_all`` kills any such session on
shutdown, and every session also tears itself down when its socket ends, so the
registry never accumulates on the normal path.
"""

import asyncio
import logging
from contextlib import suppress
from typing import Any, Dict, List, Optional

from . import constants
from .session import Emit, SessionCapacity, TerminalSession

logger = logging.getLogger("benchmarkit.terminal")


class TerminalHub:
    """Registry of live PTY sessions, bounded by ``MAX_SESSIONS``."""

    def __init__(self) -> None:
        self._sessions: Dict[str, TerminalSession] = {}

    def capacity(self) -> int:
        """Read at call time so a patched cap is honoured (constants rule)."""
        return constants.MAX_SESSIONS

    def count(self) -> int:
        return len(self._sessions)

    def is_full(self) -> bool:
        return len(self._sessions) >= self.capacity()

    def create(self, emit: Emit) -> TerminalSession:
        """Reserve a slot and build the session (the caller then ``spawn``s it).

        Raises :class:`SessionCapacity` when the app already runs the maximum
        number of shells, so the route can refuse the *upgrade* rather than
        silently opening another process tree. The session is registered before
        it spawns, so a spawn failure still lands in :meth:`drop`.
        """
        if self.is_full():
            raise SessionCapacity(
                f"Too many terminal sessions ({len(self._sessions)} of "
                f"{self.capacity()}); close one before opening another"
            )
        session = TerminalSession(emit)
        self._sessions[session.id] = session
        return session

    def drop(self, session_id: str) -> None:
        """Remove a finished session from the registry (does not kill it).

        The route calls this after ``teardown``; the process is already gone by
        then, so this only keeps ``list_sessions``/``is_full`` honest.
        """
        self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> Optional[TerminalSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """One dict per live session, in id order — the ``GET /api/terminal`` body."""
        return [self._sessions[key].status() for key in sorted(self._sessions)]

    async def stop_all(self) -> None:
        """Tear down every session that outlived its socket. Idempotent.

        A session already tearing down on its own path is safe to pass here
        too: ``teardown`` is guarded, so the second call is a no-op. Each is
        awaited through ``kill_session_tree``'s own grace, so the app exit pays
        at most ``STOP_GRACE_SECONDS`` per stubborn shell, not per process.
        """
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            with suppress(Exception):
                await asyncio.wait_for(
                    session.teardown(),
                    timeout=constants.STOP_GRACE_SECONDS + 1.0,
                )
