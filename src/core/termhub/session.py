"""One shell on one PTY: spawn, pump, resize, tear down.

``TerminalSession`` is the ed-06 core behind ``WS /api/terminal``. It knows
nothing about WebSockets: like the benchmark and chat cores it takes an
injected ``emit`` (an async callable publishing plain dicts) and is handed a
``receive_frame`` callable that yields the next client frame. Everything else —
the pty, the child, the escalation — is process work.

Why a PTY rather than a pipe pair: xterm renders an ANSI stream, and only the
kernel's terminal emulation gives colours, cursor movement, the alternate
screen, line editing and Ctrl-C-as-signal. The server only relays bytes and
updates the window size, and ``pty``/``fcntl``/``termios`` are stdlib.

The two rules inherited from AGENTS.md:

* the shell leads its own process group (``start_new_session=True``), which is
  what makes the group signal reach its children;
* teardown goes through :func:`src.core.proctree.kill_process_tree` — the same
  SIGTERM-the-group / reap / SIGKILL escalation both hubs use — on *every* exit
  path: client disconnect, ``kill`` frame, shell exit, reader ``OSError``, app
  shutdown.
"""

import asyncio
import codecs
import logging
import os
import struct
import subprocess
import time
import uuid
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, Optional

import anyio

# POSIX-only (the pty itself is): the runtime guard lets the module import on a
# non-POSIX box so the route can answer "unsupported" instead of crashing on
# import; ``TYPE_CHECKING`` lets pyright still see the real modules (the fallback
# binds them to ``None``, which would type every call below as ``None.foo``).
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import fcntl
    import pty
    import termios
else:
    try:
        import fcntl
        import pty
        import termios
    except ImportError:  # pragma: no cover - non-POSIX platform
        fcntl = pty = termios = None

from src.config import path_location
from src.core import proctree

from . import constants

logger = logging.getLogger("benchmarkit.terminal")

# A frame publisher: the transport's send, injected by the route. Raises when
# the client is gone, which the reader loop treats as "stop pumping".
Emit = Callable[[Dict[str, Any]], Awaitable[None]]
# The next client frame, already parsed to a dict; None means the client left.
ReceiveFrame = Callable[[], Awaitable[Optional[Dict[str, Any]]]]

# Frame types, both directions (ed-06's contract).
INPUT = "input"
RESIZE = "resize"
KILL = "kill"


class TerminalUnsupported(RuntimeError):
    """No PTY on this platform (``pty`` is POSIX; the app is Linux self-hosted)."""


class SessionCapacity(Exception):
    """The concurrent-session cap is reached — the upgrade is refused."""


def shell_command() -> str:
    """The shell to run: ``$SHELL``, falling back to ``/bin/sh`` (decision D2).

    Interactive and non-login (``-i``): an embedded pane should read
    ``~/.bashrc`` — the prompt, aliases, colours a user expects there — without
    paying ``~/.profile`` side effects once per tab.
    """
    configured = (os.environ.get(constants.SHELL_KEY) or "").strip()
    if configured and os.path.isabs(configured) and os.access(configured, os.X_OK):
        return configured
    return constants.FALLBACK_SHELL


class TerminalSession:
    """A shell process, its PTY, and the task pumping PTY bytes to ``emit``."""

    def __init__(self, emit: Emit, session_id: Optional[str] = None) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self._emit = emit
        self.master_fd: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None
        self.cwd = ""
        self.shell = ""
        self.started_at = time.time()
        self.bytes_out = 0
        self.exit_code: Optional[int] = None
        self.kill_requested = False
        self._closed = False
        self._transport_gone = False
        self._reader_task: Optional[asyncio.Task] = None
        # Joined across chunk boundaries, so a multi-byte character split over
        # two os.read() results still reaches xterm whole.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    # -- spawn ----------------------------------------------------------------

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process is not None else None

    def spawn(self, cols: int, rows: int) -> None:
        """Open the PTY and start the shell on it.

        The order matters in two places. The window size is set *before* the
        spawn so the first prompt already wraps at the right width, and the
        parent's copy of the slave is closed right after it: while any process
        holds the slave open, a read of the master never reports the shell's
        death, so a leftover duplicate would hang the reader forever.

        With ``start_new_session=True`` the shell becomes a session leader with
        no controlling terminal, and the ``slave`` it is given as its stdio is
        the only terminal it can acquire — so it takes the PTY as its ctty and
        job control (``^C``, job stop/resume, ``stty``) behaves exactly as in a
        real terminal window.
        """
        if os.name != "posix":
            raise TerminalUnsupported(
                "The terminal needs a POSIX pty; this app is self-hosted on Linux"
            )
        self.cwd = path_location()
        self.shell = shell_command()
        master_fd, slave_fd = pty.openpty()
        try:
            self._set_window_size(master_fd, cols, rows)
            try:
                self.process = subprocess.Popen(
                    [self.shell, "-i"],
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=self.cwd,
                    # ``env`` is inherited verbatim — the honest trade-off: the
                    # child sees BENCHMARKIT_PATH_LOCATION and the .env-derived
                    # values because the app itself has them in its environment.
                    env=dict(
                        os.environ,
                        TERM=constants.TERM_VALUE,
                        COLUMNS=str(cols),
                        LINES=str(rows),
                    ),
                    start_new_session=True,
                )
            except OSError as exc:
                self.process = None
                raise
            finally:
                os.close(slave_fd)
        except BaseException:
            os.close(master_fd)
            raise
        self.master_fd = master_fd

    @staticmethod
    def _set_window_size(fd: int, cols: int, rows: int) -> None:
        # (rows, cols) is the struct's order — the kernel's ws_row/ws_col — and
        # the ioctl is what makes the kernel SIGWINCH the foreground job.
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    # -- I/O ------------------------------------------------------------------

    def write(self, data: str) -> None:
        """Keystrokes as xterm encoded them (already control bytes)."""
        if self.master_fd is None or not data:
            return
        with suppress(OSError):
            os.write(self.master_fd, data.encode("utf-8", "surrogateescape"))

    def resize(self, cols: int, rows: int) -> None:
        """``TIOCSWINSZ`` on the master; the kernel signals ``SIGWINCH``."""
        if self.master_fd is None:
            return
        with suppress(OSError):
            self._set_window_size(self.master_fd, cols, rows)

    def terminal_size(self) -> Optional[tuple]:
        """The window size the kernel has for this PTY, as ``(cols, rows)``.

        ``TIOCGWINSZ`` — the server-side answer to "what width does the kernel
        think this PTY has". The shell's own ``stty size`` is the client-side
        view of the same number (what the resize test reads).
        """
        if self.master_fd is None:
            return None
        try:
            rows, cols, _, _ = struct.unpack(
                "HHHH",
                fcntl.ioctl(self.master_fd, termios.TIOCGWINSZ, b"\0" * 8),
            )
        except OSError:
            return None
        return (cols, rows)

    # -- the pump -------------------------------------------------------------

    async def pump(self) -> None:
        """Stream PTY bytes until the shell is gone, then report its exit code.

        One ``emit`` per ``os.read`` chunk (capped at ``PTY_READ_BYTES``) so a
        ``cat`` of a big file is many frames rather than one giant message.
        ``os.read`` blocks, hence ``asyncio.to_thread``; it is unblocked by the
        teardown, because once the shell and its children are gone the master
        answers ``EIO``.

        A publish failure is the client leaving mid-stream: stop pumping and
        remember it, so the run loop ends on the same path as a disconnect
        rather than raising out of a task nobody is awaiting.
        """
        fd = self.master_fd
        if fd is None:  # pump only ever runs after a successful spawn
            return
        while True:
            try:
                chunk = await asyncio.to_thread(
                    os.read, fd, constants.PTY_READ_BYTES
                )
            except OSError:
                break  # EIO once every holder of the slave is gone
            if not chunk:
                break
            self.bytes_out += len(chunk)
            try:
                await self._emit(
                    {"type": "output", "data": self._decoder.decode(chunk)}
                )
            except Exception as exc:  # transport gone mid-pump
                logger.info("terminal %s: output publish failed: %s", self.id, exc)
                self._transport_gone = True
                return
        # The reader ending is how the app learns the shell exited on its own
        # (``exit``, Ctrl-D, a crash): reap it and name the code. Skipped while
        # the teardown is already handling the process, and while the socket is
        # gone (nothing left to tell).
        if self.process is not None and not self._closed and not self._transport_gone:
            self.exit_code = await asyncio.to_thread(self.process.wait)
            with suppress(Exception):
                await self._emit({"type": "exit", "code": self.exit_code})

    async def _finish_pump(self) -> None:
        """Let the reader task end, then it is safe to close the master.

        The read runs in a worker thread (``asyncio.to_thread``) that a task
        cancellation cannot interrupt, so closing the master under a still
        blocked read would risk a fd-reuse race. The teardown's session kill has
        already removed every holder of the slave, so the read returns ``EIO``
        (confirmed against a real PTY) and the task resolves on its own; the
        timeout only guards against a process that escaped the kill.
        """
        task = self._reader_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(task, timeout=constants.STOP_GRACE_SECONDS + 1.0)
        except (asyncio.TimeoutError, TimeoutError):
            # A slave holder survived the kill (a setsid'd daemon, the documented
            # opt-out). Cancel the coroutine and leave the fd open: leaking one
            # descriptor beats reading it as if it were the next session's.
            task.cancel()
        except Exception:
            pass  # the reader already recorded its own failure

    # -- the loop -------------------------------------------------------------

    async def run(self, receive_frame: ReceiveFrame) -> str:
        """Drive the session until the socket, the shell or a ``kill`` ends it.

        Returns why it ended: ``disconnect``, ``kill`` or ``shell-exit``. Every
        exit path lands in :meth:`teardown`. The ``open`` frame (pid, cwd,
        shell) is published before the pump starts, so the client always sees
        it before any output — that is what lets it flush keystrokes typed
        during the handshake against the right session.
        """
        try:
            await self._emit(
                {
                    "type": "open",
                    "pid": self.pid,
                    "cwd": self.cwd,
                    "shell": self.shell,
                }
            )
        except Exception as exc:
            logger.info("terminal %s: client gone before open: %s", self.id, exc)
            self._transport_gone = True
            await self.teardown()
            return "disconnect"

        self._reader_task = asyncio.create_task(self._pump_and_watch())
        reason = "disconnect"
        try:
            while True:
                recv = asyncio.ensure_future(receive_frame())
                done, _ = await asyncio.wait(
                    {recv, self._reader_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if self._reader_task in done:
                    # The shell is gone: ``pump`` has already reported the code
                    # (unless the socket died first, in which case nobody hears).
                    recv.cancel()
                    reason = "shell-exit" if not self._transport_gone else "disconnect"
                    break
                try:
                    frame = recv.result()
                except Exception as exc:
                    logger.info("terminal %s: receive failed: %s", self.id, exc)
                    self._transport_gone = True
                    reason = "disconnect"
                    break
                if frame is None:
                    reason = "disconnect"
                    break
                if frame.get("type") == INPUT:
                    data = frame.get("data")
                    if isinstance(data, str):
                        self.write(data)
                elif frame.get("type") == RESIZE:
                    cols, rows = _positive_pair(frame.get("cols"), frame.get("rows"))
                    if cols and rows:
                        self.resize(cols, rows)
                elif frame.get("type") == KILL:
                    self.kill_requested = True
                    reason = "kill"
                    break
                # Anything else is ignored rather than fatal: a frame from a
                # newer client must not kill an older session.
        finally:
            await self.teardown()
        return reason

    async def _pump_and_watch(self) -> None:
        """The reader task: pump, and treat a raise as a transport failure."""
        try:
            await self.pump()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("terminal %s: reader failed: %s", self.id, exc)
            self._transport_gone = True

    # -- teardown -------------------------------------------------------------

    async def teardown(self) -> None:
        """Kill the process *tree* and release the PTY. Idempotent.

        Runs on every exit path, and it goes through the shared
        ``proctree.kill_session_tree`` so a ``vim`` or a
        ``python -m http.server`` started in the pane cannot outlive the tab.
        """
        if self._closed:
            return
        self._closed = True
        # A teardown must survive the cancellation that triggered it. When the
        # WebSocket handler task is cancelled (client gone, or app shutdown), a
        # plain ``await`` in the kill below would re-raise CancelledError and
        # abandon the SIGTERM→SIGKILL escalation mid-way — stranding a blocked
        # ``process.wait`` thread. An interactive shell traps SIGTERM, so
        # SIGKILL is the step that actually reaps it, which makes shielding the
        # whole teardown (kill, drain, fd release) the only way its guarantees
        # hold under cancel.
        with anyio.CancelScope(shield=True):
            pid = self.pid
            # The *session*, not the shell's group: an interactive shell gives
            # every job its own process group, so killpg(shell) would leave a
            # backgrounded ``http.server`` alive. ``pid`` is the session id
            # (start_new_session=True).
            if pid is not None and os.name == "posix":
                await proctree.kill_session_tree(
                    pid, self._reap_leader(), constants.STOP_GRACE_SECONDS
                )
            # Drain the reader (it unblocks with EIO now the session is gone)
            # before touching the fd, so the worker thread never reads a
            # closed/reused one.
            await self._finish_pump()
            if self.master_fd is not None:
                with suppress(OSError):
                    os.close(self.master_fd)
                self.master_fd = None

    def _reap_leader(self):
        """The reap step ``kill_session_tree`` runs before probing the session."""

        async def reap(deadline: float) -> None:
            process = self.process
            if process is None:
                return
            loop = asyncio.get_event_loop()
            with suppress(Exception):
                await asyncio.wait_for(
                    asyncio.to_thread(process.wait),
                    timeout=max(deadline - loop.time(), 0.01),
                )

        return reap

    # -- status ---------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """What ``GET /api/terminal`` reports — an orphan must be nameable."""
        return {
            "id": self.id,
            "pid": self.pid,
            "cwd": self.cwd,
            "shell": self.shell,
            "startedAt": self.started_at,
            "bytesOut": self.bytes_out,
            "exitCode": self.exit_code,
        }


def _positive_pair(cols: Any, rows: Any) -> tuple:
    """Both frame values as positive ints, else ``(0, 0)`` (ignore the frame)."""
    try:
        cols_i, rows_i = int(cols), int(rows)
    except (TypeError, ValueError):
        return (0, 0)
    if cols_i <= 0 or rows_i <= 0:
        return (0, 0)
    return (cols_i, rows_i)
