"""One shell on one PTY: spawn, pump, resize, tear down.

``TerminalSession`` is the core behind ``WS /api/terminal``. It knows
nothing about WebSockets: like the benchmark and chat cores it takes an
injected ``emit`` (an async callable publishing plain dicts) and is handed a
``receive_frame`` callable that yields the next client frame. Everything else —
the pty, the child, the escalation — is process work.

Why a PTY rather than a pipe pair: xterm renders an ANSI stream, and only the
kernel's terminal emulation gives colours, cursor movement, the alternate
screen, line editing and Ctrl-C-as-signal. The server only relays bytes and
updates the window size; which pseudo-terminal mechanism that is — if any —
is the seam's answer in :mod:`termhub.drivers`, whose POSIX half is
:mod:`termhub.pty_posix` and whose Windows half is :mod:`termhub.pty_conpty`
(a ConPTY pseudo-console). The halves differ in one structural way the session
branches on through the seam — :func:`termhub.drivers.pair_starts_the_child`:
a kernel PTY hands over a slave fd for this class to spawn onto, a ConPTY *is*
the child's stdio and starts the shell itself.

The two rules inherited from AGENTS.md:

* the shell's tree has one named boundary: on POSIX the shell leads its own
  process group (``start_new_session=True``), which is what makes the group
  signal reach its children; on Windows the session opens one **Job Object**
  at spawn (:mod:`src.core.proctree_jobs`) which contains the shell, the
  console helper the driver started beside it, and — by inheritance at
  creation — everything those two spawn;
* teardown asks the shared :mod:`src.core.proctree` entry point — the same
  escalation shape both hubs use, in whichever platform half that entry point
  dispatches to — on *every* exit path: client disconnect, ``kill`` frame,
  shell exit, reader ``OSError``, app shutdown. One call, on both platforms,
  guarded by nothing: there is no second kill path and no platform test here.
"""

import asyncio
import codecs
import logging
import subprocess
import time
import uuid
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, Optional

import anyio

from src.config import path_location
from src.core import proctree
from src.core import proctree_jobs

from . import constants
from . import drivers
from . import shells

logger = logging.getLogger("benchmarkit.terminal")

# A frame publisher: the transport's send, injected by the route. Raises when
# the client is gone, which the reader loop treats as "stop pumping".
Emit = Callable[[Dict[str, Any]], Awaitable[None]]
# The next client frame, already parsed to a dict; None means the client left.
ReceiveFrame = Callable[[], Awaitable[Optional[Dict[str, Any]]]]

# Frame types, both directions — the contract between this module and the pane.
INPUT = "input"
RESIZE = "resize"
KILL = "kill"


class TerminalUnsupported(RuntimeError):
    """No pseudo-terminal driver is available (see :mod:`termhub.drivers`).

    The refusal every caller already renders: an ``error`` frame plus close
    code 4501 — never a crash, never a degraded pipe pair.
    """


class SessionCapacity(Exception):
    """The concurrent-session cap is reached — the upgrade is refused."""


class TerminalSession:
    """A shell process, its PTY, and the task pumping PTY bytes to ``emit``."""

    def __init__(self, emit: Emit, session_id: Optional[str] = None) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self._emit = emit
        self.pty: Optional[drivers.TerminalPty] = None
        # The shell process. Normally the ``Popen`` this session spawned onto the
        # driver's slave; with a pair that starts its own child (the Windows
        # pseudo-console — see :func:`termhub.drivers.pair_starts_the_child`) the
        # pair spawned it and answers back with what this class uses of a
        # process: ``pid`` and ``wait()``.
        self.process: Optional[drivers.TerminalChild] = None
        # The session's process-tree boundary on the platform that has no
        # process groups: one Job Object, opened before anything spawns and
        # containing the shell plus the console helper the driver started
        # beside it (C-D5). Inert on POSIX — no handle, every ``assign``
        # answers ``None`` — where the shell's own session is the scope, so
        # the Linux path is exactly what it was.
        self.job: Optional[proctree_jobs.ProcessJob] = None
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
        """Open the driver's terminal pair and start the shell on it.

        The order matters in two places. The window size is set *before* the
        spawn so the first prompt already wraps at the right width (on a ConPTY
        that size is carried to the console's *creation* — C-D4's "set before
        spawn maps exactly"), and the parent's copy of the slave is closed right
        after it: while any process holds the slave open, a read of the master
        never reports the shell's death, so a leftover duplicate would hang the
        reader forever. A ConPTY has no slave, and its driver's ``close_slave``
        is the named no-op that says so.

        With ``start_new_session=True`` the shell becomes a session leader with
        no controlling terminal, and the ``slave`` it is given as its stdio is
        the only terminal it can acquire — so it takes the PTY as its ctty and
        job control (``^C``, job stop/resume, ``stty``) behaves exactly as in a
        real terminal window. On Windows neither idea exists — the console *is*
        the child's stdio and there is no group to signal — so the boundary is
        the **job** opened here, *before* the spawn: containment is decided at
        a process's creation, which is why the shell and the console helper
        are assigned as soon as the driver reports them and no command can
        reach the shell before they are (C-D5).
        """
        driver_module = drivers.selected_driver()
        if driver_module is None:
            raise TerminalUnsupported(constants.NO_DRIVER_MESSAGE)
        # Which shell, and what it is handed: both are the platform's answer, in
        # :mod:`termhub.shells` (C-D2 as amended). A runner with no usable shell
        # raises the same ``TerminalUnsupported`` the driver absence does — one
        # refusal shape, and its sentence names what was looked for.
        shell = shells.select_shell()
        self.cwd = path_location()
        self.shell = shell[0]
        # The tree's boundary exists before anything is started, on every
        # platform: where the platform has no jobs this object is inert and
        # costs nothing (the shell's session is the scope there), and where it
        # does, a process born later inherits the job its parent was already
        # in — so opening it late would be a different guarantee, not a
        # weaker one (see :mod:`src.core.proctree_jobs`).
        self.job = proctree_jobs.ProcessJob(f"terminal-{self.id}")
        driver = None
        try:
            driver = driver_module.open_pair()
            driver.set_size(cols, rows)
            try:
                child_env = shells.child_environment(shell, cols, rows)
                if drivers.pair_starts_the_child(driver):
                    # A pseudo-console *is* the child's stdio — there is no slave
                    # fd to hand a Popen — so the pair spawns the shell itself
                    # and returns what this class uses of a process. Same argv,
                    # same cwd, same environment as the branch below.
                    self.process = driver.spawn_shell(  # type: ignore[attr-defined]
                        list(shell), self.cwd, child_env
                    )
                else:
                    self.process = subprocess.Popen(
                        list(shell),
                        stdin=driver.slave_fd,
                        stdout=driver.slave_fd,
                        stderr=driver.slave_fd,
                        cwd=self.cwd,
                        env=child_env,
                        start_new_session=True,
                    )
                self._contain_spawned(driver)
            except OSError as exc:
                self.process = None
                raise
            finally:
                driver.close_slave()
        except BaseException:
            self._release_job()
            if driver is not None:
                driver.close_master()
            raise
        self.pty = driver

    def _contain_spawned(self, driver: Any) -> None:
        """Put every process the spawn started into the session's job.

        The shell is the **leader** — the pid the shared teardown entry point
        is handed, and the key the job is looked up by. The console helper
        beside it is the process C-D5 names as the one an implementer forgets;
        it is a member, never a key. A pair that starts no helper (a kernel
        PTY) reports no list, and on that platform the job is inert anyway —
        the one call site stays one call site.
        """
        job = self.job
        if job is None:
            return
        pid = self.pid
        if pid is not None:
            job.assign(pid, leader=True)
        started_pids = getattr(driver, "started_pids", None)
        if callable(started_pids):
            for other in started_pids():
                if other != pid:
                    job.assign(other)

    def _release_job(self) -> None:
        """Release the job handle — with kill-on-close, the last guarantee."""
        if self.job is not None:
            self.job.close()

    # -- I/O ------------------------------------------------------------------

    def write(self, data: str) -> None:
        """Keystrokes as xterm encoded them (already control bytes)."""
        if self.pty is None or not data:
            return
        with suppress(OSError):
            self.pty.write(data.encode("utf-8", "surrogateescape"))

    def resize(self, cols: int, rows: int) -> None:
        """Tell the driver the new window size (POSIX: the kernel SIGWINCHes)."""
        if self.pty is None:
            return
        with suppress(OSError):
            self.pty.set_size(cols, rows)

    def terminal_size(self) -> Optional[tuple]:
        """The window size the kernel has for this PTY, as ``(cols, rows)``.

        The server-side answer to "what width does the driver think this
        terminal has". The shell's own ``stty size`` is the client-side view
        of the same number (what the resize test reads).
        """
        if self.pty is None:
            return None
        try:
            return self.pty.get_size()
        except OSError:
            return None

    # -- the pump -------------------------------------------------------------

    async def pump(self) -> None:
        """Stream PTY bytes until the shell is gone, then report its exit code.

        One ``emit`` per driver-read chunk (capped at ``PTY_READ_BYTES``) so a
        ``cat`` of a big file is many frames rather than one giant message.
        The driver read blocks, hence ``asyncio.to_thread``; it is unblocked by the
        teardown, because once the shell and its children are gone the driver
        answers the far end as gone (POSIX: the master ``EIO``s; a ConPTY: the
        pipe EOFs and the released console answers so at once — C-D3, and the
        reason lives in the driver, not here).

        A publish failure is the client leaving mid-stream: stop pumping and
        remember it, so the run loop ends on the same path as a disconnect
        rather than raising out of a task nobody is awaiting.
        """
        driver = self.pty
        if driver is None:  # pump only ever runs after a successful spawn
            return
        while True:
            try:
                chunk = await asyncio.to_thread(
                    driver.read, constants.PTY_READ_BYTES
                )
            except OSError:
                break  # the driver's "far end gone" answer, or a driver failure
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
        timeout only guards against a process that escaped the kill. On a
        pseudo-console the same order holds for the same reason: the driver's
        released console answers the far end as gone, so its read returns to the
        thread without anything being cancelled.
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
        """Kill the process *tree* and release the terminal. Idempotent.

        Runs on every exit path, and it goes through the shared
        ``proctree.kill_session_tree`` so a ``vim`` or a
        ``python -m http.server`` started in the pane cannot outlive the tab.
        The call is **unguarded and platform-neutral** (C-D5): the entry
        point answers with the process **session** here (POSIX) or with the
        **job** this session opened at spawn (Windows — the shell, the console
        helper beside it, and every job bash backgrounded into them). The
        order the POSIX path has always had — kill, drain the reader, release
        the driver's end, release the job — is the order on both platforms.
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
        # hold under cancel. C added a platform, and this stays as true there:
        # the job's graceful step and its force step are what a cancel must not
        # be allowed to abandon halfway.
        with anyio.CancelScope(shield=True):
            pid = self.pid
            # The scope of the kill is the platform half's answer to one call:
            # here the *session*, not the shell's group — an interactive shell
            # gives every job its own process group, so killpg(shell) would
            # leave a backgrounded ``http.server`` alive, and ``pid`` is the
            # session id (start_new_session=True). On Windows that scope is the
            # session's **job**, which spawn-time containment already made
            # whole; a child that escaped the grace outlives the pane exactly
            # as a ``setsid``'d daemon does here, and is reported the same way.
            if pid is not None:
                await proctree.kill_session_tree(
                    pid, self._reap_leader(), constants.STOP_GRACE_SECONDS
                )
            # Drain the reader (it unblocks with the far-end answer now the
            # tree is gone) before touching the driver's end, so the worker
            # thread never reads a closed/reused one.
            await self._finish_pump()
            if self.pty is not None:
                with suppress(OSError):
                    self.pty.close_master()
                self.pty = None
            # The job handle is released last and always: the job path of
            # ``kill_session_tree`` already closed it when it ran (close is
            # idempotent), and this covers the paths where it did not — no
            # pid, or a leader whose assignment failed so the entry point
            # never found the job. Closing it is kill-on-close, the guarantee
            # that outlives the escalation.
            self._release_job()

    def _reap_leader(self):
        """The reap step ``kill_session_tree`` runs before probing the tree."""

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
