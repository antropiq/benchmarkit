"""The Windows driver behind the terminal's pty seam: a real pseudo-console.

C-D1's seam, second half. :mod:`termhub.pty_posix` is a kernel PTY; this module
is a Windows **pseudo-console** (ConPTY), through ``pywinpty`` — the dependency
the ``windows`` extra carries (csl-04) and the layer Jupyter's and VS Code's
terminals sit on. It answers exactly the surface csl-03's *Results* note names —
open-with-size, spawn the shell on it, read bytes, write bytes, set size, get
size, close, and *"is the far end gone?"* — and nothing wider: no policy, no
frame, no WebSocket, and **no kill** (the session's teardown plus
:mod:`src.core.proctree_jobs` own the process tree; csl-07 wires the job).

It is the accepted alternative from C-D1's table and nothing else: not raw
ConPTY through ``ctypes`` (a console-host handshake re-implemented in a module
whose failure mode is a hung terminal), not a pipe pair (no ``isatty``, no
resize, no line discipline — the pane would be a log viewer), not a WSL bridge
(the shell's filesystem would stop being the working location), and not Git's
bundled ``winpty.dll``/``winpty-agent.exe``.

Two members are this platform's answer for a pair that has no slave descriptor
to hand a ``Popen`` — the pseudo-console *is* the child's stdio:

* :attr:`ConPty.starts_child` — the capability :mod:`termhub.drivers` and
  ``session.spawn()`` branch on, so the POSIX half keeps its ``Popen`` exactly
  as it is today. It is a property of the pair, never an ``os.name`` test.
* :meth:`ConPty.spawn_shell` — start the shell on this console, answering with
  the two things the session asks of a process (``pid``, ``wait``).

The rules the session depends on, re-stated for this mechanism:

* **the size travels to the creation** — C-D4: *"the size is passed when the
  pseudo-console is created (so 'set before spawn' maps exactly)"*.
  :func:`open_pair` therefore creates nothing on its own; it records a size, and
  the session's existing order (``set_size`` before the spawn) is what supplies
  it. :meth:`ConPty.set_size` re-issues it on a live console and
  :meth:`ConPty.get_size` answers the size it was created or reflowed with —
  ``pywinpty`` exposes no read-back, so that is the driver's own record, and the
  shell's ``stty size`` stays the client-side view of the same number.
* **bytes, never decoded text**, and never more than the caller asked for.
  ``pywinpty`` answers *strings* of whatever length its agent likes (measured:
  up to 16 384 characters in one read), so the driver chunks to ``max_bytes`` itself
  and buffers the rest. The session's ``PTY_READ_BYTES`` framing and its
  incremental UTF-8 decoder stay the only place bytes become text (C §7).
* **the stream ends at an EOF, not an ``EIO``** — C-D3: *"A ConPTY pipe ends with
  EOF (an empty read), not ``EIO``, and there is no slave fd to close. … the seam
  exposes one predicate — 'is the far end gone?' — and ``pump()`` keeps its shape
  while the reason moves into the driver."* So an empty read whose shell is gone
  raises the seam's one :class:`termhub.pty_posix.FarEndGone`, and a driver
  failure raises :class:`ConPtyError`, an ``OSError`` that is *not* that: the
  session tells a stream that ended from a mechanism that broke. An EOF is not
  proof the shell exited cleanly — :meth:`ConPty.child_exit_status` reports the
  child's status separately, and the exit code the session reaps is the number
  the pane is given.
* **encoding is verified, not assumed** (C-D6). Measured on this machine with a
  Git Bash child: ``echo café中文 ☃`` round-trips identically, because the
  ConPTY buffer is UTF-16 and the agent frames its pipe as UTF-8 — no
  ``chcp``/code-page step was needed, so the driver only converts at the
  boundary, symmetrically (encode UTF-8 on read, decode it on write, both with
  ``surrogateescape``). A child byte that is not valid text in its own locale is
  repaired by the console before it reaches us (a lone ``\\xff`` arrives as
  U+FFFD), which the session's ``errors="replace"`` decoder would do in any case.
* **every process the driver started is nameable** — :meth:`ConPty.started_pids`
  gives the shell *and* the console helper beside it, which is C-D5's *"the one
  an implementer will forget"* and what lets csl-07's job contain both.
  ``pywinpty`` reports only the shell's pid, so the helper's is obtained by
  enumerating this process's children around the creation (:func:`_child_processes`).

**How a read waits**: it never leaves a native call pending — the driver polls
the agent's non-blocking read on an interval (:data:`READ_POLL_INTERVAL_SECONDS`,
a patchable tunable; the same mode ``pywinpty``'s own reader thread uses). This
is not a timeout in ``pump()`` and it adds no grace of its own: a blocking read
inside a worker thread cannot be cancelled by anything the app owns (``pywinpty``
documents it as interruptible only by killing the process), while a poll returns
to its loop every few tens of milliseconds, so a teardown ends the pump instead
of stranding a thread on a live child.

The one honest platform difference, stated rather than papered over (C-D4): a
POSIX program notifies *itself* on ``SIGWINCH``, a Windows console program is
**reflowed by the console**, and a program that reads its size once at startup
behaves differently on the two. ``^C`` is the input event it always was (C-D5):
the byte goes into the console, which turns it into a ``CTRL_C_EVENT`` for the
group — this driver special-cases nothing, and ``^Z``/job stop/resume are not
promised. Nor is the shell a session leader here (``start_new_session`` has no
Windows meaning): the job is the boundary instead.
"""

import ctypes
import errno
import logging
import os
import subprocess
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import constants
from .pty_posix import FarEndGone

if TYPE_CHECKING:  # pragma: no cover - typing only
    import winpty
else:
    # The dependency is optional by design: the ``windows`` extra carries it and a
    # Linux install never resolves it. Importing this module must never be what
    # breaks the app, so a missing module or an unloadable DLL is recorded here and
    # answered by :func:`available` — the same shape as :mod:`termhub.pty_posix`'s
    # runtime guard, and the reason csl-04's refusal is still what the pane shows.
    try:
        import winpty
    except (ImportError, OSError):  # pragma: no cover - dependency not installed
        winpty = None

logger = logging.getLogger("benchmarkit.terminal")


def available() -> bool:
    """Is this driver runnable here? Probes the dependency, never the platform.

    :mod:`termhub.drivers` asks this the same question it asks
    :func:`termhub.pty_posix.available`, and the answer is what imported: the
    pseudo-console exists exactly where ``winpty`` loaded.
    """
    return winpty is not None


# -- tunables: patch them through this module, never a copy of them -----------

# How long the driver sleeps between attempts at the agent's non-blocking read
# while the console has nothing to say. Patchable so a test can make it quick.
READ_POLL_INTERVAL_SECONDS = 0.02

# How long :meth:`ConPTYChild.wait` sleeps between probes of the child. There is
# no blocking wait behind a ConPTY — the agent only answers the current status —
# so the reap is a bounded poll (the same interval ``pywinpty``'s own
# ``PtyProcess.wait`` uses), and the caller bounds it from its own side.
EXIT_POLL_INTERVAL_SECONDS = 0.05

# The exe names the console helper can have. ``pywinpty`` starts the
# pseudo-console's agent as a child of *this* process — ``OpenConsole.exe`` on
# the ConPTY backend, ``winpty-agent.exe`` on the older one — and never reports
# its pid, so :meth:`ConPty.started_pids` finds it by name among the children
# that appeared while the console was created. Names rather than "anything new":
# another component (a hub, a second terminal) that spawns in the same instant
# must not be claimed as this session's helper.
CONSOLE_HELPER_NAMES: Tuple[str, ...] = ("openconsole.exe", "winpty-agent.exe")


class ConPtyError(OSError):
    """The driver itself failed — a console that could not be created, a pipe
    that broke mid-stream.

    An ``OSError`` deliberately **not** a :class:`termhub.pty_posix.FarEndGone`:
    ``pump()`` ends on either, but the session and the log can tell a stream that
    reached its end from a mechanism that broke. A failed spawn raises this too,
    which keeps ``session.spawn()`` on the ``OSError`` path the route already
    renders as an ``error`` frame plus close code 4501 — C §3's *"if … the
    pseudo-console cannot be created"* case, with the refusal's existing shape.
    """


def open_pair() -> "ConPty":
    """The seam's ``open_pair`` for Windows: a pair at the default size.

    Nothing native is created here, because a pseudo-console cannot be made
    without its size (C-D4) and the session's ordering is ``set_size`` *then*
    spawn: the size the session sets on the pair is the size the console is built
    with, inside :meth:`ConPty.spawn_shell`.
    """
    return ConPty(constants.DEFAULT_COLS, constants.DEFAULT_ROWS)


class ConPty:
    """One Windows pseudo-console, answering the seam's surface."""

    #: This pair starts its own child — the console *is* the child's stdio, so
    #: there is no slave fd for the session's ``Popen`` to hand over. The session
    #: branches on this (via :func:`termhub.drivers.pair_starts_the_child`),
    #: never on a platform name.
    starts_child = True

    def __init__(self, cols: int, rows: int) -> None:
        self._cols = cols
        self._rows = rows
        self._pty: Optional["winpty.PTY"] = None
        self._shell_pid: Optional[int] = None
        self._helper_pids: List[int] = []
        self._preexisting_helpers: List[int] = []
        self._pending = b""
        self._closed = False
        self._eof_seen = False

    def __repr__(self) -> str:
        return (
            f"<ConPty {self._cols}x{self._rows} "
            f"shell={self._shell_pid} helpers={list(self._helper_pids)} "
            f"{'closed' if self._closed else 'open'}>"
        )

    # -- size ---------------------------------------------------------------

    def set_size(self, cols: int, rows: int) -> None:
        """Record the window size; reflow the console once it exists.

        Before :meth:`spawn_shell` this only moves the number the pseudo-console
        will be **created** with — that is the whole of C-D4's "set before spawn"
        on this mechanism. After it, it asks the console to resize (which reflows
        its buffer); see the module docstring for how a Windows program learns
        the new size, and for what is *not* promised.
        """
        if cols <= 0 or rows <= 0:
            raise ValueError(f"the console size must be positive, got {cols}x{rows}")
        self._cols, self._rows = cols, rows
        pty = self._pty
        if pty is None:
            return
        try:
            pty.set_size(cols, rows)
        except Exception as exc:
            if not self.child_is_alive():
                return  # nothing left to reflow; the pump ends on the far end
            raise ConPtyError(errno.EIO, f"pseudo-console resize failed: {exc}") from exc

    def get_size(self) -> Tuple[int, int]:
        """The size the console was created or last reflowed to, ``(cols, rows)``.

        The driver's own record: ``pywinpty`` gives no way to ask a live
        pseudo-console its size, so there is nothing to read back. The shell's
        ``stty size`` stays the client-side view of the same number.
        """
        return (self._cols, self._rows)

    # -- the child ----------------------------------------------------------

    def spawn_shell(
        self, argv: Sequence[str], cwd: str, env: Mapping[str, str]
    ) -> "ConPTYChild":
        """Create the pseudo-console at the recorded size and start ``argv`` on it.

        The same argv and environment dict the POSIX branch would hand a
        ``Popen`` — framing them for the agent (a command line, a NUL-split env
        block) is this driver's business, not the session's. A failure raises
        :class:`ConPtyError` (an ``OSError``, which keeps the session's existing
        failure path — and the route's 4501 refusal — one shape for both
        platforms), and a console that was created before the failure is released
        here, so a half-started session never strands its helper: the session's
        failure path calls :meth:`close_master` too, and that is idempotent.
        """
        if winpty is None:  # pragma: no cover - the seam never selects this here
            raise ConPtyError(errno.ENODEV, "the pseudo-console dependency is not importable")
        if not argv:
            raise ValueError("spawn_shell needs a non-empty argv")
        program = str(argv[0])
        cmdline = subprocess.list2cmdline([str(part) for part in argv[1:]]) or None
        # The helper agent is born with the console, as a child of *this*
        # process, and the dependency never reports its pid — so the children
        # that appear around the creation are caught and named (C-D5).
        self._preexisting_helpers = self._helper_snapshot()
        try:
            pty = winpty.PTY(self._cols, self._rows)
        except Exception as exc:
            raise ConPtyError(
                errno.ENODEV, f"could not create the pseudo-console: {exc}"
            ) from exc
        self._pty = pty
        try:
            self._note_helpers()
            try:
                started = pty.spawn(
                    program, cmdline=cmdline, cwd=cwd, env=_env_block(env)
                )
            except Exception as exc:
                raise ConPtyError(
                    errno.ENOENT,
                    f"could not start {program} on the pseudo-console: {exc}",
                ) from exc
            if not started:
                raise ConPtyError(
                    errno.ENOENT, f"{program} did not start on the pseudo-console"
                )
            self._note_helpers()
            pid = pty.pid
            if pid is None:  # pragma: no cover - the agent reports the child it started
                raise ConPtyError(errno.EPROTO, "the pseudo-console reported no shell pid")
        except BaseException:
            # Nothing runs on this console: release it — and with it the helper
            # that was born beside it — rather than leave both to the collector.
            self.close_master()
            raise
        self._shell_pid = int(pid)
        if not self._helper_pids:
            # Not fatal — the shell is still there — but a helper nobody can
            # name is a process csl-07's job cannot contain, so it is said.
            logger.warning("terminal: the console helper of this pseudo-console was not identified")
        return ConPTYChild(self, self._shell_pid)

    def started_pids(self) -> Tuple[int, ...]:
        """Every process this driver started — the shell, then the console helper.

        csl-07's job contains exactly this list and everything they spawn;
        without the helper the pane's tree guarantee would be a wrapper kill.
        """
        pids = [] if self._shell_pid is None else [self._shell_pid]
        pids.extend(self._helper_pids)
        return tuple(pids)

    # -- I/O ----------------------------------------------------------------

    def read(self, max_bytes: int) -> bytes:
        """At most ``max_bytes`` of the console's output — bytes, never text.

        Waits (in the caller's worker thread) by polling the agent's
        non-blocking read; ends on the seam's predicate. The agent answers whole
        strings of its own choosing, far larger than any cap, so the chunking
        happens here and the remainder is kept for the next call: the session's
        framing stays the only framing the pane sees. Trailing output is drained
        before the far-end answer, so a shell's last line is never lost.
        """
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        # The framing ceiling is the rule, not the caller's courtesy: a caller
        # asking for more still never gets more than ``PTY_READ_BYTES``.
        cap = min(max_bytes, constants.PTY_READ_BYTES)
        while True:
            if self._pending:
                chunk, self._pending = self._pending[:cap], self._pending[cap:]
                return chunk
            text = self._read_once()
            if text:
                raw = text.encode("utf-8", "surrogateescape")
                self._pending = raw[cap:]
                return raw[:cap]
            if self._far_end_gone():
                raise FarEndGone(errno.EPIPE, "the pseudo-console reached EOF")
            time.sleep(READ_POLL_INTERVAL_SECONDS)

    def write(self, data: bytes) -> None:
        """Plain bytes in, exactly as the session's framing hands them.

        The console takes characters, so this decodes UTF-8 — :meth:`read`'s
        conversion reversed, ``surrogateescape`` both ways. Ctrl-C is byte 3
        like any other byte: the console turns it into the input event it is
        (C-D5); nothing here special-cases it.
        """
        pty = self._pty
        if pty is None:
            raise ConPtyError(errno.EPIPE, "the pseudo-console is closed")
        try:
            pty.write(data.decode("utf-8", "surrogateescape"))
        except Exception as exc:
            raise ConPtyError(errno.EPIPE, f"pseudo-console write failed: {exc}") from exc

    # -- the closes ----------------------------------------------------------

    @property
    def slave_fd(self) -> int:
        """There is no slave on this mechanism — C-D3, asked of it anyway.

        Reaching this means a caller branched wrong; it fails loudly instead of
        handing out a made-up descriptor.
        """
        raise ConPtyError(errno.ENOTSUP, "a pseudo-console has no slave descriptor")

    def close_slave(self) -> None:
        """The slave-close rule has no Windows counterpart: a named no-op.

        *"while any process holds the slave open, a read of the master never
        reports the shell's death"* — there is no slave to hold, and
        ``session.spawn()``'s ``finally`` stays one shape for both halves.
        """

    def close_master(self) -> None:
        """Release the pseudo-console. Idempotent, and **never a kill**.

        The console has no descriptor to close: releasing it is letting the last
        reference to the native object go, which also unblocks a polling
        :meth:`read` (its next pass sees no console and answers the far-end
        predicate; measured at 2–17 ms for a read blocked on a silent live
        console). Measured on this machine, the release also left both processes
        it started gone — helper, shell, and a backgrounded child of the shell
        that had been running under it. That is the dependency's and the
        console's own
        teardown, observed once, **not** a guarantee this driver holds: it makes
        no kill call, and a promise about a tree is exactly the promise
        :mod:`src.core.proctree_jobs` exists to make and csl-07 wires. Until
        then, a session's processes have one named owner — this module reports
        them, the teardown ends them.
        """
        if self._closed:
            return
        self._closed = True
        pty, self._pty = self._pty, None
        if pty is None:
            return
        with suppress(Exception):
            pty.cancel_io()
        # Dropping the last strong reference here is the release; a pending read
        # must not keep the native object alive past this method.
        del pty

    # -- the driver's own probes ---------------------------------------------

    def child_is_alive(self) -> bool:
        """Is the shell still running? ``False`` once the console is released.

        A probe the mechanism cannot answer counts as gone — the only way it
        fails is that the console (and with it the session's reason to ask) is
        broken, and a pump must end on a broken console, not spin on one.
        """
        pty = self._pty
        if pty is None or self._shell_pid is None:
            return False
        try:
            return bool(pty.isalive())
        except Exception as exc:
            logger.debug("pseudo-console liveness probe failed: %s", exc)
            return False

    def child_exit_status(self) -> Optional[int]:
        """The shell's exit code, or ``None`` while it has none to give.

        The child's status *separately* from the stream's end — which is what
        keeps the exit number honest when an EOF alone proves nothing.
        """
        pty = self._pty
        if pty is None:
            return None
        try:
            status = pty.get_exitstatus()
        except Exception as exc:  # pragma: no cover - the console is going away
            logger.debug("pseudo-console status probe failed: %s", exc)
            return None
        return None if status is None else int(status)

    def _read_once(self) -> str:
        """One attempt at the agent's output; ``""`` when the console is idle.

        A failure while the shell still runs is a :class:`ConPtyError`, never a
        silent idle: a mechanism that broke must not reach the pane as a clean
        end of stream.
        """
        pty = self._pty
        if pty is None:
            return ""
        try:
            return pty.read(blocking=False) or ""
        except Exception as exc:
            if self.child_is_alive():
                raise ConPtyError(
                    errno.EIO, f"pseudo-console read failed: {exc}"
                ) from exc
            self._eof_seen = True
            return ""

    def _far_end_gone(self) -> bool:
        """The seam's question — *"is the far end gone?"* — on this mechanism.

        Either answer counts: the agent reports the end of the stream (measured:
        it can lag the empty read by a moment after an ``exit``), or the shell
        behind the console is no longer running. The second is the firm one, and
        a released console answers gone at once — which is what unblocks the
        pump when the session's teardown closes the master.
        """
        if self._eof_seen or self._pty is None:
            return True
        try:
            if self._pty.iseof():
                self._eof_seen = True
                return True
        except Exception:  # pragma: no cover - the agent is unreachable
            pass
        return not self.child_is_alive()

    def _helper_snapshot(self) -> List[int]:
        """The pids of console-helper-named children of this process, now."""
        return sorted(
            pid
            for pid, name in _child_processes()
            if name in CONSOLE_HELPER_NAMES
        )

    def _note_helpers(self) -> None:
        """Add the console helpers that appeared since before the console was made.

        Only *new* ones: a second live session's helper is in the snapshot too,
        and claiming it would put another session's process into this one's job.
        """
        found = [pid for pid in self._helper_snapshot() if pid not in self._preexisting_helpers]
        for pid in found:
            if pid not in self._helper_pids:
                self._helper_pids.append(pid)


class ConPTYChild:
    """The shell on the pseudo-console, as the two things the session asks of it.

    ``session.py`` uses ``pid`` and ``wait()`` of its child and nothing else, so
    this answers those — through the *pair's* probes, never a strong reference to
    the native console: holding one would keep the console (and its helper) alive
    past :meth:`ConPty.close_master`. :meth:`exit_status` and :meth:`is_alive`
    are the driver's separate report of the child, for when an EOF alone is not
    evidence of how it left.
    """

    def __init__(self, pair: "ConPty", pid: int) -> None:
        self._pair = pair
        self._pid = pid

    def __repr__(self) -> str:
        return f"<ConPTYChild pid={self._pid}>"

    @property
    def pid(self) -> int:
        return self._pid

    def is_alive(self) -> bool:
        return self._pair.child_is_alive()

    def exit_status(self) -> Optional[int]:
        return self._pair.child_exit_status()

    def wait(self) -> int:
        """Block until the shell is gone, and answer its exit code.

        A poll, not a wait — behind a pseudo-console the agent only answers the
        current status (there is no blocking ``waitpid`` twin). It runs in the
        worker thread the session puts it in, and the caller bounds its own wait
        exactly as it bounds a ``Popen.wait``. If that bound cancels the task
        without stopping this thread, the poll still resolves: releasing the
        console (which the teardown always does, after the drain) makes the
        pair's liveness probe answer gone, so no thread outlives the session. A
        child that left with no status recorded answers
        :data:`EXIT_CODE_UNKNOWN`, a stated number rather than a 0 that would
        read as a clean exit.
        """
        while self._pair.child_is_alive():
            time.sleep(EXIT_POLL_INTERVAL_SECONDS)
        status = self._pair.child_exit_status()
        return EXIT_CODE_UNKNOWN if status is None else status


# A child that is gone with no status the agent will name. Reported, not guessed.
EXIT_CODE_UNKNOWN = -1


# -- framing for the dependency ------------------------------------------------


def _env_block(env: Mapping[str, str]) -> str:
    """``env`` as the agent wants it: ``NAME=VALUE`` pairs split by NULs.

    The same dict :func:`termhub.shells.child_environment` builds for a ``Popen``
    (``TERM``, ``COLUMNS``/``LINES``, Git's ``PATH`` order); this only frames it.
    A NUL cannot appear in an environment value, so nothing is lost.
    """
    return "\0".join(f"{name}={value}" for name, value in env.items()) + "\0"


def _child_processes() -> List[Tuple[int, str]]:
    """``(pid, exe name lowercased)`` for every child of this process.

    The only handle on the console helper: the dependency starts it as a child
    of the app and reports no pid, and it is exactly the process an implementer
    forgets (C-D5). The standard snapshot calls, in their wide forms (a name read
    as bytes would not compare, and every call prototyped because ctypes' default
    ``int`` return truncates a 64-bit handle — the ``proctree_jobs`` rule); empty
    where the API is not there, which is a runner that never selects this driver.

    Patchable as a whole, which is what makes the helper-capture rules testable
    on any runner.
    """
    api = _snapshot_calls()
    if api is None:  # pragma: no cover - non-Windows runner
        return []
    snapshot = api["create_snapshot"](_TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == _INVALID_SNAPSHOT:
        return []
    found: List[Tuple[int, str]] = []
    entry = _ProcessEntry32W()
    entry.dwSize = ctypes.sizeof(_ProcessEntry32W)
    me = os.getpid()
    try:
        ok = api["first"](snapshot, ctypes.byref(entry))
        while ok:
            if int(entry.th32ParentProcessID) == me:
                found.append((int(entry.th32ProcessID), entry.szExeFile.lower()))
            ok = api["next"](snapshot, ctypes.byref(entry))
    finally:
        api["close_handle"](snapshot)
    return found


# The snapshot surface, declared from ``ctypes`` primitives rather than
# ``ctypes.wintypes`` so this module imports on any runner (the precedent is
# :mod:`src.core.proctree_jobs`): ``_ProcessEntry32W`` is ``PROCESSENTRY32W``,
# and the ``W`` forms are what is called. Bound on first use rather than at
# import, so importing this module on any runner touches no DLL.
_TH32CS_SNAPPROCESS = 0x2
_INVALID_SNAPSHOT = ctypes.c_void_p(-1).value


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_int32),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),  # ``MAX_PATH``
    ]


_snapshot_api: Optional[Dict[str, Any]] = None


def _snapshot_calls() -> Optional[Dict[str, Any]]:
    """The three snapshot calls, prototyped once, or ``None`` off Windows."""
    global _snapshot_api
    if _snapshot_api is not None:
        return _snapshot_api
    if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows runner
        return None
    kernel32 = ctypes.windll.kernel32
    handle = ctypes.c_void_p
    try:
        create = kernel32.CreateToolhelp32Snapshot
        create.restype = handle
        create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        first = kernel32.Process32FirstW
        first.restype = ctypes.c_int
        first.argtypes = [handle, ctypes.POINTER(_ProcessEntry32W)]
        nxt = kernel32.Process32NextW
        nxt.restype = ctypes.c_int
        nxt.argtypes = [handle, ctypes.POINTER(_ProcessEntry32W)]
        close = kernel32.CloseHandle
        close.restype = ctypes.c_int
        close.argtypes = [handle]
    except (AttributeError, OSError):  # pragma: no cover - kernel32 lacks these
        logger.warning("terminal: the process snapshot calls are unavailable; "
                       "the console helper of a session cannot be named")
        return None
    _snapshot_api = {
        "create_snapshot": create,
        "first": first,
        "next": nxt,
        "close_handle": close,
    }
    return _snapshot_api
