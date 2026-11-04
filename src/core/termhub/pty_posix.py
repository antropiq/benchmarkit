"""The POSIX driver behind the terminal's pty seam: kernel mechanics, nothing else.

C-D1's seam in one sentence: ``session.py`` keeps every session guarantee —
the frame contract, the ``PTY_READ_BYTES`` chunk framing, the incremental
UTF-8 decoder, the session cap, the tree teardown — and the pseudo-terminal
itself sits behind one small named surface. This module is that surface for
POSIX:

* :func:`open_pair` — the terminal pair (today's ``pty.openpty()``);
* :meth:`PosixPty.set_size` / :meth:`PosixPty.get_size` — the window-size
  ioctls (``TIOCSWINSZ`` / ``TIOCGWINSZ``);
* :meth:`PosixPty.read` / :meth:`PosixPty.write` — byte-level I/O on the master;
* :meth:`PosixPty.close_slave` — the slave-close rule, with its why;
* :class:`FarEndGone` — the ``EIO`` translation, the seam's answer to
  *"is the far end gone?"*;
* :meth:`PosixPty.close_master` — releasing the master (the rule about *when*
  it is safe to close stays with the session, which owns the reader thread).

:func:`available` is this driver's answer to the seam's selection question
(csl-04): the machinery below is there exactly when it imported.

It is a mechanics layer only: it decides no policy, owns no process, spawns no
child and knows nothing about WebSockets or frames. ``read`` answers bytes —
never already-decoded text — so the session's framing and decoder stay the
single source of truth every platform's driver shares.
"""

import errno
import os
import struct
from typing import TYPE_CHECKING

# POSIX-only (the pty itself is): the runtime guard lets the module import on a
# non-POSIX box so the route can answer "unsupported" instead of crashing on
# import; ``TYPE_CHECKING`` lets pyright still see the real modules (the fallback
# binds them to ``None``, which would type every call below as ``None.foo``).
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


def available() -> bool:
    """Is this driver runnable here? Probes the machinery, never the platform.

    The runtime import block above already answers the question — the three
    modules exist exactly on POSIX — so the probe is that verdict, not an
    ``os.name`` test (roadmap §9 rule 2). ``termhub.drivers`` asks it to
    decide whether a session may spawn at all.
    """
    return pty is not None and fcntl is not None and termios is not None


class FarEndGone(OSError):
    """The seam's answer to *"is the far end gone?"*.

    :meth:`PosixPty.read` raises it for the ``EIO`` the master answers once
    every holder of the slave is gone — the fact ``session.py`` relies on
    twice: it is how the pump learns the shell exited on its own, and how a
    read blocked in a worker thread is unblocked by the teardown ("it is
    unblocked by the teardown, because once the shell and its children are
    gone the master answers ``EIO``").

    An ``OSError`` subclass on purpose: an unexpected read error stays the
    error it was and keeps the pump's single ``except OSError`` path, and a
    driver for another platform answers the same question by raising this
    same exception.
    """


class PosixPty:
    """One terminal pair as two descriptors: the master to pump, the slave to
    hand the child. Every method is exactly one syscall on one fd; the fds are
    closed once, by the session, in the order the session's rules demand.
    """

    def __init__(self, master_fd: int, slave_fd: int) -> None:
        self._master_fd = master_fd
        self._slave_fd = slave_fd

    @property
    def slave_fd(self) -> int:
        """The end to hand to the child as its stdio (the spawn's business)."""
        return self._slave_fd

    def set_size(self, cols: int, rows: int) -> None:
        # (rows, cols) is the struct's order — the kernel's ws_row/ws_col — and
        # the ioctl is what makes the kernel SIGWINCH the foreground job.
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)

    def get_size(self) -> tuple[int, int]:
        """The window size the kernel has for this PTY, as ``(cols, rows)``.

        ``TIOCGWINSZ`` — the server-side answer to "what width does the kernel
        think this PTY has". The shell's own ``stty size`` is the client-side
        view of the same number.
        """
        rows, cols, _, _ = struct.unpack(
            "HHHH",
            fcntl.ioctl(self._master_fd, termios.TIOCGWINSZ, b"\0" * 8),
        )
        return (cols, rows)

    def read(self, max_bytes: int) -> bytes:
        """One bounded read of the master — bytes, never decoded text.

        The cap and the decode belong to the session's framing rule
        (``PTY_READ_BYTES``, the incremental decoder); the driver only moves
        bytes, and translates.
        """
        try:
            return os.read(self._master_fd, max_bytes)
        except OSError as exc:
            if exc.errno == errno.EIO:
                raise FarEndGone(exc.errno, exc.strerror) from exc
            raise  # any other error stays the error it was

    def write(self, data: bytes) -> None:
        os.write(self._master_fd, data)

    def close_slave(self) -> None:
        """Close *this process's* copy of the slave, right after the spawn.

        The slave-close rule: while any process holds the slave open, a read
        of the master never reports the shell's death, so a leftover duplicate
        would hang the reader forever. It belongs in the spawning thread —
        that is the process holding the copy — and right after the spawn,
        because the descriptor only needs to stay open long enough to become
        the child's stdio.
        """
        os.close(self._slave_fd)

    def close_master(self) -> None:
        os.close(self._master_fd)


def open_pair() -> PosixPty:
    """The kernel's terminal pair, master and slave.

    Setting the window size is the caller's, *before* it spawns the child — so
    the first prompt already wraps at the right width. That ordering is session
    policy, so it lives in ``session.py``.
    """
    master_fd, slave_fd = pty.openpty()
    return PosixPty(master_fd, slave_fd)
