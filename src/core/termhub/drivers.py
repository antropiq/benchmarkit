"""The terminal driver seam: which driver can this app actually run right now?

The session's pseudo-terminal sits behind one small named surface (C-D1).
:mod:`termhub.pty_posix` is one half of it (landed by csl-03, behaviour-
identical to the inline PTY it replaced) and :mod:`termhub.pty_conpty` the
other (csl-06, the Windows pseudo-console, behind the ``windows`` extra). This
module is the **selection** question in front of both —
*"is a driver available?"* — and nothing else: no policy, no process, no
frame, and it decides no platform by name.

Roadmap §9 rule 2: *"Platform differences are selected by capability probes,
not by ``sys.platform``, wherever a probe is possible … Where it is not
possible (job objects, ConPTY), the difference sits behind one named seam,
and the other platform's tests keep running unchanged."* That is what this
is: the answer comes from trying what a driver needs — ``pty_posix`` is a
driver exactly while its POSIX machinery imported, ``pty_conpty`` exactly
while its module *and* its dependency import — never from ``os.name`` or
``sys.platform``. On Linux the answer is ``pty_posix``; on a Windows box
without the console driver installed the answer is "none", and the sentence
the refusal says is :data:`termhub.constants.NO_DRIVER_MESSAGE` — written
once, naming what was looked for.

One branch is answered here too, because it is a property of the *pair* and not
of a platform's name: :func:`pair_starts_the_child` — does the selected driver's
pair spawn the shell on itself (a pseudo-console *is* the child's stdio; there is
no slave fd to hand a ``Popen``), or hand a slave over and let the session spawn?
That keeps ``session.spawn()`` one code path that branches on a capability.

:data:`termhub.constants.DRIVER_ABSENT_FORCED` is the seam's patch-point: a
test forces the driver-absent answer through it, so the refusal is a *tested*
behaviour on every platform (roadmap §9 rule 3), not a fallback that only
exists where nobody can check it.
"""

import importlib
from typing import Optional, Protocol

from . import constants, pty_posix

# The Windows driver's module name. Naming it here is a lookup, not an
# implementation: while that module or its dependency fails to import, the seam
# honestly answers "no driver" — the state of any install without the
# ``windows`` extra.
WINDOWS_DRIVER_MODULE = "pty_conpty"


class TerminalPty(Protocol):
    """The pair a driver hands the session — csl-03's landed surface.

    The driver answers byte ``read``/``write``, ``set_size``/``get_size``,
    the two closes, and *"is the far end gone?"* (its ``read`` raises the
    driver's ``FarEndGone``). The *ordering* rules — size before the spawn,
    the slave closed right after it, the master only when no read is blocked
    — stay the session's, because the session owns the child and the reader
    thread.
    """

    @property
    def slave_fd(self) -> int: ...

    def set_size(self, cols: int, rows: int) -> None: ...

    def get_size(self) -> tuple[int, int]: ...

    def read(self, max_bytes: int) -> bytes: ...

    def write(self, data: bytes) -> None: ...

    def close_slave(self) -> None: ...

    def close_master(self) -> None: ...


class TerminalChild(Protocol):
    """The child of a pair that starts it itself — what the session asks of a
    ``Popen`` and nothing more: the pid to report and teardown, and a blocking
    ``wait()`` that answers the exit code.
    """

    @property
    def pid(self) -> int: ...

    def wait(self) -> int: ...


def pair_starts_the_child(pair: TerminalPty) -> bool:
    """Does this driver's pair spawn the shell on itself, instead of handing the
    session a slave descriptor to spawn onto? A capability of the pair — the
    pseudo-console *is* the child's stdio — never an ``os.name`` test.
    """
    return bool(getattr(pair, "starts_child", False))


class TerminalDriver(Protocol):
    """A driver module answering the seam's two questions."""

    def available(self) -> bool: ...

    def open_pair(self) -> TerminalPty: ...


def _posix_driver() -> Optional[TerminalDriver]:
    """``pty_posix`` while its kernel machinery is there — a probe, not a name."""
    return pty_posix if pty_posix.available() else None


def _windows_driver() -> Optional[TerminalDriver]:
    """The console driver while its module *and* its dependency are importable.

    A module that does not answer the seam's availability question is not a
    driver, so this refuses rather than raising: a refusal is the tested
    behaviour, a crash is not (roadmap §9 rule 3).
    """
    try:
        module = importlib.import_module(f".{WINDOWS_DRIVER_MODULE}", __package__)
    except (ImportError, OSError):
        # ImportError: the module is not there yet, or its dependency is not
        # installed. OSError: a dependency that is installed but cannot load
        # (a DLL failure) — both are "no driver", which is the honest answer
        # and the one the pane shows, never a crashed route.
        return None
    probe = getattr(module, "available", None)
    return module if callable(probe) and probe() else None


def selected_driver() -> Optional[TerminalDriver]:
    """The driver this app can run right now, or None. Never ``os.name``."""
    if constants.DRIVER_ABSENT_FORCED:
        return None
    return _posix_driver() or _windows_driver()


def driver_available() -> bool:
    """The seam's yes/no question — the same answer the refusal is built on."""
    return selected_driver() is not None
