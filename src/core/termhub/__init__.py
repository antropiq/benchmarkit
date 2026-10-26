"""Terminal surface of the Editor tab: one PTY per WebSocket connection.

Core-only (no FastAPI), a sibling of ``mcphub/``, ``lsphub/`` and
``editorfs/`` with the same one-concern-per-file layout:

* :mod:`termhub.constants` the session cap, the grace period, PTY framing sizes
* :mod:`termhub.session` ``TerminalSession`` — the pty, the shell, the pump, the
  tree teardown
* :mod:`termhub.manager` ``TerminalHub``: the ``app.state.terminal`` registry,
  the cap decision, ``stop_all`` on shutdown

A session is a **human** surface, reached only by the SPA over
``WS /api/terminal``: it is never an MCP or LSP tool, and that boundary is what
keeps "tools are read-only by design" intact while a real shell exists in the
app. Its ``cwd`` is always ``path_location()`` and the server takes no path from
the client, so the tree/editor path policy has nothing here to guard.

Like the two hubs, a session leads its own process group and is torn down as a
**tree** (via :func:`src.core.proctree.kill_session_tree`) on every exit path.
"""

from typing import TYPE_CHECKING, Any

from . import constants
from .manager import TerminalHub
from .session import (
    SessionCapacity,
    TerminalSession,
    TerminalUnsupported,
    shell_command,
)

_MISSING = object()


def __getattr__(name: str) -> Any:  # PEP 562
    """Resolve a constant from :mod:`termhub.constants` at lookup time.

    Same rule as the hubs and ``editorfs``: the caps and sizes are module
    attributes precisely so a test can patch them, so they must not be bound
    into this module's namespace (a copy would go stale under the patch).
    """
    value = getattr(constants, name, _MISSING)
    if value is not _MISSING:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from .constants import (
        CLOSE_CODE_CAPACITY,
        CLOSE_CODE_POLICY,
        CLOSE_CODE_UNSUPPORTED,
        DEFAULT_COLS,
        DEFAULT_ROWS,
        FALLBACK_SHELL,
        MAX_SESSIONS,
        PTY_READ_BYTES,
        SHELL_KEY,
        STOP_GRACE_SECONDS,
        TERM_VALUE,
    )

__all__ = [
    "SessionCapacity",
    "TerminalHub",
    "TerminalSession",
    "TerminalUnsupported",
    "shell_command",
]
