"""Static configuration of the Editor tab's terminal surface.

Module attributes on purpose (the ``lsphub.constants`` / ``editorfs.constants``
convention): a test that exercises a cap or a grace period patches it here —
``termhub.constants.MAX_SESSIONS`` — instead of reaching into the code that
reads it.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Concurrent PTY sessions the app accepts. A fifth WebSocket is refused with a
# close code and a one-line reason rather than silently piling up shells: every
# session is a real process tree costing the same as the tab that opened it.
MAX_SESSIONS = 4

# Grace between the group SIGTERM and the SIGKILL escalation of a session
# teardown — the same number and the same rule as both hubs' (AGENTS.md: kill
# the process *tree*, not the wrapper).
STOP_GRACE_SECONDS = 5.0

# One ``os.read`` of the PTY master, so a ``cat`` of a big file arrives as many
# output frames instead of one giant message (the framing rule).
PTY_READ_BYTES = 4096

# Window size a session is born with; the client's first ``resize`` frame
# replaces it (and the kernel's SIGWINCH tells the shell).
DEFAULT_COLS = 80
DEFAULT_ROWS = 24

# The terminal type a full-screen program (vim, less, top) needs to paint.
TERM_VALUE = "xterm-256color"

# D2: the shell is ``$SHELL``, falling back to ``/bin/sh`` — interactive and
# non-login, so an embedded pane reads ``~/.bashrc`` but not ``~/.profile``.
# The selection itself lives in :mod:`termhub.shells` (csl-05), which owns the
# per-platform order and the interactive flag that goes with each shell.
SHELL_KEY = "SHELL"
FALLBACK_SHELL = "/bin/sh"

# The Windows counterpart of ``SHELL``: an explicit shell for the pane, honoured
# only when it is a drive-qualified path to an executable (``termhub.shells``).
# **Process-environment only, never a managed ``.env`` key** — the same
# reasoning as ``BENCHMARKIT_PATH_LOCATION`` and the Git override: a settings
# save must not erase it, and the managed template's key set is not widened for
# a platform detail. ``cmd.exe``/``%COMSPEC%`` is deliberately not read here or
# anywhere: this pane's shell is Git Bash (the operator's amendment of C-D2).
TERMINAL_SHELL_KEY = "TERMINAL_SHELL"

# The seam's patch-point (csl-04): a test that needs the terminal to answer
# "no driver" forces it here — the refusal is a tested behaviour on every
# platform (roadmap §9 rule 3), never a fallback only a Windows box can reach.
# The drivers' own probes (:func:`termhub.pty_posix.available` and its future
# Windows twin) are what answer for real; this flag only hides every answer.
DRIVER_ABSENT_FORCED = False

# WebSocket close codes for the three refusals (4000-4999 is the application
# range). Each mirrors the HTTP status the same guard would answer on a REST
# route: 403 policy, 429 capacity, 501 platform. C makes 4501 mean "no driver
# available", which is still the 501 case.
CLOSE_CODE_POLICY = 4403
CLOSE_CODE_CAPACITY = 4429
CLOSE_CODE_UNSUPPORTED = 4501

# The one sentence for "no driver", said by :meth:`TerminalSession.spawn` and
# by the WebSocket route alike (it is also the close *reason*, which is what
# the pane shows for 4501). It names the missing dependency rather than
# "POSIX-only" — C-D3's fallback wording — and stays inside the 123-byte limit
# a WebSocket close reason has.
NO_DRIVER_MESSAGE = (
    "The terminal has no pseudo-console driver available "
    "(on Windows: install the 'windows' extra, pywinpty)"
)
