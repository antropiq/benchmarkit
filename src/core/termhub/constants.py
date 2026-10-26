"""Static configuration of the Editor tab's terminal surface (ed-06).

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
# output frames instead of one giant message (ed-06's framing rule).
PTY_READ_BYTES = 4096

# Window size a session is born with; the client's first ``resize`` frame
# replaces it (and the kernel's SIGWINCH tells the shell).
DEFAULT_COLS = 80
DEFAULT_ROWS = 24

# The terminal type a full-screen program (vim, less, top) needs to paint.
TERM_VALUE = "xterm-256color"

# D2: the shell is ``$SHELL``, falling back to ``/bin/sh`` — interactive and
# non-login, so an embedded pane reads ``~/.bashrc`` but not ``~/.profile``.
SHELL_KEY = "SHELL"
FALLBACK_SHELL = "/bin/sh"

# WebSocket close codes for the three refusals (4000-4999 is the application
# range). Each mirrors the HTTP status the same guard would answer on a REST
# route: 403 policy, 429 capacity, 501 platform.
CLOSE_CODE_POLICY = 4403
CLOSE_CODE_CAPACITY = 4429
CLOSE_CODE_UNSUPPORTED = 4501
