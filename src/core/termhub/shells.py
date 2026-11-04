"""Which shell the pane runs, and what the child's environment says (C-D2).

The session asks this module one question at spawn — *what program, with which
argv, in which environment?* — and never decides by itself: the fallback lives
with the platform, which is what keeps C-D2's one knob honest (*"the shell is
the user's choice, expressed the same way on both platforms"*). The answer is
patchable end to end — the candidates, the override name and the selection
platform are module attributes, the environment is an argument — so a test
forces either platform's answer on either runner (the hermetic-tests
convention: never a path a test computed).

**POSIX — today's behaviour verbatim.** ``$SHELL`` (``constants.SHELL_KEY``) when
it names an absolute, executable program, else ``constants.FALLBACK_SHELL``, and
the interactive flag is ``-i``: *"interactive-non-login so ``~/.bashrc`` is read
but not ``~/.profile``"*. Nothing here improves that; the Linux pane is the
reference this lot must not move.

**Windows — Git Bash, per the operator's amendment of this lot (2026-11-02).**
C-D2 originally read *"a configured ``TERMINAL_SHELL`` if set and executable →
``%COMSPEC%`` (``cmd.exe``) → refuse"*, and explicitly refused to auto-pick a
shell (*"the app should not decide a user's shell"*). The operator amended the
middle step out: the order is

1. ``TERMINAL_SHELL`` (``constants.TERMINAL_SHELL_KEY``) — accepted only when it
   is **drive-qualified** and executable;
2. **Git Bash** — the install the console connector's probe already finds
   (:func:`mcp_connectors.gitbash.discover_git_install`, then
   :func:`mcp_connectors.gitbash.git_shell_program`);
3. else **refuse**, with a sentence naming what was looked for.

``cmd.exe``/``%COMSPEC%`` is **not consulted at all** — not as a fallback and not
as a candidate — and it stays that way deliberately: the app's console story on
Windows is already Git's toolset (target B1), and a ``cmd.exe`` pane would give
one working location two different tool vocabularies. ``pwsh``/``powershell`` are
still never auto-selected, which is C-D2's rejection standing unchanged. The rest
of C-D2 survives the amendment intact: the refusal keeps its shape, the seam
still owns shell selection, and ``TERM`` is still set for the child.

**A POSIX-shaped spec is never a Windows shell.** ``TERMINAL_SHELL=/bin/bash``
(or ``/c/Program Files/Git/bin/bash.exe``) is what the environment holds when the
app itself was launched *from* a Git Bash window, and on Windows
``os.path.isabs("/bin/bash")`` answering ``False`` today is luck rather than a
rule. So the accepted shape is stated as **drive-qualified**, asked of
:func:`mcp_connectors.path_norm.is_qualified` — the module that answers what a
path text *is*, on either runner — and pinned by a test. A spec that is not
drive-qualified is refused **by name** rather than spawned, and the same rule
stops an MSYS-mount path ever reaching a Windows ``Popen``. The refusal for a
present-but-unusable spec says so instead of falling through to Git Bash: a
silent fallback is how the one knob stops being honest.

The interactive flag belongs to the shell spec, not to ``session.py``, which used
to hard-code the bash-specific ``-i``. Git Bash takes the same flag for the same
reason: interactive-non-login reads ``~/.bashrc`` — the prompt, aliases and
colours a user expects in a pane — and does not start each tab as a login shell,
which would re-run the profile's environment setup once per tab.

*Not here:* the child's ``cwd`` — the session's, always ``path_location()``, and
the server takes no path from the client; the pseudo-terminal itself —
:mod:`termhub.drivers`; any permission question. A shell is reachable only from
the Editor's WebSocket, never from a hub, a tool or a chat turn, and none of this
gives the console connector a shell: it still refuses a command named ``bash``
with its no-shell sentence (G-D3), and finding a shell program here is not
permission to run one there.
"""

import ntpath
import os
from typing import Dict, List, Mapping, Optional, Tuple

# The shared probe lives in ``mcp_connectors/`` because a connector runs in its
# own process and cannot import ``src``; ``src`` may import it (the precedent is
# ``editorfs/fscase.py`` re-exporting ``mcp_connectors/fscase.py``). One probe
# answers "where is Git" for the connector and for the pane, so the two cannot
# start disagreeing about a path. Never fork its logic.
from mcp_connectors import gitbash
from mcp_connectors.path_norm import is_qualified

from . import constants

# The selected shell: the program plus its own interactive flag.
ShellSpec = Tuple[str, ...]

# Where a Windows install of Git may be when the probe finds nothing on PATH —
# the terminal's own patch-point, seeded from the shared probe's list (which the
# console connector built the same way before csl-01 moved it).
GIT_INSTALL_CANDIDATES: List[str] = list(gitbash.GIT_INSTALL_CANDIDATES)

# The override's name, same process-environment-only reasoning as the console
# connector's and ``BENCHMARKIT_PATH_LOCATION``'s: a settings save must not erase
# it, and the managed ``.env`` template's key set is not widened for it.
GIT_ENV_OVERRIDE: str = gitbash.GIT_ENV_OVERRIDE

# Which selection runs. Named and patchable rather than read inline at the call,
# so a test forces the Windows half's answers **on the Linux runner** (roadmap
# §9 rule 2) and the POSIX half's on a Windows one. This module decides the shell
# by what the platform's environment can name; the *driver* seam beside it
# answers the mechanism by capability probe (see :mod:`termhub.drivers`).
SELECTION_PLATFORM: str = os.name


def select_shell(
    platform: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> ShellSpec:
    """The program argv for this runner's shell. Raises when there is none.

    ``platform`` and ``environ`` default to the process's own answers; a test
    hands in the other pair (or patches the candidate lists) and gets that
    platform's decision without pretending to be on it.
    """
    name = SELECTION_PLATFORM if platform is None else platform
    env = os.environ if environ is None else environ

    if name == "posix":
        return _select_posix(env)
    return _select_windows(env)


def child_environment(
    shell: ShellSpec,
    cols: int,
    rows: int,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """The environment for ``shell``: the inherited one plus what a terminal needs.

    ``TERM`` (``constants.TERM_VALUE``, so vim, less and top can paint), and
    ``COLUMNS``/``LINES`` at the size the window was opened with — today's keys,
    unchanged on both platforms.

    The process environment is inherited **verbatim** otherwise — the honest
    trade-off the session has always documented: the child sees
    ``BENCHMARKIT_PATH_LOCATION`` and the ``.env``-derived values because the app
    itself carries them.

    For the Git Bash case, ``PATH`` gets Git's directories **prepended in Windows
    form** (:func:`mcp_connectors.gitbash.git_child_env`): a bash started with
    only the ambient Windows ``PATH`` answers "command not found" for ``stty``,
    ``tput`` and ``which`` (measured by csl-01) — the tools the pane's resize test
    and a user's ``vim``/``less`` depend on. So the search order handed to the
    child is the one the console connector already proved (G-D4).

    ``environ`` defaults to the process environment, exactly as the session
    hands it today; a test passes its own base and sees the same answer.
    """
    base: Dict[str, str] = dict(os.environ if environ is None else environ)
    # The probe decides, not the platform's name: the child gets Git's search
    # order exactly when the selected program *is* a shell inside a verified Git
    # install, which is a question the shared marker answers on either runner.
    root = _git_root_of(shell[0])
    if root is not None:
        base = gitbash.git_child_env(base, root)
    base["TERM"] = constants.TERM_VALUE
    base["COLUMNS"] = str(cols)
    base["LINES"] = str(rows)
    return base


# -- the POSIX answer (unchanged behaviour, moved out of session.py) ----------


def _select_posix(env: Mapping[str, str]) -> ShellSpec:
    """``$SHELL`` when it is an absolute executable, else ``/bin/sh``.

    Verbatim what ``session.shell_command()`` did, including the interactive
    flag: a non-login ``-i`` so an embedded pane reads ``~/.bashrc`` — the prompt,
    aliases and colours a user expects there — without paying ``~/.profile`` side
    effects once per tab.
    """
    configured = (env.get(constants.SHELL_KEY) or "").strip()
    if configured and os.path.isabs(configured) and os.access(configured, os.X_OK):
        return (configured, "-i")
    return (constants.FALLBACK_SHELL, "-i")


# -- the Windows answer (C-D2 as amended: Git Bash, no cmd.exe) ---------------


def _select_windows(env: Mapping[str, str]) -> ShellSpec:
    """``TERMINAL_SHELL`` if drive-qualified and executable, else Git Bash, else refuse."""
    # Reached lazily: :mod:`termhub.session` asks *this* module for the shell and
    # owns the refusal class this module raises, so importing it here would be a
    # cycle — the same shape as ``editorfs.policy``'s lazy import of the error
    # class ``editorfs.policy_file`` needs from it.
    from .session import TerminalUnsupported

    configured = (env.get(constants.TERMINAL_SHELL_KEY) or "").strip().strip('"')
    if configured:
        program, why = _windows_shell_program(configured)
        if program is None:
            raise TerminalUnsupported(
                "the configured shell is not usable on Windows: "
                f"{constants.TERMINAL_SHELL_KEY}='{configured}' {why}. Set it to "
                "a full Windows path (a drive-qualified one such as "
                "C:\\Program Files\\Git\\bin\\bash.exe), or leave it unset to "
                "use Git Bash."
            )
        return (program, "-i")

    root = _discover_git_install(env)
    program = gitbash.git_shell_program(root)
    if program is None:
        raise TerminalUnsupported(_no_shell_refusal(env))
    return (program, "-i")


def _windows_shell_program(spec: str) -> Tuple[Optional[str], str]:
    r"""``spec`` as a Windows shell program, or ``(None, why not)``.

    The accepted shape is stated as **drive-qualified** — a drive spec
    (``C:\…``, ``C:/…``) or a UNC root (``\\srv\share\…``) — rather than left to
    ``os.path.isabs``, which answers ``False`` for ``/bin/bash`` on Windows by
    luck rather than by rule. Two questions, and both are needed:

    * :func:`mcp_connectors.path_norm.is_qualified` — the shared answer for "this
      text cannot be a workspace-relative path", which refuses a bare ``bash``
      and any relative name. It is *not* the drive rule on its own: a leading
      slash is qualified (that is how it closes the ``\etc\passwd`` hole), so
      ``/bin/bash`` passes it.
    * ``ntpath.splitdrive`` on the spec as written — a non-empty drive part is
      what makes it a path a Windows child can be started from. That is what
      refuses ``/bin/bash``, a bare ``bash``, and the MSYS-mount spelling
      ``/c/Program Files/Git/bin/bash.exe`` a Git Bash window puts in the
      environment. Asking it of the raw text, not of the canonical form, because
      the canonical form folds a UNC root's leading separators away.
    """
    if not is_qualified(spec):
        return (None, "is a relative name, not a path")
    if not ntpath.splitdrive(spec)[0]:
        return (
            None,
            "names no Windows drive or share, so it is a POSIX-shaped path — "
            "the kind a Git Bash window carries, not a program this app can "
            "start as a Windows child (use a drive-qualified one such as "
            "C:\\Program Files\\Git\\bin\\bash.exe)",
        )
    if not _program_is_runnable(spec):
        return (None, "is not an existing executable file")
    return (spec, "")


def _program_is_runnable(spec: str) -> bool:
    """Does ``spec`` name a file this runner could start? Named, so a test can
    hand the selection order a yes for a spec it never plants on disk (the
    qualification rule and the filesystem answer are separate questions, and a
    Linux runner can only fake the second)."""
    return os.path.isfile(spec) and os.access(spec, os.X_OK)


def _discover_git_install(env: Mapping[str, str]) -> Optional[str]:
    """The Git install root, asked of the shared probe with this module's inputs."""
    return gitbash.discover_git_install(GIT_INSTALL_CANDIDATES, GIT_ENV_OVERRIDE, env)


def _git_root_of(program: str) -> Optional[str]:
    r"""The Git root a selected ``…\bin\bash.exe`` belongs to, or None.

    Asked of the program that was selected rather than carried across from the
    selection, so the child environment works off one answer: the root is the
    verified install whose ``bin\bash.exe`` this is (the parent of ``bin``),
    confirmed by the same ``usr\bin\ls.exe`` marker discovery used. A
    ``TERMINAL_SHELL`` pointing somewhere that is not a Git install therefore
    keeps the ambient ``PATH`` untouched rather than borrowing a toolset.
    """
    if not program:
        return None
    directory = os.path.dirname(os.path.abspath(program))
    if os.path.basename(directory).lower() != "bin":
        return None
    root = os.path.dirname(directory)
    return gitbash.has_git_toolset(root)


def _no_shell_refusal(env: Mapping[str, str]) -> str:
    """The sentence for "no shell here", naming what was looked for (win-07's shape).

    It also names the two things that are **not** and will not be tried, so the
    absence is not mistaken for an unfinished lookup: ``%COMSPEC%``/``cmd.exe``
    (one working location, one tool vocabulary) and ``pwsh``/``powershell`` (the
    app does not decide a user's shell).
    """
    searched = ", ".join(GIT_INSTALL_CANDIDATES)
    shell_key = constants.TERMINAL_SHELL_KEY
    return (
        f"the terminal found no shell to run: {shell_key} is not set, `git` was "
        "not on PATH, none of the Git install candidates "
        f"({searched}) carries usr\\bin\\ls.exe, and {GIT_ENV_OVERRIDE} names no "
        "usable install. This pane runs Git Bash; install Git for Windows, point "
        f"{GIT_ENV_OVERRIDE} at its install root, or set {shell_key} to a "
        "drive-qualified shell path. cmd.exe and pwsh/powershell are not "
        "started here."
    )
