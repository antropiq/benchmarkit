"""Suite-wide fixtures and the platform gate (win-01).

The two markers below are how the suite stays a usable signal on a platform
where some of these tests cannot run: a test whose subject the platform will
not provide is *skipped with a sentence naming the invariant it would have
checked*, never weakened. On Linux neither marker ever fires, so the whole
suite still runs there.
"""

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Iterator
from typing import Callable
from typing import cast
from typing import Optional
from typing import Dict
from typing import Tuple

import pytest

from src.core import editorfs
from src.core.job import RunJob
from src.core.termhub import drivers as termhub_drivers
from src.core.termhub import pty_conpty as termhub_conpty
from src.core.termhub import shells as termhub_shells


# --- the platform gate (win-01) ---------------------------------------------
#
# Every platform guard in the suite consults one of the ``_Probe`` objects
# below: a per-file copy of a probe is how a guard drifts from what it guards.
# A probe answers by *trying*, not by the platform's name, where a try is
# possible (symlinks, a shell through the terminal's own selection seam, a
# driver through the terminal's driver seam, ``/proc``); where it is not
# (``/etc/shadow`` exists, ``chmod`` is meaningful), the difference sits behind
# the named seam ``posix_behaviour`` instead. The attempt is lazy --
# ``bool(probe)`` runs it once and caches the verdict -- so collection-only
# commands touch no filesystem.


class _Probe:
    """A platform capability, attempted on first use and kept for the session.

    A probe is truthy while the capability *is* there. pytest calls ``bool()`` on
    a ``skipif`` condition at setup time, which is exactly where the attempt
    belongs, so a probe (or its negation, as :func:`skip_unless` builds one) is
    itself a usable condition. The guard markers below consult their probes the
    same way.
    """

    def __init__(
        self,
        attempt: Callable[[], bool],
        description: str,
        detail: Optional[Callable[[], str]] = None,
    ) -> None:
        self._attempt = attempt
        self.description = description
        # What the attempt *learned* when it failed, for the skip sentence. A
        # capability a seam refuses usually says why, and naming that is what
        # makes "Git is missing" distinguishable from "the driver is missing"
        # in a runner's output -- so the detail travels with the verdict.
        self._detail = detail
        self._verdict: Optional[bool] = None
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        with self._lock:
            if self._verdict is None:
                self._verdict = bool(self._attempt())
            return self._verdict

    def refusal_detail(self) -> str:
        """The failed attempt's own sentence, if it has one (asked only once
        the verdict is no)."""
        return "" if self._detail is None else str(self._detail())

    def clear_verdict(self) -> None:
        """Forget the cached attempt, so the next question asks again."""
        with self._lock:
            self._verdict = None

    def __repr__(self) -> str:
        # Never answers the probe -- a repr during collection must not touch the
        # filesystem, so the verdict is shown only once it is already known.
        return f"<probe: {self.description} -> {self._verdict}>"


def _symlink_probe_forms(directory: Path) -> None:
    """Create every shape of symlink a test in this suite asks for.

    All three are attempted, because the tests use all three and a platform may
    grant them differently: a plain file link, a *directory* link
    (``target_is_directory=True``), and a ``target_is_directory=True`` link to a
    name that does not exist -- the dangling form the Editor's create, rename
    and delete tests plant. Any one raising means an unprivileged process
    cannot make the link that test needs.
    """
    target_file = directory / "probe-target.txt"
    target_file.write_text("x", encoding="utf-8")
    target_dir = directory / "probe-target-dir"
    target_dir.mkdir()

    os.symlink(target_file, directory / "probe-file")
    os.symlink(target_dir, directory / "probe-dir", target_is_directory=True)
    os.symlink(
        directory / "probe-never-existed",
        directory / "probe-dangling",
        target_is_directory=True,
    )


def _symlinks_are_allowed() -> bool:
    """Try once, in a private temporary directory, to make every symlink shape."""
    try:
        with tempfile.TemporaryDirectory(
            prefix="benchmarkit-symlink-probe-"
        ) as raw:
            _symlink_probe_forms(Path(raw))
    except OSError:
        return False
    return True


symlink_probe = _Probe(_symlinks_are_allowed, "create a symlink")
posix_probe = _Probe(lambda: os.name == "posix", "POSIX process semantics")
windows_probe = _Probe(lambda: os.name == "nt", "Windows process semantics")
proc_fs_probe = _Probe(lambda: Path("/proc").is_dir(), "a readable /proc")


# --- the volume's case answer (tst-01) ----------------------------------------
#
# Several suites pin a *pair* of verdicts by forcing ``fscase.case_folded`` and
# planting the two spellings the pair is about — ``.nojae`` and ``.NOJAE``,
# ``denied-root`` and ``Denied-Root``. That only holds where the volume can hold
# both names at once: NTFS answers one folder for both, so the second
# ``mkdir`` raises and the planted premise is false. The probe is the app's own
# question (plant a cased name, ``stat`` the other spelling, compare
# ``st_dev``/``st_ino`` — the same measurement as ``mcp_connectors/fscase``),
# so a test of a folded pair and a test that skips one cannot disagree about
# this volume. A case-sensitive volume — the Linux runner, any ext4 — keeps
# both spellings, so the marker never fires there and the whole suite runs.


def _volume_folds_case() -> bool:
    with tempfile.TemporaryDirectory(prefix="benchmarkit-case-probe-") as raw:
        planted = Path(raw) / "ProbeName"
        planted.write_text("x", encoding="utf-8")
        try:
            cased = os.stat(planted)
            try:
                other = os.stat(Path(raw) / "probename")
            except OSError:
                return False
            return (cased.st_dev, cased.st_ino) == (other.st_dev, other.st_ino)
        finally:
            planted.unlink(missing_ok=True)


case_folded_probe = _Probe(_volume_folds_case, "a volume that folds name case")


# --- the terminal's capability probes (csl-08) --------------------------------
#
# The pane's own seams answer these questions, so the suite asks the seams
# rather than re-implementing them: ``termhub.shells`` decides *which program*
# the terminal would run (``$SHELL`` → ``/bin/sh`` on POSIX, ``TERMINAL_SHELL``
# → Git Bash on Windows, else a refusal naming what was looked for) and
# ``termhub.drivers`` decides *whether a pseudo-terminal mechanism exists at
# all*. The old probe (``$SHELL`` or ``/bin/sh``, judged with ``shutil.which``)
# answered "no shell" on every Windows machine, which is why a real session
# under test was impossible there. One probe per question, consulted by the
# guards *and* by the tests that want the program itself, so a test that spawns
# a session and a test that skips one can never disagree about this machine.

_selected_shell: Optional[str] = None
_shell_refusal = ""


def _offer_a_shell() -> bool:
    """Ask ``termhub.shells`` once; keep its answer for the tests and the skip."""
    global _selected_shell, _shell_refusal
    try:
        _selected_shell = termhub_shells.select_shell()[0]
    except Exception as exc:
        # Whatever the seam refuses with (no shell on the platform, no Git
        # install), it is the same refusal ``spawn()`` would raise, so this is
        # the pane's own answer rather than a second opinion on it — and its
        # sentence is what a skip gets to quote, so "Git is missing" and "the
        # driver is missing" stay tellable apart in a runner's output.
        _shell_refusal = str(exc)
        return False
    return True


shell_probe = _Probe(
    _offer_a_shell,
    "a shell this machine offers the terminal",
    lambda: _shell_refusal,
)
driver_probe = _Probe(
    termhub_drivers.driver_available,
    "a pseudo-terminal driver (the kernel PTY, or the 'windows' extra's ConPTY)",
)
# The one answer to "is the *Windows* driver installed", so a test class that
# needs a real pseudo-console and the seam's own selection cannot disagree about
# whether this box has one (the same rule as the shell probe above).
windows_driver_probe = _Probe(
    termhub_conpty.available,
    "the Windows pseudo-console driver (pywinpty, the 'windows' extra)",
)


def selected_shell() -> Optional[str]:
    """The program a terminal session would run here, or ``None``."""
    if not shell_probe:
        return None
    return _selected_shell


def _pane_can_run(tool: str) -> bool:
    """Could the *pane's* shell run ``tool``? Asked of the environment the seam
    would hand that child, not of this process's ``PATH``.

    A Git Bash session's child is given Git's own directories first
    (:func:`termhub.shells.child_environment`), so ``stty`` is reachable there
    while ``shutil.which`` in the test process answers nothing — the app itself
    is not launched from Git Bash. The child's answer is the one a terminal
    assertion needs, and asking it through the seam keeps the test and the pane
    from disagreeing about the search path.
    """
    program = selected_shell()
    if program is None:
        return False
    child = termhub_shells.child_environment((program, "-i"), 80, 24)
    for entry in child.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        for suffix in ("", ".exe"):
            if os.path.isfile(os.path.join(entry, tool + suffix)):
                return True
    return False


# C §6's resize assertion comes in two halves: the size the *driver* reports is
# platform-neutral and asserted wherever a session can open; the size the
# *shell* re-reads needs a shell carrying `stty`. Whether it does is an
# environment fact about the program the seam selected, so a probe answers it —
# the terminal twin of ``console_tool`` below, one probe for the one tool the
# suite runs through the pane.
stty_probe = _Probe(
    lambda: _pane_can_run("stty"), "`stty` on the pane shell's own search path"
)


# The marker names, each answering to one probe plus the sentence a bare marker
# skips with. A skipped test must say which invariant it would have checked, so
# a test whose own name does not say it passes its own sentence as the marker's
# argument:
# ``@pytest.mark.requires_symlink("a symlinked lot file must never be read")``.
_GUARD_MARKERS: Dict[str, Tuple[_Probe, str]] = {
    "requires_symlink": (
        symlink_probe,
        "the invariant needs a symlink this platform will not let an "
        "unprivileged process create",
    ),
    "posix_behaviour": (
        posix_probe,
        "the invariant is a POSIX-only fact with no probe that answers it",
    ),
    "windows_behaviour": (
        windows_probe,
        "the invariant is a Windows-only fact (the Windows trees, the Windows "
        "environment values, Git's toolset) with no probe that answers it",
    ),
    # The terminal's capabilities (csl-08), marked rather than skipif-ed so the
    # sentence can carry what the probe *learned* — the seam's own refusal —
    # which a skipif reason, fixed at import time, cannot. A session test names
    # the ones it opens through; a skip then says which capability is absent
    # rather than repeating a sentence that could describe any of them.
    "needs_shell": (
        shell_probe,
        "the terminal's own shell selection (termhub.shells: POSIX $SHELL -> "
        "/bin/sh, Windows TERMINAL_SHELL -> Git Bash) names no program here",
    ),
    "needs_driver": (
        driver_probe,
        "no pseudo-terminal driver answers available (the kernel PTY, or the "
        "'windows' extra's ConPTY), so no session can spawn at all",
    ),
    "needs_stty": (
        stty_probe,
        "the invariant is that the *shell* re-reads the window the server set, "
        "which needs `stty` on the pane shell's own search path",
    ),
    "needs_windows_driver": (
        windows_driver_probe,
        "the invariant can only be measured on a real pseudo-console, which "
        "needs the 'windows' extra installed",
    ),
    # A pair of verdicts obtained by forcing ``fscase.case_folded`` while both
    # spellings sit on disk as two folders. On a volume that folds case they are
    # one folder, so the "not folded" half is no longer the pair's other answer
    # — it describes a name that cannot exist there.
    "case_sensitive_volume": (
        _Probe(
            lambda: not bool(case_folded_probe),
            "a volume that keeps two names differing only in case apart",
        ),
        "the test plants both spellings of one name and asks the fold probe to "
        "answer each way, which only a case-sensitive volume can hold",
    ),
}


def skip_unless(probe: _Probe, reason: str) -> pytest.MarkDecorator:
    """A ``skipif`` for one platform capability a test needs, with its sentence.

    A fact that a named marker does not cover (``/proc`` only, not all of
    POSIX) keeps its own guard, but built on this one helper so the probe and
    the reason live together. ``skipif(not probe)`` would answer the probe at
    import time; the wrapper below defers the verdict to the test's setup.
    """
    absent = _Probe(lambda: not bool(probe), f"absence of {probe.description}")
    # ``cast`` only for the stub: pytest's own skipif signature wants a plain
    # bool, and it evaluates the condition by calling bool() on it.
    return pytest.mark.skipif(cast(bool, absent), reason=reason)


# A few tests can only observe the rule they pin by *running* a program (the
# console connector's decoder, its timeout, its output cap). Whether that
# program is on PATH is an environment fact, so a probe answers it rather than a
# platform name — the same answer on Windows and on a Linux box with a thin
# PATH. One probe per tool, cached, so the tests asking about `printf` attempt
# it once.
_tool_probes: Dict[str, _Probe] = {}


def console_tool(tool: str, reason: str) -> pytest.MarkDecorator:
    """Skip a test whose subject can only be seen through ``tool``.

    ``reason`` names the invariant the test would have checked, as every skip's
    sentence must.
    """
    probe = _tool_probes.get(tool)
    if probe is None:
        probe = _Probe(lambda: shutil.which(tool) is not None, f"`{tool}` on PATH")
        _tool_probes[tool] = probe
    return skip_unless(probe, reason)


def pytest_configure(config: pytest.Config) -> None:
    """Register the guard markers, so ``--strict-markers`` stays clean."""
    for name, (_, default_reason) in _GUARD_MARKERS.items():
        config.addinivalue_line("markers", f"{name}(reason=None): {default_reason}")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Turn a guard marker into the skip its probe says the platform needs."""
    for name, (probe, default_reason) in _GUARD_MARKERS.items():
        for mark in item.iter_markers(name=name):
            if probe:
                continue
            reason = str(mark.args[0]) if mark.args else default_reason
            detail = probe.refusal_detail()
            if detail:
                reason = f"{reason} ({detail})"
            pytest.skip(f"{name}: {reason}")


@pytest.fixture(autouse=True)
def _no_mcp_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never spawn real MCP connector subprocesses from app lifespans in tests."""
    monkeypatch.setenv("BENCHMARKIT_DISABLE_MCP", "1")


@pytest.fixture(autouse=True)
def _no_lsp_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never spawn real language servers from app lifespans in tests."""
    monkeypatch.setenv("BENCHMARKIT_DISABLE_LSP", "1")


@pytest.fixture(autouse=True)
def _no_trash_of_its_own(monkeypatch: pytest.MonkeyPatch,
                         tmp_path: Path) -> None:
    """Keep every test's sweep off the app repo's legacy ``logs/trash``.

    ``sweep_trash`` prunes that folder as well as the workspace's own, and the
    app sweeps once at boot — so without this pin, running the suite would
    expire the developer's real old batches. Tests that assert about batches
    patch ``TRASH_DIR``/``LEGACY_TRASH_DIR`` themselves (the ``trash`` /
    ``legacy_trash`` fixtures); this only closes the paths that never ask.
    """
    monkeypatch.setattr(
        editorfs.constants, "LEGACY_TRASH_DIR", tmp_path / "legacy-trash-unused"
    )


@pytest.fixture(autouse=True)
def _no_legacy_tool_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tool-loop cap deterministic.

    ``create_app()`` loads the developer's own ``.env`` once, so a legacy
    ``MAX_MCP_ITERATION`` (still honoured as a fallback by
    ``config.max_tool_iteration()``) or an ``LSP_ENABLED`` list would
    otherwise leak into every test and depend on the machine it runs on.
    ``MCP_ENABLED`` is gone from the config entirely (connectors activate only
    through their ``enable`` flag in ``mcp-settings.json``); clearing it keeps a
    stale entry in a developer's ``.env`` from confusing the suite.
    """
    monkeypatch.delenv("MAX_TOOL_ITERATION", raising=False)
    monkeypatch.delenv("MAX_MCP_ITERATION", raising=False)
    monkeypatch.delenv("LSP_ENABLED", raising=False)
    monkeypatch.delenv("MCP_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def _no_debug_switch_leak(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the LLM debug switch off, and process state clean, for every test.

    Two leaks would make that switch machine-dependent otherwise:
    ``create_app()`` loads the developer's own ``.env`` (which may carry
    ``LLM_DEBUG``, or the legacy key it falls back to), and ``save_settings``
    writes the key straight into ``os.environ``, which ``monkeypatch`` never
    undoes. The switch itself also mutates what no fixture can undo on its own —
    the level and handler of the ``benchmarkit.llm`` logger, and the payload
    hook of ``core/llmkit/client``, which an app boot re-points at ``config``.

    So both keys are cleared up front, and the logger and the hook are handed
    back as they were found — except the handler the switch itself attaches,
    which is the switch's own state to rebuild and must not come back bound to a
    stream that has since gone (and never without the ``propagate`` off that
    belongs to it). Without this, a test asserting the client's hook defaults to
    off would pass or fail on the order the settings tests ran in.
    """
    from src import config as config_module
    from src.core.llmkit import client as llmkit_client
    from src.core.llmkit import constants as llmkit_constants

    logger = logging.getLogger(llmkit_constants.DEBUG_LOGGER_NAME)
    saved = (
        logger.level,
        list(logger.handlers),
        logger.propagate,
        llmkit_client.debug_enabled,
    )
    monkeypatch.delenv(config_module.LLM_DEBUG_KEY, raising=False)
    monkeypatch.delenv(config_module.LEGACY_LLM_DEBUG_KEY, raising=False)
    yield
    level, handlers, propagate, hook = saved
    kept = [
        handler
        for handler in handlers
        if not isinstance(handler, config_module._DebugStreamHandler)
    ]
    logger.setLevel(level)
    logger.handlers[:] = kept
    # The handler being dropped takes its ``propagate = False`` with it: leaving
    # that behind would silence a logger that now has nowhere to write.
    logger.propagate = propagate or len(kept) != len(handlers)
    llmkit_client.debug_enabled = hook


@pytest.fixture(autouse=True)
def _pane_shell_is_the_apps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the terminal pane's shell to the app's own default for every test.

    On POSIX the pane runs ``$SHELL`` when it names an executable program
    (:mod:`src.core.termhub.shells`), so the suite would otherwise drive each
    test shell in whatever the developer has as a login shell — and a configured
    interactive shell loads that developer's own startup files. A zsh with a
    prompt theme was measured to swallow a keystroke of whatever it types first
    (here, the redraw ate the ``s`` of ``sleep`` in the command a tree test
    types), so a session assertion can fail on one machine and pass on another.
    Same class as the cap leak above: the environment, not the code, decides the
    outcome.

    Clearing the key asks the selection seam its documented POSIX question,
    whose answer is the fallback shell — no test computes a path. It is *not* a
    skip condition: every platform's own answer still runs (Windows names Git
    Bash and ignores this key entirely), and a runner that offers no fallback
    keeps skipping with the seam's own sentence through ``shell_probe``.
    """
    monkeypatch.delenv(termhub_shells.constants.SHELL_KEY, raising=False)
    # The probe caches its verdict for the session, and the markers consult it;
    # a verdict taken from the ambient shell is the machine's, not this test's.
    shell_probe.clear_verdict()


@pytest.fixture
def job() -> Iterator[RunJob]:
    """Provide a fresh, empty benchmark run job."""
    yield RunJob()
