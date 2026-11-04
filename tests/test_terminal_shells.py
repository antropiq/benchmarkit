"""Which shell the pane selects, and what the child is handed (csl-05, C-D2 amended).

Every assertion runs on **either** runner by forcing the seam: the selection
platform and the environment are arguments, the Git candidates, the override name
and the filesystem probe are patched module attributes, and a fake Git tree is
planted in ``tmp_path`` rather than trusting the machine's Git (roadmap §9 rule 2,
and AGENTS.md's hermetic-tests rule). Nothing pretends to be another OS — the
Windows half is asked of ``shells.select_shell("nt", {...})``, the same call a
Windows ``spawn()`` makes.

The POSIX half is asserted as the **unchanged** behaviour it must stay: ``$SHELL``
when it is absolute and executable, else ``/bin/sh``, and ``-i`` either way.

The last class asks this module's *unforced* answer — the one a real ``spawn()``
makes here — and requires it to agree with the suite's one shell-capability probe
(``tests/conftest.py``), which is what the ``needs_shell`` guard consults. One
probe, and the tests that pick a shell and the tests that skip a session read
the same verdict (csl-08).
"""

import os
from pathlib import Path
from typing import Dict, List

import pytest

from mcp_connectors import gitbash as probe
from tests.conftest import selected_shell, shell_probe
from src.core.termhub import drivers, shells
from src.core.termhub import session as terminal_session
from src.core.termhub.session import TerminalUnsupported


@pytest.fixture
def no_git_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe's PATH step answers "no git", so the candidate list decides."""
    monkeypatch.setattr(probe.shutil, "which", lambda name, *a, **k: None)


def fake_git_tree(directory: Path) -> Path:
    """A directory spelled like a Git install root, toolset marker included."""
    (directory / "usr" / "bin").mkdir(parents=True)
    (directory / "usr" / "bin" / "ls.exe").write_text("x", encoding="utf-8")
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "bash.exe").write_text("#!/bin/sh\n", encoding="utf-8")
    return directory


def plant_in(tmp_path: Path) -> Path:
    """A planted Git root at the spelling every Windows test patches to find."""
    return fake_git_tree(tmp_path / "Git")


class TestTheSuitesShellProbe:
    """The guard and the selection are one question, answered once (csl-08).

    No fixture here patches the seam, so the answer is this machine's own — the
    same call ``spawn()`` makes with no arguments. A drift between the two means
    the suite is gating sessions on a different capability than the one that
    runs them, which is exactly what the shared probe exists to prevent.
    """

    def test_the_probe_offers_a_shell_exactly_when_this_module_names_one(self):
        try:
            named = shells.select_shell()[0]
        except TerminalUnsupported:
            named = None
        assert bool(shell_probe) is (named is not None), (
            "conftest's shell_probe and termhub.shells.select_shell() disagree "
            "about this machine — the needs_shell guard no longer gates the "
            "sessions this seam would run"
        )
        assert selected_shell() == named

    def test_the_named_program_is_the_one_the_pane_would_run(self):
        """The probe keeps the program, so a session test never re-asks for it.

        The rest of this file plants fake Git trees through patched module
        attributes; ``monkeypatch`` is undone before the next test's setup, and
        the probe answers once, so the value handed to a test is this machine's
        selection — the same program the ``open`` frame will name.
        """
        program = selected_shell()
        if program is None:
            assert not bool(shell_probe)
            return
        assert program == shells.select_shell()[0]
        assert Path(program).is_file()


class TestPosixSelection:
    """Today's answer, verbatim — the Linux pane is the reference."""

    def test_a_configured_absolute_executable_shell_wins(self, tmp_path):
        chosen = tmp_path / "my-shell"
        chosen.write_text("#!/bin/sh\n", encoding="utf-8")
        os.chmod(chosen, 0o755)
        assert shells.select_shell("posix", {"SHELL": str(chosen)}) == (
            str(chosen),
            "-i",
        )

    def test_the_fallback_covers_unset_relative_and_missing_specs(self):
        for env in ({}, {"SHELL": "  "}, {"SHELL": "zsh"}, {"SHELL": "/nope/nope"}):
            assert shells.select_shell("posix", env) == (
                shells.constants.FALLBACK_SHELL,
                "-i",
            ), env

    def test_the_posix_branch_never_reads_the_windows_knob(self, tmp_path, monkeypatch):
        """``TERMINAL_SHELL`` is a Windows name; POSIX keeps choosing as it did.

        And the candidate list is never consulted there either — a Linux box's own
        Git does not select the pane's shell.
        """
        monkeypatch.setattr(
            shells, "GIT_INSTALL_CANDIDATES", [str(fake_git_tree(tmp_path / "Git"))]
        )
        assert shells.select_shell("posix", {"TERMINAL_SHELL": "/somewhere/bash"}) == (
            shells.constants.FALLBACK_SHELL,
            "-i",
        )


class TestWindowsSelection:
    """C-D2 as amended: ``TERMINAL_SHELL``, then Git Bash, then refuse. No cmd.exe."""

    def test_a_planted_git_tree_is_answered_and_the_refusal_never_fires(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        root = plant_in(tmp_path)
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        assert shells.select_shell("nt", {}) == (
            os.path.join(str(root), "bin", "bash.exe"),
            "-i",
        )

    def test_the_override_names_the_install_and_a_stray_git_exe_does_not(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        root = fake_git_tree(tmp_path / "override-git")
        monkeypatch.setattr(
            shells, "GIT_INSTALL_CANDIDATES", [str(tmp_path / "no-git-here")]
        )
        env: Dict[str, str] = {probe.GIT_ENV_OVERRIDE: str(root)}
        assert shells.select_shell("nt", env)[0] == os.path.join(
            str(root), "bin", "bash.exe"
        )

        # G-D4's verification carries into the terminal: a git.exe whose tree has
        # no usr\bin\ls.exe is not an install, so it selects no shell.
        stray_cmd = tmp_path / "stray" / "cmd"
        stray_cmd.mkdir(parents=True)
        (stray_cmd / "git.exe").write_text("x", encoding="utf-8")
        monkeypatch.setattr(
            probe.shutil, "which", lambda name, *a, **k: str(stray_cmd / "git.exe")
        )
        with pytest.raises(TerminalUnsupported):
            shells.select_shell("nt", {})

    def test_a_root_without_a_bash_selects_nothing_rather_than_another_shell(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        """The refusal, not a consolation prize: pwsh/powershell are never picked."""
        root = fake_git_tree(tmp_path / "no-bash")
        os.remove(root / "bin" / "bash.exe")
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        with pytest.raises(TerminalUnsupported) as caught:
            shells.select_shell("nt", {})
        sentence = str(caught.value)
        for name in ("pwsh", "powershell", "cmd.exe"):
            assert name in sentence
        assert "not started here" in sentence

    def test_terminal_shell_wins_over_the_discovered_one(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        """C-D2's honest knob: the user's choice is what runs, flag included.

        A drive-qualified spec the runner has no such drive for, judged by the
        filesystem probe answered True — the qualification (the rule under test)
        is the module's own, so both halves of the Windows answer are exercised
        on either runner.
        """
        root = plant_in(tmp_path)
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        monkeypatch.setattr(shells, "_program_is_runnable", lambda spec: True)
        spec = "R:\\chosen\\bin\\bash.exe"
        assert shells.select_shell("nt", {"TERMINAL_SHELL": spec}) == (spec, "-i")

    def test_a_posix_shaped_terminal_shell_is_refused_by_name_not_spawned(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        """The rule stated, not left to ``os.path.isabs`` answering False by luck.

        ``/bin/bash`` (what a Git Bash window leaves in ``SHELL``) and the
        MSYS-mount spelling of the very install the pane would otherwise pick are
        refused **by name**. The sentence says which knob to fix, and the
        discovered bash is never substituted for it — a silent fallback is how
        the one knob stops being honest.
        """
        root = plant_in(tmp_path)
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        planted_bash = os.path.join(str(root), "bin", "bash.exe")
        mount_form = "/c/Program Files/Git/bin/bash.exe"
        for spec in ("/bin/bash", mount_form, "bash", "bash.exe", "Git/bin/bash.exe"):
            with pytest.raises(TerminalUnsupported) as caught:
                shells.select_shell("nt", {"TERMINAL_SHELL": spec})
            sentence = str(caught.value)
            # The spec is named, the knob is named, and the refusal says what to
            # set it to — never the discovered bash substituted quietly.
            assert spec in sentence, (spec, sentence)
            assert "TERMINAL_SHELL" in sentence, (spec, sentence)
            assert "drive-qualified" in sentence, (spec, sentence)
            assert planted_bash not in sentence, (spec, sentence)
            # The two explanations differ by shape, so each is pinned.
            if spec.startswith("/"):
                assert "POSIX-shaped" in sentence, (spec, sentence)
            else:
                assert "relative" in sentence, (spec, sentence)

    def test_a_qualified_spec_that_is_not_an_executable_refuses_with_the_spec(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        """The other half of "accepted only if executable", named in the sentence."""
        root = plant_in(tmp_path)
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        monkeypatch.setattr(shells, "_program_is_runnable", lambda spec: False)
        spec = "R:\\gone\\bin\\bash.exe"
        with pytest.raises(TerminalUnsupported) as caught:
            shells.select_shell("nt", {"TERMINAL_SHELL": spec})
        sentence = str(caught.value)
        assert spec in sentence
        assert "executable" in sentence
        assert os.path.join(str(root), "bin", "bash.exe") not in sentence

    def test_comspec_is_never_consulted(self, tmp_path, monkeypatch, no_git_on_path):
        """COMSPEC pointing at an absurd program changes the answer to nothing.

        With a Git install found the answer is still Git Bash; with none it is the
        refusal — and the refusal never names that program as the thing it tried.
        """
        absurd = str(tmp_path / "COMSPEC-SHOULD-NOT-RUN.exe")
        monkeypatch.setattr(shells, "_program_is_runnable", lambda spec: True)

        root = plant_in(tmp_path)
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        env: Dict[str, str] = {"COMSPEC": absurd, "SHELL": "/bin/bash"}
        assert shells.select_shell("nt", env)[0] == os.path.join(
            str(root), "bin", "bash.exe"
        )

        monkeypatch.setattr(
            shells, "GIT_INSTALL_CANDIDATES", [str(tmp_path / "no-git-here")]
        )
        with pytest.raises(TerminalUnsupported) as caught:
            shells.select_shell("nt", {"COMSPEC": absurd})
        sentence = str(caught.value)
        assert absurd not in sentence
        assert "cmd.exe" in sentence

    def test_the_no_shell_refusal_names_everything_it_looked_for(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        candidates = [str(tmp_path / "first-git"), str(tmp_path / "second-git")]
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", candidates)
        with pytest.raises(TerminalUnsupported) as caught:
            shells.select_shell("nt", {})
        sentence = str(caught.value)
        assert "TERMINAL_SHELL" in sentence
        assert probe.GIT_ENV_OVERRIDE in sentence
        assert "git" in sentence
        for candidate in candidates:
            assert candidate in sentence
        assert "usr\\bin\\ls.exe" in sentence

    def test_a_git_bash_launch_of_the_app_still_selects_git_bash(
        self, tmp_path, monkeypatch, no_git_on_path
    ):
        """The hazard, named: MSYS-form ``SHELL`` and ``PATH`` change nothing.

        Launched from a Git Bash window, the environment carries a POSIX-shaped
        ``SHELL`` and ``/usr/bin``-style ``PATH`` entries. The Windows branch
        reads neither as a shell, and lands on the planted Git Bash.
        """
        root = plant_in(tmp_path)
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        env = {
            "SHELL": "/bin/bash",
            "COMSPEC": r"C:\WINDOWS\system32\cmd.exe",
            "PATH": "/usr/bin:/bin:" + os.environ.get("PATH", ""),
        }
        assert shells.select_shell("nt", env)[0] == os.path.join(
            str(root), "bin", "bash.exe"
        )


class TestChildEnvironment:
    """What the shell is handed — the answer that reaches the child's ``PATH``.

    Asked of a planted Git root, so the same assertions hold on either runner:
    the Git search order is attached to the *program*, through the shared
    toolset marker, not to the platform's name.
    """

    def test_the_git_case_prepends_gits_directories_and_keeps_the_rest(self, tmp_path):
        root = fake_git_tree(tmp_path / "Git with space")  # a space inside an entry
        (root / "ucrt64" / "bin").mkdir(parents=True)
        shell = (os.path.join(str(root), "bin", "bash.exe"), "-i")
        base = {
            "PATH": os.pathsep.join([str(tmp_path / "spaced dir"), str(tmp_path), ""]),
            "SYSTEMROOT": str(tmp_path),
        }
        child = shells.child_environment(shell, 132, 41, base)

        entries: List[str] = child["PATH"].split(os.pathsep)
        assert entries[:3] == [
            os.path.join(str(root), "usr", "bin"),
            os.path.join(str(root), "bin"),
            os.path.join(str(root), "ucrt64", "bin"),
        ]
        # The ambient entries keep their order; no empty part survives.
        assert entries[3:] == [str(tmp_path / "spaced dir"), str(tmp_path)]
        # The terminal's own keys, unchanged on both platforms.
        assert child["TERM"] == shells.constants.TERM_VALUE
        assert (child["COLUMNS"], child["LINES"]) == ("132", "41")
        assert child["SYSTEMROOT"] == str(tmp_path)
        assert set(child) - {"PATH", "TERM", "COLUMNS", "LINES"} == set(base) - {"PATH"}

    def test_a_shell_outside_a_git_install_keeps_the_ambient_path(self, tmp_path):
        """A TERMINAL_SHELL that is not a Git toolset borrows nobody's search order."""
        elsewhere = tmp_path / "tools" / "bin" / "bash.exe"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_text("#!/bin/sh\n", encoding="utf-8")
        base = {"PATH": os.pathsep.join([str(tmp_path), str(tmp_path / "x")])}
        child = shells.child_environment((str(elsewhere), "-i"), 80, 24, base)
        assert child["PATH"] == base["PATH"]
        assert (child["COLUMNS"], child["LINES"]) == ("80", "24")


class TestSpawnAsksTheSeam:
    """``session.spawn()`` stops deciding anything about the shell.

    Driven with a planted driver answering the seam's surface and a recorded
    ``Popen``, so it runs wherever the suite runs — the guarantee under test is
    that the argv and the environment come from :mod:`termhub.shells`, including
    the interactive flag that used to be hard-coded in the session.
    """

    class FakePty:
        def __init__(self) -> None:
            self.size = (0, 0)

        @property
        def slave_fd(self) -> int:
            return -1

        def set_size(self, cols: int, rows: int) -> None:
            self.size = (cols, rows)

        def get_size(self) -> tuple:
            return self.size

        def read(self, max_bytes: int) -> bytes:
            return b""

        def write(self, data: bytes) -> None:
            pass

        def close_slave(self) -> None:
            pass

        def close_master(self) -> None:
            pass

    def test_the_windows_spec_and_environment_reach_popen(
        self, monkeypatch, tmp_path, no_git_on_path
    ):
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        root = fake_git_tree(tmp_path / "Git")
        monkeypatch.setattr(shells, "GIT_INSTALL_CANDIDATES", [str(root)])
        # The session asks select_shell() with no arguments, so the platform the
        # *module* decides on is the seam a test forces (the same patch-point a
        # real Windows spawn answers through). The two knobs spawn() reads from
        # the process environment are cleared, so a developer's own settings
        # cannot change which shell this asserts about.
        monkeypatch.setattr(shells, "SELECTION_PLATFORM", "nt")
        monkeypatch.delenv(shells.constants.TERMINAL_SHELL_KEY, raising=False)
        monkeypatch.delenv(probe.GIT_ENV_OVERRIDE, raising=False)

        class FakeDriver:
            @staticmethod
            def open_pair():
                return TestSpawnAsksTheSeam.FakePty()

        monkeypatch.setattr(drivers, "selected_driver", lambda: FakeDriver)
        started: Dict[str, object] = {}

        def record_popen(argv, **kwargs):
            started["argv"] = list(argv)
            started["env"] = kwargs["env"]
            started["cwd"] = kwargs["cwd"]
            raise OSError("stop here — nothing is really spawned")

        monkeypatch.setattr(terminal_session.subprocess, "Popen", record_popen)

        async def emit(_event: Dict[str, object]) -> None:  # never reached
            pass

        live = terminal_session.TerminalSession(emit)
        with pytest.raises(OSError):
            live.spawn(cols=100, rows=30)
        # The argv is the seam's spec, flag and all — session.py no longer spells
        # it out, and self.shell (the `open` frame) names the program.
        assert started["argv"] == [os.path.join(str(root), "bin", "bash.exe"), "-i"]
        env = started["env"]
        assert env["TERM"] == shells.constants.TERM_VALUE
        assert (env["COLUMNS"], env["LINES"]) == ("100", "30")
        assert started["cwd"] == str(tmp_path)
        assert live.shell == os.path.join(str(root), "bin", "bash.exe")
        assert live.process is None and live.pty is None

    def test_a_shell_refusal_reaches_the_session_as_the_same_refusal(
        self, monkeypatch, tmp_path, no_git_on_path
    ):
        """One refusal shape for both gaps: the route already renders this one.

        A runner whose driver is available but whose shell selection finds nothing
        must not crash or open a degraded pane — ``spawn()`` raises
        ``TerminalUnsupported`` with the sentence, which is what the WebSocket
        turns into an ``error`` frame plus 4501 (csl-04's landed path).
        """
        monkeypatch.setattr(shells, "SELECTION_PLATFORM", "nt")
        monkeypatch.setattr(
            shells, "GIT_INSTALL_CANDIDATES", [str(tmp_path / "no-git-here")]
        )
        # spawn() reads the process environment, so the two knobs it consults are
        # cleared here — the suite must not depend on a developer's own override.
        monkeypatch.delenv(shells.constants.TERMINAL_SHELL_KEY, raising=False)
        monkeypatch.delenv(probe.GIT_ENV_OVERRIDE, raising=False)

        class FakeDriver:
            @staticmethod
            def open_pair():
                return TestSpawnAsksTheSeam.FakePty()

        monkeypatch.setattr(drivers, "selected_driver", lambda: FakeDriver)
        live = terminal_session.TerminalSession(lambda _e: None)
        with pytest.raises(TerminalUnsupported) as caught:
            live.spawn(cols=80, rows=24)
        assert "TERMINAL_SHELL" in str(caught.value)
