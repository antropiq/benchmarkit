"""The Windows pseudo-console driver (csl-06): its surface, and the real console.

Two halves, and which one runs where is the point of the split.

**The seam half runs on every runner.** ``pty_conpty`` must import on Linux (a
driver that cannot be imported breaks the suite there) and must never be
*selected* while the POSIX driver answers; its own rules — chunking to the
caller's cap, the EOF predicate, the failure-vs-end distinction, the size
carried to the console's creation, the argv/env framing, naming the console
helper — are asked of a planted fake ``winpty`` and a planted child list. The
session's job rules (csl-07) are asked the same way: a planted console plus a
recording ``ProcessJob`` assert that the session opens one job before the
spawn, assigns the shell as the leader and the helper as a plain member, and
releases it on every exit path — with the job seam forced either way, so both
platform halves of the session's one code path are proven wherever the suite
runs (roadmap §9 rule 2), and no test depends on a Git install path.

**The real-console half is marked ``windows_behaviour``** and additionally needs
the dependency installed — with the ``windows`` extra absent these tests skip
with that sentence rather than pass vacuously. They drive the ordinary
``TerminalSession`` against a real pseudo-console and a real shell, so what is
asserted is the frame sequence a pane would see: C-D3's two end-of-stream cases,
the resize round-trip, C-D6's non-ASCII round-trip, and C-D5's tree — a
disconnect takes the shell, the ``OpenConsole.exe`` helper and a backgrounded
child (the ghost ``proctree`` names, on the platform where no group signal
reaches it), asserted by the job's own member list and the shared liveness
probe. The resize assertion here is the **server-side round-trip**: whether the
*shell* also re-reads its size is a fact about that shell — measured on this
machine (Git Bash does; the unchanged
``test_terminal.py::TestSession::test_resize_updates_the_window_size`` passes
against this driver), and stated in the task's Results note rather than
promised as cross-platform parity by this file (C-D4's honest difference).
"""

import asyncio
import os
import threading
import time
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.core import proctree
from src.core import proctree_jobs
from src.core.termhub import constants as termhub_constants
from src.core.termhub import drivers as termhub_drivers
from src.core.termhub import pty_conpty, pty_posix, shells
from src.core.termhub import session as terminal_session


# -- the planted console -------------------------------------------------------


class FakeConsole:
    """A stand-in for ``winpty.PTY``, answering exactly what the driver asks.

    Records what it was created with (the size), what it was spawned with
    (program, command line, cwd, env block) and what was written to it, and
    replays a scripted list of read results. Every knob a test needs is a plain
    attribute, so the driver's rules can be asserted on either runner.
    """

    def __init__(self, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows
        self.pid: Optional[int] = None
        self.spawned: Optional[Dict[str, Any]] = None
        self.written: List[str] = []
        self.sizes: List[Tuple[int, int]] = []
        self.reads: List[str] = []
        self.read_error: Optional[Exception] = None
        self.alive = True
        self.eof = False
        self.status: Optional[int] = None
        self.cancelled = False
        self.released = False
        self.spawn_raises = False
        self.spawn_answers = True
        # The pid the console reports for its shell (the agent's ``pid`` prop).
        self.shell_pid: Optional[int] = 4242
        self.hide_pid = False

    # the surface the driver uses
    def spawn(self, appname, cmdline=None, cwd=None, env=None) -> bool:
        if self.spawn_raises:
            raise RuntimeError("the console refused the child")
        if not self.spawn_answers:
            return False
        self.spawned = {
            "appname": appname,
            "cmdline": cmdline,
            "cwd": cwd,
            "env": env,
        }
        self.pid = None if self.hide_pid else self.shell_pid
        return True

    def read(self, blocking: bool = False) -> str:
        if self.released:
            raise RuntimeError("the console is gone")
        if self.read_error is not None:
            raise self.read_error
        if self.reads:
            return self.reads.pop(0)
        return ""

    def write(self, to_write: str) -> int:
        self.written.append(to_write)
        return len(to_write)

    def set_size(self, cols: int, rows: int) -> None:
        self.sizes.append((cols, rows))

    def isalive(self) -> bool:
        if self.released:
            raise RuntimeError("the console is gone")
        return self.alive

    def iseof(self) -> bool:
        return self.eof

    def get_exitstatus(self) -> Optional[int]:
        return self.status

    def cancel_io(self) -> bool:
        self.cancelled = True
        return True


class FakeWinpty:
    """The planted dependency module: the consoles it made, in creation order."""

    def __init__(self) -> None:
        self.consoles: List[FakeConsole] = []
        self.create_fails = False
        # Knobs applied to every console this factory makes, so a test can arm
        # the failure before ``spawn_shell`` creates the console itself.
        self.spawn_raises = False
        self.spawn_answers = True

    def PTY(self, cols: int, rows: int) -> FakeConsole:
        if self.create_fails:
            raise RuntimeError("no pseudo-console here")
        console = FakeConsole(cols, rows)
        console.spawn_raises = self.spawn_raises
        console.spawn_answers = self.spawn_answers
        self.consoles.append(console)
        return console

    @property
    def last(self) -> FakeConsole:
        return self.consoles[-1]


@pytest.fixture
def fake_winpty(monkeypatch: pytest.MonkeyPatch) -> FakeWinpty:
    """The driver's dependency, planted — so its rules are testable anywhere.

    Patched on the driver module itself (the hermetic rule: a test patches the
    seam's own attribute, never a path it computed). The process enumeration is
    the driver's other OS touch, planted as an empty answer by default; the poll
    intervals are shortened so the waiting tests stay quick.
    """
    module = FakeWinpty()
    monkeypatch.setattr(pty_conpty, "winpty", module)
    monkeypatch.setattr(pty_conpty, "_child_processes", lambda: [])
    monkeypatch.setattr(pty_conpty, "READ_POLL_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(pty_conpty, "EXIT_POLL_INTERVAL_SECONDS", 0.001)
    return module


def planted_children(monkeypatch: pytest.MonkeyPatch, *snapshots):
    """The driver's child enumeration, answering scripted snapshots in order.

    ``spawn_shell`` asks it three times — before the console is created, after
    it, and after the shell — which is exactly the before/after the helper
    capture turns on. The last snapshot repeats once the list runs out.
    """
    state = {"i": 0}

    def enumerate_children():
        index = min(state["i"], len(snapshots) - 1)
        state["i"] += 1
        return list(snapshots[index])

    monkeypatch.setattr(pty_conpty, "_child_processes", enumerate_children)
    return enumerate_children

def spawned_pair(
    module: FakeWinpty,
    argv: Tuple[str, ...] = ("C:\\Git\\bin\\bash.exe", "-i"),
    cols: int = 80,
    rows: int = 24,
) -> Tuple[pty_conpty.ConPty, pty_conpty.ConPTYChild]:
    """A pair the session would have opened and spawned onto, in that order."""
    pair = pty_conpty.ConPty(cols, rows)
    child = pair.spawn_shell(list(argv), "C:\\work", {"TERM": "xterm-256color"})
    return pair, child


class TestSelectionOfTheDriver:
    """The Linux promises: importable, probed honestly, never chosen there."""

    def test_the_module_imports_whether_or_not_the_dependency_is_there(self):
        # This test *is* the import: reaching it means the module loaded on a
        # runner that may have no ``winpty`` at all (and on this one, may have
        # it — either way the module import cannot be what breaks the suite).
        assert hasattr(pty_conpty, "available")
        assert hasattr(pty_conpty, "open_pair")

    def test_the_probe_answers_the_dependency_not_the_platform(self, fake_winpty):
        assert pty_conpty.available() is True
        fake_winpty.create_fails = True  # a console failing is not a missing driver
        assert pty_conpty.available() is True

    def test_no_dependency_means_the_probe_says_no(self, monkeypatch):
        monkeypatch.setattr(pty_conpty, "winpty", None)
        assert pty_conpty.available() is False

    def test_a_probe_of_false_takes_the_driver_out_of_the_seam(self, monkeypatch):
        # Selection follows the probe (csl-04's rule, now for the Windows half):
        # hide the dependency and the module is no longer a driver.
        monkeypatch.setattr(pty_posix, "available", lambda: False)
        monkeypatch.setattr(pty_conpty, "winpty", None)
        assert termhub_drivers.selected_driver() is None
        monkeypatch.setattr(pty_conpty, "winpty", FakeWinpty())
        assert termhub_drivers.selected_driver() is pty_conpty

    def test_the_seam_never_selects_it_while_the_posix_driver_answers(
        self, fake_winpty, monkeypatch
    ):
        """An installed Windows driver must not take over a working POSIX pane."""
        monkeypatch.setattr(pty_posix, "available", lambda: True)
        assert termhub_drivers.selected_driver() is pty_posix
        # …and the Windows half really is a candidate here, so the order is what
        # decides, not an inability to be selected at all.
        monkeypatch.setattr(pty_posix, "available", lambda: False)
        assert termhub_drivers.selected_driver() is pty_conpty

    def test_starting_the_child_is_a_property_of_the_pair_not_of_the_os(self):
        # ``session.spawn()`` branches on this; both answers exist on one runner.
        assert termhub_drivers.pair_starts_the_child(pty_posix.PosixPty(3, 4)) is False
        assert termhub_drivers.pair_starts_the_child(pty_conpty.ConPty(80, 24)) is True


class TestSizeTravelsToTheCreation:
    """C-D4: *"the size is passed when the pseudo-console is created"*."""

    def test_the_console_is_born_at_the_size_set_before_the_spawn(self, fake_winpty):
        pair = pty_conpty.ConPty(80, 24)
        pair.set_size(132, 41)
        _child = pair.spawn_shell(["bash", "-i"], "C:\\work", {})
        assert (fake_winpty.last.cols, fake_winpty.last.rows) == (132, 41)

    def test_the_open_size_is_the_session_s_default(self, fake_winpty):
        pair = pty_conpty.open_pair()
        pair.spawn_shell(["bash"], "C:\\work", {})
        assert (fake_winpty.last.cols, fake_winpty.last.rows) == (
            termhub_constants.DEFAULT_COLS,
            termhub_constants.DEFAULT_ROWS,
        )

    def test_a_resize_round_trips_and_reaches_the_live_console(self, fake_winpty):
        pair, _child = spawned_pair(fake_winpty)
        pair.set_size(132, 41)
        assert pair.get_size() == (132, 41)
        assert fake_winpty.last.sizes == [(132, 41)]

    def test_a_non_positive_size_is_refused_before_the_agent_sees_it(self, fake_winpty):
        pair = pty_conpty.ConPty(80, 24)
        with pytest.raises(ValueError):
            pair.set_size(0, 24)
        with pytest.raises(ValueError):
            pair.set_size(80, -1)
        assert pair.get_size() == (80, 24)


class TestFramingOfBytes:
    """Bytes out, at most what was asked — the session's framing stays the only one."""

    def test_an_agent_answer_larger_than_the_cap_is_chunked_to_it(self, fake_winpty):
        # Measured: the agent answers ~20 000 characters in one read. The driver
        # caps its own answer and keeps the remainder.
        pair, _child = spawned_pair(fake_winpty)
        blob = "x" * 9000
        fake_winpty.last.reads = [blob]
        first = pair.read(4096)
        second = pair.read(4096)
        third = pair.read(4096)
        assert (len(first), len(second), len(third)) == (4096, 4096, 808)
        assert first + second + third == blob.encode("utf-8")

    def test_the_ceiling_is_the_constant_not_the_call_s_number(self, fake_winpty):
        """*"never larger than ``constants.PTY_READ_BYTES``"* — enforced, not polite.

        A caller that asks for more still gets at most the framing ceiling, so a
        slip at one call site cannot turn the pane's framing into one giant
        message.
        """
        pair, _child = spawned_pair(fake_winpty)
        fake_winpty.last.reads = ["y" * 10000]
        chunk = pair.read(10 * termhub_constants.PTY_READ_BYTES)
        assert len(chunk) == termhub_constants.PTY_READ_BYTES

    def test_read_answers_bytes_never_decoded_text(self, fake_winpty):
        pair, _child = spawned_pair(fake_winpty)
        fake_winpty.last.reads = ["hi"]
        chunk = pair.read(4096)
        assert isinstance(chunk, bytes) and chunk == b"hi"

    def test_a_character_split_across_chunks_stays_recoverable(self, fake_winpty):
        # The driver cuts *bytes*; the session's incremental decoder joins them.
        pair, _child = spawned_pair(fake_winpty)
        fake_winpty.last.reads = ["café中文"]
        gathered = b""
        while True:
            chunk = pair.read(3)
            gathered += chunk
            if len(chunk) < 3:
                break
        assert gathered == "café中文".encode("utf-8")

    def test_write_is_a_plain_byte_writer(self, fake_winpty):
        # C-D5: Ctrl-C is an input event, not a byte this code special-cases.
        pair, _child = spawned_pair(fake_winpty)
        pair.write(b"echo\r")
        pair.write(b"\x03")
        assert fake_winpty.last.written == ["echo\r", "\x03"]

    def test_non_ascii_input_bytes_reach_the_console_as_their_characters(self, fake_winpty):
        pair, _child = spawned_pair(fake_winpty)
        pair.write("café中文 ☃".encode("utf-8"))
        assert fake_winpty.last.written == ["café中文 ☃"]


class TestTheEndOfTheStream:
    """C-D3 re-stated: an EOF is the far-end answer; a broken mechanism is not."""

    def test_an_empty_read_with_the_shell_gone_raises_the_seam_s_predicate(
        self, fake_winpty
    ):
        pair, _child = spawned_pair(fake_winpty)
        fake_winpty.last.alive = False
        with pytest.raises(pty_posix.FarEndGone):
            pair.read(4096)

    def test_the_agent_s_eof_flag_counts_even_before_the_shell_is_reaped(self, fake_winpty):
        pair, _child = spawned_pair(fake_winpty)
        fake_winpty.last.eof = True
        with pytest.raises(pty_posix.FarEndGone):
            pair.read(4096)

    def test_trailing_output_is_drained_before_the_far_end_is_answered(self, fake_winpty):
        # An EOF is not an excuse to lose the shell's last line.
        pair, _child = spawned_pair(fake_winpty)
        fake_winpty.last.alive = False
        fake_winpty.last.reads = ["last words"]
        assert pair.read(4096) == b"last words"
        with pytest.raises(pty_posix.FarEndGone):
            pair.read(4096)

    def test_an_empty_read_with_a_live_shell_keeps_the_pump_waiting(self, fake_winpty):
        pair, _child = spawned_pair(fake_winpty)
        answer: List[Any] = []

        def reader() -> None:
            try:
                answer.append(pair.read(64))
            except BaseException as exc:  # reported through the list
                answer.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(0.05)
        assert thread.is_alive(), "the pump ended on an idle console"
        fake_winpty.last.reads = ["now"]
        thread.join(2.0)
        assert not thread.is_alive()
        assert answer == [b"now"]

    def test_a_read_failure_while_the_shell_lives_is_not_the_far_end(self, fake_winpty):
        # A mechanism that broke must not reach the pane as a clean end of stream.
        pair, _child = spawned_pair(fake_winpty)
        fake_winpty.last.read_error = RuntimeError("agent pipe broke")
        with pytest.raises(pty_conpty.ConPtyError) as caught:
            pair.read(4096)
        assert not isinstance(caught.value, pty_posix.FarEndGone)
        assert isinstance(caught.value, OSError)

    def test_releasing_the_pair_answers_the_far_end_without_waiting(self, fake_winpty):
        # What ends a blocked pump on this mechanism, from the teardown's side.
        pair, _child = spawned_pair(fake_winpty)
        answer: List[Any] = []

        def reader() -> None:
            try:
                answer.append(pair.read(64))
            except BaseException as exc:
                answer.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(0.05)
        pair.close_master()
        thread.join(2.0)
        assert not thread.is_alive(), "the pump is still reading a released console"
        assert answer and isinstance(answer[0], pty_posix.FarEndGone)
        assert fake_winpty.last.cancelled, "a pending read was not cancelled"

    def test_close_master_is_idempotent_and_never_a_kill(self, fake_winpty):
        # The kill is the job's and the teardown's (csl-07); the driver releases
        # the console and reports the child it can no longer see.
        pair, child = spawned_pair(fake_winpty)
        pair.close_master()
        pair.close_master()
        assert child.is_alive() is False
        assert child.wait() == pty_conpty.EXIT_CODE_UNKNOWN

    def test_there_is_no_slave_and_the_driver_says_so(self, fake_winpty):
        pair, _child = spawned_pair(fake_winpty)
        assert pair.close_slave() is None  # the session's ``finally``, one shape
        with pytest.raises(pty_conpty.ConPtyError):
            pair.slave_fd  # reaching this is a caller that branched wrong


class TestTheChildReportsSeparatelyFromTheStream:
    """"A stream cut looks finished" — the child's status is its own answer."""

    def test_the_exit_status_arrives_even_though_the_stream_ends_first(self, fake_winpty):
        pair, child = spawned_pair(fake_winpty)
        fake_winpty.last.alive = False
        fake_winpty.last.status = 7
        assert child.exit_status() == 7
        assert child.wait() == 7

    def test_a_child_gone_with_no_status_is_named_not_invented(self, fake_winpty):
        pair, child = spawned_pair(fake_winpty)
        fake_winpty.last.alive = False
        fake_winpty.last.status = None
        assert child.exit_status() is None
        assert child.wait() == pty_conpty.EXIT_CODE_UNKNOWN


class TestSpawnFraming:
    """The argv, cwd and environment ``termhub.shells`` produced, delivered."""

    def test_the_argv_is_framed_into_the_agent_s_command_line(self, fake_winpty):
        spawned_pair(
            fake_winpty, ("C:\\Program Files\\Git\\bin\\bash.exe", "-i")
        )
        spawned = fake_winpty.last.spawned
        assert spawned["appname"] == "C:\\Program Files\\Git\\bin\\bash.exe"
        assert spawned["cmdline"] == "-i"
        assert spawned["cwd"] == "C:\\work"

    def test_an_argument_needing_quotes_keeps_them(self, fake_winpty):
        spawned_pair(fake_winpty, ("bash.exe", "-c", "echo a b"))
        assert fake_winpty.last.spawned["cmdline"] == '-c "echo a b"'

    def test_the_environment_arrives_as_the_agent_s_block(self, fake_winpty):
        pair = pty_conpty.ConPty(80, 24)
        pair.spawn_shell(
            ["bash", "-i"], "C:\\work", {"TERM": "xterm-256color", "LINES": "30"}
        )
        assert fake_winpty.last.spawned["env"] == "TERM=xterm-256color\0LINES=30\0"

    def test_a_console_that_cannot_be_created_is_a_renderable_oserror(self, fake_winpty):
        # C §3: *"if … the pseudo-console cannot be created"* — the session's
        # OSError path and the route's 4501 refusal stay one shape.
        fake_winpty.create_fails = True
        pair = pty_conpty.ConPty(80, 24)
        with pytest.raises(pty_conpty.ConPtyError):
            pair.spawn_shell(["bash"], "C:\\work", {})

    def test_a_child_that_refuses_to_start_releases_the_console_it_made(self, fake_winpty):
        pair = pty_conpty.ConPty(80, 24)
        fake_winpty.spawn_raises = True
        with pytest.raises(pty_conpty.ConPtyError):
            pair.spawn_shell(["no-such-shell"], "C:\\work", {})
        # A half-started session must not strand the helper born with the console.
        assert fake_winpty.last.cancelled is True

    def test_a_spawn_that_answers_false_is_a_failure_too(self, fake_winpty):
        pair = pty_conpty.ConPty(80, 24)
        fake_winpty.spawn_answers = False
        with pytest.raises(pty_conpty.ConPtyError):
            pair.spawn_shell(["bash"], "C:\\work", {})
        assert fake_winpty.last.cancelled is True


class TestTheConsoleHelperIsNamed:
    """C-D5: *"That second process is the one an implementer will forget."*

    The dependency never reports the helper's pid, so the driver enumerates this
    process's children around the console's creation and matches the helper's
    name. Planted here — no test depends on a real console or a real pid.
    """

    def test_a_helper_that_appears_with_the_console_is_reported(
        self, fake_winpty, monkeypatch
    ):
        planted_children(
            monkeypatch,
            [],  # before the console exists
            [(777, "openconsole.exe")],  # the helper, born with it
        )
        pair, child = spawned_pair(fake_winpty)
        assert pair.started_pids() == (4242, 777)
        assert child.pid == 4242

    def test_a_helper_that_belongs_to_another_session_is_not_claimed(
        self, fake_winpty, monkeypatch
    ):
        # The snapshot before the console was made holds the other session's
        # helper; only what *appeared* is this pair's.
        planted_children(
            monkeypatch,
            [(700, "openconsole.exe")],  # another session's, already there
            [(700, "openconsole.exe"), (800, "openconsole.exe")],
        )
        pair, _child = spawned_pair(fake_winpty)
        assert pair.started_pids() == (4242, 800)

    def test_a_foreign_child_of_the_same_instant_is_not_claimed(
        self, fake_winpty, monkeypatch
    ):
        # Name-matched on purpose: a hub's child that appears mid-spawn is not
        # this session's helper and must not land in this session's job.
        planted_children(
            monkeypatch,
            [],
            [(900, "node.exe"), (800, "winpty-agent.exe")],
        )
        pair, _child = spawned_pair(fake_winpty)
        assert pair.started_pids() == (4242, 800)


# -- the session's job: one tree, both processes (csl-07) ----------------------


class RecordingJob:
    """A stand-in for ``proctree_jobs.ProcessJob`` recording every call.

    The session's containment rules — who is a leader, who is a plain member,
    when the handle is released — are what csl-07 adds; they are asserted here
    on **any** runner through the seams (the driver seam answering the planted
    console, the job class replaced), never left to a Windows-only test.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self.assignments: List[Tuple[int, bool]] = []
        self.closed = 0

    def assign(self, pid: Optional[int], *, leader: bool = False) -> Optional[bool]:
        self.assignments.append((pid, leader))
        return True

    def close(self) -> None:
        self.closed += 1


class TestTheSessionOwnsOneJob:
    """The session opens one job **before the spawn** and contains every
    process the spawn started (C-D5). Driven with the planted console and a
    recorded job — the driver seam forced to the planted pair by hiding the
    kernel PTY's probe, so these run identically on either runner (roadmap
    §9 rule 2) and a Linux box never spawns a real shell here. What the
    session assigns is the driver's ``started_pids()`` — the shell as the
    leader the shared teardown looks the job up by, the console helper beside
    it as a plain member."""

    def _force_the_planted_driver(self, monkeypatch) -> None:
        # On this Windows runner the POSIX probe already answers False; on the
        # Linux runner it answers True and would win the seam, so the test
        # asks the seam's own question the way csl-04/csl-06's tests do.
        monkeypatch.setattr(pty_posix, "available", lambda: False)

    def _spawn_with_recording_job(self, monkeypatch, fake_winpty):
        """A session spawn whose job is a :class:`RecordingJob`."""
        self._force_the_planted_driver(monkeypatch)
        monkeypatch.setattr(proctree_jobs, "ProcessJob", RecordingJob)

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        session = terminal_session.TerminalSession(emit)
        session.spawn(cols=80, rows=24)  # planted console: no real process
        return session

    def test_the_job_is_named_for_the_session_and_contains_both_processes(
        self, fake_winpty, monkeypatch
    ):
        # The helper is planted through the driver's own enumeration seam — the
        # same snapshot the real console answers — so what is under test here
        # is the *session's* containment of it, not the driver's naming.
        planted_children(monkeypatch, [], [(777, "openconsole.exe")])
        session = self._spawn_with_recording_job(monkeypatch, fake_winpty)
        try:
            job = session.job
            assert isinstance(job, RecordingJob)
            # The name carries the session id: a process list or a log line
            # reads as *this* session's tree.
            assert job.name == f"terminal-{session.id}"
            # Shell first, as the leader (the pid the teardown entry point is
            # handed and the job is keyed by); helper second, contained but
            # never a key — one child, one owner of its death. Both before
            # spawn() returns: containment is decided at a process's creation,
            # and the shell cannot have run a command before this point.
            assert job.assignments == [(4242, True), (777, False)]
        finally:
            _release_session(session)

    def test_a_driver_that_starts_no_helper_leaves_the_shell_the_only_member(
        self, fake_winpty, monkeypatch
    ):
        # ``fake_winpty`` plants an empty child list by default: no helper was
        # born, so nothing beyond the shell is assigned — the session asks the
        # driver, it does not guess.
        session = self._spawn_with_recording_job(monkeypatch, fake_winpty)
        try:
            assert session.job.assignments == [(4242, True)]
        finally:
            _release_session(session)

    def test_a_failed_spawn_releases_the_job_it_opened(
        self, fake_winpty, monkeypatch
    ):
        """No handle is left armed against nothing: a spawn that failed before
        any process reached the console still closes the job it opened."""
        self._force_the_planted_driver(monkeypatch)
        monkeypatch.setattr(proctree_jobs, "ProcessJob", RecordingJob)
        fake_winpty.create_fails = True

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        session = terminal_session.TerminalSession(emit)
        with pytest.raises(pty_conpty.ConPtyError):
            session.spawn(cols=80, rows=24)
        assert session.job is not None
        assert session.job.closed == 1
        assert session.pty is None and session.process is None

    def test_teardown_runs_the_shared_entry_point_on_both_platforms(
        self, fake_winpty, monkeypatch
    ):
        """No ``os.name`` guard survives in the session's teardown path: the
        entry point is asked with the leader pid and the session's grace on
        **either runner**, and the job the session opened is released by the
        teardown. (What the real entry point does with that pid — find the
        job, drain it — is ``test_proctree.py``'s dispatch and escalation
        tests; this one pins the *call*, which is the session's half.)"""
        self._force_the_planted_driver(monkeypatch)
        monkeypatch.setattr(proctree_jobs, "ProcessJob", RecordingJob)
        asked: List[Tuple[Any, ...]] = []

        async def fake_kill_session_tree(
            pid: int, reap_leader: Any, grace_seconds: float
        ) -> None:
            asked.append((pid, grace_seconds))
            with suppress(Exception):
                await reap_leader(0.0)  # the injected step stays injected

        monkeypatch.setattr(proctree, "kill_session_tree", fake_kill_session_tree)

        async def scenario() -> None:
            async def emit(_event: Dict[str, Any]) -> None:
                pass

            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=80, rows=24)
            await session.teardown()
            assert asked == [(4242, termhub_constants.STOP_GRACE_SECONDS)]
            assert session.pty is None
            assert session.job is not None
            assert session.job.closed == 1, (
                "the teardown left the session's job handle open"
            )
            # Idempotent: the second teardown asks for nothing again.
            asked.clear()
            await session.teardown()
            assert asked == []

        asyncio.run(scenario())

    def test_where_the_platform_has_no_jobs_the_session_is_unchanged(
        self, fake_winpty, monkeypatch
    ):
        """The Linux promise, forced on **either** runner through the seam's
        own patch-point: with the job half unavailable, the session still runs
        one code path — it asks, and the answer is ``None`` — its job object
        has no handle, nothing registers as a teardown key, and the teardown
        still asks the shared entry point. Opening an inert job is therefore
        what leaves the POSIX pane exactly as it was."""

        class ObservingJob(proctree_jobs.ProcessJob):
            """The real job class, recording what the session asked and what
            the inert half answered."""

            def __init__(self, name: Optional[str] = None) -> None:
                super().__init__(name)
                self.asked: List[Tuple[int, bool, Optional[bool]]] = []
                self.close_calls = 0

            def assign(self, pid, *, leader: bool = False):
                answer = super().assign(pid, leader=leader)
                self.asked.append((pid, leader, answer))
                return answer

            def close(self) -> None:
                self.close_calls += 1
                super().close()

        self._force_the_planted_driver(monkeypatch)
        monkeypatch.setattr(proctree_jobs, "JOBS_SUPPORTED", False)
        monkeypatch.setattr(proctree_jobs, "ProcessJob", ObservingJob)
        planted_children(monkeypatch, [], [(777, "openconsole.exe")])
        asked: List[Tuple[Any, ...]] = []

        async def fake_kill_session_tree(
            pid: int, reap_leader: Any, grace_seconds: float
        ) -> None:
            asked.append((pid, grace_seconds))

        monkeypatch.setattr(proctree, "kill_session_tree", fake_kill_session_tree)

        async def scenario() -> None:
            async def emit(_event: Dict[str, Any]) -> None:
                pass

            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=80, rows=24)
            job = session.job
            assert job is not None and job.available is False
            # The session still contained both processes — one call site — and
            # the half with no jobs answered ``None`` (the platform's answer,
            # not a failure), registering no teardown key for anyone.
            assert [(pid, leader) for pid, leader, _ in job.asked] == [
                (4242, True),
                (777, False),
            ]
            assert all(answer is None for _, _, answer in job.asked)
            assert proctree_jobs.job_for(4242) is None
            await session.teardown()
            assert asked == [(4242, termhub_constants.STOP_GRACE_SECONDS)]
            assert job.close_calls == 1  # idempotent close, inert handle

        asyncio.run(scenario())


# -- the Linux import promise, forced on this runner ---------------------------


class TestTheModuleImportsWithoutItsDependency:
    """*"The module must import cleanly on Linux and must not be selected there."*

    The Linux half of that cannot be run from a Windows box, so it is proven the
    way roadmap §9 rule 2 asks — by **hiding** the dependency and re-importing
    the module. If it comes back with ``winpty`` bound to ``None`` and its probe
    answering False, then the import is not what breaks a Linux suite, and the
    seam's answer there is the refusal. (The other half — POSIX selection winning
    whenever the kernel PTY answers — is tested above by forcing that probe.)
    """

    def test_a_reload_with_the_dependency_blocked_leaves_no_driver(self):
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def import_without_winpty(name, *args, **kwargs):
            if name == "winpty" or name.startswith("winpty."):
                raise ImportError("hidden for this test — the Linux state")
            return real_import(name, *args, **kwargs)

        saved = sys.modules.pop("winpty", None)
        builtins.__import__ = import_without_winpty
        try:
            module = importlib.reload(pty_conpty)
            assert module.winpty is None
            assert module.available() is False
        finally:
            builtins.__import__ = real_import
            if saved is not None:
                sys.modules["winpty"] = saved
            importlib.reload(pty_conpty)


# -- the real pseudo-console (Windows only, driver installed) ------------------
#
# The gates are csl-08's consolidated ones — conftest's markers over the pane's
# own probes, each skipping **with the seam's sentence naming the missing
# capability** rather than passing vacuously: ``windows_behaviour`` for the
# Windows process facts inside (cleanup and liveness go through the job half's
# primitives, and ``os.kill(pid, 0)`` is never used — it would kill what it
# inspects here), ``needs_windows_driver`` for the driver's own ``available()``
# answer, and ``needs_shell`` for the shell this class spawns through
# ``termhub.shells`` (a box with the extra but no Git says so). The Linux runner
# proves the Windows half's *rules* through the seam, in the classes above, with
# a planted dependency.


@pytest.mark.windows_behaviour(
    "C-D3's end-of-stream and C-D6's encoding facts can only be measured on a "
    "real pseudo-console, which only exists on Windows"
)
@pytest.mark.needs_windows_driver
@pytest.mark.needs_shell
class TestRealPseudoConsole:
    """The session over a real ConPTY and a real shell — measured, not assumed.

    These are C §6's Windows-only cases: the two ways the stream ends (C-D3),
    the resize round-trip (C-D4), C-D5's tree teardown (csl-07's shell-child and
    console-helper cases), and the non-ASCII round-trip (C-D6). The shell is
    whatever :mod:`termhub.shells` selects on this runner (Git Bash, csl-05), so
    no path here is written by the test.
    """

    def test_the_shell_exits_on_its_own_and_the_exit_frame_carries_the_code(
        self, monkeypatch, tmp_path
    ):
        """C-D3 case 1: an ``exit``/Ctrl-D ends the pump on the driver's EOF."""
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        frames: List[Dict[str, Any]] = []

        async def emit(event: Dict[str, Any]) -> None:
            frames.append(event)

        async def scenario() -> Dict[str, Any]:
            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            try:
                pump = asyncio.ensure_future(session.pump())
                session.write("exit 7\r")
                await asyncio.wait_for(pump, timeout=20.0)
                exits = [f for f in frames if f["type"] == "exit"]
                assert exits, f"no exit frame in {[f['type'] for f in frames]}"
                return exits[0]
            finally:
                _release_session(session)

        frame = asyncio.run(scenario())
        assert frame["code"] == 7

    def test_a_ctrl_d_eof_also_ends_the_pump_with_an_exit_frame(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        frames: List[Dict[str, Any]] = []

        async def emit(event: Dict[str, Any]) -> None:
            frames.append(event)

        async def scenario() -> Dict[str, Any]:
            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            try:
                pump = asyncio.ensure_future(session.pump())
                session.write("\x04")  # Ctrl-D: the shell reads an EOF and leaves
                await asyncio.wait_for(pump, timeout=20.0)
                exits = [f for f in frames if f["type"] == "exit"]
                assert exits, f"no exit frame in {[f['type'] for f in frames]}"
                return exits[0]
            finally:
                _release_session(session)

        frame = asyncio.run(scenario())
        assert frame["code"] == 0

    def test_a_teardown_requested_ends_the_pump_inside_the_grace(
        self, monkeypatch, tmp_path
    ):
        """C-D3 case 2: the pump stops within the grace, no reader left running.

        The teardown half that exists today (release the console; the job kill
        lands with csl-07) is what ends a read that is waiting on a live shell:
        the driver's next poll pass sees the released console and answers the
        far end. The wait is bounded by ``STOP_GRACE_SECONDS`` — the same number
        and rule as both hubs', unchanged by this platform.
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        grace = termhub_constants.STOP_GRACE_SECONDS

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        async def scenario() -> None:
            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            session.kill_requested = True
            pump = asyncio.ensure_future(session.pump())
            try:
                await asyncio.sleep(0.3)  # the read is genuinely blocked by then
                started = time.monotonic()
                session.pty.close_master()
                try:
                    await asyncio.wait_for(pump, timeout=grace)
                except (asyncio.TimeoutError, TimeoutError):
                    pytest.fail(
                        f"the pump outlived the {grace}s grace on a released console"
                    )
                elapsed = time.monotonic() - started
                assert elapsed < grace
                assert pump.done() and not pump.cancelled()
            finally:
                _release_session(session)

        asyncio.run(scenario())

    def test_a_resize_frame_moves_the_size_the_driver_reports(
        self, monkeypatch, tmp_path
    ):
        """The server-side round-trip (the shell's side is in the Results note)."""
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        session = terminal_session.TerminalSession(emit)
        session.spawn(cols=termhub_constants.DEFAULT_COLS, rows=termhub_constants.DEFAULT_ROWS)
        try:
            assert session.terminal_size() == (
                termhub_constants.DEFAULT_COLS,
                termhub_constants.DEFAULT_ROWS,
            )
            session.resize(132, 41)
            assert session.terminal_size() == (132, 41)
        finally:
            _release_session(session)

    def test_the_console_helper_is_named_as_a_live_process_of_this_app(
        self, monkeypatch, tmp_path
    ):
        """What csl-07's job needs: the helper's pid, not just the shell's.

        ``pywinpty`` never reports it, so the driver's child enumeration is the
        only way to name it — and csl-07's tree guarantee fails silently if that
        answer is empty (C-D5's forgotten process). Asserted here against a real
        console: two distinct pids, both live, neither the app itself.
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        session = terminal_session.TerminalSession(emit)
        session.spawn(cols=80, rows=24)
        try:
            pids = session.pty.started_pids()
            assert len(pids) >= 2, (
                f"the driver named only {pids} — the console helper beside the "
                "shell is the process a teardown must not leave behind"
            )
            assert len(set(pids)) == len(pids)
            assert os.getpid() not in pids
            for pid in pids:
                assert proctree_jobs.process_is_gone(pid) is False
        finally:
            _release_session(session)

    def test_a_disconnect_takes_the_shell_the_helper_and_a_backgrounded_child(
        self, monkeypatch, tmp_path
    ):
        """C-D5 on the platform where ``killpg`` cannot reach it (C §6's
        Windows-only case, and the ghost ``proctree``'s session-scope paragraph
        names).

        A **backgrounded** child is started through the ordinary frame loop, the
        socket drops, and the session's teardown — the shared entry point,
        answering with the job this session opened at spawn — must end the
        shell, the ``OpenConsole.exe`` helper the driver started beside it, and
        the child bash put in the background. Liveness is the shared probe,
        never ``os.kill(pid, 0)`` (which here would *kill* what it inspected).
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        monkeypatch.setattr(termhub_constants, "STOP_GRACE_SECONDS", 1.0)

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        async def scenario() -> Dict[str, Any]:
            incoming: asyncio.Queue = asyncio.Queue()

            async def receive_frame() -> Optional[Dict[str, Any]]:
                return await incoming.get()

            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            shell_pid = session.pid
            started = list(session.pty.started_pids())
            helper_pids = [pid for pid in started if pid != shell_pid]
            assert helper_pids, (
                f"the driver named no console helper beside {shell_pid} — a "
                "tree assertion without it would be the wrapper kill C-D5 is "
                "here to rule out"
            )
            job = session.job
            assert job is not None and job.available, (
                "the session opened no job on the platform that has jobs"
            )
            run = asyncio.ensure_future(session.run(receive_frame))
            try:
                await incoming.put({"type": "input", "data": "sleep 40 &\r"})
                # Wait until the backgrounded child is *in the tree* — the
                # assertion below is only about a ghost that really existed.
                members = await asyncio.to_thread(
                    _wait_for_member_outside, job, set(started)
                )
                assert members, (
                    "no child of the shell ever joined the session's job — the "
                    "spawn-time containment that makes the job a tree does not "
                    "hold for this shell"
                )
                ghost = [pid for pid in members if not proctree_jobs.process_is_gone(pid)]
                assert ghost, "the backgrounded child exited before the disconnect"
                await incoming.put(None)  # the socket drops
                reason = await asyncio.wait_for(run, timeout=20.0)
            finally:
                if not run.done():
                    run.cancel()
            return {
                "reason": reason,
                "shell": shell_pid,
                "helpers": helper_pids,
                "ghosts": ghost,
                "job": job,
            }

        outcome = asyncio.run(scenario())
        assert outcome["reason"] == "disconnect"
        # The whole tree, by the OS's own liveness answer — and the ghost is
        # the point: on Windows it is not in any group a signal could reach.
        for label, pid in (
            [("shell", outcome["shell"])]
            + [("helper", pid) for pid in outcome["helpers"]]
            + [("child", pid) for pid in outcome["ghosts"]]
        ):
            assert _wait_until_gone(pid), f"the {label} {pid} outlived the teardown"
        assert outcome["job"].handle is None, (
            "the teardown left the session's job handle open: kill-on-close "
            "would still be armed, and the pid could be recycled under it"
        )

    def test_a_shell_that_exits_on_its_own_still_takes_the_console_helper(
        self, monkeypatch, tmp_path
    ):
        """The second half of C-D5's forgotten process: the helper outlives the
        shell by design (``pywinpty`` keeps it while the console is referenced —
        measured by csl-06), so a teardown reached by a *self-exiting* shell,
        with no kill frame and no disconnect pressure, must still end it.
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        monkeypatch.setattr(termhub_constants, "STOP_GRACE_SECONDS", 1.0)

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        async def scenario() -> Dict[str, Any]:
            incoming: asyncio.Queue = asyncio.Queue()

            async def receive_frame() -> Optional[Dict[str, Any]]:
                return await incoming.get()

            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            shell_pid = session.pid
            helpers = [
                pid for pid in session.pty.started_pids() if pid != shell_pid
            ]
            assert helpers, "no console helper was named to contain"
            run = asyncio.ensure_future(session.run(receive_frame))
            try:
                await incoming.put({"type": "input", "data": "exit\r"})
                # The pump ends on the driver's EOF, the run loop reports
                # ``shell-exit``, and the teardown runs with nothing else
                # asked of it.
                reason = await asyncio.wait_for(run, timeout=20.0)
            finally:
                if not run.done():
                    run.cancel()
            return {"reason": reason, "helpers": helpers}

        outcome = asyncio.run(scenario())
        assert outcome["reason"] == "shell-exit"
        assert all(_wait_until_gone(pid) for pid in outcome["helpers"]), (
            "the console helper outlived a session whose shell exited on its "
            "own — the teardown reached the wrapper but not the driver's "
            "second process"
        )

    def test_the_teardown_returns_inside_the_grace_it_was_given(
        self, monkeypatch, tmp_path
    ):
        """The ``stop_all`` promise on Windows: *at most* ``STOP_GRACE_SECONDS``
        per stubborn shell, not per process — a session with a live child that
        no graceful step reaches is force-killed through the job and the call
        still lands inside the patched grace.
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        grace = 1.0
        monkeypatch.setattr(termhub_constants, "STOP_GRACE_SECONDS", grace)

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        async def scenario() -> float:
            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            session.write("sleep 40 &\r")
            member = await asyncio.to_thread(
                _wait_for_member_outside, session.job, set(session.pty.started_pids())
            )
            assert member, "the child never became a member of the session's job"
            started = time.monotonic()
            await session.teardown()
            elapsed = time.monotonic() - started
            # ``stop_all`` gives each session ``STOP_GRACE_SECONDS + 1`` — the
            # job's escalation ends inside the grace, the extra second is the
            # reader drain — and the observed figure on this machine is ~0.2s.
            assert elapsed < grace + 1.0, (
                f"the teardown took {elapsed:.2f}s against a {grace}s grace — "
                "the app's exit does not pay the promised ceiling per shell"
            )
            return elapsed

        asyncio.run(scenario())

    def test_a_kill_frame_ends_the_session_and_the_tree_inside_the_grace(
        self, monkeypatch, tmp_path
    ):
        """The Windows twin of ``test_kill_frame_closes_session_and_kills_pid``
        and of the grace promise, against the job-based teardown: a ``kill``
        frame ends the run loop, the shell pid is gone by the shared probe,
        and the teardown returned inside the patched grace.
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        grace = 1.0
        monkeypatch.setattr(termhub_constants, "STOP_GRACE_SECONDS", grace)

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        async def scenario() -> Tuple[str, int]:
            incoming: asyncio.Queue = asyncio.Queue()

            async def receive_frame() -> Optional[Dict[str, Any]]:
                return await incoming.get()

            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            shell_pid = session.pid
            run = asyncio.ensure_future(session.run(receive_frame))
            try:
                await asyncio.sleep(0.3)  # the pump is genuinely reading
                started = time.monotonic()
                await incoming.put({"type": "kill"})
                reason = await asyncio.wait_for(run, timeout=grace + 1.0)
                elapsed = time.monotonic() - started
            finally:
                if not run.done():
                    run.cancel()
            assert elapsed < grace + 1.0, (
                f"the kill-frame teardown took {elapsed:.2f}s against a "
                f"{grace}s grace"
            )
            return reason, shell_pid

        reason, shell_pid = asyncio.run(scenario())
        assert reason == "kill"
        assert _wait_until_gone(shell_pid), (
            f"shell {shell_pid} survived the kill frame — the job-based "
            "teardown did not reach it"
        )

    def test_where_the_driver_named_no_helper_the_shell_is_still_contained(
        self, monkeypatch, tmp_path
    ):
        """The containment promise, not the naming promise: when the helper's
        pid could not be obtained (the driver's own caution), the shell is
        still in the job and dies through the shared teardown — and the
        *missing* helper is a thing the driver says, not one the session hides.
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        monkeypatch.setattr(termhub_constants, "STOP_GRACE_SECONDS", 1.0)
        # The driver's enumeration seam, made blind for this session (the
        # hermetic rule: patch the seam's own attribute, not a path).
        monkeypatch.setattr(pty_conpty, "_child_processes", lambda: [])

        async def emit(_event: Dict[str, Any]) -> None:
            pass

        async def scenario() -> int:
            session = terminal_session.TerminalSession(emit)
            session.spawn(cols=100, rows=30)
            shell_pid = session.pid
            assert session.pty.started_pids() == (shell_pid,)
            assert shell_pid in (session.job.members() or ()), (
                "the shell was not put in the job even though the helper could "
                "not be named"
            )
            await session.teardown()
            return shell_pid

        shell_pid = asyncio.run(scenario())
        assert _wait_until_gone(shell_pid), (
            f"shell {shell_pid} survived a teardown with no named helper"
        )

    def test_a_non_ascii_echo_comes_back_as_the_same_bytes(self, monkeypatch, tmp_path):
        """C-D6, measured: the round-trip, not an assumption about UTF-8.

        The bytes written in are the characters the shell sees, and the
        characters the shell writes come back as the same bytes — accents, CJK
        and a snowman. If the console ever hands back something the driver
        cannot promise is UTF-8, this is the test that says so.
        """
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        payload = "café中文 ☃"
        gathered: List[bytes] = []

        spec = shells.select_shell()  # the pane's own choice — never a test path
        pair = pty_conpty.ConPty(100, 30)
        _child = pair.spawn_shell(
            list(spec), str(tmp_path), shells.child_environment(spec, 100, 30)
        )
        try:
            pair.write(f"echo {payload}\r".encode("utf-8"))
            deadline = time.time() + 20.0
            stream = b""
            while time.time() < deadline:
                try:
                    gathered.append(pair.read(termhub_constants.PTY_READ_BYTES))
                except pty_posix.FarEndGone:
                    break
                stream = b"".join(gathered)
                if payload.encode("utf-8") in stream:
                    break
                time.sleep(0.05)
            stream = b"".join(gathered)
            assert payload.encode("utf-8") in stream, (
                f"the pseudo-console did not round-trip {payload!r}: "
                f"{stream[-400:]!r}"
            )
        finally:
            _release_session_pair(pair)


def _wait_for_member_outside(job: Any, known: set, timeout: float = 10.0):
    """The job's members that are not in ``known``, once there are any.

    The OS's own list of who is in the tree — the answer a backgrounded child
    has to show up in, since a shell's ``$!`` is an MSYS pid and not the
    Windows pid the job names.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        members = job.members()
        if members is not None:
            extra = [pid for pid in members if pid not in known]
            if extra:
                return extra
        time.sleep(0.05)
    return None


def _wait_until_gone(pid: Optional[int], timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proctree_jobs.process_is_gone(pid):
            return True
        time.sleep(0.05)
    return proctree_jobs.process_is_gone(pid)


def _release_session(session: terminal_session.TerminalSession) -> None:
    if session.job is not None:  # csl-07: the session owns the handle
        session.job.close()
    _release_pair(session.pty)


def _release_session_pair(pair: Any) -> None:
    _release_pair(pair)


def _release_pair(pair: Any) -> None:
    """End a real session's driver claim on the console and on its processes.

    csl-06 deliberately implements no kill — csl-07 wires the job that owns the
    tree — so these tests clean up with the job half's own primitives (the calls
    csl-07 will make), and the driver's ``started_pids()`` is what makes the list
    complete: the shell *and* the console helper. Liveness comes from the shared
    Windows probe, never ``os.kill(pid, 0)``, which would kill what it inspects.
    """
    pids = pair.started_pids()
    with suppress(Exception):
        pair.close_master()
    for pid in pids:
        with suppress(Exception):
            proctree_jobs.terminate_process(pid)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if all(proctree_jobs.process_is_gone(pid) for pid in pids):
            break
        time.sleep(0.05)
