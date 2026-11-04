"""The shared process-tree teardown, and both mechanisms behind it.

``proctree`` is one teardown with two halves, so this file asserts the contract
once per half: the **group/session** half against a real POSIX process tree, and
the **Job Object** half against a real Windows one. The parts that belong to
neither platform — which half an entry point dispatches to, and the escalation's
shape — are forced on **both** runners through the named seam
(``proctree_jobs.JOBS_SUPPORTED``, its primitives, its tunables), never by
pretending to be the other platform (roadmap §9 rule 2). A skip's sentence names
the invariant it would have checked.

Every process here is a throw-away child this file spawns and cleans up — never
a hub, never a language server — and every tunable is patched through the module
that owns it.

Liveness always comes from :func:`src.core.proctree_jobs.process_is_gone`.
``os.kill(pid, 0)`` is *not* a liveness probe on Windows, where CPython maps a
plain signal number to ``TerminateProcess`` and the "probe" therefore kills the
process it inspected; one test below asserts no such call exists in the suite.
"""

import ast
import asyncio
import os
import logging
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, List, Optional, Tuple

import pytest

from src.core import proctree
from src.core import proctree_jobs

from tests.conftest import proc_fs_probe, skip_unless

# Long enough that nothing this file kills dies by luck, short enough that a
# leaked child is an inconvenience rather than a hang.
LIFETIME_SECONDS = 40

# The grace a teardown test runs with. Short, because the promise under test is
# that the escalation *returns inside* it.
TEARDOWN_GRACE_SECONDS = 0.5

# A plain long-lived child.
SLEEP_SRC = "import time; time.sleep(%d)" % LIFETIME_SECONDS

# A child that spawns a descendant of its own and reports its pid — the ghost
# AGENTS.md names: *"children a hub spawned of its own never outlive the app
# that started them"*. Cross-platform (nothing here is POSIX-specific), because
# the job half needs the same tree to prove it reaches descendants.
#
# The descendant is born only when a line arrives on stdin, because containment
# in a job is decided at a process's **creation** (measured: assigning a parent
# later does not migrate a descendant that already exists). A test that wants
# its descendant inside the job therefore has to put the parent in the job
# *first* — so the gate is what makes that order testable rather than lucky.
TREE_SRC = (
    "import subprocess, sys, time\n"
    "sys.stdin.readline()\n"
    "child = subprocess.Popen([sys.executable, '-c', %r])\n"
    "print(child.pid, flush=True)\n"
    "time.sleep(%d)\n" % (SLEEP_SRC, LIFETIME_SECONDS)
)


async def _noop_reap(deadline: float) -> None:
    """A reap step that resolves at once, for a process this test does not own."""


def _reap_of(proc: subprocess.Popen) -> Callable[[float], Awaitable[None]]:
    """The caller's reap step, in the shape ``proctree`` documents.

    Bounded by the deadline it is handed, exactly as both hubs' and the
    terminal's is, so a test of the grace also proves the step cannot hold it up.
    """

    async def reap(deadline: float) -> None:
        loop = asyncio.get_event_loop()
        with suppress(Exception):
            await asyncio.wait_for(
                asyncio.to_thread(proc.wait),
                timeout=max(deadline - loop.time(), 0.01),
            )

    return reap


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0,
                step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


@pytest.fixture
def spawned() -> Iterator[List[subprocess.Popen]]:
    """Every child a test starts, killed and reaped when the test ends.

    CPython waits for the ``Popen`` objects a process leaves unfinished at exit,
    so a leaked long-lived child would *hang* the suite — indistinguishable from
    a teardown bug. The descendant a test's child spawned is not a ``Popen`` of
    this process, so it is tracked and killed by pid too: a test that fails to
    tear a tree down must not leave ghosts for the next one.
    """
    started: List[subprocess.Popen] = []
    yield started
    for proc in started:
        for pid in (proc.pid,) + tuple(getattr(proc, "reported_children", ())):
            if proctree_jobs.process_is_gone(pid):
                continue
            if os.name == "nt":
                # ``signal.SIGKILL`` does not exist here, and the module's own
                # primitive is the only kill this file uses on Windows — a
                # cleanup, never an assertion.
                proctree_jobs.terminate_process(pid)
            else:
                with suppress(Exception):
                    os.kill(pid, signal.SIGKILL)
        with suppress(Exception):
            proc.wait(timeout=10)


def _spawn(spawned: List[subprocess.Popen], src: str,
           new_session: bool = False,
           gated: bool = False) -> subprocess.Popen:
    """Start one throw-away child, remembering it so the fixture can clean up
    whatever the test asserts.

    ``new_session=True`` leads a group and session of its own — the shape a hub
    child and a terminal shell are spawned in, and the precondition for the
    group- and session-scoped teardowns to mean anything.

    ``gated=True`` opens the child's stdin so its descendant is born only when
    :func:`_release` says so, which is how a test controls whether the
    descendant exists before or after the job assignment.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", src],
        stdin=subprocess.PIPE if gated else None,
        stdout=subprocess.PIPE if gated else None,
        text=True,
        start_new_session=new_session,
    )
    proc.reported_children: Tuple[int, ...] = ()  # type: ignore[attr-defined]
    spawned.append(proc)
    return proc


def _release(proc: subprocess.Popen) -> int:
    """Let a gated child spawn its descendant; return that descendant's pid."""
    assert proc.stdin is not None and proc.stdout is not None, (
        "the child was not started gated"
    )
    proc.stdin.write("go\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line.strip().isdigit(), (
        f"the child never reported its descendant's pid (got {line!r})"
    )
    pid = int(line)
    proc.reported_children = (pid,)  # type: ignore[attr-defined]
    return pid


# -- one entry point, two mechanisms -----------------------------------------


class TestDispatch:
    """The callers do not change: each entry point lands in the right half.

    Both directions are forced by patching the seam through the module that
    owns it, so either runner proves the branch its own platform does not take.
    """

    def test_both_entry_points_take_the_job_branch_when_the_seam_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Forcing ``JOBS_SUPPORTED`` moves *both* scopes, on either runner.

        The two scopes are POSIX ideas, so on the job platform they answer the
        same way — and the pid is handed to the job half as **the caller's
        child**, never signalled as a group id: on that platform a ``killpg``
        aimed at it would signal the app's own group.
        """
        asked: List[int] = []

        async def fake_kill(pid: Optional[int], reap_leader: Any,
                            grace_seconds: Optional[float]) -> None:
            asked.append(int(pid or 0))

        monkeypatch.setattr(proctree_jobs, "JOBS_SUPPORTED", True)
        monkeypatch.setattr(proctree_jobs, "kill_job_tree", fake_kill)

        def no_group_signals(*args: Any) -> None:
            pytest.fail(
                "the group path ran on the job platform, where a pid is not a "
                "pgid — it would signal the app itself"
            )

        monkeypatch.setattr(proctree.os, "killpg", no_group_signals,
                            raising=False)
        for entry in (proctree.kill_process_tree, proctree.kill_session_tree):
            asyncio.run(entry(4242, _noop_reap, TEARDOWN_GRACE_SECONDS))
        assert asked == [4242, 4242]

    @pytest.mark.posix_behaviour(
        "the group half signals a process group, which exists only on POSIX"
    )
    def test_the_group_half_runs_untouched_when_the_seam_says_no_jobs(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The same forcing the other way: the POSIX path is not diverted.

        The graceful signal is still the group, still SIGTERM, and the job half
        is never entered.
        """
        signalled: List[Tuple[int, int]] = []
        monkeypatch.setattr(proctree_jobs, "JOBS_SUPPORTED", False)

        async def no_job_tree(*args: Any) -> None:
            pytest.fail("the job half ran on a platform with process groups")

        monkeypatch.setattr(proctree_jobs, "kill_job_tree", no_job_tree)

        def record(pgid: int, sig: int) -> None:
            signalled.append((pgid, sig))
            raise ProcessLookupError  # "the group is already gone": end it here

        monkeypatch.setattr(proctree.os, "killpg", record)
        asyncio.run(proctree.kill_process_tree(4242, _noop_reap,
                                               TEARDOWN_GRACE_SECONDS))
        assert signalled == [(4242, signal.SIGTERM)]

    def test_the_seam_here_is_a_capability_answer(self):
        """What this runner answers, so a skip on the other one is explainable.

        A caller asks :func:`jobs_available` *before* it spawns into a job: the
        patchable flag **and** the calls being bound, so a half-working job is
        never offered.
        """
        assert proctree_jobs.jobs_available() is proctree_jobs.JOBS_SUPPORTED

    def test_the_job_is_inert_when_the_seam_says_there_are_no_jobs(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A spawn site can be written once for both platforms.

        With the seam off, the object opens as nothing and :meth:`assign`
        answers ``None`` — the platform's answer, not a failure the caller has
        to branch around — and closing it twice is still not an error. This is
        what keeps today's POSIX callers byte-for-byte unchanged.
        """
        monkeypatch.setattr(proctree_jobs, "JOBS_SUPPORTED", False)
        job = proctree_jobs.ProcessJob("no-jobs-here")
        assert job.available is False
        assert job.handle is None
        assert job.assign(os.getpid(), leader=True) is None
        assert job.members() is None
        assert proctree_jobs.job_for(os.getpid()) is None
        job.close()
        job.close()

    def test_the_primitives_answer_nothing_when_the_calls_are_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The module is safe to hold on a runner that will never use it.

        ``proctree`` imports both halves everywhere, so a missing mechanism is a
        value, never an import error or a raise at the first teardown.
        """
        monkeypatch.setattr(proctree_jobs, "_api", None)
        assert proctree_jobs.jobs_available() is False
        assert proctree_jobs.open_job() is None
        assert proctree_jobs.assign_pid_to_job("handle", 1234) is False
        assert proctree_jobs.job_member_pids("handle") is None
        assert proctree_jobs.terminate_job("handle") is False
        assert proctree_jobs.terminate_process(1234) is False
        assert proctree_jobs.close_job_handle("handle") is False


# -- the escalation's shape: one contract, both halves -----------------------


class _StubJob:
    """What :func:`proctree_jobs.terminate_and_drain` asks of a job.

    The order, the injected reap and the grace deadline are platform-neutral;
    the member list is what decides them. Handing the loop a schedule lets a
    test produce an outcome no real process would deliver in time — a tree that
    never drains — and assert the escalation anyway.
    """

    def __init__(self, schedule: List[Optional[Tuple[int, ...]]],
                 leaders: Tuple[int, ...] = (111,)) -> None:
        self._schedule = list(schedule)
        self.leaders = list(leaders)
        self.handle = "stub-handle"
        self.probes = 0
        self.closes = 0

    def members(self) -> Optional[Tuple[int, ...]]:
        self.probes += 1
        return self._schedule[min(self.probes - 1, len(self._schedule) - 1)]

    def close(self) -> None:
        self.closes += 1


@pytest.fixture
def job_kills(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """The job half's two kill steps, recorded, with its poll shortened.

    Patched through ``proctree_jobs`` — the module that owns them — so the
    escalation is observed as a shape, with no real process to time.
    """
    events: List[str] = []
    monkeypatch.setattr(proctree_jobs, "JOB_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        proctree_jobs, "terminate_process",
        lambda pid: (events.append(f"leader:{pid}"), True)[1],
    )
    monkeypatch.setattr(
        proctree_jobs, "terminate_job",
        lambda handle: (events.append(f"job:{handle}"), True)[1],
    )
    return events


class TestJobEscalation:
    """Graceful first, bounded wait, then force — the POSIX half's shape."""

    def test_the_graceful_step_aims_at_the_leader_alone(
        self, job_kills: List[str]
    ):
        """No member is killed by the *graceful* step; the force step reaches them.

        There is no advisory signal for a console process on Windows, so the
        first step terminates the caller's own child and leaves the rest of the
        job alone: a descendant that exits because its parent did gets the grace
        for exactly that. Here the tree never goes, so the force step — the
        job-wide terminate — is what ends the escalation.
        """
        async def reap(deadline: float) -> None:
            job_kills.append("reap")

        job = _StubJob([(7, 8, 9)])  # never drains
        started = time.monotonic()
        asyncio.run(proctree_jobs.terminate_and_drain(job, reap, 0.3))
        assert job_kills == ["leader:111", "reap", "job:stub-handle"]
        assert job.closes == 1, "the job handle was left open by the teardown"
        assert time.monotonic() - started < 2.0, "the teardown outlived its grace"

    def test_a_tree_that_drains_early_is_never_forced(
        self, job_kills: List[str]
    ):
        """The grace is a budget, not a wait.

        The POSIX half returns as soon as its group probe is empty; the job half
        must do the same on its own probe, or every teardown would pay the full
        grace and the terminal's *"at most STOP_GRACE_SECONDS per stubborn
        shell"* would become "per shell".
        """
        job = _StubJob([(7,), ()])  # a member, then empty
        started = time.monotonic()
        asyncio.run(proctree_jobs.terminate_and_drain(job, _noop_reap, 30.0))
        assert not [e for e in job_kills if e.startswith("job:")], (
            "the force step ran on a tree that had already gone"
        )
        assert job.closes == 1
        assert time.monotonic() - started < 2.0

    def test_the_leader_is_reaped_before_the_tree_is_probed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """``proctree``'s rule, honoured by the job half too.

        *"The leader is reaped before the tree is probed"*: the reap step is
        injected, handed this loop's deadline, called once, and is the first
        await — the member query (the job half's probe) is what follows it.
        """
        order: List[str] = []
        handed: List[Tuple[float, float]] = []
        monkeypatch.setattr(proctree_jobs, "JOB_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(proctree_jobs, "terminate_process",
                            lambda pid: (order.append("leader"), True)[1])
        monkeypatch.setattr(proctree_jobs, "terminate_job",
                            lambda handle: True)

        async def reap(deadline: float) -> None:
            order.append("reap")
            # Read the same loop's clock from inside the step, so the deadline
            # can be checked against it rather than against the wall clock —
            # ``loop.time()`` has no fixed relation to ``time.monotonic()``.
            handed.append((deadline, asyncio.get_event_loop().time()))

        job = _StubJob([()])
        real_members = job.members

        def probed() -> Optional[Tuple[int, ...]]:
            order.append("probe")
            return real_members()

        job.members = probed  # type: ignore[method-assign]
        asyncio.run(proctree_jobs.terminate_and_drain(job, reap, 0.3))
        assert order[:2] == ["leader", "reap"], (
            "the tree was probed, or the leader killed, out of order"
        )
        assert "probe" in order[2:], "the member query ran before the reap"
        assert len(handed) == 1, "the reap step is called exactly once"
        deadline, loop_now = handed[0]
        assert loop_now < deadline <= loop_now + 0.3, (
            "the deadline is not this loop's clock inside the grace — the step "
            "could not bound itself by it"
        )

    def test_a_reap_step_that_overshoots_or_raises_does_not_escape(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The step is bounded by the deadline, and its failure is swallowed.

        Both hubs' and the terminal's reap waits that way; a teardown that let a
        stuck or failed wait escape would abandon the force step — the one part
        that reaches the descendants.
        """
        monkeypatch.setattr(proctree_jobs, "JOB_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(proctree_jobs, "terminate_process", lambda pid: True)
        monkeypatch.setattr(proctree_jobs, "terminate_job", lambda handle: True)

        async def hanging_reap(deadline: float) -> None:
            loop = asyncio.get_event_loop()
            # Deliberately past its own budget: the caller's poll is what has
            # to keep the whole teardown inside the grace.
            await asyncio.sleep(max(deadline - loop.time(), 0.0) + 0.1)

        async def raising_reap(deadline: float) -> None:
            raise RuntimeError("the wait failed")

        for step in (hanging_reap, raising_reap):
            job = _StubJob([()])
            started = time.monotonic()
            asyncio.run(proctree_jobs.terminate_and_drain(job, step, 0.3))
            assert time.monotonic() - started < 2.0, step.__name__
            assert job.closes == 1, step.__name__

    def test_the_grace_fallback_is_read_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """``JOB_GRACE_SECONDS`` is a module attribute, not a captured default.

        The constants-module rule both hubs follow: a test that shortens it must
        move the fallback, or the fallback is the one number nothing can test.
        """
        monkeypatch.setattr(proctree_jobs, "JOB_GRACE_SECONDS", 0.2)
        monkeypatch.setattr(proctree_jobs, "JOB_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(proctree_jobs, "terminate_process", lambda pid: True)
        monkeypatch.setattr(proctree_jobs, "terminate_job", lambda handle: True)
        job = _StubJob([(7,)])  # never drains, so the grace is what ends it
        started = time.monotonic()
        asyncio.run(proctree_jobs.terminate_and_drain(job, _noop_reap, None))
        assert time.monotonic() - started < 2.0
        assert job.closes == 1

    def test_an_uncontained_child_is_escalated_and_named(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """A spawn site that forgot the job has to be findable, not silent.

        With no job there is no reachable tree on that platform, so the leader
        alone is escalated, twice if it will not die, and the log line is the
        only signal that its descendants are unbounded.
        """
        monkeypatch.setattr(proctree_jobs, "JOBS_SUPPORTED", True)
        monkeypatch.setattr(proctree_jobs, "job_for", lambda pid: None)
        killed: List[int] = []
        monkeypatch.setattr(proctree_jobs, "terminate_process",
                            lambda pid: (killed.append(pid), True)[1])
        monkeypatch.setattr(proctree_jobs, "JOB_POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr(proctree_jobs, "process_is_gone",
                            lambda pid: False)  # stubborn: force must follow
        with caplog.at_level(logging.WARNING):
            asyncio.run(proctree_jobs.kill_job_tree(4242, _noop_reap, 0.2))
        assert killed == [4242, 4242], (
            "the leader was not escalated twice — graceful then force"
        )
        assert any("never assigned to a job" in r.getMessage()
                   for r in caplog.records), (
            "the missing containment was not reported"
        )


# -- the liveness probe: one answer, both platforms --------------------------


class TestLivenessProbe:
    def test_it_agrees_with_a_child_this_test_spawned_and_then_lost(
        self, spawned: List[subprocess.Popen]
    ):
        """A live pid is not gone; the same pid, once it exited, is."""
        proc = _spawn(spawned, "import sys; print(1)")
        assert proctree_jobs.process_is_gone(proc.pid) is False, (
            "a running child was reported gone"
        )
        proc.wait(timeout=10)
        assert _wait_until(lambda: proctree_jobs.process_is_gone(proc.pid)), (
            "a pid that exited is still reported live"
        )

    def test_a_working_child_survives_being_asked(
        self, spawned: List[subprocess.Popen]
    ):
        """The probe is a query, not a side effect.

        Asked repeatedly about a child that is still working, it must say "not
        gone" **and** leave the child working. That is the whole reason the suite
        may not use a zero signal instead: on Windows that call ends the process
        it was inspecting (see the guard below).
        """
        proc = _spawn(spawned, "import time; time.sleep(2)")
        for _ in range(20):
            assert proctree_jobs.process_is_gone(proc.pid) is False
            time.sleep(0.02)
        assert proc.poll() is None, "the probe interrupted the process it read"

    def test_no_pid_is_gone_by_definition(self):
        assert proctree_jobs.process_is_gone(None) is True
        assert proctree_jobs.process_is_gone(0) is True

    def test_no_test_in_the_suite_uses_a_zero_signal_as_a_probe(self):
        """The guard the platform trap demands.

        ``os.kill(pid, 0)`` is ``TerminateProcess`` on Windows: a "liveness"
        check written that way would kill what it inspected and then truthfully
        report it gone. Checked on the parsed call, not the text, because this
        file's docstring names the idiom it forbids. (The probe's own POSIX
        fallback is in ``src``, where signal 0 really is a query.)
        """
        offenders: List[str] = []
        for path in sorted(Path(__file__).parent.glob("test_*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or len(node.args) != 2:
                    continue
                called = node.func
                if not (
                    isinstance(called, ast.Attribute)
                    and called.attr == "kill"
                    and isinstance(called.value, ast.Name)
                    and called.value.id == "os"
                ):
                    continue
                argument = node.args[1]
                if isinstance(argument, ast.Constant) and argument.value == 0:
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            "ask proctree_jobs.process_is_gone instead; zero-signal probes at "
            f"{offenders}"
        )


# -- the POSIX half: the promise, unchanged ----------------------------------


@pytest.mark.posix_behaviour(
    "the group and session scopes, killpg and SIGTERM/SIGKILL are POSIX facts; "
    "the Windows job equivalent is asserted by the forced-seam tests above and "
    "by the real-process class below"
)
class TestPosixHalf:
    """The escalation the hubs and the terminal have always run.

    Not weakened or reshaped for the second mechanism: the same graceful→force
    order, the leader reaped before the probe, and the session scope reaching a
    job that runs in a process group of its own.
    """

    def test_a_group_member_that_ignores_the_graceful_signal_is_forced(
        self, spawned: List[subprocess.Popen], monkeypatch: pytest.MonkeyPatch
    ):
        """The wrapper-kill case, on the mechanism that has groups.

        The descendant ignores SIGTERM on purpose, so only reaching the *group*
        and escalating can clear it — the same ghost the job half exists for on
        the platform without groups.
        """
        monkeypatch.setattr(proctree, "GROUP_POLL_INTERVAL_SECONDS", 0.01)
        proc = _spawn(spawned, _IGNORING_TREE_SRC, new_session=True, gated=True)
        grandchild = _release(proc)
        assert not proctree_jobs.process_is_gone(grandchild)
        asyncio.run(proctree.kill_process_tree(
            proc.pid, _reap_of(proc), TEARDOWN_GRACE_SECONDS
        ))
        assert _wait_until(lambda: proctree_jobs.process_is_gone(grandchild)), (
            "the teardown killed the wrapper but not its group"
        )

    def test_the_session_scope_reaches_a_job_of_its_own(
        self, spawned: List[subprocess.Popen], monkeypatch: pytest.MonkeyPatch
    ):
        """The ghost ``proctree``'s docstring was written for.

        A job-controlling shell gives every job a process group of its own, all
        inside the shell's session, so ``killpg`` on the shell's group reaches
        the shell and nothing else. The session probe is what finds the
        backgrounded job — the promise the terminal's teardown depends on, and
        the reason it calls this entry point rather than the group one.
        """
        monkeypatch.setattr(proctree, "GROUP_POLL_INTERVAL_SECONDS", 0.01)
        shell = _spawn(spawned, _NEW_GROUP_CHILD_SRC, new_session=True,
                       gated=True)
        job_pid = _release(shell)
        assert not proctree_jobs.process_is_gone(job_pid)
        # The job is *not* in the shell's group: that is what makes this test
        # about the session and not a repeat of the one above.
        assert os.getpgid(job_pid) != shell.pid, (
            "the job does not lead a group of its own, so the session scope is "
            "not being exercised"
        )
        assert job_pid in proctree.session_members(shell.pid), (
            "the session probe does not see the job it exists to find"
        )
        asyncio.run(proctree.kill_session_tree(
            shell.pid, _reap_of(shell), TEARDOWN_GRACE_SECONDS
        ))
        assert _wait_until(lambda: proctree_jobs.process_is_gone(job_pid)), (
            "a job of the session outlived the session teardown"
        )

    @skip_unless(proc_fs_probe, "uses /proc to name a session's members")
    def test_the_session_probe_reads_this_process_session(self):
        """The read the session half is built on, seen from the inside.

        The answer must exclude the reader — the scan that finds a shell's
        backgrounded job must not find the app doing the finding.
        """
        sid = os.getsid(0)
        assert os.getpid() not in proctree.session_members(sid)


# A tree whose members ignore SIGTERM, so the escalation to SIGKILL is what
# clears it. POSIX-only source (signal traps), used only by ``TestPosixHalf``;
# gated like ``TREE_SRC`` so the two tree helpers are one shape.
_IGNORING_TREE_SRC = (
    "import signal, subprocess, sys, time\n"
    "sys.stdin.readline()\n"
    "child = subprocess.Popen([sys.executable, '-c', "
    "'import signal, time\\n'\n"
    "                          'signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n'\n"
    "                          'time.sleep(%d)'])\n"
    "print(child.pid, flush=True)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(%d)\n" % (LIFETIME_SECONDS, LIFETIME_SECONDS)
)

# A job running in a process group of its own, inside its parent's *session* —
# what a job-controlling shell does for a backgrounded job. POSIX-only
# (``setpgid``). The **job** prints its own pid, after it has led its group:
# had the parent printed it right after ``Popen``, the test would be reading a
# pid whose pgid is still the parent's, and its "this job leads its own group"
# assertion would be a race rather than a fact.
_NEW_GROUP_CHILD_SRC = (
    "import subprocess, sys, time\n"
    "sys.stdin.readline()\n"
    "child = subprocess.Popen([sys.executable, '-c', "
    "'import os, time\\n'\n"
    "                          'os.setpgid(0, 0)\\n'\n"
    "                          'print(os.getpid(), flush=True)\\n'\n"
    "                          'time.sleep(%d)'])\n"
    "time.sleep(%d)\n" % (LIFETIME_SECONDS, LIFETIME_SECONDS)
)


# -- the job half, against real processes (Windows) --------------------------


@pytest.mark.windows_behaviour(
    "the invariant is the Job Object mechanism acting on real processes: it has "
    "no POSIX equivalent, and its kill of a descendant is what the group half "
    "gets from killpg"
)
class TestJobHalfAgainstRealProcesses:
    """Observed, not assumed, on the platform that has the mechanism.

    The seam-forced tests above prove the *shape*; these prove the thing the
    shape is for — a process the app never spawned dies with the tree.
    """

    def test_terminating_the_job_takes_the_grandchild_too(
        self, spawned: List[subprocess.Popen]
    ):
        """AGENTS.md's case, where ``killpg`` cannot reach it.

        The parent spawns a descendant of its own **after** it was put in the
        job, and the test never assigns that descendant to anything: a member's
        children join the same job by themselves. The job's termination must
        end both.
        """
        job = proctree_jobs.ProcessJob("grandchildren")
        assert job.available, "the seam said jobs exist; the job disagrees"
        parent = _spawn(spawned, TREE_SRC, gated=True)
        assert job.assign(parent.pid, leader=True) is True
        grandchild = _release(parent)
        assert not proctree_jobs.process_is_gone(grandchild), (
            "the descendant never came up"
        )
        members = job.members()
        assert members is not None and grandchild in members, (
            f"the descendant the jobbed parent spawned is not itself a member "
            f"(members={members}, want {grandchild} in it): the job does not "
            "reach a tree, so terminating it cannot reach one either"
        )
        assert proctree_jobs.terminate_job(job.handle) is True
        assert _wait_until(lambda: proctree_jobs.process_is_gone(parent.pid)), (
            "the leader survived the job termination"
        )
        assert _wait_until(lambda: proctree_jobs.process_is_gone(grandchild)), (
            f"descendant {grandchild} survived the job termination — the "
            "teardown killed the wrapper but not its descendants"
        )
        assert _wait_until(lambda: job.members() == ()), (
            "the job still lists members after all of them died"
        )
        job.close()

    def test_a_descendant_born_before_the_assignment_is_not_migrated(
        self, spawned: List[subprocess.Popen]
    ):
        """The boundary of the mechanism, recorded because it binds every caller.

        Measured on this machine: containment is decided at a process's
        **creation**. Assigning a parent after its descendant exists lists only
        the parent, and terminating the job leaves the earlier descendant
        alive. That is the fact behind the seam's rule that *"the job is created
        by the caller's side of the seam … a process becomes a member when it is
        spawned"* — a hub or a session that opened its job at spawn has no such
        gap, and a caller that assigns late cannot claim tree coverage.
        """
        job = proctree_jobs.ProcessJob("late-assignment")
        parent = _spawn(spawned, TREE_SRC, gated=True)
        grandchild = _release(parent)  # born while the job holds nothing
        assert job.assign(parent.pid, leader=True) is True
        members = job.members()
        assert members is not None and parent.pid in members
        assert grandchild not in members, (
            "a descendant that pre-dated the assignment is now contained: the "
            "mechanism changed, and this is the moment to simplify the seam's "
            "spawn-time rule"
        )
        assert proctree_jobs.terminate_job(job.handle) is True
        assert _wait_until(lambda: proctree_jobs.process_is_gone(parent.pid))
        assert proctree_jobs.process_is_gone(grandchild) is False, (
            "the earlier descendant died with the job after all — update the "
            "rule and this file's docstring together"
        )
        job.close()

    def test_a_member_that_outlives_the_graceful_step_is_forced_within_it(
        self, spawned: List[subprocess.Popen], monkeypatch: pytest.MonkeyPatch
    ):
        """The grace deadline is honoured, and the call returns inside it.

        The graceful step kills the leader; a descendant the leader only
        spawned does not go with it, so it is still there when the grace runs
        out and the force step is what ends it.
        """
        monkeypatch.setattr(proctree_jobs, "JOB_POLL_INTERVAL_SECONDS", 0.01)
        job = proctree_jobs.ProcessJob("stubborn")
        parent = _spawn(spawned, TREE_SRC, gated=True)
        assert job.assign(parent.pid, leader=True) is True
        grandchild = _release(parent)
        started = time.monotonic()
        asyncio.run(proctree_jobs.terminate_and_drain(
            job, _reap_of(parent), 0.4
        ))
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, (
            f"the teardown took {elapsed:.2f}s against a 0.4s grace"
        )
        assert proctree_jobs.process_is_gone(parent.pid)
        assert proctree_jobs.process_is_gone(grandchild), (
            "the descendant that survived the graceful step was not forced"
        )
        assert job.handle is None, "the teardown left the job handle open"

    def test_closing_the_last_handle_is_a_kill(
        self, spawned: List[subprocess.Popen]
    ):
        """The *"dies with the app"* half of the mechanism, observed.

        kill-on-close is what makes an app that exits without tearing down still
        take its trees. If this stops being true, every sentence about the app's
        exit becomes a promise about the OS instead.
        """
        job = proctree_jobs.ProcessJob("kill-on-close")
        proc = _spawn(spawned, SLEEP_SRC)
        assert job.assign(proc.pid) is True
        assert not proctree_jobs.process_is_gone(proc.pid)
        job.close()
        assert _wait_until(lambda: proctree_jobs.process_is_gone(proc.pid)), (
            "closing the job's last handle did not kill its member — "
            "kill-on-close is not in effect"
        )

    def test_a_process_the_caller_did_not_spawn_is_assignable_by_pid(
        self, spawned: List[subprocess.Popen]
    ):
        """The answer csl-07 depends on.

        The terminal's session must contain the shell **and** the console helper
        its driver started: neither is a process the session class spawned with
        its own hands, and a helper cannot be put in a job before it exists. So
        assignment by *pid*, after the fact, has to work.
        """
        job = proctree_jobs.ProcessJob("driver-spawned")
        helper = _spawn(spawned, SLEEP_SRC)
        assert job.assign(helper.pid) is True, (
            "a process the caller did not spawn cannot be put in the job — the "
            "console helper would be outside the session's tree"
        )
        assert helper.pid in (job.members() or ())
        assert proctree_jobs.terminate_job(job.handle) is True
        assert _wait_until(lambda: proctree_jobs.process_is_gone(helper.pid))
        job.close()

    def test_the_registry_keys_the_job_by_the_caller_child(
        self, spawned: List[subprocess.Popen]
    ):
        """The teardown is handed a pid, so that pid is the lookup key.

        A helper assigned without ``leader=True`` is *contained* but is not a
        key — ``proctree``'s entry points are called with the caller's own
        child. And the key is gone once the job is released, so a recycled pid
        can never be pointed at a closed job.
        """
        job = proctree_jobs.ProcessJob("registry")
        parent = _spawn(spawned, SLEEP_SRC)
        helper = _spawn(spawned, SLEEP_SRC)
        assert job.assign(parent.pid, leader=True) is True
        assert job.assign(helper.pid) is True
        assert proctree_jobs.job_for(parent.pid) is job
        assert proctree_jobs.job_for(helper.pid) is None, (
            "a helper became a lookup key; the teardown is called with the "
            "caller's child"
        )
        job.close()
        assert proctree_jobs.job_for(parent.pid) is None, (
            "a closed job is still reachable by pid — a recycled pid would be "
            "torn down against the wrong tree"
        )
        assert proctree_jobs.terminate_job(job.handle) is False
        job.close()  # idempotent after the release

    def test_the_shared_entry_point_reaches_a_real_tree(
        self, spawned: List[subprocess.Popen], monkeypatch: pytest.MonkeyPatch
    ):
        """The whole path unforced: entry point → registry → job → tree.

        ``proctree.kill_process_tree`` called exactly as a hub calls it — the
        point of the task is that the callers do not change.
        """
        monkeypatch.setattr(proctree_jobs, "JOB_POLL_INTERVAL_SECONDS", 0.01)
        job = proctree_jobs.ProcessJob("through-the-entry-point")
        parent = _spawn(spawned, TREE_SRC, gated=True)
        assert job.assign(parent.pid, leader=True) is True
        grandchild = _release(parent)
        asyncio.run(proctree.kill_process_tree(
            parent.pid, _reap_of(parent), TEARDOWN_GRACE_SECONDS
        ))
        assert _wait_until(lambda: proctree_jobs.process_is_gone(grandchild)), (
            "the shared entry point did not reach the descendant"
        )
        assert job.handle is None

    def test_a_process_already_in_a_job_is_contained_by_both(
        self, spawned: List[subprocess.Popen]
    ):
        """What the job refuses to contain — measured, and recorded.

        Verified on this machine: there is **no** "already in another job"
        refusal. Assigning a jobbed process to a second job succeeds, both jobs
        list it, and terminating either one kills it. So the seam has no failure
        to report for it — and the discipline is the app's instead: one child,
        one job, because two jobs over one tree means two owners of its death.
        """
        first = proctree_jobs.ProcessJob("first")
        second = proctree_jobs.ProcessJob("second")
        proc = _spawn(spawned, SLEEP_SRC)
        assert first.assign(proc.pid) is True
        assert second.assign(proc.pid) is True, (
            "the second assignment no longer succeeds: the behaviour recorded "
            "here has changed, and the seam should report it"
        )
        assert proc.pid in (first.members() or ()), (
            "the first job stopped listing a process it contains: membership "
            "is no longer additive, which changes what two jobs over one mean"
        )
        assert proc.pid in (second.members() or ())
        assert proctree_jobs.terminate_job(first.handle) is True
        assert _wait_until(lambda: proctree_jobs.process_is_gone(proc.pid)), (
            "terminating the first job did not reach a member both jobs list"
        )
        first.close()
        second.close()

    def test_an_exited_process_assigns_false_and_registers_nothing(
        self, spawned: List[subprocess.Popen]
    ):
        """A failed assignment is not a failed spawn.

        The usual reason is that the process already exited — not a tree to
        contain — so the answer is False, the caller keeps going, and no pid is
        registered under it.
        """
        job = proctree_jobs.ProcessJob("dead-pid")
        proc = _spawn(spawned, "import sys; print(1)")
        proc.wait(timeout=10)
        assert _wait_until(lambda: proctree_jobs.process_is_gone(proc.pid))
        assert job.assign(proc.pid, leader=True) is False
        assert proctree_jobs.job_for(proc.pid) is None
        assert proctree_jobs.assign_pid_to_job(job.handle, 0) is False
        job.close()
