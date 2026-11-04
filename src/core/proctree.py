"""Process-tree teardown, shared by the hubs and the terminal sessions, on
either platform.

AGENTS.md: *"Hub teardown kills process \\*trees\\*, not wrappers"*. The escalation
is the same everywhere a child leads its own process group
(``start_new_session=True``): SIGTERM the group, reap the leader, probe the
group until the grace period expires, then SIGKILL the stragglers. The hubs and
the terminal share this one implementation because two copies is how they drift.

The **mechanism** behind that escalation is platform-half, chosen by capability
at each entry point, never by an ``if os.name`` inside the algorithm:

* **POSIX** — process **groups** and signals, this module's own half:
  ``killpg`` reaches every descendant because they all inherited the leader's
  group, and the probe reads ``/proc``.
* **Windows** — :mod:`src.core.proctree_jobs`, a **Job Object** with
  kill-on-close (roadmap §5: *"not a `killpg` shim"*). There is no group to
  signal there (``start_new_session`` is silently ignored), so containment is
  established at spawn time — the caller puts the child in a job — and the
  teardown terminates the job, which reaches the grandchildren the same way.
  Both entry points land in the job half there; the scope distinction below is
  a POSIX one.

The leader's pid *is* both the pgid and (with ``start_new_session=True``) the
session id, which is why no ``getpgid()`` is used — and none could work here: it
fails once the leader has been reaped, even while other members are still
running. The leader is reaped *before* the tree is probed, because a tree only
disappears once every member is dead **and** reaped, so an unreaped leader would
keep the probe — and so the whole grace period — alive.

Two scopes share that escalation (both are how a *POSIX* tree is found; the job
half needs neither, because the job already names its members):

* :func:`kill_process_tree` signals the leader's **group** — right for a hub
  child, whose descendants inherit its group.
* :func:`kill_session_tree` signals the whole **session**. An interactive shell
  runs with job control: every job it starts gets a process group of its own,
  all inside the shell's session, so a ``killpg`` on the shell's group reaches
  the shell and nothing else. A terminal teardown that stopped there would let
  a backgrounded ``python -m http.server`` outlive the tab — the exact ghost
  this module exists to prevent. Its members are found by scanning ``/proc``
  for the session id, so no syscall on the (possibly already reaped) leader is
  needed.

Callers differ only in how they wait for their own child, so that step is
injected: each killer takes a coroutine that reaps the leader and returns by
deadline — the same injected step on both halves, which is what keeps the two
implementations honest to one contract.
"""

import asyncio
import logging
import os
import signal
from contextlib import suppress
from typing import Awaitable, Callable, List

from . import proctree_jobs

logger = logging.getLogger(__name__)

# How often to probe for surviving group members between SIGTERM and SIGKILL.
GROUP_POLL_INTERVAL_SECONDS = 0.01

# A reap-the-leader step: given the deadline (``loop.time()``-based) so it can
# bound its own wait, resolves once the leader is reaped or the wait failed.
ReapLeader = Callable[[float], Awaitable[None]]


def group_alive(pgid: int) -> bool:
    """Probe the group with signal 0 — only ESRCH proves it is gone.

    EPERM is ambiguous (a foreign-euid member, or a zombie this process may not
    signal), so it counts as alive and keeps waiting rather than pretending the
    group drained.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


async def kill_process_tree(
    pgid: int, reap_leader: ReapLeader, grace_seconds: float
) -> None:
    """SIGTERM the group, reap the leader, SIGKILL whatever outlives it.

    ``pgid`` is the pid of the group leader as spawned with
    ``start_new_session=True``. ``reap_leader`` must return (by deadline) so the
    probe below can tell a live member from an unreaped leader.

    On Windows this is :func:`src.core.proctree_jobs.kill_job_tree` — the same
    escalation over the job that contains that pid, because there is no group
    for ``pgid`` to name. The pid is *not* treated as a group id: a ``killpg``
    aimed at it there would signal the app's own group, so the dispatch happens
    before anything is signalled.
    """
    if proctree_jobs.JOBS_SUPPORTED:
        await proctree_jobs.kill_job_tree(pgid, reap_leader, grace_seconds)
        return
    loop = asyncio.get_event_loop()
    deadline = loop.time() + grace_seconds
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return  # the whole group is already gone
    except PermissionError:
        logger.warning("No permission to signal process group %d", pgid)
    with suppress(Exception):
        await reap_leader(deadline)
    while loop.time() < deadline:
        if not group_alive(pgid):
            return
        await asyncio.sleep(GROUP_POLL_INTERVAL_SECONDS)
    # ESRCH: died since the last probe. EPERM: we killed what we were allowed to.
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def session_members(sid: int) -> List[int]:
    """Pids whose *session id* is ``sid``, this process excluded.

    ``/proc/<pid>/stat`` field 6 is the session id, and a process only changes
    it with ``setsid`` — which is exactly what ``start_new_session=True`` does —
    so every member found descends from that session's leader. A job that
    ``setsid``s out (a double-forking daemon) leaves the session and so opts out
    by design, the same exception the group path documents. Zombies keep a
    ``/proc`` entry, which is what makes the probe wait for them to be reaped
    rather than declaring the tree gone while one still holds the terminal.
    """
    members: List[int] = []
    sid_text = str(sid)
    me = os.getpid()
    try:
        names = os.listdir("/proc")
    except OSError:  # no procfs: the caller treats "no members" as already gone
        return members
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as stat:
                # 'pid (comm) state ppid pgrp session …' — comm may hold
                # spaces, hence the rsplit. A process gone mid-scan is simply
                # not a member any more.
                fields = stat.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 3 and fields[3] == sid_text:
            members.append(pid)
    return members


def _signal_session(sid: int, sig: int) -> None:
    """Signal the session: its leader group first, then every other member.

    The leader leads its own group (``killpg`` is atomic for it); the other job
    groups are reached one pid at a time — a job that already exited raises
    ESRCH and is skipped.
    """
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(sid, sig)
    for pid in session_members(sid):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)


async def kill_session_tree(
    sid: int, reap_leader: ReapLeader, grace_seconds: float
) -> None:
    """SIGTERM the session, reap the leader, SIGKILL whatever outlives it.

    The session-scoped twin of :func:`kill_process_tree`, for leaders whose
    children run in process groups of their own (a job-controlling shell). The
    probe reads ``/proc`` rather than ``killpg … 0`` because that tree outlives
    its leader's *group* by construction: the members are exactly the other
    groups. ``sid`` is the leader's pid (``start_new_session=True``).

    On Windows the scope question does not arise: the session's job contains
    the shell *and* the console helper beside it, and every job the shell
    started is inside them by inheritance — that is the promise the session
    scope makes here, moved to the caller's side of the seam (the session
    creates the job at spawn). ``kill_session_tree`` and ``kill_process_tree``
    answer the same way on that platform, and both land in the job half.
    """
    if proctree_jobs.JOBS_SUPPORTED:
        await proctree_jobs.kill_job_tree(sid, reap_leader, grace_seconds)
        return
    loop = asyncio.get_event_loop()
    deadline = loop.time() + grace_seconds
    _signal_session(sid, signal.SIGTERM)
    with suppress(Exception):
        await reap_leader(deadline)
    while loop.time() < deadline:
        if not session_members(sid):
            return
        await asyncio.sleep(GROUP_POLL_INTERVAL_SECONDS)
    # Whatever survives to here traps or ignores SIGTERM: escalate.
    _signal_session(sid, signal.SIGKILL)
