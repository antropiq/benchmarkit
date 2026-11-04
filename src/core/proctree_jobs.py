"""The Windows half of the shared process-tree teardown: a Job Object.

AGENTS.md: *"Teardown kills process \\*trees\\*, not wrappers"*. The POSIX half
(:mod:`src.core.proctree`) does that with process **groups**, and on Windows
there is no group to signal: CPython's Windows ``_execute_child`` takes
``unused_start_new_session``, so a child spawned with ``start_new_session=True``
leads nothing, and a kill aimed at the child the app spawned reaches that
wrapper alone — every process it spawned of its own stays alive. This module is
roadmap §5's decided replacement: *"the Windows equivalent is a Job Object with
kill-on-close, not a ``killpg`` shim"*.

A job moves the boundary from a signal to a **handle**:

* a process joins the job when it is spawned (:meth:`ProcessJob.assign`), and
  from then on every process it **creates** joins the same job by itself —
  measured on this machine: assign the parent, then let it spawn a grandchild,
  and the grandchild is a member nobody assigned. That is how a tree is reached
  with no group, and it is why the boundary is a *spawn-time* one: containment
  is decided when a process is created, so assigning a parent after its child
  exists covers the parent only (also measured). A caller that opens its job at
  the spawn has no such gap; one that assigns late cannot claim tree coverage;
* the teardown terminates **the job**, so the grandchildren die with it;
* kill-on-close ends the members when the app's last handle to the job closes,
  so an app that exits without tearing down still takes its trees with it (also
  measured: closing the handle of a job holding a live ``sleep`` killed it
  immediately). The handle is therefore the guarantee *and* the thing that must
  not leak: a ctypes handle is a plain integer, so nothing closes it for the
  caller, and a caller that replaces a child (a hub restart) closes the old job
  rather than dropping the reference.

The escalation shape is the one :mod:`src.core.proctree` documents and the POSIX
half runs: a graceful step, the **injected** reap step bounded by a deadline,
then probe the tree, then force — the whole thing inside the grace the caller
named, with :data:`JOB_FORCE_DRAIN_SHARE` of it held back so the force step can
be issued *and* verified rather than fired at the deadline. Two differences
are stated rather than smoothed over:

* **there is no advisory signal for a console process here.** The graceful step
  is the caller's own protocol shutdown, which both hubs already perform *before*
  asking for the teardown; inside the teardown the first step terminates the
  leader **alone** — the same call today's Windows fallback makes — and the
  grace window is the tree's chance to leave with it. What this module adds over
  the fallback is the force step reaching the whole job.
* **a job has no unreaped state.** An exited member leaves
  :meth:`ProcessJob.members` at once, waited-for or not, so the job's verdict is
  "every member is dead" where the group probe's is "dead **and** reaped". The
  caller's reap step still runs first, because it is what frees the caller's own
  handle on its child.

``ctypes`` only — no new dependency (roadmap §8 scoped B-D5 as *"none
(ctypes/Win32 for jobs)"*). The module imports on every platform, because
``proctree`` imports both halves unconditionally: the *absence* of the mechanism
is an answer (:func:`jobs_available`), never an ImportError. The primitives
below (:func:`open_job`, :func:`assign_pid_to_job`, :func:`job_member_pids`,
:func:`terminate_job`, :func:`terminate_process`, :func:`close_job_handle`) and
the tunables are the seam a test patches to force either half on one runner
(roadmap §9 rule 2).
"""

import asyncio
import ctypes
import logging
import os
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -- tunables: patch them through this module, never a copy of them -----------

# How often to ask the job for its members between the graceful step and the
# force step: the job twin of ``proctree.GROUP_POLL_INTERVAL_SECONDS``.
JOB_POLL_INTERVAL_SECONDS = 0.01

# The grace a job teardown runs when its caller names none — the same number and
# the same rule as both hubs' and the terminal's ``STOP_GRACE_SECONDS``, so one
# wait is the budget everywhere a tree is torn down.
JOB_GRACE_SECONDS = 5.0

# The exit code the job's kills stamp on what they terminate.
JOB_TERMINATE_EXIT_CODE = 1

# The share of the grace reserved for the **force** step's own drain. The
# graceful phase may poll no further than the rest, so a teardown that had to
# escalate still returns inside ``grace_seconds`` — the promise the terminal's
# *"at most STOP_GRACE_SECONDS per stubborn shell"* depends on. Without a
# reserve, the force step would be issued at the deadline and never verified.
JOB_FORCE_DRAIN_SHARE = 0.25

# The dispatch seam: does this platform have the job mechanism at all? Named and
# patchable, so a test on either runner forces the other half's branch (roadmap
# §9 rule 2) without pretending to be that platform.
JOBS_SUPPORTED = os.name == "nt"

# A reap-the-leader step, exactly as :mod:`src.core.proctree` states the
# contract: given the deadline (``loop.time()``-based) so it can bound its own
# wait, resolves once the leader is reaped or the wait failed. Injected, so the
# two halves stay honest to one contract.
ReapLeader = Callable[[float], Awaitable[None]]


# -- the Win32 surface -------------------------------------------------------

# Declared from ``ctypes`` primitives rather than ``ctypes.wintypes`` so the
# module imports (and its names mean the same thing) on a runner that has no
# Windows types: ``c_size_t`` is ``ULONG_PTR``/``DWORD_PTR``, ``c_int`` is
# ``BOOL``. A handle is ``c_void_p`` — with ctypes' default ``int`` return a
# 64-bit handle is truncated, which is why every call below is prototyped.
_HANDLE = ctypes.c_void_p
_BOOL = ctypes.c_int
_DWORD = ctypes.c_uint32
_UINT = ctypes.c_uint32
_LPVOID = ctypes.c_void_p
_LPCWSTR = ctypes.c_wchar_p

# Rights: assignment needs PROCESS_SET_QUOTA, a kill needs PROCESS_TERMINATE, a
# liveness probe needs only the *limited* query right (it works on a process at
# another elevation level).
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# The information classes, as verified on this machine: class 9 is
# ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` — the only one that carries
# kill-on-close, and the ones around it answered ``ERROR_BAD_LENGTH`` — and
# class 3 is ``JOBOBJECT_BASIC_PROCESS_ID_LIST``, the job's live members.
_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_PROCESS_ID_LIST_CLASS = 3

# ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: closing the last handle is a kill.
_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

# ``ERROR_MORE_DATA``: the member list is bigger than the buffer offered.
_ERROR_MORE_DATA = 234

# ``GetExitCodeProcess``'s "still running" answer. A program may also *exit*
# with 259, so the liveness probe never caches an answer: it opens a handle on
# the pid it is being asked about and reads the code from that.
_STILL_ACTIVE = 259


class _BasicLimitInformation(ctypes.Structure):
    """``JOBOBJECT_BASIC_LIMIT_INFORMATION`` — only ``LimitFlags`` is used; the
    fields around it exist so size and layout match what the API expects."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", _DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", _DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", _DWORD),
        ("SchedulingClass", _DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` — the only class that carries
    kill-on-close. The length passed to the call comes from ``sizeof``, so the
    runner's own pointer width is what the API sees (144 bytes measured here)."""

    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _ProcessIdList(ctypes.Structure):
    """``JOBOBJECT_BASIC_PROCESS_ID_LIST`` — the job's live members.

    The entries are ``ULONG_PTR`` (``c_size_t``): a 32-bit array would truncate
    a pid on x64. The struct's array is a fixed capacity while the real reply is
    variable-sized, so the capacity is what the call is told was offered; a job
    holding more answers ``ERROR_MORE_DATA``, which the probe reads as "at least
    this many members" rather than as an empty tree. Nothing the app starts (a
    shell and its jobs, a language server and its workers) nears the capacity.
    """

    _CAPACITY = 256

    _fields_ = [
        ("NumberOfProcessIds", _DWORD),
        ("MaximumNumberOfProcessIds", _DWORD),
        ("ProcessIds", ctypes.c_size_t * _CAPACITY),
    ]


def _bind_job_calls() -> Optional[Dict[str, Any]]:
    """The kernel32 calls, each with its prototype, or ``None`` if unavailable.

    Bound once at import, so a missing call is one answer (``jobs_available()``
    is False) rather than a failure at the first teardown.
    """
    if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows runner
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        signatures = {
            "create_job_object": (
                "CreateJobObjectW", _HANDLE, [_LPVOID, _LPCWSTR]),
            "set_information_job_object": (
                "SetInformationJobObject", _BOOL,
                [_HANDLE, ctypes.c_int, _LPVOID, _DWORD]),
            "query_information_job_object": (
                "QueryInformationJobObject", _BOOL,
                [_HANDLE, ctypes.c_int, _LPVOID, _DWORD, ctypes.POINTER(_DWORD)]),
            "assign_process_to_job_object": (
                "AssignProcessToJobObject", _BOOL, [_HANDLE, _HANDLE]),
            "terminate_job_object": (
                "TerminateJobObject", _BOOL, [_HANDLE, _UINT]),
            "open_process": (
                "OpenProcess", _HANDLE, [_DWORD, _BOOL, _DWORD]),
            "get_exit_code_process": (
                "GetExitCodeProcess", _BOOL, [_HANDLE, ctypes.POINTER(_DWORD)]),
            "terminate_process": (
                "TerminateProcess", _BOOL, [_HANDLE, _UINT]),
            "close_handle": ("CloseHandle", _BOOL, [_HANDLE]),
        }
        bound: Dict[str, Any] = {}
        for key, (name, restype, argtypes) in signatures.items():
            function = getattr(kernel32, name)
            function.restype = restype
            function.argtypes = argtypes
            bound[key] = function
        return bound
    except (AttributeError, OSError):  # pragma: no cover
        logger.warning("kernel32 job-object calls unavailable; job teardown off")
        return None


# ``proctree`` imports this module on every platform, so the absence of the
# mechanism has to be data, never an ImportError. Every primitive re-reads this
# at call time, which makes it the seam a test patches to ask "what if the calls
# were never bound?" — the one branch no non-Windows runner can reach by itself.
_api: Optional[Dict[str, Any]] = _bind_job_calls() if os.name == "nt" else None


def jobs_available() -> bool:
    """Can a process tree be held in a job here at all?

    The capability answer a caller consults *before* spawning into a job, the
    same shape as the terminal's driver seam: ``JOBS_SUPPORTED`` is the patchable
    half, the bound calls are the half no test can fake into existing.
    """
    return JOBS_SUPPORTED and _api is not None


# -- the job primitives (the seam) -------------------------------------------

def open_job(name: Optional[str] = None) -> Optional[Any]:
    """Create a job with kill-on-close, or ``None`` where jobs do not exist.

    The limit is set while the job is still empty, so no member is ever born
    into a job that would not die with the app. Either step failing closes the
    handle again: a half-configured job is not returned to be trusted.
    """
    api = _api
    if api is None:
        return None
    handle = api["create_job_object"](None, name)
    if not handle:
        logger.warning("CreateJobObjectW failed: %s", ctypes.GetLastError())
        return None
    limits = _ExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = _LIMIT_KILL_ON_JOB_CLOSE
    if not api["set_information_job_object"](
        handle,
        _EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        logger.warning(
            "SetInformationJobObject(kill-on-close) failed: %s",
            ctypes.GetLastError(),
        )
        api["close_handle"](handle)
        return None
    return handle


def assign_pid_to_job(handle: Any, pid: int) -> bool:
    """Make the process ``pid`` a member of the job.

    Called after the spawn, which is what lets a process the *driver* started
    (the terminal's shell, the console helper beside it) join a job the app
    opened: a process cannot be assigned before it exists. Descendants created
    *after* this call need no call of their own — they inherit the job — but
    descendants that already existed do, and are better never left behind:
    containment is decided at creation (see the module docstring).
    """
    api = _api
    if api is None or not handle or pid <= 0:
        return False
    process = api["open_process"](
        _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid
    )
    if not process:
        # The usual reason is that the process already exited, which is not a
        # tree to contain. The caller's probe settles whether anything of it is
        # left, so this stays a log line and never a failed spawn.
        logger.debug("OpenProcess(%d) for job assignment failed: %s",
                     pid, ctypes.GetLastError())
        return False
    try:
        assigned = bool(api["assign_process_to_job_object"](handle, process))
    finally:
        api["close_handle"](process)
    if not assigned:
        logger.warning(
            "AssignProcessToJobObject(pid %d) failed: %s",
            pid, ctypes.GetLastError(),
        )
    return assigned


def job_member_pids(handle: Any) -> Optional[Tuple[int, ...]]:
    """The job's live members, or ``None`` when the job could not be asked.

    An exited member leaves the list immediately, waited-for or not: a Windows
    process has no unreaped state, the difference this half documents against
    the group probe. ``ERROR_MORE_DATA`` means the tree has more members than
    the buffer holds, and the reply fills exactly ``MaximumNumberOfProcessIds``
    of them (measured: ``n`` still reports the *total*, larger than what was
    written) — so the count is clamped by that, and the result is a non-empty
    prefix: the probe keeps waiting rather than declaring a crowded tree drained.
    """
    api = _api
    if api is None or not handle:
        return None
    listing = _ProcessIdList()
    returned = _DWORD()
    if not api["query_information_job_object"](
        handle,
        _PROCESS_ID_LIST_CLASS,
        ctypes.byref(listing),
        ctypes.sizeof(listing),
        ctypes.byref(returned),
    ):
        error = ctypes.GetLastError()
        if error != _ERROR_MORE_DATA:
            logger.warning("QueryInformationJobObject(members) failed: %s", error)
            return None
    written = min(int(listing.MaximumNumberOfProcessIds), _ProcessIdList._CAPACITY)
    listed = min(int(listing.NumberOfProcessIds), written)
    return tuple(int(listing.ProcessIds[i]) for i in range(listed))


def terminate_job(handle: Any, exit_code: int = JOB_TERMINATE_EXIT_CODE) -> bool:
    """Terminate **every member of the job** — the force step, and the reason a
    grandchild dies with the wrapper.

    The call is asynchronous: it returns once the termination is *requested*, so
    members are still listed for a moment afterwards. :func:`terminate_and_drain`
    is what waits for them to actually go.
    """
    api = _api
    if api is None or not handle:
        return False
    terminated = bool(api["terminate_job_object"](handle, exit_code))
    if not terminated:
        logger.warning("TerminateJobObject failed: %s", ctypes.GetLastError())
    return terminated


def terminate_process(pid: int, exit_code: int = JOB_TERMINATE_EXIT_CODE) -> bool:
    """Terminate one process and nothing else — the job half's graceful step.

    A console process has no advisory signal, so the first phase of the
    escalation is the same call today's Windows fallback already makes on its own
    child. It deliberately leaves the rest of the tree alone: a member that
    exits because its parent did gets the grace to do that, and one that does
    not is what the force step exists for.
    """
    api = _api
    if api is None or pid <= 0:
        return False
    process = api["open_process"](_PROCESS_TERMINATE, False, pid)
    if not process:
        # Already gone, or not ours to signal: either way the graceful step has
        # nothing to do and the probe below decides.
        return False
    try:
        terminated = bool(api["terminate_process"](process, exit_code))
    finally:
        api["close_handle"](process)
    if not terminated:
        logger.warning(
            "TerminateProcess(%d) failed: %s", pid, ctypes.GetLastError()
        )
    return terminated


def close_job_handle(handle: Any) -> bool:
    """Release one job handle. With kill-on-close set, releasing it is a kill."""
    api = _api
    if api is None or not handle:
        return False
    return bool(api["close_handle"](handle))


def process_is_gone(pid: Optional[int]) -> bool:
    """Is ``pid`` no longer a live process? The cross-platform liveness probe.

    The tests need one, because the usual idiom is a weapon on the other
    platform: CPython maps a plain signal number to ``TerminateProcess`` on
    Windows, so ``os.kill(pid, 0)`` there **kills** the process it meant to
    inspect. Nothing in this suite may use it as a probe.

    * **Windows**: ``OpenProcess`` + ``GetExitCodeProcess`` — the answer the OS
      gives, needing only the limited query right. A process the app has not
      waited for still counts as gone once it has exited, because a Windows
      process has no unreaped state.
    * **POSIX**: ``/proc/<pid>/stat``'s state, so a zombie counts as gone — it
      is dead, still holds a name, and signal 0 still answers "there". Where
      there is no ``procfs``, signal 0 *is* the probe on this platform
      (``EPERM`` means a process this user may not signal, so it is not gone).
    """
    if not pid or pid <= 0:
        return True
    if os.name == "nt":
        api = _api
        if api is None:  # pragma: no cover - kernel32 has these calls
            # Cannot be asked. "Still there" makes a test fail loudly rather
            # than pass on an assumption.
            return False
        process = api["open_process"](
            _PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not process:
            return True
        code = _DWORD()
        try:
            if not api["get_exit_code_process"](process, ctypes.byref(code)):
                return True  # it went between the open and the query
        finally:
            api["close_handle"](process)
        return int(code.value) != _STILL_ACTIVE
    try:  # pragma: no cover - the other runner
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as stat:
            # 'pid (comm) state …' — comm may hold spaces, hence the rsplit.
            state = stat.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        if os.path.isdir("/proc"):
            return True  # procfs is there and the entry is not: the pid is gone
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return True
    return state == "Z"


# -- one job per child tree ---------------------------------------------------

# The lookup the shared entry points use: a teardown is handed a *pid*, and the
# job that contains it is found here. Keyed by the pid the caller registered with
# ``assign(pid, leader=True)`` — the process the caller spawned itself, which is
# the pid ``proctree``'s entry points are given.
_JOBS_BY_LEADER: Dict[int, "ProcessJob"] = {}


def job_for(pid: Optional[int]) -> Optional["ProcessJob"]:
    """The job holding the tree led by ``pid``, if one was ever opened for it."""
    if pid is None:
        return None
    return _JOBS_BY_LEADER.get(pid)


class ProcessJob:
    """One job object, holding one child's whole tree.

    Created **by the caller's side of the seam**, before the spawn: a process
    joins a job when it is created, so containment is decided by whoever starts
    the process, not by whoever later kills it. The caller keeps the object for
    as long as its child lives — a hub singleton holds one per child, a terminal
    session one per session — and this stays true across a hub restart, which
    rebuilds a hub's internals without replacing the object the app holds: the
    job belongs to the *child*, and :meth:`close` is what releases it when that
    child is replaced.

    Inert where the platform has no jobs: :attr:`available` is False and
    :meth:`assign` answers ``None``, so a spawn site can be written once for
    both platforms and the POSIX path stays exactly what it is today.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._handle: Optional[Any] = open_job(name) if jobs_available() else None
        self._leaders: List[int] = []
        self._closed = False

    def __repr__(self) -> str:  # a raw handle in a log line is not a clue
        return (
            f"<ProcessJob {self.name or '(unnamed)'} "
            f"leaders={list(self._leaders)} {'closed' if self._closed else 'open'}>"
        )

    @property
    def available(self) -> bool:
        """Does this job actually exist on this platform?"""
        return self._handle is not None

    @property
    def handle(self) -> Optional[Any]:
        return self._handle

    @property
    def leaders(self) -> Tuple[int, ...]:
        """The pids registered with ``leader=True`` — the caller's own children."""
        return tuple(self._leaders)

    def assign(self, pid: Optional[int], *, leader: bool = False) -> Optional[bool]:
        """Contain ``pid`` in the job; call it right after each spawn.

        ``leader=True`` marks the caller's **own** child, the pid the shared
        teardown entry points are later handed — pass it for that one process
        and leave it out for a helper the driver started next to it.

        Answers ``None`` where there is no job mechanism (the platform's answer,
        not a failure), ``True`` when the process is contained, and ``False``
        when it is not — already-exited, or already in another job. The caller
        keeps its process either way: a failed assignment costs the tree
        guarantee, which the teardown then reports, and is not a reason to fail
        a spawn.
        """
        if not jobs_available() or self._handle is None:
            return None
        if not pid or not assign_pid_to_job(self._handle, pid):
            return False
        if leader and pid not in self._leaders:
            self._leaders.append(pid)
            _JOBS_BY_LEADER[pid] = self
        return True

    def members(self) -> Optional[Tuple[int, ...]]:
        """The job's live members; ``None`` when the job could not be asked."""
        if self._handle is None:
            return None
        return job_member_pids(self._handle)

    def close(self) -> None:
        """Release the handle — idempotent, and with kill-on-close it is a kill.

        The teardown calls it. A caller that abandons a child **without** a
        teardown (a hub restart that replaces its session) must call it too: a
        ctypes handle is a plain integer, so dropping the reference leaks it and
        leaves the old tree's kill-on-close armed against a handle nobody has.
        """
        if self._closed:
            return
        self._closed = True
        for pid in self._leaders:
            if _JOBS_BY_LEADER.get(pid) is self:
                del _JOBS_BY_LEADER[pid]
        self._leaders.clear()
        handle, self._handle = self._handle, None
        if handle:
            close_job_handle(handle)


# -- the escalation -----------------------------------------------------------

async def terminate_and_drain(
    job: ProcessJob,
    reap_leader: ReapLeader,
    grace_seconds: Optional[float] = None,
) -> None:
    """Terminate the job's tree with the escalation ``proctree`` documents.

    ``grace_seconds`` defaults to :data:`JOB_GRACE_SECONDS`, read at call time so
    a test that patches the module attribute moves the fallback too.

    Graceful step (the leader **alone** — there is no advisory signal for a
    console process here), then the injected reap bounded by a deadline, then
    poll the job's own member list, then terminate **the job**: every member,
    and every descendant it inherited, dies — which is what ``killpg`` buys the
    POSIX half.

    The force is asynchronous, so it gets a bounded drain of its own, and the
    graceful phase stops short of the deadline by :data:`JOB_FORCE_DRAIN_SHARE`
    of the grace so that drain has room inside the *same* deadline: the
    teardown returns within ``grace_seconds`` whether or not it escalated, which
    is what the terminal's *"at most STOP_GRACE_SECONDS per stubborn shell"*
    sentence needs. ``close()`` at the end is the last guarantee — kill-on-close
    puts a kill over whatever the terminate call had not finished taking — and a
    tree still listed at the deadline is reported, not assumed gone.
    """
    loop = asyncio.get_event_loop()
    grace = JOB_GRACE_SECONDS if grace_seconds is None else grace_seconds
    deadline = loop.time() + grace
    graceful_deadline = deadline - grace * JOB_FORCE_DRAIN_SHARE
    # The graceful step aims at the caller's own child only: a member that
    # leaves because its parent did gets the grace to do it, and one that does
    # not is what the force step is for.
    for pid in job.leaders:
        terminate_process(pid)
    with suppress(Exception):
        await reap_leader(graceful_deadline)

    # Poll the *job*, not the leader: the ghost this module exists for is a
    # member the leader spawned, and the leader may already be gone by itself.
    while loop.time() < graceful_deadline:
        members = job.members()
        if members is not None and not members:
            job.close()
            return
        await asyncio.sleep(JOB_POLL_INTERVAL_SECONDS)

    # Whatever survives to here took no notice of the graceful step: escalate
    # to the whole tree, then drain what the asynchronous terminate left listed.
    handle = job.handle
    members = job.members()
    if handle:
        terminate_job(handle)
        while loop.time() < deadline and members:
            await asyncio.sleep(JOB_POLL_INTERVAL_SECONDS)
            members = job.members()
        if members:
            # Still listed at the deadline. close() below puts kill-on-close
            # over them again; naming it keeps the leak loud rather than
            # pretending the tree went.
            logger.warning("job %r still holds %d member(s) at the deadline",
                           job, len(members))
    job.close()


async def kill_job_tree(
    pid: Optional[int],
    reap_leader: ReapLeader,
    grace_seconds: Optional[float] = None,
) -> None:
    """The job half's twin of ``proctree``'s two entry points.

    ``pid`` is the caller's own child — the one it registered with
    ``assign(pid, leader=True)`` at spawn time — which is also what
    ``proctree.kill_process_tree``/``kill_session_tree`` are handed, so a caller
    changes nothing about how it asks. Both entry points land here on Windows:
    the group and the session are POSIX ideas, while the job's membership is
    exactly the scope the session half promises (the shell, the console helper
    beside it, and everything the shell started).

    A child that was never put in a job has **no reachable tree** on this
    platform — the group signal that would reach it does not exist — so the
    leader alone is escalated and the missing containment is logged rather than
    silently accepted. That is the leak the job exists to close, and a spawn
    site that forgot the call has to be findable.
    """
    job = job_for(pid)
    if job is not None:
        await terminate_and_drain(job, reap_leader, grace_seconds)
        return

    loop = asyncio.get_event_loop()
    grace = JOB_GRACE_SECONDS if grace_seconds is None else grace_seconds
    deadline = loop.time() + grace
    if pid:
        terminate_process(pid)
    with suppress(Exception):
        await reap_leader(deadline)
    while loop.time() < deadline:
        if process_is_gone(pid):
            break
        await asyncio.sleep(JOB_POLL_INTERVAL_SECONDS)
    if pid and not process_is_gone(pid):
        terminate_process(pid)
    logger.warning(
        "pid %s was never assigned to a job: its descendants are not bounded",
        pid,
    )
