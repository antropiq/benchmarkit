"""pol-05: a workspace gets its policy file the moment it becomes active.

Seeding rather than prompting: ``ensure_policy_file`` leaves a default
``<root>/.nojae/read-write-policy.json`` behind at the two moments a folder
becomes the app's working location — boot, and the settings switch — so the
syntax arrives as a worked example the user can read in the tree, and the first
listing of the new root is already filtered by rules they can see and edit.

Those seeded rules *are* the visibility mechanism: pol-00 D7 removed the
hardcoded ignored-directory constant, so ``node_modules`` or ``__pycache__`` are
shown or hidden only because a rule says so. That is why the content below is a
Python literal rendered with ``json.dumps`` rather than a hand-maintained blob:
the same bytes go through pol-04's loader, and a test asserts they round-trip —
otherwise the two halves of this feature could drift apart silently, the
written file and the read one agreeing only by luck.

Two non-negotiables, both easy to get wrong later:

* **the create is exclusive** (``O_CREAT | O_EXCL``, one syscall), so a
  hand-edited policy is never clobbered and two concurrent switches have one
  winner. A ``if os.path.isfile(): …`` before the write would be a TOCTOU and
  buys nothing the flag does not already buy;
* **never called from ``config.path_location()``** — that is a per-request
  *reader*, and a reader that writes is how a stray folder turns up in a
  project someone only meant to browse.

A seed that cannot happen (read-only mount, a root owned by someone else, no
disk) is a **note** for the UI, not a reason to refuse opening a project:
everything here answers a dict and raises nothing.
"""

import json
import os
import stat as stat_module
from typing import Any, Dict, Tuple

from . import constants, fscase, policy_file
from .constants import policy_shared as shared

# The seeded document, in the shape pol-04 validates: ``version``, a ``readme``
# (JSON has no comments, and this file is a user's first look at the syntax, so
# the syntax explains itself from inside the file) and the rules.
#
# The mode split is deliberate and matches pol-00's reasoning: ``hidden`` for
# pure derived noise nobody ever wants to browse (version-control internals,
# caches), ``ro`` for dependency and build trees whose *documentation* is the
# reason to keep them visible — browsing ``node_modules`` to read a README is
# legitimate, editing it is not.
DEFAULT_POLICY_README: Tuple[str, ...] = (
    "benchmarkit reads this file when it browses this folder.",
    "Rules are matched in order and the LAST match wins, so a later rule carves",
    "an exception out of an earlier one:",
    '    {"path": "**/node_modules", "mode": "ro"},',
    '    {"path": "**/node_modules/**/README.md", "mode": "rw"}',
    "path is relative to this folder. * stays inside one name, ** crosses",
    "directories, and a rule naming a folder covers everything inside it.",
    "mode: hidden = not shown and not reachable; ro = browsable and readable but",
    "nothing under it changes; rw = the default, and what an unmatched path gets.",
    ".nojae itself is always hidden and is never listed by a rule.",
    "This file is machine-local: it is gitignored, and benchmarkit does not edit it.",
)

DEFAULT_POLICY_RULES: Tuple[Dict[str, str], ...] = (
    {"path": ".git", "mode": shared.HIDDEN},
    {"path": ".hg", "mode": shared.HIDDEN},
    {"path": ".svn", "mode": shared.HIDDEN},
    {"path": "**/__pycache__", "mode": shared.HIDDEN},
    {"path": "**/.pytest_cache", "mode": shared.HIDDEN},
    {"path": "**/.mypy_cache", "mode": shared.HIDDEN},
    {"path": "**/.ruff_cache", "mode": shared.HIDDEN},
    {"path": "**/node_modules", "mode": shared.READ_ONLY,
     "reason": "browse the docs, edit nothing"},
    {"path": "**/.venv", "mode": shared.READ_ONLY},
    {"path": "**/venv", "mode": shared.READ_ONLY},
    {"path": "**/dist", "mode": shared.READ_ONLY},
    {"path": "**/build", "mode": shared.READ_ONLY},
)


def policy_file_path(root: str) -> str:
    """Where the policy lives under ``root`` — built from ``constants``, per call.

    Both names are module attributes precisely so a test can move the lookup
    (the ``mcphub.constants`` convention this repo uses for every cap), so
    nothing here may capture them at import time.
    """
    return os.path.join(root, constants.POLICY_DIRNAME, constants.POLICY_FILENAME)


def default_policy_text() -> str:
    """The bytes :func:`ensure_policy_file` writes, rendered from the literal."""
    document: Dict[str, Any] = {
        "version": constants.POLICY_VERSION,
        "readme": list(DEFAULT_POLICY_README),
        "policies": [dict(rule) for rule in DEFAULT_POLICY_RULES],
    }
    return json.dumps(document, indent=2) + "\n"


def _created(path: str) -> Dict[str, Any]:
    return {"state": "created", "path": path, "error": None}


def _existing(path: str) -> Dict[str, Any]:
    return {"state": "existing", "path": path, "error": None}


def _absent(path: str) -> Dict[str, Any]:
    return {"state": "absent", "path": path, "error": None}


def _failed(path: str, reason: str) -> Dict[str, Any]:
    return {"state": "failed", "path": path, "error": reason}


def _is_regular_file(path: str) -> bool:
    """Whether ``path`` is an ordinary file (a folder there is a finding).

    pol-04 reports ``is a folder, not a file`` as an error finding when it tries
    to read one, so the seeder says the same thing rather than calling the
    obstruction a policy the loader would then reject — the two readers of this
    folder must not disagree about what is standing there.
    """
    try:
        return stat_module.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def ensure_policy_file(root: str) -> Dict[str, Any]:
    """Create ``<root>/.nojae/read-write-policy.json`` if absent. Never overwrites.

    Returns ``{"state": "created" | "existing" | "failed", "path": …,
    "error": Optional[str]}`` — a shape the app bar can render without knowing
    anything about the filesystem.

    The whole function is one ``try``/``except``: a seed that cannot happen
    (read-only mount, root owned by someone else, no disk) is a **NOTE** for the
    UI, never a reason to refuse opening a project. Refusing to point the Editor
    at someone's folder because a file could not be left behind in it is worse
    than an unfiltered tree.

    Two syscalls carry the guarantees, and both are the point:

    * ``os.mkdir`` — **one level, not ``makedirs``**. The root exists by
      contract, so a missing parent is a symptom worth failing on rather than a
      chain to paper over (and ``EEXIST`` here is the ordinary case: the folder
      is where the file goes);
    * ``os.open(..., O_CREAT | O_EXCL)`` — check and create in one atomic call,
      so a user's edited policy is never clobbered and two concurrent switches
      have exactly one winner. ``EEXIST`` is the ``"existing"`` state, not an
      error. There is deliberately no ``isfile()`` guard before it: that is a
      TOCTOU, and the flag already answers the same question.

    Nothing here raises, for the same reason: the caller is a boot or a switch
    that has already succeeded, and the worst it may do is report.
    """
    try:
        return _ensure(root)
    except Exception as exc:  # noqa: BLE001 — a seed is a note, never a failure
        return _failed(policy_file_path(root), f"{type(exc).__name__}: {exc}")


def policy_file_state(root: str) -> Dict[str, Any]:
    """What a **reader** may say about the policy file, writing nothing.

    ``GET /api/path-location`` asks this rather than :func:`ensure_policy_file`:
    the working location is read far more often than a workspace is switched,
    and a read that leaves a folder behind in someone's project is exactly the
    mistake the seeding rules forbid. Same shape as the other answers, plus
    ``"absent"`` — a workspace whose seed failed, or which predates the feature.
    """
    path = policy_file_path(root)
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return _absent(path)
    except OSError as exc:
        return _failed(path, exc.strerror or str(exc))
    if not stat_module.S_ISREG(info.st_mode):
        # The same condition pol-04 reports as a file-level error finding;
        # naming it here stops the answer claiming a policy the loader rejects.
        return _failed(path, "the policy path is a folder, not a file")
    return _existing(path)


def _ensure(root: str) -> Dict[str, Any]:
    path = policy_file_path(root)
    folder = os.path.dirname(path)
    try:
        os.mkdir(folder, mode=0o755)
    except FileExistsError:
        pass
    except OSError as exc:
        return _failed(path, exc.strerror or str(exc))

    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        # The name is taken. ``"existing"`` is the ordinary answer, but a
        # *folder* there is not a policy file — the same condition pol-04
        # reports as a file-level error finding, said here in the same voice
        # rather than claimed as a policy that does not exist.
        if not _is_regular_file(path):
            return _failed(path, "a folder stands where the policy file belongs")
        return _existing(path)
    except PermissionError:
        # Which error a **folder** at that path answers with is the platform's
        # choice: ``EEXIST`` on POSIX, ``EACCES`` on Windows — the exclusive
        # open cannot create over a directory there. Same condition, same
        # sentence, so a user is told what stands where the file belongs rather
        # than being told the folder they can see is an unreadable file. A real
        # permission problem on an ordinary name falls through to the branch
        # below, which is the platform's own word for it.
        if not _is_regular_file(path):
            return _failed(path, "a folder stands where the policy file belongs")
        return _failed(path, "the policy path cannot be written")
    except OSError as exc:
        return _failed(path, exc.strerror or str(exc))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(default_policy_text())
    except OSError as exc:
        # The bytes did not land, and the file is *ours by construction*: the
        # exclusive open above means no one else's bytes are in it. So it is
        # removed, which puts the workspace back exactly as it was — absence is
        # a state pol-04 treats as "no rules", while a half-written file is an
        # error finding that refuses every write in that project (D4) for a
        # mistake this call made. The empty ``.nojae`` left behind is harmless
        # and is not removed: the folder is the app's reserved name anyway, and
        # deleting from a tree that may be someone else's is not this
        # function's business.
        _discard(path)
        return _failed(path, exc.strerror or str(exc))

    # The cached rules for this workspace were about a file that did not exist
    # a moment ago, so forget them — a switch lands on the new root's rules,
    # not the ones cached while its policy was still absent.
    policy_file.invalidate_policy_cache()
    # Same moment, same reason, for the case answer (win-05): this is one of the
    # two points where a folder *becomes* the working location, so the probe
    # belongs to the new filesystem rather than to whatever the previous root
    # was mounted on.
    fscase.invalidate_case_cache()
    return _created(path)


def _discard(path: str) -> None:
    """Best-effort removal of the file this call just created."""
    try:
        os.unlink(path)
    except OSError:
        pass
