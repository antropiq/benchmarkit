"""The sibling mutations of the guarded write: create, rename, and delete.

``files.py`` stays about reading plus one whole buffer back to disk; the
operations that make, move and remove *paths* live here, as the write half's
sibling of ``tree.py``/``files.py``. Entry points:

* :func:`create_path` — the create route: one empty file or one folder, never
  over anything that is already there.
* :func:`rename_path` — the rename route: one path to a new workspace path. A
  rename with a different parent *is* a move, so it is one operation — and the
  one where the policy has the most ways to go wrong, because two
  user-supplied paths describe one filesystem action.
* :func:`move_paths_batch` — the move-batch route: N files into one folder that
  already exists, the same rules as a rename asked of every entry **before the
  first one moves**, so the set either moves whole or nothing moves at all.
* :func:`trash_path` — the delete route: one file or one whole folder, **moved**
  into a dated batch under the active workspace's own trash (``trash_dir()``,
  ``<workspace>/.nojae/trash``) and never ``unlink``ed, so a
  mistaken delete is a ``mv`` away from being undone.

The gate is the one the save goes through — the whole ancestor chain, each level
asked of the policy verdict — and the same promise holds: this is a browser
surface, reached from the SPA and from nothing else. No MCP connector, LSP tool
or chat code path may call it; the guard suite in ``tests/test_editorfs_write.py``
asserts that instead of trusting the prose.

What the verdict means for a *path* operation (pol-00 D3): an ``ro`` folder
cannot be deleted, emptied, renamed **or added to** — the rule is about its
subtree *and* about the folder's own directory entries, because a move out of it
rewrites them just as surely as a move into it. A ``hidden`` name cannot be
created either: the write gate answers 422 naming the rule, never the 404 that
would tell the caller the name is free.
"""

import errno
import hashlib
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import constants, fscase
from .files import fsync_dir
from .policy import (
    WorkspacePathError,
    _absolute_parts,
    _is_within,
    missing_path_error,
    normalise_rel,
    resolve_existing_in_workspace,
    resolve_parent_in_workspace,
    resolve_write_target,
    workspace_root,
)
from .policy_modes import mode_for, refusal_message
from .constants import policy_shared as shared


class WorkspacePathExists(WorkspacePathError):
    """The 409 a create answers when the name is taken — carrying the path.

    ``detail`` is the sentence for a log; :attr:`detail_object` is what goes on
    the wire, because the client's answer to "already there" is a *re-action*
    (re-list the folder, select the row that exists), not a message to read.
    Same shape as the save guard's conflict object, for the same reason.
    """

    def __init__(self, path: str, *, is_dir: bool) -> None:
        super().__init__(
            409,
            f"{path} already exists "
            f"({'a folder' if is_dir else 'a file'}), and a create never replaces it",
        )
        self.detail_object: Dict[str, Any] = {
            "reason": "exists",
            "path": path,
            "dir": is_dir,
        }


def _exists_answer(normalised: str, target: str) -> WorkspacePathExists:
    """The 409 for an occupied name — a dangling symlink included.

    ``lstat``, not ``stat``/``exists``: a symlink whose target is gone has no
    ``stat`` at all, and calling that "free" would let a create write *through*
    someone's link. What matters is that the name is taken.
    """
    try:
        info = os.lstat(target)
    except OSError:
        info = None
    is_dir = info is not None and (info.st_mode & 0o170000) == 0o040000
    return WorkspacePathExists(normalised, is_dir=is_dir)


def _missing_parent_error(normalised: str) -> WorkspacePathError:
    """The gate's own sentence, for a folder that went away mid-create."""
    parent_rel = normalised.rpartition("/")[0]
    return WorkspacePathError(
        409, f"Parent folder does not exist: {parent_rel or '.'}"
    )


def create_path(rel: str, *, is_dir: bool) -> Dict[str, Any]:
    """The create route's core: one new name inside the workspace, or a refusal.

    Rules, each one mirrored by a test:

    * the **write** gate over the whole chain — the parent and its ancestors
      through :func:`policy.resolve_parent_in_workspace`, the leaf name by its
      own verdict — so a name a policy hides and a name under a read-only folder
      are both refused, 422 naming the rule, **before the name is looked for on
      disk**;
    * the parent must already be a directory; a missing one is **409** naming it.
      A typo in a name input becomes a refusal rather than a stray tree, and a
      deeper path stays the client's decision, one gated request per level —
      there is no ``mkdir -p`` side effect here;
    * **never overwrite**. An occupied name — file, folder, or dangling symlink —
      answers 409 ``{"reason": "exists", …}`` with the disk untouched. A create
      that silently truncated an existing file is the worst bug this function
      could have, so the race is *closed* rather than checked: ``os.mkdir`` and
      ``O_CREAT | O_EXCL`` fail on an existing name, so of two tabs creating the
      same one, one wins and the other gets this same 409;
    * ``is_dir`` picks ``os.mkdir`` over an ``O_WRONLY`` open of an empty file;
    * the parent's entry is fsync'd, so the new name survives a crash;
    * the payload answers ``path``/``dir``/``size``/``mtime`` — the same shape a
      save answers, so the client re-baselines on one shape.

    There is one policy model here, not two: **``hidden`` is a write rule** (a
    name the policy hides cannot be created — D8, and the answer is 422 naming
    the rule, never the 409 that would confirm the name is free), **``ro`` is a
    write rule**, and neither is a *read* rule except ``hidden``. The old note
    explained why the tree's ignored-names set and the connector deny-list
    disagreed about a folder called ``node_modules``; under one policy there is
    nothing to disagree about — the seeded file makes that name read-only, so
    creating it is refused and the answer says so.
    """
    normalised, parent, name = resolve_parent_in_workspace(rel)

    target = parent / name
    if os.path.lexists(str(target)):
        raise _exists_answer(normalised, str(target))

    try:
        if is_dir:
            os.mkdir(target)
        else:
            handle = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(handle)
    except FileExistsError as exc:  # the race closed: another writer won the name
        raise _exists_answer(normalised, str(target)) from exc
    except PermissionError as exc:
        raise WorkspacePathError(
            403, f"Cannot create {normalised}: {exc.strerror or exc}"
        ) from exc
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            # The folder went away between the gate and here: the same answer
            # the gate gives, never an unhandled OSError.
            raise _missing_parent_error(normalised) from exc
        raise

    fsync_dir(str(parent))

    try:
        stat = os.stat(target)
    except OSError as exc:  # pragma: no cover - the create only just landed
        raise WorkspacePathError(
            500, f"Created {normalised} but could not stat it: {exc}"
        ) from None
    return {
        "path": normalised,
        "dir": is_dir,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


# ---------------------------------------------------------------------------
# Rename / move: one path, and everything that answered for it, to a new path.
# ---------------------------------------------------------------------------


def _parent_missing_error(parent_rel: str) -> WorkspacePathError:
    """The create's sentence, for the folder a rename target asked for."""
    return WorkspacePathError(
        409, f"Parent folder does not exist: {parent_rel or '.'}"
    )


def _rename_oserror(
    from_rel: str, to_rel: str, target: str, exc: OSError
) -> WorkspacePathError:
    """The one answer for an ``os.rename`` that failed — the two movers share it.

    ``EXDEV`` is refused by name rather than worked around (copy-then-delete is
    not atomic and loses the mode and the extended attributes); ``ENOENT`` /
    ``ENOTDIR`` mean a folder went away between the gate and here, which is the
    same answer the gate gives; ``ENOTEMPTY`` / ``EEXIST`` mean something arrived
    at the target name after the occupied check, which is the same answer the
    check gives. Anything else is a 500 with both paths in the sentence.
    """
    if exc.errno == errno.EXDEV:
        return WorkspacePathError(
            409,
            f"Cannot move {from_rel} to {to_rel}: they are on different "
            "filesystems, and a copy-then-delete would not be atomic",
        )
    if exc.errno in (errno.ENOENT, errno.ENOTDIR):
        return _parent_missing_error(to_rel.rpartition("/")[0])
    if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
        return _exists_answer(to_rel, target)
    return WorkspacePathError(
        500, f"Could not rename {from_rel} to {to_rel}: {exc}"
    )


def rename_path(frm: str, to: str) -> Dict[str, Any]:
    """The rename route's core: move one path, or refuse.

    Taking the **whole target path** (not a bare new name) is what makes a move
    the same route as a rename — and it is why the target's parent needs the
    same treatment as everything else here.

    Steps, in this order — the order is the test list:

    * gate ``from`` — itself **and every ancestor of it** — with the write
      verdict, then resolve it (404 when it is not there), and refuse a
      top-level **symlink** (400, the delete's argument: the link's target may be
      outside the workspace, and moving the link is not a rename of a file);
    * gate ``to`` **and every ancestor of ``to``** the same way. This is the
      both-ends rule: ``from`` may be perfectly innocent while ``to`` sits in a
      folder the policy hides or locks, and the refusal has to arrive from the
      target's side too. It is also what makes pol-00 D3's "renaming **out** of
      an ``ro`` folder is refused" need no new code — a move out of a locked
      folder is refused by its own source end, because a move rewrites that
      folder's directory entries exactly as a move into one would.
    * the target's parent must already be a directory — **409** naming it, no
      implicit ``mkdir`` (the create's rule, restated for the same reason: a
      typo in a name input becomes a refusal rather than a stray tree);
    * ``from`` and ``to`` normalising to the same path is **400**. A case-only
      rename is *not* that (on a case-sensitive filesystem the two normalised
      paths differ), and it must work;
    * **a folder cannot be moved into itself**: ``to`` starting with
      ``from + '/'`` is 409 naming the rule. The trailing slash is what makes
      the two near-misses legal — ``src`` → ``src2`` and ``src/sub`` →
      ``src/sub2`` are siblings, not descendants;
    * an occupied target — file, folder, dangling symlink — is 409
      ``{"reason": "exists", …}``. ``os.rename`` would *silently replace* a file
      here, which is the data loss this route exists to prevent;
    * the working location itself (``from == ""``) is refused outright (400);
    * ``os.rename`` — same filesystem, so atomic. An ``EXDEV`` (a bind mount
      inside the workspace) is refused by name rather than worked around:
      copy-then-delete is not atomic and loses the mode and the extended
      attributes that make a file what it is.

    There is deliberately **no backup** for a rename, unlike a save: the bytes
    did not change, and the trash plus ``git status`` already describe the move.
    A backup tree of every file that was ever renamed would be a second copy of
    the workspace for information that costs nothing to recover by renaming back.

    The payload answers ``from``/``to``/``dir``/``size``/``mtime``. ``mtime`` is
    the moved file's own — a rename does not touch content, so an open buffer
    of it stays fresh and its save token stays valid at the new path.
    """
    from_rel = normalise_rel(frm)
    to_rel = normalise_rel(to)
    if from_rel == "":
        raise WorkspacePathError(
            400, "The working location itself cannot be renamed"
        )

    # ``from`` first: it exists on disk, so the whole realpath chain is checked,
    # and a missing source is the plain 404 every other mutation answers.
    from_path = resolve_write_target(from_rel)[-1][1]
    if not from_path.exists():
        raise missing_path_error(from_rel)

    spelled = Path(os.path.join(workspace_root(), from_rel))
    if spelled.is_symlink():
        raise WorkspacePathError(
            400,
            f"Path is a symbolic link, not a file to rename: {from_rel} "
            f"(its target may be outside the working location, and renaming "
            f"the link would move it without moving what it points at)",
        )

    # ``to`` and its whole ancestor chain. The leaf name is not on disk, so it
    # is judged by its own verdict rather than through the realpath locks — the
    # same split :func:`create_path` relies on.
    _, to_parent, to_name = resolve_parent_in_workspace(to_rel)

    if to_rel == from_rel:
        raise WorkspacePathError(400, "Source and target are the same path")

    if to_rel.startswith(f"{from_rel}/"):
        raise WorkspacePathError(
            409,
            f"Cannot move {from_rel} inside itself: {to_rel} is under it",
        )

    target = to_parent / to_name
    if os.path.lexists(str(target)):
        raise _exists_answer(to_rel, str(target))

    try:
        os.rename(str(from_path), str(target))
    except OSError as exc:
        raise _rename_oserror(from_rel, to_rel, str(target), exc) from exc

    fsync_dir(str(to_parent))
    fsync_dir(str(from_path.parent))

    try:
        # Not ``lstat``: a top-level symlink was refused above, so what is here
        # now is the file or folder that was moved.
        stat = os.stat(target)
    except OSError as exc:  # pragma: no cover - the rename only just landed
        raise WorkspacePathError(
            500, f"Renamed {from_rel} but could not stat it: {exc}"
        ) from None
    return {
        "from": from_rel,
        "to": to_rel,
        "dir": (stat.st_mode & 0o170000) == 0o040000,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


# ---------------------------------------------------------------------------
# Batch move: N files into one folder that is already there, or nothing at all.
# ---------------------------------------------------------------------------


def move_paths_batch(paths: List[str], to_parent: str) -> Dict[str, Any]:
    """The move-batch route's core: N files to one folder, or a refusal.

    This is **not a second rename** — it is :func:`rename_path`'s rules applied
    N times with the checks finished before the first move starts. The all-or-
    nothing the UI promises cannot be had by firing N rename requests from the
    browser: the successes would be on disk when the fifth is refused. So the
    transaction here is the *ordering* — validate every entry, then apply none
    until all are valid.

    Steps, in this order — the order is the test list:

    * ``paths`` empty is 400; more than :data:`editorfs.constants.MAX_MOVE_BATCH`
      entries is 400 naming the cap (read at call time, so a test patches the
      constant by name);
    * ``to_parent`` is normalised, never joined by the client — the target is
      ``to_parent + "/" + basename(source)``, computed here, because a client
      that joined them could hide a target the ancestor gate would never see. The
      no-op case (the destination *is* the sources' own folder) is not refused
      here the way the dialog refuses it: it falls out of the occupied check as
      the 409 it is, since each target is the source itself;
    * every **source**: gated through itself and its whole ancestor chain, then
      it must exist (**404** for the first absent one), must be a **file** —
      a folder is 400 naming the path, so the route cannot be talked into moving
      a tree — and must not be a top-level **symlink** (400, the delete's and the
      rename's argument verbatim);
    * ``to_parent`` must already be a directory — **409** naming it, no implicit
      ``mkdir``. Asked before the targets, because one folder describes the
      whole batch: a destination that is a file gets *this* sentence rather than
      the create's different one for a file parent;
    * every **target**, gated through its whole ancestor chain (the both-ends
      rule; an innocent file may be moving into a folder the policy hides);
    * no target occupied (**409** ``{"reason": "exists", …}``), and no two
      sources landing on one target name. That last check is unreachable from
      the UI (a batch's files share one parent, so distinct names stay distinct);
      it is the guard against a hand-built body that would otherwise let the
      second ``os.rename`` silently replace the first;
    * **then** the renames, in the given order.

    Step 7 is not atomic across N paths — there is no filesystem transaction to
    borrow — and that is owned out loud rather than papered over: the loop stops
    at the first ``OSError`` and never continues past one, so a partial apply can
    only ever leave a **prefix** of the list moved — and ``moved`` reports exactly
    what landed rather than what was asked for. The window between the occupied
    check and here is the same one a single :func:`rename_path` already accepts,
    and ``EXDEV`` / ``ENOENT`` map to the same 409 sentences through
    :func:`_rename_oserror`, naming the entry that failed.

    There is deliberately **no backup**, as for a rename: the bytes did not
    change, and the trash plus ``git status`` already describe the move.

    The payload is ``{"moved": [{"from", "to", "mtime"}…], "to_parent"}`` — the
    rename's own shape per entry, so the caller's reconciliation is the rename's
    loop — and ``to_parent`` is the *normalised* folder, so what gets re-listed
    is the one the server actually used.
    """
    cap = constants.MAX_MOVE_BATCH
    if not paths:
        raise WorkspacePathError(400, "A move needs at least one file")
    if len(paths) > cap:
        raise WorkspacePathError(
            400,
            f"A move names at most {cap} files at once, and this names "
            f"{len(paths)}",
        )

    parent_rel = normalise_rel(to_parent)

    # Step 3 — every source, gated and answered for, before any target is
    # looked at. The first entry that fails decides the answer.
    entries: List[Dict[str, Any]] = []
    for raw in paths:
        from_rel = normalise_rel(raw)
        if from_rel == "":
            raise WorkspacePathError(
                400, "The working location itself cannot be moved"
            )
        from_path = resolve_write_target(from_rel)[-1][1]
        if not from_path.exists():
            raise missing_path_error(from_rel)
        if from_path.is_dir():
            raise WorkspacePathError(
                400,
                f"Path is a folder, not a file to move: {from_rel} "
                f"(a batch moves files into one folder; moving a whole tree "
                f"is what Rename does)",
            )
        # The **spelled** path, not the resolved one — the delete's argument:
        # the gate hands back the realpath, which is a link's target and never a
        # link itself, so symlink-ness has to be judged one level up.
        spelled = Path(os.path.join(workspace_root(), from_rel))
        if spelled.is_symlink():
            raise WorkspacePathError(
                400,
                f"Path is a symbolic link, not a file to rename: {from_rel} "
                f"(its target may be outside the working location, and renaming "
                f"the link would move it without moving what it points at)",
            )

        # Step 4's input, joined **here** (D11): the client never sends a target,
        # so the ancestor gate cannot be shown one folder and landed in another.
        name = from_rel.rpartition("/")[2]
        to_rel = f"{parent_rel}/{name}" if parent_rel else name
        entries.append({"from": from_rel, "to": to_rel, "source": from_path})

    # Step 5 (R3) — asked **before** each target, deliberately: one folder
    # describes the whole batch, so a locked one is 422 naming the rule and a
    # folder that is missing *or is a file* is the same 409 sentence naming it.
    # ``resolve_parent_in_workspace`` would answer a file with its own 400, which
    # is not what a destination field is shown — and there is no implicit
    # ``mkdir`` anywhere in this function.
    parent_path = resolve_write_target(parent_rel)[-1][1]
    if not parent_path.is_dir():
        raise _parent_missing_error(parent_rel)

    # Step 4 — every target through its own ancestor chain. The leaf name is not
    # on disk, so it is judged by its own verdict exactly as a create judges it:
    # a name the policy hides or locks *at the destination* is refused there even
    # though the same name was innocent at the source.
    targets: Dict[str, Tuple[Path, str]] = {}
    for entry in entries:
        _, to_parent_path, to_name = resolve_parent_in_workspace(entry["to"])
        targets[entry["to"]] = (to_parent_path, to_name)

    # Nothing occupied, and no two entries aiming at one name. The second is a
    # hand-built-body guard (see the docstring): three lines, no collision table.
    seen: Dict[str, str] = {}
    for entry in entries:
        target = targets[entry["to"]][0] / targets[entry["to"]][1]
        if entry["to"] in seen:
            raise WorkspacePathError(
                400,
                f"Two moved files would land on the same name: "
                f"{seen[entry['to']]} and {entry['from']} both to {entry['to']}",
            )
        seen[entry["to"]] = entry["from"]
        if os.path.lexists(str(target)):
            raise _exists_answer(entry["to"], str(target))

    # Step 7 — only now the moves, in the given order, stopping at the first
    # failure (there is no filesystem transaction to borrow, so the payload says
    # what landed rather than pretending the whole list did).
    moved: List[Dict[str, Any]] = []
    for entry in entries:
        from_rel = str(entry["from"])
        to_rel = str(entry["to"])
        from_path = entry["source"]
        to_parent_path, to_name = targets[to_rel]
        target = to_parent_path / to_name
        try:
            os.rename(str(from_path), str(target))
        except OSError as exc:
            raise _rename_oserror(from_rel, to_rel, str(target), exc) from exc
        fsync_dir(str(to_parent_path))
        fsync_dir(str(from_path.parent))
        try:
            # Not ``lstat``: a symlink source was refused above, so what is here
            # now is the file that was moved.
            stat = os.stat(target)
        except OSError as exc:  # pragma: no cover - the move only just landed
            raise WorkspacePathError(
                500, f"Moved {from_rel} but could not stat it: {exc}"
            ) from None
        moved.append({"from": from_rel, "to": to_rel, "mtime": stat.st_mtime})

    return {"moved": moved, "to_parent": parent_rel}


# ---------------------------------------------------------------------------
# Delete: one path, moved into the trash — never ``unlink``ed.
# ---------------------------------------------------------------------------


def trash_dir() -> Path:
    """The active workspace's trash — ``.nojae/trash`` under the resolved root.

    Resolved **per call**, never captured at import: a settings switch changes
    :func:`editorfs.policy.workspace_root`, and the very next delete has to file
    into the new root rather than into the install directory it was pointed at
    when the module was imported.

    ``constants.TRASH_DIR`` stays as the pin for a test (or a deployment) that
    wants one absolute folder instead; it is read here, at call time.

    Nothing in :mod:`editorfs.constants` may import :mod:`editorfs.policy` (the
    package imports ``policy`` *from* ``constants``, and a cycle through a
    partially-initialised module is not worth one helper), which is why the
    resolver lives here — the one module that already imports both sides.

    The destination is app-chosen space inside the reserved folder, so it is
    **never** put through the write gate: the gate refuses every ``.nojae/...``
    path with a 422 naming the reservation, and a delete that gated its own
    destination could never move anything. Only the *source* is gated.

    ``resolve()`` normalises both sides of every later comparison — the batch
    label rebases on this path, and :func:`_rel_in_workspace` works on the
    resolved root, so a symlinked workspace spelling must not leave the label
    absolute by an accident of spelling.
    """
    override = constants.TRASH_DIR
    if override is not None:
        return Path(override).resolve()
    return Path(
        workspace_root(), constants.POLICY_DIRNAME, constants.TRASH_SUBDIR
    ).resolve()


def trash_digest(resolved: Path) -> str:
    """The digest identifying *what* a trashed batch holds.

    Keyed on the resolved path, not the spelling in the request (the reasoning
    the editor's backups use too): two mount spellings of one tree are one
    directory, and two batches that could collide by name are easier to read
    when the name they collide on is the digest of the same file.
    """
    return hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]


def trash_batches() -> List[Path]:
    """The batch folders currently in the trash — what a sweep walks over.

    Public because a test asserts the shape of what a delete leaves behind
    without re-deriving the naming scheme.
    """
    trash = trash_dir()
    if not trash.is_dir():
        return []
    return sorted(item for item in trash.iterdir() if item.is_dir())


def _legacy_batches() -> List[Path]:
    """Batches the *old* code wrote under ``logs/trash`` — swept, never written.

    Separate from :func:`trash_batches` so that one stays exactly what its
    docstring promises: the active workspace's own batches.
    """
    trash = Path(constants.LEGACY_TRASH_DIR)
    if not trash.is_dir():
        return []
    return sorted(item for item in trash.iterdir() if item.is_dir())


def sweep_trash(*, older_than_days: Optional[int] = None,
                now: Optional[float] = None) -> int:
    """Delete every trash batch older than the retention window. Returns how many.

    The shape of the fedit connector's ``prune_old_backups`` — age is the only
    signal that still works for a path that no longer exists, the scan is one
    directory level, and it is called from the write path so it never raises: a
    batch that cannot be deleted is not worth failing a delete over.

    It prunes **both** the active workspace's trash and the legacy
    ``logs/trash`` folder the batches written before this task live in. The
    legacy folder is only ever expired, never *moved* into the new shape:
    rewriting bytes nobody is looking at is a worse surprise than a batch that
    outlives its usefulness one day longer, and age is the only signal left for
    a path the app no longer addresses. Because the sweep is best-effort, it is
    also called once from the app's startup, so the legacy folder drains even
    when no further delete ever happens.

    Like the move it follows, the sweep walks these folders with plain ``os`` /
    ``shutil`` calls — the trash sits inside the reserved folder, which the
    gate refuses to *address*, and the sweep is the app tidying its own space
    rather than a request naming a path. It is deliberately *not* imported
    across the connector boundary (``src/`` does not depend on
    ``mcp_connectors/`` except for the shared policy modules).
    """
    days = (
        constants.TRASH_RETENTION_DAYS if older_than_days is None else older_than_days
    )
    if days <= 0:
        return 0
    cutoff = (now if now is not None else time.time()) - days * 86400
    removed = 0
    folders = [trash_dir(), Path(constants.LEGACY_TRASH_DIR)]
    for item in trash_batches() + _legacy_batches():
        try:
            if item.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(item)
            removed += 1
        except OSError:
            continue
    if removed:
        for folder in folders:
            fsync_dir(str(folder))
    return removed


def _walk_descendants(root: Path) -> List[Path]:
    """Every path below ``root``, symlinks **not** followed.

    ``followlinks=False`` is the whole point: a link inside a folder may point
    at ``/etc``, and walking there would both read outside the workspace and
    report a descendant that is not really inside the subtree. The link *itself*
    is reported, so a link to a denied name still refuses the delete.
    """
    found: List[Path] = []
    for current, dirs, names in os.walk(str(root), followlinks=False):
        for entry in list(dirs) + list(names):
            found.append(Path(current) / entry)
    return found


def _rel_in_workspace(resolved: Path) -> Optional[str]:
    """``resolved`` as a workspace-relative path, or ``None`` when it is not in.

    The verdict is written against names inside the workspace, so a subtree
    check that handed it an absolute path would be answering a different
    question. ``None`` is a real answer here, not a failure: it means the path —
    always a symlink's target, since the walk starts at a resolved root — sits
    where no policy of this workspace reaches, and the caller refuses that.

    Containment is the one shared rule (A-D2.2, win-06) rather than a prefix
    built here: what this function used to do was trim the root's trailing
    separator, append one, and compare that against slash-joined text — which
    never matched a ``C:\\…`` root, so it answered ``None`` for *every* path on
    Windows and each folder delete was refused with the "outside the working
    location" wording (roadmap §2's loud-but-correct row).
    The fold comes from the workspace's own filesystem, and the remainder keeps
    ``/`` separators like every other Editor payload.
    """
    workspace = workspace_root()
    text = str(resolved)
    if not _is_within(workspace, text, fold=fscase.case_folded(workspace)):
        return None
    # Same component list the containment compare used, so the two cannot
    # disagree about where the workspace stops. The remainder keeps the
    # spelling of the path on disk — folding is comparison only (A-D3).
    parent_parts = _absolute_parts(workspace)
    return "/".join(_absolute_parts(text)[len(parent_parts) :])


def _subtree_refusal(
    root: Path, descendants: List[Path]
) -> Optional[WorkspacePathError]:
    """The first path at or below ``root`` that a policy locks away, else None.

    All-or-nothing by construction: this runs before a single byte moves, and
    the caller refuses the whole delete on its answer. Deleting ``src/`` because
    its own name is innocent while it holds a read-only or hidden file is the
    accident this exists to prevent, so the refusal names the **descendant** —
    that is the path the user has to move away from.

    Both strict modes count: a folder rule matches everything under it, so this
    is what makes ``ro`` mean "this subtree does not move", and ``hidden`` mean
    the same about a subtree the tree never showed. A descendant whose answer
    cannot be settled is refused too — an unchecked subtree is not a checked one.

    ``descendants`` is the caller's walk (the same one that counts ``entries``),
    so a delete of a big folder walks it once, not twice.
    """
    for candidate in [root] + descendants:
        try:
            if candidate.is_symlink():
                # The link's own *target* is what has to be judged: a link to a
                # locked path inside an innocent folder is a route to it, and a
                # link pointing outside the workspace cannot be vouched for at
                # all — no rule of this workspace reaches it, so nothing here
                # says it may move.
                target = Path(os.path.realpath(str(candidate)))
            else:
                target = candidate
        except OSError:  # pragma: no cover - realpath does not raise here
            return WorkspacePathError(
                422, f"Cannot check the subtree under {root.name}: {candidate}"
            )
        rel = _rel_in_workspace(target)
        if rel is None:
            return WorkspacePathError(
                422,
                f"Path is outside the working location: {target} "
                f"(a link under {root.name}) — no policy can vouch for it",
            )
        verdict = mode_for(rel)
        if verdict.mode in (shared.HIDDEN, shared.READ_ONLY):
            message = refusal_message(verdict, for_write=True)
            return WorkspacePathError(422, f"{message} — inside {root.name}")
    return None


def _batch_label(batch: Path) -> str:
    """How the trash folder is named back to the browser.

    **Workspace-relative** (``.nojae/trash/<batch>``) whenever the batch sits in
    the active workspace's trash — the same frame every other Editor payload
    uses, and the one the e2e witness joins onto the working location. It names
    a folder the pane *deliberately cannot open* (the reserved folder is hidden
    from the tree and the read route), so the confirmation has to phrase it as a
    path to ``cd`` to in a terminal, not as a browsable location.

    Absolute when a deployment (or a test) pinned ``TRASH_DIR`` outside the
    workspace — a label that is wrong is worse than one that is longer.
    """
    workspace = Path(workspace_root())
    try:
        rel = batch.relative_to(workspace)
    except ValueError:
        return str(batch)
    # ``/``-joined whatever the host separator (A §7, roadmap §9 rule 1): this
    # label is Editor payload text, and a payload field never carries ``\``.
    return rel.as_posix()


def trash_path(rel: str) -> Dict[str, Any]:
    """The delete route's core: move one path into the trash, or refuse.

    Steps, in this order — the order is the test list:

    * the **write** gate over the whole ancestor chain, so a path under a hidden
      folder, one marked read-only, and a NUL are all refused — 422 naming the
      rule — before the path is looked for on disk;
    * the working location itself is refused outright (400): deleting "" is not
      a thing the tree offers and not a thing a route should ever move;
    * **404** for an absent path — absent is *not* "already deleted, fine". The
      caller's view of the tree is wrong and has to be told;
    * a **symlink** at the top level is refused (400): its target may live
      outside the workspace and the policy cannot vouch for it. Unlinking a
      link is a human decision for the terminal;
    * the **subtree check, all-or-nothing** (:func:`_subtree_refusal`): every
      descendant with the write flavour, symlinks not followed. One denied
      path → 422 naming it, and **nothing moves**;
    * the move: ``shutil.move`` of the top-level entry into
      ``trash_dir()/<digest>_<stamp>/<the path exactly as it was, relative to the
      workspace>`` — ``trash_dir()`` being the **active** workspace's
      ``.nojae/trash``, asked here rather than at import so a settings switch is
      followed by the very next delete. Keeping the relative layout is what makes
      a restore a plain ``mv`` — with a bare ``f.py`` one cannot tell ``src/f.py``
      from ``web/f.py``, and a name collision inside one batch would be data loss.
      ``shutil.move`` preserves the subtree, and on a workspace that sits on
      another filesystem it degrades to a copy + delete: slower, and correct,
      which is the trade being taken here.
      The **destination** is built with plain ``mkdir`` calls and is never put
      through :func:`resolve_write_target`: it lives under the reserved folder,
      which that gate refuses for every path that names it, and it is app-chosen
      space no route can address anyway. Only the source is gated.
      A batch can never be trashed into itself either — nothing can delete
      ``.nojae`` or anything inside it, and no folder under the root contains the
      trash path — so the ``shutil.move`` "directory into itself" hazard does not
      arise.
    * the sweep (:func:`sweep_trash`) runs after the move, so a failed prune can
      never lose the copy that was just made.

    The payload answers ``path``, ``dir``, the batch (``trashedTo``, relative to
    the working location — a folder the pane cannot open, so the confirmation
    names it as something to ``cd`` to in a terminal) and ``entries`` — the
    descendant count from the server's own walk, which is the number the
    confirmation dialog already promised.
    """
    normalised = normalise_rel(rel)
    if normalised == "":
        raise WorkspacePathError(
            400, "The working location itself cannot be deleted"
        )
    _, resolved = resolve_existing_in_workspace(normalised)

    # The **spelled** path, not the resolved one: the gate hands back the
    # realpath, which is the *target* of a link and therefore never a link
    # itself. Judging symlink-ness has to happen one level up.
    spelled = Path(os.path.join(workspace_root(), normalised))
    if spelled.is_symlink():
        raise WorkspacePathError(
            400,
            f"Path is a symbolic link, not a file to delete: {normalised} "
            f"(its target may be outside the working location, and moving the "
            f"link to the trash would break it)",
        )

    # One walk, used twice: the subtree check has to see every descendant, and
    # the answer's ``entries`` is that same count — so the number the
    # confirmation dialog promised comes from the server's own walk.
    descendants = _walk_descendants(resolved)
    subtree_error = _subtree_refusal(resolved, descendants)
    if subtree_error is not None:
        raise subtree_error

    trash = trash_dir()
    trash.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    batch = trash / f"{trash_digest(resolved)}_{stamp}"
    # A same-second pair of deletes is rare at microsecond stamps but not
    # impossible (two tabs, one keystroke each), and a second batch *inside* the
    # first would be a confusing thing to restore. The suffix is what makes one
    # delete one directory, whatever the clock's resolution.
    suffix = 0
    while batch.exists():
        suffix += 1
        batch = trash / f"{trash_digest(resolved)}_{stamp}-{suffix}"
    destination = batch / normalised
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(resolved), str(destination))
    except OSError as exc:
        raise WorkspacePathError(
            500, f"Could not move {normalised} to the trash: {exc}"
        ) from exc

    fsync_dir(str(batch))
    fsync_dir(str(destination.parent))
    fsync_dir(workspace_root())
    sweep_trash()

    return {
        "path": normalised,
        "dir": not os.path.isfile(str(destination)),
        "trashedTo": _batch_label(batch),
        "entries": len(descendants),
    }
