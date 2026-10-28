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
* :func:`trash_path` — the delete route: one file or one whole folder, **moved**
  into a dated batch under ``constants.TRASH_DIR`` and never ``unlink``ed, so a
  mistaken delete is a ``mv`` away from being undone.

The gate is the one the save goes through — the whole ancestor chain, with the
**write** flavour of the connector policy — and the same promise holds: this is
a browser surface, reached from the SPA and from nothing else. No MCP connector,
LSP tool or chat code path may call it; the guard suite in
``tests/test_editorfs_write.py`` asserts that instead of trusting the prose.
"""

import errno
import hashlib
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import constants
from .files import fsync_dir
from .policy import (
    WorkspacePathError,
    missing_path_error,
    normalise_rel,
    refusal,
    resolve_existing_in_workspace,
    resolve_parent_in_workspace,
    resolve_write_target,
    workspace_root,
)


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
      through :func:`policy.resolve_parent_in_workspace`, the leaf name through
      the same list — so ``mcp_connectors/x.py``, ``.git/config``, ``.env`` and
      ``id_rsa`` are refused in the deny-list's own words, before the name is
      looked for on disk;
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

    ``constants.IGNORED_DIR_NAMES`` is deliberately **not** a write rule: a
    folder called ``node_modules`` is legal to create and simply invisible
    afterwards, because the read surface hides that name. It looks like a bug
    from the browser, so say it plainly — the create landed; the tree just does
    not show that name.
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


def rename_path(frm: str, to: str) -> Dict[str, Any]:
    """The rename route's core: move one path, or refuse.

    Taking the **whole target path** (not a bare new name) is what makes a move
    the same route as a rename — and it is why the target's parent needs the
    same treatment as everything else here.

    Steps, in this order — the order is the test list:

    * gate ``from`` with the **write** flavour, then resolve it (404 when it is
      not there), and refuse a top-level **symlink** (400, the delete's
      argument: the link's target may be outside the workspace, and moving the
      link is not a rename of a file);
    * gate ``to`` **and every ancestor of ``to``** with the write flavour. This
      is the both-ends rule: ``from`` may be perfectly innocent while ``to`` is
      ``.git/hooks/pre-commit`` or ``mcp_connectors/policy_patch.py``, and the
      refusal has to arrive from the target's side too;
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
    # goes through the deny-list directly (:func:`policy.refusal`) rather than
    # through the realpath locks — the same split :func:`create_path` uses.
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
        if exc.errno == errno.EXDEV:
            raise WorkspacePathError(
                409,
                f"Cannot move {from_rel} to {to_rel}: they are on different "
                "filesystems, and a copy-then-delete would not be atomic",
            ) from exc
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            # The folder went away between the gate and here.
            raise _parent_missing_error(to_rel.rpartition("/")[0]) from exc
        if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
            raise _exists_answer(to_rel, str(target)) from exc
        raise WorkspacePathError(
            500, f"Could not rename {from_rel} to {to_rel}: {exc}"
        ) from exc

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
# Delete: one path, moved into the trash — never ``unlink``ed.
# ---------------------------------------------------------------------------


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
    trash = Path(constants.TRASH_DIR)
    if not trash.is_dir():
        return []
    return sorted(item for item in trash.iterdir() if item.is_dir())


def sweep_trash(*, older_than_days: Optional[int] = None,
                now: Optional[float] = None) -> int:
    """Delete every trash batch older than the retention window. Returns how many.

    The shape of the fedit connector's ``prune_old_backups`` — age is the only
    signal that still works for a path that no longer exists, the scan is one
    directory level, and it is called from the write path so it never raises: a
    batch that cannot be deleted is not worth failing a delete over. It is
    deliberately *not* imported across the connector boundary (``src/`` does not
    depend on ``mcp_connectors/`` except for the one lazily-imported policy
    module).
    """
    days = (
        constants.TRASH_RETENTION_DAYS if older_than_days is None else older_than_days
    )
    trash = Path(constants.TRASH_DIR)
    if days <= 0:
        return 0
    cutoff = (now if now is not None else time.time()) - days * 86400
    removed = 0
    for item in trash_batches():
        try:
            if item.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(item)
            removed += 1
        except OSError:
            continue
    if removed:
        fsync_dir(str(trash))
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


def _subtree_refusal(
    root: Path, descendants: List[Path]
) -> Optional[WorkspacePathError]:
    """The first denied path at or below ``root``, or None when all are writable.

    All-or-nothing by construction: this runs before a single byte moves, and
    the caller refuses the whole delete on its answer. Deleting ``src/`` because
    its own name is innocent while it holds ``.env`` is the accident this exists
    to prevent, so the refusal names the **descendant** — that is the path the
    user has to move away from.

    A descendant whose policy answer cannot be settled is refused too: an
    unchecked subtree is not a checked one, and the promise is that nothing
    denied was moved.

    ``descendants`` is the caller's walk (the same one that counts ``entries``),
    so a delete of a big folder walks it once, not twice.
    """
    for candidate in [root] + descendants:
        try:
            if candidate.is_symlink():
                # The link's own *target* is what the policy has to judge: a
                # link to ``.ssh/id_rsa`` inside an innocent folder is a write
                # route to a secret, and a link pointing outside the workspace
                # cannot be vouched for at all.
                target = Path(os.path.realpath(str(candidate)))
            else:
                target = candidate
        except OSError:  # pragma: no cover - realpath does not raise here
            return WorkspacePathError(
                422, f"Cannot check the subtree under {root.name}: {candidate}"
            )
        deny = refusal(target, for_write=True)
        if deny is not None:
            return WorkspacePathError(422, f"{deny} — inside {root.name}")
    return None


def _batch_label(batch: Path) -> str:
    """How the trash folder is named back to the browser.

    Repo-relative (``logs/trash/<batch>``) when it is the repo's own trash,
    because that is a path the user can ``cd`` to from the app's directory and
    the confirmation can quote. Absolute when a deployment (or a test) moved
    ``TRASH_DIR`` elsewhere — a label that is wrong is worse than one that is
    longer.
    """
    try:
        return str(batch.relative_to(constants.PROJECT_ROOT))
    except ValueError:
        return str(batch)


def trash_path(rel: str) -> Dict[str, Any]:
    """The delete route's core: move one path into the trash, or refuse.

    Steps, in this order — the order is the test list:

    * the **write** gate over the whole ancestor chain, so ``.git/config``,
      ``mcp_connectors/x.py``, ``.env`` and a NUL are refused in the deny-list's
      own words, before the path is looked for on disk;
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
      ``TRASH_DIR/<digest>_<stamp>/<the path exactly as it was, relative to the
      workspace>``. Keeping the relative layout is what makes a restore a plain
      ``mv`` — with a bare ``f.py`` one cannot tell ``src/f.py`` from
      ``web/f.py``, and a name collision inside one batch would be data loss.
      ``shutil.move`` preserves the subtree, and on a workspace that sits on
      another filesystem it degrades to a copy + delete: slower, and correct,
      which is the trade being taken here;
    * the sweep (:func:`sweep_trash`) runs after the move, so a failed prune can
      never lose the copy that was just made.

    The payload answers ``path``, ``dir``, the batch (``trashedTo``, relative to
    the app repo so the confirmation can name a path the user can ``cd`` to) and
    ``entries`` — the descendant count from the server's own walk, which is the
    number the confirmation dialog already promised.
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

    trash = Path(constants.TRASH_DIR)
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
