"""The file-level half of the Editor tab's surface: reading, stat'ing, saving.

Entry points, all gated exactly like the tree:

* :func:`read_file` — the whole text of one file (``GET /api/fs/file``), with
  the size cap, the binary sniff and the language guess;
* :func:`stat_paths` — the cheap ``mtime``/``size`` probe the client polls for
  its open buffers only, instead of a watcher or a whole-tree poll;
* :func:`write_file` — one whole buffer back to disk (the save route): the
  write gate, the ``mtime`` guard, a pre-save backup and an atomic replace.
  The guarded write is browser-only — no connector or LSP tool reaches it —
  and it is the only code here that opens a file for writing.

A create, a rename or a delete is not here — this module is about reading plus
one whole buffer back to disk. Those operations on *paths* live in the sibling
:mod:`editorfs.mutate`, on the same gate.
"""

import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import constants
from .policy import (
    WorkspacePathError,
    missing_path_error,
    normalise_rel,
    resolve_save_target,
    resolve_in_workspace,
)

logger = logging.getLogger("benchmarkit.editor")


class WorkspaceSaveConflict(WorkspacePathError):
    """The 409 a stale ``mtime`` token answers — carrying both numbers.

    ``detail`` is a sentence for a log; :attr:`detail_object` is what the route
    puts on the wire, because the client has to *act* on a conflict (reload, or
    overwrite on purpose), not read about it.
    """

    def __init__(self, path: str, disk_mtime: float, buffer_mtime: float) -> None:
        super().__init__(
            409,
            f"{path} changed on disk since it was read "
            f"(disk {disk_mtime}, buffer {buffer_mtime})",
        )
        self.detail_object: Dict[str, Any] = {
            "reason": "stale",
            "path": path,
            "diskMtime": disk_mtime,
            "bufferMtime": buffer_mtime,
        }


# Extension -> editor language id. Computed server-side so the tree, the read
# endpoint and the browser cannot disagree about what a file *is*: the editor
# keys its grammar on exactly these ids, and an unknown extension (or a dotfile
# such as ``.gitignore``) is ``plaintext`` — never a guess. Ids the editor has
# no grammar for are honest labels it renders as plain text.
LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".vue": "vue",
    ".json": "json",
    ".jsonc": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shell",
}


def language_for(name: str) -> str:
    """The editor language id for one file name (``plaintext`` when unknown)."""
    lower = name.lower()
    index = lower.rfind(".")
    # index 0 is a dotfile (``.gitignore``) and -1 a name with no dot: neither
    # has an extension to key a grammar on, so neither gets a guess.
    if index <= 0:
        return "plaintext"
    return LANGUAGE_BY_EXTENSION.get(lower[index:], "plaintext")


def read_file(rel: str) -> Dict[str, Any]:
    """The payload of ``GET /api/fs/file`` for one workspace-relative path.

    Rules, each one mirrored by a test:

    * the same gate as the tree (:func:`policy.resolve_in_workspace`), so a path
      the tree would not have listed answers **404** here — an observation of a
      hidden name is indistinguishable from absence (pol-00 D8). A read-only
      name is read normally: ``ro`` constrains what changes, not what is seen,
      and the tree promised this row;
    * past ``constants.MAX_EDITOR_BYTES``: 413 naming the size, before any
      bytes are read — a browser tab is as little use for a giant file as the
      context window is (the same argument as ``lsphub.MAX_DOCUMENT_BYTES``);
    * a NUL in the first ``BINARY_SNIFF_BYTES``, or bytes that are not valid
      UTF-8: ``binary: true`` and **no ``text`` key** — an answer, not an
      error, so the pane can say "binary file, 2.1 MB" instead of garbage;
    * ``truncated`` plus the byte offset when a read ran past the cap (the file
      grew between the stat and the read): a partial is never presented as a
      whole, the same honesty rule as the ``tokensEstimated`` flag;
    * the text is the file's exact characters, CRLF included. Line endings are
      the editor's concern (Monaco keeps its own model ``EOL``), never something
      the server silently rewrites.
    """
    normalised, resolved = resolve_in_workspace(rel)
    try:
        stat = os.stat(resolved)
    except OSError:
        raise WorkspacePathError(
            404, f"File does not exist: {normalised or rel}"
        ) from None
    mode = stat.st_mode & 0o170000
    if mode == 0o040000:  # S_ISDIR
        raise WorkspacePathError(400, f"Path is a directory, not a file: {normalised}")
    if mode != 0o100000:  # S_ISREG — fifos, sockets, devices are not buffers
        raise WorkspacePathError(400, f"Path is not a regular file: {normalised}")

    size = stat.st_size
    cap = constants.MAX_EDITOR_BYTES
    if size > cap:
        raise WorkspacePathError(
            413,
            f"File is {size} bytes, over the {cap}-byte cap the editor loads",
        )

    try:
        with open(resolved, "rb") as handle:
            # cap + 1: that extra byte is what makes a read which ran past the
            # cap detectable rather than silently short.
            raw = handle.read(cap + 1)
    except OSError as exc:
        raise WorkspacePathError(404, f"File cannot be read: {exc}") from exc

    truncated = len(raw) > cap
    if truncated:
        raw = raw[:cap]
        # The stat said the file fit; the bytes say otherwise — it grew while
        # we read it. Ask again so ``size`` is not the stale number, and keep
        # the lower bound we proved (cap + 1 bytes) if the file is gone now.
        try:
            size = max(size, len(raw) + 1, os.stat(resolved).st_size)
        except OSError:
            size = max(size, cap + 1)

    # The name of the file as it was asked for ("" is impossible here: the root
    # is a directory and answered 400 above). Keying the language on it rather
    # than on the realpath's basename keeps the guess about the row the user
    # clicked, and keeps the payload free of any other name on disk.
    name = normalised.rsplit("/", 1)[-1]

    # The sniff itself: a NUL anywhere in the leading window. A NUL decodes
    # perfectly well as UTF-8, so the decode below cannot be the only test —
    # and the window is exactly why the constant exists (a real text file may
    # hold a stray NUL far into a long document; a binary one leads with one).
    binary_sniff = b"\0" in raw[: constants.BINARY_SNIFF_BYTES]

    payload: Dict[str, Any] = {
        "path": normalised,
        "size": size,
        "mtime": stat.st_mtime,
        "language": language_for(name),
        "binary": False,
        "truncated": truncated,
        "encoding": "utf-8",
    }
    if truncated:
        payload["truncatedAt"] = cap

    text = None if binary_sniff else _decode_utf8(raw, truncated=truncated)
    if text is None:
        payload["binary"] = True
        return payload  # no "text" key at all: the pane must not render garbage
    payload["text"] = text
    return payload


def _decode_utf8(raw: bytes, *, truncated: bool) -> Optional[str]:
    """Strict UTF-8 decode of the bytes, or None when they are not text.

    ``truncated`` only matters at the tail: a read cut mid-character would make
    an ordinary UTF-8 file look binary, so the incomplete sequence is dropped
    instead (``truncated: true`` already says the rest is missing). Anywhere
    else an undecodable byte means the file is not text, and the caller answers
    ``binary: true``.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        if not truncated:
            return None
        if exc.start >= len(raw) - 4 and exc.end >= len(raw):
            try:
                return raw[:exc.start].decode("utf-8")
            except UnicodeDecodeError:
                return None
        return None


def stat_paths(paths: Iterable[str]) -> List[Dict[str, Any]]:
    """``mtime``/``size``/``exists`` for each exact workspace-relative path.

    A path that is simply absent answers ``exists: false`` — that is the answer
    the freshness probe needs (the model deleted an open buffer's file), not an error. A path
    a policy **hides** answers that same ``exists: false`` (pol-00 D8): the probe
    only *looks*, and an observation of a hidden name has to be indistinguishable
    from absence — the opposite of the old gate's 422, where the refusal is what
    kept the probe from confirming a secret was there. A hidden name is not a
    secret from the browser any more: the user wrote the rule (pol-07 shows it),
    and the pane never receives the name from a listing that pruned it.

    What still raises :class:`WorkspacePathError` (422) is a **lock**: a path
    outside the working location, or a working location the app itself refuses.
    That is a refusal about the request rather than an answer about a file, so it
    travels for the whole batch instead of becoming a list of absences.
    """
    results: List[Dict[str, Any]] = []
    for raw in paths:
        try:
            normalised, resolved = resolve_in_workspace(raw)
        except WorkspacePathError as exc:
            if exc.status_code != 404:
                raise  # a lock, not a hidden name
            results.append(
                {"path": normalise_rel(raw), "exists": False, "mtime": None,
                 "size": None, "dir": False}
            )
            continue
        try:
            stat = os.stat(resolved)
        except OSError:
            results.append(
                {"path": normalised, "exists": False, "mtime": None, "size": None,
                 "dir": False}
            )
            continue
        results.append(
            {
                "path": normalised,
                "exists": True,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "dir": os.path.isdir(resolved),
            }
        )
    return results


# ---------------------------------------------------------------------------
# The guarded write: one whole buffer, back to disk.
# ---------------------------------------------------------------------------


def backup_key(resolved: Path) -> str:
    """The digest identifying *the file* a backup belongs to.

    Keyed on the resolved path rather than the spelling in the request, so the
    two mount names of one tree share a chain — the same reasoning the fedit
    connector's backups use (two spellings, two chains, and ``revert`` answers
    "never edited" about a file that was).
    """
    return hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]


def list_backups(resolved: Path) -> List[str]:
    """The chain of pre-save copies for one file, oldest first."""
    backup_dir = Path(constants.BACKUP_DIR)
    if not backup_dir.is_dir():
        return []
    return sorted(
        str(item) for item in backup_dir.glob(f"{backup_key(resolved)}_*.orig")
    )


def backup_for_write(resolved: Path) -> Optional[Path]:
    """Copy the current bytes aside before they are replaced.

    Returns the backup's path, or ``None`` when there was nothing on disk to
    copy — a save never creates, so that only happens if the file vanished
    between the gate and here, which the caller must not treat as an error.
    """
    if not resolved.is_file():
        return None
    backup_dir = Path(constants.BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", resolved.name)[:60]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    destination = backup_dir / f"{backup_key(resolved)}_{safe_name}_{stamp}.orig"
    shutil.copy2(resolved, destination)
    # ``copy2`` carries the *source's* mtime over, which would make a copy of a
    # long-untouched file be born past the age limit and get swept seconds
    # later. The sweep's clock is "how long has this been recoverable", so the
    # copy is stamped with the time of the copy. (The name already carries it;
    # this just makes the filesystem agree.)
    try:
        os.utime(destination, None)
    except OSError:
        pass
    prune_backup_chain(resolved)
    prune_old_backups()
    return destination


def prune_backup_chain(resolved: Path) -> int:
    """Keep the newest ``constants.BACKUP_KEEP`` copies of one file."""
    keep = constants.BACKUP_KEEP
    if keep <= 0:
        return 0
    chain = list_backups(resolved)
    removed = 0
    for stale in chain[:-keep] if len(chain) > keep else []:
        try:
            os.unlink(stale)
            removed += 1
        except OSError:
            continue
    return removed


def prune_old_backups(*, older_than_days: Optional[int] = None,
                      now: Optional[float] = None) -> int:
    """Delete every editor backup older than the retention window.

    The per-file keep above is right but incomplete: once a file is renamed or
    deleted, nothing collects the copies that referred to it, so age is the
    only signal that still works for a path that no longer exists. Cheap by
    design (one directory scan) and called from the write path, so it never
    raises — a copy that cannot be deleted is not worth failing a save over.
    """
    days = constants.BACKUP_PRUNE_DAYS if older_than_days is None else older_than_days
    backup_dir = Path(constants.BACKUP_DIR)
    if days <= 0 or not backup_dir.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - days * 86400
    removed = 0
    for item in backup_dir.glob("*.orig"):
        try:
            if item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def fsync_dir(path: str) -> None:
    """Durability hint for a directory whose entries just changed.

    A create, a rename and an atomic replace all leave a new *entry* behind, and
    that is what has to survive a crash — so the directory itself gets the fsync
    the file got. Failure is not a failure: the mutation already happened.
    """
    try:
        dir_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _atomic_replace(resolved: Path, payload: bytes, *, mode: Optional[int]) -> None:
    """``payload`` into a temp file in the target's directory, then rename over.

    Same-directory is the rule: a temp on another filesystem would turn
    ``os.replace`` into a copy, which is neither atomic nor cheap. Both fsyncs
    matter — the file's bytes, then the parent directory's entry, so the rename
    itself survives a crash. ``mode`` carries the target's permission bits onto
    the temp: a save must not quietly turn a ``0755`` script into ``0600``.
    """
    parent = os.path.dirname(str(resolved))
    handle = tempfile.NamedTemporaryFile(
        dir=parent, prefix=f".{os.path.basename(str(resolved))}.", suffix=".tmp",
        delete=False,
    )
    temp_path = handle.name
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_path, mode)
        os.replace(temp_path, resolved)
    except BaseException:
        # Never leave the temp behind on the failure path — it holds a full
        # copy of the buffer and no one will ever look for it.
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    # The replace already happened; a missed durability hint is not a failure.
    fsync_dir(parent)


def write_file(
    rel: str, text: str, *, if_unmodified_since: Optional[float] = None
) -> Dict[str, Any]:
    """The save route's core: one whole buffer back to disk, or a refusal.

    Rules, each one mirrored by a test:

    * the write gate on the **leaf** (:func:`policy.resolve_save_target`): the
      path itself must not be hidden or read-only, and a name the policy locks
      answers 422 naming the rule and the level that wrote it. A folder's own
      rule reaches every path under it, so a save inside a read-only folder is
      refused for that reason — while a later rule that marks one file ``rw``
      saves it, which is the carve-out the policy file is designed around;
    * 404 for an absent path, 400 for a directory or anything that is not a
      regular file — a save is not how a file is created;
    * the **mtime guard**: ``if_unmodified_since`` is the token ``GET
      /api/fs/file`` handed out, and the disk's current mtime differs from it
      (beyond ``MTIME_GUARD_EPSILON``, because a filesystem tick and a JSON
      float round-trip are not exact) → 409 carrying **both** numbers, as an
      object, so the client can offer reload-or-overwrite without parsing
      prose. Absent token means "no expectation" and skips the guard; that is
      a client bug, and it is logged rather than punished.
    * a file we never fully read is never written back: disk over
      ``MAX_EDITOR_BYTES`` → 413, whatever the buffer claims. The read route
      refused to load it, so nothing here has the right to replace it;
    * the payload itself is capped by the same number before a byte is
      written;
    * backup, then an atomic replace (see :func:`_atomic_replace`);
    * the payload answers the **re-stat'ed** ``size``/``mtime``, so the
      caller's next token is the server's truth and not a guess.

    Bytes go in exactly as they came: UTF-8, no line-ending normalisation, no
    BOM touched. A buffer that holds ``\\r\\n`` had it read from disk, and the
    write route keeps the read route's promise of exact characters.
    """
    # The write gate on the leaf, and a 404 when the leaf is not there — save
    # never creates, and never changes a folder's listing either.
    normalised, resolved = resolve_save_target(rel)

    # One stat for the two shape checks and the size cap. A vanished file here
    # still answers the write family's 404, never an unhandled OSError.
    try:
        stat = os.stat(resolved)
    except OSError:
        raise missing_path_error(normalised or rel) from None
    mode = stat.st_mode & 0o170000
    if mode == 0o040000:  # S_ISDIR
        raise WorkspacePathError(400, f"Path is a directory, not a file: {normalised}")
    if mode != 0o100000:  # S_ISREG — fifos, sockets, devices are not buffers
        raise WorkspacePathError(400, f"Path is not a regular file: {normalised}")

    if if_unmodified_since is None:
        # The client always sends the token; a save without one is the shape of
        # a bug, and logging it is how that bug stops being invisible.
        logger.warning("save of %s without an ifUnmodifiedSince token", normalised)
    elif abs(stat.st_mtime - float(if_unmodified_since)) > constants.MTIME_GUARD_EPSILON:
        # The detail is an object, not a sentence: the answer has to be acted
        # on (reload, or overwrite on purpose), not read.
        raise WorkspaceSaveConflict(normalised, stat.st_mtime, float(if_unmodified_since))

    cap = constants.MAX_EDITOR_BYTES
    if stat.st_size > cap:
        raise WorkspacePathError(
            413,
            f"File is {stat.st_size} bytes, over the {cap}-byte cap the editor loads",
        )

    payload = text.encode("utf-8")
    if len(payload) > cap:
        raise WorkspacePathError(
            413,
            f"Buffer is {len(payload)} bytes, over the {cap}-byte cap the editor writes",
        )

    backup_for_write(resolved)
    _atomic_replace(resolved, payload, mode=stat.st_mode & 0o7777)

    fresh = os.stat(resolved)
    return {
        "path": normalised,
        "size": fresh.st_size,
        "mtime": fresh.st_mtime,
        "backedUp": True,
    }
