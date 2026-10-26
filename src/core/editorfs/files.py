"""The file-level half of the Editor tab's surface: reading and stat'ing.

Two entry points, both gated exactly like the tree:

* :func:`read_file` — the whole text of one file (``GET /api/fs/file``), with
  the size cap, the binary sniff and the language guess;
* :func:`stat_paths` — the cheap ``mtime``/``size`` probe ed-09's live reload
  polls for the open buffers only, instead of a watcher or a whole-tree poll.

The write half belongs to a future version (ed-10) — this lot has no write
route at all (decision D4), so nothing here opens a file for writing, and
``policy.writable`` is the only write-shaped thing in the package.
"""

import os
from typing import Any, Dict, Iterable, List, Optional

from . import constants
from .policy import WorkspacePathError, resolve_in_workspace


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

    * the same gate as the tree (``resolve_in_workspace``, read flavour), so a
      path the connectors refuse is refused here too, naming the rule;
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
    ed-09 needs (the model deleted an open buffer's file), not an error. A path
    the policy refuses still raises :class:`WorkspacePathError` (422), so the
    probe can never be used to *stat* a secret file into existence: the answer
    for a denied path is a refusal, never ``exists: false``, which would leak
    that it is there.
    """
    results: List[Dict[str, Any]] = []
    for raw in paths:
        normalised, resolved = resolve_in_workspace(raw)
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
