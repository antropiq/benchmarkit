"""Batch mode: enumerate a ``todos/``-style lot of numbered task files.

Core-only (no FastAPI, no browser concerns), a sibling of :mod:`core.job`:
this is the *whole* file surface batch mode has. It lists a folder -- names,
sizes and the first heading -- and hands over the one body the client must be
able to send (the ``NN == 0`` big picture, which ``core.chat`` injects into
the system prompt). **Task bodies are never read here**: a task is named by
its root-relative path and the model opens the file with the file-touching
connectors, which is what keeps batch mode free of any file-read route.

Three locks bound the one path the caller controls, implemented here rather
than delegated to the (not yet existing) editor code:

* no absolute path and no ``..`` component;
* a ``realpath`` prefix check against ``config.path_location()``;
* the shared connector deny-list (``mcp_connectors/path_policy``) -- the same
  list both file-touching connectors use. Without it, ``?dir=../.ssh`` would
  make batch mode a GUI for reading a secret tree.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import PROJECT_ROOT, path_location

# Lot file names are ``<prefix><NN>-<slug>.md``: a leading alnum, then word
# characters. This one check is what keeps the query out of glob-injection
# territory (``*``, ``?``, ``[``, ``]``, ``/`` and NUL are all refused).
PREFIX_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
_PREFIX_RE = re.compile(PREFIX_PATTERN)

# Lot folder used when the caller names none, relative to the working location.
DEFAULT_LOT_DIR = "todos"

# Cap on the big-picture body the listing returns *and* the chat accepts (one
# number guards both sides, see ``routes.ChatIn.batchContext``). Over the cap
# the lot stays runnable -- it just loses its extra system-prompt context.
MAX_BATCH_LOT_BYTES = 64 * 1024
# Cap on how many tasks one lot may hand over. Entries past it are reported in
# ``skipped`` rather than silently vanishing.
MAX_BATCH_TASKS = 100
# How much of a file is scanned for its first heading (a dialog-only label).
HEADING_SCAN_BYTES = 4096

# The folder the per-task instruction moves finished files into; its absence
# is reported (``doneFolder``) because the instruction depends on it, and
# batch mode never creates a folder of its own.
DONE_SUBFOLDER = "done"


class BatchListingError(Exception):
    """A listing request that must be refused, carrying its HTTP status.

    The status belongs to the core because the rules below *are* the endpoint's
    contract -- 404 for "folder missing" versus 422 for "not a folder / not
    allowed" are two different user messages. ``routes`` only re-raises this as
    an ``HTTPException``.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# The number may only be followed by the usual slug separator (or the end of
# the stem): a prefix+number+letter stem like ``xx-1x`` is a file name, not
# task 1.
NUMBER_SEPARATORS = "-_"


def leading_digit_run(stem: str, prefix: str) -> str:
    """Digits right after ``prefix`` in ``stem`` ("" when the prefix differs)."""
    if not stem.startswith(prefix):
        return ""
    rest = stem[len(prefix):]
    index = 0
    while index < len(rest) and rest[index].isdigit():
        index += 1
    return rest[:index]


def parse_task_number(stem: str, prefix: str) -> Optional[int]:
    """The ``NN`` of ``stem``, or None when the stem carries no task number.

    The digit run *immediately after* the prefix, ``int()``-normalised
    (``xx-01`` -> 1, ``xx-007`` -> 7), so two- and three-digit numbers sort
    numerically and the ``NN`` need not be contiguous. Two shapes are not a
    number and land in ``skipped``: none at all (``xx-foo``) and a run that is
    not a token of its own (``xx-1x``) — accepting the latter would silently
    give one lot two task 1s.
    """
    if not stem.startswith(prefix):
        return None
    run = leading_digit_run(stem, prefix)
    if not run:
        return None
    rest = stem[len(prefix) + len(run):]
    if rest and rest[0] not in NUMBER_SEPARATORS:
        return None
    return int(run)


def skipped_reason(stem: str, prefix: str) -> str:
    """Why a prefix-matching stem is not a task (for the ``skipped`` list).

    Report, don't guess: the dialog explains every file that matched the
    prefix but is not runnable, so ``Found tasks 8`` next to an explanation
    line never reads as "the app lost a file".
    """
    run = leading_digit_run(stem, prefix)
    if not run:
        return "no task number after the prefix"
    return (
        f"task number {run} is not a separate token "
        f"(expected {prefix}NN-…)"
    )


def format_size(num_bytes: int) -> str:
    """Human-readable size for the notes ("big picture too large: 812 KB")."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes // 1024} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def entry_size(entry: os.DirEntry) -> int:
    """Size of a directory entry; 0 when its stat failed (it vanished)."""
    try:
        return entry.stat(follow_symlinks=False).st_size
    except OSError:
        return 0


def read_heading(path: Path) -> Optional[str]:
    """First ``# `` heading of ``path`` (marker stripped), None when none.

    Reads only the head of the file and stops at the first heading: this is a
    dialog label, not task content -- the number stays the authority, and a
    dropped heading costs the run nothing.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(HEADING_SCAN_BYTES)
    except OSError:
        return None
    for line in head.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def read_lot_text(path: Path, size: int) -> Tuple[Optional[str], Optional[str]]:
    """``(text, note)`` for the big-picture file; exactly one of them is set.

    Over :data:`MAX_BATCH_LOT_BYTES` the body is dropped and the note explains
    why -- the lot stays runnable, it only loses the extra system-prompt
    context, so this is reported rather than failing with a 422. An unreadable
    file is reported the same way instead of killing the listing.
    """
    if size > MAX_BATCH_LOT_BYTES:
        return None, (
            f"big picture too large to inject: {format_size(size)} "
            f"(max {format_size(MAX_BATCH_LOT_BYTES)})"
        )
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, f"big picture could not be read: {exc}"


def _path_policy():
    """The shared connector deny-list, imported the connectors' own way.

    ``mcp_connectors`` lives at the project root, outside the ``src`` package,
    so it is importable as a package only when that root is on ``sys.path``;
    the fallback mirrors what the connector modules do when launched as plain
    script paths. The module is resolved at call time so a patch of
    ``path_policy`` (the tests') is seen here too.
    """
    try:
        from mcp_connectors import path_policy
    except ImportError:  # pragma: no cover - depends on how the app is started
        import sys

        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from mcp_connectors import path_policy  # type: ignore[no-redef]
    return path_policy


def resolve_lot_dir(raw: Optional[str]) -> Tuple[str, Path]:
    """Normalise ``raw`` and resolve it to a folder inside the workspace.

    Three locks, in order: no absolute path and no ``..`` component; a
    ``realpath`` prefix check against the working location; then the connector
    deny-list, read-only flavour (a lot folder is only ever listed here). A name
    that exists but is not a folder is a 422 (a naming mistake), a name that is
    absent a 404 ("create the folder") -- two different dialog messages.
    """
    text = (raw or "").strip() or DEFAULT_LOT_DIR
    if os.path.isabs(text) or text.startswith("~"):
        raise BatchListingError(
            422,
            f"Batch folder must be relative to the workspace: {raw} "
            f"(workspace {path_location()})",
        )
    if ".." in Path(text).parts:
        raise BatchListingError(422, f"Batch folder must not contain '..': {raw}")
    normalised = os.path.normpath(text)
    if normalised in ("", ".", "..") or os.path.isabs(normalised):
        # A named folder, so every taskLocation handed to the model (and shown
        # in the dialog) is a clean root-relative path rather than "./x.md".
        raise BatchListingError(
            422,
            f"Batch folder must be a named folder inside the workspace: {raw}",
        )

    workspace = Path(path_location()).resolve()
    resolved = os.path.realpath(workspace / normalised)
    if resolved != str(workspace) and not resolved.startswith(
        str(workspace) + os.sep
    ):
        raise BatchListingError(
            422,
            f"Batch folder is outside the working location: {raw} "
            f"(workspace {workspace})",
        )
    deny = _path_policy().deny_reason(Path(resolved), for_write=False)
    if deny is not None:
        raise BatchListingError(
            422,
            f"Batch folder is refused by the connector path policy: {deny}",
        )
    if os.path.isfile(resolved):
        # A file is a naming mistake, not an absent lot: the dialog has to be
        # able to say "that is not a folder" rather than "create the folder".
        raise BatchListingError(422, f"Batch folder is not a directory: {raw}")
    if not os.path.isdir(resolved):
        raise BatchListingError(
            404,
            f"Batch folder does not exist: {text}",
        )
    return normalised, Path(resolved)


def task_lot(prefix: str, raw_dir: Optional[str] = None) -> Dict[str, Any]:
    """List the task lot of ``prefix`` in ``raw_dir``: the endpoint's core.

    Rules, in the order they apply (each unit-testable on its own):

    1. ``prefix`` must match :data:`PREFIX_PATTERN` (422 otherwise) -- the one
       check that keeps the query out of glob-injection territory.
    2. ``dir`` goes through :func:`resolve_lot_dir` (the three locks).
    3. One ``os.scandir`` of that folder, non-recursive: a subfolder (notably
       ``todos/done/``) is never descended into, so a file cannot belong to two
       lots and a task ``mv``'d away is simply gone from the next listing. The
       ``done`` entry itself is only *observed*, as ``doneFolder``.
    4. Only ``.md`` files whose name starts with the prefix are candidates
       (case-sensitive). A file that does not match the prefix is outside the
       lot by definition, so it is not mentioned at all -- not even in
       ``skipped``.
    5. ``NN == 0`` is the lot's big picture, ``NN >= 1`` a task, no digits a
       ``skipped`` entry. Tasks are sorted by number ascending.
    6. More tasks than :data:`MAX_BATCH_TASKS` sets ``truncated`` and moves the
       overflow into ``skipped`` with its number, instead of dropping it.

    ``taskCount`` is ``max(number)`` -- the highest task number, not a file
    count -- so one big-picture file plus tasks 1..8 reads ``8``, gaps included.
    Nothing matching is *not* an error: the payload reports ``taskCount: 0``
    with empty lists, because "no such folder" and "nothing here yet" are
    different things for the dialog to say.

    ``lot.text`` is the only file *content* batch mode ever returns: the
    client needs it to send the big picture (``ChatIn.batchContext``) while no
    file-read route exists. Tasks carry ``taskLocation`` -- the exact string
    interpolated into the per-task message -- so the frontend never assembles a
    path from parts.
    """
    if not prefix or not _PREFIX_RE.match(prefix):
        raise BatchListingError(
            422,
            "prefix must match ^[A-Za-z0-9][A-Za-z0-9_-]*$ -- a leading letter "
            "or digit, then letters, digits, '_' or '-'; no path or glob "
            "characters",
        )
    normalised, folder = resolve_lot_dir(raw_dir)

    try:
        with os.scandir(folder) as scan:
            entries = sorted(scan, key=lambda item: item.name)
    except OSError as exc:
        raise BatchListingError(
            404, f"Batch folder does not exist: {raw_dir or DEFAULT_LOT_DIR}"
        ) from exc

    tasks: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    lot_file: Optional[os.DirEntry] = None
    done_folder = False
    for entry in entries:
        if entry.is_dir():
            if entry.name == DONE_SUBFOLDER:
                done_folder = True
            continue
        name = entry.name
        if not name.startswith(prefix) or not name.endswith(".md"):
            continue
        # A symlink is a pointer somewhere else: following it would let a lot
        # folder read (and leak into the dialog) a file the folder does not
        # actually hold. The model's own tools resolve paths the same bounded
        # way, so refusing here costs nothing a run would otherwise get.
        if entry.is_symlink():
            skipped.append(
                {"name": name, "reason": "not a regular file (a symlink)"}
            )
            continue
        size = entry_size(entry)
        number = parse_task_number(name[: -len(".md")], prefix)
        if number is None:
            skipped.append(
                {
                    "name": name,
                    "reason": skipped_reason(name[: -len(".md")], prefix),
                }
            )
            continue
        if number == 0:
            if lot_file is None:
                lot_file = entry
            else:
                skipped.append(
                    {
                        "name": name,
                        "reason": "only one big-picture (00) file is used",
                    }
                )
            continue
        tasks.append(
            {
                "number": number,
                "name": name,
                "size": size,
                "taskLocation": f"{normalised}/{name}",
                "heading": read_heading(Path(entry.path)),
            }
        )

    tasks.sort(key=lambda task: (task["number"], task["name"]))
    truncated = len(tasks) > MAX_BATCH_TASKS
    if truncated:
        for task in tasks[MAX_BATCH_TASKS:]:
            skipped.append(
                {
                    "name": task["name"],
                    "reason": f"number {task['number']} > MAX_BATCH_TASKS "
                    f"({MAX_BATCH_TASKS})",
                }
            )
        tasks = tasks[:MAX_BATCH_TASKS]

    lot: Optional[Dict[str, Any]] = None
    lot_note: Optional[str] = None
    if lot_file is not None:
        lot_size = entry_size(lot_file)
        text, lot_note = read_lot_text(Path(lot_file.path), lot_size)
        lot = {
            "name": lot_file.name,
            "path": f"{normalised}/{lot_file.name}",
            "size": lot_size,
            "heading": read_heading(Path(lot_file.path)),
            "text": text,
        }

    return {
        "dir": normalised,
        "workspace": str(Path(path_location()).resolve()),
        "prefix": prefix,
        "taskCount": max((task["number"] for task in tasks), default=0),
        "doneFolder": done_folder,
        "lot": lot,
        "lotNote": lot_note,
        "tasks": tasks,
        "skipped": skipped,
        "truncated": truncated,
    }

