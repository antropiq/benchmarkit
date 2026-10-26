"""Which workspace files git considers *unstaged*, and the two sides of a diff.

Two entry points for the Editor tab's navigator:

* :func:`git_status` — the changed paths of the working location
  (``GET /api/fs/status``), so a tree row can say "this file changed outside
  the app";
* :func:`file_diff` — the committed side and the on-disk side of one file
  (``GET /api/fs/diff``), for a diff tab.

Both are **read-only git queries**: nothing here stages, commits or checks out
anything, and the write stance of this package (decision D4) is untouched.

Four rules the two endpoints share:

* **the same gate as the tree.** Every requested path goes through
  :func:`~editorfs.policy.resolve_in_workspace`, and every path *git* reported
  is re-filtered with the tree's two omission rules (an ignored directory
  name, the connector deny-list). Otherwise ``/api/fs/status`` would be a way
  to learn that a denied path holds a modified file the tree refuses to show.
* **one definition of "unstaged"**: the *worktree* column of ``porcelain v1``
  is not clean. A tracked file modified, deleted, type-changed or unmerged, and
  an untracked file, are listed. A change that lives only in the index
  (``git add`` and nothing since) is not.
* **paths are workspace-relative**, like every other Editor payload. ``git``
  always reports repository-toplevel-relative names, so ``--show-prefix`` maps
  them back and the pathspec (``-- .``) keeps the query inside the browsable
  subtree. When the working location is a subdirectory of a checkout, the rest
  of that repo is simply not listed and ``atToplevel: false`` says why the list
  can look short.
* **a location with no repository is not an error**: ``repo: false`` and an
  empty list, so the navigator draws no markers and offers no "View diff…"
  item. A *broken* git run (no binary, a timeout, a corrupt repo) answers 504
  instead — a marker is decoration the front end may ignore, but it must never
  be handed "nothing changed" as a lie.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import constants
from .files import language_for, read_file
from .policy import WorkspacePathError, listable, resolve_in_workspace, workspace_root

def _git(args: List[str], cwd: str) -> subprocess.CompletedProcess:
    """Run one git command in ``cwd``, capturing bytes; a non-zero exit is the
    caller's to interpret, so this never raises for one.

    ``GIT_OPTIONAL_LOCKS=0`` stops ``status`` from refreshing the index on
    disk: this surface *reports* the working tree, it does not tidy it. Every
    caller that parses paths passes ``-z``, which keeps names byte-exact — a
    name with a space, a quote or a newline then needs no unquoting.
    """
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=constants.GIT_COMMAND_TIMEOUT,
        )
    except FileNotFoundError as exc:  # no git on this machine
        raise WorkspacePathError(
            504, "git is not available, so file status cannot be reported"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkspacePathError(
            504, f"git timed out after {constants.GIT_COMMAND_TIMEOUT}s"
        ) from exc


def _stderr(proc: subprocess.CompletedProcess) -> str:
    return proc.stderr.decode("utf-8", "replace").strip()


def repo_info() -> Optional[Dict[str, str]]:
    """``{root, prefix}`` for the repository holding the workspace, else None.

    ``prefix`` is the workspace's own path *inside* the repository ("" at the
    toplevel): the map between git's toplevel-relative names and the
    workspace-relative names every Editor payload uses. None means "not a
    working tree" — the silent answer, not an error.
    """
    proc = _git(
        ["rev-parse", "--show-toplevel", "--show-prefix", "--is-inside-work-tree"],
        workspace_root(),
    )
    if proc.returncode != 0:
        return None
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    if len(lines) < 3 or lines[2].strip() != "true":
        return None  # inside a .git dir, or a bare repo: no worktree to mark
    return {"root": lines[0].strip(), "prefix": lines[1].strip().strip("/")}


def _to_workspace_rel(path: str, prefix: str) -> Optional[str]:
    """Repository-relative -> workspace-relative, or None when it is outside."""
    normalised = path.replace(os.sep, "/").lstrip("/")
    if prefix:
        if not normalised.startswith(f"{prefix}/"):
            return None
        normalised = normalised[len(prefix) + 1 :]
    return normalised or None


def _displayable(rel: str) -> bool:
    """True when ``rel`` may appear in a payload the navigator renders.

    The tree's two omission rules, applied to a path *git* chose: without them
    a status list would name files the tree deliberately hides, which is the
    leak ``editorfs.policy`` exists to prevent.
    """
    if any(part in constants.IGNORED_DIR_NAMES for part in rel.split("/")):
        return False
    return listable(Path(os.path.join(workspace_root(), rel)))


def _records(raw: bytes) -> List[str]:
    """Split a ``-z`` field list into text.

    ``surrogateescape`` is what ``os.fsdecode`` uses, so a name that is not
    valid UTF-8 survives the round trip unchanged — it has to, because the same
    text goes back into a pathspec and into ``resolve_in_workspace`` for the
    diff of that very file.
    """
    return [
        chunk.decode("utf-8", "surrogateescape")
        for chunk in raw.split(b"\0")
        if chunk
    ]


def classify(index_code: str, worktree_code: str) -> Tuple[str, bool]:
    """``(kind, comparable)`` for one porcelain status pair.

    ``comparable`` answers "is there a committed side to diff against?" — false
    for an untracked file and for one that is new since the last commit (its
    index column is ``A``). That is exactly what lets the UI mark such a file
    without offering a diff against nothing.
    """
    if index_code == "?" and worktree_code == "?":
        return "untracked", False
    if "U" in (index_code, worktree_code):
        return "conflict", True
    if worktree_code == "D":
        return "deleted", True
    if index_code == "A":
        return "added", False
    if index_code in ("R", "C"):
        return "renamed", True
    return "modified", True


def _status_records(workspace: str, scope: str) -> List[List[str]]:
    """The parsed ``porcelain v1 -z`` records for one scope.

    Each record is ``[index_code, worktree_code, path, old_path_or_empty]``;
    the fourth field only exists for a rename/copy, whose old path arrives in
    the *next* ``-z`` field. ``-M`` is what makes a worktree rename visible at
    all — without it, a moved tracked file is a delete plus an untracked add,
    which would both hide the rename and claim the new name has no baseline.
    """
    proc = _git(
        [
            "status",
            "--porcelain=v1",
            "-z",
            "-M",
            "--untracked-files=all",
            "--",
            scope,
        ],
        workspace,
    )
    if proc.returncode != 0:
        raise WorkspacePathError(
            504, f"git status failed: {_stderr(proc) or f'exit {proc.returncode}'}"
        )
    chunks = _records(proc.stdout)
    records: List[List[str]] = []
    index = 0
    while index < len(chunks):
        record = chunks[index]
        index += 1
        old_path = ""
        # The rename's source only follows a rename record; anything else is a
        # real path and must not be swallowed as its partner.
        if record[:1] in ("R", "C") and index < len(chunks):
            old_path = chunks[index]
            index += 1
        if len(record) < 4 or record[3:] == "":
            continue
        records.append([record[0], record[1], record[3:], old_path])
    return records


def git_status() -> Dict[str, Any]:
    """The unstaged files of the working location, workspace-relative."""
    repo = repo_info()
    if repo is None:
        return {
            "repo": False,
            "root": None,
            "atToplevel": True,
            "entries": [],
            "truncated": False,
        }

    workspace = workspace_root()
    entries: List[Dict[str, Any]] = []
    truncated = False
    for index_code, worktree_code, path, old_path in _status_records(workspace, "."):
        if worktree_code == " ":
            # The worktree matches the index: whatever changed is staged, and
            # "staged" is not "unstaged".
            continue
        rel = _to_workspace_rel(path, repo["prefix"])
        if rel is None or not _displayable(rel):
            continue
        kind, comparable = classify(index_code, worktree_code)
        if len(entries) >= constants.MAX_STATUS_ENTRIES:
            truncated = True
            break
        entry: Dict[str, Any] = {"path": rel, "kind": kind, "comparable": comparable}
        old_rel = _to_workspace_rel(old_path, repo["prefix"]) if old_path else None
        if old_rel:
            entry["oldPath"] = old_rel
        entries.append(entry)

    return {
        "repo": True,
        "root": repo["root"],
        "atToplevel": repo["prefix"] == "",
        "entries": entries,
        "truncated": truncated,
    }


def _head_blob(workspace: str, *repo_paths: str) -> Tuple[Optional[str], bool]:
    """``(text_or_None, binary)`` for the first of ``repo_paths`` in ``HEAD``.

    ``(None, False)`` means no committed side exists at all — the empty left
    half of a file that is new since the last commit. The paths are tried in
    order so a rename can fall back to the name the blob is stored under (the
    worktree copy has moved, the blob has not).

    A NUL in the leading window, or bytes that are not UTF-8, answer
    ``("", True)``: the *same* binary answer the file endpoint gives, decided
    by the *same* sniff window. The blob side is checked here because its own
    bytes only exist on this path.
    """
    for repo_path in repo_paths:
        proc = _git(["show", f"HEAD:{repo_path}"], workspace)
        if proc.returncode != 0:
            continue
        raw = proc.stdout
        if b"\0" in raw[: constants.BINARY_SNIFF_BYTES]:
            return "", True
        try:
            # Stricter than the file reader's decode on purpose: a blob arrives
            # whole, so there is no cut-mid-character tail to forgive and an
            # undecodable byte simply means this pane cannot diff it.
            return raw.decode("utf-8"), False
        except UnicodeDecodeError:
            return "", True
    return None, False


def _status_one(workspace: str, repo_path: str) -> Tuple[str, Optional[str]]:
    """``(kind, oldPath)`` for one repository-relative path, asked of git.

    The diff endpoint does not trust the marker the click came from — the tree
    may be showing a status from before a commit or a ``git checkout`` — so the
    answer's ``status`` is what git thinks *now*.

    The query is the whole work tree, the same one ``git_status`` runs, and the
    path is then matched in Python. Naming the file in the pathspec instead
    would be both less safe (a name that looks like a glob would have to be
    escaped with ``:(literal)``) and **less correct**: rename detection pairs a
    path with its source, so a query scoped to one path reports a rename as a
    plain add and the file would wrongly look like it has no committed side.
    """
    for index_code, worktree_code, path, old_path in _status_records(workspace, "."):
        if path != repo_path:
            continue
        kind, _ = classify(index_code, worktree_code)
        return kind, old_path or None
    # Unchanged since HEAD: the empty diff, honestly reported. The UI only
    # offers the item for a marked file, so this is a refresh race, not the norm.
    return "clean", None


def file_diff(rel: str) -> Dict[str, Any]:
    """The payload of ``GET /api/fs/diff``: the two sides of one file.

    ``original`` is the file **as of ``HEAD``** and ``modified`` the file on
    disk, so the answer is "what has changed since the last commit" — which
    also covers a file that was staged and then edited again. Returning the two
    sides rather than a patch is what lets the editor compute and render the
    diff itself (word-level, inline, collapsible) instead of the app parsing a
    patch into a widget. A file not in ``HEAD`` (added, untracked) gets an empty
    left side, and the UI does not offer the item for one at all.
    """
    normalised, _resolved = resolve_in_workspace(rel)
    repo = repo_info()
    if repo is None:
        raise WorkspacePathError(504, "The working location is not a git repository")

    prefix = repo["prefix"]
    repo_path = f"{prefix}/{normalised}" if prefix else normalised
    kind, old_path = _status_one(workspace_root(), repo_path)
    old_repo_path = old_path.replace(os.sep, "/") if old_path else None

    payload: Dict[str, Any] = {
        "path": normalised,
        "language": language_for(normalised.rsplit("/", 1)[-1]),
        "status": kind,
        "baseline": "head",
        "repoRoot": repo["root"],
        "atToplevel": prefix == "",
        "binary": False,
        "truncated": False,
    }
    if old_repo_path:
        payload["renamedFrom"] = _to_workspace_rel(old_repo_path, prefix) or old_repo_path

    # The disk side first: the *same* reader as the file endpoint, so the size
    # cap, the binary sniff and the policy answer for a diff exactly as they do
    # for a buffer.
    if kind == "deleted":
        # Deleted in the worktree: there is no disk side and ``read_file``
        # would answer 404. The empty right half *is* the answer.
        current_text, truncated = "", False
    else:
        current = read_file(normalised)
        if current.get("binary"):
            payload["binary"] = True
            return payload  # no sides at all, exactly as the file endpoint
        current_text = current.get("text", "")
        truncated = bool(current.get("truncated"))
        if truncated:
            payload["truncatedAt"] = current.get("truncatedAt")

    # The old name second: a rename has moved the worktree copy, but the blob
    # is still stored under the name HEAD knows it by.
    candidates = (repo_path, old_repo_path) if old_repo_path else (repo_path,)
    original, original_binary = _head_blob(workspace_root(), *candidates)
    if original_binary:
        payload["binary"] = True
        return payload

    payload["original"] = original or ""
    payload["modified"] = current_text
    payload["truncated"] = truncated
    return payload
