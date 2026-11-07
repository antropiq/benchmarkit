"""Seeding of the workspace's MCP declaration file.

``<working location>/.nojae/mcp-settings.json`` is the one source of MCP
connectors (see :mod:`mcphub.constants`). It joins the reserved folder's
family — ``read-write-policy.json``, ``agent-modes.json``,
``prompt-helpers.json`` — and arrives by the same rule: **seeded by
exclusive create at the two moments a folder becomes the working location
(boot, and ``POST /api/path-location``), never overwritten**.

The seeded bytes are the committed template
(``mcp_connectors/mcp-settings.default.json``) copied verbatim. Nothing is
substituted at seed time on purpose: its ``${BENCHMARKIT_PYTHON_ROOT}``,
``${PROJECT_ROOT}`` and ``${BENCHMARKIT_PATH_LOCATION}`` references resolve
on *every read* (:func:`mcphub.validation.expand_env`), so one seeded file
stays correct after a workspace switch and on a different machine — and the
template, which is what an operator reads to learn the format, is exactly
what the workspace gets.
The copy is made **as bytes** — read with ``read_bytes`` and written through a
binary handle — because a text-mode write on Windows turns every ``\n`` of the
template into ``\r\n``, and the copy would no longer be the template's bytes.

Unlike the policy and the modes there is **no cache to invalidate**: the hub
re-scans the declaration file on every load, so the first read after the
seed is the file itself.

The seeder lives in the hub package (the ``constants``/``discovery``/
``manager`` mirror) for the reason ``core.prompt_helpers`` gives for the
same choice: a seeder has to write exactly the bytes its loader reads, and
this loader is the hub's own.
"""

import os
import stat as stat_module
from typing import Any, Dict

from . import constants


def _state_dict(state: str, path: Any, error: Any = None) -> Dict[str, Any]:
    """The one payload shape the two state readers and the seeder answer with."""
    return {"state": state, "path": str(path), "error": error}


def _is_regular_file(path: str) -> bool:
    try:
        return stat_module.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def _discard_seeded_file(path: str) -> None:
    """Remove a file this call created but could not finish writing.

    Absence is the state the loader reads as "no connectors declared"; a
    half-written file would be a finding of its own about a mistake this
    call made. The failure the caller reports is the write error, not this
    cleanup's.
    """
    try:
        os.remove(path)
    except OSError:
        pass


def mcp_settings_file_path(root: str) -> str:
    """Where the declaration file goes under ``root`` — one level of the reserved folder.

    ``constants.mcp_settings_path()`` is the *active* root (what the loader
    reads); this takes the root explicitly, because the workspace switch
    seeds the folder it is moving to before ``path_location()`` is what the
    rest of the app sees.
    """
    return os.path.join(root, constants.POLICY_DIRNAME, constants.MCP_SETTINGS_FILENAME)


def default_mcp_settings_bytes() -> bytes:
    """The template's bytes, read fresh through the hub's ``constants``.

    The one place that touches the template. Read at call time (no cache, no
    copy at import) so a patched ``MCP_SETTINGS_TEMPLATE`` — the test that
    pins a missing template — moves the lookup like every other hub path.
    """
    template = constants.MCP_SETTINGS_TEMPLATE
    return template.read_bytes()


def ensure_mcp_settings_file(root: str) -> Dict[str, Any]:
    """Create ``<root>/.nojae/mcp-settings.json`` from the template if absent.

    Never overwrites: a hand-edited declaration — connectors added, entries
    disabled — survives every later seed byte for byte.

    Returns ``{"state": "created" | "existing" | "failed", "path": str,
    "error": Optional[str]}``. The whole function is one ``try``/``except``:
    a seed that cannot happen (read-only mount, a root owned by someone
    else, no disk, missing template) is a **NOTE** for the UI, never a
    reason to refuse opening a project. A workspace with no declaration file
    still chats — it simply has no MCP connectors — and the file can be
    written by hand afterwards, which is how every connector beyond the
    seeded two arrives anyway.
    """
    try:
        return _ensure_mcp_settings(root)
    except Exception as exc:  # noqa: BLE001 — a seed is a note, never a failure
        return _state_dict("failed", mcp_settings_file_path(root),
                           f"{type(exc).__name__}: {exc}")


def _ensure_mcp_settings(root: str) -> Dict[str, Any]:
    # The template first, the destination second: a template that cannot be
    # read must leave no half-seeded file behind at the workspace.
    try:
        data = default_mcp_settings_bytes()
    except OSError as exc:
        return _state_dict(
            "failed", constants.MCP_SETTINGS_TEMPLATE,
            f"the MCP settings template is not readable: {exc.strerror or exc}",
        )

    path = mcp_settings_file_path(root)
    folder = os.path.dirname(path)
    try:
        os.mkdir(folder, mode=0o755)
    except FileExistsError:
        pass
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))

    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if not _is_regular_file(path):
            # The loader reads a folder at this path as its own failure; the
            # seeder says the same thing rather than claiming a declaration
            # file the loader would reject.
            return _state_dict(
                "failed", path, "a folder stands where the MCP settings file belongs"
            )
        return _state_dict("existing", path)
    except PermissionError:
        # Which error a **folder** at that path answers with is the platform's
        # choice: ``EEXIST`` on POSIX, ``EACCES`` on Windows. Same condition,
        # same sentence, so the seeder says what stands there rather than
        # quoting an errno.
        if not _is_regular_file(path):
            return _state_dict(
                "failed", path, "a folder stands where the MCP settings file belongs"
            )
        return _state_dict("failed", path, "the MCP settings path cannot be written")
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except OSError as exc:
        _discard_seeded_file(path)
        return _state_dict("failed", path, exc.strerror or str(exc))

    return _state_dict("created", path)


def mcp_settings_file_state(root: str) -> Dict[str, Any]:
    """What a **reader** may say about the declaration file, writing nothing.

    ``GET /api/path-location`` asks this instead of
    :func:`ensure_mcp_settings_file` — the working location is read far more
    often than a workspace is switched, and a read that leaves a folder in
    someone's project is the mistake the seeding rules forbid. ``absent`` is
    a normal answer here: a workspace whose seed failed, or one that
    predates the file, simply has no MCP connectors.
    """
    path = mcp_settings_file_path(root)
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return _state_dict("absent", path)
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))
    if not stat_module.S_ISREG(info.st_mode):
        return _state_dict("failed", path, "the MCP settings path is a folder, not a file")
    return _state_dict("existing", path)
