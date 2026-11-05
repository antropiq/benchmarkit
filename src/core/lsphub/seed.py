"""Seeding of the workspace's LSP template file.

``<working location>/.nojae/lsp-settings.json`` is the one source of
language servers (see :mod:`lsphub.constants`), but unlike the MCP
declaration it is **never seeded as a working file**: every server entry
names machine-local paths (its ``workspaceRoot``, its interpreter shim), so
a seeded declaration would be a file the app refuses to start on every
machine but one. What the seed leaves behind instead is the *template* —
``<working location>/.nojae/lsp-settings.template.json``, a byte-for-byte
copy of the committed example — and the operator copies it to
``lsp-settings.json`` and adapts it.

The template joins the reserved folder's family — ``read-write-policy.json``,
``agent-modes.json``, ``prompt-helpers.json``, ``mcp-settings.json`` — and
arrives by the same rule: **seeded by exclusive create at the two moments a
folder becomes the working location (boot, and ``POST /api/path-location``),
never overwritten**, and created whenever it is absent (a folder that
gained ``.nojae`` by other means is offered the template too).

The seeded bytes are the committed example
(``lsp_connectors/lsp_servers.example.json``) copied verbatim. Nothing is
substituted at seed time on purpose: its ``${NAME}`` references resolve on
*every read* (:func:`mcphub.validation.expand_env`), so one seeded template
stays correct after a workspace switch and on a different machine.

Like the MCP seed there is **no cache to invalidate**: the hub re-scans the
declaration file on every load, and the template is never read by the hub
at all.

The seeder lives in the hub package (the ``constants``/``discovery``/
``manager``/``seed`` mirror) for the reason ``core.prompt_helpers`` gives
for the same choice: a seeder has to write exactly the bytes its loader
would read, and this loader is the hub's own.
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

    Absence is the state the operator reads as "no template here yet"; a
    half-written template would be a finding of its own about a mistake this
    call made. The failure the caller reports is the write error, not this
    cleanup's.
    """
    try:
        os.remove(path)
    except OSError:
        pass


def lsp_settings_template_file_path(root: str) -> str:
    """Where the template file goes under ``root`` — one level of the reserved folder.

    ``constants.lsp_settings_path()`` is the *active* root (what the loader
    reads); this takes the root explicitly, because the workspace switch
    seeds the folder it is moving to before ``path_location()`` is what the
    rest of the app sees.
    """
    return os.path.join(
        root, constants.POLICY_DIRNAME, constants.LSP_SETTINGS_TEMPLATE_FILENAME
    )


def lsp_settings_file_path(root: str) -> str:
    """Where the hand-written declaration goes under ``root`` (never seeded)."""
    return os.path.join(root, constants.POLICY_DIRNAME, constants.LSP_SETTINGS_FILENAME)


def default_lsp_settings_text() -> str:
    """The example's bytes, read fresh through the hub's ``constants``.

    The one place that touches the example. Read at call time (no cache, no
    copy at import) so a patched ``LSP_SETTINGS_TEMPLATE`` — the test that
    pins a missing example — moves the lookup like every other hub path.
    """
    template = constants.LSP_SETTINGS_TEMPLATE
    return template.read_text(encoding="utf-8")


def ensure_lsp_settings_template(root: str) -> Dict[str, Any]:
    """Create ``<root>/.nojae/lsp-settings.template.json`` from the example if absent.

    Never overwrites: a template someone trimmed or annotated — or replaced
    with their own house style — survives every later seed byte for byte.
    The declaration file itself is never created here; only the operator
    writes that, from this template.

    Returns ``{"state": "created" | "existing" | "failed", "path": str,
    "error": Optional[str]}``. The whole function is one ``try``/``except``:
    a seed that cannot happen (read-only mount, a root owned by someone
    else, no disk, missing example) is a **NOTE** for the UI, never a reason
    to refuse opening a project. A workspace with no template can still get
    a declaration — the example is committed — it just loses the copy-me
    shortcut.
    """
    try:
        return _ensure_lsp_settings_template(root)
    except Exception as exc:  # noqa: BLE001 — a seed is a note, never a failure
        return _state_dict("failed", lsp_settings_template_file_path(root),
                           f"{type(exc).__name__}: {exc}")


def _ensure_lsp_settings_template(root: str) -> Dict[str, Any]:
    # The example first, the destination second: an example that cannot be
    # read must leave no half-seeded file behind at the workspace.
    try:
        text = default_lsp_settings_text()
    except OSError as exc:
        return _state_dict(
            "failed", constants.LSP_SETTINGS_TEMPLATE,
            f"the LSP settings example is not readable: {exc.strerror or exc}",
        )

    path = lsp_settings_template_file_path(root)
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
            # A folder at the template path is the same broken shape the MCP
            # seeder reports: say what stands there rather than claim a file.
            return _state_dict(
                "failed", path, "a folder stands where the LSP settings template belongs"
            )
        return _state_dict("existing", path)
    except PermissionError:
        # Which error a **folder** at that path answers with is the platform's
        # choice: ``EEXIST`` on POSIX, ``EACCES`` on Windows. Same condition,
        # same sentence, so the seeder says what stands there rather than
        # quoting an errno.
        if not _is_regular_file(path):
            return _state_dict(
                "failed", path, "a folder stands where the LSP settings template belongs"
            )
        return _state_dict("failed", path, "the LSP settings template path cannot be written")
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        _discard_seeded_file(path)
        return _state_dict("failed", path, exc.strerror or str(exc))

    return _state_dict("created", path)


def lsp_settings_template_state(root: str) -> Dict[str, Any]:
    """What a **reader** may say about the template file, writing nothing.

    ``GET /api/path-location`` asks this instead of
    :func:`ensure_lsp_settings_template` — the settings dialog opens far more
    often than a workspace is switched, and a read that leaves a folder in
    someone's project is the mistake the seeding rules forbid. ``absent`` is
    a normal answer here: a workspace whose seed failed, or one that
    predates the file, simply has no template.
    """
    path = lsp_settings_template_file_path(root)
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return _state_dict("absent", path)
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))
    if not stat_module.S_ISREG(info.st_mode):
        return _state_dict("failed", path, "the LSP settings template path is a folder, not a file")
    return _state_dict("existing", path)


def lsp_settings_file_state(root: str) -> Dict[str, Any]:
    """What a **reader** may say about the declaration file, writing nothing.

    The declaration is hand-written (the app only ever *toggles* it), so this
    is the one report of whether the workspace has servers declared at all:
    ``absent`` is a plain fact — the panel can then point the operator at the
    seeded template.
    """
    path = lsp_settings_file_path(root)
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return _state_dict("absent", path)
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))
    if not stat_module.S_ISREG(info.st_mode):
        return _state_dict("failed", path, "the LSP settings path is a folder, not a file")
    return _state_dict("existing", path)
