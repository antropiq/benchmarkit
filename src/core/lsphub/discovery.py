"""Discovery and validation of the declared language servers.

Reads the workspace's ``.nojae/lsp-settings.json`` into the dicts the rest
of the package consumes. The declaration path is looked up through
``lsphub.constants.lsp_settings_path()`` *inside* each function call, so a
switched working location moves the read and a patched pin redirects it —
one attribute, not an import dance. The deny-list is still an
``lsphub.constants`` attribute read the same way.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# The path-*text* questions the shared connector policy answers (win-02): what
# root a prefix is spelled with, on either platform, with no ``os.path`` branch.
# Imported the connectors' own way, because ``mcp_connectors`` sits outside
# ``src`` and is not always on ``sys.path`` (the same fallback
# ``mcphub.process`` and ``editorfs.constants`` use).
try:
    from mcp_connectors import path_policy
except ImportError:  # pragma: no cover - depends on how the app is started
    import sys

    _root = str(Path(__file__).resolve().parents[3])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from mcp_connectors import path_policy  # type: ignore[no-redef]

# The one containment rule (A-D2.2, win-06). Direction of this import: the
# editorfs surface reaches *this* module for ``restricted_lsp_roots`` — lazily,
# inside the one function that asks — while this module needs only the pure
# compare, which imports nothing from ``lsphub``. So no cycle: importing
# ``editorfs.policy`` here pulls in ``src.config`` and the ``mcp_connectors``
# policy modules, never ``lsphub``. ``fscase`` is where the fold answer comes
# from, and it is the *workspace's* filesystem that is probed for every one of
# these compares (see ``_is_within``'s docstring for that choice) — aliased,
# because this module's own entries carry a ``workspaceRoot`` and the local
# spelling of that name is a string, not the function.
from src.core.editorfs import fscase
from src.core.editorfs.policy import _is_within
from src.core.editorfs.policy import workspace_root as active_workspace_root

from src.core.mcphub.validation import (
    argv_from_command,
    expand_env,
    expand_env_json,
    expand_env_list,
    expand_env_map,
    str_map,
    unresolved_env_note,
)

from . import constants

logger = logging.getLogger("benchmarkit.lsp")


def lsp_installed() -> bool:
    """True when the optional LSP deps (the ``[lsp]`` extra) are importable."""
    try:
        return (
            importlib.util.find_spec("pygls") is not None
            and importlib.util.find_spec("lsprotocol") is not None
        )
    except (ImportError, ValueError):
        return False


def restricted_lsp_roots(path: Optional[Path] = None) -> List[str]:
    r"""Workspace-root prefixes refused by the hub, from ``restricted-paths.txt``.

    One absolute prefix per line, ``#`` starts a comment, and a line may name an
    environment variable as ``${NAME}`` (or ``${NAME:-fallback}``): it is expanded
    here, at read time, through the same helper a declaration's values go
    through, so one committed list stays machine-neutral and carries **both**
    platforms' path forms (B-D1). Nothing in the file may spell a drive letter or
    a home directory — ``${USERPROFILE}/.ssh`` and ``${SystemRoot}`` resolve to
    whatever the running OS states, and a runner that states neither gains
    nothing, which is what keeps the Linux verdicts unchanged.

    A line whose reference does **not** resolve is dropped and reported, not kept
    as written. For a declaration the rule is the opposite (an unresolved
    ``${NAME}`` stays verbatim so the typo is visible in that entry's note), and
    the difference is what makes an un-expanded line dangerous *here*: it is a
    prefix, so it reads to ``guardrails_speak_here`` as an entry that speaks this
    filesystem's root form while protecting nothing — the list would look
    meaningful, the reach seam would be lifted on that reading, and every entry
    that *did* resolve would go unenforced. A guardrail that cannot be honoured
    is absent, not guessed: the same sentence ``path_policy._env_absolute``
    already applies to the connector lists.
    """
    root = path or constants.LSP_RESTRICTED_PATHS
    try:
        lines = root.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries: List[str] = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        entry, missing = expand_env(text)
        entry = entry.strip()
        # ``missing`` first and on its own: a half-resolved line — ``/home/
        # ${NOPE}/.ssh`` — still has a rooted shape, so testing only the shape
        # would let the literal reference into the list as a working prefix.
        honoured = not missing and _is_prefix(entry)
        if not honoured:
            if missing:
                names = ", ".join(f"${{{name}}}" for name in dict.fromkeys(missing))
                reason = f"{names} is not set in the environment"
            else:
                reason = f"{text!r} is not an absolute path"
            _report_dropped(root, text, reason)
        # Only a resolved, absolute line joins the effective list: see the
        # docstring — a prefix that cannot be honoured must not even be counted
        # as one, and one guardrail stated twice is noise in a deny-list.
        if honoured and entry not in entries:
            entries.append(entry)
    return entries


# A dropped guardrail is *reported*, and the report is one line per process, not
# one per read: this list is re-read on every Editor path resolution and every
# hub scan, and a file written for the other platform drops its other-platform
# lines by design — three warnings per file read would bury the one line that
# names a real typo. Keyed by file and line, so a different file (or a test's own
# tmp copy) still gets its own sentence. Bounded so it cannot grow.
_REPORTED_DROPS: Set[Tuple[str, str]] = set()
_REPORTED_DROPS_MAX = 200


def _report_dropped(root: Path, text: str, reason: str) -> None:
    """Say once that a guardrail line is not in force (see ``_REPORTED_DROPS``)."""
    key = (str(root), text)
    if key in _REPORTED_DROPS:
        return
    if len(_REPORTED_DROPS) >= _REPORTED_DROPS_MAX:
        _REPORTED_DROPS.clear()
    _REPORTED_DROPS.add(key)
    logger.warning(
        "%s: %s — the guardrail %r is not in force", root.name, reason, text
    )


def _is_prefix(text: str) -> bool:
    r"""Is ``text`` an absolute prefix at all — on **either** platform?

    Asked of the shared text classifier, never of ``os.path.isabs``: that answers
    ``False`` for ``/etc`` on Windows, and a line is either a real POSIX guardrail
    or a dead one depending on the runner it was read on. What makes a prefix
    speak a path is its *root* (``path_policy.root_form``), the same basis
    ``lists_are_meaningful`` and the hub's own reach seam compare — so the file's
    POSIX half stays in force on a Windows runner, and only text with no root at
    all (``~/.ssh``, a relative path, a leading reference that resolved to
    nothing) is dropped for being unrooted.
    """
    return path_policy.root_form(text) in ("absolute", "drive", "unc")


def lsp_reach_is_refused() -> bool:
    """Must a declared language server be refused here? (win-07 / A-D4.)

    The LSP hub's call of the shared predicate (``mcphub.process``, which the
    MCP hub calls too), so a refusal cannot be argued differently in the two
    hubs; what differs per hub is *which* guardrails are asked about. For a
    language server that is :func:`restricted_lsp_roots` — the deny-list that
    would confine its ``workspaceRoot`` — and, when that file carries no
    entries, the connector prefix lists, so an absent list is not mistaken for
    a bound that has stopped matching.

    That is the whole question (lspfinal-04): what refuses here is an **inert
    list**, never an operating system. A rule that stops matching produces no
    error — the verdict falls back to allowing — so a language server whose
    ``workspaceRoot`` no prefix can name is one whose reach the app has bounded
    in words only, and that is refused. Everything else about the bound is a
    capability this hub holds on either platform: the spawn puts the child in a
    Job Object where there are no process groups, the teardown is the one shared
    entry point (B-D5, ``proctree`` / ``proctree_jobs``), and the list gains its
    Windows forms from the running environment at read time (B-D1). A
    deployment is therefore fixed by *writing values*, not by the code knowing
    which OS it is on.
    """
    return not constants.guardrails_speak_here(restricted_lsp_roots() or None)


def load_lsp_servers(
    path: Optional[Path] = None, reserved_ids: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read the workspace's ``.nojae/lsp-settings.json`` into server dicts.

    A missing file yields ``([], [])``. Unreadable/invalid JSON yields one
    file-level error and no servers. A single malformed entry is skipped with
    a per-entry note; unknown keys are ignored. Entries keep their ``enable``
    flag; the caller decides what to do with disabled ones. Never raises.

    ``command``/``args``, ``env``, ``workspaceRoot``, ``initializationOptions``
    and ``settings`` may reference environment variables as ``${NAME}`` (or
    ``${NAME:-fallback}``): they are expanded here, once, before any of the
    path/shape checks below run, so the rest of the package never sees a
    reference and one declaration file works on every machine. An entry that
    references an unset name is still loaded (subject to the usual checks)
    with a note saying so.
    """
    root = path or constants.lsp_settings_path()
    servers: List[Dict[str, Any]] = []
    errors: List[str] = []
    reserved = {str(entry) for entry in reserved_ids or ()}
    if not root.is_file():
        return servers, errors
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"{root.name} is not readable JSON: {exc}"]
    if not isinstance(data, dict):
        return [], [f"{root.name} must contain an object of servers keyed by id"]
    denied = [os.path.realpath(entry) for entry in restricted_lsp_roots()]
    # One reach answer per scan, not per server: the capability is a property of
    # the deployment, so every note in one load has to agree with every other.
    # This hub is asked about its own deny-list, the rules that would confine a
    # server's reach (``lsp_reach_is_refused``).
    refused_reach = lsp_reach_is_refused()
    for server_id in sorted(data):
        prefix = f"{root.name}: '{server_id}'"
        if server_id in reserved:
            errors.append(f"{prefix}: id is reserved — skipped")
            continue
        if server_id.startswith("_"):
            # Convention: an underscore key is a comment, not a server.
            continue
        entry = data[server_id]
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: entry must be a JSON object — skipped")
            continue
        # Everything that may carry a ``${NAME}`` reference is expanded here,
        # in one block, before the checks that consume it — so an unresolved
        # reference is reported by name *and* explains the shape error it
        # causes below (an unset ``${ROOT}`` turns into a relative path, which
        # would otherwise read like a typo in the file).
        missing: List[str] = []
        command = argv_from_command(entry.get("command"), entry.get("args"))
        if command is None:
            errors.append(f"{prefix}: an LSP server needs a command — skipped")
            continue
        command, names = expand_env_list(command)
        missing.extend(names)
        env = str_map(entry.get("env"))
        if env is None:
            errors.append(f"{prefix}: env must be an object of string values — skipped")
            continue
        env, names = expand_env_map(env)
        missing.extend(names)
        workspace_root = entry.get("workspaceRoot")
        if isinstance(workspace_root, str):
            workspace_root, names = expand_env(workspace_root)
            missing.extend(names)
        initialization_options = entry.get("initializationOptions")
        if initialization_options is not None and not isinstance(
            initialization_options, dict
        ):
            errors.append(
                f"{prefix}: initializationOptions must be an object — skipped"
            )
            continue
        initialization_options, names = expand_env_json(initialization_options)
        missing.extend(names)
        # Answers handed back to the server's own workspace/configuration
        # requests, keyed by LSP section ('html.customData',
        # 'vue.suggest.propNameCasing', …). Anything not listed is answered
        # null, i.e. "not set", which is what the LSP spec prescribes and
        # what makes a server's own ?? fallback land on its default.
        settings = entry.get("settings")
        if settings is not None and not isinstance(settings, dict):
            errors.append(f"{prefix}: settings must be an object — skipped")
            continue
        settings, names = expand_env_json(settings)
        missing.extend(names)
        note = unresolved_env_note(prefix, missing)
        if note is not None:
            errors.append(note)
        extensions = entry.get("extensions")
        if extensions is None:
            extensions = []
        if not isinstance(extensions, list) or not all(
            isinstance(item, str) and item.startswith(".") for item in extensions
        ):
            errors.append(
                f"{prefix}: extensions must be a list of '.ext' strings — skipped"
            )
            continue
        if not isinstance(workspace_root, str) or not os.path.isabs(workspace_root):
            errors.append(f"{prefix}: workspaceRoot must be an absolute path — skipped")
            continue
        real_root = os.path.realpath(workspace_root)
        if not os.path.isdir(real_root):
            errors.append(f"{prefix}: workspaceRoot does not exist — skipped")
            continue
        # The fold is the active workspace's answer, not this entry's: one
        # answer per running configuration, and no probe of a folder the hub is
        # about to refuse (``_is_within``'s docstring records that choice). Asked
        # only when the list has entries, so an empty deny-list costs no probe.
        fold = fscase.case_folded(active_workspace_root()) if denied else False
        if any(
            _is_within(denied_entry, real_root, fold=fold) for denied_entry in denied
        ):
            errors.append(f"{prefix}: workspaceRoot is on the deny-list — skipped")
            continue
        tools = entry.get("tools")
        if tools is not None and (
            not isinstance(tools, list)
            or not all(isinstance(item, str) and item for item in tools)
        ):
            errors.append(f"{prefix}: tools must be a list of names — skipped")
            continue
        max_diagnostics = entry.get("maxDiagnostics", constants.DEFAULT_MAX_DIAGNOSTICS)
        if not isinstance(max_diagnostics, int) or max_diagnostics < 1:
            errors.append(
                f"{prefix}: maxDiagnostics must be a positive integer — skipped"
            )
            continue
        idle_shutdown = entry.get(
            "idleShutdownSeconds", constants.DEFAULT_IDLE_SHUTDOWN_SECONDS
        )
        if not isinstance(idle_shutdown, int) or idle_shutdown < 0:
            errors.append(
                f"{prefix}: idleShutdownSeconds must be an integer (0 disables) — skipped"
            )
            continue
        if refused_reach and bool(entry.get("enable", False)):
            # Every declared server here is a stdio child process, and the lists
            # above are what confines its path reach — so when no entry in them
            # can name this filesystem's paths (win-07 / A-D4), there is no bound
            # to claim. The tree half of that bound is a capability on either
            # platform (the spawn's Job Object, the shared ``proctree`` teardown),
            # so it is no longer part of this question. The entry is still
            # loaded, because a card naming the gap beats a vanished server, and
            # the hub marks it ``error`` with these same words rather than
            # spawning it. Asked only of an entry that wants to run: a disabled
            # server is not a missing capability, it is a switch the user made.
            errors.append(
                f"{prefix}: {constants.LOCAL_REACH_REFUSAL} — not started"
            )
        label = entry.get("label")
        language_id = entry.get("languageId")
        servers.append(
            {
                "id": server_id,
                "label": label if isinstance(label, str) and label else server_id,
                "description": (
                    entry["description"]
                    if isinstance(entry.get("description"), str)
                    else ""
                ),
                "command": command,
                "workspaceRoot": workspace_root,
                "extensions": sorted({item.lower() for item in extensions}),
                "languageId": language_id if isinstance(language_id, str) else "",
                "enable": bool(entry.get("enable", False)),
                "lazy": bool(entry.get("lazy", False)),
                "initializationOptions": initialization_options or {},
                "tools": list(tools) if tools is not None else None,
                "env": env or {},
                "settings": settings or {},
                "maxDiagnostics": max_diagnostics,
                "idleShutdownSeconds": idle_shutdown,
            }
        )
    return servers, errors


def all_servers(
    path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Every declared language server plus the file's validation errors."""
    return load_lsp_servers(path)


def set_lsp_enabled(
    enabled: Dict[str, bool], path: Optional[Path] = None
) -> List[str]:
    """Write ``enable`` flags back to ``.nojae/lsp-settings.json``, keeping everything else.

    Only entries already present in the file are touched; other keys, entry
    order, and unknown ids are preserved. A missing file is a no-op; a file
    that cannot be parsed is left untouched and reported as an error list.
    """
    root = path or constants.lsp_settings_path()
    if not root.is_file():
        return []
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"Could not update {root.name}: {exc}"]
    if not isinstance(data, dict):
        return [f"Could not update {root.name}: top level is not an object"]
    for server_id, value in enabled.items():
        entry = data.get(server_id)
        if isinstance(entry, dict):
            entry["enable"] = bool(value)
    try:
        root.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return [f"Could not write {root.name}: {exc}"]
    return []
