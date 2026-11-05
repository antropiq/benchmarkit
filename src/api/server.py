"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import RunState, build_router
from src.config import (
    apply_llm_debug,
    ENV_PATH,
    load_dotenv_once,
    PROJECT_ROOT,
    path_location,
    questions_path,
    setup_path_location,
)
from src.core.mcphub import (
    McpManager,
    ensure_mcp_settings_file,
    load_externals,
)
from src.core.lsphub import (
    LspManager,
    all_servers,
    ensure_lsp_settings_template,
)
from src.core.editorfs import mutate as editorfs_mutate, policy_seed
from src.core.termhub import TerminalHub
from src.core import agent_modes
from src.core import prompt_helpers
from src.config import lsp_enabled_ids

WEB_DIST = PROJECT_ROOT / "web" / "dist"

_log = logging.getLogger("benchmarkit")


def _seed_workspace_policy() -> Dict[str, Any]:
    """Leave the workspace's default policy file behind; never raise.

    ``ensure_policy_file`` already answers a dict instead of raising, so this
    wrapper only guards the *unexpected* — and its one job is that a policy file
    this app could not write must not take the boot down with it: an unseeded
    workspace simply has no level-B rules yet (level A, the app's own rules, is
    built in and still applies).
    """
    try:
        result = policy_seed.ensure_policy_file(path_location())
    except Exception as exc:  # noqa: BLE001 — boot must survive anything here
        _log.warning("could not seed the workspace policy: %s", exc)
        return {"state": "failed", "path": None, "error": str(exc)}
    if result["state"] == "failed":
        _log.warning(
            "could not seed the workspace policy at %s: %s",
            result["path"],
            result["error"],
        )
    return result


def _seed_agent_modes() -> Dict[str, Any]:
    """Leave the workspace's default agent-modes file behind; never raise.

    am-03's first of two entry points, sitting beside the policy seed because it
    is the same contract: ``ensure_agent_modes_file`` already answers a dict
    instead of raising, so this wrapper only guards the *unexpected*, and a
    modes file this app could not write must not take the boot down with it. An
    unseeded workspace chats on the built-in modes — its dropdown simply has
    nothing of its own to customise yet.
    """
    try:
        result = agent_modes.ensure_agent_modes_file(path_location())
    except Exception as exc:  # noqa: BLE001 — boot must survive anything here
        _log.warning("could not seed the agent-modes file: %s", exc)
        return {"state": "failed", "path": None, "error": str(exc)}
    if result["state"] == "failed":
        _log.warning(
            "could not seed the agent-modes file at %s: %s",
            result["path"],
            result["error"],
        )
    return result


def _seed_prompt_helpers() -> Dict[str, Any]:
    """Leave the workspace's default prompt-helpers file behind; never raise.

    mbl-02's first of two entry points, the third sibling of the policy and
    modes seeds and the same contract: ``ensure_prompt_helpers_file`` already
    answers a dict instead of raising, so this wrapper only guards the
    *unexpected*, and a helpers file this app could not write must not take the
    boot down with it. An unseeded workspace chats on nine empty bullets —
    :func:`load_prompt_helpers` answers the defaults for an absent file.

    This is also the one moment the old app-level ``resources/`` file is read
    (spec D6): the lift happens on the branch where the file was just created,
    and the legacy file itself is left in place.
    """
    try:
        result = prompt_helpers.ensure_prompt_helpers_file(path_location())
    except Exception as exc:  # noqa: BLE001 — boot must survive anything here
        _log.warning("could not seed the prompt-helpers file: %s", exc)
        return {"state": "failed", "path": None, "error": str(exc)}
    if result["state"] == "failed":
        _log.warning(
            "could not seed the prompt-helpers file at %s: %s",
            result["path"],
            result["error"],
        )
    return result


def _seed_mcp_settings() -> Dict[str, Any]:
    """Leave the workspace's MCP declaration file behind; never raise.

    The second half of the pair the declaration file needs: it lives inside
    ``.nojae`` now, so the boot seed must land *before* the lifespan's first
    connector scan — a hub rebuilt over an unseeded folder would come up with
    zero connectors. Same contract as the policy, modes and helpers seeds:
    ``ensure_mcp_settings_file`` already answers a dict instead of raising, so
    this wrapper only guards the *unexpected*, and a declaration file this app
    could not write must not take the boot down with it. An unseeded workspace
    simply has no MCP connectors — its chat works, its tools do not, and the
    Settings → MCP panel says so.
    """
    try:
        result = ensure_mcp_settings_file(path_location())
    except Exception as exc:  # noqa: BLE001 — boot must survive anything here
        _log.warning("could not seed the MCP settings file: %s", exc)
        return {"state": "failed", "path": None, "error": str(exc)}
    if result["state"] == "failed":
        _log.warning(
            "could not seed the MCP settings file at %s: %s",
            result["path"],
            result["error"],
        )
    return result


def _seed_lsp_template() -> Dict[str, Any]:
    """Leave the workspace's LSP declaration *template* behind; never raise.

    The language-server declaration lives inside ``.nojae`` now, but unlike
    the MCP one it is **never seeded as a working file** — every entry names
    machine-local paths — so what lands here is the template
    (``.nojae/lsp-settings.template.json``) the operator copies to
    ``lsp-settings.json``. Same contract as the other four seeds:
    ``ensure_lsp_settings_template`` already answers a dict instead of
    raising, so this wrapper only guards the *unexpected*, and a template
    this app could not write must not take the boot down with it. A
    workspace with no declaration simply has no language servers — its chat
    works without ``lsp_*`` tools, and the Settings → LSP panel says so.
    """
    try:
        result = ensure_lsp_settings_template(path_location())
    except Exception as exc:  # noqa: BLE001 — boot must survive anything here
        _log.warning("could not seed the LSP settings template: %s", exc)
        return {"state": "failed", "path": None, "error": str(exc)}
    if result["state"] == "failed":
        _log.warning(
            "could not seed the LSP settings template at %s: %s",
            result["path"],
            result["error"],
        )
    return result


def _sweep_trash() -> None:
    """Expire old trash batches — including the pre-``.nojae`` ones in ``logs/``.

    The sweep normally runs after a delete, so a folder nothing is ever deleted
    into again would rot forever. One best-effort call per boot is the cheapest
    way to keep the legacy ``logs/trash`` batches from outliving their
    retention, and the same rule as the seed above applies: a folder this app
    could not prune must not take the boot down with it. It never *moves* those
    batches, only lets them expire.
    """
    try:
        removed = editorfs_mutate.sweep_trash()
    except Exception as exc:  # noqa: BLE001 — boot must survive anything here
        _log.warning("could not sweep the trash: %s", exc)
        return
    if removed:
        _log.info("removed %d expired trash batch(es)", removed)


def create_app() -> FastAPI:
    """Build the FastAPI app: API routes first, static SPA fallback after."""
    load_dotenv_once()
    # The debug switch is read from the environment, so it is applied right
    # after the .env has been loaded (and again on every settings save).
    # Keeping it here rather than in the module's import side effects means the
    # tests build an app without touching the loggers of the whole process
    # unless they ask for it.
    apply_llm_debug()
    # The working location the MCP connectors and language servers point at
    # (``${BENCHMARKIT_PATH_LOCATION}`` in both declaration files) must exist
    # before the very first scan of those files — which happens below, in the
    # lifespan — so it is seeded here, right after the .env is read. Booting
    # always lands on the install directory unless the environment already
    # carries a value.
    setup_path_location()
    # Seed the workspace policy for whichever location that chose — the second
    # of the two entry points ``POST /api/path-location`` is the other. This is
    # how benchmarkit gets its own ``.nojae/read-write-policy.json`` on first
    # launch, and why a project opened from the settings dialog arrives with a
    # tree already filtered by rules the user can read.
    #
    # Called from here rather than from inside ``setup_path_location()`` on
    # purpose: ``src/config.py`` sits at the bottom of the dependency graph
    # (``editorfs`` imports ``path_location`` *from* it), so a call there would
    # need an import cycle or a local import in a config module. The app is the
    # only thing that boots, so one line beside the line it follows is the
    # honest place. Failure is a note in the log, never a boot failure: an
    # unseeded workspace still works, it just has no level-B rules yet.
    _seed_workspace_policy()
    # The agent-modes file (am-03) follows the same rule and the same moment:
    # seeded here at boot, and by ``POST /api/path-location`` on a switch. It
    # goes after the policy seed because both belong to the same reserved
    # folder, and the policy seed is the one that creates it — so in the usual
    # case this is a plain file write, not a second mkdir. Failure is a note in
    # the log (and, on the switch, in the payload), never a boot failure: a
    # workspace with no modes file chats on the built-in defaults.
    _seed_agent_modes()
    # The prompt-helpers file follows the same rule and the same two moments:
    # same reserved folder, so this runs after the seeds that create it and is a
    # plain file write in the usual case. Failure is a note in the log (and, on
    # the switch, in the payload), never a boot failure — a workspace with no
    # helpers file chats on nine empty bullets. It is *not* called from
    # ``config.path_location()`` (spec D5): that is a per-request reader, and a
    # reader that writes is how a stray ``.nojae`` turns up in a project someone
    # only meant to browse; ``POST /api/path-location`` is the other moment.
    _seed_prompt_helpers()
    # And the workspace's MCP declaration file: it lives inside ``.nojae``
    # now, so this seed must land before the lifespan's first connector scan
    # below — that is the whole ordering rule for this one. Same reserved
    # folder, same contract: a note in the log, never a boot failure; a
    # workspace with no declaration file simply has no MCP connectors.
    _seed_mcp_settings()
    # And the workspace's LSP declaration *template*: the declaration itself
    # moved into ``.nojae`` too, but it is hand-written (its entries are
    # per-machine), so this seed only offers the copy-me template beside it.
    # Same reserved folder, same contract: a note in the log, never a boot
    # failure; a workspace with no declaration file simply has no language
    # servers. Nothing scans the template, so — unlike the MCP seed — its
    # ordering against the lifespan's first server scan carries no rule.
    _seed_lsp_template()
    # And drain the trash of whichever location that is, plus the legacy
    # ``logs/trash`` batches the pre-``.nojae`` code wrote here. Age is the only
    # signal for a batch nobody addresses any more, and a boot is the one moment
    # guaranteed to happen even if no further delete ever does.
    _sweep_trash()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # MCP connector sessions live exactly as long as the web app: connected
        # here on startup, closed on shutdown. The env var is a test/dev switch
        # that keeps the endpoints alive but connects nothing. Every connector
        # is declared in the workspace's .nojae/mcp-settings.json (seeded at
        # boot above); entries with ``enable: false`` stay stopped until
        # toggled on.
        connectors = (
            [] if os.environ.get("BENCHMARKIT_DISABLE_MCP") else load_externals()[0]
        )
        manager = McpManager(connectors)
        app.state.mcp = manager
        await manager.start_all()
        # Language servers follow the same rule (own env-var switch, own log
        # dir): spawned with the app, shut down (shutdown/exit then
        # SIGTERM/SIGKILL) when it quits. Servers marked ``"lazy": true`` in
        # .nojae/lsp-settings.json spawn on their first tool call instead, so
        # the app stays snappy and tsserver does not sit in RAM for nothing.
        servers = (
            [] if os.environ.get("BENCHMARKIT_DISABLE_LSP") else all_servers()[0]
        )
        # Two gates must agree for a server to run: its ``enable`` flag in
        # .nojae/lsp-settings.json and the LSP_ENABLED key in .env, so a
        # prepared config file can stay entirely dormant.
        active_ids = {
            server["id"] for server in servers if server["enable"]
        } & set(lsp_enabled_ids())
        lsp = LspManager(servers)
        app.state.lsp = lsp
        await lsp.start_enabled(sorted(active_ids))
        # The terminal registry is a plain dict (no hub thread): sessions are
        # driven by their WebSocket coroutine, on this loop. Nothing spawns at
        # boot — a session is only born on a WS upgrade — but it is created
        # here so every route can find it, and torn down in the same block as
        # the hubs, so a shell that outlived its socket cannot outlive the app.
        terminal = TerminalHub()
        app.state.terminal = terminal
        try:
            yield
        finally:
            await terminal.stop_all()
            await lsp.stop_all()
            await manager.stop_all()

    app = FastAPI(title="benchmarkit", version="0.2.0", lifespan=lifespan)
    app.state.run_state = RunState()
    app.include_router(build_router(app.state.run_state))

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "env": ENV_PATH.is_file(),
            "questions": questions_path().is_file(),
        }

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="static")

    return app


def run() -> None:
    """Console-script entry: serve the API and built frontend on :8000."""
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000)


app = create_app()
