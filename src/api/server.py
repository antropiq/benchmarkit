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
    apply_litellm_debug,
    ENV_PATH,
    load_dotenv_once,
    PROJECT_ROOT,
    path_location,
    questions_path,
    register_configured_models,
    setup_path_location,
)
from src.core.mcphub import McpManager, load_externals
from src.core.lsphub import LspManager, all_servers
from src.core.editorfs import policy_seed
from src.core.termhub import TerminalHub
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


def create_app() -> FastAPI:
    """Build the FastAPI app: API routes first, static SPA fallback after."""
    load_dotenv_once()
    # The LiteLLM debug switch is read from the environment, so it is applied
    # right after the .env has been loaded (and again on every settings save).
    # Keeping it here rather than in the module's import side effects means the
    # tests build an app without touching the litellm loggers of the whole
    # process unless they ask for it.
    apply_litellm_debug()
    # And the same for the cost-table noise: a local quantised model name is
    # unknown to litellm's pricing map, so its cost calculator raises — caught
    # internally, but dumped in full once per call while LITELLM_DEBUG is on.
    # Done here (and on every settings save) so no LLM call is ever preceded by
    # a model litellm cannot price.
    register_configured_models()
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # MCP connector sessions live exactly as long as the web app: connected
        # here on startup, closed on shutdown. The env var is a test/dev switch
        # that keeps the endpoints alive but connects nothing. Every connector
        # is declared in mcp_connectors/externals.json; entries with
        # ``enable: false`` stay stopped until toggled on.
        connectors = (
            [] if os.environ.get("BENCHMARKIT_DISABLE_MCP") else load_externals()[0]
        )
        manager = McpManager(connectors)
        app.state.mcp = manager
        await manager.start_all()
        # Language servers follow the same rule (own env-var switch, own log
        # dir): spawned with the app, shut down (shutdown/exit then
        # SIGTERM/SIGKILL) when it quits. Servers marked ``"lazy": true`` in
        # lsp_servers.json spawn on their first tool call instead, so the app
        # stays snappy and tsserver does not sit in RAM for nothing.
        servers = (
            [] if os.environ.get("BENCHMARKIT_DISABLE_LSP") else all_servers()[0]
        )
        # Two gates must agree for a server to run: its ``enable`` flag in
        # lsp_servers.json and the LSP_ENABLED key in .env, so a prepared
        # config file can stay entirely dormant.
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
