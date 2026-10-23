"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import RunState, build_router
from src.config import ENV_PATH, load_dotenv_once, PROJECT_ROOT, questions_path
from src.core.mcp_hub import McpManager, all_connectors
from src.core.lsphub import LspManager, all_servers
from src.config import lsp_enabled_ids

WEB_DIST = PROJECT_ROOT / "web" / "dist"


def create_app() -> FastAPI:
    """Build the FastAPI app: API routes first, static SPA fallback after."""
    load_dotenv_once()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # MCP connector processes live exactly as long as the web app:
        # spawned here on startup, terminated on shutdown. The env var is a
        # test/dev switch that keeps the endpoints alive but spawns nothing.
        # Externals declared in mcp_connectors/externals.json join the set;
        # those with ``enable: false`` stay stopped until toggled on.
        connectors = (
            [] if os.environ.get("BENCHMARKIT_DISABLE_MCP") else all_connectors()[0]
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
        try:
            yield
        finally:
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
