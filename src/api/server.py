"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import RunState, build_router
from src.config import ENV_PATH, load_dotenv_once, PROJECT_ROOT, questions_path
from src.core.mcp_hub import McpManager, discover_connectors

WEB_DIST = PROJECT_ROOT / "web" / "dist"


def create_app() -> FastAPI:
    """Build the FastAPI app: API routes first, static SPA fallback after."""
    load_dotenv_once()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # MCP connector processes live exactly as long as the web app:
        # spawned here on startup, terminated on shutdown. The env var is a
        # test/dev switch that keeps the endpoints alive but spawns nothing.
        connectors = (
            [] if os.environ.get("BENCHMARKIT_DISABLE_MCP") else discover_connectors()
        )
        manager = McpManager(connectors)
        app.state.mcp = manager
        await manager.start_all()
        try:
            yield
        finally:
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
