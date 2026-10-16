"""REST + SSE endpoints for the benchmark run."""

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from src.api.events import EventBus, sse_format
from src.config import load_endpoints_config, load_questions, QUESTIONS_PATH
from src.core.bench import run_benchmark
from src.core.job import RunJob
from src.core.models import EndpointsConfig
from src.core.utils import RESULTS_PATH


class RunState:
    """Process-wide state for the single active (or last) benchmark run."""

    def __init__(self) -> None:
        self.status: str = "idle"  # idle | running | finished | failed
        self.run_id: Optional[str] = None
        self.job: Optional[RunJob] = None
        self.bus: Optional[EventBus] = None
        self.error: Optional[str] = None

    def reset(self) -> None:
        self.job = RunJob()
        self.bus = EventBus()
        self.error = None


def build_router(state: RunState) -> APIRouter:
    """Create the API router bound to a run-state instance."""
    router = APIRouter(prefix="/api")

    def _load_config() -> EndpointsConfig:
        config, error = load_endpoints_config()
        if error is not None:
            raise HTTPException(
                status_code=422,
                detail=json.loads(error.json()),
            )
        assert config is not None
        return config

    @router.get("/config")
    def get_config() -> Dict[str, Any]:
        config = _load_config()
        try:
            questions = load_questions()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not load {QUESTIONS_PATH}: {exc}"
            ) from exc
        models = config.model_list()
        sampling = config.to_kwargs()
        sampling_summary = (
            ", ".join(f"{k}={v}" for k, v in sampling.items()) or "defaults"
        )
        return {
            "models": models,
            "samplingSummary": sampling_summary,
            "questionCount": len(questions),
            "totalTests": len(questions) * len(models),
            "verifierConfigured": bool(config.verifierAgent and config.verifierAgent.endpoint),
            "runStatus": state.status,
            "hasResults": RESULTS_PATH.is_file(),
        }

    @router.post("/run", status_code=202)
    async def start_run() -> Dict[str, Any]:
        config = _load_config()
        questions = load_questions()
        models = config.model_list()
        if not models:
            raise HTTPException(status_code=400, detail="No models configured (ENDPOINT/ENDPOINTS)")
        if not questions:
            raise HTTPException(status_code=400, detail="No questions loaded")
        if state.status == "running":
            raise HTTPException(status_code=409, detail="A benchmark run is already in progress")

        state.reset()
        assert state.job is not None and state.bus is not None
        state.run_id = uuid.uuid4().hex
        state.status = "running"
        state.bus.bind_loop(asyncio.get_running_loop())

        sampling = config.to_kwargs()
        state.bus.publish({
            "event": "run_started",
            "runId": state.run_id,
            "models": models,
            "totalTests": len(questions) * len(models),
            "samplingSummary": ", ".join(f"{k}={v}" for k, v in sampling.items())
            or "defaults",
        })
        asyncio.create_task(_execute_run(state, questions, config))
        return {"runId": state.run_id}

    @router.get("/run")
    def get_run() -> Dict[str, Any]:
        events: List[Any] = []
        if state.bus is not None:
            events = [
                {"seq": seq, "event": event} for seq, event in state.bus.snapshot()
            ]
        return {
            "status": state.status,
            "runId": state.run_id,
            "error": state.error,
            "lastSeq": state.bus.last_seq if state.bus else -1,
            "events": events,
        }

    @router.get("/run/events")
    async def run_events(request: Request, since: int = -1) -> StreamingResponse:
        last_event_id = request.headers.get("last-event-id")
        if last_event_id and last_event_id.isdigit():
            since = max(since, int(last_event_id))
        if state.bus is None:
            raise HTTPException(status_code=404, detail="No benchmark run started yet")

        async def event_stream() -> AsyncIterator[str]:
            async for seq, event in state.bus.subscribe(since):
                if await request.is_disconnected():
                    return
                yield sse_format(seq, event)
            # Bus closed — let the EventSource client know the stream is over.
            yield (
                "event: stream_end\n"
                + f"data: {json.dumps({'status': state.status})}\n\n"
            )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.get("/results")
    def get_results() -> FileResponse:
        if not RESULTS_PATH.is_file():
            raise HTTPException(status_code=404, detail="No results.json yet")
        return FileResponse(
            RESULTS_PATH, filename="results.json", media_type="application/json"
        )

    return router


async def _execute_run(
    state: RunState, questions: List[Dict], config: EndpointsConfig
) -> None:
    """Run the benchmark in a worker thread, publishing events to the bus."""
    assert state.job is not None and state.bus is not None
    bus = state.bus
    try:
        await asyncio.to_thread(run_benchmark, questions, config, state.job, bus.publish)
        state.status = "finished"
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        state.error = str(exc)
        state.status = "failed"
        bus.publish({"event": "run_error", "message": str(exc)})
    finally:
        bus.close()
