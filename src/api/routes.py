"""REST + SSE endpoints for the benchmark run."""

import asyncio
import json
import threading
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import os

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.api.events import EventBus, sse_format
from src.config import (
    QUESTION_SET_DEFAULT,
    list_question_sets,
    load_endpoints_config,
    load_questions,
    load_settings,
    save_settings,
    questions_path,
)
from src.core.bench import run_benchmark
from src.core.job import RunJob
from src.core.llm import RunCancelled
from src.core.models import EndpointsConfig
from src.core.utils import RESULTS_PATH


class SettingsFieldsMixin(BaseModel):
    """Tolerant request fields: JSON scalars are accepted and stringified.

    ``null`` means "unset" and unknown keys are ignored, so hand-written
    clients may post numbers (``"temp": 0.2``) or omit optional fields.
    """

    model_config = ConfigDict(extra="ignore")

    @field_validator("*", mode="before")
    @classmethod
    def _stringify_scalars(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (dict, list, BaseModel)) or isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return str(value)
        raise ValueError("expected a string or number")


class VerifierSettingsIn(SettingsFieldsMixin):
    """Verifier agent settings as sent by the settings dialog."""

    serverUrl: str = ""
    endpoint: str = ""
    sysprompt: str = ""


class SettingsIn(SettingsFieldsMixin):
    """Benchmark settings as sent by the settings dialog (raw strings)."""

    serverUrl: str = ""
    endpoint: str = ""
    temp: str = ""
    topP: str = ""
    topK: str = ""
    minP: str = ""
    repeatPenalty: str = ""
    presencePenalty: str = ""
    reasoning: str = ""
    questionSet: str = ""
    verifier: VerifierSettingsIn = VerifierSettingsIn()

    @field_validator("verifier", mode="before")
    @classmethod
    def _empty_verifier(cls, value: Any) -> Any:
        # Accept "verifier": null / "" / {} as "all verifier fields unset".
        return value or VerifierSettingsIn()


class QuestionSetIn(SettingsFieldsMixin):
    """Question set selection as sent by the settings dialog."""

    questionSet: str = ""


# Dialog field -> .env key, used for both directions of the settings API.
SETTINGS_ENV_KEYS: Dict[str, str] = {
    "serverUrl": "SERVER_URL",
    "endpoint": "ENDPOINT",
    "temp": "TEMP",
    "topP": "TOP_P",
    "topK": "TOP_K",
    "minP": "MIN_P",
    "repeatPenalty": "REPEAT_PENALTY",
    "presencePenalty": "PRESENCE_PENALTY",
    "reasoning": "REASONING",
}
VERIFIER_ENV_KEYS: Dict[str, str] = {
    "serverUrl": "VERIFIER_AGENT_SERVERURL",
    "endpoint": "VERIFIER_AGENT_ENDPOINT",
    "sysprompt": "VERIFIER_AGENT_SYSPROMPT",
}


def _settings_json(values: Dict[str, str]) -> Dict[str, Any]:
    """Shape raw env values into the settings dialog payload."""
    data: Dict[str, Any] = {
        field: values.get(key, "") for field, key in SETTINGS_ENV_KEYS.items()
    }
    data["questionSet"] = values.get("QUESTION_SET", "")
    data["verifier"] = {
        field: values.get(key, "") for field, key in VERIFIER_ENV_KEYS.items()
    }
    return data


def _models_url(server_url: str) -> str:
    """Build the OpenAI-compatible models listing URL for a server URL.

    ``SERVER_URL`` usually already ends with ``/v1`` (it is used as the
    litellm ``api_base``), so only ``/models`` is appended in that case.
    """
    url = server_url.strip().rstrip("/")
    if url.endswith("/v1"):
        return f"{url}/models"
    return f"{url}/v1/models"


def _model_ids(payload: Any) -> List[str]:
    """Extract model ids from an OpenAI-compatible /models payload."""
    data = payload.get("data") if isinstance(payload, dict) else None
    ids = [
        item["id"]
        for item in data or []
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    return sorted(set(ids))


def _settings_env_values(settings: SettingsIn) -> Dict[str, str]:
    """Convert a settings payload into the managed .env key/value mapping."""
    payload = settings.model_dump()
    values: Dict[str, str] = {
        key: (payload.get(field) or "").strip()
        for field, key in SETTINGS_ENV_KEYS.items()
    }
    if values.get("REASONING", "").strip().lower() == "none":
        values["REASONING"] = ""
    verifier = payload.get("verifier") or {}
    for field, key in VERIFIER_ENV_KEYS.items():
        values[key] = (verifier.get(field) or "").strip()
    question_set = (payload.get("questionSet") or "").strip()
    if question_set:
        values["QUESTION_SET"] = question_set
    return values


class RunState:
    """Process-wide state for the single active (or last) benchmark run."""

    def __init__(self) -> None:
        self.status: str = "idle"  # idle | running | finished | failed | cancelled
        self.run_id: Optional[str] = None
        self.job: Optional[RunJob] = None
        self.bus: Optional[EventBus] = None
        self.error: Optional[str] = None
        self.cancel_event: Optional[threading.Event] = None

    def reset(self) -> None:
        self.job = RunJob()
        self.bus = EventBus()
        self.error = None
        self.cancel_event = threading.Event()


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
                status_code=500, detail=f"Could not load {questions_path()}: {exc}"
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

    @router.get("/settings")
    def get_settings() -> Dict[str, Any]:
        return _settings_json(load_settings())

    @router.post("/settings")
    def post_settings(settings: SettingsIn) -> Dict[str, Any]:
        # The .env is process-wide configuration; changing it mid-run would
        # silently mix two configurations into one result set.
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot change settings while a benchmark run is in progress",
            )
        env_values = _settings_env_values(settings)
        merged = {
            key: value
            for key, value in os.environ.items()
            if key not in env_values or env_values[key]
        }
        merged.update({k: v for k, v in env_values.items() if v})
        try:
            EndpointsConfig.from_env(merged)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
        # Note: keys that were set as real process environment variables before
        # startup keep precedence in the live process until restart; the file
        # is still updated. The file itself is regenerated from a fixed
        # template, so custom comments in .env are not preserved.
        save_settings(env_values)
        return _settings_json(load_settings())

    @router.get("/question-sets")
    def get_question_sets() -> Dict[str, Any]:
        """List available question sets and the currently selected one."""
        return {
            "questionSets": list_question_sets(),
            "current": os.environ.get("QUESTION_SET", QUESTION_SET_DEFAULT),
        }

    @router.post("/question-sets")
    def post_question_set(body: QuestionSetIn) -> Dict[str, Any]:
        """Select the active question set source.

        Persists the choice to ``.env`` via ``save_settings`` so the next run
        uses the selected file. Rejected with 409 while a run is in progress
        and 422 for an unknown question set id.
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot change question set while a benchmark run is in progress",
            )
        selected = (body.questionSet or "").strip()
        available = [s["id"] for s in list_question_sets()]
        if not selected:
            raise HTTPException(
                status_code=422, detail="No question set selected"
            )
        if selected not in available:
            raise HTTPException(
                status_code=422, detail=f"Unknown question set: {selected}"
            )
        # Merge with current settings so unrelated keys are preserved.
        current = load_settings()
        merged = dict(current)
        merged["QUESTION_SET"] = selected
        save_settings(merged)
        return {
            "questionSets": list_question_sets(),
            "current": os.environ.get("QUESTION_SET", QUESTION_SET_DEFAULT),
        }

    @router.get("/models")
    async def get_models(serverUrl: str = "") -> Dict[str, Any]:
        """List models exposed by an OpenAI-compatible server URL."""
        url = serverUrl.strip()
        if not url:
            raise HTTPException(status_code=422, detail="No server URL provided")
        if not url.lower().startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422, detail="Server URL must start with http:// or https://"
            )
        models_url = _models_url(url)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(models_url)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not fetch models from {models_url}: {exc}",
            ) from exc
        return {"models": _model_ids(payload)}

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

    @router.post("/run/stop", status_code=202)
    async def stop_run() -> Dict[str, Any]:
        if state.status != "running" or state.cancel_event is None:
            raise HTTPException(status_code=409, detail="No benchmark run in progress")
        state.cancel_event.set()
        return {"runId": state.run_id, "cancelling": True}

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
        await asyncio.to_thread(
            run_benchmark, questions, config, state.job, bus.publish, state.cancel_event
        )
        state.status = "finished"
    except RunCancelled:
        state.status = "cancelled"
        bus.publish({
            "event": "run_finished",
            "cancelled": True,
            "totalRunningTime": state.job.total_running_time,
            "totalTests": len(questions) * len(config.model_list()),
        })
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        state.error = str(exc)
        state.status = "failed"
        bus.publish({"event": "run_error", "message": str(exc)})
    finally:
        bus.close()
