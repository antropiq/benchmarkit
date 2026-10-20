"""REST + SSE endpoints for the benchmark run."""

import asyncio
import base64
import binascii
import json
import re
import threading
import uuid
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

import os

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.api.events import EventBus, sse_format
from src.config import (
    QUESTION_SET_DEFAULT,
    list_question_sets,
    load_endpoints_config,
    load_questions,
    load_settings,
    mcp_enabled_ids,
    max_mcp_iteration,
    save_settings,
    questions_path,
)
from src.core.mcp_hub import all_connectors, load_externals, set_externals_enabled
from src.core.bench import run_benchmark
from src.core.chat import stream_chat
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
    apiKey: str = ""
    endpoint: str = ""
    sysprompt: str = ""


class SettingsIn(SettingsFieldsMixin):
    """Benchmark settings as sent by the settings dialog (raw strings)."""

    serverUrl: str = ""
    apiKey: str = ""
    endpoint: str = ""
    sysprompt: str = ""
    temp: str = ""
    topP: str = ""
    topK: str = ""
    minP: str = ""
    repeatPenalty: str = ""
    presencePenalty: str = ""
    reasoning: str = ""
    supportsVision: str = ""
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


class McpIn(BaseModel):
    """MCP connector activation as sent by the settings dialog."""

    model_config = ConfigDict(extra="ignore")

    enabled: List[str] = []
    maxIterations: int = Field(default=25, ge=1, le=500)


# Base64-decoded image size limit for multimodal chat messages (the client
# applies the same limit before reading a file).
MAX_CHAT_IMAGE_BYTES = 10 * 1024 * 1024
_CHAT_IMAGE_DATA_URL = re.compile(
    r"^data:image/(png|jpeg|webp|gif);base64,([A-Za-z0-9+/]+={0,2})$"
)


class ChatImageUrl(BaseModel):
    """Image reference inside a multimodal chat content part."""

    model_config = ConfigDict(extra="ignore")

    url: str

    @field_validator("url")
    @classmethod
    def _require_data_url(cls, value: str) -> str:
        match = _CHAT_IMAGE_DATA_URL.match(value)
        if match is None:
            raise ValueError(
                "image_url.url must be a base64 data URL "
                "(data:image/png|jpeg|webp|gif;base64,...)"
            )
        try:
            decoded = base64.b64decode(match.group(2), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image data is not valid base64: {exc}") from exc
        if len(decoded) > MAX_CHAT_IMAGE_BYTES:
            raise ValueError(
                f"image is too large ({len(decoded)} bytes, max "
                f"{MAX_CHAT_IMAGE_BYTES})"
            )
        return value


class ChatContentPart(BaseModel):
    """One OpenAI-style multimodal content part (text or image_url)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ChatImageUrl] = None

    @model_validator(mode="after")
    def _require_payload(self) -> "ChatContentPart":
        if self.type == "text" and self.text is None:
            raise ValueError("text content part requires text")
        if self.type == "image_url" and self.image_url is None:
            raise ValueError("image_url content part requires image_url")
        return self


class ChatMessage(BaseModel):
    """One chat turn as sent by the chat screen.

    ``content`` is a plain string or a list of multimodal parts (user
    messages only) so images travel as base64 data URLs in the standard
    OpenAI format, passed through litellm to the OpenAI-compatible server.
    """

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: Union[str, List[ChatContentPart]]

    @model_validator(mode="after")
    def _require_payload(self) -> "ChatMessage":
        if not isinstance(self.content, str):
            if self.role == "assistant":
                raise ValueError("assistant messages must use plain text content")
            if not self.content:
                raise ValueError("message content must not be empty")
            has_image = any(part.type == "image_url" for part in self.content)
            has_text = any(
                part.type == "text" and (part.text or "").strip()
                for part in self.content
            )
            if not has_image and not has_text:
                raise ValueError("message content must not be empty")
        return self

    def payload(self) -> Dict[str, Any]:
        """Shape the message for litellm (parts dumped to plain dicts)."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [part.model_dump(exclude_none=True) for part in self.content],
        }


class ChatIn(BaseModel):
    """Chat request body: the full conversation history."""

    model_config = ConfigDict(extra="ignore")

    messages: List[ChatMessage]

    @field_validator("messages")
    @classmethod
    def _require_turns(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        for message in value:
            if isinstance(message.content, str) and not message.content.strip():
                raise ValueError("message content must not be empty")
        return value


# Strong references to in-flight chat pump tasks (the event loop only keeps
# weak references to bare tasks).
_CHAT_TASKS: set = set()


# Dialog field -> .env key, used for both directions of the settings API.
SETTINGS_ENV_KEYS: Dict[str, str] = {
    "serverUrl": "SERVER_URL",
    "apiKey": "SERVER_URL_API_KEY",
    "endpoint": "ENDPOINT",
    "sysprompt": "LLM_SYSPROMPT",
    "temp": "TEMP",
    "topP": "TOP_P",
    "topK": "TOP_K",
    "minP": "MIN_P",
    "repeatPenalty": "REPEAT_PENALTY",
    "presencePenalty": "PRESENCE_PENALTY",
    "reasoning": "REASONING",
    "supportsVision": "SUPPORTS_VISION",
}
VERIFIER_ENV_KEYS: Dict[str, str] = {
    "serverUrl": "VERIFIER_AGENT_SERVERURL",
    "apiKey": "VERIFIER_AGENT_SERVERURL_API_KEY",
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
    vision = values.get("SUPPORTS_VISION", "").strip().lower()
    if vision in {"1", "true", "yes", "on"}:
        values["SUPPORTS_VISION"] = "1"
    elif vision in {"", "0", "false", "no", "off"}:
        values["SUPPORTS_VISION"] = ""
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


def _active_connector_ids() -> List[str]:
    """Connector ids whose tools chat may use.

    Built-ins come from ``MCP_ENABLED`` (.env); externals are activated by
    their ``enable`` flag in ``externals.json`` (single source of truth, so
    they never enter ``MCP_ENABLED``).
    """
    ids = list(mcp_enabled_ids())
    connectors, _errors = load_externals()
    for connector in connectors:
        if connector["enable"] and connector["id"] not in ids:
            ids.append(connector["id"])
    return ids


def _mcp_tool_kwargs(request: Request) -> Dict[str, Any]:
    """Extra ``stream_chat`` kwargs when MCP connectors are active for chat.

    Returns ``{}`` when nothing is active so the no-MCP call shape (and the
    existing chat tests that stub ``stream_chat``) stays untouched.
    """
    manager = getattr(request.app.state, "mcp", None)
    if manager is None:
        return {}
    try:
        tools, tool_to_server = manager.active_tool_defs(_active_connector_ids())
    except Exception:  # noqa: BLE001 — never block chat because of the hub
        return {}
    if not tools:
        return {}

    def tool_executor(name: str, arguments: Dict[str, Any]) -> str:
        return manager.call_tool_sync(tool_to_server[name], name, arguments)

    return {
        "tools": tools,
        "tool_executor": tool_executor,
        "tool_servers": tool_to_server,
        "max_tool_iterations": max_mcp_iteration(),
    }


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
            "serverUrl": config.serverUrl,
            "supportsVision": config.supports_vision,
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
        # Settings keys not covered by this dialog (question set, MCP) are
        # merged from the current values so saving here never drops them.
        env_values = {**load_settings(), **env_values}
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

    def _mcp_payload(
        request: Request, extra_errors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Connector list (live folder scan + externals.json) merged with manager status.

        External connectors are active on their file ``enable`` flag; built-ins
        on ``MCP_ENABLED``. Secret material (``headers``/``env``) is stripped.
        """
        manager = getattr(request.app.state, "mcp", None)
        statuses = {s["id"]: s for s in (manager.status() if manager else [])}
        connectors, errors = all_connectors()
        if extra_errors:
            errors = errors + list(extra_errors)
        active_builtins = set(mcp_enabled_ids())
        payload_connectors = []
        for connector in connectors:
            status = statuses.get(connector["id"], {})
            payload_connectors.append(
                {
                    key: value
                    for key, value in connector.items()
                    if key not in ("headers", "env")
                }
                | {
                    "active": connector["enable"]
                    if connector["external"]
                    else connector["id"] in active_builtins,
                    "status": status.get("status", "stopped"),
                    "error": status.get("error"),
                }
            )
        return {
            "installed": bool(manager and manager.installed),
            "connectors": payload_connectors,
            "maxIterations": max_mcp_iteration(),
            "externalsError": "\n".join(errors) if errors else None,
        }

    @router.get("/mcp")
    def get_mcp(request: Request) -> Dict[str, Any]:
        """List built-in and external MCP connectors with live status."""
        return _mcp_payload(request)

    @router.post("/mcp")
    def post_mcp(body: McpIn, request: Request) -> Dict[str, Any]:
        """Activate MCP connectors for the Chat tab and set the iteration cap.

        Built-in activations persist to ``MCP_ENABLED`` / ``MAX_MCP_ITERATION``
        in ``.env``; external toggles write the ``enable`` flag back to
        ``mcp_connectors/externals.json`` and start/stop the connector right
        away (its process otherwise lives with the app).
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot change MCP settings while a benchmark run is in progress",
            )
        requested = [item.strip() for item in body.enabled if item.strip()]
        connectors, _errors = all_connectors()
        builtin_ids = {c["id"] for c in connectors if not c["external"]}
        external_ids = {c["id"] for c in connectors if c["external"]}
        unknown = [
            item for item in requested if item not in builtin_ids and item not in external_ids
        ]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown MCP connectors: {', '.join(unknown)}",
            )
        enabled = set(requested)
        merged = dict(load_settings())
        merged["MCP_ENABLED"] = ",".join(
            dict.fromkeys(item for item in requested if item in builtin_ids)
        )
        merged["MAX_MCP_ITERATION"] = str(body.maxIterations)
        save_settings(merged)
        write_errors = (
            set_externals_enabled({cid: cid in enabled for cid in external_ids})
            if external_ids
            else []
        )
        manager = getattr(request.app.state, "mcp", None)
        if manager is not None and hasattr(manager, "start_connector"):
            for connector in connectors:
                if not connector["external"]:
                    continue
                connector_id = connector["id"]
                current = manager.connector_status(connector_id)
                if connector_id in enabled and current not in ("running", "starting"):
                    manager.start_connector(connector_id)
                elif connector_id not in enabled and current in ("running", "starting"):
                    manager.stop_connector(connector_id)
        return _mcp_payload(request, extra_errors=write_errors)

    @router.get("/models")
    async def get_models(serverUrl: str = "", apiKey: str = "") -> Dict[str, Any]:
        """List models exposed by an OpenAI-compatible server URL."""
        url = serverUrl.strip()
        if not url:
            raise HTTPException(status_code=422, detail="No server URL provided")
        if not url.lower().startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422, detail="Server URL must start with http:// or https://"
            )
        models_url = _models_url(url)
        headers = {}
        if apiKey.strip():
            headers["Authorization"] = f"Bearer {apiKey.strip()}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(models_url, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not fetch models from {models_url}: {exc}",
            ) from exc
        return {"models": _model_ids(payload)}

    @router.post("/chat")
    async def post_chat(body: ChatIn, request: Request) -> StreamingResponse:
        """Stream one chat reply from the benchmark-configured LLM as SSE.

        Sync litellm runs in a worker thread feeding an ``asyncio.Queue``
        (project convention: sync litellm in a thread + ``emit``); the SSE
        generator polls the queue and disconnect, closing the upstream
        stream via ``cancel_event`` when the client goes away.
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot chat while a benchmark run is in progress",
            )
        config = _load_config()
        models = config.model_list()
        if not models:
            raise HTTPException(status_code=400, detail="No models configured (ENDPOINT)")

        messages = [m.payload() for m in body.messages]
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel_event = threading.Event()
        tool_kwargs = _mcp_tool_kwargs(request)

        def emit(event: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def pump() -> None:
            try:
                await asyncio.to_thread(
                    stream_chat,
                    messages,
                    models[0],
                    config.serverUrl,
                    emit,
                    config.to_kwargs(),
                    cancel_event,
                    config.apiKey or "none",
                    config.sysprompt,
                    **tool_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
                emit({"event": "error", "message": str(exc)})
            finally:
                emit(None)  # sentinel: worker finished

        task = asyncio.create_task(pump())
        _CHAT_TASKS.add(task)
        task.add_done_callback(_CHAT_TASKS.discard)

        async def event_stream() -> AsyncIterator[str]:
            seq = 0
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    if item is None:
                        return
                    yield sse_format(seq, item)
                    seq += 1
                    if item.get("event") in ("done", "error"):
                        return
            finally:
                # Client gone (or terminal event): stop the worker thread,
                # which closes the upstream litellm stream.
                cancel_event.set()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.post("/run", status_code=202)
    async def start_run() -> Dict[str, Any]:
        config = _load_config()
        questions = load_questions()
        models = config.model_list()
        if not models:
            raise HTTPException(status_code=400, detail="No models configured (ENDPOINT)")
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
