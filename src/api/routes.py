"""REST + SSE endpoints for the benchmark run."""

import asyncio
import base64
import binascii
import json
import re
import threading
import uuid
from contextlib import suppress
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

import os

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.api.events import EventBus, sse_format
from src.config import (
    MEMORY_COLOR_RE,
    MEMORY_CONTENT_MAX_CHARS,
    MEMORY_DEFAULT_COLOR,
    MEMORY_SLOT_COUNT,
    QUESTION_SET_DEFAULT,
    PATH_LOCATION_KEY,
    agent_instructions_block,
    list_question_sets,
    load_endpoints_config,
    load_memory_registrations,
    load_questions,
    load_settings,
    max_tool_iteration,
    lsp_enabled_ids,
    path_location,
    normalize_env_flag,
    default_path_location,
    save_memory_registration,
    save_settings,
    questions_path,
)
from src.core.mcphub import load_externals, set_externals_enabled
from src.core.lsphub import all_servers, restricted_lsp_roots, set_lsp_enabled
from src.core import editorfs
from src.core.editorfs import policy_seed
from src.core import termhub
from src.core.termhub import policy as term_policy
from src.core.bench import run_benchmark
from src.core import batch as batch_core
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
    litellmDebug: str = ""
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


class PathLocationIn(SettingsFieldsMixin):
    """Working-location change as sent by the settings dialog."""

    path: str = ""


class MemorySlotIn(BaseModel):
    """One memory-bullet slot as sent by the registration dialog.

    ``index`` addresses one of the fixed slots (the file always holds exactly
    ``MEMORY_SLOT_COUNT``, so there is nothing to create or delete); saving
    empty ``content`` clears the slot.
    """

    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=0, le=MEMORY_SLOT_COUNT - 1)
    color: str = MEMORY_DEFAULT_COLOR
    content: str = ""

    @field_validator("color")
    @classmethod
    def _require_hex_color(cls, value: str) -> str:
        if not MEMORY_COLOR_RE.match(value):
            raise ValueError("color must be a 6-digit hex color (e.g. #808080)")
        return value

    @field_validator("content")
    @classmethod
    def _cap_content(cls, value: str) -> str:
        if len(value) > MEMORY_CONTENT_MAX_CHARS:
            raise ValueError(
                f"content exceeds the {MEMORY_CONTENT_MAX_CHARS} character cap"
            )
        return value


class McpIn(BaseModel):
    """MCP connector activation as sent by the settings dialog."""

    model_config = ConfigDict(extra="ignore")

    enabled: List[str] = []
    maxIterations: int = Field(default=25, ge=1, le=500)


class LspIn(BaseModel):
    """Language-server activation as sent by the settings dialog.

    ``maxIterations`` is the same shared MCP+LSP tool-loop cap the MCP
    section edits — posting it from either section keeps one value.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: List[str] = []
    maxIterations: int = Field(default=25, ge=1, le=500)


class FileSaveIn(BaseModel):
    """A save of one whole editor buffer (``PUT /api/fs/file``).

    ``ifUnmodifiedSince`` is the ``mtime`` the ``GET /api/fs/file`` that filled
    this buffer reported — the optimistic-concurrency token the save guard
    compares against the disk. ``None`` means the client holds no expectation
    and skips the guard; the route logs that, because the SPA always sends the
    token and its absence is the shape of a client bug.

    The size cap is deliberately *not* a validator here: over-cap is a refusal
    of the same shape as the read route's (``413`` naming the size), and a
    Pydantic validator would answer ``422`` instead. ``editorfs.write_file``
    owns it.
    """

    model_config = ConfigDict(extra="ignore")

    path: str
    text: str = ""
    ifUnmodifiedSince: Optional[float] = None


class FileRenameIn(BaseModel):
    """A rename, and therefore a move (``POST /api/fs/rename``).

    ``to`` is the whole workspace-relative **target path**, not a new leaf name:
    that is what makes a move the same route as a rename, and it means the
    target's parent chain is a user-supplied path like any other, gated the same
    way. A name-only field would push the joining into the client and leave the
    server unable to check the folder a file lands in.
    """

    model_config = ConfigDict(extra="ignore")

    frm: str = Field(alias="from")
    to: str


class FileCreateIn(BaseModel):
    """A new file or folder (``POST /api/fs/create``).

    ``path`` is the whole workspace-relative path of the thing to create — its
    parent must already exist, so this is one name per request and never a
    ``mkdir -p``. ``dir`` picks a folder over an empty file.

    A name-only field is deliberately *not* offered: the client joins the name
    it collected to the folder it pointed at, so the route sees one path and the
    gate that checks the ancestor chain sees the whole of it.
    """

    model_config = ConfigDict(extra="ignore")

    path: str
    dir: bool = False


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
    # Conversation-scoped MCP filter: ids of the connectors whose tools this
    # conversation may use (intersection with the Settings-activated set).
    # ``None`` means "no filter" — every activated connector stays eligible,
    # which keeps older clients and the existing chat tests unchanged.
    mcpEnabled: Optional[List[str]] = None
    # Same conversation-scoped filter for the LSP language servers (the Chat
    # tab's per-conversation LSP toggles).
    lspEnabled: Optional[List[str]] = None

    # Batch mode only: the task lot's big-picture text (the ``NN == 0`` file),
    # which ``core.chat`` appends to the system message of *every* task turn.
    # Size-capped with the very number the listing enforced when it handed the
    # body over, so the two sides cannot drift. Not otherwise validated as
    # trusted input: this is a single-user app and the same person already owns
    # ``LLM_SYSPROMPT`` through the settings dialog, so a caller-chosen system
    # block grants no new privilege.
    batchContext: Optional[str] = None

    @field_validator("batchContext")
    @classmethod
    def _cap_batch_context(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        # Read through the module so a patch of the cap (the tests) applies.
        cap = batch_core.MAX_BATCH_LOT_BYTES
        if len(value.encode("utf-8")) > cap:
            raise ValueError(
                f"batchContext is too large ({len(value.encode('utf-8'))} "
                f"bytes, max {cap})"
            )
        return value

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


class ChatHandle:
    """The one in-flight chat reply, so another request can pause it.

    ``POST /api/chat/pause`` is a separate HTTP request from the reply's own
    stream, so the pause flag has to live in a process-wide slot — the same
    shape ``RunState`` uses for the run. Mono-user app, one tab: one handle,
    no chat id.
    """

    def __init__(self, pause_event: threading.Event, cancel_event: threading.Event) -> None:
        self.pause_event = pause_event
        self.cancel_event = cancel_event


# Set for the duration of one streaming reply (cleared when its SSE generator
# ends), so a finished or aborted reply can never be paused afterwards.
_CHAT_HANDLE: Optional[ChatHandle] = None


def _arm_chat_handle(cancel_event: threading.Event, pause_event: threading.Event) -> None:
    """Publish the in-flight reply so /chat/pause can reach it."""
    global _CHAT_HANDLE
    _CHAT_HANDLE = ChatHandle(pause_event=pause_event, cancel_event=cancel_event)


def _clear_chat_handle(cancel_event: threading.Event) -> None:
    """Retire the handle, but only if it is still this reply's (a newer
    conversation must never be un-registered by an old stream's ``finally``)."""
    global _CHAT_HANDLE
    if _CHAT_HANDLE is not None and _CHAT_HANDLE.cancel_event is cancel_event:
        _CHAT_HANDLE = None


def _refuse_if_chat_streaming(what: str) -> None:
    """409 while a chat reply is in flight — the one place that rule is written.

    A reply is the only other writer of the workspace, so a mutation landing in
    the middle of one races the model's own tool edits. The check is the same
    process-wide slot ``POST /api/chat/pause`` uses; the client-side rule in
    ``useEditorMode`` is the *display* of this and the server is the enforcement.

    (The hub-restart and path-location routes ask the wider question instead —
    ``_CHAT_TASKS``, which also covers a reply whose handle is not armed yet —
    because they tear down what that reply is using. A file write only needs to
    know a reply is streaming.)
    """
    if _CHAT_HANDLE is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot {what} while a chat reply is streaming",
        )


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
    "litellmDebug": "LITELLM_DEBUG",
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


# Last known context window per (models url, model id): success is cached
# indefinitely (n_ctx only changes with a server restart, which also flips
# the url/model key), failures fall back to the stale value or None.
_CTX_WINDOW_CACHE: Dict[tuple, int] = {}


def _context_window(server_url: str, api_key: str, model: str) -> Optional[int]:
    """Context window (tokens) of the configured model, None when unknown.

    Reads ``meta.n_ctx`` from an OpenAI-compatible ``/models`` entry
    (llama.cpp exposes it); a server that omits it is not an error — the
    chat occupancy badge then just shows the used side without a limit.
    """
    if not server_url.strip():
        return None
    url = _models_url(server_url)
    key = (url, model)
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with httpx.Client(timeout=3.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        entries = [item for item in data or [] if isinstance(item, dict)]
        entry = next((item for item in entries if item.get("id") == model), None)
        if entry is None and entries:
            entry = entries[0]
        meta = entry.get("meta") if entry else None
        n_ctx = meta.get("n_ctx") if isinstance(meta, dict) else None
        if isinstance(n_ctx, int) and n_ctx > 0:
            _CTX_WINDOW_CACHE[key] = n_ctx
            return n_ctx
    except (httpx.HTTPError, ValueError):
        pass
    return _CTX_WINDOW_CACHE.get(key)


def _settings_env_values(settings: SettingsIn) -> Dict[str, str]:
    """Convert a settings payload into the managed .env key/value mapping."""
    payload = settings.model_dump()
    values: Dict[str, str] = {
        key: (payload.get(field) or "").strip()
        for field, key in SETTINGS_ENV_KEYS.items()
    }
    if values.get("REASONING", "").strip().lower() == "none":
        values["REASONING"] = ""
    # Boolean-ish keys: the dialog sends "1"/"" from a checkbox but hand-written
    # clients may send "true"/"on", so both are normalised to the .env form.
    values["SUPPORTS_VISION"] = normalize_env_flag(values.get("SUPPORTS_VISION"))
    values["LITELLM_DEBUG"] = normalize_env_flag(values.get("LITELLM_DEBUG"))
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

    Every connector is declared in ``mcp_connectors/externals.json`` and
    activated by its ``enable`` flag there (single source of truth, no .env
    key involved).
    """
    connectors, _errors = load_externals()
    return [connector["id"] for connector in connectors if connector["enable"]]


def _active_server_ids() -> List[str]:
    """Language-server ids whose tools chat may use.

    Both gates must agree: the id is listed in ``LSP_ENABLED`` (the .env
    switch that lets a whole prepared config file stay dormant) *and* its
    ``enable`` flag is true in ``lsp_servers.json`` (the MCP-externals
    precedent for file-declared entries).
    """
    enabled = set(lsp_enabled_ids())
    servers, _errors = all_servers()
    return [
        server["id"]
        for server in servers
        if server["enable"] and server["id"] in enabled
    ]


def _mcp_tool_kwargs(
    request: Request, enabled: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extra ``stream_chat`` kwargs when MCP connectors are active for chat.

    ``enabled`` is the conversation-scoped connector allowlist from the chat
    body (the Chat tab's per-conversation MCP toggles); it is intersected
    with the Settings-activated ids so Settings stays the authority — unknown
    or stale ids silently drop and a connector deactivated in Settings can
    never leak back into a running conversation. ``None`` means no filter.
    Returns ``{}`` when nothing is active so the no-MCP call shape (and the
    existing chat tests that stub ``stream_chat``) stays untouched.
    """
    manager = getattr(request.app.state, "mcp", None)
    if manager is None:
        return {}
    connector_ids = _active_connector_ids()
    if enabled is not None:
        allowed = set(enabled)
        connector_ids = [cid for cid in connector_ids if cid in allowed]
    try:
        tools, tool_to_server = manager.active_tool_defs(connector_ids)
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
        "max_tool_iterations": max_tool_iteration(),
    }


# Chat-side budget for one LSP tool call: covers a lazy server's cold start
# (initialize can take INIT_TIMEOUT seconds while it indexes the project)
# plus the request itself and the diagnostics wait.
LSP_TOOL_TIMEOUT = 90.0


def _lsp_tool_kwargs(
    request: Request, enabled: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extra ``stream_chat`` kwargs when language servers are active for chat.

    Same conversation-scoped filter as MCP: ``enabled`` is intersected with
    the Settings-activated ids, so a server disabled in Settings can never
    leak back into a running conversation. ``{}`` when nothing is active.
    """
    manager = getattr(request.app.state, "lsp", None)
    if manager is None:
        return {}
    server_ids = _active_server_ids()
    if enabled is not None:
        allowed = set(enabled)
        server_ids = [server_id for server_id in server_ids if server_id in allowed]
    try:
        tools, tool_to_server = manager.active_tool_defs(server_ids)
    except Exception:  # noqa: BLE001 — never block chat because of the hub
        return {}
    if not tools:
        return {}

    def tool_executor(name: str, arguments: Dict[str, Any]) -> str:
        return manager.call_tool_sync(
            tool_to_server[name], name, arguments, timeout=LSP_TOOL_TIMEOUT
        )

    return {
        "tools": tools,
        "tool_executor": tool_executor,
        "tool_servers": tool_to_server,
    }


def _chat_tool_kwargs(
    request: Request,
    mcp_enabled: Optional[List[str]] = None,
    lsp_enabled: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The single tool loop of ``stream_chat``, fed by both hubs.

    The merge lives here (not in ``core/chat.py``): ``stream_chat`` keeps
    taking one ``tools`` list, one executor and one iteration cap, and the
    ``tool_servers`` map is what makes every tool call attributable to its
    hub in the Chat UI. MCP tools win a name clash (LSP tool names are
    namespaced, so in practice they cannot collide).
    """
    mcp_kwargs = _mcp_tool_kwargs(request, mcp_enabled)
    lsp_kwargs = _lsp_tool_kwargs(request, lsp_enabled)
    if not mcp_kwargs and not lsp_kwargs:
        return {}
    if not lsp_kwargs:
        return mcp_kwargs
    if not mcp_kwargs:
        # _mcp_tool_kwargs owns the shared iteration cap; add it when only
        # LSP contributes tools.
        return {**lsp_kwargs, "max_tool_iterations": max_tool_iteration()}

    tools = list(mcp_kwargs["tools"])
    tool_to_server = dict(mcp_kwargs["tool_servers"])
    for definition in lsp_kwargs["tools"]:
        name = definition["function"]["name"]
        if name in tool_to_server:
            continue
        tool_to_server[name] = lsp_kwargs["tool_servers"][name]
        tools.append(definition)
    mcp_executor = mcp_kwargs["tool_executor"]
    lsp_executor = lsp_kwargs["tool_executor"]
    lsp_tool_names = set(lsp_kwargs["tool_servers"])

    def tool_executor(name: str, arguments: Dict[str, Any]) -> str:
        if name in lsp_tool_names:
            return lsp_executor(name, arguments)
        return mcp_executor(name, arguments)

    return {
        "tools": tools,
        "tool_executor": tool_executor,
        "tool_servers": tool_to_server,
        "max_tool_iterations": max_tool_iteration(),
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
            # Chat occupancy gauge cap: the configured model's context window
            # (meta.n_ctx), None when the server does not expose it.
            "contextWindow": _context_window(
                config.serverUrl, config.apiKey or "none", models[0] if models else ""
            ),
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

    @router.get("/memory")
    def get_memory() -> Dict[str, Any]:
        """Return the memory-bullet slots (self-heals the file to 9 defaults)."""
        return {"registration": load_memory_registrations()}

    @router.post("/memory")
    def post_memory(body: MemorySlotIn) -> Dict[str, Any]:
        """Persist one memory-bullet slot and return the full registration.

        Chat-adjacent configuration, so (unlike settings/question sets) it is
        not gated while a benchmark run is in progress. Invalid index, color
        or oversized content are rejected with 422 by the request model.
        """
        slots = save_memory_registration(body.index, body.color, body.content)
        return {"registration": slots}

    def _mcp_payload(
        request: Request, extra_errors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Connector list (live ``externals.json`` scan) merged with manager status.

        Connectors are active on their file ``enable`` flag — the single
        source of truth. Secret material (``headers``/``env``) is stripped.
        """
        manager = getattr(request.app.state, "mcp", None)
        statuses = {s["id"]: s for s in (manager.status() if manager else [])}
        connectors, errors = load_externals()
        if extra_errors:
            errors = errors + list(extra_errors)
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
                    "active": connector["enable"],
                    "status": status.get("status", "stopped"),
                    "error": status.get("error"),
                }
            )
        return {
            "installed": bool(manager and manager.installed),
            "connectors": payload_connectors,
            "maxIterations": max_tool_iteration(),
            "externalsError": "\n".join(errors) if errors else None,
        }

    @router.get("/mcp")
    def get_mcp(request: Request) -> Dict[str, Any]:
        """List the declared MCP connectors with live status."""
        return _mcp_payload(request)

    @router.post("/mcp")
    def post_mcp(body: McpIn, request: Request) -> Dict[str, Any]:
        """Activate MCP connectors for the Chat tab and set the iteration cap.

        Toggles write the entry ``enable`` flag back to
        ``mcp_connectors/externals.json`` (the only activation source) and
        start/stop the connector right away; only ``MAX_TOOL_ITERATION``
        persists to ``.env``.
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot change MCP settings while a benchmark run is in progress",
            )
        requested = [item.strip() for item in body.enabled if item.strip()]
        connectors, _errors = load_externals()
        connector_ids = {connector["id"] for connector in connectors}
        unknown = [item for item in requested if item not in connector_ids]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown MCP connectors: {', '.join(unknown)}",
            )
        enabled = set(requested)
        merged = dict(load_settings())
        merged["MAX_TOOL_ITERATION"] = str(body.maxIterations)
        save_settings(merged)
        write_errors = (
            set_externals_enabled({cid: cid in enabled for cid in connector_ids})
            if connector_ids
            else []
        )
        manager = getattr(request.app.state, "mcp", None)
        if manager is not None and hasattr(manager, "start_connector"):
            for connector in connectors:
                connector_id = connector["id"]
                current = manager.connector_status(connector_id)
                if connector_id in enabled and current not in ("running", "starting"):
                    manager.start_connector(connector_id)
                elif connector_id not in enabled and current in ("running", "starting"):
                    manager.stop_connector(connector_id)
        return _mcp_payload(request, extra_errors=write_errors)

    async def _restart_mcp_hub(request: Request) -> Dict[str, Any]:
        """Tear the MCP hub down and rebuild it from a fresh file scan.

        Shared by ``POST /mcp/restart`` and the working-location endpoint: a
        changed ``${BENCHMARKIT_PATH_LOCATION}`` only takes effect once every
        connector is relaunched with the new argv.
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot restart the MCP hub while a benchmark run is in progress",
            )
        if _CHAT_TASKS:
            raise HTTPException(
                status_code=409,
                detail="Cannot restart the MCP hub while a chat reply is streaming",
            )
        manager = getattr(request.app.state, "mcp", None)
        if manager is None or not hasattr(manager, "restart"):
            raise HTTPException(status_code=503, detail="MCP hub is not available")
        if os.environ.get("BENCHMARKIT_DISABLE_MCP"):
            # Same dev/test switch the app lifespan honours: endpoints stay
            # usable but nothing is connected.
            connectors = []
        else:
            connectors, _errors = load_externals()
        await manager.restart(connectors)
        # The payload re-scans the file, so it re-derives the same validation
        # errors — they must not be passed as extra_errors or they appear twice.
        return _mcp_payload(request)

    @router.post("/mcp/restart")
    async def post_mcp_restart(request: Request) -> Dict[str, Any]:
        """Restart the MCP hub: re-read ``externals.json`` and reconnect.

        Everything is torn down (client sessions and, with them, every local
        stdio child process) and rebuilt from a fresh scan of the declaration
        file, so an edit made by hand — a new connector, a changed ``command``,
        another ``cwd`` — takes effect without restarting the web app.
        Validation problems do not abort the restart: the connectors that did
        parse come back up and the errors are returned in ``externalsError``.

        Awaited, so the response carries the real post-restart status of every
        connector (a failing one burns its whole retry window — the button
        stays busy while that happens).
        """
        return await _restart_mcp_hub(request)

    def _lsp_payload(
        request: Request, extra_errors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Language-server list (file scan + validation errors) + live status.

        A server is active when its ``enable`` flag and ``LSP_ENABLED`` agree.
        Secret material (``env``) is stripped.
        """
        manager = getattr(request.app.state, "lsp", None)
        statuses = {s["id"]: s for s in (manager.status() if manager else [])}
        servers, errors = all_servers()
        if extra_errors:
            errors = errors + list(extra_errors)
        enabled_ids = set(lsp_enabled_ids())
        payload_servers = []
        for server in servers:
            status = statuses.get(server["id"], {})
            payload_servers.append(
                {
                    key: value
                    for key, value in server.items()
                    if key not in ("env", "initializationOptions")
                }
                | {
                    "active": server["enable"] and server["id"] in enabled_ids,
                    "status": status.get("status", "stopped"),
                    "error": status.get("error"),
                    "toolCount": status.get("toolCount", 0),
                }
            )
        return {
            "installed": bool(manager and manager.installed),
            "servers": payload_servers,
            "maxIterations": max_tool_iteration(),
            "serversError": "\n".join(errors) if errors else None,
        }

    @router.get("/lsp")
    def get_lsp(request: Request) -> Dict[str, Any]:
        """List declared language servers with live status."""
        return _lsp_payload(request)

    @router.post("/lsp")
    def post_lsp(body: LspIn, request: Request) -> Dict[str, Any]:
        """Activate language servers for the Chat tab.

        Activation writes both places: the ids persist to ``LSP_ENABLED`` in
        ``.env`` and the ``enable`` flag is written back to
        ``lsp_servers.json`` (both must agree for a server to run, so a
        prepared config file can stay entirely dormant). Running servers are
        hot-started/stopped right away; ``maxIterations`` updates the shared
        MCP+LSP tool-loop cap.
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot change LSP settings while a benchmark run is in progress",
            )
        requested = [item.strip() for item in body.enabled if item.strip()]
        servers, _errors = all_servers()
        known_ids = {server["id"] for server in servers}
        unknown = [item for item in requested if item not in known_ids]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown LSP servers: {', '.join(unknown)}",
            )
        enabled = set(requested)
        merged = dict(load_settings())
        merged["LSP_ENABLED"] = ",".join(dict.fromkeys(requested))
        merged["MAX_TOOL_ITERATION"] = str(body.maxIterations)
        save_settings(merged)
        write_errors = (
            set_lsp_enabled({sid: sid in enabled for sid in known_ids})
            if known_ids
            else []
        )
        manager = getattr(request.app.state, "lsp", None)
        if manager is not None and hasattr(manager, "start_server"):
            for server in servers:
                server_id = server["id"]
                current = manager.server_status(server_id)
                # Only ``running`` means "nothing to do". The hub decides
                # whether a start is worth starting (a live session or an
                # attempt already in flight is a no-op), so a server stuck in
                # ``starting`` can be re-activated from the panel instead of
                # being un-clickable until a full hub restart.
                if server_id in enabled and current != "running":
                    manager.start_server(server_id)
                elif server_id not in enabled and current in ("running", "starting"):
                    manager.stop_server(server_id)
        return _lsp_payload(request, extra_errors=write_errors)

    async def _restart_lsp_hub(request: Request) -> Dict[str, Any]:
        """Tear the LSP hub down and rebuild it from a fresh file scan.

        Shared by ``POST /lsp/restart`` and the working-location endpoint: a
        changed ``${BENCHMARKIT_PATH_LOCATION}`` only takes effect once every
        language server is respawned on the new ``workspaceRoot``.
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot restart the LSP hub while a benchmark run is in progress",
            )
        if _CHAT_TASKS:
            raise HTTPException(
                status_code=409,
                detail="Cannot restart the LSP hub while a chat reply is streaming",
            )
        manager = getattr(request.app.state, "lsp", None)
        if manager is None or not hasattr(manager, "restart"):
            raise HTTPException(status_code=503, detail="LSP hub is not available")
        if os.environ.get("BENCHMARKIT_DISABLE_LSP"):
            # Same dev/test switch the app lifespan honours: endpoints stay
            # usable but nothing is spawned.
            servers, active_ids = [], []
        else:
            servers, _errors = all_servers()
            # Same two-gate rule as the startup lifespan, kept in one place.
            active_ids = _active_server_ids()
        await manager.restart(servers, active_ids)
        # The payload re-scans the file, so it re-derives the same validation
        # errors — they must not be passed as extra_errors or they appear twice.
        return _lsp_payload(request)

    @router.post("/lsp/restart")
    async def post_lsp_restart(request: Request) -> Dict[str, Any]:
        """Restart the LSP hub: re-read ``lsp_servers.json`` and respawn.

        Every language server is shut down (``shutdown``/``exit``, then
        SIGTERM/SIGKILL) and respawned from a fresh scan of the declaration
        file, so a hand-edited entry — a new server, a changed ``command``,
        another ``workspaceRoot`` — takes effect without restarting the web
        app. The two activation gates (the entry ``enable`` flag and
        ``LSP_ENABLED``) are re-applied exactly as at startup, so a dormant
        file stays dormant. Invalid entries never spawn: the valid ones come
        back up and the problems are reported in ``serversError``.

        Awaited, so the response carries real post-restart status (a cold
        language-server start is multi-second — the button stays busy).
        """
        return await _restart_lsp_hub(request)

    def _path_location_payload() -> Dict[str, Any]:
        return {
            "key": PATH_LOCATION_KEY,
            "path": path_location(),
            "default": default_path_location(),
        }

    def _policy_block() -> Dict[str, Any]:
        """What the workspace's policy file looks like, **read-only**.

        pol-05 seeds a default ``.nojae/read-write-policy.json`` at the two
        moments a folder becomes the working location (boot, and the switch
        below), so a GET must not write one: the settings dialog is opened far
        more often than a workspace is switched, and a read that leaves a folder
        in someone's project is the mistake the seeding rules forbid. ``state``
        is ``existing`` / ``absent`` / ``failed`` here, and the dialog turns
        ``absent`` into "this project has no policy file yet".
        """
        return policy_seed.policy_file_state(path_location())

    @router.get("/path-location")
    def get_path_location() -> Dict[str, Any]:
        """Effective working location of the MCP/LSP hubs.

        The value ``${BENCHMARKIT_PATH_LOCATION}`` expands to in
        ``mcp_connectors/externals.json`` and ``lsp_connectors/lsp_servers.json``
        — read live from the environment, so it reflects a change made through
        ``POST /path-location`` rather than a stale startup snapshot.
        """
        return _path_location_payload() | {"policy": _policy_block()}

    @router.post("/path-location")
    async def post_path_location(
        body: PathLocationIn, request: Request
    ) -> Dict[str, Any]:
        """Point both hubs at another directory and reload their tools.

        Sets ``BENCHMARKIT_PATH_LOCATION`` in this process' environment, then
        restarts the MCP and LSP hubs so both declaration files are re-read and
        every connector/language server is relaunched against the new value —
        the same effect as the refresh buttons, in one call.

        The value is deliberately *not* persisted: it is process state, and the
        next boot always re-seeds it to the install directory (see
        ``config.setup_path_location``), so a machine move can never be broken
        by a remembered path. Rejected with 422 when the directory is missing
        or sits on the LSP workspace-root deny-list (a language server indexing
        a secret tree would hand its contents to the model); in that case the
        environment is left untouched and nothing is restarted.
        """
        if state.status == "running":
            raise HTTPException(
                status_code=409,
                detail="Cannot change the working location while a benchmark run is in progress",
            )
        # Every precondition of the two restarts is checked *before* the
        # environment is touched: an env value the hubs never picked up is the
        # worst possible outcome (the panel would disagree with the tools).
        if _CHAT_TASKS:
            raise HTTPException(
                status_code=409,
                detail="Cannot change the working location while a chat reply is streaming",
            )
        for attr, label in (("mcp", "MCP"), ("lsp", "LSP")):
            hub = getattr(request.app.state, attr, None)
            if hub is None or not hasattr(hub, "restart"):
                raise HTTPException(
                    status_code=503, detail=f"{label} hub is not available"
                )
        raw = (body.path or "").strip()
        if not raw:
            raise HTTPException(
                status_code=422, detail="Working location must not be empty"
            )
        if not os.path.isabs(raw):
            raise HTTPException(
                status_code=422, detail="Working location must be an absolute path"
            )
        # Expanduser + realpath: a '~' or a symlinked checkout must land on the
        # same canonical prefix the hubs compare against, and what the panel
        # shows must be what the declaration files expand to.
        expanded = os.path.realpath(os.path.expanduser(raw))
        if not os.path.isdir(expanded):
            raise HTTPException(
                status_code=422, detail=f"Working location does not exist: {raw}"
            )
        for denied in restricted_lsp_roots():
            entry = os.path.realpath(denied)
            if expanded == entry or expanded.startswith(entry + os.sep):
                raise HTTPException(
                    status_code=422,
                    detail=f"Working location is on the deny-list: {denied}",
                )
        previous = os.environ.get(PATH_LOCATION_KEY)
        os.environ[PATH_LOCATION_KEY] = expanded
        try:
            mcp = await _restart_mcp_hub(request)
            lsp = await _restart_lsp_hub(request)
        except BaseException:
            # Guards above mean a refusal can no longer happen here, so this
            # only fires on an unexpected hub failure: put the environment back
            # so the panel and the next file scan agree. (If the MCP restart
            # already landed, its live sessions keep the new argv until the
            # next restart — a restart button or a reboot re-aligns them.)
            if previous is None:
                os.environ.pop(PATH_LOCATION_KEY, None)
            else:
                os.environ[PATH_LOCATION_KEY] = previous
            raise
        # Seed the new root's default policy file here — the second of pol-05's
        # two entry points (the other is boot, ``api/server.py``). It goes
        # *after* the two hub restarts on purpose: a seeding hiccup must not be
        # mistaken for a hub failure, because the except above would then roll
        # the environment back for a reason that has nothing to do with the
        # switch. And it goes before the response is built, so the Editor's
        # tree re-roots onto a workspace whose rules already exist.
        #
        # A failure is a note in the payload, never an error: an unseeded
        # workspace opens fine, it just has no level-B rules to filter it.
        return (
            _path_location_payload()
            | {
                "mcp": mcp,
                "lsp": lsp,
                "policy": policy_seed.ensure_policy_file(expanded),
            }
        )

    @router.get("/fs/tree")
    def get_fs_tree(path: str = "") -> Dict[str, Any]:
        """One directory level of the active workspace, for the Editor tab.

        ``path`` is relative to ``BENCHMARKIT_PATH_LOCATION`` ("" is its root)
        — the same working location both hubs point at, so the tree browses
        exactly what the tools are working on. One level per call keeps the
        first paint at one ``listdir`` however big the checkout is.

        All the rules live in ``core/editorfs`` (the three locks on the path,
        the entry cap, and one policy verdict per path — built in for the app's
        own files, and read from the workspace's ``.nojae/read-write-policy.json``
        for everyone else's); this route only turns a
        :class:`~src.core.editorfs.policy.WorkspacePathError` into its HTTP
        status. This route reads: every write route under ``/api/fs`` is gated by
        the same verdict, belongs to the browser alone, and no hub or tool may
        ever gain one.
        """
        try:
            return editorfs.list_dir(path)
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.get("/fs/file")
    def get_fs_file(path: str) -> Dict[str, Any]:
        """The whole text of one workspace-relative file, for the Editor tab.

        The same gate as the tree — ``editorfs`` resolves the path inside the
        working location and asks the policy verdict — so a name the listing
        pruned answers this read as absent (404), never as a refusal that would
        confirm it exists. One verdict, every surface that touches disk. Past
        ``editorfs.constants.MAX_EDITOR_BYTES``
        the answer is 413 naming the size; a binary file answers ``binary: true``
        with no ``text`` key rather than an error, so the pane can label it.

        ``mtime`` in the payload is the token the save guard compares: the
        client sends it back as ``ifUnmodifiedSince`` and a disk mtime that has
        moved since answers 409 instead of overwriting someone else's change.
        It is also what the freshness probe (``/api/fs/stat``) compares for the
        open buffers. This route reads; the write routes are browser-only.
        """
        try:
            return editorfs.read_file(path)
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    # -- the guarded write surface (browser-only) ------------------------------

    @router.put("/fs/file")
    def put_fs_file(body: FileSaveIn) -> Dict[str, Any]:
        """One whole editor buffer, back to disk (``PUT /api/fs/file``).

        The rules all live in :func:`src.core.editorfs.files.write_file` —
        the write gate on the leaf (a path the policy hides or marks read-only
        is refused 422, naming the level and the rule), the ``mtime`` guard
        (409 whose ``detail`` is an object carrying both mtimes), the size cap
        on the way in *and* on the file on disk (413: a buffer we never fully
        read is never written back), a pre-save backup, and an atomic temp-file
        + ``os.replace`` that keeps the file's mode and its exact bytes (CRLF
        included, no normalisation).

        Refused while a chat reply streams: that reply may be editing these
        very files through its tools, so a save landing in the middle of one
        would race the model. The payload answers the new ``size``/``mtime``
        so the caller re-baselines on the server's truth.

        This is a **browser** surface. No MCP connector, LSP tool or chat code
        path may call it — the tools stay read-only by design, and the guard
        suite in ``tests/test_editorfs_write.py`` asserts exactly that.
        """
        _refuse_if_chat_streaming("save")
        try:
            return editorfs.write_file(
                body.path, body.text, if_unmodified_since=body.ifUnmodifiedSince
            )
        except editorfs.WorkspaceSaveConflict as exc:
            # The object, not the sentence: the client offers reload-or-
            # overwrite, and that choice needs both numbers, not prose.
            raise HTTPException(status_code=exc.status_code, detail=exc.detail_object)
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.post("/fs/create")
    def post_fs_create(body: FileCreateIn) -> Dict[str, Any]:
        """One new file or folder inside the working location.

        The rules live in :func:`src.core.editorfs.mutate.create_path` — the
        write gate over the parent chain *and* the leaf name (422 naming the
        policy level and the rule that matched, never the 404 a caller would
        read as "the name is free"), the parent must already be a directory
        (409 naming it: a typo becomes a refusal, never a stray tree — there is
        no ``mkdir -p`` here), and never over an existing name, dangling symlink
        included (409 whose ``detail`` is ``{"reason": "exists", …}``, closed
        against a race with ``mkdir``/``O_EXCL`` rather than checked).

        Refused while a chat reply streams, like every other write. The payload
        answers ``path``/``dir``/``size``/``mtime`` — the shape a save answers,
        so the client re-baselines the new empty buffer on the server's truth.

        Browser-only, like ``PUT /api/fs/file``: no connector or LSP tool may
        reach a route under ``/fs`` that mutates disk.
        """
        _refuse_if_chat_streaming("create a file or folder")
        try:
            return editorfs.create_path(body.path, is_dir=body.dir)
        except editorfs.WorkspacePathExists as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail_object)
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.delete("/fs/file")
    def delete_fs_file(path: str) -> Dict[str, Any]:
        """One file or one whole folder, moved to the trash (``DELETE /api/fs/file``).

        The rules live in :func:`src.core.editorfs.mutate.trash_path` — the
        write gate over the chain (422 naming the policy rule), the working
        location itself refused outright (400), an absent path answered **404**
        rather than "already deleted, fine", a top-level symlink refused (400,
        its target may be outside the workspace), and the **all-or-nothing
        subtree check**: one hidden or read-only descendant refuses the whole
        delete, naming that descendant, and nothing at all moves. What is
        deleted is never ``unlink``ed — it lands in
        ``logs/trash/<batch>/<the relative path>`` and is swept after
        ``TRASH_RETENTION_DAYS``, which is why the payload names the batch.

        Refused while a chat reply streams, like every other write. The answer
        carries ``entries`` — the descendant count from the server's own walk —
        so the confirmation the UI already showed matches what happened.

        Browser-only, like the other write routes: no connector or LSP tool may
        reach a route under ``/fs`` that mutates disk.
        """
        _refuse_if_chat_streaming("delete a file or folder")
        try:
            return editorfs.trash_path(path)
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.post("/fs/rename")
    def post_fs_rename(body: FileRenameIn) -> Dict[str, Any]:
        """One path to another path — a rename, which is also a move.

        The rules live in :func:`src.core.editorfs.mutate.rename_path`: **both
        ends** go through the write gate (``from`` whole, and ``to`` plus every
        one of its ancestors — the rule that refuses an innocent file moving
        into ``.git/…`` or ``mcp_connectors/…``), an absent source is 404, the
        target's parent must already exist (409 naming it, no implicit
        ``mkdir``), a folder cannot move into itself (409), and an occupied
        target is 409 rather than ``os.rename``'s silent replacement. A
        cross-device move (``EXDEV``) is refused rather than copied.

        Refused while a chat reply streams, like every other write. Nothing is
        backed up here, deliberately: the bytes did not change and the trash
        plus git already describe the move. The payload answers
        ``from``/``to``/``dir``/``size``/``mtime`` so the caller can re-point
        its open buffers and re-baseline on the server's truth.

        Browser-only, like the other write routes: no connector or LSP tool may
        reach a route under ``/fs`` that mutates disk.
        """
        _refuse_if_chat_streaming("rename a file or folder")
        try:
            return editorfs.rename_path(body.frm, body.to)
        except editorfs.WorkspacePathExists as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail_object)
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.get("/fs/stat")
    def get_fs_stat(path: List[str] = Query(default_factory=list)) -> Dict[str, Any]:
        """``mtime``/``size``/``exists`` for a handful of exact paths.

        The freshness probe: the client names the files it has open, so nothing
        has to poll a tree or run a file watcher. Every path goes through the
        same gate as the tree.
        """
        try:
            return {"paths": editorfs.stat_paths(path)}
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.get("/fs/status")
    def get_fs_status() -> Dict[str, Any]:
        """Which files of the working location git considers unstaged.

        One call for the whole tree (not one per row), so the navigator can mark
        the modified and untracked files of every branch it has on screen. The
        answer is workspace-relative, and a working location that is not a git
        repository answers ``repo: false`` with an empty list — the silent,
        non-error case the UI is expected to ignore.

        Read-only like the rest of ``/api/fs``: the query is ``git status``, and
        ``GIT_OPTIONAL_LOCKS=0`` keeps even it from touching the index on disk.
        """
        try:
            return editorfs.git_status()
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.get("/fs/diff")
    def get_fs_diff(path: str) -> Dict[str, Any]:
        """The two sides of one file: ``original`` as of ``HEAD``, ``modified``
        as it sits on disk.

        Returned as two texts rather than a patch so the editor renders the diff
        itself. The path goes through the same gate as the tree and the file
        read, and the disk side is read by that same reader — so the size cap,
        the binary answer and the policy refusal are identical here. A file with
        no committed side (added, untracked) gets an empty ``original``; the UI
        does not offer the item for one.
        """
        try:
            return editorfs.file_diff(path)
        except editorfs.WorkspacePathError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    # -- terminal: one PTY per WebSocket ---------------------------------------

    @router.get("/terminal")
    def get_terminal(request: Request) -> Dict[str, Any]:
        """The live terminal sessions (pid, cwd, started, bytes) — for the
        settings panel. Mirrors ``GET /api/mcp``: an orphan must be nameable, so
        whatever the WebSocket registry holds is reported here. Read-only; the
        shells themselves are driven by the WS, never by this route."""
        hub = getattr(request.app.state, "terminal", None)
        return {"sessions": hub.list_sessions() if hub is not None else []}

    @router.websocket("/terminal")
    async def terminal_ws(websocket: WebSocket) -> None:
        """One shell on one PTY, bridged to the browser's xterm pane.

        Guards, in order, each refusing *before* any process is spawned:
        the Origin check (the app has no CORS/CSRF layer, so this is the one
        cheap defence against a random page reaching a shell), the platform
        check (``pty`` is POSIX; the app is self-hosted on Linux), and the
        concurrent-session cap. A real shell can change anything the app's user
        can — which is why it is *never* an MCP or LSP tool, only this route,
        used by the SPA. The tools stay read-only; a shell is a human surface.

        ``cwd`` is always ``path_location()``; the server takes no path from the
        client. Frames are JSON text (``input``/``resize``/``kill`` in,
        ``open``/``output``/``exit``/``error`` out); the session owns the
        framing and teardown, this route only adapts the socket to it.
        """
        hub = getattr(websocket.app.state, "terminal", None)
        if hub is None:  # pragma: no cover - lifespan always sets it
            await websocket.close(termhub.CLOSE_CODE_UNSUPPORTED, "Terminal unavailable")
            return

        headers = {k.lower(): v for k, v in websocket.headers.raw}
        allowed, reason = term_policy.origin_allowed(headers)
        if not allowed:
            await websocket.close(
                termhub.CLOSE_CODE_POLICY, f"Origin refused: {reason}"
            )
            return

        if os.name != "posix":
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "The terminal is POSIX-only; this app is self-hosted on Linux",
                }
            )
            await websocket.close(termhub.CLOSE_CODE_UNSUPPORTED, "Terminal unsupported")
            return

        if hub.is_full():
            await websocket.close(
                termhub.CLOSE_CODE_CAPACITY,
                f"Too many terminal sessions (cap {hub.capacity()})",
            )
            return

        # Publish a frame straight to the socket. Raises when the client is
        # gone, which the session's reader treats as "stop pumping".
        async def emit(event: Dict[str, Any]) -> None:
            await websocket.send_json(event)

        async def receive_frame():
            try:
                text = await websocket.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                return None
            try:
                frame = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # Malformed frame: ignored by the session, not fatal.
                return {"type": ""}
            return frame if isinstance(frame, dict) else {"type": ""}

        try:
            session = hub.create(emit)
        except termhub.SessionCapacity as exc:
            await websocket.close(termhub.CLOSE_CODE_CAPACITY, str(exc))
            return

        await websocket.accept()
        try:
            # One session per connection: no attach, no reuse.
            session.spawn(cols=termhub.DEFAULT_COLS, rows=termhub.DEFAULT_ROWS)
        except (termhub.TerminalUnsupported, OSError) as exc:
            hub.drop(session.id)
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(termhub.CLOSE_CODE_UNSUPPORTED, "Could not spawn shell")
            return

        try:
            await session.run(receive_frame)
        finally:
            hub.drop(session.id)
            with suppress(Exception):
                await websocket.close()

    @router.get("/lsp/log")
    def get_lsp_log(request: Request, server: str = "") -> Dict[str, Any]:
        """Tail of one language server's log file (for the settings panel)."""
        manager = getattr(request.app.state, "lsp", None)
        if manager is None or not server:
            raise HTTPException(status_code=404, detail="No LSP log available")
        log = manager.log_tail(server) if hasattr(manager, "log_tail") else ""
        return {"server": server, "log": log}

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

    @router.get("/batch/tasks")
    def get_batch_tasks(prefix: str = "", dir: str = "") -> Dict[str, Any]:
        """List the numbered task lot of ``prefix`` in the workspace's ``dir``.

        The whole backend surface batch mode has: names, sizes, the first
        heading and the big-picture body — never a task body (the model opens
        its task file with the connectors, which is why there is no file-read
        route here). All the rules (prefix shape, the three folder locks,
        numbering, caps) live in ``core/batch.py``; this route only turns a
        :class:`core.batch.BatchListingError` into its HTTP status.
        """
        try:
            return batch_core.task_lot(prefix, dir)
        except batch_core.BatchListingError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    @router.post("/chat")
    async def post_chat(body: ChatIn, request: Request) -> StreamingResponse:
        """Stream one chat reply from the benchmark-configured LLM as SSE.

        Sync litellm runs in a worker thread feeding an ``asyncio.Queue``
        (project convention: sync litellm in a thread + ``emit``); the SSE
        generator polls the queue and disconnect, closing the upstream
        stream via ``cancel_event`` when the client goes away.

        While the reply streams, its events are also *pausable*: the handle
        registered here lets ``POST /api/chat/pause`` arm ``pause_event``, which
        ``core.chat.stream_chat`` checks at its next message boundary.
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
        pause_event = threading.Event()
        _arm_chat_handle(cancel_event, pause_event)
        tool_kwargs = _chat_tool_kwargs(request, body.mcpEnabled, body.lspEnabled)

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
                    cancel_event=cancel_event,
                    pause_event=pause_event,
                    api_key=config.apiKey or "none",
                    sysprompt=config.sysprompt,
                    # Keywords from here on: the flag pair is optional on
                    # ``stream_chat``, so a stub may omit it (tests do).
                    # Project instructions ride on the system message: when the
                    # working location carries AGENTS.md, its text is appended
                    # there so every conversation starts already knowing the
                    # project's conventions (see ``config.load_agent_instructions``).
                    agent_instructions=agent_instructions_block(),
                    # The lot's big picture (batch mode only) rides alongside
                    # the project instructions, never merged into ``sysprompt``:
                    # that separation is what keeps the TOOL_NOTICE word check
                    # judging only the prompt the user configured. It is passed
                    # only when set, so a plain chat request keeps exactly the
                    # call shape it always had.
                    **({"batch_instructions": body.batchContext}
                       if body.batchContext else {}),
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
                    if item.get("event") in ("done", "error", "pause_ack"):
                        return
            finally:
                # Client gone (or terminal event): stop the worker thread,
                # which closes the upstream litellm stream.
                cancel_event.set()
                _clear_chat_handle(cancel_event)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.post("/chat/pause", status_code=200)
    async def post_chat_pause() -> Dict[str, Any]:
        """Ask the one in-flight chat reply to stop at its next boundary.

        Non-destructive twin of ``/run/stop``: the worker closes the upstream
        stream, reports its exact usage and terminates the reply with
        ``pause_ack``, so the client keeps the partial turn in its history.
        Idempotent — a second call while the first is pending says so.
        """
        handle = _CHAT_HANDLE
        if handle is None:
            raise HTTPException(status_code=409, detail="No chat reply is streaming")
        if handle.pause_event.is_set():
            return {"paused": "pending"}
        handle.pause_event.set()
        return {"paused": "requested"}

    @router.post("/run", status_code=202)
    async def start_run() -> Dict[str, Any]:
        # Exclusion in the direction ``POST /chat`` already enforces the other
        # way (409 while a run is in progress): a benchmark must not start
        # under a streaming chat reply, or two generations fight over the
        # single _CHAT_HANDLE and the endpoint's own sampling configuration.
        # Enforced here rather than only by a disabled button, because batch
        # mode keeps a chat turn streaming for minutes at a time.
        if _CHAT_TASKS:
            raise HTTPException(
                status_code=409,
                detail="Cannot start a run while a chat reply is streaming",
            )
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

        # Local alias: pyright does not keep `state.bus` narrowed inside the
        # closure below (it could be reassigned between the check and the call).
        bus = state.bus
        async def event_stream() -> AsyncIterator[str]:
            async for seq, event in bus.subscribe(since):
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

    def emit(event: Dict[str, Any]) -> None:
        bus.publish(event)

    try:
        await asyncio.to_thread(
            run_benchmark, questions, config, state.job, emit, state.cancel_event
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
