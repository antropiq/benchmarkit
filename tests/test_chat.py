"""Tests for the streaming chat core and the /api/chat SSE endpoint."""

import base64
import json
import threading
import time
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.core import chat as chat_core
from src.core.models import DEFAULT_LLM_SYSPROMPT, EndpointsConfig


def content_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def reasoning_chunk(text):
    return {"choices": [{"delta": {"reasoning_content": text}}]}


def usage_chunk(prompt=None, completion=None, total=None):
    usage = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if total is not None:
        usage["total_tokens"] = total
    return {"choices": [], "usage": usage}


class TestStreamChat:
    def _run(self, monkeypatch, chunks, cancel_event=None):
        monkeypatch.setattr("src.core.chat.completion", lambda **kwargs: iter(chunks))
        events = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi?"}],
            "model-a",
            "http://server:8080/v1",
            events.append,
            cancel_event=cancel_event,
        )
        return events

    def test_emits_reasoning_once_then_tokens_then_done(self, monkeypatch):
        events = self._run(
            monkeypatch,
            [
                reasoning_chunk("thinking"),
                reasoning_chunk(" more"),
                content_chunk("Hel"),
                content_chunk("lo"),
                usage_chunk(prompt=5, completion=7, total=12),
            ],
        )
        names = [e["event"] for e in events]
        # The one-shot `reasoning` flag opens the phase, then every thinking
        # delta streams as its own `reasoning_delta` frame.
        assert names == [
            "reasoning",
            "reasoning_delta", "reasoning_delta",
            "token", "token",
            "done",
        ]
        assert [e["text"] for e in events if e["event"] == "reasoning_delta"] == [
            "thinking", " more",
        ]
        text = "".join(e["text"] for e in events if e["event"] == "token")
        assert text == "Hello"
        # Reasoning is never ANSWER text: it stays out of `token` frames and
        # out of the assistant turn fed back to the model.
        assert all("thinking" not in e["text"] for e in events if e["event"] == "token")
        assert events[0] == {"event": "reasoning"}  # flag carries no text
        done = events[-1]
        assert done["tokens"] == 7
        assert done["tokensEstimated"] is False
        # Consumed = prompt + completion: exactly the server's total here.
        assert done["consumed"] == 12
        assert done["consumedEstimated"] is False
        # Single round: the occupancy figure is that round's total, exact.
        assert done["contextTokens"] == 12
        assert done["contextEstimated"] is False

    def test_consumed_estimated_from_prompt_when_server_reports_no_usage(
        self, monkeypatch,
    ):
        # No usage chunk at all: completion keeps its char estimate, and the
        # consumed figure adds the char estimate of the prompt the request
        # carries (system prompt + user text) — flagged as estimated.
        from src.core.models import DEFAULT_LLM_SYSPROMPT
        from src.core.utils import estimate_completion_tokens

        events = self._run(monkeypatch, [content_chunk("Hello there")])
        done = events[-1]
        prompt_est = estimate_completion_tokens(DEFAULT_LLM_SYSPROMPT + "hi?")
        assert done["tokens"] == estimate_completion_tokens("Hello there")
        assert done["consumed"] == prompt_est + done["tokens"]
        assert done["consumedEstimated"] is True
        # Without usage the occupancy figure is the same char-based round
        # estimate, flagged.
        assert done["contextTokens"] == done["consumed"]
        assert done["contextEstimated"] is True

    def test_consumed_prompt_exact_when_only_completion_is_estimated(
        self, monkeypatch,
    ):
        # prompt without completion or total: the exact prompt is kept, only
        # the generation share is char-estimated (flagged).
        events = self._run(
            monkeypatch,
            [content_chunk("abcd"), usage_chunk(prompt=40)],
        )
        done = events[-1]
        assert done["tokens"] == 1  # 4 chars / 4
        assert done["tokensEstimated"] is True
        assert done["consumed"] == 41
        assert done["consumedEstimated"] is True

    def test_reasoning_deltas_open_and_extend_decode_window(self, monkeypatch):
        # t0, reasoning, reasoning, content — the usage-only chunk is not
        # timed. The window must start at the first reasoning delta, like in
        # the benchmark loop, so tok/s does not pretend thinking was free.
        clock = [0.0, 1.0, 2.0, 6.0]  # t0 + one arrival per generated chunk
        monkeypatch.setattr(
            "src.core.chat.time.monotonic", lambda: clock.pop(0)
        )
        events = self._run(
            monkeypatch,
            [
                reasoning_chunk("thinking"),
                reasoning_chunk(" more"),
                content_chunk("Hi"),
                usage_chunk(prompt=5, completion=9, total=14),
            ],
        )
        done = events[-1]
        assert clock == []
        assert done["ttft"] == 1.0  # first generated delta was reasoning
        assert done["took"] == 6.0
        assert done["tokens"] == 9  # completion usage counts reasoning tokens
        # Single round: the decode window equals took - ttft, thinking included.
        assert done["decode"] == 5.0

    def test_prepends_system_prompt(self, monkeypatch):
        from src.core.models import DEFAULT_LLM_SYSPROMPT

        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s/v1", lambda e: None
        )
        assert captured["messages"][0] == {
            "role": "system",
            "content": DEFAULT_LLM_SYSPROMPT,
        }

    def test_configured_sysprompt_is_prepended(self, monkeypatch):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s/v1", lambda e: None,
            sysprompt="Answer concisely.",
        )
        assert captured["messages"][0] == {
            "role": "system",
            "content": "Answer concisely.",
        }

    def test_existing_system_prompt_kept(self, monkeypatch):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}],
            "m",
            "http://s/v1",
            lambda e: None,
        )
        assert captured["messages"][0]["content"] == "be nice"

    def test_call_failure_emits_error(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("down")

        monkeypatch.setattr("src.core.chat.completion", boom)
        events = []
        chat_core.stream_chat([{"role": "user", "content": "hi"}], "m", "http://s", events.append)
        assert [e["event"] for e in events] == ["error"]
        assert "down" in events[0]["message"]

    def test_cancel_stops_without_done(self, monkeypatch):
        cancel = threading.Event()
        cancel.set()
        closed = threading.Event()

        class Gen:
            def __iter__(self):
                yield content_chunk("x")

            def close(self):
                closed.set()

        monkeypatch.setattr("src.core.chat.completion", lambda **kwargs: Gen())
        events = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s", events.append,
            cancel_event=cancel,
        )
        assert events == []
        assert closed.is_set()

    def test_cancel_closes_underlying_completion_stream(self, monkeypatch):
        # litellm >= 1.105 CustomStreamWrapper has no sync close(): the raw
        # provider stream on .completion_stream must be the one that closes,
        # otherwise the server keeps generating into a dead connection.
        cancel = threading.Event()
        cancel.set()
        closed = threading.Event()

        class InnerStream:
            def close(self):
                closed.set()

        class Wrapper:
            completion_stream = InnerStream()

            def __iter__(self):
                yield content_chunk("x")

        monkeypatch.setattr("src.core.chat.completion", lambda **kwargs: Wrapper())
        events = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s", events.append,
            cancel_event=cancel,
        )
        assert events == []
        assert closed.is_set()

    def test_cancel_close_errors_are_swallowed(self, monkeypatch):
        cancel = threading.Event()
        cancel.set()

        class Gen:
            def __iter__(self):
                yield content_chunk("x")

            def close(self):
                raise RuntimeError("socket already gone")

        monkeypatch.setattr("src.core.chat.completion", lambda **kwargs: Gen())
        events = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s", events.append,
            cancel_event=cancel,
        )
        assert events == []

    def test_estimates_tokens_without_usage(self, monkeypatch):
        events = self._run(monkeypatch, [content_chunk("hello world!!!")])
        done = events[-1]
        assert done["event"] == "done"
        assert done["tokens"] == 4
        assert done["tokensEstimated"] is True

    def test_char_estimate_covers_reasoning_text(self, monkeypatch):
        # No usage at all: the ~4 chars/token fallback must count the
        # reasoning deltas too (they were generated tokens), not just answer
        # text: 8 reasoning chars + "hi" = 10 chars -> 3 tokens.
        events = self._run(
            monkeypatch,
            [reasoning_chunk("thinking"), content_chunk("hi")],
        )
        done = events[-1]
        assert done["tokens"] == 3
        assert done["tokensEstimated"] is True

    def test_total_minus_prompt_derived_across_rounds_without_completion(
        self, monkeypatch,
    ):
        # A server that omits completion_tokens but reports total/prompt per
        # request: without tools the single usage chunk still yields total -
        # prompt, not flagged as estimated.
        events = self._run(
            monkeypatch,
            [content_chunk("hi"), usage_chunk(prompt=5, total=12)],
        )
        done = events[-1]
        assert done["tokens"] == 7
        assert done["tokensEstimated"] is False

    def test_multimodal_content_passed_through_untouched(self, monkeypatch):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        parts = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
        ]
        chat_core.stream_chat(
            [{"role": "user", "content": parts}], "m", "http://s/v1", lambda e: None
        )
        # Only the system message is prepended; the multimodal parts are
        # forwarded to litellm as-is.
        assert captured["messages"][1] == {"role": "user", "content": parts}


def _config(endpoint: Optional[str] = "model-a") -> EndpointsConfig:
    return EndpointsConfig(
        serverUrl="http://localhost:8080/v1",
        endpoint=endpoint,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    app = create_app()
    monkeypatch.setattr(routes, "load_endpoints_config", lambda: (_config(), None))
    results = tmp_path / "results.json"
    monkeypatch.setattr(routes, "RESULTS_PATH", results)
    monkeypatch.setattr("src.core.utils.RESULTS_PATH", results)
    with TestClient(app) as test_client:
        yield test_client


def _events(body: str):
    parsed = []
    for frame in body.strip().split("\n\n"):
        name = None
        data = None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if name is not None:
            parsed.append((name, data))
    return parsed


# Real 1x1 transparent PNG.
_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _oversized_data_url() -> str:
    from src.api.routes import MAX_CHAT_IMAGE_BYTES

    payload = base64.b64encode(b"x" * (MAX_CHAT_IMAGE_BYTES + 1)).decode()
    return f"data:image/png;base64,{payload}"


class TestChatEndpoint:
    def test_happy_path_streams_frames(self, client, monkeypatch):
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 api_key="none", sysprompt=None):
            emit({"event": "token", "text": "Hel"})
            emit({"event": "token", "text": "lo"})
            emit({"event": "done", "took": 1.0, "ttft": 0.1, "tokens": 3,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        names = [name for name, _ in _events(response.text)]
        assert names == ["token", "token", "done"]

    def test_uses_first_configured_model_and_server(self, client, monkeypatch):
        seen = {}

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 api_key="none", sysprompt=None):
            seen["model"] = model
            seen["server_url"] = server_url
            seen["messages"] = messages
            seen["api_key"] = api_key
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        client.post(
            "/api/chat",
            json={"messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]},
        )
        assert seen["model"] == "model-a"
        assert seen["server_url"] == "http://localhost:8080/v1"
        assert seen["api_key"] == "none"
        assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user"]

    def test_multimodal_message_forwarded_as_parts(self, client, monkeypatch):
        seen = {}

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 api_key="none", sysprompt=None):
            seen["messages"] = messages
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat",
            json={"messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
                ],
            }]},
        )
        assert response.status_code == 200
        content = seen["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1] == {"type": "image_url",
                              "image_url": {"url": _PNG_DATA_URL}}

    def test_image_only_message_accepted(self, client, monkeypatch):
        seen = {}

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 api_key="none", sysprompt=None):
            seen["messages"] = messages
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat",
            json={"messages": [{
                "role": "user",
                "content": [{"type": "image_url",
                             "image_url": {"url": _PNG_DATA_URL}}],
            }]},
        )
        assert response.status_code == 200
        assert len(seen["messages"][0]["content"]) == 1

    @pytest.mark.parametrize(
        "parts",
        [
            [],  # empty parts list
            [{"type": "text", "text": "   "}],  # blank text only
            [{"type": "text"}],  # text part without text
            [{"type": "image_url"}],  # image part without image_url
            # Not a data URL
            [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}],
            # Disallowed media type
            [{"type": "image_url", "image_url":
              {"url": "data:image/svg+xml;base64,AAA="}}],
            # Not base64
            [{"type": "image_url", "image_url":
              {"url": "data:image/png;base64,!!!not base64!!!"}}],
            # Over the 10 MB decoded limit
            [{"type": "image_url", "image_url": {"url": _oversized_data_url()}}],
        ],
    )
    def test_invalid_multimodal_bodies_return_422(self, client, parts):
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": parts}]}
        )
        assert response.status_code == 422

    def test_assistant_multimodal_content_returns_422(self, client):
        response = client.post(
            "/api/chat",
            json={"messages": [
                {"role": "assistant",
                 "content": [{"type": "text", "text": "hi"}]},
            ]},
        )
        assert response.status_code == 422

    def test_forwards_configured_api_key(self, monkeypatch, tmp_path):
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 api_key="none", sysprompt=None):
            emit({"api_key_seen": api_key, "event": "done", "took": 0.0,
                  "ttft": 0.0, "tokens": 0, "tokensEstimated": False})

        app = create_app()
        monkeypatch.setattr(
            routes,
            "load_endpoints_config",
            lambda: (
                EndpointsConfig(
                    serverUrl="http://localhost:8080/v1",
                    apiKey="sk-42",
                    endpoint="model-a",
                ),
                None,
            ),
        )
        monkeypatch.setattr(routes, "stream_chat", fake)
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
            )
        assert '"api_key_seen": "sk-42"' in response.text

    def test_error_event_terminates_stream(self, client, monkeypatch):
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 api_key="none", sysprompt=None):
            emit({"event": "error", "message": "boom"})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        frames = _events(response.text)
        assert [name for name, _ in frames] == ["error"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"messages": []},
            {"messages": [{"role": "user", "content": "   "}]},
            {"messages": [{"role": "wizard", "content": "abracadabra"}]},
            {"messages": [{"role": "user"}]},
        ],
    )
    def test_invalid_bodies_return_422(self, client, payload):
        assert client.post("/api/chat", json=payload).status_code == 422

    def test_no_model_configured_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(routes, "load_endpoints_config", lambda: (_config(None), None))
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 400

    def test_running_benchmark_returns_409(self, client, monkeypatch):
        gate = threading.Event()

        def blocked(questions, config, job, emit, cancel_event=None):
            gate.wait()

        monkeypatch.setattr(routes, "run_benchmark", blocked)
        monkeypatch.setattr(routes, "load_questions", lambda: [{"uuid": "q", "question": "q?", "expected": "e"}])
        try:
            assert client.post("/api/run").status_code == 202
            deadline = time.time() + 5
            while time.time() < deadline:
                if client.get("/api/run").json()["status"] == "running":
                    break
                time.sleep(0.02)
            response = client.post(
                "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
            )
            assert response.status_code == 409
        finally:
            gate.set()


class TestAgentDocInjection:
    """The user-scope AGENTS.md reaches chat, and only chat."""

    @pytest.fixture
    def doc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        directory = tmp_path / "xdg" / "benchmarkit"
        directory.mkdir(parents=True)
        path = directory / "AGENTS.md"
        # Mentions tools explicitly, which is what suppresses TOOL_NOTICE.
        path.write_text(
            "Tools are available for this conversation. "
            "Never pipe inside execute_command.",
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        return path

    def test_chat_receives_the_agent_doc(self, client, monkeypatch, doc):
        seen = {}

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 api_key="none", sysprompt=None):
            seen["sysprompt"] = sysprompt
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert "Never pipe inside execute_command." in seen["sysprompt"]
        # The configured prompt is kept, not replaced.
        assert seen["sysprompt"].startswith(DEFAULT_LLM_SYSPROMPT)

    def test_benchmark_run_never_sees_the_agent_doc(self, client, monkeypatch, doc):
        """Scores must stay comparable regardless of what the file contains."""
        seen = {}

        def fake_run(questions, config, job, emit, cancel_event=None):
            seen["sysprompt"] = config.sysprompt

        monkeypatch.setattr(routes, "run_benchmark", fake_run)
        monkeypatch.setattr(
            routes, "load_questions",
            lambda: [{"uuid": "q", "question": "q?", "expected": "e"}],
        )
        assert client.post("/api/run").status_code == 202
        deadline = time.time() + 5
        while time.time() < deadline:
            if client.get("/api/run").json()["status"] != "running":
                break
            time.sleep(0.02)
        assert seen["sysprompt"] == DEFAULT_LLM_SYSPROMPT

    def test_doc_mentioning_tools_suppresses_the_notice(self, monkeypatch, doc):
        """Documents stating tool availability take over from TOOL_NOTICE.

        ``stream_chat`` appends TOOL_NOTICE only when the system prompt does not
        already mention tools, so an agent doc that states tools are available
        replaces the generic notice instead of doubling it.
        """
        captured = []

        def fake_completion(**kwargs):
            captured.append(kwargs["messages"][0])
            return iter([{"choices": [{"delta": {"content": "hi"}}]}])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}],
            "m",
            "http://s/v1",
            lambda e: None,
            sysprompt=routes.chat_sysprompt(DEFAULT_LLM_SYSPROMPT),
            tools=[{"type": "function"}],
            tool_executor=lambda name, args: "ok",
        )
        assert chat_core.TOOL_NOTICE.strip() not in captured[0]["content"]
        assert "Never pipe inside execute_command." in captured[0]["content"]

    def test_framing_text_alone_does_not_suppress_the_notice(self, monkeypatch, tmp_path):
        """A document saying nothing about tools must not disable TOOL_NOTICE.

        The wrapper around the document is worded to avoid the trigger word, so
        suppression stays driven by the document's own content.
        """
        directory = tmp_path / "xdg" / "benchmarkit"
        directory.mkdir(parents=True)
        (directory / "AGENTS.md").write_text(
            "Answer in French. Keep replies under five lines.", encoding="utf-8"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        captured = []

        def fake_completion(**kwargs):
            captured.append(kwargs["messages"][0])
            return iter([{"choices": [{"delta": {"content": "hi"}}]}])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}],
            "m",
            "http://s/v1",
            lambda e: None,
            sysprompt=routes.chat_sysprompt(DEFAULT_LLM_SYSPROMPT),
            tools=[{"type": "function"}],
            tool_executor=lambda name, args: "ok",
        )
        assert "Answer in French." in captured[0]["content"]
        assert chat_core.TOOL_NOTICE.strip() in captured[0]["content"]


class TestChatMcpFilter:
    """``mcpEnabled`` (Chat tab conversation toggles) intersects the active ids."""

    class RecordingManager:
        installed = True

        def __init__(self):
            self.requested_ids = None

        def status(self):
            return []

        def active_tool_defs(self, ids):
            self.requested_ids = list(ids)
            tools = [
                {"type": "function", "function": {"name": f"use_{cid}"}}
                for cid in ids
            ]
            return tools, {f"use_{cid}": cid for cid in ids}

        def call_tool_sync(self, cid, name, args):
            return "ok"

    @pytest.fixture
    def mcp_setup(self, client, monkeypatch):
        monkeypatch.setattr(routes, "_active_connector_ids", lambda: ["alpha", "beta"])
        manager = self.RecordingManager()
        client.app.state.mcp = manager

        seen = {}

        def fake(messages, model, server_url, emit, sampling=None,
                 cancel_event=None, api_key="none", sysprompt=None, **kwargs):
            seen.update(kwargs)
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        return client, manager, seen

    @staticmethod
    def _post(client, extra=None):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        body.update(extra or {})
        response = client.post("/api/chat", json=body)
        assert response.status_code == 200
        return response

    def test_mcp_enabled_intersects_active_ids(self, mcp_setup):
        client, manager, seen = mcp_setup
        self._post(client, {"mcpEnabled": ["beta"]})
        assert manager.requested_ids == ["beta"]
        assert seen["tool_servers"] == {"use_beta": "beta"}

    def test_mcp_enabled_absent_keeps_all_active(self, mcp_setup):
        client, manager, seen = mcp_setup
        self._post(client)
        assert manager.requested_ids == ["alpha", "beta"]
        assert set(seen["tool_servers"]) == {"use_alpha", "use_beta"}

    def test_empty_mcp_enabled_disables_all_tools(self, mcp_setup):
        client, manager, seen = mcp_setup
        self._post(client, {"mcpEnabled": []})
        assert manager.requested_ids == []
        # No tools at all: the no-MCP stream_chat call shape is preserved.
        assert "tools" not in seen

    def test_unknown_ids_are_dropped_by_the_intersection(self, mcp_setup):
        client, manager, _seen = mcp_setup
        self._post(client, {"mcpEnabled": ["beta", "gone_mcp"]})
        assert manager.requested_ids == ["beta"]

    def test_mcp_enabled_wrong_type_returns_422(self, mcp_setup):
        client, _manager, _seen = mcp_setup
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}],
                  "mcpEnabled": "beta"},
        )
        assert response.status_code == 422
