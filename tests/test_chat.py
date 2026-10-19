"""Tests for the streaming chat core and the /api/chat SSE endpoint."""

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
from src.core.models import EndpointsConfig


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
        assert names == ["reasoning", "token", "token", "done"]
        text = "".join(e["text"] for e in events if e["event"] == "token")
        assert text == "Hello"
        # Reasoning text must never leak into emitted text.
        assert all("thinking" not in json.dumps(e) for e in events)
        done = events[-1]
        assert done["tokens"] == 7
        assert done["tokensEstimated"] is False

    def test_reasoning_deltas_open_and_extend_decode_window(self, monkeypatch):
        # t0, reasoning, reasoning, content — the usage-only chunk is not
        # timed. The window must start at the first reasoning delta, like in
        # the benchmark loop, so tok/s does not pretend thinking was free.
        clock = [0.0, 1.0, 2.0, 6.0]
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

    def test_prepends_system_prompt(self, monkeypatch):
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
            "content": chat_core.CHAT_SYSTEM_PROMPT,
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

    def test_estimates_tokens_without_usage(self, monkeypatch):
        events = self._run(monkeypatch, [content_chunk("hello world!!!")])
        done = events[-1]
        assert done["event"] == "done"
        assert done["tokens"] == 4
        assert done["tokensEstimated"] is True


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


class TestChatEndpoint:
    def test_happy_path_streams_frames(self, client, monkeypatch):
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None):
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

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None):
            seen["model"] = model
            seen["server_url"] = server_url
            seen["messages"] = messages
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
        assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user"]

    def test_error_event_terminates_stream(self, client, monkeypatch):
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None):
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
