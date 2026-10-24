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


class TestStreamChatTruncation:
    """A server-cut reply must end on ``error``, not a tidy ``done``.

    Without this, a stream that ends with no finish_reason chunk (llama.cpp
    losing its upstream mid-generation) is indistinguishable from a finished
    answer: the UI would settle to idle with a half-finished task, and a
    half-streamed tool call would be executed with truncated arguments.
    """

    class Wrapper:
        def __init__(self, chunks, received=None):
            self._chunks = chunks
            self.received_finish_reason = received

        def __iter__(self):
            return iter(self._chunks)

    def _run(self, monkeypatch, generator, **kwargs):
        monkeypatch.setattr("src.core.chat.completion", lambda **kw: generator)
        events = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi?"}],
            "model-a",
            "http://server:8080/v1",
            events.append,
            **kwargs,
        )
        return events

    def test_cut_reply_ends_on_error_not_done(self, monkeypatch):
        events = self._run(
            monkeypatch,
            self.Wrapper([content_chunk("half an "), content_chunk("answer")]),
        )
        assert [e["event"] for e in events][-1] == "error"
        assert events[-1]["message"] == chat_core.TRUNCATION_MESSAGE
        assert "done" not in [e["event"] for e in events]

    def test_completed_reply_still_ends_on_done(self, monkeypatch):
        events = self._run(
            monkeypatch, self.Wrapper([content_chunk("hi")], received="stop")
        )
        assert events[-1]["event"] == "done"

    def test_plain_iterable_is_left_alone(self, monkeypatch):
        # Fails open, exactly as in the benchmark loop.
        events = self._run(monkeypatch, iter([content_chunk("hi")]))
        assert events[-1]["event"] == "done"

    def test_cut_tool_call_is_not_executed(self, monkeypatch):
        # The dangerous case: a tool call whose arguments stopped streaming
        # would run with truncated JSON, and the model would be told its own
        # arguments were invalid — blaming it for a transport cut.
        executed = []
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": '{"command": "ls -l',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        events = self._run(
            monkeypatch,
            self.Wrapper([chunk]),
            tools=[{"type": "function", "function": {"name": "execute_command"}}],
            tool_executor=lambda name, args: executed.append(name) or "ok",
        )
        assert executed == []
        assert not [e for e in events if e["event"] == "tool_call"]
        assert events[-1]["event"] == "error"

    def test_reasoning_only_cut_ends_on_error(self, monkeypatch):
        events = self._run(
            monkeypatch, self.Wrapper([reasoning_chunk("thinking…")])
        )
        assert events[-1]["event"] == "error"

    def test_empty_stream_is_not_flagged(self, monkeypatch):
        events = self._run(monkeypatch, self.Wrapper([]))
        assert events[-1]["event"] == "done"


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
        # Single round: the occupancy figure is that round's total, exact.
        assert done["contextTokens"] == 12
        assert done["contextEstimated"] is False

    def test_context_estimated_from_prompt_when_server_reports_no_usage(
        self, monkeypatch,
    ):
        # No usage chunk at all: completion keeps its char estimate, and the
        # occupancy figure adds the char estimate of the prompt the request
        # carries (system prompt + user text) — flagged as estimated.
        from src.core.models import DEFAULT_LLM_SYSPROMPT
        from src.core.utils import estimate_completion_tokens

        events = self._run(monkeypatch, [content_chunk("Hello there")])
        done = events[-1]
        prompt_est = estimate_completion_tokens(DEFAULT_LLM_SYSPROMPT + "hi?")
        assert done["tokens"] == estimate_completion_tokens("Hello there")
        # Without usage the occupancy figure is the char-based round
        # estimate (prompt + estimated generation), flagged.
        assert done["contextTokens"] == prompt_est + done["tokens"]
        assert done["contextEstimated"] is True

    def test_context_prompt_exact_when_only_completion_is_estimated(
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
        assert done["contextTokens"] == 41
        assert done["contextEstimated"] is True

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

    def test_agent_instructions_appended_to_the_system_message(self, monkeypatch):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s/v1", lambda e: None,
            sysprompt="Answer concisely.",
            agent_instructions="project root is /srv/proj\n\nuse pytest",
        )
        system = captured["messages"][0]
        assert system["role"] == "system"
        assert system["content"].startswith("Answer concisely.")
        assert system["content"].endswith("project root is /srv/proj\n\nuse pytest")

    def test_blank_agent_instructions_leave_the_prompt_untouched(self, monkeypatch):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s/v1", lambda e: None,
            sysprompt="Answer concisely.", agent_instructions="",
        )
        assert captured["messages"][0]["content"] == "Answer concisely."

    def test_agent_instructions_do_not_suppress_the_tool_notice(self, monkeypatch):
        # Instruction files mention "tool" everywhere; the notice-skipping
        # heuristic must judge only the configured prompt, so passing the
        # instructions separately keeps the nudge alive.
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return iter([])

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s/v1", lambda e: None,
            sysprompt="Answer concisely.",
            tools=[{"type": "function", "function": {"name": "use_thing"}}],
            tool_executor=lambda name, args: "ok",
            agent_instructions="tools are documented in AGENTS.md",
        )
        system = captured["messages"][0]["content"]
        assert chat_core.TOOL_NOTICE.strip() in system
        assert system.endswith("tools are documented in AGENTS.md")

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
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
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
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
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
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
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
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
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
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
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

    def test_agent_instructions_reach_the_chat_call(
        self, client, monkeypatch, tmp_path: Path
    ):
        # The working location's AGENTS.md is injected on every request, so a
        # conversation starts knowing the project's conventions.
        (tmp_path / "AGENTS.md").write_text("# rules\n\nrun pytest first\n", encoding="utf-8")
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        seen = {}

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
            seen["instructions"] = agent_instructions
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert seen["instructions"].endswith("run pytest first")
        assert f"project root is {tmp_path}" in seen["instructions"]

    def test_agent_instructions_empty_without_the_file(
        self, client, monkeypatch, tmp_path: Path
    ):
        # No AGENTS.md at the location: the configured prompt stands alone.
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
        seen = {}

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
            seen["instructions"] = agent_instructions
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert seen["instructions"] == ""

    def test_error_event_terminates_stream(self, client, monkeypatch):
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
            emit({"event": "error", "message": "boom"})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        frames = _events(response.text)
        assert [name for name, _ in frames] == ["error"]

    def test_truncation_flag_survives_the_sse_frame(self, client, monkeypatch):
        # The whole event dict is JSON-serialized into the frame, so the flag
        # the WorkBench view needs to mark the bubble reaches the browser.
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
            emit({"event": "error", "message": "cut short", "truncated": True})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        name, payload = _events(response.text)[0]
        assert name == "error"
        assert payload["message"] == "cut short"
        assert payload["truncated"] is True

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


class TestChatPauseEndpoint:
    """``POST /api/chat/pause`` arms the flag of the one in-flight reply."""

    def test_no_reply_streaming_returns_409(self, client):
        response = client.post("/api/chat/pause")
        assert response.status_code == 409
        assert "No chat reply is streaming" in response.json()["detail"]

    def test_pause_arms_the_flag_and_the_stream_ends_on_pause_ack(
        self, client, monkeypatch
    ):
        seen = {}
        started = threading.Event()
        released = threading.Event()

        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
            seen["pause_event"] = pause_event
            started.set()
            # Stands in for the core: the flag is what the real loop checks.
            while not pause_event.is_set():
                time.sleep(0.01)
            # Hold the reply open across both pause calls: the second one must
            # still see the handle, so no ack may go out before it landed.
            released.wait(timeout=5)
            emit({"event": "token", "text": "partial"})
            emit({"event": "usage", "tokens": 3, "decode": 0.2,
                  "contextTokens": 12, "contextEstimated": False})
            emit({"event": "pause_ack", "partial": True})

        monkeypatch.setattr(routes, "stream_chat", fake)

        def arm_pause():
            # The TestClient buffers a whole SSE body before returning, so the
            # reply below only ends once the flag is armed: the POST has to come
            # from another thread, exactly as it does in the browser.
            started.wait(timeout=5)
            arm_pause.results["first"] = client.post("/api/chat/pause")
            arm_pause.results["again"] = client.post("/api/chat/pause")
            released.set()

        arm_pause.results = {}
        poster = threading.Thread(target=arm_pause)
        poster.start()
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        poster.join(timeout=5)

        assert seen["pause_event"] is not None
        first = arm_pause.results["first"]
        assert first.status_code == 200
        assert first.json() == {"paused": "requested"}
        # Idempotent: a second call while the ack is pending says so.
        again = arm_pause.results["again"]
        assert again.status_code == 200
        assert again.json() == {"paused": "pending"}
        names = [name for name, _data in _events(response.text)]
        assert names == ["token", "usage", "pause_ack"]

    def test_handle_is_retired_once_the_reply_ended(self, client, monkeypatch):
        def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
                 pause_event=None, api_key="none", sysprompt=None, agent_instructions=None):
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 1,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert [name for name, _ in _events(response.text)] == ["done"]
        assert client.post("/api/chat/pause").status_code == 409


class TestStreamChatPause:
    """``pause_event`` interrupts a reply without throwing its context away.

    The ack must be terminal and unambiguous (no ``done``, no ``error``, no
    truncation frame — the pause itself closed the stream, so no finish_reason
    was ever expected), and it must carry the exact usage of the partial turn.
    """

    class Wrapper:
        def __init__(self, chunks, received=None):
            self._chunks = chunks
            self.received_finish_reason = received
            self.closed = False

        def __iter__(self):
            return iter(self._chunks)

        def close(self):
            self.closed = True

    def _run(self, monkeypatch, generator, **kwargs):
        monkeypatch.setattr("src.core.chat.completion", lambda **kw: generator)
        events = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi?"}],
            "model-a",
            "http://server:8080/v1",
            events.append,
            **kwargs,
        )
        return events

    def test_paused_reply_ends_on_pause_ack(self, monkeypatch):
        pause = threading.Event()
        pause.set()
        events = self._run(
            monkeypatch,
            self.Wrapper([content_chunk("half an "), content_chunk("answer")]),
            pause_event=pause,
        )
        names = [e["event"] for e in events]
        # Nothing streamed (the flag was armed before the first chunk), but the
        # ack is still terminal and preceded by the usage frame.
        assert names == ["usage", "pause_ack"]
        assert events[-1]["partial"] is True
        assert "done" not in names and "error" not in names

    def test_pause_is_never_reported_as_a_truncation(self, monkeypatch):
        pause = threading.Event()
        pause.set()
        events = self._run(
            monkeypatch,
            self.Wrapper([content_chunk("partial")]),
            pause_event=pause,
        )
        assert not [e for e in events if e["event"] == "error"]
        assert not [e for e in events if e.get("truncated")]

    def test_pause_closes_the_upstream_stream(self, monkeypatch):
        pause = threading.Event()
        pause.set()
        wrapper = self.Wrapper([content_chunk("x")])
        self._run(monkeypatch, wrapper, pause_event=pause)
        assert wrapper.closed

    def test_pause_mid_stream_keeps_only_the_tokens_so_far(self, monkeypatch):
        # The check is per chunk: a flag armed after N chunks must stop the
        # loop there, not at the end of the stream.
        pause = threading.Event()

        class Gen:
            received_finish_reason = None

            def __iter__(self):
                for i in range(10):
                    if i == 3:
                        pause.set()
                    yield content_chunk(f"t{i}")

        events = self._run(monkeypatch, Gen(), pause_event=pause)
        tokens = [e["text"] for e in events if e["event"] == "token"]
        assert tokens == ["t0", "t1", "t2"]
        assert events[-1]["event"] == "pause_ack"

    def test_partial_tokens_then_usage_then_the_ack(self, monkeypatch):
        # The flag is armed after the server's usage chunk: what streamed so
        # far is kept, the usage frame carries the exact cumulative figures,
        # and the ack terminates the reply (the chunk after the flag is gone).
        pause = threading.Event()

        class Gen:
            received_finish_reason = None

            def __iter__(self):
                yield content_chunk("par")
                yield usage_chunk(prompt=40, completion=8, total=48)
                pause.set()
                yield content_chunk("tially-streamed-but-skipped")

        events = self._run(monkeypatch, Gen(), pause_event=pause)
        names = [e["event"] for e in events]
        assert names == ["token", "usage", "pause_ack"]
        usage = events[1]
        assert usage["tokens"] == 8
        assert usage["contextTokens"] == 48
        assert usage["contextEstimated"] is False
        assert not [e for e in events if e.get("text", "").startswith("tially")]

    def test_flag_armed_after_the_last_chunk_still_ends_on_done(self, monkeypatch):
        # The race: the round streamed to its end with a real finish_reason and
        # the flag was armed only after that last chunk. The reply is complete,
        # so `done` wins — a round that carries its finish_reason was never
        # interrupted, whatever the flag says afterwards.
        pause = threading.Event()

        class Gen:
            received_finish_reason = "stop"

            def __iter__(self):
                yield content_chunk("whole answer")
                pause.set()

        events = self._run(monkeypatch, Gen(), pause_event=pause)
        names = [e["event"] for e in events]
        assert names == ["token", "done"]

    def test_cancel_wins_over_a_pause_on_the_same_chunk(self, monkeypatch):
        # Stop after Pause: the client is gone, so nothing may be emitted —
        # the cancel check comes first at every pause site.
        cancel = threading.Event()
        cancel.set()
        pause = threading.Event()
        pause.set()
        wrapper = self.Wrapper([content_chunk("x")])
        events = self._run(
            monkeypatch, wrapper, cancel_event=cancel, pause_event=pause
        )
        assert events == []
        assert wrapper.closed

    def test_pause_while_tool_calls_stream_never_runs_them(self, monkeypatch):
        # The dangerous shape: tool-call arguments only half-streamed, so
        # executing them would feed the model invalid JSON of its own cutting.
        executed = []
        pause = threading.Event()

        class Gen:
            received_finish_reason = None

            def __iter__(self):
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "execute_command",
                                            "arguments": '{"command": "ls',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
                pause.set()
                yield content_chunk("must not stream")

        events = self._run(
            monkeypatch,
            Gen(),
            pause_event=pause,
            tools=[{"type": "function", "function": {"name": "execute_command"}}],
            tool_executor=lambda name, args: executed.append(name) or "ok",
        )
        assert executed == []
        assert not [e for e in events if e["event"] == "tool_call"]
        assert events[-1]["event"] == "pause_ack"

    def test_pause_asked_during_a_tool_run_ends_between_rounds(self, monkeypatch):
        # A running tool is allowed to finish (its `tool_result` still lands,
        # so the badge never sticks on "running"), and the next round is never
        # started. No extra `usage` frame: this round's figures went out as the
        # pre-tool one.
        pause = threading.Event()

        def executor(name, args):
            pause.set()
            return "done"

        chunks_round_one = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "execute_command",
                                        "arguments": '{"command": "ls"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            usage_chunk(prompt=10, completion=4, total=14),
        ]

        class Wrapper:
            received_finish_reason = "tool_calls"

            def __iter__(self):
                return iter(chunks_round_one)

        monkeypatch.setattr("src.core.chat.completion", lambda **kw: Wrapper())
        events = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi?"}],
            "model-a",
            "http://server:8080/v1",
            events.append,
            pause_event=pause,
            tools=[{"type": "function", "function": {"name": "execute_command"}}],
            tool_executor=executor,
        )
        names = [e["event"] for e in events]
        # `usage` precedes the tool badges (it is emitted as soon as that
        # round stops generating), and the ack is the only terminal frame.
        assert names == ["usage", "tool_call", "tool_result", "pause_ack"]


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
