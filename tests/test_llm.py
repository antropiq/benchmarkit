import threading
from unittest.mock import MagicMock

import pytest

from src.core.llm import (
    _extract_total_tokens,
    _extract_usage,
    call_llm_and_stream,
    safe_get_choice_content,
    warmup_model,
)


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


class TestCallLlmAndStreamTiming:
    def _run(self, monkeypatch, chunks, clock_values):
        clock = list(clock_values)

        def fake_monotonic():
            return clock.pop(0)

        monkeypatch.setattr("src.core.llm.time.monotonic", fake_monotonic)
        monkeypatch.setattr(
            "src.core.llm.completion", lambda **kwargs: iter(chunks)
        )
        events = []
        result = call_llm_and_stream(
            "m", "http://server:8080/v1",
            {"uuid": "u", "question": "q", "expected": "e"}, events.append,
        )
        assert clock == []  # every monotonic() call consumed a tick
        return result, events

    def test_usage_chunk_arrival_does_not_extend_timing(self, monkeypatch):
        # t0, first content, last content, stream-end (never used as took).
        chunks = [content_chunk("a"), content_chunk("bc"),
                  usage_chunk(prompt=20, completion=5, total=25)]
        result, _ = self._run(monkeypatch, chunks, [0.0, 1.0, 3.0, 100.0])

        assert result.text == "abc"
        assert result.took == 3.0   # t_last - t0, ignoring the 100.0 drain
        assert result.ttft == 1.0
        assert result.gen_duration == 2.0
        assert result.completion_tokens == 5
        assert result.prompt_tokens == 20
        assert result.tokens_estimated is False

    def test_missing_usage_estimates_tokens(self, monkeypatch):
        chunks = [content_chunk("hello world!!!")]  # 14 chars -> 4 tokens
        result, _ = self._run(monkeypatch, chunks, [0.0, 0.5, 2.0])

        assert result.completion_tokens == 4
        assert result.tokens_estimated is True
        assert result.prompt_tokens == 0

    def test_completion_derived_from_total_minus_prompt(self, monkeypatch):
        chunks = [content_chunk("x"),
                  usage_chunk(prompt=10, total=18)]
        result, _ = self._run(monkeypatch, chunks, [0.0, 0.2, 1.0])

        assert result.completion_tokens == 8
        assert result.tokens_estimated is False

    def test_empty_stream_has_zero_generation_window(self, monkeypatch):
        result, _ = self._run(monkeypatch, [], [0.0, 5.0])

        assert result.text == ""
        assert result.took == 5.0
        assert result.ttft == 0.0
        assert result.gen_duration == 0.0
        assert result.completion_tokens == 0
        assert result.tokens_estimated is False

    def test_call_failure_returns_zeroed_result(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("connect failed")

        clock = [0.0, 0.25]
        monkeypatch.setattr("src.core.llm.time.monotonic", lambda: clock.pop(0))
        monkeypatch.setattr("src.core.llm.completion", boom)
        events = []
        result = call_llm_and_stream(
            "m", "http://server:8080/v1",
            {"uuid": "u", "question": "q", "expected": "e"}, events.append,
        )

        assert result.text == ""
        assert result.took == 0.25
        assert result.gen_duration == 0.0
        assert events[0]["event"] == "run_error"

    def test_emits_token_events_in_order(self, monkeypatch):
        chunks = [content_chunk("he"), content_chunk("llo")]
        result, events = self._run(monkeypatch, chunks, [0.0, 0.1, 0.2, 0.3])

        assert [e["text"] for e in events if e["event"] == "token"] == ["he", "llo"]

    def test_reasoning_deltas_anchor_the_generation_window(self, monkeypatch):
        # Prefill ends, reasoning streams first, then the answer: the decode
        # window must start at the first reasoning token, not the first
        # content token, or the rate is inflated by the reasoning wait.
        chunks = [
            reasoning_chunk("thinking…"),
            reasoning_chunk("more…"),
            content_chunk("4"),
            content_chunk("2"),
            usage_chunk(prompt=20, completion=40, total=60),
        ]
        result, events = self._run(
            monkeypatch, chunks, [0.0, 9.9, 11.0, 12.04, 16.5, 100.0]
        )

        assert result.text == "42"
        assert result.ttft == 9.9
        assert result.took == 16.5
        assert result.gen_duration == 16.5 - 9.9
        assert result.completion_tokens == 40
        assert [e["text"] for e in events if e["event"] == "token"] == ["4", "2"]

    def test_reasoning_only_stream_has_zero_answer_text(self, monkeypatch):
        chunks = [reasoning_chunk("only "), reasoning_chunk("thinking")]
        result, events = self._run(monkeypatch, chunks, [0.0, 1.0, 2.0, 3.0])

        assert result.text == ""
        assert result.gen_duration == 1.0
        assert [e["event"] for e in events if e["event"] == "token"] == []


class TestSafeGetChoiceContent:
    def test_dict_with_delta_content(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        assert safe_get_choice_content(chunk) == "hello"

    def test_dict_missing_delta_returns_empty(self):
        chunk = {"choices": [{}]}
        assert safe_get_choice_content(chunk) == ""

    def test_dict_with_no_choices_returns_empty(self):
        chunk = {}
        assert safe_get_choice_content(chunk) == ""

    def test_object_with_choices_and_delta(self):
        delta = MagicMock()
        delta.content = "world"
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        assert safe_get_choice_content(chunk) == "world"

    def test_object_missing_delta_attribute_returns_empty(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = None
        assert safe_get_choice_content(chunk) == ""

    def test_non_dict_non_object_returns_empty(self):
        assert safe_get_choice_content("string") == ""
        assert safe_get_choice_content(123) == ""

    def test_dict_with_empty_choices_returns_empty(self):
        chunk = {"choices": [], "usage": {"total_tokens": 12}}
        assert safe_get_choice_content(chunk) == ""


class TestExtractTotalTokens:
    def test_dict_chunk_with_usage(self):
        chunk = {"choices": [], "usage": {"total_tokens": 42}}
        assert _extract_total_tokens(chunk) == 42

    def test_object_chunk_with_usage(self):
        usage = MagicMock()
        usage.total_tokens = 7
        chunk = MagicMock()
        chunk.usage = usage
        assert _extract_total_tokens(chunk) == 7

    def test_chunk_without_usage_returns_none(self):
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        assert _extract_total_tokens(chunk) is None

    def test_zero_total_returns_none(self):
        assert _extract_total_tokens({"usage": {"total_tokens": 0}}) is None


class TestExtractUsage:
    def test_dict_chunk_with_full_usage(self):
        chunk = {"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                           "total_tokens": 15}}
        assert _extract_usage(chunk) == {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }

    def test_object_chunk_with_usage(self):
        usage = MagicMock()
        usage.prompt_tokens = 3
        usage.completion_tokens = 2
        usage.total_tokens = 5
        chunk = MagicMock()
        chunk.usage = usage
        assert _extract_usage(chunk) == {
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
        }

    def test_missing_fields_are_none(self):
        chunk = {"usage": {"completion_tokens": 4}}
        assert _extract_usage(chunk) == {
            "prompt_tokens": None, "completion_tokens": 4, "total_tokens": None,
        }

    def test_chunk_without_usage_returns_none(self):
        assert _extract_usage({"choices": [{"delta": {"content": "hi"}}]}) is None

    def test_empty_usage_returns_none(self):
        assert _extract_usage({"usage": {}}) is None


class TestWarmupModel:
    def test_sends_minimal_non_streaming_request(self, monkeypatch):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr("src.core.llm.completion", fake_completion)
        warmup_model("my-model", "http://server:8080/v1")

        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["model"] == "openai/my-model"
        assert kwargs["api_base"] == "http://server:8080/v1"
        assert kwargs["max_tokens"] == 1
        assert "stream" not in kwargs
        assert "stream_options" not in kwargs

    def test_swallows_errors(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("model not found")

        monkeypatch.setattr("src.core.llm.completion", boom)
        warmup_model("missing", "http://server:8080/v1")

    def test_cancelled_before_call_skips_request(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "src.core.llm.completion", lambda **kwargs: called.append(kwargs)
        )
        cancel = threading.Event()
        cancel.set()
        warmup_model("my-model", "http://server:8080/v1", cancel)
        assert called == []
