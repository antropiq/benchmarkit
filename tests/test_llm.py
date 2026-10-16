import threading
from unittest.mock import MagicMock

import pytest

from src.core.llm import _extract_total_tokens, safe_get_choice_content, warmup_model


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
