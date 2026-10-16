from unittest.mock import MagicMock

import pytest

from src.core.llm import _extract_total_tokens, safe_get_choice_content


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
