from unittest.mock import MagicMock

import pytest

from src.llm import safe_get_choice_content, safe_get_token_usage


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


class TestSafeGetTokenUsage:
    def test_dict_with_usage_total_tokens(self):
        chunk = {"usage": {"total_tokens": 42}}
        assert safe_get_token_usage(chunk) == 42

    def test_dict_missing_usage_returns_zero(self):
        chunk = {}
        assert safe_get_token_usage(chunk) == 0

    def test_dict_usage_without_total_tokens_returns_zero(self):
        chunk = {"usage": {}}
        assert safe_get_token_usage(chunk) == 0

    def test_object_with_usage_total_tokens(self):
        chunk = MagicMock()
        chunk.usage = {"total_tokens": 99}
        assert safe_get_token_usage(chunk) == 99

    def test_object_missing_usage_attribute_returns_zero(self):
        chunk = MagicMock(spec=[])
        assert safe_get_token_usage(chunk) == 0
