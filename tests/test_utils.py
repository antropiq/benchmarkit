from io import StringIO

import pytest

from src.utils import format_time, load_json_from_file


class TestFormatTime:
    def test_zero_seconds(self):
        assert format_time(0) == "00:00:00"

    def test_exactly_one_minute(self):
        assert format_time(60) == "00:01:00"

    def test_exactly_one_hour(self):
        assert format_time(3600) == "01:00:00"

    def test_one_hour_one_minute_one_second(self):
        assert format_time(3661) == "01:01:01"

    def test_large_value(self):
        assert format_time(90061) == "25:01:01"

    def test_fractional_seconds_truncated(self):
        assert format_time(59.9) == "00:00:59"


class TestLoadJsonFromFile:
    def test_valid_json(self, sample_questions_file: StringIO):
        result = load_json_from_file(sample_questions_file)
        assert isinstance(result, dict)
        assert "questions" in result
        assert len(result["questions"]) == 3

    def test_invalid_json_calls_error_and_returns_empty_dict(
        self, mocker
    ):
        mock_error = mocker.patch("src.utils.st.error")
        bad_file = StringIO("{bad json}")
        result = load_json_from_file(bad_file)
        assert result == {}
        mock_error.assert_called_once()
