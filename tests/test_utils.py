import json
from pathlib import Path

from src.core.job import RunJob
from src.core.utils import estimate_completion_tokens, format_time, save_results


class TestEstimateCompletionTokens:
    def test_empty_text_returns_zero(self):
        assert estimate_completion_tokens("") == 0

    def test_rounds_up_quarter_length(self):
        assert estimate_completion_tokens("abcd") == 1
        assert estimate_completion_tokens("abcde") == 2
        assert estimate_completion_tokens("a" * 10) == 3


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


class TestSaveResults:
    def test_writes_expected_schema(self, job: RunJob, tmp_path: Path):
        job.add_result("gpt-4", "q1", True, "hello", 2.0, 100)
        path = tmp_path / "results.json"
        data = save_results(job, path)
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written == json.loads(data)
        assert written["totalRunningTime"] == "00:00:02"
        assert written["results"][0]["llm"] == "gpt-4"
        assert written["results"][0]["score"] == "1/1"
