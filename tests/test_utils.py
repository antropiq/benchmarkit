from src.utils import format_time


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
