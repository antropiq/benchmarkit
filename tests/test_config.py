import json
from pathlib import Path

import pytest

from src import config as app_config
from src.config import load_endpoints_config, load_questions


class TestLoadEndpointsConfig:
    def test_valid_env_builds_config(self, monkeypatch: pytest.MonkeyPatch):
        for key in (
            "SERVER_URL", "ENDPOINT", "ENDPOINTS", "TEMP", "TOP_P", "TOP_K",
            "MIN_P", "REPEAT_PENALTY", "PRESENCE_PENALTY", "REASONING",
            "VERIFIER_AGENT_SYSPROMPT", "VERIFIER_AGENT_SERVERURL",
            "VERIFIER_AGENT_ENDPOINT",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SERVER_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("ENDPOINTS", "model-a, model-b")
        monkeypatch.setenv("TEMP", "0.7")
        monkeypatch.setenv("VERIFIER_AGENT_ENDPOINT", "judge")

        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.model_list() == ["model-a", "model-b"]
        assert config.temperature == 0.7
        assert config.verifierAgent is not None
        assert config.verifierAgent.endpoint == "judge"

    def test_invalid_env_returns_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("TEMP", "not-a-number")
        config, error = load_endpoints_config()
        assert config is None
        assert error is not None
        # Must be JSON-serializable for the API response.
        json.loads(error.json())


class TestLoadQuestions:
    def test_reads_questions_key(self, tmp_path: Path):
        path = tmp_path / "questions.json"
        path.write_text(
            json.dumps({"questions": [{"uuid": "1", "question": "q", "expected": "e"}]}),
            encoding="utf-8",
        )
        assert load_questions(path) == [
            {"uuid": "1", "question": "q", "expected": "e"}
        ]

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "nope.json")
