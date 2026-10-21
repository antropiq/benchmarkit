import json
from pathlib import Path

import pytest

from src import config as app_config
from src.config import load_endpoints_config, load_questions


class TestLoadEndpointsConfig:
    def test_valid_env_builds_config(self, monkeypatch: pytest.MonkeyPatch):
        for key in (
            "SERVER_URL", "SERVER_URL_API_KEY", "ENDPOINT", "TEMP",
            "TOP_P", "TOP_K", "MIN_P", "REPEAT_PENALTY", "PRESENCE_PENALTY",
            "REASONING", "VERIFIER_AGENT_SYSPROMPT", "VERIFIER_AGENT_SERVERURL",
            "VERIFIER_AGENT_SERVERURL_API_KEY", "VERIFIER_AGENT_ENDPOINT",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SERVER_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("ENDPOINT", "model-a")
        monkeypatch.setenv("TEMP", "0.7")
        monkeypatch.setenv("VERIFIER_AGENT_ENDPOINT", "judge")

        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.model_list() == ["model-a"]
        assert config.temperature == 0.7
        assert config.verifierAgent is not None
        assert config.verifierAgent.endpoint == "judge"

    def test_api_keys_flow_into_config(self, monkeypatch: pytest.MonkeyPatch):
        for key in ("SERVER_URL_API_KEY", "VERIFIER_AGENT_SERVERURL_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SERVER_URL_API_KEY", "sk-target")
        monkeypatch.setenv("VERIFIER_AGENT_SERVERURL_API_KEY", "sk-judge")
        monkeypatch.setenv("VERIFIER_AGENT_ENDPOINT", "judge")

        config, error = load_endpoints_config()
        assert error is None
        assert config.apiKey == "sk-target"
        assert config.verifierAgent.apiKey == "sk-judge"

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


class TestAgentDoc:
    """The user-scope chat agent document (XDG config dir)."""

    def _doc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Path:
        directory = tmp_path / "xdg" / "benchmarkit"
        directory.mkdir(parents=True)
        path = directory / "AGENTS.md"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        return path

    def test_dir_follows_xdg_config_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert app_config.agent_doc_dir() == tmp_path / "xdg" / "benchmarkit"
        assert app_config.agent_doc_path() == tmp_path / "xdg" / "benchmarkit" / "AGENTS.md"

    def test_dir_falls_back_to_home_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        expected = Path.home() / ".config" / "benchmarkit"
        assert app_config.agent_doc_dir() == expected

    def test_empty_xdg_falls_back_to_home_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "  ")
        assert app_config.agent_doc_dir() == Path.home() / ".config" / "benchmarkit"

    def test_missing_doc_loads_as_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nothing"))
        assert app_config.load_agent_doc() == ""

    def test_doc_is_loaded_and_stripped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self._doc(tmp_path, monkeypatch, "\n  never use pipes  \n")
        assert app_config.load_agent_doc() == "never use pipes"

    def test_env_var_overrides_location(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self._doc(tmp_path, monkeypatch, "from xdg")
        custom = tmp_path / "elsewhere.md"
        custom.write_text("from override", encoding="utf-8")
        monkeypatch.setenv("AGENT_DOC", str(custom))
        assert app_config.load_agent_doc() == "from override"

    def test_env_var_can_disable_the_feature(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self._doc(tmp_path, monkeypatch, "from xdg")
        monkeypatch.setenv("AGENT_DOC", str(tmp_path / "absent.md"))
        assert app_config.load_agent_doc() == ""

    def test_oversized_doc_is_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self._doc(tmp_path, monkeypatch, "x" * (app_config.AGENT_DOC_MAX_BYTES + 500))
        assert len(app_config.load_agent_doc()) == app_config.AGENT_DOC_MAX_BYTES

    def test_unreadable_doc_degrades_to_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # A directory where the document is expected must not raise out of chat.
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / "xdg").mkdir(parents=True)
        (tmp_path / "xdg" / "benchmarkit").mkdir()
        assert app_config.load_agent_doc() == ""

    def test_chat_sysprompt_appends_the_doc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self._doc(tmp_path, monkeypatch, "prefer fedit over sed")
        prompt = app_config.chat_sysprompt("You are a helpful assistant.")
        assert prompt.startswith("You are a helpful assistant.")
        assert "prefer fedit over sed" in prompt

    def test_chat_sysprompt_unchanged_without_doc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "nothing"))
        assert app_config.chat_sysprompt("base prompt") == "base prompt"

    def test_empty_doc_leaves_sysprompt_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self._doc(tmp_path, monkeypatch, "   \n\n  \n")
        assert app_config.chat_sysprompt("base prompt") == "base prompt"
