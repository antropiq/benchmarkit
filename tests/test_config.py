import json
import os
from pathlib import Path

import pytest

from src.config import (
    PATH_LOCATION_KEY,
    agent_instructions_block,
    load_agent_instructions,
    default_path_location,
    load_endpoints_config,
    load_questions,
    path_location,
    setup_path_location,
)


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


class TestPathLocation:
    """The boot default of BENCHMARKIT_PATH_LOCATION (the hubs' working dir)."""

    def test_seeded_from_the_install_dir_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(PATH_LOCATION_KEY, raising=False)
        setup_path_location()
        assert os.environ[PATH_LOCATION_KEY] == default_path_location()
        assert path_location() == default_path_location()

    def test_an_existing_value_wins(self, monkeypatch: pytest.MonkeyPatch):
        # Exported by the launcher or written in .env: boot must not clobber
        # it, which is what lets an operator pin the location.
        monkeypatch.setenv(PATH_LOCATION_KEY, "/somewhere/else")
        setup_path_location()
        assert path_location() == "/somewhere/else"

    def test_blank_value_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(PATH_LOCATION_KEY, "   ")
        setup_path_location()
        assert path_location() == default_path_location()

    def test_is_idempotent(self, monkeypatch: pytest.MonkeyPatch):
        # create_app() runs per test; re-seeding must not drift the value.
        monkeypatch.delenv(PATH_LOCATION_KEY, raising=False)
        setup_path_location()
        first = os.environ[PATH_LOCATION_KEY]
        setup_path_location()
        assert os.environ[PATH_LOCATION_KEY] == first

    def test_read_live_after_an_api_style_change(self, monkeypatch: pytest.MonkeyPatch):
        # The endpoint mutates os.environ directly; path_location() must see it
        # without any reload, which is what makes the tool restart pick it up.
        monkeypatch.setenv(PATH_LOCATION_KEY, "/before")
        assert path_location() == "/before"
        monkeypatch.setenv(PATH_LOCATION_KEY, "/after")
        assert path_location() == "/after"


class TestAgentInstructions:
    """The project instructions (AGENTS.md) injected into the Chat prompt."""

    def test_reads_the_file_at_the_location_root(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("# rules\n\nUse pytest.\n", encoding="utf-8")
        assert load_agent_instructions(str(tmp_path)) == "# rules\n\nUse pytest."

    def test_missing_file_yields_empty(self, tmp_path: Path):
        assert load_agent_instructions(str(tmp_path)) == ""

    def test_unreadable_file_yields_empty(self, tmp_path: Path):
        # A directory named AGENTS.md raises OSError: never propagate it, a
        # chat request must work even when the lookup is broken.
        (tmp_path / "AGENTS.md").mkdir()
        assert load_agent_instructions(str(tmp_path)) == ""

    def test_resolves_the_location_live(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # No explicit location: the working location is read at call time, so
        # moving it through the settings dialog moves the lookup too.
        (tmp_path / "AGENTS.md").write_text("from the env", encoding="utf-8")
        monkeypatch.setenv(PATH_LOCATION_KEY, str(tmp_path))
        assert load_agent_instructions() == "from the env"

    def test_block_frames_the_text_with_the_project_root(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(PATH_LOCATION_KEY, "/srv/proj")
        block = agent_instructions_block("mind the conventions")
        assert "The current project root is /srv/proj" in block
        assert block.endswith("mind the conventions")
        assert "AGENTS.md" in block

    def test_block_is_empty_without_instructions(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # No file (and no text) means the system prompt is left untouched.
        monkeypatch.setenv(PATH_LOCATION_KEY, str(tmp_path))
        assert agent_instructions_block() == ""
        assert agent_instructions_block("   ") == ""

    def test_block_reads_the_file_when_no_text_is_given(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        (tmp_path / "AGENTS.md").write_text("body text", encoding="utf-8")
        monkeypatch.setenv(PATH_LOCATION_KEY, str(tmp_path))
        assert agent_instructions_block().endswith("body text")

    def test_a_renamed_file_is_not_picked_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        (tmp_path / "AGENTS.md").write_text("nope", encoding="utf-8")
        monkeypatch.setattr("src.config.agent_instructions_filename", lambda: "OTHER.md")
        monkeypatch.setenv(PATH_LOCATION_KEY, str(tmp_path))
        assert load_agent_instructions() == ""
