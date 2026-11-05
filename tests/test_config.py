import json
import logging
import os
from pathlib import Path

import pytest

from src.core.llmkit import client as llmkit_client
from src.core.llmkit import constants as llmkit_constants
from src.config import (
    LLM_DEBUG_KEY,
    LEGACY_LLM_DEBUG_KEY,
    PATH_LOCATION_KEY,
    _DebugStreamHandler,
    agent_instructions_block,
    apply_llm_debug,
    llm_debug_enabled,
    load_agent_instructions,
    normalize_env_flag,
    default_path_location,
    load_endpoints_config,
    load_questions,
    path_location,
    setup_path_location,
)


class TestLoadEndpointsConfig:
    def test_valid_env_builds_config(self, monkeypatch: pytest.MonkeyPatch):
        for key in (
            "SERVER_URL", "SERVER_URL_API_KEY", "ENDPOINT", "TEMPERATURE",
            "TOP_P", "TOP_K", "MIN_P", "REPEAT_PENALTY", "PRESENCE_PENALTY",
            "REASONING", "VERIFIER_AGENT_SYSPROMPT", "VERIFIER_AGENT_SERVERURL",
            "VERIFIER_AGENT_SERVERURL_API_KEY", "VERIFIER_AGENT_ENDPOINT",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SERVER_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("ENDPOINT", "model-a")
        monkeypatch.setenv("TEMPERATURE", "0.7")
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
        assert config is not None
        assert config.apiKey == "sk-target"
        assert config.verifierAgent is not None
        assert config.verifierAgent.apiKey == "sk-judge"

    def test_invalid_env_returns_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("TEMPERATURE", "not-a-number")
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


def _llm_logger() -> logging.Logger:
    """The logger ``core/llmkit`` writes its request payloads to."""
    return logging.getLogger(llmkit_constants.DEBUG_LOGGER_NAME)


def _switch_handlers(logger: logging.Logger) -> list:
    """Only the handlers the debug switch itself attaches.

    Asked by class, because the logger legitimately carries others: the one an
    app boot attached when the developer's ``.env`` turns the switch on, and
    pytest's own capture handlers. A total over ``logger.handlers`` would make an
    assertion here depend on the machine and on the tests that ran before it.
    """
    return [h for h in logger.handlers if isinstance(h, _DebugStreamHandler)]


class TestLLMDebugFlag:
    """The LLM_DEBUG switch: how it is read, what it does to ``benchmarkit.llm``.

    Both keys are unset, and the logger and the client's payload hook restored,
    around every test of the suite (``conftest._no_debug_switch_leak``), so each
    case states only the key it cares about and the switch's writes to
    process-wide state stay inside the test that made them.
    """

    def test_truthy_spellings_enable_it(self, monkeypatch):
        for raw in ("1", "true", "TRUE", " on "):
            monkeypatch.setenv(LLM_DEBUG_KEY, raw)
            assert llm_debug_enabled() is True
        for raw in ("", "0", "false", "off", "yes please"):
            monkeypatch.setenv(LLM_DEBUG_KEY, raw)
            assert llm_debug_enabled() is False

    def test_unset_means_off(self):
        assert llm_debug_enabled() is False

    def test_the_fallback_key_is_the_pre_rename_name(self):
        """The one place the old name is spelled outside ``config`` itself.

        The whole D6 fallback hangs on this exact string, so renaming the
        constant would be a behaviour change and not a tidy-up — which is what
        makes it worth pinning by value, not by the constant's own name.
        """
        assert LEGACY_LLM_DEBUG_KEY == "LITELLM_DEBUG"
        assert LLM_DEBUG_KEY == "LLM_DEBUG"

    def test_the_legacy_key_alone_still_enables_it(self, monkeypatch):
        # A .env written by hand before the rename keeps working until the
        # settings dialog regenerates that file from its template.
        monkeypatch.setenv(LEGACY_LLM_DEBUG_KEY, "1")
        assert llm_debug_enabled() is True

    def test_the_new_key_wins_when_both_are_set(self, monkeypatch):
        monkeypatch.setenv(LLM_DEBUG_KEY, "0")
        monkeypatch.setenv(LEGACY_LLM_DEBUG_KEY, "1")
        assert llm_debug_enabled() is False
        monkeypatch.setenv(LLM_DEBUG_KEY, "1")
        monkeypatch.setenv(LEGACY_LLM_DEBUG_KEY, "off")
        assert llm_debug_enabled() is True

    def test_normalising_a_submitted_flag(self):
        assert normalize_env_flag("true") == "1"
        assert normalize_env_flag(" 1 ") == "1"
        assert normalize_env_flag("off") == ""
        assert normalize_env_flag(None) == ""

    def test_apply_on_raises_the_clients_logger_and_owns_its_hook(self, monkeypatch):
        monkeypatch.setenv(LLM_DEBUG_KEY, "1")
        logger = _llm_logger()

        assert apply_llm_debug() is True
        assert logger.level == logging.DEBUG
        # Exactly one handler *of ours*, counted as such rather than as a total:
        # an app that booted with the switch on, and pytest's own capture, also
        # put handlers on this logger and the switch must not disturb them. The
        # handler is what makes the flag observable at all — a level alone would
        # print nothing, since uvicorn attaches handlers to its own loggers only.
        assert len(_switch_handlers(logger)) == 1
        assert logger.propagate is False
        # The payload hook follows the switch (the payload half of D8).
        assert llmkit_client.debug_enabled is llm_debug_enabled
        assert llmkit_client.debug_enabled() is True

    def test_apply_off_leaves_the_logger_as_it_was_found(self, monkeypatch):
        monkeypatch.setenv(LLM_DEBUG_KEY, "1")
        logger = _llm_logger()

        apply_llm_debug()
        # Boot plus one call per settings save: a second must not stack another,
        # or every payload is printed once per save.
        apply_llm_debug()
        assert len(_switch_handlers(logger)) == 1

        monkeypatch.setenv(LLM_DEBUG_KEY, "")
        assert apply_llm_debug() is False
        # Ours gone (whichever of them the boot had attached), the level back to
        # NOTSET — inherited from the root logger, as it stood before the switch
        # existed — and propagation restored, so a process is never left chatty
        # (nor silent) after the flag is cleared.
        assert _switch_handlers(logger) == []
        assert logger.level == logging.NOTSET
        assert logger.propagate is True
        assert llmkit_client.debug_enabled is llm_debug_enabled
        assert llmkit_client.debug_enabled() is False
