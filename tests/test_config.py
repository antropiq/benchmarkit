import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from src.core.llmkit import client as llmkit_client
from src.core.llmkit import constants as llmkit_constants
from src.config import (
    DEFAULT_MAX_TOOL_ITERATION,
    LLM_DEBUG_KEY,
    LEGACY_LLM_DEBUG_KEY,
    PATH_LOCATION_KEY,
    _DebugStreamHandler,
    agent_instructions_block,
    apply_llm_debug,
    llm_debug_enabled,
    lsp_enabled_ids,
    load_agent_instructions,
    max_tool_iteration,
    default_path_location,
    load_endpoints_config,
    load_questions,
    path_location,
    setup_path_location,
    workspace_settings_refusal,
)
from src.core import workspace_settings
from src.core.models import DEFAULT_LLM_SYSPROMPT
from tests.conftest import neutralise_workspace_settings, write_workspace_settings


class TestLoadEndpointsConfig:
    """The seam where the two stores meet: group 1 from the file, verifier from ``.env``.

    Every case here points the workspace settings lookup at its own ``tmp_path``
    (the ``conftest`` helper) and writes its own file, because the alternative —
    an ambient answer — would make the assertions about this machine's model
    connection rather than about the function.
    """

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ambient group-2 keys: the verifier half comes from the env."""
        for key in (
            "VERIFIER_AGENT_SYSPROMPT",
            "VERIFIER_AGENT_SERVERURL",
            "VERIFIER_AGENT_SERVERURL_API_KEY",
            "VERIFIER_AGENT_ENDPOINT",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_reads_group_one_from_the_file_and_the_verifier_from_the_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """One test, both halves: this is the whole point of the split."""
        neutralise_workspace_settings(
            monkeypatch,
            tmp_path,
            serverUrl="http://localhost:8080/v1",
            serverApiKey="sk-target",
            endpoint="model-a",
            sysprompt="Answer briefly.",
            supportsVision=True,
            temperature=0.7,
            reasoning="low",
        )
        monkeypatch.setenv("VERIFIER_AGENT_ENDPOINT", "judge")
        monkeypatch.setenv("VERIFIER_AGENT_SERVERURL_API_KEY", "sk-judge")

        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.serverUrl == "http://localhost:8080/v1"
        assert config.apiKey == "sk-target"
        assert config.model_list() == ["model-a"]
        assert config.sysprompt == "Answer briefly."
        assert config.supports_vision is True
        assert config.temperature == 0.7
        assert config.reasoning_effort == "low"
        assert config.verifierAgent is not None
        assert config.verifierAgent.endpoint == "judge"
        assert config.verifierAgent.apiKey == "sk-judge"

    def test_a_group_one_value_only_in_the_env_is_not_picked_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """D2: a ``.env`` line left behind is dead bytes — no fallback."""
        neutralise_workspace_settings(
            monkeypatch, tmp_path, serverUrl="", serverApiKey="", endpoint=""
        )
        monkeypatch.setenv("SERVER_URL", "http://left-over:9/v1")
        monkeypatch.setenv("SERVER_URL_API_KEY", "sk-left-over")
        monkeypatch.setenv("ENDPOINT", "left-over-model")
        monkeypatch.setenv("TEMPERATURE", "0.9")

        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.serverUrl == ""
        assert config.apiKey == ""
        assert config.model_list() == []
        # The file's own 0.2 answers, never the env's 0.9.
        assert config.temperature == 0.2

    def test_an_absent_temperature_omits_the_key_from_the_request_body(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        neutralise_workspace_settings(monkeypatch, tmp_path)
        path = write_workspace_settings(tmp_path)
        document: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        assert "temperature" in document  # the template holds one
        document.pop("temperature")
        path.write_text(json.dumps(document), encoding="utf-8")
        workspace_settings.invalidate_workspace_settings()

        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.temperature is None
        assert "temperature" not in config.to_kwargs()

    def test_the_native_vision_flag_is_true(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """D6, the native path: a JSON ``true`` reaches pydantic as ``True``.

        The other half — a string ``"true"`` in the file is a **failed** read,
        not a truthy one — is the reader's own rule and is asserted in
        ``tests/test_workspace_settings.py``.
        """
        neutralise_workspace_settings(monkeypatch, tmp_path, supportsVision=True)
        config, error = load_endpoints_config()
        assert error is None
        assert config is not None
        assert config.supports_vision is True

    def test_an_unusable_file_is_a_sentence_naming_the_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """D9 (malformed) and D8 (still the seed text) both refuse, never default.

        A caller that benchmarked on defaults would measure the defaults, and one
        that dialled the template's sentence would report a transport error
        quoting prose — so both states come back as ``(None, sentence)``.
        """
        neutralise_workspace_settings(monkeypatch, tmp_path, absent=True)
        path = (
            tmp_path
            / workspace_settings.POLICY_DIRNAME
            / workspace_settings.WORKSPACE_SETTINGS_FILENAME
        )

        # absent — the reader writes nothing, and the refusal names where it looked
        config, error = load_endpoints_config()
        assert config is None
        assert isinstance(error, str) and str(path) in error

        # malformed (D9)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{ "serverUrl": "http://x/v1", ', encoding="utf-8")
        workspace_settings.invalidate_workspace_settings()
        config, error = load_endpoints_config()
        assert config is None
        assert isinstance(error, str) and str(path) in error

        # a seeded file is valid JSON and unusable (D8): reported, not dialled
        path.unlink()
        workspace_settings.ensure_workspace_settings_file(str(tmp_path))
        config, error = load_endpoints_config()
        assert config is None
        assert isinstance(error, str) and str(path) in error

    def test_the_refusal_sentence_is_built_in_one_place(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """``load_endpoints_config`` and ``workspace_settings_refusal`` agree.

        ws-04's ``GET /api/settings`` reuses that function; if the two sentences
        could differ, the dialog and the refusal would disagree about the same
        file.
        """
        neutralise_workspace_settings(monkeypatch, tmp_path)
        assert workspace_settings_refusal() is None
        assert load_endpoints_config()[1] is None

        path = write_workspace_settings(tmp_path)
        path.write_text("{ nope", encoding="utf-8")  # not JSON at all
        workspace_settings.invalidate_workspace_settings()
        sentence = workspace_settings_refusal()
        assert sentence
        assert load_endpoints_config()[1] == sentence


class TestWorkspaceSourcedConfigHelpers:
    """``lsp_enabled_ids()`` / ``max_tool_iteration()``: same names, new source."""

    def test_lsp_enabled_deduplicates_and_keeps_the_files_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        neutralise_workspace_settings(
            monkeypatch, tmp_path, lspEnabled=["vue", "pyright", "vue", "typescript"]
        )
        assert lsp_enabled_ids() == ["vue", "pyright", "typescript"]

    def test_lsp_enabled_is_empty_without_a_file_or_an_env_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        neutralise_workspace_settings(monkeypatch, tmp_path)
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        assert lsp_enabled_ids() == []

    def test_the_files_value_wins_over_the_code_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        neutralise_workspace_settings(monkeypatch, tmp_path, maxToolIteration=9)
        assert max_tool_iteration() == 9

    def test_absent_member_falls_back_to_the_code_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The member being *absent* is what defaults; the template holds 300."""
        path = write_workspace_settings(tmp_path)
        neutralise_workspace_settings(monkeypatch, tmp_path)
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["maxToolIteration"] == 300
        assert max_tool_iteration() == 300
        document.pop("maxToolIteration")
        path.write_text(json.dumps(document), encoding="utf-8")
        workspace_settings.invalidate_workspace_settings()
        assert max_tool_iteration() == DEFAULT_MAX_TOOL_ITERATION

    def test_zero_or_negative_clamps_to_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        for value in (0, -5):
            neutralise_workspace_settings(
                monkeypatch, tmp_path, maxToolIteration=value
            )
            assert max_tool_iteration() == 1

    def test_the_env_names_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """``MAX_TOOL_ITERATION`` / ``MAX_MCP_ITERATION`` reach nothing.

        The first left ``.env`` with the rest of group 1 (D2: no fallback); the
        second was the hand-written-``.env`` escape hatch for a file that no
        longer holds the key, so it stopped being honoured (spec section 5).
        """
        neutralise_workspace_settings(monkeypatch, tmp_path, maxToolIteration=4)
        monkeypatch.setenv("MAX_TOOL_ITERATION", "7")
        monkeypatch.setenv("MAX_MCP_ITERATION", "7")
        assert max_tool_iteration() == 4


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
