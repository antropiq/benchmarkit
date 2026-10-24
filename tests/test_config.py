import json
import os
from pathlib import Path

import pytest

from src.config import (
    LITELLM_MODEL_PREFIX,
    LITELLM_LOGGER_NAMES,
    PATH_LOCATION_KEY,
    ZERO_COST_MODEL_INFO,
    agent_instructions_block,
    apply_litellm_debug,
    litellm_debug_enabled,
    load_agent_instructions,
    normalize_env_flag,
    register_configured_models,
    register_unmapped_models,
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
        assert config is not None
        assert config.apiKey == "sk-target"
        assert config.verifierAgent is not None
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


class TestLiteLLMDebugFlag:
    """The LITELLM_DEBUG switch: how it is read, what it does to litellm."""

    def test_truthy_spellings_enable_it(self, monkeypatch):
        for raw in ("1", "true", "TRUE", " on "):
            monkeypatch.setenv("LITELLM_DEBUG", raw)
            assert litellm_debug_enabled() is True
        for raw in ("", "0", "false", "off", "yes please"):
            monkeypatch.setenv("LITELLM_DEBUG", raw)
            assert litellm_debug_enabled() is False

    def test_unset_means_off(self, monkeypatch):
        monkeypatch.delenv("LITELLM_DEBUG", raising=False)
        assert litellm_debug_enabled() is False

    def test_normalising_a_submitted_flag(self):
        assert normalize_env_flag("true") == "1"
        assert normalize_env_flag(" 1 ") == "1"
        assert normalize_env_flag("off") == ""
        assert normalize_env_flag(None) == ""

    def test_apply_on_raises_litellm_loggers_and_hides_its_hint(self, monkeypatch):
        import litellm

        calls = []
        monkeypatch.setattr(litellm, "_turn_on_debug", lambda: calls.append("on"))
        monkeypatch.setattr(litellm, "suppress_debug_info", False, raising=False)
        monkeypatch.setenv("LITELLM_DEBUG", "1")

        assert apply_litellm_debug() is True
        assert calls == ["on"]
        # litellm's own "use litellm._turn_on_debug()" banner would only repeat
        # advice this function already followed, and it buries tracebacks.
        assert litellm.suppress_debug_info is True

    def test_apply_off_restores_logger_levels(self, monkeypatch):
        import logging
        import litellm

        monkeypatch.setattr(litellm, "suppress_debug_info", True, raising=False)
        monkeypatch.delenv("LITELLM_DEBUG", raising=False)
        for name in LITELLM_LOGGER_NAMES:
            logging.getLogger(name).setLevel(logging.DEBUG)

        assert apply_litellm_debug() is False
        for name in LITELLM_LOGGER_NAMES:
            assert logging.getLogger(name).level == logging.NOTSET
        assert litellm.suppress_debug_info is False


class TestUnmappedModelCostRegistration:
    """Zero-cost litellm entries for model names it cannot price.

    Without them the per-call cost calculator raises ModelNotMappedError, and
    with LITELLM_DEBUG=1 that lands on the console as a full traceback per
    call — noise that buries the failure debug mode exists to reveal.
    """

    PREFIX = "zz-benchmarkit-unmapped"

    def fresh_name(self, request: pytest.FixtureRequest) -> str:
        """A model name no other test has registered yet.

        Registration mutates litellm's process-wide cost map, so a name shared
        between tests would be mapped for whichever test runs second and the
        guard would (correctly) skip it — making the assertions order-dependent.
        """
        return f"{self.PREFIX}-{request.node.name}"

    def test_unmapped_name_becomes_priceable_at_zero(self, request):
        import litellm

        name = self.fresh_name(request)
        assert register_unmapped_models([name]) == [name]
        info = litellm.get_model_info(
            model=f"openai/{name}", custom_llm_provider="openai"
        )
        assert info["input_cost_per_token"] == 0
        assert info["output_cost_per_token"] == 0
        cost = litellm.completion_cost(
            completion_response={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            },
            model=f"openai/{name}",
        )
        assert cost == 0.0

    def test_entry_is_registered_in_the_provider_qualified_form(self, request, capsys):
        """The key/value shape litellm's own cost map uses for an entry.

        Keying by the bare model name makes ``register_model`` resolve the
        provider from the key — with none in it, ``get_llm_provider`` fails and
        litellm prints its red "Provider List: https://docs.litellm.ai/…"
        banner to stdout on every boot (twice per model), and the stored entry
        ends up with ``mode: None`` / no provider-list membership.
        """
        import litellm

        name = self.fresh_name(request)
        register_unmapped_models([name])

        out = capsys.readouterr().out
        assert "Provider List" not in out
        assert f"{LITELLM_MODEL_PREFIX}{name}" in litellm.model_cost

        info = litellm.get_model_info(model=f"openai/{name}")
        assert info["litellm_provider"] == "openai"
        assert info["mode"] == "chat"
        assert (
            f"{LITELLM_MODEL_PREFIX}{name}" in litellm.open_ai_chat_completion_models
        )

    def test_template_carries_provider_and_mode(self):
        # The two non-price fields are what tell litellm the entry is an
        # OpenAI-compatible chat model; a template without them silently
        # regresses to the banner + mode-less entry above.
        assert ZERO_COST_MODEL_INFO["litellm_provider"] == "openai"
        assert ZERO_COST_MODEL_INFO["mode"] == "chat"
        assert LITELLM_MODEL_PREFIX == "openai/"

    def test_registered_prefix_matches_the_one_calls_use(self, request):
        """The entry must be found by the lookup ``completion()`` performs.

        ``core/llm.py`` / ``core/chat.py`` call litellm with
        ``model=f"openai/{ENDPOINT}"``, so a cost entry keyed any other way is
        dead weight — the cost calculator would keep raising.
        """
        import litellm

        name = self.fresh_name(request)
        register_unmapped_models([name])
        # No custom_llm_provider: the prefix alone has to route it, exactly as
        # inside a completion() call.
        cost = litellm.completion_cost(
            completion_response={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            },
            model=f"openai/{name}",
        )
        assert cost == 0.0

    def test_already_mapped_models_keep_their_real_pricing(self):
        # A model litellm knows must not be re-registered at zero cost: the
        # guard has to agree with the lookup the cost calculator performs.
        assert register_unmapped_models(["gpt-4o"]) == []

    def test_blank_names_are_skipped(self):
        assert register_unmapped_models(["", "   ", None]) == []

    def test_reregistration_is_a_no_op(self, request):
        # Passing the same name twice registers it once, and a second call over
        # an already-known name stays quiet — so a repeated settings save adds
        # nothing.
        name = self.fresh_name(request)
        assert register_unmapped_models([name, name]) == [name]
        assert register_unmapped_models([name]) == []

    def test_registration_failures_never_propagate(self, request, monkeypatch):
        # Log hygiene must not be able to break app startup.
        import litellm

        monkeypatch.setattr(litellm, "register_model", lambda *a, **k: 1 / 0)
        assert register_unmapped_models([self.fresh_name(request)]) == []

    def test_reads_the_configured_endpoints_from_the_env(self, request, monkeypatch):
        target = self.fresh_name(request)
        monkeypatch.setenv("ENDPOINT", f"{target}-a")
        monkeypatch.setenv("VERIFIER_AGENT_ENDPOINT", f"{target}-b")
        assert register_configured_models() == [f"{target}-a", f"{target}-b"]

    def test_unset_endpoints_register_nothing(self, monkeypatch):
        monkeypatch.delenv("ENDPOINT", raising=False)
        monkeypatch.delenv("VERIFIER_AGENT_ENDPOINT", raising=False)
        assert register_configured_models() == []
