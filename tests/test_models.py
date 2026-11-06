import pytest
from pydantic import ValidationError

from src.core.models import (
    DEFAULT_LLM_SYSPROMPT,
    EndpointsConfig,
    SamplingParams,
    VerifierAgent,
)


class TestVerifierAgent:
    def test_parses_full_block(self):
        v = VerifierAgent.model_validate({
            "sysprompt": "be strict",
            "serverUrl": "http://verifier:8080/v1",
            "endpoint": "strata",
        })
        assert v.sysprompt == "be strict"
        assert v.serverUrl == "http://verifier:8080/v1"
        assert v.endpoint == "strata"

    def test_defaults_are_empty_strings(self):
        v = VerifierAgent.model_validate({})
        assert v.sysprompt == ""
        assert v.serverUrl == ""
        assert v.endpoint == ""
        assert v.apiKey == ""


class TestSamplingParams:
    def test_full_config_string_values_coerced(self):
        p = SamplingParams.model_validate({
            "temp": "1.0",
            "top-p": "0.95",
            "top-k": "20",
            "min-p": "0",
            "repeat-penalty": "1.0",
            "presence-penalty": "0",
        })
        assert p.to_kwargs() == {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        }

    def test_numeric_values_pass_through(self):
        p = SamplingParams.model_validate({"temp": 0.7, "top-k": 40})
        assert p.to_kwargs() == {"temperature": 0.7, "top_k": 40}

    def test_missing_keys_excluded(self):
        assert SamplingParams.model_validate({}).to_kwargs() == {}
        assert SamplingParams.model_validate({"temp": "0.5"}).to_kwargs() == {
            "temperature": 0.5
        }

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            SamplingParams.model_validate({"top-p": "O.95"})

    def test_top_k_rejects_non_integer(self):
        with pytest.raises(ValidationError):
            SamplingParams.model_validate({"top-k": "abc"})

    def test_reasoning_mapped_and_normalized(self):
        p = SamplingParams.model_validate({"reasoning": "Medium"})
        assert p.to_kwargs() == {"reasoning_effort": "medium"}

    def test_reasoning_off_and_xhigh_valid(self):
        assert SamplingParams.model_validate(
            {"reasoning": "off"}
        ).to_kwargs() == {"reasoning_effort": "off"}
        assert SamplingParams.model_validate(
            {"reasoning": "xhigh"}
        ).to_kwargs() == {"reasoning_effort": "xhigh"}

    def test_invalid_reasoning_raises(self):
        with pytest.raises(ValidationError):
            SamplingParams.model_validate({"reasoning": "ultra"})

    def test_empty_reasoning_treated_as_absent(self):
        assert SamplingParams.model_validate({"reasoning": "  "}).to_kwargs() == {}

    def test_canonical_names_accepted(self):
        p = SamplingParams.model_validate({"temperature": 0.3})
        assert p.to_kwargs() == {"temperature": 0.3}


class TestEndpointsConfig:
    def test_parses_real_world_shape(self):
        cfg = EndpointsConfig.model_validate({
            "serverUrl": "http://127.0.0.1:8080/v1",
            "endpoint": "qwen3.8-flash-next-iq3_xxs",
            "temp": "0.2",
            "top-p": "0.95",
            "top-k": "20",
            "min-p": "0",
            "repeat-penalty": "1.0",
            "presence-penalty": "0",
            "reasoning": "low",
            "verifierAgent": {
                "sysprompt": "analyse",
                "serverUrl": "http://192.168.1.4:8080/v1",
                "endpoint": "strata",
            },
        })
        assert cfg.serverUrl == "http://127.0.0.1:8080/v1"
        assert cfg.model_list() == ["qwen3.8-flash-next-iq3_xxs"]
        assert cfg.to_kwargs() == {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "reasoning_effort": "low",
        }
        # The validated document carries a verifierAgent entry.
        assert cfg.verifierAgent is not None
        assert cfg.verifierAgent.endpoint == "strata"
        assert cfg.verifierAgent.serverUrl == "http://192.168.1.4:8080/v1"

    def test_to_kwargs_excludes_config_fields(self):
        cfg = EndpointsConfig.model_validate({
            "serverUrl": "http://x",
            "endpoint": "m",
            "verifierAgent": {"endpoint": "v"},
        })
        assert cfg.to_kwargs() == {}

    def test_single_endpoint_string(self):
        cfg = EndpointsConfig.model_validate({"endpoint": "my-model"})
        assert cfg.model_list() == ["my-model"]

    def test_blank_endpoint_ignored(self):
        cfg = EndpointsConfig.model_validate({"endpoint": "  "})
        assert cfg.endpoint is None
        assert cfg.model_list() == []

    def test_empty_config(self):
        cfg = EndpointsConfig.model_validate({})
        assert cfg.serverUrl == ""
        assert cfg.model_list() == []
        assert cfg.verifierAgent is None

    def test_unknown_keys_ignored(self):
        cfg = EndpointsConfig.model_validate({"serverUrl": "u", "bogus": 1})
        assert cfg.serverUrl == "u"

    def test_invalid_sampling_raises(self):
        with pytest.raises(ValidationError):
            EndpointsConfig.model_validate({"temp": "hot"})


class TestFromEnv:
    """``from_env`` after group 1 moved to the workspace file: ``.env``'s half only.

    The class is the D2 guard as much as it is a builder test — a group-1 name
    set here must make **no** difference to the config, because the fallback that
    would have honoured it was rejected on purpose (spec D2). So the cases that
    used to feed ``SERVER_URL`` / ``TEMPERATURE`` / ``SUPPORTS_VISION`` through
    the env are gone rather than translated: the file's own builder is
    :class:`TestFromWorkspace`, and the pair belongs in this one file.
    """

    def test_the_verifier_block_is_read_from_the_env(self):
        cfg = EndpointsConfig.from_env({
            "SERVER_URL": "http://host:8080/v1",  # group 1: read by nothing
            "ENDPOINT": "model-a",                # group 1: read by nothing
            "TEMPERATURE": "0.2",                 # group 1: read by nothing
            "VERIFIER_AGENT_SYSPROMPT": "verify",
            "VERIFIER_AGENT_SERVERURL": "http://verif:8080/v1",
            "VERIFIER_AGENT_SERVERURL_API_KEY": "sk-judge",
            "VERIFIER_AGENT_ENDPOINT": "strata",
        })
        assert cfg.verifierAgent is not None
        assert cfg.verifierAgent.endpoint == "strata"
        assert cfg.verifierAgent.serverUrl == "http://verif:8080/v1"
        assert cfg.verifierAgent.sysprompt == "verify"
        assert cfg.verifierAgent.apiKey == "sk-judge"
        # …and the group-1 names above left every group-1 field at its default.
        assert cfg.serverUrl == ""
        assert cfg.model_list() == []
        assert cfg.temperature is None

    def test_blank_and_missing_keys(self):
        cfg = EndpointsConfig.from_env({"ENDPOINT": "  ", "SERVER_URL": ""})
        assert cfg.model_list() == []
        assert cfg.serverUrl == ""
        assert cfg.verifierAgent is None

    def test_verifier_created_from_key_only(self):
        cfg = EndpointsConfig.from_env({"VERIFIER_AGENT_SERVERURL_API_KEY": "sk"})
        assert cfg.verifierAgent is not None
        assert cfg.verifierAgent.apiKey == "sk"

    def test_the_group_one_names_it_does_not_read(self):
        """The one sentence D2 is about, pinned as a fact about this builder."""
        cfg = EndpointsConfig.from_env({
            "SERVER_URL": "http://left-over:9/v1",
            "SERVER_URL_API_KEY": "sk-left-over",
            "ENDPOINT": "left-over-model",
            "LLM_SYSPROMPT": "left-over prompt",
            "TEMPERATURE": "0.9",
            "TOP_P": "0.5",
            "TOP_K": "5",
            "MIN_P": "0.1",
            "REPEAT_PENALTY": "1.2",
            "PRESENCE_PENALTY": "0.3",
            "REASONING": "sideways",  # would raise if this builder read it
            "SUPPORTS_VISION": "1",
            "LSP_ENABLED": "pyright",
            "MAX_TOOL_ITERATION": "9",
        })
        assert cfg.serverUrl == ""
        assert cfg.apiKey == ""
        assert cfg.endpoint is None
        assert cfg.model_list() == []
        assert cfg.sysprompt == DEFAULT_LLM_SYSPROMPT
        assert cfg.to_kwargs() == {}
        assert cfg.supports_vision is False


class TestFromWorkspace:
    """``from_workspace``: group 1 from the JSON mapping, verifier from the env."""

    def test_full_document(self):
        cfg = EndpointsConfig.from_workspace(
            {
                "serverUrl": "http://host:8080/v1",
                "serverApiKey": "sk-target",
                "endpoint": "model-a",
                "sysprompt": "Answer briefly.",
                "supportsVision": True,
                "temperature": 0.2,
                "topP": 0.95,
                "topK": 20,
                "minP": 0,
                "repeatPenalty": 1.0,
                "presencePenalty": 0,
                "reasoning": "low",
                "lspEnabled": ["pyright"],
                "maxToolIteration": 300,
            },
            {
                "VERIFIER_AGENT_SYSPROMPT": "verify",
                "VERIFIER_AGENT_SERVERURL": "http://verif:8080/v1",
                "VERIFIER_AGENT_ENDPOINT": "strata",
            },
        )
        assert cfg.serverUrl == "http://host:8080/v1"
        assert cfg.apiKey == "sk-target"
        assert cfg.model_list() == ["model-a"]
        assert cfg.sysprompt == "Answer briefly."
        assert cfg.supports_vision is True
        assert cfg.to_kwargs() == {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "reasoning_effort": "low",
        }
        assert cfg.verifierAgent is not None
        assert cfg.verifierAgent.endpoint == "strata"
        assert cfg.verifierAgent.serverUrl == "http://verif:8080/v1"
        assert cfg.verifierAgent.sysprompt == "verify"

    def test_native_types_pass_straight_through(self):
        """D6: ``True`` / ``0.7`` / ``40`` need no coercion, and no string parser."""
        cfg = EndpointsConfig.from_workspace(
            {"supportsVision": True, "temperature": 0.7, "topK": 40}
        )
        assert cfg.supports_vision is True
        assert cfg.temperature == 0.7
        assert cfg.top_k == 40

    def test_an_absent_sampler_member_is_omitted_from_the_request_body(self):
        cfg = EndpointsConfig.from_workspace({"serverUrl": "http://x/v1"})
        assert cfg.temperature is None
        assert cfg.top_p is None
        assert cfg.top_k is None
        assert cfg.min_p is None
        assert cfg.repeat_penalty is None
        assert cfg.presence_penalty is None
        assert cfg.reasoning_effort is None
        assert cfg.to_kwargs() == {}

    def test_absent_members_mean_the_documented_default(self):
        # The verifier half comes from the *env* mapping, so it is passed empty
        # here: the ambient environment is the developer's own and group 2 stays
        # in ``.env``, which is a fact about the machine, not about this builder.
        cfg = EndpointsConfig.from_workspace({}, {})
        assert cfg.serverUrl == ""
        assert cfg.apiKey == ""
        assert cfg.endpoint is None
        assert cfg.model_list() == []
        assert cfg.sysprompt == DEFAULT_LLM_SYSPROMPT
        assert cfg.supports_vision is False
        assert cfg.verifierAgent is None

    def test_a_blank_string_member_is_absent_not_a_value(self):
        cfg = EndpointsConfig.from_workspace(
            {"endpoint": "  ", "sysprompt": "   ", "serverUrl": "  http://x/v1  "}
        )
        assert cfg.endpoint is None
        assert cfg.sysprompt == DEFAULT_LLM_SYSPROMPT
        assert cfg.serverUrl == "http://x/v1"

    def test_an_empty_mapping_is_valid_and_says_no_server(self):
        """How an absent or malformed file reaches a caller: nothing to dial.

        The refusal is the reader's job (D8/D9), so the builder neither raises
        nor invents — a workspace with no file is a config with no server.
        """
        cfg = EndpointsConfig.from_workspace({})
        assert cfg.serverUrl == ""
        assert cfg.model_list() == []

    def test_a_bad_reasoning_still_raises(self):
        with pytest.raises(ValidationError):
            EndpointsConfig.from_workspace({"reasoning": "sideways"})

    def test_the_verifier_half_comes_from_the_env_mapping(self):
        cfg = EndpointsConfig.from_workspace(
            {"serverUrl": "http://x/v1"},
            {"VERIFIER_AGENT_SERVERURL_API_KEY": "sk-judge"},
        )
        assert cfg.verifierAgent is not None
        assert cfg.verifierAgent.apiKey == "sk-judge"
        # The model connection is unaffected by the env mapping's verifier keys.
        assert cfg.apiKey == ""
