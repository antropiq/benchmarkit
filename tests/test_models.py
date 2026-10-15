import pytest
from pydantic import ValidationError

from src.models import EndpointsConfig, SamplingParams, VerifierAgent


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

    def test_single_endpoint_preferred_over_legacy_list(self):
        cfg = EndpointsConfig.model_validate({
            "endpoint": "new",
            "endpoints": ["old-a", "old-b"],
        })
        assert cfg.model_list() == ["new"]

    def test_legacy_endpoints_list(self):
        cfg = EndpointsConfig.model_validate({"endpoints": ["a", "b"]})
        assert cfg.model_list() == ["a", "b"]

    def test_legacy_endpoints_string(self):
        cfg = EndpointsConfig.model_validate({"endpoints": "a"})
        assert cfg.model_list() == ["a"]

    def test_blank_endpoint_ignored(self):
        cfg = EndpointsConfig.model_validate({"endpoint": "  "})
        assert cfg.endpoint is None
        assert cfg.model_list() == []

    def test_blank_entries_stripped_from_list(self):
        cfg = EndpointsConfig.model_validate({"endpoints": ["a", " ", "b"]})
        assert cfg.model_list() == ["a", "b"]

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
    def test_full_env(self):
        cfg = EndpointsConfig.from_env({
            "SERVER_URL": "http://host:8080/v1",
            "ENDPOINT": "model-a",
            "TEMP": "0.2",
            "TOP_P": "0.95",
            "TOP_K": "20",
            "MIN_P": "0",
            "REPEAT_PENALTY": "1.0",
            "PRESENCE_PENALTY": "0",
            "REASONING": "low",
            "VERIFIER_AGENT_SYSPROMPT": "verify",
            "VERIFIER_AGENT_SERVERURL": "http://verif:8080/v1",
            "VERIFIER_AGENT_ENDPOINT": "strata",
        })
        assert cfg.serverUrl == "http://host:8080/v1"
        assert cfg.model_list() == ["model-a"]
        assert cfg.temperature == 0.2
        assert cfg.top_k == 20
        assert cfg.reasoning_effort == "low"
        assert cfg.verifierAgent.endpoint == "strata"
        assert cfg.verifierAgent.serverUrl == "http://verif:8080/v1"
        assert cfg.verifierAgent.sysprompt == "verify"

    def test_blank_and_missing_keys(self):
        cfg = EndpointsConfig.from_env({"ENDPOINT": "  ", "SERVER_URL": ""})
        assert cfg.model_list() == []
        assert cfg.serverUrl == ""
        assert cfg.verifierAgent is None

    def test_env_endpoints_list(self):
        cfg = EndpointsConfig.from_env({"ENDPOINTS": "a, b ,c"})
        assert cfg.model_list() == ["a", "b", "c"]

    def test_endpoint_takes_precedence_over_endpoints(self):
        cfg = EndpointsConfig.from_env({"ENDPOINT": "x", "ENDPOINTS": "a,b"})
        assert cfg.model_list() == ["x"]

    def test_invalid_reasoning_raises(self):
        with pytest.raises(ValidationError):
            EndpointsConfig.from_env({"REASONING": "sideways"})
