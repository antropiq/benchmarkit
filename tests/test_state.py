from src.models import EndpointsConfig
from src.state import add_result_to_state, get_session_data


class TestAddResultToState:
    def test_first_result_creates_model_entry(self, session_state):
        add_result_to_state(
            model="gpt-4",
            question_uuid="q1",
            succeeded=True,
            response_text="hello",
            took=1.5,
            tokens=100,
        )
        assert len(session_state["results"]) == 1
        entry = session_state["results"][0]
        assert entry["llm"] == "gpt-4"
        assert entry["score"] == "1/1"
        assert len(entry["responses"]) == 1
        resp = entry["responses"][0]
        assert resp["uuid"] == "q1"
        assert resp["response"] == "hello"
        assert resp["took"] == 1.5
        assert resp["consumedTokens"] == 100
        assert resp["succeeded"] is True

    def test_second_result_updates_score(self, session_state):
        add_result_to_state("gpt-4", "q1", True, "a", 1.0, 50)
        add_result_to_state("gpt-4", "q2", False, "b", 2.0, 60)
        entry = session_state["results"][0]
        assert entry["score"] == "1/2"
        assert len(entry["responses"]) == 2

    def test_different_models_are_separate(self, session_state):
        add_result_to_state("gpt-4", "q1", True, "a", 1.0, 50)
        add_result_to_state("llama", "q1", False, "b", 2.0, 60)
        assert len(session_state["results"]) == 2
        gpt = next(e for e in session_state["results"] if e["llm"] == "gpt-4")
        llama = next(e for e in session_state["results"] if e["llm"] == "llama")
        assert gpt["score"] == "1/1"
        assert llama["score"] == "0/1"


class TestGetSessionData:
    def test_returns_expected_tuple(self, session_state):
        session_state["questions"] = [{"uuid": "q1", "question": "?", "expected": "a"}]
        config = EndpointsConfig.model_validate({
            "serverUrl": "http://localhost:8080",
            "endpoint": "model-a",
            "temp": "1.0",
            "verifierAgent": {"serverUrl": "http://verifier", "endpoint": "v"},
        })
        session_state["endpoints_config"] = config

        questions, returned = get_session_data()
        assert questions == session_state["questions"]
        assert returned is config
        assert returned.serverUrl == "http://localhost:8080"
        assert returned.model_list() == ["model-a"]
        assert returned.to_kwargs() == {"temperature": 1.0}
        assert returned.verifierAgent.serverUrl == "http://verifier"

    def test_missing_config_defaults_to_none(self, session_state):
        session_state.pop("endpoints_config")
        (_, config) = get_session_data()
        assert config is None
