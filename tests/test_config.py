from src.config import initialize_session_state


class TestInitializeSessionState:
    def test_sets_default_current_index(self, session_state):
        del session_state["current_index"]
        initialize_session_state()
        assert session_state["current_index"] == 0

    def test_sets_default_results(self, session_state):
        del session_state["results"]
        initialize_session_state()
        assert session_state["results"] == []

    def test_does_not_overwrite_existing_current_index(self, session_state):
        session_state["current_index"] = 42
        initialize_session_state()
        assert session_state["current_index"] == 42

    def test_does_not_overwrite_existing_results(self, session_state):
        session_state["results"] = [{"llm": "gpt-4"}]
        initialize_session_state()
        assert session_state["results"] == [{"llm": "gpt-4"}]
