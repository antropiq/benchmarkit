from typing import Any, Dict

import pytest
import streamlit as st


class _SessionStateDict(dict):
    """A dict subclass that also supports attribute-style access."""
    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


@pytest.fixture
def session_state() -> Dict[str, Any]:
    """Provide a mock st.session_state dict."""
    state: Dict[str, Any] = _SessionStateDict({
        "current_index": 0,
        "results": [],
        "questions": [],
        "endpoints_config": None,
        "total_running_time": 0.0,
    })
    # Patch st.session_state so modules see our dict
    original = st.session_state
    try:
        st.session_state = state
        yield state
    finally:
        st.session_state = original
