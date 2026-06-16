import json
from io import StringIO
from typing import Dict, Any

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
        "server_url": "",
        "endpoints": [],
        "verifier_agent": {},
        "total_running_time": 0.0,
    })
    # Patch st.session_state so modules see our dict
    original = st.session_state
    try:
        st.session_state = state
        yield state
    finally:
        st.session_state = original


@pytest.fixture
def sample_questions_json() -> str:
    """Return a JSON string with three sample questions."""
    return json.dumps({
        "questions": [
            {"uuid": "00000001-0001-0001-0001-000000000001", "question": "What is 2+2?", "expected": "4"},
            {"uuid": "00000001-0001-0001-0001-000000000002", "question": "Capital of France?", "expected": "Paris"},
            {"uuid": "00000001-0001-0001-0001-000000000003", "question": "Is water wet?", "expected": "yes"},
        ]
    })


@pytest.fixture
def sample_questions_file(sample_questions_json: str) -> StringIO:
    """Return a file-like object containing sample questions JSON."""
    return StringIO(sample_questions_json)
