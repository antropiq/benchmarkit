import json
from pathlib import Path
from typing import Tuple

import streamlit as st

from src.state import get_answered_time

LLMResult = Tuple[str, float, int]  # (response_text, took, tokens_used)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results.json"


def format_time(seconds: float) -> str:
    """Return a string HH:MM:SS for given seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_results() -> None:
    """Save benchmark results to JSON file."""
    total_running_time: float = get_answered_time()
    st.session_state["total_running_time"] = total_running_time
    output = {
        "results": st.session_state.get("results", []),
        "totalRunningTime": format_time(total_running_time),
    }
    data = json.dumps(output, indent=2)
    try:
        RESULTS_PATH.write_text(data, encoding="utf-8")
    except Exception as exc:
        st.error(f"Could not write `{RESULTS_PATH}`: {exc}")
        return
    st.download_button(
        label="Download results.json",
        icon=":material/download:",
        data=data,
        file_name="results.json",
        mime="application/json",
    )
