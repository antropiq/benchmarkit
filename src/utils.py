import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import streamlit as st

JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
LLMResult = Tuple[str, float, int]  # (response_text, took, tokens_used)
VerificationResult = Tuple[bool, str]  # (succeeded, response_text)


def load_json_from_file(file) -> JSONType:
    """Read a JSON file and return the dict."""
    try:
        return json.load(file)
    except Exception as exc:
        st.error(f"Could not parse JSON: {exc}")
        return {}


def format_time(seconds: float) -> str:
    """Return a string HH:MM:SS for given seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def save_results() -> None:
    """Save benchmark results to JSON file."""
    t: str = format_time(st.session_state["total_running_time"])
    output = {"results": st.session_state["results"], "totalRunningTime": t}
    output_path = Path.cwd() / "results.json"
    try:
        with open(output_path, "w") as fp:
            json.dump(output, fp, indent=2)
        st.download_button(
            label="📥 Download results.json",
            data=output_path.read_bytes(),
            file_name="results.json",
            mime="application/json",
        )
    except Exception as exc:
        st.error(f"⚠️ Could not write `{output_path}`: {exc}")
