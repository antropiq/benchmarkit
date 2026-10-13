from typing import Dict, List, Optional, Tuple

import streamlit as st

from src.models import EndpointsConfig


def add_result_to_state(
    model: str,
    question_uuid: str,
    succeeded: bool,
    response_text: str,
    took: float,
    tokens: int,
) -> None:
    """Store per-question result in SessionState."""
    if "results" not in st.session_state:
        st.session_state.results = []

    model_entry = next(
        (e for e in st.session_state.results if e["llm"] == model),
        None
    )
    if not model_entry:
        model_entry = {"llm": model, "score": "", "responses": []}
        st.session_state.results.append(model_entry)

    model_entry["responses"].append({
        "uuid": question_uuid,
        "response": response_text,
        "took": took,
        "consumedTokens": tokens,
        "succeeded": succeeded,
    })

    # Update model score
    succeeded_count = sum(r["succeeded"] for r in model_entry["responses"])
    model_entry["score"] = f"{succeeded_count}/{len(model_entry['responses'])}"


def get_session_data() -> Tuple[List[Dict], Optional[EndpointsConfig]]:
    """Retrieve benchmark configuration from session state."""
    return (
        st.session_state["questions"],
        st.session_state.get("endpoints_config"),
    )
