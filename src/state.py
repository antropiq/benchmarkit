from typing import Dict, Tuple

import streamlit as st


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


def get_overall_stats() -> Tuple[int, int]:
    """Return ``(successes, answered)`` across all models and questions."""
    responses = [
        response
        for model_entry in st.session_state.get("results", [])
        for response in model_entry["responses"]
    ]
    successes = sum(1 for response in responses if response["succeeded"])
    return successes, len(responses)
