from typing import Dict, List

import streamlit as st

from src.llm import process_question
from src.models import EndpointsConfig
from src.state import get_session_data
from src.utils import save_results


def run_benchmark(
    questions: List[Dict],
    config: EndpointsConfig,
    total_tests: int,
) -> None:
    """Execute benchmark for all models and questions."""
    progress_bar = st.progress(0)
    score_progress = st.empty()
    st.session_state.current_index = 0

    for model in config.model_list():
        st.subheader(f"🔍 Model **{model}**")
        for idx, question in enumerate(questions, 1):
            question["index"] = idx  # Add index for display
            st.session_state.current_index += 1
            process_question(
                model,
                question,
                config,
                progress_bar,
                score_progress,
                total_tests
            )

    st.success("✅ Benchmark finished!")
    save_results()
