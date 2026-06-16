import json
import sys
import time

from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bench import run_benchmark
from src.config import initialize_session_state, setup_ui
from src.state import get_session_data
from src.utils import load_json_from_file


def main() -> None:
    setup_ui()
    initialize_session_state()

    # File upload section
    col1, col2 = st.columns([1, 1])
    with col1:
        questions_file = st.file_uploader(
            "Upload *resources/questions.json*", type="json", key="questions_file"
        )
    with col2:
        endpoints_file = st.file_uploader(
            "Upload *conf/endpoints.json*", type="json", key="endpoints_file"
        )

    # Handle file uploads
    if questions_file and endpoints_file:
        questions_data = load_json_from_file(questions_file)
        endpoints_data = load_json_from_file(endpoints_file)

        # Ensure we have dictionaries
        if not isinstance(questions_data, dict):
            questions_data = {}
        if not isinstance(endpoints_data, dict):
            endpoints_data = {}

        st.session_state["questions"] = questions_data.get("questions", [])
        st.session_state["server_url"] = endpoints_data.get("serverUrl", "")
        st.session_state["endpoints"] = endpoints_data.get("endpoints", [])
        st.session_state["verifier_agent"] = endpoints_data.get(
            "verifierAgent", {}
        )

        # Calculate total tests
        questions = st.session_state["questions"]
        endpoints = st.session_state["endpoints"]
        st.session_state["total_tests"] = len(questions) * len(endpoints)

    # Start benchmark button
    if st.button("🏁 Start Benchmark") and questions_file and endpoints_file:
        questions, server_url, endpoints, verifier = get_session_data()
        st.session_state["start_time"] = time.time()
        st.session_state["total_running_time"] = 0
        run_benchmark(
            questions,
            server_url,
            endpoints,
            verifier,
            st.session_state["total_tests"],
        )
        duration = time.time() - st.session_state["start_time"]
        st.session_state["total_running_time"] = duration


if __name__ == "__main__":
    main()
