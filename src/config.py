import streamlit as st


def initialize_session_state() -> None:
    """Initialize required session state variables."""
    defaults = {
        "current_index": 0,
        "results": [],
        "total_running_time": 0.0,
        "benchmark_started_at": None,
        "benchmark_finished_at": None,
        "benchmark_running": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def setup_ui() -> None:
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="LLM benchmark",
        page_icon=":material/science:",
        layout="wide",
    )
