import streamlit as st


def initialize_session_state() -> None:
    """Initialize required session state variables."""
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "results" not in st.session_state:
        st.session_state.results = []


def setup_ui() -> None:
    """Configure Streamlit UI components."""
    st.set_page_config(page_title="LLM Benchmark UI", layout="wide")
    st.title("🔬 LLM Benchmark")
    st.markdown(
        """
        Provide two JSON files:
        * **resources/questions.json** – list of questions (must contain `uuid`, `question` & `expected`)
        * **conf/endpoints.json** – list of LLM endpoints + verifier settings

        Click **Start Benchmark** to run tests with real-time streaming.
        """
    )
