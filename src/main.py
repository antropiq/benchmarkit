import json
import sys
import time

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from pydantic import ValidationError

# Ensure project root is on sys.path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bench import run_benchmark
from src.config import initialize_session_state, setup_ui
from src.models import EndpointsConfig
from src.state import get_session_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = PROJECT_ROOT / "resources" / "questions.json"
ENDPOINTS_PATH = PROJECT_ROOT / "conf" / "endpoints.json"


def _load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file from disk, or return an empty dict on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load {path}: {exc}")
        return {}


def _load_config() -> None:
    """Auto-load endpoints and questions on page load."""
    if "config_loaded" in st.session_state:
        return

    endpoints_data = _load_json_file(ENDPOINTS_PATH)
    questions_data = _load_json_file(QUESTIONS_PATH)

    if not endpoints_data and not questions_data:
        st.error(f"Failed to load {ENDPOINTS_PATH} and {QUESTIONS_PATH}")
        return

    if not endpoints_data:
        st.error(f"Failed to load {ENDPOINTS_PATH}")
    if not questions_data:
        st.error(f"Failed to load {QUESTIONS_PATH}")

    st.session_state["config_loaded"] = True
    st.session_state["questions"] = questions_data.get("questions", [])
    _apply_endpoints_config(endpoints_data)

    _update_total_tests()


def _apply_endpoints_config(endpoints_data: Dict[str, Any]) -> None:
    """Validate the endpoints config and store it in session state."""
    try:
        st.session_state["endpoints_config"] = EndpointsConfig.model_validate(
            endpoints_data
        )
    except ValidationError as exc:
        st.error(f"⚠️ Invalid endpoints configuration: {exc}")


def _update_total_tests() -> None:
    """Recalculate total_tests from current questions and endpoints."""
    questions: List[Dict] = st.session_state.get("questions", [])
    config: Optional[EndpointsConfig] = st.session_state.get("endpoints_config")
    endpoints: List[str] = config.model_list() if config else []
    st.session_state["total_tests"] = len(questions) * len(endpoints)


def main() -> None:
    setup_ui()
    initialize_session_state()

    # Auto-load from disk
    _load_config()

    # File upload section (overrides auto-loaded config)
    col1, col2 = st.columns([1, 1])
    with col1:
        questions_file = st.file_uploader(
            "Upload *resources/questions.json*", type="json", key="questions_file"
        )
    with col2:
        endpoints_file = st.file_uploader(
            "Upload *conf/endpoints.json*", type="json", key="endpoints_file"
        )

    # Handle file uploads (override auto-loaded config)
    if questions_file:
        questions_data: Dict[str, Any] = json.load(questions_file)
        st.session_state["questions"] = questions_data.get("questions", [])
        _update_total_tests()

    if endpoints_file:
        endpoints_data: Dict[str, Any] = json.load(endpoints_file)
        _apply_endpoints_config(endpoints_data)
        _update_total_tests()

    questions = st.session_state.get("questions", [])
    config: Optional[EndpointsConfig] = st.session_state.get("endpoints_config")
    endpoints: List[str] = config.model_list() if config else []
    server_url: str = config.serverUrl if config else ""
    sampling: Dict[str, Any] = config.to_kwargs() if config else {}
    total = st.session_state.get("total_tests", 0)

    sampling_summary = (
        ", ".join(f"{k}={v}" for k, v in sampling.items()) or "defaults"
    )
    st.markdown(
        f"- **Questions:** {len(questions)}  "
        f"**- Models:** {len(endpoints)}  "
        f"**- Total tests:** {total}  "
        f"**- Server:** `{server_url}`"
    )
    st.markdown(f"- **Sampling:** `{sampling_summary}`")

    # Start benchmark button
    if st.button("🏁 Start Benchmark"):
        questions, config = get_session_data()
        st.session_state["start_time"] = time.time()
        st.session_state["total_running_time"] = 0
        run_benchmark(
            questions,
            config,
            st.session_state["total_tests"],
        )
        duration = time.time() - st.session_state["start_time"]
        st.session_state["total_running_time"] = duration


if __name__ == "__main__":
    main()
