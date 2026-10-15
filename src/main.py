import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

# Ensure project root is on sys.path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bench import BenchmarkUi, run_benchmark, update_stats
from src.config import initialize_session_state, setup_ui
from src.models import EndpointsConfig
from src.ui import render_layout_style, render_chronometer, render_stream_box_style

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = PROJECT_ROOT / "resources" / "questions.json"
ENV_PATH = PROJECT_ROOT / ".env"

STREAM_HEIGHT_PX = 448


def _load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file from disk, or return an empty dict on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load {path}: {exc}")
        return {}


def _load_config() -> None:
    """Auto-load the .env configuration and questions on page load."""
    if "config_loaded" in st.session_state:
        return

    if ENV_PATH.exists():
        # Existing environment variables take precedence over the file.
        load_dotenv(ENV_PATH)
    else:
        st.warning(f"No `{ENV_PATH.name}` found — using environment variables only.")

    questions_data = _load_json_file(QUESTIONS_PATH)
    if not questions_data:
        st.error(f"Failed to load {QUESTIONS_PATH}")

    questions: List[Dict] = questions_data.get("questions", [])
    st.session_state["config_loaded"] = True
    st.session_state["questions"] = questions
    _apply_env_config()

    config: Optional[EndpointsConfig] = st.session_state.get("endpoints_config")
    endpoints: List[str] = config.model_list() if config else []
    st.session_state["total_tests"] = len(questions) * len(endpoints)


def _apply_env_config() -> None:
    """Build and validate the endpoints config from the environment."""
    try:
        st.session_state["endpoints_config"] = EndpointsConfig.from_env()
    except ValidationError as exc:
        st.error(f"Invalid endpoints configuration: {exc}")


def main() -> None:
    setup_ui()
    initialize_session_state()

    # Auto-load .env configuration and questions from resources
    _load_config()

    # Brand row
    render_layout_style()
    st.title(":material/science: Benchmark'it")

    questions = st.session_state.get("questions", [])
    config: Optional[EndpointsConfig] = st.session_state.get("endpoints_config")
    total_tests: int = st.session_state.get("total_tests", 0)
    sampling: Dict[str, Any] = config.to_kwargs() if config else {}
    sampling_summary = (
        ", ".join(f"{k}={v}" for k, v in sampling.items()) or "defaults"
    )

    col_stats, col_stream = st.columns([1, 4], gap="medium")

    with col_stats:
        start_clicked = st.button(
            "Start benchmark",
            icon=":material/flag:",
            type="primary",
            disabled=total_tests == 0 or not config or not config.model_list(),
        )
        st.caption(f"{len(questions)} questions · `{sampling_summary}`")
        st.caption(":material/timer: Elapsed")
        clock_slot = st.empty()
        question_slot = st.empty()
        score_slot = st.empty()

    with col_stream:
        render_stream_box_style()
        ui = BenchmarkUi(
            stream=st.container(
                height=STREAM_HEIGHT_PX, border=True, autoscroll=True
            ),
            question=question_slot,
            score=score_slot,
        )

    update_stats(ui, total_tests)

    if start_clicked and config:
        st.session_state["benchmark_started_at"] = time.time()
        st.session_state["benchmark_finished_at"] = None
        render_chronometer(clock_slot, st.session_state["benchmark_started_at"])

        run_benchmark(questions, config, total_tests, ui)

        st.session_state["benchmark_finished_at"] = time.time()
        st.session_state["total_running_time"] = (
            st.session_state["benchmark_finished_at"]
            - st.session_state["benchmark_started_at"]
        )

    render_chronometer(
        clock_slot,
        st.session_state.get("benchmark_started_at"),
        st.session_state.get("benchmark_finished_at"),
    )


if __name__ == "__main__":
    main()
