from dataclasses import dataclass
from typing import Dict, List

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.llm import process_question
from src.models import EndpointsConfig
from src.state import get_overall_stats, get_overall_token_rate
from src.ui import render_token_rate
from src.utils import save_results


@dataclass
class BenchmarkUi:
    """Render targets the benchmark streams its progress into."""

    stream: DeltaGenerator
    question: DeltaGenerator
    score: DeltaGenerator
    clock: DeltaGenerator
    rate: DeltaGenerator


def update_stats(ui: BenchmarkUi, total_tests: int) -> None:
    """Refresh the current-question and score metric cards."""
    current_index: int = st.session_state.current_index
    successes, answered = get_overall_stats()
    render_token_rate(ui.rate, get_overall_token_rate())
    ui.question.metric(
        "Question in progress",
        f"{current_index:02d}" if current_index else "--",
        delta=f"{current_index} of {total_tests}",
        delta_color="off",
        border=True,
    )
    ui.score.metric(
        "Score",
        f"{successes}/{answered}",
        delta=f"{successes / answered * 100:.0f}%" if answered else "0%",
        delta_color="off",
        border=True,
    )


def run_benchmark(
    questions: List[Dict],
    config: EndpointsConfig,
    total_tests: int,
    ui: BenchmarkUi,
) -> None:
    """Execute benchmark for all models and questions."""
    progress_bar = ui.stream.progress(0.0)
    st.session_state.current_index = 0
    update_stats(ui, total_tests)

    for model in config.model_list():
        ui.stream.badge(model, icon=":material/memory:", color="violet")
        for idx, question in enumerate(questions, 1):
            st.session_state.current_index += 1
            update_stats(ui, total_tests)
            process_question(model, question, idx, config, ui)
            progress_bar.progress(st.session_state.current_index / total_tests)
            update_stats(ui, total_tests)

    ui.stream.success(
        "Benchmark finished",
        icon=":material/task_alt:",
    )
    save_results()
