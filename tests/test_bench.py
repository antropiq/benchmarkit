import threading
from typing import Any, Dict, List

import pytest

from src.core.bench import run_benchmark
from src.core.job import RunJob
from src.core.llm import RunCancelled
from src.core.models import EndpointsConfig


@pytest.fixture
def config() -> EndpointsConfig:
    return EndpointsConfig(serverUrl="http://server:8080/v1", endpoint="m1")


@pytest.fixture
def questions() -> List[Dict[str, Any]]:
    return [
        {"uuid": "u1", "question": "q1", "expected": "a1"},
        {"uuid": "u2", "question": "q2", "expected": "a2"},
    ]


def patch_llm(monkeypatch, events: List[Any], warmup=None):
    """Replace the LLM entry points in ``bench`` with event-recording stubs."""

    def fake_warmup(model, server_url, cancel_event=None):
        events.append(("warmup", model))
        if warmup is not None:
            warmup()

    def fake_call(model, server_url, question, emit, sampling=None, cancel_event=None):
        return "answer", 1.5, 10

    def fake_verify(verifier, question, response_text, emit):
        return True

    monkeypatch.setattr("src.core.bench.warmup_model", fake_warmup)
    monkeypatch.setattr("src.core.bench.call_llm_and_stream", fake_call)
    monkeypatch.setattr("src.core.bench.verify_response", fake_verify)
    monkeypatch.setattr("src.core.bench.save_results", lambda job, *args: None)


class TestWarmupOrdering:
    def test_warmup_runs_after_model_started_before_first_question(
        self, monkeypatch, job, config, questions
    ):
        events: List[Any] = []
        patch_llm(monkeypatch, events)

        run_benchmark(questions, config, job, events.append)

        assert events[0]["event"] == "model_started"
        assert events[1] == ("warmup", "m1")
        assert events[2]["event"] == "question_started"

    def test_warmup_once_per_model(self, monkeypatch, job, questions):
        config = EndpointsConfig(
            serverUrl="http://server:8080/v1", endpoints=["m1", "m2"]
        )
        events: List[Any] = []
        patch_llm(monkeypatch, events)

        run_benchmark(questions, config, job, events.append)

        warmed = [e[1] for e in events if isinstance(e, tuple)]
        assert warmed == ["m1", "m2"]
        model_starts = [
            e["model"] for e in events if not isinstance(e, tuple)
            and e["event"] == "model_started"
        ]
        assert model_starts == ["m1", "m2"]


class TestCancellationDuringWarmup:
    def test_stop_during_warmup_raises_without_starting_questions(
        self, monkeypatch, job, config, questions
    ):
        events: List[Any] = []
        cancel = threading.Event()

        def cancel_during_warmup():
            cancel.set()

        patch_llm(monkeypatch, events, warmup=cancel_during_warmup)

        with pytest.raises(RunCancelled):
            run_benchmark(questions, config, job, events.append, cancel)

        names = [e["event"] for e in events if not isinstance(e, tuple)]
        assert names == ["model_started"]
        assert not job.results
