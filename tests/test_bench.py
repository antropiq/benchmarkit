import threading
from typing import Any, Dict, List

import pytest

from src.core.bench import run_benchmark
from src.core.job import RunJob
from src.core.llm import RunCancelled
from src.core.models import EndpointsConfig
from src.core.utils import LLMResult


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

    def fake_warmup(model, server_url, cancel_event=None, api_key="none"):
        events.append(("warmup", model))
        if warmup is not None:
            warmup()

    def fake_call(
        model,
        server_url,
        question,
        emit,
        sampling=None,
        cancel_event=None,
        api_key="none",
        sysprompt=None,
    ):
        return LLMResult(
            text="answer",
            took=1.5,
            ttft=0.3,
            gen_duration=1.2,
            completion_tokens=10,
            prompt_tokens=7,
        )

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
        class TwoModelConfig(EndpointsConfig):
            def model_list(self):
                return ["m1", "m2"]

        config = TwoModelConfig(serverUrl="http://server:8080/v1", endpoint="m1")
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


class TestTimingEvents:
    def test_answer_done_carries_generation_timing(
        self, monkeypatch, job, config, questions
    ):
        events: List[Any] = []
        patch_llm(monkeypatch, events)

        run_benchmark([questions[0]], config, job, events.append)

        done = next(e for e in events if isinstance(e, dict) and e["event"] == "answer_done")
        assert done["took"] == 1.5
        assert done["ttft"] == 0.3
        assert done["genTook"] == 1.2
        assert done["tokens"] == 10
        assert done["promptTokens"] == 7
        assert done["tokensEstimated"] is False
        # Displayed figure: what the answer consumed (prompt + completion),
        # while `tokens` stays the completion-only rate numerator.
        assert done["consumed"] == 17

    def test_stats_token_rate_is_generation_based(
        self, monkeypatch, job, config, questions
    ):
        events: List[Any] = []
        patch_llm(monkeypatch, events)

        run_benchmark([questions[0]], config, job, events.append)

        done = next(e for e in events if isinstance(e, dict) and e["event"] == "question_done")
        # completion tokens / gen duration, not total_tokens / took.
        assert done["stats"]["tokenRate"] == pytest.approx(10 / 1.2)


class TestTruncatedAnswer:
    """A server-cut answer is never verified, and is marked everywhere.

    Sending a fragment to the verifier measures the transport failure rather
    than the model, and the resulting verdict would land in the score as if the
    model had answered badly.
    """

    def _patch_truncated(self, monkeypatch, events, verifier_calls):
        def fake_call(*args, **kwargs):
            return LLMResult(
                text="half an answer",
                took=1.0,
                ttft=0.2,
                gen_duration=0.8,
                completion_tokens=4,
                prompt_tokens=6,
                truncated=True,
            )

        def fake_verify(verifier, question, response_text, emit):
            verifier_calls.append(response_text)
            return True

        monkeypatch.setattr("src.core.bench.warmup_model", lambda *a, **k: None)
        monkeypatch.setattr("src.core.bench.call_llm_and_stream", fake_call)
        monkeypatch.setattr("src.core.bench.verify_response", fake_verify)
        monkeypatch.setattr("src.core.bench.save_results", lambda job, *a: None)

    def test_verifier_is_skipped_and_the_question_is_failed(
        self, monkeypatch, job, config, questions
    ):
        events: List[Any] = []
        verifier_calls: List[str] = []
        self._patch_truncated(monkeypatch, events, verifier_calls)

        run_benchmark([questions[0]], config, job, events.append)

        assert verifier_calls == []  # the fragment never reached the verifier
        assert job.get_failed_questions() == [1]
        assert job.results[0]["score"] == "0/1"

    def test_events_and_results_carry_the_flag(
        self, monkeypatch, job, config, questions
    ):
        events: List[Any] = []
        self._patch_truncated(monkeypatch, events, [])

        run_benchmark([questions[0]], config, job, events.append)

        answer_done = next(
            e for e in events if isinstance(e, dict) and e["event"] == "answer_done"
        )
        question_done = next(
            e for e in events if isinstance(e, dict) and e["event"] == "question_done"
        )
        assert answer_done["truncated"] is True
        assert question_done["truncated"] is True
        assert question_done["succeeded"] is False
        # The fragment is still recorded (it is what came back), flagged.
        stored = job.results[0]["responses"][0]
        assert stored["response"] == "half an answer"
        assert stored["truncated"] is True
        assert stored["succeeded"] is False

    def test_normal_answers_are_not_flagged(self, monkeypatch, job, config, questions):
        # Same run through the shared patch_llm (whose stub verifier returns
        # True): the flag must be False, and the answer must have gone through
        # verification — evidenced by the success it can only get there.
        events: List[Any] = []
        patch_llm(monkeypatch, events)

        run_benchmark([questions[0]], config, job, events.append)

        done = next(e for e in events if isinstance(e, dict) and e["event"] == "answer_done")
        assert done["truncated"] is False
        assert job.get_failed_questions() == []
        stored = job.results[0]["responses"][0]
        assert stored["truncated"] is False
        assert stored["succeeded"] is True
