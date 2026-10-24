import threading
from typing import Any, Dict, List, Optional

from src.core.job import RunJob
from src.core.llm import (
    Emit,
    RunCancelled,
    call_llm_and_stream,
    verify_response,
    warmup_model,
)
from src.core.models import EndpointsConfig
from src.core.utils import save_results


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RunCancelled()


def process_question(
    model: str,
    question: Dict,
    number: int,
    index: int,
    config: EndpointsConfig,
    job: RunJob,
    emit: Emit,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Process a single question for a model, emitting stream events."""
    _check_cancelled(cancel_event)
    emit({
        "event": "question_started",
        "index": index,
        "number": number,
        "model": model,
        "uuid": question["uuid"],
        "question": question["question"],
    })

    # Elapsed time is the cumulative answer time only. The client live-ticks
    # from question_started until answer_done, then shows "Checking…" while
    # the verifier runs — so verification time is never counted.
    result = call_llm_and_stream(
        model,
        config.serverUrl,
        question,
        emit,
        config.to_kwargs(),
        cancel_event,
        api_key=config.apiKey or "none",
        sysprompt=config.sysprompt,
    )
    _check_cancelled(cancel_event)
    timing = {
        "took": result.took,
        "ttft": result.ttft,
        "genTook": result.gen_duration,
        # `tokens` stays completion-only (the tok/s numerator); `consumed`
        # is what the answer actually cost (prompt + completion), the figure
        # displayed to the user. promptTokens is 0 when the server reports no
        # usage, in which case consumed equals the (estimated) completion.
        "tokens": result.completion_tokens,
        "consumed": result.completion_tokens + result.prompt_tokens,
        "promptTokens": result.prompt_tokens,
        "tokensEstimated": result.tokens_estimated,
        # Marks the turn as a fragment rather than a finished answer: its
        # duration, token rate and verdict all describe an incomplete reply.
        "truncated": result.truncated,
    }
    emit({"event": "answer_done", "index": index, **timing})

    # A truncated answer is never handed to the verifier: judging a fragment
    # against the expected answer measures the transport failure, not the
    # model, and the verdict would land in the score as if the model had
    # answered badly. The question is failed instead, so the failed-question
    # list names the turns a run cannot be trusted on.
    if result.truncated:
        job.mark_question_failed(number)
        job.add_result(
            model=model,
            question_uuid=question["uuid"],
            succeeded=False,
            response_text=result.text,
            took=result.took,
            tokens=result.completion_tokens,
            ttft=result.ttft,
            gen_duration=result.gen_duration,
            prompt_tokens=result.prompt_tokens,
            tokens_estimated=result.tokens_estimated,
            truncated=True,
        )
        emit({
            "event": "question_done",
            "index": index,
            "number": number,
            "model": model,
            "uuid": question["uuid"],
            "succeeded": False,
            **timing,
            "failedLabel": job.failed_label(),
            "stats": _stats_payload(job),
        })
        return

    succeeded = verify_response(config.verifierAgent, question, result.text, emit)
    if not succeeded:
        job.mark_question_failed(number)

    job.add_result(
        model=model,
        question_uuid=question["uuid"],
        succeeded=succeeded,
        response_text=result.text,
        took=result.took,
        tokens=result.completion_tokens,
        ttft=result.ttft,
        gen_duration=result.gen_duration,
        prompt_tokens=result.prompt_tokens,
        tokens_estimated=result.tokens_estimated,
    )
    emit({
        "event": "question_done",
        "index": index,
        "number": number,
        "model": model,
        "uuid": question["uuid"],
        "succeeded": succeeded,
        **timing,
        "failedLabel": job.failed_label(),
        "stats": _stats_payload(job),
    })


def _stats_payload(job: RunJob) -> Dict[str, Any]:
    """Return the aggregate stats carried on question_done events."""
    successes, answered = job.get_overall_stats()
    return {
        "current": job.current_index,
        "successes": successes,
        "answered": answered,
        "tokenRate": job.get_overall_token_rate(),
        "answeredTime": job.get_answered_time(),
    }


def run_benchmark(
    questions: List[Dict],
    config: EndpointsConfig,
    job: RunJob,
    emit: Emit,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Execute benchmark for all models and questions, emitting run events.

    Raises ``RunCancelled`` (without saving results) when ``cancel_event``
    is set — checked between questions and during answer streaming.
    """
    job.reset()
    models = config.model_list()
    total_tests = len(questions) * len(models)

    for model in models:
        _check_cancelled(cancel_event)
        emit({"event": "model_started", "model": model})
        # Load the model before the first question so neither the chronometer
        # (which shows "Starting…" until question_started) nor the first
        # question's ``took`` includes model-loading wait.
        warmup_model(model, config.serverUrl, cancel_event, api_key=config.apiKey or "none")
        _check_cancelled(cancel_event)
        for idx, question in enumerate(questions, 1):
            job.current_index += 1
            process_question(
                model, question, idx, job.current_index, config, job, emit, cancel_event
            )

    save_results(job)
    emit({
        "event": "run_finished",
        "totalRunningTime": job.total_running_time,
        "totalTests": total_tests,
    })
