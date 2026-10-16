from typing import Any, Callable, Dict, List

from src.core.job import RunJob
from src.core.llm import Emit, call_llm_and_stream, verify_response
from src.core.models import EndpointsConfig
from src.core.utils import save_results

Emit = Callable[[Dict[str, Any]], None]


def process_question(
    model: str,
    question: Dict,
    number: int,
    index: int,
    config: EndpointsConfig,
    job: RunJob,
    emit: Emit,
) -> None:
    """Process a single question for a model, emitting stream events."""
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
    response_text, took, tokens_used = call_llm_and_stream(
        model, config.serverUrl, question, emit, config.to_kwargs()
    )
    emit({"event": "answer_done", "index": index, "took": took, "tokens": tokens_used})

    succeeded = verify_response(config.verifierAgent, question, response_text, emit)
    if not succeeded:
        job.mark_question_failed(number)

    job.add_result(
        model=model,
        question_uuid=question["uuid"],
        succeeded=succeeded,
        response_text=response_text,
        took=took,
        tokens=tokens_used,
    )
    emit({
        "event": "question_done",
        "index": index,
        "number": number,
        "model": model,
        "uuid": question["uuid"],
        "succeeded": succeeded,
        "took": took,
        "tokens": tokens_used,
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
) -> None:
    """Execute benchmark for all models and questions, emitting run events."""
    job.reset()
    models = config.model_list()
    total_tests = len(questions) * len(models)

    for model in models:
        emit({"event": "model_started", "model": model})
        for idx, question in enumerate(questions, 1):
            job.current_index += 1
            process_question(model, question, idx, job.current_index, config, job, emit)

    save_results(job)
    emit({
        "event": "run_finished",
        "totalRunningTime": job.total_running_time,
        "totalTests": total_tests,
    })
