import json
import threading
import time
from typing import Any, Callable, Dict, Optional

from src.core.llmkit import chat_completion, stream_completion

from src.core.models import DEFAULT_LLM_SYSPROMPT, VerifierAgent
from src.core.utils import LLMResult, estimate_completion_tokens

Emit = Callable[[Dict[str, Any]], None]


class RunCancelled(Exception):
    """Raised inside the streaming loop when a stop has been requested."""


# --- chunk reading ------------------------------------------------------------
#
# One read path, one protocol: a chunk is read as a mapping (``.get``), never
# partly as the attribute shape (``chunk.choices[0].delta.content``). ``llmkit``
# hands over plain chunk dicts, which read as mappings, and a stub that answers
# ``.get`` without being a dict reads the same way — so a chunk of any shape
# either yields its fields or reads as absent, and a read never raises out of
# the streaming loop. Anything else (a string, ``None``, a drained stream) reads
# as absent rather than raising.


def _field(container: Any, name: str) -> Any:
    """One mapping read, ``None`` when ``container`` carries no such field."""
    get = getattr(container, "get", None)
    if not callable(get):
        return None
    return get(name)


def _delta(chunk: Any) -> Any:
    """The chunk's generation delta, ``None`` when it carries none."""
    choices = _field(chunk, "choices")
    if not choices:
        return None
    return _field(choices[0], "delta")


def safe_get_choice_content(chunk: Any) -> str:
    """Safely extract content from a streamed chunk."""
    return str(_field(_delta(chunk), "content") or "")


def _extract_reasoning(chunk: Any) -> str:
    """Safely extract streamed reasoning text (thinking) from a chunk.

    Reasoning-capable servers emit thinking deltas under ``reasoning_content``
    or ``reasoning`` instead of ``content``; they are generated tokens too,
    so they must anchor the decode-window timing even though they are not
    part of the answer text.
    """
    delta = _delta(chunk)
    return str(
        _field(delta, "reasoning_content") or _field(delta, "reasoning") or ""
    )


def _extract_usage(chunk: Any) -> Optional[Dict[str, Optional[int]]]:
    """Read the usage block from a streamed chunk, when the server sent one.

    Returns ``{prompt_tokens, completion_tokens, total_tokens}`` with
    ``None`` for missing/zero fields, ``None`` overall when the chunk
    carries no usage.
    """
    usage = _field(chunk, "usage")
    if not usage:
        return None

    def field(name: str) -> Optional[int]:
        value = _field(usage, name)
        return int(value) if value else None

    return {
        "prompt_tokens": field("prompt_tokens"),
        "completion_tokens": field("completion_tokens"),
        "total_tokens": field("total_tokens"),
    }


def _extract_total_tokens(chunk: Any) -> Optional[int]:
    """Read total token usage from a streamed chunk, when the server sent one."""
    usage = _extract_usage(chunk)
    return usage["total_tokens"] if usage else None


# Shared by both streaming loops (chat imports it from here). One sentence,
# deliberately free of provider jargon: it becomes the visible message in the
# WorkBench error line and in the run's error entry.
TRUNCATION_MESSAGE = (
    "The answer was cut short by the server before it finished "
    "(the stream ended without a finish reason). The reply above is "
    "incomplete — the task was not carried through."
)

# Carried on the terminal event so a client can style this failure differently
# from a plain one — and, in WorkBench, mark the *message* rather than only
# showing a transient error line (which the next send clears).
TRUNCATION_EVENT_FLAG = "truncated"


def saw_finish_reason(response_generator: Any) -> bool:
    """Whether the stream ended with a chunk that carried a finish reason.

    A stream records this while it is iterated, so the answer is only
    meaningful once the loop has drained it. A stream that ends with no chunk
    ever carrying a reason is a generation the server cut short, and nothing
    downstream would notice otherwise: the chunks of a cut answer and of a
    finished one look the same.

    Fails open: a stream without the attribute (``None``, or a stub) returns
    ``True``, so detection never raises a false alarm on a complete answer.
    """
    return bool(getattr(response_generator, "saw_finish_reason", True))


def close_llm_stream(response_generator: Any) -> None:
    """Close a streamed response so the server aborts generation.

    Best-effort: the socket teardown must never raise out of a cancel path, and
    a stream without a ``close()`` has nothing to close.
    """
    if response_generator is None:
        return
    close = getattr(response_generator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001 — closing on cancel is best-effort
        pass


def call_llm_and_stream(
    model_name: str,
    server_url: str,
    question: Dict,
    emit: Emit,
    sampling: Optional[Dict] = None,
    cancel_event: Optional[threading.Event] = None,
    api_key: str = "none",
    sysprompt: Optional[str] = None,
) -> LLMResult:
    """Make LLM call, stream the answer as ``token`` events, return results.

    Timing uses ``time.monotonic()`` and is anchored to *generated* chunks —
    content or reasoning deltas (the trailing usage-only chunk is not timed):
    ``took`` runs from request start to the last generated chunk, ``ttft`` to
    the first generated chunk, and ``gen_duration`` covers first-to-last
    generated chunk — the window matching llama.cpp's generation (decode)
    speed, since reasoning deltas are part of ``completion_tokens``.
    """
    messages = [
        {"role": "system", "content": sysprompt or DEFAULT_LLM_SYSPROMPT},
        {"role": "user", "content": question["question"]},
    ]
    t0 = time.monotonic()
    buffer = ""
    response_generator: Optional[Any] = None

    try:
        sampling_kwargs: Dict[str, Any] = dict(sampling or {})
        response_generator = stream_completion(
            model=model_name,
            server_url=server_url,
            messages=messages,
            api_key=api_key,
            sampling=sampling_kwargs,
        )
    except Exception as exc:
        emit({"event": "run_error", "message": f"LLM call failed: {exc}"})
        return LLMResult(took=time.monotonic() - t0)

    completion_tokens: Optional[int] = None
    last_usage: Optional[Dict[str, Optional[int]]] = None
    prompt_tokens = 0
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    # Any generated text at all — answer OR reasoning. A thinking model cut
    # mid-reasoning streams an empty answer, so gating the truncation check on
    # ``buffer`` alone would leave that turn looking like a clean empty answer.
    generated_any = False
    try:
        for chunk in response_generator:
            if cancel_event is not None and cancel_event.is_set():
                close_llm_stream(response_generator)
                raise RunCancelled()
            usage = _extract_usage(chunk)
            if usage is not None:
                last_usage = usage
                if usage["completion_tokens"] is not None:
                    completion_tokens = usage["completion_tokens"]
                if usage["prompt_tokens"]:
                    prompt_tokens = usage["prompt_tokens"]
            content = safe_get_choice_content(chunk)
            reasoning = _extract_reasoning(chunk)

            if not content and not reasoning:
                continue

            generated_any = True
            # Time the last generated chunk, not the stream drain: with
            # include_usage the server appends a usage-only chunk afterwards.
            # Reasoning deltas open/extend the window (they are decoded
            # tokens counted in completion_tokens) but are not answer text.
            arrival = time.monotonic()
            if t_first is None:
                t_first = arrival
            t_last = arrival
            if not content:
                continue
            buffer += content
            emit({"event": "token", "text": content})
    except RunCancelled:
        raise
    except Exception as exc:
        emit({"event": "run_error", "message": f"Stream error: {exc}"})
        buffer = ""

    # A stream that ends with no chunk ever carrying a finish_reason is a cut
    # generation, not a finished answer, and nothing downstream would notice:
    # the fragment would be scored as if it were a final answer and its timing
    # and token rate would enter the run aggregates. Checked after the loop
    # because the stream records the reason while iterating (see
    # ``saw_finish_reason``). Only judged when something was actually generated:
    # an empty stream (or a stream that died before any chunk) has nothing to
    # call incomplete, and a hard transport error already went down the
    # ``except`` path above.
    truncated = generated_any and not saw_finish_reason(response_generator)
    if truncated:
        emit({"event": "run_error", "message": TRUNCATION_MESSAGE})

    t_end = time.monotonic()
    # t_last is set whenever t_first is, but narrow both for the type checker.
    if t_first is None or t_last is None:
        took, ttft, gen_duration = t_end - t0, 0.0, 0.0
    else:
        took = t_last - t0
        ttft = t_first - t0
        # Single-chunk answers have no measurable decode window.
        gen_duration = t_last - t_first

    tokens_estimated = False
    if not completion_tokens:
        # Some servers ignore include_usage or omit completion_tokens; derive
        # it from total - prompt when possible, else fall back to an estimate.
        if completion_tokens is None and last_usage is not None:
            total = last_usage.get("total_tokens")
            prompt = last_usage.get("prompt_tokens")
            if total and prompt:
                completion_tokens = max(total - prompt, 0)
        if not completion_tokens:
            completion_tokens = estimate_completion_tokens(buffer)
            tokens_estimated = completion_tokens > 0

    return LLMResult(
        text=buffer,
        took=took,
        ttft=ttft,
        gen_duration=gen_duration,
        completion_tokens=completion_tokens,
        tokens_estimated=tokens_estimated,
        prompt_tokens=prompt_tokens,
        truncated=truncated,
    )


def warmup_model(
    model_name: str,
    server_url: str,
    cancel_event: Optional[threading.Event] = None,
    api_key: str = "none",
) -> None:
    """Best-effort request that forces the server to load ``model_name``.

    The answer is discarded; a failure is ignored so real errors still
    surface on the actual questions. Keeps elapsed time (and the first
    question's ``took``) free of model-loading wait.
    """
    if cancel_event is not None and cancel_event.is_set():
        return
    try:
        chat_completion(
            model=model_name,
            server_url=server_url,
            messages=[{"role": "user", "content": "ping"}],
            api_key=api_key,
            max_tokens=1,
        )
    except Exception:  # noqa: BLE001 — warm-up is best-effort
        pass


def verify_response(
    verifier: Optional[VerifierAgent],
    question: Dict,
    response_text: str,
    emit: Emit,
) -> bool:
    """Verify LLM response against expected answer."""
    if not verifier or not verifier.endpoint:
        emit(
            {
                "event": "run_error",
                "message": "No verifier agent configured in the endpoints config",
            }
        )
        return False
    try:
        messages = [
            {"role": "system", "content": verifier.sysprompt},
            {
                "role": "user",
                "content": json.dumps({
                    "question": question["question"],
                    "response": response_text,
                    "expected": question.get("expected"),
                }),
            },
        ]
        v_resp = chat_completion(
            model=verifier.endpoint,
            server_url=verifier.serverUrl,
            messages=messages,
            api_key=verifier.apiKey or "none",
        )

        # ``choices[0].message.content`` can be ``null`` — a reasoning-only
        # judge answer cut by ``max_tokens`` really does come back that way —
        # so the verdict of a judge that says nothing is ``False``, never a
        # crash. Normalised to "" rather than relied on as falsy.
        content = (
            (v_resp.get("choices") or [{}])[0]
            .get("message", {})
            .get("content")
            or ""
        )

        return content.strip().lower() == "true"
    except Exception as exc:
        emit({"event": "run_error", "message": f"Verifier failed: {exc}"})
        return False
