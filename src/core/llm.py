import json
import threading
import time
from typing import Any, Callable, Dict, Optional

from litellm import completion

from src.core.models import VerifierAgent
from src.core.utils import LLMResult, estimate_completion_tokens

Emit = Callable[[Dict[str, Any]], None]


class RunCancelled(Exception):
    """Raised inside the streaming loop when a stop has been requested."""


def safe_get_choice_content(chunk: Any) -> str:
    """Safely extract content from various response types."""
    if isinstance(chunk, dict):
        return (
            (chunk.get("choices") or [{}])[0]
            .get("delta", {})
            .get("content", "")
        )
    elif hasattr(chunk, "choices"):
        try:
            return chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            return ""
    return ""


def _extract_reasoning(chunk: Any) -> str:
    """Safely extract streamed reasoning text (thinking) from a chunk.

    Reasoning-capable servers emit thinking deltas under ``reasoning_content``
    or ``reasoning`` instead of ``content``; they are generated tokens too,
    so they must anchor the decode-window timing even though they are not
    part of the answer text.
    """
    if isinstance(chunk, dict):
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        if not isinstance(delta, dict):
            return ""
        return str(delta.get("reasoning_content") or delta.get("reasoning") or "")
    if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta is None:
            return ""
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(delta, attr, None)
            if value:
                return str(value)
    return ""


def _extract_usage(chunk: Any) -> Optional[Dict[str, Optional[int]]]:
    """Read the usage block from a streamed chunk, when the server sent one.

    Returns ``{prompt_tokens, completion_tokens, total_tokens}`` with ``None``
    for missing/zero fields, or ``None`` when the chunk carries no usage.
    """
    usage = (
        chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
    )
    if not usage:
        return None

    def field(name: str) -> Optional[int]:
        value = (
            usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        )
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


def call_llm_and_stream(
    model_name: str,
    server_url: str,
    question: Dict,
    emit: Emit,
    sampling: Optional[Dict] = None,
    cancel_event: Optional[threading.Event] = None,
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
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question["question"]},
    ]
    t0 = time.monotonic()
    buffer = ""
    response_generator: Optional[Any] = None

    try:
        sampling_kwargs: Dict[str, Any] = dict(sampling or {})
        # litellm gates reasoning_effort to OpenAI reasoning models only;
        # allow it through so custom servers (llama.cpp) receive it.
        allowed: Optional[list] = (
            ["reasoning_effort"] if "reasoning_effort" in sampling_kwargs else None
        )
        response_generator = completion(
            model=f"openai/{model_name}",
            api_key="none",
            messages=messages,
            api_base=server_url,
            stream=True,
            # Ask the server to report token usage in the final stream chunk.
            stream_options={"include_usage": True},
            allowed_openai_params=allowed,
            **sampling_kwargs,
        )
    except Exception as exc:
        emit({"event": "run_error", "message": f"LLM call failed: {exc}"})
        return LLMResult(took=time.monotonic() - t0)

    completion_tokens: Optional[int] = None
    last_usage: Optional[Dict[str, Optional[int]]] = None
    prompt_tokens = 0
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    try:
        for chunk in response_generator:
            if cancel_event is not None and cancel_event.is_set():
                close = getattr(response_generator, "close", None)
                if close:
                    close()
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

    t_end = time.monotonic()
    if t_first is None:
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
    )


def warmup_model(
    model_name: str,
    server_url: str,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Best-effort request that forces the server to load ``model_name``.

    The answer is discarded; a failure is ignored so real errors still
    surface on the actual questions. Keeps elapsed time (and the first
    question's ``took``) free of model-loading wait.
    """
    if cancel_event is not None and cancel_event.is_set():
        return
    try:
        completion(
            model=f"openai/{model_name}",
            api_key="none",
            messages=[{"role": "user", "content": "ping"}],
            api_base=server_url,
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
        v_resp = completion(
            model=f"openai/{verifier.endpoint}",
            api_key="none",
            messages=messages,
            api_base=verifier.serverUrl,
        )

        # Handle different response types safely
        content = ""
        if isinstance(v_resp, dict):
            content = (
                v_resp.get("choices", [{}])[0]  # type: ignore
                .get("message", {})
                .get("content", "")
            )
        elif hasattr(v_resp, "choices") and v_resp.choices:  # type: ignore
            try:
                content = v_resp.choices[0].message.content  # type: ignore
            except (AttributeError, IndexError):
                pass

        return content.strip().lower() == "true" if content else False
    except Exception as exc:
        emit({"event": "run_error", "message": f"Verifier failed: {exc}"})
        return False
