import json
import threading
import time
from typing import Any, Callable, Dict, Optional

from litellm import completion

from src.core.models import DEFAULT_LLM_SYSPROMPT, VerifierAgent
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

    Returns ``{prompt_tokens, completion_tokens, total_tokens}`` with
    ``None`` for missing/zero fields, ``None`` overall when the chunk
    carries no usage.
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

# Sentinel for "this litellm version does not expose the attribute at all".
_ATTR_MISSING = object()


def saw_finish_reason(response_generator: Any) -> bool:
    """Whether the provider really ended the stream with a finish reason.

    This is the only reliable truncation signal, and it is *not* on the chunks:
    when a stream ends without a ``finish_reason`` chunk — which is what
    llama.cpp does when its own upstream disconnects mid-generation — litellm
    **fabricates** ``finish_reason: "stop"`` on the final chunk it hands the
    caller (``CustomStreamWrapper.finish_reason_handler``). So the chunks cannot
    tell a cut answer from a finished one; they say ``stop`` either way.

    The truth survives on the wrapper object: ``received_finish_reason`` (and
    ``intermittent_finish_reason``, for servers that send it out of band) stay
    ``None`` when no chunk ever carried one.

    Fails open: if the wrapper exposes neither attribute (another litellm
    version), this returns ``True`` so detection never raises a false alarm on
    a complete answer. It is only meaningful once the stream has been fully
    drained — the attributes are set while iterating.
    """
    received = getattr(response_generator, "received_finish_reason", _ATTR_MISSING)
    if received is _ATTR_MISSING:
        return True
    if received:
        return True
    return bool(getattr(response_generator, "intermittent_finish_reason", None))


def close_llm_stream(response_generator: Any) -> None:
    """Close a streamed litellm response so the server aborts generation.

    litellm's ``CustomStreamWrapper`` (>= some 1.x releases) has no sync
    ``close()`` — only an async ``aclose()`` — so ``getattr(gen, "close")``
    is a silent no-op and the underlying HTTP connection stays open, letting
    the LLM server keep streaming into a socket nobody reads. The raw
    provider stream (openai's sync ``Stream``) lives on ``.completion_stream``
    and does expose ``close()``, which closes the httpx connection; servers
    like llama.cpp then abort the generation slot on their next failed write.
    Best-effort: cleanup must never raise out of a cancel path.
    """
    if response_generator is None:
        return
    target = getattr(response_generator, "completion_stream", None)
    if target is None:
        target = response_generator
    close = getattr(target, "close", None)
    if not callable(close):
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
        # litellm gates reasoning_effort to OpenAI reasoning models only;
        # allow it through so custom servers (llama.cpp) receive it.
        allowed: Optional[list] = (
            ["reasoning_effort"] if "reasoning_effort" in sampling_kwargs else None
        )
        response_generator = completion(
            model=f"openai/{model_name}",
            api_key=api_key,
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
    # generation, not a finished answer. litellm labels such a stream "stop"
    # (see saw_finish_reason), so nothing downstream would notice: the fragment
    # would be scored as if it were a final answer and its timing and token
    # rate would enter the run aggregates. Checked after the loop because the
    # wrapper records the reason while iterating. Only judged when something
    # was actually generated: an empty stream (or a stream that died before any
    # chunk) has nothing to call incomplete, and a hard transport error already
    # went down the ``except`` path above.
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
        completion(
            model=f"openai/{model_name}",
            api_key=api_key,
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
            api_key=verifier.apiKey or "none",
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
