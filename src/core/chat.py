"""Streaming chat with the benchmark-configured LLM.

Pure loop + event emission, like ``core/llm.py``: takes an ``emit``
callable, no FastAPI or browser concerns. Reasoning/thinking deltas are
never emitted as text — a single ``reasoning`` event flags that the model
is thinking — so the chat UI can show an indicator instead.
"""

import threading
import time
from typing import Any, Dict, List, Optional

from litellm import completion

from src.core.llm import Emit, _extract_reasoning, _extract_usage, safe_get_choice_content
from src.core.utils import estimate_completion_tokens

CHAT_SYSTEM_PROMPT = "You are a helpful assistant."


def stream_chat(
    messages: List[Dict[str, str]],
    model_name: str,
    server_url: str,
    emit: Emit,
    sampling: Optional[Dict[str, Any]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Stream one chat reply, emitting ``token`` / ``reasoning`` / ``done``.

    Terminal event is exactly one of ``done`` or ``error`` (unless the call
    was cancelled — then the client is gone and nothing is emitted).
    """
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *messages]

    t0 = time.monotonic()
    sampling_kwargs: Dict[str, Any] = dict(sampling or {})
    # litellm gates reasoning_effort to OpenAI reasoning models only;
    # allow it through so custom servers (llama.cpp) receive it.
    allowed: Optional[list] = (
        ["reasoning_effort"] if "reasoning_effort" in sampling_kwargs else None
    )
    try:
        response_generator = completion(
            model=f"openai/{model_name}",
            api_key="none",
            messages=messages,
            api_base=server_url,
            stream=True,
            stream_options={"include_usage": True},
            allowed_openai_params=allowed,
            **sampling_kwargs,
        )
    except Exception as exc:
        emit({"event": "error", "message": f"LLM call failed: {exc}"})
        return

    buffer = ""
    completion_tokens: Optional[int] = None
    last_usage: Optional[Dict[str, Optional[int]]] = None
    reasoning_emitted = False
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    try:
        for chunk in response_generator:
            if cancel_event is not None and cancel_event.is_set():
                close = getattr(response_generator, "close", None)
                if close:
                    close()
                return
            usage = _extract_usage(chunk)
            if usage is not None:
                last_usage = usage
                if usage["completion_tokens"] is not None:
                    completion_tokens = usage["completion_tokens"]
            content = safe_get_choice_content(chunk)
            reasoning = _extract_reasoning(chunk)
            if reasoning and not reasoning_emitted:
                reasoning_emitted = True
                emit({"event": "reasoning"})
            # Reasoning deltas open and extend the decode window (like the
            # benchmark loop in core/llm.py): they are generated tokens too,
            # so tok/s must not pretend the thinking phase was free.
            if not content and not reasoning:
                continue
            arrival = time.monotonic()
            if t_first is None:
                t_first = arrival
            t_last = arrival
            if not content:
                continue
            buffer += content
            emit({"event": "token", "text": content})
    except Exception as exc:
        emit({"event": "error", "message": f"Stream error: {exc}"})
        return

    took = (t_last or time.monotonic()) - t0
    ttft = t_first - t0 if t_first is not None else 0.0
    tokens_estimated = False
    if not completion_tokens:
        if last_usage is not None:
            total = last_usage.get("total_tokens")
            prompt = last_usage.get("prompt_tokens")
            if total and prompt:
                completion_tokens = max(total - prompt, 0)
        if not completion_tokens:
            completion_tokens = estimate_completion_tokens(buffer)
            tokens_estimated = completion_tokens > 0

    emit({
        "event": "done",
        "took": took,
        "ttft": ttft,
        "tokens": completion_tokens or 0,
        "tokensEstimated": tokens_estimated,
    })
