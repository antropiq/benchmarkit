import json
import threading
import time
from typing import Any, Callable, Dict, Optional

from litellm import completion

from src.core.models import VerifierAgent
from src.core.utils import LLMResult

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


def _extract_total_tokens(chunk: Any) -> Optional[int]:
    """Read total token usage from a streamed chunk, when the server sent one."""
    usage = (
        chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
    )
    if usage is None:
        return None
    total = (
        usage.get("total_tokens")
        if isinstance(usage, dict)
        else getattr(usage, "total_tokens", None)
    )
    return int(total) if total else None


def call_llm_and_stream(
    model_name: str,
    server_url: str,
    question: Dict,
    emit: Emit,
    sampling: Optional[Dict] = None,
    cancel_event: Optional[threading.Event] = None,
) -> LLMResult:
    """Make LLM call, stream the answer as ``token`` events, return results."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question["question"]},
    ]
    start_time = time.time()
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
        return "", time.time() - start_time, 0

    tokens_used = 0
    try:
        for chunk in response_generator:
            if cancel_event is not None and cancel_event.is_set():
                close = getattr(response_generator, "close", None)
                if close:
                    close()
                raise RunCancelled()
            tokens_used = _extract_total_tokens(chunk) or tokens_used
            content = safe_get_choice_content(chunk)

            if not content:
                continue

            buffer += content
            emit({"event": "token", "text": content})
    except RunCancelled:
        raise
    except Exception as exc:
        emit({"event": "run_error", "message": f"Stream error: {exc}"})
        buffer = ""

    return buffer, time.time() - start_time, tokens_used


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
