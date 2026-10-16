import json
import time
from typing import Any, Dict, Optional

import streamlit as st
from litellm import completion
from streamlit.delta_generator import DeltaGenerator

from src.models import EndpointsConfig, VerifierAgent
from src.state import add_result_to_state, get_answered_time
from src.ui import render_chronometer, render_checking
from src.utils import LLMResult


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
    slot: DeltaGenerator,
    sampling: Optional[Dict] = None,
) -> LLMResult:
    """Make LLM call, stream the answer into ``slot``, return results."""
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
        st.error(f"LLM call failed: {exc}", icon=":material/error:")
        return "", time.time() - start_time, 0

    tokens_used = 0
    try:
        for chunk in response_generator:
            tokens_used = _extract_total_tokens(chunk) or tokens_used
            content = safe_get_choice_content(chunk)

            if not content:
                continue

            buffer += content
            slot.markdown(buffer)
            time.sleep(0.05)  # Allow UI to update
    except Exception as exc:
        st.error(f"Stream error: {exc}", icon=":material/error:")
        buffer = ""

    return buffer, time.time() - start_time, tokens_used


def verify_response(
    verifier: Optional[VerifierAgent],
    question: Dict,
    response_text: str,
) -> bool:
    """Verify LLM response against expected answer."""
    if not verifier or not verifier.endpoint:
        st.error(
            "No verifier agent configured in the endpoints config",
            icon=":material/error:",
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
        st.error(f"Verifier failed: {exc}", icon=":material/error:")
        return False


def process_question(
    model: str,
    question: Dict,
    number: int,
    config: EndpointsConfig,
    ui: Any,
) -> None:
    """Process single question for a model inside the stream container."""
    with ui.stream.container(border=True):
        st.caption(f"Question {number} · `{model}`")
        st.markdown(f"**{question['question']}**")

        response_slot = st.empty()

        # Elapsed time is the cumulative answer time only. Tick live from the
        # answered-so-far baseline while the LLM streams this answer.
        answered_before = get_answered_time()
        render_chronometer(ui.clock, answered_before, time.time())
        response_text, took, tokens_used = call_llm_and_stream(
            model, config.serverUrl, question, response_slot, config.to_kwargs()
        )

        # Freeze the clock at the new cumulative value and show a static
        # "Checking" indicator so verifier time is never counted.
        render_checking(ui.clock)
        succeeded = verify_response(config.verifierAgent, question, response_text)
        render_chronometer(ui.clock, answered_before + took)

        if succeeded:
            st.badge("Correct", icon=":material/check_circle:", color="green")
        else:
            st.badge("Incorrect", icon=":material/error:", color="red")
        st.caption(f"{took:.1f}s · {tokens_used:,} tokens")

    add_result_to_state(
        model=model,
        question_uuid=question["uuid"],
        succeeded=succeeded,
        response_text=response_text,
        took=took,
        tokens=tokens_used,
    )
