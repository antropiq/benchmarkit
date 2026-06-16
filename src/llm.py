import json
import time
from typing import Any, Dict, Optional

import streamlit as st
from litellm import completion

from src.state import add_result_to_state
from src.utils import format_time, LLMResult


def safe_get_choice_content(chunk: Any) -> str:
    """Safely extract content from various response types."""
    if isinstance(chunk, dict):
        return (
            chunk.get("choices", [{}])[0]
            .get("delta", {})
            .get("content", "")
        )
    elif hasattr(chunk, "choices"):
        try:
            return chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            return ""
    return ""


def safe_get_token_usage(chunk: Any) -> int:
    """Safely extract token usage from various response types."""
    if isinstance(chunk, dict):
        return chunk.get("usage", {}).get("total_tokens", 0)
    elif hasattr(chunk, "usage"):
        return getattr(chunk, "usage", {}).get("total_tokens", 0)
    return 0


def call_llm_and_stream(
    model_name: str,
    server_url: str,
    question: Dict,
) -> LLMResult:
    """Make LLM call with streaming and return results."""
    sysmsg: str = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": question["question"]}
    ]
    start_time = time.time()
    buffer = ""
    last_chunk: Optional[Any] = None

    try:
        response_generator = completion(
            model=f"openai/{model_name}",
            api_key="none",
            messages=messages,
            api_base=server_url,
            stream=True,
        )
    except Exception as exc:
        st.error(f"⚠️ LLM call failed: {exc}")
        return "", time.time() - start_time, 0

    stream_placeholder = st.empty()
    try:
        for chunk in response_generator:
            content = safe_get_choice_content(chunk)

            if not content:
                continue

            last_chunk = chunk
            buffer += content
            stream_placeholder.markdown(buffer, unsafe_allow_html=True)
            time.sleep(0.05)  # Allow UI to update
    except Exception as exc:
        st.error(f"⚠️ Stream error: {exc}")
        buffer = ""

    # Extract token usage safely
    tokens_used = 0
    if hasattr(response_generator, "last_response"):
        tokens_used = response_generator.last_response.usage.total_tokens
    # tokens_used = safe_get_token_usage(last_chunk) if last_chunk else 0
    st.session_state["total_running_time"] = st.session_state["total_running_time"] + time.time() - start_time
    return buffer, time.time() - start_time, tokens_used


def verify_response(
    verifier: Dict,
    question: Dict,
    response_text: str,
) -> bool:
    """Verify LLM response against expected answer."""
    try:
        messages = [
            {"role": "system", "content": verifier.get("sysprompt", "")},
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
            model=f"openai/{verifier['endpoint']}",
            api_key="none",
            messages=messages,
            api_base=verifier["serverUrl"],
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
        st.error(f"⚠️ Verifier failed: {exc}")
        return False


def process_question(
    model: str,
    question: Dict,
    server_url: str,
    verifier: Dict,
    progress_bar,
    score_progress,
    total_tests: int,
) -> None:
    """Process single question for a model."""
    st.subheader(f"Question `{question['index']}`")
    st.text(f"❓ {question['question']}")

    # LLM call with streaming
    response_text, took, tokens_used = call_llm_and_stream(
        model, server_url, question
    )

    # Verification
    succeeded = verify_response(verifier, question, response_text)

    # Update UI and state
    st.metric(
        label="✅ Succeeded" if succeeded else "❌ Failed",
        value=None,
        delta=None,
    )
    add_result_to_state(
        model=model,
        question_uuid=question["uuid"],
        succeeded=succeeded,
        response_text=response_text,
        took=took,
        tokens=tokens_used,
    )
    progress_bar.progress(st.session_state.current_index / total_tests)
    current_successes = sum(
        1 for model_entry in st.session_state.results
        for resp in model_entry["responses"] if resp["succeeded"]
    )
    t: str = format_time(st.session_state["total_running_time"])
    score_progress.markdown(f"""
        **Score:** {current_successes}/{total_tests} **Running for:** {t}
        Fail count: {st.session_state.current_index - current_successes}
        """.strip()
    )
