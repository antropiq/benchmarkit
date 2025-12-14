# benchmark_app.py
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
from litellm import completion  # pip: streamlit, litellm

# --------------------------------------------------------------------------- #
#                                  Typing                                     #
# --------------------------------------------------------------------------- #
JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
LLMResult = Tuple[str, float, int]  # (response_text, took, tokens_used)
VerificationResult = Tuple[bool, str]  # (succeeded, response_text)


# --------------------------------------------------------------------------- #
#                                  Helpers                                    #
# --------------------------------------------------------------------------- #
def load_json_from_file(file) -> JSONType:
    """Read a JSON file and return the dict."""
    try:
        return json.load(file)
    except Exception as exc:
        st.error(f"Could not parse JSON: {exc}")
        return {}


def add_result_to_state(
    model: str,
    question_uuid: str,
    succeeded: bool,
    response_text: str,
    took: float,
    tokens: int,
) -> None:
    """Store per-question result in SessionState."""
    if "results" not in st.session_state:
        st.session_state.results = []

    model_entry = next(
        (e for e in st.session_state.results if e["llm"] == model),
        None
    )
    if not model_entry:
        model_entry = {"llm": model, "score": "", "responses": []}
        st.session_state.results.append(model_entry)

    model_entry["responses"].append({
        "uuid": question_uuid,
        "response": response_text,
        "took": took,
        "consumedTokens": tokens,
        "succeeded": succeeded,
    })

    # Update model score
    succeeded_count = sum(r["succeeded"] for r in model_entry["responses"])
    model_entry["score"] = f"{succeeded_count}/{len(model_entry['responses'])}"


def get_session_data() -> Tuple[
    List[Dict], str, List[str], Dict
]:
    """Retrieve benchmark configuration from session state."""
    return (
        st.session_state["questions"],
        st.session_state["server_url"],
        st.session_state["endpoints"],
        st.session_state["verifier_agent"],
    )


# --------------------------------------------------------------------------- #
#                                  LLM Flow                                   #
# --------------------------------------------------------------------------- #
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
    messages = [{"role": "user", "content": question["question"]}]
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
            last_chunk = chunk
            content = safe_get_choice_content(chunk)

            if not content:
                continue

            buffer += content
            stream_placeholder.markdown(buffer, unsafe_allow_html=True)
            time.sleep(0.05)  # Allow UI to update
    except Exception as exc:
        st.error(f"⚠️ Stream error: {exc}")
        buffer = ""

    # Extract token usage safely
    tokens_used = safe_get_token_usage(last_chunk) if last_chunk else 0
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


# --------------------------------------------------------------------------- #
#                                Main Flow                                    #
# --------------------------------------------------------------------------- #
def process_question(
    model: str,
    question: Dict,
    server_url: str,
    verifier: Dict,
    progress_bar,
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


def run_benchmark(
    questions: List[Dict],
    server_url: str,
    endpoints: List[str],
    verifier: Dict,
    total_tests: int,
) -> None:
    """Execute benchmark for all models and questions."""
    progress_bar = st.progress(0)
    st.session_state.current_index = 0

    for model in endpoints:
        st.subheader(f"🔍 Model **{model}**")
        for idx, question in enumerate(questions, 1):
            question["index"] = idx  # Add index for display
            st.session_state.current_index += 1
            process_question(
                model, question, server_url, verifier, progress_bar, total_tests
            )

    st.success("✅ Benchmark finished!")
    save_results()


def save_results() -> None:
    """Save benchmark results to JSON file."""
    output = {"results": st.session_state["results"]}
    output_path = Path.cwd() / "results.json"
    try:
        with open(output_path, "w") as fp:
            json.dump(output, fp, indent=2)
        st.download_button(
            label="📥 Download results.json",
            data=output_path.read_bytes(),
            file_name="results.json",
            mime="application/json",
        )
    except Exception as exc:
        st.error(f"⚠️ Could not write `{output_path}`: {exc}")


# --------------------------------------------------------------------------- #
#                                  Setup                                      #
# --------------------------------------------------------------------------- #
def initialize_session_state() -> None:
    """Initialize required session state variables."""
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "results" not in st.session_state:
        st.session_state.results = []


def setup_ui() -> None:
    """Configure Streamlit UI components."""
    st.set_page_config(page_title="LLM Benchmark UI", layout="wide")
    st.title("🔬 LLM Benchmark")
    st.markdown(
        """
        Provide two JSON files:
        * **questions.json** – list of questions (must contain `uuid`, `question` & `expected`)
        * **endpoints.json** – list of LLM endpoints + verifier settings

        Click **Start Benchmark** to run tests with real-time streaming.
        """
    )


# --------------------------------------------------------------------------- #
#                                Main Script                                  #
# --------------------------------------------------------------------------- #
def main() -> None:
    setup_ui()
    initialize_session_state()

    # File upload section
    col1, col2 = st.columns([1, 1])
    with col1:
        questions_file = st.file_uploader(
            "Upload *questions.json*", type="json", key="questions_file"
        )
    with col2:
        endpoints_file = st.file_uploader(
            "Upload *endpoints.json*", type="json", key="endpoints_file"
        )

    # Handle file uploads
    if questions_file and endpoints_file:
        questions_data = load_json_from_file(questions_file)
        endpoints_data = load_json_from_file(endpoints_file)

        # Ensure we have dictionaries
        if not isinstance(questions_data, dict):
            questions_data = {}
        if not isinstance(endpoints_data, dict):
            endpoints_data = {}

        st.session_state["questions"] = questions_data.get("questions", [])
        st.session_state["server_url"] = endpoints_data.get("serverUrl", "")
        st.session_state["endpoints"] = endpoints_data.get("endpoints", [])
        st.session_state["verifier_agent"] = endpoints_data.get(
            "verifierAgent", {}
        )

        # Calculate total tests
        questions = st.session_state["questions"]
        endpoints = st.session_state["endpoints"]
        st.session_state["total_tests"] = len(questions) * len(endpoints)

    # Start benchmark button
    if st.button("🏁 Start Benchmark") and questions_file and endpoints_file:
        questions, server_url, endpoints, verifier = get_session_data()
        run_benchmark(
            questions,
            server_url,
            endpoints,
            verifier,
            st.session_state["total_tests"],
        )


if __name__ == "__main__":
    main()
