# AGENTS.md — benchmarkit

## Project overview

A Streamlit app that benchmarks LLM endpoints by streaming questions, collecting responses, and verifying them against expected answers using a secondary LLM verifier agent. Data is loaded via two JSON files uploaded through the UI.

## Commands

```bash
# Setup (creates .venv + installs package in editable mode)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the app
streamlit run src/main.py

# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest
```

A pytest-based test suite lives under `tests/`. No lint or build step exists. Dependencies are managed via `pyproject.toml` (hatchling backend).

## Code structure

Package layout under `src/`, organized into these modules:

| Module | Purpose |
|---|---|
| **`main.py`** | Streamlit UI scaffolding, file upload handling, benchmark trigger |
| **`bench.py`** | `run_benchmark()` — orchestrates model/question iteration |
| **`llm.py`** | `call_llm_and_stream()`, `verify_response()`, `process_question()` |
| **`config.py`** | Streamlit page config, session-state initialization |
| **`state.py`** | Session-state result management |
| **`utils.py`** | JSON loading, result saving, helpers |

## Data files

- **`questions.json`** — Array of `{ uuid, question, expected }` objects. UUIDs are sequential (`01234567-0123-0123-0123-0000000000XX`). 30 questions from matchingpennies.com.
- **`endpoints.json`** — Contains `serverUrl` (target LLM), `endpoints` (model names to test), and `verifierAgent` (a separate LLM used to verify responses).
- **`results.json`** — Generated output, gitignored.

## Key patterns & conventions

- **LiteLLM abstraction**: All LLM calls use `litellm.completion()` with `openai/<model>` prefix and `api_base` pointing to a compatible server (e.g., llama.cpp). `api_key="none"` — no real API keys used.
- **Streaming**: `call_llm_and_stream()` iterates chunks, extracts content via `safe_get_choice_content()`, and updates a Streamlit placeholder in real time. Token usage is read from `response_generator.last_response.usage.total_tokens`.
- **Session state**: All mutable state lives in `st.session_state` — results, current index, running time, config. Initialize with `initialize_session_state()`.
- **Verifier agent**: A separate LLM call that evaluates whether a model's response matches the expected answer. Returns `true`/`false` (case-insensitive).
- **Type annotations**: Explicit `: str`, `: float`, `: Dict` etc. with `typing` imports.
- **Error handling**: All external calls wrapped in try/except with `st.error()` UI feedback.

## Gotchas

- The verifier agent and the target LLM can be **different endpoints** — `serverUrl` is the target, `verifierAgent.serverUrl` is the verifier.
- Token counting relies on `response_generator.last_response` attribute from LiteLLM, not the last chunk. The chunk-based approach is commented out (l.160).
- `time.sleep(0.05)` in the streaming loop is intentional — it allows Streamlit's UI to update between chunks.
- The app reuses the same `endpoints.json` `serverUrl` for both the target LLM and the verifier's `api_base` unless overridden in `verifierAgent`.
- Results are saved only at the end of the full benchmark run, not incrementally.

## Adding questions

Add entries to `questions.json` under the `"questions"` key. Each needs a unique `uuid`, the `question` text, and the `expected` answer. UUIDs should follow the existing sequential pattern.

## Adding endpoints

Edit `endpoints.json` — add model names to the `"endpoints"` array. The `serverUrl` should point to your llama.cpp or compatible inference server.
