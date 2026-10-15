# AGENTS.md — benchmarkit

## Project overview

A Streamlit app that benchmarks LLM endpoints by streaming questions, collecting responses, and verifying them against expected answers using a secondary LLM verifier agent. Questions are auto-loaded from `resources/questions.json` and configuration from `.env` on page load.

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
| **`main.py`** | Streamlit UI: brand row + two-column layout (`st.columns([1, 4])`), auto-loading of `.env` + questions, benchmark trigger |
| **`bench.py`** | `run_benchmark()`, `BenchmarkUi` dataclass (stream/question/score slots), `update_stats()` metric cards |
| **`llm.py`** | `call_llm_and_stream()`, `verify_response()`, `process_question()` |
| **`ui.py`** | `chronometer_html()` / `render_chronometer()` — client-side JS digital clock (no native Streamlit equivalent) |
| **`config.py`** | Streamlit page config, session-state initialization |
| **`models.py`** | Pydantic models for the benchmark config (`EndpointsConfig`, `SamplingParams`, `VerifierAgent`), built from env vars via `EndpointsConfig.from_env()` |
| **`state.py`** | Session-state result management |
| **`utils.py`** | JSON loading, result saving, helpers |

## Data files

- **`questions.json`** — Array of `{ uuid, question, expected }` objects. UUIDs are sequential (`01234567-0123-0123-0123-0000000000XX`). 30 questions from matchingpennies.com.
- **`.env`** (project root, gitignored; see `.env.example`) — Platform-independent benchmark configuration. Keys: `SERVER_URL`, `ENDPOINT` (or comma-separated `ENDPOINTS`), `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`, and the verifier block `VERIFIER_AGENT_SYSPROMPT`, `VERIFIER_AGENT_SERVERURL`, `VERIFIER_AGENT_ENDPOINT`. Loaded via `python-dotenv` (real env vars take precedence) and parsed by `EndpointsConfig.from_env()` into the same pydantic models (`EndpointsConfig`, `SamplingParams`, `VerifierAgent`); string values are coerced, invalid values raise a `ValidationError` shown in the UI.
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
- Token counting relies on `stream_options={"include_usage": True}`: the server reports usage in the final stream chunk, read via `_extract_total_tokens()`. Servers that ignore `include_usage` report 0 tokens.
- `time.sleep(0.05)` in the streaming loop is intentional — it allows Streamlit's UI to update between chunks.
- The chronometer ticks **client-side** (JS in `ui.py`, rendered via `st.html(unsafe_allow_javascript=True)`) because the benchmark run blocks the Streamlit script server-side; it is seeded from `benchmark_started_at`/`benchmark_finished_at` epoch timestamps.
- Look & feel (colors, Inter/JetBrains Mono fonts, radii) is set in `.streamlit/config.toml` — do not style with custom CSS.
- The app reuses the same `SERVER_URL` for both the target LLM and the verifier's `api_base` unless `VERIFIER_AGENT_SERVERURL` is set.
- Results are saved only at the end of the full benchmark run, not incrementally.

## Adding questions

Add entries to `questions.json` under the `"questions"` key. Each needs a unique `uuid`, the `question` text, and the `expected` answer. UUIDs should follow the existing sequential pattern.

## Adding endpoints

Edit `.env` — set the model name in `ENDPOINT`. `SERVER_URL` should point to your llama.cpp or compatible inference server. Sampling parameters (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`) are optional and forwarded to `litellm.completion()` for the tested model. `REASONING` is optional too and sets the reasoning effort (`off`/`low`/`medium`/`high`/`xhigh`).
