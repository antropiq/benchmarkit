# AGENTS.md — benchmarkit

## Project overview

A FastAPI + Vue 3 app that benchmarks LLM endpoints by streaming questions, collecting responses, and verifying them against expected answers using a secondary LLM verifier agent. Questions are auto-loaded from `resources/questions.json` and configuration from `.env` on server startup. The frontend is a Vite-built SPA served by FastAPI; run progress is pushed to the browser over SSE.

## Commands

```bash
# Setup (creates .venv + installs package in editable mode)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Frontend build (requires Node.js; output in web/dist, served by FastAPI)
cd web && npm install && npm run build

# Run the app (serves API + UI on http://127.0.0.1:8000)
benchmarkit
# or: uvicorn src.api.server:app

# Frontend dev server (hot reload on :5173, proxies /api to :8000)
cd web && npm run dev

# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest
```

A pytest-based test suite lives under `tests/`. No lint or build step exists beyond `vite build`. Dependencies are managed via `pyproject.toml` (hatchling backend).

## Code structure

Python package layout under `src/`, split into a UI-free core and the API:

| Module | Purpose |
|---|---|
| **`api/server.py`** | `create_app()` (routers + `StaticFiles("web/dist")`), `run()` console entry |
| **`api/routes.py`** | REST + SSE endpoints, `RunState` (single active run), worker-thread launch |
| **`api/events.py`** | `EventBus`: seq-numbered in-memory event buffer, thread-safe publish, async subscribe with replay; `sse_format()` |
| **`core/bench.py`** | `run_benchmark()`, `process_question()` — pure loop + event emission |
| **`core/llm.py`** | `call_llm_and_stream()`, `verify_response()` — take an `emit` callable, no UI code |
| **`core/job.py`** | `RunJob` class: results, failed questions, score/token/time aggregates |
| **`core/models.py`** | Pydantic models for the benchmark config, built from env vars via `EndpointsConfig.from_env()` |
| **`core/utils.py`** | `format_time()`, `save_results(job, path)` |
| **`config.py`** | `.env` loading + questions + `load_endpoints_config()` (config lives in the process environment, not the browser) |

Frontend under `web/src/` (Vite + Vue 3, plain JS, Pico.css):

| File | Purpose |
|---|---|
| **`composables/useBenchmark.js`** | Singleton reactive state + EventSource client (SSE event → UI state), snapshot-rebuild reconnect |
| **`composables/useTheme.js`** | Dark/light theme singleton (persisted to `localStorage`, applied via `data-theme`) |
| **`App.vue`** | Sticky `AppBar` + grid: sticky stats column (chronometer, metric cards) beside `StreamPanel` and progress bar |
| **`components/`** | `AppBar` (brand left; status pill, play/stop run controls, settings cog, theme toggle right), `StreamPanel` (autoscroll stream box), `QuestionCard`, `Chronometer` (idle/ticking/checking/frozen), `MetricCard`, `RunControls` (play icon + stop icon, stop only active while running), `SettingsDialog` (native `<dialog>`; edit benchmark + verifier settings via `/api/settings`, opens from the cog, disabled while running), `ThemeToggle` |
| **`styles/theme.css`** | Pico CSS variable overrides (palette formerly in `.streamlit/config.toml`) + metric card / badge / stream box classes |

## HTTP API

- `GET /api/config` — models, sampling summary, question count, `runStatus`, `hasResults`
- `GET /api/settings` — raw `.env` values for the settings dialog (empty strings when unset)
- `POST /api/settings` — save settings: rewrites `.env` (regenerated from a fixed template, empty optional keys commented out) and refreshes `os.environ` so the next run uses them (409 while running; 422 with pydantic detail on invalid values). Real process env vars set before startup keep precedence until restart.
- `POST /api/run` — start a run (202 `{runId}`; 409 if running; 400/422 on bad config)
- `POST /api/run/stop` — cancel the active run (202; 409 if not running). The run loop aborts between questions and mid-stream, then emits `run_finished` with `cancelled: true` (partial results are not saved). `runStatus` becomes `cancelled`.
- `GET /api/run` — status snapshot + all buffered events (client rebuilds UI from this)
- `GET /api/run/events` — SSE stream (`since` query or `Last-Event-ID` header for replay)
- `GET /api/results` — download `results.json`
- Everything else → SPA static files

## SSE event contract

Events carry an SSE `id:` (sequence number). `run_started` → `model_started` → per question `question_started` / `token` / `answer_done` / `question_done` → `run_finished`; `run_error` for per-question failures. The client clock live-ticks between `question_started` and `answer_done`, shows `Checking…` between `answer_done` and `question_done`, and accumulates only `answer_done.took` — verifier time is excluded.

## Data files

- **`questions.json`** — Array of `{ uuid, question, expected }` objects. UUIDs are sequential (`01234567-0123-0123-0123-0000000000XX`). 30 questions from matchingpennies.com.
- **`.env`** (project root, gitignored; see `.env.example`) — Platform-independent benchmark configuration. Keys: `SERVER_URL`, `ENDPOINT` (or comma-separated `ENDPOINTS`), `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`, and the verifier block `VERIFIER_AGENT_SYSPROMPT`, `VERIFIER_AGENT_SERVERURL`, `VERIFIER_AGENT_ENDPOINT`. Loaded via `python-dotenv` at server startup (real env vars take precedence) and parsed by `EndpointsConfig.from_env()` into the pydantic models; string values are coerced, invalid values raise a `ValidationError` returned as HTTP 422.
- **`results.json`** — Generated output, gitignored. Saved once at the end of a run (`save_results()` in `core/bench.py`).

## Key patterns & conventions

- **LiteLLM abstraction**: All LLM calls use `litellm.completion()` with `openai/<model>` prefix and `api_base` pointing to a compatible server (e.g., llama.cpp). `api_key="none"` — no real API keys used. The benchmark loop runs sync litellm in a worker thread (`asyncio.to_thread`) publishing to the `EventBus`; the SSE endpoint subscribes on the event loop.
- **Event emission**: core modules never render — they call an injected `emit(event: dict)`. Keep `src/core/` free of FastAPI and browser concerns; `src/api/` owns transport.
- **Single run**: one run at a time, state in memory (`RunState`); restarting the server abandons the active run. Cancellation goes through `POST /api/run/stop`, which sets a `threading.Event` the benchmark loop and stream reader poll (`RunCancelled` in `core/llm.py`).
- **Frontend state**: one reactive singleton in `useBenchmark.js`; components read/mutate it directly. SSE reconnects by fetching the `/api/run` snapshot and replaying all events, so no client-side dedupe is needed. Markdown answers go through `marked` + `DOMPurify` (never `v-html` raw).
- **Type annotations**: Explicit `: str`, `: float`, `: Dict` etc. with `typing` imports (Python side).

## Gotchas

- The verifier agent and the target LLM can be **different endpoints** — `serverUrl` is the target, `verifierAgent.serverUrl` is the verifier.
- Token counting relies on `stream_options={"include_usage": True}`: the server reports usage in the final stream chunk, read via `_extract_total_tokens()`. Servers that ignore `include_usage` report 0 tokens.
- There is **no `sleep()` throttle** in the streaming loop anymore: token events go straight to SSE, and the client re-renders markdown at most once per animation frame.
- The chronometer is pure frontend state (`Chronometer.vue`) driven by event boundaries; its source of truth is the cumulative `took` values (LLM answer time only, verifier time excluded).
- Look & feel is Pico.css + `web/src/styles/theme.css` (colors, Inter/JetBrains Mono, 8px radii, light + dark). Fonts are self-hosted via `@fontsource/*`. `index.html` sets `data-theme` from the saved `benchmarkit:theme` choice, falling back to `prefers-color-scheme`; the `ThemeToggle` in the app bar switches and persists it.
- The app reuses the same `SERVER_URL` for both the target LLM and the verifier's `api_base` unless `VERIFIER_AGENT_SERVERURL` is set.
- `web/dist` must exist (or `npm run build` run) for FastAPI to serve the UI; the API works without it. In dev use the Vite server instead — it proxies `/api`.
- The old Streamlit app is gone (`src/ui.py`, `.streamlit/`, `streamlit` dep removed); run state is no longer in `st.session_state`.

## Adding questions

Add entries to `questions.json` under the `"questions"` key. Each needs a unique `uuid`, the `question` text, and the `expected` answer. UUIDs should follow the existing sequential pattern.

## Adding endpoints

Edit `.env` — set the model name in `ENDPOINT`. `SERVER_URL` should point to your llama.cpp or compatible inference server. Sampling parameters (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`) are optional and forwarded to `litellm.completion()` for the tested model. `REASONING` is optional too and sets the reasoning effort (`off`/`low`/`medium`/`high`/`xhigh`).
