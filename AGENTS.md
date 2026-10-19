# AGENTS.md — benchmarkit

## Project overview

A FastAPI + Vue 3 app that benchmarks LLM endpoints by streaming questions, collecting responses, and verifying them against expected answers using a secondary LLM verifier agent. Question sets are JSON files in `resources/` (active set chosen by `QUESTION_SET`, default `legacy`) and configuration comes from `.env` on server startup. The SPA has two tabs: **Benchmark** (run + live SSE progress) and **Chat** (interactive streaming chat with the configured model). The frontend is a Vite-built SPA served by FastAPI.

## Commands

```bash
# Setup — this project uses a pyenv virtualenv named `benchmarkit`
# (no .venv directory). If pyenv is configured, `python` already resolves
# to it; otherwise: pyenv shell benchmarkit
python --version   # sanity check: should be the benchmarkit env
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
| **`core/chat.py`** | `stream_chat()` — one streamed chat reply, emits `token` / `reasoning` / `done` / `error`; reuses `_extract_usage`/`_extract_reasoning` from `core/llm.py` |
| **`core/job.py`** | `RunJob` class: results, failed questions, score/token/time aggregates |
| **`core/models.py`** | Pydantic models for the benchmark config, built from env vars via `EndpointsConfig.from_env()` |
| **`core/utils.py`** | `format_time()`, `save_results(job, path)` |
| **`config.py`** | `.env` loading + question-set discovery/loading (`QUESTION_SET`, `resources/*.json`) + `load_endpoints_config()` (config lives in the process environment, not the browser) |

Frontend under `web/src/` (Vite + Vue 3, plain JS, Pico.css):

| File | Purpose |
|---|---|
| **`composables/useBenchmark.js`** | Singleton reactive state (incl. active tab) + EventSource client (SSE event → UI state), snapshot-rebuild reconnect |
| **`composables/useChat.js`** | Chat singleton: message list, POST + SSE-style fetch pump to `/api/chat`, phases `idle`/`waiting`/`reasoning`/`answering`, rAF-throttled markdown re-render |
| **`composables/useAutoScroll.js`** | Autoscroll helper shared by stream/chat panels |
| **`composables/useTheme.js`** | Dark/light theme singleton (persisted to `localStorage`, applied via `data-theme`) |
| **`composables/useSplitter.js`** | Draggable horizontal separator in Chat (output ↔ input row): pointer + keyboard resize, clamped, persisted to `localStorage`, double-click resets to the default 4:1 ratio |
| **`lib/markdown.js`** | Shared markdown renderer: `marked` + `DOMPurify` + `marked-katex-extension` (KaTeX) + `marked-highlight` (highlight.js, `lib/hljs.js`) + optional Mermaid fences (`lib/mermaid.js`) |
| **`App.vue`** | Sticky `AppBar` + tab views: benchmark grid (sticky stats column: chronometer, metric cards) beside `StreamPanel`, and `ChatView` (`v-show` on `state.activeTab`) |
| **`components/`** | `AppBar` (brand left; centered `TabNav` — Benchmark/Chat — then status pill, settings cog, theme toggle right), `StreamPanel` (autoscroll stream box), `ChatView` + `ChatInput` (chat transcript + composer, send/stop), `MarkdownStream` (renders a live markdown message), `QuestionCard`, `Chronometer` (idle/ticking/checking/frozen), `MetricCard`, `RunControls` (play icon + stop icon, stop only active while running; rendered centered in the run-summary card of the stats column, below the sampling-parameters caption), `SettingsDialog` (native `<dialog>`; edit benchmark + verifier + question-set settings via `/api/settings` and `/api/question-sets`, opens from the cog, disabled while running), `ThemeToggle` |
| **`styles/theme.css`** | Pico CSS variable overrides (palette formerly in `.streamlit/config.toml`) + metric card / badge / stream box classes |

## HTTP API

- `GET /api/config` — models, sampling summary, question count, `runStatus`, `hasResults`
- `GET /api/settings` — raw `.env` values for the settings dialog (empty strings when unset)
- `POST /api/settings` — save settings: rewrites `.env` (regenerated from a fixed template, empty optional keys commented out) and refreshes `os.environ` so the next run uses them (409 while running; 422 with pydantic detail on invalid values). Real process env vars set before startup keep precedence until restart.
- `GET /api/question-sets` — available question sets (`{id, label, count}` per `resources/*.json`) + `current` selection
- `POST /api/question-sets` — select the active question set (persists `QUESTION_SET` to `.env`; 409 while running; 422 unknown id)
- `GET /api/models?serverUrl=&apiKey=` — list model ids from an OpenAI-compatible `/models` endpoint (used by the settings dialog when the server URL or API key changes; the optional `apiKey` is sent as `Authorization: Bearer`; 502 on fetch failure)
- `POST /api/run` — start a run (202 `{runId}`; 409 if running; 400/422 on bad config)
- `POST /api/run/stop` — cancel the active run (202; 409 if not running). The run loop aborts between questions and mid-stream, then emits `run_finished` with `cancelled: true` (partial results are not saved). `runStatus` becomes `cancelled`.
- `GET /api/run` — status snapshot + all buffered events (client rebuilds UI from this)
- `GET /api/run/events` — SSE stream (`since` query or `Last-Event-ID` header for replay)
- `POST /api/chat` — stream one chat reply from the first configured model as SSE (body: full `messages` history; 409 while a run is in progress). Worker-thread litellm feeds an `asyncio.Queue`; client disconnect sets `cancel_event` and closes the upstream stream.
- `GET /api/results` — download `results.json`
- Everything else → SPA static files

## SSE event contract

Events carry an SSE `id:` (sequence number). `run_started` → `model_started` → per question `question_started` / `token` / `answer_done` / `question_done` → `run_finished`; `run_error` for per-question failures. The client clock live-ticks between `question_started` and `answer_done`, shows `Checking…` between `answer_done` and `question_done`, shows `Starting…` between `run_started`/`model_started` and the first `question_started` (the backend warms the model up with a `max_tokens=1` request first, so model-loading time is excluded from elapsed time), and accumulates only `answer_done.took` — verifier time is excluded.

`answer_done` and `question_done` carry the timing payload: `took` (request start → last **generated** chunk, TTFT included, `time.monotonic()`), `ttft` (request start → first generated chunk), `genTook` (first → last generated chunk — the decode window; **reasoning/thinking deltas open and extend this window** even though they are not answer text, because `completion_tokens` counts them), `tokens` (**completion tokens only**, never `total_tokens`), `promptTokens`, and `tokensEstimated` (true when the server omitted usage and tokens were estimated at ~4 chars/token). `stats.tokenRate` is the **average of the per-question rates** (`tokens / genTook`) over questions with a measurable decode window — computed entirely app-side from `time.monotonic()` chunk arrivals; only the token count comes from the server's usage chunk.

Chat SSE (`/api/chat`) is a separate, simpler contract: `token` (answer text delta) / `reasoning` (emitted once, the first time reasoning deltas appear — reasoning text itself is never sent) → exactly one terminal `done` (`took`, `ttft`, `tokens`, `tokensEstimated`) or `error`. Nothing is emitted when the client disconnected (cancelled).

## Data files

- **`resources/*.json`** — question sets, one file per set (file stem = set id): `legacy.json` (30 questions from matchingpennies.com, the default), `78qcm.json`, `122qcm.json`, `30hardmath.json`. Each file is `{"questions": [{ uuid, question, expected, ... }]}`; extra keys like `category`/`subcategory` are allowed. UUIDs in `legacy` are sequential (`01234567-0123-0123-0123-0000000000XX`). The active set is chosen by `QUESTION_SET` (default `legacy`).
- **`.env`** (project root, gitignored; see `.env.example`) — Platform-independent benchmark configuration. Keys: `SERVER_URL`, `SERVER_URL_API_KEY`, `ENDPOINT` (or comma-separated `ENDPOINTS`), `LLM_SYSPROMPT` (system prompt for benchmark + chat, defaults to `You are a helpful assistant.`), `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`, `QUESTION_SET`, and the verifier block `VERIFIER_AGENT_SYSPROMPT`, `VERIFIER_AGENT_SERVERURL`, `VERIFIER_AGENT_SERVERURL_API_KEY`, `VERIFIER_AGENT_ENDPOINT`. API keys are optional (empty → litellm sends `"none"`, like keyless llama.cpp servers) and are forwarded as the `api_key` of every `completion()` call. Loaded via `python-dotenv` at server startup (real env vars take precedence) and parsed by `EndpointsConfig.from_env()` into the pydantic models; string values are coerced, invalid values raise a `ValidationError` returned as HTTP 422.
- **`results.json`** — Generated output, gitignored. Saved once at the end of a run (`save_results()` in `core/bench.py`). Each response carries `took`, `ttft`, `genDuration`, `completionTokens` (aliased as `consumedTokens` for older consumers), `promptTokens`, and `tokensEstimated`.

## Key patterns & conventions

- **LiteLLM abstraction**: All LLM calls use `litellm.completion()` with `openai/<model>` prefix and `api_base` pointing to a compatible server (e.g., llama.cpp). `api_key="none"` — no real API keys used. The benchmark loop runs sync litellm in a worker thread (`asyncio.to_thread`) publishing to the `EventBus`; the SSE endpoint subscribes on the event loop.
- **Event emission**: core modules never render — they call an injected `emit(event: dict)`. Keep `src/core/` free of FastAPI and browser concerns; `src/api/` owns transport.
- **Single run**: one run at a time, state in memory (`RunState`); restarting the server abandons the active run. Cancellation goes through `POST /api/run/stop`, which sets a `threading.Event` the benchmark loop and stream reader poll (`RunCancelled` in `core/llm.py`). Chat (`/api/chat`) uses the same pattern (worker thread + `asyncio.Queue` + `cancel_event`) and is rejected with 409 while a run is in progress.
- **Frontend state**: one reactive singleton in `useBenchmark.js`; components read/mutate it directly. SSE reconnects by fetching the `/api/run` snapshot and replaying all events, so no client-side dedupe is needed. Markdown answers go through `marked` + `DOMPurify` (never `v-html` raw). The Chat tab has its own singleton (`useChat.js`) and shares the markdown pipeline in `lib/markdown.js`.
- **Type annotations**: Explicit `: str`, `: float`, `: Dict` etc. with `typing` imports (Python side).

## Gotchas

- The verifier agent and the target LLM can be **different endpoints** — `serverUrl` is the target, `verifierAgent.serverUrl` is the verifier.
- Token counting relies on `stream_options={"include_usage": True}`: the server reports usage in the final stream chunk, read via `_extract_usage()`. The rate uses **`completion_tokens`** only (`total_tokens` = prompt + completion would inflate it); when the server omits it, tokens are derived as `total - prompt` or estimated from text (~4 chars/token) and flagged `tokensEstimated`. Timing is `time.monotonic()` anchored to content chunks, so the trailing usage-only chunk never extends `took`.
- There is **no `sleep()` throttle** in the streaming loops: token events go straight to SSE, and the client re-renders markdown at most once per animation frame.
- The chronometer is pure frontend state (`Chronometer.vue`) driven by event boundaries; its source of truth is the cumulative `took` values (LLM answer time only, verifier time excluded).
- Look & feel is Pico.css + `web/src/styles/theme.css` (colors, Inter/JetBrains Mono, 8px radii, light + dark). Fonts are self-hosted via `@fontsource/*`. `index.html` sets `data-theme` from the saved `benchmarkit:theme` choice, falling back to `prefers-color-scheme`; the `ThemeToggle` in the app bar switches and persists it.
- Rendered answers can contain LaTeX (`marked-katex-extension`), fenced code highlighted by highlight.js, and ` ```mermaid ` fences rendered by Mermaid — all through the single `lib/markdown.js` renderer.
- The app reuses the same `SERVER_URL` for both the target LLM and the verifier's `api_base` unless `VERIFIER_AGENT_SERVERURL` is set.
- `web/dist` must exist (or `npm run build` run) for FastAPI to serve the UI; the API works without it. In dev use the Vite server instead — it proxies `/api`.

## Adding questions

Add entries to a question-set file in `resources/` (e.g. `resources/legacy.json`) under its `"questions"` key. Each needs a unique `uuid`, the `question` text, and the `expected` answer. A new `resources/<name>.json` file automatically becomes a selectable question set (id = file stem).

## Adding endpoints

Edit `.env` — set the model name in `ENDPOINT`. `SERVER_URL` should point to your llama.cpp or compatible inference server. Sampling parameters (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`) are optional and forwarded to `litellm.completion()` for the tested model. `REASONING` is optional too and sets the reasoning effort (`off`/`low`/`medium`/`high`/`xhigh`).
