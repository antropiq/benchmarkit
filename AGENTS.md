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

# Optional MCP tools for the Chat tab (mcp SDK + playwright + browser binaries)
pip install -e ".[mcp]"
playwright install chromium   # once, downloads Chromium

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
| **`core/chat.py`** | `stream_chat()` — one streamed chat reply, emits `token` / `reasoning` (one-shot thinking flag) / `reasoning_delta` (thinking text, streamed per delta) / `done` / `error` (+ `tool_call` — carrying `iteration` / `maxIterations` — and `tool_result` per call, plus a cumulative `usage` (`tokens` / `decode` seconds / `consumed`) event right after each round's generation ends — before the tool executes — in the MCP tool loop, and a `TOOL_NOTICE` appended to the system prompt only while tools are active; reasoning never enters `token` frames nor the assistant turn fed back to the model; `done` carries the cumulative `decode` window so the chat tok/s excludes tool execution time, and the per-round-summed `consumed` (prompt + completion) token count); reuses `_extract_usage`/`_extract_reasoning` from `core/llm.py` |
| **`core/mcp_hub.py`** | `discover_connectors()` (docstring scan of `mcp_connectors/*_mcp.py`), `load_externals()` / `all_connectors()` / `set_externals_enabled()` (`mcp_connectors/externals.json`: kilo-style `remote`/`local` connectors with an `enable` flag), OpenAI tool-format helpers, `McpManager`: spawns one subprocess per built-in connector for the app's lifetime, connects externals over Streamable HTTP (remote) or stdio (local, launched via `stdio_client`), supports `start_connector()`/`stop_connector()` for hot toggles, and bridges sync chat threads to persistent MCP clients on a private asyncio thread |
| **`core/job.py`** | `RunJob` class: results, failed questions, score/token/time aggregates |
| **`core/models.py`** | Pydantic models for the benchmark config, built from env vars via `EndpointsConfig.from_env()` |
| **`core/utils.py`** | `format_time()`, `save_results(job, path)` |
| **`config.py`** | `.env` loading + question-set discovery/loading (`QUESTION_SET`, `resources/*.json`, skipping `memory_registration.json`) + memory-bullet registration (`load/save_memory_registration*`, self-healing 9-slot `resources/memory_registration.json`) + `load_endpoints_config()` (config lives in the process environment, not the browser) |

Frontend under `web/src/` (Vite + Vue 3, plain JS, Pico.css):

| File | Purpose |
|---|---|
| **`composables/useBenchmark.js`** | Singleton reactive state (incl. active tab) + EventSource client (SSE event → UI state), snapshot-rebuild reconnect |
| **`composables/useChat.js`** | Chat singleton: message list, POST + SSE-style fetch pump to `/api/chat`, phases `idle`/`waiting`/`reasoning`/`answering`, rAF-throttled markdown re-render, pending-image attach (client MIME/size validation, multimodal content parts in history), conversation-scoped MCP toggles (`state.mcp`: snapshot of the Settings-activated connectors from `/api/mcp`, refetched on Chat-tab entry, MCP-settings save and restart, all ON by default, sticky-locked by the first accepted send, unlocked by restart with the toggle states kept — only a page reload re-defaults them to ON; the ON ids are sent as `mcpEnabled` in the chat body, never persisted) |
| **`composables/useAutoScroll.js`** | Autoscroll helper shared by stream/chat panels |
| **`composables/useMemory.js`** | Memory-bullet singleton: `{color, content}` slot registration from `/api/memory` (always 9 slots, refetched on Chat-tab entry), `saveMemory()` replaces the registration from the POST response, sets `error` on failure and returns an ok flag for the dialog |
| **`composables/useTheme.js`** | Dark/light theme singleton (persisted to `localStorage`, applied via `data-theme`) |
| **`composables/useSplitter.js`** | Draggable horizontal separator in Chat (output ↔ input row): pointer + keyboard resize, clamped, persisted to `localStorage`, double-click resets to the default 4:1 ratio |
| **`lib/markdown.js`** | Shared markdown renderer: `marked` + `DOMPurify` + `marked-katex-extension` (KaTeX) + `marked-highlight` (highlight.js, `lib/hljs.js`) + optional Mermaid fences (`lib/mermaid.js`) |
| **`App.vue`** | Sticky `AppBar` + tab views: benchmark grid (sticky stats column: chronometer, metric cards) beside `StreamPanel`, and `ChatView` (`v-show` on `state.activeTab`) |
| **`components/`** | `AppBar` (brand left; centered `TabNav` — Benchmark/Chat — then status pill, settings cog, theme toggle right), `StreamPanel` (autoscroll stream box), `ChatView` + `ChatInput` (chat transcript + composer, send/stop; the textarea column carries `MemoryBullets` under the textarea and hosts the slot dialog), `ChatSidePanel` (fixed-width left panel of the input row: t/s + tokens badges — kept live during the MCP tool loop via `usage` events — image-attach button (hidden file input, disabled unless `supportsVision`), copy/restart buttons, send/stop — width `--bi-chat-panel-width`), `ChatMcpPanel` (scrollable panel between `ChatSidePanel` and the textarea listing only the Settings-activated MCP connectors, each with a conversation-scoped ON/OFF switch + a status dot beside it — green running, red error, gray otherwise, non-running rows dimmed, hidden when nothing is activated, locked once the first message is sent, unlocked again by restart — own width `--bi-chat-mcp-width`), `MemoryBullets` (centered row of 9 round numbered slots under the textarea, backed by `useMemory`: empty slot → click opens `RegistrationMemoryDlg`, filled → click appends the content to the draft via `useChat`'s `appendDraft`, filled + Ctrl/Cmd+click → dialog; background = slot color with luminance-picked black/white number, native `title` tooltip truncated at 150 chars), `RegistrationMemoryDlg` (native `<dialog>` editing one slot — markdown-message textarea + `<input type="color">`; Save persists via `/api/memory` then closes and appends the content to the draft unless it is empty, Cancel/Esc discards), `ChatToolPanel` (always-visible timeline left of the chat stream of everything the stream does not show: streamed `reasoning_delta` thinking flows interleaved with the MCP tool-call badges (badge tooltip = truncated args + result preview) in emission order, built from the per-message `blocks` array, reasoning as plain pre-wrap muted mono text (never markdown, never copied by Copy conversation), `GaugeBar` bottom-pinned and always visible showing tool-loop iterations vs `MAX_MCP_ITERATION`, cleared on restart; width `--bi-chat-tool-width` = the input row's whole left zone (stats panel + MCP panel − one gap, via a `.chat-shell:has(.chat-mcp-panel)` override, falling back to `--bi-chat-panel-width` without a visible MCP panel) so the chat stream and the textarea columns stay aligned), `GaugeBar` (reusable horizontal meter: fixed green→yellow→orange→red gradient masked to `value`/`max`, so the fill tip's color reflects the level), `MarkdownStream` (renders a live markdown message), `QuestionCard`, `Chronometer` (idle/ticking/checking/frozen), `MetricCard`, `RunControls` (play icon + stop icon, stop only active while running; rendered centered in the run-summary card of the stats column, below the sampling-parameters caption), `SettingsDialog` (native `<dialog>`; edit benchmark + verifier + question-set + MCP-connector settings via `/api/settings`, `/api/question-sets` and `/api/mcp`, opens from the cog, disabled while running), `ThemeToggle` |
| **`styles/theme.css`** | Pico CSS variable overrides (palette formerly in `.streamlit/config.toml`) + metric card / badge / stream box classes |

## HTTP API

- `GET /api/config` — models, sampling summary, question count, `supportsVision`, `runStatus`, `hasResults`
- `GET /api/settings` — raw `.env` values for the settings dialog (empty strings when unset)
- `POST /api/settings` — save settings: rewrites `.env` (regenerated from a fixed template, empty optional keys commented out) and refreshes `os.environ` so the next run uses them (409 while running; 422 with pydantic detail on invalid values). Real process env vars set before startup keep precedence until restart.
- `GET /api/mcp` — built-in connectors scanned from `mcp_connectors/` plus external ones from `mcp_connectors/externals.json` (`{id, label, description, port, endpoint, command, external, kind, active, status, error}`; `headers`/`env` are never returned) + `installed` flag + `maxIterations` + `externalsError` (validation errors, or `null`)
- `POST /api/mcp` — activate connectors for the Chat tab and set the iteration cap. Built-in activations persist to `MCP_ENABLED` / `MAX_MCP_ITERATION` in `.env`; external toggles write the `enable` flag back to `mcp_connectors/externals.json` and hot start/stop the connector. Externals are activated solely by their file `enable` flag, never `MCP_ENABLED` (409 while running; 422 unknown id)
- `GET /api/question-sets` — available question sets (`{id, label, count}` per `resources/*.json`, excluding `memory_registration.json`) + `current` selection
- `POST /api/question-sets` — select the active question set (persists `QUESTION_SET` to `.env`; 409 while running; 422 unknown id)
- `GET /api/memory` — memory-bullet registration: `{"registration": [9 × {color, content}]}` from `resources/memory_registration.json` (missing/corrupt file self-heals to 9 normalized defaults, written back)
- `POST /api/memory` — persist one memory slot (`{index 0..8, color #rrggbb, content}`; saving empty content clears the slot, the file always keeps exactly 9 slots — no DELETE). Returns the full GET payload; 422 on bad index/color or content over 32 000 chars. Not 409-gated while a run is in progress (chat-adjacent)
- `GET /api/models?serverUrl=&apiKey=` — list model ids from an OpenAI-compatible `/models` endpoint (used by the settings dialog when the server URL or API key changes; the optional `apiKey` is sent as `Authorization: Bearer`; 502 on fetch failure)
- `POST /api/run` — start a run (202 `{runId}`; 409 if running; 400/422 on bad config)
- `POST /api/run/stop` — cancel the active run (202; 409 if not running). The run loop aborts between questions and mid-stream, then emits `run_finished` with `cancelled: true` (partial results are not saved). `runStatus` becomes `cancelled`.
- `GET /api/run` — status snapshot + all buffered events (client rebuilds UI from this)
- `GET /api/run/events` — SSE stream (`since` query or `Last-Event-ID` header for replay)
- `POST /api/chat` — stream one chat reply from the first configured model as SSE (body: full `messages` history; 409 while a run is in progress). Optional `mcpEnabled` (list of connector ids): the conversation-scoped allowlist from the Chat tab's MCP panel, intersected with the Settings-activated ids to pick the tool sources (absent = no filter = all activated connectors; unknown/stale ids are silently dropped). User message `content` is a plain string or a list of OpenAI multimodal parts (`text` + `image_url` with a base64 data URL `data:image/(png|jpeg|webp|gif);base64,…`, decoded ≤ 10 MB; image-only messages allowed) — image parts are validated with 422 and passed through litellm untouched; assistant content must stay a plain string. Worker-thread litellm feeds an `asyncio.Queue`; client disconnect sets `cancel_event` and closes the upstream stream.
- `GET /api/results` — download `results.json`
- Everything else → SPA static files

## SSE event contract

Events carry an SSE `id:` (sequence number). `run_started` → `model_started` → per question `question_started` / `token` / `answer_done` / `question_done` → `run_finished`; `run_error` for per-question failures. The client clock live-ticks between `question_started` and `answer_done`, shows `Checking…` between `answer_done` and `question_done`, shows `Starting…` between `run_started`/`model_started` and the first `question_started` (the backend warms the model up with a `max_tokens=1` request first, so model-loading time is excluded from elapsed time), and accumulates only `answer_done.took` — verifier time is excluded.

`answer_done` and `question_done` carry the timing payload: `took` (request start → last **generated** chunk, TTFT included, `time.monotonic()`), `ttft` (request start → first generated chunk), `genTook` (first → last generated chunk — the decode window; **reasoning/thinking deltas open and extend this window** even though they are not answer text, because `completion_tokens` counts them), `tokens` (**completion tokens only**, never `total_tokens`), `consumed` (**prompt + completion** — what the answer actually cost; the figure the UI displays), `promptTokens`, and `tokensEstimated` (true when the server omitted usage and tokens were estimated at ~4 chars/token). `stats.tokenRate` is the **average of the per-question rates** (`tokens / genTook`) over questions with a measurable decode window — computed entirely app-side from `time.monotonic()` chunk arrivals; only the token count comes from the server's usage chunk.

Chat SSE (`/api/chat`) is a separate, simpler contract: `token` (answer text delta) / `reasoning` (emitted once, the first time reasoning deltas appear — a phase flag, carries no text) + `reasoning_delta` (`text`, one frame per thinking delta — reasoning text streams live to the Chat tab's left timeline panel and is never answer text nor fed back to the model) → exactly one terminal `done` (`took`, `ttft`, `decode`, `tokens`, `tokensEstimated`, `consumed`, `consumedEstimated`) or `error`. `done.decode` is the cumulative generation (decode) window in seconds across all tool-loop rounds — the chat tok/s denominator; `took - ttft` would wrongly include tool execution and inter-round re-prefill gaps once tools ran (it equals `decode` for a single-round reply). `done.consumed` is the **tokens actually consumed** (prompt + completion, summed **per round** — each tool round re-sends and re-processes the whole history, so its prompt counts again; a round's server-reported `total_tokens` is exactly that figure), what the session token badge shows, while `tokens` stays completion-only (the tok/s numerator); `consumedEstimated` flags rounds whose prompt or generation had to be char-estimated (servers reporting no usage at all). When MCP connectors are active for Chat, `tool_call` (`id`, `server`, `tool`, `argsPreview`, `iteration`, `maxIterations`) and `tool_result` (`id`, `ok`, `preview`) frames appear per executed tool call, and a `usage` (`tokens`, `decode`, `consumed`, `consumedEstimated`) frame with the cumulative figures opens each tool round right after its generation ends — before the tool executes, so the badges tick when generation stops (content, reasoning and tool-call argument deltas all open/extend the decode window; tool execution time excluded). Nothing is emitted when the client disconnected (cancelled).

## Data files

- **`resources/*.json`** — question sets, one file per set (file stem = set id): `legacy.json` (30 questions from matchingpennies.com, the default), `78qcm.json`, `122qcm.json`, `30hardmath.json`. Each file is `{"questions": [{ uuid, question, expected, ... }]}`; extra keys like `category`/`subcategory` are allowed. UUIDs in `legacy` are sequential (`01234567-0123-0123-0123-0000000000XX`). The active set is chosen by `QUESTION_SET` (default `legacy`).
- **`resources/memory_registration.json`** — the Chat tab's 9 memory bullets: `{"registration": [{color, content} ×9]}` (`#rrggbb` colors, default `#808080` with empty content). Managed only through `/api/memory`; `load_memory_registrations()` in `src/config.py` normalizes and self-heals the file on every read (pad/truncate to 9, coerce malformed entries). `list_question_sets()` skips this filename so it never appears as a question set.
- **`.env`** (project root, gitignored; see `.env.example`) — Platform-independent benchmark configuration. Keys: `SERVER_URL`, `SERVER_URL_API_KEY`, `ENDPOINT`, `LLM_SYSPROMPT` (system prompt for benchmark + chat, defaults to `You are a helpful assistant.`), `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`, `SUPPORTS_VISION` (declare the endpoint accepts image input — gates the Chat image-attach button), `QUESTION_SET`, the MCP block `MCP_ENABLED` (comma-separated connector ids activated for Chat) and `MAX_MCP_ITERATION` (default 25), and the verifier block `VERIFIER_AGENT_SYSPROMPT`, `VERIFIER_AGENT_SERVERURL`, `VERIFIER_AGENT_SERVERURL_API_KEY`, `VERIFIER_AGENT_ENDPOINT`. API keys are optional (empty → litellm sends `"none"`, like keyless llama.cpp servers) and are forwarded as the `api_key` of every `completion()` call. Loaded via `python-dotenv` at server startup (real env vars take precedence) and parsed by `EndpointsConfig.from_env()` into the pydantic models; string values are coerced, invalid values raise a `ValidationError` returned as HTTP 422.
- **`results.json`** — Generated output, gitignored. Saved once at the end of a run (`save_results()` in `core/bench.py`). Each response carries `took`, `ttft`, `genDuration`, `completionTokens` (aliased as `consumedTokens` for older consumers), `promptTokens`, and `tokensEstimated`.
- **`mcp_connectors/externals.json`** (gitignored; may hold API keys) — Declarative external MCP connectors, an object keyed by id: `{ type: "remote", enable, url, headers? }` (Streamable HTTP) or `{ type: "local", enable, command, args?, env? }` (stdio, launched via the MCP SDK's `stdio_client`; `command` is an argv array or a shell string). `enable` is the single on/off source of truth (checked at startup, and rewritten in place by `POST /api/mcp` without disturbing other keys). An id that collides with a built-in connector, an unknown `type`, or a remote without an `http(s)` url is skipped with a note surfaced as `externalsError`. A missing file means no externals; invalid JSON yields no externals plus a file-level error (never overwritten by a toggle).

## Key patterns & conventions

- **LiteLLM abstraction**: All LLM calls use `litellm.completion()` with `openai/<model>` prefix and `api_base` pointing to a compatible server (e.g., llama.cpp). `api_key="none"` — no real API keys used. The benchmark loop runs sync litellm in a worker thread (`asyncio.to_thread`) publishing to the `EventBus`; the SSE endpoint subscribes on the event loop.
- **Event emission**: core modules never render — they call an injected `emit(event: dict)`. Keep `src/core/` free of FastAPI and browser concerns; `src/api/` owns transport.
- **Single run**: one run at a time, state in memory (`RunState`); restarting the server abandons the active run. Cancellation goes through `POST /api/run/stop`, which sets a `threading.Event` the benchmark loop and stream reader poll (`RunCancelled` in `core/llm.py`). Chat (`/api/chat`) uses the same pattern (worker thread + `asyncio.Queue` + `cancel_event`) and is rejected with 409 while a run is in progress.
- **Frontend state**: one reactive singleton in `useBenchmark.js`; components read/mutate it directly. SSE reconnects by fetching the `/api/run` snapshot and replaying all events, so no client-side dedupe is needed. Markdown answers go through `marked` + `DOMPurify` (never `v-html` raw). The Chat tab has its own singleton (`useChat.js`) and shares the markdown pipeline in `lib/markdown.js`.
- **Type annotations**: Explicit `: str`, `: float`, `: Dict` etc. with `typing` imports (Python side).

## Gotchas

- The verifier agent and the target LLM can be **different endpoints** — `serverUrl` is the target, `verifierAgent.serverUrl` is the verifier.
- Token counting relies on `stream_options={"include_usage": True}`: the server reports usage in the final stream chunk, read via `_extract_usage()`. The rate uses **`completion_tokens`** only (`total_tokens` = prompt + completion would inflate it); when the server omits it, tokens are derived as `total - prompt` or estimated from text (~4 chars/token) and flagged `tokensEstimated`. The displayed **consumed** figure is the opposite pairing: prompt + completion (per-round sum in Chat — every tool round re-sends the history — and simply `prompt + completion` for a single benchmark answer), never mixed into the rate. Timing is `time.monotonic()` anchored to content chunks, so the trailing usage-only chunk never extends `took`.
- There is **no `sleep()` throttle** in the streaming loops: token events go straight to SSE, and the client re-renders markdown at most once per animation frame.
- The chronometer is pure frontend state (`Chronometer.vue`) driven by event boundaries; its source of truth is the cumulative `took` values (LLM answer time only, verifier time excluded).
- Look & feel is Pico.css + `web/src/styles/theme.css` (colors, Inter/JetBrains Mono, 8px radii, light + dark). Fonts are self-hosted via `@fontsource/*`. `index.html` sets `data-theme` from the saved `benchmarkit:theme` choice, falling back to `prefers-color-scheme`; the `ThemeToggle` in the app bar switches and persists it.
- Rendered answers can contain LaTeX (`marked-katex-extension`), fenced code highlighted by highlight.js, and ` ```mermaid ` fences rendered by Mermaid — all through the single `lib/markdown.js` renderer.
- The app reuses the same `SERVER_URL` for both the target LLM and the verifier's `api_base` unless `VERIFIER_AGENT_SERVERURL` is set.
- MCP connector processes start/stop with the app via the FastAPI **lifespan** (`server.py`) — the hub is the port authority: each built-in connector gets a port allocated deterministically from its id within `AUTO_PORT_BASE..AUTO_PORT_BASE + AUTO_PORT_SPAN` (8940–8999, `core/mcp_hub.py`), probed for availability, passed to the child as `--port`/`--host`, and displayed as a port badge in the settings panel; a connector module must accept `--port` and must not assume a port of its own. Logs in `logs/mcp/`. `BENCHMARKIT_DISABLE_MCP=1` skips spawning (set by the test-suite autouse fixture). The `mcp` package is imported lazily so the app runs fine without the extra.
- External connectors (`mcp_connectors/externals.json`) are declared, not spawned as `*_mcp.py` modules: `remote` ones connect over Streamable HTTP (custom `headers` carried via a configured `httpx` client), `local` ones are launched through the SDK's `stdio_client` with the project environment (`connector_environment()`) merged with their `env`. Only `enable: true` ones start at startup; the settings dialog toggles `enable` and calls `McpManager.start_connector()`/`stop_connector()` for a live toggle. Secrets (`headers`/`env`) never leave the server — they are stripped from every API payload and the file is gitignored.
- `web/dist` must exist (or `npm run build` run) for FastAPI to serve the UI; the API works without it. In dev use the Vite server instead — it proxies `/api`.

## Adding questions

Add entries to a question-set file in `resources/` (e.g. `resources/legacy.json`) under its `"questions"` key. Each needs a unique `uuid`, the `question` text, and the `expected` answer. A new `resources/<name>.json` file automatically becomes a selectable question set (id = file stem).

## Adding endpoints

Edit `.env` — set the model name in `ENDPOINT`. `SERVER_URL` should point to your llama.cpp or compatible inference server. Sampling parameters (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`) are optional and forwarded to `litellm.completion()` for the tested model. `REASONING` is optional too and sets the reasoning effort (`off`/`low`/`medium`/`high`/`xhigh`). `SUPPORTS_VISION=1` declares that the model accepts image input and enables the image-attach button in the Chat tab (it can also be toggled in Settings → LLM).
