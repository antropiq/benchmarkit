# AGENTS.md — benchmarkit

## Project overview

A FastAPI + Vue 3 app that benchmarks LLM endpoints: it streams questions to a configured model, collects responses, and verifies them against expected answers with a secondary LLM verifier agent. Question sets are JSON files in `resources/` (active set chosen by `QUESTION_SET`); configuration comes from `.env`. The frontend is a Vite-built SPA served by FastAPI, with two tabs: **Benchmark** (run + live SSE progress) and **Chat** (interactive streaming chat, optionally augmented with MCP connectors). It targets desktop browsers only.

## Commands

```bash
# Setup (pyenv virtualenv named `benchmarkit`, no .venv directory)
pip install -e .
pip install -e ".[test]"     # test deps
pip install -e ".[mcp]"      # optional MCP extras, then: playwright install chromium
pip install -e ".[lsp]"      # optional LSP extras (pygls, lsprotocol)

cd web && npm install && npm run build   # frontend -> web/dist, served by FastAPI
cd web && npm run dev                    # Vite dev server :5173, proxies /api to :8000

benchmarkit                              # serve API + UI on http://127.0.0.1:8000
pytest                                   # test suite (tests/)
```

Dependencies are managed via `pyproject.toml` (hatchling backend). No lint step; the only frontend build is `vite build`.

## Code structure

Python package layout under `src/`, split into a UI-free core and the API:

| Module | Purpose |
|---|---|
| `api/server.py` | `create_app()` (routers + static `web/dist`), MCP connector lifespan start/stop, `run()` entry |
| `api/routes.py` | REST + SSE endpoints, `RunState` (single active run), worker-thread launch |
| `api/events.py` | `EventBus`: seq-numbered in-memory event buffer with replay; SSE formatting |
| `core/bench.py` | Benchmark loop (`run_benchmark()` / `process_question()`), saves `results.json` |
| `core/llm.py` | `call_llm_and_stream()` / `verify_response()` — take an `emit` callable, no UI concerns |
| `core/chat.py` | `stream_chat()` — one streamed chat reply, with MCP tool-loop support |
| `core/mcphub/` | MCP connector management: every connector comes from the declarative `externals.json` (remote Streamable HTTP or local stdio), hot toggles. One concern per module: `constants` (paths/timeouts), `validation` (shared `headers`/`env` + command shapes, also used by lsphub), `discovery` (read/validate/write `externals.json`), `formatting` (MCP tool/result objects → OpenAI/plain text), `process` (`connector_environment`, also used by lsphub), `manager` (`McpManager`) |
| `core/lsphub/` | Language-server management, mirroring `mcphub`: `lsp_servers.json` entries spawned as stdio LSP sessions (handshake, document store, diagnostics inbox), capabilities synthesized into `lsp_<id>_<action>` tools, hot toggles, idle reaper. One concern per module: `constants` (paths/timeouts/relay method names), `discovery` (read/validate/write the file), `tools` (action catalogue), `session` (`DiagnosticsInbox`/`DocumentStore`/`ServerSession`), `features` (server→client handlers incl. the tsserver relay), `actions` (document sync + the per-action requests), `rendering` (LSP structures → text), `tsserver` (relay wire shapes), `manager` (`LspManager`) |
| `core/job.py` | `RunJob`: results, failures, score/token/time aggregates |
| `core/models.py` | Pydantic config models, built from env via `EndpointsConfig.from_env()` |
| `config.py` | `.env` loading, question-set discovery, memory-bullet registration |

Frontend under `web/src/` (Vite + Vue 3, plain JS, Pico.css):

| File | Purpose |
|---|---|
| `composables/` | One reactive singleton per concern: `useBenchmark` (run state + EventSource client), `useChat` (SSE fetch pump), `useMemory`, `useTheme`, `useAutoScroll`, `useSplitter` |
| `lib/markdown.js` | Shared markdown renderer: `marked` + `DOMPurify` + KaTeX + highlight.js + optional Mermaid (never raw `v-html`) |
| `App.vue` | App bar + tab views: benchmark grid (stats column + `StreamPanel`) and `ChatView` |
| `components/` | Benchmark: `StreamPanel`, `QuestionCard`, `Chronometer`, `MetricCard`, `RunControls`. Chat: `ChatView`/`ChatInput`, `ChatSidePanel`/`ChatMcpPanel`/`ChatLspPanel`/`ChatToolPanel`, memory bullets + registration dialog, `MarkdownStream`, `GaugeBar`. Shared: `AppBar`, `SettingsDialog`, `ThemeToggle` |
| `styles/theme.css` | Pico CSS variable overrides + app classes (palette, cards, badges) |

## HTTP API

All app endpoints live under `/api/` (everything else serves the SPA):

- `GET /api/config` — models, sampling summary, question count, `supportsVision`, `contextWindow`, `runStatus`, `hasResults`
- `GET|POST /api/settings` — read/save `.env`-backed settings (rewrites `.env`, refreshes `os.environ`; 409 while running, 422 on invalid values)
- `GET|POST /api/mcp` — list / activate MCP connectors (activation is the entry `enable` flag in `externals.json`, written back with hot start/stop; only `MAX_TOOL_ITERATION` persists to `.env`)
- `GET|POST /api/lsp` — list / activate language servers (activation writes both `LSP_ENABLED` in `.env` and the entry `enable` flag in `lsp_servers.json`, hot start/stop; `env` values never returned) · `GET /api/lsp/log?server=` — tail of one server's log
- `POST /api/mcp/restart` / `POST /api/lsp/restart` — rebuild a hub from a fresh scan of its declaration file (sessions/children are torn down and respawned, so a hand-edited `externals.json`/`lsp_servers.json` applies without restarting the app). Awaited, so the returned payload carries real post-restart statuses. Same payload as the matching GET. 409 while a run is in progress *or* a chat reply is streaming (a restart would kill a tool call mid-flight); 503 when the hub is absent. `BENCHMARKIT_DISABLE_MCP`/`_LSP` blank the scan, so the endpoints stay testable without spawning anything.
- `GET|POST /api/question-sets` — list / select the active question set (persists `QUESTION_SET`)
- `GET|POST /api/memory` — read / update the 9 memory-bullet slots (self-healing file, always exactly 9 slots)
- `GET /api/models?serverUrl=&apiKey=` — list model ids from an OpenAI-compatible server (settings dialog)
- `POST /api/run` / `POST /api/run/stop` — start / cancel the single active run (409 on conflicts)
- `GET /api/run` — status snapshot + all buffered events (UI rebuild point) · `GET /api/run/events` — SSE stream with replay (`since` / `Last-Event-ID`)
- `POST /api/chat` — stream one chat reply as SSE (full `messages` history, optional multimodal image parts, optional `mcpEnabled`/`lspEnabled` allowlists; 409 while a run is in progress)
- `GET /api/results` — download `results.json`

## SSE event contract

Events carry an SSE `id:` (sequence number) so clients can reconnect and replay. Two contracts:

- **Benchmark run**: `run_started` → `model_started` → per question `question_started` / `token` / `answer_done` / `question_done` → `run_finished` (`run_error` for per-question failures). Timing fields (`took`, `ttft`, `genTook`) are measured app-side with `time.monotonic()`; the chronometer accumulates only answer time (model warmup and verifier time excluded). Token fields distinguish completion-only `tokens` (rate numerator) from `consumed` = prompt + completion (the displayed cost).
- **Chat** (`/api/chat`): `token` / `reasoning` (phase flag) / `reasoning_delta` (thinking text, never answer text) → exactly one terminal `done` or `error`. `done` reports the cumulative `decode` window (the chat tok/s denominator, excludes tool execution) and `contextTokens` (context-window occupancy, never summed across rounds). With MCP tools active, `tool_call` / `tool_result` frames plus a cumulative `usage` frame appear per tool round.

## Data files

- **`resources/*.json`** — question sets, one file per set (file stem = set id); each is `{"questions": [{ uuid, question, expected, ... }]}`. A new file automatically becomes a selectable set; `QUESTION_SET` picks the active one (default `legacy`).
- **`resources/memory_registration.json`** — the Chat tab's 9 memory bullets (`{color, content}` slots). Managed only through `/api/memory`, self-heals to 9 normalized slots on every read, and is skipped by question-set discovery.
- **`.env`** (project root, gitignored; see `.env.example`) — all configuration: `SERVER_URL` + API key + `ENDPOINT`, `LLM_SYSPROMPT`, sampling params (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, penalties, `REASONING`), `SUPPORTS_VISION`, `QUESTION_SET`, `LSP_ENABLED`, `MAX_TOOL_ITERATION` (one tool-loop cap shared by MCP + LSP; the legacy `MAX_MCP_ITERATION` is still honoured), and the `VERIFIER_AGENT_*` block. MCP connectors have **no** env key: they activate through `externals.json`. A leftover `MCP_ENABLED` is ignored (and no longer written by the settings dialog). Loaded with `python-dotenv` at startup (real env vars take precedence), parsed by `EndpointsConfig.from_env()` into pydantic models; invalid values surface as HTTP 422.
- **`results.json`** — generated run output, gitignored; saved once at the end of a run.
- **`mcp_connectors/externals.json`** (gitignored; may hold secrets) — the one and only connector declaration, keyed by id: `remote` (Streamable HTTP) or `local` (stdio command); `enable` is the single on/off source of truth (no `.env` key involved). Their `headers`/`env` secrets are never returned by the API.
- **`lsp_connectors/lsp_servers.json`** (gitignored; may hold secrets; `lsp_servers.example.json` is the committed template) — declarative language servers keyed by id: always local stdio (`command` argv, never a shell), with `extensions` (which files the server answers for), absolute `workspaceRoot` (refused when inside a `restricted-paths.txt` prefix), `lazy` spawn flag, `tools` list, `maxDiagnostics`/`idleShutdownSeconds`. A server runs when its `enable` flag **and** `LSP_ENABLED` agree. `env` values are never returned by the API.

## Key patterns & conventions

- **LiteLLM abstraction**: all LLM calls go through `litellm.completion()` with the `openai/<model>` prefix against an OpenAI-compatible server (e.g., llama.cpp); `api_key="none"` — no real API keys.
- **Core/API separation**: core modules never render or touch FastAPI — they call an injected `emit(event: dict)`; `src/api/` owns transport. Sync litellm runs in a worker thread publishing to the `EventBus`; SSE subscribes on the event loop.
- **Single run**: one run at a time, state in memory (`RunState`); cancellation via a `threading.Event` polled by the loop and stream reader. Chat uses the same worker-thread + cancel pattern, and is 409-rejected while a run is active.
- **Frontend state**: one reactive singleton per concern (`useBenchmark`, `useChat`, …); components read/mutate directly. SSE reconnect = refetch the `/api/run` snapshot and replay. Markdown always through `lib/markdown.js` (`marked` + `DOMPurify`), never raw `v-html`.
- **Type annotations**: explicit Python types (`: str`, `: Dict`, `typing` imports).

## Gotchas

- The verifier and the target LLM can be **different endpoints**; without `VERIFIER_AGENT_SERVERURL`, the verifier reuses `SERVER_URL`.
- Token counting relies on `stream_options={"include_usage": True}` (final stream chunk, read via `_extract_usage()`). Keep the distinction straight: the rate uses **completion tokens only**, the displayed cost is **prompt + completion** (`consumed`); missing usage is char-estimated and flagged `tokensEstimated`. Timing is `time.monotonic()` anchored to content chunks, so the trailing usage-only chunk never extends `took`.
- MCP connectors follow the lsphub pattern: no built-in modules, no ports — `remote` entries connect over Streamable HTTP, `local` entries are spawned as **stdio children owned by the client transport** (closing the client's exit stack shuts them down, the hub keeps no process handle of its own). `BENCHMARKIT_DISABLE_MCP=1` skips connecting (used by tests); the `mcp` package is imported lazily. The patchable file locations (`MCP_CONNECTORS_DIR`, `EXTERNALS_PATH`) live on `mcphub.constants` and are read through that module at call time — a test that redirects them must patch `src.core.mcphub.constants`.
- Language servers are managed the same way by the `core/lsphub/` package (`src.core.lsphub`), but LSP is **stdio** (no ports): one private `lsp-hub` asyncio thread owns every session, `lsp_<id>_<action>` tools come from the server's declared `tools` intersected with its real `initialize` capabilities, and tools are read-only (formatting/rename return proposed edits as text — `fedit_mcp` is the only writer). `BENCHMARKIT_DISABLE_LSP=1` skips spawning (used by tests); `pygls`/`lsprotocol` are imported lazily. Logs in `logs/lsp/`. No file outside a server's `workspaceRoot` (and `restricted-paths.txt`) can be opened through it. The patchable file locations (`LSP_SERVERS_PATH`, `LSP_RESTRICTED_PATHS`, `LSP_CONNECTORS_DIR`) live on `lsphub.constants` and are read through that module at call time — a test that redirects them must patch `src.core.lsphub.constants`.
- Node language servers need a real TypeScript compiler: `typescript-language-server` and `vue-language-server` only work when `web/node_modules/typescript` exists (devDependency, `tsserver.js`) and, for Vue, `--tsdk=<…>/typescript/lib` is passed in `command`. Volar owns no tsserver: it delegates `.vue` TypeScript semantics to its *client* over the private `tsserver/request` notification and **awaits the answer inside `getLanguageService()`** — an unanswered request hangs every later request for that file, so the hub implements the relay (`lsphub/tsserver.py` + the handler in `lsphub/features.py`): it answers `_vue:projectInfo` with the server's `workspaceRoot` tsconfig/jsconfig (or `null`, which makes Volar fall back to its inferred project) and answers the component-detail commands (`_vue:getComponentProps` and friends) with `null` so their promises settle. Watch the wire shape: both directions carry a **doubly nested array** (`params: [[id, …]]`) because vscode-jsonrpc treats a single array argument as positional params. The hub runs no tsserver, so `lsp_vue_diagnostics` reports Volar's template/SFC checks, not `<script>` type errors — the `typescript` server covers plain `.js`/`.ts`.
- External connector secrets (`headers`/`env`) must never leave the server — they are stripped from every API payload.
- A hub restart reuses the **same** `McpManager`/`LspManager` object and only rebuilds its internals (new hub thread, new sessions). Chat worker threads hold that reference across a request, so replacing the object in `app.state` instead would break an in-flight conversation — keep `restart()` in-place.
- Look & feel is Pico.css + `web/src/styles/theme.css` (self-hosted fonts, light + dark via `data-theme`). Rendered markdown supports KaTeX, highlighted code and Mermaid fences through the single `lib/markdown.js` renderer.
- `web/dist` must exist for FastAPI to serve the UI (the API works without it); in dev use the Vite server, which proxies `/api`.

## Adding questions

Add entries under the `"questions"` key of a file in `resources/` — each needs a unique `uuid`, `question`, and `expected`. A new `resources/<name>.json` file automatically becomes a selectable set (id = file stem).

## Adding endpoints

Edit `.env`: `SERVER_URL` (llama.cpp or compatible server) and `ENDPOINT` (model name). Optional sampling params (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, penalties, `REASONING` effort) are forwarded to `litellm.completion()`. `SUPPORTS_VISION=1` enables the Chat image-attach button.
