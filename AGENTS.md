# AGENTS.md — benchmarkit

## Project overview

A FastAPI + Vue 3 app that benchmarks LLM endpoints: it streams questions to a configured model, collects responses, and verifies them against expected answers with a secondary LLM verifier agent. Question sets are JSON files in `resources/` (active set chosen by `QUESTION_SET`); configuration comes from `.env`. The frontend is a Vite-built SPA served by FastAPI, with two tabs: **Benchmark** (run + live SSE progress) and **WorkBench** (interactive streaming chat, optionally augmented with MCP connectors and language servers). It targets desktop browsers only.

Naming note: the tab was renamed Chat → WorkBench, but everything behind it kept the `chat` name — `/api/chat`, `core/chat.py`, the `useChat` composable, `ChatStopwatch.vue`, and `state.activeTab === 'chat'`. Only the *view components* were renamed `WorkBench*`.

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
| `api/server.py` | `create_app()` (routers + static `web/dist` + `GET /api/health`), MCP and LSP hub lifespan start/stop, `run()` entry (uvicorn on 127.0.0.1:8000) |
| `api/routes.py` | REST + SSE endpoints, `RunState` (single active run), worker-thread launch |
| `api/events.py` | `EventBus`: seq-numbered in-memory event buffer with replay; SSE formatting |
| `core/bench.py` | Benchmark loop (`run_benchmark()` / `process_question()`), saves `results.json` |
| `core/llm.py` | `call_llm_and_stream()` / `warmup_model()` / `verify_response()` — take an `emit` callable, no UI concerns |
| `core/chat.py` | `stream_chat()` — one streamed chat reply, with the shared MCP + LSP tool loop |
| `core/utils.py` | `save_results()` (`results.json`), `estimate_completion_tokens()` fallback, `format_time()` |
| `core/mcphub/` | MCP connector management: every connector comes from the declarative `externals.json` (remote Streamable HTTP or local stdio), hot toggles. One concern per module: `constants` (paths/timeouts), `validation` (shared `headers`/`env` + command shapes, also used by lsphub), `discovery` (read/validate/write `externals.json`), `formatting` (MCP tool/result objects → OpenAI/plain text), `process` (`connector_environment`, also used by lsphub), `manager` (`McpManager`) |
| `core/lsphub/` | Language-server management, mirroring `mcphub`: `lsp_servers.json` entries spawned as stdio LSP sessions (handshake, document store, diagnostics inbox), capabilities synthesized into `lsp_<id>_<action>` tools, hot toggles, idle reaper. One concern per module: `constants` (paths/timeouts/relay method names), `discovery` (read/validate/write the file), `tools` (action catalogue), `session` (`DiagnosticsInbox`/`DocumentStore`/`ServerSession`), `features` (server→client handlers incl. the tsserver relay), `actions` (document sync + the per-action requests), `rendering` (LSP structures → text), `tsserver` (relay wire shapes), `manager` (`LspManager`) |
| `core/job.py` | `RunJob`: results, failures, score/token/time aggregates |
| `core/models.py` | Pydantic config models, built from env via `EndpointsConfig.from_env()` |
| `config.py` | `.env` loading and the managed-key template (`SETTING_KEYS`), question-set discovery, memory-bullet registration, `BENCHMARKIT_PATH_LOCATION` seeding, `load_agent_instructions()` |

Frontend under `web/src/` (Vite + Vue 3, plain JS, Pico.css):

| File | Purpose |
|---|---|
| `composables/` | One reactive singleton per concern: `useBenchmark` (run state, tab selection + EventSource client), `useChat` (SSE fetch pump + WorkBench hub toggles + session save/load), `useMemory`, `useTheme`, `useAutoScroll`, `useSplitter` |
| `lib/` | `markdown.js` — shared markdown renderer: `marked` + `DOMPurify` + KaTeX + highlight.js + optional Mermaid (never raw `v-html`); `hljs.js` — the registered highlight.js languages; `markedKatexBrackets.js` — teaches marked the `\[ \]` / `\( \)` LaTeX delimiters; `mermaid.js` — lazy-loaded, cached diagram renderer |
| `App.vue` | App bar + tab views: benchmark grid (stats column + `StreamPanel`) and `WorkBenchView` (tab state lives in `useBenchmark`: `benchmark` \| `chat`) |
| `components/` | Benchmark: `StreamPanel`, `QuestionCard`, `Chronometer`, `MetricCard`, `RunControls`. WorkBench: `WorkBenchView`/`WorkBenchInput`, `WorkBenchSidePanel`/`WorkBenchMcpPanel`/`WorkBenchLspPanel`/`WorkBenchToolPanel`, plus `ChatStopwatch`, `MemoryBullets` + `RegistrationMemoryDlg`, `ImageLightbox`, `MarkdownStream`, `GaugeBar`. Shared: `AppBar` (with `TabNav`), `SettingsDialog` (with `ApiKeyField`), `ThemeToggle` |
| `styles/theme.css` | Pico CSS variable overrides + app classes (palette, cards, badges) |

## HTTP API

All app endpoints live under `/api/` (everything else serves the SPA):

- `GET /api/config` — models, `serverUrl`, sampling summary, question count + `totalTests`, `supportsVision`, `contextWindow` (polled from the server's `/v1/models` `meta.n_ctx`, `null` when absent), `verifierConfigured`, `runStatus`, `hasResults`
- `GET|POST /api/settings` — read/save `.env`-backed settings (rewrites `.env`, refreshes `os.environ`; 409 while running, 422 on invalid values)
- `GET|POST /api/mcp` — list / activate MCP connectors (activation is the entry `enable` flag in `externals.json`, written back with hot start/stop; only `MAX_TOOL_ITERATION` persists to `.env`)
- `GET|POST /api/lsp` — list / activate language servers (activation writes both `LSP_ENABLED` in `.env` and the entry `enable` flag in `lsp_servers.json`, hot start/stop; `env` and `initializationOptions` values never returned; `maxIterations` updates the cap shared with MCP) · `GET /api/lsp/log?server=` — tail of one server's log
- `POST /api/mcp/restart` / `POST /api/lsp/restart` — rebuild a hub from a fresh scan of its declaration file (sessions/children are torn down and respawned, so a hand-edited `externals.json`/`lsp_servers.json` applies without restarting the app). Awaited, so the returned payload carries real post-restart statuses. Same payload as the matching GET. 409 while a run is in progress *or* a chat reply is streaming (a restart would kill a tool call mid-flight); 503 when the hub is absent. `BENCHMARKIT_DISABLE_MCP`/`_LSP` blank the scan, so the endpoints stay testable without spawning anything.
- `GET|POST /api/path-location` — read / change `BENCHMARKIT_PATH_LOCATION`, the single folder both declaration files point their file-touching tools at (`--root-dir`, `workspaceRoot`, `--tsdk`). POST validates (absolute, exists after `expanduser` + `realpath`, not on the `restricted-paths.txt` deny-list), sets `os.environ`, and restarts **both** hubs in one call, returning `{key, path, default, mcp, lsp}` so the dialog repaints both panels without extra round-trips. Never persisted — the next boot re-seeds it from the install dir (`config.setup_path_location` in `create_app`). Guards checked before the env is touched: 409 during a run / streaming reply, 503 when a hub is absent, 422 on a rejected path (env unchanged, nothing restarted).
- `GET|POST /api/question-sets` — list / select the active question set (persists `QUESTION_SET`)
- `GET|POST /api/memory` — read / update the 9 memory-bullet slots (self-healing file, always exactly 9 slots)
- `GET /api/models?serverUrl=&apiKey=` — list model ids from an OpenAI-compatible server (settings dialog)
- `POST /api/run` / `POST /api/run/stop` — start / cancel the single active run (both 202; 409 on conflicts, 400 when no model or no question is configured)
- `GET /api/run` — status snapshot + all buffered events (UI rebuild point) · `GET /api/run/events` — SSE stream with replay (`since` / `Last-Event-ID`), closed by a final `stream_end` frame
- `POST /api/chat` — stream one chat reply as SSE (full `messages` history, optional multimodal image parts, optional `mcpEnabled`/`lspEnabled` allowlists; 409 while a run is in progress)
- `GET /api/results` — download `results.json`
- `GET /api/health` — liveness probe (declared in `server.py`, not on the `/api` router): `.env` and the active question-set file exist

## SSE event contract

Events carry an SSE `id:` (sequence number) so clients can reconnect and replay. Two contracts:

- **Benchmark run**: `run_started` (carries `runId`, `models`, `totalTests`, `samplingSummary`) → `model_started` (per model, after its warmup) → per question `question_started` / `token` / `answer_done` / `question_done` → `run_finished` (`run_error` for per-question failures). Timing fields (`took`, `ttft`, `genTook`) are measured app-side with `time.monotonic()`; the chronometer accumulates only answer time (model warmup and verifier time excluded). Token fields distinguish completion-only `tokens` (rate numerator) from `consumed` = prompt + completion (the displayed cost). `question_done` additionally carries the running aggregate `stats` (`current`, `successes`, `answered`, `tokenRate`, `answeredTime`); a cancelled run ends on `run_finished` with `cancelled: true`.
- **WorkBench / chat** (`/api/chat`): `token` / `reasoning` (phase flag) / `reasoning_delta` (thinking text, never answer text) → exactly one terminal `done` or `error`. `done` reports the cumulative `decode` window (the chat tok/s denominator, excludes tool execution) and `contextTokens` (context-window occupancy, never summed across rounds). With MCP **or** LSP tools active, `tool_call` / `tool_result` frames (carrying `iteration` / `maxIterations` for the UI gauge) plus a `usage` frame appear per tool round — `usage` is emitted as soon as a round finishes generating, *before* the tool runs.

## Data files

- **`resources/*.json`** — question sets, one file per set (file stem = set id); each is `{"questions": [{ uuid, question, expected, ... }]}`. A new file automatically becomes a selectable set; `QUESTION_SET` picks the active one (default `legacy`).
- **`resources/memory_registration.json`** — the WorkBench tab's 9 memory bullets (`{color, content}` slots). Managed only through `/api/memory`, self-heals to 9 normalized slots on every read, and is skipped by question-set discovery.
- **`.env`** (project root, gitignored; see `.env.example`) — all configuration: `SERVER_URL` + `SERVER_URL_API_KEY` + `ENDPOINT`, `LLM_SYSPROMPT`, sampling params (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`), `SUPPORTS_VISION`, `QUESTION_SET`, `LSP_ENABLED`, `MAX_TOOL_ITERATION` (one tool-loop cap shared by MCP + LSP; the legacy `MAX_MCP_ITERATION` is still honoured), and the `VERIFIER_AGENT_*` block. MCP connectors have **no** env key: they activate through `externals.json`. A leftover `MCP_ENABLED` is ignored (it is not a managed key any more, so saving settings drops it from the file). Loaded with `python-dotenv` at startup (real env vars take precedence), parsed by `EndpointsConfig.from_env()` into pydantic models; invalid values surface as HTTP 422.
  The file is **regenerated from a fixed template** on every save (`SETTING_KEYS` + `_render_env()` in `src/config.py`): key set and order are fixed, empty optional keys become commented-out lines, and custom keys or hand-written comments do not survive a save. `BENCHMARKIT_PATH_LOCATION` is deliberately absent from that list. `.env.example` is the hand-written template and still carries a stale `MCP_ENABLED` block — it is *not* kept in sync with `_render_env()`, so adding or renaming a managed key means editing `SETTING_KEYS`, `_render_env()` and `.env.example` together.
- **`results.json`** — generated run output, gitignored; saved once at the end of a run.
- **`mcp_connectors/externals.json`** (gitignored; may hold secrets) — the one and only connector declaration, keyed by id: `remote` (Streamable HTTP) or `local` (stdio command); `enable` is the single on/off source of truth (no `.env` key involved). Their `headers`/`env` secrets are never returned by the API. `url`/`command`/`headers`/`env` values may hold `${NAME}` (or `${NAME:-fallback}`) environment references — see *Env references in declaration files* below.
- **`lsp_connectors/lsp_servers.json`** (gitignored; may hold secrets; `lsp_servers.example.json` is the committed template) — declarative language servers keyed by id: always local stdio (`command` argv, never a shell), with `extensions` (which files the server answers for), absolute `workspaceRoot` (refused when inside a `restricted-paths.txt` prefix), `lazy` spawn flag, `tools` list, `maxDiagnostics`/`idleShutdownSeconds`. A server runs when its `enable` flag **and** `LSP_ENABLED` agree. `env` and `initializationOptions` are never returned by the API. `command`/`args`/`env`/`workspaceRoot`/`initializationOptions`/`settings` values may hold `${NAME}` environment references (see below).

### Env references in declaration files

Both declaration files accept `${NAME}` and `${NAME:-fallback}` in their string values, expanded **at read time** by the two *discovery* modules through the shared helpers in `core/mcphub/validation.py` (`expand_env`, `expand_env_map`, `expand_env_list`, `expand_env_json`, `unresolved_env_note`) — so the rest of each package never sees a reference, and a per-machine path (a `workspaceRoot`, a `--tsdk`) or an API key can live in the environment instead of in the file. Conventions worth keeping:

- Source is `os.environ`, which already carries `.env` (`config.load_dotenv_once`, real env vars winning) — `${HOME}` and project keys like `${CONTEXT7_API_KEY}` are equally available.
- **Only the braced form is special.** A bare `$` stays literal, so npm pins (`pkg@$latest`) and regexes in a `command` are never touched; do not add shell-style `$NAME` support, it would silently eat those pins.
- Expansion runs **before** the shape/path checks (the absolute-`workspaceRoot` and http(s)-`url` checks must see the expanded value), and the unresolved-name note is appended **before** the error it causes, so the note explains the error.
- An unresolved name is left written verbatim plus one validation note (surfaced by `GET /api/mcp` / `GET /api/lsp`) instead of being blanked — a typo stays visible rather than becoming an empty path or key.
- No shell semantics: argv is expanded per element and never re-split, so a value containing spaces stays one argument. Dict keys are never expanded; non-strings pass through `expand_env_json` untouched.
- `set_externals_enabled` / `set_lsp_enabled` write only the `enable` flag back to the **raw** file, so an expanded value is never persisted and the reference stays portable.
- **`BENCHMARKIT_PATH_LOCATION`** is the working-location reference both files use (connector `--root-dir`, every `workspaceRoot`/`--tsdk`). `config.setup_path_location()` seeds it at boot with the install dir — deliberately *not* a `SETTING_KEYS` key, so `save_settings` (which regenerates `.env` from a fixed template) can neither erase it nor resurrect a stale value. An exported/pre-set value always wins. Keep the closing brace right after the name (`${BENCHMARKIT_PATH_LOCATION}/web`, never `${BENCHMARKIT_PATH_LOCATION/web}`): a reference with anything else inside the braces resolves to nothing *and* produces no note.

## Key patterns & conventions

- **LiteLLM abstraction**: all LLM calls go through `litellm.completion()` with the `openai/<model>` prefix against an OpenAI-compatible server (e.g., llama.cpp); `api_key="none"` — no real API keys.
- **Core/API separation**: core modules never render or touch FastAPI — they call an injected `emit(event: dict)`; `src/api/` owns transport. Sync litellm runs in a worker thread publishing to the `EventBus`; SSE subscribes on the event loop.
- **Single run**: one run at a time, state in memory (`RunState`); cancellation via a `threading.Event` polled by the loop and stream reader. Chat uses the same worker-thread + cancel pattern, and is 409-rejected while a run is active.
- **Frontend state**: one reactive singleton per concern (`useBenchmark`, `useChat`, …); components read/mutate directly. The active tab lives in `useBenchmark` (`state.activeTab`, `'benchmark'` / `'chat'`) and `setTab()` refuses to open WorkBench while a run is running. SSE reconnect = refetch the `/api/run` snapshot and replay. Markdown always through `lib/markdown.js` (`marked` + `DOMPurify`), never raw `v-html`.
- **AGENTS.md feeds the chat**: when the working location carries `AGENTS.md`, its text is appended to every chat system message (`config.load_agent_instructions()` / `agent_instructions_block()`, passed to `stream_chat()` as `agent_instructions`). Keep it accurate — the model follows it in every WorkBench conversation.
- **Type annotations**: explicit Python types (`: str`, `: Dict`, `typing` imports).
- **WorkBench session files** (the chat): Save/Load in the WorkBench side panel exchange a `{format, version, messages}` JSON file (`format: "benchmarkit-chat-session"`, current version 3) holding the OpenAI `user`/`assistant` turns — user turns keep their attached images as standard `image_url` base64 parts (plus one benchmarkit-only `name` key per image part, for the restored thumbnail: `ChatContentPart` has `extra="ignore"`, so it never reaches the model), so a loaded conversation re-posts the images on the next send. An image part whose data URL is unusable degrades to a placeholder string, so an image-only turn stays postable. Tool-loop turns and the per-turn badges are never saved, and the system prompt is rebuilt server-side per send. An assistant turn may carry an optional `reasoning` list (added in v2; absent in v1 files): that reply's streamed thinking flows — one entry per *reasoning* block of the `WorkBenchToolPanel` timeline, tool blocks are not saved — restored into `message.blocks` for the panel; it sits beside `content` and is never re-posted, since `send()` builds its history through `messagePayload()` (`raw`/`images` only). Both directions are client-side only (`useChat.js`: `conversationJson` / `parseSessionFile` / `loadSession`); the backend needs no change because every send re-posts the whole history.

## Gotchas

- The verifier and the target LLM can be **different endpoints**; without `VERIFIER_AGENT_SERVERURL`, the verifier reuses `SERVER_URL`.
- Token counting relies on `stream_options={"include_usage": True}` (final stream chunk, read via `_extract_usage()`). Keep the distinction straight: the rate uses **completion tokens only**, the displayed cost is **prompt + completion** (`consumed`); missing usage is char-estimated and flagged `tokensEstimated`. Timing is `time.monotonic()` anchored to content chunks, so the trailing usage-only chunk never extends `took`.
- MCP connectors follow the lsphub pattern: there is no built-in connector list and no port range to scan — every connector is an `externals.json` entry, `remote` ones connect over Streamable HTTP, `local` ones are spawned as **stdio children owned by the client transport** (closing the client's exit stack shuts them down, the hub keeps no process handle of its own). What *looks* like built-in connectors is just `mcp_connectors/*.py` (`console_mcp.py`, `fedit_mcp.py`, `playwright_mcp.py`) being referenced as `local` commands — adding a connector means adding a file *and* an entry there, never a module in `src/`. `BENCHMARKIT_DISABLE_MCP=1` skips connecting (used by tests); the `mcp` package is imported lazily. The patchable file locations (`MCP_CONNECTORS_DIR`, `EXTERNALS_PATH`) live on `mcphub.constants` and are read through that module at call time — a test that redirects them must patch `src.core.mcphub.constants`.
- Language servers are managed the same way by the `core/lsphub/` package (`src.core.lsphub`), but LSP is **stdio** (no ports): one private `lsp-hub` asyncio thread owns every session, `lsp_<id>_<action>` tools come from the server's declared `tools` intersected with its real `initialize` capabilities, and tools are read-only — the catalogue (`lsphub/tools.py`) is `diagnostics` / `hover` / `definition` / `references` / `workspace_symbols` / `document_symbols` / `completion` / `signature_help` / `formatting`, and even `formatting` only returns the proposed edits as a unified diff (apply them through the file-edit tools); `fedit_mcp` is the only writer. `BENCHMARKIT_DISABLE_LSP=1` skips spawning (used by tests); `pygls`/`lsprotocol` are imported lazily. Logs in `logs/lsp/`. No file outside a server's `workspaceRoot` (and `restricted-paths.txt`) can be opened through it. The patchable file locations (`LSP_SERVERS_PATH`, `LSP_RESTRICTED_PATHS`, `LSP_CONNECTORS_DIR`) live on `lsphub.constants` and are read through that module at call time — a test that redirects them must patch `src.core.lsphub.constants`.
- Node language servers need a real TypeScript compiler: `typescript-language-server` and `vue-language-server` only work when `web/node_modules/typescript` exists (devDependency, `tsserver.js`) and, for Vue, `--tsdk=<…>/typescript/lib` is passed in `command`. Volar owns no tsserver: it delegates `.vue` TypeScript semantics to its *client* over the private `tsserver/request` notification and **awaits the answer inside `getLanguageService()`** — an unanswered request hangs every later request for that file, so the hub implements the relay (`lsphub/tsserver.py` + the handler in `lsphub/features.py`): it answers `_vue:projectInfo` with the server's `workspaceRoot` tsconfig/jsconfig (or `null`, which makes Volar fall back to its inferred project) and answers the component-detail commands (`_vue:getComponentProps` and friends) with `null` so their promises settle. Watch the wire shape: both directions carry a **doubly nested array** (`params: [[id, …]]`) because vscode-jsonrpc treats a single array argument as positional params. The hub runs no tsserver, so `lsp_vue_diagnostics` reports Volar's template/SFC checks, not `<script>` type errors — the `typescript` server covers plain `.js`/`.ts`.
- External connector secrets (`headers`/`env`) must never leave the server — they are stripped from every API payload.
- A hub restart reuses the **same** `McpManager`/`LspManager` object and only rebuilds its internals (new hub thread, new sessions). Chat worker threads hold that reference across a request, so replacing the object in `app.state` instead would break an in-flight conversation — keep `restart()` in-place.
- The two deny-lists are different files for different purposes: `lsp_connectors/restricted-paths.txt` (absolute path prefixes no `workspaceRoot` may sit inside — also gates `POST /api/path-location`) and `mcp_connectors/restricted-fqdn.txt` (hosts `playwright_mcp.py` refuses to navigate to). Don't confuse them.
- Look & feel is Pico.css + `web/src/styles/theme.css` (self-hosted fonts via `@fontsource`, light + dark via `data-theme`). Rendered markdown supports KaTeX, highlighted code and Mermaid fences through the single `lib/markdown.js` renderer.
- `web/dist` must exist for FastAPI to serve the UI (the API works without it); in dev use the Vite server, which proxies `/api`.

## Adding questions

Add entries under the `"questions"` key of a file in `resources/` — each needs a unique `uuid`, `question`, and `expected`. A new `resources/<name>.json` file automatically becomes a selectable set (id = file stem).

## Adding endpoints

Edit `.env`: `SERVER_URL` (llama.cpp or compatible server) and `ENDPOINT` (model name). Optional sampling params (`TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING` effort) are forwarded to `litellm.completion()`. `SUPPORTS_VISION=1` enables the WorkBench image-attach button. The same fields are editable in the settings dialog (`/api/settings`), which rewrites `.env` from its fixed template.
