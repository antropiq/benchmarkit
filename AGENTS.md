# AGENTS.md — benchmarkit

## What it is

A self-hosted FastAPI + Vue 3 app that benchmarks LLM inference endpoints
(llama.cpp or any OpenAI-compatible server). It streams each question of a
question set to the configured model, measures how fast and how well the model
answers, and grades every answer against the expected one with a secondary
**verifier LLM**. Everything reaches the browser live over Server-Sent Events.
Desktop browsers only.

The app has two tabs, sharing one LLM configuration and one streaming layer:

- **Benchmark** — runs a question set against the configured model, showing
  answers, scores, timings and token/s as they arrive.
- **WorkBench** — interactive streaming chat with that same model, optionally
  augmented with MCP connectors and language servers.

## Commands

```bash
# Setup — pyenv virtualenv named `benchmarkit`, no .venv directory
pip install -e .
pip install -e ".[test]"     # test deps
pip install -e ".[mcp]"      # MCP extras, then: playwright install chromium
pip install -e ".[lsp]"      # LSP extras (pygls, lsprotocol)

cd web && npm install && npm run build   # SPA -> web/dist, served by FastAPI
cd web && npm run dev                    # Vite dev server :5173, proxies /api

benchmarkit                              # app on http://127.0.0.1:8000
pytest                                   # test suite (tests/)
```

No lint step; the only frontend build is `vite build`. `web/dist` must exist
for FastAPI to serve the UI — the API works without it.

## Code structure

### Backend (`src/`)

Two layers: a UI-free **core** and the **API** in front of it. Core functions
never touch FastAPI — they take an injected `emit(event: dict)` callable and
report progress through it.

| Module | Purpose |
|---|---|
| `config.py` | `.env` loading, the managed-key template, question-set discovery. |
| `api/server.py` | `create_app()`: routers, static SPA, hub startup/shutdown, uvicorn entry. |
| `api/routes.py` | REST + SSE endpoints; `RunState` holds the single active run. |
| `api/events.py` | `EventBus`: seq-numbered in-memory buffer with replay, SSE framing. |
| `core/llm.py` | One streamed LLM call, model warm-up, and the verifier call. |
| `core/bench.py` | Benchmark loop: question → answer → verify → emit. |
| `core/chat.py` | One streamed chat reply, including the shared tool-call loop. |
| `core/job.py` | `RunJob`: per-run results and score/token/time aggregates. |
| `core/models.py` | Pydantic config models, built from env. |
| `core/mcphub/` | MCP connectors: declared in `externals.json`, hot toggles. |
| `core/lsphub/` | Language servers: declared in `lsp_servers.json`, hot toggles. |

`mcphub/` and `lsphub/` mirror each other — `constants` (paths, timeouts),
`discovery` (read/validate/write the declaration file), `manager` (the live
hub), plus formatting/rendering of tool results. Keep that symmetry when
touching either one.

### Frontend (`web/src/`, Vite + Vue 3, plain JS, Pico.css)

| Path | Purpose |
|---|---|
| `composables/` | One reactive singleton per concern; components use them directly. |
| `lib/markdown.js` | The single markdown renderer (marked + DOMPurify + KaTeX + hljs). |
| `App.vue` | App bar and the two tab views. |
| `components/` | Benchmark widgets, WorkBench widgets, shared chrome. |
| `styles/theme.css` | Pico CSS variable overrides + app classes (light and dark). |

Two frontend invariants: rendered HTML always goes through `lib/markdown.js`,
never raw `v-html`; and run/stream state lives in the composables, which own
the EventSource and its reconnect.

## How a request flows

All app endpoints live under `/api/`; anything else serves the SPA. The exact
route list belongs to `api/routes.py` — what matters architecturally:

- **Run**: starts the benchmark in a worker thread, watched live over SSE.
  One run at a time (WorkBench is refused while one runs), cancelled through a
  `threading.Event` polled both between questions and inside the stream reader.
- **Chat**: one request per reply, streamed back over SSE. The client re-posts
  the whole history each time; the server keeps no conversation state.
- **Configuration**: `.env` and the two declaration files are edited from the
  UI and take effect hot — an app restart is never needed.

### SSE is the UI's only source of truth

Core emits plain dicts into the `EventBus`, which numbers them so a
reconnecting client replays what it missed. Two independent streams: the run
stream (`run_started` → `model_started` → per-question `question_started` /
`token` / `answer_done` / `question_done` → `run_finished`, with `run_error`
on failure) and the per-reply chat stream (`token`, `reasoning_delta`, and
`tool_call` / `tool_result` when tools are active, ending in exactly one
`done` or `error`).

Two measurement rules hold across both:

- **Timing is app-side** (`time.monotonic()`, anchored to content chunks) —
  warm-up and verifier time never enter the numbers.
- **Cost and rate are different numbers**: token/s uses completion tokens only,
  the displayed cost is prompt + completion.

## Data files

| File | Role |
|---|---|
| `.env` | Run configuration: endpoint, key, prompt, sampling, set, verifier. |
| `resources/<set>.json` | Question sets — `{uuid, question, expected}`. |
| `resources/memory_registration.json` | WorkBench memory-bullet slots (fixed, self-healing). |
| `results.json` | Run output, written once when a run ends. |
| `mcp_connectors/externals.json` | MCP connectors: `remote` (HTTP) or `local` (stdio). |
| `lsp_connectors/lsp_servers.json` | Language servers: `command` (stdio), `workspaceRoot`. |

Any new file in `resources/` becomes a selectable set. The `.env` and the two
hub files are gitignored, may hold secrets, and each has a committed template.
Four rules tie them together:

- **`.env` is regenerated from a fixed template on every save** (`SETTING_KEYS`
  and `_render_env()` in `src/config.py`): fixed key set and order, empty
  optional keys commented out, hand-written extras lost. A new or renamed key
  means editing `SETTING_KEYS`, `_render_env()` and `.env.example` together.
- **The declaration files are the only source of truth for what runs.** Their
  `enable` flag is the switch, toggled hot from the UI; their secrets
  (`headers`, `env`, `initializationOptions`) never leave the server.
- **Both accept `${NAME}` / `${NAME:-fallback}`**, expanded at read time by each
  hub's `discovery` module via shared helpers in `mcphub/validation.py`, so a
  per-machine path or key stays in the environment. Expansion runs *before*
  validation; only the braced form is special (a bare `$` stays literal, which
  protects npm pins and regexes); an unresolved name is kept verbatim with a
  validation note rather than blanked.
- **`BENCHMARKIT_PATH_LOCATION`** is the working-location both files point their
  file-touching tools at (`--root-dir`, `workspaceRoot`, `--tsdk`), so one value
  moves them together. Seeded at boot from the install dir; deliberately not a
  managed `.env` key, so a settings save can neither erase it nor resurrect a
  stale value.

## Key patterns

- **LiteLLM everywhere**: every LLM call is `litellm.completion()` with the
  `openai/<model>` prefix against an OpenAI-compatible server, `api_key="none"`.
  The verifier is its own optional endpoint block (`VERIFIER_AGENT_*`); with no
  verifier endpoint configured, an answer cannot be graded and is failed.
- **Core owns logic, API owns transport**: blocking LLM work runs in a worker
  thread publishing to the `EventBus`; the SSE generator consumes it on the
  event loop. Run and chat use the same worker + cancel pattern.
- **Hubs are long-lived singletons**: a restart rebuilds the *internals* of the
  same `McpManager`/`LspManager`. Never swap the object in `app.state` — chat
  workers hold the reference across a request.
- **Tools are read-only by design**: LSP tools return information (even
  `formatting` returns a proposed diff), the `console` connector execs one argv
  with no shell (pipes, chains, redirections and `$VAR` expansion are refused,
  `check_command_allowed` is the same parse), and each connector's reach is
  bounded by a deny-list — `lsp_connectors/restricted-paths.txt` for paths,
  `mcp_connectors/restricted-fqdn.txt` for hosts.
- **AGENTS.md feeds the chat**: when the working location carries an
  `AGENTS.md`, its text is appended to every chat system prompt, re-read on
  each request. Keep it accurate and short — the model acts on it.
- **Python style**: explicit type annotations (`: str`, `Dict`, `typing`
  imports).
- **Tests stay hermetic**: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1` skip
  connecting and spawning; the patchable file locations live on
  `mcphub.constants` / `lsphub.constants` and must be patched through those
  modules.

## Gotchas

- **A server-cut stream is indistinguishable from a finished answer at the
  chunk level**: litellm fabricates `finish_reason: "stop"` when no chunk
  carried one. The honest signal is on the wrapper once the loop is drained —
  `core.llm.saw_finish_reason()` reads it and fails open. Both loops act on it:
  a truncated answer is failed and **never sent to the verifier** (grading a
  fragment measures the transport, not the model), and a truncated chat reply
  ends on `error` rather than `done` — which also stops a half-streamed tool
  call from being executed.
- **Token counts can look server-reported when they are not**: if a stream dies
  before any usage chunk, litellm back-fills usage by re-counting the chunks it
  saw, so `tokensEstimated` stays `False`. Missing usage on a normal stream is
  char-estimated and does get flagged.
- **litellm cannot price local model names**, and under `LITELLM_DEBUG=1` it
  logs that swallowed error once per call. `config` registers a zero-cost,
  provider-qualified entry per configured model to keep it silent.
- **Node language servers need a real TypeScript compiler** — the frontend's
  `node_modules/typescript`, passed to Vue via `--tsdk`. Volar delegates `.vue`
  TypeScript semantics back to its client and awaits the answer, so the hub
  implements that relay (`lsphub/tsserver.py`); an unanswered request hangs
  every later request for that file. So `lsp_vue_*` reports Volar's template
  checks only, not `<script>` type errors.
- **Adding a connector is never a module in `src/`**: the local connectors are
  the scripts in `mcp_connectors/`, referenced as `local` entries in
  `externals.json`. Add the file *and* the entry.

## Editing AGENTS.md

- This file is injected into every WorkBench system prompt, so its size is paid
  for on each message. Stay under ~200 lines.
- Describe the **big picture**: what the pieces are and which invariants hold.
  Route-level detail, exhaustive lists and per-feature essays belong in the
  code, the README, or a doc comment next to the thing they describe.
- One idea per bullet, no single long line — wrap prose at ~100 columns.
- Describe the app as it is now. Nothing about a past session, a rename, a
  previous state of the file, or what used to be true.
- When code changes, delete or shrink the paragraph here rather than appending
  a newer, longer one beside it.
