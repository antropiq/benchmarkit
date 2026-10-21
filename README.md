# Benchmark'it

A self-hosted web app to benchmark LLM inference endpoints — llama.cpp or any
OpenAI-compatible server — with natural language questions everybody
understands 😆.

It streams each question through the model(s) you configure, measures how fast
and how well they answer, and verifies every response against the expected
answer using a secondary **verifier LLM**. Everything updates live in the
browser over Server-Sent Events. A companion **Chat** tab lets you talk to the
same configured model interactively.

## Features

- **Benchmark runs** — every question of a set is streamed to each configured
  model; answers, scores, and timings appear as they arrive, with a live
  chronometer and metric cards (score, generation token/s).
- **LLM-based verification** — an optional second model judges each answer
  against the expected one, so grading tolerates wording differences.
- **Question sets** — drop JSON files in `resources/` and pick the active set
  from the UI: `legacy` (30 hard questions from
  [matchingpennies.com](https://matchingpennies.com/hard_questions_for_llms/)),
  `78qcm` / `122qcm` (multi-domain multiple choice), and `30hardmath`.
- **Honest timing** — time-to-first-token, generation duration, and token/s
  are measured app-side from chunk arrivals (`time.monotonic()`), with a model
  warm-up request so model loading never pollutes the numbers.
- **Chat tab** — interactive streaming chat with the benchmark-configured
  model, including a "thinking" indicator while the model emits reasoning.
- **Optional MCP tools in Chat** — activate the bundled browser and console
  MCP connectors so the chat model can browse the web and run whitelisted
  shell commands; tool calls appear as badges in the transcript.
- **Rich rendering** — answers render as Markdown with KaTeX math,
  syntax-highlighted code blocks, and Mermaid diagrams.
- **In-app settings** — a dialog to edit endpoint, sampling parameters,
  question set, and verifier; changes are persisted to `.env`.
- **Dark / light theme**, self-hosted fonts, Pico.css.
- **Stop anytime** — cancellation aborts between questions and mid-stream.

## Running the app

Requirements: Python ≥ 3.10, Node.js (to build the frontend).

### 1. Install Python dependencies

Either way works — pick one virtualenv manager, not both at once:

**Using pyenv** (this project ships a `.python-version` pointing to a
`benchmarkit` virtualenv, which auto-activates when you `cd` into the project):

```bash
cd /path/to/benchmarkit
pip install -e .
```

**Using plain venv:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> If you use the venv approach while pyenv is installed, make sure only **one**
> environment is active — pyenv shims otherwise shadow `.venv/bin`.

#### MCP connectors (Chat tab, optional)

To let the Chat model use the bundled MCP connectors
(`mcp_connectors/`), install the optional `mcp` extra and the Chromium
binaries the browser connector needs:

```bash
pip install -e ".[mcp]"          # adds the mcp SDK + playwright
playwright install chromium      # Linux: playwright install --with-deps chromium
```

The connector servers start when the app starts and stop when it quits. The
hub assigns every built-in connector a listen port automatically — derived
deterministically from its id inside the `8940–8999` range, checked for
availability, and passed to the connector as `--port` — so a new
`*_mcp.py` file needs no port registration anywhere; the assigned port is
shown as a badge in **Settings → MCP**. Nothing is exposed to the model until
you tick a connector in **Settings → MCP** (persisted as `MCP_ENABLED` in
`.env`); benchmark runs never use tools. The **Max tool iterations** field
(`MAX_MCP_ITERATION`, default 25) caps the tool-calling loop of a single
chat reply and is shared by all connectors. Connector stdout/stderr is
captured in `logs/mcp/`. Without the extra installed everything keeps
working — the MCP settings panel just shows an install hint.

### 2. Build the frontend

```bash
cd web
npm install
npm run build
```

### 3. Configure

Copy `.env.example` to `.env` at the project root and point it at your server:

```bash
SERVER_URL=http://127.0.0.1:8080/v1
ENDPOINT=my-model
```

Sampling parameters, the question set, and the verifier agent are optional —
see the comments in `.env.example`, or edit them later from the settings dialog
in the UI.

### 4. Run

```bash
benchmarkit    # serves the API + UI on http://127.0.0.1:8000
```

### Development (hot reload)

Terminal 1 (API):

```bash
uvicorn src.api.server:app --reload
```

Terminal 2 (frontend dev server on :5173, proxies `/api` to :8000):

```bash
cd web && npm run dev
```

### Tests

```bash
pip install -e ".[test]"
pytest
```

If you have good questions to add please share them! 🙏

## Technical details

- **LiteLLM everywhere.** All calls (benchmark, verifier, chat) go through
  `litellm.completion()` with the `openai/` provider prefix against your
  `api_base`, so any OpenAI-compatible server works. `api_key="none"` — no
  real credentials involved.
- **Streaming architecture.** The benchmark loop and chat run synchronous
  litellm calls in worker threads that publish to an in-memory, seq-numbered
  event bus; a FastAPI SSE endpoint subscribes on the event loop. Clients
  reconnect by replaying buffered events, so a refresh never loses state.
- **Clean core.** `src/core/` has no FastAPI or browser concerns — modules
  receive an `emit(event)` callable and stay testable in isolation;
  `src/api/` owns transport only.
- **Timing methodology.** Token rate is the mean of per-question
  `completion_tokens / decode-window` rates, timed from chunk arrivals;
  reasoning deltas count toward the decode window (they consume completion
  tokens), verifier time never does. Usage comes from the final stream chunk
  (`stream_options={"include_usage": true}`), with text-length estimation as
  fallback.
- **Frontend.** Vue 3 + Vite + Pico.css, plain JS, one reactive singleton per
  screen (`useBenchmark.js`, `useChat.js`) — no store library. Rendered
  Markdown always passes through DOMPurify.
- **Single run.** One benchmark run at a time, state in memory; the run and
  the chat share the configured endpoint and refuse to overlap.
- **MCP lifecycle.** Connector servers live inside the app lifespan
  (`src/core/mcp_hub.py`): one subprocess each, one persistent MCP client
  per connector on a private asyncio thread that chat worker threads call
  through synchronously; everything is terminated on shutdown.
