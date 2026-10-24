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
- **Optional MCP tools in Chat** — declare MCP connectors (remote HTTP or
  local stdio) in `mcp_connectors/externals.json` and tick them in Settings so
  the chat model can browse the web, read docs or run whitelisted shell
  commands; tool calls appear as badges in the transcript.
- **Rich rendering** — answers render as Markdown with KaTeX math,
  syntax-highlighted code blocks, and Mermaid diagrams.
- **Save / load a WorkBench session** — park a conversation on disk as a plain
  JSON file (`{format, version, messages}` in OpenAI user/assistant form) and
  re-open it later to continue where it stopped; each assistant turn may carry
  its streamed `reasoning` for the tool panel, while images and tool calls are
  not part of the file.
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

Connectors are declared — never coded — in
`mcp_connectors/externals.json`, keyed by id: a `remote` entry (Streamable
HTTP `url` plus optional `headers`) or a `local` entry (a stdio `command`
argv, e.g. launched through `npx`). Install the optional `mcp` extra, and the
Chromium binaries when a connector drives a browser:

```bash
pip install -e ".[mcp]"          # adds the mcp SDK + playwright
playwright install chromium      # Linux: playwright install --with-deps chromium
```

Enabled connectors are connected when the app starts and closed when it
quits. Nothing is exposed to the model until its `enable` flag is on — the
single source of truth, edited from **Settings → MCP** (which writes the flag
back to the file and starts/stops the connector right away) or by hand;
benchmark runs never use tools. The file may hold API keys, so it is
gitignored and its `headers`/`env` are never returned by the API. The **Max
tool iterations** field (`MAX_TOOL_ITERATION`, default 25) caps the
tool-calling loop of a single chat reply and is shared by all connectors and
language servers. Without the extra installed everything keeps working — the
MCP settings panel just shows an install hint.

Any `url`, `command`, `headers` or `env` value may reference an environment
variable as `${NAME}` (or `${NAME:-fallback}` when the variable is optional) —
expanded when the file is read, so a per-machine path or an API key can live
in the environment instead of in the file, and one declaration works on every
checkout:

```json
{
  "context7": {
    "type": "remote",
    "enable": true,
    "url": "${MCP_BASE_URL:-https://mcp.context7.com/mcp}",
    "headers": { "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}" }
  }
}
```

The environment is the process one, so anything exported before launch and
everything in `.env` (loaded at startup) is available — `${HOME}`,
`${CONTEXT7_API_KEY}`, … Only the braced `${…}` form is special, so a bare `$`
stays literal and npm version pins (`pkg@$latest`) are untouched. An expansion
never re-splits an argument, so a value containing spaces stays one argv
element. An unset name is left written as-is and reported in the connector's
validation note (visible in **Settings → MCP**) rather than silently blanked.

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

#### Tools working directory

Both declaration files point their file-touching tools at a single folder
through one environment variable, `BENCHMARKIT_PATH_LOCATION` (the
`--root-dir` of the local connectors, and every `workspaceRoot`/`--tsdk` of the
language servers). It is seeded at startup with the install directory, so a
fresh checkout works with no configuration and moving the project needs no
file edit.

The **Settings → Working directory** field shows the live
value; **Set** repoints the MCP and LSP tools at another folder and reloads
them in one step (the same restart the refresh buttons do). The change is
process-local on purpose: the next server start goes back to the install
directory, so a remembered path can never break another machine's checkout.
Export the variable (or set it in `.env`) to pin it permanently — an existing
value is never overwritten. A folder must exist and must not sit under a
`lsp_connectors/restricted-paths.txt` prefix, or the change is refused and
nothing is restarted.

Note the reference syntax: the closing brace goes right after the variable
name, with the rest of the path outside it — `${BENCHMARKIT_PATH_LOCATION}/web`,
never `${BENCHMARKIT_PATH_LOCATION/web}`.

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
- **Session files are client-side.** WorkBench Save/Load (de)serialises the
  conversation in the browser — the file is just the history `POST /api/chat`
  already re-posts on every send, so no server state is involved; loading one
  over a live conversation asks for confirmation first.
- **MCP lifecycle.** Connector sessions live inside the app lifespan
  (`src/core/mcphub/`): one persistent MCP client per declared connector on a
  private asyncio thread that chat worker threads call through synchronously
  (a `local` connector's stdio child is owned by that client); everything is
  closed on shutdown.
