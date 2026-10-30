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
- **Editor tab** — browse the working location as a file tree and edit it in a
  Monaco pane: save, create, rename and delete files and folders, with git
  status badges and an integrated terminal.
- **Deleted files are not lost** — the Editor never removes a file outright; it
  moves it into a dated batch under `.nojae/trash/` of the **current** working
  location (kept 30 days), so the batch is a `mv` away from the tree it came
  from; each save keeps the previous bytes in `logs/editor_backups/`. That
  folder is reserved — hidden from the tree, the read and the write routes — so
  you restore it from a terminal. One wrinkle to know: `.nojae/` is only in
  benchmarkit's own `.gitignore`, so in a *foreign* project's `git status` it
  shows as untracked; the app never edits a repository it was pointed at, so it
  will not add it to your `.gitignore` for you.
- **Optional MCP tools in Chat** — declare MCP connectors (remote HTTP or
  local stdio) in `mcp_connectors/externals.json` and tick them in Settings so
  the chat model can browse the web, read docs or run whitelisted shell
  commands; tool calls appear as badges in the transcript.
- **Rich rendering** — answers render as Markdown with KaTeX math,
  syntax-highlighted code blocks, and Mermaid diagrams.
- **Save / load a WorkBench session** — park a conversation on disk as a plain
  JSON file (`{format, version, messages}` in OpenAI user/assistant form) and
  re-open it later to continue where it stopped; each assistant turn may carry
  its streamed `reasoning` for the tool panel, and attached images travel with
  their turn as base64 `image_url` parts, so a loaded conversation still shows
  (and re-sends) them; tool calls are not part of the file.
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

Whichever folder you point at gets its own read-write policy in
`.nojae/read-write-policy.json`: which of its files the Editor (and the chat
assistant's file tools) may show, read and change. It is seeded for you and
edited outside the app. See
[`docs/POLICY-MANAGEMENT-DOC-EN.md`](docs/POLICY-MANAGEMENT-DOC-EN.md) /
[`docs/POLICY-MANAGEMENT-DOC_FR.md`](docs/POLICY-MANAGEMENT-DOC_FR.md).

#### Agent modes (Chat tab)

The WorkBench composer carries a **mode** dropdown: who the assistant is being
for this conversation, and how far it may reach.

| id | label | MCP tools | language servers | editable |
|---|---|---|---|---|
| `chat` | Chat | no | no | yes |
| `code` | Code | yes | yes | yes |
| `architect` | Architect | yes | no | yes |
| `todoBuilder` | Todo builder | yes | no | **no** — defined by benchmarkit |

A mode replaces the system prompt the chat model is given (`LLM_SYSPROMPT` in
`.env` still drives the benchmark, which has no mode), and its two booleans
decide whether a whole tool hub reaches the model at all. It can only narrow
what Settings and the conversation's own toggles allow, never widen them.

Switching mode mid-conversation is free — except for Todo builder: once that
thread has written a turn, the lot conventions it wrote under stay fixed for the
rest of it. A launched batch takes its own mode for every task turn, so the
dropdown has no effect on a lot that is running.

The three editable modes live in `<working location>/.nojae/agent-modes.json`,
seeded for you when a folder becomes the working location and edited by hand
outside the app — the same rule as the read-write policy beside it. An unknown
key in that file is reported rather than ignored, and a typo never costs you the
conversation: the built-in modes answer, and the dropdown shows a line naming
what the file got wrong. To get the old behaviour back for one mode, set
`"systemMessage": ""` — that restores the prompt configured in `.env`.

The file as benchmarkit seeds it, in full:

```json
{
  "version": 1,
  "readme": [
    "benchmarkit reads this file to build the agent-mode dropdown.",
    "The keys in it are matched exactly: an unknown key is refused with an error,",
    "not ignored, so a misspelling is reported instead of quietly switching an",
    "agent off. The allowed top-level keys are version, readme, default and",
    "agents; the allowed keys of one mode are label, mcptools, lsptools and",
    "systemMessage.",
    "The entries under agents are kept in the order you wrote them, and that is",
    "the order of the dropdown.",
    "systemMessage is the persona sent to the model. Write it as a string, or as",
    "a list with one line per entry (readable inside JSON). An empty or missing",
    "systemMessage means: use LLM_SYSPROMPT from .env instead.",
    "mcptools and lsptools are JSON booleans, true or false - not 1/0, not",
    "\"yes\"/\"no\". A missing one means false. They decide whether that hub reaches",
    "the model at all; Settings still decides which of its connectors are on.",
    "\"todoBuilder\" is defined by benchmarkit and cannot be set here; an entry",
    "for it is ignored.",
    "benchmarkit seeds this file when a folder becomes the working location, then",
    "never writes it again. It is machine-local and gitignored, so each checkout",
    "edits its own agents."
  ],
  "default": "code",
  "agents": {
    "chat": {
      "label": "Chat",
      "mcptools": false,
      "lsptools": false,
      "systemMessage": [
        "You are answering in Chat mode and you are a helpful assistant.",
        "- Answer the question. Do not edit, create or move project files.",
        "- Say when you are unsure, and ask one clarifying question rather than guessing a requirement.",
        "- Use markdown for structure, fenced code for snippets, Mermaid diagram for representation.",
        "  You can answer all type of questions. Not just questions related to the project.",
        "- When the task genuinely needs the workspace (read a file, run a test),",
        "  say so and suggest the user to switch to another mode instead of improvising an answer."
      ]
    },
    "code": {
      "label": "Code",
      "mcptools": true,
      "lsptools": true,
      "systemMessage": [
        "You are working in Code mode on the project at the working",
        "location: you implement, and you leave the tree compiling and tested.",
        "- Use the tools. Read before you edit (file reads, lsp_* symbols,",
        "  diagnostics); edit through preview_edit/apply_edit with verbatim context;",
        "  use console for the test suite.",
        "- Keep edits small and in the file's own style. The project instructions",
        "  above win over your habits on anything project-specific.",
        "- A path a tool refuses is not worth retrying: the refusal names the rule,",
        "  and changing that rule is the user's decision, outside this chat.",
        "- Prove the change: run the narrowest relevant tests (pytest,",
        "  `cd web && npm test`) and re-read diagnostics after the edit. Report what",
        "  you ran; if you did not run it, say that rather than saying it works.",
        "- Do one task per turn unless asked for more. When the request is bigger",
        "  than one session, say how you would split it and stop.",
        "- Touch the project instructions only when an invariant changed."
      ]
    },
    "architect": {
      "label": "Architect",
      "mcptools": true,
      "lsptools": false,
      "systemMessage": [
        "You are working in Architect mode: you turn a rough idea into a",
        "written spec. You do not implement it.",
        "- Talk first, write second. Restate the feature, list what you need to know",
        "  (scope, edge cases, data, which surface owns it), and ask one focused",
        "  question at a time. Summarise what you heard before you commit it to a file.",
        "- Your only writes are markdown: a spec under docs/ (or the folder the user",
        "  names). No source edits, no refactors, no test runs. If implementing",
        "  something would answer a question, write the question down instead.",
        "- A spec ends up with: what the feature is for; the decisions taken, each",
        "  with the alternative rejected; what it builds on; the shape and files of",
        "  each new piece; what is out of scope; and how it is verified.",
        "- Read the real files, and name this project's real functions and the rules",
        "  its project instructions set. A design that could describe any project is",
        "  not a design for this one.",
        "- End by offering to hand the spec to the Todo builder. That is the",
        "  next step, not this one."
      ]
    }
  }
}
```

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
