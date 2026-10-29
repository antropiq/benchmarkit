# AGENTS.md — benchmarkit

## What it is

A self-hosted FastAPI + Vue 3 app that benchmarks LLM inference endpoints
(llama.cpp or any OpenAI-compatible server). It streams each question of a
question set to the configured model, measures how fast and how well the model
answers, and grades every answer against the expected one with a secondary
**verifier LLM**. Everything reaches the browser live over Server-Sent Events.
Desktop browsers only.

The app's tabs share one LLM configuration and one streaming layer:

- **Benchmark** — runs a question set against the configured model, showing
  answers, scores, timings and token/s as they arrive.
- **WorkBench** — interactive streaming chat with that same model, optionally
  augmented with MCP connectors and language servers.
- **Editor** — the working location as a file tree and a code pane. Its writes
  are one bounded surface reachable from the browser only, never from a hub or
  a tool. The app's one shell is a separate human surface (`core/termhub/`, one
  WebSocket per session), never an agent tool either.

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
cd web && npm test                       # JS unit/component tests (Vitest, web/tests)
cd web && npm run test:e2e               # Playwright e2e (web/e2e), against its own server
```

No lint step; the only frontend build is `vite build`. `web/dist` must exist
for FastAPI to serve the UI — the API works without it.

Frontend tests come as unit/component runs (Vitest+jsdom over composables and
`lib/`) and Playwright e2e against the real FastAPI app (it builds `web/dist`
itself). e2e starts its own server on a separate port and takes its LLM endpoint
from `.env.test`, never from the developer's `.env`.

## Code structure

### Backend (`src/`)

A UI-free **core** and the **API** in front of it. Core functions never touch
FastAPI — they take an injected `emit(event: dict)` callable and report
progress through it.

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
| `core/batch.py` | Batch-mode lot listing: numbered task files + the big-picture body, gated by the connector path policy — never a task body. |
| `core/editorfs/` | Editor tab's file surface: `path` relative to the working location, every call answering from the workspace policy's verdict. Reads are bounded; writes are refused by `ro`/`hidden` and never destroy bytes — a save backs up what it replaces, a delete moves the file to a dated batch. |
| `core/models.py` | Pydantic config models, built from env. |
| `core/mcphub/` | MCP connectors: declared in `externals.json`, hot toggles. |
| `core/lsphub/` | Language servers: declared in `lsp_servers.json`, hot toggles. |
| `core/termhub/` | Editor's terminal: one PTY shell per WebSocket, session registry on `app.state.terminal`. |
| `core/proctree.py` | Shared process-tree teardown (SIGTERM→reap→SIGKILL) used by both hubs and the terminal. |

`mcphub/` and `lsphub/` mirror each other — `constants` (paths, timeouts),
`discovery` (read/validate/write the declaration file), `manager` (the live
hub), plus formatting/rendering of tool results. Keep that symmetry when
touching either one.

### Frontend (`web/src/`, Vite + Vue 3, plain JS, Pico.css)

| Path | Purpose |
|---|---|
| `composables/` | One reactive singleton per concern; components use them directly. |
| `lib/markdown.js` | The single markdown renderer (marked + DOMPurify + KaTeX + hljs). |
| `App.vue` | App bar and the tab views (`v-show`, so every view stays mounted; the Editor view is a lazy chunk). |
| `components/` | Benchmark widgets, WorkBench widgets, shared chrome. |
| `styles/theme.css` | Pico CSS variable overrides + app classes (light and dark). |

Frontend invariants: rendered HTML always goes through `lib/markdown.js`, never
raw `v-html`; and run/stream state lives in the composables, which own the
EventSource and its reconnect.

## How a request flows

All app endpoints live under `/api/`; anything else serves the SPA. The exact
route list belongs to `api/routes.py` — what matters architecturally:

- **Run**: starts the benchmark in a worker thread, watched live over SSE.
  One run at a time (WorkBench is refused while one runs), cancelled through a
  `threading.Event` polled both between questions and inside the stream reader.
- **Chat**: one request per reply, streamed back over SSE. The client re-posts
  the whole history each time; the server keeps no conversation state.
- **Configuration**: `.env` and the hub declaration files are edited from the UI
  and take effect hot — an app restart is never needed.

### SSE is the UI's only source of truth

Core emits plain dicts into the `EventBus`, which numbers them so a
reconnecting client replays what it missed. Each stream has its own channel and
closes with exactly one terminal event (event names live in `api/routes.py` and
`core/`). A reply the user paused with `POST /api/chat/pause` still ends on its
own ack, keeping its partial turn.

Measurement rules hold across every stream:

- **Timing is app-side** (`time.monotonic()`, anchored to content chunks) —
  warm-up and verifier time never enter the numbers.
- **Cost and rate are different numbers**: token/s uses completion tokens only,
  the displayed cost is prompt + completion.

## Data files

| File | Role |
|---|---|
| `.env` | Run configuration: endpoint, key, prompt, sampling, set, verifier. |
| `.env.test` | e2e-only endpoint, same key set as `.env.example`. |
| `resources/<set>.json` | Question sets — `{uuid, question, expected}`. |
| `resources/memory_registration.json` | WorkBench memory-bullet slots (fixed, self-healing). |
| `results.json` | Run output, written once when a run ends. |
| `mcp_connectors/externals.json` | MCP connectors: `remote` (HTTP) or `local` (stdio). |
| `lsp_connectors/lsp_servers.json` | Language servers: `command` (stdio), `workspaceRoot`. |
| `<working location>/.nojae/read-write-policy.json` | The workspace's own read-write rules; seeded when a folder becomes the working location, edited only outside the app. |

Any new file in `resources/` becomes a selectable set. The `.env`, `.env.test`
and the hub declaration files are gitignored, may hold secrets, and each has a
committed template; `.nojae/` is gitignored too (machine-local per checkout) and
has no template — it is written by the seeding code. These rules tie them together:

- **`.env` is regenerated from a fixed template on every save** (`SETTING_KEYS`
  and `_render_env()` in `src/config.py`): fixed key set and order, empty
  optional keys commented out, hand-written extras lost. A new or renamed key
  means editing `SETTING_KEYS`, `_render_env()` and `.env.example` together.
- **The declaration files are the only source of truth for what runs.** Their
  `enable` flag is the switch, toggled hot from the UI; their secrets
  (`headers`, `env`, `initializationOptions`) never leave the server.
- **The declaration files accept `${NAME}` / `${NAME:-fallback}`**, expanded at
  read time by each hub's `discovery` module via shared helpers in
  `mcphub/validation.py`, so a
  per-machine path or key stays in the environment. Expansion runs *before*
  validation, and only the braced form is special (a bare `$` stays literal).
  An unresolved name is kept verbatim with a note, never blanked.
- **`BENCHMARKIT_PATH_LOCATION`** is the working-location every file-touching
  connector roots itself at, so one value moves them together. Seeded at boot
  from the install dir; deliberately not a managed `.env` key, so a settings
  save can neither erase it nor resurrect a stale value.
- **Every path the app holds is relative to that location**, so a switch
  invalidates them all at once: the Editor's tree re-roots (cached listings,
  branches and selection are forgotten, not re-listed) and every tab of the root
  it leaves closes, unsaved edits reported rather than negotiated with.
- **A targeted workspace gets a `.nojae/read-write-policy.json`** — seeded with
  a default at the two moments a folder becomes the working location (boot, and
  the settings switch), by exclusive create so an edited file is never
  clobbered. Seeding failure is a note, never a refused switch. The folder is
  **reserved by code**: hidden from the tree, the read and git status, and it is
  edited only outside the app.
- **Path rules come from two levels, one verdict.** Level A is the app's own
  rules about files it writes (`**/.env**`, `logs/**`), live only while the
  working location *is* the install dir — so another project's `.env` is an
  ordinary file. Level B is the workspace's file, always live. The strictest
  match wins, so B tightens but never loosens A. A hidden path answers 404 to a
  read and 422 naming the rule to a write; `ro` answers 422 to a write only.
- **The connectors answer level B for their own root.** Each local connector reads
  that workspace file in its own process, so an `ro` folder is uneditable by an
  agent too; the two stdio connector processes and the app share one parser, one
  matcher and one refusal sentence. They do not apply level A (a rule about the
  pane fighting the settings dialog over a generated file), and the connector
  deny-list still guards everything outside the root — the lists stay separate
  on purpose, and merging them would move secrets into a per-project file.

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
- **An activated language server is never invisible**: one with no live session
  (not started, mid-handshake, or reaped for idleness) is still published from
  its declared tools and revived by its first call. Only `enable: false` and
  `error` keep it out.
- **Hub teardown kills process *trees*, not wrappers**: every child leads its
  own process group and is signalled by group, so the children a language
  server spawns of its own never outlive the app that started them.
- **Tools are read-only by design**: LSP tools return information (even
  `formatting` returns a proposed diff), the `console` connector execs one argv
  with no shell (pipes, chains, redirections and `$VAR` expansion are refused,
  `check_command_allowed` is the same parse), and each connector's reach is
  bounded by a deny-list — `lsp_connectors/restricted-paths.txt` for paths,
  `mcp_connectors/restricted-fqdn.txt` for hosts.
- **A `/api/fs` write route is reachable from the SPA only**: no connector, LSP
  tool or chat code path may call one, and every write route is refused while a
  chat reply streams. `tests/test_write_surface_isolation.py` asserts that
  instead of trusting this sentence.
- **Batch mode never touches file contents**: the `todos/` lot is *listed*
  through the shared deny-list (`core/batch.py`) and *read/written* only
  through the connectors; its loop is client-side (`useBatch.js`), so a reload
  discards a running lot.
- **Tree file icons are one generated unit**: the per-file-type glyph mapping
  (`web/src/lib/file-icons.js`), the `nerd-icons-*.woff2` subsets and the
  `unicode-range` of the "JetBrains Mono Nerd Icons" faces in `nerd-fonts.css`
  are all produced from the tables in `tools/nerd_icons.py` (`generate`), and a
  test holds them in sync — a code point with no glyph renders a tofu box, so
  edit the tables and regenerate, never the artifacts by hand.
- **AGENTS.md feeds the chat**: when the working location carries an
  `AGENTS.md`, it is appended to every chat system prompt, re-read per request.
  Keep it accurate and short — the model acts on it.
- **Python style**: explicit type annotations (`: str`, `Dict`, `typing`
  imports).
- **Tests stay hermetic**: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1` skip
  connecting and spawning; patchable file locations live on `mcphub.constants`
  / `lsphub.constants` and must be patched through those modules.

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
  semantics to its client and awaits the answer, so the hub relays it
  (`lsphub/tsserver.py`); `lsp_vue_*` checks templates only, not `<script>`
  type errors.
- **Adding a connector is never a module in `src/`**: the local connectors are
  the scripts in `mcp_connectors/`, referenced as `local` entries in
  `externals.json`. Add the file *and* the entry.

## Editing AGENTS.md

- This file is injected into every WorkBench system prompt, so its size is paid
  for on each message. Stay under ~300 lines.
- Describe the **big picture**: what the pieces are and which invariants hold.
  Route-level detail, exhaustive lists and per-feature essays belong in the
  code, the README, or a doc comment next to the thing they describe.
- Keep out anything a routine code change would make wrong: counts ("three
  tabs", "four routes"), named enumerations of members, test totals and version
  numbers. State the invariant instead — *what* the write surface guarantees,
  not how many endpoints implement it. The code and its tests are the source of
  truth for those details. Naming the *one* test that enforces an invariant is
  the exception: it is a pointer, not a count.
- One idea per bullet, no single long line — wrap prose at ~100 columns.
- Describe the app as it is now. Nothing about a past session, a rename, a
  previous state of the file, or what used to be true.
- When code changes, delete or shrink the paragraph here rather than appending
  a newer, longer one beside it.
