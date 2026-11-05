# AGENTS.md — benchmarkit

## What it is

A self-hosted FastAPI + Vue 3 app that benchmarks LLM inference endpoints
(llama.cpp or any OpenAI-compatible server). It streams each question of a
question set to the configured model, measures how fast and how well the model
answers, and grades every answer against the expected one with a secondary
**verifier LLM**. Everything reaches the browser live over Server-Sent Events.
Desktop browsers only.

## Rules for this file

This file is written **for agents** — coding, review and architect agents alike
— and it is injected into every WorkBench turn, so every line is paid for on
every message.

- It states **what the app is and how its pieces fit**, not how to build it.
- No **machine-specific setup**: no virtualenv manager, package-manager or
  browser-install commands, no ports or paths tied to one machine. That differs
  between machines and belongs in the README.
- No **code details that move**: no line numbers, quoted code, exhaustive route
  or member lists, counts or version numbers. The code and its tests are the
  source of truth for those.
- A **big-picture view only**: what each part is and which invariants hold.
  Feature-by-feature detail belongs in the README, route-level detail in the
  code and its doc comments.
- Keep it **short and current** — the shorter it is, the more of it a model
  actually holds. Cut or shrink a paragraph when the code it described changes
  instead of appending a newer one beside it, and describe the app as it is now,
  never as it used to be.

Setup, run and test commands live in the README; read that when you need to
build, install or launch something on this machine.

## Shape of the app

The tabs share one LLM configuration and one streaming layer:

- **Benchmark** — runs a question set against the configured model, showing
  answers, scores, timings and token/s as they arrive.
- **WorkBench** — interactive streaming chat with that same model, optionally
  augmented with MCP connectors and language servers, plus a batch mode that
  works through a folder of task files.
- **Editor** — the working location as a file tree and a code pane. Its writes
  form one bounded surface reachable from the browser only, never from a hub or
  a tool. The app's one shell — the Editor's terminal — is a human surface too,
  never an agent tool.

## Code layout

### Backend (`src/`)

A UI-free **core** and the **API** in front of it. Core functions never touch
FastAPI — they take an injected `emit(event: dict)` callable and report progress
through it.

- `config` — `.env` loading, the managed-key template, question-set discovery,
  and the project-instructions block injected into chat.
- `api` — the REST and SSE endpoints under one route prefix (everything else
  serves the SPA), plus the event bus that numbers events so a reconnecting
  client can replay what it missed.
- `core` — one streamed LLM call with model warm-up and the verifier call; the
  benchmark loop; one streamed chat reply including the shared tool-call loop;
  per-run results and their aggregates; the batch-mode lot listing (task files
  are listed through the path policy, never read); the WorkBench agent modes.
- `core/mcphub/` and `core/lsphub/` — the MCP and language-server hubs. Both
  declarations are the **working location's** (its reserved folder): the MCP
  one is seeded by exclusive create the first time a folder is opened, from
  the committed template in `mcp_connectors/`; the language-server one is
  hand-written — its seed leaves only `.nojae/lsp-settings.template.json`, a
  copy of the committed example in `lsp_connectors/`, because a server entry
  names machine-local paths and a seeded declaration would load nowhere.
  Both are hot-toggled from the UI. The two mirror each other — `constants`
  (paths, timeouts), `discovery` (read/validate/write the declaration file),
  `manager` (the live hub), plus rendering of tool results, and `seed` on
  each side (MCP leaves the workspace its declaration file, LSP only its
  template). Keep that symmetry when touching either one.
- `core/editorfs/` — the Editor's file surface. Paths are relative to the working
  location and every call answers from the workspace policy's verdict. Reads are
  bounded; writes to `ro`/`hidden` are refused; and no write destroys bytes — a
  save backs up what it replaces, a delete moves the file into a dated batch in
  the reserved folder. What the tree lists, the search can answer about, and
  what is invisible to both is the same set of paths.
- `core/termhub/` — the Editor's terminal: one shell per WebSocket, on the
  platform's pseudo-terminal (a PTY, or ConPTY); `$SHELL` on POSIX, and on
  Windows `TERMINAL_SHELL` or, by default, Git Bash.
- `core/proctree.py` and `core/proctree_jobs.py` — the process-tree teardown the
  hubs and the terminal share: one seam, group signalling or a job object.

### Frontend (`web/src/`)

Vite + Vue 3, plain JavaScript, Pico.css.

- `composables/` hold run and stream state and own each EventSource and its
  reconnect; components use them directly.
- `lib/` holds the single markdown renderer — all rendered HTML goes through it,
  never raw `v-html`.
- `components/` and `styles/` are the widgets and the theme.
- The tree's per-file-type icons are a **generated unit**: the glyph mapping, the
  font subsets and their coverage all come from the tables in
  `tools/nerd_icons.py`. Edit the tables and regenerate; never hand-edit the
  generated artifacts.

Configuration — `.env` and the two hub declaration files — is edited from the UI
and takes effect hot, so a change never needs an app restart.

## How a request flows

The exact route list belongs to the API module; what matters architecturally:

- **Run**: the benchmark starts in a worker thread and is watched live over SSE.
  Only one run at a time (a chat request is refused while one runs), and
  cancellation is honoured both between questions and mid-stream.
- **Chat**: one request per reply, streamed back over SSE. The client re-posts
  the whole history each time; the server keeps no conversation state.
- **Every WorkBench turn runs in an agent mode** — a workspace-owned persona in
  the reserved folder's mode file that replaces the configured system prompt for
  chat (never for the benchmark) and may switch a whole hub off. A mode only ever
  narrows what settings and the conversation already allow; it never grants
  anything. This file is injected into every chat turn whatever the mode, and one
  mode is app-owned and not editable from that file.

### SSE is the UI's only source of truth

Core emits plain dicts into the event bus, which numbers them so a reconnecting
client replays what it missed. Each stream has its own channel and closes with
exactly one terminal event; a reply the user paused still ends on its own ack,
keeping its partial turn.

Measurement rules hold across every stream:

- **Timing is app-side**, anchored to content-chunk arrivals — warm-up and
  verifier time never enter the numbers.
- **Cost and rate are different numbers**: token/s uses completion tokens only,
  the displayed cost is prompt + completion.

## Data files

| File | Role |
|---|---|
| `.env` | Run configuration: endpoint, key, prompt, sampling, set, verifier. |
| `.env.test` | Test-only endpoint, same key set as the template. |
| `resources/*.json` | Question sets — every file there is one selectable set. |
| `results.json` | Run output, written once when a run ends. |
| MCP connector declarations | Which connectors exist: `remote` (HTTP) or `local` (stdio) — the working location's file in the reserved folder, seeded from a committed template. |
| Language-server declarations | Which servers exist: a stdio command and its workspace root — the working location's file in the reserved folder, hand-written from the seeded template (the committed example in `lsp_connectors/` is that template's source). |
| Reserved workspace folder | This workspace's read/write policy, its agent-mode personas, its hub declarations, and the app-owned prompt-helper store. |

Those files may hold secrets, so they are gitignored and each has a committed
template; the reserved folder's files have no template — the app seeds them.
These rules tie them together:

- **`.env` is regenerated from a fixed template on every save** — fixed key set
  and order, empty optional keys commented out, hand-written extras lost. A new
  or renamed key means changing the template, the renderer and the committed
  example together.
- **The declaration files are the only source of truth for what runs.** Their
  `enable` flag is the switch; their secrets never leave the server. A value may
  reference an environment variable so one declaration works on every machine; an
  unresolved name is kept verbatim with a note, never blanked. `${PROJECT_ROOT}`
  is no environment variable: it answers from the app's own computed install
  path, so a stray name in the environment cannot redirect a connector.
- **A test run takes its endpoint from the test env file, never from a
  developer's.**
- **One setting defines the working location** that every file-touching part of
  the app roots itself at. It is deliberately outside the settings file, so a
  settings save can neither erase it nor resurrect a stale value. Every path the
  app holds is relative to it, so moving it invalidates them all at once: cached
  views are discarded rather than re-negotiated, and unsaved work is reported
  rather than silently dropped.
- **The reserved folder is seeded, then read-only to the app**: its files are
  created only when absent, so an edited file is never clobbered, and they are
  changed only outside the app — except the app's own prompt-helper store,
  which it writes. The folder is hidden from the tree, from reads and from
  git status; failed seeding is a reported note, never a refusal. Deleted files
  land in its dated batches instead of vanishing.
- **Path rules come from two levels, one verdict: the strictest match wins.** One
  level is the app's rules about files the app itself writes, active only while
  the working location is the app's own directory — so another project's config
  files are ordinary files. The other is the workspace's own file, always active:
  it can tighten the app's rules but never loosen them. A hidden path is
  indistinguishable from an absent one to a reader and named to a writer; a
  read-only path is searchable but named to a writer.
- **The connectors answer the workspace level for their own root**, each reading
  that policy in its own process, so a read-only folder is uneditable by an agent
  too — the app and the connector processes share one parser, one matcher and one
  refusal sentence. The deny-lists guarding everything outside a root stay
  separate from the workspace's rules on purpose: merging them would move secrets
  into a per-project file.

## Key patterns

- **One LLM layer.** Every model call — benchmark answer, verifier, chat — goes
  through `core/llmkit`, the only seam that speaks HTTP to an OpenAI-compatible
  endpoint, so adding a model is configuration, never new code. It sends the
  sampling parameters **verbatim** (no whitelist), so an unknown sampler name is
  invisible rather than an error (`tests/test_llmkit_contract.py`). Without a
  verifier endpoint an answer cannot be graded, and is failed rather than guessed.
- **Core owns logic, the API owns transport.** Blocking work runs in a worker
  thread publishing to the event bus; the streaming endpoint consumes it. Run and
  chat share that worker-plus-cancellation pattern, and nothing under `core` may
  import the API layer.
- **Hubs are long-lived singletons.** A restart rebuilds a hub's internals but
  never replaces the object the app holds, because worker threads keep the
  reference across a request.
- **A declared-but-unconnected language server is still published** from its
  declared tools and revived by its first call; only being switched off or
  errored keeps it out of the catalogue.
- **Teardown kills process *trees*, not wrappers**, on either platform and by
  either mechanism, so children a hub or the terminal spawned of their own never
  outlive the app that started them.
- **Tools are read-only by design.** Every tool returns information, the shell
  connector execs a single command with no shell (so command chains, pipes and
  redirections are unrepresentable), and each connector's reach is bounded by its
  own deny-list. `tests/test_write_surface_isolation.py` asserts that instead of
  trusting this sentence: no connector, tool or chat path reaches a write route,
  and the terminal is never a tool.
- **The Editor's write surface is the browser's alone**, and it is refused while
  a chat reply streams. Batch mode lists its lot of task files through the shared
  path policy but never reads or writes their contents — only the connectors do.
- **A lot runs every task in one mode, pinned at launch** — the workspace's coding
  mode, never whatever the composer currently has selected, so moving that choice
  mid-lot reaches no task. A workspace with no such mode falls back to the
  server's default and says so rather than being refused.
- **A task is finished when its file leaves the lot folder**; that move is the
  agent's own acknowledgement, and re-listing is the only way the loop can see it.
  A file still there is relaunched up to the loop's budget, then the lot aborts.
- **This file feeds the chat**: when the working location carries it, it is
  injected into every chat turn and re-read per request. The model acts on it, so
  accuracy matters more than volume.
- **Tests stay hermetic**: connectors and language servers can be disabled by an
  environment flag, and a test patches a hub's declared file location through that
  hub's own constants rather than a path it computed.

## Gotchas

- **A stream cut by the server looks like a finished answer** at the chunk level.
  The honest signal is whether a real finish marker was ever seen, read after the
  loop is drained. A truncated benchmark answer is failed and never sent to the
  verifier (grading a fragment measures the transport, not the model), and a
  truncated chat reply ends as an error, which also stops a half-streamed tool call
  from being executed.
- **Token counts can look server-reported when they are not**: a stream that dies
  before any usage chunk gets a re-counted back-fill, so a reported figure is not
  evidence of where it came from. Keep the estimated flag honest.
- **Adding a connector is never a module under `src/`** — a local connector is a
  script (usually beside the shipped ones in `mcp_connectors/`) plus an entry in
  the workspace's declaration file naming it. Both halves are needed.

## Maintaining this file

Apply the rules above whenever you edit it: keep the big picture, cut what the
code already says, and name the invariant rather than describing an implementation
that a routine change would move. One idea per bullet, and no line so long it
stops being read. A test that exists to enforce an invariant is worth naming as a
pointer — the invariant stays the point.
