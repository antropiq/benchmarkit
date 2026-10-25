# Editor tab — big picture

Add a third tab, **Editor**, to the right of WorkBench:

* **left**, full height: file tree of the *active workspace* (the working
  location, `BENCHMARKIT_PATH_LOCATION`, already shown in the app bar);
* **bottom**, full width minus the left column: a real terminal (xterm.js);
* **centre**, remaining space: a code editor (CodeMirror 6) with Python,
  JavaScript/TypeScript and Vue support.

This folder decomposes that feature. `ed-00` is the analysis and the map;
`ed-01`…`ed-09` are the steps, each one shippable and testable on its own;
`ed-08` holds what is parked and `ed-10` the write layer deliberately left to a
**future version** (decision D4).

## Verdict: feasible, no architectural break

Every heavy piece the feature needs already exists:

| Need | Already there |
|---|---|
| A third tab | `state.activeTab` + `setTab()` (`useBenchmark.js:10`, `:408`), `TabNav.vue`, and a `<X v-show="state.activeTab === '…'">` in `App.vue` |
| The workspace to browse | `GET /api/path-location` (`routes.py:1041`) returns the live location; `pathLocationCurrent` is fetched at boot for the app bar (`App.vue:18`) |
| A path policy to reuse | `mcp_connectors/path_policy.py` — deny prefixes, secret components, `.env`/key files; `lsphub.restricted_lsp_roots()` for workspace roots |
| Bounded, hermetic tests | `tests/test_api.py` (`TestClient` on `/api/*`); hubs disabled by `BENCHMARKIT_DISABLE_MCP/_LSP` (`conftest.py`) |
| Draggable split panes | `useSplitter.js` — pointer capture, clamp, localStorage, `ResizeObserver`; the pattern for the sidebar width and the terminal height |
| Dark/light theming | Pico CSS variables in `styles/theme.css`; xterm and CodeMirror are both themed from CSS variables |

Nothing constrains the UI from the backend: the SPA is served statically, all
routes live under `/api/`, and a new tab changes no existing behaviour.

## Hard constraint inherited from the codebase

AGENTS.md: **"Tools are read-only by design"** — LSP tools return information,
`console` execs one argv with no shell, and each connector is bounded by a
deny-list. The Editor tab is a *human* surface, not an agent surface, and that
distinction must be deliberate rather than incidental:

* **Nothing in this lot lets the agent write.** The editor is read-only, so the
  guarantee is stronger than "the write endpoint is browser-only": in the
  current lot **there is no write endpoint at all** (D4). When ed-10 adds one,
  it must stay out of the hubs' reach — no MCP tool, no LSP tool and nothing in
  `core/chat.py` may ever call it.
* **The two deny-lists still apply**, so the tab cannot become a GUI way to
  read `~/.ssh` or rewrite `mcp_connectors/` that the connectors refuse.
* **Read-only describes the editor surface, not the whole tab.** The terminal of
  ed-06/ed-07 is a real shell and can change anything the app's user can change.
  That asymmetry is deliberate: keeping the *editor* read-only is what stops the
  human's buffer from racing the model's writes, while the terminal is a human
  tool the chat and the verifier can never call.

That framing is set once here and reused verbatim by the tree, the editor, the
terminal and the live-reload steps.

## Target layout

```
┌ AppBar ────────────────────── Benchmark | WorkBench | Editor ─┐
├ editor-shell ─────────────────────────────────────────────────┤
│ ┌ navigator ┐ ┌ editor-area ────────────────────────────────┐ │
│ │ workspace │ │ tabs: src/config.py ×          read-only     │ │
│ │  tree     │ ├─────────────────────────────────────────────┤ │
│ │ (full     │ │  CodeMirror 6  (editable: false)            │ │
│ │  height)  │ │   ↻ badge when the model changed the file   │ │
│             │ │                                             │ │
│ ├───────────┤ ├─────────────────────────────────────────────┤ │
│ │ bottom of │ │  xterm.js  (bash, $PWD = workspace)         │ │
│ │ the tree  │ └─────────────────────────────────────────────┘ │
│ └───────────┘   ▲ splitter: terminal height                   │
▲ splitter: sidebar width                                       │
└───────────────────────────────────────────────────────────────┘
```

The brief says the terminal spans the width *minus* the bottom of the file
explorer: the tree column and the terminal's left edge share one line, which a
two-column grid gives for free — the navigator spans both rows, the editor and
the terminal stack in the right column. Both splitters are `useSplitter`-shaped
and persist to `localStorage` (`benchmarkit:editorSidebar`,
`benchmarkit:editorTerminal`).

## Dependency decision: xterm + CodeMirror, not Monaco

| | xterm.js + CodeMirror 6 (proposed) | Monaco |
|---|---|---|
| Terminal | the only maintained browser terminal emulator | none |
| Bundle | CM6 preset ≈ 400 kB min / ~700 kB gz, lazy | ~1.5 MB min, plus a worker |
| Workers | none — CM6 and xterm are worker-free, so the `StaticFiles` mount needs no change | needs `editor.worker.js` with worker semantics |
| Theming | CSS variables; Pico light/dark map directly | own theming API |
| Vue | thin wrapper: mount in `onMounted`, `destroy` in `onBeforeUnmount` | same, plus worker plumbing |

The decisive point is the worker: the SPA is served by a plain `StaticFiles`
mount (`server.py:98`), while Monaco wants a same-origin worker file. CodeMirror
and xterm are plain ES modules — `vite build` emits chunks and nothing else has
to be configured.

`web/dist` is already ~8 MB (mermaid alone is ~630 kB in one chunk), so the
Editor must be **lazy**: `defineAsyncComponent` for the view, dynamic imports
per language package, so the Benchmark tab never pays for it.

## New backend surface (all under `/api/`)

| Endpoint | Purpose | Step |
|---|---|---|
| `GET /api/fs/tree?path=<rel>` | one directory level, dirs first, symlinked dirs not followed, denied entries omitted; **directories carry `mtime` too** — ed-09's cheap "a child appeared/disappeared" predicate | ed-01 |
| `GET /api/fs/file?path=<rel>` | content + `mtime` + size + binary flag (413 past `MAX_EDITOR_BYTES`) | ed-03 |
| `GET /api/fs/stat?path=…&path=…` | `mtime`/`size`/`exists` for a handful of exact paths — the live-reload probe | ed-09 |
| `WS /api/terminal` | one PTY per connection: raw bytes + resize frames | ed-06 |
| ~~`PUT /api/fs/file`~~ — **not in this lot** | guarded write: `ifUnmodifiedSince` mtime → 409, and 409 while a chat reply streams | ed-10 |

One relativity rule everywhere: `path` is **relative to `path_location()`**,
resolved with `os.path.realpath`, rejected with 422 when it escapes the
workspace or hits a deny-list entry. Implementation lands in a new
`src/core/editorfs/` package mirroring the hub packages (`constants` with
patchable module-level paths, `policy`, `tree`, `files`), reusing
`path_policy.deny_reason()`.

## Terminal transport — D1 **decided: WebSocket**

`ws://127.0.0.1:8000/api/terminal`: one frame per keystroke (sub-ms on
loopback), and `fit` → an 8-byte resize frame. `websockets` is already installed
through `uvicorn[standard]`, so this adds **no Python dependency**; the only
config change is `ws: true` on the `/api` proxy in `vite.config.js`.

The alternative (SSE out + `POST` in) fits today's SSE-only layer but re-frames
every keystroke and makes `Ctrl-C` latency visible. Rejected; kept here as the
record of what was considered, and expanded in ed-06.

Server side: `pty.openpty()` + `subprocess.Popen` of `$SHELL` (fallback
`/bin/sh`) with `cwd=path_location()`, a reader thread pumping chunks to the
socket, `TIOCSWINSZ` on resize, and **tree teardown on disconnect** — the same
escalation rule as the hubs (AGENTS.md: kill the process *tree*, not the
wrapper). Linux-only is acceptable for a self-hosted loopback app; the endpoint
answers 501 elsewhere.

## Steps

File names group the topic (`ed-` prefix), the `#` column is the order of work —
so `ed-08` is a parked item and `ed-09`/`ed-10` are later than the number
suggests. Nothing in the file names implies a build order.

| # | File | Delivers | Depends on |
|---|---|---|---|
| 1 | `ed-01-workspace-navigator.md` | `GET /api/fs/tree` + `WorkspaceNavigator.vue` | — |
| 2 | `ed-02-editor-tab-shell.md` | Editor tab, `activeTab`, three-pane grid, two splitters, tree wired in | 1 |
| 3 | `ed-03-tree-selection-load.md` | click a file → content in a placeholder pane (`GET /api/fs/file`) | 1, 2 |
| 4 | `ed-04-codemirror-integration.md` | CodeMirror, languages, tabs, read-only editing surface | 3 |
| 5 | `ed-05-frontend-dependencies.md` | npm install of xterm + CM6, lazy-loading rules, bundle budget | — (before 4) |
| 6 | `ed-06-xterm-terminal-backend.md` | `WS /api/terminal`, PTY lifecycle, policy, tests | 2 |
| 7 | `ed-07-xterm-frontend.md` | xterm + `fit` + `web-links`, theme sync, resize | 2, 5, 6 |
| 8 | `ed-09-live-reload.md` | open buffers and the tree refresh while the model edits | 1, 3, 4 |
| 9 | `ed-08-editor-extras.md` | parked (not scheduled): quick-open, LSP diagnostics in the gutter, watcher, handoff | 4 |
| — | `ed-10-editor-write-layer.md` | **future version, not in this lot**: create / edit / delete like a standard editor, with the write guards and the chat-stream gate | 4, 8 |

Working order in practice: **1 → 2 → 5 → 3 → 4 → 6 → 7 → 8**, with 9 parked and
ed-10 deferred to a later version. Step 2 is the first commit a user can see;
nothing in this lot writes to disk, which is what makes the read-only-by-design
framing trivially true here — there is no write endpoint for a hub to reach.

## Decisions — D1…D9 all settled

| # | Question | Decision |
|---|---|---|
| D1 | Terminal transport: WebSocket, or SSE + POST? | **DECIDED — WebSocket** (ed-06) |
| D2 | Which shell, and a login shell? | **DECIDED — `$SHELL`**, non-login, `cwd` = workspace (ed-06) |
| D3 | Reuse `path_policy.py` or keep a file list of its own? | **DECIDED — reuse it**: one policy per surface that touches disk (ed-01) |
| D4 | What may the editor do to files? | **DECIDED — read-only in this lot.** No create, no edit, no delete: no write endpoint exists at all. While WorkBench is answering, the files the inference touches are refreshed into the open buffers (the live-reload rule, specified in ed-09; "answering" = a reply is streaming, the same window D9 settles for ed-10). The full standard-editor behaviour — create, edit, delete — is a **future version**, specced in ed-10 and explicitly out of this lot |
| D5 | Are unsaved edits kept across tab switches? | Moot in this lot (nothing is editable). `v-show` still keeps the view, the tree state and the terminal session alive (ed-02) |
| D6 | Is the tab disabled while a benchmark runs? | **DECIDED — no**: browsing is harmless, and with D4 no write exists to gate (ed-02) |
| D7 | `web/dist` is ~8 MB: accept new chunks or split harder? | **DECIDED — code-split per language**; no other change (ed-05) |
| D8 | How does the UI learn the model changed a file? | **DECIDED — poll `GET /api/fs/stat` for the open paths only**, driven by the `tool_result` frames the chat stream already emits; no file watcher in this lot (ed-09, watcher parked in ed-08) |
| D9 | *(ed-10, future version)* What counts as "a conversation in progress", during which the editor is read-only? | **DECIDED — (a): while a reply is streaming** (`_CHAT_HANDLE` set). Outside that window the human edits freely; editing during a conversation is not blocked by keeping the tab closed, it is blocked only while tools can write. The workflow this assumes is explicit: stop the reply, edit in Editor, go back to WorkBench and tell the model what changed — **the human owns that step**, and nothing in the app tries to detect or narrate it for them |

## Verification convention

* Backend: new `tests/test_editorfs.py` (+ `tests/test_terminal.py` for ed-06)
  — traversal attempts, deny-list, size cap, binary, the `stat` probe, PTY
  teardown — and one assertion that **no write route exists under `/api/fs/`**,
  which is this lot's read-only promise written as a test. The existing suite
  must stay green.
* Frontend: no runner exists, so each step ships a manual checklist, as in
  `todos/done/session-saving.md`.

## Index

`ed-01-workspace-navigator.md` (tree endpoint + component) ·
`ed-02-editor-tab-shell.md` (third tab + layout) ·
`ed-03-tree-selection-load.md` (selection → content) ·
`ed-04-codemirror-integration.md` (CodeMirror, read-only) ·
`ed-05-frontend-dependencies.md` (npm, lazy loading) ·
`ed-06-xterm-terminal-backend.md` (PTY over WebSocket) ·
`ed-07-xterm-frontend.md` (terminal panel) ·
`ed-09-live-reload.md` (buffers track the model's edits) ·
`ed-08-editor-extras.md` (parked enhancements) ·
`ed-10-editor-write-layer.md` (**future version** — create / edit / delete)
