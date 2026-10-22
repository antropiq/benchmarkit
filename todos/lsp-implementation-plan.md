# LSP support plan (`lsp_hub`) — Benchmark'it

## 0. Goal and hard constraints

Expose the language servers already installed on the machine (volta shims in
`/home/vgr/.volta/bin`: `pyright-langserver`, `typescript-language-server`,
`vue-language-server`) as **tools the Chat LLM can call**, while the language
servers themselves are **spawned, initialized, kept alive and shut down by the
app** — exactly like `core/mcp_hub.py` owns the MCP connectors.

Derived requirements this plan honours:

1. **No console detour.** The LLM must never see `pyright-langserver` as a
   shell command; the console connector must not become an LSP client. The
   hub is the only party that talks JSON-RPC over stdio.
2. **Managed lifecycle**: start with the app (or lazily on first use),
   hot-toggle without restarting, stop on shutdown, logs per server, crash
   surfaced as a status — the `McpManager` contract, transposed.
3. **Declarative config**, JSON, keyed by id, with a single `enable` flag as
   the on/off source of truth (the `mcp_connectors/externals.json` precedent).
4. **Core/API separation**: `src/core/lsp_hub.py` never imports FastAPI; it
   exposes OpenAI-format tool defs + a sync call bridge, and the routes own
   transport. Type annotations explicit, as elsewhere.
5. **Benchmark untouched**: LSP tools join the Chat tool loop only, never a
   benchmark run, so scores stay comparable (same rule as the chat `AGENTS.md`).
6. Desktop-only UI, Pico.css, no new frontend dependency.

## 1. Why this is not just another MCP connector

Three properties of LSP make a straight "declare it in `externals.json`"
solution impossible — every one of those servers speaks **LSP over stdio**, not
MCP:

- **No tool concept.** LSP exposes *requests* (`textDocument/hover`, …) and
  *notifications*. Someone has to synthesise the tool catalogue, the JSON
  schemas and the argument→LSP-params mapping. That "someone" is our hub.
- **Stateful sessions.** `initialize` → `initialized` → `didOpen` (the server
  only analyses text *we* push it, with `versioned` semantics) → requests →
  `shutdown`/`exit`. A document store and a version counter are mandatory.
- **Push channel.** Diagnostics arrive as a server→client *notification*
  (`textDocument/publishDiagnostics`), so the hub needs an inbox and a
  bounded "wait for the server to catch up" step. There is no MCP-style
  `call_tool` that just returns them (unless the server implements 3.17
  `diagnosticProvider` pull diagnostics — see §6.2).

Verified on this machine (`mcp` 2.3.0 installed; **`pygls`/`lsprotocol` are
not installed** — a new optional extra is required):

| Binary | Transport facts found in the shipped code | Notes |
|---|---|---|
| `pyright-langserver` | `stdio` handled in `dist/pyright-internal.js`; the langserver entry needs `--stdio` | Works standalone; reads `pyrightconfig.json` from the workspace root |
| `typescript-language-server` 6.0.2 | `--stdio` in `lib/cli.mjs`; `initialize()` **throws** unless it finds a `tsserver.js` | See §7.1 — volta's `typescript@7.0.2` ships **no `tsserver.js`** |
| `vue-language-server` | `@volar/language-server/node` → `vscode-languageserver/node` `createConnection(ProposedFeatures.all)`; `--stdio` detection lives in that bundled `vscode-languageserver/lib/node/main.js` | Accepts `--tsdk=<typescript/lib>`; its bundled `typescript@7.0.2` exports only a stub — §7.2 |

## 2. Chosen architecture

**Native `src/core/lsp_hub.py`, mirroring `mcp_hub.py` structure, plus a
JSON declaration file.** Both hub files then look alike to the reader:
discover/validate config → spawn per entry → connect → cache tool defs →
`*_sync` bridge for chat worker threads → status list for the API.

Rejected alternative (kept as fallback, §8): an `mcp_connectors/lsp_mcp.py`
connector that embeds an LSP client and re-exposes it as MCP tools. It would
reuse the manager, hot-toggle and `/api/mcp` UI for free, at the cost of an
extra process hop, a second config language, and the same amount of LSP code
written anyway — the hard part (sessions, diagnostics inbox, tool synthesis)
does not come from MCP.

```
Chat LLM  ──tool call `lsp_pyright_diagnostics`──►  chat.py tool loop
                                                        │ tool_router
routes._lsp_tool_kwargs ──► LspManager.call_tool_sync(id, method, args)
                                                        │ asyncio thread "lsp-hub"
                                            DocumentStore / DiagnosticsInbox
                                                        │ JSON-RPC over stdio
                              pyright-langserver --stdio   tsserver   vue-language-server
```

## 3. New files

| Path | Role |
|---|---|
| `src/core/lsp_hub.py` | Discovery/validation of `lsp_servers.json`, spawn + LSP handshake per server, document store, diagnostics inbox, tool-catalogue synthesis, `call_tool_sync` bridge, status list. Core-only (no FastAPI). |
| `lsp_connectors/lsp_servers.json` | The declaration file (gitignored — see §7). |
| `lsp_connectors/lsp_servers.example.json` | Committed template documenting every key. |
| `lsp_connectors/restricted-paths.txt` *(optional)* | Deny-list of workspace roots the hub refuses to open, mirroring `mcp_connectors/restricted-fqdn.txt`. |
| `tests/test_lsp.py` + `tests/fake_langserver.py` | Config validation, tool synthesis, an in-test stdio fake language server, API endpoints, chat routing. |
| `web/src/components/ChatLspPanel.vue` | Per-conversation LSP toggles (sibling of `ChatMcpPanel.vue`). |
| `web/src/composables/useLsp.js` *(or a `lsp` slice inside `useChat`)* | Snapshot + per-conversation allowlist + status dots. |
| `web/src/components/SettingsDialog.vue` | A new `LSP` section: activation switches, status badges, workspace root, tool list. |

## 4. Configuration file (`lsp_connectors/lsp_servers.json`)

Same shape as `externals.json`: an object keyed by server id, per-entry
validation that skips a bad entry with a note instead of failing the whole
file, and `enable` as the single on/off source of truth.

```json
{
  "pyright": {
    "enable": true,
    "label": "Pyright (Python)",
    "extensions": [".py"],
    "command": ["/home/vgr/.volta/bin/pyright-langserver", "--stdio"],
    "workspaceRoot": "/home/vgr/dev/benchmarkit/web",
    "env": { "PATH": "/home/vgr/.volta/bin:/usr/bin:/bin" },
    "initializationOptions": { "typeCheckingMode": "standard" },
    "tools": ["diagnostics", "hover", "definition", "references",
              "document_symbols", "completion", "formatting"],
    "maxDiagnostics": 40,
    "idleShutdownSeconds": 300
  },
  "typescript": {
    "enable": true,
    "extensions": [".ts", ".tsx", ".js", ".mjs", ".json"],
    "command": ["/home/vgr/.volta/bin/typescript-language-server", "--stdio"],
    "workspaceRoot": "/home/vgr/dev/benchmarkit/web",
    "initializationOptions": {
      "tsserver": { "path": "/home/vgr/.volta/tools/image/packages/typescript" }
    }
  },
  "vue": {
    "enable": false,
    "extensions": [".vue"],
    "command": ["/home/vgr/.volta/bin/vue-language-server", "--stdio"],
    "workspaceRoot": "/home/vgr/dev/benchmarkit/web"
  }
}
```

Rules the loader enforces — `load_lsp_servers() -> (servers, errors)`, never
raises, a literal transcription of `load_externals()`' policy:

- No `type` key: an LSP server is always local stdio. `command` must be a
  non-empty argv array, or a command string parsed with `shlex` — reuse
  `_argv_from_command()` from `mcp_hub.py` so both config files behave
  identically. **Never a shell.**
- `extensions` (optionally `languageId`) decide which server answers for a
  path; the first server whose list matches wins. `.vue` goes to
  `vue-language-server`; routing its `<script lang="ts">` region to
  `typescript` (`nested: true`) is an M3 nice-to-have.
- `workspaceRoot` must be absolute, must exist, and is refused when it hits
  the deny-list. It deliberately does **not** default to the project root, so
  Pyright is not pointed at the whole repo (pyenv venv paths would make it
  noisy and slow) — `web/` for TS/Vue, and a Python root of your choice.
- `env` may carry secrets, so the file is gitignored and `env` values are
  stripped from every API payload (same rule as MCP external `headers`/`env`).
- Unknown keys are ignored, a malformed entry is reported per entry
  (`lsp_servers.json: 'vue': … — skipped`).

`.env` additions (added to `SETTING_KEYS` **and** `_render_env()` +
`.env.example`, so the settings dialog keeps managing the whole file):

```
LSP_ENABLED=pyright,typescript     # ids to activate, like MCP_ENABLED
LSP_SERVERS_FILE=lsp_connectors/lsp_servers.json
BENCHMARKIT_DISABLE_LSP=0          # test/dev switch, mirrors BENCHMARKIT_DISABLE_MCP
```

Open decision: a separate `MAX_LSP_ITERATION` or reuse `MAX_MCP_ITERATION`?
`stream_chat` has **one** iteration cap for the whole loop regardless of which
hub served the tool, so I recommend reusing `MAX_MCP_ITERATION` and relabeling
it "Max tool iterations" in the UI — one knob, no split-brain.

> We will go MAX_TOOL_ITERATION way to keep only one counter (no UI change for the progress bar)

## 5. `src/core/lsp_hub.py` — public surface

Deliberately parallel to `mcp_hub.py`, so a reader who knows one hub knows both.

Module level (pure functions, no I/O beyond reading the JSON file):

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LSP_CONNECTORS_DIR = PROJECT_ROOT / "lsp_connectors"
LSP_LOGS_DIR = PROJECT_ROOT / "logs" / "lsp"
LSP_SERVERS_PATH = LSP_CONNECTORS_DIR / "lsp_servers.json"

def lsp_installed() -> bool            # pygls + lsprotocol importable?
def load_lsp_servers(path=None, reserved_ids=None) -> tuple[list, list[str]]
def set_lsp_enabled(enabled: dict[str, bool], path=None) -> list[str]   # writes `enable` back
def all_servers() -> tuple[list, list[str]]                             # + validation errors
def tools_for_server(server: dict) -> list[dict]   # OpenAI function-calling defs
```

Class, with the same method names as `McpManager` where the semantics match:

```python
class LspManager:
    def __init__(self, servers, python_exe=None, project_root=PROJECT_ROOT, logs_dir=None): ...
    installed: bool
    async def start_all(self) -> None          # spawn + handshake, never raises
    async def stop_all(self) -> None           # shutdown/exit, SIGTERM→SIGKILL grace
    def start_server(self, server_id) -> None  # fire-and-forget hot start
    def stop_server(self, server_id) -> None   # blocking-ish, so status is readable after
    def server_status(self, server_id) -> str | None
    def active_tool_defs(self, server_ids) -> tuple[list[dict], dict[str, str]]  # tool → server
    def call_tool_sync(self, server_id, tool, arguments, timeout=30.0) -> str
    def status(self) -> list[dict]             # defs + live status, `env` stripped
```

Internals worth naming in the design:

1. **One private asyncio thread** (`name="lsp-hub"`), exactly like the MCP hub's
   `"mcp-hub"` thread, so the chat worker threads stay sync and bridge on with
   `run_coroutine_threadsafe(...).result(timeout=…)`. LSP servers are chatty and
   slow on cold start (tsserver indexing), so requests need per-method timeouts
   (`REQUEST_TIMEOUT = 20s`, `INIT_TIMEOUT = 60s`) rather than the MCP default.
2. **Lifecycle per server**: `create_subprocess_exec(argv…, stdin=PIPE,
   stdout=PIPE, stderr=log_file)` writing to `logs/lsp/<id>.log`, then
   `initialize` (with `clientInfo`, `processId`, `rootUri`, `capabilities`,
   `initializationOptions`) → `initialized` → ready. Teardown sends
   `shutdown` then `exit`, waits `STOP_GRACE_SECONDS`, then `terminate()`/`kill()`
   — the same pattern as `_close_connector`.
3. **DocumentStore**: `path → (uri, version, text)`. Every tool call that names
   a file first `didOpen`s it (or `didChange` with a full-text
   `contentChanges` when it is already open) so the server analyses *current*
   content, not a stale on-disk snapshot. The version counter increments
   monotonically; documents are closed (`didClose`) when the chat conversation
   ends or after `idleShutdownSeconds`.
4. **DiagnosticsInbox**: the server→client notification
   `textDocument/publishDiagnostics` is buffered per URI (latest wins).
   `lsp_<id>_diagnostics` opens the document, then awaits either the next
   publish for that URI or a deadline (Pyright publishes ~300 ms after a
   change; tsserver can take seconds on a cold index). If the server declares
   `diagnosticProvider` (LSP 3.17 pull diagnostics) the hub prefers
   `textDocument/diagnostic` and skips the wait.
5. **Server→client requests** we must answer, not crash on: `window/workDoneProgress/create`,
   `client/registerCapability`, `workspace/configuration`. Return benign
   defaults (`configuration` → `[{}]`, registerCapability → `null`) and drop
   `window/logMessage` / `$/logTrace` into the per-server log file. tsserver is
   the noisiest here, this is what makes or breaks the Vue/TS servers.
6. **Lazy start**: `enable: true` servers start with the app, but a server whose
   first call is minutes away costs nothing; conversely, to keep the app snappy
   the default should be **lazy spawn on first tool call** for servers marked
   `"lazy": true` (recommended for `typescript` and `vue`: tsserver cold start
   is multi-second and eats RAM). Status then shows `starting` — the UI already
   renders that state.
7. **Idle reaper**: a server untouched for `idleShutdownSeconds` (default 300,
   `0` disables) is torn down and re-spawned on demand. Pyright and tsserver
   are both heavy; the benchmark app should not hold them forever.

## 6. Tool catalogue: LSP requests → LLM tools

Tools are synthesised from the declared `tools` list intersected with the
server's **actual** `initialize` capabilities — a server that does not advertise
`definitionProvider` simply does not get a `definition` tool, so the model never
gets offered something it will fail on. Names are namespaced
`lsp_<serverId>_<action>` (mirrors the "tool → connector" attribution map already
built by `active_tool_defs`, and keeps ids unique across hubs).

| Tool | LSP request | Args the model sees |
|---|---|---|
| `lsp_pyright_diagnostics` | `didOpen` + wait `publishDiagnostics` (or pull `diagnostic`) | `path` |
| `lsp_pyright_hover` | `textDocument/hover` | `path`, `line`, `character` |
| `lsp_pyright_definition` | `textDocument/definition` | `path`, `line`, `character` |
| `lsp_pyright_references` | `textDocument/references` | `path`, `line`, `character`, `includeDeclaration` |
| `lsp_pyright_document_symbols` | `textDocument/documentSymbol` | `path` |
| `lsp_pyright_completion` | `textDocument/completion` | `path`, `line`, `character`, `limit` |
| `lsp_pyright_formatting` | `textDocument/formatting` (result **returned**, never written) | `path` |
| `lsp_pyright_signature_help` | `textDocument/signatureHelp` | `path`, `line`, `character` |
| `lsp_pyright_workspace_symbols` | `workspace/symbol` | `query` |
| `lsp_pyright_rename_preview` *(M3)* | `textDocument/prepareRename` + `rename` returned as a text-edit preview | `path`, `line`, `character`, `newName` |

Conventions that keep this cheap and safe for a benchmarking app:

- **Zero-based positions, documented in the tool description**, and the
  description explicitly says "line/character as reported by
  `lsp_*_diagnostics`" — otherwise models invent one-based offsets and blame the
  server. Better: accept the diagnostic's own `range` object
  (`{"range": {"start": {"line":12,"character":4}}}`) so the model can round-trip
  what it just got.
- **Text out, never raw JSON.** Convert results to the compact, line-numbered
  style the fedit connector already uses: `web/src/App.vue:41:13 error: Property
  'foo' does not exist… (ts-plugin2339)`. Positions in the text carry
  1-based line/column for human/model readability, while inputs stay
  0-based LSP-native; state that asymmetry in the description.
- Reuse `extract_tool_result_text`'s truncation idea: cap at
  `maxDiagnostics` items (default 40) and at ~12 000 chars per result —
  diagnostics dumps are the single biggest context-window risk here.
- **Read-only by default.** `apply_edit`-style writes stay out of scope: the
  hub returns proposed text edits (formatting, rename) for the LLM to feed to
  the fedit connector, which owns backups, the deny-list and revert. That keeps
  one write path in the app and avoids two tools racing on the same file
  (crucial, because the hub holds an open document whose buffer would then be
  stale). If a later `lsp_*_will_save_wait_until`/write tool is ever added, it
  must invalidate the DocumentStore entry and re-`didChange` from disk.
- **Path policy** before any `didOpen`: the file must be inside the server's
  `workspaceRoot`, must exist, must be a file, size-capped (say 1 MB), and its
  extension must be in `extensions`. This is what stops "read
  `~/.ssh/id_rsa` through the Python server" style tricks, and what makes the
  LSP hub useless as a generic file reader — the fedit connector's policy stays
  the single authority.

## 7. Machine-specific gotchas found while surveying this box

These are real, verified facts about the volta-installed servers — they decide
the config values, and each needs a workaround in the hub or in the config.

### 7.1 `typescript-language-server` cannot find a `tsserver`

`typescript-language-server@6.0.2` throws at `initialize()`: *"Could not find a
valid TypeScript installation… Please ensure that the `typescript` dependency
is installed in the workspace or that a valid `tsserver.path` is specified."*
On this machine:

- volta's `typescript@7.0.2` (`~/.volta/tools/image/packages/typescript`) ships
  only `lib/{tsc.js,version.cjs,getExePath.js}` — **there is no `tsserver.js`**
  anywhere under `~/.volta/tools/image/packages` (TypeScript 7 native removes
  the classic `tsserver.js` entry point).
- `/home/vgr/dev/benchmarkit/web/node_modules` is absent (not built yet), and it
  does not list `typescript` as a dependency anyway — so there is no *workspace*
  TypeScript either.

Consequences / options, in order of preference:

1. `cd web && npm install && npm run build` (already required by the project
   rules for frontend work) **plus** adding a real compiler to the dev
   dependencies, e.g. `npm i -D typescript@5` — then
   `web/node_modules/typescript/lib/tsserver.js` exists and the server picks it
   up as the *workspace* version automatically. Cleanest, and it also unblocks
   vue-language-server (§7.2).
2. Point the config at a tsserver that exists —
   `initializationOptions.tsserver.path` accepts a directory *or* a file, and
   resolves siblings as `lib/tsserver.js`. There is no such file on this box
   today, so this is only viable after installing TypeScript 5 somewhere
   (a pinned helper package, e.g. `lsp_connectors/ts/package.json` with
   `typescript@5.x`, kept out of `web/` so the app build is untouched).
3. Leave `typescript` `"enable": false` for M1 and ship **Pyright +
   vue-language-server first**; the TS server is the only one of the three with
   no working compiler on the machine.

> We will choose point 1: Add a real compiler dependency and use it.

The hub should not paper over this: an `initialize` failure must land in
`status[].error` with the server's own message (the MCP hub's `_log_tail()`
trick), because "TS server died: no tsserver" is a config problem the user can
only fix if it is reported verbatim.

### 7.2 `vue-language-server` needs a TypeScript SDK

Its entry (`@vue/language-server/index.js`) honours `--tsdk=<dir>` and otherwise
falls back to `require('typescript')`; the copy it bundles
(`node_modules/typescript@7.0.2`) exposes only `lib/{tsc.js,getExePath.js}` —
same stub as §7.1, so the fallback is not usable. Fix: pass the SDK explicitly
in argv — `"command": ["…/vue-language-server", "--stdio", "--tsdk=<typescript/lib>"]`
— which again requires a TypeScript 5 `lib` folder with `tsserverlibrary.js`.
Volar also drives tsserver through its own `volar-service-typescript`, so the
same missing-compiler problem applies: without a usable TypeScript, the Vue
server starts but its TS features are dead. → Practical plan: **Pyright first**
(zero external dependencies), `typescript`/`vue` in M2 once a TypeScript 5 SDK
is pinned in the config.

### 7.3 Pyright and this repo

`pyrightconfig.json` is listed in `.gitignore` and does not currently exist in
the project root. Pointing Pyright at `/home/vgr/dev/benchmarkit` would make it
scan `web/`, `logs/`, and the pyenv-virtualenv-adjacent files; it would also
resolve imports against no environment at all, so litellm/fastapi imports would
be reported as missing. Recommended: `workspaceRoot` = the Python source root
you want analysed, plus either create a `pyrightconfig.json` (a human call —
gitignored, and it is also fedit-write-adjacent policy territory) or pass
`initializationOptions.typeCheckingMode` and `venvPath`/`venv` from
`lsp_servers.json`. Note `.pyenv` is on fedit's **write** deny-list, not on the
LSP hub's read path — the hub only needs the venv *readable*, which is fine.

> pyrightconfig.json as been added to this repo (stays a per MACHINE configuration file)

### 7.4 Volta shims

All three `~/.volta/bin` entries are symlinks to `volta-shim`. Spawning the
shim directly works and is what the config should do (it resolves the pinned
tool + node), but it adds a shim hop and its `PATH`/`TMPDIR` environment matters:
reuse `connector_environment()` from `mcp_hub.py` (it already pins a valid
`TMPDIR`, because the app's own `TEMP` sampling variable leaks into
`os.environ` and Node misreads it — the same trap would bite the language
servers), and set `PATH` to include `/home/vgr/.volta/bin` so nested `node`
lookups resolve.

## 8. Fallback option: an LSP-over-MCP connector

`mcp_connectors/lsp_mcp.py` (auto-discovered by `discover_connectors()` because
of the `*_mcp.py` name) that holds the LSP client inside, exposes
`lsp_diagnostics/lsp_hover/…` as MCP tools, and appears in the existing MCP
panel. Cheapest to ship: `/api/mcp`, `MCP_ENABLED`, the settings panel, the chat
tool loop and the tests all keep working unchanged.

Costs: an extra process hop; the LSP config file still has to exist somewhere
(hardcoded or a second file); the connector's own port allocation and
`--port` convention apply; LSP status is invisible to the UI (an MCP connector
that is "running" may still have three dead language servers behind it); and
the "managed by our app, not by the LLM" requirement is satisfied only
indirectly. Keep it as a spike (M0) to prove the handshake against Pyright in
~100 lines before committing to the hub — the same LSP client code moves into
`src/core/lsp_hub.py` afterwards.

> Option rejected.

## 9. Wiring: API, chat, frontend

### 9.1 Lifespan (`src/api/server.py`)

```python
connectors = [] if os.environ.get("BENCHMARKIT_DISABLE_MCP") else all_connectors()[0]
manager = McpManager(connectors)
app.state.mcp = manager
await manager.start_all()

servers = [] if os.environ.get("BENCHMARKIT_DISABLE_LSP") else all_servers()[0]
lsp = LspManager(servers)
app.state.lsp = lsp
await lsp.start_all()
try:
    yield
finally:
    await lsp.stop_all()
    await manager.stop_all()
```

Tests stay spawn-free: `tests/conftest.py` grows an autouse
`BENCHMARKIT_DISABLE_LSP=1` fixture next to the existing MCP one.

### 9.2 Endpoints (`src/api/routes.py`)

Mirror `/api/mcp` exactly, so the frontend code is near-duplicate too:

- `GET /api/lsp` — servers (file scan + validation errors) merged with live
  status, `env` stripped: `{ installed, servers: [{id,label,description,
  extensions,workspaceRoot,command,enable,active,status,error,toolCount}],
  externalsError }`.
- `POST /api/lsp` — `{ enabled: [ids] }`; 409 while a run is in progress (same
  guard as MCP); unknown ids → 422; built-in activation persisted to
  `LSP_ENABLED` in `.env` **and** the `enable` flag written back to
  `lsp_servers.json`, then `start_server`/`stop_server` hot-toggle. Since LSP
  entries are all file-declared (no built-in/external split), the `enable`
  flag is the single source of truth — simpler than MCP: `LSP_ENABLED` can be
  dropped entirely if you prefer one knob; decision point for the user.
- `POST /api/chat` body gains `lspEnabled: Optional[List[str]] = None`, the
  same conversation-scoped filter intersected with the Settings-activated set.
- Optional `GET /api/lsp/log?id=<server>` returning the tail of
  `logs/lsp/<id>.log` for the settings panel (mirrors the "log tail on startup
  error" idea already in `McpManager._log_tail`).

`_lsp_tool_kwargs(request, body.lspEnabled)` sits next to `_mcp_tool_kwargs()`
and returns the extra `stream_chat` kwargs. **The merge must happen in the
routes, not in `chat.py`** — `stream_chat` already takes
`tools/tool_executor/tool_servers/max_tool_iterations` and must keep one
loop. Concretely:

```python
tools, tool_to_server = _merge(active_mcp_defs, active_lsp_defs)  # LSP wins/ties by order

def tool_executor(name, arguments):
    if name in lsp_map:
        return lsp.call_tool_sync(lsp_map[name], name, arguments, timeout=LSP_TIMEOUT)
    return mcp.call_tool_sync(mcp_map[name], name, arguments)
```

No change to `core/chat.py` is needed at all — the event contract
(`tool_call` / `tool_result` / `usage`) already carries a `server` field, which
is exactly what makes the LSP tools attributable in the UI. That is the strongest
argument for the hub-mirrors-hub design.

### 9.3 Frontend (`web/`)

- `SettingsDialog.vue`: add `{ id: 'lsp', label: 'LSP' }` to `SECTIONS`; reuse
  the `.mcp-panel` markup/classes (status badge classes
  `green/violet/red/gray` already exist) and the existing save-button flow, with
  `loadLsp`/`saveLsp` in `useBenchmark`.
- `ChatLspPanel.vue`: copy of `ChatMcpPanel.vue` (status dot + switch, hidden
  when nothing is activated), fed by a `lsp` slice in `useChat.js` — same
  snapshot-on-tab-entry, same `locked` freeze once the first message is sent.
  `ChatSidePanel`/`ChatView` need no change: tool badges come from
  `ChatToolPanel`, which is hub-agnostic.
- No new npm dependency, desktop layout, existing Pico classes.

## 10. Dependencies

```toml
[project.optional-dependencies]
lsp = [
    "pygls>=2.0",       # client: JsonRPCClient/LanguageServerProtocol, stdio framing
    "lsprotocol>=2024.0",  # typed LSP messages (pulled in by pygls)
]
```

- Lazy-import it, exactly like `mcp_hub.py` does with `mcp`, so discovery and
  the settings panel keep working without the extra: `lsp_installed()` +
  `CONNECTORS_MISSING_HINT`-style message (`pip install -e ".[lsp]"`).
- Alternative with no new dependency: hand-roll the JSON-RPC framing over the
  subprocess pipes (`Content-Length: N\r\n\r\n`), ~150 lines, stdlib only. More
  control over timeouts, no extra install, but then `lsprotocol`'s typed
  messages and pygls' request/notification plumbing get rewritten by hand. I
  recommend the extra (it is a separate optional group, so the base install and
  the benchmark path stay untouched).
- AGENTS.md updates: the module table (`core/lsp_hub.py`), the HTTP list
  (`GET|POST /api/lsp`), the data-files section (`lsp_connectors/lsp_servers.json`,
  gitignored, secrets never returned), the gotchas entry (ports not relevant —
  LSP is stdio, but *logs in `logs/lsp/`*, `BENCHMARKIT_DISABLE_LSP`, and the
  tsserver/`--tsdk` trap), and the commands block (`pip install -e ".[lsp]"`).
- `.gitignore`: `lsp_connectors/lsp_servers.json`, `logs/` already covers the logs.

## 11. Tests (`tests/test_lsp.py`, mirroring `tests/test_mcp.py`)

1. **Config**: missing file → `([], [])`; invalid JSON → one file-level error;
   a bad entry skipped with a note; id collision with a disabled key; `env`
   never present in the `GET /api/lsp` payload; `set_lsp_enabled()` rewrites
   `enable` and preserves unknown keys and order.
2. **Tool synthesis**: only capabilities the fake server advertises become
   tools; names are `lsp_<id>_<action>`; the OpenAI shape matches
   `tools_to_openai_format()` so the merge in routes is trivial.
3. **Fake language server** (`tests/fake_langserver.py`, stdlib only, run with
   `sys.executable`): a ~60-line stdio JSON-RPC loop that answers `initialize`,
   `initialized`, `didOpen`, and pushes one `publishDiagnostics`, then replies
   to `hover`/`definition`. This covers handshake, document store, inbox wait,
   `shutdown`/`exit`, and the crash path (exit mid-conversation → status
   `error` with the log tail). No real Pyright/tsserver in CI, mirroring the
   "`BENCHMARKIT_DISABLE_MCP` never spawns real connectors in tests" rule.
4. **Lifecycle**: `start_all`/`stop_all` idempotence; `stop_server` then
   `start_server` re-spawns; a server that fails `initialize` lands in `error`
   without breaking the others.
5. **API**: `GET/POST /api/lsp` (409 during a run, 422 on unknown ids, `.env`
   rewritten with `LSP_ENABLED`, `enable` written back).
6. **Chat routing**: a stubbed manager proves the merged tool list and the
   name→hub router, and that `lspEnabled` cannot resurrect a server disabled
   in Settings (the exact invariant `_mcp_tool_kwargs` tests today).
7. Run the full suite (`pytest`) — the frontend is untouched unless the panels
   are implemented, in which case `npm run build` in `web/` is the check.

## 12. Milestones

- **M0 — spike (½ day)**: throwaway script (not committed) that spawns
  `pyright-langserver --stdio`, does the handshake, `didOpen`s one project file
  and prints its diagnostics. Proves the client choice (pygls vs hand-rolled)
  and the diagnostics-wait model against a real server.
- **M1 — Pyright hub**: `lsp_servers.json` + loader + validation errors;
  `LspManager` with lifecycle, document store, diagnostics inbox; tools
  `diagnostics`, `hover`, `definition`, `references`, `document_symbols`;
  lifespan + `GET/POST /api/lsp` + chat merge; settings section; tests.
  **This alone satisfies the requirement end to end.**
- **M2 — TypeScript + Vue**: pin a usable TypeScript 5 SDK (see §7.1/§7.2),
  fill in `initializationOptions`, teach the hub `workspace/configuration` and
  `client/registerCapability`; add `completion`, `formatting`,
  `signature_help`; per-conversation `ChatLspPanel`.
- **M3 — polish**: `rename_preview`, `workspace/symbol`, nested `.vue` →
  `typescript` routing, log-tail endpoint, idle reaper tuning, an
  "explain this error" system-prompt hint that points the model at
  `lsp_*_diagnostics` before it reaches for the console connector.

## 13. Security and failure posture

- **Process isolation is the guarantee**: the LLM only ever calls
  `lsp_<id>_<action>` with `{path, line, character, …}`. argv comes from the
  config file only — never from a model argument — and there is no shell.
- Path policy (§6) confines reads to the declared workspace root with the
  expected extension, size-capped; the deny-list mirrors the fedit lists so a
  secret file cannot be laundered through a Python language server.
- Server output is untrusted text entering the context window: it goes through
  the same truncation as MCP results, and the tool result is presented as data
  (a prompt-injection payload inside a diagnostic message is still possible —
  worth a line in the chat agent doc, same as for console/browser output).
- Resource guards: per-request timeout, per-server `maxDiagnostics`/result-size
  cap, `idleShutdownSeconds` reaper, `toolCount` shown in the panel, one hub
  thread so a wedged server cannot block the chat worker beyond its timeout.
  `stop_all()` runs in the lifespan `finally`, so no orphaned node process on
  shutdown (SIGTERM → grace → SIGKILL, as in `_close_connector`).
- Secrets: `env` values stay server-side; the API returns
  `command`/`workspaceRoot`/`extensions`/status only.

## 14. Open questions for you

1. **One `.env` knob or two?** `enable` in `lsp_servers.json` is already the
   truth for MCP externals. Should LSP keep it *file-only* (one source, no
   `LSP_ENABLED`), or mirror the built-in MCP pattern with `LSP_ENABLED` too?
   My recommendation: file-only, simplest and least to keep in sync.

   > We need to keep that LSP_ENABLED key since we could have an existing json configuration
     but do not want to load it.

2. **Iteration cap**: reuse `MAX_MCP_ITERATION` (my recommendation, §4) or add
   `MAX_LSP_ITERATION`?

   > Point out earlier: USE a MAX_TOOL_ITERATION key for both

3. **Eager or lazy spawn** at app start? Lazy keeps startup fast and RAM low,
   at the cost of a slow first tool call (visible as `tool_result` taking
   seconds). I'd default to eager Pyright + lazy TypeScript/Vue.

   > Confirmed

4. **Which workspace roots?** For Python: the repo root (needs a
   `pyrightconfig.json` for sane import resolution — a human creates it, it is
   gitignored) or a subfolder? For TS/Vue: `web/`.

   > The configuration file as been added to the repo.

5. **Read-only or write-capable?** I strongly recommend read-only (hub returns
   proposed edits; `fedit_mcp` applies them) — one write path, one backup chain.
   Do you agree, or do you want `lsp_*_format` to write files through the hub?

   > I agree only fedit_mcp is used in that case.

6. **Should the benchmark loop ever see LSP tools?** Proposal: never
   (scores must stay comparable) — same rule as the chat-only `AGENTS.md`.

   > The benchmark does not need LSP at all.
