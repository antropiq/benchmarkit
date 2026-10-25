# Step 6 — Terminal backend: one PTY per WebSocket

`WS /api/terminal`: a real shell (PTY) on the server, bridged to the browser
xterm.js pane of ed-07. Backend-only step; nothing user-visible until ed-07.

Depends on: ed-02 (the tab and its pane exist, so the endpoint has a consumer).

## Why a PTY and not `subprocess` + pipes

xterm renders an ANSI stream: colours, cursor movement, alternate screen, line
editing. A pipe pair gives none of that — `vim`, `top`, `less`, a coloured
prompt or `git diff` would emit or expect escape sequences that never get a
terminal state machine behind them, and there is no way to send `Ctrl-C` as a
signal rather than as bytes. `pty.openpty()` makes the kernel do the terminal
emulation; the server only relays bytes and updates the window size. Stdlib
only (`pty`, `fcntl`, `termios`, `struct`) — no new Python dependency.

## Transport decision (D1): WebSocket

| | WebSocket | SSE out + `POST` in |
|---|---|---|
| Keystroke latency | one frame, ~sub-ms on loopback | one HTTP request per keystroke |
| Resize | 8-byte frame | a second channel, sequencing headaches |
| Binary fidelity | raw bytes | base64 or JSON-escaped control chars |
| Available today | yes — `websockets==17.2` is installed through `uvicorn[standard]`, and FastAPI/Starlette expose `WebSocket` | yes (the app's existing pattern) |

The app's SSE layer (`api/events.py`, the chat stream) is built for *server to
client* progress events, and it is genuinely good at that — replay by `seq`,
reconnect, one terminal event per reply. A terminal is the opposite shape:
bidirectional, ephemeral, lossy-tolerant. Forcing it through SSE would mean
inventing a second protocol inside the SSE frames. Decision: **WebSocket**, and
the SSE conventions (replay, seq) are explicitly *not* applied here — a
terminal scrollback lives in xterm's buffer, not in a server-side replay log.

## Endpoint

```python
@router.websocket("/terminal")
async def terminal_ws(websocket: WebSocket) -> None:
```

Message framing, one JSON object per frame in each direction:

```jsonc
// client → server
{ "type": "input",  "data": "ls\r" }          // keystrokes, already ANSI-encoded by xterm
{ "type": "resize", "cols": 118, "rows": 30 } // from @xterm/addon-fit
{ "type": "kill" }                             // explicit session teardown (the pane's button)

// server → client
{ "type": "open",   "pid": 41237, "cwd": "/mnt/bigfoot/dev/benchmarkit", "shell": "/bin/bash" }
{ "type": "output", "data": "…" }             // raw PTY bytes, UTF-8 (latin-1 fallback)
{ "type": "exit",   "code": 0 }               // shell gone
{ "type": "error",  "message": "…" }          // then close
```

JSON text frames rather than binary: cheap to read, trivially debuggable with
`wscat`, and the throughput of a human typing is nowhere near the point where
framing matters. One frame per `os.read()` chunk (up to 4 KiB), so a `cat` of a
big file does not become one giant message.

## Session lifecycle

```
accept → guards → pty.openpty() → Popen(argv, stdin/slave/stdout=slave,
                                        cwd=workspace, env=…, start_new_session=True)
       → reader thread (or asyncio.to_thread) pumping os.read(master, 4096)
       → loop: receive() → input / resize / kill
       → finally: teardown, close
```

Hard requirements, all inherited from the hub rules in AGENTS.md:

* **`start_new_session=True`** so the shell leads its own process group — the
  precondition for the tree kill;
* **teardown kills the process *tree*, not the wrapper** (AGENTS.md: "Hub
  teardown kills process *trees*, not wrappers"). Escalation identical to
  `LspManager._kill_process_tree` (`lsphub/manager.py:688`) and the vendored
  `mcp.os.posix.utilities.terminate_posix_process_tree` the MCP client uses:
  i.e. SIGTERM the group → reap the leader (its pid *is* the pgid, so no
  `getpgid()` is needed and none would work once the leader is reaped) → probe
  the group until the grace period expires → SIGKILL the stragglers. Extract
  that helper **once** and use it from all three places — the two hubs already
  mirror each other, and a third copy is how they would drift. This is what
  keeps a `vim` or a `python -m http.server` started in the pane from outliving
  the tab;
* teardown runs on **every** exit path: client disconnect, `kill` frame,
  shell exit, reader `OSError`, server shutdown;
* a session registry on `app.state.terminal` (id → session) so shutdown can
  kill sessions that outlive their socket, and so `GET /api/terminal` can
  report what is running (pid, cwd, started, bytes) for the settings panel —
  the same visibility rule the hubs follow: an orphan must be *nameable*
  (see `todos/opt-orphan-hub-children.md` for the boot-sweep question, whose
  signature list gains these shells if this step is accepted);
* **at most N concurrent sessions** (module constant, default 4); a fifth
  upgrade is refused with a close code and a one-line reason, rather than
  silently piling up shells;
* the app shutdown path already stops the hubs in the `lifespan` `finally`
  (`server.py:80-84`); the terminal registry is stopped in the same block.

## Guards

The pane is a human surface, not an agent surface — but the endpoint still
validates what it accepts, because a WebSocket to `127.0.0.1:8000` is reachable
by anything on the machine (a page in the browser, another local process):

* **Origin check**: accept the upgrade only when `Origin` is the app's own
  origin (or absent, for non-browser clients the user launches deliberately).
  There is no CSRF-style token in this app and no CORS layer at all
  (`grep` for `CORSMiddleware` finds none), so this is the one cheap defence
  against a random web page reaching a shell;
* **one session per connection**: no attach/reuse, no shared terminal;
* **`cwd` is `path_location()`**, never a client-supplied directory. `cd`
  inside the shell is the user's business — that is the point of a terminal —
  but the *server* never takes a path from the client, so there is nothing to
  inject. The tree/editor guards (ed-01, ed-04) govern the API surfaces that
  *do* take paths;
* **shell**: `$SHELL`, falling back to `/bin/sh`; launched non-login (an
  interactive non-login bash reads `~/.bashrc`, which is what a user expects in
  an embedded pane and avoids paying `~/.profile` side effects per tab);
  `TERM=xterm-256color`, `COLUMNS`/`LINES` set from the first resize, `PWD`
  handled by the kernel PTY;
* **`env` is inherited verbatim** — which is the honest trade-off: the child
  sees `BENCHMARKIT_PATH_LOCATION`, and `.env`-derived values, because the app
  itself has them in its environment. Documented in the file's docstring rather
  than papered over;
* non-Linux: return an immediate `error` frame + close (501-equivalent close
  code). `pty` is POSIX; the app is documented as self-hosted on Linux;
* the endpoint is *never* exposed as an MCP/LSP tool. It is a route, used by the
  SPA, and that boundary is what keeps read-only-by-design intact.

## Tests — `tests/test_terminal.py`

`TestClient` supports WebSocket (`client.websocket_connect("/api/terminal")`),
and `conftest.py` already keeps the hubs from spawning, so the suite stays
hermetic:

1. `echo hi` → an `output` frame containing `hi`, then the shell stays open;
2. `exit` → an `exit` frame with the code;
3. resize frame → no error, and `os.get_terminal_size(master)` reflects it;
4. client disconnect mid-session → the shell process is gone (poll for it), and
   a child it started is gone too — the tree-kill assertion, the same shape as
   the hub teardown tests;
5. `kill` frame → the session closes and the pid disappears;
6. the session cap: `N+1` connections, the last refused with the reason;
7. a wrong `Origin` → refused before any process is spawned;
8. `GET /api/terminal` lists the live session and empties after teardown.

If spawning a real shell in CI is judged too heavy, gate the process tests on
`shutil.which(os.environ.get("SHELL", "/bin/sh"))` — the guards and the registry
tests still run unconditionally.

## Manual checklist

* Open the tab: prompt appears with `$PWD` = the workspace; `ls` output is
  coloured.
* `vim` opens and paints; `:q` returns cleanly; `clear`, `less README.md`,
  `top` then `q` all render and quit.
* `Ctrl-C` interrupts; `Ctrl-D` exits; `↑` replays history.
* Drag the terminal splitter: text reflows to the new width, no stale rows
  (`fit` + `SIGWINCH`).
* `python -m http.server 8765` then close the tab: `pgrep -f http.server` shows
  nothing, and `ps` shows no orphaned shell.
* Switch to the Benchmark tab and back: the session survives (`v-show`).
