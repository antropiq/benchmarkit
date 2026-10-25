# Sweep orphaned hub children at boot *(optional)*

Follow-up to the process-tree teardown in `lsphub/manager.py`
(`_kill_process_tree`). That teardown kills the whole tree **when the hub gets to
run it**. This note covers the case where it does not: a child that outlives an
app which never got to clean up after itself.

Not scheduled. Filed so the decision is not lost — the *policy* has to be chosen
before any code is written (see "Why this is not free").

## The hole that is left

The `lifespan` in `api/server.py` calls `lsp.stop_all()` then `manager.stop_all()`,
and both are now tree-killers. That `finally` only runs on a polite shutdown. Three
paths skip it:

| Path | Why the teardown is skipped |
|---|---|
| **Ctrl+C twice** | uvicorn sets `force_exit` on the second SIGINT and `Server.shutdown()` guards the call (`if not self.force_exit: await self.lifespan.shutdown()`), so neither hub is ever told to stop. |
| `kill -9`, crash, OOM | No handler of any kind. Both hubs run on `daemon=True` threads, so their children are simply orphaned when the interpreter dies. |
| **Spawn in flight** | Narrower: a `_start_one` still awaiting `initialize` can insert a session *after* `stop_all` iterated `list(self._sessions)`; the loop is then shut down and nobody terminates it. |

What survives: an `npx`/`node` wrapper with its `chromium` tree (MCP, worst
offender), a `typescript-language-server` with its **`tsserver`** child, a
`vue-language-server` with its workers, `pyright-langserver` — all reparented to
init, each holding an open index of the project.

Nothing catches them today: grepping `src/` for `atexit`/`signal.signal`/
`add_signal_handler` returns nothing. The next app start does not know the
previous one left anything behind, so a ghost is permanent and invisible — absent from
`/api/lsp` and `/api/mcp`, costing RAM, possibly holding a lock or a port.

## Proposed shape

A boot sweep, between the config load and `start_enabled` in the `lifespan` (it
must run **before** the hubs start, so it cannot hit its own children).

1. Build the argv signatures from the files that are already the source of truth:
   the `command` of each entry in `lsp_connectors/lsp_servers.json`, and each
   `local` entry of `mcp_connectors/externals.json`. Reuse their `discovery`
   modules — the `${NAME}` expansion happens there; do not re-read raw JSON.
2. Scan `/proc/*/cmdline` for argv matching one signature **exactly**.
3. Narrow to orphans: `PPid == 1` in `/proc/<pid>/status`, or a `pgid` equal to its
   own pid with no living ancestor in the set. This is what separates "our ghost"
   from "the user's own language server".
4. SIGTERM the group, wait, SIGKILL — the escalation just implemented. Extract it
   into one helper for both hubs (they already mirror each other; MCP's is
   `mcp.os.posix.utilities.terminate_posix_process_tree`, LSP's is
   `LspManager._kill_process_tree`).
5. Log every kill and put the count in the status payload, so a surprise stays
   visible in the settings panel.

Cheaper and independent, worth pairing: an `atexit`/signal hook, so a plain
SIGTERM to the app still cleans up. That covers the middle row; it cannot cover
`kill -9`, which is why the boot sweep is the part that matters.

## Why this is not free — the decision to make

The sweep kills by **argv signature**, and signatures are not unique to this app:

- A developer running `typescript-language-server` in the same checkout (the
  editor's own tsserver, a manual `npx`) matches one. A naive sweep kills their
  editor's server.
- `--root-dir` / `workspaceRoot` appear in the argv of these servers, so two
  checkouts are told apart only when the declaration carries an absolute path. A
  generic signature widens the blast radius.
- `PPid == 1` filters most of that, but not a server the user started from a
  terminal that has since closed.

Pick one before writing anything:

| Option | Behaviour | Cost |
|---|---|---|
| **A. Conservative** | act only on an argv-matched, `PPid == 1` process, and skip anything older than the app's own start time | leaves some ghosts behind |
| **B. Opt-in** | sweep off by default, driven by a managed `.env` key (needs `SETTING_KEYS`, `_render_env()` and `.env.example` edited together) | a knob nobody knows about |
| **C. Report, then ask** | boot *lists* the orphans it found and kills them only after a click in the settings panel | one extra round-trip, no surprise |

**C looks right**: it keeps the no-surprise-kills property, reuses the status
payload that already enumerates hub servers, and makes the sweep observable —
which is also what makes a misfire reportable. B is the fallback if C's UI work
is judged too large.

## Deliberately not done

The `mcphub` side has no equivalent of the LSP `STARTING_STUCK_SECONDS` watchdog.
MCP was never affected by the self-blocking guard (its `_start_one` has none), so
there is no reproducer to test against; it was left alone rather than changed on
symmetry grounds. If it is added, the symmetry rule in AGENTS.md is the reason,
and the test should mirror the LSP one.

## Verify by hand

- Start the app with `pyright` enabled, then `kill -9` its pid. `pgrep -af
  pyright-langserver` must still show the orphan — that is the bug, reproduced.
- With a sweep: start again, and the orphan must be gone, named in `logs/`.
- Guard the other direction: start a `pyright-langserver` by hand from a shell,
  then start the app. It must **survive**, and the sweep must say why it skipped it.

