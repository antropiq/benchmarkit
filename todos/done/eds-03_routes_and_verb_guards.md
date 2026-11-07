# eds-03 — The two routes, and the rewritten agent-modes verb guards

**Spec:** D1 (including the paragraph that says the existing guard is
**rewritten, not left alone**), §4 "Files touched" rows 3, 8 and 9.
**Depends on:** eds-01 (the document read), eds-02 (the writer).
**Files it touches:** `src/api/routes.py`, `tests/test_api.py`,
`tests/test_write_surface_isolation.py`, `tests/test_editorfs.py`.

## What to deliver

Two routes on their own path pair, both browser-only:

| route | does |
|---|---|
| `GET /api/agent-modes/document` | the editor's read (eds-01's payload): the file's own entries, persona verbatim, findings, `mtime`, caps, defaults. No `?path=`, no raw bytes. |
| `POST /api/agent-modes/save` | the whole-document write (eds-02). Body carries the document plus the `mtime` token. Gated by `_refuse_if_chat_streaming` — the one place that rule is written, used by every Editor write. Maps the core's errors: **409** for the `mtime` conflict (detail is the object carrying both numbers, as the Editor save does) and for the streaming refusal, **422** for an invalid document naming the pointer. Answers the new `mtime` and the new verdict so the client reloads without a second round-trip (D9). |

`GET /api/agent-modes` is **unchanged** — same payload, same docstring promise
that it publishes no persona and no generated blocks.

The save is **not** gated during a benchmark run. A run never consults a mode
(`test_the_benchmark_never_consults_a_mode`), so gating it would be a rule without
a race behind it. Do not copy the run-check the settings routes use.

## Rewrite the guard in the same commit

`TestTheAgentModesReaderStaysReadOnly` walks the route registrations with
`path.startswith("/agent-modes")` and asserts the verbs are `["get"]`. Both new
paths fall inside that prefix, so its assertion **fails the day the save route is
added**. Add the routes and rewrite the guard in the same task — a task that adds
the pair and leaves the guard alone breaks the suite.

What replaces it keeps the same teeth and states the narrower truth:

* the **reader** path (`/agent-modes`) has one verb and no argument — keep the
  existing arity assertion;
* the **editor pair** exists, is pinned to `get` (`/agent-modes/document`) and
  `post` (`/agent-modes/save`), and is reachable from the browser only;
* the poisoned case in `TestTheGuardsActuallyFail` is updated to poison the **new
  pair** (a `post` on the document path, a `get` on the save path), so the
  rewritten guard is still a guard that can fail.

Update that class's docstring: it currently reasons that the modes file *"is
edited by hand outside the app (D6), so a write route for it would not just break
the reservation"*. That premise is gone. The surviving reason is that a **tool**
must not reach the switch that decides whether tools exist — say that.

`tests/test_editorfs.py`'s `TestSurfaceShape` verb map is scoped to `/api/fs` and
will **not** cover these paths. Add the pair's own pinned verb set beside it (or
in the isolation file, your call — but it must be pinned, not assumed into the
`/api/fs` allowlist).

## Rules this task must not break

* AGENTS.md: *"Core owns logic, the API owns transport"* — the routes call eds-01
  and eds-02 and decide nothing about the file themselves.
* The reserved folder stays unreachable through `/api/fs/*`: `tests/test_agent_modes_seed.py::TestTheSeededFileIsUnreachableFromTheEditor`
  must stay green — the modes file is still not readable, not listable and not
  writable through the Editor surface. The new pair is a purpose-built route for
  one file, the pattern `GET /api/path-location` + `policy_file_state` already
  establishes.
* No new write marker may appear outside `src/api/routes.py` and the Editor
  composables — the marker scan in `test_write_surface_isolation.py` decides what
  counts as a marker; if you add the pair to that vocabulary, do it in eds-04,
  which owns the reach guards, and keep this task's scan green.
* `GET /api/agent-modes` keeps zero arguments (its arity is pinned in
  `tests/test_api.py` through the OpenAPI operation).
* The save route takes a body model with the same validation discipline as the
  other write bodies; a malformed body is a 422, never a 500.

## Tests to add (`tests/test_api.py`)

* `GET /api/agent-modes/document` on a scratch root (reuse the `mode_root`
  fixture): the file's entries in the file's order, persona present verbatim,
  `mtime` present, caps present.
* A hand-broken file: 200, the broken entry **listed** with its finding — the
  contrast with the reader, which drops it.
* The document payload carries no app-owned entry.
* `POST /api/agent-modes/save` writes the file, answers the new `mtime` and the
  new verdict, and the previous bytes are in the app's backup folder.
* A save against a file that changed since the read → 409 carrying both numbers,
  and the disk is untouched.
* A save declaring a reserved id → 422, disk untouched.
* A save while a chat reply streams → 409 (follow
  `tests/test_editorfs_write.py::TestChatStreamGate`'s shape).
* A save while a benchmark run is in progress → **not** refused (pin the absence
  of that gate, so nobody adds it later by analogy).
* The file stays unreachable through `/api/fs/file` and `/api/fs/tree`.

## How to tell it is done

`pytest tests/test_api.py tests/test_write_surface_isolation.py tests/test_editorfs.py tests/test_agent_modes_write.py -q`
green, and `pytest -q` green overall. Move this file to `todos/done/`.
