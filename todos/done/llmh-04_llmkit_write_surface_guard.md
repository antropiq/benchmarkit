# llmh-04 — Guard `llmkit` the way the project guards its other boundaries

**Spec:** Phase 1 (step 6), §1 ("Tools are read-only by design" in AGENTS.md).
**Depends on:** llmh-03 (there must be a `client.py` to guard).
**Leaves the app working:** test-only change; no production file is touched.

## Deliver this

A new test class in **`tests/test_write_surface_isolation.py`** that makes the
`llmkit` boundary *asserted, not promised*. Two facts frame it (spec step 6):

* that file's `http_import_offenders` scan covers `mcp_connectors/` only, so
  `httpx` under `src/core/llmkit/` is **legal** — do not "fix" that;
* but its write-marker scan (`scan_roots()` walking `src/core`) **does** apply:
  `llmkit`'s sources must never contain a `PUT /api/fs`, `POST /api/fs` or
  `DELETE /api/fs` string — the existing
  `test_the_write_markers_appear_only_in_the_routes_and_the_callers` already
  enforces this passively; keep it green.

The new class adds what the passive scan cannot answer — the **URL shape**:

1. **Suffix lock**: for every input shape the app uses (`http://h:p/v1`,
   `http://h:p/v1/`, bare `http://h:p`, `http://h:p/`), `chat_completions_url`
   returns a URL whose path ends exactly with `/v1/chat/completions` and
   `models_url` one ending exactly with `/v1/models`. The client appends **only
   these two fixed suffixes**, so no configured `SERVER_URL` can turn it into a
   caller of an app route.
2. **No `/api/` reach**: assert the returned path never starts with (or contains)
   `/api/` for those inputs — the invariant spec step 6 states, i.e. the new
   HTTP client cannot reach the app's own write routes. If a hostile
   `SERVER_URL` itself embeds `/api/fs`, the suffix lock still means the request
   is a chat-completions POST, not a `PUT /api/fs` write — say this in the test
   docstring rather than trying to police `SERVER_URL`, which is user config.
3. **Marker self-check, in the file's own style** (`TestSelfChecks` shows the
   pattern): a poisoned scratch module carrying a write marker under a tmp_path
   is caught by `offenders_for` — reuse the shared helpers, never a second copy
   of the walk ("a second copy of the walk is how a guard drifts").

Where to import the URL helpers from: `src.core.llmkit.client`. Importing the
module in a *test* is fine; the wall this file guards is runtime, not import.

## Files it touches

| file | change |
|---|---|
| `tests/test_write_surface_isolation.py` | **new test class only** — do not restructure existing classes, do not widen `http_import_offenders` to `src/core` (big-picture out-of-scope), do not touch `ALLOWED_HOMES` |

## Rules it must not break

* The guard tests must pass **unmodified** through llmh-09: they assert
  invariants, not implementation details (no line counts, no signature text).
* AGENTS.md, quoted: *"no connector, tool or chat path reaches a write route,
  and the terminal is never a tool"*. This task adds the model-client's share of
  that sentence; it changes no runtime behaviour.
* **Tests stay hermetic** (AGENTS.md): no network — the URL helpers are pure
  string functions; call them directly.

## Acceptance

```
pytest tests/test_write_surface_isolation.py -q
```
fully green, including every pre-existing test in the file — and a quick
sanity that a deliberately broken suffix would be caught (edit a helper in a
scratch copy or via monkeypatch in the self-check, not by editing `client.py`
and forgetting).

## Notes for the reader

Read the file's module docstring and `TestSelfChecks` first; the file has a
deliberate shape (one shared walker, markers + import scan + self-checks) and
the new class should slot into it, not bolt onto it. The current write-marker
constants and helper names are in that file — cite them from the code, not from
this task file, since they may have been renamed between task 3 and now (they
should not be; this file is only listed so you can find the shape fast:
`WRITE_MARKERS`, `ALLOWED_HOMES`, `scan_roots()`, `offenders_for()`).

## What landed

`tests/test_write_surface_isolation.py` only — no production file touched, and
`http_import_offenders` was not widened to `src/core`, nor `ALLOWED_HOMES`
changed. The module docstring gained its fifth bullet ("the model client cannot
spell an app route either"), and the file gained:

* `APP_ROUTE_PREFIX = "/api/"` beside the other scan constants, and
  `builder_paths(builder, bases)` beside the other shared walkers — the URL
  half of `offenders_for`, so the two guards and their self-check read one
  implementation.
* `TestTheModelClientCannotReachAWriteRoute`, slotted in after the other
  surface guards and before `TestTheGuardsActuallyFail`: the suffix lock
  (`chat_completions_url` → `/v1/chat/completions`, `models_url` → `/v1/models`,
  over all four `SERVER_URL` shapes), the no-`/api/` lock (docstring says why a
  hostile `SERVER_URL` is user config and not policed here), a self-check
  feeding the *same* walker a builder drifted onto `/api/fs/file`, and a
  marker-self-check on a poisoned scratch `llmkit/cheeky.py` through
  `offenders_for`.

No network: the builders are pure string functions, called directly.

Acceptance run green:
`pytest tests/test_write_surface_isolation.py -q` → 24 passed (20 pre-existing,
4 new). `pytest tests/test_write_surface_isolation.py tests/test_llm.py
tests/test_chat.py tests/test_agent_modes.py -q` → 295 passed;
`tests/test_llmkit_contract.py tests/test_llmkit_sse.py
tests/test_llmkit_client.py` → 86 passed. pyright clean on the test file.
Sanity (outside the suite, by handing the guards' own table a builder returning
`http://host:8080/api/fs/file`): both the suffix lock and the no-`/api/` lock
raised. No live probe — nothing in this task talks to an endpoint.
