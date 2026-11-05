# llmh-02 — `llmkit/constants.py` + `llmkit/sse.py`, tested, still unused

**Spec:** Phase 1 (steps 3–4), §4.1, §4.2, D3, D10.
**Depends on:** nothing (independent of llmh-01).
**Leaves the app working:** new package imports nothing and nothing imports it.

## Deliver this

The package skeleton `src/core/llmkit/` (`__init__.py`, empty beyond docstring
and re-exports deferred to llmh-03) plus two modules, exactly in the spec's
`mcphub`/`lsphub-mirroring` shape — constants hold the tuning numbers, the leaf
module holds the mechanism:

1. **`src/core/llmkit/constants.py`** — §4.1 verbatim:
   `DONE_SENTINEL = "[DONE]"`, `DATA_PREFIX = "data:"`,
   `API_KEY_PLACEHOLDER = "none"`, `CONNECT_TIMEOUT = 10.0`,
   `READ_TIMEOUT: Optional[float] = None` (D10: no read deadline on a live
   stream — a stalled prefill surfacing as a "truncated answer" is the exact
   confusion this design removes), `WRITE_TIMEOUT = 30.0`, `POOL_TIMEOUT = 10.0`,
   `ERROR_BODY_LIMIT = 500`, `DEBUG_LOGGER_NAME = "benchmarkit.llm"`.

2. **`src/core/llmkit/sse.py`** — §4.2:
   * `parse_frame(raw: str) -> Optional[Dict[str, Any]]` — one SSE frame to a
     chunk dict; `None` for `data: [DONE]`, comment lines (`: keep-alive` and
     any `:`-prefixed line), blank lines, and frames carrying no JSON object;
   * `iter_chunks(lines: Iterable[str]) -> Iterator[Dict[str, Any]]` — yields
     chunk dicts until `[DONE]`, ignoring everything else.
   Rules from §4.2 that the live server exercises and the tests must pin:
   the space after `data:` is optional; a JSON body that does not parse
   **raises** (a half-read frame means the connection died — skipping it would
   hide a cut stream behind a frame that never carried a finish reason, D3).

3. **`tests/test_llmkit_sse.py`** — step 4's list, one test each: `data:` with
   and without the space; `: keep-alive` and other comment lines skipped;
   `[DONE]` ends iteration and yields no chunk; a usage-only frame with
   `"choices": []` parses into a dict; unparseable JSON raises. Add a case for
   the real final-frame shape (finish_reason `"stop"` **and** a `usage` block
   plus an unknown `timings` key — extra keys pass through untouched).

## Files it touches

| file | change |
|---|---|
| `src/core/llmkit/__init__.py` | **new** — package marker (module docstring only) |
| `src/core/llmkit/constants.py` | **new** — §4.1 |
| `src/core/llmkit/sse.py` | **new** — `parse_frame`, `iter_chunks` |
| `tests/test_llmkit_sse.py` | **new** — §6.1 unit file |

## Rules it must not break

* **Core owns logic, the API owns transport** (AGENTS.md): `sse.py` imports
  stdlib + `constants` only. No FastAPI, no `src.api`, no `emit`, no httpx —
  `sse.py` never sees a socket, only an iterable of decoded lines.
* `[DONE]` ends iteration **without becoming a chunk and without becoming a
  finish-reason signal** (D3). `sse.py` records nothing; recording is
  `ChatStream`'s job (llmh-03).
* The exception raised on unparseable JSON is a plain, named exception defined
  in `sse.py` (say `SSEFrameError`) — `client.py` will let it travel; both loops
  surface `str(exc)` in the UI error line (D8), so its message says what broke.
* **Tests stay hermetic** (AGENTS.md): pure line-iterables; no network.

## Acceptance

```
pytest tests/test_llmkit_sse.py -q
```
green, and the rest of the suite untouched:

```
pytest tests/test_llm.py tests/test_chat.py -q    # unchanged, still green
```

## Optional live sanity check (cheap, only if the server is up)

If `http://localhost:8080/v1` answers (model `qwen3.8-flash-next-iq3_xxs`), you
may pipe one `curl -sN` streaming request through `iter_chunks` in a scratch
REPL and confirm real frames — including `: keep-alive` lines — parse. Record
the outcome (or "server not reachable") in a line at the bottom of this file's
copy in `done/`. **Never** probe `http://192.168.1.4:8080/v1`. Do not commit any
scratch file.

---

**Live sanity check (done):** `http://localhost:8080/v1` was up and
`qwen3.8-flash-next-iq3_xxs` was loaded, so one `curl -sN` streaming request was
piped through `iter_chunks` in a scratch run (scratch response file deleted, not
committed). 42 wire lines → 18 chunk dicts: the 2 `: keep-alive` comment lines and
all 21 blank separators were skipped, `data: [DONE]` was present and yielded no
chunk, and the final frame parsed with `finish_reason: "length"` **and** its
`usage` block (`prompt_tokens`/`completion_tokens`/`total_tokens`/
`prompt_tokens_details`) plus the unknown llama.cpp `timings` key passed through
untouched — the shapes `tests/test_llmkit_sse.py` pins.
