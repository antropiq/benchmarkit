# llmh-03 — `llmkit/client.py`: the two call shapes, the error, the URL rules

**Spec:** Phase 1 (steps 3, 5), §4.3, D3, D8, D9, D10, D11, §2.1, §2.2.
**Depends on:** llmh-01 (the `EXPECTED_STREAM_BODY` fixture, `FakeStream`),
llmh-02 (`constants`, `sse`).
**Leaves the app working:** `client.py` is still imported by nothing outside
`tests/`.

## Deliver this

**`src/core/llmkit/client.py`**, §4.3's shape exactly — signatures, defaults and
the keyword-only style included — plus **`tests/test_llmkit_client.py`**, driven
end-to-end by `httpx.MockTransport` (no network anywhere).

The pieces:

* `LLMRequestError(Exception)` — carries `status`, `message`, `url`. Its `str()`
  names **both** the status and the upstream message (D8): both loops do
  `except Exception as exc` and emit `str(exc)` into the UI's error line, so the
  message is the surface that matters, and it must be complete with the debug
  flag **off**. A non-JSON error body is included raw, truncated to
  `ERROR_BODY_LIMIT`, never raising a decode error.
* `chat_completions_url(server_url)` / `models_url(server_url)` — one rule for
  both (D9): `SERVER_URL` already ends in `/v1` today (it is the old litellm
  `api_base`); a bare host gets `/v1` added. `routes._models_url` has the second
  rule already; import its behaviour into `llmkit`, don't reinvent it (the
  rewiring of `routes.py` itself is llmh-08 — until then the duplicate exists;
  do not touch `routes.py` in this task).
* `ChatStream` — wraps one open httpx streaming response; `__iter__` yields the
  chunk dicts from `sse.iter_chunks` and records as it yields: `finish_reason`
  (first non-null seen), `saw_finish_reason` (bool — chunk-level **only**; a
  `[DONE]` that arrives with no finish-reason chunk still leaves it `False`,
  D3), `usage` (the last usage dict seen, `Optional`). `close()` closes response
  and client, idempotent, never raises (the cancel path calls it blind).
  Iteration exhausted or raised-through also leaves the client closed (context-
  manage internally; a stream that ends must not leak the connection).
* `stream_completion(*, model, server_url, messages, api_key=API_KEY_PLACEHOLDER,
  sampling=None, tools=None) -> ChatStream` — POST with `"stream": true` and
  `"stream_options": {"include_usage": true}`; `sampling` merged into the body
  **verbatim, no whitelist** (§2.1 — the invariant); `tools` added **only when
  non-empty** — the `tests/test_chat.py` rule "no `tools` key at all rather than
  `tools: []`" travels down here with it. `Authorization: Bearer <key>` sent
  even when the key is the `"none"` placeholder (§2.1: that is what happens
  today and the server accepts it). `extra` is **dropped** (big-picture settled
  item 4: no caller). One `httpx.Client` per call with the D10 `httpx.Timeout`
  built from `constants` (D11) — no shared client.
* `chat_completion(*, model, server_url, messages, api_key=API_KEY_PLACEHOLDER,
  max_tokens=None, sampling=None) -> Dict[str, Any]` — POST without `stream`,
  returns the parsed body; raises `LLMRequestError` on any non-2xx.
* Debug logging (the payload half of D8): when `LLM_DEBUG` semantics are on —
  read through a **module-level hook**, e.g. `client.debug_enabled: Callable[[],
  bool]` defaulting to `lambda: False`, which llmh-07 re-points at `config` —
  log at DEBUG to `constants.DEBUG_LOGGER_NAME` the request line (POST url,
  model) and the JSON payload. Never log the `Authorization` header or the key.

`tests/test_llmkit_client.py` covers step 5's list as separate tests:
the streaming body equals llmh-01's `EXPECTED_STREAM_BODY` (substituting
model/max_tokens); `Authorization: Bearer none` present with no key configured;
the model name reaches the body bare; an unknown sampler key in `sampling`
appears verbatim; `tools` absent when `None`/empty and present when non-empty;
a `finish_reason` on a late chunk flips `saw_finish_reason` to `True`, and a
stream that ends with `[DONE]` but never carried one leaves it `False`; HTTP
404 with `{"error":{"message":"no router for requested model",…}}` raises
`LLMRequestError` whose `str()` holds the status **and** that message; a
non-JSON error body is truncated at `ERROR_BODY_LIMIT` without raising;
`close()` twice is a no-op and a failing underlying close never propagates;
`chat_completions_url`/`models_url` agree for a `…/v1` base **and** a bare host
(assert the D9 rule once, for both shapes).

## Files it touches

| file | change |
|---|---|
| `src/core/llmkit/client.py` | **new** — §4.3 |
| `src/core/llmkit/__init__.py` | export the public names (`stream_completion`, `chat_completion`, `ChatStream`, `LLMRequestError`, the two URL helpers) |
| `tests/test_llmkit_client.py` | **new** — step 5's list via `httpx.MockTransport` |

## Rules it must not break

* **Sampling dict passes through unexamined** (§2.1). No whitelist, no rename,
  no drop. This is the lot's first invariant.
* The bare model name is the second: nothing here may prepend a provider.
* **Core owns logic, the API owns transport** (AGENTS.md): imports stay
  `httpx` + stdlib + `llmkit` siblings. No `src.api`, no FastAPI, no `emit`.
* `httpx` reaches **only** model endpoints: URL building accepts `server_url`
  strings and produces `/chat/completions` or `/models` paths. Nothing here may
  name or reach an app route (`tests/test_write_surface_isolation.py`'s
  write-marker scan walks `src/core`; llmh-04 formalises the guard).
* No retries, no shared client, no async (§1 out of scope; D11).
* **Tests stay hermetic** (AGENTS.md): `httpx.MockTransport` only. A test that
  wants a different timeout or body limit patches `llmkit.constants`, never a
  literal inside `client.py`.

## Acceptance

```
pytest tests/test_llmkit_client.py tests/test_llmkit_sse.py tests/test_llmkit_contract.py -q
```
green; `pytest tests/test_llm.py tests/test_chat.py -q` unchanged and green.

## Optional live probe (record the outcome either way)

One streaming call against `http://localhost:8080/v1` with model
`qwen3.8-flash-next-iq3_xxs` through `stream_completion` itself, confirming:
chunks arrive, `saw_finish_reason` is `True`, `usage` is not `None`, and a
mid-stream `close()` returns promptly. **Never** `http://192.168.1.4:8080/v1`.
If the server is down, say so in the done-file and move on — the MockTransport
suite is the actual gate.

## Done — outcome (2026-11-04)

Delivered `src/core/llmkit/client.py` (§4.3 shape: `LLMRequestError`,
`chat_completions_url`/`models_url` over one `_v1_base` rule, `ChatStream`,
`stream_completion`, `chat_completion`, the `debug_enabled` module hook
defaulting off), the package exports in `__init__.py`, and
`tests/test_llmkit_client.py` (40 tests via a `httpx.MockTransport` seam —
the client itself is built for real, only the transport is a handler).
`extra` dropped per settled item 4; `routes.py` untouched.

Acceptance run green:
`pytest tests/test_llmkit_client.py tests/test_llmkit_sse.py tests/test_llmkit_contract.py -q`
→ 86 passed. `pytest tests/test_llm.py tests/test_chat.py -q` unchanged and
green (139 passed); `tests/test_write_surface_isolation.py` green (20 passed);
pyright clean on `client.py`. No `/api/` string, no `src.api`/FastAPI import
under `llmkit`.

Live probe (localhost:8080/v1 only, model `qwen3.8-flash-next-iq3_xxs`),
through `stream_completion` itself:
- full drain: 36 chunks, `saw_finish_reason=True`, `finish_reason='stop'`,
  `usage={'prompt_tokens': 16, 'completion_tokens': 37, 'total_tokens': 53, ...}`,
  stream closed on exhaustion, 1.0 s total;
- mid-stream `close()` after two chunks returned in 0.29 s, `closed=True`.
`http://192.168.1.4:8080/v1` was not contacted.
