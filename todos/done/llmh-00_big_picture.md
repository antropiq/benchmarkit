# llmh — the in-house LLM client: litellm out, httpx in

Replace `litellm.completion()` with a small in-house transport package so that
the one thing the app actually needs from a model endpoint — POST
`{SERVER_URL}/chat/completions` and read the SSE stream — is spoken by
`src/core/llmkit/` and by nothing else.

The authority for this lot is **`docs/in-house-llm-client-spec.md`**. Read it
before your task. Every task file below names the decisions it implements
(**D1 … D11**), the spec section it comes from (**§2.1**, **§4.3**, …) and the
step number it carries (**step 3**, …). This file is the map and the cross-task
contract; it does not restate that spec.

## Why this is worth doing (the invariant, not the opinion)

Three of the app's subtlest behaviours are archaeology over a library's private
attributes: `saw_finish_reason()` reads `CustomStreamWrapper.received_finish_reason`,
`close_llm_stream()` digs out `.completion_stream.close()`, and a `register_model`
cost table exists only to silence a traceback. Spec §1 states each. After this lot
each is a plain property or method on our own stream object, so the truncation
check is honest **by construction** — which is what `AGENTS.md`'s "a stream cut by
the server looks like a finished answer" actually requires.

## The two invariants that outlive the migration

1. **Sampling params reach the server verbatim** (§2.1). `top_k`, `min_p` and
   `repeat_penalty` are llama.cpp sampler names, not OpenAI ones. A client that
   whitelisted OpenAI's own list would silently change the model's sampling — and
   every score of every run — with nothing on screen to say so. `llmh`'s task
   pins this with a body fixture.
2. **The model name travels bare** (§2.1). No `openai/` prefix in the request
   body. A leftover prefix reaches llama-swap as `openai/qwen3.8-…` and comes
   back `404 {"error":{"message":"no router for requested model"}}`.

## The seam (the cross-task contract)

`src/core/llmkit/` — `constants.py`, `sse.py`, `client.py` (§4). Like `mcphub` /
`lsphub` it is a mirror of the same convention: `constants` holds the tuning
numbers, the leaf modules hold the mechanism. It imports no FastAPI, emits no
events, and never reads `.env`: it takes `model` / `server_url` / `api_key` /
`sampling` as arguments, exactly as the current `completion(**kwargs)` call does.

| name | lands in | shape |
|---|---|---|
| `constants.py` | llmh-02 | §4.1 verbatim: `DONE_SENTINEL`, `DATA_PREFIX`, `API_KEY_PLACEHOLDER`, `CONNECT_TIMEOUT`, `READ_TIMEOUT = None` (D10), `WRITE_TIMEOUT`, `POOL_TIMEOUT`, `ERROR_BODY_LIMIT`, `DEBUG_LOGGER_NAME = "benchmarkit.llm"` |
| `sse.parse_frame` / `sse.iter_chunks` | llmh-02 | frame → chunk **dict**, or `None` for `[DONE]`, comments (`: keep-alive`) and blanks; unparseable JSON **raises** (§4.2) |
| `client.LLMRequestError` | llmh-03 | carries `status`, `message`, `url`; its `str()` names status + upstream message **with debug off** (D8) |
| `client.chat_completions_url` / `client.models_url` | llmh-03 | the one URL rule for both shapes of `SERVER_URL` (D9); `models_url` replaces `routes._models_url` |
| `client.ChatStream` | llmh-03 | `__iter__` of chunk dicts, plus `finish_reason`, `saw_finish_reason`, `usage`, and idempotent never-raising `close()` |
| `client.stream_completion` / `client.chat_completion` | llmh-03 | the two call shapes; the new seam's two names (D5) |
| `tests/llmkit_fakes.FakeStream` | llmh-01 | a stub that yields dicts and carries `saw_finish_reason` + `close()` — the **shared** double every later task feeds its seam stubs, so the stub and the real client are held to one interface from the start |

`ChatStream.close()` is the socket's own teardown. It is **not** the process-tree
teardown `core/proctree.py` owns; the two are unrelated (§4.3).

## Task order and phase boundaries

| # | task | spec step(s) | depends on |
|---|---|---|---|
| llmh-01 | Pin the contract: `tests/llmkit_fakes.FakeStream` + the §2.1 request-body fixture | Phase 0, 1–2 | — |
| llmh-02 | `llmkit/constants.py` + `llmkit/sse.py` and their tests | 3 (part), 4 | — |
| llmh-03 | `llmkit/client.py`: the two call shapes, the error type, the URL rules, `ChatStream` | 3 (part), 5 | llmh-01, llmh-02 |
| llmh-04 | `llmkit` write-surface guard in `tests/test_write_surface_isolation.py` | 6 | llmh-03 |
| llmh-05 | Move the **benchmark** across: `src/core/llm.py` | 7–11 | llmh-01 … llmh-04 |
| llmh-06 | Move the **chat** across: `src/core/chat.py` (+ `core/models.py` docstring) | 12–16 | llmh-05 |
| llmh-07 | The debug switch: `LLM_DEBUG` + fallback, the logger half, cost table deleted | Phase 4, 17–18, 20–22 | llmh-03, llmh-05, llmh-06 |
| llmh-08 | The switch at the surface: `server.py`, `routes.py`, `SettingsDialog.vue`, env files, vitest fixtures | 19, 20 | llmh-07 |
| llmh-09 | Drop `litellm` from `pyproject.toml`, purge the mentions | 23–25 | llmh-05 … llmh-08 |
| llmh-10 | Verification pass: §6.1, §6.2 static scans, the §6.3 manual checklist written down | Phase 6 | llmh-09 |
| llmh-11 | Docs last: `AGENTS.md`, `README.md`, the spec's own status | 26 | llmh-09 |

**Phases are load-bearing.** 01–04 add code nothing imports; 05 and 06 are each
**one file** on a live path and each is a revert point (§6.4); 09 is the first
task that can break anything by deleting a dependency, so it lands after both
paths are switched. **11 is last**: `AGENTS.md` is injected into every WorkBench
turn, so a wrong sentence there is worse than a missing one.

01 and 02 are independent of each other. 05 and 06 must not run concurrently:
both re-point overlapping stub machinery in `tests/`.

## Endpoint rule (binding on every task)

* Live smoke, when a task calls for one: **`http://localhost:8080/v1`**, model
  **`qwen3.8-flash-next-iq3_xxs`**, only while that server is up. Probe once,
  minimally, and record what you saw in the task file.
* **Never** send a request to **`http://192.168.1.4:8080/v1`** — a developer
  endpoint. It is not a test target, not a fallback and not a probe.
* The committed `.env.test` currently points at a `192.168.1.x` host. A task that
  runs a **real benchmark** for §6.3.1 comparison points the app at localhost
  through the Settings dialog (which writes `.env`) and says so, rather than
  editing `.env.test`. A **test run** takes its endpoint from `.env.test`, never
  from a developer's `.env` — that rule stays untouched.
* Every unit test of `llmkit` goes through `httpx.MockTransport`. The suite stays
  hermetic: no test in this lot touches a network, live or not.

## Project rules that bind these tasks (quoted from `AGENTS.md`)

* **"One LLM layer. Every model call — benchmark answer, verifier call, chat —
  goes through the same client"** against an OpenAI-compatible endpoint. After
  this lot that client is `core/llmkit`, and the sentence stays true without
  naming a library.
* **"Core owns logic, the API owns transport."** Blocking work runs in a worker
  thread publishing to the event bus, and **"nothing under `core` may import the
  API layer."** `llmkit` is core: no FastAPI, no `src.api`, no `emit`.
* **Tools are read-only by design**: *"no connector, tool or chat path reaches a
  write route, and the terminal is never a tool"* —
  `tests/test_write_surface_isolation.py` asserts it. `httpx` under `core/` is
  legal (its import scan covers `mcp_connectors/` only), but its write-marker
  scan **does** walk `src/core`: `llmkit` must never contain a `PUT /api/fs`,
  `POST /api/fs` or `DELETE /api/fs` string, and its URL builders must never be
  able to produce a path under `/api/`. llmh-04 makes that asserted, not promised.
* **Timing is app-side**, anchored to content-chunk arrivals: *"warm-up and
  verifier time never enter the numbers"*, and **token/s uses completion tokens
  only** while the displayed cost is prompt + completion. This lot moves the
  transport under the timers and must not move the timers.
* **A truncated answer is failed, never graded**: *"A truncated benchmark answer
  is failed and never sent to the verifier … and a truncated chat reply ends as an
  error, which also stops a half-streamed tool call from being executed."* D3 is
  how that stays true after litellm's fabricated `"stop"` is gone.
* **`.env` is regenerated from a fixed template on every save**: *"A new or
  renamed key means changing the template, the renderer and the committed example
  together."* llmh-07 renames `LITELLM_DEBUG` → `LLM_DEBUG`, so llmh-07 touches
  `SETTING_KEYS`, `_render_env`, `.env.example`, `.env.test.example` and `.env.test`
  in the same task.
* **Hubs are long-lived singletons** — *"a restart rebuilds a hub's internals but
  never replaces the object the app holds."* Deliberately **not** copied here
  (D11): one `httpx.Client` per call, closed by `ChatStream`. Nothing in `llmkit`
  holds a reference across requests.
* **Tests stay hermetic**: *"a test patches a seam's own constants rather than a
  path it computed."* A test that needs a timeout or the body limit patches
  `llmkit.constants`, never a literal in the module under test.
* **No lint step.** `pytest` for Python, `cd web && npm test` for Vitest. Use the
  project venv (README "Tests" for this machine's invocation).
* Read a file with the read tool before editing it, dry-run the edit, and keep the
  text after the line number verbatim — indentation included.

## Cadence: run the tests you touched, not the world

> **DO NOT RUN ALL TEST SUITE. RUN ONLY TESTS TOUCHING THE AREA OF THE FEATURE.**
> Full-suite `pytest` runs **once**, in llmh-10.

Per-task command, from the spec §6.1:

```
pytest tests/test_llmkit_contract.py tests/test_llmkit_sse.py \
       tests/test_llmkit_client.py tests/test_write_surface_isolation.py \
       tests/test_llm.py tests/test_chat.py tests/test_bench.py \
       tests/test_agent_modes.py tests/test_config.py tests/test_api.py
```

Pick from that list the files your task touched. `cd web && npm test` runs only in
llmh-08, the one task with a frontend change.

## Out of scope for this lot (do not add it)

* Any provider other than OpenAI-compatible; `async`; **retries** (a retried
  stream would double-count tokens and timing); a real tokenizer; monetary cost
  tracking; a `/completions` path; embeddings; chunk dataclasses or pydantic
  response models — all of §1 "Out of scope" and §8.
* **The app's own SSE layer** (`api/`'s event bus, per-channel replay, terminal
  events), the two hubs, and the Editor's file surface and terminal. `llmkit`
  speaks to a model endpoint and to nothing else.
* `saw_finish_reason()` / `close_llm_stream()` **call sites** in `core/chat.py`
  and their cancel > error > pause precedence (D4). Names stay; bodies become
  one-liners over `ChatStream`.
* The `stream_chat` terminal-event contract: exactly one of `done` / `error` /
  `pause_ack`, a paused reply ending on its own ack. This lot changes what a
  chunk *is*, not what a pause *is* (step 15).
* The reserved workspace folder, the path policy, agent-mode personas, batch mode.
* `todos/done/**`, `.kilo/plans/**` and `logs/_baseline_check/**` — historical
  records; their litellm mentions stay exactly where they are.
* Widening `tests/test_write_surface_isolation.py`'s `http_import_offenders` scan
  to `src/core`. That scan guards the connectors; making it also condemn the
  app's own model client is a different decision. llmh-04 adds the narrow guard.

## Settled here (the spec's §7 open questions, decided by the operator's brief)

1. **D6 rename goes ahead**: `LLM_DEBUG` with the `LITELLM_DEBUG` fallback, and
   the settings field `litellmDebug` → `llmDebug`. The dialog checkbox naming a
   library that is no longer installed is a lie on screen.
2. **D2**: the attribute branches in the chunk extractors are **deleted**, and
   with them the `MagicMock` / object-shaped chunk fixtures in `tests/test_llm.py`
   (`TestSafeGetChoiceContent::test_object_with_choices_and_delta`,
   `TestExtractTotalTokens::test_object_chunk_with_usage`,
   `TestExtractUsage::test_object_chunk_with_usage` become dict cases).
3. **D3 stands as designed**: `[DONE]` is transport framing and never counts as a
   finish reason. No measurement required for the chunk-level rule to be correct.
4. **`extra` is dropped from `stream_completion`** — smaller than the spec's
   maybe, and nothing has a caller. The spec allows this (§8: names and shapes
   may move; the invariants may not).

## Definition of done for the lot

1. `grep -rn "litellm" src tests web/src pyproject.toml README.md` returns only
   the `LITELLM_DEBUG` fallback in `src/config.py` and the test that pins it, plus
   `"litellm"` in the two forbidden-import tuples of `tests/test_agent_modes.py`
   — which **keep** it and **gain** `"llmkit"` (step 25).
2. `grep -rn "completion_stream\|received_finish_reason\|intermittent_finish_reason" src`
   is empty: the workarounds are gone, not renamed.
3. `grep -rn "httpx" src/core` shows the model-facing HTTP only under
   `core/llmkit/`; `mcphub`, `lsphub`, `editorfs` unchanged.
4. `litellm` is gone from `pyproject.toml` and the full suite passes without it.
5. The §6.3 manual checklist is written down as a human-only step (with its
   localhost-only endpoint rule) and **not** attempted by any task in this lot,
   except where a task's own file says a live probe is cheap and available.
6. `AGENTS.md` *Key patterns* names the one seam that speaks HTTP to a model
   endpoint and the verbatim-sampling invariant — one or two lines, the file's own
   style, naming the invariant and not the implementation.
