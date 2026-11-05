# In-house LLM client — replace litellm with httpx

**Status**: **landed** by lot `llmh` (`todos/done/llmh-01_…` …
`todos/done/llmh-11_docs_last.md`, 2026-11-05) — `src/core/llmkit/` is now the
app's only seam that speaks HTTP to a model endpoint, `litellm` is gone from
`pyproject.toml`, and the three litellm-shaped workarounds are deleted rather
than renamed. §7 carries the answers to its four open questions, settled by the
operator's brief as recorded in `todos/llmh-00_big_picture.md`. One step of this
spec is deliberately not a task: the §6.3 manual pass, written down for a human
in [`in-house-llm-client-smoke-checklist.md`](in-house-llm-client-smoke-checklist.md)
with its localhost-only endpoint rule. This document is the lot's authority and
its historical record — it was annotated, and is not to be revised: the litellm
mentions it still carries describe the state the lot started from.

## 1. What this is for

Every model call in benchmarkit — the benchmark answer, the verifier verdict and
the WorkBench chat reply — goes through `litellm.completion()`. The app talks to
one kind of server (an OpenAI-compatible `/v1/chat/completions`, in practice
llama.cpp behind llama-swap) and uses litellm for exactly four things:

1. POST the JSON body and read the SSE stream.
2. Send `Authorization: Bearer <key>`.
3. Turn the wire bytes into chunk objects.
4. Log a failed call legibly (`LITELLM_DEBUG`).

Everything else litellm does is either unused or actively worked around:

- **A cost table the app never reads.** `register_unmapped_models()` exists only
  to stop litellm's own per-call `ModelNotMappedError` traceback from flooding
  the console while `LITELLM_DEBUG=1`. benchmarkit reports token counts, never a
  monetary figure.
- **A fabricated finish reason.** `saw_finish_reason()` has to reach into
  `CustomStreamWrapper.received_finish_reason` because litellm labels a stream
  the server cut mid-generation as `finish_reason: "stop"`. The honest signal is
  on the wire, and the app already re-reads it off a private attribute.
- **A close that is not there.** `close_llm_stream()` has to dig out
  `.completion_stream.close()` because the wrapper only offers `aclose()`, so a
  plain `close()` would silently leave the generation running.
- **A provider prefix on every call** (`model=f"openai/{name}"`) that litellm
  strips again before the request leaves.

Replacing it removes a heavy dependency and three workarounds, and makes the
truncation check honest by construction instead of by archaeology.

**Out of scope, and why:** no provider other than OpenAI-compatible (nothing else
is configured, and adding one is a different feature), no async client (both call
sites run in a worker thread by project convention), no retries (a retried stream
would double-count tokens and timing), no real tokenizer (the ~4 chars/token
estimate and its `tokensEstimated` flag stay as they are), no monetary cost
tracking, and no change to the app's own SSE layer, the hubs or the Editor.

## 2. The wire contract (measured, not assumed)

Measured on this machine against `http://localhost:8080/v1` with
`qwen3.8-flash-next-iq3_xxs`. **`http://192.168.1.4:8080/v1` is a developer
endpoint and was never probed — nor is it ever to be used by a test run, which
takes its endpoint from `.env.test`.**

### 2.1 Request

`POST {SERVER_URL}/chat/completions`, `SERVER_URL` already ending in `/v1` (it is
today's litellm `api_base`). Header: `Authorization: Bearer <api_key>`, sent even
when the key is the `"none"` placeholder — that is what happens today and the
server accepts it.

The body litellm puts on the wire today, for a call configured exactly like the
current `.env`:

```json
{"messages": [],
 "model": "mymodel",
 "max_tokens": 4,
 "presence_penalty": 0,
 "reasoning_effort": "medium",
 "temperature": 1,
 "top_p": 0.95,
 "top_k": 20,
 "min_p": 0,
 "repeat_penalty": 1}
```

and a streaming call adds `"stream": true` plus
`"stream_options": {"include_usage": true}`.

Three facts in that body decide the design:

- **`top_k`, `min_p`, `repeat_penalty` are passed through untouched.** They are
  llama.cpp sampler names, not OpenAI ones (litellm's own
  `get_supported_openai_params("openai/…")` does not list them, and it does not
  drop them either; `allowed_openai_params=["reasoning_effort"]` is what lets
  `reasoning_effort` through). `.env` sets `TEMPERATURE=1`, `TOP_K=20`, `MIN_P=0`,
  `REPEAT_PENALTY=1`, `REASONING=medium` today. **An in-house client must send the
  sampling dict as-is.** A client that whitelisted OpenAI's own list would change
  the model's sampling, and every score of every run, with nothing on screen to
  say so.
- **The model name travels bare.** litellm strips its own `openai/` prefix before
  sending. The new client sends the configured name verbatim: a leftover prefix
  reaches llama-swap as `openai/qwen3.8-…` and comes back
  `404 {"error":{"message":"no router for requested model"}}`.
- **Key order is irrelevant** (litellm sorts it; the server does not care).

### 2.2 Response

Streaming: `text/event-stream`, one `data: {json}` frame per event, blank lines as
separators, `data: [DONE]` last. The live server also emits `: keep-alive` comment
lines between frames, and its final frame carries `finish_reason: "stop"` **and**
the `usage` block (`prompt_tokens`, `completion_tokens`, `total_tokens`,
`prompt_tokens_details`) plus a llama.cpp `timings` object the app ignores.
Thinking arrives as `delta.reasoning_content`, which `_extract_reasoning` already
handles.

Non-streaming: a normal JSON body. Note that `choices[0].message.content` can be
**`null`** — observed on a reasoning-only response cut by `max_tokens` — so the
verifier's extraction must treat `None` as "no content" and answer `False` rather
than raise. Today's `verify_response` survives it only incidentally, through
`AttributeError`.

Errors carry a real HTTP status (`404`, `400`, `500`) and an OpenAI-shaped
`{"error": {"message": …, "type": …}}` body.

Unknown parameters are accepted by this server with `200` (probed with a bogus
key), so the client gets **no** signal about a typo'd sampler name. The design
must not assume a bad param is ever rejected.

## 3. Design decisions

| # | Decision | Alternative rejected |
|---|---|---|
| D1 | One new core package, `src/core/llmkit/`, as the only place in the app that speaks HTTP to a model endpoint. `core/llm.py` and `core/chat.py` keep their loops and stop knowing about transport. | Editing the two loops to call `httpx` directly. It spreads URL building, SSE parsing and error shaping over the two files that already carry the app's subtlest logic (the timing anchor, pause precedence, per-round usage). |
| D2 | The client yields **plain chunk dicts**, not objects. | Keeping litellm's attribute shape (`chunk.choices[0].delta.content`) so the extractors need no change. That means hand-rolling dataclasses to imitate a library being removed, and keeping the dual dict/attribute branches in `safe_get_choice_content`, `_extract_reasoning`, `_extract_usage` and `_collect_tool_call_deltas`, plus the `MagicMock` fixtures that feed them. The dict branch is the one the tests already use. |
| D3 | The client records **whether any chunk carried a `finish_reason`**, and nothing else decides truncation. `data: [DONE]` deliberately does **not** count as that signal. | Trusting `[DONE]` as proof the stream ended cleanly. It is transport framing, and whether it survives an upstream disconnect mid-generation on this stack is unverified; treating an unreliable marker as proof is the same mistake as litellm's fabricated `"stop"`. The `AGENTS.md` invariant ("a stream cut by the server looks like a finished answer") stays enforced by a chunk-level rule. |
| D4 | `saw_finish_reason()` and `close_llm_stream()` keep their names in `core/llm.py` (both are imported by `core/chat.py`) but become one-liners over the new stream object, still fails-open for `None` or a stub without the attribute. | Deleting them and editing the cancel/pause/truncation call sites in `chat.py`. Those sites encode precedence rules (cancel > error > pause) that are worth leaving untouched. |
| D5 | The seam both loops import is named for what it now is: `stream_completion` / `chat_completion`. Tests re-point their `monkeypatch.setattr` strings (mechanical, ~40 lines). | Keeping the name `completion` so no patch string moves. The name would then mean "litellm's function" in history and "our two-shape function" in the code — and the stubs must be rewritten anyway, because they have to carry `saw_finish_reason` and `close()`. |
| D6 | The debug switch is renamed `LITELLM_DEBUG` → `LLM_DEBUG` (settings field `llmDebug`), keeping the old key as a fallback. | Keeping the key name. The dialog labels the switch in the user's words ("Log LiteLLM internals"), and a checkbox naming a library that is no longer installed is a lie on screen. A fallback is precedent the project already follows with `MAX_TOOL_ITERATION` / `MAX_MCP_ITERATION`. |
| D7 | The zero-cost model registration is **deleted** with no replacement: `register_unmapped_models`, `register_configured_models`, `ZERO_COST_MODEL_INFO`, `LITELLM_MODEL_PREFIX`, `LITELLM_LOGGER_NAMES`, and their two call sites. | Keeping a cost table of our own. Nothing reads a price: `completion_cost` appears only inside `tests/test_config.py`, and the "cost" the UI shows is prompt + completion **tokens** (`core/bench.py`, `QuestionCard.vue`). |
| D8 | `LITELLM_DEBUG`'s real purpose — a failed call that names the upstream status and body — moves into the **exception message**, so it is available with debug off. The debug flag then adds the request payload on top. | A logger-only equivalent. Both loops do `except Exception as exc` and emit `str(exc)` straight into the UI's error line, so the message is the surface that matters. |
| D9 | URL building moves into `llmkit`; `routes._models_url` is imported from there. | Two rules for "what is the endpoint shape". They drift the first time a `SERVER_URL` without `/v1` shows up. |
| D10 | Generous read timeout, short connect timeout, in one `httpx.Timeout` in `llmkit/constants.py`. | A tight read timeout. A stalled prefill or a model load (warm-up exists precisely because that wait is long) would surface as a **truncated answer** — a transport setting masquerading as a model failure, which is the exact confusion this design exists to remove. |
| D11 | One `httpx.Client` per call, closed by the stream object; no shared long-lived client. | A hub-style singleton. Hubs are singletons because their connection is expensive and stateful (a stdio child, a session). A chat request is one POST from a transient worker thread, and `AGENTS.md`'s singleton rule is about never replacing a live hub object — nothing here holds a reference across requests. |


## 4. Shape of the new code

```
src/core/llmkit/
├── __init__.py
├── constants.py     # timeouts, the DONE sentinel, header name, key placeholder
├── sse.py           # SSE lines -> parsed chat-completion chunk dicts
└── client.py        # the two call shapes, the error type, the URL rules
```

This mirrors the `mcphub` / `lsphub` symmetry the project already keeps
(`constants` holds the tuning numbers, the leaf module holds the mechanism). Like
them, and like everything under `core/`, it imports no FastAPI and emits no
events: it is transport, not the event bus.

### 4.1 `constants.py`

```python
DONE_SENTINEL = "[DONE]"
DATA_PREFIX = "data:"
API_KEY_PLACEHOLDER = "none"
CONNECT_TIMEOUT = 10.0                    # a dead endpoint fails fast
READ_TIMEOUT: Optional[float] = None      # D10: no read deadline on a live stream
WRITE_TIMEOUT = 30.0
POOL_TIMEOUT = 10.0
ERROR_BODY_LIMIT = 500                    # unparseable body bytes shown to the UI
DEBUG_LOGGER_NAME = "benchmarkit.llm"
```

### 4.2 `sse.py`

```python
def parse_frame(raw: str) -> Optional[Dict[str, Any]]:
    """One SSE frame -> a chunk dict; None for [DONE], comments (keep-alive),
    blank lines and frames carrying no JSON object."""

def iter_chunks(lines: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Yield chunk dicts until [DONE], ignoring everything else."""
```

Rules worth stating, because the live server exercises all of them: the space
after `data:` is optional; `: keep-alive` is a comment, not an event; a frame
whose JSON does not parse **raises** — a half-read frame means the connection
died, and skipping it would hide a cut stream behind a frame that never carried a
finish reason (D3).

### 4.3 `client.py`

```python
class LLMRequestError(Exception):
    """A call the server refused. Carries the status and the upstream message."""
    def __init__(self, status: int, message: str, url: str) -> None: ...

def chat_completions_url(server_url: str) -> str: ...
def models_url(server_url: str) -> str:    # the rule today in routes._models_url

class ChatStream:
    """One streamed completion: iterable of chunk dicts plus its end state."""
    def __iter__(self) -> Iterator[Dict[str, Any]]: ...   # records as it yields
    @property
    def finish_reason(self) -> Optional[str]: ...         # first non-null seen
    @property
    def saw_finish_reason(self) -> bool: ...              # D3: chunk-level only
    @property
    def usage(self) -> Optional[Dict[str, Optional[int]]]: ...
    def close(self) -> None: ...    # idempotent, never raises (cancel path)

def stream_completion(*, model: str, server_url: str, messages: List[Dict],
                      api_key: str = API_KEY_PLACEHOLDER,
                      sampling: Optional[Dict[str, Any]] = None,
                      tools: Optional[List[Dict[str, Any]]] = None,
                      extra: Optional[Dict[str, Any]] = None) -> ChatStream:
    """POST with stream + include_usage; sampling/tools merged verbatim."""

def chat_completion(*, model: str, server_url: str, messages: List[Dict],
                    api_key: str = API_KEY_PLACEHOLDER,
                    max_tokens: Optional[int] = None,
                    sampling: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """POST without stream; returns the parsed body."""
```

`sampling` is merged into the body **without a whitelist** (§2.1). `tools` is
added only when non-empty: the existing rule that a request carries *no `tools`
key rather than `tools: []`*
(`tests/test_chat.py` — `…_leaves_no_tools_key_at_all`) travels down into the
client with it.

`ChatStream.close()` closes the `httpx` response and its client. That is the
whole of what `close_llm_stream()` digs out of litellm today, and it is what lets
llama.cpp free the generation slot on its next failed write instead of streaming
into a socket nobody reads. (This is the socket's own teardown, not the
process-tree teardown `core/proctree.py` owns — the two are unrelated.)

### 4.4 Changes to the existing modules

| File | Change |
|---|---|
| `src/core/llm.py` | `from litellm import completion` → the two `llmkit` imports. `call_llm_and_stream` uses `stream_completion` and iterates a `ChatStream`; the `close_llm_stream(gen)` call site is unchanged in name. `warmup_model` / `verify_response` use `chat_completion`; the `openai/` prefix is dropped; the verifier's content extraction handles `content: None`. `saw_finish_reason()` becomes `bool(getattr(gen, "saw_finish_reason", True))`; `close_llm_stream()` becomes a best-effort `gen.close()`; `_ATTR_MISSING` goes. The chunk extractors lose their attribute branch (D2). |
| `src/core/chat.py` | Same seam. The `kwargs` dict it builds becomes the `stream_completion` keyword set (`sampling=sampling_kwargs`, `tools=tools`); `allowed` / `allowed_openai_params` disappear, since nothing gates a param any more. The loop body, the cancel/pause precedence and every `emit` stay textually as they are. |
| `src/core/models.py` | `SamplingParams` docstring: "mapped to litellm/OpenAI-compatible names" → the request-body parameters of an OpenAI-compatible server. `to_kwargs()` keeps its name and its output. |
| `src/config.py` | `apply_litellm_debug` → `apply_llm_debug` (raise `benchmarkit.llm` to DEBUG and log the request line plus payload); `litellm_debug_enabled` → `llm_debug_enabled`, reading `LLM_DEBUG` with a `LITELLM_DEBUG` fallback (D6). `LITELLM_DEBUG_KEY` → `LLM_DEBUG_KEY`; the `SETTING_KEYS` entry renamed; the `_render_env` comment block rewritten (it currently explains litellm's three loggers). The cost-registration block deleted (D7). |
| `src/api/server.py` | One boot call instead of two (`apply_llm_debug()`; `register_configured_models()` gone) and the two comments that explain them. |
| `src/api/routes.py` | `SETTINGS_ENV_KEYS["llmDebug"] = "LLM_DEBUG"`, `_settings_env_values` normalising `LLM_DEBUG`, `_models_url` imported from `llmkit` (local definition and docstring go), and the docstrings that say "sync litellm in a worker thread" now say "the sync LLM client". |
| `src/config.py::save_settings` | Re-applies `apply_llm_debug()`; the `register_configured_models()` line and its comment go. |
| `web/src/components/SettingsDialog.vue` | `state.settings.llmDebug`, label "Log LLM request internals (debug)", caption reworded (it currently promises "LiteLLM's own DEBUG output"). |
| `.env.example`, `.env.test`, `.env.test.example` | The `#LITELLM_DEBUG=1` line becomes `#LLM_DEBUG=1`, comment included. |
| `pyproject.toml` | `"litellm"` removed from `dependencies`. `httpx` is already a dependency. |
| `AGENTS.md` | In *Key patterns*, name `core/llmkit` as the one seam that speaks HTTP to a model endpoint, and state that sampling params reach the server verbatim. Both are invariants, so they get a line; the implementation does not. |
| `README.md` | Any litellm mention (configuration/install text) reworded. |

Left alone on purpose: `todos/done/*` and `.kilo/plans/*` (historical records),
`logs/_baseline_check/*` (an untracked reference copy of two older files), and
`src/core/mcphub/formatting.py`, whose only mention is a docstring that should now
read "handed to the LLM client".

## 5. Step-by-step

Each phase ends with the suite green and the app usable, and each is a revert
point.

### Phase 0 — Pin the contract (no behaviour change)

1. Add `tests/test_llmkit_contract.py` as a **characterization** file: a
   `FakeStream` double that yields dicts (the shape every existing stub already
   feeds) and carries `saw_finish_reason` and `close()`, so the stub and the real
   client are held to one interface from the start.
2. Record §2.1 as a fixture: the exact body for `TEMPERATURE=1, TOP_P=0.95,
   TOP_K=20, MIN_P=0, REPEAT_PENALTY=1, PRESENCE_PENALTY=0, REASONING=medium`.
   The point of the fixture is that `top_k`, `min_p` and `repeat_penalty` **are**
   in it, and that `model` carries no prefix.

### Phase 1 — Build `llmkit`, still unused

3. `constants.py` (§4.1), then `sse.py` (§4.2), then `client.py` (§4.3).
4. New `tests/test_llmkit_sse.py`: `data:` with and without the space;
   `: keep-alive` and other comment lines skipped; `[DONE]` ends iteration and
   yields no chunk; a usage-only frame with `choices: []` parses; unparseable JSON
   raises.
5. New `tests/test_llmkit_client.py`, driven by a fake transport (no network):
   - a streaming body equals the Phase 0 fixture plus `stream`/`stream_options`;
   - `Authorization: Bearer none` is sent when no key is configured;
   - the model name reaches the body bare;
   - `sampling` passthrough — an unknown sampler key appears in the body verbatim;
   - `tools` absent when `None` or empty, present when non-empty;
   - a `finish_reason` seen on a late chunk → `saw_finish_reason is True`; never
     seen → `False`, **even after `[DONE]`** (D3);
   - HTTP `404` with an OpenAI error body → `LLMRequestError` whose `str()` holds
     both the status and `"no router for requested model"` (D8); a non-JSON body
     is truncated to `ERROR_BODY_LIMIT` instead of raising a decode error;
   - `close()` twice is a no-op, and a failing `close()` never propagates;
   - `chat_completions_url` and `models_url` agree for a `…/v1` base and for a
     bare host — the D9 rule, asserted once for both shapes.
6. Guard the new module the way the project guards its other boundaries: assert
   that `llmkit`'s URL helpers can never produce a path under `/api/`, so the new
   HTTP client cannot become a caller of the app's own write routes — the
   invariant `tests/test_write_surface_isolation.py` protects. (Its
   `http_import_offenders` scan covers `mcp_connectors/`, not `src/core`, so
   `httpx` under `core/llmkit/` is legal — but the write-marker scan in
   `scan_roots()` **does** walk `src/core`, so `llmkit` must never contain a
   `PUT /api/fs`, `POST /api/fs` or `DELETE /api/fs` string.)

### Phase 2 — Move the benchmark across

7. In `src/core/llm.py`, swap the seam (D5) and rewire `call_llm_and_stream`,
   `warmup_model` and `verify_response`.
8. Re-point every stub in `tests/test_llm.py` from `src.core.llm.completion` to
   `src.core.llm.stream_completion` / `src.core.llm.chat_completion`, and the
   `Wrapper` / `MagicMock` chunk fixtures to `FakeStream` and plain dicts. The
   affected classes are `TestStreamTruncationDetection`, `TestCloseLlmStream`,
   `TestWarmupModel`, `TestCallLlmApiKey` and `TestVerifyResponseApiKey`; the
   timing tests need no shape change, since they already feed dicts.
9. Where a test asserted litellm's kwargs (`kwargs["model"] == "openai/my-model"`,
   `kwargs["api_base"]`, `kwargs["api_key"]`), assert the new shape instead: the
   bare model in the body, the URL, the bearer header.
10. Add `verify_response` coverage for `content: None` (§2.2) — a judge that says
    nothing is a `False`, not a crash.
11. `pytest tests/test_llm.py tests/test_bench.py`, then a real short run (§6.3.1):
    the numbers must stay in the pre-migration band, because timing is app-side —
    a shift means the anchor or the client moved.

### Phase 3 — Move the chat across

12. Same swap in `src/core/chat.py`; delete `allowed` / `allowed_openai_params`.
13. Re-point `src.core.chat.completion` stubs to `src.core.chat.stream_completion`
    across `tests/test_chat.py` and `tests/test_agent_modes.py`; `Wrapper` becomes
    `FakeStream`.
14. `test_multimodal_content_passed_through_untouched` keeps its meaning with the
    new seam: the multimodal parts reach the request body untouched.
15. Confirm the loop's terminal-event contract is untouched — exactly one of
    `done` / `error` / `pause_ack`, a paused reply still ends on its own ack, and a
    truncated round never executes its half-streamed tool call. Those are the
    behaviours `stream_chat`'s docstring pins; this phase changes what a chunk is,
    not what a pause is.
16. `pytest tests/test_chat.py tests/test_agent_modes.py tests/test_batch.py`.

### Phase 4 — The debug switch, minus the cost table

17. `src/config.py`: rename the flag pair, add the `LITELLM_DEBUG` fallback,
    implement `apply_llm_debug` against `benchmarkit.llm` (which `llmkit` logs the
    payload to), and delete the cost-registration block (D7).
18. `src/api/server.py` and `save_settings`: one call each instead of two.
19. `src/api/routes.py` and `web/src/components/SettingsDialog.vue`: the `llmDebug`
    field, the new label and the new caption.
20. `.env.example`, `.env.test`, `.env.test.example` and `_render_env`: the renamed
    key and its comment. `.env` itself is user data, so the change note says that a
    hand-written `LITELLM_DEBUG=1` keeps working through the fallback until the
    dialog is saved.
21. Tests: `TestLiteLLMDebugFlag` → the renamed pair, plus one that the **old** key
    still enables it and the new key wins when both are set; delete
    `TestUnmappedModelCostRegistration`; update
    `test_save_normalizes_litellm_debug_and_applies_it` to the new field name.
22. Add an `llmkit` test that an HTTP failure's `str()` names the status and the
    upstream message with the debug flag **off** (D8) — that message is what the
    UI's error line shows.

### Phase 5 — Remove the dependency

23. `pyproject.toml`: drop `litellm`, then reinstall with
    `pip install -e ".[mcp,lsp,test]"` (nothing else in the tree needs it; the hubs
    pull `mcp` and `pygls`).
24. Purge the remaining mentions. `grep -rn "litellm" src tests web/src
    pyproject.toml`, plus the same scan over the committed env examples and
    `README.md`, returns nothing except (a) the D6 fallback key in `src/config.py`
    and its test, and (b) the docstring in `src/core/mcphub/formatting.py`, reworded
    to "the LLM client".
25. Keep `"litellm"` in both forbidden-import tuples in `tests/test_agent_modes.py`
    (`test_the_loader_imports_no_transport_no_model_and_no_hub` and its sibling),
    and **add `"llmkit"`** to the same tuples: a core loader must stay free of the
    LLM transport whatever that transport is called.
26. Update `AGENTS.md` as set out in §4.4 — one or two lines in *Key patterns*, in
    the file's own style, naming the invariant and not the implementation.

### Phase 6 — Verify

See §6. Nothing here is a code change except what a failure sends you back to.

## 6. How it is verified

### 6.1 Unit

```
pytest tests/test_llmkit_sse.py tests/test_llmkit_client.py \
       tests/test_llm.py tests/test_chat.py tests/test_bench.py \
       tests/test_config.py tests/test_api.py
pytest            # full suite
```

The suite stays hermetic the way the project requires: `llmkit` is exercised
through a fake transport, never a live endpoint, just as the hubs are switched off
by `BENCHMARKIT_DISABLE_MCP` / `BENCHMARKIT_DISABLE_LSP`.

### 6.2 Static

- `grep -rn "litellm" src tests web/src pyproject.toml` → only the two items of
  step 24.
- `grep -rn "completion_stream\|received_finish_reason\|intermittent_finish_reason" src`
  → empty: the litellm-shaped workarounds are gone, not renamed.
- `grep -rn "httpx" src/core` → the model-facing HTTP sits only in `core/llmkit`;
  `mcphub`, `lsphub` and `editorfs` unchanged.

### 6.3 Live smoke (manual, at phases 2, 3 and 6)

Against `http://localhost:8080/v1`, model `qwen3.8-flash-next-iq3_xxs`. **Never
`http://192.168.1.4:8080/v1`** — that is a developer endpoint; a test run takes its
endpoint from `.env.test`, never from a developer's `.env`.

1. **Benchmark, two questions, verifier on.** Tokens stream into the card; `tok/s`
   and `ttft` land in the same band as the pre-migration run of the same set;
   `consumed` equals prompt + completion; the verdicts match the baseline. A
   verdict that flips means the request body changed — re-read §2.1 before
   doubting the model.
2. **Thinking visible.** With `REASONING=medium`, the reasoning panel fills from
   `reasoning_content`, and reasoning text is never answer text or part of the
   message sent back to the model.
3. **Stop mid-answer.** Run → Stop, and the server's generation slot is free within
   a second or two — visible as the next request not queueing behind a phantom.
   That is `ChatStream.close()` doing what `close_llm_stream()` does today.
4. **WorkBench with a tool.** One turn that calls an MCP connector and then answers:
   `tool_call`, `tool_result`, the pre-tool `usage` frame and exactly one `done`,
   with the token badge reading the server's usage rather than the estimate.
5. **Pause mid-reply.** The partial text stays in history, `pause_ack` closes the
   reply, and it is **not** flagged truncated.
6. **Truncation path.** Point the endpoint at a server that closes the stream
   without ever sending a finish reason (a proxy that drops the connection works).
   Expected: the run fails that question and never sends the fragment to the
   verifier; WorkBench marks the message with the truncation error. This is D3
   earning its keep, and it is the one check that must be repeated after any change
   to `sse.py`.
7. **Debug switch.** Tick "Log LLM request internals", call a bad model name, and
   read one legible `404 … no router for requested model` line plus the payload on
   the console; untick it and confirm the console is quiet again.
8. **Settings round-trip.** Save the dialog and confirm `.env` is regenerated from
   the template with `#LLM_DEBUG=` commented out and no other managed key lost.

### 6.4 Rollback

Phases 2 and 3 are the only ones that change a live path, and each is one file:
`src/core/llm.py` or `src/core/chat.py`. Reverting either returns that path to
litellm while `llmkit` sits unused by the other — which is why Phase 1 lands before
both and Phase 5 (removing the dependency) lands after both. Do not drop `litellm`
from `pyproject.toml` until a full-suite pass and §6.3.1–6.3.5 have run with both
paths switched.

## 7. Open questions to settle before Phase 1

Settled before Phase 1 by the operator's brief (items 1–4 of "Settled here" in
`todos/llmh-00_big_picture.md`); each answer is noted under its question rather
than in place of it.

1. **D6 rename** — go ahead with `LLM_DEBUG` plus the fallback, or keep the key and
   only relabel the dialog? The spec assumes the rename, which the fallback makes
   free.
   **Answered: renamed** — `LLM_DEBUG` with the old key as the fallback in
   `llm_debug_enabled`, and the settings field `llmDebug` (llmh-07, llmh-08).
2. **D2 dual-shape extractors** — delete the attribute branches (the spec's
   position), or keep them as a defensive path now that only dicts arrive? Deleting
   also retires the `MagicMock` chunk fixtures in `tests/test_llm.py`.
   **Answered: deleted**, and with them the `MagicMock` chunk fixtures, which
   became dict cases (llmh-05, llmh-06).
3. **`[DONE]` as a signal (D3)** — rejected here on reasoning, not on measurement.
   If someone measures that llama-swap does omit `[DONE]` on an upstream cut, that
   is evidence worth recording in §3; the chunk-level rule needs no such evidence to
   stay correct.
   **Answered: as designed** — `[DONE]` is transport framing and never counts.
   What has been observed since, both against localhost: a clean stream does carry
   `data: [DONE]` (llmh-02's probe through `iter_chunks`, 2026-11-04), and llmh-10's
   hermetic end-to-end case pins that a body carrying `[DONE]` with **no**
   finish-reason chunk is still truncated through the real parse→judge chain
   (2026-11-05). Neither measures what an upstream cut omits — the fact this rule
   refuses to rely on either way.
4. **`extra` on `stream_completion`** — nothing needs it yet (warm-up's
   `max_tokens=1` rides on `chat_completion`). Keep the parameter for future
   llama.cpp extras, or drop it until a caller exists?
   **Answered: dropped** — no caller exists, and §8 leaves the shape free to move
   (llmh-03).

## 8. What this spec deliberately does not decide

- **Package name.** `llmkit` is used throughout so the steps are concrete. A name
  the maintainer prefers (`llmhttp`, `openaicompat`) changes only the paths.
- **Whether to keep `src/core/llm.py` as a module or merge it into `llmkit`.** It
  stays as the benchmark's answer/verifier/warm-up layer, because `core/bench.py`
  and its tests import four names from it and that boundary is not this change.
- **Chunk dataclasses, pydantic response models, a retry/backoff layer, a
  `/completions` path, embeddings.** None has a caller in this app today.

---

*Next step:* hand this spec to the Todo builder — one task per numbered step,
keeping the phase boundaries, since each phase is a point where the full suite is
green and the app still runs.


