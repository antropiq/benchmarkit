# llmh-05 — Move the benchmark across: `src/core/llm.py` speaks `llmkit`

**Spec:** Phase 2 (steps 7–11), §4.4 row `src/core/llm.py`, D2, D3, D4, D5, §2.2.
**Depends on:** llmh-01, llmh-02, llmh-03, llmh-04.
**Revert point:** this is one file. Reverting `src/core/llm.py` alone returns the
benchmark path to litellm with `llmkit` otherwise unused (§6.4).

## Deliver this

`src/core/llm.py` stops knowing about transport. Everything else in the file —
the timing anchor, the token back-fill ladder, the cancellation shape, every
`emit` — stays textually where it is.

The swap (steps 7, 9, 10):

* `from litellm import completion` goes; import `stream_completion` and
  `chat_completion` from `src.core.llmkit`.
* `call_llm_and_stream` calls `stream_completion(model=model_name,
  server_url=server_url, messages=messages, api_key=api_key,
  sampling=sampling_kwargs)` and iterates the returned `ChatStream`. The
  `allowed` / `allowed_openai_params` lines and their comment disappear —
  nothing gates a param any more. **Do not move `t0`, the `arrival` stamping,
  the `t_first`/`t_last` window or the usage-vs-estimate fallback**: they are
  the app-side measurement the AGENTS.md measurement rules pin.
* `warmup_model` uses `chat_completion(..., max_tokens=1)`; `verify_response`
  uses `chat_completion(...)`. The `openai/` prefix is dropped at all three call
  sites (bare model name — the lot's second invariant).
* `saw_finish_reason()` keeps its name and its signature and becomes
  `bool(getattr(response_generator, "saw_finish_reason", True))` — still
  **fails open** for `None` or a stub without the attribute (D4).
  `_ATTR_MISSING` goes. `close_llm_stream()` keeps its name and becomes a
  best-effort `close()` that swallows every error.
* The chunk extractors (`safe_get_choice_content`, `_extract_reasoning`,
  `_extract_usage`, `_extract_total_tokens`) lose their attribute branch — dict
  only (D2). `_extract_usage`'s tolerance of missing/`None` fields stays.
* `verify_response`'s content extraction handles `"content": None` explicitly
  (§2.2: a reasoning-only answer cut by `max_tokens` really does return `null`):
  a judge that says nothing is `False`, never a crash. The object branch goes
  with the extractors' (a dict arrives now), and the `content` variable is
  normalised to `""` rather than relying on falsiness.

Then `tests/test_llm.py`, re-pointed (steps 8, 9):

* every `monkeypatch.setattr("src.core.llm.completion", …)` becomes
  `…llm.stream_completion` or `…llm.chat_completion` — the generator-shaped
  stubs to `stream_completion`, the whole-response stubs (`TestWarmupModel`,
  `TestVerifyResponseApiKey`, the `TestCallLlmApiKey` non-streaming half) to
  `chat_completion`;
* the `Wrapper` / `MagicMock` stream and chunk fixtures become
  `tests.llmkit_fakes.FakeStream` (from llmh-01) and **plain dicts**;
* the affected classes named by step 8 are `TestStreamTruncationDetection`,
  `TestCloseLlmStream`, `TestWarmupModel`, `TestCallLlmApiKey`,
  `TestVerifyResponseApiKey`. `TestCloseLlmStream`'s two litellm-specific cases
  (`prefers_completion_stream_close`, `falls_back_to_generator_close`) collapse
  into one: `close()` is called once on the stream. Its
  `test_swallows_close_errors` and `test_missing_close_is_noop` stay.
  `TestStreamTruncationDetection.test_plain_iterable_without_the_attributes_is_not_flagged`
  keeps its meaning verbatim (a bare iterable fails **open**).
* where a test asserted litellm's kwargs (`kwargs["model"] == "openai/my-model"`,
  `api_base`, `api_key`), assert the new shape: bare `model`, `server_url`,
  `api_key` as passed to the seam. The URL/header assertions already live in
  `tests/test_llmkit_client.py` — do not duplicate them here.
* the object-chunk cases named in the big picture ("Settled" item 2) become dict
  cases or are dropped if they duplicate an existing dict case.
* **new** (step 10): a `verify_response` test feeding `"content": None` —
  verdict `False`, no `run_error` emitted for it.

`TestCallLlmAndStreamTiming` must pass **without any change to its assertions**
(it already feeds dicts). If a timing test needs editing, the swap is wrong —
stop and re-read the measurement rules below.

## Files it touches

| file | change |
|---|---|
| `src/core/llm.py` | edit — the seam swap above |
| `tests/test_llm.py` | edit — re-pointed stubs, dict chunks, new `content: None` case |
| `tests/test_bench.py` | expected **unchanged**; touch only if a stub there reached `src.core.llm.completion` (check) |

`src/core/chat.py` still imports `saw_finish_reason` / `close_llm_stream` from
here and still calls litellm itself — that is llmh-06. Both must keep working
after this task, which is why the two helper **names and signatures stay**.

## Rules it must not break (quoted)

* AGENTS.md: *"A truncated benchmark answer is failed and never sent to the
  verifier (grading a fragment measures the transport, not the model)"* — the
  `truncated = generated_any and not saw_finish_reason(...)` rule and its
  after-the-loop position stay exactly as they are (D3).
* AGENTS.md: *"**Timing is app-side**, anchored to content-chunk arrivals —
  warm-up and verifier time never enter the numbers."*
* *"token/s uses completion tokens only, the displayed cost is prompt +
  completion."* `completion_tokens` / `prompt_tokens` in `LLMResult` keep their
  meanings.
* *"A truncated … reply ends as an error"* / cancellation: `close_llm_stream`
  is still called on the cancel path **before** `RunCancelled` is raised.
* **Core owns logic, the API owns transport** (AGENTS.md): no `src.api` import
  appears here.
* `core/bench.py` imports four names from this module. They keep working:
  `call_llm_and_stream`, `warmup_model`, `verify_response`, `LLMResult`
  (check the actual import list; change none of it).

## Acceptance

```
pytest tests/test_llm.py tests/test_bench.py tests/test_write_surface_isolation.py -q
```
green, and nothing regressed on the still-litellm chat path:

```
pytest tests/test_chat.py tests/test_agent_modes.py -q   # unchanged, green
```

## Live smoke (spec §6.3.1) — the reason Phase 2 ends with a real run

If `http://localhost:8080/v1` is up (model `qwen3.8-flash-next-iq3_xxs`): run
two questions with the verifier on, and confirm tokens stream into the card,
`tok/s` and `ttft` land in the same band as the pre-migration run of the same
set, `consumed` equals prompt + completion, and the verdicts match. A flipped
verdict means the request body changed — re-read §2.1 before doubting the model.
**Never** use `http://192.168.1.4:8080/v1`; and if a real run is needed, point
the app at localhost through the Settings dialog rather than editing `.env.test`
(big picture, "Endpoint rule"). If the server is unreachable, record that the
smoke was skipped and why.

## Done — outcome (2026-11-04)

`src/core/llm.py` no longer knows about transport: `stream_completion` for the
answer loop, `chat_completion` for warm-up (`max_tokens=1`) and the verifier,
bare model name at all three sites, `allowed` / `allowed_openai_params` gone,
`verify_response` normalising a missing/null `content` to `""`. `t0`, the
`arrival` stamping, the `t_first`/`t_last` window and the usage-vs-estimate
ladder are untouched; `core/bench.py`'s four imports unchanged; no `src.api`.

`tests/test_llm.py` re-pointed: generator stubs → `stream_completion`,
whole-response stubs → `chat_completion`, `Wrapper`/`MagicMock` fixtures →
`FakeStream` + plain dicts, `TestCloseLlmStream`'s two litellm cases collapsed
into one `close()`-called-once case, kwargs assertions now bare `model` /
`server_url`, and the new `content: None` verdict case.
`TestCallLlmAndStreamTiming` needed **only** its two patch strings changed — no
assertion edited. `tests/test_bench.py` untouched (it never reached
`src.core.llm.completion`). `tests/test_write_surface_isolation.py`'s diff in
the tree is llmh-04's, not this task's.

Acceptance green: `pytest tests/test_llm.py tests/test_bench.py
tests/test_write_surface_isolation.py -q` → 82 passed; `pytest tests/test_chat.py
tests/test_agent_modes.py -q` → 224 passed with **`tests/test_chat.py` byte-
identical to before this task** (`git status` shows it unmodified); pyright clean
on both files.

### Deliberate deviation, and why (read this before llmh-06)

The two shared helpers and the three chunk extractors are **not** the literal
one-liners the task text gives, because `core/chat.py` still streams through the
client being replaced and shares all five. Each shortcut was measured live
(`http://localhost:8080/v1`, model `qwen3.8-flash-next-iq3_xxs`) and each breaks
an invariant this task itself lists as unbreakable, silently, for the whole
05→06 window:

* **Extractors** (`safe_get_choice_content`, `_extract_reasoning`,
  `_extract_usage`): the *attribute branch* is deleted as D2 requires, but the
  read is the **mapping protocol** (`.get`) rather than `isinstance(chunk, dict)`.
  With the dict guard, the live WorkBench path streamed **an empty answer ending
  on a clean `done`** (`answer text: ''`, `non-empty answer: False`) — the old
  attribute branch over the same live call returns `OK` plus 23 reasoning
  deltas. A dict-only read passes every dict-feeding unit test, so nothing in
  the suite would have caught it; that is the transport-masquerading-as-the-model
  failure AGENTS.md names. Pinned by `TestChunksAreReadAsMappings`.
* **`saw_finish_reason()`**: `saw_finish_reason` first, and only when absent the
  interim `received_finish_reason` / `intermittent_finish_reason` read (with
  `_ATTR_MISSING`, which the task deletes). Measured: that wrapper has **no**
  `saw_finish_reason` at all, so the literal one-liner returns `True` for every
  stream and chat's truncation check stops firing — a cut reply would end on
  `done` and **execute its half-streamed tool call**, which the rules forbid.
  Fails open exactly as before for `None` and stubs.
* **`close_llm_stream()`**: `gen.close()` first, falling back to
  `gen.completion_stream.close()` only when the stream has no `close()`.
  Measured: the wrapper exposes `aclose()` and **no** sync `close()`, so the
  literal version is a silent no-op on the chat path — Stop would leave the
  server generating into a dead socket.

All three interim branches are commented `**llmh-06 deletes this whole
branch**`. Consequences for the lot: **llmh-06 must delete them** (its own file
already says the helpers become one-liners — it now also has to drop the
fallbacks and `_ATTR_MISSING`), and until it does, the lot's definition-of-done
scans cannot be clean: `grep -rn "completion_stream\|received_finish_reason\|
intermittent_finish_reason" src` matches `core/llm.py`, and llmh-09's litellm
purge must not try to clean these names by hand — they go with chat.

### Live smoke (§6.3.1) — done, localhost only

Against `http://localhost:8080/v1`, model `qwen3.8-flash-next-iq3_xxs`, two
questions with the verifier on (`logs/probe_llmh05/probe.py`, run by hand; the
suite stayed hermetic). Body carries the sampling verbatim
(`top_k`, `min_p`, `repeat_penalty`, `reasoning_effort` all present) and the
model bare:

* tokens stream into the card (99 and 121 `token` events), no `run_error`, no
  truncation;
* `ttft` 0.92 s / 0.63 s, `tok/s` 55.5 / 60.3 — same band across three repeats
  of the same two questions (55.7, 60.8, 63.5), so the anchor did not move;
* `consumed` = prompt + completion (68+233=301, 46+631=677), server-reported,
  `tokens_estimated` False;
* both verdicts `true`, unchanged across the repeats.

`http://192.168.1.4:8080/v1` was not contacted; the probe hardcodes localhost
and the placeholder key. `.env` and `.env.test` were not edited.
