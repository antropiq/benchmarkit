# llmh-01 — Pin the contract: `FakeStream` and the §2.1 request-body fixture

**Spec:** `docs/in-house-llm-client-spec.md` Phase 0 (steps 1–2), §2.1, D2, D3.
**Depends on:** nothing. **Leaves the app working:** this task changes no
production code — it only adds test artifacts that nothing imports yet.

## Deliver this

Two test artifacts that every later task of this lot is written against:

1. **`tests/llmkit_fakes.py`** — a `FakeStream` double for the stream object the
   new client will return:
   * iterates plain **chunk dicts** (the shape the existing stubs in
     `tests/test_llm.py` / `tests/test_chat.py` already feed);
   * records and exposes `saw_finish_reason` (**bool**, set the moment any chunk
     carried a non-null `finish_reason`) and `finish_reason` (first non-null
     seen) — computed while iterating, exactly the way `ChatStream` will compute
     them (D3: `[DONE]` is not a signal, and a fake must not be sneakier than
     the real thing);
   * exposes `usage` (last usage dict seen, or `None`);
   * a `close()` that is idempotent, records that it was called
     (`closed` count or flag), and can be configured to raise — the cancel-path
     tests need a close that fails and must still not propagate;
   * optionally raises a configured exception partway through iteration, so the
     truncation tests can simulate a cut connection.
   Keep it small and constructor-driven, e.g.
   `FakeStream(chunks, close_error=None, error_after=None)`. A shared helper in
   it for building common chunks (content delta / reasoning delta / usage-only
   frame / finish frame) is welcome — later tasks reuse it instead of
   hand-rolling dicts.

2. **`tests/test_llmkit_contract.py`** — a **characterization** file. No
   `src.core.llmkit` import (it does not exist yet); it pins the *contract* and
   the *wire facts*:
   * `FakeStream` satisfies the interface: iterable of dicts, `saw_finish_reason`
     starts `False` and flips only on a chunk with a non-null `finish_reason`
     **after that chunk has been yielded**; `close()` twice is a no-op; a failing
     `close()` is survivable;
   * the **§2.1 request-body fixture**: a module-level constant
     `EXPECTED_STREAM_BODY` holding the exact JSON body litellm puts on the wire
     today for `.env`'s current settings — `TEMPERATURE=1, TOP_P=0.95, TOP_K=20,
     MIN_P=0, REPEAT_PENALTY=1, PRESENCE_PENALTY=0, REASONING=medium` — i.e. the
     body in spec §2.1 plus `"stream": true` and
     `"stream_options": {"include_usage": true}` for the streaming shape. Assert
     on the fixture itself: `top_k`, `min_p`, `repeat_penalty`, `reasoning_effort`
     **are** in it, and `model` has **no** `openai/` prefix. llmh-03 will diff
     the real client's body against this constant — the point of the fixture is
     that the comparison is made against the *measured wire*, not against the
     spec author's memory.
   * one test naming the invariant in prose (a docstring is fine): an unknown
     sampler key must survive to the body untouched — the server accepts garbage
     with `200`, so no layer may "validate" the sampling dict (§2.1 last
     paragraph).

Read spec §2.1 carefully before writing the fixture: the `max_tokens: 4` /
`"mymodel"` in the spec's example are placeholders for the call's own values —
give the fixture named constants (`MODEL = "mymodel"`, `MAX_TOKENS = 4`) so
llmh-03 can substitute its own.

## Files it touches

| file | change |
|---|---|
| `tests/llmkit_fakes.py` | **new** — `FakeStream` + chunk-builder helpers |
| `tests/test_llmkit_contract.py` | **new** — the characterization tests + `EXPECTED_STREAM_BODY` |

## Rules it must not break

* **Tests stay hermetic** (AGENTS.md): no network, no `.env` reads in these
  files.
* Chunk shape is **plain dicts** (D2) — no attribute-bearing fakes; the object
  branch of the extractors is being deleted in llmh-05, not extended.
* `FakeStream`'s `saw_finish_reason` must be **iter-driven**: constructing it
  from a list eagerly (e.g. scanning all chunks up front) would let a stub claim
  a finish reason the loop never reached, which is the exact lie D3 exists to
  prevent.

## Acceptance

```
pytest tests/test_llmkit_contract.py -q
```
green; nothing else in the suite changes (nothing imports the new files yet).

## Notes for the reader

You do not need the live server for this task. Later tasks (llmh-03, llmh-05,
llmh-06) import `FakeStream` from this file; keep its constructor kwargs
additive, never renaming what is here.

