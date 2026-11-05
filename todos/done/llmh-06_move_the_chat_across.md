# llmh-06 — Move the chat across: `src/core/chat.py` speaks `llmkit`

**Spec:** Phase 3 (steps 12–16), §4.4 rows `src/core/chat.py` and
`src/core/models.py`, D2, D4, D5.
**Depends on:** llmh-05 (llmh-05 makes `saw_finish_reason` / `close_llm_stream`
in `core/llm.py` the ChatStream-aware one-liners that chat inherits).
**Revert point:** one file — reverting `src/core/chat.py` alone returns the chat
path to litellm (§6.4).
**Sequencing:** must not overlap with llmh-05 (both re-point stub machinery).

## Deliver this

**`src/core/chat.py`** — the same seam swap, and *nothing else moves*:

* `from litellm import completion` goes; import `stream_completion` from
  `src.core.llmkit`.
* The `kwargs` dict it builds each round becomes the `stream_completion`
  keyword set: `model=model_name` (bare), `server_url`, `messages`, `api_key`,
  `sampling=sampling_kwargs`, `tools=tools` (pass `use_tools and tools` or
  `tools if use_tools else None` — the client already omits an empty/`None`
  `tools`, so the "no `tools` key at all" rule keeps holding; the existing test
  `…leaves_no_tools_key_at_all` stays green and stays the proof).
* The `allowed` / `allowed_openai_params` lines and their comment (near the
  `sampling_kwargs` build) disappear. The litellm-overloads comment at the call
  site goes too.
* The loop body, `close_llm_stream` call sites, the **cancel > pause precedence
  and its comment**, the per-round usage ladder, the truncation check via
  `saw_finish_reason(response_generator)` after the loop, and **every `emit`
  stay textually as they are**. `_collect_tool_call_deltas` keeps working on
  dict chunks (it already handles dicts; drop only its attribute branch if it
  has one).
* Docstring/comment mentions of litellm in this file become "the LLM client"
  (also in the module docstring and the `stream_chat` docstring if present).

**`src/core/models.py`** — the `SamplingParams` docstring first line changes
from "mapped to litellm/OpenAI-compatible names" to the request-body parameters
of an OpenAI-compatible server. `to_kwargs()` keeps its name and output. Nothing
else in the file.

Then the tests, re-pointed (steps 13, 14):

* every `monkeypatch.setattr("src.core.chat.completion", …)` (about 18 sites in
  `tests/test_chat.py`, one in `tests/test_agent_modes.py`) becomes
  `…chat.stream_completion`; each `Wrapper`-style stream stub becomes
  `tests.llmkit_fakes.FakeStream`; attribute-shaped chunk fixtures become dicts;
* `test_multimodal_content_passed_through_untouched` keeps its meaning with the
  new seam: assert the multimodal parts reach the **keyword arguments** of
  `stream_completion` untouched (it spies the seam now, not a body — the body
  assertions live in `tests/test_llmkit_client.py`);
* the stubs that record `**kwargs` now record the new keyword names — assertions
  like `kwargs["model"] == "openai/…"` become `model == "…"`;
* `tests/test_agent_modes.py`'s spy at `src.core.chat.completion` (its
  "assembled litellm request" comment included) re-points the same way; its two
  forbidden-import tuples **keep** `"litellm"` and are otherwise untouched here
  (adding `"llmkit"` to them is llmh-09, after the package is real policy).

Step 15 is a *confirmation*, not a code change: run the file's own terminal-event
tests and confirm by reading — exactly one of `done` / `error` / `pause_ack` per
reply, a paused reply ends on its own `pause_ack` and is **not** flagged
truncated, and a truncated round never executes its half-streamed tool call.
Those are `stream_chat`'s docstring-pinned behaviours; this task changes what a
chunk *is*, not what a pause *is*.

## Files it touches

| file | change |
|---|---|
| `src/core/chat.py` | edit — seam swap, `allowed` deleted, comments reworded |
| `src/core/models.py` | docstring line only |
| `tests/test_chat.py` | edit — re-pointed stubs, `FakeStream`, dict chunks |
| `tests/test_agent_modes.py` | edit — the one seam spy + its comments |

## Rules it must not break (quoted)

* AGENTS.md: *"a truncated chat reply ends as an error, which also stops a
  half-streamed tool call from being executed."*
* The in-file comment that pins the precedence: *"Cancel wins over pause
  (checked first, at every site)."* — unchanged text, unchanged order.
* SSE honesty contract (AGENTS.md "SSE is the UI's only source of truth"): *"a
  reply the user paused still ends on its own ack, keeping its partial turn."*
* *"token/s uses completion tokens only"* — the per-round usage ladder
  (`iteration_completion`, `usage_derived_completion`, the estimated-flag logic)
  is not this task's territory.
* The request carries *no `tools` key rather than `tools: []`* — the rule from
  `tests/test_chat.py` (step 13's spec row), now enforced one layer down in
  `llmkit.client`; both the chat test and the llmh-03 client test must be green.
* **Core owns logic, the API owns transport** (AGENTS.md): still no `src.api`.

## Acceptance (step 16)

```
pytest tests/test_chat.py tests/test_agent_modes.py tests/test_batch.py -q
pytest tests/test_llm.py tests/test_llmkit_client.py tests/test_write_surface_isolation.py -q
```
all green — the second line proves the benchmark path didn't regress while chat
moved.

## Live smoke (§6.3.2–6.3.5, only if `http://localhost:8080/v1` is up)

Same localhost-only rule as llmh-05; **never** `http://192.168.1.4:8080/v1`.

* thinking fills the reasoning panel from `reasoning_content` with
  `REASONING=medium`, and thinking never becomes answer text;
* Stop mid-answer frees the server's generation slot within a second or two
  (that is `ChatStream.close()` reaching what `close_llm_stream()` dug out of
  litellm);
* one WorkBench turn with an MCP tool: `tool_call`, `tool_result`, the pre-tool
  `usage` frame, exactly one `done`, token badge reading server usage;
* pause mid-reply: partial text stays in history, `pause_ack` closes the reply,
  not flagged truncated.

If the server or a hub is unavailable, record each skipped item in the done-file.
The re-pointed test suite is the gate; the smoke is the confirmation.

## Done — outcome (2026-11-05)

`src/core/chat.py` no longer knows about transport: `stream_completion(model=…,
server_url=…, messages=…, api_key=…, sampling=sampling_kwargs,
tools=tools if use_tools else None)`, bare model name, `allowed` /
`allowed_openai_params` and both litellm comments gone, `_collect_tool_call_deltas`'
attribute branch dropped (D2). The loop body, the `close_llm_stream` call sites,
the *"Cancel wins over pause (checked first, at every site)"* comment and order,
the per-round usage ladder and every `emit` are textually unchanged. No
`src.api`; pyright clean on `chat.py` and `llm.py`. `src/core/models.py`: the
`SamplingParams` docstring line only — `to_kwargs()` untouched, its own docstring
mention of the old library left for llmh-09's purge.

Step 15 confirmed by reading plus the file's own tests: a reply ends on exactly
one of `done` / `error` / `pause_ack`, a paused reply ends on its own `pause_ack`
and is never flagged truncated, and `test_pause_while_tool_calls_stream_never_runs_them`
and `test_cut_tool_call_is_not_executed` still prove the half-streamed tool call
is never executed.

### Two notes for the rest of the lot

* **The interim branches llmh-05 left behind are deleted here.** Its done-file
  says "llmh-06 must delete them", so `saw_finish_reason()` is
  `bool(getattr(gen, "saw_finish_reason", True))`, `close_llm_stream()` closes the
  stream's own `close()` only, and `_ATTR_MISSING` is gone. The lot's
  definition-of-done scan is clean now:
  `grep -rn --include=*.py "completion_stream|received_finish_reason|intermittent_finish_reason" src`
  → empty. The mapping-protocol reads stay (D2-compliant, pinned by
  `TestChunksAreReadAsMappings`); that class's docstring and `llm.py`'s read-path
  comment lost the *coexistence* justification this task ended and keep the
  `.get`-protocol one — no assertion in `tests/test_llm.py` was edited.
* **The no-`tools` rule moved down a layer, so four spy assertions moved with
  it.** `stream_chat` now always passes a `tools` **keyword** (`None` when the
  loop is off) and `llmkit.client` omits the body key, so the *request* rule is
  unchanged and still pinned in the body by
  `tests/test_llmkit_client.py::TestToolsKeyRule`. What changed is that a **seam
  spy** can no longer see a missing key: `test_a_tools_off_mode_reaches_the_llm_client_with_no_tools`
  (renamed from `…_reaches_litellm_with_no_tools`),
  `test_the_shipped_chat_mode_is_a_tools_off_mode`,
  `test_a_chat_mode_turn_sends_no_tools_and_keeps_the_project_block` and
  `test_the_mode_gate_survives_a_second_turn_of_the_same_conversation` now assert
  `not captured.get("tools")`, and `tests/test_mcp.py::test_no_tools_keeps_single_shot_shape`
  gates on `not kwargs.get("tools")`. The untouched route-level proof of the rule
  stays `test_both_flags_false_leaves_no_tools_key_at_all`, which asserts on
  `_chat_tool_kwargs` itself.

### Files touched beyond the four listed

`tests/test_batch.py` (one stub + one comment) and `tests/test_mcp.py` (two stubs
+ two comments) patch `src.core.chat.completion` too, so they had to follow the
seam — `test_batch.py` is in this task's own acceptance command and failed
otherwise. `tests/test_llm.py`: docstrings/comments only.

Acceptance, all green: `pytest tests/test_chat.py tests/test_agent_modes.py
tests/test_batch.py -q` → 267 passed; the second line plus the rest of the
llmkit/benchmark set (`test_llm.py test_llmkit_client.py test_llmkit_sse.py
test_llmkit_contract.py test_write_surface_isolation.py test_bench.py
test_config.py test_api.py test_models.py`) → 564 passed; `tests/test_mcp.py` →
76 passed. No full suite (that is llmh-10).

### Live smoke (§6.3.2–6.3.5) — done, localhost only

Against `http://localhost:8080/v1`, model `qwen3.8-flash-next-iq3_xxs`, key
`none`, `reasoning_effort=medium` (`logs/probe_llmh06/probe.py`, hand-run; the
suite stayed hermetic; `.env` and `.env.test` untouched):

* **thinking**: 30 frames — the one-shot `reasoning` flag, the `reasoning_delta`
  stream (87 chars), then the answer (2 chars) and one `done`. Reasoning filled
  the reasoning channel and never became answer text; `tokens=31`,
  `estimated=False`, 66 tok/s.
* **Stop mid-answer**: cancel armed mid-stream → no frames emitted, the stream
  object reports `closed: True`, and the very next request completed in 0.87 s
  (no phantom generation queueing), so `ChatStream.close()` reaches the socket.
* **tool round** (`stream_chat` with a stub executor — the loop's own contract,
  not the UI): `tool_call`, `tool_result`, the pre-tool `usage` frame ordered
  *before* the `tool_call`, exactly one terminal `done`, and the badge figure
  server-reported (`tokens=115`, `estimated=False`, `context=377`), 59 tok/s.
  The hub-and-browser half of §6.3.4 was **not** driven (no UI in this probe);
  the event contract it checks is the one above.
* **pause**: armed before the request → `['usage', 'pause_ack']`, no `done`, no
  `error`, no `truncated` flag; armed 3 answer chunks in → partial text kept
  (`'Paris\nTokyo'`), `usage` rode the ack, `pause_ack` was the last frame and
  nothing was flagged truncated.

No item was skipped for want of a server; the only sub-item not attempted is the
browser/MCP-hub wiring of §6.3.4, which is llmh-10's manual checklist.
`http://192.168.1.4:8080/v1` was not contacted — the probe hardcodes localhost
and the placeholder key.
