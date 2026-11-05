# Live smoke checklist — the in-house LLM client

The human-only half of this client's verification (spec
`in-house-llm-client-spec.md` §6.3). Everything the suite can prove hermetically
it already proves through `httpx.MockTransport`; what is listed here needs a
running model server and a pair of eyes. **No automated task runs this list.**
Re-run it after any change to `src/core/llmkit/`, and item 6 above all — it is
the one check that must be repeated after any change to `sse.py`.

## Endpoint rule

Run every item against **`http://localhost:8080/v1`**, model
**`qwen3.8-flash-next-iq3_xxs`**. **Never `http://192.168.1.4:8080/v1`** — that
is a developer endpoint: not a test target, not a fallback, not a probe.

A **test run** takes its endpoint from `.env.test`, never from a developer's
`.env`. So item 1, a real benchmark run, points the app at localhost through
**Settings** — which writes `.env` from the template — instead of editing
`.env.test`. The probe scripts the migration used hardcode localhost and the
placeholder key for exactly this reason.

## The eight items

1. **Benchmark, two questions, verifier on.** Tokens stream into the card;
   `tok/s` and `ttft` land in the same band as the pre-migration run of the same
   set; `consumed` equals prompt + completion; the verdicts match the baseline. A
   verdict that flips means the request body changed — re-read spec §2.1 before
   doubting the model.
   *Expected:* no `run_error` line, `tokensEstimated` false, verdicts identical to
   the baseline run of the same two questions.

2. **Thinking visible.** With `REASONING=medium`, the reasoning panel fills from
   `reasoning_content`, and reasoning text is never answer text or part of the
   message sent back to the model.
   *Expected:* the reasoning pane fills while the answer pane stays empty until
   the first answer delta, and the next turn's assistant message carries none of
   that reasoning text.

3. **Stop mid-answer.** Run → Stop, and the server's generation slot is free
   within a second or two — visible as the next request not queueing behind a
   phantom. That is `ChatStream.close()` doing what `close_llm_stream()` did.
   *Expected:* the following request generates immediately rather than waiting
   behind the abandoned one.

4. **WorkBench with a tool.** One turn that calls an MCP connector and then
   answers: `tool_call`, `tool_result`, the pre-tool `usage` frame and exactly one
   `done`, with the token badge reading the server's usage rather than the
   estimate.
   *Expected:* those frames in that order, one terminal `done`, the badge not
   flagged as estimated. The hub-and-browser wiring of this item is the part no
   unit test reaches.

5. **Pause mid-reply.** The partial text stays in history, `pause_ack` closes the
   reply, and it is **not** flagged truncated.
   *Expected:* the partial answer stays in the transcript, `pause_ack` is the last
   frame, no `done`, no `error`, no truncation marker on the message.

6. **Truncation path.** Point the endpoint at a server that closes the stream
   without ever sending a finish reason (a proxy that drops the connection works).
   Expected: the run fails that question and never sends the fragment to the
   verifier; WorkBench marks the message with the truncation error. This is D3
   earning its keep, and it is the one check that must be repeated after any
   change to `sse.py`.
   *Expected:* the partial text shown, the run's error line carrying the
   truncation sentence, the question in the failed list, and no verifier call for
   it. **This is the one item that needs a second terminal** — for the
   connection-dropping proxy, which cannot be driven from the app itself. Its
   transport half is automated: `tests/test_llmkit_client.py`,
   `TestTruncationReachesTheVerdictThroughTheRealParse`; what the proxy adds is a
   real socket dying mid-write.

7. **Debug switch.** Tick "Log LLM request internals", call a bad model name, and
   read one legible `404 … no router for requested model` line plus the payload on
   the console; untick it and confirm the console is quiet again.
   *Expected:* the status and the upstream message on one line, the JSON payload
   below it, no `Authorization` header and no key in either; a quiet console once
   unticked.

8. **Settings round-trip.** Save the dialog and confirm `.env` is regenerated from
   the template with `#LLM_DEBUG=` commented out and no other managed key lost.
   *Expected:* the fixed key set in its fixed order, empty optional keys commented,
   `#LLM_DEBUG=` among them. This one checks the dialog and the template, not the
   model, so it needs no live server.

## What the migration already observed live

Carried from `todos/done/llmh-05_move_the_benchmark_across.md` and
`todos/done/llmh-06_move_the_chat_across.md`, both against localhost with the
placeholder key and `reasoning_effort=medium` (`logs/probe_llmh05/probe.py`,
`logs/probe_llmh06/probe.py`, hand-run; the suite stayed hermetic and `.env` /
`.env.test` were not edited):

* **1** — 99 and 121 `token` events, no `run_error`, no truncation; `ttft`
  0.92 s / 0.63 s; 55.5 / 60.3 tok/s, and the same band across three repeats of
  the same two questions (55.7, 60.8, 63.5); `consumed` = prompt + completion
  (68+233=301, 46+631=677), server-reported, not estimated; both verdicts `true`
  and unchanged across the repeats. The body carried `top_k`, `min_p`,
  `repeat_penalty` and `reasoning_effort` verbatim and the model name bare.
* **2** — 30 frames: the one-shot `reasoning` flag, an 87-char `reasoning_delta`
  stream, then the answer; reasoning filled the reasoning channel and never
  became answer text.
* **3** — cancel armed mid-stream emits no frames, the stream reports
  `closed: True`, and the very next request completed in 0.87 s.
* **4** — `tool_call`, `tool_result`, the pre-tool `usage` frame ordered *before*
  the `tool_call`, exactly one terminal `done`, badge server-reported
  (`tokens=115`, not estimated). The hub-and-browser half was not driven.
* **5** — pause armed before the request: `['usage', 'pause_ack']`, no `done`, no
  `error`, nothing flagged truncated.

Items **6**, **7** and **8** were never driven live by any task in the lot. Their
plumbing was proven off the wire instead, so what is left here is the console and
the browser, not the mechanism: llmh-07 ran the §6.3.8 round-trip through
`save_settings` on a *temporary copy* of `.env` (`#LLM_DEBUG=` back out commented,
no managed key lost, legacy name gone) and showed `_log_request` writing the POST
line and the JSON body with the key absent; llmh-08 asserted the ticked box
survives a reload through `GET /api/settings`. Item 6's transport half is the new
automated case named above. A run of these three against a real server and a real
dialog is still owed by a human.
