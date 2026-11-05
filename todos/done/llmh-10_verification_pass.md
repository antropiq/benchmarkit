# llmh-10 — Verification pass: full suite, static scans, the manual list

**Spec:** Phase 6 (§6.1, §6.2, §6.3, and §6.4's rollback note as a risk check).
**Depends on:** llmh-09. **Code changes:** only what a failure sends you back
to — this task's own edits are the one new test and the written-down manual
checklist.

## Deliver this

1. **The full suite** (§6.1), the one place this lot runs everything:

   ```
   pytest
   cd web && npm test
   ```

   Green with litellm uninstalled. A failure names its owning task
   (05/06 = seams, 07/08 = the switch, 09 = the purge); fix it in the spirit of
   that task's file and re-run the affected files before the full pass again.

2. **The §6.2 static scans**, each run and its output recorded in the done-file:
   * `grep -rn "litellm" src tests web/src pyproject.toml` → only llmh-09's
     allowed survivors;
   * `grep -rn "completion_stream\|received_finish_reason\|intermittent_finish_reason" src`
     → empty;
   * `grep -rn "httpx" src/core` → only `src/core/llmkit/`
     (plus any pre-existing `mcphub` remote-connector use, noted as pre-existing).

3. **One new automated test — the truncation end-to-end through the real
   transport** (spec §6.3.6: *"the one check that must be repeated after any
   change to `sse.py`"*; the manual proxy trick stays manual, but the seam is
   testable hermetically): in `tests/test_llmkit_client.py` (or a small new
   class in it), a `MockTransport` streaming body that carries content chunks
   and then **ends without ever sending a `finish_reason` frame** (cut socket —
   no `[DONE]`, and also a variant **with** `[DONE]`, which must behave the
   same, D3). Feed that `ChatStream` to `call_llm_and_stream` with a stubbed
   verifier spy and assert: `LLMResult.truncated` is `True`, the
   `TRUNCATION_MESSAGE`-carrying `run_error` was emitted, and the verifier was
   **never called** — the AGENTS.md invariant, now enforced across the real
   parse→record→judge chain rather than per-layer.

4. **The §6.3 manual checklist, written down as a human-only step.** Create
   `docs/in-house-llm-client-smoke-checklist.md`: the eight §6.3 items in the
   spec's own wording, each with its expected observation, headed by the
   endpoint rule — **against `http://localhost:8080/v1` with model
   `qwen3.8-flash-next-iq3_xxs`; never `http://192.168.1.4:8080/v1`** — plus a
   line saying item 6 (truncation via a connection-dropping proxy) is the one
   that needs a second terminal, and item 8 the settings round-trip. No task in
   this lot attempts the list; llmh-05/06 already recorded whatever live smoke
   was feasible on their turn — carry their notes in.

5. **Rollback sanity read** (§6.4): confirm from `git diff --stat` that the two
   live-path files (`src/core/llm.py`, `src/core/chat.py`) each form a
   self-contained revert — reverting either one alone still leaves the *other*
   path and the suite working (helpers shared by name). If that is no longer
   true (e.g. chat now relies on an llmh-05 change inside `llm.py` that litellm
   mode can't satisfy), record it as a finding; do not restructure.

## Files it touches

| file | change |
|---|---|
| `tests/test_llmkit_client.py` | the truncation end-to-end test (new class) |
| `docs/in-house-llm-client-smoke-checklist.md` | **new** — §6.3 as a human-only list |
| anything a red test points at | the minimum the owning task's file would have done |

## Rules it must not break

* The new test stays **hermetic** (AGENTS.md): `MockTransport`, no sockets, no
  `.env` reads; a timeout or limit it needs comes from `llmkit.constants`.
* If the truncation test contradicts a decision recorded in the spec (§7 item 3
  about `[DONE]`), the **chunk-level rule wins** (D3): `[DONE]` without a
  finish-reason chunk is still truncation. Do not soften the test to match an
  observation; record the observation in the spec's §7 item 3 as evidence, which
  is what that section invited.
* Do not run the manual checklist. Do not point anything at
  `http://192.168.1.4:8080/v1`. Ever.

## Acceptance

* `pytest` — full suite, green.
* `cd web && npm test` — green.
* The three §6.2 scans, with output, in the done-file.
* The checklist file exists and matches §6.3's eight items and endpoint rule.
* The done-file states, in one line each: full-suite result, scan results,
  rollback-sanity verdict.

## Notes for the reader

This is the lot's gate. If the full suite reveals two tasks disagree about the
seam (e.g. a stub from 05 vs a stub from 06 shape `close()` differently), fix
`tests/llmkit_fakes.py` toward the **real** `ChatStream` behaviour, not the
convenient one — the fake exists so the two are held to one interface.

## Done — the gate, in one line each

* **Full suite** — `pytest` → **2343 passed, 29 skipped** (58 s) with litellm
  uninstalled; `cd web && npm test` → **494 passed, 36 files**. Nothing needed
  sending back to another task: no seam disagreement surfaced, so
  `tests/llmkit_fakes.py` is unchanged.
* **§6.2 scans** — all three clean, output below.
* **Rollback sanity (§6.4)** — holds, with one honest caveat recorded below.

## Done — what landed

* **`tests/test_llmkit_client.py`** — new class
  `TestTruncationReachesTheVerdictThroughTheRealParse`: a `MockTransport` body of
  content chunks that ends **without ever sending a `finish_reason` frame**, in
  both shapes (cut socket, and the same body with `data: [DONE]` appended —
  which behaves identically, D3), fed through the **real** chain: `sse` parse →
  `ChatStream._record` → `call_llm_and_stream` → `process_question`. Asserted:
  `LLMResult.truncated` is `True`, exactly one `run_error` carrying
  `TRUNCATION_MESSAGE`, the fragment still streamed as `token` events, the
  verifier spy **never called**, the question in `job.get_failed_questions()`,
  and `answer_done.truncated` True. A third case serves as the negative control:
  the same chain with a `finish_reason` chunk is *not* truncated and keeps the
  server's own usage (`completion_tokens` 4, `prompt_tokens` 10,
  `tokens_estimated` False), so the truncation cases cannot be passing because
  everything reads as cut.
  The file's `_Wire` fixture grew no new capability — `serve_stream(chunks, done=…)`
  already served both shapes — and the test needed no timeout or body limit, so
  it patches nothing in `llmkit.constants`. Hermetic as before: `MockTransport`,
  no sockets, no `.env`.
  **Teeth shown, not assumed**: poisoning `ChatStream._record` to always claim a
  finish reason made 4 of the 5 new cases fail (including the verifier spy
  recording the fragment it must never see); the poison was reverted and the
  file is byte-identical to before it.
* **`docs/in-house-llm-client-smoke-checklist.md`** — **new**. The eight §6.3
  items in the spec's wording, each with an *Expected* line, headed by the
  endpoint rule (localhost + `qwen3.8-flash-next-iq3_xxs`, never
  `192.168.1.4`), item 6 marked as the one needing a second terminal and item 8
  as the settings round-trip, and the live notes llmh-05/06 recorded carried in
  (items 1–5, with their numbers). It also states plainly that items 6–8 were
  never driven live and what their plumbing was proven by instead.
  No task ran the list; no request left this machine on this task's turn.

## Verification — the §6.2 scans

1. `grep -rn "litellm" src tests web/src pyproject.toml --exclude-dir=__pycache__`

   ```
   src/config.py:115:# The name this switch had while the calls went through litellm. It lives here
   tests/test_agent_modes.py:874:        forbidden = ("fastapi", "litellm", "src.api", "mcphub", "lsphub", "termhub",
   tests/test_agent_modes.py:1334:        forbidden = ("fastapi", "litellm", "src.api", "mcphub", "lsphub",
   ```

   Exactly llmh-09's allowed survivors. Case-insensitive, the same scan adds the
   two it pins: `src/config.py:119` (`LEGACY_LLM_DEBUG_KEY = "LITELLM_DEBUG"`)
   and `tests/test_config.py:224`. Over `README.md`, `.env.example`, `.env.test`
   and `.env.test.example` the scan is **empty**.
2. `grep -rn "completion_stream\|received_finish_reason\|intermittent_finish_reason" src`
   → **empty** (exit 1): the litellm-shaped workarounds are gone, not renamed.
3. `grep -rn "httpx" src/core` → model-facing HTTP only under `core/llmkit/`
   (`client.py` import + two type-annotated helpers and one comment in
   `constants.py`). The one hit outside it is **pre-existing** and untouched by
   this lot: `mcphub/manager.py` reaching `mcp.shared._httpx_utils` and falling
   back to `httpx2` for a remote connector's client. `lsphub` and `editorfs`: no
   hits. No new home for it was added.

Also confirmed: the full suite ran with `python -c "import litellm"` raising
`ModuleNotFoundError`, and `pip list` carries no `litellm`.

## Rollback sanity (§6.4)

`git diff --stat` puts each live path in exactly one file: `src/core/llm.py`
(206 changed lines) and `src/core/chat.py` (48). Chat imports its shared helpers
from `llm` **by name** — `Emit`, `TRUNCATION_EVENT_FLAG`, `TRUNCATION_MESSAGE`,
`_extract_reasoning`, `_extract_usage`, `close_llm_stream`,
`safe_get_choice_content`, `saw_finish_reason` — and that import list is
**identical to HEAD's**, so reverting `llm.py` alone hands them all back (HEAD
has every one, dual-shape versions) while `chat.py` keeps its own
`llmkit.stream_completion` call. Reverting `chat.py` alone restores
`from litellm import completion`, which is still installed at that point in the
history. Neither revert strands the other path. **Verdict: each file is still a
self-contained revert point.**

One caveat worth stating rather than hiding: a *single-file* revert is source-
compatible but not green on tests — `tests/test_chat.py` and `tests/test_llm.py`
were re-pointed at the seam in the same phases, so reverting a live file leaves
its own test module asserting against the other transport. §6.4's promise is that
the *app* runs and the *other* path keeps working, which holds; no restructure is
warranted.

## Findings

None that sent this task back to another task's file. The two stub machineries
(05's benchmark doubles and 06's chat doubles) agree on the seam's shape — the
full suite passing first time is the evidence — so `tests/llmkit_fakes.py` needed
no correction toward the real `ChatStream`.

