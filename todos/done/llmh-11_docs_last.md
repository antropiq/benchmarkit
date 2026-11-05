# llmh-11 — Docs last: `AGENTS.md`, the README, the spec's status

**Spec:** step 26, §4.4 rows `AGENTS.md` and `README.md`.
**Depends on:** llmh-09 (and ideally llmh-10's green suite — if 10 is still
open, this task may run, but its `AGENTS.md` edit must describe what 01–09
actually built, read off the code).
**Code changes:** none. This task only writes sentences.

## Why it is last

`AGENTS.md` is injected into **every WorkBench turn** (its own rule: *"The model
acts on it, so accuracy matters more than volume"*). A sentence written before
the code settles is a lie the model acts on. Write from the code as it is now,
not from the spec as it was hoped.

## Deliver this

1. **`AGENTS.md`**, *Key patterns* — the *One LLM layer* bullet, adjusted rather
   than replaced (keep "adding a model is configuration, never new code" and the
   verifier sentence). Two invariants get named, in the file's own style — one
   or two lines of net growth for the whole edit, no implementation detail:
   * the one seam that speaks HTTP to a model endpoint is `core/llmkit`;
   * sampling parameters reach the server **verbatim** (no whitelist), which is
     why an unknown sampler name is invisible rather than an error.
   Check the rest of the file for litellm-shaped sentences that are now false
   (the *Gotchas* bullet about the swallowed `ModelNotMappedError` / zero-cost
   registration describes machinery llmh-07 deleted — that bullet **goes**;
   keep any part of it still true, e.g. the client-debug note, only if it still
   describes reality). Honour the file's own rules: what the app is and how the
   pieces fit, no line numbers, no counts, no machine-specific setup, shorter is
   better, cut or shrink rather than append.
2. **`README.md`** — every litellm mention reworded: grep the file
   case-insensitively (`LiteLLM`, `litellm`) and handle each hit — install /
   optional-dependency text, the configuration section's description of what the
   `LLM_DEBUG` switch does (mirror the `_render_env` comment llmh-07 wrote), and
   any prose about the provider prefix or the debug flag. Also check the README's
   env-key table/template prose mentions `LLM_DEBUG` and not `LITELLM_DEBUG`
   (with a line on the legacy fallback if the README documents keys at that
   level of detail — it may not; do not add a section just for the fallback).
   The README owns setup/run/test commands — if it documented installing
   litellm, remove that; nothing new needs installing (`httpx` is already a
   dependency).
3. **`docs/in-house-llm-client-spec.md`** — a status line at the top, the way the
   other docs in this repo carry status (look at a `docs/windows-support/*` file
   for the house style): **landed by lot `llmh` (todos/llmh-…)**, and the §7
   open questions are answered (D6 renamed with fallback; D2 attribute branches
   deleted; D3 as designed; `extra` dropped). Do **not** rewrite the spec — it is
   the lot's authority and a historical record; annotate, don't revise. If
   llmh-05/06/10 recorded a live-server observation that bears on §7 item 3,
   append it there as evidence with its date.

## Files it touches

| file | change |
|---|---|
| `AGENTS.md` | *Key patterns* + the dead *Gotchas* bullet |
| `README.md` | litellm mentions, `LLM_DEBUG` wording |
| `docs/in-house-llm-client-spec.md` | status line + §7 answers, annotations only |

## Rules it must not break

* **AGENTS.md's own rules, quoted**: *"It states what the app is and how its
  pieces fit, not how to build it"*; *"No code details that move"*; *"Keep it
  short and current… Cut or shrink a paragraph when the code it described changes
  instead of appending a newer one beside it, and describe the app as it is
  now."* Net length must stay roughly flat. One idea per bullet. Name the
  invariant; if a test enforces it, name the test as a pointer.
* Do not restate the spec in `AGENTS.md`: no decision numbers, no module
  inventories, no wire-format detail.
* Do not touch `todos/**` (except moving this task's own file to `done/`),
  `docs/windows-support/**` or the code.
* The AGENTS.md text becomes part of every chat turn — no "recently migrated
  from litellm" framing anywhere in it. The app has an in-house client; the
  migration is history, and history belongs in the spec's status line.

## Acceptance

* `grep -rn -i "litellm" AGENTS.md README.md` → **empty** (only the spec keeps
  its historical mentions, plus its new status line).
* `AGENTS.md` read top-to-bottom still describes the app as it now is, and no
  sentence in it is longer or vaguer than it was.
* The spec carries a status line and its §7 answers.
* `pytest tests/test_config.py -q` still green (nothing under test reads prose —
  this is a paranoia check that no file was edited by accident).

## After this task

The lot is complete. A human runs `docs/in-house-llm-client-smoke-checklist.md`
(llmh-10's file) against the localhost server; that is the only step left, and
it is deliberately not a task.

## Done — what landed (2026-11-05)

Written off the code as it is now (`src/core/llmkit/`, `src/config.py`,
`tests/test_llmkit_contract.py`), not off the spec.

* **`AGENTS.md`** — *Key patterns*: the *One LLM layer* bullet adjusted in place:
  the calls now go through `core/llmkit`, named as the only seam that speaks HTTP
  to an OpenAI-compatible endpoint, and it sends the sampling parameters
  **verbatim** (no whitelist), so an unknown sampler name is invisible rather than
  an error, with `tests/test_llmkit_contract.py` as the pointer. "adding a model is
  configuration, never new code" and the verifier sentence kept verbatim. *Gotchas*:
  the **Local model names cannot be priced** bullet deleted — the cost registration
  and the swallowed mapping error it described went with D7 (grep: no
  `register_model` / `ModelNotMappedError` / zero-cost table left in `src`), and
  nothing else in it still described reality. Net: 250 → 249 lines, and the file
  names no module inventory, no wire format and no decision numbers.
* **`README.md`** — grep of the whole file (case-insensitive) found **no** litellm
  mention: it never documented installing it, the provider prefix or a debug flag.
  Its install/extras tables and its "One configuration, one streaming layer" prose
  already said "the same client against an OpenAI-compatible endpoint", which stays
  true. One line added to the `.env` key block — `#LLM_DEBUG=1`, with a
  column-aligned comment in the block's own voice, mirroring the `_render_env`
  comment llmh-07 wrote. No line on the `LITELLM_DEBUG` fallback: the README
  documents keys at this level of detail for no other key, and the task's own rule
  says not to add a section for it.
* **`docs/in-house-llm-client-spec.md`** — a `**Status**: **landed** by lot llmh …`
  block at the top in the house style of `docs/windows-support/*` (and
  `docs/tree-search-spec.md`): what landed, where §7's answers are, that the §6.3
  manual pass is a human step with its localhost-only rule, and that the document
  is annotated rather than revised — its litellm mentions in §1–§5 describe the
  state the lot started from. §7 gained a lead line plus one **Answered:** note per
  question (1 renamed with the fallback and `llmDebug`; 2 deleted, fixtures became
  dict cases; 3 as designed, with the two localhost observations it invited and
  their dates — llmh-02's `[DONE]`-carrying clean stream, 2026-11-04, and llmh-10's
  truncated-despite-`[DONE]` chain case, 2026-11-05 — stating that neither measures
  what an upstream cut omits; 4 dropped). Nothing above §7 was rewritten.

## Verification

* `grep -rn -i "litellm" AGENTS.md README.md` → **empty**; the spec keeps its
  historical mentions plus the new status line.
* `pytest tests/test_config.py -q` → 26 passed (paranoia check: no prose is read
  by a test, and nothing else was edited).
* `pytest tests/test_chat.py tests/test_agent_modes.py -q` → 223 passed, since
  `AGENTS.md` is injected into every chat turn and those modules assert on the
  injection's shape.
* No code file touched: `git diff --stat` over the task's own edits is `AGENTS.md`
  and `README.md`, plus the two spec annotations.
