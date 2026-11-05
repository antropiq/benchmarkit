# llmh-09 — Drop `litellm` from the project and purge the mentions

**Spec:** Phase 5 (steps 23–25), §6.2.
**Depends on:** llmh-05, llmh-06, llmh-07, llmh-08 — both live paths must already
be on `llmkit`. Do not run this task if any of them is still open (§6.4).
**Leaves the app working:** after this task `litellm` is not installed and
nothing imports it. If the suite turns up an importer, that is a bug in an
earlier task — report it rather than re-adding the dependency.

## Deliver this

1. **`pyproject.toml`**: remove `"litellm"` from `dependencies`. `httpx` is
   already listed (both as a runtime and a `test` extra dependency) — leave it.
   Then reinstall: `pip install -e ".[mcp,lsp,test]"` (or the project venv's
   equivalent per README). Nothing else in the tree needs litellm; the hubs pull
   `mcp` and `pygls`.
2. **Purge the remaining mentions** (step 24). Scan:
   `grep -rn "litellm" src tests web/src pyproject.toml README.md .env.example
   .env.test .env.test.example` — note grep sees source only; also confirm the
   committed env examples carry no `LITELLM_DEBUG` line except nothing (llmh-07
   renamed them). Allowed survivors, exactly:
   * `src/config.py` — the `LITELLM_DEBUG` fallback string and its comment;
   * `tests/test_config.py` — the test that pins the fallback (its name and
     docstring may mention the old key; that is the point of the test);
   * `tests/test_agent_modes.py` — `"litellm"` inside the two forbidden-import
     tuples, which **keep** it (step 25) so a core loader stays free of the old
     transport too;
   * `src/core/mcphub/formatting.py` — reword its one docstring mention to
     "handed to the LLM client" (spec §4.4 "Left alone on purpose" list keeps the
     file untouched otherwise).
   Anything else that surfaces is yours to fix — usually a docstring or a stale
   test comment.
3. **Step 25**: add `"llmkit"` to **both** forbidden-import tuples in
   `tests/test_agent_modes.py` (`test_the_loader_imports_no_transport_no_model_and_no_hub`
   and its sibling — find them by the tuple containing `"fastapi"`, `"litellm"`).
   Read each test's assertion first: they assert a core loader module's imports;
   `"llmkit"` joins the sentence *"a core loader must stay free of the LLM
   transport whatever that transport is called."* The agent-modes loader must
   genuinely not import `llmkit` — if the test now fails, the loader grew an
   import it should not have; fix the loader, not the test.
4. Re-run the §6.2 static scans and record their output in your done-file:
   * `grep -rn "litellm" src tests web/src pyproject.toml` → only the survivors
     above;
   * `grep -rn "completion_stream\|received_finish_reason\|intermittent_finish_reason" src`
     → **empty** (the workarounds are gone, not renamed);
   * `grep -rn "httpx" src/core` → model-facing HTTP only under `core/llmkit`
     (if `mcphub` legitimately imports httpx for a remote connector, note it —
     the scan's point is that this lot added no new home for it).
   Also confirm no leftover import anywhere: `grep -rn "^import litellm\|^from
   litellm\|import litellm" src tests` → empty.

## Files it touches

| file | change |
|---|---|
| `pyproject.toml` | `litellm` out of `dependencies` |
| `tests/test_agent_modes.py` | `"llmkit"` added to the two forbidden tuples (and nothing else) |
| `src/core/mcphub/formatting.py` | one docstring reword |
| whatever else the scan surfaces | docstrings/comments only — if a *code* file still imports litellm, stop and report |

## Rules it must not break

* **Never** touch `todos/done/**`, `.kilo/plans/**`, `logs/**` or
  `web/dist/**` — historical records and build artifacts, litellm mentions and
  all (spec §4.4 "Left alone on purpose").
* The suite stays **hermetic** (AGENTS.md) — this task must not make any test
  depend on a live endpoint in exchange for the removed dependency.
* Do not "helpfully" uninstall other packages, bump versions, or re-lock
  anything beyond what `pip install -e` does on its own.
* The forbidden-import tuples guard a real invariant (AGENTS.md: *"Tools are
  read-only by design"* / core-owns-logic). Add to them; never reorder or
  reformat them.

## Acceptance

```
pip show litellm        # not installed / not required by this project
pytest tests/test_llmkit_contract.py tests/test_llmkit_sse.py tests/test_llmkit_client.py \
       tests/test_write_surface_isolation.py tests/test_llm.py tests/test_chat.py \
       tests/test_bench.py tests/test_agent_modes.py tests/test_batch.py \
       tests/test_config.py tests/test_api.py -q
```
all green with litellm uninstalled. (Full suite still waits for llmh-10.)

## Done — what landed

* **`pyproject.toml`** — `"litellm"` out of `dependencies`; `httpx` left where it
  was (runtime *and* `test` extra). `pip install -e ".[mcp,lsp,test]"` rebuilt the
  editable install, after which `pip show litellm` reported `Required-by:` empty —
  nothing required it any more — so `pip uninstall -y litellm` removed it cleanly
  and `pip check` says *No broken requirements found*. No other package was
  touched, no version bumped, nothing re-locked.
* **`tests/test_agent_modes.py`** — `"llmkit"` appended to **both** forbidden
  tuples (the loader's and `defaults`'), `"litellm"` kept in place in each, no
  reordering; the loader test's one-line docstring now says "the LLM transport"
  since the tuple names every transport, whatever it is called. Both tests pass
  unmodified in logic: neither `core/agent_modes` nor `core/defaults` imports the
  new seam.
* **`src/core/mcphub/formatting.py`** — the docstring now reads "definitions
  handed to the LLM client"; nothing else in the file changed (§4.4 leaves it
  alone otherwise).
* **Two stale mentions the scan surfaced**, docstring/comment only:
  `src/core/models.py::to_kwargs` ("as litellm kwargs" → "as request-body
  kwargs") and the `tests/test_llmkit_contract.py` header above the §2.1 body
  fixture (which claimed "the exact body litellm puts on the wire today" — the
  fixture measures the wire, not a library).
* **Not touched** (allowed survivors): the `LITELLM_DEBUG` fallback key and its
  comment in `src/config.py`, and `tests/test_config.py`'s assertion that pins it.
  `todos/done/**`, `.kilo/plans/**`, `logs/**` and the spec document itself keep
  their litellm sentences exactly where they are.
* The working `.env` still carries hand-written `#LITELLM_DEBUG=` lines; the next
  settings save rewrites that file from the template, which llmh-07 already made
  emit `#LLM_DEBUG=` — so it self-heals and is not a tracked file.

## Verification — the §6.2 scans

* `grep -rn "litellm" src tests web/src pyproject.toml --exclude-dir=__pycache__`
  → the three survivors only: `src/config.py:115` (the fallback's comment) and the
  two `tests/test_agent_modes.py` tuples. Case-insensitive the same scan adds
  `src/config.py:119` (`LEGACY_LLM_DEBUG_KEY = "LITELLM_DEBUG"`) and
  `tests/test_config.py:224` (the test that pins it). The same scan over `README.md`,
  `.env.example`, `.env.test` and `.env.test.example` is **empty**.
* `grep -rn "completion_stream\|received_finish_reason\|intermittent_finish_reason" src`
  → empty: the workarounds are gone, not renamed.
* `grep -rn "httpx" src/core` → model-facing HTTP only under `core/llmkit`
  (`client.py` + one comment in `constants.py`). The one hit outside it is
  pre-existing and untouched by this lot: `mcphub/manager.py` falls back to
  `httpx2` (the MCP SDK's own httpx fork) for a remote connector's client. This
  task added no new home for it.
* `grep -rn "^import litellm\|^from litellm\|import litellm" src tests` → empty,
  and `python -c "import src.api.server, src.core.chat, src.core.bench"` imports
  fine with litellm uninstalled.

* The acceptance command, run **after** the uninstall:
  `pytest tests/test_llmkit_contract.py tests/test_llmkit_sse.py
  tests/test_llmkit_client.py tests/test_write_surface_isolation.py
  tests/test_llm.py tests/test_chat.py tests/test_bench.py
  tests/test_agent_modes.py tests/test_batch.py tests/test_config.py
  tests/test_api.py -q` → **529 passed**. No test gained a live endpoint.

## If the reinstall fails or something still imports litellm

Do **not** re-add the dependency to `pyproject.toml` to make it pass. Find the
importer (`python -c "import src.api.server"` plus the traceback names it), fix
it if it's a leftover mention, or record the task number that should have caught
it and leave this task's file in the lot folder (not `done/`) with a one-line
note. The batch runner will relaunch it; that is the honest outcome.
