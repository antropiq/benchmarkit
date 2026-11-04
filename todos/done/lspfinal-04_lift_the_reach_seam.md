# lspfinal-04 — drop the platform half of the LSP reach seam

The task the other three earn. On Windows, `lsp_reach_is_refused()` must stop
consulting `local_reach_is_bounded()` — the job-object teardown now bounds a
language server's tree, and the guardrail list now speaks this volume — so the
refusal becomes what it always claimed to be: an answer about a **missing
capability**, not about an operating system.

**Depends on: `lspfinal-01` AND `lspfinal-02`. Both, fully.** This task must not
run first, and the big picture's verified proof is the reason: with the current
`restricted-paths.txt`, lifting the seam makes a `workspaceRoot` of
`C:/Users/vgr/.ssh` or `C:/Windows` acceptable. If you are reading this lot out of
order and 01 or 02 is not in `todos/done/`, **stop** and report the ordering
rather than doing this task anyway.
Spec: `windows-support-target-B-connectors.md` B-D5's last paragraph, §6's
"`lists_are_meaningful` is true on both platforms — the test that says A's note is
no longer needed", and B1 §5's "the LSP hub on Windows, `restricted-paths.txt`
rework (B-D1's LSP half)" as the deferred piece.

## The change is small; the honesty around it is the work

`lsphub/discovery.py:lsp_reach_is_refused()` becomes the guardrail half only:

```
return not constants.guardrails_speak_here(restricted_lsp_roots() or None)
```

— which is exactly the shape `mcphub.discovery.mcp_reach_is_refused()` already has,
restoring the symmetry the two hubs are meant to keep (AGENTS.md: the two hubs
mirror each other). Then:

- **`local_reach_is_bounded()` is deleted, not left unused.** After this task no
  caller remains (`mcphub` stopped asking it in B1). A seam nobody asks about is a
  fact that will be quoted as if it were still load-bearing — its docstring is
  three paragraphs of rationale for a predicate nothing evaluates. Delete the
  function, its re-exports from `mcphub.constants` and `lsphub.constants`, and the
  docstring text that reads "this half still answers the **LSP hub**".
- **Rewrite `lsphub/constants.py`'s `LOCAL_REACH_REFUSAL` comment.** It currently
  justifies standing on the *tree* half; after this task it must justify standing
  on the guardrail half alone. The **string itself stays identical** —
  the symmetry test in both `test_mcp.py` and `test_lsp.py` asserts the two hubs
  share it, and it asserts the words `"bounded"` and `"target B"` appear. Do not
  reword it here; if "target B" is no longer the right pointer, that is
  `lspfinal-06`'s docs decision, made once for both hubs.

## The tests that must change — do not delete them, redirect them

These exist to pin the asymmetry this task removes. Each needs a real replacement
assertion, not a removal:

| test | today | after |
|---|---|---|
| `test_mcp.py::test_the_lsp_hub_still_asks_both_halves` | forcing the platform half refuses the LSP hub | delete it, and **keep the symmetry**: assert `lsp_reach_is_refused()` now asks only the guardrail half — forcing `guardrails_speak_here` to `False` refuses, forcing nothing does not |
| `test_mcp.py::test_the_platform_half_of_the_seam_is_posix_here` (`posix_behaviour`) | asserts the seam is `True` on Linux | goes with the deleted function |
| `test_lsp.py::test_the_seam_is_bounded_here_so_nothing_is_refused` (`posix_behaviour`) | the Linux answer | **the marker comes off**: this is now true on both platforms and must be asserted on both. If it cannot be, something in 01 is wrong |
| `test_lsp.py::test_its_guardrail_list_is_the_one_it_is_asked_about` (`posix_behaviour`) | a Windows-shaped list speaks nothing on a POSIX runner | this is the pair test §6 asks for: **un-mark it** and give it its Windows twin (a POSIX-shaped list must speak nothing on this runner). Two tests, one per platform, each asserting its own list shape is inert |
| the `autouse` fixture `_reach_seam_pinned_for_the_hub_tests` | patches the seam so the suite has a hub to exercise on Windows | **it should become dead code.** After 01 and this task, the real answers on Windows are the patched ones. Deleting it is the acceptance criterion — see below |

**The strongest single proof this lot worked:** delete that autouse fixture and
`pytest tests/test_lsp.py` still passes on Windows, with the same number of
collected tests. The fixture exists only because the app refused its own hub here;
when the refusal is gone the fixture is lying. If the suite needs it, the lift is
not real and this task is not done.

## And the reach-note cases

`TestLoadLspServers` has cases that patch `guardrails_speak_here` to `False` to get
the "its reach cannot be bounded" note. Those stay — the refusal is still a real
state (a deployment with an inert list must still be refused) — but re-read each
one and make sure it is forcing the **guardrail** half. Any case that forced the
platform half is now forcing a function that does not exist.

## Files

| file | change |
|---|---|
| `src/core/lsphub/discovery.py` | `lsp_reach_is_refused()` → guardrail half only; its docstring rewritten to say what the refusal now means (an *inert list*, not *this OS*) |
| `src/core/mcphub/process.py` | delete `local_reach_is_bounded()` and the rationale that only the LSP hub asked it |
| `src/core/mcphub/constants.py`, `src/core/lsphub/constants.py` | drop the re-export; rewrite `LOCAL_REACH_REFUSAL`'s comment (the string is unchanged) |
| `tests/test_mcp.py`, `tests/test_lsp.py` | the table above, plus the fixture deletion |

## Acceptance

- On **this** machine:
  - `lsp_reach_is_refused()` is **`False`** and `load_lsp_servers()` reports **no
    reach note** for `pyright`, `typescript`, `vue`. This is the first point in the
    lot where the servers stop being refused. (They may still fail to *start* —
    that is 03 and 05, and it must be a spawn error naming the reason, not the
    reach refusal.)
  - a declaration whose `workspaceRoot` is on the deny-list is still skipped, and a
    still-inert list still produces the refusal sentence — prove the guardrail half
    bites by pointing `lsphub.constants.LSP_RESTRICTED_PATHS` at a file of POSIX
    entries and asserting the refusal here, on Windows.
- On **Linux**: unchanged verdicts, and the two `posix_behaviour` markers listed
  above are **removed** — a marker that hides a now-cross-platform assertion is a
  new skip in all but name.
- `local_reach_is_bounded` appears **nowhere** in `src/` or `tests/` when you grep
  for it.
- `pytest -q` green with **no new skips**; the autouse seam fixture is gone.
- Do not weaken `guardrails_speak_here` to make the above pass. It answers "could
  these prefixes speak this path form"; an empty list answering `True` is a
  deliberate declaration (no prefixes is not a list that stopped matching) and the
  docstring says so. If you find yourself wanting it to answer `False` for an empty
  list, that is a change to a stated contract — stop and report it.
- Move this file to `todos/done/` when the above is true.
