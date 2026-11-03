# lspfinal-06 — the Windows sweep, the Linux regression run, and the docs that are now lies

The lot's last task is the only one that can prove the guarantee the whole thing
exists for, and it is deliberately last: a manual Windows run is the *only* evidence
that no process survived. §6 of the spec says so in terms — "no unit test on Linux
can make it".

Depends on: **all of 01–05.** Spec: `windows-support-target-B-connectors.md` §6
(Verification) and §7 (What B must not break), plus the AGENTS.md clause discussed
below.

## 1. The manual Windows sweep (do this first, report it as evidence)

Run the app on this machine and record what actually happened, not what the code
says should happen:

1. **Language-server check.** All three declared servers reach a state that is not
   `error`, and each answers at least one real tool: a `diagnostics` call on a file
   with a deliberate error, one `hover`, one `definition`. `typescript` and `vue` are
   `"lazy": true` in this workspace's file — so they spawn on the first tool call,
   and a settings page that shows them as declared-but-not-running is **not**
   evidence of failure. Say which state you read.
2. **A benchmark run** and **one WorkBench turn** with `fedit` and `console` both
   enabled — the spec asks for these because the same guardrail file feeds the
   connectors, and task 01 rewrote it.
3. **Quit the app and check Task Manager.** No `node`, no `pyright`, no
   `vue-language-server`, no connector process survives. This is B-D5's whole point
   and the one step a CI run cannot substitute.

Record the Task-Manager observation as a number of surviving processes per name,
not "looks clean". If anything survives, the sweep has **failed** — write that in
the report and leave this task file in `todos/`; do not move a file whose acceptance
was not met.

## 2. The regression runs — here, and on Linux

§6's cadence, per the spec and the big picture's contract 1:

Run all three **on this machine** first:

- `pytest -q` — green, **zero new skips**;
- `cd web && npm test` — green;
- `cd web && npm run test:e2e` — green wherever `.env.test` reaches an endpoint.

§6 asks for the same three on **Linux** every session. This machine is Windows, so
the honest thing is to run them where they can be run and say which you ran; do not
report a Linux result you did not observe on a Linux runner.

Baselines measured on this machine at the time of writing (so a regression is countable, not
argued): `tests/test_lsp.py` **88 passed, 5 skipped**;
`tests/test_proctree.py + tests/test_path_policy.py` **64 passed, 4 skipped**. A new
skip is a defect in this lot, not a fact about Windows. If the e2e suite cannot run
here (`.env.test` endpoint unreachable, or win-08's browser-path note), say which
check was not run and why — never report a green you did not observe.

## 3. The docs — check them, do not assume they need rewriting

**AGENTS.md's process-tree bullet is already in the platform-neutral form** §7 asks
for: *"Teardown kills process \*trees\*, not wrappers, on either platform and by
either mechanism, so children a hub or the terminal spawned of their own never
outlive the app that started them."* Target C landed it. So **the expected outcome
here is: verify the sentence is now true of the LSP hub, and change nothing.** Only
edit if the sentence is false — and if it is false, that means task 02 did not land,
which is a finding, not a docs fix.

What the AGENTS.md rules actually require of this lot:

- The **deny-list bullet** ("each connector's reach is bounded by its own
  deny-list") is unchanged in meaning and now simply true on both platforms — no
  edit. §7's suggestion of "on both platforms' path forms" is optional wording; the
  bullet is already unconditional, and AGENTS.md's own rule is to cut what the code
  already says rather than append.
- **`LOCAL_REACH_REFUSAL`'s "(see target B)"** pointer. The two hubs still refuse in
  one shared sentence (a test asserts it). Decide once, for both: does that pointer
  still mean something to a reader, now that B is finished? If it becomes stale, the
  change is one string in two `constants.py` files **and** the assertion in both
  symmetry tests — all in this task, together. Do not leave half the sentence
  pointing at a closed target.
- **README.** The LSP section needs the sentence win-08 deferred: what a Windows
  declaration looks like (the form task 05 recorded), and that
  `lsp_connectors/restricted-paths.txt` now carries both path forms. README:229-230
  already documents that the working location is refused when it sits on a prefix
  from that file — re-read it against task 01's list and correct it if the wording
  assumed POSIX entries.
- **The spec files gain a status line**, the way B1 and C did ("Status: …
  implementation tracked in §9"): mark B's LSP half landed with its task numbers, so
  the next reader does not reopen this lot. Update the B-D1/D5/D6 rows in §4 and the
  "not wired yet" sentence in B-D5's *Landed* paragraph — that paragraph will
  otherwise keep telling the next agent that the LSP refusal stands.
- Nothing else in AGENTS.md moves, and it stays under ~300 lines.

## 4. Report what this lot did **not** do

Name the known remaining gap so it does not get lost, and do not fix it here:

- the **MCP hub's** children are still not inside a job on Windows. B1 lifted the
  MCP refusal on the argument that a connector's children are its own short-lived
  direct children; a connector that spawns a grandchild can still leak one. The
  machinery exists (`ProcessJob`) and the LSP hub now uses it — the connector side
  is its own task.
- `console_mcp`'s own `shutil.which` resolution (B-D6's rule, connector side).

Both belong in a follow-up lot; the big picture lists them as out of scope.

## Acceptance

- The three servers answer real LSP tools on Windows, and **zero** language-server
  or connector processes survive app exit — with the observed numbers recorded.
- `pytest -q`, `cd web && npm test` green; `test:e2e` green or its absence
  explained. Skip counts no higher than the baseline above.
- AGENTS.md unchanged **or** a one-line justification in the report for the line
  that was changed.
- The B spec's status and the stale "not wired yet" paragraph are corrected.
- `git diff --stat` for this task touches only `README.md`,
  `docs/windows-support/windows-support-target-B-connectors.md`, and (if the pointer
  changed) the two `constants.py` files and the two symmetry tests.
- Move this file to `todos/done/` **only** when the Task-Manager check passed.
