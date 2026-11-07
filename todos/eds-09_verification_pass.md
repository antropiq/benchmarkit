# eds-09 — Verification pass: what an individual task could not see

**Spec:** §5 (Acceptance) in full.
**Depends on:** eds-01 … eds-08 — every one of them in `todos/done/`.
**Code changes:** only what a check below proves wrong. This task's default is to
report, not to redesign; a fix it makes is a fix to the thing the check named.

Work the acceptance list against the tree as it is now, and record for each item
how you established it (a test name, a command, a `curl`, a grep). Anything that
cannot be established is a finding to fix in this task or a sentence to report.

## The acceptance list, as checks

1. The pencil sits beside the mode dropdown and opens the dialog; the dialog shows
   one tab per mode the **file** declares, in the file's order, plus Common; the
   app-owned mode appears nowhere in it.
2. Changing a Label changes that tab's caption immediately.
3. **Create agent** adds a tab, focuses it, and writes nothing. Cancel leaves the
   file byte-identical.
4. Save writes the document, and the dropdown reloads: the new mode is selectable,
   the selection is the newly set default, and the file on disk holds what the
   form showed — including a persona that round-trips through the loader with no
   findings.
5. A file with a broken entry shows that entry and its finding, and cannot be
   saved until it is fixed or removed. **No entry is ever dropped silently.**
6. A file that is not valid JSON is listed and not overwritten; Save is disabled.
7. A save against a file that changed on disk since the read is refused, the
   dialog stays open, and the bytes on disk are the ones the user put there.
8. The previous content is recoverable from the app's backup folder after any save.
9. No connector tool, LSP spec, chat path or model-client URL builder can reach
   the save route; the save is refused while a reply streams and the pencil is
   inert while a lot runs.
10. `web/src` contains no reserved mode id.

## The cross-task checks (the ones no single task could make)

* **The two reads really are two.** `GET /api/agent-modes` still publishes no
  persona and no generated blocks, still has one verb and zero arguments, and
  still answers 200 with the defaults for a broken file. The document read still
  shows the broken entry. Grep the reader's payload test and run it.
* **A save the loader would refuse is impossible.** Save a document through the
  route, then read the file back with `resolve_modes()`: zero findings. If any
  finding appears, the writer wrote a file its own reader refuses — the round-trip
  bug D6 exists to prevent.
* **The loader's leniency survived.** A hand-broken modes file still chats: the
  modes lot's D11 tests are green and `resolve_modes()` never raises.
* **The benchmark is untouched.** `test_the_benchmark_never_consults_a_mode` green;
  no `agent_modes` import reached `core/bench.py`, `core/llm.py`, `core/chat.py`,
  `core/job.py` or `core/batch.py`.
* **The batch contract survived.** A lot still pins its mode at launch
  (`LOT_TASK_MODE_ID`), and a save cannot move a running lot's mode.
* **The reserved folder is still reserved.** The modes file is still not readable,
  listable or writable through `/api/fs/*`; the tree, search and git status still
  answer as if `.nojae` does not exist.
* **No new write spelling leaked into `editorfs`**, and the write-helper caller
  scans still name only the routes (plus whatever eds-02/eds-04 widened, which
  must be documented in the guard's own docstring).
* **Every guard added in eds-03/eds-04 fails when poisoned** — run the
  `TestTheGuardsActuallyFail` cases and confirm the new ones are among them.
* **No cap is hardcoded in `web/src`**: grep the client for the numbers the
  payload carries (16, 32, 40, 16384) and confirm each hit is a test fixture or a
  coincidence, not a rule.

## Full sweep

```
pytest -q
cd web && npm test
cd web && npm run build
cd web && npm run test:e2e
```

All four green. Then read `git log --stat` (or the diff against the lot's start)
for anything that landed outside the files the spec's "Files touched" table names,
and either justify it in the report or revert it.

## Report

Close with a short list: each acceptance item → the evidence, plus any finding you
fixed and any you are handing to a human. Do **not** edit `AGENTS.md` or
`README.md` — that is eds-10, and it writes from the code as it is after this
task.

## How to tell it is done

The four commands above are green and the report is written. Move this file to
`todos/done/`.
