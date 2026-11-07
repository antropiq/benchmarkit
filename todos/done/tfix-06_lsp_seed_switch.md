# tfix-06 — LSP seed: `test_switch_seeds_the_template` fails on Windows

**Failing test id**

`tests/test_lsp_settings_seed.py::TestTheTwoEntryPoints::test_switch_seeds_the_template`

**Reported failure (verbatim)**

```
assert b'{\r\n  "_no...n  }\r\n}\r\n' == b'{\n  "_note...300\n  }\n}\n'
```

**Where it is**

* Test: `tests/test_lsp_settings_seed.py`, class `TestTheTwoEntryPoints`, test name
  `test_switch_seeds_the_template` (around line 253).
* What it does: drives the app through the `seed_client` fixture (line 199), moves
  the working location with `POST /api/path-location`, and compares the bytes the
  restarted hub saw against the template's bytes (`template_bytes()` /
  `constants.LSP_SETTINGS_TEMPLATE.read_bytes()` in that file).
* Helpers in the same file: `seed_client` (line 199), `_RecordingHub` (line 183),
  `seeded_template` (line 50), `settings_file` (line 54).
* Subject under test: the two seed entry points — the `GET` state answer and the
  `POST /api/path-location` seed result in `src/api/routes.py`, backed by
  `src/core/lsphub/seed.py` and the hub restart in `src/core/lsphub/manager.py`.

**What to deliver**

The switch must seed the template and report it, and the bytes the restart reads
must be the template's. On Windows the byte comparison disagrees. Make this test
pass on Windows while it stays green on Linux.

**Files it touches**

One of: `tests/test_lsp_settings_seed.py` **or** `src/core/lsphub/seed.py` (the
copy it seeds). Only if the investigation shows the *entry point* is at fault
should it also touch `src/api/routes.py` — and then the route's existing contract
(a switch seeds and reports, a failure is a note, the settings cache is
invalidated) must stay as it is.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "the switch seeds and reports — both at `GET` (state only) and
  `POST /api/path-location` (seed result), so the Settings → LSP panel can point
  at the template."
* "the template is the example verbatim — one `O_CREAT | O_EXCL` copy ... never
  substituted at seed time"; "seeding never creates the declaration".
* "One setting defines the working location ... moving it invalidates them all at
  once: cached views are discarded rather than re-negotiated" — the switch's cache
  invalidation stays.
* "A failure is a note, never a refusal" for a failed seed.
* "Core owns logic, the API owns transport." ... "nothing under `core` may import
  the API layer."
* "Tests stay hermetic": the hubs are disabled by an environment flag elsewhere;
  this test drives a recording hub, not a real server. No model call.

**Depends on:** none. (Independent of tfix-05 even though both live in the same
test file — do not assume its fix carries over.)

**How to tell it is done**

1. `pytest "tests/test_lsp_settings_seed.py::TestTheTwoEntryPoints::test_switch_seeds_the_template"` passes on this machine.
2. `tests/test_lsp_settings_seed.py` still passes in full (run that file only,
   never the suite).
3. The test still asserts the switch seeded and reported the template, and that
   the bytes are the committed example's. Say in one sentence, in the final
   report, which side of the comparison was platform-dependent.
4. Move this file to `todos/done/`.
