# tfix-09 — workspace-settings seed: `test_the_file_the_restart_sees_is_the_new_roots` fails on Windows

**Failing test id**

`tests/test_workspace_settings_seed.py::TestTheTwoEntryPoints::test_the_file_the_restart_sees_is_the_new_roots`

**Reported failure (verbatim)**

```
assert b'{\r\n  "ser... 300\r\n}\r\n' == b'{\n  "serve...on": 300\n}\n'
```

**Where it is**

* Test: `tests/test_workspace_settings_seed.py`, class `TestTheTwoEntryPoints`,
  test name `test_the_file_the_restart_sees_is_the_new_roots` (around line 509).
* What it does: plants the target root's **own** settings file by hand
  (`seeded_path(target).write_text(own, encoding="utf-8")` where `own` comes from
  the `configured_document(...)` helper, line 82), switches the working location
  with `POST /api/path-location`, asserts the seed answers
  `workspaceSettings.state == "existing"` (so it wrote nothing), and asserts the
  bytes the restarted hub saw are `own.encode("utf-8")`.
* Helpers in the same file: `seed_client` (line 388), `_RecordingHub` (line 358),
  `_StartRecordingHub` (line 202), `template_bytes` (line 78),
  `configured_document` (line 82), and this file's `seeded_path`.
* Subject under test: `src/core/workspace_settings.py` (`ensure_workspace_settings_file`,
  `load_workspace_settings`, `workspace_settings_state`, the cache invalidation)
  reached through `POST /api/path-location` in `src/api/routes.py`.

**What to deliver**

The promise this test pins is D10's other half: *which* file the restart reads —
the new root's own bytes, never the previous workspace's and never the template.
On Windows the bytes read back from the file the test itself planted disagree with
the bytes the test built. Make this test pass on Windows while it stays green on
Linux, keeping the assertion about *which file* was read.

**Files it touches**

One of: `tests/test_workspace_settings_seed.py` **or** `src/core/workspace_settings.py`
(the reader/writer of that file). Only if the investigation shows the route is at
fault should it also touch `src/api/routes.py` — and then the route's existing
contract (seed-and-report on a switch, the cache invalidated, the test-settings
write refusal) must stay as it is.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "`.nojae/nojae-settings.json` — This workspace's model connection ... seeded
  from a committed template at the install root, written by the Settings dialog."
* "One setting defines the working location ... Every path the app holds is
  relative to it, so moving it invalidates them all at once: cached views are
  discarded rather than re-negotiated, and unsaved work is reported rather than
  silently dropped."
* "The reserved folder is seeded, then read-only to the app: its files are created
  only when absent, so an edited file is never clobbered ... except the two stores
  it writes, its prompt-helper store and this workspace's model-connection file."
* "`.env` is regenerated from a fixed template on every save" — this task must not
  change what the Settings save writes to `.env`.
* "A test run takes its endpoint from the test env file, never from a developer's."
* "Core owns logic, the API owns transport." ... "nothing under `core` may import
  the API layer."
* "Tests stay hermetic": no model call; the recording hub stands in for the real
  one.

**Depends on:** none.

**How to tell it is done**

1. `pytest "tests/test_workspace_settings_seed.py::TestTheTwoEntryPoints::test_the_file_the_restart_sees_is_the_new_roots"` passes on this machine.
2. `tests/test_workspace_settings_seed.py` and `tests/test_workspace_settings.py`
   both still pass (run those two files only, never the whole suite).
3. The test still asserts the seed answered `existing` (wrote nothing) and that the
   bytes the restart saw are the new root's own file. Say in one sentence, in the
   final report, which side of the byte comparison was platform-dependent.
4. Move this file to `todos/done/`.
