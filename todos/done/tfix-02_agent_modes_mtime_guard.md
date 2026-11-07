# tfix-02 — `test_a_conflict_writes_no_backup_and_leaves_no_temp_file` does not raise on Windows

**Failing test id**

`tests/test_agent_modes_write.py::TestTheMtimeGuard::test_a_conflict_writes_no_backup_and_leaves_no_temp_file`

**Reported failure (verbatim)**

```
Failed: DID NOT RAISE AgentModesSaveConflict
```

**Where it is**

* Test: `tests/test_agent_modes_write.py`, class `TestTheMtimeGuard`, test name
  `test_a_conflict_writes_no_backup_and_leaves_no_temp_file` (around line 713).
* The test's own shape: it writes the modes file, takes the `mtime` token with
  `token_of(workspace)`, then moves the file's timestamps forward with
  `os.utime(modes_path(workspace), ns=(int(token) + 1, int(token) + 1))` and
  expects `save(...)` to raise `writer.AgentModesSaveConflict`.
* Helpers in the same file: `write_modes`, `modes_json`, `modes_path`,
  `token_of`, `save`, `refused`, `folder_names`, and the `backups` fixture.
* Subject under test: `src/core/agent_modes_write.py` (the `mtime` guard) and
  `src/core/agent_modes.py` (where the token's value comes from).

**What to deliver**

The guard is supposed to answer "the disk moved under this save" and refuse. On
Windows the test's way of moving the disk forward does not produce a token
mismatch, so no conflict is raised. Make the test pass on Windows while it stays
green on Linux, keeping the promise the guard exists for: a stale save is a
conflict carrying both numbers.

**Files it touches**

One of: `tests/test_agent_modes_write.py` **or** `src/core/agent_modes_write.py`
/ `src/core/agent_modes.py` (the token's resolution). Nothing else — no route, no
frontend, no Editor code.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "Core owns logic, the API owns transport." ... "nothing under `core` may import
  the API layer."
* The mtime guard is the app's conflict guard: "a stale save is a conflict
  carrying both numbers" — the neighbouring tests in `TestTheMtimeGuard` and
  `TestTheGuardCanFail` pin that, and they must stay green.
* "A save keeps the bytes it replaced" — a conflict must still write **no**
  backup and leave **no** temp file, which is literally this test's name.
* "Tests stay hermetic": no model call, no real workspace, no real settings file.

**Depends on:** none.

**How to tell it is done**

1. `pytest "tests/test_agent_modes_write.py::TestTheMtimeGuard::test_a_conflict_writes_no_backup_and_leaves_no_temp_file"` passes on this machine.
2. The whole file `tests/test_agent_modes_write.py` still passes (run that file
   only, never the suite).
3. The test still asserts a refusal: it must not be weakened into accepting a
   save that went through. Say in one sentence, in the final report, what the
   platform difference was.
4. Move this file to `todos/done/`.
