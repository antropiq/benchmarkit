# tfix-01 — `test_a_key_outside_the_id_pattern_is_refused` fails on Windows

**Failing test id**

`tests/test_agent_modes_write.py::TestRefusalsLeaveTheDiskAlone::test_a_key_outside_the_id_pattern_is_refused`

**Reported failure (verbatim)**

```
assert b'{\r\n  "ver...}\r\n  }\r\n}' == b'{\n  "versi...    }\n  }\n}'
```

**Where it is**

* Test: `tests/test_agent_modes_write.py`, class `TestRefusalsLeaveTheDiskAlone`,
  test name `test_a_key_outside_the_id_pattern_is_refused` (around line 462 of
  that file; the file's own header explains what the suite pins).
* Helpers it uses, defined near the top of the same file: `write_modes`,
  `modes_json`, `modes_path`, `token_of`, `save`, `refused`, and the `workspace`
  fixture.
* Subject under test: `src/core/agent_modes_write.py` (the whole-document writer)
  read back through `src/core/agent_modes.py`.

**What to deliver**

This test asserts that a refused save leaves the modes file's bytes exactly as
they were. On Windows the two sides of that byte comparison disagree; make the
test pass on Windows while it stays green on Linux. Decide, from the code and the
suite's own promises, whether the disagreement belongs to what the writer puts on
disk or to how the test builds the bytes it compares — and fix that one thing.

**Files it touches**

One of: `tests/test_agent_modes_write.py` **or** `src/core/agent_modes_write.py`
(and, if the writer's on-disk shape changes, the reader/loader side
`src/core/agent_modes.py` only insofar as the round-trip promise must still hold).
Nothing else. No frontend file, no route.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "Core owns logic, the API owns transport." ... "nothing under `core` may import
  the API layer."
* The reserved-folder rule: "its files are created only when absent, so an edited
  file is never clobbered" — the modes save stays an exclusive create when the
  file is absent.
* "A save keeps the bytes it replaced" (the D8 backup) — the other tests in this
  class and `TestTheBackup` pin it.
* "Tests stay hermetic": no model call, no real workspace, no real settings file.

**Depends on:** none.

**How to tell it is done**

1. `pytest "tests/test_agent_modes_write.py::TestRefusalsLeaveTheDiskAlone::test_a_key_outside_the_id_pattern_is_refused"` passes on this machine.
2. The rest of `tests/test_agent_modes_write.py` still passes: run that one file,
   not the suite.
3. The fix does not delete the assertion or turn it into a no-op: the test must
   still assert that a refused save leaves the disk alone. Say in one sentence,
   in the final report, which side (writer or test) was wrong and why the other
   platform was unaffected.
4. Move this file to `todos/done/`.
