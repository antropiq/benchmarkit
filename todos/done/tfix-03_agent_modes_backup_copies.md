# tfix-03 — `test_a_second_save_keeps_two_copies` fails on Windows

**Failing test id**

`tests/test_agent_modes_write.py::TestTheBackup::test_a_second_save_keeps_two_copies`

**Reported failure (verbatim)**

```
assert b'{\r\n  "ver...}\r\n  }\r\n}' == b'{\n  "versi...    }\n  }\n}'
```

**Where it is**

* Test: `tests/test_agent_modes_write.py`, class `TestTheBackup`, test name
  `test_a_second_save_keeps_two_copies` (around line 826).
* What it does: saves twice, then reads the backup chain through
  `editorfs.files.list_backups(path)` and asserts the **oldest copy first** —
  `copies[0]` must equal the hand-written document's bytes
  (`modes_json(agents=GOOD_AGENTS).encode("utf-8")`), and `copies[1]` must still
  hold the first save's mode.
* Helpers in the same file: `write_modes`, `modes_json`, `modes_path`,
  `token_of`, `save`, `entry`, `body`, `GOOD_AGENTS`, and the `backups` fixture.
* Subject under test: `src/core/agent_modes_write.py` (the save and its D8 backup)
  plus `src/core/editorfs` (`files.list_backups`, `files.backup_key`).

**What to deliver**

The backup chain must keep the bytes it replaced, oldest first. On Windows the
byte comparison of the first backup against the bytes the test wrote disagrees.
Make this test pass on Windows while it stays green on Linux, keeping the promise
that the chain holds **what was on disk**, not a re-rendered copy of it.

**Files it touches**

One of: `tests/test_agent_modes_write.py` **or** `src/core/agent_modes_write.py`
**or** `src/core/editorfs/` (the backup writer/reader). Nothing else — no route,
no frontend.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "A save backs up what it replaces" / "a save keeps the bytes it replaced" (D8):
  "the same pre-save backup the Editor's saves use (the app's own backup folder,
  not inside the workspace)" — the modes save and the Editor's saves share **one**
  chain, which `TestTheBackup` and `tests/test_editorfs_write.py::TestBackups`
  both assert. Do not build a second chain.
* "Nothing is written on a refusal" — a conflict still writes no backup.
* "Core owns logic, the API owns transport." ... "nothing under `core` may import
  the API layer."
* "Tests stay hermetic": no model call, no real workspace, no real settings file.

**Depends on:** none. (Independent of tfix-01 and tfix-02 even though all three
live in the same test file — do not assume their fixes carry over.)

**How to tell it is done**

1. `pytest "tests/test_agent_modes_write.py::TestTheBackup::test_a_second_save_keeps_two_copies"` passes on this machine.
2. `tests/test_agent_modes_write.py` and `tests/test_editorfs_write.py` both still
   pass (run those two files only, never the whole suite).
3. The test still asserts two copies, oldest first, and that the oldest is the
   byte-for-byte document it replaced. Say in one sentence, in the final report,
   which side of the comparison was platform-dependent.
4. Move this file to `todos/done/`.
