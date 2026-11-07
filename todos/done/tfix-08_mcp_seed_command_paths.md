# tfix-08 — MCP seed: `test_the_seeded_file_loads_as_the_two_shipped_connectors` fails on Windows

**Failing test id**

`tests/test_mcp_settings_seed.py::TestTheSeedLands::test_the_seeded_file_loads_as_the_two_shipped_connectors`

**Reported failure (verbatim)**

```
AssertionError: assert ['/opt/py/pyt...ort', 'stdio'] == ['/opt/py/pyt...ort', 'stdio']
```

(The two sides print alike in the truncated summary — the difference is in the part
pytest elided. The test's expected first command element is the POSIX-looking
`/opt/py/python` it puts into `BENCHMARKIT_PYTHON_ROOT` itself.)

**Where it is**

* Test: `tests/test_mcp_settings_seed.py`, class `TestTheSeedLands`, test name
  `test_the_seeded_file_loads_as_the_two_shipped_connectors` (around line 65).
* What it does: `monkeypatch.setenv("BENCHMARKIT_PYTHON_ROOT", "/opt/py/python")`,
  seeds the declaration, then `load_externals()` and asserts the connector ids are
  `["console", "fedit"]`, `errors == []`, and the console entry's `command` is
  `[<that python>, f"{str(constants.MCP_CONNECTORS_DIR)}/console_mcp.py",
  "--root-dir", str(workspace), "--transport", "stdio"]`.
* Helpers in the same file: `seeded_path` (line 50), the `workspace` fixture (line
  42).
* Subject under test: `src/core/mcphub/discovery.py` (`load_externals`) and
  `src/core/mcphub/validation.py` (`expand_env`, which resolves the `${NAME}`
  references on every read), plus `src/core/mcphub/constants.py`
  (`MCP_CONNECTORS_DIR`).

**What to deliver**

The test pins "a fresh workspace's hub is console + fedit" with the tokens
resolved from the environment. On Windows the resolved command list and the
expected list differ. Make this test pass on Windows while it stays green on
Linux, keeping the rule that a declaration "works on every machine" because its
values are resolved at read time.

**Files it touches**

One of: `tests/test_mcp_settings_seed.py` (how it spells the path it expects)
**or** `src/core/mcphub/validation.py` / `src/core/mcphub/discovery.py` (how a
token's value is turned into a command element). No seed change, no route, no
frontend, no LSP file.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "A value may reference an environment variable so one declaration works on every
  machine; an unresolved name is kept verbatim with a note, never blanked."
* "`${PROJECT_ROOT}` is no environment variable: it answers from the app's own
  computed install path, so a stray name in the environment cannot redirect a
  connector."
* "The declaration files are the only source of truth for what runs." — the ids,
  their order and the `enable` switch this test asserts stay as they are.
* "Their secrets never leave the server."
* "Tests stay hermetic": no real connector process is started here, and no model
  call; `tests/test_env_isolation.py` keeps a test from reading a real model
  connection.
* "Core owns logic, the API owns transport." ... "nothing under `core` may import
  the API layer."

**Depends on:** none. (Independent of tfix-07 even though both sit in the same
class of the same file — do not assume its fix carries over.)

**How to tell it is done**

1. `pytest "tests/test_mcp_settings_seed.py::TestTheSeedLands::test_the_seeded_file_loads_as_the_two_shipped_connectors"` passes on this machine.
2. `tests/test_mcp_settings_seed.py` still passes in full (run that file only,
   never the suite), and `tests/test_mcp.py` still passes too (the hub's own
   suite owns the expansion rules).
3. The test still asserts both connector ids, the empty `errors`, and a fully
   resolved command list — no token left unresolved in what the hub was handed.
   Say in one sentence, in the final report, which side was spelled for POSIX.
4. Move this file to `todos/done/`.
