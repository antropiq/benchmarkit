# tfix-07 — MCP seed: `test_it_creates_the_file_byte_for_byte_from_the_template` fails on Windows

**Failing test id**

`tests/test_mcp_settings_seed.py::TestTheSeedLands::test_it_creates_the_file_byte_for_byte_from_the_template`

**Reported failure (verbatim)**

```
assert b'{\r\n  "con...n  }\r\n}\r\n' == b'{\n  "conso...  ]\n  }\n}\n'
```

**Where it is**

* Test: `tests/test_mcp_settings_seed.py`, class `TestTheSeedLands`, test name
  `test_it_creates_the_file_byte_for_byte_from_the_template` (around line 54).
* What it asserts: `seed.ensure_mcp_settings_file(str(workspace))` answers
  `state == "created"`, `error is None`, `path == str(seeded_path(workspace))`, and
  the seeded file's `read_bytes()` equals
  `constants.MCP_SETTINGS_TEMPLATE.read_bytes()`.
* Helpers in the same file: `seeded_path` (line 50), the `workspace` fixture (line
  42, which sets `BENCHMARKIT_PATH_LOCATION` to `tmp_path`).
* Subject under test: `src/core/mcphub/seed.py` with
  `src/core/mcphub/constants.py` (`MCP_SETTINGS_TEMPLATE` is the committed
  `mcp_connectors/mcp-settings.default.json`; `MCP_SETTINGS_FILENAME` is the name
  it lands under in the reserved folder).

**What to deliver**

The promise is a byte-for-byte copy of the committed template into the reserved
folder, by exclusive create. On Windows the two byte strings disagree. Make this
test pass on Windows while it stays green on Linux, keeping the copy verbatim and
its `${NAME}` tokens unresolved.

**Files it touches**

One of: `tests/test_mcp_settings_seed.py` **or** `src/core/mcphub/seed.py` (the
copy). The committed template `mcp_connectors/mcp-settings.default.json` is what
the copy must stay identical to. No route, no frontend, no LSP file.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "the MCP one is seeded by exclusive create the first time a folder is opened,
  from the committed template in `mcp_connectors/`".
* "The declaration files are the only source of truth for what runs." ... "A value
  may reference an environment variable so one declaration works on every machine;
  an unresolved name is kept verbatim with a note, never blanked."
* "never overwrites" — "its files are created only when absent, so an edited file
  is never clobbered"; "a failed seed is a note, never a refusal".
* "The connectors answer the workspace level for their own root, each reading that
  policy in its own process ... the app and the connector processes share one
  parser, one matcher and one refusal sentence."
* "Tests stay hermetic": "a test patches a hub's declared file location through
  that hub's own constants"; `BENCHMARKIT_DISABLE_MCP` switches the hub off; no
  model call, no real connector process.
* `src/core/mcphub` and `src/core/lsphub` "mirror each other ... Keep that
  symmetry when touching either one."

**Depends on:** none. (Independent of tfix-05, its LSP twin — fix them separately,
but land the same kind of fix on both sides so the mirror holds.)

**How to tell it is done**

1. `pytest "tests/test_mcp_settings_seed.py::TestTheSeedLands::test_it_creates_the_file_byte_for_byte_from_the_template"` passes on this machine.
2. `tests/test_mcp_settings_seed.py` still passes in full (run that file only,
   never the suite) — including "the template is not a working file" and "never
   overwrites a hand-edited declaration".
3. The seeded file is still byte-identical to the committed template and still
   holds its `${BENCHMARKIT_PYTHON_ROOT}` / `${PROJECT_ROOT}` /
   `${BENCHMARKIT_PATH_LOCATION}` tokens unresolved. Say in one sentence, in the
   final report, which side of the comparison was platform-dependent.
4. Move this file to `todos/done/`.
