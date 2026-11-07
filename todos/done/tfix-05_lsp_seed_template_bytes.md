# tfix-05 — LSP seed: `test_it_creates_the_template_byte_for_byte_from_the_example` fails on Windows

**Failing test id**

`tests/test_lsp_settings_seed.py::TestTheSeedLands::test_it_creates_the_template_byte_for_byte_from_the_example`

**Reported failure (verbatim)**

```
assert b'{\r\n  "_no...n  }\r\n}\r\n' == b'{\n  "_note...300\n  }\n}\n'
```

**Where it is**

* Test: `tests/test_lsp_settings_seed.py`, class `TestTheSeedLands`, test name
  `test_it_creates_the_template_byte_for_byte_from_the_example` (around line 117).
* What it asserts: `seed.ensure_lsp_settings_template(str(workspace))` answers
  `state == "created"`, `error is None`, `path == str(seeded_template(workspace))`,
  and the seeded file's `read_bytes()` equals
  `constants.LSP_SETTINGS_TEMPLATE.read_bytes()`.
* Helpers in the same file: `seeded_template` (line 50), `settings_file` (line 54),
  the `workspace` fixture.
* Subject under test: `src/core/lsphub/seed.py` with `src/core/lsphub/constants.py`
  (`LSP_SETTINGS_TEMPLATE` points at the committed example in `lsp_connectors/`,
  `LSP_SETTINGS_TEMPLATE_FILENAME` is the name it lands under in the reserved
  folder).

**What to deliver**

The promise is "the template is the example verbatim — one `O_CREAT | O_EXCL`
copy". On Windows the two byte strings disagree. Make this test pass on Windows
while it stays green on Linux, keeping the copy verbatim.

**Files it touches**

One of: `tests/test_lsp_settings_seed.py` **or** `src/core/lsphub/seed.py` (the
copy). The committed example `lsp_connectors/lsp_servers.example.json` is that
template's source — if it is involved, remember it is also the template a human
copies by hand. No route, no frontend, no MCP file.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "the language-server one is hand-written — its seed leaves only
  `.nojae/lsp-settings.template.json`, a copy of the committed example in
  `lsp_connectors/`, because a server entry names machine-local paths and a seeded
  declaration would load nowhere."
* "the template is the example verbatim — one `O_CREAT | O_EXCL` copy, its
  `${NAME}` references resolved on every read, **never substituted at seed
  time**".
* "never overwrites — a hand-edited template survives every later seed"; "a
  failure is a note" and "a reader never writes".
* "The connectors answer the workspace level for their own root, each reading that
  policy in its own process ... the app and the connector processes share one
  parser, one matcher and one refusal sentence."
* "Tests stay hermetic": a test patches a hub's declared file location through
  that hub's own constants; no model call.
* `src/core/lsphub` and `src/core/mcphub` "mirror each other — `constants`,
  `discovery`, `manager`, plus rendering of tool results, and `seed` on each side.
  Keep that symmetry when touching either one."

**Depends on:** none. (tfix-07 is the MCP twin of this test in the mirrored hub;
fix them separately and keep the two sides symmetric.)

**How to tell it is done**

1. `pytest "tests/test_lsp_settings_seed.py::TestTheSeedLands::test_it_creates_the_template_byte_for_byte_from_the_example"` passes on this machine.
2. `tests/test_lsp_settings_seed.py` still passes in full (run that file only,
   never the suite) — including "never overwrites" and "seeding never creates the
   declaration".
3. The seeded file is still a byte-identical copy of the committed example, and
   its `${NAME}` tokens are still unresolved in the copy. Say in one sentence, in
   the final report, which side of the comparison was platform-dependent.
4. Move this file to `todos/done/`.
