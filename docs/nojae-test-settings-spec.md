# E2E test configuration — `.nojae/nojae-test-settings.json`

Status: **landed** — implemented as specified, with no change to any decision
below. The evidence is `src/core/workspace_settings.py` (the switch, the two
names, the mode-aware read and the write refusal), `tests/test_workspace_settings.py`
and `tests/test_env_isolation.py` (the mode and the hermeticity), the
`TestSettingsEndpointInTestMode` cases in `tests/test_api.py`, and the
`web/e2e/03-workspace-switch.spec.js` assertions on the fixture. The README's
"Setting up tests for benchmarkit" section is the operator's step. Scope: give
the e2e suite its **own** model-connection file, so a spec can never talk to the
developer's workspace settings, and reduce `.env` and `.env.test` to app-level
keys only.

Parent spec: `docs/nojae-settings-workspace-spec.md` (the `wss` lot; ws-01 … ws-05
landed, ws-06/ws-07 open). This spec does not re-open any of its D1 … D11; it
extends them to a second file of the same shape. Its decisions are numbered
**T-D1 … T-D8** so a task can cite them without colliding.

## 1. What this is for

The `wss` lot moved group 1 (server, key, model id, prompt, sampling, vision,
`lspEnabled`, `maxToolIteration`) out of `.env` into
`<working location>/.nojae/nojae-settings.json`, and **D2 made `os.environ` stop
answering those keys at all**. The e2e suite was not moved with it:
`web/playwright.config.js` still parses `.env.test` and injects it into the
environment of the server under test. Since D2, every group-1 line in that file
is **dead bytes** — parsed, exported, read by nothing.

What the suite therefore does today is read the **boot working location's**
`.nojae/nojae-settings.json`. The boot location defaults to the install
directory, so:

- **the e2e suite benchmarks, chats and grades against the developer's own
  server, with the developer's own key** — the leak the parent spec's §10
  predicted, arriving on the test side;
- the suite is **green while it happens**.
  `web/e2e/03-workspace-switch.spec.js` compares the boot root's dialog values
  *to themselves* after a round trip, and the scratch root is freshly seeded with
  the template's instruction text, so every assertion holds whatever server the
  machine happens to point at. No spec fails when the endpoint changes; several
  specs pass that should not have run at all.
- `.env.test`'s pinned sampling (`TOP_P`, `TOP_K`, `REASONING=off`, …) pins
  nothing any more, so the specs that do stream an answer are no longer
  reproducible for the reason their own comments claim.

After this change:

- **`.nojae/nojae-test-settings.json`** (working location, reserved folder) holds
  the model connection the suite uses — the same 14 members as
  `nojae-settings.json`, same types, same reader. **Hand-authored, never created
  or written by the app**, like `.env.test` and like `lsp-settings.json`.
- **`.env.test`** keeps only the app-level keys — exactly `config.SETTING_KEYS`:
  `QUESTION_SET`, `LLM_DEBUG`, the four `VERIFIER_AGENT_*`,
  `BENCHMARKIT_PYTHON_ROOT`.
- **`.env`** is unchanged (it already holds only app-level keys since ws-04).

One switch decides which of the two settings files a process reads.

## 2. The two files, and which keys live where

| File | Holds | Read by | Written by |
|---|---|---|---|
| `.nojae/nojae-settings.json` | a workspace's model connection | the app, when the test switch is **off** | the Settings dialog; seeded from `nojae-settings.default.json` |
| `.nojae/nojae-test-settings.json` | the **e2e suite's** model connection | the app, when the test switch is **on** | nobody but a human |
| `.env` | `QUESTION_SET`, `LLM_DEBUG`, `VERIFIER_AGENT_*`, `BENCHMARKIT_PYTHON_ROOT` | `config` / `os.environ` | `save_settings()`, from the fixed template |
| `.env.test` | the same seven keys, for the server under test | `playwright.config.js` → `webServer.env` | a human, from `.env.test.example` |

The split is the parent spec's three groups, unchanged: **group 1 is a settings
file, groups 2 and 3 are `.env`'s** — and the e2e suite now follows that rule
instead of keeping its own copy of it. The verifier stays in `.env.test` on
purpose: it is group 2, i.e. the install's scope, and the suite needs a verifier
that answers (an ungraded answer is failed, which would make every benchmark spec
measure the wrong thing).

### The test file's shape

The 14 members of the parent spec's §2, with the same JSON types and the same
"member absent means" table. Unlike `nojae-settings.json` it has **no committed
template and no instruction sentences** — a hand-authored file either holds a
usable value or the reader says so. A realistic file:

```json
{
  "serverUrl": "http://<test-host>:8080/v1",
  "serverApiKey": "",
  "endpoint": "<the test model id, exactly as the server names it>",
  "supportsVision": false,
  "sysprompt": "You are a helpful assistant.",
  "temperature": 1,
  "topP": 0.95,
  "topK": 20,
  "minP": 0,
  "repeatPenalty": 1.0,
  "presencePenalty": 0,
  "reasoning": "off",
  "lspEnabled": [],
  "maxToolIteration": 25
}
```

No document about this file may hold a real key. The file itself is gitignored —
`.gitignore` already carries `.nojae/`, so **no gitignore change is needed** — and
the README section says so.

`reasoning: "off"` and the pinned samplers are not decoration: they are what
replaces `.env.test`'s dead `REASONING=off` / `TOP_P` / `TOP_K` lines, and they
are what keeps the streaming specs reproducible.

## 3. Decisions

**T-D1 — one process-level switch: `BENCHMARKIT_TEST_SETTINGS`.** When it is
truthy (`config.env_flag`, the same `1/true/yes/on` rule as
`BENCHMARKIT_DISABLE_MCP`), the settings file *in use* is
`nojae-test-settings.json`; otherwise `nojae-settings.json`. It is exported by
`playwright.config.js` in `webServer.env`, which is the one place that already
guarantees the precedence `.env.test` relies on — applied last when the server is
spawned, so it beats an ambient value inherited from a running app's process tree.
Rejected: a `BENCHMARKIT_TEST_SETTINGS_FILE` naming an arbitrary path — that lets
a settings file live outside the reserved folder, where the workspace policy and
the `.nojae` invisibility rules do not apply, and it splits "the file in use" from
"the file the dialog names". Rejected: a per-request or per-workspace flag — the
choice is about *which process this is*, not which request.

**T-D2 — the switch is read per call, never at import.** `test_settings_mode()`
answers from `os.environ` each time, so `tests/conftest.py` can clear the variable
for the pytest suite (T-D7) and a unit test can flip it with `monkeypatch.setenv`
without reloading the module. Same reason `workspace_settings_path()` is built per
call.

**T-D3 — no fallback, in either direction.** With the switch on, an absent test
file is the `absent` state and a chat/benchmark request refuses with the reader's
own sentence — it does **not** fall through to `nojae-settings.json`. Rejected:
"test file wins, workspace file as fallback" — that is the exact ambiguity D2
removed, and here it would mean a suite silently reaching the developer's key
whenever the test file's name has a typo in it. The consequence is deliberate and
loud: **`playwright.config.js` refuses to start the suite** when the file is
missing (T-D6), because a missing fixture must not look like an app bug.

**T-D4 — the switch never changes the seed.** `ensure_workspace_settings_file()`
keeps seeding `nojae-settings.json` from `nojae-settings.default.json`, in test
mode exactly as in production mode. Rejected: skipping the seed in test mode — it
would make the reserved folder's file set depend on a mode (the switch's own e2e
assertion lists the seeded files), and it would mean a workspace is not usable the
moment the switch is off. The seeded file is simply read by nothing while the
switch is on, which is D2's "dead bytes" rule applied the other way — and is
precisely what makes the developer's live key unreachable from a spec.
Corollary, and the point of the whole file: **the app never creates, seeds or
writes `nojae-test-settings.json`.**

**T-D5 — the write side refuses in test mode.** `save_workspace_settings()`
raises `WorkspaceSettingsFileError` as its **first act** when the switch is on,
with a sentence naming the test file and saying it is hand-authored. The Settings
dialog's group-1 save and `post_lsp()`'s persistence both land in that refusal.
Rejected: letting the writer write the test file — a dialog save mid-suite would
rewrite the fixture the *next* spec depends on (the e2e equivalent of a spec
editing `.env.test` while it runs), and `reuseExistingServer` would carry the
change into the next run. The refusal must be checked **before**
`routes._save_workspace_group_one()`'s remove-then-write dance, which deletes the
file before calling the writer; both call sites consult one predicate, and a test
pins that a refused save leaves the test file's bytes untouched.

**T-D6 — the suite fails loudly when the fixture is missing.**
`playwright.config.js` gains an existence check for the test settings file beside
the one it already has for `.env.test`, with the reasoning already written in that
file's comment: a silent drop would let specs pass against a configuration no
other machine can repeat. The expected path is resolved the way the app resolves
it — `BENCHMARKIT_PATH_LOCATION` if set, else the repo root, plus
`.nojae/nojae-test-settings.json` — and the error names both the path and the
README section that explains it. `E2E_ENV_FILE` keeps pointing at `.env.test` and
gains no sibling override (T-D1's rejection).

**T-D7 — the pytest suite pins the switch off.** An exported
`BENCHMARKIT_TEST_SETTINGS=1` in a developer's shell would make the *whole backend*
suite read a file that is not there and fail en masse for a reason no traceback
explains. An autouse fixture in `tests/conftest.py` clears it, beside the existing
`_no_workspace_settings_leak`, and `tests/test_env_isolation.py` gains the case.
The e2e suite is the only thing that turns it on.

**T-D8 — the test file's member table is its own, and contains the production
one.** `WORKSPACE_TEST_SETTINGS_KEYS: Dict[str, type] = {**WORKSPACE_SETTINGS_KEYS}`
plus, later, whatever test-only members appear. The strict D9 member check applies
to it verbatim — a typo in a test member is `failed`, never silently ignored,
because a silently-defaulted sampler makes a green suite mean nothing.
`_self_check()` pins that the production table is a subset of the test table and
that the two filenames differ. A test-only member is added **here**, never to
`WORKSPACE_SETTINGS_KEYS` (a production file must never gain a member the dialog
and the template do not know), and the README's table changes in the same commit.
Today the extension is empty: the file the suite needs holds the 14 members and
nothing else, so this is a stated seam with an invariant, not a speculative
feature.

## 4. What it builds on

- `src/core/workspace_settings.py` — the whole read/write/state side of a settings
  file, already parameterised enough to serve two names: the filename is a module
  attribute, the path is built per call, the member check already reads a
  table-shaped rule, and the state vocabulary (`ok`/`absent`/`failed` + the D8
  `configured` flag) needs no new word.
- `src/config.py` — `env_flag()` for the switch, `PATH_LOCATION_KEY` for where the
  file lives, and `workspace_settings_refusal()` as the one builder of the refusal
  sentence. That sentence already interpolates the resolved path, so **no new
  refusal sentence is needed**: in test mode the existing one names
  `nojae-test-settings.json` on its own.
- `web/playwright.config.js` — the `.env.test` loader, its loud missing-file check,
  and the `webServer.env` precedence that `BENCHMARKIT_DISABLE_MCP` / `_LSP`
  already use.
- `tests/conftest.py` — `neutralise_workspace_settings()`,
  `write_workspace_settings()` and `_no_workspace_settings_leak`: the guards this
  spec extends rather than duplicates.
- `web/e2e/03-workspace-switch.spec.js` — the spec that already reads
  `nojae-settings.default.json` from the checkout and asserts on the dialog's
  reported path; it becomes the spec that pins which file the suite is configured
  from.

## 5. Shape of the change

### `src/core/workspace_settings.py`

- `TEST_SETTINGS_KEY = "BENCHMARKIT_TEST_SETTINGS"`,
  `WORKSPACE_TEST_SETTINGS_FILENAME = "nojae-test-settings.json"`,
  `WORKSPACE_TEST_SETTINGS_KEYS` (T-D8) — all module attributes, patchable, and
  named in the module's existing self-check loop.
- `test_settings_mode() -> bool` — `env_flag(TEST_SETTINGS_KEY)`, per call (T-D2).
- `active_settings_filename() -> str` — the mode's filename.
- `settings_file_path(root, filename)` — the one join.
  `workspace_settings_file_path(root)` keeps its current meaning and **stays the
  production name** (it is the seeder's path, T-D4); `workspace_settings_path()`
  becomes the **file in use** (mode-aware), which is what every existing caller of
  it wants.
- The validation helpers (`_problem_of`, `_mergeable_members`, `_member_names`)
  take the table as an argument instead of closing over `WORKSPACE_SETTINGS_KEYS`;
  the reader picks the table by mode, the writer always the production one.
- `save_workspace_settings()` refuses first thing in test mode (T-D5).
- `_template_instruction_values()` is unchanged: the D8 seed-text check runs
  against the same committed template in both modes, because that is the only
  instruction text that exists. A hand-authored test file whose `serverUrl` was
  pasted from the template is therefore reported *not configured* rather than
  dialled — the same protection, for free.
- `_self_check()` gains the T-D8 subset check and the distinct-filenames check.

**No caller of the read side changes.** `load_workspace_settings()`,
`workspace_setting()`, `workspace_settings_state()` and `_current()` all resolve the
path through `workspace_settings_path()`, so `config.load_endpoints_config()`,
`config.lsp_enabled_ids()`, `config.max_tool_iteration()`,
`config.workspace_settings_refusal()` and `routes._settings_json()` follow the mode on
their own. That is the whole reason this is one seam and not fourteen: ws-01/ws-03
funnelled every group-1 read through this module already.

### `src/api/routes.py`

- `_settings_json()` and `_workspace_values()` index
  `WORKSPACE_SETTINGS_KEYS[member]` directly; both must resolve the type through
  the **table in use**, and `_WORKSPACE_ABSENT` stays as it is.
- `_save_workspace_group_one()` consults the write refusal before its
  `os.remove(...)` (T-D5).
- No new route and no new payload field: `workspaceSettings.path` already names the
  file in use, which is how the dialog tells the user which file it is looking at.

### `web/src/components/SettingsDialog.vue`

The hint hardcodes the literal `.nojae/nojae-settings.json` beside the
server-reported absolute path. In test mode that literal is simply wrong, printed
next to a path that is right. It stops hardcoding the name and renders the reported
path alone — one sentence, one source, and the dialog can never disagree with the
server about which file it edits.

`workspaceSettingsNote` shows a sentence for `failed` and for ok-but-not-configured,
and nothing for `absent` — which production can never reach, because the seed always
creates the file. Test mode **can** reach it (a switched-to scratch root has no test
settings file, T-D4), so the note gains the `absent` case with the reader's own
message: a dialog that shows empty fields and no explanation would read as an app
bug, which is the failure mode this whole spec exists to remove.

### `web/playwright.config.js`

- `TEST_SETTINGS_FILE` resolved from `BENCHMARKIT_PATH_LOCATION ?? <repo root>` +
  `.nojae/nojae-test-settings.json`, and a missing-file throw (T-D6).
- `webServer.env` gains `BENCHMARKIT_TEST_SETTINGS: '1'` beside the two
  `BENCHMARKIT_DISABLE_*` flags.
- The comment block above `readEnvFile` stops claiming `.env.test` "defines the LLM
  endpoint" — it defines the **app-level** keys; the endpoint is the settings
  file's job.

### `.env.test` and `.env.test.example`

Both lose every group-1 line (`SERVER_URL`, `SERVER_URL_API_KEY`, `ENDPOINT`,
`SUPPORTS_VISION`, `LLM_SYSPROMPT`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`,
`REPEAT_PENALTY`, `PRESENCE_PENALTY`, `REASONING`, `LSP_ENABLED`,
`MAX_TOOL_ITERATION`, and the stale `MCP_ENABLED` comment) and keep the seven
app-level keys, under a header comment that says where the endpoint went and which
README section explains it. They move in the same commit as the config change —
AGENTS.md's trio rule, applied to the test pair.

Two things the current `.env.test` carries that the rewrite settles, so they are
stated rather than changed quietly:

- its debug line is the **legacy** `#LITELLM_DEBUG=1` while the example file has
  `#LLM_DEBUG=1`. The rewritten file uses `LLM_DEBUG`, the name `config` reads;
- `QUESTION_SET` differs between the two (`teste2e` in the working file, `legacy` in
  the example). The working file's value is the one the specs are written against —
  `resources/teste2e.json` is the 30-question set a benchmark spec can wait on — so
  the **example** gains that value and the comment explaining why, and the working
  file keeps it.

### Tests

- `tests/conftest.py` — the autouse clear (T-D7); `write_workspace_settings()`
  builds its path from `active_settings_filename()`, so a test that flips the mode
  writes the file that mode reads.
- `tests/test_workspace_settings.py` — the mode cases (§8).
- `tests/test_env_isolation.py` — the switch cannot leak into the suite, and with
  the switch on no test can read the production file.

## 6. Documentation

- **README — new `### Setting up tests for benchmarkit`**, in §1 directly after
  `### Tests`. It states: the two files and which keys each holds; that
  `.nojae/nojae-test-settings.json` is **created by hand, never by the app**, and
  is invisible to the built-in Editor (`.nojae` is reserved) so it is edited
  outside the app; the JSON shape with placeholder values and no secret; that it is
  gitignored with the rest of `.nojae`; that the app will not write it, so a
  Settings-dialog save does not change what the suite talks to; that a missing file
  stops the whole suite before a spec runs, and what to do about it; and that
  `cd web && npm run test:e2e` needs the server it names to be reachable.
- **README `### LLM configuration — `.env``** — one pointer line: the e2e suite
  reads the same app-level keys from `.env.test`, and its model connection from the
  test settings file. Cross-reference to the new section.
- **`AGENTS.md`** — two edits, in the file's own voice and no longer: the `.env.test`
  row of the Data-files table (it says "same key set as the template", which stops
  being true) and one row for the test settings file, saying it is hand-authored and
  read only under the test switch. AGENTS.md is injected into every WorkBench turn,
  so no decision numbers and no member lists.
- **The two open `wss` task files** — `todos/wss-06_verification_pass.md` tells its
  runner to put "an endpoint in `.env.test`", and
  `todos/wss-07_docs_readme_agents_spec.md` tells it to document `.env.test` as the
  endpoint source. Both sentences become false with this change and are corrected as
  part of it, so ws-06 does not run its e2e pass against the developer's server and
  ws-07 does not document the old mechanism.

## 7. Out of scope

- Any change to groups 2/3, to the verifier, or to how `.env` is rendered.
- Making the test settings file seeded, templated, editable from the dialog, or
  otherwise app-owned (T-D4, T-D5).
- A path override for the test file (`E2E_TEST_SETTINGS_FILE`, or a file named by an
  env var) — T-D1's rejection.
- Making the e2e suite stub the model. The endpoint stays a real server, so the
  existing rule holds: assert on the *shape* of a stream, never on wording.
- Converting the backend pytest suite to a settings file. It keeps its `tmp_path` +
  `neutralise_workspace_settings()` style, which is stronger.
- Retiring `.env.test`, or merging it into `.env`.
- Any migration: the old group-1 lines in a developer's `.env.test` are deleted by
  hand in the same commit as the example file, exactly as D11 decided for `.env`.

## 8. How it is verified

Backend (`pytest`, hermetic, no reachable model):

- **`tests/test_workspace_settings.py`** — switch off answers
  `nojae-settings.json`; switch on answers `nojae-test-settings.json` at the
  **active** working location (a location switch moves it; D4 is unchanged); with
  the switch on and *both* files present holding different values, the reader
  answers the test file's values and never the other's (T-D3, no fallback); with
  the switch on and the test file absent, the state is `absent` and
  `config.workspace_settings_refusal()` names `nojae-test-settings.json`; an
  unknown member or a wrong type in the test file is `failed` (D9 applies to it);
  `save_workspace_settings()` refuses in test mode **and leaves the file's bytes
  untouched**; `ensure_workspace_settings_file()` still writes the production name
  in test mode (T-D4); `_self_check()` fails if the test table stops containing the
  production one.
- **`tests/test_env_isolation.py`** — the autouse fixture clears
  `BENCHMARKIT_TEST_SETTINGS`, so a value exported in the runner's shell cannot
  redirect the suite; and no test reads group-1 values from a file it did not
  create, asserted through the module's own constants.
- **`tests/test_api.py`** — in test mode `GET /api/settings` reports the test file's
  path in `workspaceSettings.path`; a group-1 `POST /api/settings` in test mode is
  refused and neither file changes.
- **Existing suites that must still pass unchanged**: `test_workspace_settings*`,
  `test_config`, `test_lsp`, `test_mcp*`, `test_editorfs*`,
  `test_write_surface_isolation` (the switch adds no write surface and reaches no
  write route), `test_llmkit_contract`.

E2E (`cd web && npm run test:e2e`, needs the test server reachable):

- `playwright.config.js` exits with its own sentence when
  `.nojae/nojae-test-settings.json` is absent — checked by running the suite with
  `BENCHMARKIT_PATH_LOCATION` pointed at an empty scratch root.
- **`web/e2e/03-workspace-switch.spec.js` is updated, not left to fail.** Its
  scratch-root block currently asserts the still-holds-the-template note (D8) and a
  hint naming `.nojae/nojae-settings.json`; under the switch both are simply
  different now. The dialog's reported path names `nojae-test-settings.json` at
  **both** roots; the boot root's `serverUrl` field equals the value the spec reads
  from `.nojae/nojae-test-settings.json` with `node:fs` — the assertion that pins
  "the suite is configured from the test file", and the one that would have caught
  this whole problem; the scratch root is the **`absent`** state ("there is no
  workspace settings file at … yet") instead of the template-text note, because the
  app does not seed the test file (T-D4); and the seeded-file list assertion stays
  **unchanged**, which is what T-D4 pins from the browser side.
- **What that costs**: the still-holds-the-template note (D8) loses its only
  browser-level assertion, because in test mode no workspace can ever be seeded into
  that state — the app seeds the *production* name only. The transfer is accepted and
  recorded: D8 stays covered by `tests/test_workspace_settings_seed.py` and
  `tests/test_workspace_settings.py` (the reader and the refusal sentence), and the
  dialog's rendering of a `message`-bearing state stays covered by the `absent` case
  the same spec now asserts — same computed property, same `.alert-error` element.
  Re-adding a browser assertion for the seeded-template dialog belongs to a future
  e2e run with the switch off, which is a manual pass, not a spec.
- One smoke that the suite really is configured: a WorkBench reply at the boot root
  streams a non-empty answer. A misconfigured test file then fails with the refusal
  sentence rather than a timeout.
- **A switched-to scratch root cannot chat or run in test mode** — it has no test
  settings file, so group-1 requests refuse with the `absent` sentence (T-D3, T-D4).
  No spec does this today (the scratch-root specs assert on files, tabs and dialogs,
  never on a reply), so nothing breaks; but it is the rule for future specs: **a spec
  that streams sends from the boot root**, the one root whose `.nojae` a human
  authored. This is the suite's version of "the endpoint comes from the fixture,
  never from wherever the app happens to be pointed".

Manual check, one machine, written into the README section as steps: run the suite
with the test file holding a deliberately wrong `endpoint` and confirm the failure
names the file; then with the file absent and confirm the suite refuses to start.

## 9. Sequencing with the `wss` lot

This lands **after ws-05 and before ws-06**: ws-06's e2e pass is the run that
currently reaches the developer's server, and ws-07's README section is the one that
would document `.env.test` as the endpoint source. It carries §6's corrections to
those two task files, so the lot cannot be resumed against the old mechanism.
Nothing in ws-01 … ws-05 needs redoing: the switch is additive to
`workspace_settings`, and the production file is the default in every case.
