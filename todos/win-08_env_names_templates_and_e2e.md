# win-08 — The honesty chores: reserved env names, the Windows-shaped templates, and an e2e that says why it did not run

Three small things that share one quality — each is a place where the app (or
its test suite) **silently means something different on Windows** — and each is
too small to deserve its own session. Together they are still one bounded task:
no path-matcher code is touched here.

Depends on: win-01 (markers, for the Playwright part), win-03 (the `\`-heavy
location test asserts the canonical form win-03 created).
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` A-D6, A-D7,
§4 (the `src/config.py` + `tests/test_config.py` piece, and *Docs* for the two
data-file notes), and §6's frontend paragraph (Playwright).

## Files

| file | change |
|---|---|
| `src/config.py` | `_RESERVED_ENV_NAMES` frozenset **next to** `SETTING_KEYS` (`:35`), plus the rule as a comment |
| `tests/test_config.py` | the A-D6 test; the `\`-survives round-trip test |
| `tests/test_mcp.py` / `tests/test_lsp.py` (the expansion helpers live in `src/core/mcphub/validation.py`: `expand_env`, `expand_env_list`) | a test that a value containing `\` survives `${…}` expansion |
| `lsp_connectors/restricted-paths.txt` | a comment: these are POSIX forms, B supplies the Windows ones |
| `lsp_connectors/lsp_servers.example.json` | a note key stating the Windows form of the two `${VOLTA_BIN:-…}` lines |
| `web/playwright.config.js` (`:22-23`), `web/e2e/global-setup.js` (`:17`) | an OS check each, so the suite either runs or says why it did not |

## 1. A-D6 — no `.env` key may collide with an OS variable

The rename `TEMP` → `TEMPERATURE` (`6a397d6` → `ef21ea3`) fixed one instance.
What A adds is the **rule**: every name in `SETTING_KEYS` must be
reserved-name-free on every supported platform.

- A `_RESERVED_ENV_NAMES: frozenset[str]` **written next to `SETTING_KEYS`**,
  with the deny-set §6 names: `TEMP`, `TMP`, `PATH`, `HOME`, `USER`, `USERNAME`,
  `USERPROFILE`, `SYSTEMROOT`, `COMSPEC`, `PROMPT`, `OS`, `APPDATA`,
  `LOCALAPPDATA`, `PROGRAMFILES`, `COMPUTERNAME`, `HOMEDRIVE`, `HOMEPATH`,
  `NUMBER_OF_PROCESSORS`, `PATHEXT`.
- A comment stating the rule, joining the existing sentence about *"a new or
  renamed key means editing `SETTING_KEYS`, `_render_env()` and `.env.example`
  together"* (AGENTS.md). **This is the place a contributor reads the rule**, so
  it belongs in the comment, not only in the test name.
- The check lives on **the key list, not the loader**: `load_dotenv_once`
  (`config.py:257`) keeps dotenv's *"existing environment wins"* precedence on
  purpose — that precedence is *why* the collision produced a folder instead of
  a value. Do not add a warning or a fix-up inside the loader.
- The test asserts `set(SETTING_KEYS) & _RESERVED_ENV_NAMES == set()`, and (so
  the test is not a tautology) that the deny-set is non-empty and that
  `TEMP` is **not** in `SETTING_KEYS` while `TEMPERATURE` is — the historical
  case, pinned.

## 2. A-D7 — the templates the Windows user is offered

A is not where language servers start working, but *"a template that cannot load
on the platform the user is on is a bug in A's honesty: the file is what the UI
offers them."*

- `lsp_connectors/lsp_servers.example.json` keeps the Linux form (it is the
  default install shape) and gains a note **in the file** giving the Windows
  form: `${VOLTA_BIN:-.}\pyright-langserver.cmd` for the command and a
  `;`-separated `PATH`. Use a key starting with `_` — the file already treats an
  underscore key as a comment, not a server (`lsphub/discovery.py:85-87`), and
  there is a `_note` key today; extend or add beside it. **No JSON syntax
  change**: the file must still parse and still load with the same servers.
- `lsp_connectors/restricted-paths.txt` gains a `#` comment stating the entries
  are POSIX forms and that **target B supplies the Windows ones**, *"so the file
  does not read as if it were complete"*.
- **Verify, do not assume**: a test that a value containing `\` survives
  `${NAME:-fallback}` expansion, exactly as §6 asks. Both declaration files
  expand through the shared helpers in `src/core/mcphub/validation.py`
  (AGENTS.md: expansion runs *before* validation, and only the braced form is
  special). Assert a Windows path round-trips through `expand_env` and through
  `expand_env_list` (a `command` argv entry), and that a bare `$` stays literal.
- Add one test for the `\`-heavy working location:
  `BENCHMARKIT_PATH_LOCATION=C:\Users\vgr\dev\benchmarkit` survives
  `save_settings` → `dotenv_values` → `load_dotenv_once` unchanged. Roadmap §7
  says `_encode_value` (`config.py:620`) already does this — the test *keeps it
  correct*, it does not fix anything. `BENCHMARKIT_PATH_LOCATION` is
  **deliberately not a managed `SETTING_KEYS` key** (AGENTS.md), so this test
  reads/writes the env file directly and must not add the key to the template.
- Do **not** add Windows guardrail values, and do not touch
  `mcp_connectors/restricted-fqdn.txt` (separator-free, already works — A-D4).

## 3. §6 frontend — Playwright runs or explains itself

Two hard-coded POSIX assumptions:

- `web/playwright.config.js:22-23` defaults `PLAYWRIGHT_EXECUTABLE_PATH` to
  `/usr/bin/chromium`. On Windows: either resolve a platform-appropriate default
  or leave `executablePath` unset so Playwright uses the browser
  `npx playwright install chromium` pinned — the comment above it already
  describes that fallback as legitimate. Choose **one**, and make the other
  impossible by accident (if you leave it unset, say so in the comment so nobody
  re-adds the path).
- `web/e2e/global-setup.js:17` calls `execFileSync('npm', …)`, which is
  `npm.cmd` on Windows.

Requirement from §6: *"Both get an OS check so the suite either runs or says why
it did not."* A silent failure is the failure shape this whole roadmap exists to
remove. So: on an unsupported/unconfigured platform, **fail the run with a
message that names the reason and the fix** (a thrown error from
`globalSetup`, or Playwright's own skip with a printed reason — not a green run
of zero specs). `process.platform === "win32"` is the right check here: this is
not a path-policy decision, it is a "which binary is this" question, and there is
no probe worth writing for it (roadmap §9 rule 2 allows the named seam).

No e2e spec is marked skipped for Windows: **the specs are not the problem** —
the browser path and the build step are. Do not touch any `web/e2e/*.spec.js`.
`cd web && npm test` (Vitest) is explicitly unchanged by §6 — *"no platform
code"*. Do not add a Vitest skip.

## Acceptance

- `pytest -q` green with: the reserved-name test, the `\`-survives-expansion
  test, and the `\`-heavy `BENCHMARKIT_PATH_LOCATION` round-trip (§6's last
  bullet).
- `cd web && npm test` green, unchanged (no Vitest file touched).
- `cd web && npm run test:e2e` on Linux still runs its specs as before (this is
  the risky one — prove it, do not assume it).
- `python -c "import json; json.load(open('lsp_connectors/lsp_servers.example.json'))"`
  succeeds, and `load_lsp_servers()` on the example file reports the same server
  ids as before the note was added.
- `git diff --stat` shows: `src/config.py`, `tests/test_config.py`, the hub
  discovery/validation test file, `lsp_connectors/restricted-paths.txt`,
  `lsp_connectors/lsp_servers.example.json`, `web/playwright.config.js`,
  `web/e2e/global-setup.js`. **Nothing** in `src/core/editorfs/`,
  `policy_glob.py`, `path_norm.py` or `path_policy.py`.
- The README is *not* touched here — that is win-09, which writes the Windows
  form sentence this task's data-file notes point at.
- Move this file to `todos/done/` when the above is true.
