# Target A — correct everywhere, honest about the rest

**Status**: planned, not started. **Depends on**: nothing (A is the first step).
**Blocks**: B and C. **Read first**: `windows-support-road_to_target.md` §2, §3, §4.
**Code referenced**: `ef21ea3`.

## 1. What A is for

A makes the app **correct on Windows for the surfaces it can support** — the
Benchmark tab, WorkBench chat, the verifier, and the Editor's file surface (tree,
read, search, git status, diff, create/rename/delete/backup) — and makes every
surface it cannot support **say so in words**.

The framing that matters: A is not "make Windows not crash". Three of the
failures in the roadmap §2 are **silently permissive** — a policy rule that stops
matching makes the verdict fall back to `rw`, so a `hidden` file becomes readable,
an `ro` folder becomes writable, and **the reserved `.nojae` folder stops being
reserved**. A is a security fix whose symptom happens to be Windows.

A does **not** make connectors, language servers or the terminal work on Windows.
It refuses them, in the same shape the terminal already refuses
(`termhub/session.py:137` → `api/routes.py:1655-1664` → a sentence in the UI).

## 2. Decisions taken

### A-D1 — Canonicalise *to* `/`; accept `\` only at the inbound lock

The canonical form stays `/`-joined workspace-relative text on every platform.
One function does the translation, at the point where a path enters the policy
system from a caller: `editorfs.policy.normalise_rel` (`src/core/editorfs/policy.py:180`).

*Rejected — normalise to the OS separator and fix the consumers.* That changes
the wire format sent to the browser (whose `split('/')` calls are correct today:
`useWorkspaceTree.js:88`, `:535`, `useEditorRename.js:50`), the form stored in
`results.json` and echoed in every `path` payload, and the frontend, all at once.
It also loses the property that a path string means the same thing in a Linux
test and a Windows bug report.

Concretely: replace `os.path.normpath(text)` (`policy.py:198`) with a separator-
agnostic normalisation that splits on both `/` and `\`, drops `.` and duplicate
separators, applies `..`, and re-joins with `/`. The existing `..`-survival check
(`policy.py:201`) keeps its meaning. Then fix the four places that consume the
form as if it could contain `\` — which, after this change, it never can:

| file:line | what it does today | why it now works |
|---|---|---|
| `policy.py:282` `normalised.split("/")[:-1]` | ancestor walk of the strict gate — empty on Windows, so the gate degrades to leaf-only (`road_to_target` §2.2) | input is always `/`-joined |
| `policy.py:370` `normalised.rpartition("/")` | parent of a create — empty parent, so a create lands at the root (§2.4) | idem |
| `search.py:113` `rel.rfind("/", 0, cut)` | `.gitignore` pruning from `b55dd20` — never fires (§2) | idem |
| `mutate.py:459` `workspace_root().rstrip("/") + "/"` | the reason **every folder delete is refused** — see A-D2 | replaced, not inherited |

Also `batch.py:216` (`os.path.normpath` on a lot folder name): normalise the same
way, so a Windows user typing `todos\drafts` gets the same folder as `todos/drafts`
and `taskLocation` strings stay `/`-joined for the model.

**Accepted input, deliberately widened**: `\` is a separator in an inbound path.
It is *not* a legal character in a POSIX filename, so this loses nothing on Linux
and it means the Editor, the API and the connectors all agree about what a path is.

### A-D2 — Absolute-path and containment locks use Windows-aware semantics

`os.path.isabs` and `os.path.realpath` are the right functions (they already mean
the OS's truth), but three holes follow from them being *OS*-relative rather than
*documented*:

1. **`normalise_rel`'s "must be relative" lock misses three Windows shapes.**
   Verified on 3.13 with `ntpath`: `isabs('/etc/x')` is **`False`**, so
   `path=/etc/x` is not refused as absolute — it normalises to `\etc\x` and then
   escapes the workspace through `os.path.join` (`join('C:\ws\a', '\etc\passwd')`
   → `C:\etc\passwd`), where the containment lock does catch it. The message is
   then *wrong* ("outside the working location" for something that was absolute),
   and the same class covers drive-relative input: `isabs('C:x')` is **`False`**,
   `normpath('C:x')` stays `'C:x'`, and **`ntpath.join(root, 'C:x')` returns the
   other drive's current directory** — a path on a drive the workspace is not on,
   which the containment lock catches by luck, not by intent.
   **Decision:** the lock rejects anything with a drive spec (`ntpath.splitdrive`
   non-empty) or a leading `/` or `\`, in addition to today's `isabs`/`~` check.
   The refusal sentence must name what was wrong, not just where it landed.
2. **Case**: every containment prefix compare
   (`policy.py:172`, `:216`, `lsphub/discovery.py:156`, `batch.py:228`,
   `routes.py:1298`) is exact-text. See A-D3.
3. **`~` is checked before expansion, which is right** — keep it, and keep
   `expanduser` only where it already runs (`routes.py:1291`, for the settings
   dialog's working-location field). No connector may expand it.

### A-D3 — Comparison folds case when the *filesystem* folds it (roadmap §4)

A probe answers one question per workspace root: **is this filesystem
case-insensitive?** Create a name and `stat` its folded variant; cache the answer
with the root and drop it when `BENCHMARKIT_PATH_LOCATION` changes (the same
moment the tree already forgets its cached listings).

Then: fold **only inside comparisons**, never in the stored/displayed/wire form.
`policy_glob._name_matches` (`mcp_connectors/policy_glob.py:219`) gains a folded
comparison path; `reserved_component_hit` (`:145`) and `_path_segments` (`:169`)
compare folded; the `os.sep` containment prefixes of A-D2 compare folded; and
`path_policy.component_hit` (`mcp_connectors/path_policy.py:120`) — which is
currently exact against `Path.parts` — compares folded.

The docstring at `policy_glob.py:226` (*"Comparison is exact — no case-folding,
the temptation that would make `Dist` and `dist` one folder"*) must be updated to
say the truth: exact where the filesystem is exact, folded where it is not. That
sentence is the reason nobody fixed this yet, and it is a good sentence about
Linux.

*Rejected — `os.path.normcase`.* It keys off the platform, so it makes the wrong
call for a case-insensitive mount on Linux and cannot be exercised on the Linux
runner at all (roadmap §9 rule 2).

### A-D4 — A guardrail that matches nothing is an error (roadmap §3, honesty half)

A does not write Windows guardrail lists — that is B. A makes sure that while
they are absent, the app does not claim a bound it does not have:

- At boot and on each hub reload, check whether the platform-shaped guardrail
  lists are meaningful (`path_policy.ALLOW_PREFIXES`/`DENY_PREFIXES`,
  `lsp_connectors/restricted-paths.txt`). On a Windows path form, POSIX prefixes
  match nothing → publish a **note** in the hub's own status, in the same words
  the declaration files already use for an unresolved `${NAME}`.
- **Local stdio connectors (`console`, `fedit`) and every language server are
  refused at start on Windows in A**, with a sentence naming the reason
  ("their reach cannot be bounded on this filesystem yet — see target B").
  A `local` entry gets an `error` status with that message, exactly as an
  unresolved variable does today. Remote `remote` HTTP connectors are unaffected:
  they have no local path reach, and their `restricted-fqdn.txt` guard is
  separator-free and already works.

*Rejected — leave the connectors running with a non-matching deny-list.* That
ships the silently-permissive shape the whole roadmap exists to remove.

*Rejected — refuse the app from starting.* The Benchmark tab, WorkBench and the
Editor all work; a missing connector capability is a note, like every other
declaration-file problem.

### A-D5 — Refusals, not crashes, for the rest of A's boundary

| surface | today on Windows | A's behaviour |
|---|---|---|
| Editor terminal | already honest (`os.name != "posix"` → `error` frame + `CLOSE_CODE_UNSUPPORTED`) | unchanged, and tested from Windows too — **superseded by target C**: the refusal is now the driver/shell **capability probe** behind `termhub/drivers.py` and `termhub/shells.py`, same `error` frame and same 4501, and a Windows shell now runs when the probe answers yes |
| Local MCP connectors | spawn `python mcp_connectors/x_mcp.py`; policy no-ops | refused, note naming target B (A-D4) |
| Language servers | spawn; `start_new_session` ignored; teardown kills the wrapper only | refused, note naming target B (A-D4) |
| `console` connector command set | `dir` is whitelisted but has no `dir.exe` → `FileNotFoundError` | not reached (row above) |
| folder delete | refused by accident (`mutate.py:459`) | **works**, after A-D1/A-D2 |
| `fsync_dir` | already swallows `OSError` (`files.py:373-380`) | unchanged |
| mode preservation | `os.chmod(temp, mode)` (`files.py:404`, `fedit_mcp.py:1186`) is a near-no-op on NTFS; `stat.st_mode & 0o170000` (`files.py:133`, `:463`) still classifies correctly on Windows | **A leaves it**, and records it: A must not lose the Linux mode round-trip (`0o7777` copy) — that is the only requirement |

### A-D6 — No `.env` key may collide with an OS variable (the `TEMP` class)

The rename `TEMP` → `TEMPERATURE` (`6a397d6`, `ef21ea3`) fixed one instance. The
rule is what A adds: **every name in `SETTING_KEYS` (`src/config.py:35`) must be
reserved-name-free on every supported platform**, checked by a test over a small
deny-set (`TEMP`, `TMP`, `PATH`, `HOME`, `USER`, `USERNAME`, `USERPROFILE`,
`SYSTEMROOT`, `COMSPEC`, `PROMPT`, `OS`, `APPDATA`, `LOCALAPPDATA`, `PROGRAMFILES`,
`COMPUTERNAME`, `HOMEDRIVE`, `HOMEPATH`, `NUMBER_OF_PROCESSORS`, `PATHEXT`).
`load_dotenv_once` (`config.py:257`) keeps `load_dotenv`'s "existing environment
wins" precedence — that precedence is *why* the collision was a folder instead of
a value — so the check belongs on the key list, not on the loader.

The rule is written next to `SETTING_KEYS` as a comment, because AGENTS.md already
says a new or renamed key means editing `SETTING_KEYS`, `_render_env()` and
`.env.example` together; this joins that list.

### A-D7 — The Windows user's templates must be usable

`lsp_connectors/lsp_servers.example.json:7,9` ships
`${VOLTA_BIN:-/usr/local/bin}/pyright-langserver` and
`PATH: "${VOLTA_BIN:-/usr/local/bin}:/usr/local/bin:/usr/bin:/bin"`. A is not
where servers start working, but a template that cannot load on the platform the
user is on is a bug in A's honesty: the file is what the UI offers them.

**Decision**: the example keeps the Linux form and gains a comment-adjacent note
in the README's Windows section (A-D8) giving the Windows form
(`${VOLTA_BIN:-.}\pyright-langserver.cmd`, `;`-separated `PATH`), *and* the `${NAME:-fallback}`
expansion in each hub's `discovery` module gains a test that a value containing
`\` survives expansion (it is text, so it should — verify, do not assume).
`lsp_connectors/restricted-paths.txt` gains a comment stating that the entries are
POSIX forms and that B supplies the Windows ones, so the file does not read as if
it were complete.

### A-D8 — A Windows section in the README, and one doc note

One section: how to install (`pip install -e .` in a venv or pyenv on Windows),
what works, **what refuses and why**, and the two template differences from A-D7.
Not a porting guide; a statement of the boundary. `docs/windows-support/` is
linked from it so the roadmap is findable.

## 3. What it builds on

- The existing lock/verdict split in `editorfs/policy_modes.py`'s module docstring:
  *"the locks run before this is consulted and are not configurable… A verdict can
  only ever narrow what the locks already allowed."* A keeps that order; it only
  makes the locks mean the same thing on both platforms.
- The single-question rule (`mode_for` is the one place the verdict is computed,
  pol-06). A must not add a second predicate — every change lands inside the
  matcher or the locks.
- `policy_glob` living in `mcp_connectors/` so both stdio connectors and the app
  share one matcher and one refusal sentence (pol-08). **A's matcher fix is free
  for the connectors**, even though B is what enables them.
- The declaration-file note mechanism (`discovery`: *"An unresolved name is kept
  verbatim with a note, never blanked"*) as the shape A-D4's refusal reuses.
- `web/src/composables/useTerminal.js`'s rendering of an `error` frame as the
  pattern for showing a hub note.

## 4. Shape and files of the pieces

### `mcp_connectors/path_norm.py` (new; shared, `src`-free)

Sits beside `policy_glob.py` for the same reason that module does: the two stdio
connector processes import it and cannot import `src`.

```python
def canonical_rel(text: str) -> str: ...      # both separators in, `/` out, `.`/dupes gone
def is_qualified(text: str) -> bool: ...      # drive spec, UNC, or leading separator
def path_parts(text: str) -> tuple[str, ...]: # one component-splitter for every consumer
```

No `sys.platform` branch in this module: it is pure text. `os.path.normpath`
disappears from the policy path (only `batch.py:216` and `policy.py:198` call it
today — verified).

### `src/core/editorfs/policy.py` (edit)

`normalise_rel` calls `path_norm.canonical_rel` + `is_qualified` (A-D1, A-D2.1).
The containment compares at `:172` and `:216` route through one
`_is_within(parent, child)` helper that applies the A-D3 fold.
`resolve_write_target:282` and `resolve_parent_in_workspace:370` use
`path_norm.path_parts` — after A-D1 they could keep `split("/")`, and they should
anyway: the ancestor walk and the matcher must agree by construction, not by
coincidence of separator.

### `mcp_connectors/policy_glob.py` (edit)

`_path_segments` uses `path_norm.path_parts`. `_name_matches`,
`reserved_component_hit` take a `fold: bool` (from the probe) threaded by callers
through `mode_for` → `_verdict` → the connector's own verdict call. The fold is
passed explicitly rather than read from a global, so the tests can force either
outcome on the Linux runner (roadmap §9 rule 2).

### `src/core/editorfs/fscase.py` (new, small)

The probe: `case_folded(workspace_root) -> bool`, cached per root, invalidated by
the same path-location switch that already re-roots the tree. Lives on the app
side because it touches disk; the matcher stays pure.

### `src/core/editorfs/mutate.py` (edit)

`_rel_in_workspace:459` stops hand-building a prefix: `Path.relative_to` cannot be
used (it is case- and separator-exact), so it asks the same
`policy._is_within(workspace, resolved)` helper and returns the remainder. This is
what un-breaks folder delete.

### `src/core/editorfs/search.py`, `src/core/batch.py` (edit)

`_ignored_hit:113` walks with `path_norm.path_parts` (or equivalently reuses the
matcher's cut loop — one implementation, per the pol-06 rule). `batch._clean_lot_dir`
uses `canonical_rel` and drops its `os.path.normpath`.

### `mcp_connectors/path_policy.py` (edit, honesty half only)

`component_hit:120` folds when told. Add
`def lists_are_meaningful(text: str) -> bool` — "does any of these prefixes speak
this path's form?" — used by A-D4's boot check. The Windows *values* arrive in B.

### `src/core/mcphub/discovery.py`, `src/core/lsphub/discovery.py` (edit)

One shared capability check: on a non-POSIX platform (or whenever A-D4's
`lists_are_meaningful` fails) a `local` entry / a declared server gets
`status: "error"` plus the sentence naming target B. Symmetric between the two
hubs, per AGENTS.md's "keep that symmetry" rule.

### `src/config.py` (edit) + `tests/test_config.py` (edit)

A-D6: a `_RESERVED_ENV_NAMES` frozenset next to `SETTING_KEYS` and a test asserting
the intersection is empty.

### Docs

`README.md` gains the Windows section (A-D8); `lsp_connectors/restricted-paths.txt`
and `lsp_connectors/lsp_servers.example.json` gain the notes from A-D7.

## 5. Out of scope for A

- Anything that makes a connector, language server or terminal **work** on
  Windows: B and C. A's deliverable for them is a refusal with a name.
- Job objects, `proctree`'s second implementation, `console_mcp`'s tokenizer and
  command set (B).
- ConPTY (C).
- Preserving NTFS attributes/ACLs through the Editor's atomic replace, or mapping
  `st_mode` to the read-only attribute. Recorded in roadmap §10; A only promises
  the Linux mode round-trip keeps working.
- Case-insensitive *git* path matching: `gitstatus` speaks git's own forms and
  `path.replace(os.sep, "/")` (`gitstatus.py:123`) is already right. The fold in
  A-D3 applies to policy comparisons, not to git output parsing.
- `.nojae` seeding semantics: `os.open(..., O_CREAT | O_EXCL)`
  (`src/core/agent_modes.py:887`, and the seeding in `api/server.py:35-58`) is
  already exclusive-create on both platforms and needs no change. A's
  reserved-folder work is the *matcher*, not the seeder.

## 6. Verification

**The gate first** (this is what makes B/C developable on Windows — roadmap §9):

- `tests/conftest.py` gains two markers, used across the suite:
  `@pytest.mark.requires_symlink` (skip when the platform will not let an
  unprivileged process make one — probe by trying once, not by name) and
  `@pytest.mark.posix_behaviour` (asserts a fact that only holds on POSIX: the
  `/etc/...` fixtures at `test_fedit_policy.py:50` and
  `test_console_policy.py:241-261`, the `chmod`/`st_mode` expectations at
  `test_fedit_policy.py:166-176`). `tests/test_terminal.py` already has the
  pattern (`needs_shell`, `:27`) and `tests/test_lsp.py` already has two
  `skipif`s (`:689`, `:700`) — generalise those, do not add a third convention.
- The 24 symlink call sites across 8 files (`test_editorfs_write.py` 9,
  `test_fedit_policy.py` 4, `test_editorfs_search.py` 3, `test_editorfs.py` 2,
  `test_editorfs_policy_file.py` 2, `test_batch.py` 2, `test_api.py` 1,
  `test_editorfs_builtin_policy.py` 1) get `requires_symlink`.
- **Rule: a skipped test must say which invariant it would have checked.** The
  reason string is part of the deliverable, because B and C are developed where
  those tests are silent.
- Definition of done, both platforms: `pytest` green with **no failures**; on
  Windows, skips only for the two marked reasons and their count reported in the
  PR/commit note; on Linux, the count is zero.
- Frontend: `cd web && npm test` unchanged (no platform code). Playwright:
  `web/playwright.config.js:23` defaults `PLAYWRIGHT_EXECUTABLE_PATH` to
  `/usr/bin/chromium`, and `web/e2e/global-setup.js:17` calls
  `execFileSync('npm', …)` — which is `npm.cmd` on Windows. Both get an OS check
  so the suite either runs or says why it did not.

**Behaviour tests A adds** (all runnable on Linux by forcing the fold, and with
`\`-shaped input):

1. `normalise_rel('src\\config.py') == 'src/config.py'`; the same for `a\\..\\b`,
   `a\\/\\/b`, and a leading `\`. (Covers A-D1; these are the regression tests for
   the roadmap §2 list.)
2. `resolve_write_target` refuses an `ro` folder rule when the leaf is *below* it,
   given `\`-separated input. **This is the test that fails today** and is the
   proof A-D1 landed.
3. `reserved_component_hit('.NOJAE\\x', '.nojae', fold=True)` hits; and the whole
   chain: a read of `.nojae\x` through the real gate answers 404, a write answers
   422 naming the rule. (Closes §2.1 — the reserved folder.)
4. The three Windows path shapes that slip past `os.path.isabs`
   (`/etc/x`, `//srv/share`, `C:x`) are each refused by `normalise_rel` with a
   message naming the shape, not by containment-by-luck (A-D2.1).
5. Folder delete under a folder rule works, and `_subtree_refusal` still names the
   offending descendant (A-D2/mutate).
6. `_ignored_hit` prunes an ignored ancestor given `/`-joined rel (guards `search.py`).
7. Case: one rule `dist/**` marked `hidden` blocks `Dist/app.js` when
   `fold=True` and not when `fold=False`; same for `.ssh` in `component_hit`
   (A-D3, both outcomes on the Linux runner).
8. The A-D6 reserved-name test; the A-D4 `lists_are_meaningful` check (a Windows
   form against POSIX prefixes is `False`, and the hub note appears for a `local`
   entry); the `SETTING_KEYS` round-trip with a `\`-heavy `BENCHMARKIT_PATH_LOCATION`
   (already correct per roadmap §7 — this test keeps it correct).

**Manual pass on Windows** (the point of A is that this list is short): boot the
app; Benchmark one question; WorkBench one streamed reply; Editor: tree, open,
search, create, rename, delete a folder, save a CRLF file and confirm no
translation; the terminal and the local connectors show their sentences.

## 7. What A must not break

Linux behaviour, exactly: the verdict for every existing rule in every existing
`.nojae/read-write-policy.json`, the 404-vs-422 asymmetry for hidden paths
(`policy.py:133`, `hidden_error`), the all-or-nothing subtree delete, the
`/`-joined payload form the frontend depends on, and the mode round-trip
(`st_mode & 0o7777`). The only user-visible Linux change is that `a\b` as a path
now means `a/b` instead of a filename containing a backslash — and no POSIX tool
in this app produces such a name.

**AGENTS.md**: when A lands, one bullet changes. *"Path rules come from two
levels, one verdict"* gains the sentence that the canonical path form is
`/`-joined on every platform and that rule comparison folds case only where the
filesystem does. Add the "a refusal names its reason" invariant once, not per
surface. Stay under ~300 lines; do not add a Windows section to AGENTS.md — that
belongs to the README.
