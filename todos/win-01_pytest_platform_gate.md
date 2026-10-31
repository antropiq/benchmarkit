# win-01 — The pytest platform gate: two markers, and the suite becomes runnable on Windows

`pytest` on Windows is currently **not a usable signal** (roadmap §9: 8 test
files create symlinks, two more assert POSIX-only facts). Everything else in
this lot has to be checkable on both platforms, so this task lands first: it
adds the two markers the rest of the lot's tests use, and marks what already
needs them. Pure test-side work — **no `src/` file changes in this task**.

Depends on: nothing.
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` §6, first
block ("The gate first"), plus roadmap §9's last paragraph.

## Files

| file | change |
|---|---|
| `tests/conftest.py` | register the two markers + the symlink probe |
| `tests/test_editorfs_write.py` (9 sites), `tests/test_fedit_policy.py` (4), `tests/test_editorfs_search.py` (3), `tests/test_editorfs.py` (2), `tests/test_editorfs_policy_file.py` (2), `tests/test_batch.py` (2), `tests/test_api.py` (1), `tests/test_editorfs_builtin_policy.py` (1) | `@pytest.mark.requires_symlink` on every test that calls `.symlink_to(...)` |
| `tests/test_fedit_policy.py`, `tests/test_console_policy.py` | `@pytest.mark.posix_behaviour` on the POSIX-fact tests |
| `tests/test_terminal.py` | generalise `needs_shell` (`:26-29`) — see §5 |
| `tests/test_lsp.py` | generalise the two existing `skipif`s (`:689`, `:700`) |
| `pyproject.toml` | only if you register the markers there instead of in conftest — either place is fine, but they must be **registered** so `--strict-markers` is clean |

## 1. The two markers

Register both in `tests/conftest.py` via `pytest_configure(config)`:

- **`requires_symlink`** — skip when the platform will not let an *unprivileged*
  process make a symlink. **Probe by trying once, not by name**: a helper that
  attempts `os.symlink(target, link)` in a temp dir (covering both a plain link
  and the `target_is_directory=True` form the tests use) and remembers the
  verdict for the session. Windows without Developer Mode /
  SeCreateSymbolicLinkPrivilege raises `OSError`; Linux and macOS pass.
  **Do not write `sys.platform == "win32"` here** — roadmap §9 rule 2, and the
  probe is the whole point.
- **`posix_behaviour`** — skip when `os.name != "posix"`. This one is a name
  check *by design*: the facts it guards (`/etc/shadow` exists, `chmod` is
  meaningful) have no probe that answers them, and roadmap §9 rule 2 says such a
  difference sits behind one named seam — that seam is this marker.

The probe must be evaluated **lazily** (a fixture, or a condition object whose
attempt happens on first use, not at conftest import): a collection-only command
must not touch the filesystem. Use **one** mechanism for the probe — a helper in
`conftest.py` that both markers consult. A per-file copy of it is how a guard
drifts from what it guards.

## 2. The rule for a skip's reason string

From §6: **"a skipped test must say which invariant it would have checked."**
The reason is part of the deliverable, because B and C are developed where these
tests are silent. Each skip therefore carries its own sentence — the batch one,
for instance: *"a symlinked lot file must never be read, whatever the platform
allows"*. Where a shared marker reason is not specific enough, use
`@pytest.mark.skipif(<same probe condition>, reason="…")` rather than dropping
the information. A bare marker is fine when the test name already says it.

## 3. The 24 symlink sites

`grep -c "symlink_to(" tests/<file>` reproduces the counts above; they match §6.
Mark the **test function** that creates the link (or `pytestmark` on a class
that is entirely about symlinks). Three traps:

- `tests/test_editorfs_write.py:634` and `:929` create **dangling** links
  (`symlink_to(workspace / "never-existed")`) — still a symlink call, still
  marked.
- `tests/test_editorfs_builtin_policy.py:87` links *to the app root* with
  `target_is_directory=True` — the probe must cover the directory form too, and
  this test guards `is_self_workspace()`'s realpath equality, so its reason
  string must say that.
- Tests that **assert** `is_symlink()` after a delete/rename refusal (e.g.
  `test_editorfs_write.py:998`, `:1200`) need the marker for the same reason:
  without the link there is nothing to assert.

Do not weaken any assertion to make it platform-free. Skipping is the answer;
"assert less" is not.

## 4. POSIX-fact sites

- `test_fedit_policy.py:45-55` (`test_reads_of_credential_stores_are_refused`,
  the `/etc/shadow` fixture at `:50`) and the `chmod`/`st_mode` round-trip at
  `:160-176`: `posix_behaviour`.
- `test_console_policy.py:241-261` (the `/etc/shadow` and `~/.ssh` refusals), and
  anything else there whose subject is a POSIX path: `posix_behaviour`.
- When one test mixes a POSIX path with a separator-independent rule, mark the
  test **and** name in the reason which half is being skipped.
- The Linux mode round-trip is **not** lost: A §7 requires `st_mode & 0o7777`
  copying to keep working. Marked ≠ deleted — on Linux these tests still run.
- `test_console_policy.py:262-268` (`cat /etc/hosts`, `ls /tmp/scratch` allowed):
  check whether the *rule* it proves is about the path existing or about the
  deny-list. If the latter, keep it running and change only the input.

## 5. Generalise what already exists — do not add a third convention

`tests/test_terminal.py:26-29` (`needs_shell`) and `tests/test_lsp.py:689`
(`skipif os.name != "posix"`, "process groups are POSIX") and `:700`
(`skipif not sys.platform.startswith("linux")`, "uses /proc") already are the
pattern. Re-express them through the same helper/marker where the meaning
matches; where a fact is genuinely a third one (`/proc` only, not all of POSIX),
keep its own `skipif` — but built on the same helper, with its own reason. Say
which you chose in the commit note. The existing reason strings are good; keep
their sentences.

## Acceptance

- `pytest -q` on Linux: **green**, and **zero** skips for the two new reasons
  (`pytest -q -rs` shows no `requires_symlink` / `posix_behaviour` skip — Linux
  has symlinks and is POSIX). Pre-existing environment-driven skips (missing
  `[lsp]` extra, no shell) may stay.
- The suite passes with `--strict-markers`: no unknown-marker warning.
- Every line `grep -rn "symlink_to(" tests --include=*.py` lists sits in a test
  (or class) carrying `requires_symlink`, or inside the conftest probe itself.
- `grep -rn "skipif" tests --include=*.py` — each remaining ad-hoc `skipif` names
  an invariant the two markers do not cover, and shares the one probe helper.
- **No `src/`, `mcp_connectors/`, `lsp_connectors/` or `web/` file is touched**:
  `git diff --stat` shows only the test files above (+ `pyproject.toml` if you
  registered markers there).
- Nothing functional moved, so the app still runs as before.
- Move this file to `todos/done/` when the above is true.
