# tfix-04 — `test_a_symlinked_task_does_not_offer_its_prefix` errors on Windows

**Failing test id**

`tests/test_batch.py::TestPrefixListingEndpoint::test_a_symlinked_task_does_not_offer_its_prefix`

**Reported failure (verbatim)**

```
OSError: [WinError 1314] Le client ne dispose pas d’un privilège nécessaire:
'C:\\Users\\vgr\\AppData\\Local\\Temp\\...'
```

(The localized Windows message is the "A client process attempted to create a
symbolic link but does not have the required privilege" one — the failure is the
test's own `symlink_to` call, not the app's.)

**Where it is**

* Test: `tests/test_batch.py`, class `TestPrefixListingEndpoint`, test name
  `test_a_symlinked_task_does_not_offer_its_prefix` (around line 290).
* What it does: builds a lot with `make_lot(tmp_path, ["zz-00-real.md"])`, then
  `(folder / "zz-01-link.md").symlink_to(folder / "zz-00-real.md")`, then asserts
  `prefixes(batch_client).json()["prefixes"] == []` — the real file is the `00`
  big picture, the only task is a symlink, so nothing runnable is offered.
* Helpers in the same file: `make_lot` (line 25), `batch_client` (line 46),
  `prefixes` (line 219), `listing` (line 53).
* Subject under test: the prefix-listing endpoint in `src/api/routes.py` backed by
  `src/core/batch.py` (the lot listing — "task files are listed through the path
  policy, never read").

**What to deliver**

On this machine an unprivileged process cannot create the symlink the test plants,
so the test body raises before it ever reaches the assertion. Make this test pass
on Windows while it stays green on Linux, using the suite's **existing** mechanism
for exactly this situation rather than inventing a new one.

Look at the sibling symlink tests in the same file — `test_a_symlinked_task_file_is_skipped_not_read`,
`test_a_symlinked_dir_cannot_escape_the_workspace` and their neighbours (around
lines 433–466) — and at `tests/conftest.py`, which defines the `requires_symlink`
marker, the probe behind it, and the rule that a skip must name the invariant it
gives up. Whatever this task lands on, it must be consistent with how those tests
already answer the same platform fact.

**Files it touches**

`tests/test_batch.py` (the test and/or its marker). Only if the investigation
shows the *listing* itself mishandles a symlink on Windows should it also touch
`src/core/batch.py` or `src/api/routes.py` — and then the endpoint's existing
refusals (404 / 422 for a dir above the workspace, 422 for a symlinked lot dir)
must stay exactly as they are.

**Rules this task must not break** (quoted from `AGENTS.md`)

* "A task is finished when its file leaves the lot folder; re-listing is the only
  way the loop can see it." — the listing's behaviour is the loop's contract; do
  not change what it reports about symlinks.
* "Batch mode lists its lot of task files through the shared path policy but
  never reads or writes their contents — only the connectors do."
* "A hidden path is indistinguishable from an absent one to a reader and named to
  a writer" — a skipped symlink is still named in `skipped` by the task listing.
* "Tests stay hermetic": no model call, no real workspace.
* A skipped test must say which invariant it would have checked (the conftest's
  own rule for guard markers) — a bare, reasonless skip is not acceptable.

**Depends on:** none.

**How to tell it is done**

1. `pytest "tests/test_batch.py::TestPrefixListingEndpoint::test_a_symlinked_task_does_not_offer_its_prefix"` no longer errors on this machine — it either passes or skips with the suite's own symlink sentence.
2. `tests/test_batch.py` still passes in full (run that file only, never the
   suite), and no other test in it gained or lost coverage of the symlink rule.
3. If the outcome is a skip, the skip's sentence names the invariant lost, in the
   conftest's existing style, and the same marker is what the sibling symlink
   tests already use.
4. Move this file to `todos/done/`.
