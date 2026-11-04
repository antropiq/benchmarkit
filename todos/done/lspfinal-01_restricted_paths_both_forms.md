# lspfinal-01 — `restricted-paths.txt` in both platforms' forms, and expanded at read time

The file that is supposed to bound every language server's reach currently bounds
nothing on Windows, and its POSIX entries resolve into folders that do not exist on
this volume. This task makes the list real on **both** platforms. It changes no
matcher and no predicate: after it, the guardrail half of the LSP reach seam
answers `True` here **on its own**, with no code deciding the platform.

Depends on: nothing in this lot. **Blocks: lspfinal-04** — the seam must not be
lifted before this lands (the big picture's verified proof of what happens if it is).
Spec: `docs/windows-support/windows-support-target-B-connectors.md` **B-D1** (and
its two rejected alternatives — read them, they are why you must not add a
`restricted-paths.windows.txt`). The connector half of B-D1 is **already done** in
`path_policy` (G-D8); this is the LSP half.

## The current file, in full

```
/etc
/root
/proc
/sys
/home/vgr/.ssh
/home/vgr/.aws
/home/vgr/.config/gcloud
```

Seven entries, all POSIX text, three of them **machine-shaped** — a real user's
home directory committed into a file every deployment reads. B-D1's own words:
"`restricted-paths.txt` is worse than POSIX-shaped: it is *machine*-shaped".

## Files

| file | change |
|---|---|
| `lsp_connectors/restricted-paths.txt` | the list reworked: `${HOME}`/`${USERPROFILE}` forms for the profile stores, the Windows system prefixes, the `/home/vgr/*` lines dropped, the header comment rewritten |
| `src/core/lsphub/discovery.py` | `restricted_lsp_roots()` expands `${NAME}` / `${NAME:-fallback}` per line, through the shared helpers |
| `tests/test_lsp.py` | the new behaviour tests (`TestLoadLspServers` deny-list cases, `TestLocalReachRefusal`) |
| `tests/test_path_policy.py` | only if a shared-text assertion is the natural home for "an unexpanded `${…}` is not a prefix" |
| `src/config.py`, `.env.example` | **only if** a new managed key turns out to be needed — and if so, check it against both the `SETTING_KEYS`/`_render_env`/template rule *and* A-D6's `_RESERVED_ENV_NAMES` rule **before** adding it. Prefer needing none. |

## 1. Expansion is the part that is missing today

`restricted_lsp_roots()` (`lsphub/discovery.py:53`) reads lines and strips them.
There is **no expansion** — verified:

```
os.path.realpath('${HOME}/.ssh')  ->  'D:\\${HOME}\\.ssh'
```

So an entry written `${USERPROFILE}/.ssh` today becomes a literal folder named
`${USERPROFILE}` under the current drive. AGENTS.md and B-D1 both say the
declaration lists "accept `${NAME}` / `${NAME:-fallback}`, expanded at read time",
and `path_policy` already honours that for the connector lists via the environment.
The LSP list is the half that never got it.

- Use the **existing** helpers — `mcphub.validation.expand_env` (already imported
  into `lsphub/discovery.py` for the declaration values). Do not write a second
  expander; the two halves must expand identically.
- An **unresolved** reference keeps its current contract for declarations: kept
  verbatim with a note, never blanked. But for this file a verbatim `${…}` entry
  is *dangerous*, because it is a prefix that speaks the working location's root
  form while protecting nothing. State that in the code comment and assert it in a
  test: an entry whose reference does not resolve must not be allowed to satisfy
  `guardrails_speak_here` for a tree it cannot protect. If keeping the verbatim
  rule, drop such a line from the *effective* list and report it — a guardrail that
  cannot be honoured is absent, not guessed (the same sentence `_env_absolute`
  already applies to the connector lists).
- **Do not change `lists_are_meaningful`** and do not change `guardrails_speak_here`
  or `lsp_reach_is_refused`. Contract 2 of the big picture: this task passes by
  *writing values*. If you find yourself editing those, the design has gone wrong.

## 2. What the list must contain

- Profile stores on **both** forms, stated once each: `${HOME}/.ssh` and
  `${USERPROFILE}/.ssh` (same for `.aws`, `.config/gcloud`). One file, one list,
  both platforms — B-D1's rejected alternative was two files.
- The Windows system trees. Prefer the **same environment-derived names
  `path_policy` already uses** (`SystemRoot`, `ProgramFiles`, `ProgramW6432`,
  `ProgramFiles(x86)`, `ProgramData`) rather than spelling `C:\Windows` into a
  committed file: a drive letter is as machine-specific as `/home/vgr` was.
  Whatever mechanism you choose, the file must contain **no literal drive letter**
  and **no literal home directory** — say so in the header.
- The POSIX entries (`/etc`, `/root`, `/proc`, `/sys`) stay: they are correct and
  load-bearing on the other platform.
- The header comment must stop saying "the Windows forms … are target B's work
  too" — after this task they are here. Keep the sentence that forbids
  hand-writing `\` entries only if it is still true; if the file now *does* carry
  both forms, that warning is stale and must be replaced by what the format is.

## 3. The consumers you must keep working (all four)

Every consumer calls `os.path.realpath` on each line and then
`_is_within(entry, root, fold=…)` — component-wise, with the *workspace's* fold
answer. Reworking the list changes verdicts in all four, so assert each:

1. `load_lsp_servers` — a declared `workspaceRoot` inside a denied prefix is
   skipped with "workspaceRoot is on the deny-list".
2. `lsp_reach_is_refused` — the guardrail half now answers `True` here **without
   patching the seam**. Add the assertion that it does; that is this task's proof.
3. `editorfs.policy._check_workspace_root` — the working location itself is
   refused if it sits on the list.
4. `POST /path-location` (`api/routes.py`) — the same rule, one sentence.

## Acceptance

- On **this** machine (Windows, working location `D:\dev\benchmarkit`):
  - `guardrails_speak_here(restricted_lsp_roots() or None)` is **`True`**, with
    `local_reach_is_bounded()` untouched and still `False` — so
    `lsp_reach_is_refused()` is still `True` and the three servers are still
    refused. **This task does not start anything.** That is the point: prove the
    guardrail half turned around while the platform half still refuses.
  - a declaration with `workspaceRoot` `C:/Windows`, `C:/Windows/System32/config`
    or this user's `.ssh` is **skipped** by `load_lsp_servers` with the deny-list
    note. This is the case the big picture shows failing today.
- On **Linux**: the same four consumers give the same verdicts as before, and
  `restricted_lsp_roots()` still yields the POSIX entries.
- `pytest -q` green with **no new skips** (baseline for this machine: `test_lsp.py`
  `88 passed, 5 skipped`; `test_path_policy.py` + `test_proctree.py` `64 passed,
  4 skipped`).
- The new tests are named for the behaviour, not the mechanism — the class to
  extend is the deny-list group in `tests/test_lsp.py`, which patches
  `lsphub.constants.LSP_RESTRICTED_PATHS` (the hub's own constants, per the
  hermetic-tests rule — never a path it computed).
- Move this file to `todos/done/` when the above is true.
