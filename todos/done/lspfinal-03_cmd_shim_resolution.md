# lspfinal-03 — resolve a declared command through `PATHEXT`, and refuse by name

A language server declared with the obvious command — the extensionless one a POSIX
user would write — **cannot spawn on Windows**, and the error the hub reports today
is a raw `WinError 193` that tells the operator nothing. This task implements B-D6's
third bullet for the LSP hub: resolve through `shutil.which` before the spawn, and
refuse with a sentence naming what was searched.

Depends on: nothing in this lot. **Blocks: lspfinal-06** (nothing works on Windows
until this lands). Do not confuse it with B-D6's *connector* half, which is out of
scope here — the big picture lists it.
Spec: `windows-support-target-B-connectors.md` **B-D6**, last bullet, and §6's
"a `.cmd` shim resolves via `PATHEXT`" and "`dir` is refused by name with a
sentence, not by `FileNotFoundError`".

## Verified on this machine — the exact failure

```
subprocess.run(['C:/Users/vgr/AppData/Local/Volta/bin/pyright-langserver'])
  -> OSError: [WinError 193] %1 n'est pas une application Win32 valide

subprocess.run(['C:/Users/vgr/AppData/Local/Volta/bin/pyright-langserver.cmd'])
  -> spawns fine
```

The extensionless file is a **`#!/bin/bash` script** (`volta run "$(basename $0)" "$@"`),
so it is not an executable image at all. Two things follow:

1. The error is **`WinError 193`, not `FileNotFoundError`** — the code in
   `_start_session` that formats `"Could not spawn language server: {exc}"` will
   carry a message the operator cannot act on, and a test written for the
   "command missing" case will not fire on it. Handle both, and name what was
   looked for in either case.
2. `shutil.which` **does** solve it, because it honours `PATHEXT` and skips the
   extensionless script (verified):

```
shutil.which('C:/Users/vgr/AppData/Local/Volta/bin/pyright-langserver')
  -> 'C:/Users/vgr/AppData/Local/Volta/bin\\pyright-langserver.CMD'
shutil.which('pyright-langserver', path=child_PATH)  -> the same .CMD
shutil.which('no-such-server', path=child_PATH)      -> None
```

## The rule, and the trap in it

- Resolve **`argv[0]` only** (the hub passes `argv[1:]` as plain arguments, which
  is right — an argument is not a program name).
- Ask `shutil.which` with **the environment the child will actually get**, not the
  app's. `_environment(info)` (`manager.py:209-216`) **replaces** `PATH` with the
  declaration's own `env.PATH` wholesale. Resolving with the app's PATH and
  spawning with the child's is two different searches, and the declaration can
  make the second one fail. Pass `path=` from the child environment, and state
  that coupling in the comment — this is the same "one parser, one verdict"
  principle the path policy runs on.
- **Never fall back to `shell=True`.** B-D6 says it, and it is the rule of the
  whole app's tool surface.
- On refusal, the sentence names **what was searched**: the command as declared and
  the `PATH` it was looked up in. A server left at `error` with that sentence beats
  a `WinError 193` in a log. The refusal must reach `info["status"]`/`info["error"]`
  the way the existing spawn failure does — visible on the settings card.
- A **POSIX** run must be untouched: `shutil.which` on an absolute POSIX path that
  exists returns it unchanged. Do not add a platform branch — resolution is a
  capability question with one answer on both sides. If an entry declares a
  command that exists but is not executable, keep the OS's own error rather than
  inventing one.

## Files

| file | change |
|---|---|
| `src/core/lsphub/manager.py` | resolve `argv[0]` via `shutil.which` against the child's environment, immediately before `client.start_io`; the refusal sentence |
| `src/core/lsphub/constants.py` | the search-paths note / message constant if one is needed — keep it next to the other hub sentences |
| `tests/test_lsp.py` | the tests below |

## Tests

- **A Windows path with no extension resolves to the `.cmd`** (or `.bat`) beside it.
  Build a `tmp_path` folder holding a `fake-server` file and a `fake-server.cmd`
  file and assert the resolved target is the `.cmd` — do **not** depend on Volta
  existing (hermetic rule).
- **A command that resolves to nothing is refused, not spawned.** `_start_session`
  must not be reached (the existing pattern: monkeypatch it to raise
  `AssertionError("spawn attempted")`). The status is `error` and the message names
  the command and the searched `PATH`.
- **`WinError 193` and `FileNotFoundError` both produce an actionable sentence**,
  not a bare OS code — a `windows_behaviour`-marked pair, mirroring §6's "refused
  by name with a sentence, not by `FileNotFoundError`".
- **The search uses the child's PATH.** Declare `env.PATH` pointing at a folder
  where the command exists and assert resolution succeeds; then point it at a
  folder where it does not and assert the refusal names *that* folder. This is the
  trap in the middle bullet — without it, the test passes for the wrong reason.
- **POSIX is unchanged**: an absolute existing command resolves to itself.

## Acceptance

- `pytest -q` green, **same skip count** as the baseline in the big picture; the
  new Windows-only tests carry the `windows_behaviour` marker and run here.
- A declaration naming the extensionless Volta command reaches a **spawned** server
  (or, while 04 has not landed, still the reach refusal — this task must not change
  that predicate; the resolution sits after it).
- The refusal sentence names the command and the searched paths, and no code path
  can produce `shell=True`.
- Nothing in `console_mcp.py` is touched (the connector half of B-D6 is out of
  scope; if you think otherwise, re-read the big picture's out-of-scope note and
  say so in the report instead of doing it).
- Move this file to `todos/done/` when the above is true.
