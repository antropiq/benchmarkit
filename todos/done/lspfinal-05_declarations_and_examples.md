# lspfinal-05 — the declaration files in a form that works on both platforms

The code can now start a language server on Windows; the **files a user is offered**
still describe only a POSIX one. `lsp_servers.example.json` has a `:`-joined `PATH`
and an `/usr/local/bin` fallback; this workspace's `lsp_servers.json` points at
extensionless Volta shims. Both halves are data, and both are the app's promise to
the operator — AGENTS.md: the declaration files are *the only source of truth for
what runs*, and a template that cannot load on the platform the user is on is a bug
in honesty (win-08 said this for target A; it is still true).

Depends on: `lspfinal-03` (resolution, so an extensionless command can work at all)
and `lspfinal-04` (so the servers are not refused while you are testing them).
Spec: `windows-support-target-B-connectors.md` §4 (the data-file row), B-D6's last
bullet; and `todos/done/win-08_…md` §2 for the precedent on the example file — it
asked for the note and the note was never written (the file still carries only the
Linux form).

## Verified on this machine, and it changes what to write

The live declarations set `"env": { "PATH": "C:/Users/vgr/AppData/Local/Volta/bin" }`.
`_environment()` **replaces** the child's `PATH` with that string wholesale
(`manager.py:215`: `env.update(info.get("env") or {})`). Measured consequences:

- with a **`/`-separated** single-entry PATH, `pyright-langserver.cmd` spawns but
  fails: `'volta' n'est pas reconnu…` — the Volta shim cannot find `volta`, because
  `volta.EXE` lives in `C:\Program Files\Volta`, which the replaced PATH excludes;
- with a **`\`-separated, `;`-joined** PATH that also carries `C:\Program Files\Volta`,
  the same command launches node.

So on Windows the separators are **`;`** and the PATH must carry every directory the
shim itself resolves through. Do not "fix" this by removing the entry's `env` — that
is the mechanism that lets a shim resolve `node` at all (the comment at
`_environment` says exactly that: *"volta shims need a PATH that resolves
`node`"*).

## Files

| file | change |
|---|---|
| `lsp_connectors/lsp_servers.example.json` | a Windows form the reader can copy, without breaking the Linux default. The `:`-joined `PATH` and the `${VOLTA_BIN:-/usr/local/bin}` fallback are POSIX-only as written |
| `lsp_connectors/lsp_servers.json` | this workspace's three entries, working on Windows |
| `README.md` | only the sentence the LSP section needs about the two forms — the full Windows write-up is `lspfinal-06`'s; keep this one line |

## Rules for what you write

- **Use `${NAME:-fallback}` rather than a literal machine path where a fallback can
  be honest.** But be precise about what that syntax does: `expand_env`
  (`mcphub/validation.py:25-48`) substitutes **one** variable; there is no
  concatenation, no conditional, and a fallback that is the *wrong platform's* path
  still lands in the value. A single template string therefore cannot be correct on
  both platforms for a `PATH`. State that limitation in the note rather than
  inventing a syntax — adding one is a change to the expansion contract and to
  AGENTS.md, and it belongs to neither this task nor a data-file task.
- **An underscore key is a comment, not a server** (`lsphub/discovery.py:85-87`),
  and the file already uses `_note`. Extend the existing top-level `_note` or add a
  `_windows_note` beside it — **no JSON syntax change**, the file must still parse
  and still load the same server ids.
- `lsp_connectors/lsp_servers.json` is **gitignored and machine-local** (AGENTS.md:
  these files may hold secrets, each has a committed template). Fixing it is for
  this machine's working setup; the example is what every deployment reads. Say in
  the report which of the two you changed and why the other is unchanged.
- Do not add a fourth server, do not re-enable anything that is off, and do not
  touch `mcp_connectors/` — this task is LSP declarations only.

## Tests

Data files need the loader asked about them, not a human read:

- `load_lsp_servers()` on the **example** file returns the same three ids as today,
  and the **same errors as today**. It cannot be a clean load: the placeholders
  `workspaceRoot: "/absolute/path/to/your/web"` do not exist, so those entries are
  skipped with *"workspaceRoot does not exist"*. A test asserting a clean load of
  the example is asserting the wrong thing and will push you to turn a placeholder
  into a real path — which would be worse. Test that the file **parses**, that the
  ids are unchanged, and that the note keys are still treated as comments.
- On the **live** `lsp_servers.json`, the acceptance is not a test but the run in
  `lspfinal-06` — do not write a test that depends on Volta existing (hermetic rule,
  and AGENTS.md on tests).
- If 01 added expansion to `restricted-paths.txt`, re-read this task's `${…}`
  guidance: the two files expand through **the same helper**, and a note in one
  file must not promise behaviour the other does not have.

## Acceptance

- `python -c "import json; json.load(open('lsp_connectors/lsp_servers.example.json'))"`
  succeeds and `load_lsp_servers()` on it reports the same server ids as before.
- On this machine, after 04, `load_lsp_servers()` on the **live** file reports
  **no note and no error** for `pyright`, `typescript` and `vue` — and one manual
  start of `pyright` (non-lazy) reaches `status: "ready"`, not `error`. Record the
  exact `command` and `env.PATH` values that made that true; that is the sentence
  the example's Windows note must carry.
- `git diff --stat` touches **only** the two `lsp_connectors/*.json` files and (at
  most) one README line. No `src/` file. If a code change seems necessary to make a
  declaration load, that is a finding for 03 or 04 — report it, do not fix it here.
- The example file still loads its servers on **Linux** exactly as before.
- Move this file to `todos/done/` when the above is true.
