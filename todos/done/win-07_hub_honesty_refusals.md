# win-07 — Hub honesty: a guardrail that matches nothing is an error, and Windows refuses the local hubs by name

AGENTS.md promises *"each connector's reach is bounded by a deny-list"*. On
Windows that sentence is **false and nothing says so**: `/etc`, `/usr`,
`/home/vgr/.ssh` match nothing, and a tool call on a protected Windows path is
refused only if its name happens to contain `.ssh`, `shadow` or `.pem`
(roadmap §3). A does **not** write Windows guardrail values — that is B. A makes
sure that while they are absent, the app does not claim a bound it does not
have, and refuses the surfaces that depend on them.

Depends on: win-01 (markers), win-06 (the fold convention and the containment
helper this task reuses for `restricted-paths.txt`).
Spec: `docs/windows-support/windows-support-target-A-no-crash.md` A-D4 and
A-D5's *Local MCP connectors* / *Language servers* rows; §4's
`path_policy.py` and `discovery.py` pieces; §6's `lists_are_meaningful` +
hub-note test and the `component_hit` half of behaviour test 7.

## Files

| file | change |
|---|---|
| `mcp_connectors/path_policy.py` | new `lists_are_meaningful(text: str) -> bool`; `component_hit` gains keyword `fold: bool = False` |
| `src/core/mcphub/constants.py`, `src/core/lsphub/constants.py` | the refusal sentence, as a named constant in each hub (mirrored) |
| `src/core/mcphub/discovery.py` | a `local` entry gets the refusal note on a platform where the guardrails are meaningless |
| `src/core/lsphub/discovery.py` | same for every declared server (they are all stdio) |
| `src/core/mcphub/manager.py`, `src/core/lsphub/manager.py` | the entry reaches `status: "error"` + that sentence, and is not started |
| tests: `test_mcp.py`, `test_lsp.py`, `test_console_policy.py`, `test_path_policy.py` (new or existing) | §6's hub-note test, the `component_hit` fold pair, the `lists_are_meaningful` pair |

## 1. `lists_are_meaningful(text) -> bool` — the zero-match check

"Does any of these prefixes speak this path's form?" A POSIX prefix list
(`DENY_PREFIXES`, `DENY_PREFIXES_READ`, `ALLOW_PREFIXES`, and the lines of
`lsp_connectors/restricted-paths.txt`) against a path in Windows form answers
`False`. §6: *"a Windows form against POSIX prefixes is `False`"*.

- One function in `path_policy.py` (A §4), consulted by **both** hubs — the two
  hub modules mirror each other and must not each grow their own copy
  (AGENTS.md: *"Keep that symmetry when touching either one"*).
- No Windows *values* here (A §4: *"The Windows values arrive in B"*). A adds
  the check, not the lists.
- Like everything in this lot, decide by shape, not by `sys.platform`, where a
  shape can decide it: the function's answer comes from comparing the form of the
  prefixes to the form of the path. The hubs additionally know the platform
  (see §3) — that seam is named and single.

`component_hit(parts, entries, *, fold: bool = False)` is the other half of
§6 test 7: `.ssh` in `C:\Users\VGR\.SSH\id_ed25519` must be refused when the
filesystem folds. `parts` comes from `Path(...).parts`, so the entries' `/`
split stays as it is; only the comparison folds. The connectors themselves keep
`fold=False` (big picture, "Fold threading") — this parameter is what lets a
Linux test prove both outcomes, and B passes real values.

## 2. The refusal, in the shape the codebase already uses

A-D4: a `local` entry gets `status: "error"` with that message, *"exactly as an
unresolved variable does today"* — compare how `mcphub/discovery.py:113` appends
`unresolved_env_note(prefix, missing)` to `errors`, and how `lsphub/discovery.py`
appends `f"{prefix}: workspaceRoot is on the deny-list — skipped"`. Same channel,
same sentence style (`{file}: '{id}' … — skipped`), no new payload key.

The sentence must **name the reason and point at the capability**, e.g.:
*"their reach cannot be bounded on this filesystem yet — see target B"*. It lands
in two places per hub, and they are the same string:

- the validation note list (`externalsError` / `serversError` in the hub
  payloads, already rendered by `useBenchmark.js` as `mcpExternalsError` /
  `lspServersError`);
- the per-entry `status`/`error` of the hub itself, so the connector card shows
  it in the same place a failed connection shows its error.

Put the sentence in `mcphub.constants` and `lsphub.constants` (the
`CONNECTORS_MISSING_HINT` / `LSP_MISSING_HINT` precedent exists at
`mcphub/manager.py:82` for exactly this shape: a known-absent capability
reported per entry, not a crash).

## 3. Where the platform decision lives

A-D4's check runs *"at boot and on each hub reload"*. Both discovery modules are
called on boot (`api/server.py`) and on every reload (`GET /mcp`,
`POST /mcp/restart`, `GET /lsp`, …), so putting the check in `discovery` gets
both for free and keeps the hubs symmetric.

The condition is: the platform cannot bound a local process **or**
`lists_are_meaningful` says the guardrails speak nothing. The first half is a
genuine `os.name != "posix"` fact (no probe can answer "would
`restricted-paths.txt` protect this tree"), which roadmap §9 rule 2 allows — but
it must be **one named seam**, not a scattered `if`: a single helper
(e.g. `local_reach_is_bounded() -> bool`, mirroring the existing
`os.name == "posix"` guards at `lsphub/manager.py:654` and
`termhub/session.py:137`), called by both hubs.

Consequences to honour:

- `enable: true` in `externals.json` does **not** start a `local` connector on
  that platform — the manager must not spawn it (A-D5: spawn = a policy no-op).
  Remote `remote` HTTP connectors are **unaffected**: no local path reach, and
  their `restricted-fqdn.txt` guard is separator-free and already works.
- Language servers are all stdio, so **every** declared server is refused the
  same way. Their tools must not be published to chat while refused — an
  "activated language server is never invisible" rule (AGENTS.md) has two
  documented exclusions, `enable: false` and `error`; a refused server is
  `error`, so it is out by the existing rule. Verify that rather than assuming
  it: the lazy-revival path revives a server with no live session *unless* it is
  `error`.
- **The refusal is a tested behaviour** (roadmap §9 rule 3): the test asserts
  the sentence *and* the status, for both hubs.
- The terminal is untouched (A-D5: *already honest, unchanged, and tested from
  Windows too* — win-01's markers are what makes its existing refusal test
  runnable there).

## 4. Tests

- `lists_are_meaningful`: a Windows form (`C:\Users\vgr\.ssh\id_ed25519`) against
  the default POSIX `DENY_PREFIXES` → `False`; a POSIX form → `True`.
- §6 test 7's second half: `component_hit` with `.ssh` matches
  `C:\Users\VGR\.SSH\id_rsa` when `fold=True`, not when `fold=False`.
- The hub note: with the platform seam forced to "unbounded", `load_externals()`
  reports a note naming target B for the `local` entries and no note for the
  remote ones, and `McpManager.status()` carries `status: "error"` + that
  sentence for the local entry while an already-running remote stays `running`.
  Force the seam with `monkeypatch` (through `mcphub.constants` /
  `lsphub.constants` — AGENTS.md's patchability rule), never by pretending to be
  another OS.
- The same shape for `load_lsp_servers()`: every declared server gets the note
  and the status, and the two sentences are **the same words** — assert the two
  constants against each other, that is the symmetry test.
- Nothing spawns: with the seam forced, `BENCHMARKIT_DISABLE_MCP`/`_LSP` stay
  unnecessary for these entries (they stay set anyway by `conftest`).

## Rules this task must not break

- **The declaration files are the only source of truth for what runs.** Do not
  add an `.env` key, a UI switch, or a second enable mechanism: the `enable`
  flag stays the switch and this is a *status*, not a setting.
- **Secrets never leave the server**: `headers`, `env`,
  `initializationOptions` stay stripped from the payloads
  (`routes.py:946-952`). A refusal note must not echo a command line that could
  carry one.
- An unresolved `${NAME}` stays *"kept verbatim with a note, never blanked"* —
  the new note joins that list, it does not replace it.
- Hubs are long-lived singletons: a restart rebuilds the *internals*, never
  swaps the object in `app.state`. Do not re-shape `McpManager`/`LspManager`
  identity here.
- `console_mcp`'s command set and tokenizer stay untouched (A §5, that is B) —
  the `dir` row of A-D5 is "not reached (row above)".
- `mcphub/` and `lsphub/` stay mirror-shaped: `constants`, `discovery`,
  `manager`. Add the piece to both or neither.

## Acceptance

- `pytest -q` green, including the hub-note pair and the `component_hit` fold
  pair.
- On Linux (the runner's platform) **nothing about the hubs changes**: every
  existing `test_mcp.py` / `test_lsp.py` case passes untouched, no note appears,
  no connector is refused. Prove it by asserting the seam's default answer is
  "bounded" on POSIX in the same test file.
- `grep -n "os.name\|sys.platform" src/core/mcphub src/core/lsphub` shows the
  decision in **one** named place per concern, not repeated at each call site.
- The refusal sentence names the capability gap and points at target B; the two
  hubs' strings are asserted identical.
- No `proctree.py`, no `console_mcp.py`/`fedit_mcp.py` behaviour change, no
  `restricted-fqdn.txt` change, no frontend change (the payload key is already
  rendered).
- Move this file to `todos/done/` when the above is true.
