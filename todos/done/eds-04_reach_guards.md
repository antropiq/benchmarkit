# eds-04 — The reach guards: nothing but the browser can name the save route

**Spec:** D1's last bullet ("the isolation suite gains the sibling check the pair
needs"), acceptance item 9.
**Depends on:** eds-03 (the pair exists and is registered — there is nothing to
guard before it does).
**Files it touches:** `tests/test_write_surface_isolation.py`,
`tests/test_editorfs_write.py` (only if the caller scan needs its allowlist
reconciled with eds-02's choice), `web/tests/` if a client-side reach check
belongs there. **No production code** unless a guard finds a real leak — if one
does, fix the leak and say so in the commit.

## What to deliver

The isolation suite asserts, for the new pair, the same class of promise it
already makes for `/api/fs`:

* **no advertised tool names it** — the MCP connectors' live `list_tools`
  listings (including their `instructions` text) and the LSP hub's published tool
  defs carry neither the save path nor a helper spelling of it;
* **no code path spells a call to it** — the save route joins the write-marker
  vocabulary the source scan walks (the existing markers are the verb plus the
  path, chosen so a docstring naming a *method* stays out of the scan; keep that
  property when you add one), and the scan's allowed homes stay "the routes plus
  the browser callers of that route" — the composables that will call it are
  eds-05's, so if the marker must be live before they exist, add the composable
  name to the allowlist in the same spirit `useEditorMove.js` was declared there
  "with the route, before its caller exists";
* **the model client cannot reach it** — the `llmkit` URL-shape guards already
  pin that the two builders append only their two fixed suffixes and never a path
  under the app's route prefix; extend the poisoned-builder self-check to name
  the save route, so the guard is shown to see *this* path;
* **the core writer is called from the routes and nowhere else** — the sibling of
  the existing "the write core is called from the routes and nowhere else" scan,
  over eds-02's public entry point;
* **the terminal is not a side door** — nothing new here, but confirm the
  terminal-stays-human-only class still reads true now that a reserved-folder
  file is app-writable: the app's one shell is a human surface, never a tool.

Every new guard gets a poisoned case in `TestTheGuardsActuallyFail`, following how
that class feeds each check a deliberately bad input. AGENTS.md's rule for this
file is that it *"asserts that instead of trusting this sentence"* — a guard that
cannot fail is not a guard.

## Rules this task must not break

* AGENTS.md: *"Tools are read-only by design. Every tool returns information, the
  shell connector execs a single command with no shell… and each connector's reach
  is bounded by its own deny-list."*
* AGENTS.md: *"The Editor's write surface is the browser's alone… Batch mode lists
  its lot of task files through the shared path policy but never reads or writes
  their contents."*
* The scan roots stay what they are (`src/core`, the two connector trees,
  `useChat.js`). Do not widen a scan so it condemns `llmkit`'s legal `httpx` —
  that decision is written in the file's docstring and the URL-shape guards are
  its answer.
* Tests stay hermetic: the connector listings come from `list_tools` with the hubs
  disabled, the LSP defs from `TOOL_SPECS` through `tools_for_server` against the
  committed template — no live session.

## How to tell it is done

`pytest tests/test_write_surface_isolation.py tests/test_editorfs_write.py -q`
green, and each new guard proven to fail by its poisoned case (run the
`TestTheGuardsActuallyFail` tests specifically). `pytest -q` green overall. Move
this file to `todos/done/`.
