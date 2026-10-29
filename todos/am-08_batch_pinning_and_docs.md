# am-08 — The runner pins its mode, then the docs and the sweep

Two things in one task, and they belong together: the last behaviour the feature
needs (a running lot decides its own mode, D13), and the documentation pass that
only makes sense once the shape is final.

Depends on: am-03 through am-07. Read this lot's **big picture** (task 0: it is
sent to you as system context, and its decisions table is the source the docs
summarise) and the shape `todos/done/pol-08_connectors_policy.md` gave the policy
docs, which is the closest precedent for "a config file the app seeds".

## 1. D13 — a batch task turn always runs in `code`

A lot is authored in Todo builder mode and *executed* by the runner. The runner
fires coding tasks: it must not inherit whatever mode the composer happens to show,
and it must not be at the mercy of a dropdown the user changed while task 4 was
streaming.

The seam is already there: `send()` and `sendBatchTurn()` both call `startTurn`
(`useChat.js:910`), which is where `payload.mcpEnabled` / `lspEnabled` and (from
am-07) `payload.agentMode` are built.

```js
// startTurn's option bag gains one field, in the batchContext's company:
//   agentMode — the mode this turn runs in. The interactive send passes the
//   composer's selection; a batch task turn passes the runner's pin, because a
//   lot is one job executed under one set of rules and the dropdown is not part
//   of it. null = "ask the server for its default" (the pre-feature shape).
async function startTurn({ userMessage, history, batchContext = null, agentMode = undefined })
```

`sendBatchTurn(text, { batchContext, agentMode })` forwards it; `useBatch`'s
`runOneTurn` (`:325-333`) passes the pinned id.

**Where the pinned id comes from.** Not a literal `"code"` in `useBatch.js` — the
mode id lives in the workspace's file, and a workspace whose author renamed the
coding mode would silently get a different agent for its tasks. Two steps:

```js
// useBatch.js — the mode a lot runs its tasks in. `code` is the *name* of the
// contract, resolved against the workspace's modes: a lot whose modes file has
// no such mode launches with the server's default instead, and says so.
export const LOT_TASK_MODE_ID = 'code'
```

1. at `launch()` (after `ensureToolsAvailable()`, `:378`), resolve it through
   `useAgentModes`: if `LOT_TASK_MODE_ID` is among the loaded modes, pin it for the
   whole run (`state.pinnedMode = LOT_TASK_MODE_ID`);
2. if it is **not** present (a workspace with a custom modes file), do not refuse
   the launch — leave `pinnedMode` null (server default), and put one line in the
   batch status area: `no "code" mode in this workspace; tasks run in "<default>"`.
   Refusing would make the feature able to break a lot that works today.

`REQUIRED_MCP_CONNECTORS` (`:68-71`) already refuses a launch whose `fedit`/
`console` connectors are off, and that check is the reason the pin can be lax: if
the pinned mode had `mcptools: false`, every task would fail the same way. So add
one clause to the *existing* `ensureToolsAvailable()` rather than a new guard:
when a mode is pinned and `mode.mcptools` is false, it is reported in the same
missing-tools list ("the `code` mode has no MCP tools; a task cannot move its own
file"). One list, one reason, one place the user learns why Launch is disabled.

## 2. D7's interaction with a running lot — restate, don't re-implement

am-07 disables the select while `batch.active`. That is a rendering rule, and this
task adds the *reason*, in one comment at each end:

* `useBatch.launch()` calls `ensureFreshChat()` (`:376`) → `resetChat()` → the D7
  lock is released by the reset, so a thread that was locked in Todo builder mode
  is unlocked the moment a lot starts. No conflict: a lot never runs *as* Todo
  builder;
* the pin is per-run, stored on `state.pinnedMode`, and read per turn — so a
  dropdown change during task 7 cannot reach task 8.

Then test the sequence end to end (§4): Todo builder → write a lot → open Batch →
Launch → every task request carries the pinned mode, and the select shows it
disabled.

## 3. Docs

Three places, in the order a reader meets them.

### `AGENTS.md` — one bullet, under "How a request flows"

This file is injected into every chat system prompt (it is the very thing the modes
sit on top of), so it stays short and it must describe the invariant, not the
member list:

```markdown
- **Every WorkBench turn runs in an agent mode** — a workspace-owned persona in
  `.nojae/agent-modes.json` that replaces `LLM_SYSPROMPT` for chat (never for the
  benchmark) and may switch a whole hub off. Settings ∩ conversation ∩ mode: the
  mode narrows, it never grants. AGENTS.md itself is always injected, and one
  mode is app-owned and not editable from the file.
```

Also update the two places that already speak about `.nojae` and `LLM_SYSPROMPT`,
each by *editing the existing sentence* rather than adding a new one:

* the `<working location>/.nojae/` row of the Data files table — it currently
  describes `read-write-policy.json` only; make it describe the folder's two files
  and their common rule (seeded on activation, hand-edited outside the app);
* the bullet that says `.env` is regenerated from a fixed template — add nothing
  (no new managed key), but if it reads as "the only config file is `.env`", fix
  that reading.

**Nothing in the `## Code structure` backend table gets a row** for the two new
`core/` modules unless the table is a complete listing of `core/` — check it; if
the table already omits several core modules, omitting these two is consistent and
cheaper. Do not grow this file past its stated ~300-line budget: `wc -l AGENTS.md`
before and after, and the after number must be under 300.

### `README.md` — one subsection

The README is for a human running the app, so it gets what the user can *see*: the
dropdown, the four modes in one table (id, label, tools, editable), the file path,
and the one escape hatch (`"systemMessage": ""` restores the configured prompt).
Include the seeded JSON verbatim — copy it out of `agent_modes_defaults.py` with a
script, do not retype it (a README that disagrees with the code about a config
file is worse than no README section). No implementation detail, no module names.

### `core/agent_modes.py` docstring — the design in four sentences

What a mode is, that the modes file is seeded and hand-edited like the policy file,
that a broken file never blocks chat (unlike a broken policy, which does), and
that the reserved mode is not in the file **on purpose**. The last one is the only
thing a future contributor cannot infer from the code, because the obvious
"consolidation" is to move it into the JSON.

## 4. Tests — the sweep

Per-task tests already exist (am-01…am-07). What this task adds is the cases that
only make sense with the whole feature in place:

**Python** (`tests/test_agent_modes.py`, one class):

* end-to-end prompt: `POST /api/chat` with `agentMode: "todoBuilder"` against a
  **seeded** file (call `ensure_agent_modes_file`, don't hand-write JSON) → the
  system message contains the persona, the AGENTS.md block, and the
  lot-conventions block, in that order;
* the same request with `agentMode: "chat"` on a workspace where **every**
  connector is Settings-enabled → no `tools` key, and the AGENTS.md block is still
  there — the two flagship invariants of am-00 in one request;
* the benchmark is mode-blind: assert nothing under `core/bench.py` or
  `core/llm.py` imports `agent_modes` (an `ast` scan, the style
  `tests/test_write_surface_isolation.py` already uses), poisoned with a fake
  import to prove the guard fails when it should;
* a modes file with an **extra unknown mode** (`"myhelper"`) is served by
  `GET /api/agent-modes` and selectable by `POST /api/chat` — the feature is
  workspace-extensible without a code change, which is the point of D8.

**Vitest** (`web/tests/`):

* `runOneTurn` passes the pinned mode into `startTurn` for every task, and the
  pin survives a dropdown change between turns (mutate the selection mid-run,
  assert the next request still carries the pinned id);
* a workspace with no `code` mode → `pinnedMode` is null, one status line, launch
  still proceeds;
* a pinned mode with `mcptools: false` → the launch guard fires and names the
  mode (the §1 clause), and it appears in the same list as the missing connectors.

**Playwright** (`web/e2e/04-agent-modes.spec.js`): boot → the select shows `code`
→ switch to Chat → the MCP panel is disabled and the select's hint names the
tools; switch back. No LLM call, no run.

## 5. Acceptance — the checklist for the whole feature

Run all four suites; a green `pytest` is not enough:

```
pytest
cd web && npm test
cd web && npm run build
cd web && npm run test:e2e
```

Then the greps, each of which is a promise someone else will later break:

| check | expected |
|---|---|
| `grep -rn "todoBuilder" web/src` | no hits |
| `grep -rn "agent_modes" src/core/bench.py src/core/llm.py` | no hits |
| `grep -rn "agent_modes" src/core/chat.py` | no hits (chat takes text) |
| `grep -c "agent_instructions=None" tests/test_chat.py` vs `mode_instructions=None` | equal |
| `wc -l AGENTS.md` | under 300 |
| `grep -rnE '"(mcptools|lsptools)": *[01]' src web docs README.md` | no hits — booleans, not `1`/`0` (D8) |

Finally, boot the app once by hand and read one real system message: log the
payload or hit the endpoint from `curl` with `agentMode: "architect"`, and confirm
with your eyes that the order in `am-00`'s §"The system message" is what actually
goes out. A test that asserts five substrings in order can still miss a doubled
blank line or a stray separator; this feature is made of prose, and prose is
checked by reading it.

## 6. Explicitly out of scope

* per-mode sampling, model choice, `maxIterations` (D10) — the vocabulary stays
  closed; adding a key later means am-01's unknown-key error fires first, which is
  the correct failure;
* a Settings UI for editing the modes file (D6) — the file stays hand-edited, and
  `GET /api/agent-modes` stays read-only;
* modes for the benchmark tab;
* moving `todoBuilder` into the JSON, in any form. If someone wants to, the
  reserved-mode test is the thing to argue with, not this file.
