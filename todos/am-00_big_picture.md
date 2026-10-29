# Agent modes — big picture

Give WorkBench a **mode**: a named persona that decides *what the agent is for*
in this conversation — its system message, and whether the MCP and LSP hubs are
available to it at all. Four modes ship: **Chat**, **Code**, **Architect** and
**Todo builder**. Three of them are user-editable per workspace in
`<workspace>/.nojae/agent-modes.json`; the fourth, `todoBuilder`, belongs to
benchmarkit and is **not in that file and not editable** (D9).

Why it exists: today every conversation gets the same system message —
`LLM_SYSPROMPT` + AGENTS.md + whatever tools Settings happens to have enabled —
so the same model is asked to write code, hold a conversation and draft a spec
with identical instructions and identical reach. The prompt is the only place
those differ, and it is one global string shared with the benchmark.

## The four modes

| id | dropdown label | persona | `mcptools` | `lsptools` | editable |
|---|---|---|---|---|---|
| `chat` | Chat | answer; never touch the workspace | no | no | yes |
| `code` | Code | implement, test, prove it | yes | yes | yes |
| `architect` | Architect | write a markdown spec, never implement | yes | no | yes |
| `todoBuilder` | Todo builder | write a `todos/` lot, never implement | yes | no | **no** |

`code` is what the app does today, minus the missing persona line; keeping it the
`default` means the feature changes nobody's current behaviour by accident.

## The system message, unchanged in shape

**No banners, no reordering.** `core/chat.py:211-240` already builds
`sysprompt → TOOL_NOTICE → AGENTS.md → batch big picture`, and that order is
kept exactly. A mode does two things to it:

1. its `systemMessage` **replaces** `LLM_SYSPROMPT` (decision D1) — empty falls
   back to the configured prompt, then to `DEFAULT_LLM_SYSPROMPT`;
2. its generated blocks append after AGENTS.md and before the batch big picture
   (only `todoBuilder` has any: the lot conventions, D9).

So the final order is: **mode persona → tool notice → AGENTS.md → mode blocks →
lot big picture**. `tests/test_chat.py` and `tests/test_config.py` pins on that
sequence stay valid; only the first element's *source* changes.

**AGENTS.md is always injected** (`config.agent_instructions_block()`,
`routes.py:1689`). No mode, no JSON key and no request field can suppress it —
and this stays true. am-04 makes it a test rather than a sentence.

## Decisions

| # | decision | why |
|---|---|---|
| D1 | mode `systemMessage` **replaces** `LLM_SYSPROMPT`, empty → fall back to it | two stacked personas contradict each other; `LLM_SYSPROMPT` stays authoritative for the benchmark, which is not mode-aware at all |
| D2 | keep the existing block order, no cosmetic separators | the order is already right and is pinned by tests |
| D3 | a mode with `mcptools: false` passes **no `tools=` kwarg** and gets no `TOOL_NOTICE` | "tools not in the system message" is not a restriction; the schema must not reach the model |
| D4 | three levels of gating: **mode ∩ conversation ∩ Settings** | the mode is the coarsest switch (whole hub off), Settings stays the authority, the per-thread toggles stay the finest. Zero new plumbing: a gated hub yields an empty allowlist in `_chat_tool_kwargs` |
| D5 | the id is `todoBuilder`, the label is **"Todo builder"** | "Batch mode" already means the *runner*; this mode *authors* the lot it runs |
| D6 | the file is edited by hand, **outside the app**; the app only validates and reports | `.nojae` is reserved by code, hidden from the tree, unreachable by any route — same rule as `read-write-policy.json` |
| D7 | every mode is switchable mid-conversation **except `todoBuilder`**, which locks once the thread has a turn | a spec draft may change its mind; a lot being written under one set of conventions must not change them halfway |
| D8 | JSON booleans (`mcptools: true`), `label`, `readme`, `version`; `systemMessage` accepts a **string or a list of lines** | mirrors the policy file the user already edits; a list keeps a hand-edited file readable inside JSON |
| D9 | `todoBuilder` is app-owned: absent from the JSON, its persona and its lot-conventions block are **rendered from code**, never configurable | the batch conventions are enforced by the app (`core/batch.py`); a JSON copy would be a third source of truth and let a user break a runner invariant |
| D10 | no per-mode sampling / `maxIterations` in this lot | deferred on purpose; see "Later". Note the consequence: the key vocabulary is closed, so a future key needs the app updated *first* |
| D11 | a broken modes file is **never fatal**: the built-in defaults are used and the findings are shown | refusing chat over a typo is a wall; the policy file's blocking error is right for the tree and wrong for a chat prompt |
| D12 | an unknown key is an **error finding**, not ignored | this file is hand-edited; `mcptool: 1` silently meaning "off" is the classic loss. D10 is the price, and it is paid in the docs |
| D13 | a batch task turn always runs in `code`, set by the runner — it does not inherit the composer's selection | the *runner* executes coding tasks; `todoBuilder` writes the lot. D7 has no conflict because a lot launch resets the thread, which releases the lock |
| D14 | an unknown `agentMode` in a request is a **422 naming the id**, never a silent fall back to the default | a silently-wrong mode is a silently-wrong agent; a typo in a client deserves an error, unlike D11 where the file is the user's own |

## The file

`<workspace>/.nojae/agent-modes.json`, seeded by exclusive create at the two
moments a folder becomes the working location — boot (`api/server.py:111`, next
to `_seed_workspace_policy`) and the settings switch (`routes.py:1263`, riding
in `_path_location_payload` beside `"policy"`). Same three non-negotiables as
`editorfs/policy_seed.py`: one `O_CREAT | O_EXCL`, never called from a *reader*
(`config.path_location()`), and a seed that cannot happen is a note, not a
refused switch.

```json
{
  "version": 1,
  "readme": ["…explains the vocabulary from inside the file…"],
  "default": "code",
  "agents": {
    "chat":      {"label": "Chat",      "mcptools": false, "lsptools": false,
                  "systemMessage": ["…", "…"]},
    "code":      {"label": "Code",      "mcptools": true,  "lsptools": true,
                  "systemMessage": ["…", "…"]},
    "architect": {"label": "Architect", "mcptools": true,  "lsptools": false,
                  "systemMessage": ["…", "…"]}
  }
}
```

`todoBuilder` deliberately has no entry. The loader injects the app-owned mode
into every resolved list; a file that declares it is answered with a **note**
("todoBuilder is defined by benchmarkit and cannot be set here") and its entry
is ignored — a note rather than an error because the user *can* fix it by
deleting it, and the fix must not cost them the other three modes (D11).

## Where the code goes

Two flat modules under `src/core/`, not a package — this is `core/batch.py`
shaped (vocabulary + loader + one renderer), not `mcphub/` shaped (no live
sessions, no hub, no hot toggle):

| file | owns |
|---|---|
| `core/agent_modes.py` | constants, validation vocabulary, the loader and its cache, `resolve_mode(id)`, `ensure_agent_modes_file` |
| `core/agent_modes_defaults.py` | the seeded document as a Python literal, the reserved `todoBuilder` definition, and the lot-conventions block rendered **from `core/batch.py` constants** |

`routes.py` reaches it through three named symbols only — the mode list for
`GET /api/agent-modes`, the resolved mode for `post_chat`, and its two derived
inputs (persona text, generated blocks). `core/chat.py` gains **one optional
kwarg** (`mode_instructions`) and does not learn what a mode is; it keeps taking
a plain `sysprompt`. That keeps the benchmark path (`core/llm.py`, `core/bench.py`)
completely untouched.

Frontend: `composables/useAgentModes.js` (one fetch, the selection, the D7 lock
rule) and a `<select>` in the composer's `.chat-bullet-row` (`WorkBenchInput.vue:287`),
in the left action group — the memory bullets stay centered, the actions stay
left, the select sits between them as asked.

## Invariants the tests must hold

* AGENTS.md is in the system message for **every** mode, including one with both
  tool flags false, whenever the working location has the file.
* `chat` mode: litellm receives **no `tools` key at all**, and the system message
  contains no `TOOL_NOTICE`.
* Settings still wins over a mode that opens a hub: `mcptools: true` with the
  connector unticked in Settings contributes nothing.
* The benchmark never consults a mode: no `agent_modes` import in `core/bench.py`
  or `core/llm.py`.
* `agent_modes_defaults` is the only place the seeded bytes and the lot
  conventions live, and the seeded document round-trips through the loader
  byte for byte (the `policy_seed.py` argument, reused).
* The lot-conventions block cannot drift from `core/batch.py`: the test asserts
  the rendered text contains the module's live `DEFAULT_LOT_DIR`,
  `MAX_BATCH_TASKS` and `MAX_BATCH_LOT_BYTES`, and a patched constant changes it.
* `POST /api/chat` with an unknown `agentMode` is a 422.
* No route can read or write `.nojae/agent-modes.json`; the app's reader
  (`GET /api/agent-modes`) is the only exposure, and it is read-only.

## Not in this lot

**Later, another set:**

* per-mode sampling (`temperature`, `maxIterations`, model choice) — D10. The
  schema is deliberately closed rather than open-ended, so this means adding
  keys to the vocabulary, not discovering a hole;
* a Settings panel that edits `agent-modes.json` — D6 says a power user edits it
  by hand first; the panel needs a JSON editor with validation feedback and is
  its own feature;
* modes for the *benchmark* (a benchmark run has no mode — it has `LLM_SYSPROMPT`
  and that is enough);
* hiding the memory bullets or the tool dock per mode — the tool panels already
  gray out from the D4 verdict, and nothing else is proven worth hiding yet.

## How to run this lot

Launch Batch mode with **prefix `am-`** (hyphen included), lot folder `todos`.
The hyphen is part of the prefix, not a separator: `core/batch.py`'s
`parse_task_number` wants the number immediately after the prefix, so `am`
matches nothing and every file lands in `skipped` with "no task number after the
prefix". Same reason the earlier lots are `pol-`, `ed-` and `work-`. Verified
against `batch.task_lot('am-', 'todos')`: 8 tasks, no skips.

## Task order

| # | task | depends on |
|---|---|---|
| am-01 | `agent_modes.py`: vocabulary, loader, validation, cache, `resolve_mode` | — |
| am-02 | `agent_modes_defaults.py`: seeded personas, the reserved mode, the conventions block | am-01 |
| am-03 | seeding at boot and on the workspace switch | am-01, am-02 |
| am-04 | prompt assembly: persona replaces `sysprompt`, `mode_instructions` kwarg | am-01, am-02 |
| am-05 | tool gating: mode ∩ conversation ∩ Settings, no `tools=` at all | am-01 |
| am-06 | API surface: `GET /api/agent-modes`, `ChatIn.agentMode` | am-04, am-05 |
| am-07 | frontend: `useAgentModes`, the select, session persistence | am-06 |
| am-08 | batch pinning (D13, D7), then the test sweep and the docs | am-03 … am-07 |

Each task ships its own pytest (and Vitest where it touches `web/src`) — the
final task is a sweep for what the individual tasks could not see, plus README
and the one AGENTS.md paragraph.

## Open questions a task must settle in code, not here

* am-02: the exact wording of the three seeded personas. The drafts agreed in
  the design session are the starting point; they are *instructions*, not
  description, and each stays under ~15 lines because this text is paid for on
  every turn of the conversation.
* am-04: whether `mode_instructions` rides in the same string as
  `agent_instructions` or its own argument. Own argument — the `TOOL_NOTICE`
  word check (`chat.py:218`) must keep judging only prompt-like text, which is
  why `agent_instructions` was separated in the first place.
* am-06: response shape for a broken file — `modes` (the usable defaults) plus
  `findings` is the shape, because a 500 would hide the fix from the UI.
* am-07: whether the select is disabled during `streaming` (it is not — D7
  allows switching between turns, and a mid-turn switch has no effect on the
  turn already running) and during `batch.active` (it is, and shows `code`).
