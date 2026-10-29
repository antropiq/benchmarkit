# am-05 — The mode gates the hubs: Chat really has no tools

D3 and D4 in code. A mode with `mcptools: false` means the MCP schemas **never
reach the model** — not "the prompt says not to use them". Today a Chat-mode
request (am-04) would still hand litellm the full `tools` array and the
`TOOL_NOTICE`, so the persona would be a polite request against a live affordance.

Depends on: am-01 (`AgentMode.mcptools` / `.lsptools`). Independent of am-04 in
the code, but land after it: both touch the `post_chat` argument list and one
rebase is cheaper than two.

## 1. Three levels, one verdict

| level | who owns it | granularity | today |
|---|---|---|---|
| Settings | `SettingsDialog` → `_active_connector_ids()` / `_active_server_ids()` | the whole install | exists |
| **mode** | `.nojae/agent-modes.json` | one hub, per mode | **this task** |
| conversation | the Chat tab's per-thread toggles → `ChatIn.mcpEnabled` / `lspEnabled` | one connector | exists |

They intersect; there is no override in any direction, and the order does not
matter because it's a set intersection. The implementation is a **narrowing of the
conversation allowlist**, which is why it needs almost no new code:

```python
def _chat_tool_kwargs(
    request: Request,
    mcp_enabled: Optional[List[str]] = None,
    lsp_enabled: Optional[List[str]] = None,
    mode: Optional[AgentMode] = None,
) -> Dict[str, Any]:
```

`mode=None` means "no mode gate" and keeps the current behaviour byte-for-byte —
the shape every existing test calls with.

```python
    if mode is not None and not mode.mcptools:
        mcp_enabled = []          # empty list = allowlist with nothing in it
    if mode is not None and not mode.lsptools:
        lsp_enabled = []
```

**Use `[]`, not `None`.** `None` is the documented "no filter" sentinel
(`routes.py:686-691`); passing `None` here would open the hub wide. This is the
single most dangerous line in the lot — write the comment next to it.

`_mcp_tool_kwargs` / `_lsp_tool_kwargs` already return `{}` when the intersection
is empty (`if not tools: return {}`), and `_chat_tool_kwargs` returns `{}` when
neither hub contributes. So the whole gate is those four lines plus the parameter.

## 2. What "no tools" must mean, exactly

With `mode=chat` and every connector enabled in Settings:

* `stream_chat` receives **no `tools` key at all** (the `{}` return path) — not
  `tools=[]`;
* no `tool_executor`, no `tool_servers`, no `max_tool_iterations`;
* therefore `use_tools` is False at `chat.py:217` and **`TOOL_NOTICE` is not
  appended**;
* no hub session is even consulted: `manager.active_tool_defs([])` is reached, but
  the earlier `_active_connector_ids()` intersection makes the list empty, so no
  tool definition is serialised into the request. (Check that no code path calls
  `list_tools` per connector before the filter — if one does, that is a wasted
  round-trip worth fixing here.)

`tools=[]` instead of omitting the key would be a real bug on some servers: an
empty `tools` array is not the same request as no `tools` field, and litellm
forwards what it is given.

## 3. The mode does not *grant* tools

`mcptools: true` cannot switch on a connector the user unticked in Settings: the
mode only *narrows*, never widens, and the existing code already guarantees it
(`_active_connector_ids()` is the authority, the allowlist filters it). Assert it
in a test anyway — it is the invariant a future "let the mode pick the connectors"
refactor would break.

The same argument is why the mode takes **booleans, not id lists**: an id list in
the modes file would be a second place to enable things, in a file the app cannot
edit, with no UI — three places to answer one question.

## 4. The UI consequence: a hub that cannot be used

A user who ticks MCP connectors in Settings and picks Chat gets nothing, with no
explanation. am-07 grays the panels out, but the *server* has to be the one that
says why, because `GET /api/mcp` and the chat request both answer the effective
state:

* extend the chat-hub listing routes with nothing new (do **not** add a mode
  parameter to them — the panels are global views, and a per-mode answer there
  would make Settings a function of the dropdown);
* the mode list from am-06 (`GET /api/agent-modes`) already carries
  `mcptools`/`lsptools` per mode, which is all the frontend needs to gray a panel
  out. Keep it that way: **one route answers the mode question.**

## 5. Tests — `tests/test_chat.py` (the gate) and `tests/test_api.py`

Hermetic: the hubs are already disabled in the suite (`BENCHMARKIT_DISABLE_MCP=1`,
`BENCHMARKIT_DISABLE_LSP=1`), so build the gate tests on a **stub manager** on
`app.state`, the way
the existing MCP-gating tests do. Grep for the fixture that fakes
`active_tool_defs` and reuse it.

Must exist:

* mode with `mcptools=False`, connector enabled in Settings and ticked in the
  conversation → the kwargs dict has **no `tools` key** (`"tools" not in kwargs`,
  not `kwargs.get("tools") == []` — the assertion *is* the requirement);
* same for `lsptools=False`;
* both false → `{}`;
* `mode=None` (the pre-feature call) → byte-identical kwargs to today for the same
  request — run it against the existing tests' expectations, don't re-derive them;
* `mcptools=True` + connector **unticked in Settings** → no tool, proving the mode
  never widens;
* `mcptools=True` + Settings-ticked + conversation-unticked → no tool (the
  three-way intersection in one case);
* the `TOOL_NOTICE` case from D3: system message for a Chat-mode request contains
  no `TOOL_NOTICE` fragment, and for a Code-mode request on the same prompt it
  does;
* a request whose mode gates the only hub that had tools ends up in the
  no-tools branch of `_chat_tool_kwargs` — i.e. no `max_tool_iterations` leaks in
  from the LSP-only branch at `routes.py:699-701`.

## 6. Acceptance

* The invariant list in `am-00` gains its two `chat`-mode bullets as green tests.
* `pytest tests/test_api.py tests/test_chat.py tests/test_mcp.py tests/test_lsp.py`
  green with **zero changes** to `test_mcp.py` / `test_lsp.py`: those files test
  the hubs, and this task must not have touched hub behaviour.
* No new module, no new route, no new frontend file. If this task grows past
  ~40 lines of production code, the gate is being implemented somewhere other than
  the four lines in §1.
* One docstring sentence on `_chat_tool_kwargs` naming the three levels, since
  that function is now where all three meet.
