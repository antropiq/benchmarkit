# am-06 — `GET /api/agent-modes` and `ChatIn.agentMode`

The two HTTP edges of the feature: one read-only route that publishes the modes,
and one request field that selects one. Everything before this task is unreachable
from a browser; everything after it is a UI.

Depends on: am-01 (loader), am-04 (persona path), am-05 (gate). The field and the
gate must land in the same task as the route, or a client can select a mode that
changes the prompt but not the tool set — half a mode.

## 1. `ChatIn.agentMode`

```python
class ChatIn(BaseModel):
    ...
    # The agent mode for this turn (see core.agent_modes). None means "the
    # workspace's default": a client predating the feature keeps working, the
    # same compatibility rule as mcpEnabled/lspEnabled above.
    agentMode: Optional[str] = None
```

**Validate in the route, not in the model.** The model can check the *shape*
(`MODE_ID_PATTERN`, length) cheaply, but "is this a mode that exists right now" is
a question about a file that may have been edited a second ago — a `Literal[...]`
or an enum here would freeze the mode list at import time and break the hot-edit
property am-04 §5 promises. So: field is `Optional[str]`, the existence check is
`resolve_mode()`'s job, called from `post_chat`.

Note `model_config = ConfigDict(extra="ignore")` is already on this model: an
older *server* ignores a new field, and this app is single-user, so no version
negotiation is needed or wanted.

## 2. The 422 (D14)

```python
mode, findings = resolve_mode(body.agentMode)
if mode is None:
    raise HTTPException(
        status_code=422,
        detail=f"Unknown agent mode {body.agentMode!r}. "
               f"Available: {', '.join(m.id for m in resolve_modes()['modes'])}",
    )
```

The detail lists what *is* available, because the only way to hit this is a typo —
in a JSON file edited by hand, or in a stale tab after someone renamed a mode. A
silent fall back to the default would answer a user's question with the wrong
agent and no clue; that is D14, and it is deliberately the opposite of D11
(a broken *file* still chats, because there the failure is the app's to absorb).

Raise **before** `_arm_chat_handle(...)` and before the SSE generator exists: this
is a plain request validation failure, and arming a pause handle for a request
that will never stream is how a later request finds a stale handle.

The error reaches the UI through the existing non-OK path in `useChat.js`
(`describeError`, around line 990) — no new client error plumbing.

## 3. `GET /api/agent-modes`

```python
@router.get("/agent-modes")
def get_agent_modes() -> Dict[str, Any]:
```

Read-only, no body, no cache-busting, mirroring `GET /api/memory` /
`GET /api/path-location` in shape.

```json
{
  "modes": [
    {"id": "chat", "label": "Chat", "mcptools": false, "lsptools": false,
     "editable": true},
    {"id": "code", "label": "Code", "mcptools": true, "lsptools": true,
     "editable": true},
    {"id": "architect", "label": "Architect", "mcptools": true,
     "lsptools": false, "editable": true},
    {"id": "todoBuilder", "label": "Todo builder", "mcptools": true,
     "lsptools": false, "editable": false}
  ],
  "default": "code",
  "source": {"path": "/srv/app/.nojae/agent-modes.json", "exists": true,
             "loaded": true, "version": 1},
  "findings": [
    {"kind": "error", "message": "agents.architect.mcptool: unknown key "
     "(expected label, mcptools, lsptools, systemMessage)"}
  ]
}

```

Deliberately **absent from the payload**:

* `systemMessage` — it is long, the dropdown never shows it, and the settings
  dialog has no editor for it yet (D6, "Later"). A client that wants to display it
  is a client that already needs the editor; adding the text now invites a
  half-built preview that will be wrong the day the editor lands. If you disagree,
  the burden is to say in the docstring *why* a ~2 KB string belongs in a request
  the composer fires on boot;
* `instructions` — the reserved mode's generated blocks, same reasoning, and it is
  derived text nobody should be copying into the DOM.

`findings` **is** present, always as a list (possibly empty), and a broken file
still answers **200** with the defaults in `modes` plus the findings that explain
them (am-00's open question, settled here). A 4xx/5xx would hide the fix: the user
has to be told *which pointer* is wrong, and the dropdown still has to render.

`editable: false` on the reserved mode is the frontend's signal for D7's lock and
for am-02's note about not editing it — so the client never hardcodes the id
`todoBuilder`. Grep-able rule: **the string `"todoBuilder"` appears zero times in
`web/src/` after am-07.**

## 4. The `.nojae` reach rule, restated for this route

`GET /api/agent-modes` answers *about* the file; it does not read it through any
file route, and it must not grow a `?path=` parameter or return the raw bytes. The
`.nojae` folder stays reserved: the tree, `GET /api/fs/*`, git status and the diff
all still answer as if it does not exist.

This is the pattern `GET /api/path-location` + `policy_file_state` already
establishes for the policy file — a small, purpose-built reader for one reserved
file, and no general escape hatch.

## 5. Tests — `tests/test_api.py`

* `GET /api/agent-modes` with no file: 200, four modes, `todoBuilder` last with
  `editable: false`, `default == "code"`, `findings == []`, `source.exists` False;
* with the seeded file (call `ensure_agent_modes_file` into a temp root):
  `source.loaded` True, `version == 1`;
* with a hand-broken file (bad key): 200, the defaults are served, and
  `findings[0]["message"]` contains the pointer — the D11 case;
* the payload contains **no** `systemMessage` and no `instructions` key anywhere
  (a `json.dumps` of the response asserting the two strings are absent — cheap and
  it survives refactors that add the fields back by accident);
* `POST /api/chat` with `agentMode: "nope"` → **422**, and the detail names the
  id and lists the real ones;
* `POST /api/chat` with no `agentMode` → uses the file default, and the recorded
  `stream_chat` kwargs match what an old client produced **for the tool
  arguments** — `"tools"`, `"tool_executor"`, `"tool_servers"` and
  `"max_tool_iterations"` identical. They do *not* match on `sysprompt`: from
  am-04 on, the default mode (`code`) supplies a persona, so the configured
  `LLM_SYSPROMPT` is no longer what an old client saw. Assert the tool kwargs and
  the mode id separately; a test that demands the whole old call shape would have
  to pin a persona string, and personas are prose someone will reword (am-02);
* `POST /api/chat` with `agentMode: "todoBuilder"` → 200 and the mode blocks
  reached `mode_instructions` (the reserved mode is reachable, which is the point);
* one test that the reserved mode's presence in the list does **not** depend on
  the file: two different files, both list `todoBuilder`.

## 6. The write-surface test

`tests/test_write_surface_isolation.py` scans `src/` for write markers and for
HTTP clients in connectors. This task adds a read-only route, so the scan should
stay green untouched — **but** add one assertion there, in the same spirit as the
existing ones: no route in `routes.py` whose path starts with `/agent-modes`
accepts a mutation. One extra check in that file, poisoned with a deliberately
added `@router.post("/agent-modes")` in a fixture string, following how the last
test in that file feeds its own guard a bad input.

## 7. Acceptance

* `pytest tests/test_api.py tests/test_write_surface_isolation.py` green.
* `curl -s localhost:8000/api/agent-modes | python -m json.tool` gives the shape
  above on a real boot, against the real `.nojae` of this repo.
* The benchmark, the Editor and the hub routes are untouched: `git diff --stat`
  shows changes only in `routes.py`, `core/agent_modes*.py`, `core/chat.py` and
  the tests.
