# am-04 — The mode's words reach the system message (order preserved)

The seam. One mode is resolved per request and contributes exactly two things to
the system message: it **replaces** `LLM_SYSPROMPT` with its persona, and its
generated blocks append after AGENTS.md. Nothing else about the prompt changes.

Depends on: am-01 (`resolve_mode`, `AgentMode`), am-02 (`instructions` on the
reserved mode).

## 1. The order, and why nobody gets to move it

`core/chat.py:211-240` already builds, in this order:

1. `sysprompt` (or `DEFAULT_LLM_SYSPROMPT`)
2. `TOOL_NOTICE`, when tools are actually being passed and the prompt doesn't
   already mention tools
3. `agent_instructions` — the AGENTS.md block
4. `batch_instructions` — the lot big picture

**Keep it exactly.** No separators, no banners, no reordering: the sequence is
pinned by `tests/test_chat.py` (the `agent_instructions` tests around lines
289-337) and the recency slot is already occupied by the most operational text.
A mode therefore plugs in at **step 1** and at **a new step between 3 and 4**.

Final order:
`persona → tool notice → AGENTS.md → mode blocks → lot big picture`

## 2. `core/chat.py`: one new optional kwarg

```python
def stream_chat(
    ...,
    agent_instructions: Optional[str] = None,
    mode_instructions: Optional[str] = None,   # new, after agent_instructions
    batch_instructions: Optional[str] = None,
    ...
) -> None:
```

Appended after `agent_instructions`, same append-not-mutate shape as its two
neighbours, with a docstring paragraph in their voice:

```python
    if mode_instructions:
        # The active mode's own standing instructions (the reserved Todo
        # builder carries the batch-lot conventions). After the project
        # instructions and before the lot big picture: AGENTS.md stays the
        # closest thing to the conversation, and the lot description stays
        # the furthest. It is a separate argument from agent_instructions
        # for the same reason those are separate: the TOOL_NOTICE word
        # check above must judge prompt-like text only.
        messages[0] = {
            "role": "system",
            "content": _content_text(messages[0]) + "\n\n" + mode_instructions,
        }
```

`stream_chat` must **not** learn what a mode is: no import of `agent_modes`, no
mode object, no id. It takes text. That is what keeps the benchmark path
(`core/llm.py`, `core/bench.py`) untouched — they never call this function and
never will.

## 3. The persona replaces `sysprompt` — in the route

`post_chat` (around `routes.py:1673-1700`) resolves the mode and computes the
persona. Do the substitution in **`routes.py`**, not inside `stream_chat`: the
fallback chain involves `config.sysprompt`, which is a route-side concern
(`_load_config()`), and `stream_chat` already handles `sysprompt=None`.

```python
mode, findings = resolve_mode(body.agentMode)   # am-06 adds the field
if mode is None:
    raise HTTPException(422, …)                 # D14, see am-06
persona = mode.system_message or config.sysprompt
```

Then, in the `stream_chat` call:

```python
sysprompt=persona,
agent_instructions=agent_instructions_block(),   # unchanged, always
**({"mode_instructions": "\n\n".join(mode.instructions)}
   if mode.instructions else {}),
```

The `**({…} if …)` conditional-pass shape is the existing one for
`batch_instructions` — copy it, so a request with no mode blocks keeps exactly the
call shape the current chat tests stub.

**Why replace rather than append (D1)**: two stacked personas contradict each
other ("you are a helpful assistant" + "you never write code"). The configured
`LLM_SYSPROMPT` remains authoritative for everything that is *not* WorkBench — the
benchmark reads it through `core/models.py` and never sees a mode.

**Why the empty-string fallback matters**: an editor who deletes a persona to
"see what happens" gets the configured prompt back, not a conversation with no
instructions at all.

## 4. AGENTS.md is unconditional — make it a test (am-00's invariant)

Three assertions, all in `tests/test_chat.py` and `tests/test_api.py`:

* `config.agent_instructions_block()` is called for **every** mode, and its result
  is present in the system message sent to litellm — including a mode with both
  tool flags false;
* there is **no** parameter, JSON key or request field anywhere that suppresses
  it: `stream_chat` has no `include_agents_md`, `resolve_modes` has no such key,
  `ChatIn` has no such field;
* the persona never mentions AGENTS.md as an instruction to go read it. AGENTS.md
  is already in the prompt (see `chat.py:195-202`: it's appended, and the block
  itself names its source), so a persona that tells the model to go look for it
  wastes a turn on a tool call for text it was handed.

The last one is a real trap: `config.agent_instructions_block()` frames the text
as "this project's own instructions… take precedence over generic wording". A
persona that contradicts that precedence loses, so am-02's personas must not
claim authority over project conventions.

## 5. Per-turn re-resolution (hot edit)

The mode document is resolved **per request**, not cached on the thread: `.nojae`
is edited outside the app, and the app's stance everywhere else is that a config
file change takes effect without a restart (AGENTS.md is re-read per request for
the same reason). am-01's mtime cache makes this cheap; do not add a second cache
layer here.

Consequence to write down in the code comment: an edit to the modes file changes
the persona of the *next* turn of an existing conversation. That is consistent
with how the rest of the app behaves and needs no UI warning.

## 6. Tests — `tests/test_chat.py` (extend)

Existing tests stub `stream_chat` with a signature that includes
`agent_instructions=None` (lines 523, 542, 568, 594, 648, 682, 700, 711, 725, 795,
844). **Add `mode_instructions=None` to every one of those stubs** in the same
commit that adds the kwarg, or the suite breaks in a way that looks unrelated.
Grep for the stub signature rather than editing by eye.

New cases:

* a mode with a persona: the system message **starts** with the persona and the
  configured `LLM_SYSPROMPT` is absent from it;
* a mode with an empty persona: the configured `LLM_SYSPROMPT` is used, verbatim;
* a mode with no persona key at all: same as above;
* ordering pinned for a mode that has both a persona and blocks, with a
  `batchContext` also in play — assert the index of each of the five parts is
  strictly increasing;
* `mode_instructions` empty/`None` → the system message is byte-identical to the
  pre-feature one for the same request (the regression guard, the one test that
  proves nothing changed for people who never touch the feature);
* `TOOL_NOTICE` interaction: a persona that does *not* mention tools still gets
  the notice appended when tools are passed; a persona that *does* mention them
  doesn't get it twice; and `mode_instructions` never influences that check,
  because it is appended after it (mirror of the existing
  `test_agent_instructions_do_not_suppress_the_tool_notice` at line 321 — write
  the mode equivalent).

## 7. Acceptance

* `pytest tests/test_chat.py tests/test_api.py tests/test_bench.py` green —
  `test_bench.py` included because it is the proof the benchmark never learned
  about modes.
* The whole suite passes with no stub-signature churn left over: `grep -c
  "agent_instructions=None" tests/test_chat.py` and the count of
  `mode_instructions=None` must be equal.
* `core/chat.py` still contains no `agent_modes` import (check it, don't assert it
  in prose).
* Two docstring lines in `stream_chat` describing `mode_instructions`, in the
  voice of the two paragraphs above it — no more.

## 8. Not this task

* tool gating (am-05) — under this task a Chat-mode request still *passes* tools,
  it is simply told not to use them. Land am-05 right after; do not "just add" the
  gate here and leave the 422 for later;
* the `agentMode` request field itself (am-06).

## 9. This task changes behaviour for everyone — say so in the commit

Until am-06 lands there is no way to *pick* a mode, so every chat uses the file's
`default`, which the seeded file sets to `code`. So the moment this task merges,
chat conversations stop using `LLM_SYSPROMPT` and start using the Code persona —
intentional (it is the same agent, now told what it is for), but it is the one
task in the lot that changes what an existing user sees. Three consequences to
handle explicitly:

* `LLM_SYSPROMPT` keeps its meaning for the **benchmark** and stays a required
  settings key; for chat it becomes the *fallback* rather than the prompt. Update
  the settings dialog's help text for that key (one sentence) — a key whose value
  silently stopped being used is worse than a key that moved. `SETTING_KEYS`,
  `_render_env()` and `.env.example` do **not** change: the key set is unchanged,
  which is exactly why the wording in the dialog is the thing to fix;
* the escape hatch is `"systemMessage": ""` on the `code` entry of
  `.nojae/agent-modes.json`: empty falls back to the configured prompt, so a user
  who wants the old behaviour back restores it without a code change. Test it (§6,
  second case) so the hatch is a supported path, not an accident;
* if a machine has no seeded modes file (a workspace created before am-03
  shipped), am-01's fallback is the built-in defaults, which are also `code` — so
  the behaviour does not depend on whether seeding succeeded.
