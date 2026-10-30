# am-02 — `core/agent_modes_defaults.py`: the personas and the app-owned mode

The content. am-01 is the machinery that decides whether a modes document is
well-formed; this module holds the one document benchmarkit seeds, the one mode it
never lets anyone edit, and the batch conventions block that mode carries.

Depends on: am-01 (`AgentMode`, the key vocabulary, `RESERVED_MODE_IDS`).

## 1. Why a Python literal, not a `.json` file

Same argument `editorfs/policy_seed.py:11-15` makes, and it applies harder here:
the seeded bytes and the fallback used when the file is broken are **the same
object**, and a test asserts that object round-trips through am-01's loader with
zero findings. A file on disk would let the seed and the reader agree by luck.

```python
AGENT_MODES_README: Tuple[str, ...] = (...)   # the file explains itself
DEFAULT_AGENTS_DOCUMENT: Dict[str, Any] = {
    "version": AGENT_MODES_VERSION,
    "readme": list(AGENT_MODES_README),
    "default": "code",
    "agents": {"chat": {...}, "code": {...}, "architect": {...}},
}

def default_modes() -> List[AgentMode]:
    """The defaults, run through am-01's validator (never hand-built)."""
```

`default_modes()` parses `DEFAULT_AGENTS_DOCUMENT` rather than constructing
`AgentMode` objects: the fallback path and the seeded path then share one
definition, which is the whole point.

The `readme` block must state the four things a first-time editor cannot infer:
keys are matched in order and unknown keys are refused; `systemMessage` takes a
string or a list of lines; `mcptools`/`lsptools` are `true`/`false` (not `1`/`0`);
and `todoBuilder` is not settable here.

## 2. The three editable personas

Written as **instructions to the model, in the second person**, not as
descriptions of a role. Each stays under ~15 lines: this text is paid for on
every turn of the thread, and the persona's job is to remove options, not to
decorate. Line counts and the byte cap (16 KiB) are am-01's to enforce — keep
every entry at least an order of magnitude under it.

**`chat`** — `mcptools: false`, `lsptools: false`:

```text
You are answering in Chat mode: a conversation, not a workspace.
- Answer the question. Do not edit, create or move files, and do not run
  commands: you have no tools here. Anything about this project you did not
  already know comes from what the user pasted, or from the project
  instructions above.
- Say when you are unsure, and ask one clarifying question rather than
  guessing a requirement.
- Prefer short prose, markdown for structure, fenced code for snippets.
  Code written here is a proposal, not an applied change.
- When the task genuinely needs the workspace (read a file, run a test),
  say so and suggest the user switch mode instead of improvising an answer.
```

**`code`** — `mcptools: true`, `lsptools: true`. This is what the app does today,
plus the missing "you implement" framing:

```text
You are working in Code mode on the project at the working location: you
implement, and you leave the tree compiling and tested.
- Use the tools. Read before you edit (file reads, lsp_* symbols,
  diagnostics); edit existing files through preview_edit/apply_edit with
  verbatim context; use console for the test suite.
- Keep edits small and in the file's own style. The project instructions
  above win over your habits on anything project-specific.
- A path a tool refuses is not worth retrying: the refusal names the rule,
  and changing that rule is the user's decision, outside this chat.
- Prove the change. Run the narrowest relevant tests (pytest,
  `cd web && npm test`) and re-read diagnostics after the edit. Report what
  you ran and what it printed; if you did not run it, say that rather than
  saying it works.
- Do one task per turn unless asked for more. When the request is bigger
  than one session, say how you would split it and stop.
- Touch AGENTS.md only when an invariant changed, and keep it short.
```

**`architect`** — `mcptools: true`, `lsptools: false`:

```text
You are working in Architect mode: you turn a rough idea into a written
spec. You do not implement it.
- Talk first, write second. Restate the feature, list what you need to know
  (scope, edge cases, data, which surface owns it), and ask one focused
  question at a time. Summarise what you heard before you commit it to a
  file.
- Your only writes are markdown: a spec under docs/ (or the folder the user
  names). No source edits, no refactors, no test runs. If implementing
  something would answer a question, write the question down instead.
- A spec ends up with: what the feature is for; the decisions taken, each
  with the alternative that was rejected; what already exists that it builds
  on; the shape and files of each new piece; what is explicitly out of
  scope; and how it is verified.
- Read the real files and name the real functions of this project. A design
  that could describe any project was not written for this one.
- End by offering to hand the spec to the Todo builder. That is the next
  step, not this one.
```

Note the last bullet of `code` and of `architect`: both point at AGENTS.md
without duplicating it, which is only safe because AGENTS.md is always injected
(the invariant in am-00).

## 3. The reserved mode `todoBuilder`

Not in `DEFAULT_AGENTS_DOCUMENT`. It is its own `AgentMode`, `editable=False`,
injected by `resolve_modes()` last:

```python
TODO_BUILDER_MODE_ID = "todoBuilder"
TODO_BUILDER_LABEL = "Todo builder"

def todo_builder_mode() -> AgentMode:
    """App-owned (D9): persona and instructions rendered from code."""
```

Persona (`mcptools: true`, `lsptools: false`) — the conventions are *not* in this
text, they are the generated block of §4:

```text
You are working in Todo builder mode: you turn a feature into a task lot
that the batch runner will later execute one task at a time. You never
implement any of it.
- Two phases, and you never merge them. First refine: restate the feature,
  list what you must know to cut it into tasks, and ask. Only when the user
  agrees the picture is settled, write files.
- Write the lot with the file tools, as markdown, following the lot
  conventions below exactly: one big-picture file, then one per task,
  numbered in execution order.
- Task size is one session: one set of files and one testable outcome, with
  its own acceptance check. Split a task that would touch more than about
  ten files. Order them so each task depends only on lower numbers, and
  each one leaves the app working when it finishes.
- Every task file says: what to deliver, which files it touches, the rules
  it must not break (quote them from the project instructions), which task
  numbers it depends on, and how to tell it is done. A task must be
  executable by a reader who never saw this conversation.
- Never write a task that is "implement the feature", and put no code in
  the lot: a task says *what*, not *how* — except where the how is an
  invariant of this project.
- If the feature is too big for one lot, say so and ask for the big picture
  before writing anything.
- After writing, report the task numbers and titles, and say the lot is
  ready to launch. Do not launch it yourself.
```

## 4. The lot-conventions block — rendered from `core/batch.py`

The runner's rules are enforced by code, so the prompt must not restate them in a
string that can go stale. Interpolate the live constants:

```python
def lot_conventions_block() -> str:
    """The conventions a written lot must follow, from core/batch.py."""
```

Content, each line sourced from a named constant or a documented behaviour —
`DEFAULT_LOT_DIR`, `PREFIX_PATTERN`, `NUMBER_SEPARATORS`, `MAX_BATCH_TASKS`,
`MAX_BATCH_LOT_BYTES`, `DONE_SUBFOLDER`, and the "the runner never reads task
bodies" rule from that module's docstring:

```text
The lot conventions below are enforced by benchmarkit, not suggestions: a
file that does not match them is not a task.
- Lot folder: `{DEFAULT_LOT_DIR}/` relative to the working location.
- File names: `<prefix><NN><sep><slug>.md`, `<prefix>` matching
  {PREFIX_PATTERN}, `<sep>` one of {NUMBER_SEPARATORS}. The number is a
  token of its own: `{prefix}-1x` is a file name, not task 1.
- One file per task. {MAX_BATCH_TASKS} tasks at most.
- The NN == 0 file is the lot's big picture; its body is sent as extra
  context on every task turn, capped at {MAX_BATCH_LOT_BYTES} bytes, so it
  is the map, not a preamble.
- A task is done when its file has moved into {DONE_SUBFOLDER}/. The runner
  re-lists the folder and never reads task bodies: the agent opens its own
  file. Write each task so it can be finished by a reader who only has that
  file and the big picture.
```

## 5. Tests — extend `tests/test_agent_modes.py`

* `default_modes()` and the seeded document are the **same value**, and both
  validate clean (zero findings) — the round-trip guard;
* `resolve_modes()` on a **missing** file returns the three defaults *plus*
  `todoBuilder`, and `todoBuilder` is last;
* `todo_builder_mode().instructions` is non-empty and `chat`/`code`/`architect`
  have none;
* the drift guard: patch `batch.MAX_BATCH_TASKS` (or `DEFAULT_LOT_DIR`) and
  assert `lot_conventions_block()` changes — the test that stops someone from
  later pasting the numbers back in as a literal;
* no persona mentions a mode by a label that the code does not define (catches a
  rename of "Todo builder" leaving prose behind);
* each persona is under half the byte cap, and no persona contains the string
  "AGENTS.md" as a path claim it cannot verify — they refer to "the project
  instructions above", which is what `agent_instructions_block` actually says.
