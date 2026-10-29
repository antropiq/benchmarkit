# am-01 — `core/agent_modes.py`: the vocabulary, the loader, the cache

Pure core, no HTTP, no FastAPI, no browser concern: read one JSON file and answer
"which modes exist, what does mode *X* say". Nothing in this task talks to a
model or a route — am-04 and am-06 consume it.

Depends on: nothing. am-02, am-03, am-04, am-05 and am-06 all depend on this one,
so get its shape right before they exist.

## 1. Constants — module attributes, patchable

Follow the `editorfs/constants.py` / `lsphub.constants` convention: the numbers
live on the module so a test patches `agent_modes.MAX_MODES` rather than reaching
into the function that reads it.

```python
# src/core/agent_modes.py
AGENT_MODES_FILENAME = "agent-modes.json"      # beside read-write-policy.json
AGENT_MODES_VERSION = 1                        # the only version accepted
MAX_MODES = 16                                 # over it: error finding
MAX_MODE_ID_CHARS = 32
MAX_MODE_LABEL_CHARS = 40
MAX_MODE_SYSTEM_MESSAGE_BYTES = 16 * 1024      # per mode; this text is sent
                                               # on every turn of the thread
RESERVED_MODE_IDS = ("todoBuilder",)           # app-owned, am-02
MODE_ID_PATTERN = r"^[a-z][A-Za-z0-9_-]*$"     # JSON key, and the value of
                                               # ChatIn.agentMode
```

The directory name is **not** re-declared here: import `POLICY_DIRNAME` from
`mcp_connectors.policy_shared` the way `editorfs/constants.py:24` does (with the
same `sys.path` fallback). Two constants for one folder name is how the folder
stops being one folder.

## 2. The findings vocabulary

Same two kinds as the policy file, and the same reason: a finding the user cannot
act on is a wall, a finding they can fix is a message.

```python
KIND_ERROR = "error"   # this rule/mode is dropped; defaults cover the gap
KIND_NOTE = "note"     # nothing dropped; the file says something odd

@dataclass(frozen=True)
class ModeFinding:
    kind: str
    message: str        # names the JSON pointer ("agents.chat.lspTool")
```

Report a **pointer**, not a bare id: the user is editing JSON by hand, so
`agents.architect.mcptool` is the sentence that lets them fix it in five seconds.

## 3. `resolve_modes()` — the loader

```python
def resolve_modes() -> Dict[str, Any]:
    """Every mode usable now, in dropdown order, plus the file's findings.

    Returns {"default": str, "modes": [AgentMode…], "findings": [ModeFinding…],
             "source": {"path": str, "exists": bool, "loaded": bool,
                        "version": Optional[int]}}.

    Never raises. A missing, unreadable, unparseable or invalid file yields the
    built-in defaults (am-02) plus findings that say what happened — D11: chat
    must keep working over a typo in a file nobody can preview.
    """
```

Validation rules, each one a finding with a pointer:

| shape | verdict |
|---|---|
| root not a JSON object, or `agents` not an object | error; **whole file** falls back to defaults |
| `version` absent or ≠ `AGENT_MODES_VERSION` | error (mirrors the policy file: "expected version 1") |
| `default` absent, not a string, or naming no declared mode | error; `default` becomes the defaults' `default` |
| mode id fails `MODE_ID_PATTERN` | error; that mode is dropped |
| a key outside `{label, mcptools, lsptools, systemMessage}` | **error** naming it — D12, a typo like `mcptool` must not silently mean "off" |
| `mcptools` / `lsptools` present but not a bool | error; that mode is dropped. No truthiness coercion: `1` and `"yes"` are what the *user* wrote, and `0` for off is exactly the mistake D12 refuses to guess about |
| `systemMessage` is a string | accepted, as written |
| `systemMessage` is a list of non-empty strings | accepted, joined with `\n` (D8) |
| `systemMessage` is any other shape | error; that mode is dropped |
| `systemMessage` over `MAX_MODE_SYSTEM_MESSAGE_BYTES` (UTF-8) | error; that mode is dropped — this text is re-paid every turn |
| `label` absent | **not** an error: fall back to the id, no finding (a label is presentation) |
| `label` present but not a string | error naming the pointer |
| `agents` declares a name in `RESERVED_MODE_IDS` | **note** ("`todoBuilder` is defined by benchmarkit and cannot be set here"), entry ignored |
| more than `MAX_MODES` entries | error; the extra ones are dropped, in file order |

An entry is dropped **as a unit** — a mode with one bad key contributes nothing.
A half-validated mode whose `mcptools` survived but whose `systemMessage` did not
would be an agent with tools and no persona, which is precisely the thing this
feature exists to prevent.

**Order** is file order for declared modes, with the reserved `todoBuilder`
appended last (am-02 injects it). A dict's insertion order is the dropdown's
order; say so in a comment, because it is the kind of thing a later "just sort
it" refactor silently breaks.

```python
@dataclass(frozen=True)
class AgentMode:
    id: str
    label: str
    system_message: str      # "" = fall back to LLM_SYSPROMPT (D1)
    mcptools: bool
    lsptools: bool
    editable: bool = True    # False only for the app-owned mode
    instructions: Tuple[str, ...] = ()  # generated blocks, am-02/am-04
```

`instructions` is a tuple of already-rendered blocks. am-01 defines the field and
leaves it empty; only `todoBuilder` ever fills it.

## 4. The cache

Copy the policy loader's shape (`editorfs/policy_file.py:98-135`) — same three
arguments for the same reasons:

* key: the **workspace root** (`config.path_location()`), because a root is not
  a file and the same relative path under two roots holds two different files;
* stamp: `(st_mtime_ns, st_size)`, `None` when the file is absent — absent is
  worth caching too;
* the filename pair goes **in** the entry, so a test that patches
  `AGENT_MODES_FILENAME` invalidates rather than answers stale;
* bound: drop the whole cache past N roots (32), which costs one small re-read and
  never an answer;
* `invalidate_agent_modes_cache()` for tests and for a workspace switch.

**One reader in the process.** No connector reads this file: the modes shape what
the model is *told*, and every connector process gets its instructions through the
prompt it is called with. Do not add a shared-module reader "for symmetry" with
pol-08 — that symmetry does not exist.

## 5. `resolve_mode(mode_id)`

```python
def resolve_mode(mode_id: Optional[str]) -> Tuple[AgentMode, List[ModeFinding]]:
    """One mode by id. None → the file's default. Unknown → (None, findings).

    The caller decides what an unknown id costs: the chat route answers 422
    (D14), the fallback-to-default behaviour belongs to *no one*, so this
    function must not quietly provide it.
    """
```

Returning `(None, …)` instead of raising is deliberate: `routes.py` turns it into
an `HTTPException`, and `post_chat` runs in the SSE path where an exception has
already lost the client's stream.

## 6. Tests — `tests/test_agent_modes.py`

Hermetic: `monkeypatch` `agent_modes.AGENT_MODES_FILENAME`/the root the same way
`tests/test_editorfs_policy_file.py` patches the policy location; no LSP, no MCP,
no LLM (the module imports neither).

Must exist:

* the seeded document of am-02 round-trips through `resolve_modes()` with **no
  findings at all** — this is the test that keeps the written file and the read
  one agreeing (the `policy_seed.py` argument, reused verbatim);
* every row of the table above, each asserting the finding *names the pointer*;
* a file with a syntax error → defaults + one error, and `source.loaded` is False;
* a mode dropped as a unit (bad `systemMessage`, `mcptools` correct) — assert the
  mode is **absent**, not present-with-defaults;
* cache: two calls, one read (count through a patched `open`/`os.stat`); a
  `touch`-equivalent stamp change re-reads; a root patch re-reads;
* `resolve_mode("nope")` → `(None, …)`, no exception;
* `systemMessage` as string and as list produce byte-identical `system_message`
  for the same content.

## 7. Acceptance

* `pytest tests/test_agent_modes.py` green, and `pytest` as a whole unchanged.
* No import of FastAPI, `src.api`, `litellm` or any hub in the new module.
* Nothing writes a file in this task — seeding is am-03. If you feel the urge,
  that is am-03's exclusive-create rule leaking; keep the reader a reader.
* Docstring on the module saying what it is *not*: not a hub (no live sessions,
  nothing hot-toggled), not the policy file (a broken modes file never blocks a
  request). Two paragraphs, no more.
