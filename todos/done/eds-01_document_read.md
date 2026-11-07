# eds-01 — The document read: the modes file's own entries, verbatim

**Spec:** §2 (what exists today), D2, D3, D4, D6 (the persona form), §4 "Files
touched" row 1.
**Depends on:** nothing — this is the lot's first task. The modes feature
(`am-*`) has landed, so `src/core/agent_modes.py` already holds the vocabulary,
the validator, the cache and the seeder.
**Files it touches:** `src/core/agent_modes.py`, `tests/test_agent_modes.py`.
Nothing else — no route, no frontend, no new module.

## What to deliver

One new **public read function** in `src/core/agent_modes.py` that answers *"what
does my file say?"* — the shape the agent-modes editor dialog (later tasks) needs
and that `resolve_modes()` deliberately does not give:

| field | content |
|---|---|
| `entries` | the file's own `agents` entries, **in the file's own order**, nothing added and nothing dropped: not the built-in fallback, not the app-owned mode. Each entry carries its id, its raw `label` / `mcptools` / `lsptools` / `systemMessage` as the file spells them, and the findings that name it. |
| `default` | the file's own `default` value as written (even when it names something the loader would refuse — the dialog shows it and blocks the save). |
| `readme` | the file's own `readme` if it has one, else absent — the writer (eds-02) preserves it verbatim. |
| `findings` | every finding the loader produced about this file, each with its `kind` (`error` / `note`) and the pointer it names, so the dialog can put each one next to the field it talks about. |
| `source` | `path`, `exists`, `loaded`, plus the file's **`mtime`** — the token D7's save guard compares. |
| `caps` | the numbers the form enforces: max modes, max id chars, the id pattern, max label chars, max persona bytes, max persona lines. |
| `defaults` | the built-in document (the same object the seeder writes) — the draft the dialog offers **only** when the file is absent, and the source for the "reset this persona" affordance. |

Decide and document, in the docstring, how each unrepresentable case answers, so
eds-02/eds-06 do not have to guess: a file that is absent, a file that is not
valid JSON, a root that is not an object, a bad `version`, `agents` not an object,
an entry whose value is not an object, an id outside the pattern, and an entry
that names a reserved id. The last one keeps today's shape: a **note** naming the
pointer, and the entry is **not** in `entries` (there is nothing there to edit) —
and it stays a note, not an error, so it does not block a save.

**The document read is not cached.** `resolve_modes()`'s cache keys on
`(mtime_ns, size)` and that is right for a per-request reader; a writer's baseline
must be the disk as it is now, so a cached `mtime` would defeat the conflict
guard. Say so in the docstring.

Also add one constant: `MAX_MODE_PERSONA_LINES = 16` — the soft prose budget the
form shows as "remaining line cap". It already exists as a literal inside
`tests/test_agent_modes.py::test_an_editable_persona_stays_inside_its_line_budget`;
point that test at the new constant so the number has one home. The budget is
**soft**: it is display, never a validation rule, and nothing in the loader's
verdict changes because of it.

## Rules this task must not break

* `resolve_modes()` and `resolve_mode()` keep their exact current contracts —
  same payload, same order, same leniency. AGENTS.md: *"a broken file never raises
  and never blocks — the built-in defaults answer and `findings` name the JSON
  pointer."*
* The loader stays pure core: `tests/test_agent_modes.py::test_the_loader_imports_no_transport_no_model_and_no_hub`
  forbids importing `fastapi`, `src.api`, the hubs, `llmkit`, `chat`, `bench`,
  `batch` **and `editorfs`**. This task writes nothing, so
  `test_the_loader_writes_nothing` (a per-function write allowlist) must stay
  green without touching its allowlist.
* Do not add a second validator. Reuse `_validated_document`, `_one_mode`,
  `_read_text`, `_mode_id_problem` and the finding voice (`_pointer_error`,
  `_file_error`) — a finding the dialog shows is *the loader's own sentence*,
  which is what makes it traceable to the user's JSON.
* The caps are module attributes and stay patchable (several existing tests patch
  them); build the `caps` payload per call from the attributes, not from a
  literal.
* `agent_modes_defaults.py` is not touched: the seeded bytes and the loader must
  remain one object, and the personas are prose someone will reword.

## Tests to add (in `tests/test_agent_modes.py`)

* A well-formed file: `entries` are the file's entries in the file's order, with
  the persona text verbatim, and **no** app-owned entry and **no** built-in
  fallback appears.
* A file with one broken entry: that entry is **present** in `entries` with its
  finding attached — the case D2 exists to prevent losing.
* A file with an entry whose value is not an object, and one with an id outside
  the pattern: both present, each with its finding.
* A file that declares a reserved id: a note finding naming the pointer, and the
  entry absent from `entries`.
* An absent file: `exists` false, `entries` empty, `defaults` present, no error
  findings.
* A file that is not valid JSON, a root that is not an object, a wrong `version`,
  `agents` not an object: each answers with its file-level finding and no entries.
* `mtime` is the file's, and a rewrite of the file changes it (this is the token
  eds-02's guard compares).
* `readme` survives in the payload in the shape the file wrote it.
* `caps` carries every number, and patching one changes the payload.
* The persona line budget: the constant exists, is a positive int, and the prose
  test now reads it instead of a literal.

## How to tell it is done

`pytest tests/test_agent_modes.py tests/test_agent_modes_seed.py -q` green, and
`pytest -q` green overall (this task changes no behaviour any other test sees).
`git diff --stat` shows only `src/core/agent_modes.py` and
`tests/test_agent_modes.py`. Move this file to `todos/done/`.
