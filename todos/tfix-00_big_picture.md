# Windows test failures — big picture

**What is broken:** nine pytest cases fail on this machine (Windows) and are
green on Linux. Nothing here is a feature request: each task takes **one failing
test id**, makes it pass on Windows without breaking it on Linux, and stops.

No task here is "fix the platform issue". Each one is scoped to a single named
test, because the nine failures are three unrelated platform facts and a fix that
spans them all is how a green-on-Linux suite gets broken.

## The three groups (context only — no task depends on this reading)

| group | tests | what the failure output says |
|---|---|---|
| **A. written bytes carry `\r\n`** | tfix-01, tfix-03, tfix-05, tfix-06, tfix-07, tfix-09 | every one compares `read_bytes()` against bytes built with `\n`; on Windows the file the app wrote holds `\r\n` |
| **B. a symlink cannot be created** | tfix-04 | `OSError: [WinError 1314]` raised out of the test body |
| **C. a path spelled for POSIX** | tfix-08 | an expected `/opt/py/python` compared against a resolved command list |

Group A is one symptom in four different writers (the agent-modes save, the MCP
seed, the LSP seed, the workspace-settings seed). Group B is one symptom in one
listing endpoint. Fix them one test at a time; do not generalise across a group
in a single task.

## Rules every task in this lot must honour

Quoted from `AGENTS.md`:

* **"Tests stay hermetic**: an environment flag disables the hubs, a test patches
  a hub's declared file location through that hub's own constants, and
  `tests/test_env_isolation.py` keeps a test from reading a real model
  connection."
* **"One LLM layer."** — no task here may add a model call; none of these tests
  needs one.
* **"Core owns logic, the API owns transport."** and **"nothing under `core` may
  import the API layer."** — a fix that lands in `src/core/...` must not reach
  into `src/api/...`.
* **"The connectors answer the workspace level for their own root**, each reading
  that policy in its own process, so a read-only folder is uneditable by an agent
  too — **the app and the connector processes share one parser, one matcher and
  one refusal sentence.**" — the MCP/LSP declaration files and the path policy are
  shared between processes; a change to what is written or how it is read is a
  change on both sides.
* **"The declaration files are the only source of truth for what runs."** ...
  **"A value may reference an environment variable so one declaration works on
  every machine; an unresolved name is kept verbatim with a note, never
  blanked."**
* **"A hidden path is indistinguishable from an absent one to a reader and named
  to a writer"** — the seeds' "a failure is a note, never a refusal" behaviour
  stays.
* **"Teardown kills process *trees*, not wrappers"** — nothing in this lot touches
  teardown.
* **Reserved-folder rule**: "its files are created only when absent, so an edited
  file is never clobbered" — a seed must stay an exclusive create.

## Conventions this lot works inside

* **Run only the test you are fixing.** Never the whole suite — it times out. The
  exact command is written in each task file.
* **The suite already has a way to say "this platform cannot run this test"**:
  `tests/conftest.py` defines probes and guard markers (`requires_symlink`,
  `posix_behaviour`, `windows_behaviour`, `case_sensitive_volume`, ...) built on
  one `skip_unless` helper, and each marker carries the sentence naming *which
  invariant the skip gives up*. A test that needs a capability the platform will
  not grant is **marked**, not deleted, and a skip must say what it would have
  checked. `tests/test_batch.py` already marks its other symlink tests this way —
  the failing one is the odd one out.
* **Byte-for-byte promises are the point of several of these tests** ("the
  template is the example verbatim", "one `O_CREAT | O_EXCL` copy"). A task may
  not resolve a byte comparison by making the assertion loose unless it says
  which promise it is keeping and how.
* **AGENTS.md is injected into every chat turn and is re-read per request** — if a
  task changes an invariant it describes, it updates that line in the same task
  instead of leaving the file lying.
* **A task is done when its file moves to `todos/done/`.**

## Order

tfix-01 … tfix-09 are independent of each other (no task depends on another).
They are numbered in the order the failures were reported: the three agent-modes
tests, then the batch symlink, then the LSP, MCP and workspace-settings seeds.
Run them one at a time; each one ends with its own test id passing and the tests
that were already passing in the same file still passing.
