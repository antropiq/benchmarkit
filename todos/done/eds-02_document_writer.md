# eds-02 — `agent_modes_write.py`: the whole-document writer

**Spec:** D3 (the server-side half), D4 (the reserved-id refusal), D6, D7, D8 and
its constraint; §4 "Files touched" row 2.
**Depends on:** eds-01 (the document read — the shape this writer is the mirror
of, and the `mtime` token it compares).
**Files it touches:** `src/core/agent_modes_write.py` (new),
`tests/test_agent_modes_write.py` (new). `src/core/agent_modes.py` may gain only
what the writer genuinely needs from the read side (a shared finding/pointer
helper, the path builder) — no write lands in it.

## Why a new module

The backup lives in `editorfs.constants.BACKUP_DIR`, and
`tests/test_agent_modes.py::test_the_loader_imports_no_transport_no_model_and_no_hub`
forbids `agent_modes.py` from importing `editorfs`. The spec records this so it is
not discovered as a failing test mid-task: the **write half lives in its own
module** beside the loader. The seeder stays where it is — the seeded bytes and
the loader must remain one object.

## What to deliver

One public entry point, roughly `save_agent_modes_document(document, mtime_token)`
(shape is yours to settle), which:

1. **Validates the incoming document as a writer**, refusing rather than dropping
   (D3's server-side half): a body that declares a **reserved id** is refused; a
   body carrying an entry the loader would drop (unknown key, a tool flag that is
   not a boolean, a persona over the byte cap, an id outside the pattern or over
   the length cap, an entry that is not an object) is refused; more entries than
   the mode cap is refused; a `default` that names no declared mode is refused.
   Each refusal names the pointer, in the loader's own finding voice, so the
   dialog can show the same sentence the reader would.
2. **Writes D6's shape**: `version` from the app's constant (never from the body);
   `readme` preserved verbatim when the file had one, absent when it had none, and
   the built-in one when the save **creates** the file; `agents` in the order the
   body gave; `default` from the body.
3. **Chooses the persona form correctly**: emit the **list form only when no line
   is blank**, and the string form otherwise — a list holding a blank line is
   invalid to this app's own reader, and writing a file the reader refuses is the
   exact bug the round-trip tests exist to catch. The text itself is never
   stripped or rewritten.
4. **Guards against a stale write (D7)**: the body carries the `mtime` the read
   returned; a disk `mtime` that has moved answers a conflict error carrying both
   numbers (mirror `editorfs.WorkspaceSaveConflict`'s shape — an object the client
   can act on, not prose). Nothing is merged and nothing is overwritten.
5. **Backs up what it replaces (D8)**: the same pre-save backup the Editor's saves
   use — call the public `editorfs` backup helper rather than inventing a second
   chain, so one folder, one retention sweep and one digest rule cover it.
6. **Creates exclusively** when the file is absent (`O_CREAT | O_EXCL`): a file
   that appeared between the read and the save is a conflict, not something to
   overwrite. A missing `.nojae` folder is created with one `os.mkdir` level, the
   way the seeder does — never `makedirs`.
7. Replaces atomically (temp file + `os.replace`) and answers the **new** `mtime`
   plus the new verdict — the resolved mode list in the shape `GET
   /api/agent-modes` publishes — so eds-03's client can reload the dropdown
   without a second round-trip (D9).

Raise typed errors (conflict / invalid document / unwritable) that eds-03 maps to
409 / 422 / the write family's usual codes. No FastAPI import in this module.

## Rules this task must not break

* `agent_modes.py` gains no write and no `editorfs` import. If the writer needs a
  helper from it, the helper is a read-side one and stays read-only.
* The loader's leniency is untouched: this module is a **writer** and may be
  strict; a broken file still chats (the modes lot's D11).
* AGENTS.md: *"no write destroys bytes — a save backs up what it replaces"*. A
  save that fails validation writes nothing at all.
* `tests/test_editorfs.py::test_editorfs_exposes_the_guarded_write_and_nothing_beyond_it`
  refuses any *new* write spelling exported from `editorfs`. Do not add one: reuse
  the existing backup helper and do the temp-file + `os.replace` here.
* `tests/test_write_surface_isolation.py` and `tests/test_editorfs_write.py` scan
  `src` for the editor write helpers (`write_file`, `backup_for_write`,
  `create_path`, `rename_path`, `move_paths_batch`, `trash_path`) and assert the
  only caller outside `editorfs` is `src/api/routes.py`. **eds-02 must not call
  them from a core module** — the backup call belongs in eds-03's route, or the
  allowlist of that scan is widened deliberately and named in the test's
  docstring. Pick one, keep it consistent with the guard, and say which in the
  module docstring.
* Nothing under `src/core` imports the API layer.
* The reserved folder stays invisible to the tree, `/api/fs/*`, search and git
  status: this writer is reached only by its own purpose-built route, never by a
  path parameter.

## Tests to add (`tests/test_agent_modes_write.py`)

* **Round-trip**: save a document, then read it back through `resolve_modes()` —
  zero findings, same modes, same order, same persona text byte for byte.
* A persona containing a blank line round-trips (it lands in the string form); a
  persona with none lands in the list form.
* `readme` is preserved verbatim; a file with no `readme` gains none; a save that
  creates the file gets the built-in one.
* `version` in the body is ignored — the app's constant is what lands.
* A reserved id in the body is refused and the file on disk is unchanged.
* Each invalid-entry kind is refused naming its pointer.
* A `default` naming no declared mode is refused.
* The mode cap is refused (patch the cap down, as the existing tests do).
* **Conflict**: a stale `mtime` token answers the conflict error carrying both
  numbers and the disk keeps the bytes the user put there.
* **Exclusive create**: no file → created; a file that appeared after the read →
  conflict, not an overwrite.
* **Backup**: after any save the previous bytes are recoverable from the app's
  backup folder (follow how `tests/test_editorfs_write.py::TestBackups` locates
  and asserts it).
* A save that failed validation leaves no temp file and no half-written file.
* The save response carries the new `mtime` and the new verdict in the reader's
  shape.

## How to tell it is done

`pytest tests/test_agent_modes_write.py tests/test_agent_modes.py -q` green and
`pytest -q` green overall. `git diff --stat` shows the two new files plus, if you
chose the widened-allowlist option, the one guard test you named. Move this file
to `todos/done/`.
