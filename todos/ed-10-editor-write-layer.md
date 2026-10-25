# Future version — the write layer (create / edit / delete)

**Not part of the current lot.** Decision D4: this lot ships a read-only editor.
This file is the design for the version that turns the Editor tab into a standard
editor — create new files, edit existing ones, delete them — written down now so
the read-only steps do not paint themselves into a corner.

Depends on: ed-01 … ed-04 and ed-09. It is the only step that gives the app's own
API a way to modify or destroy a file — the terminal of ed-06/ed-07 can already
do that through a shell, but only a human typing at it, never through a route the
UI calls — so this step carries the guard suite and the conflict protocol.

## Scope

| Capability | Shape |
|---|---|
| **Edit** | the CodeMirror pane becomes writable; `Ctrl+S` and a `Save` button; `dirty` state; `beforeunload` warning |
| **Create** | a new file (with parent directories) from the navigator's context menu or a `New file` button; the empty buffer then opens in the editor |
| **Delete** | a file **or** a directory from the navigator, always with a confirm, always behind a backup |
| **Rename / move** | both ends policy-checked; open tabs re-pointed |

Explicitly out, still: chmod, symlinks, and any operation whose effect is not
expressible as a path inside the workspace.

## The two invariants this step must not break

1. **AGENTS.md's "Tools are read-only by design".** The write endpoints are
   browser-only. Nothing in `core/mcphub/`, `core/lsphub/`, `core/chat.py` or a
   connector may call them, and no new MCP/LSP tool wraps them — a test should
   assert the absence of such a tool rather than trust a comment.
2. **The model and the human must not overwrite each other silently.** ed-09
   made the buffer follow the model; this step gives the human a way to write,
   which re-opens the race ed-09 was allowed to ignore. Hence the gate below.

## Gate: the editor is read-only while a conversation is in progress

The rule as stated: **no create, no edit, no delete while a WorkBench
conversation is running**; outside one, the editor behaves like a standard
editor. It is enforced twice, deliberately.

**In the UI** — while a reply is streaming, the pane flips to the same
`editable: false` + `readOnly: true` pair this lot uses permanently, the
`read-only` badge of ed-02 reappears, and the navigator hides New / Rename /
Delete. The user sees the mode they are in rather than discovering it by
pressing `Ctrl+S` and being refused.

**On the server** — because a browser is not an authority, the write endpoints
also refuse. `src/api/routes.py` already keeps process-wide state for the one
in-flight reply: `_CHAT_HANDLE` (`:298-327`, armed by `_arm_chat_handle` when
`/api/chat` starts and cleared when it ends) and `_CHAT_TASKS` (`:294`), which
three existing endpoints use to answer **409 "… while a chat reply is
streaming"** — the MCP restart (`:854`), the LSP restart (`:995`) and the
working-location change (`:1079`). The fs write endpoints reuse that exact
idiom, so the app gains no new kind of lock:

```python
if _CHAT_HANDLE is not None:          # a reply is streaming, tools included
    raise HTTPException(409, detail="Cannot edit files while a chat reply is streaming")
```

Why a hard mode switch instead of a merge or a three-way diff: the tool loop can
run for minutes and may re-read a file after any human edit, so a human buffer
held across it is not merely stale, it is *in the way*. Switching the mode is
honest, cheap, testable, consistent with the three endpoints above, and it is
what the read-only lot of this feature already implements.

Note the asymmetry to keep: the **model** keeps writing during a reply — that is
the whole point of WorkBench — and ed-09 pushes those changes into the buffer.
Only the human's side is shut.

### What counts as "a conversation in progress" — decided: a reply is streaming

Two readings were on the table; **(a)** was chosen, and the reasoning belongs to
the design because it states what the app does *not* try to do:

| Reading | Server can know it? | Verdict |
|---|---|---|
| **(a) a reply is streaming** — waiting, reasoning, answering, or a tool round (`useChat.state.phase !== 'idle'`) | yes, today, via `_CHAT_HANDLE` | **CHOSEN** — the race is closed exactly where it exists (tools writing) and nowhere else |
| **(b) a WorkBench conversation is open** — even idle, with history on screen | no: the server keeps **no conversation state** (the client re-posts the whole history each request) | **rejected** — would need a new "session open" protocol, and would shut the editor for as long as a tab is open |

The workflow that choice assumes is a deliberate part of the product:

> the model answers → the user switches to Editor and edits → back to WorkBench,
> where they *tell* the model what changed.

Nothing detects that hand-back, narrates it, or guards against doing it badly:
**the user owns it.** That is why the gate is one narrow window rather than a
session-long lock — an idle conversation is not a writer, so it has no claim on
the pane. (Recorded as D9 in ed-00.)

## Endpoints

```jsonc
// PUT /api/fs/file
{ "path": "src/config.py", "text": "…", "lineEnding": "\n",
  "ifUnmodifiedSince": 1761371340.41 }        // mtime from the last read
// 200 → { "path": "…", "size": 26010, "mtime": 1761373111.02 }
// 409 → { "error": "conflict", "diskMtime": …, "bufferMtime": …, "diskSize": … }

// POST /api/fs/create     { "path": "web/src/x.js", "text": "" }  → 409 if it exists
// POST /api/fs/rename     { "from": "a.py", "to": "b.py" }
// POST /api/fs/delete     { "path": "scratch/", "recursive": true }
```

Server side (`editorfs/files.py`), in order, for every one of them:

1. `resolve_in_workspace(rel, for_write=True)` — the **write** list
   (`DENY_COMPONENTS_WRITE`, `path_policy.py:96`), so `mcp_connectors/`,
   `lsp_connectors/`, `.git/`, `.ssh/`, `.env*` and every secret file are refused
   here too. Rename and delete check **both** ends; a recursive delete walks the
   subtree and re-checks each entry, refusing the whole call on one hit;
2. the `_CHAT_HANDLE` gate above;
3. **backup before mutation**, reusing `fedit`'s existing convention —
   `logs/fedit_backups/<sha8>_<name>_<timestamp>.orig` — so a bad save, a bad
   rename and a bad delete are all one action from a restore. A delete keeps the
   backup *and* records the tree layout, or restoring a directory loses its
   shape;
4. **atomic replace** for an edit: `<name>.tmp` in the same directory then
   `os.replace`, never a truncate-and-write;
5. `ifUnmodifiedSince` vs the current `st_mtime` → **409**, carrying both mtimes.
   This precondition is new to the app: `fedit_mcp` re-reads and fails on a
   context mismatch, which is enough for an agent and not enough for a GUI
   holding a long-lived buffer. The token comes from the read endpoint (ed-03) or
   from ed-09's probe — see ed-09's *Concurrency honesty* for why it must be
   re-statted after the last successful read;
6. create: parents are made only for the requested leaf's chain, each re-checked
   against the write policy, and an existing path is 409, never an overwrite;
7. size cap on every payload (413 past `MAX_EDITOR_BYTES`); no `chmod`, no
   symlink traversal (a symlink is resolved and re-checked), no NUL or separator
   tricks in a path (422).

## What changes in the UI built by ed-03/ed-04

* The `editable: false` + `readOnly: true` pair stops being permanent and
  becomes **derived from the conversation state**: off while no reply streams,
  on while one does — always applied as a pair, never one lever alone (ed-04
  explains why). One `watch` on `useChat.state.phase` drives it.
* `useEditorFiles.state.open[i]` gains `dirty`; `closeFile` and tab-switch-away
  confirm while dirty; `beforeunload` warns while anything is dirty.
* `Ctrl+S` (active tab) and `Ctrl+Shift+S` (all dirty tabs), both inert while
  read-only; the `read-only` badge of ed-02 shows **during a reply** instead of
  permanently.
* **ed-09's silent reload stops being silent.** It was silent *because* nothing
  could be lost; with a dirty buffer, an automatic reload destroys work. New
  rule: clean buffer → silent reload as before; dirty buffer → keep the text,
  mark the tab `conflict`, and offer *Reload* / *Diff* / *Overwrite* explicitly.
  This is the single behavioural change ed-10 forces on ed-09, and it is why the
  `stale` flag already exists.
* `autocompletion: false` flips back to `true` (the reason it was off — a buffer
  that cannot be edited — no longer holds).
* The navigator gains a context menu: New file, Rename, Delete; `refreshBranch`
  on the affected parents; a tab whose path was renamed follows the new path.

## Tests — the guard suite lives here

`tests/test_editorfs.py`, write half, one case per rule:

1. a normal edit: disk matches, `mtime` advances, a backup exists, no `.tmp` left;
2. `path=../outside.txt` → 422, the outside file untouched;
3. `mcp_connectors/console_mcp.py`, `lsp_connectors/lsp_servers.json`, `.env`,
   `id_ed25519`, `.git/config` → 4xx, each naming the rule;
4. symlink inside the workspace pointing outside → 422, target intact;
5. stale `ifUnmodifiedSince` → 409, disk unchanged;
6. a chat reply in flight (seed `_CHAT_HANDLE`, or drive `/api/chat` with a
   stubbed `stream_chat`) → 409 on **all four** write endpoints, and the message
   matches the wording used by the MCP/LSP/path-location guards;
7. create over an existing path → 409; create with denied parents → 4xx;
8. rename onto an existing path → 409; rename across the policy boundary (out of
   `web/src` into `mcp_connectors/`) → 4xx;
9. delete of a directory containing one denied entry → nothing deleted at all
   (all-or-nothing), and the backup holds the whole subtree;
10. delete a running hub's declaration file → refused; a missing file → 404;
11. payload over the cap → 413; CRLF round-trip writes CRLF;
12. read and write guards differ where `path_policy` says they should — a
    readable-but-not-writable path is served on read and refused on write,
    locking the asymmetry rather than flattening it into a blanket denial.

Cost, rough: ~150 lines Python, ~120 Vue, and the guard suite roughly doubling —
which is the reason it is its own version rather than an addendum to ed-04.

## Manual checklist

* Ask the model (WorkBench) to edit a file that is open and **unmodified**: the
  buffer refreshes silently; the pane never becomes dirty on its own.
* Type a local edit, do **not** save, go to WorkBench and start a reply that
  touches the same file: the pane goes read-only keeping the typed text, and
  when the model's change arrives the tab reads `conflict` — nothing is
  overwritten until Reload / Diff / Overwrite is chosen. Save it after the reply
  ends and the `ifUnmodifiedSince` check reports the model's newer `mtime`
  first, so the choice is asked once rather than silently lost.
* Start a reply, switch to the Editor tab: the pane is already read-only, the
  badge is back, typing does nothing, New / Rename / Delete are gone. When the
  reply ends the pane becomes writable again and ed-09's reload has already
  folded the model's changes in.
* Bypass the UI (`curl -X PUT`) during a reply: 409 — the mode is not a UI hint.
* Delete `scratch/` by hand in the tree: confirm once, and one restore action
  brings it back.
* Rename an open file: the tab follows the new path, the tree re-lists the parent,
  the terminal's `$PWD` is unaffected.
* `curl -X PUT` the endpoint by hand with a denied path: refused by policy —
  reaching it from outside the UI must be as hard as from inside it.

## Naming note for that version

The tab stays **Editor** even while read-only. A capability-qualified name
(`Files`) would have to be renamed when ed-10 lands and would not match the brief.
The read-only state is stated where it is seen — the badge ed-02 puts in the
editor area — not in the tab label.
