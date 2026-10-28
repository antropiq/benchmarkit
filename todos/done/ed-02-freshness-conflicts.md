# ed-02 — Freshness: knowing when the disk moved under a buffer

Make the buffers honest about the world. Once a save exists, the dangerous case
is not "the user typed" — it is "the WorkBench, the terminal, or another tab
wrote this file while the pane held it". Today nothing notices: the `stale` flag
on every record is set to `false` and never set again, and `reloadFile()` is
never called by anything.

The route this needs **already exists** — `GET /api/fs/stat`
(`api/routes.py:1201`, core `editorfs.stat_paths`, `files.py:175`) takes a list of
exact paths and answers `mtime`/`size`/`exists`/`dir`, answering `exists: false`
for a missing file and 422 (never `exists: false`) for a denied one. This task
only adds the client that uses it, plus the state that result needs.

Depends on: ed-01 (the write layer is what makes staleness matter; the 409 shape
is already defined there).

## 1. The poll

A new function in `useEditorFiles.js` — `probeOpenFiles()` — not a watcher, not a
tree poll:

* one request per tick: `GET /api/fs/stat?path=<a>&path=<b>…` for the **file
  records only** (a `kind: 'diff'` record has no buffer to be stale);
* started when the tab strip is non-empty and the Editor tab is the visible one
  (`useBenchmark.state.activeTab === 'editor'`), stopped otherwise — a background
  poll of a hidden pane is a request cost with no reader. Also stop while
  `useChat.state.phase !== 'idle'` (a streaming reply is a known writer; the
  pane is read-only anyway, and probing it mid-stream just produces conflicts the
  user cannot act on);
* interval `FRESHNESS_POLL_MS = 5000` as a module constant in
  `useEditorFiles.js` (mirroring the file's existing top-level constants), so a
  test can shorten it;
* concurrent calls join the running request (the `useGitStatus.load` inflight
  pattern, `useGitStatus.js:80`);
* a probe that fails (network, 422, 500) leaves every record as it was and sets
  `state.probeError` — an unavailable probe is not evidence that files changed.

Per record, compare `mtime` from the probe against `record.mtime` with the same
epsilon idea ed-01 used server-side (a `1e-6` float wobble is not an edit):

| Disk | Buffer | Result |
|---|---|---|
| mtime matches | any | clear `stale`; nothing else happens |
| gone (`exists: false`) | clean | `deleted: true` — the tab chip shows it as deleted, the pane says so; closing it needs no confirm |
| gone | dirty | `deleted: true` **and** `conflict: 'deleted'` — the work exists nowhere but the buffer |
| mtime differs | clean | silent `reloadFile()` (the buffer is the disk's copy anyway, and reload resets the baseline, clearing the dot) |
| mtime differs | dirty | `stale: true`; if it is the active tab, surface the conflict (§2); non-active tabs just carry the flag |

"Clean → reload silently" is the whole point of keeping `baseline` on the record:
the comparison is exact, so a buffer the user typed their way back to is clean
and reloading it costs nothing they can lose.

## 2. The conflict state

Add to each record: `conflict` (`null | 'stale' | 'deleted'`) and
`conflictAt`. Add `state.conflict` — `{ key, kind, diskMtime }` — for the **one**
conflict on screen, so the dialog is about the buffer the user is looking at.

A dirty stale buffer gets three answers, in this order of preference, and all
three must be reachable by keyboard:

1. **Reload** — discard my edits, take the disk version (`reloadFile()`);
2. **Compare** — open a second Monaco surface, my buffer vs the disk text, then
   choose. Implementation: `openConflictedDiff(key)` pushes a `kind: 'diff'`
   record (`original:` the freshly-read disk text, `modified:` my buffer) reusing
   the existing `MonacoDiffViewer` — no new diff component. Its note line must
   say "left: as it is on disk now — right: your unsaved buffer", which is the
   opposite polarity of the git diff tabs (there the right side is also the
   disk, and the caption in `EditorArea.vue:39-48` says so);
3. **Overwrite** — save anyway: `PUT /api/fs/file` with **no**
   `ifUnmodifiedSince`, the escape hatch for "I know, I merged by hand". This is
   the only caller allowed to omit the token, and ed-01's log line for a
   token-less save is how that shows up if it is ever a bug.

The dialog itself lives in a new `components/EditorTreeDialogs.vue`, created by
this task and extended by the later ones (create/rename/delete confirmations all
belong to one file rather than three). It follows the existing `<dialog>` +
`showModal()` pattern of `RegistrationMemoryDlg.vue`.

## 3. What the pane shows

* a stale or deleted buffer's chip carries a marker beside the dirty dot
  (`.editor-tab-stale`, style in `theme.css`, same row as `.editor-tab-dirty`),
  with a `title` naming the reason and a screen-reader span like the dirty one
  (`(changed on disk)` / `(deleted on disk)`);
* the read-only badge's `title` in `useEditorMode.js` gains the staleness case —
  the badge is where a pane-wide state is explained;
* the empty/refusal message line (`state.message`) stays what it is: the conflict
  is *not* a message, because a message is cleared by the next click and a
  conflict must survive one.

## 4. Deliberately not done here

No `fs.watch`/watchfiles (a watcher on a workspace of `$HOME` is a file-descriptor
problem, and the app already has two live writers it could have asked instead:
the chat stream and the terminal — neither can report a path reliably). No merge
editor: "Compare" is a view, the human does the merge. No conflict for a tab that
is not open: a file the user is not editing needs no ceremony. No polling of the
tree (`useWorkspaceTree` is untouched here; ed-07 adds its own `refreshBranch`).

## 5. Tests

`web/tests/useEditorWrite.test.js` (new; extend it in later tasks) — stub
`global.fetch`, drive the composable directly, the way
`useEditorFilesDiff.test.js` already does:

1. mtime unchanged → no flag, no reload;
2. mtime changed + clean → the buffer text and `baseline` are the new text,
   `dirty === false`;
3. mtime changed + dirty → `stale`, `conflict: 'stale'`, text untouched;
4. `exists: false` + dirty → `conflict: 'deleted'`;
5. float wobble within the epsilon → nothing;
6. a probe that 500s → `probeError` set, no record changed;
7. diff records are never probed (the request names file paths only);
8. two overlapping probes issue one request;
9. the conflicted-diff record has the disk text on the left and the buffer on the
   right (the polarity rule above).

A pytest case that `GET /api/fs/stat` still answers the same for a denied path
(422, not `exists: false`) belongs to ed-01's suite if it is not already in
`tests/test_editorfs.py` — do not fork the read-route tests.

## 6. Acceptance

* Edit a file in the pane, then change it from the terminal (or the WorkBench):
  within one tick the chip is marked and the conflict offers its three answers;
  Reload takes the disk version, Compare shows both, Overwrite keeps mine.
* Change a clean buffer's file externally → it updates silently, no dot, no
  dialog.
* Delete an open clean file from the terminal → the chip says deleted; nothing
  was lost.
* `cd web && npm test` green; `pytest` green and unchanged.
* No request is made while the Editor tab is hidden or a reply streams.
