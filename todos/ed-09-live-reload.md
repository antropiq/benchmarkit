# Step 9 — Open files follow the model's edits (decision D4's other half)

The pane is read-only, so the model is the only writer the human shares it
with. When a WorkBench conversation changes a file, the open buffer and the
tree must show the new state — no reload of the page, no stale text.

Depends on: ed-01 (tree, `mtime` on directories), ed-03 (`mtime` on open
records), ed-04 (a mounted CodeMirror view to swap documents in).

## The rule, as decided

Here "a conversation is running" means **a reply is streaming** — the only state
the app can observe today, and the window D9 settles on for ed-10's write gate
too. It is deliberately narrow: a WorkBench tab left open with history on screen
is *not* a running conversation. Between two prompts the human is free to edit
in the Editor tab and come back to tell the model about it; that hand-back is
the user's responsibility, not something the app enforces or detects.

* While no conversation is running: nothing is refreshed on its own. The tree's
  **Refresh** button and re-opening a file are the manual path (a watcher is
  parked in ed-08-G).
* While a conversation **is** running: every file already open, and every
  directory already expanded, is kept current. A changed file is re-read into
  its buffer; a changed directory is re-listed.
* The reload is **silent** (no confirm, no "file changed on disk" banner) — a
  legitimate consequence of read-only: there is no local edit that a reload
  could destroy. That is exactly the assumption ed-10 has to revisit when it
  makes the pane writable.

## Why a `stat` probe and not an event from the write

The chat stream already announces tool calls (`core/chat.py:482` emits
`tool_call` with `tool`, `server` and `argsPreview`; `:510` emits `tool_result`
with `id`, `ok`, `preview`), so it is tempting to read the changed path out of
those frames. That does not work, and the reason is worth recording:

* `argsPreview` is **truncated to 200 chars** (`_preview`, `core/chat.py:87`),
  so the `path` of an `apply_edit` call may not even be in it — and a payload
  carrying a diff is far longer than the cap;
* the `console` connector writes through `execute_command`, whose arguments are
  an argv, not a path: `sed -i`, `touch`, `git checkout` all change files that
  no frame names;
* the terminal (ed-06) writes with no frame at all, and a benchmark run writes
  `results.json` without a tool loop.

So the correct question is not *"which file did the model touch?"* but *"is what
I am looking at still what is on disk?"*. That is a bounded probe: the number of
paths probed is the number of tabs the human opened, regardless of how many
files a tool changed.

## Endpoint: `GET /api/fs/stat?path=…&path=…`

```jsonc
// GET /api/fs/stat?path=src/config.py&path=results.json
{
  "workspace": "/mnt/bigfoot/dev/benchmarkit",
  "entries": [
    { "path": "src/config.py", "exists": true,  "mtime": 1761373111.02, "size": 26010 },
    { "path": "results.json",  "exists": false }
  ]
}
```

* `path` is repeatable; the request carries at most
  `editorfs.constants.MAX_STAT_PATHS = 64`, otherwise 413 — the cap is what makes
  the probe a cheap operation rather than an accidental directory walk.
* **Deleted is data, not an error**: a missing file answers `exists: false` with
  200, because the whole point is to notice a deletion or a rename. 404 is
  reserved for a path that cannot be answered at all.
* Every path goes through `editorfs.policy.resolve_in_workspace(rel)` — the same
  gate as the tree and the read (D3), 422 for an escape or a deny-list entry.
  Rejected for the **whole** request rather than per entry: the client can only
  ever send paths it received from the tree or from a successful read, so a
  denial is a bug, and a per-entry `denied: true` would hide it.
* `mtime` here is the same float the read endpoint returned, so the comparison
  is exact. A `size` change is the fallback predicate: filesystems with coarse
  mtime granularity can reuse one tick, and a same-size, same-tick write is rare
  enough to accept rather than to hash for.
* Implementation in `editorfs/files.py::stat_paths()`; the route stays a thin
  delegate. No new Python dependency.

## Composable: extend `useEditorFiles.js` — not a new singleton

The freshness question belongs to the state that owns the open files, so this
step adds to ed-03's composable instead of introducing a fourth one:

```js
// state.open[i].stale     // disk differs from the buffer
// state.open[i].missing   // the path no longer exists
checkFreshness()           // one stat for every open path, then reload what moved
```

and a small trigger module, `composables/useEditorLive.js`, whose only job is
*when* to call it:

| Trigger | Source |
|---|---|
| a tool call finished | `useChat`'s existing `tool_result` handling (`useChat.js:752`) — call `checkFreshness()` debounced ~400 ms, so a five-tool reply costs one probe, not five |
| the reply ended | a `watch` on `useChat.state.phase` returning to `'idle'` — `finish()` is module-private (`useChat.js:816`, not in the returned object), so the transition is the hook rather than a new export. One final probe, which also covers a model that wrote and then paused |
| while streaming | an interval, capped at one probe per 2 s, and only while `useChat.state.phase !== 'idle'` |
| a run finished | `useBenchmark` — `results.json` is the file the run rewrites |

The predicate for "a conversation is in progress" is `useChat`'s own:
`state.phase !== 'idle'` (`useChat.js:6`), with `state.session.startedAt !== 0`
as the equivalent already used by the stopwatch. No new running flag.

Deliberately **not** filtered by tool name: skipping the probe for
`lsp_pyright_diagnostics` would save one `stat` per call and make the rule
depend on a list of which tools are read-only — a list that goes stale the day a
connector is added. The probe is bounded by the number of open tabs, so
guessing is not worth it.

## What a reload does to the view

CodeMirror's document is replaced with a `StateEffect.appendConfig`-free
transaction (`view.dispatch({ changes: { from: 0, to: oldLen, insert: text } })`),
which keeps the grammar extension, the theme and the search state alive:

* **cursor**: remembered as `{ line, column }`, then re-applied clamped to the new
  document — a tool edit above the cursor keeps the human looking at the same
  code; a tool edit that removed their line lands them at the nearest one
  instead of at position 0;
* **scroll**: `view.scrollDOM.scrollTop` restored after the transaction, then
  `EditorView.scrollIntoView` only when the remembered line left the viewport;
* **selection**: kept by the same line/column clamp, so a copy is not silently
  reset mid-flight;
* a transient `↻` badge on the tab chip for ~1.5 s, so the change is attributable
  rather than mysterious — the pane did not "blink by itself";
* `missing: true` → the buffer keeps its last text, the chip dims, and the pane
  header says `deleted from disk` in the app's `.caption` idiom. Closing such a
  tab is a plain close: no resurrection, and (this lot) nothing to lose.

For the tree, `useWorkspaceTree.refreshBranch(path)` re-lists a directory whose
recorded `mtime` (ed-01) changed, and **re-applies the set of expanded paths** so
a deep view survives; a branch that no longer exists collapses.

## Concurrency honesty

The probe reads the same file the model may be writing at that instant. Two
consequences, both accepted deliberately:

* a read can land between `fedit`'s backup and its `os.replace`, showing a
  half-applied state for one probe cycle — the next probe corrects it;
* `mtime` and `size` come from a single `os.stat`, so they are consistent with
  each other, but the following read may see a newer file: the buffer then holds
  text *newer* than the `mtime` it records. Harmless while the pane is read-only,
  since the `mtime` is only used to detect change. ed-10 must not reuse that
  value as a write precondition without re-statting after each successful read —
  noted there, not fixed here.

## Tests — extend `tests/test_editorfs.py`

1. two open paths, one edited on disk: that entry's `mtime` advances, the other's
   does not;
2. a deleted path → 200 with `exists: false` and no `mtime` key;
3. 65 paths → 413; a duplicate path is answered once (deduplicated, cheap);
4. `../outside` or `.env` in the list → 422 for the request, and no entry list
   leaks the resolved absolute path;
5. a directory path is answered (`exists: true`, its own `mtime`) — ed-09 relies
   on it for branches;
6. the response never carries an absolute path outside the workspace, matching
   the tree endpoint's rule.

## Manual checklist

* Open `src/config.py` in the Editor tab, go to WorkBench, ask the model to edit
  that file: back in the Editor tab, the buffer shows the new content, the `↻`
  badge appeared, the cursor stayed on the same line.
* Ask for a `console` command that rewrites a file no tool frame mentioned: the
  buffer still refreshes.
* `mv` a file in the terminal pane: the tab says `deleted from disk`, the tree
  branch loses it after its refresh.
* A reply with six tool calls and one open file: **one** `fs/stat` request in
  the network panel, not six.
* Idle conversation, external editor changes a file: nothing refreshes (by
  design) — the Refresh button in the navigator and re-opening the file do.
* Open a file whose language grammar is already loaded: the refresh does not
  re-download any chunk.
