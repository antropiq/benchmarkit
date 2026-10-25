# Step 1 — Workspace navigator (tree endpoint + Vue component)

Deliver a self-contained file tree of the **active workspace**, ready to be
dropped into the Editor tab by ed-02. Built standalone (temporarily rendered at
the bottom of the Benchmark tab, or through a dev-only flag) so it can be
reviewed before the layout exists.

Depends on: nothing. Everything else in the feature depends on this.

## Why the workspace is `BENCHMARKIT_PATH_LOCATION`

The app already has one notion of "the project I am working on": the working
location both hubs point at (`config.py:71` `PATH_LOCATION_KEY`, seeded by
`setup_path_location()`, returned live by `GET /api/path-location` at
`routes.py:1041`, and already shown in the app bar via
`state.pathLocationCurrent`). The tree browses exactly that, nothing else —
so moving the workspace in the Settings dialog moves the tree, the hubs and the
terminal's `cwd` together, which is the invariant AGENTS.md states for
`BENCHMARKIT_PATH_LOCATION`.

## New package: `src/core/editorfs/`

Mirrors `mcphub/` and `lsphub/` (the symmetry rule in AGENTS.md):

| Module | Role |
|---|---|
| `constants.py` | `MAX_TREE_ENTRIES_PER_DIR`, `MAX_EDITOR_BYTES`, `IGNORED_DIR_NAMES` — module attributes, so tests patch them through the package (same convention as `lsphub.constants`) |
| `policy.py` | the one `resolve_in_workspace(rel) -> Path` + `listable(path)` / `writable(path)` gate |
| `tree.py` | one directory level → list of entry dicts |
| `files.py` | read + `stat_paths()` (ed-03, ed-09). Its write half belongs to ed-10 — this lot has none (D4) |

`policy.py` **reuses `mcp_connectors/path_policy.py`** rather than inventing a
list: import `deny_reason(resolved, for_write=…)`. That module is already the
shared authority for the two file-touching connectors, its comment says a
per-surface list is how a careful policy becomes worthless, and it is on
`PYTHONPATH` for connector children (`mcphub/process.py:17`) — for the API it
is a plain project-root import. Add `editorfs` to nothing else; no new
`restricted-*` file is needed for this lot (ed-01 … ed-09).

### The gate, in full

```
workspace = os.path.realpath(path_location())
candidate = os.path.realpath(os.path.join(workspace, rel))
reject 422 unless:
    rel has no NUL, is not absolute, and no ".." survives normalisation
    candidate == workspace or candidate.startswith(workspace + os.sep)
    deny_reason(candidate, for_write=False) is None      # reads
```

Three independent locks, because each alone is bypassable in a way the others
are not: the textual check rejects the obvious attack, the `realpath` prefix
check rejects a symlink inside the workspace pointing at `/etc`, and the
deny-list rejects `~/.ssh`-shaped names that a `workspaceRoot` of `$HOME` would
otherwise expose. The workspace root itself is additionally checked against
`lsphub.restricted_lsp_roots()` when it is set (`routes.py:1107` already does
that on `POST /path-location`), so the tree inherits that guard for free.

## Endpoint: `GET /api/fs/tree?path=<rel>`

One level per call — no recursive payload, because the workspace may be a huge
checkout and lazy loading keeps the first paint at one `listdir`.

```jsonc
// GET /api/fs/tree?path=src
{
  "path": "src",                       // normalised, "" for the root
  "workspace": "/mnt/bigfoot/dev/benchmarkit",
  "entries": [
    { "name": "core",  "path": "src/core",  "dir": true,  "children": true },
    { "name": "config.py", "path": "src/config.py", "dir": false,
      "size": 25921, "mtime": 1761371340.0 }
  ],
  "truncated": false
}
```

* `path` defaults to `""` (the workspace root).
* Directories first, then files, each alphabetical, case-insensitive.
* **Both kinds carry `mtime`** — for a directory that is the mtime of the
  directory itself, i.e. when its immediate children changed. ed-09 uses it as
  the cheap "did a file appear, disappear or get renamed here?" predicate for an
  already-expanded branch, without re-listing it. A directory's mtime does
  **not** change when a descendant is edited, which is exactly why the file
  probe of ed-09 stats open paths directly instead of inferring from the tree.
* `children` is `true` when a directory is non-empty **and** not ignored —
  computed with one `any()` per directory, never a full listing.
* Ignored by default (`editorfs.constants.IGNORED_DIR_NAMES`): `.git`,
  `node_modules`, `__pycache__`, `.venv`, `dist`, `.pytest_cache`, `.coverage`.
  Present in the constants so a user can change the policy without touching the
  component; `.git` is already denied by `path_policy.DENY_COMPONENTS`, so it
  would never appear regardless.
* Denied siblings are **omitted**, not marked: `externals.json` next to
  `.env` shows one entry, and the absence of `.env` is explained by a footer
  line ("some files are hidden by the connector policy"), not by a lock icon.
* Errors: 422 escape/deny/absolute (message names the rule, like
  `post_path_location` does), 404 missing directory, 400 asked for a file.

Wire the route inside `build_router()` next to the other workspace routes; the
function stays thin and delegates to `editorfs.tree.list_dir()`, per the
"core owns logic, API owns transport" rule.

## Component: `web/src/components/WorkspaceNavigator.vue`

Plain Vue 3 `<script setup>`, no tree library: a recursive component is ~40
lines and keeps the keyboard model under our control.

| File | Role |
|---|---|
| `WorkspaceNavigator.vue` | header (workspace path + refresh), the tree, error line |
| `WorkspaceTreeItem.vue` | one row, recursive; `expand` on click, fetch on first expand |
| `composables/useWorkspaceTree.js` | singleton: `nodes` map keyed by path, `loading`, `error`, `selected`, `load(path)`, `refresh()` |

Behaviour to hold to:

* fetch on first expand, cache per path, `refresh()` clears the cache;
* opened state lives in the composable, so a tab switch (`v-show`) keeps it;
* selection is `selected` in the composable — ed-03 consumes it, nothing else;
* keyboard: `↑/↓` move, `→`/`Enter` expand, `←` collapse, `Home`/`End` jump;
  the container is `role="tree"` with `role="treeitem"` rows and
  `aria-expanded`, matching the ARIA quality of `TabNav.vue`;
* icons inline SVG (the file's convention), no icon font;
* a visible root label: the workspace basename, with the absolute path in
  `title`, so the user always knows which checkout is browsed.

Styles go in `styles/theme.css` (the single stylesheet) under
`/* -- editor screen */`: `.editor-nav`, `.tree-row`, `.tree-row.depth-N`
indent via `padding-left: calc(var(--bi-space-2) * N)`, monospace file names,
`--pico-primary` for the selected row — no scoped `<style>` blocks, matching
the existing components.

## Tests — `tests/test_editorfs.py`

`tmp_path` workspace, patching `os.environ["BENCHMARKIT_PATH_LOCATION"]` (and
restoring it), plus `editorfs.constants` where a cap is exercised:

1. root listing: dirs first, `children` flag correct, ignored names absent;
2. `?path=../etc` → 422; `?path=a/../../b` → 422; `?path=/etc` → 422;
3. symlink inside the workspace pointing outside → 422 (the `realpath` lock);
4. `.env` and `.ssh/…` inside the workspace → 404/422 per the deny-list;
5. missing directory → 404; a file path → 400;
6. entry cap: a directory with `MAX_TREE_ENTRIES_PER_DIR + 5` files returns
   `truncated: true` with exactly the cap.

## Manual checklist (no frontend runner)

* Tree of the current workspace matches `ls -A` minus the ignored names.
* Click a folder twice: no second request (cache), collapse keeps children.
* Change the workspace in Settings → the tree re-roots on refresh.
* `F5` on a deep expansion: state resets, first paint stays fast.
* Keyboard-only navigation reaches every row; screen-reader labels read out the
  depth and the expanded state.

## Delivered

| File | Change |
|---|---|
| `src/core/editorfs/constants.py` | `MAX_TREE_ENTRIES_PER_DIR`, `MAX_EDITOR_BYTES`, `BINARY_SNIFF_BYTES`, `IGNORED_DIR_NAMES` — patchable module attributes |
| `src/core/editorfs/policy.py` | `resolve_in_workspace` (the three locks + `path_policy.deny_reason`), `listable` / `writable`, `WorkspacePathError` carrying the HTTP status; the workspace root re-checked against `lsphub.restricted_lsp_roots()` |
| `src/core/editorfs/tree.py` | `list_dir` — one level, dirs first, `mtime` on both kinds, `children` from one `any()`, denied siblings omitted, symlinked dirs never followed, exact `truncated` cap |
| `src/core/editorfs/files.py` | `stat_paths` for ed-09 (denied paths refuse rather than answer `exists: false`) |
| `src/api/routes.py` | `GET /api/fs/tree`, `GET /api/fs/stat` — status mapping only |
| `web/src/composables/useWorkspaceTree.js` | singleton: `nodes` cache, `expanded`, `focused`, `selected`, `load/refresh/expand/collapse/toggle/select`, the keyboard model, `entryFor`, `rootLabel` |
| `web/src/components/WorkspaceNavigator.vue` | header (workspace basename + refresh), root row, error line, policy footer |
| `web/src/components/WorkspaceTreeItem.vue` | recursive row: chevron/file-folder SVGs, `role="treeitem"`, `aria-level/expanded/posinset/setsize`, roving tabindex |
| `web/src/styles/theme.css` | `/* -- editor screen */` block: `.editor-nav`, `.tree-row` with `--tree-depth` indent, selection in `--pico-primary`, inline-SVG icon sizing |
| `web/src/App.vue` | dev preview only: `?editor-preview=1` mounts the navigator over the Benchmark screen (removed by ed-02) |
| `tests/test_editorfs.py` | 27 tests — listing shape, the three locks incl. the symlink and the LSP root guard, 404/400, the exact entry cap, the `stat` probe, and the read-only assertion (no non-`GET` route under `/api/fs`) |
| `AGENTS.md` | one line for `core/editorfs/` in the module table |

In practice the tree is rendered behind `?editor-preview=1`; the Editor tab
itself (and the sidebar splitter) is ed-02's job.
