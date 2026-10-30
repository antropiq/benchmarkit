# Search inside the workspace tree — big picture

Give the Editor tab one new question — *"where is this string?"* — answered by a
flat table of matches the user can click to land on the line in a buffer. Right-click
a row (or press the header icon) → a `<dialog>` with a query, case and regex → rows of
`{path, line, column, snippet, matchLength}` → click one, the dialog closes, the
buffer opens with the caret on the match.

The full design (the need, the rejected alternatives, why not a tab and why not `rg`)
is **`docs/tree-search-spec.md`** — 570 lines, English, and it is the authority. This
file is the map, the decision list and the cross-task contract. Read the spec's §2.4,
§2.5 and §4 before touching either side. Nothing here restates it except what a task
cannot re-derive, or must agree with another task about.

## What is new

| piece | role |
|---|---|
| `src/core/editorfs/search.py` | core-only scan: plain values in, dict out. No FastAPI. |
| `GET /api/fs/search` | one blocking request, one JSON payload. No SSE, no worker thread. |
| `web/src/lib/search-scope.js` | one pure rule: a tree row → the scope the dialog names |
| `web/src/composables/useWorkspaceSearch.js` | the singleton: query, scope, state, result, pending reveal |
| `web/src/components/SearchDialog.vue` | `<dialog>` + `showModal()`: label, field, two checkboxes, table, footer |

Edits: `editorfs/constants.py` (the caps), `editorfs/__init__.py` (the export),
`lib/tree-menu.js` (one `ITEMS` row), `useWorkspaceTree.js` (the re-root registry),
`WorkspaceNavigator.vue`, `EditorArea.vue`, `MonacoEditor.vue`,
`lib/monaco-surface.js` (the reveal clamp) and `styles/theme.css` (the dialog card).

## The three scopes come from the row the gesture was made on

| row | searched | scope label |
|---|---|---|
| a folder (`docs/old`) | that subtree | `docs/old` (folder) |
| the root row (`path === ''`) | every accessible file | the workspace's own name (whole workspace) |
| a file (`src/config.py`) | that file's content | `src/config.py` (file) |

No new menu machinery: `scope: 'row'` in the `ITEMS` table already means file, folder
**and** root, and search needs no `needs` flag because it is honest for every row
(unlike `Close`, which needs a buffer). The root row answers `dir: true`, so the
`path === ROOT` check comes **before** the `dir` check in `scopeFor` — otherwise the
whole-workspace case reads as a folder.

## Decisions

| # | decision | why |
|---|---|---|
| D1 | one blocking `GET`: no stream, no `EventBus`, no cancel | §2.2. The scan is bounded by D3's caps and finishes in well under a second. The price is a spinner with no partial results, and the footer is what makes that honest rather than silent |
| D2 | **a search's visible set is the tree's visible set.** One verdict per path — `policy_modes.mode_for` — and no second answer anywhere | §2.5. A tree row is a *name*; a search result is **the bytes from inside the file**. Returning a snippet from a hidden file would put a secret in a payload and then in a DOM node, which makes hiding the row worthless. So `hidden` ⇒ pruned (a directory costs no `scandir` below it) and never counted; `ro` ⇒ searched, because `ro` constrains what changes, not what is seen, and the tree promised that row |
| D3 | the four caps live in `editorfs/constants.py`, read **qualified at call time** (`constants.MAX_SEARCH_MATCHES`) | §2.4. Same reason as every other Editor cap: a test patches the *name*, so `from .constants import X` freezes a copy the patch cannot reach. Mirror how `tree.py:131` reads `MAX_TREE_ENTRIES_PER_DIR` |
| D4 | over `MAX_EDITOR_BYTES` ⇒ skipped **and counted**, never a 413; binary ⇒ skipped and counted; **skipped-by-policy ⇒ skipped and not counted at all** | §2.4 + §2.5. A scan has no right to fail the whole search because one artefact is large; and a count of policy-skips is an *observation* of a hidden name, which pol-00 D8 forbids. So there is no `skipped.policy` key, deliberately |
| D5 | an invalid pattern is `SearchQueryError(400, str(exc))`, raised **before any walk**, carrying `re`'s message verbatim | §2.3, §4. The user wrote that pattern and needs `re`'s caret position back, not a generic error — and a bad pattern must cost no scan |
| D6 | the walk's order is `tree.py`'s order: directories first, then files, each sorted by `(name.lower(), name)` | §4 step 2. A truncated result must be the *same* first N rows every run, not whichever inode answered first; the exact-name tie-break is what makes that true for two names differing only in case. Reuse `tree._name_key` rather than re-deriving a key |
| D7 | the scan **never opens a symlink** and never descends one | *Traps* §1. `tree.py` lists a symlinked directory as a leaf and never follows it — but it lists a symlinked *file* as an ordinary file, and opening it reads bytes from **outside** the working location, silently bypassing the `realpath` lock `files.read_file` respects. Skipping it (not counting it) is the honest answer, and it is what `core/batch.py` does with a symlinked lot file |
| D8 | the payload carries **one server-written footer sentence** (`footerNote`) and the frontend renders it verbatim | The caps are backend numbers living in `constants`. If the dialog composed the sentence it would have to know them, and the two would drift |
| D9 | `truncated: bool` per §4 (any **cap** cut), independent of the footer, which also appears for counted skips with no cap cut | Three events need honesty: a walk that stopped early, a row list that was cut, and files that were too big or not text. Only the first two are `truncated` |
| D10 | `registerReRootHandler` gains **subscribers**, not a second single slot | search-04. It has one slot today, and §2.7 wants the search composable to join it — see *Traps* §2 |
| D11 | the reveal is **queued**, never applied synchronously from the row click | §2.6. `EditorArea` keeps *one* `MonacoEditor` instance across tab switches (`:content="tab.text"`, no `:key`), and that component's `props.content` watcher calls `model.setValue`, whose own comment says it clears the cursor. A reveal applied before that watcher ran is erased |
| D12 | snippet text is bound as a **string** — never `v-html`, never through `lib/markdown.js`; the matched span is a separate `<span>` around a slice of it | §4. It is a line of someone's file, not rendered HTML. The project rule is that rendered HTML goes through `markdown.js`; a snippet is not rendered HTML and gets no markup at all |
| D13 | the dialog closes on the row click | §2.6. A `showModal()` dialog covers the pane, so a result the user cannot yet look at is a result they cannot use |
| D14 | state lives in the composable singleton; results survive a close, a re-open and a tab switch; **nothing is re-fetched on re-open** | §2.7, and the project rule *"run/stream state lives in the composables"*. A stale hit is allowed on purpose: clicking it produces the pane's **existing** refused-open answer (`openFile` echoes the server's reason with the path), so no staleness machinery is needed |
| D15 | no keyboard chord | §2.9. `lib/editor-keys.js` owns exactly the chords the app claims; Ctrl+F is the browser's own find and taking it would be a chord the user cannot turn off |
| D16 | a `GET`, so **not** a write: no `ALLOWED_HOMES` entry, and `search.py`'s prose never spells a write marker | §4. The marker scan in `tests/test_write_surface_isolation.py` walks all of `src/core` for the three write markers; `GET /api/fs` is deliberately not one of them |

## The payload contract (four tasks must agree on exactly this)

`GET /api/fs/search?path=docs/old&q=verb&caseSensitive=false&regex=false`

```json
{
  "scope": "docs/old",
  "isFile": false,
  "matches": [
    { "path": "docs/old/notes.md", "line": 12, "column": 8,
      "snippet": "the verb of the sentence", "matchLength": 4 }
  ],
  "matchCount": 517,
  "filesScanned": 61,
  "truncated": true,
  "skipped": { "oversized": 3, "binary": 4 },
  "footerNote": "showing the first 500 matches — refine the query"
}
```

| key | rule |
|---|---|
| `scope` | the **normalised** workspace-relative path of the scope root (`""` for the whole workspace) |
| `isFile` | the scope root is a file. Derived server-side from the resolved root, so the client never sends a kind |
| `matches[].path` | workspace-relative, verbatim — the same string the tree shows and the same string `openFile(path)` takes |
| `matches[].line` / `.column` | **1-based**. `column` is the **start of the match**, not the start of the line, so the caret lands on what was searched for |
| `matches[].snippet` | the whole line, terminator stripped, `…`-appended past `MAX_SEARCH_SNIPPET_BYTES`. Multiple matches on one line are separate rows |
| `matches[].matchLength` | the match width; `0` is legal for a zero-width regex match |
| `matchCount` | **all** matches found, including ones not sent as rows |
| `filesScanned` | files the scan actually opened |
| `truncated` | `true` when any **cap** cut the scan or the row list (D9) |
| `skipped` | exactly those two keys, no policy key (D2/D4) |
| `footerNote` | one sentence, `""` when there is nothing honest to say (D8) |

Query params are camelCase like the rest of `/api/fs` (`ifUnmodifiedSince`). Statuses:
**400** invalid pattern (carrying `re`'s message), **422** the scope path escapes the
workspace or is refused by a lock, **404** the scope path is absent **or hidden**. All
three arrive from the core as `SearchQueryError` / `WorkspacePathError`; the route only
re-raises them.

`footerNote` in fixed order when several apply (D8/D9) — walk caps first, because they
bound *completeness*, then the row cap, which bounds only *what is listed*:
`searched the first 5000 files` · `read the first 32 MB` · `showing the first 500
matches`, then `3 files too large to search` / `4 binary files skipped`; ends
`— refine the query` when a cap cut. **search-02 writes that sentence once**; search-06
renders it verbatim.

## Where the scan is cut in two (the seam between search-01 and search-02)

One module, two tasks, split where *which bytes reach the matcher* stops being the
question and *what a match is* begins:

- **search-01 — the byte supply**: resolve the scope root, the walk, one policy verdict
  per child, the prune, D7's symlink rule, the size/binary/budget reader. Yields
  `(child_rel, text)` pairs **lazily** and accumulates the counts (`filesScanned`,
  `skipped.oversized`, `skipped.binary`, which caps cut). Lazy on purpose: D3's byte
  budget can only be honoured while matching consumes it, and an eager list of 5000
  decoded files is gigabytes.
- **search-02 — the matcher and the payload**: compile once, D5's error-before-any-walk,
  line-by-line matching, the columns, the snippet, the payload dict, `search_workspace`
  and its export through `editorfs/__init__.py`.

Neither half is a stub: 01's tests assert the exact sequence of paths and counts, 02's
assert the rows and the payload.

## Traps — found by reading the code, not stated in the spec

1. **A symlinked *file* inside the workspace reads bytes from outside it.**
   `tree.py:112` decides dir-ness with `is_dir(follow_symlinks=False)`, so a
   symlink-to-a-directory is listed as a leaf and never descended — but a symlink to a
   file outside the root is listed as an ordinary file, and a walk that opens
   `os.path.join(parent_abs, name)` follows it. `files.read_file` is safe only because
   it re-resolves through `resolve_in_workspace`, whose `realpath` prefix lock 422s the
   escape (`test_editorfs.py:553`). Re-resolving **every child** through that gate
   would be honest but costs a deny-list file read per file
   (`policy._check_workspace_root()` → `lsphub.discovery.restricted_lsp_roots()`), so
   the walk opens only non-symlink regular files (D7). The row still shows in the tree
   and clicking it still 422s through the read route — unchanged behaviour.
2. **`registerReRootHandler` is one slot, and it is re-registered on every mount.**
   `useWorkspaceTree.js:149-153` holds a single handler, registered from *inside*
   `useEditorFiles()`'s factory body (`:755`), which runs in several components'
   `setup`. Today that is harmless — the last writer wins with the same function. Add a
   second subscriber naively and whichever mounts last silently un-clears the other's
   state. search-04 makes it a list **and** makes registration idempotent, because a
   remount re-runs `setup` and must not accumulate handlers.
3. **Three existing tests pin exact menu lists and go red the moment the menu row
   lands**: `web/tests/useWorkspaceTreeCommands.test.js` (seven
   `treeMenuItemIds(...).toEqual([...])` around lines 417-497),
   `web/tests/workspaceTreeMenu.test.js:106-136`, and
   `web/e2e/02-editor-write.spec.js:294-321`. **search-07 updates all three in the same
   task** — otherwise it cannot be acceptance-checked and every later task inherits a
   red suite.
4. **`tests/test_editorfs.py:677` pins the whole `/api/fs` verb map** as an exact dict
   (`TestSurfaceShape`), whose docstring says a new route "has to appear here on
   purpose". search-03 adds `"/api/fs/search": ["get"]` as part of its own acceptance —
   that allowlist *is* the review of "this route is a read".
5. **jsdom has no `showModal`/`close` on `<dialog>`, and the suite has no globals.**
   `useEditorWriteMatrix.test.js:385-390` patches `Element.prototype.showModal` and
   `.close` to set `el.open` — search-06's test needs the same two stubs.
   `web/vitest.config.js` sets `globals: false`, so every test file imports
   `describe/it/expect/beforeEach` from `'vitest'` explicitly.
6. **Matching is line-by-line, so a needle containing a newline never matches** and a
   multi-line regex is out of scope (§4 step 5). Use `pattern.finditer(line)`, never a
   manual `while m := search()` loop: a zero-width pattern (`x*`) makes the manual loop
   spin forever. Split decoded text with `splitlines()`, so a CRLF file's columns agree
   with the model Monaco builds from the same text.

## Not in this lot (§5 — do not add these)

file-name search, globs, whole-word, filters or ignore patterns (the policy file is the
only exclusion mechanism this surface has, deliberately), **any** replace affordance,
grouping by file, streaming/partial results or cancel, persistence across reload,
keyboard chords, a history dropdown, search in the terminal or WorkBench.

## Project rules that bind these tasks (from `AGENTS.md`)

- *Core owns logic, API owns transport*: `search.py` is core-only like the rest of
  `editorfs/` — it takes plain values and returns a dict, and touches no FastAPI.
- *Every path the app holds is relative to the working location*, and *path rules come
  from two levels, one verdict*: search asks `mode_for`, never the connector deny-list
  (that is `core/batch.py`'s surface, and merging them is explicitly rejected).
- *A `/api/fs` write route is reachable from the SPA only* — this is a read, and D16
  keeps it that way.
- *Frontend invariants*: rendered HTML always goes through `lib/markdown.js`, never raw
  `v-html` (hence D12: a snippet is not HTML at all); *run/stream state lives in the
  composables*.
- *Python style*: explicit type annotations, `typing` imports, no bare containers.
- *Tests stay hermetic*: `BENCHMARKIT_DISABLE_MCP=1` / `_LSP=1` (set by
  `tests/conftest.py`); the workspace in a backend test is a throw-away `tmp_path`
  pointed at by `BENCHMARKIT_PATH_LOCATION`, never this checkout.
- No lint step. `pytest` for Python; `cd web && npm test` for Vitest;
  `cd web && npm run test:e2e` for Playwright (its own server, its own port, endpoint
  from `.env.test` — never the developer's `.env`); `cd web && npm run build` for
  `web/dist`, which must exist for FastAPI to serve the UI.

## Task order

| # | task | depends on |
|---|---|---|
| search-01 | the scan's byte supply: walk, verdicts, caps, reader — in `editorfs/search.py` + `constants.py` | — |
| search-02 | the matcher, the snippet, the payload dict, `search_workspace` and its export | search-01 |
| search-03 | `GET /api/fs/search` in `api/routes.py`, plus the `/api/fs` verb-map allowlist | search-02 |
| search-04 | `lib/search-scope.js`, and `registerReRootHandler` → subscribers | — |
| search-05 | `composables/useWorkspaceSearch.js`: state, the request, the retention rule, the re-root clear | search-04 |
| search-06 | `components/SearchDialog.vue` + its `theme.css` block | search-05 |
| search-07 | the two entry points: the `ITEMS` row, `MENU_ACTIONS`, the header icon, mounting the dialog — and the three pinned menu lists (Trap 3) | search-06 |
| search-08 | the reveal: `defineExpose({ revealPosition })`, `pendingReveal`, `EditorArea`'s watcher | search-05 |
| search-09 | the e2e sweep of the five §6 flows, and the one AGENTS.md clause of §7 | all of the above |

search-01 → 03 are backend; 04 → 08 are frontend; 09 is what only a real browser can
see. The frontend half needs no live route (its tests stub `fetch`), so the
backend-before-frontend order is carried by the numbering, not by a second lot.
