# search-05 — `useWorkspaceSearch.js`: the state, the request, the retention rule

The singleton every component reads: the query, the scope, the last answer, whether the
dialog is up, and the one queued reveal. No template and no CSS in this task — search-06
renders this state, search-07 raises it, search-08 consumes the reveal.

Depends on: **search-04** (`lib/search-scope.js`, the keyed re-root registry). It does
**not** need search-03's route to exist: its tests stub `fetch`, and the response shape
is the Contract in the big picture.
Read `docs/tree-search-spec.md` §2.7 and §4's composable section; decisions D11 and D14
come from `todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `web/src/composables/useWorkspaceSearch.js` | **new** |
| `web/tests/useWorkspaceSearch.test.js` | **new** |

## What to deliver

One module-level `reactive` state + a factory returning it, the way
`useEditorFiles.js` / `useWorkspaceTree.js` do it (a singleton, not per-call state —
`workbenchGutter.test.js:29-32` documents exactly why asking a fresh call for the state
would be wrong).

```js
const state = reactive({
  query: '',
  caseSensitive: false,
  regex: false,
  scope: null,          // { kind, label, path } from scopeFor(), or null
  open: false,          // the dialog's visibility, owned here (D14)
  status: 'idle',       // 'idle' | 'running' | 'done' | 'error'
  result: null,         // the Contract payload, verbatim
  error: null,          // { status, detail } for the inline 400/422/404 line
  pendingReveal: null,  // { path, line, column } — search-08's cue
})
```

### `openFor(target)` — raises the dialog, keeps the last answer

- `scope = scopeFor(target, rootLabel)` — `rootLabel` comes from `useWorkspaceTree()`,
  and is a **function**, so call it (`rootLabel()`), which is how
  `WorkspaceNavigator.vue:38,430` already uses it.
- `open = true`.
- **Clears nothing.** Not `query`, not `result`, not `status` (§2.7 / D14): re-opening
  shows the previous query in the field, the previous scope in the label and the
  previous table. This is the whole reason the state is here rather than in the
  component.
- `scopeFor` answering `null` (no row focused) ⇒ `openFor` returns without opening. A
  command with no subject declines.

`closeDialog()` sets `open = false` and **keeps everything else**. Escape and the
backdrop both end up here; search-06 wires them.

### `run()` — one search, one fetch

`GET /api/fs/search?path=…&q=…&caseSensitive=…&regex=…` built with
`URLSearchParams` (never hand-joined: `?q=verb&caseSensitive=false` is what the route
declares, and `encodeURIComponent` on a query containing `&` or `=` is not optional).

- `scope` is `null` ⇒ return, no request.
- `query` empty after trimming ⇒ return without a request *and* without touching
  `result`: an empty needle is not a search, and the server's 400 says the same. Set
  `status` to something honest (`'idle'`) so the dialog does not look like a failed
  search. Do not invent a client-side error sentence — one rule, one owner.
- `status = 'running'` while in flight; the previous `result` **stays on screen** while
  the new one loads (search-06 renders a busy marker next to it). Clearing it would
  make a re-search flash an empty table and lose the thing the user is comparing
  against.
- **A late answer must not overwrite a newer one.** Same token trick
  `useWorkspaceTree.js` uses with `generation` (`:135-140`, `:314`): stamp a counter on
  send, drop the response when the counter moved. Two `run()` calls in a row are the
  ordinary case (the user presses Enter twice), and out-of-order arrival would show a
  table for a query that is no longer in the field.
- On `!res.ok`: `status = 'error'`, `error = { status, detail }` where `detail` is the
  server's `detail` **verbatim** (the whole point of D5 is that `re`'s message reaches
  the user). **`result` is left alone** — §6's e2e row 4: an invalid pattern shows the
  inline error and *the table still holds the previous answer*. A failed search does
  not destroy the last good one.
- On ok: `result = body` verbatim, `error = null`, `status = 'done'`. Do not reshape,
  rename keys or default missing ones in the composable — the payload is the Contract
  and the dialog reads the same names.
- A network failure (fetch throws) is the same shape with `status: 0` and a sentence
  naming that it never reached the server, so it cannot be mistaken for a refusal.

### `openHit(hit)` — what a row click does

Two things, in this order (§2.6):

1. `pendingReveal = { path: hit.path, line: hit.line, column: hit.column }`;
2. publish the path as the tree's selection, which is what makes `EditorArea`'s watcher
   open or focus the buffer.

**The trap in step 2.** §2.6 writes this as `select(entryFor(path))`, and
`select()` bails on a falsy entry (`useWorkspaceTree.js:465`) — while `entryFor(path)`
answers `null` for anything whose branch has never been listed. A workspace-wide search
almost always hits a collapsed branch, so the naive form silently opens *nothing* for
the common case. The existing answer is the tree's own keyboard chord, which handles a
path it has no listing for by synthesising the entry:

```js
// `useWorkspaceTree.js:642` and :666 do exactly this for Enter and Delete.
select(
  tree.entryFor(hit.path) ??
    { path: hit.path, name: hit.path.slice(hit.path.lastIndexOf('/') + 1), dir: false },
)
```

`dir: false` is what routes it to "select a file" rather than "toggle a folder", and
`name` is derived from the path because the tree's own `:642` leaves it `undefined`
while `select()` copies `entry.name` into `state.selected` — give it a real name. Do not
use `tree.reveal(path)` as the opener: `reveal()` publishes its selection with
`source: 'tab'` (`:533`), which `EditorArea`'s watcher **skips** as the echo of a buffer
that caused it (`EditorArea.vue:119`) — it highlights the row, it never opens a file.
The reveal happens on its own anyway: `EditorArea` watches `state.active` and calls
`tree.reveal(record.path)` (`:133-139`).

Then `closeDialog()`. Not before step 2 — the dialog closing is a consequence of the
hit being taken, not a precondition of it (D13).

### The re-root clear (D10, D14)

Register through the keyed registry search-04 built, under the key
`'workspaceSearch'` — from the factory body, like `useEditorFiles.js:755` does, because
the keyed `Map` makes a re-registration a **replacement** rather than an accumulation
(that is exactly what search-04's idempotency case pins).

- forget `query`, `caseSensitive`, `regex`, `scope`, `result`, `error`, `pendingReveal`
  and set `open = false`, `status = 'idle'`;
- **all of it, including `open`**: the dialog is a `v-show`-mounted view like the rest of
  the Editor, and a dialog left open over a root that no longer contains those paths is
  the lie §2.7 names. `useEditorFiles`'s own handler closes the tabs and their prompts
  the same way (`:1425-1434`).

### The return value

Return `{ state, openFor, closeDialog, run, openHit }`. Anything search-06/07/08 need
they read off `state`; do not add getters per field.

## Tests — `web/tests/useWorkspaceSearch.test.js`

Stub `fetch` with the `jsonResponse` helper shape from `workspaceTreeMenu.test.js:26-58`
(a resolved object with `ok`/`status`/`json()`). Drive everything through the exported
functions — no component mount in this task.

1. `openFor` on a folder row sets `scope.kind === 'folder'` and `open`, and leaves a
   previous `query`/`result` **untouched** — the retention rule, stated as the spec states
   it;
2. `openFor(null)` opens nothing;
3. **retention across close and re-open with no second fetch**: run once (stub counts
   calls), `closeDialog()`, `openFor(same target)` again — `fetch` call count is still
   **1**, and query/scope/result are all still there. This is §6's headline row;
4. `run()` sends the four params the route declares: assert the URL the stub received
   contains `path=`, `q=`, `caseSensitive=` and `regex=`, and that a needle with `&`
   arrives encoded;
5. an empty query fires **no** request and leaves `result` as it was;
6. a 400 leaves `result` intact and puts the server's `detail` string into
   `error.detail` **verbatim** (use `unterminated character set at index 3` as the
   canned message);
7. out-of-order answers: start two runs with different results, resolve the first one
   last, and assert `result` is the **second** one (the token rule);
8. `run()` while running keeps the old `result` on screen (`status === 'running'`,
   `result` non-null);
9. `openHit` publishes a selection for a path **with no loaded listing** — the trap:
   stub the tree so `entryFor` returns null, and assert `state.selected` ends up with
   that path and `dir: false`, and that `pendingReveal` carries
   `{ path, line, column }` from the hit, and that `open` is now false;
10. `openHit` for a path whose listing *is* loaded reuses the tree's own entry (assert
    the `size`/`mtime` that came from the listing survive — no synthetic entry when a
    real one exists);
11. a working-location switch (`tree.reRoot('/new/root')`) empties query, scope, result,
    error, pendingReveal **and** sets `open` false. Then a further `openFor` shows an
    empty field, not the old one.

Note for 11: `useEditorFiles.js`'s subscriber is registered when its factory runs, so
importing `useEditorFiles` in this file is the honest way to get the real registry
state; a test that only registers the search handler itself proves the search half and
nothing about the two-subscriber fix. Pair it with search-04's case 7.

## Acceptance

- `cd web && npm test` green, and `cd web && npm run build` green.
- `git diff --stat` shows exactly the two files above. No `.vue` file, no CSS, no
  `tree-menu.js`.
- The composable imports `scopeFor` from `lib/search-scope` and
  `registerReRootHandler`/`useWorkspaceTree` from `useWorkspaceTree` — and **nothing
  from `useEditorFiles.js`**: the buffer half stays reached through the tree's
  selection, exactly so the Editor's read model never imports its write half (the
  reason every other registry in that file is one-way).
- Nothing is visible in the app yet: no dialog is mounted. Expected; say so rather than
  mounting one to prove it works.
- Move this file to `todos/done/` when the above is true.
