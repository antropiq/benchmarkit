# search-04 — The scope rule (`lib/search-scope.js`) and a registry that holds two holders

Two small, pure things the rest of the frontend is built on: the rule that turns a
tree row into the scope the dialog names, and the fix that lets a second composable
subscribe to a working-location switch. No component, no fetch, no dialog yet.

Depends on: nothing (first frontend task; it does not need the backend).
Read `docs/tree-search-spec.md` §2.8, §2.7 and §4's `search-scope.js` section;
decisions D10 and D14 and *Traps* §2 come from `todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `web/src/lib/search-scope.js` | **new** — `scopeFor(target, rootLabel)` |
| `web/src/composables/useWorkspaceTree.js` | `registerReRootHandler` → a subscriber list (D10) |
| `web/tests/searchScope.test.js` | **new** |
| `web/tests/useWorkspaceSwitch.test.js` | extend with the two-subscriber case |

## 1. `lib/search-scope.js`

Kebab-case, like its neighbours in `lib/` (`editor-keys.js`, `tab-strip-wheel.js`).
Those exist for one reason, which is also this file's reason: **it is the rule two
components must not disagree about, and jsdom can test it without mounting either.**

```js
// One row of the tree -> the scope the search dialog names and asks the server for.
// `target` is `commandTarget()`'s shape ({ path, name, dir, parent }) from
// useWorkspaceTree; `rootLabel` is that composable's own `rootLabel()`.
export function scopeFor(target, rootLabel) { … }   // -> { kind, label, path }
```

The rule, in this order (the order *is* the logic — the root row answers `dir: true`,
so checking `dir` first silently turns "the whole workspace" into "a folder"):

1. `target.path === ROOT` ⇒ `{ kind: 'workspace', label: <rootLabel()>, path: '' }`.
   The root row has no relative path — it *is* `""` — so it cannot render as an empty
   label (§2.8); it renders the workspace's own name, which is what the tree's root row
   already does via `rootLabel()`.
2. `target.dir` ⇒ `{ kind: 'folder', label: target.path, path: target.path }`.
3. otherwise ⇒ `{ kind: 'file', label: target.path, path: target.path }`.

`ROOT` is exported by `useWorkspaceTree.js` — import it, do not write `''` inline, so
the two files cannot drift. *(If that import pulls the whole tree composable into a
chunk that should not have it, hoisting `ROOT` into a tiny shared module is the right
call — but check first: `useWorkspaceTree.js` already imports nothing heavy, and the
Editor's chunk is lazy as a whole.)*

Also carry `kind === 'file'` as the one source of the request's `isFile`. §4 says the
`isFile` the request needs is derived *here* rather than passed around as a second
flag — so if you expose it, expose it as `isFile: kind === 'file'` computed from
`kind`, never as an independent field the caller sets.

**`null` is a real input.** `commandTarget()` returns `null` when no row is focused or
the focused row stopped existing (`useWorkspaceTree.js:450-455`), and every other
command declines on it. `scopeFor(null, …)` returns `null`; the caller then declines.
Do not invent a scope for a row that is not there.

## 2. The registry: one slot → subscribers (D10, *Traps* §2)

Today (`useWorkspaceTree.js:149-153`):

```js
let reRootHandler = null
export function registerReRootHandler(handler) {
  reRootHandler = typeof handler === 'function' ? handler : null
}
```

and `reRoot()` calls `reRootHandler?.({ workspace: rootPath })` (`:327`).

`useEditorFiles.js:755` registers from **inside the factory body**, and
`useEditorFiles()` runs in several components' `setup` — so the slot is written again on
every mount, and it writes a **fresh arrow each time**
(`registerReRootHandler(({ workspace = '' } = {}) => forgetWorkspaceBuffers({ workspace }))`),
so it is never the *same* function object twice. Today that is harmless only because the
last writer writes the same *behaviour*. Add search as a second subscriber naively and
whichever holder mounts last quietly un-clears the other's state: the tabs would survive
a switch whose results were cleared, or the results would survive a switch whose tabs
were closed. Either half is a bug no single test file can see.

Make it a **keyed** list — one entry per holder, replaced on re-registration:

- module-level `const reRootHandlers = new Map()` — key is a stable holder name, value
  is the handler; `set` replaces, so the existing call site re-registering on every
  mount is a no-op rather than an accumulation.
- **Dedup by identity does not work here and must not be used**: the one existing
  caller passes a new arrow every time, so an `includes` check would let it append
  itself once per mount. The key is what makes registration idempotent. (Hoisting that
  arrow to a module-level function instead is *also* a fix, but it would move
  `useEditorNotices()` out of the factory into module scope and change the singleton
  init order — not worth it for this.)
- Signature: `registerReRootHandler(key, handler)` with `key` a short non-empty string
  naming the holder (`'editorBuffers'` today). `useEditorFiles.js:755` gains the key
  argument — the one existing caller, and the only place a key is invented here.
  search-05 will use `'workspaceSearch'`.
- Passing a non-function `handler` **deletes that key** — today's "pass a non-function
  to unhook" affordance, one key at a time.
- `reRoot()` calls every subscriber in registration order, and **one subscriber that
  throws must not stop the others**: wrap each call in `try/catch` and report through
  the tree's own error channel (`state.error`) rather than letting one holder's failure
  cancel another holder's cleanup. `reRoot()` already sets `state.error = null` before
  this point (the `state.error = null` assignment at `:325`), so a failure lands on a
  field that is not about to be cleared.

Keep the existing comment block above the registry honest: it explains *why* the
buffer half registers itself rather than being called from the save path. That sentence
still holds for a list — extend it with the one new reason (a second holder now exists,
so the registry is a list and registration is idempotent), and do not rewrite the
paragraph.

Do **not** register the search composable here: search-05 does that. This task only
makes the registry able to hold it.

## Tests

### `web/tests/searchScope.test.js` (new)

Import `describe/it/expect` from `'vitest'` explicitly (`globals: false`). Cases:

1. the root row (`{ path: '', dir: true, … }`) ⇒ `kind: 'workspace'` and the label is
   the `rootLabel()` value passed in — **not** `''`;
2. a folder row ⇒ `kind: 'folder'`, label is its path;
3. a file row ⇒ `kind: 'file'`, label is its path, and `isFile` is true only here;
4. **the file case is decided by `dir` being false, not by the name looking like one**:
   a row named `index.html` with `dir: true` is a folder, and a row named `LICENSE`
   with `dir: false` is a file;
5. `scopeFor(null, () => 'ws')` ⇒ `null`;
6. `scopeFor(undefined, …)` and a target whose `path` is `undefined` ⇒ `null`, without
   throwing (a command with no subject declines, it does not guess).

### `web/tests/useWorkspaceSwitch.test.js` (extend)

This file already drives `tree.reRoot(NEW_ROOT)` against the buffers. Add one case for
the registry itself:

7. **two subscribers both run** on one `reRoot`: register a probe under its own key
   alongside the buffers', re-root, and assert the probe ran **and** the tabs closed —
   the whole point of D10 is that neither one displaces the other;
8. **registration is idempotent per key**: register the *same key* three times with
   three different fresh closures (exactly the shape the existing call site has), re-root
   once, and assert only the **last** one ran, once. This is the case that an
   identity-based `includes` implementation fails, so write it before choosing;
9. a subscriber that throws leaves the other subscribers running and the buffers
   forgotten (assert the tabs still closed);
10. a key can be unhooked by re-registering it with a non-function.

## Acceptance

- `cd web && npm test` green, including `useWorkspaceSwitch.test.js`'s existing cases
  unchanged: the buffers' behaviour across a switch must be exactly what it was.
- `grep -n "reRootHandler\b" web/src/composables/useWorkspaceTree.js` finds no singular
  slot left, and `grep -n "registerReRootHandler" web/src` still shows exactly **one**
  caller — `useEditorFiles.js:755`, now passing a key. search-05 adds the second; if
  this task adds one, it has overreached.
- `useEditorFiles.js`'s diff is that one call line and (if the comment above it needs
  it) a comment. Nothing else in that file.
- No new component, no `.vue` change, no fetch, no `SearchDialog` import. `git diff
  --stat` shows exactly the four files listed above.
- The app is unchanged on screen: nothing is wired yet. Expected — say so rather than
  reaching for a template.
- Move this file to `todos/done/` when the above is true.
