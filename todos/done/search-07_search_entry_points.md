# search-07 — The two entry points: the menu row, the header icon, mounting the dialog

Make the dialog reachable. One row of menu data, one action that hands it the focused
row, one header button, and `<SearchDialog />` mounted next to the button that raises
it. From here a user can search — with the caret still landing wherever the buffer
already was, because the reveal is search-08's.

Depends on: **search-06** (the dialog exists), **search-05** (the composable's
`openFor`). Also **search-03** for the app to work end to end, but no test here needs
the route.
Read `docs/tree-search-spec.md` §2.9 and §4's edits table; *Traps* §3 comes from
`todos/search-00_big_picture.md`.

## Files

| file | change |
|---|---|
| `web/src/lib/tree-menu.js` | one `ITEMS` row |
| `web/src/components/WorkspaceNavigator.vue` | `MENU_ACTIONS.search`, the header button, `<SearchDialog />` |
| `web/tests/useWorkspaceTreeCommands.test.js` | the seven pinned id lists |
| `web/tests/workspaceTreeMenu.test.js` | the four pinned label lists |
| `web/e2e/02-editor-write.spec.js` | the four pinned menu-label lists (the suite stays green; search-09 writes the new spec) |

## 1. The menu row

Exactly §2.9's item, added to `ITEMS` — and **nothing else in that file**:
`treeMenuItems` / `inScope` already cover it.

```js
{
  id: 'search',
  label: 'Search…',
  title: 'Search inside this folder, this file, or the whole workspace',
  scope: 'row',
},
```

`scope: 'row'` is already file **and** folder **and** root, and there is no `needs`
flag because search is honest for every row — unlike `Close` (needs a buffer) or
`View diff…` (needs a committed side). The three scope cases of §1 therefore need no
new machinery. The `…` follows the table's own convention for "asks for something"
(`Rename…`, `New file…`, `View diff…`).

**Where in the order.** `ITEMS` is "the one order every row kind is built from", so the
position is a decision, not a free slot. Put it **immediately after `open`** — both
answers to "show me this thing" at the top (`open` then `preview` already sit together
for that reason), and a folder or the root, which have no `open`, gets Search as their
**first** item rather than buried under New file / New folder. Say the reasoning in a
one-line comment above the row: an order that is an accident is the thing this file's
header comment exists to prevent.

## 2. `WorkspaceNavigator.vue`

Three additions, all small:

```js
// MENU_ACTIONS, next to `open` (the same one-row-one-action shape):
search: (target) => search.openFor(target),
```

- the action receives the whole `commandTarget()` shape (`{ path, name, dir, parent }`),
  which is what `scopeFor` needs to tell folder from file from root — the same subject
  `menuFlags` already reads, and the same row the mouse and the keyboard agree on;
- `const search = useWorkspaceSearch()` beside the other composables at the top
  (`files`, `creator`, `renamer`), and the import;
- the header button, inside `.editor-nav-actions` (`:363-393`), after the collapse
  button, with that pair's exact shape — `class="icon-btn editor-nav-search"`,
  `type="button"`, an `aria-label`, a `title`, and a 14×14 stroked inline `<svg>` with
  `aria-hidden="true"`. Disabled when there is nothing to act on:
  `:disabled="!commandTarget()"` — but `commandTarget()` is a **function**, so calling it
  in a template re-evaluates per render; make it a `computed` (`canSearch`) reading
  `state.focused` and `entryFor`, on the model of `canCollapse` (`:98`). "Enabled only
  when a row is focused" is §2.9's rule and the collapse button's shape;
- `<SearchDialog />` mounted in this component's template, and the import. §2.9's reason
  for putting the dialog here rather than beside the tab strip: the scope label needs the
  navigator's own subject, and mounting it next to the button that raises it avoids one
  component reaching into another's state.

The click handler is `() => search.openFor(commandTarget())` — the identical subject the
menu actions get. `openFor` handles the `null` (declines), so no guard is needed here
beyond the disabled state.

**Do not add a keyboard chord** (D15). `lib/editor-keys.js` is not in this task's file
list for exactly that reason.

## 3. The pinned lists (Trap 3 — this is half the task)

A new `ITEMS` row makes three existing tests red, because they pin exact lists rather
than containment. Update all three **in this task**: leaving them red hands every later
task a failing suite and makes this one unverifiable.

| file | what to change |
|---|---|
| `web/tests/useWorkspaceTreeCommands.test.js:417-497` | seven `treeMenuItemIds(...).toEqual([...])` — root, folder, file, markdown file, the diffable/hasBuffer pair (two of them), and the `{ dir: false, isRoot: true }` floor case at `:495`. Every one gains `'search'` in the position the new order gives it. The floor case is the interesting one: it asserts a row that offers almost nothing still gets a menu, and Search joins it |
| `web/tests/workspaceTreeMenu.test.js:106-136` | four `menuFor(wrapper, path)` label lists gain `'Search…'` |
| `web/e2e/02-editor-write.spec.js:294-321` | four `menuItems(page, path)` label lists gain `'Search…'` — this spec is the *existing* suite; keeping it green is part of this task's acceptance even though search-09 writes the new spec |

Also **add** two rows to `workspaceTreeMenu.test.js`, which is §6's row for this file:
every row kind, **root included**, now offers Search; and the case the spec names — a
row that offers *only* search still gets a menu (the floor at `:495` of the commands
file is where the "no menu at all" rule is decided, so assert the menu opens for that
shape too: `openMenu` returns early on an empty list, and Search is what makes that list
non-empty).

Do not touch anything else in those three files: no re-formatting, no re-ordering of
their other cases, no weakening of an `toEqual` into a `toContain`. The exactness is the
point of those tests — a menu whose order is an accident is the bug they exist for.

## Tests

Beyond the list updates, add to `workspaceTreeMenu.test.js` (that file already mounts
the navigator with a stubbed `fetch`, and its `menuFor` helper reads the real buttons):

1. right-click a folder row → **Search…** → the composable's `scope.kind` is `'folder'`
   and `state.open` is true;
2. the same for a file row ⇒ `'file'`; for the root row ⇒ `'workspace'`, with the label
   from `rootLabel()`;
3. the header button: disabled when no row is focused, enabled once one is, and a click
   calls `openFor` with `commandTarget()` — assert on the composable's state, not on the
   dialog's DOM (that is search-06's already-covered ground);
4. **the dialog is mounted inside the navigator**: after a click that opens it, the
   `.editor-search-dialog` element exists in the document. (jsdom needs the
   `showModal`/`close` prototype patches from `useEditorWriteMatrix.test.js:385-390` for
   this.)

## Acceptance

- `cd web && npm test` green — **all** of it, including the three files whose lists this
  task had to update. `cd web && npm run test:e2e` green for the existing
  `02-editor-write.spec.js` (its menu-list rows now include `Search…`; if the e2e run is
  not possible on this machine because `.env.test` names an unreachable endpoint, say so
  explicitly in the final note rather than claiming it passed, and make the edit to that
  spec anyway — search-09 runs it for real).
- `cd web && npm run build` green.
- `git diff --stat` shows exactly the five files above. **Not** `SearchDialog.vue`, not
  the composable, not `EditorArea.vue`/`MonacoEditor.vue`.
- `grep -n "search" web/src/lib/tree-menu.js` shows one `ITEMS` row (plus its comment)
  and nothing else — no special-casing in `inScope` or `treeMenuItems` (that is the file's
  whole reason for existing).
- The feature is now usable end to end except for the caret: clicking a hit opens the
  file and the dialog closes, and the buffer shows wherever its cursor already was.
  That is the honest state between the two tasks — say so in the final note rather than
  reaching into `MonacoEditor.vue`.
- Move this file to `todos/done/` when the above is true.
