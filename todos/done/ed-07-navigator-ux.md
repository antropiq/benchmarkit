# ed-07 — Navigator UX: making the three operations feel like one feature

Create, delete and rename now work; this task makes the tree behave like the one
place they are done from — one menu for every row kind, one keyboard model, one
way to select a folder, and no stale listing after a mutation. It adds **no new
route**.

Depends on: ed-04, ed-05, ed-06.

## 1. One context menu, three row kinds

Today the menu has four items max and knows only files. It becomes the tree's
single command surface, built from one small table instead of four `v-if`s so a
later item is one row of data:

| Row | Items (in this order) |
|---|---|
| root | New file… · New folder… · Refresh · Copy relative path (the workspace path) |
| folder | New file… · New folder… · Rename… · Delete · Refresh · Copy relative path |
| file | Open · View diff… (existing rule) · Rename… · Delete · Close (existing rule) · Copy relative path |

Keep the existing rule that an item is **absent rather than disabled-and-lying**
(the comment at `WorkspaceNavigator.vue:137-147` states it). So: no `Close`
unless a buffer is open; no `View diff…` unless `git.isDiffable`; no `Rename…` /
`Delete` when the row is the workspace root.

Refuse the menu entirely for a row that cannot be mutated **and** has nothing else
to offer — a menu with one useless item is worse than no menu.

Update the block comment that still says folder rows do not get a menu.

## 2. Selection semantics: a folder can be *selected*

`useWorkspaceTree.select()` currently only toggles a folder (`:217-231`), so
"rename this folder" has no subject other than the roving focus. Keep the click
behaviour (a folder click toggles — that is muscle memory) but make the **focused
row** the subject of every tree command, which is what `handleKey` already does:

* `state.focused` is the subject. Add a `commandTarget()` helper returning
  `{ path, name, dir }` for it (using `entryFor`), and have the keyboard chords and
  the menu both go through it;
* the menu sets focus to the row it was opened on before doing anything, so the
  keyboard and the mouse agree about the subject immediately after a right-click;
* `selected` (the file-selection channel that `EditorArea` watches) stays exactly
  as it is — it means "open this file", not "this row is the subject". Conflating
  them would make every folder click try to open a buffer.

## 3. Keyboard model, finished

Into `useWorkspaceTree.handleKey` (one chain, existing keys untouched):

| Key | Action |
|---|---|
| `n` / `N` | New file / New folder (ed-04) |
| `F2` | Rename the focused row (ed-06) |
| `Delete` | Delete the focused row (ed-05) |
| `r` | Refresh the focused folder (or the file's parent) |
| `Enter` | keep: open/select (unchanged) |

Guards the chain needs and must state: a key is ignored while `creating` is
active (the inline input owns the keyboard then — a stray `n` typing into itself
is the bug), while a dialog is open, or when the tree is not the focused surface.
`Delete` on the root row does nothing at all.

Single-letter chords in a tree are only safe because the tree container takes
focus as a `role="tree"` widget; say that in the comment, and confirm `←`/`→`/
`↑`/`↓`/`Home`/`End`/`Enter`/`Space` keep their current meaning.

## 4. Listings that tell the truth after a mutation

Every mutation already calls `refreshBranch(parent)` (ed-04/05/06). Three gaps
left:

* **the moved row's subtree.** After a rename of a *folder*, every cache entry
  keyed under the old prefix (`nodes[old]`, `nodes[old/…]`, `expanded[old]`) is
  dead. Drop them (`useWorkspaceTree.dropBranch(path)` — delete the matching keys,
  do not re-list), then re-expand at the new path if the user had it open.
* **an ancestor's `children` flag.** Creating the first file in an empty folder
  leaves `children: false` on its row until that parent is re-listed. Accept it
  (the arrow is drawn from the listing) but re-list the **parent of the parent**
  when the flag would otherwise lie about a folder the user is staring at:
  `refreshBranch(parentOf(parentOf(newPath)))` when that folder's listing is
  cached. Cheap, and it keeps the rule "`children: false` never promises an empty
  branch that then lists something" (`tree.py:44`).
* **a mutation that moves a row off-screen** (rename sorts it elsewhere) —
  `reveal()` already re-lists a stale listing once, so the follow-up `reveal` of
  the renamed path fixes the row. Assert that, do not add a second mechanism.

## 5. The name input, generalised

ed-04's create input and ed-06's rename input are the same widget: one row,
prefilled, `Enter`/`Escape`, validated client-side then refused server-side. Pull
it into `WorkspaceTreeItem.vue` (or a tiny `TreeNameInput.vue` if the item grows
past readability) driven by `useWorkspaceTree.editing`:

```
editing: { mode: 'create' | 'rename', parent, path, isDir, value, error }
```

* prefill for rename = the current name, with the **extension not selected**
  (selecting the basename and leaving `.py` unselected is the convention that
  avoids accidentally renaming the type; do it with `setSelectionRange`);
* `error` renders inside the row (`.tree-row-error`), not in a toast — the fix is
  to type again;
* a second `Enter` while a request is in flight does nothing (the same
  re-entrancy rule as `saveFile`'s `saving` flag).

## 6. Tests

Vitest (`web/tests/useWorkspaceTreeCommands.test.js`, plus the existing suites):

1. `commandTarget()` follows `focused` after a menu open, for a file and a folder;
2. the keyboard table: `n`, `N`, `F2`, `Delete`, `r` reach the right handler, and
   the arrow/`Enter` model is unchanged;
3. a key is ignored while `creating`/`editing` is active;
4. `dropBranch('src')` removes `nodes['src']`, `nodes['src/core']` and
   `expanded['src']` but leaves `nodes['']` and `expanded['']` intact;
5. the menu item table per row kind (root / folder / file), including the
   absent-rather-than-disabled cases;
6. the create and rename input share the editor state and validate the same way.

## 7. Acceptance

* Right-click anything: the menu offers exactly the items that apply.
* Focus a folder, press `F2`, rename it, the branch re-appears at the new name
  with its children still expanded.
* Focus a file, press `Delete`, the confirm dialog appears; `Escape` cancels.
* No listing on screen shows a row that is not on disk, after any of the
  operations.
* `cd web && npm test`, `pytest` green.
