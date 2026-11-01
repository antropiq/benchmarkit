# mov-05 — the `Move to…` item, and every pinned menu list it breaks

Wire the gesture up: one row in `lib/tree-menu.js`, the `moveCount` flag, the
`MENU_ACTIONS` entry that hands the dialog the subject `batchFor` decided. The second
half of this task is the part that cannot be deferred: **several existing tests assert the
menu's contents verbatim, and every one of them goes red the moment the item lands.**
Fixing them here is what makes this task acceptance-checkable; leaving them for a later
task ships a knowingly-red suite and a lot that cannot be verified.

When this task is done the feature is reachable end to end: mark rows, right-click,
`Move to…`, type, Move.

Depends on: **mov-02** (`batchFor`, `moveCount`'s source), **mov-03** (`startMove`),
**mov-04** (the dialog that must already be mounted, or the item opens nothing).
Read `docs/tree-move-batch-spec.md` §2.2 and §2.2b in full, and §7's correction about the
verbatim arrays. Decisions D2, D3, D4 and *Trap 3* come from this task.

## Files

| file | change |
|---|---|
| `web/src/lib/tree-menu.js` | one `ITEMS` row: `move-batch`, `scope: 'file'`, immediately after `rename`, no `needs`; `title` reads `flags.moveCount` |
| `web/src/components/WorkspaceNavigator.vue` | `menuFlags()` gains `moveCount`; `MENU_ACTIONS['move-batch']` |
| `web/tests/useWorkspaceTreeCommands.test.js` | the verbatim `treeMenuItemIds(...)` arrays (~417-507) gain `'move-batch'` after `'rename'`; the leanest-shape `['search', 'copy-path']` stays **unchanged** and says why |
| `web/tests/workspaceTreeMenu.test.js` | the `menuFor(...)` **label** arrays (~131-165) gain `Move to…` after `Rename…`; its `['search', 'copy-path']` assertion stays unchanged |
| `web/e2e/02-editor-write.spec.js` | the two arrays in `each row offers exactly what it can honour` — the plain file row and the markdown row — gain `'Move to…'` |

Five files, three of them test fixes. Do not add any new e2e *flow* here — that is
mov-06; this task only keeps the existing spec honest about the new item.

## What to deliver

### The item (§2.2)

```js
{
  id: 'move-batch',
  label: 'Move to…',
  scope: 'file',
  title: (flags) =>
    flags.moveCount > 1
      ? `Move these ${flags.moveCount} files into another folder`
      : 'Move this file into another folder',
}
```

placed **directly after `rename`** — same mutation family — and `Delete` stays the only
`danger` row. A one-line comment in that table's voice for why there is no `needs` flag
(D2: `scope: 'file'` is exactly the set of rows the item can honour; a flag would answer
"is this writable", which this file's header says is never its business). Also note in the
comment why `Rename…` and `Move to…` coexist: one asks for the full target path in a field
labelled as a name, the other asks for a destination **folder** and lists every path it is
about to move.

### The flag

`menuFlags(path, isDir)` gains `moveCount` = the size of the batch `batchFor(path)` would
answer for this row — so `1` for an unmarked file, the mark count when the row sits in a
legal same-parent batch, and `1` when the marks cross parents. It is a **count of rows**,
never a claim about writability, and it feeds hint text only (D3): a row never gains or
loses the item because of it.

### The action

`MENU_ACTIONS['move-batch'] = (target) => move.startMove(tree.batchFor(target.path))`.
The subject is resolved **once, here** (D4) — do not read `commandTarget()` or
`state.marked` inside the move composable at submit time.

### The pinned lists (Trap 3)

The full inventory, so nothing is missed and nothing is over-edited:

| file | which array | change |
|---|---|---|
| `web/e2e/02-editor-write.spec.js` | the **plain file** row, and the **markdown file** row | gain `'Move to…'` after `'Rename…'` |
| `web/e2e/02-editor-write.spec.js` | the **root** row, and the **scratch folder** row | **unchanged** — `scope: 'file'` excludes them |
| `web/tests/useWorkspaceTreeCommands.test.js` | every `treeMenuItemIds` call handed a **file** shape (plain, markdown, `diffable+hasBuffer`, and that pair with markdown) | gain `'move-batch'` after `'rename'` |
| `web/tests/useWorkspaceTreeCommands.test.js` | the `root` and `folder` arrays, and the two "these flags are never read on a folder/root" comparisons | **unchanged** |
| `web/tests/useWorkspaceTreeCommands.test.js` | the leanest-shape floor `['search', 'copy-path']` | **unchanged** — a file-only item cannot raise the floor of a root row. Keep the existing comment and add half a sentence that `Move to…` is deliberately absent from it |
| `web/tests/workspaceTreeMenu.test.js` | the two **file-row label** arrays (`notes.md`, `src/a.py`) | gain `'Move to…'` after `'Rename…'` |
| `web/tests/workspaceTreeMenu.test.js` | the root and `src` label arrays, and its own `['search', 'copy-path']` | **unchanged** |

Also assert the item's **hint** in `workspaceTreeMenu.test.js` the way that file asserts
the Search hint, for both `moveCount` branches — that is the only place the singular/plural
copy is pinned.

## Rules this task must not break

- The menu table's two standing rules: an item is absent rather than disabled-and-lying,
  and **no deny-list, path policy or writability answer** in `tree-menu.js` (its header).
  `moveCount` is a count; the only flags that gate presence stay `scope` + `needs`.
- An item is a new **row of data**, never a new `v-if` in the template.
- `Delete` remains the only `danger` row; `Move to…` is not one (it refuses rather than
  destroy, and the destination is chosen in a modal that lists the paths first).
- No keyboard chord for Move (D16). If a chord feels missing next to `F2`, that is the
  spec's decision, not an oversight.
- Do not touch the mark model, the dialog, or the composable — if the item cannot be wired
  without changing one of them, an earlier task is wrong; fix that task's code rather than
  working around it here.
- The e2e spec file's other tests stay untouched and green.

## Done when

```bash
cd web && npm test              # the whole Vitest suite green, with the new item in the arrays
cd web && npm run build
cd web && npm run test:e2e -g "each row offers exactly what it can honour"
                                # the one existing e2e test this task touches
```

Then, by hand or in a follow-up assertion: mark two files of one folder, right-click one,
see `Move to…`, click it, and the dialog lists both paths with the field holding their
parent. Right-click a third, unmarked file and the same item names one path.

## Out of scope here

new e2e flows (mov-06); a submenu, an icon, or a shortcut hint in the label; gating the
item on marks (it must answer a plain right-click on one file — that is the N=1 case, not
a second feature); a `Move to…` on a **folder** row (R4); batch drag-and-drop; a
`Copy to…` sibling.
