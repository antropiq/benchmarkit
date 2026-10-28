// ---------------------------------------------------------------------------
// The workspace tree's context menu, as data.
//
// The menu was four `v-if`s over a fixed list of buttons, which meant a fifth
// item was a fifth button and the *order* of items was an accident of the
// template. It is the tree's single command surface now — one menu for every
// row kind — so the shape of it lives here: one ordered table of items, each
// saying which rows it applies to, and the template is a `v-for` over the
// answer. Adding an item is one row of data.
//
// Two rules the table keeps, both restated where the menu opens:
//
//   * an item is **absent rather than disabled-and-lying**: `Close` only when a
//     buffer is open for that file, `View diff…` only when git has a committed
//     side to compare against, `Rename…` / `Delete` never on the root row (the
//     working location itself is neither renamable nor deletable, and there is
//     no listing above it to type a new name into);
//   * a row that cannot be mutated *and* has nothing else to offer gets no menu
//     at all — an empty list is the caller's signal not to open one, because a
//     menu of one useless item is worse than no menu.
//
// Nothing here knows whether a path is *allowed* to be written: that answer is
// the server's, so there is no deny-list and no path policy in this file — only
// the flags the caller already holds.
// ---------------------------------------------------------------------------

// The one order every row kind is built from. `scope` says which rows the item
// exists for at all; `needs` names a flag the caller must confirm.
const ITEMS = [
  {
    id: 'open',
    label: 'Open',
    title: 'Open this file in the editor',
    scope: 'file',
  },
  {
    id: 'new-file',
    label: 'New file…',
    title: 'Create an empty file in this folder',
    scope: 'dir',
  },
  {
    id: 'new-folder',
    label: 'New folder…',
    title: 'Create a folder in this folder',
    scope: 'dir',
  },
  {
    id: 'view-diff',
    label: 'View diff…',
    title: 'Show this file beside its committed version',
    scope: 'file',
    needs: 'diffable',
  },
  {
    id: 'rename',
    label: 'Rename…',
    title: 'Rename this in place — type a folder path in the name field to move it (F2)',
    scope: 'row',
    needs: 'mutable',
  },
  {
    id: 'delete',
    label: 'Delete',
    scope: 'row',
    needs: 'mutable',
    danger: true,
    title: (flags) =>
      `Move this to the trash — it stays there for ${flags.trashRetentionDays} days`,
  },
  {
    id: 'close',
    label: 'Close',
    title: 'Close this file\u2019s tab (it asks first if the buffer has unsaved changes)',
    scope: 'file',
    needs: 'hasBuffer',
  },
  {
    id: 'refresh',
    label: 'Refresh',
    title: 'Re-list this folder',
    scope: 'dir',
  },
  {
    id: 'copy-path',
    label: 'Copy relative path',
    title: 'Copy this path, relative to the working location',
    scope: 'row',
  },
]

// Which rows each scope covers. `row` is any row; `dir` is a folder *or* the
// synthetic root (both are folders a new name goes into); `file` is a file row.
function inScope(scope, { dir, isRoot }) {
  if (scope === 'row') return true
  if (scope === 'dir') return dir
  return !dir && !isRoot
}

// The items for one row, in the table's order. `flags`:
//   dir                — the row is a folder (the root row counts as one)
//   isRoot             — the row is the workspace root itself
//   diffable           — git says this file has a committed side to compare
//   hasBuffer          — a tab already holds this file
//   mutable            — the row may be renamed/deleted (false for the root)
//   trashRetentionDays — quoted in the delete item's hint
export function treeMenuItems(flags) {
  const out = []
  for (const item of ITEMS) {
    if (!inScope(item.scope, flags)) continue
    if (item.needs && !flags[item.needs]) continue
    out.push({
      id: item.id,
      label: item.label,
      danger: !!item.danger,
      title: typeof item.title === 'function' ? item.title(flags) : item.title,
    })
  }
  return out
}

// The ids a row offers — for a test, or for a caller that keys behaviour on them.
export function treeMenuItemIds(flags) {
  return treeMenuItems(flags).map((item) => item.id)
}
