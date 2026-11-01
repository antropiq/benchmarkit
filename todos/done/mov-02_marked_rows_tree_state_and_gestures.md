# mov-02 — marked rows: the tree's new state and its gestures

Give the tree a **batch**, as state and as gestures: `state.marked` /
`state.markAnchor`, the `Ctrl`/`Cmd`+click and `Shift`+click/arrow chords that fill
them, the row style and `aria-selected` that make them visible, the footer line that
says how many are marked, and `batchFor(path)` — the one place that decides what a
`Move to…` opened on a row would cover. Ship it with its unit tests.

Nothing moves in this task. It ends with a selection the user can make, see and lose —
and with the subject rule tested where it lives, so mov-03 and mov-05 can build on a
settled answer.

Depends on: **nothing** (independent of mov-01; either may run first).
Read `docs/tree-move-batch-spec.md` §2.1, §2.2b and §5.3 in full, and §4's row for
`useWorkspaceTree.js`. Decisions D1, D4, D5, D15, D16 and *Traps* 6, 7, 8 come from
this task.

## Files

| file | change |
|---|---|
| `web/src/composables/useWorkspaceTree.js` | `state.marked`, `state.markAnchor`; `toggleMark`, `rangeMark`, `clearMarks`, `batchFor`; `select(entry, event)`; the `Shift` chords; the `dropBranch`/`reRoot` clears |
| `web/src/components/WorkspaceTreeItem.vue` | `tree-row--marked` from `state.marked.includes(path)`; `@click` passes the event through; `aria-selected` reports the batch |
| `web/src/components/WorkspaceNavigator.vue` | the "N selected" line in the footer while a batch is live |
| `web/src/styles/theme.css` | `.tree-row--marked` — a third row tone, distinct from `--focused` and `--selected`, in **both** themes |
| `web/tests/useWorkspaceTreeCommands.test.js` | the marks: toggle, range, plain-click clear, `batchFor` across parents, the `dropBranch`/`reRoot` clears, `Shift`+arrow |

Five files. The menu table (`lib/tree-menu.js`) is **not** touched here — that is
mov-05, and this task must leave every existing menu assertion green.

## What to deliver

### The state (D1)

```js
marked: [],        // ordered paths of the rows in the batch — files only, by construction
markAnchor: null,  // the path a Shift-range grows from
```

They live in the composable like every other tree fact: the navigator sits in a
`v-show` view, so a selection survives a tab switch, and mov-03's write composable must
be able to read it without going through a component. **Never** reuse `state.selected`
or `state.focused` for this — `selected` carries `size`/`mtime`/`source` for the pane
and answers a folder click by toggling; a batch would inherit all of it (D1).

### The gestures

- **Plain click** (no modifier): replaces the marks with nothing, and focus/`selected`
  behave exactly as today. A plain click is a fresh look at one row, not a way to
  extend anything.
- **`Ctrl`/`Cmd`+click**: toggles one path in/out of `marked`, and `select()` still runs
  so the row also opens — the row under the cursor is both the row you look at and the
  row you add. A **folder row is refused entry** (R4) and keeps toggling exactly as
  today.
- **`Shift`+click** and **`Shift`+ArrowUp/Down/Home/End**: mark every *file* row in the
  visual span between `markAnchor` and the new focus, taken from `visiblePaths()` — the
  same list arrow navigation already walks, so a range can never disagree with what is
  rendered (*Trap 8*'s sibling: the row's `@click` currently calls `select(entry)` with
  no event; it must pass the event through, or none of this reaches the composable).
- A range **may cross parents** (D5). That is not softened, and the narrowing happens
  later in `batchFor`, not here.
- `Ctrl`/`Cmd` stays **out** of the `handleKey` chord chain (D16, *Trap 6*): the existing
  early return on `ctrlKey || metaKey || altKey` stays. `Shift` joins only the
  Arrow/Home/End chords — there is no chord key for Move, and do not invent one.
- Keyboard `Shift`+arrows move the focus as today **and** extend the marks; the anchor
  is the row the last non-Shift gesture landed on.

### `batchFor(path)` — the subject rule (§2.2b), decided once, when the menu opens

```
marks = state.marked                       // files only, by construction
if path in marks and marks.length >= 2 and every mark shares parentOf(path)
     -> { paths: [...marks], parent: parentOf(path) }
otherwise
     -> { paths: [path], parent: parentOf(path) }   // the N=1 case
```

Returns `{ paths, parent }`. It reads nothing outside the tree's own state, so the
navigator can call it the moment a menu opens and mov-03 can copy the result into its
prompt immediately. The cases to test are exactly the spec's table: no marks → 1;
two same-parent marks, menu on a marked row → both; menu on an **unmarked** file while a
batch is live → that one (the marks stay highlighted); two marks **across parents**,
menu on a marked row → that one.

### Where the marks die (*Trap 7*)

- `dropBranch(path)`: forget every mark at or under `path`, next to the existing
  `focused`/`selected` lines (~`:602`).
- `reRoot()`: clear all marks and the anchor, next to `focused`, `selected`, `editing`
  (~`:332`) — those paths mean nothing in the new root, which is the argument `reRoot`
  already makes for `selected`.
- `clearMarks()` is exported for mov-03's success path (one call after the loop).

### What the user sees

- `.tree-row--marked` on a marked row. It must read as a **third** tone: a marked row
  that is also focused, or also the pane's file, stays distinguishable from either alone.
  Both themes (`styles/theme.css`, beside the existing `.tree-row--focused` /
  `.tree-row--selected` block).
- `aria-selected` follows `marked` when there are marks, otherwise the single pane file
  as today (D15). This is the feature's one accessibility change and is required by the
  multi-selectable `role="tree"` pattern; do not "fix" it back.
- A footer line while a batch is live — `3 selected` in the navigator's existing
  `.editor-nav-footer` caption area — because a row style in a scrolling tree is
  invisible the moment the batch is long. It disappears with the marks; it never shows
  `0 selected`.

## Rules this task must not break

- *Run/stream state lives in the composables* — the marks are composable state, and the
  component reads them, never owns them.
- `marked` is **not** the code pane's input: `state.selected` stays the single file the
  pane follows, and marking never opens or closes a buffer or a tab (§5.3).
- A plain click, a folder click, arrow navigation, `F2`, `Delete`, `n`, `r`, `Enter`
  all behave exactly as before. The only keyboard change is `Shift` on the three
  navigation chords.
- The menu table's two rules survive untouched: an item is absent rather than
  disabled-and-lying, and no file that decides the menu asks whether a path is
  *allowed* — `moveCount` is a count of rows, never a writability claim (that arrives in
  mov-05 as hint text only).
- No `v-html`, no raw HTML anywhere (the footer line is text).
- The tree's lazy-listing contract is unchanged: no new listing request is issued by
  marking.

## Done when

```bash
cd web && npm test          # the whole Vitest suite green, including every existing
                            # menu-list assertion (mov-05 owns changing those)
cd web && npm run build     # no template/style error
```

And the new assertions in `useWorkspaceTreeCommands.test.js` cover: toggle on/off, a
folder row refusing entry (asserted explicitly, so a future gesture that marks folders
fails here rather than shipping a batch that moves a tree), `Shift`+click and
`Shift`+ArrowDown producing the same span as the rendered rows, plain click clearing,
`batchFor` in all four cases above, `dropBranch` forgetting a mark under the removed
path, `reRoot` clearing all.

## Out of scope here

any `Move to…` UI (no menu item, no dialog, no composable — mov-03/04/05); moving
folders, or marking a folder row; checkboxes or any column in the row; a drag handle;
a "Select all" affordance; a chord key for Move; touching `state.selected`'s meaning.
