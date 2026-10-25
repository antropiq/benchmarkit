# Step 2 — The Editor tab and its three-pane layout

Create the third tab and the shell: navigator on the left (full height), editor
area in the centre, terminal strip at the bottom. The tree from ed-01 is wired
in; the editor and terminal panes are placeholders (a `<div>` and a "terminal
coming in ed-07" label) so the layout can be reviewed and resized on its own.

Depends on: ed-01. Required by ed-03 … ed-09.

## Tab registration — three touched sites, no new concept

| Site | Change |
|---|---|
| `composables/useBenchmark.js:10` | comment becomes `// benchmark | chat | editor` |
| `composables/useBenchmark.js:408` `setTab()` | the running-run guard: today `if (tab === 'chat' && state.status === 'running') return`. Decision D6: the Editor tab is reachable while a run is in progress (reading files is harmless), so the guard stays keyed on `chat` only |
| `components/TabNav.vue` | a third `.tab-btn`, `role="tab"`, `aria-selected`, `title="Editor"`; **not** disabled during a run, unlike WorkBench |
| `App.vue` | `<EditorView v-show="state.activeTab === 'editor'" />` next to `<WorkBenchView v-show="…'chat'" />`, imported through `defineAsyncComponent` (see below) |

`v-show`, not `v-if`, is deliberate: it is what keeps the chat tab's scroll and
split state across switches (`useSplitter.js:112` depends on exactly that), and
it is what keeps the open files (read-only buffers, decision D4) and a live
terminal session alive when the user peeks at the benchmark (decision D5).
Consequence to remember:
a hidden `.chat-shell`/`.editor-shell` has no layout and `clientHeight === 0`,
which is why `useSplitter.onResize` bails on a zero-height shell — the new
splitters must copy that guard or they will collapse a restored size to the
minimum on the next refresh.

Lazy loading: `const EditorView = defineAsyncComponent(() => import('./components/EditorView.vue'))`.
CodeMirror and xterm land in ed-04/ed-07 and are the heavy part; keeping the
view async means the Benchmark tab (the app's default screen) never downloads
them, and the tab's first open pays ~100 ms.

## Layout — CSS grid, one column pair

```
.editor-shell              display: grid
  grid-template-columns: <sidebar> auto 1fr      /* auto = the resizer */
  grid-template-rows:    1fr auto <terminal>     /* auto = the resizer */
  height: calc(100vh - var(--bi-appbar-height))  /* as .chat-shell does */

┌ .editor-nav (grid-row: 1 / -1) ─┬─ .editor-resizer-v ─┬─ .editor-area ─┐
│  WorkspaceNavigator, full height│                     │ editor tabs    │
│                                 │                     │ <placeholder>  │
│                                 │                     ├─ .editor-resizer-h ┤
│                                 │                     │ .editor-terminal │
└─────────────────────────────────┴─────────────────────┴────────────────┘
```

The navigator spans **both rows**, which is precisely the brief's "terminal
full width minus the bottom of the file explorer": its right border and the
terminal's left border are the same line. Add the grid-area declarations
explicitly (`grid-area: nav | resizer-v | editor | resizer-h | term`) rather
than relying on auto-placement, so a future pane cannot reshuffle the grid.

Sizing defaults: sidebar `minmax(200px, 260px)`, terminal `220px`, both
overridden by the splitter's flex/grid basis once dragged. The resizers reuse
the `.chat-resizer` visual (10 px hit area, pill on hover/focus/drag,
`touch-action: none`) — extract `.pane-resizer` in `theme.css` and let
`.chat-resizer` keep its current appearance through it, so one stylesheet
change restyles both.

## Splitters — extend `useSplitter` rather than fork it

`composables/useSplitter.js` is already the vertical-splitter pattern (pointer
capture, `clamp()` against both minima, `localStorage`, keyboard `↑↓`/`Home`,
double-click reset, `ResizeObserver` validation on becoming visible). Two
generalisations, both backward compatible:

1. **axis** — an `orientation: 'vertical' | 'horizontal'` option so the sidebar
   splitter measures width instead of height (`startX`, `col-resize`,
   `aria-orientation="vertical"`, `←→` keys);
2. **storage key + minima** — currently hardcoded (`STORAGE_KEY`,
   `MIN_INPUT_HEIGHT`, `MIN_OUTPUT_HEIGHT`); move them into an options object,
   and keep the existing call signature working for the chat tab.

New keys: `benchmarkit:editorSidebar`, `benchmarkit:editorTerminal`. Minima for
the sidebar: 160 px, and never more than half the shell. For the terminal: 96 px
(two lines + the panel header) against a 200 px editor minimum.

For the grid, apply the size through a CSS custom property on the shell rather
than an inline `style` on one child: `--bi-editor-sidebar: 240px` /
`--bi-editor-terminal: 220px`, consumed by `grid-template-columns/rows`. The
composable already returns computed styles; returning the variables instead of
flex bases keeps the grid declarative.

## Pane chrome

| Element | Content in this step |
|---|---|
| `.editor-area` header | an empty tab strip + the empty state (icon, "Open a file", one caption line) — same `stream-empty` idiom as `WorkBenchView.vue:185` |
| `.editor-terminal` header | a `Terminal` label, a reconnect dot and a **Kill session** button (wired for real in ed-07; disabled here) |
| `.editor-area` header, right side | an **`read-only`** static badge: a tab labelled "Editor" promises an editable surface, and decision D4 says otherwise. Hidden once ed-10 makes the pane writable |
| `.editor-status` (optional, bottom of the editor area) | `path · language · Ln, Col` — populated in ed-04 |

Empty-state and error text follow the app's convention: `class="caption"` for
the helper line, `role="alert"` + `.alert-error` for a failed request (see
`App.vue:76-80`).

## Interaction rules that must be stated once

* **A run does not block the tab** (D6), but the Settings dialog's workspace
  change does affect it: on a successful `POST /api/path-location`,
  `useWorkspaceTree.refresh()` and the terminal restart must both be triggered.
  The response already returns the new path plus both hub payloads, so one hook
  in the save path covers all four surfaces (tree, editor, terminal, hubs).
* **A WorkBench conversation does not block the tab either.** The pane stays
  open and readable while the model runs tools, and the files those tools touch
  are refreshed into the open buffers — the rule decision D4 attaches to the
  read-only stance. Nothing is wired in this step; ed-09 owns the probe and the
  refresh, and it hangs off `useChat`'s existing `tool_result` frames, so this
  step only has to keep the pane mounted (`v-show`) rather than unmounted, which
  it already does.
* **No keyboard conflict**: the editor pane claims `Ctrl+F` (CodeMirror search)
  and `Ctrl+W` (close tab) only while it has focus; there is no `Ctrl+S` in this
  lot (read-only, D4 — it arrives with ed-10). The terminal owns everything else
  xterm receives. No global shortcut is added, so the benchmark and chat
  keyboard behaviour is untouched.
* **Responsive**: below 1024 px the grid collapses to a single column with the
  navigator as a collapsible `<details>`-style drawer, mirroring the existing
  `@media` rules that flatten `.app-shell` and `.chat-shell` (theme.css:404,
  :1953). Desktop-only is the documented scope, so this is a usability floor,
  not a mobile layout.

## Files touched

```
web/src/App.vue                                +1 import, +1 line in template
web/src/components/TabNav.vue                  +1 button
web/src/composables/useBenchmark.js            comment + (optional) setTab
web/src/components/EditorView.vue              new, ~90 lines
web/src/components/EditorArea.vue              new placeholder, ~30 lines
web/src/components/EditorTerminal.vue          new placeholder, ~25 lines
web/src/composables/useSplitter.js             orientation + options
web/src/styles/theme.css                       editor grid, resizers, media query
```

No backend change in this step.

## Manual checklist

* Third tab appears after WorkBench, labelled **Editor**, reachable while a
  benchmark run is in progress; WorkBench's disabled-during-run behaviour is
  unchanged.
* Drag the sidebar: 160 px ≤ width ≤ half the window; reload: restored.
* Double-click a resizer: default size returns; `Home` does the same from the
  keyboard; arrow keys step.
* Switch to Benchmark mid-drag and back: both sizes survive, and a page refresh
  after a tab switch does not collapse the terminal to its minimum (the
  zero-height-shell guard).
* Resize the window narrow: single-column fallback, nothing clipped, the tree
  still reachable.
* Light and dark theme: the grid lines, resizers and placeholders read cleanly
  in both.
* Network tab on first load of the Benchmark tab: **no** CodeMirror or xterm
  chunk downloaded.

## Delivered

| File | Change |
|---|---|
| `web/src/components/EditorView.vue` | new: the three-pane shell — both splitters wired, the sizes carried as `--bi-editor-sidebar` / `--bi-editor-terminal` on the shell, drawer toggle for narrow screens |
| `web/src/components/EditorArea.vue` | new placeholder: tab strip, `read-only` badge (D4), `stream-empty` centre |
| `web/src/components/EditorTerminal.vue` | new placeholder: `Terminal` header, reconnect dot, disabled Kill session |
| `web/src/composables/useSplitter.js` | generalised, the chat call site unchanged: `orientation` (width vs height, `←→` vs `↑↓`), `storageKey` / `minResized` / `minOther` / `maxRatio`, and `sizeVar` → `shellStyle` for grid shells |
| `web/src/styles/theme.css` | `.editor-shell` grid (navigator spans every row), explicit `grid-column/row` per pane, `.pane-resizer` (+ `--vertical`) extracted from `.chat-resizer`, pane chrome, `@media (max-width: 1024px)` fallback; the ed-01 dev-preview wrapper deleted |
| `web/src/App.vue` | `EditorView` via `defineAsyncComponent`, `v-if` once then `v-show` |
| `web/src/components/TabNav.vue` | third `role="tab"` button, enabled during a run (D6) |
| `web/src/composables/useBenchmark.js` | `activeTab` comment, `setTab` guard documented as chat-keyed, `setPathLocation` refreshes the tree |
| `web/src/components/WorkBenchView.vue` | resizer markup `pane-resizer chat-resizer` |
| `AGENTS.md` | three tabs; the `App.vue` row notes `v-show` and the lazy Editor chunk |

### What differs from the plan, and why

* **Lazy loading needs a first-mount gate.** An async component behind a plain
  `v-show` resolves its import at boot, so the default screen would have paid
  for the tab. `App.vue` flips `editorMounted` the first time `activeTab`
  reaches `'editor'`; from then on only `v-show` acts, so the view never
  unmounts again — which is what D5 (and ed-07's terminal session) needs.
  Verified in the build output: the tab ships as its own chunk
  (`EditorView-*.js`, ~9.7 kB) and nothing in the entry references it eagerly.
* **The workspace-change hook sits in the save path, not in the tab.** The
  plan had `EditorView` watch `pathLocationCurrent`; the tab may never be
  mounted, so that watch would not exist yet. `setPathLocation` (which already
  refreshes both hubs) now calls `useWorkspaceTree().refresh()`, keeping the
  "one hook, all four surfaces" rule. The terminal restart joins that same call
  in ed-07.
* **Editor inert during a *batch*.** D6 keys off a benchmark run, and the
  Editor stays reachable there. A running batch is the app-wide pin from
  batch-mode B11 (every other tab goes inert, and `setTab` refuses), so the new
  button follows `batch.active` only.
* **`useSplitter` measures the shell's content box.** The panes live inside the
  shell's padding, so `clientHeight` alone counted it twice and let the
  terminal grow past the bottom edge; the clamp now subtracts the padding, and
  a hidden shell (`<= 0`) is treated as "nothing to clamp against" — the same
  zero-layout guard the chat splitter needed.
* **Default sidebar track is the brief's `minmax(200px, 260px)`**, so a
  double-click or `Home` returns to the CSS default rather than to a fixed
  240 px that would ignore a narrow window.

No backend change, no new dependency: ed-01's read-only assertion (no non-`GET`
route under `/api/fs`) still holds because no route was added.
