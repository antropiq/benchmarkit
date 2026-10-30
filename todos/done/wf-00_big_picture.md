# WorkBench focus layout — big picture

Give the WorkBench screen **two focalisations**: put the reasoning timeline alone
on the whole window, or take the whole left column away so the conversation owns
it. Three states, two controls, one gutter.

The full design (need, rejected alternatives, side effects) is written in
**`docs/WORKBENCH-FOCUS-LAYOUT-SPEC.md`** — 330 lines, in French, and it is the
authority. This file is the map and the decision list; read the spec's §5 and §6
before touching geometry. Nothing here restates it except the decisions a task
cannot re-derive.

Why it exists: on a long answer the reasoning is the most interesting thing on
screen and the smallest — `--bi-chat-tools-col-width`, i.e. two 15.5 rem panels
plus 0.5 rem (`theme.css:27-28`), and only as tall as the dock leaves room. And
when you only want the conversation, the tool panels and the gauge hold a
vertical strip for nothing.

## The three states

| state | left column | right column | gutter |
|---|---|---|---|
| `normal` | in flow | in flow | visible |
| `reasoning` | `flex: 1 1 auto`, all the place | `display: none` | visible |
| `conversation` | `display: none` | in flow | visible |

Mutually exclusive, and the exclusivity lives **in the state type** (one axis of
three values), not in a CSS rule and not in two booleans that can both be `true`.

## The gutter

A third flex item between `.chat-col--left` and `.chat-col--right`
(`.chat-shell` is `display: flex; gap: var(--bi-space-2)`, `theme.css:1267-1275`),
28 px wide, holding both controls stacked — focus on top, hide below.

It is not an overlay and not a child of the tool panel. Two reasons, both load-
bearing:

* it depends on **no width number**. An absolutely-positioned control would have
  to re-derive its `left` from `--bi-chat-tools-col-width` for the `reasoning`
  state, the `conversation` state, the `chat-col--single` variant
  (`theme.css:1293`) and the two breakpoints that redefine `--bi-chat-panel-width`
  to 11 rem then 7 rem (`theme.css:2273-2283`). A flex item crosses all four
  without knowing they exist;
* it is the **only surface no state hides**, so both controls sit at the same
  place in the same order in every state. The earlier placements failed here: a
  chevron on the gauge bar (`chat-tool-gauge`, `WorkBenchToolPanel.vue:179`) is
  inside the column the `conversation` state hides, so the control that lets you
  in vanishes once you are in — and `.chat-tool-panel` is `overflow: hidden`
  (`theme.css:2103`), so a mark meant to overhang would be clipped anyway.

## Decisions

| # | decision | why |
|---|---|---|
| D1 | three values on **one axis** (`focus`), one setter arbitrates | two independent booleans can both be `true` and the screen goes empty. Unsayable in the type beats checked in the CSS |
| D2 | hide with `display: none` from the shell, **never `v-if`** | the tool panel keeps its line cache in a component-local `WeakMap` (`WorkBenchToolPanel.vue:90`) and its column count in a `ResizeObserver` (`:109-110`). A `v-if` unmounts it: cache empty, budget back to `MIN_BUDGET` = 4, every reasoning line of the session re-cut in front of the user, scroll lost. It also keeps `useChatLayout`'s refs, both splitters and `useAutoScroll` alive for free |
| D3 | **no state writes** `benchmarkit:chatToolsColumn` or `benchmarkit:chatSplit`, and none reads them to size anything | states take elements out of the flow, they never resize. Returning to `normal` restores dragged heights as they were — the rule the dock fold already follows ("Folding is a flag, not a size", `useChatLayout.js:48-52`) |
| D4 | `conversation` is persisted (`benchmarkit:chatLeftHidden`), `reasoning` is **not** | one is a furnishing preference (same order as `benchmarkit:chatToolsCollapsed`), the other a reading mode. Persisting `reasoning` would boot the app with no composer: a screen with no way to speak to the model |
| D5 | in `reasoning`, the **dock and its bar leave the flow too** | otherwise `reasoning` only widens the column and the dock still claims `DOCK_MIN = 104` px (`useChatLayout.js:23`): the timeline gains width but not height, and the "maximised" sketch is not reached. The dragged height is untouched (D3), so it comes back as it was |
| D6 | in `reasoning`, the right column is **fully** hidden — no composer, no Stop button | `reasoning` is a reading surface, not a control station. To interrupt a running answer: click the focus control again (the column returns), then press Stop. Two steps, on purpose |
| D7 | no keyboard shortcut in this iteration | the AppBar already carries the activity indicator; an Escape that interrupts while the user expects an Escape that shrinks is a trap |
| D8 | state classes go on **`.chat-shell`**, not on the columns | the shell is where the chat geometry is already decided (it carries the breakpoint-overridden `--bi-chat-panel-width`). Names: `chat-shell--reasoning`, `chat-shell--conversation` |
| D9 | both controls are `<button type="button">` with an English `aria-label` and `aria-pressed` | they are state toggles, not one-shot actions. English labels like the rest of the chrome ("Hide the tool panels", `WorkBenchToolsDock.vue:56`) |
| D10 | the mark is small, the target is not: ~14 px glyph inside a ≥ 28 px target, and the gutter cancels its own two gaps (`margin-inline: calc(-1 * var(--bi-space-2))`) | a third flex child would otherwise cost 2×14 px of gap, i.e. ~48 px taken from the conversation; cancelling brings the net cost to ~28 px and leaves the visual spacing between columns unchanged |
| D11 | the glyph: double dashed square at rest, **single** dashed square when focused, brighter on hover/focus, no size animation | the user's drawing. "Restore" must not reuse the "maximise" motif; the target must not move under the cursor |

## Accepted side effects — do not "fix" them

Spec §6 in full; the two that look like bugs and are not:

* **changing state re-cuts every displayed reasoning line.** The row budget comes
  from the `ResizeObserver`, and `displayLines` re-wraps from scratch when
  `cache.budget !== width` (`reasoning-lines.js:81`). That is already what a
  breakpoint change does — "a rendered line never moves" holds *at constant
  width*, not between widths. A test pins that contract today
  (`web/tests/reasoningLines.test.js:116`).
* **the budget falls to `MIN_BUDGET` while a column is `display: none`.** The
  measure host is 0×0, so `advance > 0` fails in `remeasure()`
  (`WorkBenchToolPanel.vue:73-75`), `columns = 0`, budget clamps to 4. The source
  text (`block.text`) stays the truth — that is what a session save reads. Do not
  add code to preserve the budget across a hidden state: re-cutting is unavoidable
  anyway, so the code would buy nothing.

## Not in this lot

* the same mechanism for the Editor tab or the terminal;
* a fourth state "conversation only" (composer visible, timeline full height) —
  the `conversation` state already covers the want, and a fourth value makes D1
  harder to explain, not easier;
* keyboard shortcuts (D7);
* a draggable gutter width — a third splitter in a view that has two needs a
  reason to exist.

## How to run this lot

Launch Batch mode with **prefix `wf-`** (hyphen included), lot folder `todos`.
The hyphen is part of the prefix: `core/batch.py`'s `parse_task_number` wants the
digits immediately after the prefix, so `wf` matches nothing and every file lands
in `skipped`. Note `todos/` still holds the finished `am-00_big_picture.md` — a
different prefix, so `task_lot('wf-', 'todos')` will not list it.

Everything is frontend: no backend file, no route, no SSE event, no `.env` key.

## Task order

| # | task | depends on |
|---|---|---|
| wf-01 | the three-value `focus` axis and persisted `leftHidden` in `useChatLayout.js` — no CSS, no template | — |
| wf-02 | the gutter as a third flex item in `WorkBenchView.vue`, with the two buttons | wf-01 |
| wf-03 | the two state rules on `.chat-shell` and the dashed glyph — from here the feature works | wf-02 |
| wf-04 | the test sweep (Vitest + one Playwright spec) and the header comment of `useChatLayout.js` | wf-01 … wf-03 |

Each of wf-01 → wf-03 ships its own Vitest. wf-04 is the sweep for what the
individual tasks could not see — chiefly `display: none`, which only a real
browser lays out.

## Project rules that bind these tasks

From `AGENTS.md`, and enforced by the acceptance checks:

* *"run/stream state lives in the composables, which own the EventSource and its
  reconnect"* — by extension, layout state lives in `useChatLayout.js`; a `.vue`
  renders it and decides nothing;
* new structure carries no inline styles: Pico variable overrides and app classes
  go in `web/src/styles/theme.css`. The `:style` bindings that exist today
  (the dock's split basis) stay as they are;
* `web/dist` must exist for FastAPI to serve the UI: run `npm run build`;
* tests: `cd web && npm test` (Vitest over `web/tests`), and
  `cd web && npm run test:e2e` (Playwright over `web/e2e`, against its own server
  and the endpoint from `.env.test`, never the developer's `.env`).

## Open questions a task must settle in code, not here

* **wf-02**: the vertical rhythm of two buttons in 28 px of width — whether the
  gutter carries any visual separator between them, and how it reads at the
  820 px breakpoint where each panel is 7 rem. Also whether the gutter is
  reachable before the first turn, when the reasoning panel has nothing to show:
  the controls must not look like they are waiting for content.
* **wf-03**: the `stroke-dasharray` pattern of the dashed square, and the "lit"
  state's contrast in **both** themes — light and dark live in one file and only
  one is checked by eye;
* **wf-03**: whether the gutter ever hides on narrow screens. Spec §3 says the
  breakpoints must not have to know about it, so the default answer is *no* — but
  at 820 px the whole screen is scarce. If it has to hide, it needs a way back
  that does not contradict D1: never hide the control of the state you are in;
* **wf-04**: the dragged-dock-height assertion (D3, D5) as an e2e assertion or a
  composable assertion, whichever is deterministic — jsdom cannot lay out, and a
  real browser drag is timing-sensitive.
