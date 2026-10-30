# wf-03 — The two state rules and the dashed glyph

The geometry. Two classes on the shell move the columns in and out of the flow,
and wf-02's placeholder marks become the dashed squares and chevron the user
asked for. **After this task the feature works** — wf-04 only proves it.

Depends on: wf-01 (the axis), wf-02 (the gutter and its buttons).

Read first: spec §2 (the state table), §4 (the glyph), §5.1, §5.2, §5.5, §5.6,
§6 (what to leave alone). Decisions D2, D3, D5, D8, D11.

## What to deliver

### 1. The state classes on the shell (D8)

`web/src/components/WorkBenchView.vue`, line 78 — the only template change:

```html
<div
  class="chat-shell"
  :class="{
    'chat-shell--reasoning': focus === 'reasoning',
    'chat-shell--conversation': leftHidden,
  }"
>
```

The two names and their meaning are fixed by D8. They go on `.chat-shell` and
**not** on the columns — that element is where the chat geometry is already
decided (it carries the breakpoint-overridden `--bi-chat-panel-width`,
`theme.css:2275,2281`).

### 2. The two state rules

In `web/src/styles/theme.css`, inside the `/* -- chat screen */` block next to
wf-02's gutter rules. The geometry of the feature is these rules and nothing
else:

```css
.chat-shell--conversation .chat-col--left { display: none; }

.chat-shell--reasoning .chat-col--right { display: none; }
.chat-shell--reasoning .chat-col--left { flex: 1 1 auto; }
.chat-shell--reasoning .chat-tools-dock,
.chat-shell--reasoning .chat-col-resizer { display: none; }
```

* `display: none`, never `v-if` (D2). The panel's line cache is a
  component-local `WeakMap` (`WorkBenchToolPanel.vue:90`) and its column count
  comes from a `ResizeObserver` on the line host (`:109-110`) — unmounting it
  re-cuts every reasoning line of the session from `MIN_BUDGET` and loses the
  scroll position. Hiding in CSS also keeps `useChatLayout`'s refs, both
  splitters and `useAutoScroll` alive without any help;
* the `reasoning` rule takes the **dock and its bar out of the flow too** (D5):
  `.chat-tools-dock` (`WorkBenchToolsDock.vue:36`) claims `DOCK_MIN = 104` px
  (`useChatLayout.js:23`), so with only the right column hidden the timeline
  gains width but not height, and the user's "maximised" sketch is not reached.
  The dock's dragged height in `benchmarkit:chatToolsColumn` is *not* written or
  zeroed (D3) — returning to `normal` restores it;
* `.chat-col--single` (`theme.css:1293`) must keep working in all three states:
  `reasoning`'s `flex: 1 1 auto` overrides its basis, and `conversation` hides
  the column entirely, so neither needs a special case — verify both, do not
  assume;
* no new `--bi-chat-*` token, and **no rule under a `@media` block** unless the
  narrow-screen question in §4 forces one. The breakpoints at
  `theme.css:2273-2283` must not have to learn about the gutter (spec §3).

### 3. The glyph (D11)

Replace wf-02's placeholders with inline `<svg>` inside the two buttons,
`stroke="currentColor"`, no `fill`:

* **focus control**: two concentric squares (`viewBox="0 0 24 24"`, outer span
  ~4→20, inner ~8→16), both with `stroke-dasharray`; visible mark ~14 px inside
  the ≥ 28 px target;
* when focused (`focus === 'reasoning'`): **one** square only — restore must not
  reuse the maximise motif. Drive that with a `v-if` on the inner `<rect>`:
  hiding an SVG child is not the D2 problem, D2 is about the panel component;
* **hide control**: a dashed chevron whose direction says what the click does —
  pointing left while the left column is shown (it will fold away), right while
  it is hidden (it will come back). `aria-hidden="true"` on both SVGs: the
  button already carries the label (wf-02);
* the "lit" hover state (D11): color and opacity come forward and the dashes go
  solid on `:hover` and `:focus-visible` (a `stroke-dasharray: none` rule on the
  same paths). **No size animation** — the target must not move under the cursor.

Pick the dash pattern and the colors in code, and check the "lit" state in
**both** themes: `theme.css` carries light and dark, and only one of them gets
checked by eye by default.

### 4. Settle, in code, the narrow-screen question

Whether the gutter ever hides below 1100 px / 820 px. Default answer: **no**
(spec §3 — the breakpoints must not know about it). If it must hide, it needs a
way back that does not strand the user: never hide the control of the state you
are in — a hidden left column with no chevron is an unrecoverable screen. Write
the answer, and the reasoning, in the commit message either way.

## Hard rules

* `WorkBenchToolPanel.vue` stays untouched (wf-02's rule still holds). The panel
  already fills its column: `.chat-tool-panel { flex: 1 1 0 }`
  (`theme.css:2096`). If a change there seems necessary, the geometry is in the
  wrong file — say so instead of making it.
* Nothing writes or reads `benchmarkit:chatToolsColumn` / `benchmarkit:chatSplit`
  (D3). No state transition is computed in the template: the two `:class` reads
  above are the whole of it.
* No Escape handler, no keyboard shortcut (D7), and do **not** keep the composer
  visible in `reasoning` (D6): the right column goes entirely, Stop included. The
  way out is the top gutter control, then Stop — two steps, on purpose.
* Do not try to preserve the reasoning row budget across a hidden state (spec
  §6): it falls to `MIN_BUDGET` while the measure host is 0×0 and the lines
  re-cut on return. Expected — a breakpoint change already does the same, and
  `web/tests/reasoningLines.test.js:116` pins that contract.
* No `v-html`, no raw `innerHTML`; the SVGs are markup in the template.
* Nothing at the end of `theme.css` — every rule in the `/* -- chat screen */`
  block.

## Tests

Extend wf-02's file (`web/tests/workbenchGutter.test.js`, or the extracted
gutter component's) — jsdom reads classes, it does not lay out:

* `setFocus('reasoning')` → the shell element carries `chat-shell--reasoning` and
  **not** `chat-shell--conversation`; `toggleLeftHidden()` → the reverse pair;
  `normal` → neither. Assert both classes for all three states, so D1's
  exclusivity is visible on the DOM as well as in the composable;
* the focus control's inner square is absent when `focus === 'reasoning'` and
  present otherwise (the D11 motif difference);
* the chevron's direction flips with `leftHidden`;
* both SVGs carry `aria-hidden="true"`, and the buttons keep wf-02's
  `aria-label` and `aria-pressed`.

No CSS assertion: `web/vitest.config.js` stubs CSS (`css: false`, *"the theme is
verified visually in e2e, not asserted here"*). The real geometry is wf-04's
Playwright spec.

## Acceptance

* `cd web && npm test` green; `cd web && npm run build` green.
* By hand, in the browser, **in both themes**, WorkBench tab:
  1. top control → the reasoning panel alone, edge to edge at full window
     height, with **no dock strip left at the bottom** (D5);
  2. top control again → exactly the layout as before, **including a dock height
     you dragged first** (D3);
  3. bottom control → the left column gone, conversation full width, the gutter
     still visible and clickable;
  4. resize across 1100 px and 820 px in each state: nothing overlaps, the
     gutter is present, the state survives;
  5. the mark reads dashed at rest, solid and brighter on hover, in light *and*
     dark.
* With a long answer streaming, enter and leave `reasoning`: the lines re-cut
  (expected, spec §6) and nothing is lost — the answer text stays intact.
* `grep -n "chat-shell--" web/src/styles/theme.css` shows the state rules, and
  `git diff` shows no `.vue` change beyond the shell's `:class` and wf-02's
  gutter markup.
