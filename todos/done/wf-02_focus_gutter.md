# wf-02 — The gutter: a third flex item with the two controls

The two controls appear on screen, in a third column between the two existing
ones, and they already flip the real state. Nothing moves visually yet — wf-03
supplies the geometry. What this task proves is that the controls exist, are
keyboard-reachable, carry the right labels, and drive the axis from wf-01.

Depends on: wf-01 (the `focus` / `leftHidden` axis must exist to be driven).

Read first: spec §3 (why a third flex item, and the two placements that were
rejected) and §4 (labels, target size). Decisions D8, D9, D10.

## What to deliver

### 1. `web/src/components/WorkBenchView.vue`

Destructure the new names from the existing `useChatLayout` call (`:70-74`) and
add **one element** between the two columns in the template — after the
`</div>` that closes `.chat-col--left` (line 107) and before
`<div ref="rightCol" class="chat-col chat-col--right">` (line 109):

```html
<div class="chat-col-gutter">
  <button
    type="button"
    class="chat-gutter-btn"
    :aria-pressed="focus === 'reasoning'"
    :aria-label="focus === 'reasoning' ? 'Restore the tool column' : 'Show the reasoning only'"
    @click="toggleFocus"
  >
    <!-- glyph placeholder: wf-03 replaces this with the dashed square -->
  </button>
  <button
    type="button"
    class="chat-gutter-btn"
    :aria-pressed="leftHidden"
    :aria-label="leftHidden ? 'Show the tool column' : 'Hide the tool column'"
    @click="toggleLeftHidden"
  >
    <!-- glyph placeholder: wf-03 replaces this with the dashed chevron -->
  </button>
</div>
```

Rules this template must obey:

* **focus control on top, hide control below** — the order the user's sketch
  draws, and the reason it is that way: the frequent gesture is the top one;
* both labels are English and describe the *action about to happen*, matching
  `"Hide the tool panels"` (`WorkBenchToolsDock.vue:56`);
* `aria-pressed` because both are state toggles (D9);
* no `v-if` on the gutter, in this task or any later one: it is the one surface
  that must survive every state, and it is where the way back lives.

The labels above are the spec's, but if wf-01's shape means `focus ===
'reasoning'` is not the readable test (for instance `leftHidden` is what derives
`conversation`), keep the *semantics* — "what this click does" — and adapt.

### 2. `web/src/styles/theme.css`

Add the gutter rules **inside the `/* -- chat screen */` block**, right after
`.chat-col--right` (line 1297) — not at the end of the file. Two rules only in
this task:

```css
.chat-col-gutter { … }      /* flex: 0 0 auto; width: 28px; align-self: flex-start;
                               margin-inline: calc(-1 * var(--bi-space-2));
                               display: flex; flex-direction: column; … */
.chat-gutter-btn { … }      /* the ≥28px target, currentColor, no border/background,
                               :hover / :focus-visible lit state */
```

* the negative `margin-inline` cancels the two gaps a third child would add
  (D10) — net cost to the conversation ~28 px, and the visible space between the
  columns stays what it is today;
* use the Pico/theme variables already in that block for colors
  (`--bi-muted`, `--bi-fg`, `--bi-border`, `--bi-space-*`) so light and dark both
  work without a second rule;
* **no inline style** and no size driven from the template;
* the buttons need `flex: none` on themselves or the gutter so the two targets
  cannot collapse at short window heights;
* the placeholder glyph may be a plain text-safe stand-in (an empty `<span>`
  sized by CSS, or `+`/`−` as a temporary mark) — do **not** design the final
  dashed SVG here, that is wf-03's deliverable and it needs its own pass at
  both themes.

### 3. Settle, in code, what the lot left open

* vertical rhythm of two 28 px targets inside 28 px of width: pick the padding
  and the gap, and check it at the 820 px breakpoint (`theme.css:2279-2283`,
  where each panel drops to 7 rem) — the gutter must not overlap either column
  at that width;
* reachability **before the first turn**: with no reasoning lines yet, the panel
  is empty but the controls must still read as available. Do not disable them
  for lack of content; if anything, the reasoning control is what the user needs
  most when a long answer is arriving.

## Hard rules

* **`WorkBenchToolPanel.vue` is not touched.** If it seems to need a change for
  the gutter to sit right, the geometry is in the wrong file — say so instead of
  doing it.
* No state decided in the template: `:class`, `:aria-pressed` and `:aria-label`
  read wf-01's values, they never compute a transition. `@click` calls the
  setter and nothing else.
* No `v-if`/`v-show` on `.chat-col-gutter` (see above), and no new
  `useChatLayout` argument — the gutter needs no ref, no splitter.
* Both columns' existing markup, both splitters and the `:class` binding at
  `:82` (`chat-col--single`) stay as they are.
* No state class on `.chat-shell` yet — D8's `chat-shell--reasoning` /
  `chat-shell--conversation` are wf-03's.

## Tests

Add to `web/tests/` (new file `web/tests/workbenchGutter.test.js`, Vitest +
`@vue/test-utils` `mount`, as in `web/tests/agentModes.test.js`):

* the gutter renders **exactly two buttons**, in that order, each with
  `type="button"`, an `aria-label`, and an `aria-pressed` that tracks wf-01's
  state after a click;
* clicking the top button moves `focus` to `'reasoning'` and back (assert through
  `useChatLayout`'s state, which the component and the test both import — the
  composable is a singleton-style module, so reading it after a click is the
  honest check);
* clicking the bottom one flips `leftHidden`, and its `aria-pressed` follows;
* the gutter is rendered when neither hub is active (the `chat-col--single`
  case, `bothHubs(state)` false at `:82`) — a control that disappears with the
  dock would strand the user in a hidden column;
* the buttons are real `<button>`s, so tab focus reaches them (assert
  `attributes().type === 'button'`; do not assert tab order, jsdom does not
  implement it faithfully).

If mounting `WorkBenchView.vue` pulls in too much (Monaco, EventSource, the
lightbox), it is legitimate to extract the gutter into
`web/src/components/WorkBenchFocusGutter.vue` — a presentational child taking
`{ focus, leftHidden }` as props and emitting the two clicks — and test that
instead. Note the trade-off in the commit: the composable stays the owner either
way, and wf-03's CSS then targets the extracted class names.

## Acceptance

* `cd web && npm test` green; `cd web && npm run build` green.
* Open the app (`benchmarkit`, WorkBench tab): two small controls sit between the
  columns, at the top of the gutter, and hovering them visibly "lights" them.
  Nothing else on screen changes when clicked — that is correct at this stage,
  and the e2e spec in wf-04 is what proves the geometry.
* `grep -n "chat-shell--" web/src web/dist` → **no hits** (wf-03 owns those names).
* `grep -rn "v-if" web/src/components/WorkBenchView.vue` shows no `v-if` on a
  `.chat-col-gutter` element.
* The gutter's width and its negative margins appear in exactly one CSS rule
  block, inside the `/* -- chat screen */` section.
