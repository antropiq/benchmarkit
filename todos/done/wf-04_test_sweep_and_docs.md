# wf-04 — The test sweep, and the header comment that must not go stale

The lot's last task: the checks the individual tasks could not make — chiefly
`display: none`, which **only a real browser lays out** (jsdom reports a hidden
column as present in the DOM) — and the one comment in the codebase that
describes this screen's geometry and would otherwise become a false statement.

Depends on: wf-01, wf-02, wf-03 (it tests their finished result).

Read first: spec §8 (both verification levels, and what is deliberately **not**
tested) and §7's closing paragraph (the header comment).

## What to deliver

### 1. `web/e2e/05-workbench-focus.spec.js` (Playwright)

Numbered `05-` because `web/e2e/` currently stops at
`04-agent-modes.spec.js`. Style that file: `import { expect, test } from
'@playwright/test'`, `page.goto('/')`, `page.getByRole('tab', { name:
'WorkBench' }).click()`, `page.getByLabel(...)`.

No LLM call and no run is needed for anything below — `.chat-tool-panel` is
unconditionally rendered (`WorkBenchToolPanel.vue:139-141` has no `v-if` at its
root), so the geometry is measurable on a cold, empty WorkBench screen. Keep it
that way: a spec that needs a streamed answer is a spec that depends on
`.env.test` and turns flaky.

Cases, in the order the user would perform them:

* **baseline**: in `normal`, read and record `.chat-col--left`'s and
  `.chat-tool-panel`'s bounding boxes, and the gutter's visibility;
* **`reasoning`**: click the focus control (by its `aria-label`, not by index).
  Then:
  - `.chat-col--right` → `toBeHidden()` — Playwright's hidden verdict is exactly
    `display: none`, which is the whole point of D2;
  - `.chat-tool-panel`'s width is larger than the baseline width, and its height
    too (D5 — the height assertion is what catches a dock strip left behind);
  - `.chat-tools-dock` → `toBeHidden()`;
  - the gutter and both its controls are still visible;
* **the way back**: click the same control, both boxes are back to the baseline
  within a pixel;
* **`conversation`**: click the hide control. `.chat-col--left` →
  `toBeHidden()`, the conversation's width grew, and — the assertion that
  protects spec §3, the one the rejected gauge-bar placement would have failed —
  **the same control is still visible and clickable**, and clicking it restores
  the baseline;
* **persistence** (D4): reload in `conversation` → still hidden. Reload in
  `reasoning` → back to `normal`. Do it with `page.reload()`, and be ready for
  the `conversation` reload to be the app's own startup state;
* **dragged geometry survives** (D3, D5): drag the dock's bar
  (`.chat-col-resizer`, `WorkBenchView.vue:85-99`, a `role="separator"` with
  `aria-label="Resize the tool panels (drag, arrow keys, double-click to
  reset)"`), record the dock's height, enter and leave `reasoning`, and assert
  the height came back. `toBeHidden()` cannot be used to prove
  `display: none` — assert the *box*, not the visibility flag, for this one;
* **keyboard parity**: focus the control with `page.keyboard.press('Tab')` up to
  it, or `page.focus(...)`, and press `Enter`/`Space` — a `<button>` must work
  without a mouse, since it is chrome the user hits repeatedly.

Settle in code, not here: whether the dragged-height case is asserted in e2e at
all. A real browser drag is timing-sensitive; if it will not hold, assert the
same rule in `web/tests/useChatLayoutFocus.test.js` (wf-01 case 5 already covers
"the storage keys are untouched") and say in the spec's comments that the e2e
half is the visibility, the jsdom half is the storage. The rule must be covered
somewhere, in one of the two layers, visibly.

### 2. The Vitest sweep

Add to `web/tests/useChatLayoutFocus.test.js` (wf-01's) and
`web/tests/workbenchGutter.test.js` (wf-02/wf-03's) only what those files could
not see while their own task was in flight:

* a **full three-state cycle** through the composable — every control clicked in
  the order a user would (`normal → reasoning → normal → conversation →
  reasoning → conversation → normal`) — with the `(focus, leftHidden)` pair and
  the two shell classes asserted at each step. wf-01 tested transitions, this
  tests a walk;
* **the dock fold is orthogonal**: fold the dock (its own persisted flag), then
  walk all three states, then unfold — the fold is exactly as it was. And the
  reverse order. This is the pair of cases that shows the two independent flags
  were never silently merged;
* `chat-col--single` (one hub or none active) renders the gutter and both
  controls in all three states — the case wf-02 listed, worth re-checking once
  the state classes exist;
* the gutter has **no `v-if` anywhere in `WorkBenchView.vue`** — a source-level
  assertion is acceptable here if a DOM one is awkward, and it is the guard for
  D2's whole premise.

If wf-02 extracted a `WorkBenchFocusGutter.vue`, target that component instead
and note it in the commit.

### 3. The header comment of `useChatLayout.js`

Lines 4-22 of `web/src/composables/useChatLayout.js` **describe the screen's
geometry**: the two-column diagram, and *"Folding is the one control that spans
both"*. After this lot that sentence is false — the gutter spans both, and there
are three controls, not one.

Update it to the shape the code ended up with: the three-child shell, the two
state classes, and the reason the states are flags rather than sizes (D3's
"Folding is a flag, not a size" line stays valid and is still the reference for
that principle). Keep it in the file's existing voice — dense, one idea per
line, ~20 lines — and keep the `DOCK_MIN` explanation intact.

This is the **only** documentation change. Deliberately **not** touched:

* `AGENTS.md` — verified: it never mentions the chat screen's columns, and its own
  rules forbid the kind of enumeration that ages badly. Adding a paragraph about
  two focus states would be paid for on every WorkBench turn for no invariant;
* `README` and the spec file — the spec is the design record, it does not get
  edited to match the implementation. If the implementation legitimately
  departed from it, write the departure in the commit message instead, so the
  document keeps showing what was decided and why.

## Hard rules

* **No backend file, no route, no SSE event, no `.env` key** is touched by this
  task or the lot. If `pytest` collects anything new, that is a mistake.
* `cd web && npm run test:e2e` runs against its own server and takes its LLM
  endpoint from `.env.test`, never the developer's `.env` (project rule). The
  spec must not need an endpoint at all.
* Do not add CSS assertions to the Vitest layer: `web/vitest.config.js` stubs CSS
  (`css: false`) and its comment says the theme is verified visually in e2e.
* Do not "fix" anything wf-03 shipped in passing. If the sweep exposes a real
  defect, fix it and say which case exposes it; if it exposes an accepted side
  effect from spec §6 (lines re-cut on state change, budget at `MIN_BUDGET` while
  hidden), leave the code and assert nothing about it.
* The e2e spec reads labels from the DOM the app renders, not from string
  constants copied into the spec — if a label has to be matched, derive it or
  locate it by role. That is what keeps `04-agent-modes.spec.js` honest.

## Acceptance

* `cd web && npm test` green (all files, none skipped); `cd web && npm run build`
  green; `cd web && npm run test:e2e` green, with `05-workbench-focus.spec.js`
  present and its cases named as run.
* `pytest` green and unchanged in count — this lot is frontend-only, and a
  passing, untouched Python suite is the proof.
* The four rules of the lot each have a named test pointing at them, and the
  task's final note lists which: D1 exclusivity, D2 (no `v-if` / `display: none`),
  D3+D5 (dragged geometry survives), D4 (persistence asymmetry).
* Reading `useChatLayout.js`'s first 25 lines now describes the screen as it
  actually is: three children, two state classes, and why neither resizes
  anything.
* One last manual pass, both themes, both breakpoints: enter `reasoning` while a
  long answer is streaming, leave it, and confirm the answer and the reasoning
  text are intact afterwards (spec §6's re-cut is visible but not corrupting —
  spec §8 says this one is checked by hand, not by a spec).
