# wf-01 — The focus state axis in useChatLayout

The screen gets three states, and this task makes them **expressible and
exclusive in the composable alone** — no CSS, no template, nothing visible on
screen yet. Later tasks render what this decides.

Depends on: nothing (first task of the lot).

Read first: `docs/WORKBENCH-FOCUS-LAYOUT-SPEC.md` §2, §5.3, §5.4. Decisions
D1, D4 and D8 come from the lot's big picture.

## What to deliver

In `web/src/composables/useChatLayout.js` (67 lines today), a focus axis and a
persisted left-column flag, returned beside the existing `{ column, composer,
dockCollapsed }`:

```js
// Three values, one axis: exactly one focalisation can be in play, so "both at
// once, screen empty" is not a state this composable can be in (D1).
const FOCUS_NORMAL = 'normal'
const FOCUS_REASONING = 'reasoning'
const FOCUS_CONVERSATION = 'conversation'
```

and, as the return value of `useChatLayout(...)`:

```js
return { column, composer, dockCollapsed, focus, setFocus, leftHidden, toggleLeftHidden }
```

* `focus` — a `ref` holding one of the three strings. `setFocus(next)` validates
  against that set and ignores anything else; `toggleFocus()` maximises when
  `focus === 'reasoning'` and enters `reasoning` otherwise (so a second click on
  the same control restores — that is the user's gesture, spec §2).
* `leftHidden` — the `conversation` state's flag, **persisted** under
  `benchmarkit:chatLeftHidden`, on the exact pattern of `COLLAPSE_KEY`
  (`useChatLayout.js:26-34` read side, `:58-62` write side: `try/catch` around
  `localStorage`, a failed write commented as "storage disabled: still works for
  the session"). `toggleLeftHidden()` flips it.
* `focus` is **not** persisted (D4).

## The single arbitration that matters

`leftHidden` and `focus` are two different storages of one axis. `conversation`
*is* `leftHidden`; `reasoning` *is* `focus`. So the setter rules are:

* `setFocus('conversation')` sets `leftHidden = true`;
* `setFocus('normal')` and `setFocus('reasoning')` set `leftHidden = false`;
* `toggleLeftHidden()` entering hidden sets `focus = 'conversation'`; leaving
  hidden sets `focus = 'normal'`;
* entering `reasoning` while hidden uncovers the left column — the two can never
  be on together.

One way to keep that honest: derive the returned `focus` from `leftHidden` when
`leftHidden` is true, rather than storing `conversation` twice. Either shape is
fine as long as no caller can observe `reasoning` **and** a hidden left column
at the same time. Say in one comment line which shape you picked, because the
next task reads this file to know where truth lives.

`toggleFocus()` must not fight `leftHidden`: if the left column is hidden, the
reasoning panel is not on screen, so `toggleFocus()` uncovers it and enters
`reasoning` in one step.

## Hard rules

* **`useSplitter` is not touched, and neither are its two storage keys.** D3: no
  focus state writes or reads `benchmarkit:chatToolsColumn` or
  `benchmarkit:chatSplit`. The two `useSplitter(...)` calls at `:37-46` stay
  byte-identical.
* **No new state in any `.vue`.** Project rule: state lives in the composables.
  This task adds no template, no CSS, no class name. `chat-shell--reasoning` and
  `chat-shell--conversation` (D8) are wf-03's, not this one's — do not add them
  here.
* Keep `dockCollapsed` independent: the dock fold and the focus axis must not
  become one flag. `DOCK_MIN`, `DOCK_MIN_OTHER`, the comment block at `:4-22` and
  the `collapsed` write-through stay as they are. (Updating that header comment
  to describe the gutter is **wf-04's** job — it is the last task to see the
  final shape. Leave it alone here.)
* Plain JS, matching the file's existing style (no TypeScript, no new import
  beyond what the file already uses: `computed`, `ref` are already imported).

## Tests — `web/tests/useChatLayoutFocus.test.js` (Vitest, jsdom)

New file. Model it on `web/tests/chatHubs.test.js` for structure (explicit
`import { describe, it, expect } from 'vitest'` — the suite has no globals, see
`web/vitest.config.js`). `useChatLayout(ref, ref, ref, ref)` can be called
directly with plain Vue refs; no component needed.

Note: **no existing frontend test touches `localStorage`**, so this file is the
first harness for it in `web/tests`. jsdom provides a working
`window.localStorage`; for the throws case, replace the accessor for one test:

```js
const original = window.localStorage
Object.defineProperty(window, 'localStorage', {
  configurable: true,
  get() {
    throw new Error('storage disabled')
  },
})
// … assert, then restore with Object.defineProperty(window, 'localStorage',
//    { configurable: true, value: original })
```

`vitest.config.js` sets `restoreMocks: true` but that does not undo a
redefined property — restore it yourself, in `afterEach`, or the next test file
in the run inherits a throwing `window`.

Cases:

1. the three values are reachable, and `toggleFocus()` round-trips
   `normal → reasoning → normal`;
2. **exclusivity**: from `reasoning`, going to `conversation` leaves `focus` not
   `'reasoning'` and `leftHidden` true; from `conversation`, `setFocus('reasoning')`
   leaves `leftHidden` false. Assert the pair `(focus, leftHidden)` after each
   transition for all six ordered pairs of states — that is the D1 table;
3. `setFocus('nonsense')` and `setFocus(undefined)` leave the state unchanged;
4. `leftHidden` **survives a re-mount**: call `useChatLayout` twice, flip the
   flag on the first, the second reads it back from storage. `focus` does not:
   a fresh call after `setFocus('reasoning')` starts at `'normal'` (D4);
5. **no writes to the splitter keys**: seed `benchmarkit:chatToolsColumn` and
   `benchmarkit:chatSplit` with values, run the whole transition matrix, and
   assert both raw values are untouched (D3);
6. `dockCollapsed` stays independent: fold the dock, enter `reasoning`, leave
   it — `dockCollapsed` never changed, and the dock's key still holds what it
   held;
7. throwing `localStorage`: every toggle still works for the session, and no
   exception escapes (mirrors `loadCollapsed()`'s `catch`).

## Acceptance

* `cd web && npm test` green, including the new file; `cd web && npm run build`
  green.
* `git diff` shows this task touched exactly two files:
  `web/src/composables/useChatLayout.js` and the new test. If `WorkBenchView.vue`
  or `theme.css` appear, this task overreached — those are wf-02 and wf-03.
* `grep -n "chatSplit\|chatToolsColumn" web/src/composables/useChatLayout.js`
  still shows exactly the two pre-existing `useSplitter` option lines and nothing
  else.
* The feature is invisible in the app at this point (no control renders yet).
  That is expected: say so in the commit message rather than reaching for the
  template to prove it works.
* One sentence in the task's final note: where the truth of `conversation`
  lives (derived or stored), since wf-02 and wf-03 read it from here.
