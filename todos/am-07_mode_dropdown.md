# am-07 — The mode dropdown in the composer

The user picks the mode before the first turn and sees, at a glance, which hubs
that mode can use. One composable owns the state (project invariant: *run/stream
state lives in the composables*), one small `<select>` renders it.

Depends on: am-06 (the route and the request field).

## 1. `web/src/composables/useAgentModes.js`

Shape it on `loadChatMcp` (`useChat.js:163-186`) — that function is the house
pattern for "fetch a server-owned list, degrade quietly on failure":

```js
export const state = {
  modes: [],            // [{id, label, mcptools, lsptools, editable}]
  current: null,        // the mode id in play; null until loaded
  default: null,        // what the server says the workspace default is
  loaded: false,        // false = the fetch failed; send() omits agentMode
  locked: false,        // set by D7 (see §3)
  findings: [],         // validation findings from the file, if any
}

export async function loadAgentModes() { … }
export function setAgentMode(id) { … }        // the one guard: §3
export function currentMode() { … }           // the object, or null
```

Rules copied from the hub loaders, and why:

* **failure hides nothing and changes nothing**: `loaded = false`, `current`
  stays `null`, and `send()` omits `agentMode` — the server then uses its own
  default, which is exactly what an old client gets. A failed dropdown must not
  silently downgrade anybody to `chat`;
* `state.findings` is kept but **not** rendered in v1 beyond the one line in §5 —
  the server already logged nothing, and this is the only place a user ever sees
  that their hand-edited file is wrong.

`current` is initialised to `body.default` on load, and a **workspace switch**
(`useBenchmark`'s `pathLocation` handling, near its existing
`reRoot(...)`/`state.mcpEnabled` refresh at `:865-874`) calls `loadAgentModes()`
again — the modes are per-workspace, so the list must follow the root. Reset
`current` to the new default on that reload: a mode id that existed in the old
workspace's file probably does not exist in the new one, and a 422 on every send
would be the alternative.

## 2. `send()` — one payload line

In `useChat.js` next to the existing `payload.mcpEnabled` / `lspEnabled` block
(`:963-981`):

```js
    if (agentModesLoaded()) {
      // The mode for this turn. Omitted when the list failed to load, so the
      // server's own default applies — the same rule as the hub allowlists
      // above, for the same reason.
      payload.agentMode = currentModeId()
    }
```

`useChat` may **not** import `useAgentModes` if that creates a cycle — check the
direction first. `useAgentModes` needs nothing from `useChat`, so
`useChat → useAgentModes` is safe and is what `useBatch → useChat` already does.
If a cycle does appear, use the existing `registerChatBusyGuard`-style injection
(`useChat.js:196-200`) rather than duplicating state.

## 3. The D7 lock — `todoBuilder` cannot be left once the thread has turned

Client-side, from data the server already sends (no `todoBuilder` string in the
client — am-06's grep rule):

```js
// `locked` is a *derived* flag, recomputed on every turn: the current mode is
// non-editable AND the thread already has a turn. `editable: false` is the
// server's flag for "this mode owns a contract" (the lot conventions), so the
// rule is data, not an id check.
function syncAgentModeLock(hasTurns) {
  const mode = currentMode()
  state.locked = !!mode && !mode.editable && hasTurns
}

function setAgentMode(id) {
  if (state.locked) return          // leaving a locked mode: refused
  const next = state.modes.find((mode) => mode.id === id)
  if (!next) return                 // never the server's list, never a guess
  state.current = next.id
  syncAgentModeLock(threadHasTurns())   // entering it can lock immediately
}
```

"Entering is always allowed; leaving is what locks" is the whole rule. Restated
from D7:

* entering `todoBuilder` at any point before the thread has a turn: allowed;
* **leaving** it once the thread has ≥ 1 turn: refused (the select keeps showing
  it, other options disabled);
* every other mode: switchable freely between turns, including mid-`streaming`
  (the switch simply has no effect on the turn already in flight — say that in a
  comment, because it looks like a bug otherwise);
* Restart (`useChat.reset`, `:1209`) **unlocks**: a cleared thread has no turns.
  Hook it the way the hub `locked` flags are cleared there, and keep the
  `refetchHubs=false` early-return in mind — the unlock must happen before that
  return, next to the other two.

Note the asymmetry to write down in one comment line: the lock is *client-side
convenience*, not enforcement — the server happily accepts `agentMode` changes
mid-thread for every mode, including this one. That is deliberate (a resumed
session JSON may legitimately want a different mode for its next turn) and the
guard exists to stop an accidental click, not to police the user.

## 4. Placement and markup

`WorkBenchInput.vue`, inside `.chat-bullet-row` (line 287) — in the **left**
action group (`chat-bullet-actions--left`, after the existing buttons), so the
memory bullets stay centered between the two button groups as the layout promises:

```html
<select
  class="chat-mode-select"
  :value="agentModes.current"
  :disabled="agentModes.locked || batch.active"
  :title="modeHint"
  aria-label="Agent mode"
  @change="onModeChange"
>
  <option v-for="mode in agentModes.modes" :key="mode.id" :value="mode.id">
    {{ mode.label }}
  </option>
</select>
```

* `modeHint` = the label plus which hubs the mode can use, e.g.
  `"Code — MCP and language servers available"` / `"Chat — no tools"`. This is
  the *only* place the tool gating is explained in the UI, so it has to be there;
* while `batch.active`, the select is disabled and shows `code` — the runner's
  pinned mode (D13, am-08), not the composer's choice;
* styles go in `web/src/styles/theme.css` as a `.chat-mode-select` rule using Pico
  variables (project invariant: no inline styles, no raw HTML rendering — and
  mode labels are text, `{{ }}` only, never `v-html`);
* if `agentModes.loaded` is false, render nothing rather than an empty select: a
  control that cannot be used is worse than no control.

## 5. The hub panels, grayed out

`WorkBenchMcpPanel.vue` / `WorkBenchLspPanel.vue` (or the dock that hosts them):
when `currentMode().mcptools` is false, the MCP panel renders disabled with a
one-line reason ("not available in Chat mode"). Do **not** hide the panel — the
toggles are still real (Settings owns them) and a hidden control that came back on
mode change reads as a bug. `currentMode()` returns `null` when the list failed to
load, and then the panels render exactly as they do today.

If a `findings` array is non-empty, add one quiet line under the select:
`"{n} problem(s) in the workspace agent-modes file — see .nojae/agent-modes.json"`.
Not a modal, not a banner: the file is the user's, outside the app, and a warning
that cannot be dismissed would nag on every turn.

## 6. Session save/load

`conversationJson` / `parseSessionFile` (`useChat.js:446-520`, `SESSION_FILE_VERSION = 3`):

* the mode id is **not** part of the message list — it is conversation state, so
  it goes in the session document's envelope next to `format`/`version`, as
  `agentMode: "code"`. Bump `SESSION_FILE_VERSION` to 4 and keep the loader
  tolerant: a v3 file has no `agentMode` and simply leaves `current` at the
  workspace default (that tolerance already exists for the `stats` block, `:499`);
* loading a session whose `agentMode` is unknown to the *current* workspace gets
  the default plus one line in `state.error`-adjacent text — never a silent
  substitution of a different persona, and never a refusal to load;
* if the loaded session is non-empty and the mode is the non-editable one, the
  lock is restored (`locked = true`) — the rule is about the thread, and a
  restored thread is that thread.

## 7. Tests — `web/tests/agentModes.test.js` (Vitest)

Model the file on `web/tests/chatHubs.test.js` (it already stubs `fetch` for the
hub loaders — reuse its harness rather than writing a third one):

* load → `current` is the server default; a 500 → `loaded === false`, `current`
  unchanged, and `send()` sends **no** `agentMode` key;
* the payload of a normal send carries `agentMode` with the selected id;
* `setAgentMode` matrix: every editable mode switchable at any time; the
  non-editable one enterable while unlocked and **not** leaveable once locked;
  `reset()` unlocks;
* a workspace-switch reload re-seeds `current` from the new default;
* `currentMode().mcptools === false` → the hub-panel disabled state (a component
  test on the panel, or a pure predicate test if the panel's disable logic is a
  computed you can export — prefer exporting the predicate);
* `conversationJson` round-trip: save with a mode, parse, `agentMode` survives; a
  hand-made v3 payload parses with `agentMode` absent and no error.

Add one Playwright case to `web/e2e/01-app-boot.spec.js` (or a new
`04-agent-modes.spec.js` if 01 is already long): boot → the select shows the
seeded default → switch to Chat → the MCP panel is disabled. No LLM call needed
for any of that, which is why it belongs in e2e at all.

## 8. Acceptance

* `cd web && npm test` green; `npm run build` green.
* `grep -rn "todoBuilder" web/src` → **no hits** (am-06's rule, enforced here).
* With the network failing for `/api/agent-modes`, the composer behaves exactly as
  it did before this feature. Say which test proves that (§7, second case).
* No mode logic in any `.vue` beyond rendering: every decision (lock, gating
  display, default selection) is a function in `useAgentModes.js`, so the next
  task in the "Later" list (a settings editor for modes) can reuse it.
