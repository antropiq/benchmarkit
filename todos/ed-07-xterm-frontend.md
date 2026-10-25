# Step 7 — Terminal pane (xterm.js)

Replace the terminal placeholder with the live xterm.js pane: emulator, `fit`,
`web-links`, the WebSocket bridge, and the reconnect/kill affordances.

Depends on: ed-02 (the pane and its splitter), ed-05 (packages), ed-06 (the
endpoint). This completes the feature as briefed: tree + editor + terminal.

## Files

| File | Role |
|---|---|
| `components/EditorTerminal.vue` | the pane: header (label, status dot, Kill, New), the `<div class="editor-term-host">` xterm mounts into |
| `composables/useTerminal.js` | singleton: socket, `Terminal` instance, status (`idle`/`connecting`/`open`/`closed`/`error`), `connect()`, `send()`, `kill()`, `attachCustomKeyEventHandler` |
| `styles/theme.css` | `.editor-term-host`, `.term-header`, xterm colour overrides |

Import inside the component, not in `main.js` (ed-05 rule 3):

```js
// module top — bundled with this component only, never in main.js
import '@xterm/xterm/css/xterm.css'

// inside the mount routine
const { Terminal } = await import('@xterm/xterm')
const { FitAddon } = await import('@xterm/addon-fit')
const { WebLinksAddon } = await import('@xterm/addon-web-links')
```

## Mount lifecycle — the part that bites

The tab is `v-show`, so the pane is in the DOM but has **no layout while
hidden**: `fit.fit()` on a zero-height host computes 0 rows and xterm paints a
blank grid. This is the exact trap `useSplitter.onResize` documents
(`useSplitter.js:112-120`). Handling:

* a `ResizeObserver` on the host: `fit()` whenever the host gains size or
  changes it, and **skip the call while `clientHeight === 0`**;
* `watch(state.activeTab)`: on entering the Editor tab, `fit()` then send a
  `resize` frame — a hidden terminal keeps receiving output into its scrollback
  and only re-lays-out when it becomes visible again;
* the splitter's drag end also triggers `fit()`;
* `dispose()` in `onBeforeUnmount`, and `term.dispose()` before replacing a
  session.

Debounce the `resize` frame to ~50 ms: dragging a splitter fires `ResizeObserver`
many times per second, and each PTY resize is a `TIOCSWINSZ` plus a `SIGWINCH`.

## Bridge

```js
const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}//${location.host}/api/terminal`
socket = new WebSocket(url)
socket.onmessage  → JSON.parse → open|output|exit|error → term.write(data)
term.onData       → socket.send({type:'input', data})
fit+observer      → socket.send({type:'resize', cols, rows})
```

Rules:

* **`term.onData`, not `onkeydown`** — xterm turns keystrokes into the exact
  control bytes (`^C` → `\x03`); intercepting key events instead would break
  Ctrl-combinations and the IME path;
* **no auto-reconnect while a session may still be alive**: a dropped socket
  means a possibly-orphaned shell server-side, so the pane shows *"connection
  closed"* with an explicit **New session** button. Silent auto-reconnect would
  either leak shells or hand the user a terminal whose scrollback belongs to a
  dead one — the app's reconnect philosophy (`EventBus` replay) exists because
  a benchmark run is resumable; a shell is not;
* `open` frame: put pid + cwd in the header, so a second tab or a lost session
  is diagnosable;
* `exit` frame: print one dim line (`[process exited — N]`) and stop the bridge;
* **buffer while connecting**: keystrokes typed during the handshake are queued
  and flushed on `open`, otherwise the first characters of the user's first
  command vanish.

## Theme

xterm colours are JavaScript, not CSS, but they must follow `useTheme()`
(`composables/useTheme.js`) — the app supports light and dark. One palette
object per mode, derived from the same Pico variables the stylesheet uses
(`--pico-bg`, `--pico-color`, `--pico-primary`) read through
`getComputedStyle` at mount, so the terminal matches the app instead of
shipping a third palette:

```js
const style = getComputedStyle(document.documentElement)
term.options.theme = {
  background: style.getPropertyValue('--pico-bg').trim(),
  foreground: style.getPropertyValue('--pico-color').trim(),
  cursor:     style.getPropertyValue('--pico-primary').trim(),
}
```

`watch` the theme and reassign `term.options.theme` on flip. Font: JetBrains
Mono (already loaded via `@fontsource/jetbrains-mono`, and `lineHeight: 1.25`),
`fontSize: 13`, `scrollback: 2000` — bounded, because this pane shares the page
with mermaid and KaTeX.

## `web-links`

`WebLinksAddon` with the default handler (`window.open`) is fine locally, but a
terminal that prints a URL should not be a way to reach a host on the FQDN
deny-list with one click. Pass a custom handler that reuses the MCP hub's
`restricted-fqdn.txt` check when the URL is local-network-shaped, or, simpler
and consistent with the read-only stance: render the link with a title showing
the full URL and let the user's browser decide. Decision recorded here:
**default handler + visible URL in the tooltip**; the deny-list stays a
connector policy, not a browser policy (the app cannot police what the user
opens in their own browser, and pretending otherwise would be a false promise).

## Interaction rules

* Focus follows the pane: clicking the terminal focuses it; clicking the editor
  unfocuses it. `Ctrl+\``-style global toggles are **not** added (ed-02: no new
  global shortcut).
* The header shows: `Terminal · pid 41237 · /mnt/bigfoot/dev/benchmarkit`, a
  status dot in the app's existing status-pill colours, and two buttons —
  **Kill session**, **New session** — mirroring the hub restart buttons in
  `SettingsDialog.vue`.
* `beforeunload`: if a session is open, warn once ("a shell is running"). The
  server also tears the session down when the socket closes, so the warning is
  courtesy, not the safety net.
* While `state.status === 'running'` (a benchmark run) the pane stays usable:
  the terminal is independent of the run loop, and a user may legitimately want
  to watch a log. It must **not** be used to start a second run — the app's
  single-run invariant lives in `/api/run`, which answers 409 regardless of
  where the request came from, so the shell cannot break it either.

## Manual checklist

* Type into the pane immediately after opening the tab, before the connection
  settles: nothing is lost.
* Resize the pane by dragging: text reflows within ~100 ms, one resize frame per
  settle (network panel), no `SIGWINCH` storm.
* Switch tabs away and back: the terminal repaints correctly, scrollback intact,
  no blank grid (the zero-height guard).
* Flip light/dark: foreground, background and cursor follow the app palette.
* Kill: the shell and its children die (`ps`); New session starts a fresh one.
* A URL in output (`curl` a page that prints one) is underlined and clickable.
* Reload the page: the old socket closes, its session is torn down server-side,
  a new session starts on demand — `GET /api/terminal` never accumulates.
