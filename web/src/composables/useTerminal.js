import { reactive, watch } from 'vue'
import { useTheme } from './useTheme'

// ---------------------------------------------------------------------------
// Editor terminal bridge: the state behind EditorTerminal.vue (ed-07).
//
// One singleton, like every other composable. It owns the WebSocket to
// `WS /api/terminal`, the xterm `Terminal` instance, and the connection state,
// so the shell survives a peek at another tab (App keeps the Editor view
// mounted with v-show) — and so a second mount of the pane never opens a second
// socket behind the user's back.
//
// The xterm classes (Terminal, FitAddon, WebLinksAddon) are imported *lazily*,
// inside `attach()`, never at module top and never in main.js (ed-05 rule 3):
// the Benchmark tab must not download the terminal. Their base CSS is imported
// statically by the component (EditorTerminal.vue), which keeps it in the same
// lazy Editor chunk as the classes.
//
// The frame contract is ed-06's: `input`/`resize`/`kill` out,
// `open`/`output`/`exit`/`error` in, all as JSON text. The server owns the
// shell, the PTY and the process-tree teardown; this layer only relays frames
// and reflects the connection to the header.
// ---------------------------------------------------------------------------

const CLOSE_POLICY = 4403
const CLOSE_CAPACITY = 4429
const CLOSE_UNSUPPORTED = 4501

const state = reactive({
  // idle -> connecting -> open -> (closed | error). No silent reconnect: a
  // dropped socket may have orphaned a shell, so the pane shows its reason and
  // waits for an explicit New session (see the todo's no-auto-reconnect rule).
  status: 'idle',
  pid: null,
  cwd: '',
  shell: '',
  message: '', // human line shown under the status when not open
})

// Transport / view internals — deliberately non-reactive. The reactive header
// reads `state`; xterm mutates its own DOM, not Vue's.
let socket = null
let term = null
let fit = null
let hostEl = null
let teardownLink = null // removes the link tooltip element
let resizeTimer = null
let sendQueue = [] // keystrokes typed before the `open` frame
let ready = false // true once the `open` frame has landed
let disposed = false

const theme = useTheme().theme

// The palette is JavaScript, not CSS, but it must follow the app's own tokens:
// read them off the live custom properties the stylesheet already maintains
// (`--bi-surface` for the panel this host sits in, `--pico-color` for text,
// `--pico-primary` for the cursor) instead of shipping a third palette.
function terminalTheme() {
  const style = getComputedStyle(document.documentElement)
  const read = (name, fallback) => (style.getPropertyValue(name) || fallback).trim()
  return {
    background: read('--bi-surface', '#0f172a'),
    foreground: read('--pico-color', '#e2e8f0'),
    cursor: read('--pico-primary', '#6366f1'),
  }
}

function setStatus(status, message = '') {
  state.status = status
  state.message = message
}

function flushQueue() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return
  for (const chunk of sendQueue) socket.send(JSON.stringify({ type: 'input', data: chunk }))
  sendQueue = []
}

function closeSocket() {
  if (!socket) return
  // Drop the handlers first: a socket we close on purpose still fires onclose,
  // and we do not want that to overwrite a status the action already set.
  socket.onopen = null
  socket.onmessage = null
  socket.onerror = null
  socket.onclose = null
  try {
    socket.close()
  } catch {
    // Already closing — nothing to do.
  }
  socket = null
}

function connect() {
  if (disposed) return
  closeSocket()
  ready = false
  sendQueue = []
  state.pid = null
  state.cwd = ''
  state.shell = ''
  setStatus('connecting', 'Starting a shell…')

  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  socket = new WebSocket(`${proto}://${location.host}/api/terminal`)

  socket.onopen = () => {
    // The socket is up but the shell is not yet announced — stay `connecting`
    // until the `open` frame, which is what the keystroke queue flushes on.
  }

  socket.onmessage = (event) => {
    let frame
    try {
      frame = JSON.parse(event.data)
    } catch {
      return // a non-JSON frame is not our contract; ignore it
    }
    switch (frame.type) {
      case 'open':
        ready = true
        state.pid = frame.pid ?? null
        state.cwd = frame.cwd ?? ''
        state.shell = frame.shell ?? ''
        setStatus('open', '')
        // The PTY was born at the server's default 80x24; push the real fitted
        // size now (not debounced) so the kernel SIGWINCHes the shell and its
        // first prompt wraps at the host's width, not 80 columns.
        pushResize()
        flushQueue()
        break
      case 'output':
        if (typeof frame.data === 'string') term?.write(frame.data)
        break
      case 'exit':
        // The shell left on its own (exit, Ctrl-D, a crash). One dim line, then
        // the bridge is over — no reconnect (its scrollback belongs to a dead
        // process).
        term?.write(`\r\n\x1b[2m[process exited — ${frame.code}]\x1b[0m\r\n`)
        ready = false
        setStatus('closed', 'The shell exited.')
        break
      case 'error':
        ready = false
        setStatus('error', frame.message || 'The terminal reported an error.')
        break
      default:
        break
    }
  }

  socket.onerror = () => {
    // onerror is always followed by onclose; the close handler owns the status,
    // so this is intentionally quiet (no browser-supplied detail to show).
  }

  socket.onclose = (event) => {
    ready = false
    sendQueue = []
    if (disposed) return
    // A refused upgrade names itself with a close code; surface it instead of a
    // generic "connection closed".
    if (event.code === CLOSE_POLICY) {
      setStatus('error', `Connection refused: ${event.reason || 'origin not allowed'}`)
    } else if (event.code === CLOSE_CAPACITY) {
      setStatus('error', event.reason || 'Too many terminal sessions — close one and retry.')
    } else if (event.code === CLOSE_UNSUPPORTED) {
      setStatus('error', event.reason || 'The terminal is unavailable on this platform.')
    } else {
      // A clean `exit`/`kill` leaves the status already `closed` with a
      // specific reason (the shell announced it, or the user asked for it).
      // The server then closes the socket normally, which fires onclose here;
      // do not downgrade that reason to a generic line. Only an *unexpected*
      // drop (socket died while the session looked open/connecting) gets the
      // generic "connection closed".
      if (state.status === 'open' || state.status === 'connecting') {
        setStatus('closed', 'Connection closed.')
      }
    }
  }
}

// Keystrokes, as xterm encoded them (already control bytes: ^C -> 3).
// `term.onData`, not a key handler — intercepting key events instead would
// break Ctrl-combinations and the IME path.
function send(data) {
  if (!data) return
  if (ready && socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'input', data }))
  } else if (state.status === 'connecting') {
    // Typed during the handshake: queue so the first characters of the first
    // command are not lost; flushed on the `open` frame.
    sendQueue.push(data)
  }
  // Any other status (idle/closed/error): no live session to receive it.
}

function kill() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return
  // Explicit tree kill (the server's `kill_session_tree`). The socket then
  // closes on its own; do not reconnect.
  setStatus('closed', 'Session killed.')
  socket.send(JSON.stringify({ type: 'kill' }))
}

// A fresh shell in the same pane: reset the scrollback (the old one belonged to
// a dead process) and open a new socket, which the server answers with a new
// session. The old socket's close tears its shell down server-side.
function newSession() {
  term?.reset()
  connect()
}

// Fit the grid to the host and tell the PTY, debounced: dragging the splitter
// fires the ResizeObserver many times a second and each PTY resize is a
// TIOCSWINSZ plus a SIGWINCH.
function sendResize() {
  if (resizeTimer) clearTimeout(resizeTimer)
  resizeTimer = setTimeout(pushResize, 50)
}

function pushResize() {
  if (!term || !fit) return
  // A hidden host (v-show) has no layout: fitting it yields 0 rows and xterm
  // paints a blank grid. Skip until it has real size.
  if (!hostEl || hostEl.clientHeight === 0 || hostEl.clientWidth === 0) return
  try {
    fit.fit()
  } catch {
    return // racing a teardown / a zero-size frame
  }
  if (!socket || socket.readyState !== WebSocket.OPEN) return
  const { cols, rows } = term
  if (!cols || !rows) return
  socket.send(JSON.stringify({ type: 'resize', cols, rows }))
}

// The one global-ish shortcut the pane must NOT swallow: the app already binds
// Ctrl+W (close the active file tab) at window level in EditorArea. Returning
// false here lets xterm skip the chord so that window handler fires.
function keyHandler(event) {
  const mod = event.ctrlKey || event.metaKey
  if (mod && event.key.toLowerCase() === 'w') return false
  return true
}

// web-links decision: default click handler (window.open), but show the full
// URL on hover so a printed link is diagnosable before the user opens it.
function attachLinkTooltip() {
  let tip = null
  const hide = () => {
    tip?.remove()
    tip = null
  }
  const show = (event, text) => {
    if (!tip) {
      tip = document.createElement('div')
      tip.className = 'term-link-tooltip'
      term.element.appendChild(tip)
    }
    tip.textContent = text
    const rect = term.element.getBoundingClientRect()
    tip.style.left = `${event.clientX - rect.left + 8}px`
    tip.style.top = `${event.clientY - rect.top + 16}px`
  }
  teardownLink = hide
  return { hover: show, leave: hide }
}

// The component calls this on mount with its host element. Everything xterm is
// created lazily here so the pane, not the app shell, pays for it.
async function attach(host) {
  if (disposed) return
  hostEl = host

  // The xterm classes are lazy so the Benchmark tab never downloads them; their
  // CSS is imported statically at the component's module top (see the file
  // header), which bundles it into the same Editor chunk.
  const [{ Terminal }, { FitAddon }, { WebLinksAddon }] = await Promise.all([
    import('@xterm/xterm'),
    import('@xterm/addon-fit'),
    import('@xterm/addon-web-links'),
  ])
  if (disposed) return // unmounted before the import resolved

  term = new Terminal({
    // The plain JetBrains Mono carries ordinary text; the 'JetBrains Mono Nerd'
    // face (web/src/fonts/nerd-fonts.css, unicode-range-gated to PUA icons) is
    // what makes shell-prompt icons render instead of xterm's missing-glyph box.
    fontFamily: "'JetBrains Mono', 'JetBrains Mono Nerd', ui-monospace, SFMono-Regular, monospace",
    fontSize: 13,
    lineHeight: 1.25,
    scrollback: 2000, // bounded: this pane shares the page with mermaid + KaTeX
    theme: terminalTheme(),
    cursorBlink: true,
  })
  fit = new FitAddon()
  term.loadAddon(fit)
  term.loadAddon(new WebLinksAddon(undefined, attachLinkTooltip()))

  term.attachCustomKeyEventHandler(keyHandler)
  term.onData(send)

  term.open(hostEl)
  pushResize() // set the initial window size before the shell prints its prompt

  if (state.status === 'idle' || state.status === 'closed' || state.status === 'error') {
    connect()
  }
  // A session that survived the pane being unmounted/remounted stays as-is.
}

// The tab-visible path: a hidden terminal kept receiving output into its
// scrollback but only re-lays-out now. Fit, then push the size to the PTY.
function fitVisible() {
  pushResize()
}

function dispose() {
  disposed = true
  if (resizeTimer) clearTimeout(resizeTimer)
  teardownLink?.()
  closeSocket()
  term?.dispose()
  term = null
  fit = null
  hostEl = null
}

// Follow the app palette on a light/dark flip.
watch(theme, () => {
  if (term) term.options.theme = terminalTheme()
})

// Unload: no confirmation, and an explicit teardown of the remote tty.
//
// Nothing on the page holds unsaved user input today — the file surface is
// read-only and a shell has no buffer worth keeping — so a reload must not be
// blocked. What unload *does* owe the server is prompt teardown of the PTY, so
// a refresh cannot leave a shell (and anything it started) alive until the
// browser's socket cleanup happens to land: `kill` asks for the session's
// explicit tree kill, `dispose()` then closes the socket, which is the
// belt-and-braces path — if the frame is lost in the unload race, the closed
// socket makes `Session.run` see the disconnect and `Session.teardown` runs the
// same SIGTERM→reap→SIGKILL (src/core/termhub/session.py). Teardown is
// idempotent, so the two paths cannot double-kill.
//
// `pagehide`, not `beforeunload`: it is the event that fires whenever the
// document really goes away (reload, tab close, navigation), whereas a
// `beforeunload` listener that only calls preventDefault can be cancelled by the
// user, and a listener that closes the socket there would then have destroyed a
// session for a page that stayed.
window.addEventListener('pagehide', () => {
  if (disposed) return
  if (socket && socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(JSON.stringify({ type: 'kill' }))
    } catch {
      // Dropped during unload — the close below still ends the session.
    }
  }
  dispose() // closes the socket and releases xterm
})

// Kept for when the Editor gains a write path (ed-10): a dirty Monaco buffer
// *is* worth a "leave without saving?" prompt. Re-enable by checking the buffers
// — `useEditorFiles().state.open.some((file) => file.dirty)` — not the terminal
// status, and keep this a separate listener from the teardown above: cancelling
// this one means the user stays, so it must never close the socket.
// window.addEventListener('beforeunload', (event) => {
//   if (state.status === 'open') {
//     event.preventDefault()
//     event.returnValue = ''
//   }
// })

export function useTerminal() {
  return {
    state,
    attach,
    connect,
    send,
    kill,
    newSession,
    sendResize,
    fitVisible,
    dispose,
  }
}
