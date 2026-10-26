<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useBenchmark } from '../composables/useBenchmark'
import { useTerminal } from '../composables/useTerminal'

// Bundled with this component only (ed-05 rule 3), never in main.js. The xterm
// classes load lazily inside useTerminal.attach(); the emulator's base CSS rides
// this static import so it still lands in the Editor chunk, not the entry one.
import '@xterm/xterm/css/xterm.css'
// Shell-prompt icons need a Nerd Font. This rides the same Editor chunk as the
// xterm CSS above; its @font-face is unicode-range-gated, so the woff2 is only
// fetched when a prompt glyph actually falls in those ranges.
import '../fonts/nerd-fonts.css'

// The live xterm.js pane (ed-07): an emulator mounted into the host div below,
// bridged to `WS /api/terminal` (ed-06). All the socket/PTY/state logic lives
// in useTerminal — a singleton, so the shell survives the v-show tab switch and
// a remount of this component never opens a second socket behind the user's
// back. This component owns only the DOM lifecycle and the header.
//
// The mount trap, documented in useSplitter.js:112-120: the tab is v-show, so
// the pane is in the DOM but has no layout while hidden — `fit()` on a
// zero-height host computes 0 rows and xterm paints a blank grid. Handling:
// a ResizeObserver on the host that fits whenever the host has size (and skips
// while clientHeight === 0), which also covers entering the tab and the
// splitter drag; the host regains size in every one of those cases.
const host = ref(null)
const terminal = useTerminal()
const state = terminal.state
const bench = useBenchmark()

// The header meta, only while a session is live: pid + cwd from the `open`
// frame, so a second tab or a lost session is diagnosable.
const meta = computed(() =>
  state.status === 'open' && state.pid
    ? ` · pid ${state.pid} · ${state.cwd}`
    : '',
)

// Status dot reuses the app's status-pill palette: blue+pulse while connecting,
// green when open, red on error, muted otherwise.
const DOT_CLASS = {
  connecting: 'editor-terminal-dot--connecting',
  open: 'editor-terminal-dot--open',
  error: 'editor-terminal-dot--error',
}
const dotClass = computed(() => DOT_CLASS[state.status] ?? '')

const STATUS_LABEL = {
  idle: 'no session',
  connecting: 'starting…',
  open: 'running',
  closed: 'closed',
  error: 'error',
}
const statusLabel = computed(() => STATUS_LABEL[state.status] ?? state.status)

// The one caption shown when the pane is not running its shell — the reason a
// session is not there (killed, exited, refused, closed), which is also the
// cue that a manual New session is what brings one back.
const banner = computed(() => (state.status === 'open' ? '' : state.message))

// Kill only makes sense with a live socket; New makes sense any time the pane
// is not mid-handshake (it tears down any old socket first).
const canKill = computed(() => state.status === 'open')
const canNew = computed(() => state.status !== 'connecting')

// A ResizeObserver on the host is the single layout driver: it fires when the
// host gains size on mount, when the pane becomes visible again (v-show back
// on), and on every splitter-drag frame — so fit + the debounced `resize`
// frame ride one mechanism. (No separate "drag end" hook is needed: the drag
// changes the host's box, which is exactly what the observer reports.)
let observer = null
onMounted(async () => {
  await nextTick()
  if (host.value) terminal.attach(host.value)
  if (typeof ResizeObserver !== 'undefined' && host.value) {
    observer = new ResizeObserver(() => terminal.sendResize())
    observer.observe(host.value)
  }
})

// Entering the Editor tab is the explicit re-lay-out moment: a hidden terminal
// kept streaming into its scrollback and only re-fits now. (The ResizeObserver
// would also fire, but the activeTab watch makes the intent explicit and the
// two collapse via the debounce / zero-size guards.)
watch(
  () => bench.state.activeTab,
  (tab) => {
    if (tab === 'editor') nextTick(() => terminal.fitVisible())
  },
)

onBeforeUnmount(() => {
  observer?.disconnect()
  observer = null
  // The socket and the xterm instance are singletons owned by the composable
  // and outlive this component (the Editor view is v-show, not v-if), so the
  // shell is NOT torn down here. dispose() runs only when the whole app goes.
})
</script>

<template>
  <section class="editor-terminal" aria-label="Workspace terminal">
    <header class="editor-terminal-head">
      <span class="editor-terminal-title">
        Terminal<span class="editor-terminal-meta">{{ meta }}</span>
      </span>
      <span class="editor-terminal-state" role="status">
        <span class="status-dot editor-terminal-dot" :class="dotClass" aria-hidden="true"></span>
        <span class="caption">{{ statusLabel }}</span>
      </span>
      <button
        class="icon-btn editor-terminal-new"
        type="button"
        :disabled="!canNew"
        aria-label="Start a new terminal session"
        title="New shell (a dropped or closed session starts fresh here)"
        @click="terminal.newSession()"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M12 5v14" />
          <path d="M5 12h14" />
        </svg>
      </button>
      <button
        class="icon-btn editor-terminal-kill"
        type="button"
        :disabled="!canKill"
        aria-label="Kill the terminal session"
        title="Kills the shell and everything it started"
        @click="terminal.kill()"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M18 6 6 18" />
          <path d="m6 6 12 12" />
        </svg>
      </button>
    </header>

    <div class="editor-terminal-body">
      <div ref="host" class="editor-term-host"></div>
      <p v-if="banner" class="caption editor-term-banner">{{ banner }}</p>
    </div>
  </section>
</template>
