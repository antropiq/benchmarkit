<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import GaugeBar from './GaugeBar.vue'
import { useChat } from '../composables/useChat'
import { useAutoScroll } from '../composables/useAutoScroll'
import { createLineCache, displayLines, MIN_BUDGET } from '../lib/reasoning-lines'
import { boxCapacity, selectWindow } from '../lib/reasoning-window'

const { state, chatBusy } = useChat()

// Chronological timeline of everything the right-hand chat stream does NOT
// show: the model's streamed reasoning flows and the MCP tool-call badges,
// in the order the backend emitted them (reasoning → calls → next round's
// reasoning…). Blocks are reactive objects, so tool_result state flips
// re-render the badges in place. Append-only, hence index keys are stable.
const blocks = computed(() =>
  state.messages
    .filter((m) => m.role === 'assistant')
    .flatMap((m) => m.blocks ?? []),
)
const reasoningChars = computed(() =>
  blocks.value.reduce(
    (sum, block) => sum + (block.kind === 'reasoning' ? block.text.length : 0),
    0,
  ),
)

// Iteration gauge tracks the latest assistant reply: 0 while it streams
// before its first tool round, holds its final level afterwards, and drops
// to the previous reply (or 0) when Stop/editLast splices messages away.
const lastAssistant = computed(() => {
  for (let i = state.messages.length - 1; i >= 0; i -= 1) {
    if (state.messages[i].role === 'assistant') return state.messages[i]
  }
  return null
})
const gaugeValue = computed(() => lastAssistant.value?.mcpIteration ?? 0)
const gaugeMax = computed(() => lastAssistant.value?.mcpMax ?? 0)

// A frame's worth of thinking arriving between two frames is what drives both
// the autoscroll and the Live View's window: one reactive source, one path.
const list = ref(null)
const signature = computed(() => {
  const last = blocks.value[blocks.value.length - 1]
  const tail = last ? `${last.kind}-${last.kind === 'tool' ? `${last.tool.id}-${last.tool.state}` : ''}` : ''
  return `${blocks.value.length}:${reasoningChars.value}:${tail}`
})
const { onScroll } = useAutoScroll(list, signature)

// ---------------------------------------------------------------------------
// Live View — the mode of `docs/spec-live-reasoning-view.md` §3.3: while it is
// on, the panel paints only what its box can hold, and the reasoning that
// scrolled off above the top leaves the DOM. That bounded list is the answer to
// a slow machine: a reasoning row is a fixed-height block (`.chat-reasoning-line`
// clips its own overflow, so a row is one line tall whatever its text holds),
// so an unbounded timeline is an unbounded layout on every painted frame.
//
// What stays exact, and why: the rows are still ALL cut by lib/reasoning-lines,
// incrementally and from each flow's own cache — only the *painting* is capped,
// so no cut can drift and the panel still scrolls by arithmetic. `block.text`
// stays the source of truth for saving and reloading a session (see useChat's
// `reasoningFlows`), so the loss §3.4 accepts is a loss of screen, never of
// bytes, and a reloaded session paints its rows again whole: this capacity is
// render state, held here and not in the messages. Rows stay grouped per block
// exactly as the unpruned list had them, so the list's gap keeps separating
// blocks rather than rows — which is why the window hands back blocks carrying
// rows, and why its budget is spent in rows and not in characters.
//
// One window for the whole timeline, spent from the newest block backwards
// (see lib/reasoning-window): a flow scrolls out of its oldest rows too, so the
// thinking being read is always whole and the edge lands inside the oldest flow
// on screen; a tool badge costs one row and is shown whole or not at all.
//
// The window's tail and the reader's eye stay together on their own — `useAutoScroll`
// already pins this box's scroll to its bottom, which is the automatic follow of
// §3.3 — so the panel needs no second notion of where the visible text is.
//
// The box is measured in rows ONCE per inference, never per chunk: the row's
// own box and the list's gap, read together, say how much of the box the
// timeline costs. While a reply streams the number is frozen (see `pinnedRows`
// below) — the box can change under the panel mid-flux (the dock folded, the
// screen focalised) and a window that followed it would prune or resurrect
// whole blocks under the reader's eyes, which is what §3.3 forbids.
// ---------------------------------------------------------------------------
const busy = computed(chatBusy)

// The mode itself belongs to the conversation, not to this component: useChat
// owns the flag and the gutter carries the control that switches it, so the
// panel only ever reads it. One owner per mode, no second notion of it.
const liveEnabled = computed(() => state.liveView)

// Rows of the box the timeline may cost. 0 = never measured, which reads as
// "no window" rather than as a window of nothing.
const windowRows = ref(0)

// The same number, frozen for the length of one inference while `busy` is on,
// so no box change mid-flux (the dock folded, the screen focalised, the
// composer dragged) can prune or resurrect blocks under the reader's eyes
// (§3.3). 0 = not decided yet, which paints everything: a window nobody can
// justify is worse than no window.
const pinnedRows = ref(0)

// A floor of rows the panel may show, so a font that measures oddly (or a
// dock folded to nothing) cannot reduce the panel to a sliver the reader
// cannot read; a ceiling so the timeline stays bounded whatever the box's
// height is — a tall box is still a box, and rows above it are off screen.
const LIVE_MIN_ROWS = 8
const LIVE_MAX_ROWS = 400

// Rows of the box the timeline costs, via `boxCapacity` (see lib/reasoning-window
// for why the divisor is the row's height alone). Both numbers are read off the
// elements, never assumed, because both belong to the theme.
function rowsInBox() {
  const scroll = list.value
  // The scroller gives the box's height, the list its padding and the row its
  // height: values that belong to the theme, all read off the elements.
  const strip = scroll?.querySelector('.chat-tool-list')
  if (!strip) return 0
  const line = strip.querySelector('.chat-reasoning-line')
  if (!line) return 0 // no list laid out (a headless host): no box to fill
  const style = getComputedStyle(strip)
  // The height belongs to the scroller (the list grows with its content, so its
  // own height says nothing about the box). Only the list's bottom padding is
  // subtracted: the panel pins this box's scroll to its bottom, so the bottom
  // padding is the one thing always inside the box, while its top padding sits
  // above the top edge — charging for it too would leave a row's sliver of
  // blank panel under the last line of the flow.
  const inside = scroll.clientHeight - parseFloat(style.paddingBottom || 0)
  const capacity = boxCapacity({
    inside,
    rowHeight: line.getBoundingClientRect().height,
  })
  if (!capacity) return 0
  return Math.min(LIVE_MAX_ROWS, Math.max(LIVE_MIN_ROWS, capacity))
}

function measureWindow() {
  const measured = rowsInBox()
  if (measured) windowRows.value = measured
}

// At rest the window follows the box, so a folded dock or a focalised screen
// resizes it before the next inference measures against it. A ResizeObserver
// callback is the layout's own notification, which is why the box read there
// is settled and reading it forces no reflow.
let windowObserver = null

function remeasureWindow() {
  if (busy.value) return // mid-inference the window is frozen, whatever moved
  measureWindow()
}

// The window of one inference is decided as the reply starts, from the box the
// panel had at rest — which is the box it has, since at rest the window follows
// it and a reply never resizes the box. Deciding it there rather than at the
// first painted row is what keeps the very first frame of thinking from painting
// the whole timeline of a long conversation and pruning it after. Releasing it
// when the reply ends is what returns the button to use (§4 of the spec).
watch(busy, (streaming) => {
  pinnedRows.value = streaming && liveEnabled.value ? windowRows.value : 0
})

const capacity = computed(() => {
  if (!liveEnabled.value) return Infinity // off: the whole history, no window
  const rows = busy.value ? pinnedRows.value : windowRows.value
  return rows || Infinity
})

// The window's input: every block of the timeline, with the rows it paints and
// therefore the rows of height it costs. `reasoningRows` is only ever read
// here, so each flow's row cache stays incremental whatever the window does
// with the result. `id` is the block's place in the emission order, which
// never moves for a block — so it is the key the list patches by, and a
// window that slides moves blocks out at one end without renaming the ones
// still on screen. (A position key would shift every remaining block on every
// pruned one and re-render the whole list, which is the cost that would make
// the mode a pessimization instead of a fix.)
function units() {
  return blocks.value.map((block, i) => {
    if (block.kind === 'tool') return { id: i, block, rows: null, cost: 1 }
    const rows = reasoningRows(block)
    return { id: i, block, rows, cost: rows.length }
  })
}

// What the panel paints: the whole timeline when the mode is off, and the
// window of lib/reasoning-window when it is on. `units()` is what makes the
// rows of every flow a cached, incremental cut, so the only thing a chunk costs
// here is the slice that spent the budget — never a re-wrap.
const view = computed(() => selectWindow(units(), capacity.value))

// `dropped` is deliberately never read here: the mode says nothing about what
// left the screen. The eye reads the state off the control's own glyph (open
// eye = the whole history is on screen, closed one = only the visible part is),
// and a count of removed lines beside it would restyle the strip on every
// painted frame and put a moving number in the panel's footer.

// ---------------------------------------------------------------------------
// Reasoning text is rendered as fixed-height rows, cut by this panel itself
// (see lib/reasoning-lines) instead of by the browser: one big `pre-wrap`
// text node gets re-wrapped on every streamed chunk, so a line the reader is
// looking at can jump to the next row when the row above it is still being
// filled. Cutting in columns makes every finished row final — its text and its
// height can never change again — and keeps each chunk's work proportional to
// the characters that just arrived.
//
// The cut width is measured, in columns, from the panel itself: monospace
// means one advance width per character, so the row count is arithmetic.
// ---------------------------------------------------------------------------
// Long probe on purpose: one advance width measured over N characters divides
// away the sub-pixel error of a single glyph, which is what keeps the cut
// width exact enough that no row ever wraps by one character too few.
const MEASURE_CHARS = 'x'.repeat(100)

// One row's worth of text for the row's own height, which the Live View's
// window measures the same way. One character, not a probe: the row's height
// is the font's line box whatever the row holds, so nothing is averaged here.
const MEASURE_ROW = 'x'

const rowBudget = ref(MIN_BUDGET)
const measure = ref(null)
let resizeObserver = null
let measureFrame = 0

function remeasure() {
  const host = measure.value
  const probe = host?.firstElementChild
  if (!host || !probe) return
  const advance = probe.getBoundingClientRect().width / probe.textContent.length
  const columns = advance > 0 ? Math.floor(host.clientWidth / advance) : 0
  const next = Math.max(MIN_BUDGET, columns)
  if (next !== rowBudget.value) rowBudget.value = next
}

function scheduleRemeasure() {
  if (measureFrame) return
  measureFrame = requestAnimationFrame(() => {
    measureFrame = 0
    remeasure()
  })
}

// Row caches live outside the reactive state: they are render cache, while
// `block.text` stays the source of truth (session saving reads it, see
// reasoningFlows). Keyed by the block, so a reloaded session starts fresh.
const rowCaches = new WeakMap()

function reasoningRows(block) {
  // Reading `block.text` here is what makes this render depend on the streamed
  // text: the panel re-renders as thinking arrives.
  const text = block.text ?? ''
  let cache = rowCaches.get(block)
  if (!cache) {
    cache = createLineCache()
    rowCaches.set(block, cache)
  }
  return displayLines(text, cache, rowBudget.value)
}

onMounted(() => {
  remeasure()
  // Observe the row host itself, not the scroller: its content box is the
  // width a reasoning row really gets, so this also fires when a scrollbar
  // appears and steals a few columns from the rows.
  resizeObserver = new ResizeObserver(scheduleRemeasure)
  if (measure.value) resizeObserver.observe(measure.value)
  // A webfont swapping in changes the advance width without changing any box,
  // so it needs its own nudge — otherwise the column count stays the fallback
  // font's, and every row is cut a few characters short.
  document.fonts?.ready?.then(scheduleRemeasure)

  // The window's own observer, on the scroller rather than on the row host:
  // the scroller's content box is the height the timeline is being asked to
  // fit into, and this is the only moment its height may change the window —
  // at rest. The first callback doubles as the initial measurement, the way
  // the splitters' observer reads one.
  if (typeof ResizeObserver !== 'undefined' && list.value) {
    windowObserver = new ResizeObserver(remeasureWindow)
    windowObserver.observe(list.value)
  }
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  resizeObserver = null
  windowObserver?.disconnect()
  windowObserver = null
  if (measureFrame) cancelAnimationFrame(measureFrame)
  measureFrame = 0
})

// Badge text: ``<tool>::<method>`` — the hub the call was routed to (MCP
// connector id, or the language-server id for `lsp_<id>_<action>` tools)
// then the callable name. Older payloads without a server fall back to the
// bare method name. LSP tool names already carry their server id
// (`lsp_vue_diagnostics`), so the redundant prefix is dropped there.
function toolLabel(tool) {
  if (!tool.server) return tool.tool
  const namespaced = `lsp_${tool.server}_`
  const method = tool.tool.startsWith(namespaced)
    ? tool.tool.slice(namespaced.length)
    : tool.tool
  return `${tool.server}::${method}`
}
</script>

<template>
  <aside
    class="chat-tool-panel"
    aria-label="Model reasoning and MCP tool calls"
  >
    <div
      ref="list"
      class="chat-tool-scroll"
      :class="{ 'chat-tool-scroll--live': liveEnabled }"
      @scroll.passive="onScroll"
    >
      <ul class="chat-tool-list">
        <!-- What the window paints, not the whole timeline (see `view` above):
             the same blocks, in the same order, keyed by `unit.id` — the block's
             place in the emission order, which never moves for a block, so a
             sliding window moves units out at one end only and the ones still
             on screen keep their nodes. A position key would rename every one
             of them on every pruned row and re-render the whole list. A
             reasoning block paints the rows of `unit.rows` — the ones lib/reasoning-lines
             cut, and the pruned flow paints the tail of them. -->
        <template v-for="unit in view.units" :key="unit.id">
          <li
            v-if="unit.block.kind === 'tool'"
            class="chat-tool"
            :class="unit.block.tool.state"
            :title="`${unit.block.tool.server ? unit.block.tool.server + ' · ' : ''}${unit.block.tool.preview}${unit.block.tool.resultPreview ? '\n→ ' + unit.block.tool.resultPreview : ''}`"
          >
            <span class="chat-tool-state" aria-hidden="true">
              {{ unit.block.tool.state === 'running' ? '…' : unit.block.tool.state === 'ok' ? '✓' : '✗' }}
            </span>
            <span class="chat-tool-name">
              {{ toolLabel(unit.block.tool) }}
            </span>
          </li>
          <li v-else class="chat-reasoning"><span
            v-for="(row, j) in unit.rows"
            :key="j"
            class="chat-reasoning-line"
          >{{ row }}</span></li>
        </template>
        <!-- Zero-height measuring row, last so it cannot shift any content: it
             sits in the list's own content box, so its width is exactly the
             width available to a reasoning row, and the probe inside gives the
             font's advance width. Both are read, never assumed. -->
        <li ref="measure" class="chat-reasoning-measure" aria-hidden="true"><span
          class="chat-reasoning-probe"
        >{{ MEASURE_CHARS }}</span><span
          class="chat-reasoning"
        ><!-- One real row, for the Live View's window: the row's height is the
              theme's to decide, and the panel cannot know it before the first
              reasoning row is painted. Clipped by this host's zero height and
              hidden by its visibility, so it costs no layout and is never seen. -->
          <span class="chat-reasoning-line">{{ MEASURE_ROW }}</span></span></li>
      </ul>
    </div>
    <GaugeBar
      class="chat-tool-gauge"
      :value="gaugeValue"
      :max="gaugeMax"
      label="Tool-loop iterations of the current reply vs MAX_TOOL_ITERATION — MCP tools and language servers share the counter; one iteration may run several calls (badges count individual calls)"
    />
  </aside>
</template>
