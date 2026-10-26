<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import GaugeBar from './GaugeBar.vue'
import { useChat } from '../composables/useChat'
import { useAutoScroll } from '../composables/useAutoScroll'
import { createLineCache, displayLines, MIN_BUDGET } from '../lib/reasoning-lines'

const { state } = useChat()

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

// Width follows the input row's toggle column below (same `visible`
// conditions as WorkBenchInput.vue): two panels wide only when both the MCP and
// the LSP panel are rendered, one panel wide when just one of them is, and
// one panel wide when neither is (no column to align with).
const singlePanel = computed(
  () =>
    !(
      state.mcp.loaded &&
      state.mcp.connectors.length > 0 &&
      state.lsp.loaded &&
      state.lsp.servers.length > 0
    ),
)

const list = ref(null)
const signature = computed(() => {
  const last = blocks.value[blocks.value.length - 1]
  const tail = last ? `${last.kind}-${last.kind === 'tool' ? `${last.tool.id}-${last.tool.state}` : ''}` : ''
  return `${blocks.value.length}:${reasoningChars.value}:${tail}`
})
const { onScroll } = useAutoScroll(list, signature)

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
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  resizeObserver = null
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
    :class="{ 'chat-tool-panel--single': singlePanel }"
    aria-label="Model reasoning and MCP tool calls"
  >
    <div
      ref="list"
      class="chat-tool-scroll"
      @scroll.passive="onScroll"
    >
      <ul class="chat-tool-list">
        <template v-for="(block, i) in blocks" :key="i">
          <li
            v-if="block.kind === 'tool'"
            class="chat-tool"
            :class="block.tool.state"
            :title="`${block.tool.server ? block.tool.server + ' · ' : ''}${block.tool.preview}${block.tool.resultPreview ? '\n→ ' + block.tool.resultPreview : ''}`"
          >
            <span class="chat-tool-state" aria-hidden="true">
              {{ block.tool.state === 'running' ? '…' : block.tool.state === 'ok' ? '✓' : '✗' }}
            </span>
            <span class="chat-tool-name">
              {{ toolLabel(block.tool) }}
            </span>
          </li>
          <li v-else class="chat-reasoning"><span
            v-for="(row, j) in reasoningRows(block)"
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
        >{{ MEASURE_CHARS }}</span></li>
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
