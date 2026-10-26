// Thinking arrives as one `reasoning_delta` frame per streamed chunk: a chunk
// rate set by the network, far above the frame clock. Writing that text into
// the reactive block per chunk makes every chunk re-cut the panel's rows,
// re-patch the row list and re-run its autoscroll — which reads `scrollHeight`
// and so forces a synchronous layout — dozens of times per painted frame.
//
// This buffer defers only the DRAWING: deltas accumulate in a plain,
// non-reactive string and the reactive block is touched once per animation
// frame, so the panel's work lands on the frame boundary instead of on an
// arrival.
//
// What is deliberately NOT deferred is the measurement: the per-character
// `reasoningChars` counter and the generation window stay anchored to the chunk
// that arrived (see the `reasoning_delta` case in useChat). A `usage` frame
// snapshots an exact baseline built from that counter, so deferring the counter
// would count the buffered tail a second time when estimating what was
// generated after that baseline.
//
// Chronology rule, now enforced here because the write happens later than the
// frame that requested it: consecutive thinking extends the current reasoning
// block, and anything else appended in between (a tool-call badge, written
// synchronously by its own frame) starts a new one. Reading the block back off
// `blocks` at flush time is what keeps that order exact — and it is also what
// makes the write land on the reactive proxy rather than on the raw object
// pushed earlier, without which the panel would never see the text at all.

// The one write: append to the flowing block, or open a new one when the
// timeline's tail is not thinking (start of the reply, or a tool round between
// two thinking runs).
export function appendReasoning(blocks, text) {
  const last = blocks[blocks.length - 1]
  if (last && last.kind === 'reasoning') last.text += text
  else blocks.push({ kind: 'reasoning', text })
}

// `schedule`/`unschedule` default to the browser's frame clock and are
// injectable so the coalescing itself is testable without one.
export function createReasoningBuffer({
  schedule = (fn) => requestAnimationFrame(fn),
  unschedule = (id) => cancelAnimationFrame(id),
} = {}) {
  // Keyed by the reply's `blocks` array, which keeps its identity through the
  // reactive proxy, so a buffered turn cannot be confused with the next one.
  const pending = new Map()
  let frame = 0

  function applyAll() {
    if (!pending.size) return
    for (const [blocks, text] of pending) appendReasoning(blocks, text)
    pending.clear()
  }

  function append(blocks, text) {
    const held = pending.get(blocks)
    pending.set(blocks, held ? held + text : text)
    if (!frame) {
      frame = schedule(() => {
        frame = 0
        applyAll()
      })
    }
  }

  // Draws everything held, now. Called when a reply ends: its thinking must
  // already be in its blocks before anything reads them — session saving and
  // the payload of the next send both do, and neither waits for a frame.
  function flush() {
    if (frame) {
      unschedule(frame)
      frame = 0
    }
    applyAll()
  }

  return {
    append,
    flush,
    // Characters still awaiting a frame. Only the tests read it: it is what
    // lets them assert coalescing happened without a frame clock.
    get pendingChars() {
      let total = 0
      for (const text of pending.values()) total += text.length
      return total
    },
  }
}
