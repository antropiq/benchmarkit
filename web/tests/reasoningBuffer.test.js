import { beforeEach, describe, expect, it } from 'vitest'

import { appendReasoning, createReasoningBuffer } from '../src/lib/reasoning-buffer'

// Layer 1 (jsdom): the coalescing rule itself, with the frame clock replaced by
// a manual one — what matters is *how many times* the reactive timeline is
// touched, and in which order, not when a frame happens to fire.

function fakeClock() {
  const queue = []
  let issued = 0
  return {
    schedule: (fn) => {
      queue.push(fn)
      issued += 1
      return issued
    },
    unschedule: () => {
      queue.length = 0
    },
    // Runs every queued callback, as one frame would. Returns how many did.
    run: () => {
      const fns = queue.splice(0, queue.length)
      for (const fn of fns) fn()
      return fns.length
    },
  }
}

function textOf(blocks) {
  return blocks.map((block) => `${block.kind}:${block.text ?? ''}`)
}

describe('appendReasoning', () => {
  it('opens a flow block for the first thinking text', () => {
    const blocks = []
    appendReasoning(blocks, 'think')
    expect(textOf(blocks)).toEqual(['reasoning:think'])
  })

  it('extends the flowing block instead of opening a new one', () => {
    const blocks = []
    appendReasoning(blocks, 'a')
    appendReasoning(blocks, 'b')
    expect(textOf(blocks)).toEqual(['reasoning:ab'])
  })

  it('starts a new flow after a tool badge, keeping the badges in place', () => {
    const blocks = [{ kind: 'reasoning', text: 'first' }]
    blocks.push({ kind: 'tool', tool: { tool: 'x' } })
    appendReasoning(blocks, 'second')
    expect(textOf(blocks)).toEqual(['reasoning:first', 'tool:', 'reasoning:second'])
  })
})

describe('createReasoningBuffer', () => {
  let clock
  let buffer

  beforeEach(() => {
    clock = fakeClock()
    buffer = createReasoningBuffer({
      schedule: clock.schedule,
      unschedule: clock.unschedule,
    })
  })

  it('holds every delta until one frame writes them all', () => {
    const blocks = []
    buffer.append(blocks, 'o')
    buffer.append(blocks, 'ne ')
    buffer.append(blocks, 'two')
    expect(buffer.pendingChars).toBe(7)
    expect(blocks).toEqual([]) // nothing drawn yet: this is the whole point
    expect(clock.run()).toBe(1) // one frame scheduled for three deltas
    expect(textOf(blocks)).toEqual(['reasoning:one two'])
    expect(buffer.pendingChars).toBe(0)
  })

  it('flush draws the tail before a badge written after it', () => {
    // A tool_call frame is synchronous, so a delta still buffered when it
    // arrives belongs EARLIER in the timeline.
    const blocks = []
    buffer.append(blocks, 'thought')
    buffer.flush()
    blocks.push({ kind: 'tool', tool: { tool: 'x' } })
    buffer.append(blocks, 'after')
    buffer.flush()
    expect(textOf(blocks)).toEqual(['reasoning:thought', 'tool:', 'reasoning:after'])
  })

  it('keeps two turns apart', () => {
    const first = []
    const second = []
    buffer.append(first, 'one')
    buffer.append(second, 'two')
    buffer.flush()
    expect(textOf(first)).toEqual(['reasoning:one'])
    expect(textOf(second)).toEqual(['reasoning:two'])
  })
})
