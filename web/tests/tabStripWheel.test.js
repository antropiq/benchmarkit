import { describe, expect, it, vi } from 'vitest'

import { scrollStripWithWheel, wheelScrollDelta } from '../src/lib/tab-strip-wheel'

// jsdom lays nothing out, so the strip's overflow is stated directly: a plain
// object carrying just what the helper reads.
function strip({ scrollWidth, clientWidth, scrollLeft = 0 }) {
  return { scrollWidth, clientWidth, scrollLeft }
}

function wheel({ deltaX = 0, deltaY = 0, deltaMode = 0 }) {
  return { deltaX, deltaY, deltaMode, preventDefault: vi.fn() }
}

describe('wheelScrollDelta', () => {
  it('scrolls sideways with a vertical notch once the tabs overflow', () => {
    const stripEl = strip({ scrollWidth: 1000, clientWidth: 400 })
    expect(wheelScrollDelta(stripEl, wheel({ deltaY: 120 }))).toBe(120)
  })

  it('leaves the event alone while every tab fits', () => {
    const stripEl = strip({ scrollWidth: 400, clientWidth: 400 })
    expect(wheelScrollDelta(stripEl, wheel({ deltaY: 120 }))).toBeNull()
  })

  it('prefers the horizontal axis of a diagonal trackpad gesture', () => {
    const stripEl = strip({ scrollWidth: 1000, clientWidth: 400 })
    expect(wheelScrollDelta(stripEl, wheel({ deltaX: -80, deltaY: 10 }))).toBe(-80)
  })

  it('scales line-mode deltas (Firefox) instead of reading lines as pixels', () => {
    const stripEl = strip({ scrollWidth: 1000, clientWidth: 400 })
    expect(wheelScrollDelta(stripEl, wheel({ deltaY: 3, deltaMode: 1 }))).toBe(48)
  })

  it('scales page-mode deltas by the strip width', () => {
    const stripEl = strip({ scrollWidth: 1000, clientWidth: 400 })
    expect(wheelScrollDelta(stripEl, wheel({ deltaY: 1, deltaMode: 2 }))).toBe(400)
  })

  it('ignores an empty notch', () => {
    const stripEl = strip({ scrollWidth: 1000, clientWidth: 400 })
    expect(wheelScrollDelta(stripEl, wheel({}))).toBeNull()
  })

  it('survives a missing strip (the pane is not mounted yet)', () => {
    expect(wheelScrollDelta(null, wheel({ deltaY: 120 }))).toBeNull()
  })
})

describe('scrollStripWithWheel', () => {
  it('moves scrollLeft and consumes the event', () => {
    const stripEl = strip({ scrollWidth: 1000, clientWidth: 400, scrollLeft: 30 })
    const event = wheel({ deltaY: 120 })
    scrollStripWithWheel(stripEl, event)
    expect(stripEl.scrollLeft).toBe(150)
    expect(event.preventDefault).toHaveBeenCalledTimes(1)
  })

  it('consumes nothing when the strip does not overflow', () => {
    const stripEl = strip({ scrollWidth: 400, clientWidth: 400 })
    const event = wheel({ deltaY: 120 })
    scrollStripWithWheel(stripEl, event)
    expect(stripEl.scrollLeft).toBe(0)
    expect(event.preventDefault).not.toHaveBeenCalled()
  })
})
