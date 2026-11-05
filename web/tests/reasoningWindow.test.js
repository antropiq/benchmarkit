import { describe, expect, it } from 'vitest'

import { boxCapacity, selectWindow } from '../src/lib/reasoning-window'

// The Live View's memory rule, asserted where it is one decision (see the
// module's own header). The unit of the budget is a DISPLAY ROW and units come
// already cut by lib/reasoning-lines.js, so what this file pins down is which
// units the panel paints and which it leaves out — never how text is wrapped.
//
// A unit is `{ id, block, rows, cost }`, with `rows: null` for a tool badge:
// a badge has no rows to show, is one row of height, and cannot be cut.
function flow(id, rows) {
  return { id, block: { kind: 'reasoning' }, rows, cost: rows.length }
}

function badge(id) {
  return { id, block: { kind: 'tool' }, rows: null, cost: 1 }
}

// `[rows, dropped]`: the rows the window paints, in order, and the rows of
// reasoning it left out.
function painted(result) {
  return [result.units.flatMap((unit) => unit.rows ?? []), result.dropped]
}

describe('boxCapacity', () => {
  // The height of one reasoning row and the gap the list puts between BLOCKS,
  // as the theme has them (`.chat-reasoning-line` is 0.8rem x 1.45, and
  // `--bi-chat-tool-gap` is 0.35rem at the 16px root). Named here because the
  // bug this pins was a divisor: the row's height is what a row costs, and the
  // gap belongs to the blocks and their `cost`, never to the row's height.
  const ROW = 18.56 // 0.8rem x line-height 1.45
  const GAP = 5.6 // 0.35rem

  it('divides the box by the row height, not by row plus gap', () => {
    expect(boxCapacity({ inside: 340, rowHeight: ROW })).toBe(
      Math.floor(340 / ROW),
    )
    // The old answer: charging every row for a gap it never gets, which left
    // 80px — four rows — of blank panel under the last line of the flow.
    expect(boxCapacity({ inside: 340, rowHeight: ROW + GAP })).toBe(14)
    expect(boxCapacity({ inside: 340, rowHeight: ROW })).toBeGreaterThan(14)
  })

  it('fills a box whose height is an exact number of rows', () => {
    expect(boxCapacity({ inside: 4 * ROW, rowHeight: ROW })).toBe(4)
  })

  it('leaves the remainder it cannot show, never an extra row', () => {
    // One row of slack under the box's height must not become a row of
    // content taller than the box.
    expect(boxCapacity({ inside: 4 * ROW + 1, rowHeight: ROW })).toBe(4)
  })

  it('is 0 for a box or a row that has no height', () => {
    expect(boxCapacity({ inside: 0, rowHeight: ROW })).toBe(0)
    expect(boxCapacity({ inside: -12, rowHeight: ROW })).toBe(0)
    expect(boxCapacity({ inside: 340, rowHeight: 0 })).toBe(0)
    expect(boxCapacity({ inside: 340, rowHeight: NaN })).toBe(0)
  })
})

describe('selectWindow', () => {
  it('paints everything when no window was measured (Infinity)', () => {
    const all = [flow('a', ['one', 'two', 'three'])]
    expect(selectWindow(all, Infinity)).toEqual({ units: all, dropped: 0 })
  })

  it('paints everything when the capacity already fits it', () => {
    // A box taller than the timeline: the mode is on and still costs nothing,
    // which is what keeps the mode honest on a short stream.
    const all = [flow('a', ['one', 'two']), badge('b')]
    expect(selectWindow(all, 9)).toEqual({ units: all, dropped: 0 })
  })

  it('leaves the mode off for a capacity that is not positive', () => {
    const all = [flow('a', ['one'])]
    expect(selectWindow(all, 0)).toEqual({ units: all, dropped: 0 })
    expect(selectWindow(all, -3)).toEqual({ units: all, dropped: 0 })
  })

  it('keeps the newest rows and drops the oldest of one long flow', () => {
    const long = flow('a', ['r1', 'r2', 'r3', 'r4', 'r5'])
    const [rows, dropped] = painted(selectWindow([long], 3))
    expect(rows).toEqual(['r3', 'r4', 'r5'])
    expect(dropped).toBe(2)
  })

  it('never empties the panel with a flow taller than the box', () => {
    // A window of one row onto a five-row flow paints that one newest row,
    // never nothing: the line being read has to stay on screen.
    const long = flow('a', ['r1', 'r2', 'r3', 'r4', 'r5'])
    const [rows, dropped] = painted(selectWindow([long], 1))
    expect(rows).toEqual(['r5'])
    expect(dropped).toBe(4)
  })

  it('spends the budget in reading order, oldest first out', () => {
    const timeline = [
      flow('old', ['o1', 'o2']),
      flow('mid', ['m1', 'm2']),
      flow('new', ['n1', 'n2']),
    ]
    const [rows, dropped] = painted(selectWindow(timeline, 3))
    // The newest flow whole, the one before it cut to its tail, the oldest
    // gone: the edge lands inside one flow, never inside a row. Three rows
    // left the screen — the oldest flow's two and the cut flow's head.
    expect(rows).toEqual(['m2', 'n1', 'n2'])
    expect(dropped).toBe(3)
  })

  it('cuts the oldest flow on screen and stops the budget there', () => {
    // A flow is the one unit that can be cut, so it always shows its tail: the
    // thinking being read stays whole and only its head leaves the screen.
    // Once the budget has been spent on that cut, the units older than it are
    // gone too — a badge is never resurrected to fill the top of the box.
    const timeline = [
      badge('a'),
      flow('old', ['o1', 'o2', 'o3']),
      flow('new', ['n1']),
    ]
    const [rows, dropped] = painted(selectWindow(timeline, 2))
    expect(rows).toEqual(['o3', 'n1'])
    expect(dropped).toBe(2)
  })

  it('paints everything a capacity fits exactly', () => {
    // A capacity equal to the timeline's cost is not a window: nothing had to
    // be left out, and this is the boundary above which the skip rule matters.
    const timeline = [flow('old', ['o1', 'o2']), flow('new', ['n1'])]
    expect(selectWindow(timeline, 3).dropped).toBe(0)
  })

  it('shows a badge whole or not at all', () => {
    const timeline = [badge('a'), badge('b'), flow('c', ['r1'])]
    const [rows, dropped] = painted(selectWindow(timeline, 2))
    // The budget is rows: the flow costs one, the badge before it one, the
    // oldest badge is out — and no reasoning line was lost, so no notice.
    expect(rows).toEqual(['r1'])
    expect(dropped).toBe(0)
    expect(selectWindow(timeline, 2).units.map((unit) => unit.id)).toEqual(['b', 'c'])
  })

  it('counts the pruned thinking, never the badges', () => {
    const timeline = [badge('a'), flow('c', ['r1', 'r2', 'r3'])]
    expect(selectWindow(timeline, 1).dropped).toBe(2)
    expect(selectWindow([badge('a')], 0).dropped).toBe(0)
  })

  it('takes a window of an empty timeline as costing nothing', () => {
    expect(selectWindow([], 4)).toEqual({ units: [], dropped: 0 })
  })

  it('keeps an empty flow harmless', () => {
    // A flow whose text has not arrived yet costs no row: it must not eat the
    // budget and must not leave the window stranded on it.
    const timeline = [flow('empty', []), flow('new', ['n1', 'n2'])]
    const [rows, dropped] = painted(selectWindow(timeline, 2))
    expect(rows).toEqual(['n1', 'n2'])
    expect(dropped).toBe(0)
  })

  it('never re-cuts, only slices', () => {
    // The cost claim of the mode: the rows kept are a suffix of the rows the
    // flow already cut, and a flow the window did not touch is the very same
    // object — an identity the list patches by, so the flows still on screen
    // do not re-render when the window slides.
    const rows = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
    const long = flow('a', rows)
    const timeline = [flow('old', ['o1']), long]
    const kept = selectWindow(timeline, 2).units
    expect(kept.map((unit) => unit.id)).toEqual(['a'])
    expect(kept[0].rows).toEqual(rows.slice(4))
    const whole = selectWindow([long, flow('new', ['n1'])], 7).units
    expect(whole[0]).toBe(long)
  })
})
