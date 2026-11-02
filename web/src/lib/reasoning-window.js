// The Live View of the reasoning panel (spec `docs/spec-live-reasoning-view.md`
// §3.3): the panel keeps only what its box can hold, and what scrolled off
// above the top leaves the DOM. This is that one decision, as a pure function,
// so the rule can be read and tested without a browser, a stream or a layout —
// what the panel adds around it is only the box's height in rows and the
// painting of the result.
//
// The unit of the budget is a DISPLAY ROW, and units arrive already cut by
// lib/reasoning-lines.js: this never re-cuts, it only ever takes a slice of a
// row list that is already final. Cutting here would make the window a second
// description of the same text, able to drift from the rows the panel paints.
//
// The window is spent from the newest unit backwards, because a flow scrolls
// out of its oldest rows: the newest thinking is therefore always whole, and
// only the oldest unit the window still shows has its head left out.
//
// The two kinds of unit are cut differently on purpose. A flow always shows
// its tail, because the thinking being read is the part in the reader's eyes
// and an empty panel is worse than a shortened one. A badge is whole or left
// out: it is one call, and half a call is not a thing to show.
//
// A capacity that the list cannot exceed — `Infinity`, which is how the panel
// spells "the mode is off" — answers with the list itself, unpainted.
//
// Where that capacity comes from: the panel reads a box's height and the height
// of one row off its own elements, and this turns them into the number of rows
// the box may be filled with. The divisor is the row's own height, because the
// rows of one flow are stacked with no gap between them — the list's gap sits
// between BLOCKS, so it belongs to the units and their cost, never to the
// height of a row. Dividing by `row + gap` instead made every row pay for a gap
// it never gets and left the bottom of the box blank: a 340px box with 16px rows
// and a 5px gap answered 15 rows, 244px of the 340 it was handed.
//
// What the answer does not charge for is a badge's extra height and the gaps the
// block boundaries add. That is deliberate, and it is the only direction that
// can be wrong safely: `useAutoScroll` pins the box's scroll to its bottom, so
// what does not fit is clipped at the TOP — the end the mode discards anyway —
// while a capacity that is too small shows as a strip of blank panel under the
// last line of the flow being read.
export function boxCapacity({ inside, rowHeight }) {
  if (!(inside > 0) || !(rowHeight > 0)) return 0
  return Math.floor(inside / rowHeight)
}

export function selectWindow(units, capacity) {
  if (!(capacity > 0)) return { units, dropped: 0 }

  let left = capacity
  if (left >= units.reduce((sum, unit) => sum + unit.cost, 0)) {
    return { units, dropped: 0 }
  }
  const kept = []
  for (let i = units.length - 1; i >= 0; i -= 1) {
    if (left <= 0) break
    const unit = units[i]
    // A badge is whole or left out: it is one row of height and there is no
    // half a call to show. A flow, in contrast, always shows its tail, because
    // the thinking that just arrived is the part being read — which is also
    // what keeps a flow taller than the box from emptying the panel.
    if (unit.rows === null) {
      if (unit.cost > left) continue
      kept.unshift(unit)
      left -= unit.cost
      continue
    }
    // A flow always shows its tail. The break below is what makes the cut
    // happen at most once, so the units older than the cut flow are out as a
    // whole and the window is one contiguous run of the timeline.
    const rows = unit.rows.length > left ? unit.rows.slice(unit.rows.length - left) : unit.rows
    kept.unshift(rows.length === unit.rows.length ? unit : { ...unit, rows })
    left -= unit.cost
    // A flow took part of the budget, so the budget is spent: older flows are
    // off the top of the box.
    if (rows.length < unit.rows.length) break
  }

  // Only the rows of reasoning count as pruned: a badge left out is a call the
  // panel no longer shows, not a line of thinking the user lost, and §3.4's
  // notice is about the thinking.
  const rows = (list) =>
    list.reduce((sum, unit) => sum + (unit.rows === null ? 0 : unit.rows.length), 0)
  return { units: kept, dropped: rows(units) - rows(kept) }
}

