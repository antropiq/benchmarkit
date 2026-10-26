// The WorkBench reasoning panel shows streamed thinking as a chronology of
// fixed-height rows. Browsers would re-wrap one big `pre-wrap` text node on
// every chunk, so this module does the wrapping itself, in columns, and can do
// it INCREMENTALLY: a row is final once the text past it has arrived, and only
// the row currently being filled is ever re-derived.
//
// Column arithmetic (not pixels) is what makes that possible: the panel is
// monospace, so one column has one advance width and a row has one fixed
// height — which is also what lets the panel scroll by arithmetic instead of
// by measuring the DOM.
//
// Prefix-stability is the contract: cutting row k uses only the characters
// between the start of row k and its own budget, so appending text to a flow
// can never move a break that has already been made. A row the reader's eye is
// on therefore stays put.

// A tab advances to the next multiple of this many columns, matching the
// `tab-size` the panel sets. Counting a tab as one character would drift the
// budget, and drift would break the fixed-height arithmetic.
export const TAB_WIDTH = 4

// Never wrap narrower than one tab, and it is what guarantees progress in the
// wrap loop (a first character wider than the budget would cut nothing).
export const MIN_BUDGET = TAB_WIDTH

// Per-flow resume state. Kept out of the reactive message state on purpose:
// it is render cache, while `block.text` stays the single source of truth for
// saving and reloading a session.
export function createLineCache() {
  return { budget: 0, offset: 0, lines: [], provisional: false }
}

// Columns a character at column `col` occupies: 1, or for a tab the distance to
// the next tab stop — the way `tab-size` lays them out, measured from the start
// of the display row it is rendered in.
function columnWidth(ch, col) {
  return ch === '\t' ? TAB_WIDTH - (col % TAB_WIDTH) : 1
}

// How far a row starting at `from` reaches inside [from, lineEnd), plus the
// last position where a break could be taken (just after a space or tab, which
// CSS-style wrapping consumes).
function fillRow(text, from, lineEnd, budget) {
  let col = 0
  let end = from
  let breakAt = -1
  while (end < lineEnd) {
    const width = columnWidth(text[end], col)
    if (col + width > budget) break
    col += width
    end += 1
    if (text[end - 1] === ' ' || text[end - 1] === '\t') breakAt = end
  }
  return { end, breakAt }
}

// True when [from, to) holds no visible character — the case where taking the
// space as the cut would render a row of nothing but whitespace.
function isBlankRun(text, from, to) {
  for (let i = from; i < to; i += 1) {
    if (text[i] !== ' ' && text[i] !== '\t') return false
  }
  return true
}

// The display rows of one reasoning flow, computed incrementally.
//
// `text` is the whole flow as received so far (it only ever grows), `cache` is
// the flow's resume state, `budget` the row width in columns. Returns the rows,
// in order, without their line breaks.
//
// A row fills greedily to the budget, then breaks at the last space it holds
// (that space ends the row, as wrapping consumes it) — or mid-word when it
// holds none, the way `overflow-wrap: anywhere` would. A hard line break in the
// model's output always ends a row.
//
// A space-only break is skipped rather than taken: a row of blank characters
// would waste a fixed-height row and read as a gap in the text.
export function displayLines(text, cache, budget) {
  const width = Math.max(MIN_BUDGET, Math.floor(budget) || MIN_BUDGET)
  if (cache.budget !== width) {
    // Re-wrap from scratch: a new width invalidates every previous cut.
    cache.budget = width
    cache.offset = 0
    cache.provisional = false
    cache.lines.length = 0
  } else if (cache.provisional) {
    // Drop the row that was still being filled; it is re-derived below.
    cache.lines.pop()
    cache.provisional = false
  }
  // Defensive: if the flow ever shrinks, it was replaced rather than appended
  // to, so nothing already cut still describes it — start over.
  if (text.length < cache.offset) {
    cache.offset = 0
    cache.lines.length = 0
  }

  let i = cache.offset
  while (i < text.length) {
    const newline = text.indexOf('\n', i)
    let lineEnd = newline === -1 ? text.length : newline
    if (lineEnd > i && text[lineEnd - 1] === '\r') lineEnd -= 1 // CRLF

    let rowStart = i
    let p = i
    for (;;) {
      rowStart = p
      const { end, breakAt } = fillRow(text, p, lineEnd, width)
      if (end >= lineEnd) {
        cache.lines.push(text.slice(p, end))
        p = end
        break
      }
      // Mid-word cut when the row holds no break at all, so a long
      // unbreakable run still wraps instead of overflowing.
      const atSpace = breakAt > p && !isBlankRun(text, p, breakAt)
      const cut = atSpace ? breakAt : Math.max(end, p + 1)
      cache.lines.push(text.slice(p, cut))
      p = cut
    }

    if (newline === -1) {
      // This was the flow's last source line and it is still streaming: the
      // row that was just emitted can still grow, so remember where it began
      // and re-derive it on the next call. Nothing else of this flow is left
      // to cut, so the scan ends here — resuming at `rowStart` would re-run
      // the same line forever.
      cache.offset = rowStart
      cache.provisional = true
      break
    }
    cache.offset = newline + 1
    cache.provisional = false
    i = cache.offset
  }
  return cache.lines
}
