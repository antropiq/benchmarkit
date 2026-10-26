import { describe, expect, it } from 'vitest'

import {
  MIN_BUDGET,
  TAB_WIDTH,
  createLineCache,
  displayLines,
} from '../src/lib/reasoning-lines'

// A flow arrives one streamed chunk at a time, so the only property that
// matters is what happens to already-rendered rows as more text arrives: they
// must not move. Feed chunks, assert the row list's prefix never changes.
function stream(text, budget) {
  const cache = createLineCache()
  return displayLines(text, cache, budget)
}

describe('displayLines', () => {
  it('keeps a short flow on one row', () => {
    expect(stream('short', 10)).toEqual(['short'])
  })

  it('splits a hard line break into its own row, empty lines included', () => {
    expect(stream('one\n\ntwo', 10)).toEqual(['one', '', 'two'])
  })

  it('drops the carriage return of a CRLF flow', () => {
    expect(stream('one\r\ntwo', 10)).toEqual(['one', 'two'])
  })

  it('fills a row greedily, cutting at the last space that fits', () => {
    // 'ab cd' is 5 columns, so 'ef' cannot join it; the break lands after the
    // space that sits just past the budget, and the next row fills again.
    expect(stream('ab cd ef', 5)).toEqual(['ab ', 'cd ef'])
  })

  it('cuts a word mid-way when the row holds no space at all', () => {
    expect(stream('abcdefgh', 4)).toEqual(['abcd', 'efgh'])
  })

  it('never wraps narrower than one tab', () => {
    // A budget below TAB_WIDTH would cut a leading tab in half, so it is
    // raised to one tab instead.
    expect(stream('abcdefgh', 1)).toEqual(['abcd', 'efgh'])
  })

  it('ends a space-only row at the budget instead of overflowing', () => {
    expect(stream('     ', 4)).toEqual(['    ', ' '])
  })

  it('never emits an empty row while text remains', () => {
    // Guards the wrap loop against a no-progress cut.
    const rows = stream('x'.repeat(40), MIN_BUDGET)
    expect(rows).toHaveLength(10)
    expect(rows.every((row) => row.length > 0)).toBe(true)
  })

  it('treats a tab as the run of columns to the next tab stop', () => {
    // A leading tab reaches column TAB_WIDTH on its own, so it fills the row
    // and can never be cut in half, whatever the budget.
    expect(stream('\tab', TAB_WIDTH)).toEqual(['\t', 'ab'])
    expect(stream('\tab', 3)).toEqual(['\t', 'ab'])
    // One column already used: the tab still needs to reach the next stop, so
    // it occupies the 3 remaining columns and the word starts a fresh row.
    expect(stream('a\tbc', 4)).toEqual(['a\t', 'bc'])
  })
})

describe('incremental streaming', () => {
  it('recomputes only the row still being filled', () => {
    const cache = createLineCache()
    const first = displayLines('hello wor', cache, 7)
    expect([...first]).toEqual(['hello ', 'wor'])
    const before = [...first]
    // More text on the same source line: the finished row keeps its exact
    // text, only the trailing row grows.
    const after = displayLines('hello world!', cache, 7)
    // The already-FINISHED rows keep their exact text; only the row being
    // filled (the last one) is re-derived, and it only ever grows.
    expect(after.slice(0, before.length - 1)).toEqual(before.slice(0, -1))
    expect(after[before.length - 1].startsWith(before[before.length - 1])).toBe(
      true,
    )
    expect(after).toEqual(['hello ', 'world!'])
  })

  it('keeps every already-rendered row stable chunk after chunk', () => {
    const text = 'the quick brown fox jumps over the lazy dog, again and again.'
    const cache = createLineCache()
    const budget = 12
    let seen = []
    for (let i = 1; i <= text.length; i += 1) {
      const rows = displayLines(text.slice(0, i), cache, budget)
      // Every row but the one still being filled is frozen: it can neither
      // change text nor move to another index. That is the whole point of the
      // module — the line under the reader's eye does not jump.
      expect(rows.slice(0, seen.length - 1)).toEqual(seen.slice(0, -1))
      seen = [...rows]
    }
    expect(seen).toEqual(stream(text, budget))
  })

  it('agrees with a from-scratch cut after every chunk of a tabbed flow', () => {
    const text = 'def f():\n\treturn\t\tvalue\n\tif x:\n\t\treturn\tx\n'
    const cache = createLineCache()
    const budget = 8
    let seen = []
    for (let i = 1; i <= text.length; i += 1) {
      const rows = displayLines(text.slice(0, i), cache, budget)
      expect(rows.slice(0, seen.length - 1)).toEqual(seen.slice(0, -1))
      seen = [...rows]
    }
    expect(seen).toEqual(stream(text, budget))
  })

  it('re-wraps from scratch when the column budget changes', () => {
    const cache = createLineCache()
    displayLines('hello world', cache, 11)
    expect([...displayLines('hello world', cache, 5)]).toEqual([
      'hello',
      ' worl',
      'd',
    ])
  })

  it('resumes from the start if the flow is replaced rather than appended', () => {
    const cache = createLineCache()
    displayLines('some long previous flow', cache, 6)
    expect([...displayLines('ab', cache, 6)]).toEqual(['ab'])
  })
})
