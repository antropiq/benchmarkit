import { describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { clampReveal } from '../src/lib/monaco-surface'

// The clamp is the half of the search reveal that is testable without Monaco:
// `MonacoEditor.vue` cannot be driven in jsdom (see `probe-monaco.test.js` and
// the coverage exclusion in vitest.config.js), so the queue's *decision* —
// which position, if any, gets handed to `revealLineInCenter`/`setPosition` —
// lives here as a pure function over the two facts a model can answer: the line
// count and a line's max column.
//
// A stale hit (the file changed since the scan) must clamp rather than throw:
// Monaco's own behaviour for an out-of-range position is not part of the
// contract, and a silent clamp beats an exception inside a click handler.

// A 3-line model; every line is 10 chars wide, so `getLineMaxColumn` is 11.
const LINE_COUNT = 3
const maxColumnFor = () => 11

function clamp(position) {
  return clampReveal(position, LINE_COUNT, maxColumnFor)
}

describe('clampReveal', () => {
  it('passes a position inside the file through unchanged', () => {
    expect(clamp({ line: 2, column: 4 })).toEqual({ line: 2, column: 4 })
    expect(clamp({ line: 1, column: 1 })).toEqual({ line: 1, column: 1 })
    expect(clamp({ line: 3, column: 11 })).toEqual({ line: 3, column: 11 })
  })

  it('clamps a line past the end to the last line', () => {
    expect(clamp({ line: 999, column: 1 })).toEqual({ line: 3, column: 1 })
  })

  it('clamps a column past the line end to that line max column', () => {
    expect(clamp({ line: 1, column: 400 })).toEqual({ line: 1, column: 11 })
    // …and it asks about the *clamped* line, not the one the hit named.
    const spy = vi.fn((line) => (line === 3 ? 5 : 11))
    expect(clampReveal({ line: 9, column: 400 }, LINE_COUNT, spy)).toEqual({
      line: 3,
      column: 5,
    })
    expect(spy).toHaveBeenCalledWith(3)
  })

  it('keeps an empty single-line model addressable', () => {
    // A buffer Monaco reports as 1 line with max column 1: a hit from a scan
    // of a file that has since been emptied still lands, at (1, 1).
    expect(clampReveal({ line: 4, column: 7 }, 1, () => 1)).toEqual({ line: 1, column: 1 })
  })

  it.each([
    ['null', null],
    ['undefined', undefined],
    ['a string', '12:4'],
    ['a missing column', { line: 2 }],
    ['a zero line', { line: 0, column: 1 }],
    ['a negative line', { line: -3, column: 1 }],
    ['a zero column', { line: 2, column: 0 }],
    ['a negative column', { line: 2, column: -1 }],
    ['a fractional line', { line: 2.5, column: 1 }],
    ['a fractional column', { line: 2, column: 1.5 }],
    ['a NaN line', { line: NaN, column: 1 }],
    ['a string line', { line: '2', column: 1 }],
  ])('rejects %s as null', (_label, position) => {
    expect(clamp(position)).toBeNull()
  })
})

// The rule that puts the find widget's buttons back on Monaco's icon font,
// asserted on the stylesheet's source. What the *browser* decides (which family
// a button ends up drawing with) is the e2e layer's; what belongs here is the
// shape of the fix, because both halves of it are easy to undo by accident: the
// selector has to reach the buttons Monaco's own widget rule reaches (its
// specificity is what beat the icon font's `font` shorthand), and the rule has
// to name the face by its real family. A weaker selector silently stops applying
// the moment the widget's markup shifts, and then the buttons are boxes again.
describe('the find widget icon rule', () => {
  const css = readFileSync(
    join(resolve(), 'src', 'styles', 'monaco-pico.css'),
    'utf8',
  )

  it('names the codicon family on the find widget buttons', () => {
    const rules = css.match(/\.monaco-editor \.find-widget[^{]*\{[^}]*\}/g) ?? []
    expect(rules.length, 'the widget buttons carry a rule of their own').toBeGreaterThan(0)
    for (const rule of rules) {
      expect(rule).toMatch(/font-family:\s*codicon\s*;/)
    }
    // Both kinds of mark the widget builds: the action buttons and the
    // in-widget toggle whose class is the icon's own name.
    expect(css).toMatch(/\.monaco-editor \.find-widget \.button,/)
    expect(css).toMatch(/\.monaco-editor \.find-widget \.codicon-find-selection/)
  })

  it('restates only the family, leaving the widget its own size and colours', () => {
    // A `font` shorthand here would reset the size Monaco authored for the
    // widget's buttons — the family is the only property this fix owns.
    const bodies = css.match(/\.monaco-editor \.find-widget[^{]*\{([^}]*)\}/g) ?? []
    for (const body of bodies) {
      expect(body.match(/[^;\n]+;/g).map((line) => line.trim())).toEqual([
        'font-family: codicon;',
      ])
    }
  })
})
