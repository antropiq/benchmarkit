import { describe, expect, it } from 'vitest'

import { renderMarkdown } from '../src/lib/markdown'

// The renderer is the one seam every markdown surface paints through,
// and the one place the browser can be saved from a chat-template
// control token: fed to innerHTML, a wrapped run becomes a bogus element
// that swallows the text after it, and the panel reads as if the reply
// stopped. The delimiters here are built from characters so this file
// never spells a complete run either (the mirror on the Python side is
// tests/test_llmkit_sanitize.py; the guard itself is neutralizeControl-
// TokenRuns in src/lib/markdown.js).

const PIPE = '|'
const run = (name) => `<${PIPE}${name}${PIPE}>`

describe('renderMarkdown control-token neutralisation', () => {
  it('removes a wrapped run from the rendered output', () => {
    const html = renderMarkdown(`it is written ${run('evil')} in ChatML`)
    expect(html).not.toContain(run('evil'))
    expect(html).toContain('it is written')
    expect(html).toContain('in ChatML')
  })

  it('removes a run that only forms after an inner one is gone', () => {
    const spliced = `<${PIPE}im_${run('x')}start${PIPE}>`
    const html = renderMarkdown(`before ${spliced} after`)
    expect(html).not.toContain(run('im_start'))
    expect(html).not.toContain(run('x'))
    expect(html).toContain('before')
    expect(html).toContain('after')
  })

  it('removes runs from fenced code as well', () => {
    const html = renderMarkdown('```\n' + run('evil') + ' ok\n```')
    expect(html).not.toContain(run('evil'))
    expect(html).toContain('ok')
  })

  it('leaves benign angle brackets and pipes alone', () => {
    const html = renderMarkdown('2 < 3 and a|b and x<y|z>')
    expect(html).toContain('2 &lt; 3')
    expect(html).toContain('a|b')
    expect(html).toContain('x&lt;y|z&gt;')
  })
})
