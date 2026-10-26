import { describe, expect, it } from 'vitest'

window.matchMedia =
  window.matchMedia ||
  ((query) => ({
    matches: false,
    media: query,
    addEventListener() {},
    removeEventListener() {},
  }))
import * as monaco from 'monaco-editor'

describe('probe', () => {
  it('tokenizes a registered monarch grammar', () => {
    monaco.languages.register({ id: 'tomlprobe' })
    monaco.languages.setMonarchTokensProvider('tomlprobe', {
      tokenizer: { root: [[/foo/, 'keyword']] },
    })
    console.log(JSON.stringify(monaco.editor.tokenize('foo bar', 'tomlprobe')))
  })
})
