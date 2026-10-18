import DOMPurify from 'dompurify'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import { markedHighlight } from 'marked-highlight'
import { highlight } from './hljs'
import { encodeSource } from './mermaid'
import { katexBrackets } from './markedKatexBrackets'

marked.setOptions({ breaks: true, gfm: true })
marked.use(markedKatex({ throwOnError: false, output: 'html' }))
marked.use(katexBrackets({ throwOnError: false }))
marked.use(
  markedHighlight({
    langPrefix: 'hljs language-',
    emptyLangClass: 'hljs',
    highlight: (code, lang) => highlight(code, lang),
  }),
)

// Only *closed* ```mermaid fences become diagram placeholders; an
// unterminated fence (still streaming) falls through to the normal
// highlighted code block and becomes a diagram on the next render.
const MERMAID_FENCE = /^[ \t]{0,3}```mermaid[ \t]*\r?\n([\s\S]*?)^[ \t]{0,3}```[ \t]*$/gm

function mermaidPlaceholders(raw) {
  return raw.replace(
    MERMAID_FENCE,
    (_match, source) =>
      `<div class="mermaid-block" data-src="${encodeSource(source)}"></div>\n`,
  )
}

export function renderMarkdown(raw, { mermaid = false } = {}) {
  const source = raw || ''
  const text = mermaid ? mermaidPlaceholders(source) : source
  return DOMPurify.sanitize(marked.parse(text))
}
