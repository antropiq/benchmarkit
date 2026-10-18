import DOMPurify from 'dompurify'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import { markedHighlight } from 'marked-highlight'
import { highlight } from './hljs'
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

export function renderMarkdown(raw) {
  return DOMPurify.sanitize(marked.parse(raw || ''))
}
