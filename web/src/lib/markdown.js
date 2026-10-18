import DOMPurify from 'dompurify'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import { katexBrackets } from './markedKatexBrackets'

marked.setOptions({ breaks: true, gfm: true })
marked.use(markedKatex({ throwOnError: false, output: 'html' }))
marked.use(katexBrackets({ throwOnError: false }))

export function renderMarkdown(raw) {
  return DOMPurify.sanitize(marked.parse(raw || ''))
}
