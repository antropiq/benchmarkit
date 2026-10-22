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

// Sentinel injected by useChat right after an MCP tool_result: the next
// round's streamed answer opens with it so each post-tool continuation
// starts on its own line, rendered as a small inline tool icon (the
// replacement of the former "> " blockquote prefix). It is a Private Use
// Area character no model output or user text ever contains.
export const TOOL_LINE_MARKER = '\uE000'

// Inline SVG (lucide "wrench"), one line so marked passes it through as
// raw inline HTML; every attribute is in DOMPurify's default allow-list.
const TOOL_LINE_ICON =
  '<span class="tool-line-icon" aria-hidden="true">' +
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
  '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>' +
  '</svg></span>'

function toolLineIcons(raw) {
  return raw.split(TOOL_LINE_MARKER).join(TOOL_LINE_ICON)
}

export function renderMarkdown(raw, { mermaid = false } = {}) {
  const source = toolLineIcons(raw || '')
  const text = mermaid ? mermaidPlaceholders(source) : source
  return DOMPurify.sanitize(marked.parse(text))
}
