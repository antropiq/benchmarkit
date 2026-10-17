import katex from 'katex'

// marked-katex-extension only understands $ and $$ delimiters. LLMs frequently
// emit LaTeX with \[ ... \] (display) and \( ... \) (inline) instead, which
// plain markdown renders as escaped brackets eating the backslashes. This
// companion extension teaches marked those delimiters too. Unclosed
// delimiters (mid-stream) simply never match and stay raw text.
const blockRule = /^ {0,3}\\\[(((?:[^\n]|\n(?!\s*\n))+?))\\\][ \t]*(?:\n|$)/
const inlineParenRule = /^\\\(((?:\\.|[^\\])+?)\\\)/
const inlineBracketRule = /^\\\[(((?:\\.|[^\]])+?))\\\]/

function katexHtml(text, displayMode, options) {
  return katex.renderToString(text, {
    throwOnError: false,
    output: 'html',
    displayMode,
    ...options,
  })
}

export function katexBrackets(options = {}) {
  return {
    extensions: [
      {
        name: 'katexBracketBlock',
        level: 'block',
        tokenizer(src) {
          const match = src.match(blockRule)
          if (!match) return undefined
          return {
            type: 'katexBracketBlock',
            raw: match[0],
            text: match[1].trim(),
            displayMode: true,
          }
        },
        renderer(token) {
          return katexHtml(token.text, true, options) + '\n'
        },
      },
      {
        name: 'katexBracketInline',
        level: 'inline',
        start(src) {
          const parens = src.indexOf('\\(')
          const brackets = src.indexOf('\\[')
          const found = [parens, brackets].filter((i) => i !== -1)
          return found.length ? Math.min(...found) : -1
        },
        tokenizer(src) {
          const paren = src.match(inlineParenRule)
          if (paren) {
            return {
              type: 'katexBracketInline',
              raw: paren[0],
              text: paren[1].trim(),
              displayMode: false,
            }
          }
          const bracket = src.match(inlineBracketRule)
          if (bracket) {
            return {
              type: 'katexBracketInline',
              raw: bracket[0],
              text: bracket[1].trim(),
              displayMode: false,
            }
          }
          return undefined
        },
        renderer(token) {
          return katexHtml(token.text, false, options)
        },
      },
    ],
  }
}
