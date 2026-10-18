const CACHE_LIMIT = 50

const cache = new Map()
let mermaidPromise = null
let idSeq = 0

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'default',
      })
      return mermaid
    })
  }
  return mermaidPromise
}

export function encodeSource(src) {
  const bytes = new TextEncoder().encode(src)
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary)
}

export function decodeSource(encoded) {
  const binary = atob(encoded)
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

const FAILED = Symbol('mermaid failed')

// Cached per source, like hljs.js: completed fences are immutable while
// streaming, so each rAF re-render reuses the rendered SVG instead of
// re-parsing the diagram. Failures are cached too, so a broken fence does
// not re-parse on every streamed frame.
async function renderDiagram(src) {
  const hit = cache.get(src)
  if (hit === FAILED) throw new Error('mermaid render failed')
  if (hit !== undefined) return hit
  const mermaid = await loadMermaid()
  try {
    const ok = await mermaid.parse(src, { suppressErrors: true })
    if (!ok) throw new Error('mermaid parse failed')
    const { svg } = await mermaid.render(`benchmarkit-mermaid-${++idSeq}`, src)
    // No re-sanitization here on purpose: DOMPurify's SVG profile drops
    // foreignObject, which is where mermaid renders every label. The
    // structure is mermaid's own code and securityLevel 'strict' already
    // sanitizes user-controlled label text inside mermaid.
    if (cache.size > CACHE_LIMIT) cache.clear()
    cache.set(src, svg)
    return svg
  } catch (error) {
    cache.set(src, FAILED)
    throw error
  }
}

function fallbackToCode(block, src) {
  const pre = document.createElement('pre')
  const code = document.createElement('code')
  code.className = 'hljs language-mermaid'
  code.textContent = src
  pre.appendChild(code)
  block.replaceChildren(pre)
}

// Replaces `.mermaid-block[data-src]` placeholders (emitted by
// renderMarkdown when mermaid mode is on) with rendered SVGs. `isValid`
// is checked after every await so renders superseded by a newer stream
// re-render are dropped instead of touching replaced DOM nodes.
export async function hydrateMermaid(rootEl, { isValid = () => true } = {}) {
  const blocks = rootEl.querySelectorAll('.mermaid-block[data-src]')
  for (const block of blocks) {
    if (block.dataset.hydrated) continue
    block.dataset.hydrated = '1'
    const src = decodeSource(block.dataset.src)
    let svg = null
    try {
      svg = await renderDiagram(src)
    } catch {
      /* fall back to the code block below */
    }
    if (!isValid() || !block.isConnected) return
    if (svg === null) {
      fallbackToCode(block, src)
    } else {
      block.innerHTML = svg
    }
  }
}
