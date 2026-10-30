import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

import SearchDialog from '../src/components/SearchDialog.vue'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useWorkspaceSearch } from '../src/composables/useWorkspaceSearch'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// ---------------------------------------------------------------------------
// The search popup, mounted for real.
//
// Everything it shows is `useWorkspaceSearch` state, so this suite never stubs
// the composable: it *drives* it (openFor / run / a payload placed on `result`)
// and reads the DOM back. That is what makes these the tests for the rendering
// rules rather than for the state machine — the state machine has its own file
// (useWorkspaceSearch.test.js).
//
// The two rules worth jsdom are D12 (a snippet is a string, never markup) and
// the highlight slice (a wrong offset shows the mark on the wrong text), and
// both are only observable on real DOM nodes. The rest is the copy the user
// reads: the three scope labels, the footer sentence, and the inline error that
// §6 insists is *not* a notice.
//
// jsdom has no `showModal`/`close` on `<dialog>`, so both are patched on
// `Element.prototype` the way useEditorWriteMatrix.test.js:385-390 does — they
// only have to keep `el.open` truthful, which is all the component's watch reads.
// ---------------------------------------------------------------------------

const search = useWorkspaceSearch()
const tree = useWorkspaceTree()
const notices = useEditorNotices()

const FOLDER_TARGET = { path: 'docs/old', name: 'old', dir: true, parent: 'docs' }
const FILE_TARGET = { path: 'src/config.py', name: 'config.py', dir: false, parent: 'src' }
const ROOT_TARGET = { path: '', name: '', dir: true, parent: '' }

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// A Contract payload; `matches` defaults to the one row most cases need.
function payload(overrides = {}) {
  return {
    scope: 'docs/old',
    isFile: false,
    matches: [
      {
        path: 'docs/old/notes.md',
        line: 12,
        column: 8,
        snippet: 'search verb found',
        matchLength: 4,
      },
    ],
    matchCount: 1,
    filesScanned: 3,
    truncated: false,
    skipped: { oversized: 0, binary: 0 },
    footerNote: '',
    ...overrides,
  }
}

let host = null

function mountDialog() {
  host = document.createElement('div')
  document.body.appendChild(host)
  return mount(SearchDialog, { attachTo: host })
}

// Raise the popup through the composable, as search-07's gesture will.
async function openOn(target) {
  search.openFor(target)
  await flushPromises()
}

beforeEach(() => {
  Element.prototype.showModal = function showModal() {
    this.open = true
  }
  Element.prototype.close = function close() {
    this.open = false
  }
  // A fresh dialog per test, and the singleton state to match — the composable
  // deliberately keeps its answer across closes, so a test that did not reset
  // would read the previous test's table.
  search.state.query = ''
  search.state.caseSensitive = false
  search.state.regex = false
  search.state.scope = null
  search.state.open = false
  search.state.status = 'idle'
  search.state.result = null
  search.state.error = null
  search.state.pendingReveal = null
  tree.state.workspace = '/w/benchmarkit'
  tree.state.nodes = {}
  tree.state.selected = null
  notices.state.notices.length = 0
  vi.restoreAllMocks()
})

afterEach(() => {
  host?.remove()
  host = null
  document.body.querySelectorAll('dialog').forEach((el) => el.remove())
})

describe('the scope label names what is searched', () => {
  it('a folder row gives the folder path', async () => {
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    expect(w.find('.editor-search-scope').text().replace(/\s+/g, ' ')).toBe(
      'Search in: docs/old (folder)',
    )
    w.unmount()
  })

  it('a file row gives the file path', async () => {
    const w = mountDialog()
    await openOn(FILE_TARGET)

    expect(w.find('.editor-search-scope').text().replace(/\s+/g, ' ')).toBe(
      'Search in: src/config.py (file)',
    )
    w.unmount()
  })

  // §6's row, verbatim: the root case is the workspace's own name, and the empty
  // string the root row's path *is* must never reach the label.
  it('the root row gives the workspace basename, not an empty label', async () => {
    tree.state.workspace = '/opt/checkouts/benchmarkit'
    const w = mountDialog()
    await openOn(ROOT_TARGET)

    expect(w.find('.editor-search-scope').text().replace(/\s+/g, ' ')).toBe(
      'Search in: benchmarkit (whole workspace)',
    )
    w.unmount()
  })
})

describe('the controls', () => {
  it('the regex switch names the engine the pattern runs in', async () => {
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    const labels = w.findAll('.editor-search-switches label')
    expect(labels.map((node) => node.text().replace(/\s+/g, ' ').trim())).toEqual([
      'Case sensitive',
      'Regex',
    ])
    // §2.3's wording verbatim: the honest thing is to say where the pattern ran.
    expect(labels[1].attributes('title')).toBe('regex — Python syntax')
    // Both default off, whatever the state happened to hold before.
    expect(labels[0].find('input').element.checked).toBe(false)
    expect(labels[1].find('input').element.checked).toBe(false)
    w.unmount()
  })

  // No `v-if` on the `<dialog>`: unmounting it throws the field's caret and the
  // results' scroll position away, which is what §2.7's retention is worth. The
  // element is mounted closed, and the watch moves it.
  it('is in the DOM before anything raises it', () => {
    const w = mountDialog()

    expect(search.state.open).toBe(false)
    expect(w.find('dialog').exists()).toBe(true)
    expect(w.find('dialog').element.open).toBe(false)
    w.unmount()
  })

  it('follows state.open with showModal, and state.open=false with close', async () => {
    const w = mountDialog()
    await openOn(FOLDER_TARGET)
    expect(w.find('dialog').element.open).toBe(true)

    search.closeDialog()
    await flushPromises()
    expect(w.find('dialog').element.open).toBe(false)
    w.unmount()
  })
})

describe('a refusal is said in the dialog, not in the notice stack', () => {
  // The difference §6 calls out. A 400 for an invalid pattern is `re`'s own
  // message next to the caret that typed it; the bottom-centre stack is for
  // writes, and routing this there would put the answer far from the fix.
  it('renders re\'s message inline, keeps the old table, and adds no notice', async () => {
    search.state.result = payload()
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse(
        { detail: 'unterminated character set at index 3' },
        false,
        400,
      ),
    )
    search.state.query = '['
    await search.run()
    await flushPromises()

    const inline = w.find('.editor-search-error')
    expect(inline.exists()).toBe(true)
    expect(inline.text()).toBe('unterminated character set at index 3')
    // The notice stack: no entry in the composable, and none on screen.
    expect(notices.state.notices).toHaveLength(0)
    expect(document.querySelector('.editor-notice, .editor-notice--error')).toBeNull()
    // The previous answer is what the failing query is compared against.
    expect(w.findAll('.editor-search-row')).toHaveLength(1)
  })
})

// The component's own source, for the few rules that are about *who calls what*
// and cannot be seen on a DOM node (same approach as workbenchGutter.test.js's
// template scan). `resolve()` is the `web/` root under Vitest.
const componentSource = readFileSync(
  join(resolve(), 'src', 'components', 'SearchDialog.vue'),
  'utf8',
)

describe('a snippet is never injected as HTML', () => {
  it('renders the bytes as text, with no element in the cell', async () => {
    search.state.result = payload({
      matches: [
        {
          path: 'web/index.html',
          line: 4,
          column: 10,
          snippet: 'a <img src=x onerror=alert(1)> tag',
          matchLength: 4,
        },
      ],
    })
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    const cell = w.find('.editor-search-cell-snippet')
    expect(cell.text()).toContain('<img src=x onerror=alert(1)>')
    expect(cell.element.innerHTML).not.toContain('<img')
    // Exactly two elements in the cell — the snippet wrapper and the highlight
    // span — so nothing in the line became a node of its own.
    expect(Array.from(cell.element.querySelectorAll('*')).map((el) => el.className)).toEqual([
      'editor-search-snippet',
      'editor-search-mark',
    ])
  })

  // The rule D12 states in code terms, asserted on the source: no `v-html` and no
  // import of the markdown renderer, which is for *rendered* HTML.
  it('uses no v-html and no markdown renderer', () => {
    expect(componentSource).not.toMatch(/v-html/)
    expect(componentSource).not.toMatch(/markdown/)
  })

  // The path column is the same kind of untrusted text, and it is bound through
  // a button rather than an attribute-only label.
  it('shows the workspace-relative path verbatim as the row button', async () => {
    search.state.result = payload({
      matches: [{ path: 'src/a <b>.py', line: 2, column: 1, snippet: 'x', matchLength: 1 }],
    })
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    const button = w.find('.editor-search-hit')
    expect(button.text()).toBe('src/a <b>.py')
    expect(button.element.innerHTML).not.toContain('<b')
  })
})

describe('the highlight span', () => {
  function snippetNodes(cell) {
    return Array.from(cell.element.querySelector('.editor-search-snippet').childNodes).map(
      (node) =>
        node.nodeType === 3
          ? { text: node.textContent }
          : { tag: node.tagName.toLowerCase(), text: node.textContent },
    )
  }

  it('wraps exactly the matched characters, in three nodes', async () => {
    search.state.result = payload()
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    // column 8 is the *start of the match*, 1-based: offset 7 in the snippet,
    // so 'verb' and not the character before or after it.
    expect(snippetNodes(w.find('.editor-search-cell-snippet'))).toEqual([
      { text: 'search ' },
      { tag: 'span', text: 'verb' },
      { text: ' found' },
    ])
  })

  it('a zero-width match highlights nothing', async () => {
    search.state.result = payload({
      matches: [{ path: 'a.py', line: 1, column: 4, snippet: 'search verb', matchLength: 0 }],
    })
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    expect(snippetNodes(w.find('.editor-search-cell-snippet'))).toEqual([
      { text: 'sea' },
      { tag: 'span', text: '' },
      { text: 'rch verb' },
    ])
    // Not the whole line: the mark is the empty span between the two texts.
    expect(w.find('.editor-search-mark').text()).toBe('')
  })

  // A line trimmed at MAX_SEARCH_SNIPPET_BYTES can end *before* the match, and
  // the row still has to render: the mark belongs at the end, with no width.
  it('a snippet trimmed before the match position renders, and does not throw', async () => {
    const trimmed = `${'a'.repeat(240)}…`
    search.state.result = payload({
      matches: [
        { path: 'long.txt', line: 7, column: 400, snippet: trimmed, matchLength: 9 },
      ],
    })
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    expect(snippetNodes(w.find('.editor-search-cell-snippet'))).toEqual([
      { text: trimmed },
      { tag: 'span', text: '' },
      { text: '' },
    ])
  })
})

describe('the footer sentence', () => {
  // D8: the server wrote this one sentence from the caps in `editorfs/constants`.
  // The dialog renders it and adds nothing — no client-side count, no "N matches".
  it('renders footerNote verbatim when there is one', async () => {
    search.state.result = payload({
      truncated: true,
      footerNote: 'showing the first 500 matches — refine the query',
    })
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    expect(w.find('.editor-search-footer').text()).toBe(
      'showing the first 500 matches — refine the query',
    )
  })

  it('is absent when the server had nothing to say', async () => {
    search.state.result = payload({ footerNote: '' })
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    expect(w.find('.editor-search-footer').exists()).toBe(false)
    w.unmount()
  })
})

describe('the busy state keeps the answer it is being compared against', () => {
  it('shows the marker and the old table together', async () => {
    let release
    const gate = new Promise((resolve) => {
      release = resolve
    })
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => gate)
    search.state.result = payload()
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    search.state.query = 'verbs'
    const inFlight = search.run()
    await flushPromises()

    expect(w.find('.editor-search-busy').exists()).toBe(true)
    expect(w.find('.editor-search-submit').element.disabled).toBe(true)
    expect(w.findAll('.editor-search-row')).toHaveLength(1)

    release(jsonResponse(payload({ matches: [] })))
    await inFlight
    await flushPromises()

    // The empty state is an answer, so it appears only once one has arrived.
    expect(w.find('.editor-search-empty').text()).toBe('no matches')
    w.unmount()
  })
})

describe('a row click', () => {
  // D13/D14: the close belongs to `openHit`. If the component also called
  // `closeDialog()`, the close would have two owners and the retention rule
  // would be enforced in two places.
  it('calls openHit, and the composable is what closes the dialog', async () => {
    const hit = payload().matches[0]
    search.state.result = payload({ matches: [hit] })
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    // The path is absent from the tree's listings, so this is the synthesised
    // -entry case the composable exists for.
    w.find('.editor-search-hit').trigger('click')
    await flushPromises()

    expect(search.state.open).toBe(false)
    expect(search.state.pendingReveal).toEqual({
      path: hit.path,
      line: hit.line,
      column: hit.column,
    })
    expect(tree.state.selected?.path).toBe(hit.path)
    // The element followed the state rather than being closed by hand.
    expect(w.find('dialog').element.open).toBe(false)

    // The other half: the row's handler is `openHit` and nothing else. A spy on
    // the composable's exported `closeDialog` cannot see the internal call
    // `openHit` makes, so this is read off the template — one click owner, no
    // second `closeDialog()` beside it (D13).
    const handler = componentSource.match(/@click="([^"]*)"/g).filter((tag) =>
      tag.includes('openHit'),
    )
    expect(handler).toEqual(['@click="openHit(row.hit)"'])
  })
})

describe('Escape (the native close)', () => {
  // The retention rule seen from the element: the *element* is allowed to close
  // on its own (that is what Escape does), and the `@close` handler's job is to
  // write the state — never to clear the answer with it.
  it('leaves state.open false with the query, scope and table still set', async () => {
    search.state.result = payload()
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)
    search.state.query = 'verb'

    w.find('dialog').element.close()
    w.find('dialog').trigger('close')
    await flushPromises()

    expect(search.state.open).toBe(false)
    expect(search.state.query).toBe('verb')
    expect(search.state.scope.path).toBe('docs/old')
    expect(search.state.result.matches).toHaveLength(1)
    expect(search.state.status).toBe('done')
    w.unmount()
  })
})

describe('the backdrop', () => {
  // A click on the dimmed area outside the card is the same answer as Escape —
  // the popup goes, the state does not. The `event.target === dialogEl` test is
  // what tells the two apart: a click inside the card reaches the dialog too.
  it('a click on the dialog itself closes it and keeps the answer', async () => {
    search.state.result = payload()
    search.state.status = 'done'
    const w = mountDialog()
    await openOn(FOLDER_TARGET)
    search.state.query = 'verb'

    w.find('dialog').trigger('click')
    await flushPromises()

    expect(search.state.open).toBe(false)
    expect(search.state.query).toBe('verb')
    expect(search.state.result.matches).toHaveLength(1)

    // The same gesture one level in — on the card — is not a close.
    search.state.open = true
    await flushPromises()
    w.find('article').trigger('click')
    await flushPromises()
    expect(search.state.open).toBe(true)
    w.unmount()
  })
})

describe('Enter in the query field', () => {
  // No `<form>` in this dialog, so Enter cannot navigate — and the point of the
  // explicit keydown is that it *does* search. One request, one answer.
  it('runs exactly one request and submits no form', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(() => jsonResponse(payload()))
    const w = mountDialog()
    await openOn(FOLDER_TARGET)

    // A form that did exist would navigate; assert none is in the subtree.
    expect(w.find('form').exists()).toBe(false)
    const submitted = vi.fn()
    w.element.addEventListener('submit', submitted)

    await w.find('.editor-search-query').setValue('verb')
    await w.find('.editor-search-query').trigger('keydown.enter')
    await flushPromises()

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/fs/search?')
    expect(search.state.query).toBe('verb')
    expect(submitted).not.toHaveBeenCalled()
    // The table that answer produced is on screen.
    expect(w.findAll('.editor-search-row')).toHaveLength(1)
    w.unmount()
  })
})
