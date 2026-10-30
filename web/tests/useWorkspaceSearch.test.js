import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useWorkspaceSearch } from '../src/composables/useWorkspaceSearch'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// ---------------------------------------------------------------------------
// The search composable's state, driven through its own functions.
//
// The rule this suite exists for is the one the dialog cannot hold on its own:
// **a search's answer outlives the popup that showed it** (§2.7). Closing and
// re-opening must show the same query, the same scope and the same table, and
// must not ask the server again — the answer is only replaced by a new search.
// The other half of that promise is the re-root clear: those paths are relative
// to a root that a settings switch makes meaningless, so a switch forgets all of
// it, the open dialog included.
//
// The server's own rules (the caps, the footer sentence, the policy pruning)
// belong to tests/test_editorfs_search.py; here the transport is a stubbed
// `fetch` and the payload is the Contract read verbatim.
//
// `useEditorFiles` is imported for its side effect, not for its own assertions:
// its factory body registers the buffer holder of the re-root registry, so the
// switch case at the end exercises the real registry with *both* subscribers
// rather than only the search half. (The buffers' own half is asserted in
// useWorkspaceSwitch.test.js; `files` is touched only to show a switch does not
// stop at the search state.)
// ---------------------------------------------------------------------------

const tree = useWorkspaceTree()
const search = useWorkspaceSearch()
const files = useEditorFiles()

const FOLDER_TARGET = { path: 'docs/old', name: 'old', dir: true, parent: 'docs' }
const ROOT_TARGET = { path: '', name: '', dir: true, parent: '' }
const FILE_TARGET = { path: 'src/a.py', name: 'a.py', dir: false, parent: 'src' }
// One row of the Contract's `matches`.
const HIT = { path: 'docs/old/notes.md', line: 12, column: 8, snippet: 'a verb', matchLength: 4 }

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// A Contract payload, distinguishable by its one snippet.
function payload(snippet, path = 'docs/old/notes.md') {
  return {
    scope: 'docs/old',
    isFile: false,
    matches: [{ path, line: 12, column: 8, snippet, matchLength: 4 }],
    matchCount: 1,
    filesScanned: 3,
    truncated: false,
    skipped: { oversized: 0, binary: 0 },
    footerNote: '',
  }
}

function resetSearch() {
  search.state.query = ''
  search.state.caseSensitive = false
  search.state.regex = false
  search.state.scope = null
  search.state.open = false
  search.state.status = 'idle'
  search.state.result = null
  search.state.error = null
  search.state.pendingReveal = null
}

function resetTree() {
  tree.state.nodes = {}
  tree.state.expanded = { '': true }
  tree.state.focused = ''
  tree.state.selected = null
  tree.state.editing = null
  tree.state.error = null
  tree.state.workspace = '/w/demo'
}

// The routes a working-location switch touches: the tree's listing of the new
// root and the git marks the buffer holder re-reads.
function stubWorkspaceRoutes() {
  vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
    const key = String(url)
    if (key.startsWith('/api/fs/tree')) {
      const path = decodeURIComponent(key.slice('/api/fs/tree?path='.length))
      return jsonResponse({
        path,
        workspace: '/w/other',
        entries: [{ name: 'src', path: 'src', dir: true, children: true }],
        truncated: false,
      })
    }
    if (key.startsWith('/api/fs/status')) {
      return jsonResponse({ repo: false, entries: [] })
    }
    return jsonResponse({})
  })
}

describe('openFor: the scope comes from the row, the last answer stays', () => {
  beforeEach(() => {
    resetSearch()
    resetTree()
    vi.restoreAllMocks()
  })

  it('names a folder row and clears nothing', () => {
    search.state.query = 'verb'
    search.state.result = payload('the verb of the sentence')
    search.state.status = 'done'

    expect(search.openFor(FOLDER_TARGET)).toBe(true)
    expect(search.state.scope.kind).toBe('folder')
    expect(search.state.open).toBe(true)
    // The retention rule, stated as the spec states it: re-opening shows the
    // previous query and the previous table.
    expect(search.state.query).toBe('verb')
    expect(search.state.result?.matches[0].snippet).toBe('the verb of the sentence')
    expect(search.state.status).toBe('done')
  })

  it('names the workspace for the root row, through the tree\'s own label', () => {
    tree.state.workspace = '/w/benchmarkit'
    expect(search.openFor(ROOT_TARGET)).toBe(true)
    expect(search.state.scope.kind).toBe('workspace')
    expect(search.state.scope.label).toBe('benchmarkit')
  })

  it('declines when there is no row to search', () => {
    expect(search.openFor(null)).toBe(false)
    expect(search.state.open).toBe(false)
    expect(search.state.scope).toBeNull()
  })
})

describe('run: one request, and the answer it keeps', () => {
  beforeEach(() => {
    resetSearch()
    resetTree()
    vi.restoreAllMocks()
  })

  // Nothing is re-fetched on re-open: the answer is replaced only by a *new*
  // search, which is §6's headline row.
  it('a close and a re-open cost no second request', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(() => jsonResponse(payload('the verb of the sentence')))
    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    await search.run()
    expect(search.state.status).toBe('done')

    search.closeDialog()
    search.openFor(FOLDER_TARGET)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(search.state.open).toBe(true)
    expect(search.state.query).toBe('verb')
    expect(search.state.scope.path).toBe('docs/old')
    expect(search.state.result?.matches).toHaveLength(1)
  })

  it('sends the four params the route declares, with the needle encoded', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(() => jsonResponse(payload('a & b')))
    search.openFor(FOLDER_TARGET)
    search.state.query = 'a&b'
    search.state.caseSensitive = true
    search.state.regex = true
    await search.run()

    const url = String(fetchMock.mock.calls[0][0])
    expect(url.startsWith('/api/fs/search?')).toBe(true)
    for (const param of ['path=docs%2Fold', 'q=a%26b', 'caseSensitive=true', 'regex=true']) {
      expect(url).toContain(param)
    }
    // The payload is the Contract verbatim — no reshaping, no defaulted keys.
    expect(search.state.result).toEqual(payload('a & b'))
  })

  it('an empty query asks nothing and keeps the answer that is on screen', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({}))
    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    await search.run()
    const before = search.state.result
    const calls = fetchMock.mock.calls.length

    search.state.query = '   '
    await search.run()

    expect(fetchMock).toHaveBeenCalledTimes(calls)
    expect(search.state.result).toBe(before)
    // 'idle', not 'error': the dialog must not look like a failed search.
    expect(search.state.status).toBe('idle')
    expect(search.state.error).toBeNull()
  })

  it('a refused search leaves the last good answer and re-says the server', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation((url) =>
        String(url).includes('q=%5B')
          ? jsonResponse({ detail: 'unterminated character set at index 3' }, false, 400)
          : jsonResponse(payload('the verb of the sentence')),
      )
    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    await search.run()

    search.state.query = '['
    search.state.regex = true
    await search.run()

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(search.state.status).toBe('error')
    expect(search.state.error.status).toBe(400)
    // re's own message, caret position and all (D5) — verbatim.
    expect(search.state.error.detail).toBe('unterminated character set at index 3')
    expect(search.state.result?.matches[0].snippet).toBe('the verb of the sentence')
  })

  it('a late answer does not overwrite a newer one', async () => {
    const gates = []
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      let release
      const gate = new Promise((resolve) => {
        release = resolve
      })
      gates.push(release)
      return gate
    })
    search.openFor(FOLDER_TARGET)
    search.state.query = 'first'
    const older = search.run()
    search.state.query = 'second'
    const newer = search.run()
    expect(gates).toHaveLength(2)

    gates[1](jsonResponse(payload('newer answer')))
    await newer
    expect(search.state.result.matches[0].snippet).toBe('newer answer')

    gates[0](jsonResponse(payload('older answer')))
    await older
    // The table always belongs to the text in the field, never to whichever
    // request happened to answer last.
    expect(search.state.result.matches[0].snippet).toBe('newer answer')
    expect(search.state.status).toBe('done')
  })

  it('a re-search keeps the previous table on screen while it loads', async () => {
    let release
    const gate = new Promise((resolve) => {
      release = resolve
    })
    let calls = 0
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      calls += 1
      return calls === 1 ? jsonResponse(payload('the verb of the sentence')) : gate
    })
    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    await search.run()

    search.state.query = 'verbs'
    const inFlight = search.run()
    expect(search.state.status).toBe('running')
    expect(search.state.result?.matches).toHaveLength(1)

    release(jsonResponse(payload('verbs answer')))
    await inFlight
    expect(search.state.status).toBe('done')
  })

  it('a request that never arrived is named as one, not as a refusal', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      throw new TypeError('Failed to fetch')
    })
    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    await search.run()

    expect(search.state.status).toBe('error')
    expect(search.state.error.status).toBe(0)
    expect(search.state.error.detail).toContain('could not be reached')
  })
})

// A row click, and the trap in it. The tree is lazy, so the path a hit names is
// usually in a branch that has never been listed — `entryFor` answers null there
// and `select(null)` does nothing, so the naive `select(entryFor(path))` would
// open *nothing* for the common case. Both cases are asserted: the synthesised
// entry that makes a collapsed path open, and the real listing entry being
// preferred when one exists (its size and mtime are what the pane's guards read).
describe('openHit: the reveal is queued, the selection opens the buffer', () => {
  beforeEach(() => {
    resetSearch()
    resetTree()
    vi.restoreAllMocks()
  })

  it('selects a path whose listing is not loaded — the collapsed-branch case', () => {
    search.openFor(ROOT_TARGET)
    search.state.open = true
    expect(tree.entryFor(HIT.path)).toBeNull()

    search.openHit(HIT)

    expect(tree.state.selected).toEqual({
      path: HIT.path,
      name: 'notes.md', // derived from the path: select() copies it into the selection
      dir: false, // a file, so the pane opens it instead of toggling a folder
      size: null,
      mtime: null,
    })
    // No `source: 'tab'`: that marker makes EditorArea's watcher *skip* the
    // selection, so `reveal()` can never be the opener.
    expect(tree.state.selected.source).toBeUndefined()
    expect(search.state.pendingReveal).toEqual({
      path: HIT.path,
      line: HIT.line,
      column: HIT.column,
    })
    expect(search.state.open).toBe(false)
  })

  it('reuses the tree\'s own entry when the listing is loaded', () => {
    tree.state.nodes['docs/old'] = {
      loaded: true,
      loading: false,
      entries: [{ name: 'notes.md', path: HIT.path, dir: false, size: 2048, mtime: 1700 }],
      truncated: false,
      error: null,
    }
    search.openFor(FOLDER_TARGET)

    search.openHit(HIT)

    // The listing's own size and mtime survive — a synthetic entry would throw
    // them away, and the pane's guards read them.
    expect(tree.state.selected).toEqual({
      path: HIT.path,
      name: 'notes.md',
      dir: false,
      size: 2048,
      mtime: 1700,
    })
    expect(search.state.open).toBe(false)
  })

  it('takes a file row as its own scope, and a root-level hit names the bare file', () => {
    search.openFor(FILE_TARGET)
    expect(search.state.scope.kind).toBe('file')

    search.openHit({ path: 'LICENSE', line: 1, column: 0 })
    expect(tree.state.selected.name).toBe('LICENSE')
    expect(search.state.pendingReveal).toEqual({ path: 'LICENSE', line: 1, column: 0 })
  })

  it('a hit with no path opens nothing and queues nothing', () => {
    search.openFor(FOLDER_TARGET)
    search.state.open = true
    search.openHit(null)
    search.openHit({ line: 3 })

    expect(tree.state.selected).toBeNull()
    expect(search.state.pendingReveal).toBeNull()
    expect(search.state.open).toBe(true)
  })
})

// The working-location switch: every path in a search result is relative to the
// root that was just left, so the answer, the query, the scope and the queued
// reveal all go — and the dialog goes with them, because it is a `v-show`
// mounted view like the rest of the Editor and a popup over a root that does not
// contain its own rows is the lie §2.7 names. Both registry subscribers run:
// `useEditorFiles`'s body registered the buffer holder when this file imported
// it, so the tabs closing here is the two-subscriber fix, not a second test of
// the buffers.
describe('a working-location switch empties the search', () => {
  beforeEach(() => {
    resetSearch()
    resetTree()
    files.state.open.length = 0
    files.state.active = null
    files.state.message = null
    files.stopFreshnessProbe()
    vi.restoreAllMocks()
  })

  it('forgets query, scope, result, error, the reveal and the open dialog', async () => {
    stubWorkspaceRoutes()
    const fetchMock = globalThis.fetch
    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    search.state.caseSensitive = true
    search.state.regex = true
    search.state.result = payload('the verb of the sentence')
    search.state.error = { status: 400, detail: 'unterminated character set at index 3' }
    search.state.status = 'error'
    search.state.pendingReveal = { path: HIT.path, line: 12, column: 8 }
    search.state.open = true

    await tree.reRoot('/w/other')

    expect(search.state.query).toBe('')
    expect(search.state.caseSensitive).toBe(false)
    expect(search.state.regex).toBe(false)
    expect(search.state.scope).toBeNull()
    expect(search.state.result).toBeNull()
    expect(search.state.error).toBeNull()
    expect(search.state.pendingReveal).toBeNull()
    expect(search.state.status).toBe('idle')
    expect(search.state.open).toBe(false)
    // A switch re-roots and re-reads; it never asks for a new answer.
    const asked = fetchMock.mock.calls.some(([url]) => String(url).startsWith('/api/fs/search'))
    expect(asked).toBe(false)

    // And the field really is empty afterwards, not holding the old needle.
    search.openFor({ path: 'src', name: 'src', dir: true, parent: '' })
    expect(search.state.query).toBe('')
    expect(search.state.result).toBeNull()
    expect(search.state.scope.path).toBe('src')
    expect(search.state.open).toBe(true)
  })

  // The two subscribers on one event. A single-slot registry would make whichever
  // holder registered last the only one that cleans up, and neither half is
  // visible from the file that covers the other — so the *pair* is asserted here:
  // `useEditorFiles`'s body registered the buffers, this one registered the
  // results, and one `reRoot` runs both.
  it('the buffers and the results go on the same switch', async () => {
    stubWorkspaceRoutes()
    files.state.open.push({
      kind: 'file',
      key: 'src/a.py',
      path: 'src/a.py',
      name: 'a.py',
      text: 'a\n',
      baseline: 'a\n',
      dirty: false,
      mtime: 1,
    })
    files.state.active = 'src/a.py'
    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    search.state.result = payload('the verb of the sentence')

    await tree.reRoot('/w/other')

    expect(files.state.open).toHaveLength(0)
    expect(search.state.result).toBeNull()
    expect(search.state.query).toBe('')
  })

  // A search that was in flight when the root moved answers about the folder that
  // was just left; applying it would put the old workspace's rows under the new
  // root's label. Same token rule the tree's listings use.
  it('an answer that lands after the switch is dropped', async () => {
    let release
    const gate = new Promise((resolve) => {
      release = resolve
    })
    // One implementation for both halves: the search request that must be
    // abandoned, and the routes the switch itself asks about.
    vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      const key = String(url)
      if (key.startsWith('/api/fs/search')) return gate
      if (key.startsWith('/api/fs/tree')) {
        return jsonResponse({
          path: '',
          workspace: '/w/other',
          entries: [{ name: 'src', path: 'src', dir: true, children: true }],
          truncated: false,
        })
      }
      if (key.startsWith('/api/fs/status')) {
        return jsonResponse({ repo: false, entries: [] })
      }
      return jsonResponse({})
    })

    search.openFor(FOLDER_TARGET)
    search.state.query = 'verb'
    const inFlight = search.run()
    await tree.reRoot('/w/other')
    release(jsonResponse(payload('the verb of the sentence')))
    await inFlight

    expect(search.state.result).toBeNull()
    expect(search.state.status).toBe('idle')
  })
})
