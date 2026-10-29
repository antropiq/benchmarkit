import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useGitStatus } from '../src/composables/useGitStatus'

// Layer 1 (jsdom): the tab-strip state a diff tab plugs into, and the git-status
// index the tree rows read. Both composables are singletons with module-level
// state — that is the app's convention — so each test resets it by hand rather
// than assuming a fresh import.
//
// What is asserted here is the *contract* the components rely on: that a diff
// of a file and that file's own buffer are two tabs about one path, and that
// "marked" and "diffable" are two different questions.

const files = useEditorFiles()
const git = useGitStatus()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

function resetState() {
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  files.state.loading = false
  git.state.entries = []
  git.state.repo = false
  git.state.error = null
  git.state.loadedAt = 0
}

describe('useEditorFiles: diff tabs in the same strip', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('opens a diff as its own record, keyed apart from the file', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ path: 'a.js', original: 'one\n', modified: 'two\n', language: 'javascript', status: 'modified' }),
    )
    const record = await files.openDiff('a.js')
    expect(record.kind).toBe('diff')
    expect(record.key).toBe('diff:a.js')
    expect(record.path).toBe('a.js')
    expect(record.name).toBe('a.js (diff)')
    expect(files.state.active).toBe('diff:a.js')
    expect(files.state.open).toHaveLength(1)
  })

  it('keeps a file buffer and its diff open as two tabs', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockImplementationOnce(() =>
      jsonResponse({ path: 'a.js', text: 'one\n', language: 'javascript' }),
    )
    fetchMock.mockImplementationOnce(() =>
      jsonResponse({ path: 'a.js', original: 'one\n', modified: 'two\n' }),
    )
    await files.openFile('a.js')
    await files.openDiff('a.js')
    expect(files.state.open.map((item) => item.key)).toEqual(['a.js', 'diff:a.js'])
    // Closing one leaves the other, and the *file* lookup still finds only the
    // file — the tree's "Close" item must not be satisfied by a diff tab.
    expect(files.recordFor('a.js').kind).toBe('file')
    expect(files.diffRecordFor('a.js').kind).toBe('diff')
  })

  it('focuses an open diff instead of fetching it twice', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ path: 'a.js', original: '', modified: '' }),
    )
    await files.openDiff('a.js')
    files.focusFile('a.js')
    await files.openDiff('a.js')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(files.state.active).toBe('diff:a.js')
  })

  it('closes a diff tab and focuses its neighbour', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockImplementationOnce(() =>
      jsonResponse({ path: 'a.js', text: 'one\n' }),
    )
    fetchMock.mockImplementationOnce(() =>
      jsonResponse({ path: 'b.js', text: 'one\n' }),
    )
    fetchMock.mockImplementationOnce(() =>
      jsonResponse({ path: 'b.js', original: 'one\n', modified: 'two\n' }),
    )
    await files.openFile('a.js')
    await files.openFile('b.js')
    await files.openDiff('b.js')
    expect(files.state.active).toBe('diff:b.js')
    files.closeTab('diff:b.js')
    expect(files.state.active).toBe('b.js')
    expect(files.state.open).toHaveLength(2)
    files.closeFile('b.js')
    expect(files.state.open.map((item) => item.key)).toEqual(['a.js'])
  })

  it('shows an absent path as a message, without a tab', async () => {
    // pol-06: a hidden path answers a *look* with 404, not with the refusal it
    // used to answer. Either way the pane shows the server's sentence and opens
    // nothing — what matters here is that no half-loaded diff tab appears.
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ detail: 'Path does not exist: .env' }, false, 404),
    )
    const record = await files.openDiff('.env')
    expect(record).toBeNull()
    expect(files.state.open).toHaveLength(0)
    expect(files.state.message.tone).toBe('error')
    expect(files.state.message.text).toContain('does not exist')
    expect(files.state.message.path).toBe('.env')
  })

  it('says so for a binary file rather than rendering two blobs', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ path: 'x.bin', binary: true }),
    )
    const record = await files.openDiff('x.bin')
    expect(record).toBeNull()
    expect(files.state.open).toHaveLength(0)
    expect(files.state.message.tone).toBe('info')
  })
})

describe('useGitStatus: what the tree asks per row', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  function setEntries(entries) {
    git.state.entries = entries
  }

  it('marks a modified file and calls it diffable', () => {
    setEntries([{ path: 'web/package.json', kind: 'modified', comparable: true }])
    expect(git.statusFor('web/package.json').kind).toBe('modified')
    expect(git.isUnstaged('web/package.json')).toBe(true)
    expect(git.isDiffable('web/package.json')).toBe(true)
  })

  it('marks an untracked file but offers no diff for it', () => {
    setEntries([{ path: 'new.txt', kind: 'untracked', comparable: false }])
    expect(git.isUnstaged('new.txt')).toBe(true)
    expect(git.isDiffable('new.txt')).toBe(false)
  })

  it('knows a file added to the index has no committed side either', () => {
    setEntries([{ path: 'staged.txt', kind: 'added', comparable: false }])
    expect(git.isDiffable('staged.txt')).toBe(false)
  })

  it('marks every ancestor folder of a changed file', () => {
    setEntries([{ path: 'web/src/App.vue', kind: 'modified', comparable: true }])
    for (const folder of ['', 'web', 'web/src']) {
      expect(git.hasUnstagedBelow(folder)).toBe(true)
    }
    expect(git.hasUnstagedBelow('web/src/App.vue')).toBe(false)
    expect(git.hasUnstagedBelow('other')).toBe(false)
  })

  it('one fetch for the whole work tree, joined when already running', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ repo: true, atToplevel: true, entries: [{ path: 'a', kind: 'modified', comparable: true }], truncated: false }),
    )
    await Promise.all([git.load({ force: true }), git.load({ force: true })])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(git.state.repo).toBe(true)
    expect(git.isUnstaged('a')).toBe(true)
  })

  it('keeps a failed query apart from a clean work tree', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ detail: 'git is not available' }, false, 504),
    )
    await git.load({ force: true })
    expect(git.state.error).toContain('git is not available')
    expect(git.state.entries).toEqual([])
    expect(git.state.repo).toBe(false)
  })

  it('a location with no repository is not an error', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ repo: false, root: null, atToplevel: true, entries: [], truncated: false }),
    )
    await git.load({ force: true })
    expect(git.state.error).toBeNull()
    expect(git.state.repo).toBe(false)
    expect(git.isUnstaged('anything')).toBe(false)
  })
})
