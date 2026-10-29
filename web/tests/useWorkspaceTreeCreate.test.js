import { beforeEach, describe, expect, it, vi } from 'vitest'

import { registerCreateChord, useWorkspaceTree } from '../src/composables/useWorkspaceTree'
import { useEditorCreate, validateCreateName } from '../src/composables/useEditorCreate'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useGitStatus } from '../src/composables/useGitStatus'

// The create flow as the client owns it: what a name may look like, one request
// per typed name, the server's refusal shown in its own words, and what success
// does to the tree (one re-listed folder) and to the buffers (a new file opens).
// The route's rules — the write gate, the 409 on a taken name, the streaming
// refusal — are asserted in tests/test_editorfs_write.py; nothing here re-litigates
// them, and in particular nothing here knows the deny-list.

const tree = useWorkspaceTree()
const creator = useEditorCreate()
const files = useEditorFiles()
const notices = useEditorNotices()
const git = useGitStatus()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// Folder listings, wired through the tree's own cache so `startCreate` and the
// "already in this folder" check read real state. Any other request answers an
// empty root listing, which is never what a test asserts on.
function stubListings(listings) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
    const key = String(url)
    for (const [path, entries] of Object.entries(listings)) {
      if (key === `/api/fs/tree?path=${encodeURIComponent(path)}`) {
        return jsonResponse({ path, workspace: '/w', entries, truncated: false })
      }
    }
    return jsonResponse({ path: '', workspace: '/w', entries: [], truncated: false })
  })
}

function stubListing(path, entries) {
  return stubListings({ [path]: entries })
}

function resetState() {
  tree.state.nodes = {}
  tree.state.expanded = { '': true }
  tree.state.focused = ''
  tree.state.selected = null
  tree.state.editing = null
  tree.state.error = null
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  notices.clear()
  git.state.entries = []
  git.state.loadedAt = 0
  git.state.error = null
}

describe('validateCreateName: the shape half of the answer', () => {
  it('accepts an ordinary name', () => {
    expect(validateCreateName('notes.py')).toBeNull()
    expect(validateCreateName('  spaced.py  ')).toBeNull()
  })

  it('rejects the shapes that cannot be a name', () => {
    expect(validateCreateName('')).toMatch(/name is needed/)
    expect(validateCreateName('a/b')).toMatch(/slash/)
    expect(validateCreateName('a\\b')).toMatch(/backslash/)
    expect(validateCreateName('.')).toMatch(/not a name/)
    expect(validateCreateName('..')).toMatch(/not a name/)
    expect(validateCreateName('x\0y')).toMatch(/that character/)
  })

  it('rejects a name the listing already shows', () => {
    expect(validateCreateName('README.md', { taken: ['README.md', 'src'] })).toMatch(
      /already in this folder/,
    )
  })

  // The other half is the server's, so nothing here knows about `.env` or
  // `mcp_connectors/`: a protected name is *not* rejected locally.
  it('says nothing about a name the connector policy protects', () => {
    expect(validateCreateName('.env')).toBeNull()
    expect(validateCreateName('id_rsa.pem')).toBeNull()
  })
})

describe('useEditorCreate: the request and its answers', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('submitting sends one POST with the joined path', async () => {
    stubListings({
      '': [{ name: 'src', path: 'src', dir: true, children: true }],
      src: [{ name: 'a.py', path: 'src/a.py', dir: false }],
    })
    await tree.load('')
    await tree.load('src')
    expect(tree.startCreate('src', false)).toBe(true)
    tree.updateCreateValue('new.py')

    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ path: 'src/new.py', dir: false, size: 0, mtime: 5000 }),
    )
    const result = await creator.submitCreate()
    expect(result.ok).toBe(true)

    const posts = fetchMock.mock.calls.filter(
      ([url, init]) => String(url) === '/api/fs/create' && init?.method === 'POST',
    )
    expect(posts).toHaveLength(1)
    expect(JSON.parse(posts[0][1].body)).toEqual({ path: 'src/new.py', dir: false })
  })

  it('a root create sends the bare name, not a leading slash', async () => {
    tree.startCreate('', true)
    tree.updateCreateValue('scratch')
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ path: 'scratch', dir: true, size: 0, mtime: 5000 }),
    )
    await creator.submitCreate()
    const post = fetchMock.mock.calls.find(([url]) => String(url) === '/api/fs/create')
    expect(JSON.parse(post[1].body)).toEqual({ path: 'scratch', dir: true })
  })

  it('a name already in the listing is refused without a request', async () => {
    stubListings({
      '': [{ name: 'src', path: 'src', dir: true, children: true }],
      src: [{ name: 'taken.py', path: 'src/taken.py', dir: false }],
    })
    await tree.load('')
    await tree.load('src')
    tree.startCreate('src', false)
    tree.updateCreateValue('taken.py')

    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({}))
    const result = await creator.submitCreate()

    expect(result.ok).toBe(false)
    expect(result.reason).toBe('invalid-name')
    expect(tree.state.editing.error).toMatch(/already in this folder/)
    // Only the listing call, never a POST.
    expect(
      fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST'),
    ).toHaveLength(0)
  })

  it('a name with a slash is refused without a request', async () => {
    stubListing('', [])
    tree.startCreate('', false)
    tree.updateCreateValue('nested/x.py')
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({}))

    const result = await creator.submitCreate()
    expect(result.reason).toBe('invalid-name')
    expect(tree.state.editing.error).toMatch(/slash/)
    expect(
      fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST'),
    ).toHaveLength(0)
  })

  // The one that surprises people: a name that looks fine and sits under a rule
  // the workspace wrote. The server's sentence is the answer, verbatim — and it
  // quotes the pattern, so the user can go find that line (pol-06).
  it('a 422 from the server surfaces its own wording, with no local rule added', async () => {
    stubListing('', [])
    tree.startCreate('', false)
    tree.updateCreateValue('.env')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse(
        {
          detail:
            'Refused by the workspace policy: "**/.env**" (hidden)',
        },
        false,
        422,
      ),
    )

    const result = await creator.submitCreate()
    expect(result.ok).toBe(false)
    expect(result.reason).toBe('refused')
    expect(tree.state.editing.error).toContain('workspace policy')
    // The input stays open with the refusal under the caret: the fix is to type
    // another name. The notice stack carries the same refusal for a user whose
    // eyes are on the code pane, prefixed by what was attempted (D9).
    expect(tree.state.editing).not.toBeNull()
    expect(notices.noticeForPath('.env').text).toBe(
      'Could not create .env: Refused by the workspace policy: "**/.env**" (hidden)',
    )
  })

  it('a 409 on a taken name keeps the input and phrases both copies from the server', async () => {
    stubListing('', [])
    tree.startCreate('', false)
    tree.updateCreateValue('README.md')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse(
        { detail: { reason: 'exists', path: 'README.md', dir: false } },
        false,
        409,
      ),
    )

    const result = await creator.submitCreate()
    expect(result.reason).toBe('refused')
    // The object the server answers with is phrased for the input, from the
    // reason the server gave — no local guess about the name.
    expect(tree.state.editing.error).toBe('README.md is already a file on disk')
    expect(notices.noticeForPath('README.md').text).toBe(
      'Could not create README.md: README.md is already a file on disk',
    )
  })

  it('success re-lists the parent, re-reads the marks, and opens a new file', async () => {
    stubListings({
      '': [{ name: 'src', path: 'src', dir: true, children: true }],
      src: [{ name: 'a.py', path: 'src/a.py', dir: false }],
    })
    await tree.load('')
    await tree.load('src')
    tree.startCreate('src', false)
    tree.updateCreateValue('new.py')

    vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
      if (String(url) === '/api/fs/create') {
        return jsonResponse({ path: 'src/new.py', dir: false, size: 0, mtime: 5000 })
      }
      if (String(url).startsWith('/api/fs/tree?path=src')) {
        return jsonResponse({
          path: 'src',
          workspace: '/w',
          entries: [
            { name: 'a.py', path: 'src/a.py', dir: false },
            { name: 'new.py', path: 'src/new.py', dir: false, size: 0, mtime: 5000 },
          ],
          truncated: false,
        })
      }
      if (String(url).startsWith('/api/fs/status')) {
        return jsonResponse({ repo: true, entries: [{ path: 'src/new.py', kind: 'untracked', comparable: false }] })
      }
      if (String(url).startsWith('/api/fs/file')) {
        return jsonResponse({ path: 'src/new.py', text: '', size: 0, mtime: 5000, language: 'python' })
      }
      return jsonResponse({ path: '', workspace: '/w', entries: [], truncated: false })
    })

    const result = await creator.submitCreate()
    expect(result.ok).toBe(true)
    // The listing the row appears in, and the marks — one each, not a whole-tree
    // refresh.
    expect(tree.state.nodes['src'].entries.map((entry) => entry.name)).toEqual([
      'a.py',
      'new.py',
    ])
    expect(git.state.entries).toHaveLength(1)
    // The empty buffer is a tab immediately, clean, with the create's mtime as
    // its save token.
    const record = files.recordFor('src/new.py')
    expect(record).not.toBeNull()
    expect(record.text).toBe('')
    expect(record.dirty).toBe(false)
    expect(record.mtime).toBe(5000)
    expect(tree.state.editing).toBeNull()
  })

  it('a new folder is expanded and focused, and opens no buffer', async () => {
    stubListing('', [])
    tree.startCreate('', true)
    tree.updateCreateValue('scratch')
    vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url) === '/api/fs/create') {
        return jsonResponse({ path: 'scratch', dir: true, size: 0, mtime: 4000 })
      }
      if (String(url).startsWith('/api/fs/status')) return jsonResponse({ repo: false, entries: [] })
      return jsonResponse({
        path: '',
        workspace: '/w',
        entries: [{ name: 'scratch', path: 'scratch', dir: true, children: false, mtime: 4000 }],
        truncated: false,
      })
    })

    const result = await creator.submitCreate()
    expect(result.ok).toBe(true)
    expect(tree.state.expanded['scratch']).toBe(true)
    expect(tree.state.focused).toBe('scratch')
    expect(files.state.open).toHaveLength(0)
  })

  it('a second Enter while one is in flight issues one request', async () => {
    stubListing('', [])
    tree.startCreate('', false)
    tree.updateCreateValue('one.py')

    let resolveFetch
    const pending = new Promise((resolve) => {
      resolveFetch = resolve
    })
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => pending)

    const first = creator.submitCreate()
    const second = creator.submitCreate()
    resolveFetch(jsonResponse({ path: 'one.py', dir: false, size: 0, mtime: 1 }))
    const [a, b] = await Promise.all([first, second])

    expect(a.ok).toBe(true)
    expect(b.ok).toBe(true) // joined, not duplicated
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url) === '/api/fs/create'),
    ).toHaveLength(1)
  })
})

describe('useWorkspaceTree: the create input and refreshBranch', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // The components ask the *create flow* for an input, so that is what the test
  // reaches for — a handshake the flow does not return is a runtime TypeError in
  // the browser, not something `tree.startCreate` would notice.
  it('startCreate opens the parent, takes the focus and puts a pending name in state', async () => {
    stubListings({
      '': [{ name: 'src', path: 'src', dir: true, children: true }],
      src: [{ name: 'a.py', path: 'src/a.py', dir: false }],
    })
    await tree.load('')
    await tree.load('src')
    expect(creator.startCreate('src', true)).toBe(true)

    expect(tree.state.expanded['src']).toBe(true)
    expect(tree.state.focused).toBe('src')
    // `mode` says which of the two flows owns the one name input (see
    // useEditorRename); for `startCreate` it is always 'create'.
    expect(tree.state.editing).toEqual({
      mode: 'create',
      parent: 'src',
      isDir: true,
      value: '',
      error: null,
    })
  })

  it('a typing round-trip clears the previous refusal', () => {
    tree.startCreate('', false)
    tree.state.editing.error = 'nope'
    tree.updateCreateValue('a')
    expect(tree.state.editing.value).toBe('a')
    expect(tree.state.editing.error).toBeNull()
  })

  it('Escape-style cancel drops the input', () => {
    tree.startCreate('', false)
    tree.cancelCreate()
    expect(tree.state.editing).toBeNull()
  })

  // The chords belong to the tree's keyboard model (which key, in which folder);
  // the answer belongs to the create flow, which *registers* itself as the
  // handler — so the read model never imports the write half.
  it('n and N name the kind and the folder of the focused row', async () => {
    const seen = []
    registerCreateChord((action, folder) => seen.push([action, folder]))
    try {
      stubListing('', [{ name: 'dir', path: 'dir', dir: true, children: false }])
      await tree.load('')

      tree.state.focused = ''
      tree.handleKey({ key: 'n', shiftKey: false, preventDefault: () => {} })
      tree.handleKey({ key: 'N', shiftKey: true, preventDefault: () => {} })

      // A file row: the new thing goes next to that file, not "inside" it.
      stubListing('', [{ name: 'a.py', path: 'a.py', dir: false }])
      tree.state.focused = 'a.py'
      tree.handleKey({ key: 'n', shiftKey: false, preventDefault: () => {} })
    } finally {
      registerCreateChord(null)
    }

    expect(seen).toEqual([
      ['new-file', ''],
      ['new-folder', ''],
      ['new-file', ''],
    ])
    expect(tree.state.editing).toBeNull() // the model asks, it does not create
  })

  it('the create flow is what answers those chords', async () => {
    useEditorCreate() // registers the chord handler, as the components do
    stubListings({ '': [{ name: 'src', path: 'src', dir: true, children: true }] })
    await tree.load('')
    tree.state.focused = 'src'
    tree.handleKey({ key: 'N', shiftKey: true, preventDefault: () => {} })

    expect(tree.state.editing).toEqual({
      mode: 'create',
      parent: 'src',
      isDir: true,
      value: '',
      error: null,
    })
    expect(tree.state.expanded['src']).toBe(true) // the row it goes into is opened
  })

  it('a chord is ignored while the name input owns the keyboard', () => {
    tree.startCreate('', false)
    tree.updateCreateValue('pa')
    const event = { key: 'n', shiftKey: false, preventDefault: () => {} }
    tree.handleKey(event)
    // The typed text is untouched and the preventDefault was never called: the
    // input owns the keystroke.
    expect(tree.state.editing.value).toBe('pa')
    expect(tree.state.editing.parent).toBe('')
  })

  it('the arrow and Enter model is unchanged by the new chords', async () => {
    stubListing('', [
      { name: 'a.py', path: 'a.py', dir: false },
      { name: 'dir', path: 'dir', dir: true, children: false },
    ])
    await tree.load('')
    expect(tree.state.focused).toBe('')
    tree.handleKey({ key: 'ArrowDown', shiftKey: false, preventDefault: () => {} })
    expect(tree.state.focused).toBe('a.py')
    tree.handleKey({ key: 'ArrowDown', shiftKey: false, preventDefault: () => {} })
    expect(tree.state.focused).toBe('dir')
    const opened = { key: 'Enter', shiftKey: false, preventDefault: () => {} }
    tree.handleKey(opened)
    expect(tree.state.expanded['dir']).toBe(true)
  })

  it('refreshBranch re-lists a loaded folder only', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url) =>
      jsonResponse({ path: '', workspace: '/w', entries: [], truncated: false }),
    )
    await tree.load('') // root: loaded
    const before = fetchMock.mock.calls.length

    await tree.refreshBranch('')
    expect(fetchMock.mock.calls.length).toBe(before + 1)

    await tree.refreshBranch('never-loaded')
    expect(fetchMock.mock.calls.length).toBe(before + 1)
  })
})
