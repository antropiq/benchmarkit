import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useEditorMove, validateMoveDestination } from '../src/composables/useEditorMove'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useGitStatus } from '../src/composables/useGitStatus'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// The batch move flow as the client owns it: the shape of the destination field,
// one request for the whole set, the server's refusal in its own words, and above
// all what success does to every holder of every moved path — the open records,
// the active tab key, the tree cache and the git marks.
//
// The route's rules (both ends of the gate, the 409s, the chat-stream refusal)
// are asserted in tests/test_editorfs_write.py; nothing here re-litigates them,
// and in particular nothing here knows the deny-list.

describe('validateMoveDestination', () => {
  it('answers the empty field', () => {
    expect(validateMoveDestination('', { parent: 'src' })).toMatch(/destination folder is needed/i)
    expect(validateMoveDestination('   ', { parent: 'src' })).toMatch(/destination folder is needed/i)
    expect(validateMoveDestination(null, { parent: 'src' })).toMatch(/destination folder is needed/i)
  })

  it('refuses an absolute path and a home-relative one', () => {
    expect(validateMoveDestination('/etc', { parent: 'src' })).toMatch(/relative to the working location/)
    expect(validateMoveDestination('~/notes', { parent: 'src' })).toMatch(/relative to the working location/)
  })

  it('refuses any . or .. segment — stricter than the server, on purpose', () => {
    expect(validateMoveDestination('..', { parent: 'src' })).toMatch(/cannot contain \. or \.\./)
    expect(validateMoveDestination('notes/../archive', { parent: 'notes' })).toMatch(
      /cannot contain \. or \.\./,
    )
    expect(validateMoveDestination('notes/./x', { parent: 'notes' })).toMatch(
      /cannot contain \. or \.\./,
    )
    // A name that merely *starts* with the dot is a folder like any other.
    expect(validateMoveDestination('notes/..hidden', { parent: 'notes' })).toBeNull()
  })

  it('refuses a backslash and a NUL', () => {
    expect(validateMoveDestination('notes\\archive', { parent: 'notes' })).toMatch(/backslash/)
    expect(validateMoveDestination('notes\0', { parent: 'notes' })).toMatch(/that character/)
  })

  // The prefill produces this one at N=1: the field opens holding the sources'
  // own folder, so the Move button starts disabled with the reason already shown.
  it('refuses the destination that is where they already are', () => {
    expect(validateMoveDestination('notes', { parent: 'notes' })).toMatch(/already are/)
    expect(validateMoveDestination('  notes  ', { parent: 'notes' })).toMatch(/already are/)
    // The working location is a real destination for a root child.
    expect(validateMoveDestination('', { parent: '' })).toMatch(/destination folder is needed/)
  })

  it('accepts a nested destination, and a bare name (the server answers that one)', () => {
    expect(validateMoveDestination('notes/archive', { parent: 'notes' })).toBeNull()
    expect(validateMoveDestination('archive', { parent: 'notes' })).toBeNull()
    expect(validateMoveDestination('notes/.nojae/x', { parent: 'notes' })).toBeNull()
  })

  // Whether a folder is *allowed* is the server's rule: nothing here knows
  // `.env`, a locked folder or a hidden one — no copy of the deny-list.
  it('says nothing about a destination the connector policy protects', () => {
    expect(validateMoveDestination('.git', { parent: 'src' })).toBeNull()
    expect(validateMoveDestination('mcp_connectors', { parent: '' })).toBeNull()
  })

  it('has no opinion when the batch has no parent to compare against', () => {
    expect(validateMoveDestination('notes', {})).toBeNull()
  })
})

const mover = useEditorMove()
const tree = useWorkspaceTree()
const files = useEditorFiles()
const notices = useEditorNotices()
const git = useGitStatus()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// The scratch tree: three files in `notes`, and `notes/archive` as the
// destination folder — already there, because this feature never makes one.
function entry(path, dir = false) {
  return {
    name: path.split('/').pop(),
    path,
    dir,
    ...(dir ? { children: true } : { size: 4, mtime: 1000 }),
  }
}

// The scratch tree, as the *disk* answers it: three files in `notes`, and
// `notes/archive` as the destination folder — already there, because this
// feature never makes one. A successful batch move is what flips the answer:
// after it, the three rows are listed under `notes/archive` and not under
// `notes`, so the re-lists and the reveal all read a tree that agrees with the
// payload the route sent.
function diskListing(path, { moved = false } = {}) {
  if (path === '') return [entry('notes', true)]
  if (path === 'notes') {
    const left = moved ? [] : [entry('notes/a.md'), entry('notes/b.md'), entry('notes/c.md')]
    return [...left, entry('notes/archive', true)]
  }
  if (path === 'notes/archive') {
    return moved
      ? [entry('notes/archive/a.md'), entry('notes/archive/b.md'), entry('notes/archive/c.md')]
      : []
  }
  return []
}

// Every request the flow issues goes through this one stub, so the counting
// assertions (two re-lists, one git read, one notice) are about real calls.
// `moveBody` is what the batch route answers.
function stubRoutes(moveBody, { ok = true, status = 200 } = {}) {
  const disk = { moved: false }
  return vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
    const target = String(url)
    if (target === '/api/fs/move-batch' && init?.method === 'POST') {
      if (ok) disk.moved = true
      return jsonResponse(moveBody, ok, status)
    }
    if (target.startsWith('/api/fs/status')) {
      return jsonResponse({ repo: true, entries: [], atToplevel: true })
    }
    if (target.startsWith('/api/fs/tree')) {
      const path = decodeURIComponent(target.slice('/api/fs/tree?path'.length + 1)) || ''
      return jsonResponse({
        path,
        workspace: '/w',
        entries: diskListing(path, disk),
        truncated: false,
      })
    }
    return jsonResponse({ path: 'x', text: 'one\n', mtime: 1000, size: 4 })
  })
}

function listingsFor(mock, path) {
  return mock.mock.calls
    .map(([url]) => String(url))
    .filter((url) => url === `/api/fs/tree?path=${encodeURIComponent(path)}`)
}

function resetState() {
  tree.state.nodes = {}
  tree.state.expanded = { '': true }
  tree.state.focused = ''
  tree.state.selected = null
  tree.state.editing = null
  tree.state.error = null
  tree.state.marked = []
  tree.state.markAnchor = null
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  files.state.conflict = null
  files.state.deletePrompt = null
  notices.clear()
  git.state.entries = []
  git.state.loadedAt = 0
  mover.state.movePrompt = null
}

async function openRecord(path, { text = 'one\n', mtime = 1000 } = {}) {
  vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
    jsonResponse({ path, text, mtime, size: text.length }),
  )
  return files.openFile(path)
}

// The three notes files, as `batchFor` answers them for a row of `notes`.
const THREE = ['notes/a.md', 'notes/b.md', 'notes/c.md']

function startBatch(paths = THREE, parent = 'notes') {
  return mover.startMove({ paths, parent })
}

describe('useEditorMove: the prompt', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // The list is copied at once (D4): the marks may have moved by the time a
  // button is pressed, and the dialog must not end up about a set that no
  // longer exists.
  it('startMove copies the batch and prefills its parent', () => {
    const batch = { paths: ['notes/a.md', 'notes/b.md'], parent: 'notes' }
    expect(startBatch(batch.paths, batch.parent)).toBe(true)
    expect(mover.state.movePrompt.paths).toEqual(['notes/a.md', 'notes/b.md'])
    expect(mover.state.movePrompt.parent).toBe('notes')
    // The prefill is the sources' parent, workspace-relative (D7, D8) — not a
    // bare name and never an absolute path.
    expect(mover.state.movePrompt.value).toBe('notes')
    expect(mover.state.movePrompt.error).toBeNull()
    expect(mover.state.movePrompt.busy).toBe(false)

    tree.state.marked.length = 0
    batch.paths.length = 0
    expect(mover.state.movePrompt.paths).toEqual(['notes/a.md', 'notes/b.md'])
  })

  it('an empty batch opens nothing', () => {
    expect(mover.startMove({ paths: [], parent: 'notes' })).toBe(false)
    expect(mover.startMove(null)).toBe(false)
    expect(mover.state.movePrompt).toBeNull()
  })

  it('cancelMove drops the prompt and issues nothing', async () => {
    const fetchMock = stubRoutes({ moved: [], to_parent: 'notes/archive' })
    startBatch()
    mover.state.movePrompt.value = 'notes/archive'
    mover.cancelMove()
    expect(mover.state.movePrompt).toBeNull()
    const result = await mover.submitMove()
    expect(result.reason).toBe('no-prompt')
    expect(fetchMock.mock.calls).toHaveLength(0)
  })
})

describe('useEditorMove.submitMove: a refusal moves nothing', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // The field answers before the network: a shape the server would refuse
  // anyway costs no request at all.
  it('an invalid destination is answered in the field, with no request', async () => {
    const fetchMock = stubRoutes({ moved: [] })
    startBatch()
    mover.state.movePrompt.value = '../outside'

    const result = await mover.submitMove()
    expect(result.reason).toBe('invalid-destination')
    expect(mover.state.movePrompt.error).toMatch(/\.\./)
    expect(mover.state.movePrompt.value).toBe('../outside') // still editable
    expect(
      fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST'),
    ).toHaveLength(0)
  })

  // R3: the destination folder must already exist, and the server's sentence
  // naming it is what the field shows.
  it('a missing destination keeps the prompt open with the typed value and re-lists nothing', async () => {
    await tree.load('')
    await tree.load('notes')
    const record = await openRecord('notes/a.md')
    tree.state.marked = [...THREE]

    const fetchMock = stubRoutes(
      { detail: 'Parent folder does not exist: notes/archiv' },
      { ok: false, status: 409 },
    )
    startBatch()
    mover.state.movePrompt.value = 'notes/archiv'

    const result = await mover.submitMove()
    expect(result.ok).toBe(false)
    expect(result.reason).toBe('refused')
    // The prompt is still up, holding what was typed: the fix is an edit.
    expect(mover.state.movePrompt.value).toBe('notes/archiv')
    expect(mover.state.movePrompt.error).toBe('Parent folder does not exist: notes/archiv')
    // Nothing moved: no re-point, no re-list, no git re-read, marks intact.
    expect(record.path).toBe('notes/a.md')
    expect(tree.state.marked).toEqual(THREE)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/fs/tree')),
    ).toHaveLength(0)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/fs/status')),
    ).toHaveLength(0)
    // The server's sentence, in the one notice that names both ends.
    expect(notices.state.notices.map((item) => item.tone)).toEqual(['error'])
    expect(notices.state.notices[0].text).toBe(
      'Could not move 3 files → notes/archiv: Parent folder does not exist: notes/archiv',
    )
  })

  // A taken target answers with the {reason, path, dir} object rather than a
  // sentence — read the way `submitRename` reads it.
  it('an occupied target is phrased from the server reason', async () => {
    stubRoutes(
      { detail: { reason: 'exists', path: 'notes/archive/a.md', dir: false } },
      { ok: false, status: 409 },
    )
    startBatch()
    mover.state.movePrompt.value = 'notes/archive'

    await mover.submitMove()
    expect(mover.state.movePrompt.error).toBe('notes/archive/a.md is already a file on disk')
    expect(notices.state.notices[0].text).toContain('is already a file on disk')
    expect(mover.state.movePrompt.busy).toBe(false)
  })

  // The contract both halves agreed on: whole source paths, and a *parent*
  // field the server joins (D11) — never a target computed here.
  it('sends the paths and the destination parent, verbatim and workspace-relative', async () => {
    const fetchMock = stubRoutes({ moved: [], to_parent: 'notes/archive' })
    startBatch()
    mover.state.movePrompt.value = 'notes/archive'

    await mover.submitMove()
    const post = fetchMock.mock.calls.find(
      ([url, init]) => String(url) === '/api/fs/move-batch' && init?.method === 'POST',
    )
    expect(JSON.parse(post[1].body)).toEqual({
      paths: THREE,
      to_parent: 'notes/archive',
    })
  })

  it('a second Enter while one is in flight issues one request', async () => {
    startBatch()
    mover.state.movePrompt.value = 'notes/archive'

    let resolveFetch
    const pending = new Promise((resolve) => {
      resolveFetch = resolve
    })
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
      if (String(url) === '/api/fs/move-batch' && init?.method === 'POST') return pending
      return jsonResponse({})
    })

    const first = mover.submitMove()
    const second = mover.submitMove()
    resolveFetch(jsonResponse({ moved: [], to_parent: 'notes/archive' }))
    const [a, b] = await Promise.all([first, second])

    expect(a.ok).toBe(true)
    expect(b.ok).toBe(true)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url) === '/api/fs/move-batch'),
    ).toHaveLength(1)
  })
})

describe('useEditorMove.submitMove: success reconciles the batch', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  const MOVED = [
    { from: 'notes/a.md', to: 'notes/archive/a.md', mtime: 1712345.0 },
    { from: 'notes/b.md', to: 'notes/archive/b.md', mtime: 1712345.0 },
    { from: 'notes/c.md', to: 'notes/archive/c.md', mtime: 1712345.0 },
  ]

  // The batch's whole point on this side: it touched two listings, so two are
  // re-fetched — whatever N is. A loop that re-lists once per file passes every
  // other assertion in this file, which is why the requests are counted.
  it('re-lists two folders, not one per file', async () => {
    const fetchMock = stubRoutes({ moved: MOVED, to_parent: 'notes/archive' })
    await tree.load('')
    await tree.load('notes')
    await tree.load('notes/archive')
    // Only what the flow itself asks is counted: `refreshBranch` re-lists a
    // folder only when something on screen shows it, so the listings have to be
    // on the cache already — and their own requests are not the batch's.
    fetchMock.mockClear()
    startBatch()
    mover.state.movePrompt.value = 'notes/archive'

    const result = await mover.submitMove()
    expect(result.ok).toBe(true)
    expect(listingsFor(fetchMock, 'notes')).toHaveLength(1)
    expect(listingsFor(fetchMock, 'notes/archive')).toHaveLength(1)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/fs/tree')),
    ).toHaveLength(2)
    // The batch is spent, and confirmed once.
    expect(tree.state.marked).toEqual([])
    expect(tree.state.markAnchor).toBeNull()
    expect(mover.state.movePrompt).toBeNull()
    expect(notices.state.notices.map((item) => item.tone)).toEqual(['ok'])
    expect(notices.state.notices[0].text).toBe('Moved 3 files → notes/archive')
  })

  // The user-visible loop: the rows move, the tab follows, and the buffer is the
  // same buffer — its unsaved work about to go to the *new* path.
  it('re-points an open buffer, keeps it dirty, and reveals the last moved row', async () => {
    stubRoutes({ moved: [] })
    await tree.load('')
    await tree.load('notes')
    await tree.load('notes/archive')
    const record = await openRecord('notes/b.md')
    files.updateText('notes/b.md', 'mine\n')
    expect(files.state.active).toBe('notes/b.md')
    tree.state.marked = [...THREE]

    const fetchMock = stubRoutes({ moved: MOVED, to_parent: 'notes/archive' })
    startBatch()
    mover.state.movePrompt.value = 'notes/archive'

    const result = await mover.submitMove()
    expect(result.ok).toBe(true)
    expect(record.path).toBe('notes/archive/b.md')
    expect(record.key).toBe('notes/archive/b.md')
    expect(record.text).toBe('mine\n')
    expect(record.dirty).toBe(true)
    expect(files.state.active).toBe('notes/archive/b.md')
    // One git re-read for the whole batch — read before the save below, which
    // re-reads them again for its own reason.
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/fs/status')),
    ).toHaveLength(1)
    // The destination is open so the rows that went there are visible, and the
    // eye is sent to the last moved one.
    expect(tree.state.expanded['notes/archive']).toBe(true)
    expect(tree.state.focused).toBe('notes/archive/c.md')

    // The next save goes to the new path, with the buffer's own token.
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'notes/archive/b.md', size: 5, mtime: 2000 }),
    )
    const saved = await files.saveFile('notes/archive/b.md')
    expect(saved.ok).toBe(true)
    const put = globalThis.fetch.mock.calls.find(([, init]) => init?.method === 'PUT')
    expect(JSON.parse(put[1].body)).toEqual({
      path: 'notes/archive/b.md',
      text: 'mine\n',
      ifUnmodifiedSince: 1000,
    })
    expect(record.dirty).toBe(false)
  })

  // The single move is the N=1 case of this flow, not a second one.
  it('a batch of one is the same flow, with a singular notice', async () => {
    await tree.load('')
    await tree.load('notes')
    await tree.load('notes/archive')
    stubRoutes({ moved: [MOVED[0]], to_parent: 'notes/archive' })
    expect(mover.startMove({ paths: ['notes/a.md'], parent: 'notes' })).toBe(true)
    mover.state.movePrompt.value = 'notes/archive'

    const result = await mover.submitMove()
    expect(result.ok).toBe(true)
    expect(notices.state.notices.map((item) => item.text)).toEqual([
      'Moved 1 file → notes/archive',
    ])
  })

  // The prompt carries paths of the root being left, so it closes the way the
  // tree forgets `selected`, `editing` and the marks — through the tree's *list*
  // of re-root subscribers, not a slot of its own.
  it('a working-location switch cancels an open prompt and asks nothing of it', async () => {
    const fetchMock = stubRoutes({ moved: [], to_parent: 'notes/archive' })
    startBatch()
    mover.state.movePrompt.value = 'notes/archive'

    await tree.reRoot('/other/place')
    expect(mover.state.movePrompt).toBeNull()
    expect(
      fetchMock.mock.calls.filter(
        ([url, init]) => String(url) === '/api/fs/move-batch' || init?.method === 'POST',
      ),
    ).toHaveLength(0)
  })
})
