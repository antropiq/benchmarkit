import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useEditorRename, validateRenameTarget } from '../src/composables/useEditorRename'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useGitStatus } from '../src/composables/useGitStatus'
import { registerRenameChord, useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// The rename/move flow as the client owns it: the shape of a typed target
// (per mode — a rename may name a *path*), one request per submitted value, the
// server's refusal shown in its own words, and above all what success does to
// every holder of the old path: the open records, the active tab key, the diff
// tab's key and the tree. The route's rules — both ends of the gate, the 409s,
// the streaming refusal — are asserted in tests/test_editorfs_write.py; nothing
// here re-litigates them, and in particular nothing here knows the deny-list.

const renamer = useEditorRename()
const tree = useWorkspaceTree()
const files = useEditorFiles()
const notices = useEditorNotices()
const git = useGitStatus()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// Folder listings through the tree's own cache, so `entryFor`/`startRename` and
// the re-list after a move read real state.
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

const TREE = {
  '': [
    { name: 'src', path: 'src', dir: true, children: true },
    { name: 'src2', path: 'src2', dir: true, children: false },
  ],
  src: [{ name: 'a.py', path: 'src/a.py', dir: false, size: 4, mtime: 1000 }],
  src2: [],
}

async function openRecord(path, { text = 'one\n', mtime = 1000 } = {}) {
  vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
    jsonResponse({ path, text, mtime, size: text.length }),
  )
  return files.openFile(path)
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
  files.state.conflict = null
  notices.clear()
  git.state.entries = []
  git.state.loadedAt = 0
}

describe('validateRenameTarget: the shape half, per mode', () => {
  it('a bare name stays in the row folder, a path is taken as given', () => {
    expect(validateRenameTarget('b.py', { from: 'src/a.py', parent: 'src' })).toBeNull()
    expect(validateRenameTarget('src2/a.py', { from: 'src/a.py', parent: 'src' })).toBeNull()
  })

  // The rule that separates the two modes: a create forbids the slash, a rename
  // *needs* it — that is how a move is typed.
  it('allows a slash where validateCreateName forbids one', async () => {
    const { validateCreateName } = await import('../src/composables/useEditorCreate')
    expect(validateCreateName('src2/a.py')).toMatch(/slash/)
    expect(validateRenameTarget('src2/a.py', { from: 'src/a.py', parent: 'src' })).toBeNull()
  })

  it('still refuses the shapes that cannot be a path', () => {
    const from = 'src/a.py'
    expect(validateRenameTarget('', { from, parent: 'src' })).toMatch(/name is needed/)
    expect(validateRenameTarget('../escape.py', { from, parent: 'src' })).toMatch(/\.\./)
    expect(validateRenameTarget('src/../x.py', { from, parent: 'src' })).toMatch(/\.\./)
    expect(validateRenameTarget('/etc/x', { from, parent: 'src' })).toMatch(/relative/)
    expect(validateRenameTarget('~/x', { from, parent: 'src' })).toMatch(/relative/)
    expect(validateRenameTarget('a\\b', { from, parent: 'src' })).toMatch(/backslash/)
  })

  it('refuses the no-ops locally, saving the round trip', () => {
    expect(validateRenameTarget('a.py', { from: 'src/a.py', parent: 'src' })).toMatch(/already/)
    expect(validateRenameTarget('src/a.py', { from: 'src/a.py', parent: 'src' })).toMatch(/already/)
    expect(validateRenameTarget('src/sub/a.py', { from: 'src', parent: '' })).toMatch(
      /inside itself/,
    )
  })

  // The sibling whose name *starts with* the same string is not a descendant.
  it('allows a similar-named sibling target', () => {
    expect(validateRenameTarget('src2', { from: 'src', parent: '' })).toBeNull()
    expect(validateRenameTarget('src/sub2', { from: 'src/sub', parent: 'src' })).toBeNull()
  })

  // Nothing here knows `.env` or `mcp_connectors/`: that answer is the server's.
  it('says nothing about a target the connector policy protects', () => {
    expect(validateRenameTarget('.env', { from: 'a.py', parent: '' })).toBeNull()
    expect(validateRenameTarget('mcp_connectors/x.py', { from: 'a.py', parent: '' })).toBeNull()
  })
})

describe('useEditorFiles.repointPath: every holder of the old path', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('updates path, key and name of the file record', async () => {
    const record = await openRecord('src/a.py')
    expect(files.repointPath('src/a.py', 'src/b.py')).toBe(1)
    expect(record.path).toBe('src/b.py')
    expect(record.key).toBe('src/b.py')
    expect(record.name).toBe('b.py')
    // The record read back through the reactive state is the same buffer (its
    // identity is compared by key, since Vue hands back a proxy).
    expect(files.recordFor('src/b.py').key).toBe(record.key)
    expect(files.recordFor('src/b.py').text).toBe(record.text)
    expect(files.recordFor('src/a.py')).toBeNull()
    expect(files.state.open).toHaveLength(1)
  })

  it('re-keys the diff tab of the same file and keeps its kind', async () => {
    await openRecord('src/a.py')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'src/a.py', original: 'a\n', modified: 'b\n', status: 'modified' }),
    )
    const diff = await files.openDiff('src/a.py')

    files.repointPath('src/a.py', 'src/b.py')

    expect(diff.kind).toBe('diff')
    expect(diff.key).toBe('diff:src/b.py')
    expect(diff.path).toBe('src/b.py')
    expect(diff.name).toBe('b.py (diff)')
    // The two tabs about one path are still two tabs, not one overwritten.
    expect(files.state.open.map((item) => item.key)).toEqual(['src/b.py', 'diff:src/b.py'])
  })

  it('rewrites the records under a renamed folder, longest prefix first', async () => {
    await openRecord('src/a.py')
    await openRecord('src/core/job.py')
    // A path that merely *starts with* the folder's name is not under it.
    await openRecord('src2/a.py')

    files.repointPath('src', 'web')

    expect(files.state.open.map((item) => item.path)).toEqual(['web/a.py', 'web/core/job.py', 'src2/a.py'])
    expect(files.state.open.map((item) => item.key)).toEqual(['web/a.py', 'web/core/job.py', 'src2/a.py'])
    expect(files.state.open.map((item) => item.name)).toEqual(['a.py', 'job.py', 'a.py'])
  })

  it('state.active follows the key, so the pane keeps its buffer', async () => {
    await openRecord('src/a.py')
    await openRecord('src/core/job.py')
    files.focusTab('src/a.py')
    expect(files.state.active).toBe('src/a.py')

    files.repointPath('src', 'web')

    expect(files.state.active).toBe('web/a.py')
    expect(files.recordForKey(files.state.active).path).toBe('web/a.py')
  })

  it('leaves text, baseline, dirty and mtime untouched', async () => {
    const record = await openRecord('src/a.py', { text: 'one\n', mtime: 1000 })
    files.updateText('src/a.py', 'mine\n')

    files.repointPath('src/a.py', 'src/b.py')

    // The real end-to-end claim: a dirty buffer stays dirty, is not stale, and
    // its save token is still the disk's truth at the new path.
    expect(record.text).toBe('mine\n')
    expect(record.baseline).toBe('one\n')
    expect(record.dirty).toBe(true)
    expect(record.mtime).toBe(1000)
    expect(record.stale).toBe(false)
    expect(record.deleted).toBe(false)
  })

  // The notice stack holds paths too, and the two kinds of notice answer a move
  // differently: a *refusal* is about a buffer the user still holds, so it
  // follows it to the new path (and the read-only badge can still quote it); a
  // *confirmation* names where a past mutation happened, a path the move just
  // un-named, so it goes away rather than lying.
  it('moves a refusal to the new path, and drops a confirmation of the old one', async () => {
    await openRecord('src/a.py')
    notices.notice('error', 'Could not save src/a.py: refused', 'src/a.py')
    notices.notice('ok', 'Created src/a.py', 'src/a.py')

    files.repointPath('src/a.py', 'src/b.py')

    expect(notices.noticeForPath('src/b.py').text).toBe('Could not save src/a.py: refused')
    expect(notices.noticeForPath('src/a.py')).toBeNull()
    expect(notices.state.notices.map((item) => item.tone)).toEqual(['error'])
  })

  it('ignores a no-op and unrelated records', async () => {
    await openRecord('src/a.py')
    expect(files.repointPath('src/a.py', 'src/a.py')).toBe(0)
    expect(files.repointPath('other', 'elsewhere')).toBe(0)
    expect(files.recordFor('src/a.py').key).toBe('src/a.py')
  })
})

describe('useEditorRename: the request and what success does to the tree', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // The input opens where the row is, prefilled with its name: the fix to a
  // refusal is an edit, and the caret starts on the part people change.
  it('startRename prefills the current name in the row listing', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    expect(renamer.startRename('src/a.py')).toBe(true)
    expect(tree.state.editing).toEqual({
      mode: 'rename',
      path: 'src/a.py',
      parent: 'src',
      isDir: false,
      value: 'a.py',
      error: null,
    })
    expect(tree.state.expanded['src']).toBe(true)
  })

  it('submitting a bare name sends the joined target path', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('b.py')

    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url) === '/api/fs/rename') {
        return jsonResponse({ from: 'src/a.py', to: 'src/b.py', dir: false, size: 4, mtime: 1000 })
      }
      if (String(url).startsWith('/api/fs/status')) return jsonResponse({ repo: false, entries: [] })
      return jsonResponse({
        path: 'src',
        workspace: '/w',
        entries: [{ name: 'b.py', path: 'src/b.py', dir: false, size: 4, mtime: 1000 }],
        truncated: false,
      })
    })

    const result = await renamer.submitRename()
    expect(result.ok).toBe(true)
    const post = fetchMock.mock.calls.find(([url, init]) => String(url) === '/api/fs/rename' && init?.method === 'POST')
    expect(JSON.parse(post[1].body)).toEqual({ from: 'src/a.py', to: 'src/b.py' })
  })

  it('a typed path is sent as the move it means', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('src2/a.py')

    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url) === '/api/fs/rename') {
        return jsonResponse({ from: 'src/a.py', to: 'src2/a.py', dir: false, size: 4, mtime: 1000 })
      }
      if (String(url).startsWith('/api/fs/status')) return jsonResponse({ repo: false, entries: [] })
      return jsonResponse({ path: '', workspace: '/w', entries: TREE[''], truncated: false })
    })

    const result = await renamer.submitRename()
    expect(result.ok).toBe(true)
    const post = fetchMock.mock.calls.find(([url]) => String(url) === '/api/fs/rename')
    expect(JSON.parse(post[1].body)).toEqual({ from: 'src/a.py', to: 'src2/a.py' })
  })

  it('an invalid target is answered in the row, with no request at all', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('../escape.py')

    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({}))
    const result = await renamer.submitRename()
    expect(result.reason).toBe('invalid-name')
    expect(tree.state.editing.error).toMatch(/\.\./)
    expect(tree.state.editing.value).toBe('../escape.py') // still editable
    expect(
      fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST'),
    ).toHaveLength(0)
  })

  it('a refusal keeps every record pointing at the old path', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    const record = await openRecord('src/a.py')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('.env')

    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse(
        {
          detail:
            "Path is refused by the connector path policy: a secrets/config file " +
            "(its name starts with '.env') (connectors never write to it)",
        },
        false,
        422,
      ),
    )
    const result = await renamer.submitRename()

    expect(result.reason).toBe('refused')
    expect(record.path).toBe('src/a.py')
    expect(record.key).toBe('src/a.py')
    // The refusal is the server's sentence, in the row, and the input is open.
    expect(tree.state.editing.error).toContain('connector path policy')
    // The same sentence in the notice stack, naming both ends of the move that
    // did not happen — keyed by the *target* the request carried, not the row.
    expect(notices.noticeForPath('src/.env').text).toBe(
      'Could not rename src/a.py → src/.env: ' +
        "Path is refused by the connector path policy: a secrets/config file " +
        "(its name starts with '.env') (connectors never write to it)",
    )
  })

  it('a 409 on a taken target is phrased from the server reason', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('taken.py')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ detail: { reason: 'exists', path: 'src/taken.py', dir: false } }, false, 409),
    )

    await renamer.submitRename()
    expect(tree.state.editing.error).toBe('src/taken.py is already a file on disk')
  })

  // The user-visible loop in one assertion: the row moves, the tab follows, and
  // the buffer is the same buffer.
  it('success re-points the buffer, re-lists the folder and reveals the row', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    const record = await openRecord('src/a.py')
    files.updateText('src/a.py', 'mine\n')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('b.py')

    vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url) === '/api/fs/rename') {
        return jsonResponse({ from: 'src/a.py', to: 'src/b.py', dir: false, size: 4, mtime: 1000 })
      }
      if (String(url).startsWith('/api/fs/status')) {
        return jsonResponse({ repo: true, entries: [{ path: 'src/b.py', kind: 'renamed', comparable: true }] })
      }
      return jsonResponse({
        path: 'src',
        workspace: '/w',
        entries: [{ name: 'b.py', path: 'src/b.py', dir: false, size: 4, mtime: 1000 }],
        truncated: false,
      })
    })

    const result = await renamer.submitRename()
    expect(result.ok).toBe(true)
    expect(record.path).toBe('src/b.py')
    expect(record.dirty).toBe(true)
    expect(files.state.active).toBe('src/b.py')
    expect(tree.state.editing).toBeNull()
    // The row the eye is sent to exists in the listing, and the marks were re-read.
    expect(tree.entryFor('src/b.py')).not.toBeNull()
    expect(tree.state.focused).toBe('src/b.py')
    // The marks were re-read: joining the request the flow fired gives its
    // answer, so the assertion does not race a `void`ed promise.
    await git.load()
    expect(git.state.entries).toHaveLength(1)
  })

  it('a renamed folder is dropped from the cache and re-opened under its new name', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    tree.expand('src')
    // A stale descendant entry is exactly what `dropBranch` exists to remove.
    tree.state.nodes['src/old'] = { loaded: true, loading: false, entries: [], truncated: false, error: null }
    tree.state.expanded['src/old'] = true

    renamer.startRename('src')
    tree.updateCreateValue('web')
    vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url) === '/api/fs/rename') {
        return jsonResponse({ from: 'src', to: 'web', dir: true, size: 0, mtime: 1000 })
      }
      if (String(url).startsWith('/api/fs/status')) return jsonResponse({ repo: false, entries: [] })
      if (String(url).startsWith('/api/fs/tree?path=web')) {
        return jsonResponse({
          path: 'web',
          workspace: '/w',
          entries: [{ name: 'a.py', path: 'web/a.py', dir: false }],
          truncated: false,
        })
      }
      return jsonResponse({
        path: '',
        workspace: '/w',
        entries: [{ name: 'web', path: 'web', dir: true, children: true }],
        truncated: false,
      })
    })

    const result = await renamer.submitRename()
    expect(result.ok).toBe(true)
    expect(tree.state.nodes['src']).toBeUndefined()
    expect(tree.state.nodes['src/old']).toBeUndefined()
    expect(tree.state.expanded['src']).toBeUndefined()
    expect(tree.state.expanded['src/old']).toBeUndefined()
    expect(tree.state.expanded['web']).toBe(true)
    expect(tree.state.focused).toBe('web')
    // A folder is never published as a file selection: that would ask the pane
    // to open a folder.
    expect(tree.state.selected).toBeNull()
  })

  // The subtree rule ed-07 states: a folder the user was looking at re-appears
  // with its children; one they had left closed stays closed.
  it('a folder that was closed stays closed under its new name', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    // Listed but *collapsed* — the case the rule is about: its children are not
    // on screen, so they must not be pushed open by a rename.
    tree.collapse('src')

    renamer.startRename('src')
    tree.updateCreateValue('web')
    vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url) === '/api/fs/rename') {
        return jsonResponse({ from: 'src', to: 'web', dir: true, size: 0, mtime: 1000 })
      }
      if (String(url).startsWith('/api/fs/status')) return jsonResponse({ repo: false, entries: [] })
      return jsonResponse({
        path: '',
        workspace: '/w',
        entries: [{ name: 'web', path: 'web', dir: true, children: true }],
        truncated: false,
      })
    })

    const result = await renamer.submitRename()
    expect(result.ok).toBe(true)
    expect(tree.state.expanded['web']).toBeUndefined()
    expect(tree.state.focused).toBe('web')
    expect(tree.state.nodes['web']).toBeUndefined() // nothing was listed for it
  })

  // The row a move can sort off-screen: the renamed file is not in the cached
  // listing under either old or new name, and `reveal`'s one re-list is what
  // puts it back — no second mechanism for finding a moved row.
  it('a move the cached listing knows nothing about is found by the reveal re-list', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('src/b-name-later.py')

    vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url) === '/api/fs/rename') {
        return jsonResponse({ from: 'src/a.py', to: 'src/b-name-later.py', dir: false, size: 4, mtime: 1000 })
      }
      if (String(url).startsWith('/api/fs/status')) return jsonResponse({ repo: false, entries: [] })
      // The re-list of `src` after the move still does not carry the new name
      // (stale on the server's side too); it is *reveal's* forced re-list that
      // answers it, one call later.
      if (String(url).startsWith('/api/fs/tree?path=src')) {
        const calls = listSrcCalls()
        return jsonResponse({
          path: 'src',
          workspace: '/w',
          entries:
            calls <= 1
              ? [{ name: 'a.py', path: 'src/a.py', dir: false, size: 4, mtime: 1000 }]
              : [{ name: 'b-name-later.py', path: 'src/b-name-later.py', dir: false, size: 4, mtime: 1000 }],
          truncated: false,
        })
      }
      return jsonResponse({ path: '', workspace: '/w', entries: TREE[''], truncated: false })
    })
    function listSrcCalls() {
      return globalThis.fetch.mock.calls.filter(([url]) =>
        String(url).startsWith('/api/fs/tree?path=src'),
      ).length
    }

    const result = await renamer.submitRename()
    expect(result.ok).toBe(true)
    expect(tree.entryFor('src/b-name-later.py')).not.toBeNull()
    expect(tree.state.focused).toBe('src/b-name-later.py')
  })

  it('the F2 chord reaches this flow through the tree, not the other way round', () => {
    stubListings(TREE)
    tree.state.nodes = {}
    tree.state.focused = 'x.py'
    const seen = []
    registerRenameChord((path) => seen.push(path))
    try {
      tree.handleKey({ key: 'F2', preventDefault: () => {} })
      // The root is never a candidate.
      tree.state.focused = ''
      tree.handleKey({ key: 'F2', preventDefault: () => {} })
    } finally {
      registerRenameChord((path) => renamer.startRename(path))
    }
    expect(seen).toEqual(['x.py'])
  })

  it('a rename flow answers the F2 chord with an open input', async () => {
    stubListings({ '': [{ name: 'a.py', path: 'a.py', dir: false }], src: [] })
    await tree.load('')
    tree.state.focused = 'a.py'
    tree.handleKey({ key: 'F2', preventDefault: () => {} })
    expect(tree.state.editing).toEqual({
      mode: 'rename',
      path: 'a.py',
      parent: '',
      isDir: false,
      value: 'a.py',
      error: null,
    })
  })

  it('a second Enter while one is in flight issues one request', async () => {
    stubListings(TREE)
    await tree.load('')
    await tree.load('src')
    renamer.startRename('src/a.py')
    tree.updateCreateValue('b.py')

    let resolveFetch
    const pending = new Promise((resolve) => {
      resolveFetch = resolve
    })
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => pending)

    const first = renamer.submitRename()
    const second = renamer.submitRename()
    resolveFetch(jsonResponse({ from: 'src/a.py', to: 'src/b.py', dir: false, size: 4, mtime: 1 }))
    const [a, b] = await Promise.all([first, second])

    expect(a.ok).toBe(true)
    expect(b.ok).toBe(true)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url) === '/api/fs/rename'),
    ).toHaveLength(1)
  })

  it('submitting a create-mode input never reaches the rename route', async () => {
    tree.state.editing = { mode: 'create', parent: '', isDir: false, value: 'x.py', error: null }
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({}))
    const result = await renamer.submitRename()
    expect(result.reason).toBe('no-input')
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url) === '/api/fs/rename'),
    ).toHaveLength(0)
  })
})
