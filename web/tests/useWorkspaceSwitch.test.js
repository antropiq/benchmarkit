import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useGitStatus } from '../src/composables/useGitStatus'

// ---------------------------------------------------------------------------
// A working-location switch, seen from the client: everything that holds a
// workspace-relative path has to let go of it.
//
// The bug this suite exists for: the tree re-listed the new root while the tab
// strip still showed a file of the previous one, so the pane and the tree
// disagreed about which checkout was open. Every path the app holds is relative
// to the root, so `src/a.py` here and `src/a.py` there are two different files
// — a switch is therefore *not* a refresh. `refresh()` re-fetches the paths it
// already knows; `reRoot()` forgets them, and the tabs built on them close.
//
// The server's own rules for the switch (the deny-list, the 409 while a run is
// streaming, the realpath canonicalisation) are asserted in tests/test_api.py.
// ---------------------------------------------------------------------------

const tree = useWorkspaceTree()
const files = useEditorFiles()
const notices = useEditorNotices()
const git = useGitStatus()

const OLD_ROOT = '/w/newproj'
const NEW_ROOT = '/w/benchmarkit'

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// The roots, mutable, because that is what a switch is: the *same* request
// answers differently once the server has moved. `README.md` exists in the old
// tree only, so a tab holding it can only be holding a file of the root that
// was just left.
function stubRoots(state = { root: OLD_ROOT }) {
  vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
    const key = String(url)
    if (key.startsWith('/api/fs/tree')) {
      const path = decodeURIComponent(key.slice('/api/fs/tree?path='.length))
      const entries =
        state.root === OLD_ROOT
          ? [{ name: 'README.md', path: 'README.md', dir: false, size: 8, mtime: 1000 }]
          : [{ name: 'src', path: 'src', dir: true, children: true, mtime: 2000 }]
      return jsonResponse({ path, workspace: state.root, entries, truncated: false })
    }
    if (key.startsWith('/api/fs/file')) {
      return jsonResponse({ path: 'README.md', text: 'old one\n', mtime: 1000, size: 8 })
    }
    if (key.startsWith('/api/fs/status')) {
      return jsonResponse({ repo: false, entries: [] })
    }
    return jsonResponse({})
  })
  return state
}

function resetState() {
  tree.state.nodes = {}
  tree.state.expanded = { '': true }
  tree.state.focused = ''
  tree.state.selected = null
  tree.state.editing = null
  tree.state.error = null
  tree.state.workspace = ''
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  files.state.probeError = null
  files.state.conflict = null
  files.state.conflictDismissedAt = 0
  files.state.closePrompt = null
  files.state.deletePrompt = null
  files.stopFreshnessProbe()
  notices.clear()
  git.state.entries = []
  git.state.repo = false
  git.state.error = null
  git.state.loadedAt = 0
}

describe('useWorkspaceTree.reRoot: the tree lets go of the old root', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('drops the listings, the branches, the focus and the selection', async () => {
    const roots = stubRoots()
    await tree.load('')
    tree.expand('notes') // a branch opened in the old tree
    tree.select({ path: 'README.md', name: 'README.md', dir: false })
    expect(tree.rootLabel()).toBe('newproj')

    roots.root = NEW_ROOT
    await tree.reRoot(NEW_ROOT)

    // Only the new root's own listing survives, and the root row names the new
    // folder rather than one nothing is browsing any more.
    expect(Object.keys(tree.state.nodes)).toEqual([''])
    expect(tree.state.nodes[''].entries.map((entry) => entry.path)).toEqual(['src'])
    expect(tree.state.expanded).toEqual({ '': true })
    expect(tree.state.focused).toBe('')
    expect(tree.state.selected).toBeNull()
    expect(tree.state.workspace).toBe(NEW_ROOT)
    expect(tree.rootLabel()).toBe('benchmarkit')
  })

  it('a half-typed name does not survive the switch', async () => {
    stubRoots()
    await tree.load('')
    tree.startCreate('', false)
    tree.updateCreateValue('notes')
    expect(tree.state.editing?.value).toBe('notes')

    await tree.reRoot(NEW_ROOT)

    // The input renders inside a listing that is gone, and submitting the text
    // would name a file in a folder the user was not typing in.
    expect(tree.state.editing).toBeNull()
  })

  // The switch also races the listings it did not start: a folder fetch that
  // was already out answers *after* the root moved, and its rows belong to the
  // tree that was just left. Applying it would put a file of the old workspace
  // into the new one — the same bug, arriving late.
  it('a late listing of the old root cannot repopulate the new tree', async () => {
    let release = null
    const pending = new Promise((resolve) => {
      release = resolve
    })
    stubRoots()
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      const key = String(url)
      const path = key.startsWith('/api/fs/tree')
        ? decodeURIComponent(key.slice('/api/fs/tree?path='.length))
        : ''
      if (path === 'old') return pending
      return jsonResponse({
        path,
        workspace: NEW_ROOT,
        entries: path === '' ? [{ name: 'src', path: 'src', dir: true, children: true }] : [],
        truncated: false,
      })
    })

    await tree.load('')
    const inFlight = tree.load('old') // answers only after the switch
    await tree.reRoot(NEW_ROOT)
    release(
      jsonResponse({
        path: 'old',
        workspace: NEW_ROOT,
        entries: [{ name: 'ghost.py', path: 'old/ghost.py', dir: false, size: 1, mtime: 1 }],
        truncated: false,
      }),
    )
    await inFlight

    // The new root was listed for real, and the abandoned answer left nothing
    // behind: no entry, and no node claiming that branch was ever loaded.
    expect(fetchMock.mock.calls.filter(([u]) => String(u).includes('/api/fs/tree?path=')).length)
      .toBeGreaterThan(1)
    expect(tree.state.nodes[''].entries.map((entry) => entry.path)).toEqual(['src'])
    expect(tree.entryFor('old/ghost.py')).toBeNull()
    expect(tree.state.nodes['old']).toBeUndefined()
    expect(tree.state.workspace).toBe(NEW_ROOT)
  })
})

describe('the tab strip: every tab of the old root closes', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('closes file buffers and diff tabs alike, leaving an empty pane', async () => {
    stubRoots()
    await files.openFile('README.md')
    await files.openDiff('README.md')
    expect(files.state.open.map((item) => item.key)).toEqual(['README.md', 'diff:README.md'])

    await tree.reRoot(NEW_ROOT)

    // Nothing is left pointing at the previous workspace: not a record, not the
    // active key, and not a notice about one.
    expect(files.state.open).toHaveLength(0)
    expect(files.state.active).toBeNull()
    expect(files.recordFor('README.md')).toBeNull()
    expect(files.diffRecordFor('README.md')).toBeNull()
    expect(notices.state.notices).toHaveLength(0)
  })

  it('the git marks of the previous work tree go with the tabs', async () => {
    stubRoots()
    await files.openFile('README.md')
    git.state.entries = [{ path: 'README.md', kind: 'modified', comparable: true }]
    git.state.repo = true
    git.state.loadedAt = 3
    expect(git.isDiffable('README.md')).toBe(true)

    await tree.reRoot(NEW_ROOT)

    // The old answer is gone rather than kept: a row of the new root must not
    // get its marker, or its "View diff…" item, from a file elsewhere.
    expect(git.state.entries).toEqual([])
    expect(git.state.repo).toBe(false)
    expect(git.isDiffable('README.md')).toBe(false)
  })

  // The one case worth a line of its own: unsaved edits live only in this page,
  // and closing the tab that holds them is the switch's doing, not the user's
  // gesture — so `requestCloseTab`'s confirm does not apply (its Save answer
  // would write to a path of a root that is gone). The loss is reported.
  it('unsaved work is reported, and the open prompts go with the root', async () => {
    stubRoots()
    await files.openFile('README.md')
    files.updateText('README.md', 'mine, never written\n')
    expect(files.dirtyCount.value).toBe(1)
    files.requestDelete({ path: 'README.md', name: 'README.md', dir: false })
    expect(files.state.deletePrompt?.path).toBe('README.md')

    await tree.reRoot(NEW_ROOT)

    expect(files.state.open).toHaveLength(0)
    expect(files.dirtyCount.value).toBe(0)
    // That confirmation would now delete a different folder's file that nobody
    // asked about, so it goes with the root it named.
    expect(files.state.deletePrompt).toBeNull()
    expect(files.state.closePrompt).toBeNull()

    expect(notices.state.notices).toHaveLength(1)
    expect(notices.state.notices[0].tone).toBe('warn')
    expect(notices.state.notices[0].text).toContain('1 tab closed with unsaved changes')
    expect(notices.state.notices[0].text).toContain(NEW_ROOT)
  })
})
