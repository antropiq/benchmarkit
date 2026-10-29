import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import EditorTreeDialogs from '../src/components/EditorTreeDialogs.vue'
import WorkspaceNavigator from '../src/components/WorkspaceNavigator.vue'
import {
  registerCreateChord,
  registerDeleteChord,
  registerRefreshChord,
  registerRenameChord,
  useWorkspaceTree,
} from '../src/composables/useWorkspaceTree'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useChat } from '../src/composables/useChat'
import { useEditorCreate } from '../src/composables/useEditorCreate'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useEditorRename } from '../src/composables/useEditorRename'
import { useGitStatus } from '../src/composables/useGitStatus'

// ---------------------------------------------------------------------------
// The combinations that belong to no single route's suite.
//
// Each write flow has its own file (save, create, rename, notices). What lives
// here is what only appears when two of them are strung together, or when one
// of them meets the layer that is *not* its own: the token's life across a
// rename, a folder rename with a dirty buffer inside it, a delete that swallows
// one, the freshness probe landing on a save in flight, and the tree's chords
// while a dialog holds the keyboard. Those are the shapes a user actually
// performs, and each of them is a bug a per-route suite would pass right over.
//
// The route rules themselves (the gate, the 409s, the streaming refusal) stay
// the server's and are asserted in tests/test_editorfs_write.py; nothing here
// re-litigates them, and nothing here knows the deny-list.
// ---------------------------------------------------------------------------

const files = useEditorFiles()
const tree = useWorkspaceTree()
const renamer = useEditorRename()
const creator = useEditorCreate()
const notices = useEditorNotices()
const git = useGitStatus()
const bench = useBenchmark()
const chat = useChat()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

// A fake disk, because these tests are about a *sequence*: a listing that never
// changed after a create or a rename would let the next step pass on state the
// previous step never produced. One entry per path, kept by the route stub
// below, so the tree re-lists what the mutation claims it wrote.
const SEED = [
  { path: 'src', dir: true, size: null, mtime: 1000 },
  { path: 'src/a.py', dir: false, size: 4, mtime: 1000 },
  { path: 'src/b.py', dir: false, size: 4, mtime: 1000 },
]

function parentOf(path) {
  const index = path.lastIndexOf('/')
  return index === -1 ? '' : path.slice(0, index)
}

function nameOf(path) {
  return path.slice(path.lastIndexOf('/') + 1)
}

// Each write answers a *later* mtime than the one before it (the clock the fake
// disk keeps), and a rename answers the moved entry's own — that asymmetry is
// the thing the loop test is about: the save token survives a move untouched
// and moves on only when bytes are written.
function stubRoutes({ saveAnswer = null, texts = {} } = {}) {
  const disk = new Map(SEED.map((entry) => [entry.path, { ...entry }]))
  let clock = 1000
  const later = () => {
    clock += 1000
    return clock
  }

  const listingFor = (folder) =>
    [...disk.values()]
      .filter((entry) => parentOf(entry.path) === folder)
      .map((entry) => ({
        name: nameOf(entry.path),
        path: entry.path,
        dir: entry.dir,
        mtime: entry.mtime,
        ...(entry.dir ? { children: true } : { size: entry.size }),
      }))

  return vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
    const key = String(url)
    if (key.startsWith('/api/fs/status')) {
      return jsonResponse({ repo: true, entries: [] })
    }
    if (key === '/api/fs/rename' && init?.method === 'POST') {
      const { from, to } = JSON.parse(init.body)
      const entry = disk.get(from)
      const isDir = entry?.dir ?? false
      // The whole subtree travels, as it does on a real filesystem.
      for (const [path, value] of [...disk.entries()]) {
        if (path !== from && !path.startsWith(`${from}/`)) continue
        const moved = to + path.slice(from.length)
        disk.delete(path)
        disk.set(moved, { ...value, path: moved })
      }
      return jsonResponse({
        from,
        to,
        dir: isDir,
        size: entry?.size ?? 4,
        // The moved entry's own mtime: a rename writes no bytes, so an open
        // buffer of it stays fresh and its save token stays valid.
        mtime: disk.get(to).mtime,
      })
    }
    if (key === '/api/fs/create' && init?.method === 'POST') {
      const { path, dir } = JSON.parse(init.body)
      const mtime = later()
      disk.set(path, { path, dir, size: dir ? null : 0, mtime })
      return jsonResponse({ path, dir, size: 0, mtime })
    }
    if (key.startsWith('/api/fs/file')) {
      const path = decodeURIComponent(key.split('?path=')[1] ?? '')
      if (init?.method === 'DELETE') {
        const entry = disk.get(path)
        for (const candidate of [...disk.keys()]) {
          if (candidate === path || candidate.startsWith(`${path}/`)) disk.delete(candidate)
        }
        return jsonResponse({
          path,
          dir: !!entry?.dir,
          entries: 0,
          trashedTo: `.nojae/trash/abc-${nameOf(path)}`,
        })
      }
      if (init?.method === 'PUT') {
        const body = JSON.parse(init.body)
        if (saveAnswer) return jsonResponse({ detail: saveAnswer }, false, 409)
        const mtime = later()
        const entry = disk.get(body.path)
        if (entry) entry.mtime = mtime
        return jsonResponse({ path: body.path, size: 6, mtime })
      }
      return jsonResponse({
        path,
        text: texts[path] ?? 'one\n',
        mtime: disk.get(path)?.mtime ?? 1000,
        size: 4,
      })
    }
    if (key.startsWith('/api/fs/stat')) {
      return jsonResponse({ paths: [] })
    }
    const folder = decodeURIComponent(key.replace('/api/fs/tree?path=', ''))
    return jsonResponse({
      path: folder,
      workspace: '/w',
      entries: listingFor(folder),
      truncated: false,
    })
  })
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
  files.state.conflictDismissedAt = 0
  files.state.closePrompt = null
  files.state.deletePrompt = null
  notices.clear()
  git.state.entries = []
  git.state.loadedAt = 0
  files.stopFreshnessProbe()
  // The probe asks only for a visible pane and an idle WorkBench.
  bench.state.activeTab = 'editor'
  chat.state.phase = 'idle'
}

describe('a buffer the whole way round: create, type, save, rename, save', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // One stub for the whole loop, so the fake clock is continuous and every
  // step reads a disk the previous step actually wrote. The claim is about the
  // save *token*: the create answers one, a save re-baselines on its payload, a
  // rename leaves it alone (no bytes were written), and the next save is
  // accepted *because* of that. A step that re-stamped the number on its own
  // would make the last save look stale.
  it('the token survives a rename, and the second save sends it', async () => {
    const fetchMock = stubRoutes()
    await tree.load('')
    await tree.load('src')

    // Create, through the flow the menu items and the n/N chords both use.
    creator.startCreate('src', false)
    tree.updateCreateValue('new.py')
    expect((await creator.submitCreate()).ok).toBe(true)

    const record = files.recordFor('src/new.py')
    expect(record).not.toBeNull()
    expect(record.dirty).toBe(false)
    const createdToken = record.mtime

    files.updateText('src/new.py', 'mine\n')
    expect(record.dirty).toBe(true)
    expect((await files.saveFile('src/new.py')).ok).toBe(true)
    expect(record.dirty).toBe(false)
    const savedToken = record.mtime
    expect(savedToken).toBeGreaterThan(createdToken)

    // The rename, through the flow the menu and F2 both use.
    renamer.startRename('src/new.py')
    tree.updateCreateValue('moved.py')
    expect((await renamer.submitRename()).ok).toBe(true)

    const moved = files.recordFor('src/moved.py')
    expect(moved).not.toBeNull()
    expect(moved.mtime).toBe(savedToken) // the move wrote no bytes
    expect(moved.dirty).toBe(false)
    expect(moved.text).toBe('mine\n')

    files.updateText('src/moved.py', 'more\n')
    const result = await files.saveFile('src/moved.py')
    const puts = calls(fetchMock, '/api/fs/file')
      .filter(([, init]) => init?.method === 'PUT')
      .map(([, init]) => JSON.parse(init.body))

    expect(result.ok).toBe(true)
    expect(puts).toHaveLength(2)
    expect(puts[1].path).toBe('src/moved.py')
    expect(puts[1].ifUnmodifiedSince).toBe(savedToken)
    expect(files.dirtyCount.value).toBe(0)
    // The row exists under its new name in the re-listed folder, which is what
    // the second rename step of the loop would need.
    expect(tree.entryFor('src/moved.py')).not.toBeNull()
  })

  // A folder rename is the same promise multiplied: every buffer under it moves
  // whether or not the user was looking at it, and each keeps its own edits.
  it('a renamed folder repoints every buffer under it, still dirty, saving to the new path', async () => {
    const fetchMock = stubRoutes()
    await tree.load('')
    await tree.load('src')
    await files.openFile('src/a.py')
    await files.openFile('src/b.py')
    const tokens = files.state.open.map((item) => item.mtime)
    files.updateText('src/a.py', 'dirty a\n')
    files.updateText('src/b.py', 'dirty b\n')

    renamer.startRename('src')
    tree.updateCreateValue('renamed')
    const result = await renamer.submitRename()
    expect(result.ok).toBe(true)

    const paths = files.state.open.map((item) => item.path)
    expect(paths).toEqual(['renamed/a.py', 'renamed/b.py'])
    files.state.open.forEach((item, index) => {
      expect(item.dirty).toBe(true)
      expect(item.stale).toBe(false)
      expect(item.deleted).toBe(false)
      expect(item.mtime).toBe(tokens[index])
      expect(item.text).toContain('dirty')
    })

    const saved = await files.saveAllDirty()
    expect(saved.saved).toEqual(['renamed/a.py', 'renamed/b.py'])
    expect(files.dirtyCount.value).toBe(0)
    const written = calls(fetchMock, '/api/fs/file')
      .filter(([, init]) => init?.method === 'PUT')
      .map(([, init]) => JSON.parse(init.body).path)
    expect(written).toEqual(['renamed/a.py', 'renamed/b.py'])
  })
})

describe('a mutation meeting the layer that is not its own', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // The probe and the save are both readers/writers of one record, and the
  // order of their awaits is not fixed. The `saving` guard has to win: a stat
  // taken before the write landed must not put another writer's text into the
  // buffer and call it the baseline.
  it('a probe landing while a save is in flight leaves the buffer alone', async () => {
    stubRoutes({})
    const record = await files.openFile('src/a.py')
    files.updateText('src/a.py', 'mine\n')

    let resolveSave
    const savePending = new Promise((resolve) => {
      resolveSave = resolve
    })
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
      const key = String(url)
      if (init?.method === 'PUT') return savePending
      // A stat saying the disk moved to a *different* mtime, which is what a
      // silent reload would follow.
      if (key.startsWith('/api/fs/stat')) {
        return jsonResponse({ paths: [{ path: 'src/a.py', exists: true, mtime: 9999, size: 6 }] })
      }
      if (key.startsWith('/api/fs/file')) {
        return jsonResponse({ path: 'src/a.py', text: 'written elsewhere\n', mtime: 9999, size: 18 })
      }
      return jsonResponse({})
    })

    const saving = files.saveFile('src/a.py')
    expect(record.saving).toBe(true)

    await files.probeOpenFiles()
    // Nothing moved: the buffer keeps its text and its old token, and no file
    // read was issued for it (the read below the probe would be a reload).
    expect(record.text).toBe('mine\n')
    expect(record.mtime).toBe(1000)
    expect(record.stale).toBe(false)
    expect(calls(fetchMock, '/api/fs/stat')).toHaveLength(1)
    expect(calls(fetchMock, '/api/fs/file').filter(([, init]) => init?.method === 'GET'))
      .toHaveLength(0)

    // The save then lands, and its own answer is what the record takes.
    resolveSave(jsonResponse({ path: 'src/a.py', size: 5, mtime: 5000 }))
    expect((await saving).ok).toBe(true)
    expect(record.text).toBe('mine\n')
    expect(record.baseline).toBe('mine\n')
    expect(record.dirty).toBe(false)
    expect(record.mtime).toBe(5000)
  })

  // The delete's promise is that unsaved work is *named* before it goes, once:
  // one dialog for one action, not a confirm per buffer.
  it('deleting a folder with dirty buffers warns once and closes every buffer under it', async () => {
    stubRoutes({ mtimes: { 'src/a.py': 1000, 'src/b.py': 1000 } })
    await tree.load('')
    await tree.load('src')
    await files.openFile('src/a.py')
    await files.openFile('src/b.py')
    files.updateText('src/a.py', 'unsaved a\n')
    files.updateText('src/b.py', 'unsaved b\n')

    files.requestDelete({ path: 'src', name: 'src', dir: true })
    expect(files.state.deletePrompt).toMatchObject({ path: 'src', dir: true, dirtyBuffers: 2 })

    stubRoutes({})
    const result = await files.confirmDelete()
    expect(result.ok).toBe(true)
    expect(files.state.open).toHaveLength(0)
    // One confirmation about the folder, not one per buffer.
    expect(notices.state.notices.map((item) => item.tone)).toEqual(['ok'])
    // The notice quotes the *route's* batch, not a client-side guess: the
    // server named `.nojae/trash/abc-src`, and that is the word of record the
    // confirmation shows (the reserved folder is not listable in the tree).
    expect(notices.state.notices[0].text).toContain('.nojae/trash/abc-src')
    // And the work is gone, which is why the dialog asked first.
    expect(files.dirtyCount.value).toBe(0)
  })
})

describe('the tree and the dialogs, side by side', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
    document.body.querySelectorAll('dialog, .tree-context-menu').forEach((el) => el.remove())
  })

  // The keyboard model needs no "a dialog is open" test in `handleKey`, and
  // this says why: the dialogs are not inside the tree container, and a
  // `showModal()` dialog keeps the caret inside itself. So a key pressed while
  // a dialog is on screen is never seen by the tree at all — asserted on the
  // mounted DOM rather than reasoned about.
  it('a key pressed while a dialog is open reaches neither the tree nor a chord', async () => {
    Element.prototype.scrollIntoView = () => {}
    Element.prototype.showModal = function showModal() {
      this.open = true
    }
    Element.prototype.close = function close() {
      this.open = false
    }
    stubRoutes({})
    const host = document.createElement('div')
    document.body.appendChild(host)
    const nav = mount(WorkspaceNavigator, { attachTo: host })
    await flushPromises()
    const dialogs = mount(EditorTreeDialogs, { attachTo: host })
    await flushPromises()

    const seen = []
    registerDeleteChord((entry) => seen.push(entry.path))
    registerRenameChord((path) => seen.push(path))
    registerCreateChord(() => seen.push('create'))
    registerRefreshChord(() => seen.push('refresh'))

    // A folder row has the focus, and the delete confirm is open for it.
    await tree.load('')
    tree.state.focused = 'src'
    files.requestDelete({ path: 'src', name: 'src', dir: true })
    await flushPromises()
    const dialog = host.querySelector('.editor-delete-dialog')
    expect(dialog?.open).toBe(true)

    // The promise the confirmation makes *before* the request exists: the
    // workspace's reserved trash folder, spelled like the server's, and said to
    // be a place the pane cannot open. The exact batch belongs to the route's
    // `trashedTo`, quoted by the notice once the delete answers.
    const keeps = host.querySelector('.editor-delete-keeps')
    expect(keeps.textContent).toContain('.nojae/trash')
    expect(keeps.textContent).toContain('reserved')
    expect(keeps.textContent).not.toContain('logs/trash')

    for (const key of ['n', 'N', 'F2', 'Delete', 'r', 'ArrowDown']) {
      dialog.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }))
    }
    await flushPromises()

    expect(seen).toEqual([])
    expect(tree.state.focused).toBe('src')
    expect(tree.state.editing).toBeNull()
    expect(files.state.deletePrompt?.path).toBe('src') // still being answered

    // Control: the same key *does* reach the tree from inside the tree, so the
    // silence above is the DOM, not a chord that stopped working.
    const treeEl = nav.element.querySelector('.editor-nav-tree')
    treeEl.dispatchEvent(new KeyboardEvent('keydown', { key: 'F2', bubbles: true, cancelable: true }))
    await flushPromises()
    expect(seen).toEqual(['src'])

    files.cancelDelete()
    registerDeleteChord(null)
    registerRenameChord(null)
    registerCreateChord(null)
    registerRefreshChord(null)
    nav.unmount()
    dialogs.unmount()
    host.remove()
  })
})

function calls(mock, prefix) {
  return mock.mock.calls.filter(([url]) => String(url).startsWith(prefix))
}
