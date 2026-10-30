import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  registerCreateChord,
  registerDeleteChord,
  registerRefreshChord,
  registerRenameChord,
  useWorkspaceTree,
} from '../src/composables/useWorkspaceTree'
import { useEditorCreate } from '../src/composables/useEditorCreate'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorRename } from '../src/composables/useEditorRename'
import { treeMenuItemIds, treeMenuItems } from '../src/lib/tree-menu'

// ---------------------------------------------------------------------------
// The tree as the *place commands are issued from*: one subject (the focused
// row), one keyboard model over it, one menu table per row kind, and cache
// bookkeeping that leaves no listing claiming a row the disk no longer has.
//
// The request halves are not here — create, rename and delete each have their
// own suite. What these tests hold is the tree's own promises: which row a
// command means, which key reaches which handler, which items a row may offer,
// and which cached entries a mutation has to forget.
// ---------------------------------------------------------------------------

const tree = useWorkspaceTree()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

const LISTINGS = {
  '': [
    { name: 'src', path: 'src', dir: true, children: true },
    { name: 'notes.md', path: 'notes.md', dir: false, size: 4, mtime: 1000 },
  ],
  src: [
    { name: 'core', path: 'src/core', dir: true, children: true },
    { name: 'a.py', path: 'src/a.py', dir: false, size: 4, mtime: 1000 },
  ],
  'src/core': [{ name: 'job.py', path: 'src/core/job.py', dir: false, size: 4, mtime: 1000 }],
}

function stubListings(listings = LISTINGS) {
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

function press(event) {
  return { shiftKey: false, ...event, preventDefault: vi.fn() }
}

async function loaded() {
  await tree.load('')
  await tree.load('src')
  await tree.load('src/core')
  tree.expand('src')
  tree.expand('src/core')
}

function resetState() {
  tree.state.nodes = {}
  tree.state.expanded = { '': true }
  tree.state.focused = ''
  tree.state.selected = null
  tree.state.editing = null
  tree.state.error = null
}

const REGISTRARS = [
  ['create', registerCreateChord],
  ['delete', registerDeleteChord],
  ['rename', registerRenameChord],
  ['refresh', registerRefreshChord],
]

// Chord handlers are module singletons (the registries in useWorkspaceTree), so
// a test that installs a spy has to hand the real ones back afterwards — calling
// the flows' composables again re-registers them, exactly as the components do.
function withChords(handlers, run) {
  for (const [name, register] of REGISTRARS) register(handlers[name])
  try {
    return run()
  } finally {
    useEditorCreate()
    useEditorRename()
    useEditorFiles()
    registerRefreshChord(null) // the navigator owns this one; tests do not mount it
  }
}

describe('commandTarget: one subject per tree command', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('is the focused row — its path, name and folder — for a file and a folder', async () => {
    stubListings()
    await loaded()
    tree.state.focused = 'src'
    expect(tree.commandTarget()).toEqual({ path: 'src', name: 'src', dir: true, parent: '' })
    tree.state.focused = 'src/core'
    expect(tree.commandTarget()).toEqual({ path: 'src/core', name: 'core', dir: true, parent: 'src' })
    tree.state.focused = 'src/a.py'
    expect(tree.commandTarget()).toEqual({ path: 'src/a.py', name: 'a.py', dir: false, parent: 'src' })
  })

  it('the root row answers as the workspace itself', async () => {
    stubListings()
    await loaded()
    tree.state.focused = ''
    const target = tree.commandTarget()
    expect(target).toEqual({ path: '', name: 'w', dir: true, parent: '' })
  })

  // A row dropped by a re-list leaves no subject; commands decline instead of
  // acting on a path that is not there.
  it('is null when the focused row is not in a listing', () => {
    tree.state.nodes = { '': { loaded: true, loading: false, entries: [], truncated: false, error: null } }
    tree.state.focused = 'gone.py'
    expect(tree.commandTarget()).toBeNull()
  })

  // The menu and the keyboard agree because the menu *sets* the subject: a
  // right-click is a focus change first, so the chord after it means that row.
  it('follows the row a rename was asked for, which is what the menu does', async () => {
    stubListings()
    await loaded()
    // The menu's path handling lives in the navigator; what the tree guarantees
    // is that opening an input for a row moves the focus to it.
    tree.startRename('src/a.py')
    expect(tree.state.focused).toBe('src/a.py')
    expect(tree.commandTarget().path).toBe('src/a.py')
    tree.cancelCreate()
    tree.startCreate('src', false)
    expect(tree.state.focused).toBe('src')
    expect(tree.commandTarget().dir).toBe(true)
  })
})

describe('the keyboard model', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  function isOpen(path) {
    return !!tree.state.expanded[path]
  }

  // Which key reaches which handler, and with which subject. The tree names the
  // key and the row; the flows behind the registries own the request.
  it('n, N, F2, Delete and r answer with the focused row', async () => {
    stubListings()
    await loaded()
    const seen = []
    await withChords(
      {
        create: (action, folder) => seen.push([action, folder]),
        rename: (path) => seen.push(['rename', path]),
        delete: (entry) => seen.push(['delete', entry.path, !!entry.dir]),
        refresh: (path) => seen.push(['refresh', path]),
      },
      () => {
        // A folder row: it *is* the folder a new thing goes into, and the folder
        // a refresh re-lists.
        tree.state.focused = 'src'
        tree.handleKey(press({ key: 'n' }))
        tree.handleKey(press({ key: 'N', shiftKey: true }))
        tree.handleKey(press({ key: 'F2' }))
        tree.handleKey(press({ key: 'Delete' }))
        tree.handleKey(press({ key: 'r' }))

        // A file row: the new thing goes next to it, and the refresh re-lists
        // the folder that lists it.
        tree.state.focused = 'src/a.py'
        tree.handleKey(press({ key: 'n' }))
        tree.handleKey(press({ key: 'F2' }))
        tree.handleKey(press({ key: 'Delete' }))
        tree.handleKey(press({ key: 'R' }))

        // The root: never renameable, never deletable, but listable.
        tree.state.focused = ''
        tree.handleKey(press({ key: 'F2' }))
        tree.handleKey(press({ key: 'Delete' }))
        tree.handleKey(press({ key: 'r' }))
      },
    )

    expect(seen).toEqual([
      ['new-file', 'src'],
      ['new-folder', 'src'],
      ['rename', 'src'],
      ['delete', 'src', true],
      ['refresh', 'src'],
      ['new-file', 'src'],
      ['rename', 'src/a.py'],
      ['delete', 'src/a.py', false],
      ['refresh', 'src'],
      ['refresh', ''],
    ])
  })

  // A single letter is only ours when no modifier is on it: Ctrl/Alt/Meta make
  // it the browser's or another surface's chord.
  it('a modifier chord is never a tree chord', async () => {
    stubListings()
    await loaded()
    const seen = []
    await withChords({ refresh: (path) => seen.push(path) }, () => {
      tree.state.focused = 'src'
      for (const mod of ['ctrlKey', 'metaKey', 'altKey']) {
        tree.handleKey(press({ key: 'r', [mod]: true }))
      }
    })
    expect(seen).toEqual([])
  })

  // The name input owns the keyboard: a chord that typed into itself is the bug.
  it('every chord is ignored while the name input is on screen', async () => {
    stubListings()
    await loaded()
    tree.state.focused = 'src/a.py'
    tree.startRename('src/a.py')
    const before = tree.state.editing.value

    const seen = []
    await withChords(
      {
        create: () => seen.push('create'),
        rename: () => seen.push('rename'),
        delete: () => seen.push('delete'),
        refresh: () => seen.push('refresh'),
      },
      () => {
        for (const key of ['n', 'N', 'F2', 'Delete', 'r', 'ArrowDown', 'Enter']) {
          const event = press({ key })
          tree.handleKey(event)
          expect(event.preventDefault).not.toHaveBeenCalled()
        }
      },
    )

    expect(seen).toEqual([])
    expect(tree.state.editing.value).toBe(before)
    expect(tree.state.focused).toBe('src/a.py')
  })

  // The rest of the model is unchanged by the new chords: the arrows, the edges,
  // the open/close pair, and Enter/Space as select.
  it('arrows, Home/End, Enter and Space keep their meaning', async () => {
    stubListings()
    await loaded()
    tree.state.focused = ''

    tree.handleKey(press({ key: 'ArrowDown' }))
    expect(tree.state.focused).toBe('src')
    tree.handleKey(press({ key: 'ArrowDown' }))
    expect(tree.state.focused).toBe('src/core')
    tree.handleKey(press({ key: 'ArrowLeft' }))
    expect(isOpen('src/core')).toBe(false)
    tree.handleKey(press({ key: 'ArrowRight' }))
    expect(isOpen('src/core')).toBe(true)
    tree.handleKey(press({ key: 'End' }))
    expect(tree.state.focused).toBe('notes.md')
    tree.handleKey(press({ key: 'Home' }))
    expect(tree.state.focused).toBe('')

    // Enter on a file publishes the selection the pane watches; Space on a
    // folder toggles it instead.
    tree.state.focused = 'notes.md'
    tree.handleKey(press({ key: 'Enter' }))
    expect(tree.state.selected?.path).toBe('notes.md')
    tree.state.focused = 'src'
    tree.handleKey(press({ key: ' ' }))
    expect(isOpen('src')).toBe(false)
  })
})

describe('dropBranch: forgetting a moved subtree', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('removes the branch and its flags, and leaves the rest of the cache', async () => {
    stubListings()
    await loaded()
    // The descendant listing a rename leaves behind: keyed under the old prefix,
    // so nothing can re-point it correctly.
    tree.state.nodes['src/core/deep'] = {
      loaded: true,
      loading: false,
      entries: [],
      truncated: false,
      error: null,
    }
    tree.state.expanded['src/core/deep'] = true
    tree.state.focused = 'src/core/job.py'

    tree.dropBranch('src')

    expect(tree.state.nodes['src']).toBeUndefined()
    expect(tree.state.nodes['src/core']).toBeUndefined()
    expect(tree.state.nodes['src/core/deep']).toBeUndefined()
    expect(tree.state.expanded['src']).toBeUndefined()
    expect(tree.state.expanded['src/core']).toBeUndefined()
    expect(tree.state.expanded['src/core/deep']).toBeUndefined()
    // The rest of the tree is untouched, and nothing was re-listed.
    expect(tree.state.nodes['']).toBeDefined()
    expect(tree.state.expanded['']).toBe(true)
    // A focus inside the dropped branch lands on the row *above* the branch
    // that is going away — the last one that certainly still exists — which is
    // where the rename flow then re-lists.
    expect(tree.state.focused).toBe('')
  })

  // A sibling whose name merely *starts with* the prefix is not in the branch.
  it('leaves a similarly-named sibling alone', async () => {
    stubListings({
      ...LISTINGS,
      '': [...LISTINGS[''], { name: 'src2', path: 'src2', dir: true, children: false }],
      src2: [],
    })
    await loaded()
    await tree.load('src2')
    tree.dropBranch('src')
    expect(tree.state.nodes['src2']).toBeDefined()
  })
})

describe('the children flag: no listing may lie about a folder', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // `tree.py` promises `children: false` never promises an empty branch that
  // then lists something. Creating the first file in a folder breaks that
  // promise *one level above*, unless the listing holding that folder's row is
  // re-listed too.
  it('re-lists the holding listing only when its flag would lie', async () => {
    const fetchMock = stubListings(LISTINGS)
    await loaded()
    const listed = () =>
      fetchMock.mock.calls
        .filter(([url]) => String(url).startsWith('/api/fs/tree'))
        .map(([url]) => decodeURIComponent(String(url)).replace('/api/fs/tree?path=', ''))
    const before = listed().length

    // src/a.py: the row of `src` lives in the root listing, which the mutation
    // re-listed itself, and its flag is not lying.
    await tree.refreshListingOfParent('src/a.py')
    // src/core/job.py: the row of `src/core` is in `src`'s listing and says
    // children: true — honest, so no request.
    await tree.refreshListingOfParent('src/core/job.py')
    expect(listed()).toHaveLength(before)

    // Now make it lie: a folder listed as empty gains its first file.
    tree.state.nodes['src'].entries = [
      { name: 'empty', path: 'src/empty', dir: true, children: false },
    ]
    await tree.refreshListingOfParent('src/empty/new.py')
    expect(listed().slice(before)).toEqual(['src'])
  })

  // The other half of "no listing may lie": a trashed *folder* leaves its own
  // cached listing and expanded flags behind unless the delete forgets them, and
  // re-creating that name would then expand into children that are on no disk.
  it('a trashed folder leaves no cache under its name', async () => {
    stubListings()
    await loaded()
    tree.expand('src')
    tree.state.focused = 'src'

    vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
      if (String(url).startsWith('/api/fs/file') && init?.method === 'DELETE') {
        return jsonResponse({ path: 'src', dir: true, entries: 2, trashedTo: '.nojae/trash/ab/src' })
      }
      return jsonResponse({
        path: '',
        workspace: '/w',
        entries: [{ name: 'notes.md', path: 'notes.md', dir: false, size: 4, mtime: 1000 }],
        truncated: false,
      })
    })

    const result = await useEditorFiles().trashPath('src')
    expect(result.ok).toBe(true)
    expect(tree.state.nodes['src']).toBeUndefined()
    expect(tree.state.nodes['src/core']).toBeUndefined()
    expect(tree.state.expanded['src']).toBeUndefined()
    expect(tree.state.expanded['src/core']).toBeUndefined()
    // The root's listing is the one that was re-listed, and it no longer has the row.
    expect(tree.entryFor('src')).toBeNull()
  })
})

describe('the menu table: exactly the items a row kind can honour', () => {
  // Root, folder, file — and each of the "absent rather than disabled-and-lying"
  // rules. These are the flags WorkspaceNavigator builds from a row's `data-*`
  // attributes, git's answer and the open buffers.
  const retention = { trashRetentionDays: 30 }
  const root = { ...retention, dir: true, isRoot: true, mutable: false }
  const folder = { ...retention, dir: true, isRoot: false, mutable: true }
  const file = { ...retention, dir: false, isRoot: false, mutable: true }

  it('the root offers creating, listing and its own path — never mutating itself', () => {
    expect(treeMenuItemIds(root)).toEqual([
      'search',
      'new-file',
      'new-folder',
      'refresh',
      'copy-path',
    ])
  })

  it('a folder offers creating, mutating, listing and its path', () => {
    expect(treeMenuItemIds(folder)).toEqual([
      'search',
      'new-file',
      'new-folder',
      'rename',
      'delete',
      'refresh',
      'copy-path',
    ])
  })

  it('a file offers opening and its path, and nothing it cannot do', () => {
    expect(treeMenuItemIds(file)).toEqual(['open', 'search', 'rename', 'delete', 'copy-path'])
  })

  it('Preview only for a markdown row, between Open and View diff…', () => {
    // The flag is the row's *name* answered by lib/markdown-preview before any
    // read happens; without it the item is absent, not disabled-and-lying.
    expect(treeMenuItemIds(file)).not.toContain('preview')
    expect(treeMenuItemIds({ ...file, isMarkdown: true })).toEqual([
      'open',
      'search',
      'preview',
      'rename',
      'delete',
      'copy-path',
    ])
  })

  it('View diff… only with a committed side, Close only with an open buffer', () => {
    expect(treeMenuItemIds({ ...file, diffable: true })).toContain('view-diff')
    expect(treeMenuItemIds({ ...file, hasBuffer: true })).toContain('close')
    expect(treeMenuItemIds({ ...file, diffable: true, hasBuffer: true })).toEqual([
      'open',
      'search',
      'view-diff',
      'rename',
      'delete',
      'close',
      'copy-path',
    ])
    expect(
      treeMenuItemIds({ ...file, diffable: true, hasBuffer: true, isMarkdown: true }),
    ).toEqual([
      'open',
      'search',
      'preview',
      'view-diff',
      'rename',
      'delete',
      'close',
      'copy-path',
    ])
    // None of these flags is ever read on a folder or the root: they describe a
    // file, so `scope: 'file'` must exclude them even when a caller sets them.
    expect(
      treeMenuItemIds({
        ...folder,
        diffable: true,
        hasBuffer: true,
        isMarkdown: true,
      }),
    ).toEqual(treeMenuItemIds(folder))
    expect(
      treeMenuItemIds({
        ...root,
        diffable: true,
        hasBuffer: true,
        isMarkdown: true,
      }),
    ).toEqual(treeMenuItemIds(root))
  })

  // The caller opens *no* menu when this answers with nothing (a menu of one
  // useless item is worse than no menu). No row of a real tree is in that case —
  // every kind has an honest minimum — but the table decides that, not a `v-if`,
  // so a caller can always ask. This shape (not a folder, but the root, so not
  // mutable and with no Open) is the leanest one the table can be handed, and
  // even it gets a menu: Search and Copy relative path are both honest for it.
  it('always has a floor, so no real row is left menu-less', () => {
    expect(treeMenuItemIds({ ...retention, dir: false, isRoot: true, mutable: false })).toEqual([
      'search',
      'copy-path',
    ])
  })

  it('every item carries a label and a hint that says what it does', () => {
    for (const flags of [root, folder, file]) {
      for (const item of treeMenuItems(flags)) {
        expect(item.label).toBeTruthy()
        expect(item.title).toBeTruthy()
      }
    }
    const items = treeMenuItems({ ...folder, trashRetentionDays: 45 })
    expect(items.find((item) => item.id === 'delete').title).toContain('45 days')
    expect(items.find((item) => item.id === 'delete').danger).toBe(true)
    expect(items.find((item) => item.id === 'refresh').danger).toBe(false)
  })
})

describe('one name input, two modes', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // The widget (TreeNameInput.vue) reads this one object either way, so the two
  // flows cannot disagree about what is being typed, where the caret is, or
  // which row a refusal lands in.
  it('create and rename share the state, and only the mode and the row differ', async () => {
    stubListings()
    await loaded()

    tree.startCreate('src', false)
    const created = { ...tree.state.editing }
    tree.cancelCreate()
    tree.startRename('src/a.py')
    const renamed = { ...tree.state.editing }

    for (const key of ['parent', 'isDir', 'value', 'error']) {
      expect(Object.keys(renamed)).toContain(key)
      expect(Object.keys(created)).toContain(key)
    }
    expect(created).toEqual({ mode: 'create', parent: 'src', isDir: false, value: '', error: null })
    expect(renamed).toEqual({
      mode: 'rename',
      path: 'src/a.py',
      parent: 'src',
      isDir: false,
      value: 'a.py', // prefilled, so the fix to a refusal is an edit
      error: null,
    })
    // Both sit in the same listing, and both leave the focused row as the
    // subject while the input is up.
    expect(renamed.parent).toBe(created.parent)
    expect(tree.state.focused).toBe('src/a.py')
  })

  // "Validated the same way": the refusal is the input's, not a message
  // elsewhere — same field, same shape, the typed value still editable in both
  // modes. The *content* of the answer differs per mode (a slash is a mistake
  // when creating and a move when renaming), which is the flows' business.
  it('a refusal leaves the same answer shape in both modes', async () => {
    stubListings()
    await loaded()
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({}))

    useEditorCreate().startCreate('src', false)
    tree.updateCreateValue('a/b.py')
    const create = await useEditorCreate().submitCreate()
    const createState = { ...tree.state.editing }

    tree.cancelCreate()
    useEditorRename().startRename('src/a.py')
    tree.updateCreateValue('../escape.py')
    const rename = await useEditorRename().submitRename()
    const renameState = { ...tree.state.editing }

    for (const result of [create, rename]) {
      expect(result.reason).toBe('invalid-name')
    }
    for (const state of [createState, renameState]) {
      expect(state.error).toBeTruthy() // the refusal is in the row…
      expect(state.value).toBeTruthy() // …and what was typed is still editable
    }
    // Neither bad value ever reached the API: only the listing calls were made.
    expect(
      fetchMock.mock.calls.filter(
        ([url, init]) =>
          init?.method === 'POST' &&
          (String(url) === '/api/fs/create' || String(url) === '/api/fs/rename'),
      ),
    ).toHaveLength(0)
  })
})
