import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import WorkspaceNavigator from '../src/components/WorkspaceNavigator.vue'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorMove } from '../src/composables/useEditorMove'
import { useGitStatus } from '../src/composables/useGitStatus'
import { useWorkspaceSearch } from '../src/composables/useWorkspaceSearch'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'
import { treeMenuItems } from '../src/lib/tree-menu'

// The menu as the user meets it: right-click a row of each kind and read out
// what the tree offered. The item *table* is asserted in
// useWorkspaceTreeCommands.test.js; what this file holds is the navigator's
// half — that it builds that table from the row's own attributes, that it
// refuses to open an empty one, and that the row it was opened on becomes the
// tree's subject, so an item and a key press act on the same row.

const tree = useWorkspaceTree()
const files = useEditorFiles()
const git = useGitStatus()
const search = useWorkspaceSearch()
const move = useEditorMove()

const LISTINGS = {
  // Two files of the root folder, so a test can build a legal two-row batch
  // (R1: one parent) without inventing a second listing.
  '': [
    { name: 'src', path: 'src', dir: true, children: true },
    { name: 'notes.md', path: 'notes.md', dir: false, size: 4, mtime: 1000 },
    { name: 'todo.txt', path: 'todo.txt', dir: false, size: 4, mtime: 1000 },
  ],
  src: [{ name: 'a.py', path: 'src/a.py', dir: false, size: 4, mtime: 1000 }],
}

function jsonResponse(body) {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) })
}

function stubFetch() {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
    const key = String(url)
    if (key.startsWith('/api/fs/status')) {
      return jsonResponse({
        repo: true,
        entries: [
          { path: 'notes.md', kind: 'modified', comparable: true },
          { path: 'src/a.py', kind: 'untracked', comparable: false },
        ],
      })
    }
    for (const [path, entries] of Object.entries(LISTINGS)) {
      if (key === `/api/fs/tree?path=${encodeURIComponent(path)}`) {
        return jsonResponse({ path, workspace: '/work/machine/demo', entries, truncated: false })
      }
    }
    if (key.startsWith('/api/fs/file')) {
      return jsonResponse({
        path: 'notes.md',
        text: '# Title\n',
        language: 'markdown',
        size: 8,
        mtime: 1000,
      })
    }
    return jsonResponse({ path: '', workspace: '/work/machine/demo', entries: [], truncated: false })
  })
}

// jsdom has no layout, so the navigator's "scroll the focused row into view"
// step has nothing to do — it is stubbed rather than asserted on (that promise
// is e2e's to check).
Element.prototype.scrollIntoView = () => {}
// …and jsdom has no `showModal`/`close` on `<dialog>`. The navigator mounts
// `<SearchDialog />` in *every* test of this file, so the two stubs go on the
// prototype for all of them rather than per case — they only have to keep
// `el.open` truthful, which is all the dialog's own watch reads (the same pair
// useEditorWriteMatrix.test.js:385-390 patches).
Element.prototype.showModal = function showModal() {
  this.open = true
}
Element.prototype.close = function close() {
  this.open = false
}

// One navigator per test, then the menu's own buttons read out of the teleported
// `<body>` (that is where they live; the tree's scroll container would clip them).
async function menuFor(wrapper, path) {
  const row = wrapper.find(`.tree-row[data-path="${path}"]`)
  if (!row.exists()) throw new Error(`no row for ${path}`)
  await row.trigger('contextmenu', { clientX: 10, clientY: 10 })
  await flushPromises()
  const menu = document.querySelector('.tree-context-menu')
  if (!menu) return null
  return [...menu.querySelectorAll('[role="menuitem"]')].map((item) => item.textContent.trim())
}

async function mounted() {
  const wrapper = mount(WorkspaceNavigator, { attachTo: document.body })
  await flushPromises()
  tree.expand('src')
  await flushPromises()
  return wrapper
}

// The gesture that builds a batch: Ctrl+click each named row. Through the real
// rows, so the marks are what a user would have (a direct write to
// `state.marked` would let a broken click pass-through go unnoticed).
async function markRows(wrapper, ...paths) {
  for (const path of paths) {
    await wrapper.find(`.tree-row[data-path="${path}"]`).trigger('click', { ctrlKey: true })
  }
  await flushPromises()
}

beforeEach(() => {
  tree.state.nodes = {}
  tree.state.expanded = { '': true }
  tree.state.focused = ''
  tree.state.selected = null
  tree.state.editing = null
  tree.state.workspace = ''
  tree.state.marked = []
  tree.state.markAnchor = null
  files.state.open.length = 0
  files.state.active = null
  git.state.entries = []
  git.state.loadedAt = 0
  // The search composable keeps its answer on purpose, so a test that reads it
  // has to start from an empty one rather than from the previous test's scope.
  search.state.query = ''
  search.state.scope = null
  search.state.open = false
  search.state.status = 'idle'
  search.state.result = null
  search.state.error = null
  search.state.pendingReveal = null
  // The move prompt is composable state too, and an open one would answer a
  // later test's read of it.
  move.state.movePrompt = null
  // A wrapper is never unmounted (the tree rows of the previous test are
  // harmless), but a `<dialog>` left open would answer a later test's query.
  document.body.querySelectorAll('.tree-context-menu, dialog').forEach((el) => el.remove())
})

describe('the context menu, per row kind', () => {
  it('offers each row exactly the items it can honour', async () => {
    stubFetch()
    const wrapper = await mounted()

    expect(await menuFor(wrapper, '')).toEqual([
      'Search…',
      'New file…',
      'New folder…',
      'Refresh',
      'Copy relative path',
    ])
    expect(await menuFor(wrapper, 'src')).toEqual([
      'Search…',
      'New file…',
      'New folder…',
      'Rename…',
      'Delete',
      'Refresh',
      'Copy relative path',
    ])
    // notes.md is `comparable`, so View diff… is there; src/a.py is untracked,
    // so the item is absent rather than disabled-and-lying. Neither has a buffer.
    // The one difference between the two rows is *their own name*: notes.md ends
    // in `.md`, so it gets Preview and a.py does not, whatever git says.
    expect(await menuFor(wrapper, 'notes.md')).toEqual([
      'Open',
      'Search…',
      'Preview',
      'View diff…',
      'Rename…',
      'Move to…',
      'Delete',
      'Copy relative path',
    ])
    expect(await menuFor(wrapper, 'src/a.py')).toEqual([
      'Open',
      'Search…',
      'Rename…',
      'Move to…',
      'Delete',
      'Copy relative path',
    ])
  })

  // Preview belongs to a file row only: a folder and the root are not documents
  // to render, and `scope: 'file'` is what keeps it that way even though the
  // navigator's own flag builder is the only thing that could have set it.
  it('never offers Preview on a folder or the root', async () => {
    stubFetch()
    const wrapper = await mounted()
    expect(await menuFor(wrapper, 'src')).not.toContain('Preview')
    expect(await menuFor(wrapper, '')).not.toContain('Preview')
  })

  it('Close appears only once a tab holds the file', async () => {
    stubFetch()
    const wrapper = await mounted()
    files.state.open.push({
      kind: 'file',
      key: 'notes.md',
      path: 'notes.md',
      name: 'notes.md',
      text: 'x',
      baseline: 'x',
      dirty: false,
    })
    const without = await menuFor(wrapper, 'notes.md')
    files.state.open.length = 0
    const bare = await menuFor(wrapper, 'notes.md')
    expect(without).toContain('Close')
    expect(bare).not.toContain('Close')
  })

  // A *preview* tab of the same path does not make that promise true: the tree's
  // `Close` means "that file's buffer", and a rendered view has no buffer — it is
  // dismissed by its own ×. `hasBuffer` reads `files.recordFor`, which is
  // file-only, and that is what keeps the item honest.
  it('a preview tab alone does not offer Close', async () => {
    stubFetch()
    const wrapper = await mounted()
    files.state.open.push({
      kind: 'preview',
      key: 'preview:notes.md',
      path: 'notes.md',
      name: 'notes.md (preview)',
      text: '# x',
      html: '<h1>x</h1>',
    })
    expect(await menuFor(wrapper, 'notes.md')).not.toContain('Close')
  })

  // The right-click *is* a focus change: the keyboard chords answer about the
  // focused row, so a menu item and the key for the same action must land on the
  // same row without either of them remembering a path of its own.
  it('takes the tree focus to the row it was opened on', async () => {
    stubFetch()
    const wrapper = await mounted()
    tree.state.focused = ''
    await menuFor(wrapper, 'src/a.py')
    expect(tree.state.focused).toBe('src/a.py')
  })

  // The table is only half the feature: each item has to reach the flow that
  // owns its answer. One click per kind, asserted on the state that flow owns.
  it('routes each item to the flow that owns it', async () => {
    stubFetch()
    const wrapper = await mounted()

    // Open — the click's own channel, the one EditorArea watches.
    await menuFor(wrapper, 'src/a.py')
    await clickMenuItem('Open')
    expect(tree.state.selected?.path).toBe('src/a.py')

    // Rename… — the tree's name input, prefilled, for the row clicked.
    await menuFor(wrapper, 'notes.md')
    await clickMenuItem('Rename…')
    expect(tree.state.editing).toMatchObject({ mode: 'rename', path: 'notes.md', value: 'notes.md' })
    tree.cancelCreate()

    // New folder… — the create flow, into the folder that was clicked.
    await menuFor(wrapper, 'src')
    await clickMenuItem('New folder…')
    expect(tree.state.editing).toMatchObject({ mode: 'create', parent: 'src', isDir: true })
    tree.cancelCreate()

    // Delete — asks, and nothing else: the trash move is the dialog's.
    await menuFor(wrapper, 'src/a.py')
    await clickMenuItem('Delete')
    expect(files.state.deletePrompt).toMatchObject({ path: 'src/a.py', name: 'a.py', dir: false })
    files.cancelDelete()
    expect(files.state.open).toHaveLength(0)

    // Refresh — re-lists just that folder.
    const before = fetchCalls('/api/fs/tree?path=src')
    await menuFor(wrapper, 'src')
    await clickMenuItem('Refresh')
    await flushPromises()
    expect(fetchCalls('/api/fs/tree?path=src')).toBe(before + 1)
  })

  // Preview is the one item that must *not* go through the tree's selection:
  // publishing one would make EditorArea's watcher open the Monaco buffer beside
  // the rendered tab. So its answer is a record of its own kind, and no buffer.
  it('routes Preview to the preview record, not to a buffer or a selection', async () => {
    stubFetch()
    const wrapper = await mounted()
    await menuFor(wrapper, 'notes.md')
    await clickMenuItem('Preview')
    await flushPromises()

    expect(files.state.open).toHaveLength(1)
    expect(files.state.open[0]).toMatchObject({
      kind: 'preview',
      key: 'preview:notes.md',
      path: 'notes.md',
      name: 'notes.md (preview)',
    })
    expect(files.state.open[0].html).toContain('<h1')
    // No Monaco buffer was created for it, and no tree selection was published.
    expect(files.recordFor('notes.md')).toBeNull()
    expect(tree.state.selected).toBeNull()
  })

  // Search is the one item `scope: 'row'` gives to every row kind, so it reaches
  // the composable rather than the tree: the menu's job is to hand over the row,
  // and `lib/search-scope` is what turns it into folder / file / workspace. The
  // dialog's own rendering is searchDialog.test.js; what is checked here is that
  // the navigator's gesture lands on the right scope.
  it('offers Search on every row kind, with the row it was made on as the scope', async () => {
    stubFetch()
    const wrapper = await mounted()

    await menuFor(wrapper, 'src')
    await clickMenuItem('Search…')
    expect(search.state.open).toBe(true)
    expect(search.state.scope).toMatchObject({ kind: 'folder', label: 'src', path: 'src' })

    search.closeDialog()
    await menuFor(wrapper, 'src/a.py')
    await clickMenuItem('Search…')
    expect(search.state.scope).toMatchObject({ kind: 'file', label: 'src/a.py', path: 'src/a.py' })

    // The root row's scope is the workspace itself, so its label is the
    // workspace's own name — never the empty string that row's path *is*.
    search.closeDialog()
    tree.state.workspace = '/work/machine/demo'
    await menuFor(wrapper, '')
    await clickMenuItem('Search…')
    expect(search.state.scope).toMatchObject({
      kind: 'workspace',
      label: tree.rootLabel(),
      path: '',
    })
    search.closeDialog()
  })

  it("the Search item's hint names all three scopes", async () => {
    stubFetch()
    const wrapper = await mounted()
    await menuFor(wrapper, 'src')
    expect(menuItemTitle('Search…')).toBe(
      'Search inside this folder, this file, or the whole workspace',
    )
  })

  // Move to… is on *every* file row (D2: `scope: 'file'` alone, no `needs` and no
  // gating on marks — the single move is the N=1 case of one code path), and
  // `moveCount` is the count `batchFor` settles for the clicked row, read here as
  // hint text only. This is the one place the singular/plural copy is pinned.
  it("the Move item's hint counts the batch this row would move", async () => {
    stubFetch()
    const wrapper = await mounted()

    // No marks at all: the item is there and names one file.
    await menuFor(wrapper, 'notes.md')
    expect(menuItemTitle('Move to…')).toBe('Move this file into another folder')

    // Two marks of one folder, menu on a marked row: the whole batch.
    await markRows(wrapper, 'notes.md', 'todo.txt')
    await menuFor(wrapper, 'notes.md')
    expect(menuItemTitle('Move to…')).toBe('Move these 2 files into another folder')

    // The same marks, menu on an *unmarked* file: the narrowing shows in the copy
    // before anything is sent, and the marks stay highlighted.
    await menuFor(wrapper, 'src/a.py')
    expect(menuItemTitle('Move to…')).toBe('Move this file into another folder')
    expect(tree.state.marked).toEqual(['notes.md', 'todo.txt'])

    // Marks that cross parents: `batchFor` narrows to the clicked row.
    await markRows(wrapper, 'notes.md', 'src/a.py')
    await menuFor(wrapper, 'notes.md')
    expect(menuItemTitle('Move to…')).toBe('Move this file into another folder')

    // Presence never moves with the count, and a folder row or the root row has
    // no such item at all, marked or not (R4).
    expect(await menuFor(wrapper, 'src')).not.toContain('Move to…')
    expect(await menuFor(wrapper, '')).not.toContain('Move to…')
  })

  // The subject is resolved once, at the moment of the click (D4), so the move
  // composable never reads `state.marked` later.
  it('routes Move to… to the batch move flow, on that row', async () => {
    stubFetch()
    const wrapper = await mounted()

    await markRows(wrapper, 'notes.md', 'todo.txt')
    await menuFor(wrapper, 'notes.md')
    await clickMenuItem('Move to…')

    expect(move.state.movePrompt).toMatchObject({
      paths: ['notes.md', 'todo.txt'],
      parent: '',
      value: '',
    })
    move.cancelMove()
    // The menu reads the batch, it never clears it.
    expect(tree.state.marked).toEqual(['notes.md', 'todo.txt'])

    // A plain right-click on one file is the N=1 case of the same item.
    tree.clearMarks()
    await menuFor(wrapper, 'src/a.py')
    await clickMenuItem('Move to…')
    expect(move.state.movePrompt).toMatchObject({ paths: ['src/a.py'], parent: 'src' })
    move.cancelMove()
  })

  // The table, not a `v-if`, decides whether a menu is opened at all — `openMenu`
  // returns early on an empty list. Search is a `scope: 'row'` item, so it is now
  // part of the floor of every shape the table can be handed, and the leanest one
  // the navigator actually has (the root row: no Open, not mutable) still gets a
  // menu.
  it('a row that offers almost nothing still gets a menu, because of Search', async () => {
    expect(
      treeMenuItems({ dir: false, isRoot: true, mutable: false, trashRetentionDays: 30 }).map(
        (item) => item.id,
      ),
    ).toEqual(['search', 'copy-path'])

    stubFetch()
    const wrapper = await mounted()
    expect(await menuFor(wrapper, '')).toContain('Search…')
  })
})

describe('the header search button', () => {
  // The button's subject is the tree's, so the disabled state is the same answer
  // `commandTarget()` gives the chords: no focused row that exists, no button.
  it('is disabled while there is no row to search, enabled once one is', async () => {
    stubFetch()
    const wrapper = await mounted()
    const button = wrapper.find('.editor-nav-search')
    expect(button.exists()).toBe(true)

    tree.state.focused = 'src'
    await wrapper.vm.$nextTick()
    expect(button.attributes('disabled')).toBeUndefined()

    // A focused row that stopped existing (a re-list dropped it) leaves the
    // command with no subject, so the button goes idle again rather than
    // promising a search of a path the tree no longer shows.
    tree.state.focused = 'gone'
    await wrapper.vm.$nextTick()
    expect(button.attributes('disabled')).toBeDefined()
  })

  it('a click raises the dialog on the focused row', async () => {
    stubFetch()
    const wrapper = await mounted()
    tree.state.focused = 'src'
    await wrapper.vm.$nextTick()
    await wrapper.find('.editor-nav-search').trigger('click')

    expect(search.state.open).toBe(true)
    expect(search.state.scope).toMatchObject({ kind: 'folder', label: 'src', path: 'src' })
    // The tree's selection is untouched: a search is not a file open.
    expect(tree.state.selected).toBeNull()
    search.closeDialog()
  })

  // §2.9's reason for mounting the dialog inside the navigator: the button that
  // raises it and the scope label it renders need the same subject. A click on
  // the row must therefore find a real `<dialog>` in the document.
  it('the dialog is mounted inside the navigator, and the click shows it', async () => {
    stubFetch()
    const wrapper = await mounted()
    tree.state.focused = 'src/a.py'
    await wrapper.vm.$nextTick()
    await wrapper.find('.editor-nav-search').trigger('click')
    await flushPromises()

    const dialog = document.querySelector('.editor-search-dialog')
    expect(dialog).not.toBeNull()
    expect(dialog.open).toBe(true)
    expect(dialog.querySelector('.editor-search-scope-path')?.textContent.trim()).toBe('src/a.py')
    search.closeDialog()
    await flushPromises()
    expect(document.querySelector('.editor-search-dialog').open).toBe(false)
  })
})

// The marks as the user and a screen reader meet them: a third row class, the
// `role="tree"` selection reporting the batch rather than the pane's file, and
// the count in the footer. The state itself is the composable's, asserted in
// useWorkspaceTreeCommands.test.js — what lives here is the rendering.
describe('the batch on screen', () => {
  it('marks a row on Ctrl+click, without opening anything new', async () => {
    stubFetch()
    const wrapper = await mounted()
    expect(wrapper.find('.editor-nav-selected').exists()).toBe(false)

    await wrapper.find('.tree-row[data-path="src/a.py"]').trigger('click', { ctrlKey: true })
    await wrapper.find('.tree-row[data-path="notes.md"]').trigger('click', { ctrlKey: true })
    await flushPromises()

    expect(wrapper.find('.tree-row[data-path="src/a.py"]').classes()).toContain('tree-row--marked')
    expect(wrapper.find('.tree-row[data-path="notes.md"]').classes()).toContain('tree-row--marked')
    expect(wrapper.find('.tree-row[data-path="src"]').classes()).not.toContain('tree-row--marked')
    // The footer says how many, in words a scrolling tree cannot hide.
    expect(wrapper.find('.editor-nav-selected').text()).toBe('2 selected')
    // A marked row is still just a row: the batch never touches the pane's file.
    expect(tree.state.selected?.path).toBe('notes.md')
  })

  // D15: a multi-selectable `role="tree"` has to report its selected set, so the
  // marks answer `aria-selected` while they last — and the pane's file does,
  // when they do not.
  it('aria-selected follows the batch while there are marks', async () => {
    stubFetch()
    const wrapper = await mounted()

    await wrapper.find('.tree-row[data-path="notes.md"]').trigger('click')
    await flushPromises()
    expect(wrapper.find('.tree-row[data-path="notes.md"]').attributes('aria-selected')).toBe('true')
    expect(wrapper.find('.tree-row[data-path="src/a.py"]').attributes('aria-selected')).toBe('false')

    // Ctrl+click two rows, then take one back out: the batch and the pane's
    // file are now two different rows, and the tree reports the batch.
    await wrapper.find('.tree-row[data-path="src/a.py"]').trigger('click', { ctrlKey: true })
    await wrapper.find('.tree-row[data-path="notes.md"]').trigger('click', { ctrlKey: true })
    await wrapper.find('.tree-row[data-path="notes.md"]').trigger('click', { ctrlKey: true })
    await flushPromises()
    expect(tree.state.marked).toEqual(['src/a.py'])
    expect(tree.state.selected?.path).toBe('notes.md')
    expect(wrapper.find('.tree-row[data-path="src/a.py"]').attributes('aria-selected')).toBe('true')
    expect(wrapper.find('.tree-row[data-path="notes.md"]').attributes('aria-selected')).toBe('false')

    // A plain click ends the batch, and the single pane file is what the tree
    // reports again.
    await wrapper.find('.tree-row[data-path="src/a.py"]').trigger('click')
    await flushPromises()
    expect(wrapper.find('.tree-row[data-path="src/a.py"]').attributes('aria-selected')).toBe('true')
    expect(wrapper.find('.tree-row[data-path="notes.md"]').attributes('aria-selected')).toBe('false')
    expect(wrapper.find('.editor-nav-selected').exists()).toBe(false)
  })

  // The keyboard reaches the marks through the same composable, so a Shift+arrow
  // in a real row has to show up as a mark on that row.
  it('Shift+ArrowUp marks the row the focus moved to', async () => {
    stubFetch()
    const wrapper = await mounted()
    await wrapper.find('.tree-row[data-path="notes.md"]').trigger('click', { ctrlKey: true })
    await flushPromises()

    await wrapper.find('.editor-nav-tree').trigger('keydown', { key: 'ArrowUp', shiftKey: true })
    await flushPromises()
    expect(wrapper.find('.tree-row[data-path="src/a.py"]').classes()).toContain('tree-row--marked')
    expect(wrapper.find('.editor-nav-selected').text()).toBe('2 selected')
  })
})

// The hint of one menu item, read off the button that carries it — the same
// buttons `clickMenuItem` finds, so a test cannot pass on a label whose title
// belongs to another item.
function menuItemTitle(label) {
  const item = [...document.querySelectorAll('.tree-context-menu [role="menuitem"]')].find(
    (node) => node.textContent.trim() === label,
  )
  if (!item) throw new Error(`no menu item ${label}`)
  return item.getAttribute('title')
}

async function clickMenuItem(label) {
  const items = [...document.querySelectorAll('.tree-context-menu [role="menuitem"]')]
  const item = items.find((node) => node.textContent.trim() === label)
  if (!item) throw new Error(`no menu item ${label} (${items.map((n) => n.textContent.trim())})`)
  item.click()
  await flushPromises()
}

function fetchCalls(prefix) {
  return [...globalThis.fetch.mock.calls].filter(([url]) =>
    String(url).startsWith(prefix),
  ).length
}
