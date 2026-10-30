import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { h } from 'vue'

// The two Monaco surfaces are replaced before either module is loaded: a real
// one needs a browser (see probe-monaco.test.js), and this file is about the
// *wiring* — the pane's watcher on the search composable's `pendingReveal`,
// reaching the surface's exposed method and clearing the queue. The queue's
// decision (clamp or reject) is `monacoRevealClamp.test.js`; what Monaco is
// finally handed is e2e's (search-09).
const revealSpy = vi.hoisted(() => vi.fn())

vi.mock('../src/components/MonacoEditor.vue', () => ({
  default: {
    name: 'MonacoEditorStub',
    props: ['content', 'language', 'readonly', 'ariaLabel'],
    methods: { revealPosition: revealSpy },
    render() {
      return h('div', { class: 'stub-monaco' })
    },
  },
}))
vi.mock('../src/components/MonacoDiffViewer.vue', () => ({
  default: {
    name: 'MonacoDiffViewerStub',
    render() {
      return h('div', { class: 'stub-diff' })
    },
  },
}))

import EditorArea from '../src/components/EditorArea.vue'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useWorkspaceSearch } from '../src/composables/useWorkspaceSearch'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// ---------------------------------------------------------------------------
// search-08: the pane is the consumer of `pendingReveal`.
//
// Three cases, and all of them are the half that can break silently:
//   1. the coordinates arrive on the surface, exactly once;
//   2. the queue is cleared by the consumer, so a re-render cannot re-apply it;
//   3. with no code surface mounted (a diff tab), asking for a reveal neither
//      throws nor leaves the queue filled with a hit nobody can land on.
// ---------------------------------------------------------------------------

const files = useEditorFiles()
const tree = useWorkspaceTree()
const search = useWorkspaceSearch()
const bench = useBenchmark()

function fileRecord(path, text) {
  return {
    kind: 'file',
    key: path,
    path,
    name: path.slice(path.lastIndexOf('/') + 1),
    text,
    baseline: text,
    dirty: false,
    mtime: 1000,
    language: 'plaintext',
    size: text.length,
    binary: false,
    truncated: false,
    truncatedAt: null,
    encoding: 'utf-8',
    stale: false,
    deleted: false,
    conflict: null,
    conflictAt: 0,
    conflictDetail: null,
    saving: false,
    reloadedAt: 0,
  }
}

function diffRecord(path) {
  return {
    kind: 'diff',
    key: `diff:${path}`,
    path,
    name: `${path} (diff)`,
    original: 'before\n',
    modified: 'after\n',
    language: 'plaintext',
    status: 'modified',
    baseline: 'head',
    renamedFrom: null,
    truncated: false,
    truncatedAt: null,
  }
}

async function mountPane(records, activeKey) {
  files.state.open.splice(0, files.state.open.length, ...records)
  files.state.active = activeKey
  const wrapper = mount(EditorArea, { attachTo: document.body })
  await flushPromises()
  return wrapper
}

beforeEach(() => {
  revealSpy.mockClear()
  files.state.open.splice(0, files.state.open.length)
  files.state.active = null
  files.state.message = null
  search.state.pendingReveal = null
  tree.state.selected = null
  // The freshness tick is off for these tests (it would poll), and it only
  // starts on the Editor tab.
  bench.state.activeTab = 'benchmark'
  vi.spyOn(globalThis, 'fetch').mockImplementation((url) =>
    Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ path: String(url), entries: [], truncated: false }),
    }),
  )
})

describe('the pane applies a queued reveal', () => {
  it('hands the code surface the hit coordinates once, and clears the queue', async () => {
    const wrapper = await mountPane([fileRecord('notes.md', 'one\ntwo\n')], 'notes.md')
    expect(wrapper.find('.stub-monaco').exists()).toBe(true)

    search.state.pendingReveal = { path: 'notes.md', line: 12, column: 8 }
    await flushPromises()

    expect(revealSpy).toHaveBeenCalledTimes(1)
    expect(revealSpy).toHaveBeenCalledWith({ line: 12, column: 8 })
    expect(search.state.pendingReveal).toBeNull()
  })

  it('does not re-apply it on a later render of the same buffer', async () => {
    const wrapper = await mountPane([fileRecord('notes.md', 'one\ntwo\n')], 'notes.md')
    search.state.pendingReveal = { path: 'notes.md', line: 2, column: 1 }
    await flushPromises()
    expect(revealSpy).toHaveBeenCalledTimes(1)

    files.state.open[0].text = 'one\ntwo\nthree\n'
    await wrapper.vm.$nextTick()
    await flushPromises()

    expect(revealSpy).toHaveBeenCalledTimes(1)
  })

  it('survives a diff tab: no surface, no throw, no queue left behind', async () => {
    await mountPane([diffRecord('notes.md')], 'diff:notes.md')
    expect(document.querySelector('.stub-monaco')).toBeNull()

    search.state.pendingReveal = { path: 'notes.md', line: 12, column: 8 }
    await flushPromises()

    expect(revealSpy).not.toHaveBeenCalled()
    expect(search.state.pendingReveal).toBeNull()
  })

  it('ignores an empty queue without touching the surface', async () => {
    await mountPane([fileRecord('notes.md', 'one\n')], 'notes.md')
    await flushPromises()
    expect(revealSpy).not.toHaveBeenCalled()
  })

  it('does not move the caret in the buffer that happens to be active', async () => {
    // The ordinary case: two tabs open, the hit belongs to the one that is not
    // on screen. The tree's selection switches the buffer, and the coordinates
    // must not be spent on the buffer that was showing when the row was
    // clicked — that is the pane's own race, not the surface's queue.
    const other = fileRecord('other.md', 'x\n')
    const hitFile = fileRecord('notes.md', 'one\ntwo\n')
    await mountPane([other, hitFile], 'other.md')

    search.state.pendingReveal = { path: 'notes.md', line: 2, column: 3 }
    await flushPromises()
    expect(revealSpy).not.toHaveBeenCalled()
    expect(search.state.pendingReveal).not.toBeNull()

    files.state.active = 'notes.md'
    await flushPromises()

    expect(revealSpy).toHaveBeenCalledTimes(1)
    expect(revealSpy).toHaveBeenCalledWith({ line: 2, column: 3 })
    expect(search.state.pendingReveal).toBeNull()
  })

  it('lands when the file opens during the click, not before', async () => {
    // No tab at all, so there is no surface when the queue is filled; the
    // reveal arrives once `openFile` has published the record and the pane has
    // rendered the surface for it.
    const wrapper = await mountPane([], null)
    expect(wrapper.find('.stub-monaco').exists()).toBe(false)

    search.state.pendingReveal = { path: 'fresh.md', line: 7, column: 2 }
    await flushPromises()
    expect(revealSpy).not.toHaveBeenCalled()

    files.state.open.push(fileRecord('fresh.md', 'a\nb\n'))
    files.state.active = 'fresh.md'
    await flushPromises()

    expect(revealSpy).toHaveBeenCalledTimes(1)
    expect(revealSpy).toHaveBeenCalledWith({ line: 7, column: 2 })
  })
})
