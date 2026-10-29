import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBenchmark } from '../src/composables/useBenchmark'
import { useChat } from '../src/composables/useChat'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useGitStatus } from '../src/composables/useGitStatus'
import { useWorkspaceTree } from '../src/composables/useWorkspaceTree'

// Layer 1 (jsdom): the third record kind in the tab strip — a rendered markdown
// preview. What is asserted is the *contract*, not the pixels (the branch itself
// is e2e's, since mounting EditorArea in jsdom would mean mounting Monaco):
//
//   * a preview is a **view**, shaped like a diff: its own `preview:` key beside
//     the file's own buffer, no `dirty`/`baseline`/`conflict`, not savable, not
//     probed, closed on the gesture that asked for it;
//   * it is a **snapshot** (D2): rendered once from the read that opened it, and
//     re-clicking Preview re-reads it rather than appending a second tab;
//   * every path-moving flow (rename, delete, a workspace switch) re-keys or
//     closes it, so no tab is left pointing at a path that is gone.

const files = useEditorFiles()
const notices = useEditorNotices()
const git = useGitStatus()
const tree = useWorkspaceTree()
const bench = useBenchmark()
const chat = useChat()

// The freshness probe's own request, counted separately from the reads this file
// also stubs, so "a preview is never probed" is asserted on the right call.
function statCalls() {
  return [...globalThis.fetch.mock.calls].filter(([url]) =>
    String(url).startsWith('/api/fs/stat'),
  )
}

function fetchUrlOfLastStat() {
  const calls = statCalls()
  return calls.length ? calls[calls.length - 1][0] : ''
}

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

const DOC = {
  path: 'docs/a.md',
  text: '# Heading\n\nbody\n',
  language: 'markdown',
  size: 16,
  mtime: 1000,
}

function stubFile(body, ok = true, status = 200) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse(body, ok, status))
}

function fileBodies(bodies) {
  const fetchMock = vi.spyOn(globalThis, 'fetch')
  let i = 0
  fetchMock.mockImplementation(() => jsonResponse(bodies[i++] ?? bodies[bodies.length - 1]))
  return fetchMock
}

function resetState() {
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  files.state.loading = false
  files.state.closePrompt = null
  files.state.deletePrompt = null
  files.state.conflict = null
  files.state.probeError = null
  notices.clear()
  git.state.entries = []
  git.state.repo = false
  git.state.error = null
  git.state.loadedAt = 0
}

describe('useEditorFiles: a preview tab in the same strip', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('opens one record, keyed apart from the file and its diff', async () => {
    stubFile(DOC)
    const record = await files.openPreview('docs/a.md')
    expect(record.kind).toBe('preview')
    expect(record.key).toBe('preview:docs/a.md')
    expect(record.path).toBe('docs/a.md')
    expect(record.name).toBe('a.md (preview)')
    // Rendered once, through the app's one renderer.
    expect(record.html).toContain('<h1')
    expect(files.state.active).toBe('preview:docs/a.md')
    expect(files.state.open).toHaveLength(1)
  })

  it('is a view: no buffer fields, and not savable', async () => {
    stubFile(DOC)
    const record = await files.openPreview('docs/a.md')
    for (const field of ['dirty', 'baseline', 'conflict', 'saving', 'stale', 'deleted']) {
      expect(record[field]).toBeUndefined()
    }
    const result = await files.saveFile(files.previewKeyFor('docs/a.md'))
    expect(result).toEqual({ ok: false, reason: 'not-savable' })
    expect(files.dirtyCount.value).toBe(0)
  })

  it('re-clicking Preview re-reads and re-renders the same tab (D2)', async () => {
    const fetchMock = fileBodies([
      DOC,
      { ...DOC, text: '# Heading\n\nbody\n\nmore\n', size: 22 },
    ])
    const first = await files.openPreview('docs/a.md')
    await files.openFile('docs/a.md') // a different tab about the same path
    const second = await files.openPreview('docs/a.md')

    // Focus *and* refresh: one record about that path, and it read the file
    // again. (Compared through `state.open`: the singleton is reactive, so the
    // proxy the lookup hands back is the record, not the object pushed first.)
    expect(files.state.open).toHaveLength(2)
    expect(files.state.open.map((item) => item.key)).toEqual([
      'preview:docs/a.md',
      'docs/a.md',
    ])
    expect(files.state.active).toBe('preview:docs/a.md')
    expect(files.previewRecordFor('docs/a.md')).not.toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(3)
    const live = files.previewRecordFor('docs/a.md')
    expect(live).toBe(second)
    expect(live.text).toContain('more')
    expect(live.html).toContain('<p>body</p>')
    expect(first.key).toBe('preview:docs/a.md')
  })

  it('keeps the buffer and the preview of one path as two separate tabs', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockImplementationOnce(() => jsonResponse(DOC))
    fetchMock.mockImplementationOnce(() => jsonResponse(DOC))
    await files.openFile('docs/a.md')
    await files.openPreview('docs/a.md')
    expect(files.state.open.map((item) => item.key)).toEqual(['docs/a.md', 'preview:docs/a.md'])
    expect(files.recordFor('docs/a.md').kind).toBe('file')
    expect(files.previewRecordFor('docs/a.md').kind).toBe('preview')
  })

  it('closes on the spot even while the file buffer of it is dirty', async () => {
    stubFile(DOC)
    const buffer = await files.openFile('docs/a.md')
    files.updateText('docs/a.md', 'edited, not saved')
    expect(buffer.dirty).toBe(true)

    const preview = await files.openPreview('docs/a.md')
    const result = files.requestCloseTab(preview.key)
    expect(result).toEqual({ closed: true })
    expect(files.state.closePrompt).toBeNull()
    // The dirty buffer the preview says nothing about is still there.
    expect(files.state.open.map((item) => item.key)).toEqual(['docs/a.md'])
    expect(files.recordFor('docs/a.md').dirty).toBe(true)
  })

  it('closing a preview dismisses no notices about the path (it is not a buffer)', async () => {
    stubFile(DOC)
    const preview = await files.openPreview('docs/a.md')
    notices.notice('error', 'Could not save docs/a.md: refused', 'docs/a.md')
    files.closeTab(preview.key)
    expect(notices.noticeForPath('docs/a.md')).toBeTruthy()
  })

  it('says so for a binary body rather than rendering a tab', async () => {
    stubFile({ path: 'docs/a.md', binary: true, size: 2200 })
    const record = await files.openPreview('docs/a.md')
    expect(record).toBeNull()
    expect(files.state.open).toHaveLength(0)
    expect(files.state.message.tone).toBe('info')
    expect(files.state.message.text).toContain('binary file, 2.1 kB')
  })

  it('echoes the server sentence for a refused or oversized read', async () => {
    stubFile({ detail: 'Path is read-only by workspace policy: docs/a.md' }, false, 422)
    expect(await files.openPreview('docs/a.md')).toBeNull()
    expect(files.state.open).toHaveLength(0)
    expect(files.state.message).toMatchObject({
      tone: 'error',
      path: 'docs/a.md',
    })
    expect(files.state.message.text).toContain('read-only by workspace policy')

    files.clearMessage()
    stubFile({ detail: 'File is too large to open: 12.0 MB' }, false, 413)
    await files.openPreview('docs/a.md')
    expect(files.state.message.text).toContain('too large to open')
  })

  it('carries the truncation answer onto the record', async () => {
    stubFile({ ...DOC, truncated: true, truncatedAt: 4096 })
    const record = await files.openPreview('docs/a.md')
    expect(record.truncated).toBe(true)
    expect(record.truncatedAt).toBe(4096)
  })

  it('is not a buffer the freshness probe asks about', async () => {
    // Every other gate open, so the *records* are the only thing that can make
    // this probe worth a request.
    bench.state.activeTab = 'editor'
    chat.state.phase = 'idle'
    stubFile(DOC)
    await files.openPreview('docs/a.md')
    await files.probeOpenFiles()
    expect(statCalls()).toHaveLength(0)

    // The same probe with a file record does ask — so the zero above is the
    // record kind, not the harness.
    await files.openFile('docs/a.md')
    await files.probeOpenFiles()
    expect(statCalls()).toHaveLength(1)
    expect(String(fetchUrlOfLastStat())).toContain('docs%2Fa.md')
    bench.state.activeTab = 'benchmark'
  })

  it('re-keys and re-names on rename — the tab follows the moved file', async () => {
    stubFile(DOC)
    await files.openPreview('docs/a.md')
    const moved = files.repointPath('docs/a.md', 'docs/b.md')
    expect(moved).toBe(1)
    const record = files.state.open[0]
    expect(record.key).toBe('preview:docs/b.md')
    expect(record.path).toBe('docs/b.md')
    expect(record.name).toBe('b.md (preview)')
    expect(record.kind).toBe('preview')
    // The pane still shows it, and it is still findable by its new path.
    expect(files.state.active).toBe('preview:docs/b.md')
    expect(files.previewRecordFor('docs/b.md')).toBe(record)
    expect(files.previewRecordFor('docs/a.md')).toBeNull()
  })

  it('a rename of an ancestor folder re-keys the preview too', async () => {
    stubFile(DOC)
    await files.openPreview('docs/a.md')
    files.repointPath('docs', 'notes')
    expect(files.state.open[0].key).toBe('preview:notes/a.md')
    expect(files.state.active).toBe('preview:notes/a.md')
  })

  it('a delete under the path closes it, without inflating the dirty count', async () => {
    stubFile(DOC)
    const buffer = await files.openFile('docs/a.md')
    files.updateText('docs/a.md', 'edited')
    await files.openPreview('docs/a.md')

    files.requestDelete({ path: 'docs', name: 'docs', dir: true })
    // The warning counts buffers, never the rendered view beside them.
    expect(files.state.deletePrompt).toMatchObject({ path: 'docs', dir: true, dirtyBuffers: 1 })
    files.cancelDelete()

    // The tree asks the *file*'s Close; a preview is not a buffer, so the promise
    // it makes is about the buffer only (and the preview keeps its own ×). The
    // dirty buffer answers with its own prompt — the preview does not.
    files.closeFile('docs/a.md')
    expect(files.state.closePrompt).toMatchObject({ key: 'docs/a.md' })
    files.discardClose()
    expect(files.state.open.map((item) => item.key)).toEqual(['preview:docs/a.md'])

    // A trashed path closes every record under it, preview included.
    vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
      if (init?.method === 'DELETE') {
        return jsonResponse({ path: 'docs', dir: true, entries: 2, trashedTo: '.nojae/trash/x/docs' })
      }
      return jsonResponse({ path: '', workspace: '/w', entries: [], truncated: false })
    })
    tree.state.nodes = { '': { loaded: true, entries: [], loading: false, truncated: false } }
    await files.trashPath('docs')
    expect(files.state.open).toHaveLength(0)
    expect(files.state.active).toBeNull()
  })

  it('a workspace switch closes it with every other tab', async () => {
    stubFile(DOC)
    await files.openPreview('docs/a.md')
    await files.openFile('docs/a.md')
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ repo: false, root: null, atToplevel: true, entries: [], truncated: false }),
    )
    files.forgetWorkspaceBuffers({ workspace: '/other' })
    expect(files.state.open).toHaveLength(0)
    expect(files.state.active).toBeNull()
  })

  it('Compare still opens a diff with a preview active', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    fetchMock.mockImplementationOnce(() => jsonResponse(DOC)) // preview
    fetchMock.mockImplementationOnce(() => jsonResponse(DOC)) // buffer
    const buffer = await files.openPreview('docs/a.md')
    await files.openFile('docs/a.md')
    // Compare needs a *conflicted buffer*; the preview beside it has no conflict
    // field at all, and Compare must not be blocked by one being the active tab.
    expect(buffer.conflict).toBeUndefined()
    expect(files.state.active).toBe('docs/a.md')
    const fileRecord = files.recordFor('docs/a.md')
    fileRecord.conflict = 'stale'
    fileRecord.conflictAt = 1
    fileRecord.conflictDetail = { diskMtime: 2000 }
    fetchMock.mockImplementationOnce(() => jsonResponse({ ...DOC, text: 'disk copy\n' }))
    const diff = await files.openConflictedDiff('docs/a.md')
    expect(diff.kind).toBe('diff')
    expect(files.state.open.map((item) => item.key)).toEqual([
      'preview:docs/a.md',
      'docs/a.md',
      'diff:docs/a.md',
    ])
    expect(files.state.active).toBe('diff:docs/a.md')
  })
})
