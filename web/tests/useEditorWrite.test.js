import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useEditorNotices } from '../src/composables/useEditorNotices'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useChat } from '../src/composables/useChat'

// The freshness half of the buffer records: what a `GET /api/fs/stat` answer
// *means* for an open buffer. The route's own rules (the gate, `exists: false`
// vs the 422 for a denied path) are the server's and already covered in
// tests/test_editorfs.py; what is covered here is the client's decision table —
// when a buffer follows the disk silently, when it only gets a marker, and when
// the two become a conflict the user has to answer.

const files = useEditorFiles()
const notices = useEditorNotices()
const bench = useBenchmark()
const chat = useChat()

// The one refusal line a path carries in the write-feedback stack — the shape
// the pane's badge and the acceptance copy are both written against.
function refusal(path) {
  return notices.noticeForPath(path)
}

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

async function openRecord(path = 'src/a.py', mtime = 1000) {
  vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
    jsonResponse({ path, text: 'one\n', mtime, size: 4 }),
  )
  return files.openFile(path)
}

// A stat stub that also answers the file read a silent reload may issue, so a
// test can assert the *whole* consequence of a probe with one mock.
function stubProbe(entries, fileBody = null) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
    if (String(url).startsWith('/api/fs/stat')) return jsonResponse({ paths: entries })
    if (String(url).startsWith('/api/fs/file')) return jsonResponse(fileBody ?? {})
    return jsonResponse({})
  })
}

// The spy instance is shared with the `openRecord` call above it, so a count is
// only meaningful per route: what a probe costs is one stat request, plus one
// file read if (and only if) a clean buffer was told to follow the disk.
function callsTo(mock, prefix) {
  return mock.mock.calls.filter(([url]) => String(url).startsWith(prefix))
}

function resetState() {
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  files.state.probeError = null
  files.state.conflict = null
  files.state.conflictDismissedAt = 0
  files.state.loading = false
  files.state.closePrompt = null
  notices.clear()
  files.stopFreshnessProbe()
  // The probe asks only for a visible pane and an idle WorkBench; tests that
  // want a request flip these, the two gate tests leave them alone.
  bench.state.activeTab = 'editor'
  chat.state.phase = 'idle'
}

describe('useEditorFiles: the freshness probe', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('a matching mtime changes nothing', async () => {
    const record = await openRecord()
    const fetchMock = stubProbe([{ path: 'src/a.py', exists: true, mtime: 1000, size: 4 }])
    await files.probeOpenFiles()

    expect(record.stale).toBe(false)
    expect(record.deleted).toBe(false)
    expect(record.conflict).toBeNull()
    expect(files.state.conflict).toBeNull()
    // One request: the probe, and no reload of an unchanged file.
    expect(callsTo(fetchMock, '/api/fs/stat')).toHaveLength(1)
    expect(callsTo(fetchMock, '/api/fs/file')).toHaveLength(1) // openFile's own
  })

  it('a float wobble within the epsilon is not an edit', async () => {
    const record = await openRecord()
    const fetchMock = stubProbe([
      { path: 'src/a.py', exists: true, mtime: 1000.004, size: 4 },
    ])
    await files.probeOpenFiles()

    expect(record.stale).toBe(false)
    expect(record.conflict).toBeNull()
    expect(callsTo(fetchMock, '/api/fs/stat')).toHaveLength(1)
    expect(callsTo(fetchMock, '/api/fs/file')).toHaveLength(1)
  })

  it('a changed mtime on a clean buffer reloads it silently', async () => {
    const record = await openRecord()
    stubProbe(
      [{ path: 'src/a.py', exists: true, mtime: 2000, size: 6 }],
      { path: 'src/a.py', text: 'disk\n', mtime: 2000, size: 5 },
    )
    await files.probeOpenFiles()

    expect(record.text).toBe('disk\n')
    expect(record.baseline).toBe('disk\n')
    expect(record.dirty).toBe(false)
    expect(record.stale).toBe(false)
    expect(record.conflict).toBeNull()
    expect(files.state.conflict).toBeNull()
    expect(record.mtime).toBe(2000)
  })

  it('a changed mtime on a dirty buffer marks it and keeps the text', async () => {
    const record = await openRecord()
    files.updateText('src/a.py', 'mine\n')
    const fetchMock = stubProbe([{ path: 'src/a.py', exists: true, mtime: 2000, size: 6 }])
    await files.probeOpenFiles()

    expect(record.text).toBe('mine\n')
    expect(record.dirty).toBe(true)
    expect(record.stale).toBe(true)
    expect(record.conflict).toBe('stale')
    expect(record.conflictDetail.diskMtime).toBe(2000)
    // The buffer on screen is the one the pane dialogs about.
    expect(files.state.conflict.key).toBe('src/a.py')
    expect(files.state.conflict.kind).toBe('stale')
    expect(files.state.conflict.diskMtime).toBe(2000)
    // No reload of a dirty buffer: nothing was read back.
    expect(callsTo(fetchMock, '/api/fs/file')).toHaveLength(1) // openFile's own
  })

  it('a dirty buffer whose file is gone conflicts as deleted', async () => {
    const record = await openRecord()
    files.updateText('src/a.py', 'mine\n')
    stubProbe([{ path: 'src/a.py', exists: false, mtime: null, size: null, dir: false }])
    await files.probeOpenFiles()

    expect(record.deleted).toBe(true)
    expect(record.conflict).toBe('deleted')
    expect(record.text).toBe('mine\n')
    expect(files.state.conflict.kind).toBe('deleted')
    // There is no disk mtime to compare against, so nothing invents one.
    expect(files.state.conflict.diskMtime).toBeNull()
  })

  it('a clean buffer whose file is gone is only marked', async () => {
    const record = await openRecord()
    stubProbe([{ path: 'src/a.py', exists: false, mtime: null, size: null, dir: false }])
    await files.probeOpenFiles()

    expect(record.deleted).toBe(true)
    expect(record.conflict).toBeNull()
    expect(files.state.conflict).toBeNull()
  })

  it('a failed probe marks nothing and says the probe failed', async () => {
    const record = await openRecord()
    stubProbe([])
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ detail: 'Working location is on the deny-list' }, false, 422),
    )
    await files.probeOpenFiles()

    expect(files.state.probeError).toContain('deny-list')
    expect(record.stale).toBe(false)
    expect(record.deleted).toBe(false)
    expect(record.conflict).toBeNull()
  })

  it('names file paths only — a diff tab is never probed', async () => {
    await openRecord('src/a.py')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'src/a.py', original: 'one\n', modified: 'two\n' }),
    )
    await files.openDiff('src/a.py')

    const fetchMock = stubProbe([{ path: 'src/a.py', exists: true, mtime: 1000, size: 4 }])
    await files.probeOpenFiles()

    const [url] = fetchMock.mock.calls.at(-1)
    expect(url).toBe('/api/fs/stat?path=src%2Fa.py')
  })

  it('two overlapping probes issue one request', async () => {
    await openRecord()
    let resolveFetch
    const pending = new Promise((resolve) => {
      resolveFetch = resolve
    })
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => pending)

    const first = files.probeOpenFiles()
    const second = files.probeOpenFiles()
    resolveFetch(jsonResponse({ paths: [] }))
    await Promise.all([first, second])

    expect(callsTo(fetchMock, '/api/fs/stat')).toHaveLength(1)
  })

  it('asks nothing while the pane is hidden or a reply streams', async () => {
    await openRecord()
    const fetchMock = stubProbe([{ path: 'src/a.py', exists: true, mtime: 2000, size: 6 }])

    bench.state.activeTab = 'benchmark'
    await files.probeOpenFiles()
    chat.state.phase = 'answering'
    bench.state.activeTab = 'editor'
    await files.probeOpenFiles()

    // Two gates, one assertion: no stat request at all, and no record moved.
    expect(callsTo(fetchMock, '/api/fs/stat')).toHaveLength(0)
    expect(files.recordFor('src/a.py').mtime).toBe(1000)
    expect(files.recordFor('src/a.py').stale).toBe(false)
  })

  it('Compare puts the disk on the left and the buffer on the right', async () => {
    const record = await openRecord()
    files.updateText('src/a.py', 'mine\n')
    stubProbe(
      [{ path: 'src/a.py', exists: true, mtime: 2000, size: 6 }],
      { path: 'src/a.py', text: 'disk\n', mtime: 2000, size: 5, language: 'python' },
    )
    await files.probeOpenFiles()

    const diff = await files.openConflictedDiff('src/a.py')
    expect(diff.kind).toBe('diff')
    expect(diff.key).toBe('diff:src/a.py')
    expect(diff.original).toBe('disk\n') // left: what the disk holds now
    expect(diff.modified).toBe('mine\n') // right: the unsaved buffer
    expect(diff.source).toBe('conflict')
    expect(files.state.active).toBe('diff:src/a.py')
    // The file record is untouched by the comparison.
    expect(record.text).toBe('mine\n')
    expect(record.conflict).toBe('stale')
  })

  it('answering the conflict drops the comparison it opened', async () => {
    await openRecord()
    files.updateText('src/a.py', 'mine\n')
    stubProbe(
      [{ path: 'src/a.py', exists: true, mtime: 2000, size: 6 }],
      { path: 'src/a.py', text: 'disk\n', mtime: 2000, size: 5 },
    )
    await files.probeOpenFiles()
    await files.openConflictedDiff('src/a.py')
    expect(files.state.open.map((item) => item.key)).toEqual(['src/a.py', 'diff:src/a.py'])

    await files.reloadFile('src/a.py')
    expect(files.state.open.map((item) => item.key)).toEqual(['src/a.py'])
  })

  it('a dismissed conflict stays on the record and can be raised again', async () => {
    // Real clocks tick too fast for this: the test is about two conflicts
    // raised at *different* moments, so the clock is moved on by hand.
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'))
    const record = await openRecord()
    files.updateText('src/a.py', 'mine\n')
    stubProbe([{ path: 'src/a.py', exists: true, mtime: 2000, size: 6 }])
    await files.probeOpenFiles()
    const first = files.state.conflict

    files.dismissConflict()
    // The next tick re-reads the same news: the dialog must not nag.
    await files.probeOpenFiles()
    expect(files.state.conflict.raisedAt).toBe(first.raisedAt)
    expect(record.conflict).toBe('stale')

    vi.setSystemTime(new Date('2026-01-01T00:00:05Z'))
    files.raiseConflict('src/a.py')
    expect(files.state.conflict.raisedAt).not.toBe(first.raisedAt)
    vi.useRealTimers()
  })
})

// The save half as the *user* reaches it: one entry point that every affordance
// calls, the state each answer leaves behind, and the close prompt that stands
// between a gesture and lost work. The wire rules stay the server's
// (tests/test_editorfs_write.py); these are the client's decisions.
describe('useEditorFiles: saving, and never losing work on the way out', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  // The token is what the *last* confirmed answer said, so a second save of a
  // buffer that saved once cannot carry a stale one.
  it('a second save sends the token the first save got back', async () => {
    await openRecord()
    files.updateText('src/a.py', 'one edit\n')
    // Stubs answer by route, not by call order: a confirmed save also asks the
    // git status for the tree's marks, and that request would otherwise eat the
    // next one-shot answer.
    const mtimes = [2000, 3000]
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
      if (String(url).startsWith('/api/fs/status')) {
        return jsonResponse({ repo: true, entries: [] })
      }
      return jsonResponse({
        path: 'src/a.py',
        size: 9,
        mtime: mtimes[Number(init?.method === 'PUT') - 1] ?? 1000,
      })
    })
    expect((await files.saveFile('src/a.py')).ok).toBe(true)

    files.updateText('src/a.py', 'two edits\n')
    expect((await files.saveFile('src/a.py')).ok).toBe(true)

    const puts = callsTo(fetchMock, '/api/fs/file').filter(([, init]) => init?.method === 'PUT')
    expect(puts.map(([, init]) => JSON.parse(init.body).ifUnmodifiedSince)).toEqual([1000, 2000])
  })

  // A held Ctrl+S must not fire a request per repeat: `saving` is both the
  // chip's busy class and the guard.
  it('a save already in flight answers the second call without a request', async () => {
    const record = await openRecord()
    files.updateText('src/a.py', 'mine\n')
    let resolveFetch
    const pending = new Promise((resolve) => {
      resolveFetch = resolve
    })
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => pending)

    const first = files.saveFile('src/a.py')
    expect(record.saving).toBe(true)
    const second = await files.saveFile('src/a.py')
    expect(second.ok).toBe(false)
    expect(second.reason).toBe('in-flight')

    resolveFetch(jsonResponse({ path: 'src/a.py', size: 5, mtime: 2000 }))
    expect((await first).ok).toBe(true)
    expect(callsTo(fetchMock, '/api/fs/file')).toHaveLength(2) // open + one PUT
    expect(record.saving).toBe(false)
  })

  // The tree's marks are drawn from one query for the work tree; a save that
  // lands has to ask again or the row keeps looking unmodified.
  it('a saved file refreshes the git marks', async () => {
    await openRecord()
    files.updateText('src/a.py', 'mine\n')
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((url) => {
      if (String(url).startsWith('/api/fs/status')) {
        return jsonResponse({ repo: true, entries: [{ path: 'src/a.py', kind: 'modified', comparable: true }] })
      }
      return jsonResponse({ path: 'src/a.py', size: 5, mtime: 2000 })
    })
    await files.saveFile('src/a.py')
    expect(callsTo(fetchMock, '/api/fs/status')).toHaveLength(1)
  })

  it('a refused save leaves the dot, and says why in a notice', async () => {
    const record = await openRecord()
    files.updateText('src/a.py', 'mine\n')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ detail: 'Cannot save while a chat reply is streaming' }, false, 409),
    )
    const result = await files.saveFile('src/a.py')

    expect(result.reason).toBe('refused')
    expect(record.dirty).toBe(true)
    expect(record.text).toBe('mine\n')
    // The server's sentence, verbatim, behind what was attempted and naming the
    // path the request carried.
    expect(refusal('src/a.py').text).toBe(
      'Could not save src/a.py: Cannot save while a chat reply is streaming',
    )

    notices.dismiss(refusal('src/a.py').id)
    expect(notices.state.notices.map((item) => [item.id, item.tone, item.text])).toEqual([])
    // Dismissing the message does not dismiss the work.
    expect(record.dirty).toBe(true)
  })

  it('saveAllDirty writes one per dirty buffer, in tab order, and stops at the first failure', async () => {
    await openRecord('one.py')
    await openRecord('two.py')
    await openRecord('three.py')
    for (const path of ['one.py', 'two.py', 'three.py']) files.updateText(path, 'edit\n')
    // The diff tab of a dirty path is never part of the lot.
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'one.py', original: 'a\n', modified: 'b\n' }),
    )
    await files.openDiff('one.py')

    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation((url, init) => {
        if (init?.method !== 'PUT') return jsonResponse({})
        const path = JSON.parse(init.body).path
        if (path === 'two.py') return jsonResponse({ detail: 'Could not write' }, false, 413)
        return jsonResponse({ path, size: 5, mtime: 2000 })
      })

    const { saved, failure } = await files.saveAllDirty()
    expect(saved).toEqual(['one.py'])
    expect(failure.reason).toBe('refused')

    const order = callsTo(fetchMock, '/api/fs/file')
      .filter(([, init]) => init?.method === 'PUT')
      .map(([url, init]) => JSON.parse(init.body).path)
    // three.py was never asked about: the run stopped at the failure.
    expect(order).toEqual(['one.py', 'two.py'])
    expect(files.recordFor('one.py').dirty).toBe(false)
    expect(files.recordFor('two.py').dirty).toBe(true)
    expect(files.recordFor('three.py').dirty).toBe(true)
  })

  it('a dirty buffer cannot be closed by one gesture; Cancel keeps it, Discard drops it', async () => {
    const record = await openRecord()
    files.updateText('src/a.py', 'mine\n')

    expect(files.requestCloseTab('src/a.py').reason).toBe('confirm')
    expect(files.state.closePrompt.key).toBe('src/a.py')
    expect(files.state.open).toHaveLength(1)

    files.cancelClose()
    expect(files.state.closePrompt).toBeNull()
    expect(files.state.open).toHaveLength(1)
    expect(files.recordFor('src/a.py').text).toBe('mine\n')

    files.requestCloseTab('src/a.py')
    files.discardClose()
    expect(files.state.open).toHaveLength(0)
  })

  it('a clean buffer and a diff tab close without asking', async () => {
    await openRecord()
    expect(files.requestCloseTab('src/a.py').closed).toBe(true)

    await openRecord('two.py')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'two.py', original: 'a\n', modified: 'b\n' }),
    )
    await files.openDiff('two.py')
    expect(files.requestCloseTab('diff:two.py').closed).toBe(true)
    expect(files.state.closePrompt).toBeNull()
    // A diff tab is not savable whatever the state around it.
    expect((await files.saveFile('diff:two.py')).reason).toBe('not-savable')
  })

  it('saveAndClose closes only when the write landed', async () => {
    await openRecord()
    files.updateText('src/a.py', 'mine\n')
    files.requestCloseTab('src/a.py')

    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ detail: 'Cannot save while a chat reply is streaming' }, false, 409),
    )
    const failed = await files.saveAndClose()
    // The refusal ends the *dialog*, not the buffer: closing would be the
    // destructive answer to a question that was never answered.
    expect(failed.ok).toBe(false)
    expect(files.state.closePrompt).toBeNull()
    expect(files.state.open).toHaveLength(1)
    expect(files.recordFor('src/a.py').dirty).toBe(true)
    expect(refusal('src/a.py').text).toContain('chat reply is streaming')

    // Asked again with a disk that accepts it, it saves and closes.
    files.requestCloseTab('src/a.py')
    vi.spyOn(globalThis, 'fetch').mockImplementation((url) =>
      String(url).startsWith('/api/fs/status')
        ? jsonResponse({ repo: true, entries: [] })
        : jsonResponse({ path: 'src/a.py', size: 5, mtime: 2000 }),
    )
    expect((await files.saveAndClose()).ok).toBe(true)
    expect(files.state.open).toHaveLength(0)
    expect(files.state.closePrompt).toBeNull()
  })
})
