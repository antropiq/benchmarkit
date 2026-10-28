import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useEditorFiles } from '../src/composables/useEditorFiles'

// The save half of the buffer records: the client-side part of the guarded
// write. The wire rules (deny-list, size cap, chat-stream gate) are the
// server's and are tested in tests/test_editorfs_write.py; what is tested here
// is the two things only this layer knows — that a half-read buffer must not go
// back, and that a 409 becomes a re-baselined record or a recorded conflict.

const files = useEditorFiles()

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

async function openRecord(path = 'src/a.py', overrides = {}) {
  vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
    jsonResponse({ path, text: 'one\n', mtime: 1000, size: 4, ...overrides }),
  )
  return files.openFile(path)
}

function resetState() {
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  files.state.loading = false
  files.state.closePrompt = null
}

describe('useEditorFiles.saveFile', () => {
  beforeEach(() => {
    resetState()
    vi.restoreAllMocks()
  })

  it('sends the buffer with the read mtime as the guard token', async () => {
    const record = await openRecord()
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'src/a.py', size: 6, mtime: 2000 }),
    )
    files.updateText('src/a.py', 'changed\n')
    const result = await files.saveFile('src/a.py')

    expect(result.ok).toBe(true)
    // The first call is openFile's GET (one argument); the save is the second.
    const put = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')
    expect(put).toBeTruthy()
    const [url, init] = put
    expect(url).toBe('/api/fs/file')
    expect(JSON.parse(init.body)).toEqual({
      path: 'src/a.py',
      text: 'changed\n',
      ifUnmodifiedSince: 1000,
    })
  })

  it('re-baselines on success, so the dot clears and the token moves', async () => {
    const record = await openRecord()
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'src/a.py', size: 6, mtime: 2000 }),
    )
    files.updateText('src/a.py', 'changed\n')
    expect(record.dirty).toBe(true)

    await files.saveFile('src/a.py')

    expect(record.dirty).toBe(false)
    expect(record.baseline).toBe('changed\n')
    // The next token is the server's answer, not the local one.
    expect(record.mtime).toBe(2000)
    expect(record.size).toBe(6)
  })

  it('records a 409 conflict instead of overwriting', async () => {
    const record = await openRecord()
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse(
        { detail: { reason: 'stale', path: 'src/a.py', diskMtime: 1500, bufferMtime: 1000 } },
        false,
        409,
      ),
    )
    files.updateText('src/a.py', 'mine\n')
    const result = await files.saveFile('src/a.py')

    expect(result.ok).toBe(false)
    expect(result.reason).toBe('conflict')
    // `conflict` names what happened; both numbers ride beside it so the pane
    // can name them (the shape the freshness probe uses too).
    expect(record.conflict).toBe('stale')
    expect(record.conflictDetail.diskMtime).toBe(1500)
    expect(record.conflictDetail.bufferMtime).toBe(1000)
    // The buffer is untouched and still dirty: nothing was lost either way.
    expect(record.text).toBe('mine\n')
    expect(record.dirty).toBe(true)
  })

  it('a reload answers the conflict it recorded', async () => {
    const record = await openRecord()
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ detail: { reason: 'stale', diskMtime: 1500, bufferMtime: 1000 } }, false, 409),
    )
    files.updateText('src/a.py', 'mine\n')
    await files.saveFile('src/a.py')
    expect(record.conflict).not.toBeNull()

    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'src/a.py', text: 'disk\n', mtime: 1500, size: 5 }),
    )
    await files.reloadFile('src/a.py')
    expect(record.conflict).toBeNull()
    expect(record.text).toBe('disk\n')
  })

  it('refuses a truncated buffer before any request', async () => {
    const record = await openRecord('big.py', { truncated: true, truncatedAt: 100 })
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    const result = await files.saveFile('big.py')

    expect(result.ok).toBe(false)
    expect(result.reason).toBe('truncated')
    // openFile's fetch, and no PUT: a half file never goes back to disk.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(record.dirty).toBe(false)
  })

  it('refuses a binary buffer too, even though one is never opened', async () => {
    const record = await openRecord('blob.bin')
    record.binary = true // openFile would have returned null; the guard is cheap
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    const result = await files.saveFile('blob.bin')
    expect(result.reason).toBe('binary')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('keeps a policy or size refusal as a message-shaped answer', async () => {
    await openRecord('mcp_connectors/x.py')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ detail: 'Path is refused by the connector path policy' }, false, 422),
    )
    const result = await files.saveFile('mcp_connectors/x.py')
    expect(result.ok).toBe(false)
    expect(result.reason).toBe('refused')
    expect(result.text).toContain('connector path policy')
  })

  // Line endings are the editor's, never the server's (D5): Monaco's model
  // takes its EOL from the file it was loaded with, and the save sends that
  // text back character for character. This is the client half of the CRLF
  // round-trip the server asserts in tests/test_editorfs_write.py.
  it('sends the line endings exactly as the model holds them', async () => {
    await openRecord('settings.ini')
    files.updateText('settings.ini', 'a = 1\r\nb = 2\r\n')
    vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) =>
      init?.method === 'PUT'
        ? jsonResponse({ path: 'settings.ini', size: 14, mtime: 2000 })
        : jsonResponse({}),
    )
    const fetchMock = globalThis.fetch
    await files.saveFile('settings.ini')

    const put = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')
    expect(JSON.parse(put[1].body).text).toBe('a = 1\r\nb = 2\r\n')
  })

  it('says so when the API cannot be reached at all', async () => {
    await openRecord()
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() => Promise.reject(new Error('boom')))
    const result = await files.saveFile('src/a.py')
    expect(result.ok).toBe(false)
    expect(result.reason).toBe('network')
  })

  it('has nothing to save for a path that is not open, or for a diff tab', async () => {
    const result = await files.saveFile('gone.py')
    expect(result.ok).toBe(false)
    // One reason for both shapes of "not a buffer": a diff tab has two texts
    // and no buffer to write, and it must not be savable by accident.
    expect(result.reason).toBe('not-savable')
  })
})
