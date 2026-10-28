import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  MAX_VISIBLE_NOTICES,
  NOTICE_DEDUPE_WINDOW_MS,
  NOTICE_TTL_MS,
  useEditorNotices,
} from '../src/composables/useEditorNotices'
import { useEditorFiles } from '../src/composables/useEditorFiles'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useChat } from '../src/composables/useChat'

// Layer 1 (jsdom): the write-feedback channel. The routes' rules (which status
// means what) are the server's and live in tests/test_editorfs_write.py; what
// lives here is the *persistence* contract — a refusal that stays, an identical
// refusal that does not stack, a confirmation that expires, and a success that
// answers the failure it was about.

const notices = useEditorNotices()
const files = useEditorFiles()
const bench = useBenchmark()
const chat = useChat()

const texts = () => notices.state.notices.map((item) => item.text)

function jsonResponse(body, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) })
}

async function openRecord(path = 'src/a.py', mtime = 1000) {
  vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
    jsonResponse({ path, text: 'one\n', mtime, size: 4 }),
  )
  return files.openFile(path)
}

function reset() {
  notices.clear()
  files.state.open.length = 0
  files.state.active = null
  files.state.message = null
  files.state.probeError = null
  files.state.conflict = null
  // The probe asks only for a visible pane and an idle WorkBench.
  bench.state.activeTab = 'editor'
  chat.state.phase = 'idle'
}

describe('useEditorNotices: the stack itself', () => {
  beforeEach(() => {
    vi.useRealTimers()
    reset()
    vi.restoreAllMocks()
  })

  // The bug this channel exists to avoid: a refusal that vanishes on a timer.
  it('a refusal never expires, a confirmation does', () => {
    vi.useFakeTimers()
    notices.notice('error', 'Could not save a.py: refused', 'a.py')
    notices.notice('warn', 'Could not check the open files against disk: boom')
    notices.notice('ok', 'Created b.py', 'b.py')

    vi.advanceTimersByTime(NOTICE_TTL_MS * 3)

    expect(texts()).toEqual([
      'Could not save a.py: refused',
      'Could not check the open files against disk: boom',
    ])
    vi.useRealTimers()
  })

  it('the same failure twice in the window is one notice', () => {
    notices.notice('error', 'Could not save a.py: refused', 'a.py')
    notices.notice('error', 'Could not save a.py: refused', 'a.py')
    expect(notices.state.notices).toHaveLength(1)

    // Different news about the same path is not a duplicate.
    notices.notice('error', 'Could not save a.py: over the cap', 'a.py')
    expect(notices.state.notices).toHaveLength(2)
  })

  // A held Ctrl+S on a failing buffer, and a poll that keeps failing, are the
  // two things the window protects against; both are longer-lived than the
  // window, so it has to be *longer* than the poll interval.
  it('the window outlasts the freshness poll, and a later repeat is news', () => {
    vi.useFakeTimers()
    notices.notice('warn', 'Could not check the open files against disk: boom')
    vi.advanceTimersByTime(NOTICE_DEDUPE_WINDOW_MS + 1000)
    notices.notice('warn', 'Could not check the open files against disk: boom')
    expect(notices.state.notices).toHaveLength(2)
    vi.useRealTimers()
  })

  it('the cap drops the oldest first', () => {
    for (const path of ['a.py', 'b.py', 'c.py', 'd.py']) {
      notices.notice('error', `Could not save ${path}: refused`, path)
    }
    expect(notices.state.notices).toHaveLength(MAX_VISIBLE_NOTICES)
    expect(texts()).toEqual([
      'Could not save b.py: refused',
      'Could not save c.py: refused',
      'Could not save d.py: refused',
    ])
  })

  it('dismiss removes one line and clear removes all', () => {
    const first = notices.notice('error', 'Could not save a.py: refused', 'a.py')
    notices.notice('error', 'Could not save b.py: refused', 'b.py')

    notices.dismiss(first.id)
    expect(texts()).toEqual(['Could not save b.py: refused'])
    // A second dismiss of the same id is silence, not an error.
    notices.dismiss(first.id)
    expect(notices.state.notices).toHaveLength(1)

    notices.clear()
    expect(notices.state.notices).toHaveLength(0)
  })

  // A save that now works removes its own earlier failure line — for the path
  // and for anything under it.
  it('a success dismisses the matching earlier failure', () => {
    notices.notice('error', 'Could not save src/a.py: refused', 'src/a.py')
    notices.notice('error', 'Could not save src/core/job.py: refused', 'src/core/job.py')
    notices.notice('ok', 'Created src', 'src')

    notices.dismissForPath('src/core')
    expect(texts()).toEqual(['Could not save src/a.py: refused', 'Created src'])

    // A path that merely *starts with* the name is not under it: saving `src`
    // dismisses `src/a.py` but leaves `src2/a.py` standing.
    notices.notice('error', 'Could not save src2/a.py: refused', 'src2/a.py')
    notices.dismissForPath('src')
    expect(texts()).toEqual(['Created src', 'Could not save src2/a.py: refused'])
  })

  it('noticeForPath answers with a refusal, never with a confirmation', () => {
    notices.notice('ok', 'Created a.py', 'a.py')
    expect(notices.noticeForPath('a.py')).toBeNull()

    notices.notice('error', 'Could not save a.py: refused', 'a.py')
    expect(notices.noticeForPath('a.py').tone).toBe('error')
    expect(notices.noticeForPath('b.py')).toBeNull()
  })

  // A move re-points what the user still holds and drops what already happened
  // somewhere else (the same split useEditorFiles.repointPath makes).
  it('a rename re-points a refusal and drops a confirmation', () => {
    notices.notice('error', 'Could not save src/a.py: refused', 'src/a.py')
    notices.notice('ok', 'Created src/a.py', 'src/a.py')
    notices.notice('error', 'Could not save src2/a.py: refused', 'src2/a.py')

    notices.repointNotices('src', 'web')

    expect(texts()).toEqual([
      'Could not save src/a.py: refused',
      'Could not save src2/a.py: refused',
    ])
    expect(notices.noticeForPath('web/a.py').tone).toBe('error')
    expect(notices.noticeForPath('src/a.py')).toBeNull()
  })
})

describe('what the callers put in the stack', () => {
  beforeEach(() => {
    vi.useRealTimers()
    reset()
    vi.restoreAllMocks()
  })

  // The wording rule, end to end: the server's `detail` is shown as it was
  // written, prefixed by what was attempted, naming the path the request
  // carried — never rewritten, never a bare status code.
  it('a refused save becomes the attempt plus the server sentence', async () => {
    const detail =
      "Path is refused by the connector path policy: inside 'mcp_connectors', " +
      'which connectors never write to it'
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ path: 'mcp_connectors/x.py', text: 'one\n', mtime: 1000, size: 4 }),
    )
    await files.openFile('mcp_connectors/x.py')
    files.updateText('mcp_connectors/x.py', 'mine\n')

    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse({ detail }, false, 422),
    )
    const result = await files.saveFile('mcp_connectors/x.py')

    expect(result.reason).toBe('refused')
    expect(notices.state.notices).toHaveLength(1)
    expect(notices.noticeForPath('mcp_connectors/x.py')).toMatchObject({
      tone: 'error',
      text: `Could not save mcp_connectors/x.py: ${detail}`,
    })

    // A held chord re-asking about the same buffer is one line, not a stack.
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({ detail }, false, 422))
    await files.saveFile('mcp_connectors/x.py')
    await files.saveFile('mcp_connectors/x.py')
    expect(notices.state.notices).toHaveLength(1)

    // And the save that lands dismisses it.
    vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) =>
      String(url).startsWith('/api/fs/status')
        ? jsonResponse({ repo: true, entries: [] })
        : jsonResponse({ path: 'mcp_connectors/x.py', size: 5, mtime: 2000 }),
    )
    expect((await files.saveFile('mcp_connectors/x.py')).ok).toBe(true)
    expect(notices.state.notices).toHaveLength(0)
  })
})

describe('conflicts: which copy of the news the stack carries', () => {
  beforeEach(() => {
    vi.useRealTimers()
    reset()
    vi.restoreAllMocks()
    bench.state.activeTab = 'editor'
    chat.state.phase = 'idle'
  })

  // The buffer on screen is answered by the pane's dialog, so a conflict raised
  // for it raises no line at all — the acceptance's "no stack of notices hides
  // behind the conflict dialog".
  it('a conflict on the buffer on screen stays out of the stack', async () => {
    await openRecord('src/a.py')
    files.updateText('src/a.py', 'mine\n')
    vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
      jsonResponse(
        { detail: { reason: 'stale', path: 'src/a.py', diskMtime: 1500, bufferMtime: 1000 } },
        false,
        409,
      ),
    )
    const result = await files.saveFile('src/a.py')

    expect(result.reason).toBe('conflict')
    expect(notices.state.notices).toHaveLength(0)
    expect(files.state.conflict.kind).toBe('stale')
  })

  // A background buffer gets no dialog, so its conflict needs the one line —
  // and it names what to do, not just what happened.
  it('a conflict on a buffer that is not on screen gets one line', async () => {
    await openRecord('src/a.py')
    await openRecord('src/b.py')
    files.focusTab('src/a.py') // the pane shows a.py only
    files.updateText('src/b.py', 'mine\n')

    vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
      if (String(url).startsWith('/api/fs/status')) {
        return jsonResponse({ repo: true, entries: [] })
      }
      const path = init?.method === 'PUT' ? JSON.parse(init.body).path : 'src/a.py'
      if (path === 'src/b.py') {
        return jsonResponse(
          { detail: { reason: 'stale', path, diskMtime: 1500, bufferMtime: 1000 } },
          false,
          409,
        )
      }
      return jsonResponse({ path, size: 5, mtime: 2000 })
    })

    files.updateText('src/a.py', 'mine too\n')

    const { saved, failure } = await files.saveAllDirty()
    expect(saved).toEqual(['src/a.py'])
    expect(failure.reason).toBe('conflict')
    expect(notices.state.notices.map((item) => item.text)).toEqual([
      'Could not save src/b.py: the file changed on disk — resolve the conflict first',
    ])

    // Turning to that buffer hands the news to the dialog, so the line goes.
    files.focusTab('src/b.py')
    expect(notices.state.notices).toHaveLength(0)
    expect(files.state.conflict.key).toBe('src/b.py')
  })

  // The probe's half: the same one line for a buffer the user is not looking
  // at, silence for the one they are, and the warning is not repeated per tick.
  it('a probe-detected conflict warns once, and stops when the dialog takes over', async () => {
    await openRecord('src/a.py')
    await openRecord('src/b.py')
    files.focusTab('src/a.py')
    files.updateText('src/a.py', 'mine a\n')
    files.updateText('src/b.py', 'mine b\n')

    const probe = () =>
      vi.spyOn(globalThis, 'fetch').mockImplementationOnce(() =>
        jsonResponse({
          paths: [
            { path: 'src/a.py', exists: true, mtime: 2000, size: 7 },
            { path: 'src/b.py', exists: true, mtime: 2000, size: 7 },
          ],
        }),
      )

    probe()
    await files.probeOpenFiles()
    probe()
    await files.probeOpenFiles()

    // a.py is on screen: the dialog owns it, so only b.py has a line — and the
    // second tick added nothing.
    expect(files.state.conflict.key).toBe('src/a.py')
    expect(notices.state.notices.map((item) => item.path)).toEqual(['src/b.py'])
    expect(notices.state.notices[0].tone).toBe('warn')

    // The disk agreeing again is the end of that news.
    probe()
    files.recordFor('src/b.py').mtime = 2000
    await files.probeOpenFiles()
    expect(notices.state.notices.filter((item) => item.tone === 'warn')).toHaveLength(0)
  })

  // The 5 s poll must not produce a notice per tick: one for the episode, and
  // the next episode is news again.
  it('a broken probe reports once, not once per tick', async () => {
    files.state.open.push({ kind: 'file', key: 'a.py', path: 'a.py', mtime: 1000 })
    files.state.active = 'a.py'
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      jsonResponse({ detail: 'Working location is on the deny-list' }, false, 422),
    )

    await files.probeOpenFiles()
    await files.probeOpenFiles()
    await files.probeOpenFiles()

    expect(fetchMock).toHaveBeenCalledTimes(3)
    const warns = notices.state.notices.filter((item) => item.tone === 'warn')
    expect(warns).toHaveLength(1)
    expect(warns[0].text).toContain('deny-list')
    // The pane's own line still carries the reason (it clears on recovery).
    expect(files.state.probeError).toContain('deny-list')
  })
})
