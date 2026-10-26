import { reactive } from 'vue'

// ---------------------------------------------------------------------------
// Open files: the state behind the Editor tab's code pane (ed-03).
//
// One singleton like every other screen's: the open buffers, the active tab
// and what the pane should say live *here*, so a tab switch (`v-show`) keeps
// them and the Monaco component only renders and disposes a view.
//
// The pane's edit mode follows the WorkBench: it is read_write only while the
// chat is idle, read-only whenever a reply is streaming (EditorArea owns that
// rule and passes `readonly` down to Monaco). In read_write an edit updates the
// in-memory buffer and its `dirty` flag, but nothing reaches disk — there is no
// save route yet — so there is still nothing to confirm on close and no
// `beforeunload` warning. Save, and new file/folder, are a later session.
//
// `dirty` lives on the record, not on Monaco: the pane reuses one editor model
// across every buffer (EditorArea's `v-if`/model swap), so Monaco's own dirty
// flag would be lost the moment a tab is switched. Comparing the buffer's
// current `text` against the `baseline` it was read with is both per-file and
// survives the switch — and a reload resets the baseline, clearing the dot.
//
// The tree's `selected` (useWorkspaceTree) and this `active` are deliberately
// separate: selecting a row highlights it, opening a file puts it in the tab
// strip. They coincide in the common case and diverge exactly when a click is
// refused (binary, over the size cap) — which is when one shared value would
// lie about what is on screen.

// The pane renders every buffer through Monaco (virtualised), so line count
// costs nothing here — the old hand-rolled gutter cap it needed is gone.
const state = reactive({
  // Open buffers, in the order they were opened. Only *text* files get a
  // record: a binary or oversized file is a message in the pane, not a tab.
  // A diff (ed: "View diff…") is a record too — same strip, same rules — with
  // `kind: 'diff'` and two texts instead of one.
  open: [],
  active: null, // key of the visible buffer (the tab strip's selection)
  loading: false,
  // What the last click answered when it did not produce a buffer:
  // { path, text, tone }, tone 'error' being a refusal (policy, size cap,
  // missing) and 'info' an answer that is not an error — "binary file,
  // 2.1 MB". It rides above whatever buffer is still on screen, and clears on
  // the next focus or close.
  message: null,
})

function nameOf(path) {
  const index = path.lastIndexOf('/')
  return index === -1 ? path : path.slice(index + 1)
}

function humanSize(bytes) {
  if (bytes === null || bytes === undefined) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function detailOf(body, fallback) {
  const detail = typeof body?.detail === 'string' ? body.detail : ''
  return detail || fallback
}

// The tab strip is keyed by `key`, not by `path`: a file buffer and a diff of
// that same file are two tabs about one path, and they must not overwrite each
// other. For a file record `key === path`; for a diff it is `diff:<path>`, a
// prefix no workspace path can have (the server normalises them and they never
// start with a scheme-like token).
const DIFF_PREFIX = 'diff:'

function diffKeyFor(path) {
  return `${DIFF_PREFIX}${path}`
}

// The record for a tab key, or null when nothing is open under it.
function recordForKey(key) {
  return state.open.find((item) => item.key === key) ?? null
}

// The *file* buffer for a path, or null. Diff tabs are deliberately not found
// here: the tree's "Close" item means "close that file's buffer", and a diff
// tab of the same file must not make that promise true.
function recordFor(path) {
  return recordForKey(path)
}

function diffRecordFor(path) {
  return recordForKey(diffKeyFor(path))
}

export function useEditorFiles() {
  // Fetch and append-or-focus. A path that is already open is *focused*, not
  // refetched: with a read-only pane a refetch would jump the scroll position
  // for no benefit, and the freshness question belongs to ed-09's stat probe
  // rather than to every click.
  async function openFile(path) {
    if (!path) return null
    const existing = recordFor(path)
    if (existing) {
      state.active = path
      state.message = null
      return existing
    }
    state.loading = true
    state.message = null
    try {
      const res = await fetch(`/api/fs/file?path=${encodeURIComponent(path)}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        // 422 deny/escape, 413 over the cap, 404 gone, 400 a directory: all
        // carry a server-written reason, echoed with the path so a policy
        // refusal reads as "this file is protected", not as a broken tab.
        state.message = {
          path,
          tone: 'error',
          text: detailOf(body, `Could not open the file (${res.status})`),
        }
        return null
      }
      if (body.binary) {
        // Not an error — the file is fine, it is just not text. No tab chip:
        // there is nothing to render and nothing to keep open.
        state.message = {
          path,
          tone: 'info',
          text: `binary file, ${humanSize(body.size)}`,
        }
        return null
      }
      const record = {
        kind: 'file',
        key: body.path ?? path,
        path: body.path ?? path,
        name: nameOf(body.path ?? path),
        text: body.text ?? '',
        // The text as it was when this buffer was read (or last reloaded). A
        // buffer is dirty while `text` differs from it; an edit never moves the
        // baseline, a reload does.
        baseline: body.text ?? '',
        dirty: false,
        mtime: body.mtime ?? null, // ed-09 compares this against /api/fs/stat
        language: body.language ?? 'plaintext',
        size: body.size ?? (body.text ?? '').length,
        binary: false,
        truncated: !!body.truncated,
        truncatedAt: body.truncatedAt ?? null,
        encoding: body.encoding ?? 'utf-8',
        stale: false, // set by ed-09 when the stat probe disagrees
        reloadedAt: 0, // bumped by reloadFile so ed-04 can swap the document
      }
      state.open.push(record)
      state.active = record.key
      return record
    } catch (exc) {
      state.message = { path, tone: 'error', text: `Could not reach the API: ${exc}` }
      return null
    } finally {
      state.loading = false
    }
  }

  // Fetch and append-or-focus the *diff* of a path: the same append-or-focus
  // rule as a file, on the tab keyed `diff:<path>`. The pane offers this only
  // for a file git reports as comparable, so there is always a committed side;
  // the server's own answer is what the tab renders, never a local guess.
  async function openDiff(path) {
    if (!path) return null
    const key = diffKeyFor(path)
    const existing = recordForKey(key)
    if (existing) {
      state.active = key
      state.message = null
      return existing
    }
    state.loading = true
    state.message = null
    try {
      const res = await fetch(`/api/fs/diff?path=${encodeURIComponent(path)}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.message = {
          path,
          tone: 'error',
          text: detailOf(body, `Could not read the diff (${res.status})`),
        }
        return null
      }
      if (body.binary) {
        // The same honest answer as a file click: a diff of two opaque blobs
        // is not something to render, and the message says why.
        state.message = { path, tone: 'info', text: 'binary file — no diff to show' }
        return null
      }
      const record = {
        kind: 'diff',
        key,
        path: body.path ?? path,
        // The chip says which file *and* that it is a diff, so two chips for
        // one path are never confusable when a long name is ellipsised.
        name: `${nameOf(body.path ?? path)} (diff)`,
        original: body.original ?? '',
        modified: body.modified ?? '',
        language: body.language ?? 'plaintext',
        // 'modified' | 'added' | 'untracked' | 'deleted' | 'renamed' | 'clean'
        status: body.status ?? 'modified',
        baseline: body.baseline ?? 'head',
        renamedFrom: body.renamedFrom ?? null,
        truncated: !!body.truncated,
        truncatedAt: body.truncatedAt ?? null,
      }
      state.open.push(record)
      state.active = key
      return record
    } catch (exc) {
      state.message = { path, tone: 'error', text: `Could not reach the API: ${exc}` }
      return null
    } finally {
      state.loading = false
    }
  }

  // Close one tab by its key — the shared half of close/focus, so the
  // neighbour rule is written once for files and diffs alike.
  function closeTab(key) {
    const index = state.open.findIndex((item) => item.key === key)
    if (index === -1) return
    state.open.splice(index, 1)
    // The message belongs to the click that produced it, not to whatever
    // buffer is on screen now.
    state.message = null
    if (state.active !== key) return
    // Focus the neighbour, the way a browser closes a tab: the one to the
    // left when there was one, otherwise the new first tab, otherwise empty.
    const next = state.open[index - 1] ?? state.open[0] ?? null
    state.active = next ? next.key : null
  }

  // Close the *file* buffer for a path (a diff tab of the same path is a
  // different tab and stays open).
  function closeFile(path) {
    closeTab(path)
  }

  function closeDiff(path) {
    closeTab(diffKeyFor(path))
  }

  function focusTab(key) {
    if (!recordForKey(key)) return
    state.active = key
    // Switching to an open buffer is a new intent: a refusal or a binary note
    // from an earlier click stops being relevant to what is on screen.
    state.message = null
  }

  function focusFile(path) {
    focusTab(path)
  }

  // Apply an edit from Monaco to the active buffer. Only reached while the pane
  // is in read_write mode (the chat is idle); a read-only surface cannot type.
  // The baseline is untouched, so `dirty` is exactly "differs from what was
  // read" — and an edit that types its way back to the original clears the dot
  // on its own.
  function updateText(path, text) {
    const record = recordFor(path)
    if (!record || record.text === text) return
    record.text = text
    record.size = text.length
    record.dirty = text !== record.baseline
  }

  // Re-GET and swap the text. Driven by ed-09's probe (the model wrote the
  // file), never by a Save — there is nothing to save in this lot.
  async function reloadFile(path) {
    const record = recordFor(path)
    if (!record) return null
    try {
      const res = await fetch(`/api/fs/file?path=${encodeURIComponent(path)}`)
      const body = await res.json().catch(() => null)
      if (!res.ok || body?.binary) return null
      const fresh = body.text ?? ''
      record.text = fresh
      // A reload is a fresh read from disk, so the disk text becomes the new
      // baseline: the buffer is clean against it again.
      record.baseline = fresh
      record.dirty = false
      record.size = body.size ?? record.size
      record.mtime = body.mtime ?? record.mtime
      record.truncated = !!body.truncated
      record.truncatedAt = body.truncatedAt ?? null
      record.stale = false
      record.reloadedAt += 1
      return record
    } catch {
      return null
    }
  }

  function clearMessage() {
    state.message = null
  }

  return {
    state,
    openFile,
    openDiff,
    closeFile,
    closeDiff,
    closeTab,
    focusFile,
    focusTab,
    updateText,
    reloadFile,
    recordFor,
    recordForKey,
    diffRecordFor,
    diffKeyFor,
    clearMessage,
    humanSize,
  }
}
