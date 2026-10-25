import { reactive } from 'vue'

// ---------------------------------------------------------------------------
// Open files: the state behind the Editor tab's code pane (ed-03).
//
// One singleton like every other screen's: the open buffers, the active tab
// and what the pane should say live *here*, so a tab switch (`v-show`) keeps
// them and ed-04's CodeMirror component only renders and disposes a view.
//
// Read-only by design (decision D4): there is no `dirty` field, nothing to
// confirm on close and no `beforeunload` warning, because there is no write
// endpoint to be stale about. Those belong to the write layer of ed-10.
//
// The tree's `selected` (useWorkspaceTree) and this `active` are deliberately
// separate: selecting a row highlights it, opening a file puts it in the tab
// strip. They coincide in the common case and diverge exactly when a click is
// refused (binary, over the size cap) — which is when one shared value would
// lie about what is on screen.

// Above this many lines the raw `<pre>` preview of ed-03 stops rendering a gutter and
// says so instead: a 100k-line v-for would hang the tab, and ed-04's
// CodeMirror (virtualised) replaces this preview entirely.
export const MAX_GUTTER_LINES = 2000

const state = reactive({
  // Open buffers, in the order they were opened. Only *text* files get a
  // record: a binary or oversized file is a message in the pane, not a tab.
  open: [],
  active: null, // path of the visible buffer (the tab strip's selection)
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

// The record for a path, or null when it is not open.
function recordFor(path) {
  return state.open.find((item) => item.path === path) ?? null
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
        path: body.path ?? path,
        name: nameOf(body.path ?? path),
        text: body.text ?? '',
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
      state.active = record.path
      return record
    } catch (exc) {
      state.message = { path, tone: 'error', text: `Could not reach the API: ${exc}` }
      return null
    } finally {
      state.loading = false
    }
  }

  function closeFile(path) {
    const index = state.open.findIndex((item) => item.path === path)
    if (index === -1) return
    state.open.splice(index, 1)
    // The message belongs to the click that produced it, not to whatever
    // buffer is on screen now.
    state.message = null
    if (state.active !== path) return
    // Focus the neighbour, the way a browser closes a tab: the one to the
    // left when there was one, otherwise the new first tab, otherwise empty.
    const next = state.open[index - 1] ?? state.open[0] ?? null
    state.active = next ? next.path : null
  }

  function focusFile(path) {
    if (!recordFor(path)) return
    state.active = path
    // Switching to an open buffer is a new intent: a refusal or a binary note
    // from an earlier click stops being relevant to what is on screen.
    state.message = null
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
      record.text = body.text ?? ''
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
    closeFile,
    focusFile,
    reloadFile,
    recordFor,
    clearMessage,
    humanSize,
    MAX_GUTTER_LINES,
  }
}
