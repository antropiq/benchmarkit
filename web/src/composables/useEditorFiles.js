import { computed, reactive } from 'vue'
// The one renderer the app owns (marked + KaTeX + hljs + DOMPurify). A preview
// tab renders through it exactly as the chat does — one renderer, one owner.
import { renderMarkdown } from '../lib/markdown'
import { useBenchmark } from './useBenchmark'
import { useChat } from './useChat'
import { useEditorNotices } from './useEditorNotices'
import { useGitStatus } from './useGitStatus'
import { registerDeleteChord, registerReRootHandler, useWorkspaceTree } from './useWorkspaceTree'

// ---------------------------------------------------------------------------
// Open files: the state behind the Editor tab's code pane.
//
// One singleton like every other screen's: the open buffers, the active tab
// and what the pane should say live *here*, so a tab switch (`v-show`) keeps
// them and the Monaco component only renders and disposes a view.
//
// The pane's edit mode follows the WorkBench: it is read_write only while the
// chat is idle, read-only whenever a reply is streaming (EditorArea owns that
// rule and passes `readonly` down to Monaco). In read_write an edit updates the
// in-memory buffer and its `dirty` flag. `saveFile` sends one buffer to the
// server's guarded write route (`PUT /api/fs/file`), reached from the pane's
// Save buttons and the Ctrl+S / Ctrl+Shift+S chords — `saveFile` is the only
// thing that writes, and `requestCloseTab` is the only way a tab closes. What
// makes a save safe is the record's `mtime`: it goes back as the guard's
// `ifUnmodifiedSince` token, and a disk that moved since the read answers 409
// instead of being overwritten.
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
//
// Freshness: the WorkBench and the terminal are live writers of the same files,
// so a buffer is only trustworthy until the disk says otherwise. Nothing here
// watches the filesystem (a watcher on a working location of `$HOME` is a
// file-descriptor problem, and neither of those writers can report a path
// reliably); instead the pane asks `GET /api/fs/stat` for exactly the paths it
// holds open — one request per tick, file records only. What that answer
// *means* per buffer is the table in `applyProbeAnswer`: a clean buffer follows
// the disk silently (it has nothing to lose), a dirty one carries a marker and,
// if it is the buffer on screen, a conflict the user answers.

// The pane renders every buffer through Monaco (virtualised), so line count
// costs nothing here — the old hand-rolled gutter cap it needed is gone.
const state = reactive({
  // Open buffers, in the order they were opened. Only *text* files get a
  // record: a binary or oversized file is a message in the pane, not a tab.
  // A diff (ed: "View diff…") is a record too — same strip, same rules — with
  // `kind: 'diff'` and two texts instead of one. A rendered markdown preview
  // (the tree's "Preview") is a record too, `kind: 'preview'`, holding the
  // snapshot it was rendered from rather than a live buffer.
  open: [],
  active: null, // key of the visible buffer (the tab strip's selection)
  loading: false,
  // What the last click answered when it did not produce a buffer:
  // { path, text, tone }, tone 'error' being a refusal (policy, size cap,
  // missing) and 'info' an answer that is not an error — "binary file,
  // 2.1 MB". It rides above whatever buffer is still on screen, and clears on
  // the next focus or close.
  message: null,
  // The freshness probe could not answer (network, 422, 500): a sentence, or
  // null. A failed probe is *not* evidence that a file changed, so while this
  // is set no record moves. It is the *pane's* line about a broken poll (it
  // clears itself on the next good answer); the copy that reaches a user whose
  // eyes are on the tree is the one `warn` notice the failure raises.
  probeError: null,
  // The close confirmation on screen: `{ key, name }` for a tab whose buffer is
  // dirty, or null. Only a dirty buffer can raise it — a clean one and any diff
  // tab close on the gesture that asked for it.
  closePrompt: null,
  // The delete confirmation on screen: `{ path, name, dir, dirtyBuffers }` for
  // the path the tree asked to delete, or null. Raised by `requestDelete` for
  // every path — a delete is destructive whatever the buffers hold, so the
  // gesture is never the whole decision. `dirtyBuffers` is the count of open
  // buffers under that path with unsaved edits: the one case the dialog warns
  // about in its own line (one dialog for one action, not two).
  deletePrompt: null,
  // The one conflict on screen: `{ key, kind, diskMtime, raisedAt }` for the
  // *active* buffer, or null. Separate from the per-record flags so the dialog
  // the pane shows is about what the user is looking at, while a buffer they are
  // not looking at just carries its marker. Unlike `message`, this is not
  // cleared by the next click — a conflict survives until it is answered.
  conflict: null,
  // Which conflict the user has already turned down (`raisedAt` of it). The
  // dialog shows the current one only when it is not this, which is what keeps
  // a dismissed conflict from returning as a modal on the next probe tick while
  // keeping its marker on the chip.
  conflictDismissedAt: 0,
})

// How many buffers hold unsaved work. The pane reads it twice: it shows the
// Save-all button, and it words the tab strip's title ("3 unsaved"). A computed
// on the singleton rather than a field, so it can never drift from the records
// it counts — the dot on each chip is the same predicate.
const dirtyCount = computed(
  () => state.open.filter((item) => item.kind === 'file' && item.dirty).length,
)

// A reload or a closed tab with a dirty buffer loses work that exists nowhere
// else, so the browser asks first. It lives here because the condition belongs to
// the buffers, not to the terminal (whose own `pagehide` teardown sits next to
// the commented-out version this replaced — see useTerminal.js).
//
// Two separate listeners, and that separation is the point: this one may be
// cancelled by the user choosing to stay, and staying must *not* have torn down
// a shell — so the terminal's kill stays on `pagehide`, which is the event that
// only fires when the document really goes away. A `beforeunload` that both
// prompted and closed the socket would destroy a session for a page that stayed.
//
// Registered once per module rather than per component: the pane is a lazy chunk
// that mounts and unmounts, and the question outlives any one view of it.
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', (event) => {
    if (dirtyCount.value === 0) return
    // The standard shape: preventDefault plus a returnValue, which is what makes
    // the browsers that honour it show their own "Leave this site?" prompt.
    event.preventDefault()
    event.returnValue = ''
  })
}

// How long a buffer may be trusted before the pane asks again. A module
// constant, and a named export rather than one more function return, because
// the tick is state of the open buffers: the component that mounts the pane only
// starts and stops it. Mirrors the patchable caps in `editorfs.constants`, so a
// test can shorten the interval instead of waiting for it.
export const FRESHNESS_POLL_MS = 5000

// Slack when comparing a buffer's mtime against the disk one: the same number
// and the same argument as `editorfs.constants.MTIME_GUARD_EPSILON`. A
// filesystem tick and a JSON float round-trip are not exact, so an exact
// comparison would report a change for a save that only just landed.
const MTIME_EPSILON = 0.01

// How long the server keeps a trashed path, quoted in the delete confirmation.
// The mirror of `editorfs.constants.TRASH_RETENTION_DAYS`, same reasoning as
// MTIME_EPSILON: the *question* is asked before the request exists, so the
// promise has to be statable client-side. The server's answer names the folder
// it actually used.
export const TRASH_RETENTION_DAYS = 30

// Where a delete files the copy: the workspace's reserved folder. The mirror
// of `editorfs.constants.TRASH_SUBDIR` under `.nojae` — same "asked before the
// request exists" reasoning as TRASH_RETENTION_DAYS, and the reason the label
// is only a *folder name*: the exact batch is the route's `trashedTo`, and
// `.nojae` is hidden from the tree, so the pane can never open either one.
// `/api/policy` publishes the real reserved name if this mirror ever needs to
// stop being one.
export const TRASH_FOLDER = '.nojae/trash'

// The two halves of one path, joined/split the way the server spells them (""
// is the workspace root).
function parentOf(path) {
  const index = path.lastIndexOf('/')
  return index === -1 ? '' : path.slice(0, index)
}

// True when `candidate` is the path itself or something under it — the prefix
// rule a delete uses to find every buffer whose file is about to be trashed.
// The trailing '/' is what keeps `src/x.pyw` from matching `src/x.py`.
//
// A missing candidate (no open record, no tree selection) is not under anything:
// the callers pass optional chains, and one undefined must not turn a delete
// into a TypeError — the same rule `useWorkspaceTree.sameOrUnder` keeps.
function underPath(candidate, path) {
  if (typeof candidate !== 'string') return false
  return candidate === path || candidate.startsWith(`${path}/`)
}

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
// A rendered preview is a third record kind with a prefix of its own, so a file
// buffer, its diff and its preview are three tabs about one path that can never
// collide (see `openPreview`).
const PREVIEW_PREFIX = 'preview:'

function diffKeyFor(path) {
  return `${DIFF_PREFIX}${path}`
}

function previewKeyFor(path) {
  return `${PREVIEW_PREFIX}${path}`
}

// The tab key and the chip label for one record kind. Both are *derived* from
// (kind, path) — the only place that spells the rules out, so `repointPath` can
// re-key a renamed record of any kind without a chain of `kind ===` tests that a
// fourth kind would silently fall through.
function keyFor(kind, path) {
  if (kind === 'diff') return diffKeyFor(path)
  if (kind === 'preview') return previewKeyFor(path)
  return path
}

function labelFor(kind, path) {
  if (kind === 'diff') return `${nameOf(path)} (diff)`
  if (kind === 'preview') return `${nameOf(path)} (preview)`
  return nameOf(path)
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

function previewRecordFor(path) {
  return recordForKey(previewKeyFor(path))
}

export function useEditorFiles() {
  // Write feedback goes to the app's one notice stack (useEditorNotices), not
  // to `state.message`: a refused save is news about work the buffer still
  // holds, and a refused delete or rename happens while the user is looking at
  // the tree. `state.message` keeps its exact job — the one-line answer to a
  // refused *open*, cleared by the next click.
  const notices = useEditorNotices()

  // Fetch and append-or-focus. A path that is already open is *focused*, not
  // refetched: a refetch would jump the scroll position for no benefit, and the
  // freshness question belongs to the stat probe rather than to every click.
  async function openFile(path) {
    if (!path) return null
    const existing = recordFor(path)
    if (existing) {
      state.active = path
      state.message = null
      syncConflict()
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
        mtime: body.mtime ?? null, // compared against /api/fs/stat, and the
        // save guard sends it back as the ifUnmodifiedSince token
        language: body.language ?? 'plaintext',
        size: body.size ?? (body.text ?? '').length,
        binary: false,
        truncated: !!body.truncated,
        truncatedAt: body.truncatedAt ?? null,
        encoding: body.encoding ?? 'utf-8',
        // What the freshness probe says about this buffer against the disk:
        // `stale` when the disk's mtime moved, `deleted` when the path is gone.
        stale: false,
        deleted: false,
        // `null | 'stale' | 'deleted'` — a conflict is the *marked* state that
        // needs an answer: the disk moved and this buffer is dirty, so one of
        // the two copies is going to be lost. A clean buffer whose file changed
        // is `stale` but never in conflict (see `applyProbeAnswer`).
        // `conflictDetail` carries both mtimes when a 409 raised it, and
        // `conflictAt` when it was raised.
        conflict: null,
        conflictAt: 0,
        conflictDetail: null,
        // In flight to `PUT /api/fs/file`. Also the re-entrancy guard: a held
        // Ctrl+S must not fire a second save of the same buffer.
        saving: false,
        reloadedAt: 0, // bumped by reloadFile so the pane swaps the document
      }
      state.open.push(record)
      state.active = record.key
      syncConflict()
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
        name: labelFor('diff', body.path ?? path),
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

  // Fetch and show a markdown file *rendered*, through the app's one renderer
  // (`lib/markdown.js`) — a sibling surface in the same pane, not a Monaco
  // feature: the bundled monaco ESM build ships no markdown preview widget, and
  // rendering through anything else would fork the app's markdown rules.
  //
  // The record owns a **snapshot**: `text` and `html` are computed once, from
  // the one read that opened the tab, exactly as a diff owns its two immutable
  // sides. That is what buys no re-render on keystrokes, no debounce and no
  // second mutable copy of a buffer — and it is why a preview is deliberately
  // outside the freshness probe. The cost is stated in the pane's note line
  // ("rendered from the copy read when this tab opened"), and the refresh is the
  // user's: unlike `openDiff`'s pure append-or-focus, re-picking **Preview** on a
  // row whose tab already exists *re-reads and re-renders* it. That re-read is
  // the whole staleness story for v1 (it resets the panel's scroll position — a
  // rendered DOM has no model to restore it from).
  //
  // A preview is not a buffer: it has no `dirty`, no `baseline`, no `conflict`
  // and no save. Every buffer rule in this file already filters
  // `kind === 'file'`, so it is excluded from the dirty count, the close prompt,
  // `saveFile`, `saveAllDirty`, the probe and the delete warning for free — do
  // not widen any of those.
  //
  // Deliberately *not* reachable as a tree selection: the menu's Preview item
  // calls this directly (as `view-diff` does), because publishing a selection
  // would make EditorArea's watcher open the Monaco buffer beside it.
  async function openPreview(path) {
    if (!path) return null
    const existing = previewRecordFor(path)
    if (existing) {
      state.active = existing.key
      state.message = null
      // …then read again: focus *and* refresh, the manual half of D2.
      await refreshPreview(existing)
      return existing
    }
    state.loading = true
    state.message = null
    try {
      const res = await fetch(`/api/fs/file?path=${encodeURIComponent(path)}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        // The same answers `openFile` gives, so refusal wording has one home:
        // 404 for a hidden or missing path, 422 a policy refusal, 413 over the
        // cap — the server's sentence, with the path beside it.
        state.message = {
          path,
          tone: 'error',
          text: detailOf(body, `Could not open the file (${res.status})`),
        }
        return null
      }
      if (body.binary) {
        state.message = {
          path,
          tone: 'info',
          text: `binary file, ${humanSize(body.size)}`,
        }
        return null
      }
      const recordPath = body.path ?? path
      const text = body.text ?? ''
      const record = {
        kind: 'preview',
        key: previewKeyFor(recordPath),
        path: recordPath,
        name: labelFor('preview', recordPath),
        // What this render was made of. Kept on the record so the pane can say
        // so, and so a refresh has something to compare its new read against.
        text,
        html: renderMarkdown(text, { mermaid: true }),
        language: body.language ?? 'plaintext',
        size: body.size ?? text.length,
        truncated: !!body.truncated,
        truncatedAt: body.truncatedAt ?? null,
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

  // Re-read the file behind an open preview and re-render it. A read that fails
  // or turns out binary leaves the *existing* render standing — half a preview
  // is worse than a stale one, and the answer to a refusal is the message line,
  // not the disappearance of what the user was reading.
  async function refreshPreview(record) {
    if (!record || record.kind !== 'preview') return null
    try {
      const res = await fetch(`/api/fs/file?path=${encodeURIComponent(record.path)}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.message = {
          path: record.path,
          tone: 'error',
          text: detailOf(body, `Could not open the file (${res.status})`),
        }
        return record
      }
      if (body.binary) {
        state.message = {
          path: record.path,
          tone: 'info',
          text: `binary file, ${humanSize(body.size)}`,
        }
        return record
      }
      const text = body.text ?? ''
      record.text = text
      record.html = renderMarkdown(text, { mermaid: true })
      record.language = body.language ?? record.language
      record.size = body.size ?? text.length
      record.truncated = !!body.truncated
      record.truncatedAt = body.truncatedAt ?? null
      return record
    } catch (exc) {
      state.message = {
        path: record.path,
        tone: 'error',
        text: `Could not reach the API: ${exc}`,
      }
      return record
    }
  }

  // Close one tab by its key — the shared half of close/focus, so the
  // neighbour rule is written once for files and diffs alike.
  function closeTab(key) {
    const index = state.open.findIndex((item) => item.key === key)
    if (index === -1) return
    const record = state.open[index]
    const path = record.path
    const wasFile = record.kind === 'file'
    state.open.splice(index, 1)
    // The message belongs to the click that produced it, not to whatever
    // buffer is on screen now. A *conflict* is different: it lives on the
    // record, so closing a conflicted buffer answers it — discarding the
    // buffer is a choice the user made on purpose, not something to warn
    // them about twice.
    state.message = null
    // A *file* buffer going away ends the work its write feedback was about, so
    // the lines about that path leave with it. A diff tab is a view, not a
    // buffer: closing it leaves the file (and its unsaved edits, and the notices
    // about them) standing, so it dismisses nothing.
    if (wasFile) notices.dismissForPath(path)
    if (state.active !== key) return
    // Focus the neighbour, the way a browser closes a tab: the one to the
    // left when there was one, otherwise the new first tab, otherwise empty.
    const next = state.open[index - 1] ?? state.open[0] ?? null
    state.active = next ? next.key : null
    syncConflict()
  }

  // Every close gesture — the chip's ×, the middle-click, Ctrl+W, the tree's
  // "Close" — goes through here rather than `closeTab` directly, because a
  // dirty buffer cannot be closed by one gesture: its edits live only in this
  // page, so there is no trash can behind a closed tab and this prompt is the
  // last line of defence. A clean buffer and any diff tab have nothing to lose
  // and close on the spot.
  //
  // The answer set itself is the pane's dialog (state.closePrompt); what lives
  // here is the decision to show it and the two answers that end in a close.
  function requestCloseTab(key) {
    const record = recordForKey(key)
    if (!record) return { closed: false, reason: 'no-buffer' }
    if (record.saving) return { closed: false, reason: 'in-flight' }
    if (record.kind === 'file' && record.dirty) {
      state.closePrompt = { key: record.key, name: record.name }
      return { closed: false, reason: 'confirm' }
    }
    closeTab(key)
    return { closed: true }
  }

  // Cancel is the default answer of that dialog: a destructive default in a
  // confirm is a bug. Nothing moves.
  function cancelClose() {
    state.closePrompt = null
  }

  // Discard, said plainly because it is what it is: the buffer is dropped and
  // with it the only copy of the edits.
  function discardClose() {
    const prompt = state.closePrompt
    state.closePrompt = null
    if (prompt) closeTab(prompt.key)
  }

  // Save, and close only if the save actually landed. When it did not — a
  // reply streaming, the disk having moved — the buffer stays open and the
  // refusal is shown instead: closing would throw the work away, which is the
  // one thing this dialog exists to prevent.
  async function saveAndClose() {
    const prompt = state.closePrompt
    if (!prompt) return null
    const result = await saveFile(prompt.key)
    if (!result.ok) {
      state.closePrompt = null
      return result
    }
    state.closePrompt = null
    closeTab(prompt.key)
    return result
  }

  // -- delete: the tree asks, this layer answers ------------------------------
  //
  // The route's rules are the server's (see DELETE /api/fs/file); what lives
  // here is the client's half of the promise: *nothing* is deleted on one
  // gesture, and when a delete does land, no tab is left pointing at a path
  // that is in the trash.
  //
  // Every open record whose path is the deleted one *or sits under it* closes —
  // its file has no future, so a confirm for each would only negotiate with a
  // fact. The one exception is information, not an extra dialog: `requestDelete`
  // counts the dirty buffers under the path and the single dialog says what will
  // be lost, because unsaved edits are the one thing the trash cannot bring back.

  // Every record — file buffer, diff tab or preview — whose path is the deleted
  // one or sits under it. A diff tab is closed too: `refreshBranch` drops its row
  // and a reveal of it would fail, so leaving the chip behind points at nothing.
  // A preview goes for the same reason — its file has no future.
  function recordsUnder(path) {
    return state.open.filter((item) => underPath(item.path, path))
  }

  // Only file buffers can hold unsaved work, so that is what the warning
  // counts (a diff tab is a view, never a buffer).
  function buffersUnder(path) {
    return state.open.filter((item) => item.kind === 'file' && underPath(item.path, path))
  }

  // Ask. `entry` is the tree row ({ path, name, dir }) — the row is what the
  // user looked at, so it is what the dialog names. Raised for every path: a
  // delete is destructive whatever the buffers hold.
  function requestDelete(entry) {
    if (!entry?.path) return { deleted: false, reason: 'no-path' }
    state.deletePrompt = {
      path: entry.path,
      name: entry.name ?? nameOf(entry.path),
      dir: !!entry.dir,
      dirtyBuffers: buffersUnder(entry.path).filter((item) => item.dirty).length,
    }
    return { deleted: false, reason: 'confirm' }
  }

  // Cancel is the default answer of that dialog too — the destructive one is
  // the second button, so Esc, the backdrop and Enter-on-first-focus all keep
  // the file.
  function cancelDelete() {
    state.deletePrompt = null
  }

  // The one function that deletes; the dialog's Delete button and nothing else
  // calls it. Whether the path was a file or a folder is the server's answer
  // (`dir` in the payload), never a guess from the row that was clicked — the
  // row may predate a change on disk.
  //
  // State first, like every other mutation here: any earlier refusal about this
  // path is dismissed before the request, so the stack never shows the *previous*
  // delete as news about this one. On success the records under the path close
  // (which dismisses their refusal lines too), the one folder that showed the row
  // is re-listed, the git marks are re-read, and the confirmation names the trash
  // folder the server actually used plus the retention the dialog promised. On
  // refusal nothing moves — no tab closes, the tree keeps its row, and the
  // server's own sentence is the notice.
  async function trashPath(path) {
    if (!path) return { ok: false, reason: 'no-path' }
    notices.dismissForPath(path)
    let res
    let body = null
    try {
      res = await fetch(`/api/fs/file?path=${encodeURIComponent(path)}`, {
        method: 'DELETE',
      })
      body = await res.json().catch(() => null)
    } catch (exc) {
      const text = `Could not delete ${path}: the API could not be reached`
      notices.notice('error', text, path)
      return { ok: false, reason: 'network', text }
    }

    if (!res.ok) {
      // The connector's words, the deny-list's words, "not there", "a symlink":
      // all are the server's sentences, shown as it wrote them, prefixed by what
      // was attempted.
      const text = `Could not delete ${path}: ${detailOf(body, `the request was refused (${res.status})`)}`
      notices.notice('error', text, path)
      return { ok: false, reason: 'refused', path, text }
    }

    for (const record of recordsUnder(path)) closeTab(record.key)
    const tree = useWorkspaceTree()
    // The row is about to disappear from the listing, and a `selected` pointing
    // at it is a selection the tree can no longer show — the same reason
    // `revalidateFocus` walks the focus up after a re-list.
    if (underPath(tree.state.selected?.path, path)) tree.state.selected = null
    // A trashed *folder* leaves more than its own row behind: the listings the
    // tree cached for it and everything below it, plus their expanded flags.
    // Those keys are the deleted path's, so they must go — a folder re-created
    // under the same name would otherwise expand into children that are on no
    // disk. `dropBranch` is the same forget-the-branch step a rename does.
    tree.dropBranch(path)
    await tree.refreshBranch(parentOf(path))
    // One level higher as well: the folder this row lived in may just have gone
    // empty, and the arrow on *its* row is drawn from the listing above.
    await tree.refreshListingOfParent(path)
    void useGitStatus().load({ force: true })

    const trashedTo = body?.trashedTo ?? ''
    notices.notice(
      'ok',
      `Deleted ${path}` +
        (trashedTo ? ` — kept in ${trashedTo} for ${TRASH_RETENTION_DAYS} days` : ''),
      path,
    )
    return {
      ok: true,
      path,
      dir: !!body?.dir,
      entries: body?.entries ?? 0,
      trashedTo,
    }
  }

  // The dialog's destructive answer. The prompt is cleared before the request
  // so the dialog is gone while the tree re-lists — a modal over a changing
  // tree is the worst of both.
  async function confirmDelete() {
    const prompt = state.deletePrompt
    state.deletePrompt = null
    if (!prompt) return { ok: false, reason: 'no-prompt' }
    return trashPath(prompt.path)
  }

  // The tree's `Delete` key reaches `requestDelete` through the registry in
  // useWorkspaceTree, the same one-way shape the create chord uses: the tree
  // owns the keyboard, this layer owns the question and the request.
  registerDeleteChord((entry) => requestDelete(entry))

  // …and the same one-way registry for a working-location switch: the tree owns
  // the re-rooting (its listings are the paths that moved), this layer owns what
  // happens to the tabs that held them. Registered from here rather than called
  // from the save path, so the app's default screen never pulls the Editor's
  // chunk in for state that only the Editor has — and a location the Editor tab
  // was never opened for has no tabs to forget.
  registerReRootHandler(({ workspace = '' } = {}) => forgetWorkspaceBuffers({ workspace }))

  // Close the *file* buffer for a path (a diff tab of the same path is a
  // different tab and stays open).
  //
  // A preview tab of that path stays open too, and that is the promise, not an
  // oversight: the tree's `Close` means "that file's buffer" — the same promise a
  // diff already refuses to make (`hasBuffer` in WorkspaceNavigator reads
  // `recordFor`, which is file-only). A rendered snapshot has no buffer to close,
  // and it is dismissed by its own ×.
  function closeFile(path) {
    requestCloseTab(path)
  }

  function closeDiff(path) {
    closeTab(diffKeyFor(path))
  }

  function focusTab(key) {
    if (!recordForKey(key)) return
    state.active = key
    // The conflict on screen follows the buffer on screen: another tab's
    // conflict is not a dialog about what is being read now.
    syncConflict()
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
  //
  // `text` is what the Monaco model reports, line endings included: the model
  // takes its EOL from the file it was loaded with and a save sends this string
  // back verbatim, so the server never has to normalise anything (D5).
  function updateText(path, text) {
    const record = recordFor(path)
    if (!record || record.text === text) return
    record.text = text
    record.size = text.length
    record.dirty = text !== record.baseline
  }

  // Send one buffer to the server's guarded save route. The wire rule is the
  // server's (see PUT /api/fs/file); what belongs *here* is only the two
  // things the client alone can know:
  //
  //   * a buffer that was read `truncated` is half a file, so sending it back
  //     would delete the rest — refused before the request, and the server
  //     refuses on its own terms too;
  //   * the token is the mtime that read reported, sent back as
  //     `ifUnmodifiedSince`, so a disk that moved underneath this buffer comes
  //     back as 409 with both numbers instead of a silent overwrite.
  //
  // A successful save re-baselines: the saved text becomes the new baseline,
  // `dirty` clears, and the payload's mtime becomes the next token — the
  // server's truth, never a local guess. A conflict is recorded on the record
  // (the reload-or-overwrite choice is the pane's to offer, not this layer's
  // to make). No keybinding is attached here: the pane wires that.
  //
  // `{ force: true }` omits the token — "I know the disk moved, I merged by
  // hand". It is the *only* caller allowed to do that, and the server logs a
  // token-less save for exactly that reason.
  //
  // Everything else about it is state the UI reads: `saving` marks the chip and
  // re-entrancy-guards a held chord, `baseline` moves *only* on a confirmed 200
  // (a failed save keeps the dot), and a refusal that is not a conflict — the
  // chat-stream 409, a policy 422, an over-cap 413, the API being away — becomes
  // a notice in the app's stack, because a buffer that could not be written is
  // work the user still holds. A *conflict* deliberately raises none: the
  // conflict dialog is the interaction for that one, and a refusal line behind a
  // modal is noise (the probe raises one for a conflict on a buffer the user is
  // *not* looking at, where no dialog appears).
  //
  // A 200 also refreshes the git marks: the tree's modified/untracked dots are
  // one query for the whole work tree, and a file the user *just* saved would
  // otherwise keep looking unmodified until something else asked.
  async function saveFile(key, { force = false } = {}) {
    const record = recordForKey(key)
    if (!record || record.kind !== 'file') return { ok: false, reason: 'not-savable' }
    if (record.truncated || record.binary) {
      // A half-read buffer is not the file, and a buffer was never opened for
      // binary content in the first place — either would destroy what is on
      // disk. The server refuses both on its own terms too.
      return { ok: false, reason: record.truncated ? 'truncated' : 'binary' }
    }
    if (record.saving) return { ok: false, reason: 'in-flight' }
    record.saving = true
    let res
    let body = null
    try {
      res = await fetch('/api/fs/file', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: record.path,
          text: record.text,
          ...(force ? {} : { ifUnmodifiedSince: record.mtime }),
        }),
      })
      body = await res.json().catch(() => null)
    } catch (exc) {
      const text = `Could not save ${record.path}: the API could not be reached`
      notices.notice('error', text, record.path)
      return { ok: false, reason: 'network', text }
    } finally {
      record.saving = false
    }

    if (res.ok) {
      // The notice about a failed write to *this* path is answered by this one
      // landing; leaving it up would say the buffer is still unwritten.
      notices.dismissForPath(record.path)
      record.baseline = record.text
      record.dirty = false
      record.size = body?.size ?? record.size
      record.mtime = body?.mtime ?? record.mtime
      // A save that landed means the path is there and it agrees with the
      // buffer, so every freshness marker this record carried is now false.
      record.deleted = false
      record.stale = false
      record.conflict = null
      record.conflictDetail = null
      syncConflict()
      // The two sides are the same text now, so a comparison of them has
      // nothing left to show.
      dropConflictDiff(record.path)
      // A write that landed is news the tree draws, so the marks are asked for
      // again — *forced*, because the cached answer predates this save. (A
      // status query already in flight is joined rather than repeated.)
      void useGitStatus().load({ force: true })
      return { ok: true, path: record.path }
    }

    // 409 whose detail is the {reason, path, diskMtime, bufferMtime} object:
    // the disk moved, this save did not land, and both numbers are kept so the
    // pane can name them when it offers reload-or-overwrite.
    if (res.status === 409 && body?.detail && typeof body.detail === 'object') {
      record.stale = true
      setConflict(record, 'stale', {
        diskMtime: body.detail.diskMtime ?? null,
        bufferMtime: body.detail.bufferMtime ?? null,
      })
      // For the buffer on screen, the *dialog* is the interaction and a refusal
      // line behind a modal is noise. For one the pane is not showing — a
      // save-all that stopped on a background tab — no dialog will appear, so
      // the conflict needs its one line here.
      //
      // `warn`, not `error`, on purpose: this is not the server refusing a write
      // (those are `error` and survive a focus change — a click must not eat the
      // reason a buffer is still unsaved). It is news that the disk moved, which
      // is exactly what `syncConflict` hands to the dialog when the user turns
      // to that tab.
      if (state.active !== record.key) {
        notices.notice(
          'warn',
          `Could not save ${record.path}: the file changed on disk — resolve the conflict first`,
          record.path,
        )
      }
      syncConflict()
      return { ok: false, reason: 'conflict', path: record.path }
    }

    // Everything else — the chat-stream 409 (whose detail is a sentence), a
    // policy 422, an over-cap 413, a 5xx: the server named the rule, the buffer
    // keeps its edits, and the refusal stays on screen until it is dismissed or
    // a write to that path lands. The server's sentence is shown verbatim behind
    // the attempt, never rewritten or shortened.
    const text = `Could not save ${record.path}: ${detailOf(body, `the request was refused (${res.status})`)}`
    notices.notice('error', text, record.path)
    return { ok: false, reason: 'refused', path: record.path, text }
  }

  // Save every dirty file buffer, one at a time and in the order of the tab
  // strip. Sequentially on purpose: parallel writes from one client are how a
  // self-inflicted 409 happens, and the answer the user sees should arrive in
  // the order they can see the chips in.
  //
  // A refusal stops the run: it usually names a reason that applies to the rest
  // (a reply streaming, the working location moving), and a half-written lot is
  // easier to read when the failure is the last thing that happened. The
  // buffers before it are saved, the rest keep their dots.
  //
  // One run at a time, joined like the probe and the git query: a held
  // Ctrl+Shift+S would otherwise queue a second pass over buffers the first pass
  // is still writing, and its first in-flight guard would read as a failure.
  let saveAllInflight = null

  async function saveAllDirty() {
    if (saveAllInflight) return saveAllInflight
    const run = (async () => {
      const keys = state.open
        .filter((item) => item.kind === 'file' && item.dirty)
        .map((item) => item.key)
      const saved = []
      let failure = null
      for (const key of keys) {
        const result = await saveFile(key)
        if (result.ok) saved.push(key)
        else {
          failure = result
          break
        }
      }
      return { saved, failure }
    })()
    saveAllInflight = run
    try {
      return await run
    } finally {
      saveAllInflight = null
    }
  }

  // Re-GET and swap the text. Driven by the freshness probe (the model wrote
  // the file underneath a buffer), never by a Save — a save already has the
  // disk's newest bytes.
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
      record.deleted = false
      record.stale = false
      // Disk's text is now the buffer's text, so any pending conflict about
      // this buffer is answered — whichever side was right.
      record.conflict = null
      record.conflictDetail = null
      record.reloadedAt += 1
      syncConflict()
      dropConflictDiff(record.path)
      // Taking the disk's copy is an answer to everything this buffer's write
      // feedback was about — the conflict and any refusal to send it.
      notices.dismissForPath(record.path)
      return record
    } catch {
      return null
    }
  }

  // -- freshness: does the disk still agree with this buffer? -----------------
  //
  // A watcher would need one file descriptor per path of a working location
  // that may be `$HOME`, and the app already knows two writers it could ask
  // instead — the chat stream and the terminal — neither of which can report
  // *which* file they touched. So the pane asks for the exact paths it holds
  // open: one `GET /api/fs/stat` per tick, and the answer is read against each
  // record's `mtime`.

  // Whether asking is worth a request at all. The tab strip is the only reader
  // of the answer, so a hidden Editor tab gets no poll; a streaming reply is a
  // known writer, the pane is read-only meanwhile, and a conflict the user
  // cannot act on is noise. And with only diff tabs open there is no buffer
  // that can be stale.
  function probeIsUseful() {
    if (useBenchmark().state.activeTab !== 'editor') return false
    if (useChat().state.phase !== 'idle') return false
    return state.open.some((item) => item.kind === 'file')
  }

  // What one stat answer means for one record: each of the four branches below
  // is a row of the table the pane's wording follows. A probe that says nothing
  // (no `mtime` to compare) leaves the record exactly as it was, and the probe
  // awaits this per record so a tick settles with its records settled.
  //
  // A buffer whose save is in flight is skipped whole: the payload of that save
  // is the freshest truth about the path, and a stat taken *before* it landed
  // describes a disk that no longer exists. Applying it would swap another
  // writer's text into the buffer and the confirmed save would then call that
  // text its own baseline — clearing the dot for bytes that were never written.
  async function applyProbeAnswer(record, probe) {
    if (record.saving) return null

    if (probe.exists === false) {
      // The path is gone. A clean buffer loses nothing (it *is* the disk's
      // copy, minus the file); a dirty one holds work that lives nowhere else,
      // which is the only case that becomes a conflict.
      record.deleted = true
      record.stale = false
      if (record.dirty) {
        const fresh = setConflict(record, 'deleted', {
          diskMtime: null,
          bufferMtime: record.mtime,
        })
        // Same rule as the stale case below: no dialog will appear for a buffer
        // the pane is not showing, so the news needs its own line — one, since a
        // path that stays gone stamps only once.
        if (fresh && state.active !== record.key) {
          notices.notice(
            'warn',
            `${record.path} was deleted on disk while its unsaved buffer is open here`,
            record.path,
          )
        }
      } else {
        record.conflict = null
        record.conflictDetail = null
      }
      return null
    }

    const diskMtime = probe.mtime ?? null
    if (diskMtime === null) return null // nothing to compare against
    if (record.mtime === null) {
      // No token of our own (an old record, or a read that carried no mtime):
      // adopt the disk's as the baseline rather than inventing a change.
      record.mtime = diskMtime
      return null
    }

    if (Math.abs(diskMtime - record.mtime) <= MTIME_EPSILON) {
      // Matches, so the world moved on without this buffer — or a save of ours
      // did. Either way the flags from an earlier tick are now false.
      record.deleted = false
      record.stale = false
      record.size = probe.size ?? record.size
      if (record.conflict === 'stale' || record.conflict === 'deleted') {
        // A conflict left over from a save's 409 that the disk has since
        // resolved (the user reloaded from elsewhere, or merged by hand).
        record.conflict = null
        record.conflictDetail = null
      }
      // The same silence for the warning this buffer may be carrying: the disk
      // agrees again, so the news is over. Only the `warn` line goes — a save
      // refusal about this path is still true (the buffer is still unsaved),
      // and eating it would hide the reason the dot is there.
      notices.dismissForPath(record.path, 'warn')
      return null
    }

    record.deleted = false
    if (!record.dirty) {
      // Nothing to lose: the buffer is the disk's copy character for character,
      // so re-reading it is free and it resets the baseline.
      record.stale = false
      if (!(await reloadFile(record.path))) {
        // The read back failed (it outgrew the cap, it went away between the
        // stat and here). Following the disk is what made silence honest, so a
        // buffer that could not follow it gets the marker instead — without a
        // conflict, since nothing of the user's is at stake.
        record.stale = true
      }
      return null
    }

    // The dangerous cell: the disk moved and this buffer has edits. Marked,
    // and answered only by the user.
    record.stale = true
    const fresh = setConflict(record, 'stale', { diskMtime, bufferMtime: record.mtime })
    // A conflict about the buffer on screen is answered by the pane's dialog, so
    // it raises no notice (a refusal line behind a modal is noise). About a
    // buffer the user is *not* looking at, no dialog will appear, so this is the
    // only copy of the news — one line, because a disk that stays moved stamps
    // once (see `setConflict`) and the stack dedupes anything that slips past.
    if (fresh && state.active !== record.key) {
      notices.notice(
        'warn',
        `${record.path} changed on disk while its unsaved buffer is open here`,
        record.path,
      )
    }
    return null
  }

  // Record a conflict, keeping its age when it is the *same* news. The probe
  // re-reads the same moved file on every tick; re-stamping each time would
  // make a dismissed dialog pop back up five seconds later. A new kind, or a
  // disk that moved *again*, is news and is stamped — and returns true, which is
  // how the callers tell news from the same news re-read.
  function setConflict(record, kind, detail) {
    const same =
      record.conflict === kind &&
      (detail.diskMtime ?? null) === (record.conflictDetail?.diskMtime ?? null) &&
      (detail.bufferMtime ?? null) === (record.conflictDetail?.bufferMtime ?? null)
    record.conflict = kind
    record.conflictDetail = detail
    if (same) return false
    record.conflictAt = Date.now()
    return true
  }

  // Concurrent calls join the running request (the `useGitStatus.load`
  // pattern): a tick landing while one is in flight has nothing new to ask.
  let probeInflight = null
  let probeTimer = null

  // A broken poll is a 5 s loop, so it must report once, not once per tick.
  // `state.probeError` is both the pane's own line (it clears itself when the
  // probe recovers) and the latch: a failure that arrives while it is already
  // set is the same continuing episode, so it stays silent. The notice is the
  // copy for a user whose eyes are on the tree rather than on the pane.
  function reportProbeFailure(reason) {
    const isNewEpisode = state.probeError === null
    state.probeError = `Could not check the open files against disk: ${reason}`
    if (isNewEpisode) notices.notice('warn', state.probeError)
  }

  async function probeOpenFiles() {
    if (!probeIsUseful()) return null
    if (probeInflight) return probeInflight

    const records = state.open.filter((item) => item.kind === 'file')
    const query = records
      .map((item) => `path=${encodeURIComponent(item.path)}`)
      .join('&')

    const run = (async () => {
      try {
        const res = await fetch(`/api/fs/stat?${query}`)
        const body = await res.json().catch(() => null)
        if (!res.ok) {
          // A refused or failed probe is not evidence that files changed: no
          // record moves, and the pane says the probe itself is what failed.
          reportProbeFailure(detailOf(body, `the request was refused (${res.status})`))
          return
        }
        const answers = new Map()
        for (const item of body?.paths ?? []) answers.set(item.path, item)
        for (const record of records) {
          // The buffer may have closed while this request was out — a tab the
          // user closed, or a workspace switch that closed every one of them.
          // An answer about a path no buffer is held open for any more is not
          // news about the pane, it is a reply about the past, and applying it
          // would raise a warning about work that is already gone.
          if (!state.open.some((item) => item.key === record.key)) {
            continue
          }
          const probe = answers.get(record.path)
          if (!probe) continue // the answer did not name it: leave it alone
          await applyProbeAnswer(record, probe)
        }
        // A good answer ends the failure, and with it the latch: the next
        // break is a new episode and gets its one `warn` again.
        state.probeError = null
      } catch (exc) {
        reportProbeFailure('the API could not be reached')
      } finally {
        probeInflight = null
      }
    })()

    probeInflight = run
    await run
    syncConflict()
    return null
  }

  // The one conflict on screen: the active tab's, or none. A stale buffer in
  // the tenth chip carries its marker without shouting over the buffer being
  // edited, and the compare tab of a conflict shows no dialog — a modal would
  // cover the two sides the user opened it to read.
  //
  // An already-published conflict is *not* republished: the probe calls this
  // on every tick, and re-issuing the same object would re-open a dialog the
  // user just dismissed. Only a change of subject, of kind, or of `raisedAt`
  // (a fresh raise, see `raiseConflict`) counts as news.
  function syncConflict() {
    const active = recordForKey(state.active)
    const current = state.conflict
    if (!active?.conflict) {
      if (current) state.conflict = null
      return
    }
    const raisedAt = active.conflictAt
    if (
      current &&
      current.key === active.key &&
      current.kind === active.conflict &&
      current.raisedAt === raisedAt
    ) {
      return
    }
    state.conflict = {
      key: active.key,
      kind: active.conflict,
      diskMtime: active.conflictDetail?.diskMtime ?? null,
      raisedAt,
    }
    // Publishing is what opens the pane's dialog, and a notice behind a modal
    // is noise: the warning this buffer raised while the user was elsewhere is
    // answered here, by the answer set itself.
    notices.dismissForPath(active.path, 'warn')
  }

  // Turning the dialog down without answering: the record keeps its conflict
  // (marker, edits and all), only the modal stops. `raisedAt` is the identity
  // of this news, so the same conflict stays down and a *new* one — the disk
  // moving again, or a focus round-trip — is news again.
  function dismissConflict() {
    if (state.conflict) state.conflictDismissedAt = state.conflict.raisedAt
  }

  // Ask again, on purpose: the marker on a dismissed conflict's chip. It
  // re-stamps the conflict, which is what makes the answer set reachable a
  // second time without a watcher nagging about it.
  function raiseConflict(key) {
    const record = recordForKey(key)
    if (!record?.conflict) return
    record.conflictAt = Date.now()
    state.conflictDismissedAt = 0
    state.conflict = {
      key: record.key,
      kind: record.conflict,
      diskMtime: record.conflictDetail?.diskMtime ?? null,
      raisedAt: record.conflictAt,
    }
  }

  // A resolved conflict has no comparison left to show: the two sides are the
  // same text again, so the tab a Compare opened goes away with the answer.
  // Only a conflict-sourced diff tab is touched — a git diff tab of the same
  // path is the user's own and says something else.
  function dropConflictDiff(path) {
    const record = diffRecordFor(path)
    if (record?.source === 'conflict') closeTab(diffKeyFor(path))
  }

  // Compare: a diff tab of *my buffer* against the disk as it is now — the
  // opposite polarity of a git diff tab (there the right side is also disk),
  // which is why the record carries `source: 'conflict'` and the pane's note
  // line says which is which. The disk side is re-read rather than trusted
  // from the probe, because the probe only answers mtime/size. Choosing what
  // survives is the human's; this is a view, not a merge.
  async function openConflictedDiff(key) {
    const record = recordForKey(key)
    if (!record || record.kind !== 'file') return null
    let fresh = null
    try {
      const res = await fetch(`/api/fs/file?path=${encodeURIComponent(record.path)}`)
      const body = await res.json().catch(() => null)
      if (res.ok && !body?.binary) {
        fresh = { text: body.text ?? '', language: body.language ?? record.language }
      }
    } catch {
      fresh = null
    }
    if (!fresh) {
      // Gone, denied, oversized or unreadable: say which, rather than opening a
      // diff whose left side silently claims to be empty.
      state.message = {
        path: record.path,
        tone: 'error',
        text: 'Could not read the copy on disk to compare against.',
      }
      return null
    }

    const diffPath = record.path
    const diffRecord = {
      kind: 'diff',
      key: diffKeyFor(diffPath),
      path: diffPath,
      name: labelFor('diff', diffPath),
      // Left is disk, right is the buffer — see the note rule above.
      original: fresh.text,
      modified: record.text,
      language: fresh.language,
      status: 'modified',
      baseline: 'disk',
      source: 'conflict',
      renamedFrom: null,
      truncated: false,
      truncatedAt: null,
    }
    // One diff tab per path: an open git-diff tab of the same file is *replaced*
    // rather than left beside it, because the two would say opposite things
    // about the same pair of texts.
    const existing = diffRecordFor(diffPath)
    if (existing) Object.assign(existing, diffRecord)
    else state.open.push(diffRecord)
    state.active = diffRecord.key
    state.message = null
    syncConflict()
    return diffRecord
  }

  // -- rename: re-pointing everything that holds a path -----------------------
  //
  // The route's rules are the server's (see POST /api/fs/rename); what lives
  // here is the app's own view of the moved path. Five things hold a path, and
  // a rename that updates only the row on disk leaves four of them lying: the
  // open records, the active tab key, the tree cache and the git marks.
  //
  // `from` matches a record exactly or as a folder prefix (`from + '/'`), which
  // is the same subtree rule a delete uses — and it is what a *folder* rename
  // needs, because the buffers under it each name the full path themselves.
  // `key` is updated with `path` (for a file the key *is* the path), `state.active`
  // is the same value so the pane keeps showing the buffer it was showing, and a
  // diff tab of the same file keeps its tab kind with a re-derived `diff:` key.
  //
  // Nothing else on a record moves. A rename does not touch content, so `text`,
  // `baseline`, `dirty` and — the load-bearing one — `mtime` stay as they were:
  // the buffer does not become stale, and its save token is still the disk's
  // truth at the new path. That is why this is a re-key, not a close-and-reopen.
  function repointPath(from, to) {
    if (!from || !to || from === to) return 0
    const renamed = state.open.filter((item) => underPath(item.path, from))
    // Read *before* the rewrite below: `active` is an old key, and after the
    // loop that key exists nowhere, so the record it names has to be found while
    // the keys still say what the old keys said.
    const activeRecord =
      state.active === null
        ? null
        : state.open.find((item) => item.key === state.active) ?? null
    for (const record of renamed) {
      const newPath = to + record.path.slice(from.length)
      record.path = newPath
      // A file's key is its path; a diff's is `diff:` + its path and a preview's
      // is `preview:` + it. Both are *derived* (keyFor/labelFor), so a record of
      // any kind re-keys and re-names with its tab kind intact — an inline
      // `kind === 'diff' ? … : …` here would leave a renamed preview keyed at
      // the old path while its `path` moved, and the pane would lose it.
      record.key = keyFor(record.kind, newPath)
      record.name = labelFor(record.kind, newPath)
    }
    // `active` is a *key*, so it travels with the record it names — otherwise
    // the pane loses the buffer it is showing while the strip still lists it.
    if (state.active !== null) state.active = activeRecord ? activeRecord.key : null

    // The notice stack is about paths too, and a path that has moved stops
    // being the address of anything: a *refusal* is about a buffer the user
    // still holds, so it re-points at the new path; a *confirmation* of a past
    // mutation names the path it happened at, and the move just un-named it, so
    // it goes away rather than lying. `repointNotices` is that split.
    notices.repointNotices(from, to)
    syncConflict()
    return renamed.length
  }

  // The Editor tab's open paths are relative to the working location, so a new
  // root makes every one of them name nothing: `src/a.py` here and `src/a.py`
  // there are two different files, and a tab that keeps showing one while the
  // tree browses the other is the bug this prevents. So all tabs — buffers and
  // diff views alike — close with the old root, and with them the state that
  // points at them: the conflict on screen, the close confirm and the probe's
  // complaint about a disk that is no longer the one being read.
  //
  // A dirty buffer is the one real cost of the switch, and it is *reported*
  // rather than negotiated: `requestCloseTab` asks about one tab at a time and
  // offers to save it, and neither answer exists any more — the path it would
  // save to belongs to the folder that was just left. So the loss gets one line
  // on the channel that survives the click that closed the tabs, naming how
  // many buffers carried work that only this page held.
  //
  // `useWorkspaceTree.reRoot` calls this through the registry it is registered
  // in below, and the git marks go with the buffers because they were read from
  // the same paths. A delete confirmation is cleared rather than left open: its
  // path belonged to the root that was just replaced, so answering its Delete
  // button would now destroy a *different* folder's file that no one asked
  // about. The conflict needs nothing — the last close ends it.
  function forgetWorkspaceBuffers({ workspace = '' } = {}) {
    const dirty = state.open.filter(
      (item) => item.kind === 'file' && item.dirty,
    ).length
    // A copy, because `closeTab` splices `state.open` while this walks it.
    for (const record of [...state.open]) closeTab(record.key)
    state.closePrompt = null
    state.deletePrompt = null
    state.message = null
    state.probeError = null
    // The marks are dropped *and* re-read: they were computed over the previous
    // work tree, and the rows that carry them now belong to another folder.
    // Reset first so the tree repaints without them at once, then ask for the
    // new location's answer — the same pair the refresh button and the delete
    // flow perform, and the reason a switch does not leave the marks blank
    // until the next tab change.
    useGitStatus().reset()
    void useGitStatus().load({ force: true })
    if (dirty > 0) {
      // After the closes on purpose: `closeTab` dismisses the notices about a
      // path that stopped being work in progress, and this one has to survive
      // them. No path is attached — it is about the tabs, which are gone.
      notices.notice(
        'warn',
        `Working location switched${workspace ? ` to ${workspace}` : ''} — ` +
          `${dirty} ${dirty === 1 ? 'tab' : 'tabs'} closed with unsaved changes, ` +
          'which are lost',
      )
    }
  }

  function clearMessage() {
    state.message = null
  }

  // The tick. The Editor pane owns the timer (started when its tab is visible
  // and something is open, stopped otherwise), and `probeIsUseful` re-checks
  // every gate on each tick — so a hidden tab, a streaming reply or an empty
  // strip costs no request even if a timer were somehow still running.
  function startFreshnessProbe(intervalMs = FRESHNESS_POLL_MS) {
    if (probeTimer !== null) return
    probeTimer = setInterval(() => {
      void probeOpenFiles()
    }, intervalMs)
    void probeOpenFiles()
  }

  function stopFreshnessProbe() {
    if (probeTimer === null) return
    clearInterval(probeTimer)
    probeTimer = null
  }

  return {
    state,
    dirtyCount,
    openFile,
    openDiff,
    openPreview,
    closeFile,
    closeDiff,
    closeTab,
    requestCloseTab,
    cancelClose,
    discardClose,
    saveAndClose,
    requestDelete,
    cancelDelete,
    confirmDelete,
    trashPath,
    buffersUnder,
    repointPath,
    focusFile,
    focusTab,
    updateText,
    saveFile,
    saveAllDirty,
    reloadFile,
    probeOpenFiles,
    startFreshnessProbe,
    stopFreshnessProbe,
    raiseConflict,
    dismissConflict,
    openConflictedDiff,
    dropConflictDiff,
    recordFor,
    recordForKey,
    diffRecordFor,
    diffKeyFor,
    previewRecordFor,
    previewKeyFor,
    // `refreshPreview` is deliberately not on this surface: a re-read only ever
    // comes from re-picking Preview, i.e. through `openPreview` (D2).
    clearMessage,
    humanSize,
    forgetWorkspaceBuffers,
  }
}
