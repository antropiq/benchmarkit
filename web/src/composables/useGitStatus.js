import { computed, reactive } from 'vue'

// ---------------------------------------------------------------------------
// Git status: which files of the working location are unstaged.
//
// One singleton like every other concern: the tree rows, the context menu and
// the diff tabs all need the same answer, and it is one request for the whole
// work tree — never one per row.
//
// The server (`core/editorfs/gitstatus.py`) is the one that decides what
// "unstaged" means and applies the connector path policy to every path it
// reports, so this module only indexes what it is given. A working location
// that is not a git repository answers `repo: false` with no entries: no
// marker, no menu item, no message. A *failed* query is kept apart from the
// empty answer (`error`), because "nothing changed" and "git could not run"
// are two different things a marker must not blur — but it is a decoration, so
// a failure never blocks the tree.
//
// `comparable` is the flag the context menu keys on: an untracked or newly
// added file has no committed side, so it is marked but has nothing to diff.
// ---------------------------------------------------------------------------

const state = reactive({
  entries: [],
  repo: false,
  atToplevel: true,
  truncated: false,
  loading: false,
  error: null,
  loadedAt: 0, // bumped on every successful fetch, so a caller can watch it
})

// path -> { path, kind, comparable, oldPath }, rebuilt whenever `entries` is
// replaced. The tree asks per row; a linear scan of a 5k-entry list per row per
// paint would be the cost this index exists to avoid.
const byPath = computed(() => {
  const map = new Map()
  for (const entry of state.entries) map.set(entry.path, entry)
  return map
})

// The folders that *contain* at least one unstaged file, so a collapsed branch
// still hints that something below it moved. A folder is never itself a status
// entry (git reports files), so this set and `byPath` cannot disagree about a
// path — and a folder's mark carries no kind, only "something inside".
const touchedDirs = computed(() => {
  const out = new Set()
  for (const entry of state.entries) {
    // Every ancestor, the workspace root ("") included — the root row is a
    // folder like any other and gets the same hint.
    let index = entry.path.indexOf('/')
    while (index !== -1) {
      out.add(entry.path.slice(0, index))
      index = entry.path.indexOf('/', index + 1)
    }
    out.add('')
  }
  return out
})

function statusFor(path) {
  return byPath.value.get(path) ?? null
}

function hasUnstagedBelow(path) {
  return touchedDirs.value.has(path)
}

function isUnstaged(path) {
  return byPath.value.has(path)
}

// True only for a file with a committed side to compare against.
function isDiffable(path) {
  return byPath.value.get(path)?.comparable === true
}

// Fetch the status of the whole work tree. Concurrent calls join the running
// request rather than firing a second git query.
let inflight = null

async function load({ force = false } = {}) {
  if (inflight) return inflight
  if (state.loadedAt && !force) return
  state.loading = true
  const run = (async () => {
    try {
      const res = await fetch('/api/fs/status')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        // Kept apart from "no entries": a 504 from git must not read as a
        // clean work tree. The tree stays usable either way.
        state.error =
          (typeof body?.detail === 'string' ? body.detail : '') ||
          `Could not read the git status (${res.status})`
        return
      }
      state.entries = body.entries || []
      state.repo = !!body.repo
      state.atToplevel = body.atToplevel !== false
      state.truncated = !!body.truncated
      state.error = null
      state.loadedAt += 1
    } catch (exc) {
      state.error = `Could not reach the API: ${exc}`
    } finally {
      state.loading = false
      inflight = null
    }
  })()
  inflight = run
  await run
}

export function useGitStatus() {
  return {
    state,
    statusFor,
    isUnstaged,
    isDiffable,
    hasUnstagedBelow,
    load,
  }
}
