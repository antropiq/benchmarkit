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
    const askedAt = generation
    try {
      const res = await fetch('/api/fs/status')
      const body = await res.json().catch(() => null)
      if (askedAt !== generation) {
        // The location moved while this was in flight: the answer — good or
        // refused — describes the previous workspace, so it is dropped whole.
        // `load()` is called again for the new one.
        return
      }
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
      // Ours, or abandoned by a `reset()` that left no replacement: either way
      // this request is the last one, so `loading` ends with it. Only a run
      // that a *newer* one replaced leaves its marker standing.
      if (inflight === run || inflight === null) {
        state.loading = false
        inflight = null
      }
    }
  })()
  inflight = run
  await run
}

// Forget the answer without asking for a new one. A working-location switch
// re-roots the tree onto a different folder, and the marks the app holds were
// computed in the *previous* one: kept, they would decorate unrelated rows and
// offer "View diff…" for paths the new repository knows nothing about. An
// empty answer is honest while the new one is on its way — it says "nothing is
// known to be unstaged", which is what the tree draws no marker from.
//
// The token bump is what makes this stick: a query that was already in flight
// against the old location lands *after* the reset, and its answer describes a
// work tree nothing is showing any more, so `load` drops it instead of applying
// it. Clearing `inflight` is the other half — that abandoned query must not be
// the one a new `load()` joins, or the switch would leave the marks unswept.
let generation = 0

function reset() {
  generation += 1
  inflight = null
  state.entries = []
  state.repo = false
  state.atToplevel = true
  state.truncated = false
  state.error = null
  state.loadedAt = 0
}

export function useGitStatus() {
  return {
    state,
    statusFor,
    isUnstaged,
    isDiffable,
    hasUnstagedBelow,
    load,
    reset,
  }
}
