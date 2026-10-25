import { reactive } from 'vue'

// ---------------------------------------------------------------------------
// Workspace tree: the state behind WorkspaceNavigator.vue (ed-01).
//
// One singleton, like every other composable: the opened branches, the focus
// and the selection live *here*, not in the component, so a tab switch
// (`v-show`) keeps the tree exactly as the user left it — and ed-03 can
// consume `selected` without touching the navigator.
//
// The tree is lazy by construction: one `GET /api/fs/tree` per directory, on
// first expand, cached by path. The workspace may be a huge checkout; the
// first paint always costs exactly one listing.
//
// Paths are workspace-relative ("" is the root) and the server is the one
// that applies the policy — denied files are simply absent from the payload,
// so the client never learns they exist. `refresh()` drops every cache entry
// and re-lists what is on screen, which is what a working-location change
// triggers (the tree re-roots with it).
// ---------------------------------------------------------------------------

const ROOT = ''

const state = reactive({
  // Cache of one listing per path, plus the per-path fetch status.
  nodes: {},
  // Opened branches. The root starts open — the first paint shows the
  // workspace's own entries — and every path opened by the user survives a tab
  // switch, because this object is not owned by the component.
  expanded: { '': true },
  focused: ROOT, // roving-focus path (the tree row in the tab order)
  selected: null, // { path, name, dir, size, mtime } — ed-03 consumes this
  workspace: '', // absolute path of the browsed workspace, from the server
  error: null,
})

function blankNode() {
  return { loaded: false, loading: false, entries: [], truncated: false, error: null }
}

function node(path) {
  if (!state.nodes[path]) state.nodes[path] = blankNode()
  return state.nodes[path]
}

function parentOf(path) {
  const index = path.lastIndexOf('/')
  return index === -1 ? ROOT : path.slice(0, index)
}

function depthOf(path) {
  return path ? path.split('/').length : 0
}

// The rows currently on screen, in visual order: every child of an expanded
// directory, followed by that directory's own children. Keyboard navigation
// and `aria-posinset` are both derived from this one list, so they can never
// disagree with what is rendered.
function visiblePaths() {
  const out = [ROOT]
  const walk = (path) => {
    if (!state.expanded[path]) return
    for (const entry of state.nodes[path]?.entries ?? []) {
      out.push(entry.path)
      walk(entry.path)
    }
  }
  walk(ROOT)
  return out
}

export function useWorkspaceTree() {
  // A path is fetched at most once concurrently; a second call while one is in
  // flight is a no-op (double clicks never double-list).
  async function load(path = ROOT, { force = false } = {}) {
    const entry = node(path)
    if (entry.loading) return
    if (entry.loaded && !force) return
    entry.loading = true
    state.error = null
    try {
      const res = await fetch(`/api/fs/tree?path=${encodeURIComponent(path)}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        const detail = typeof body?.detail === 'string' ? body.detail : ''
        entry.error = detail || `Could not list the folder (${res.status})`
        state.error = entry.error
        return
      }
      state.workspace = body.workspace || state.workspace
      entry.entries = body.entries || []
      entry.truncated = !!body.truncated
      entry.loaded = true
      entry.error = null
    } catch (exc) {
      entry.error = `Could not reach the API: ${exc}`
      state.error = entry.error
    } finally {
      entry.loading = false
    }
  }

  // Drop the cache and re-list what is on screen: the refresh button, and the
  // hook every surface gets when the working location changes. The opened
  // branches stay open — a refresh is not a collapse.
  async function refresh() {
    // Snapshot the on-screen paths *before* the cache is dropped — after that,
    // the walk that answers "what is on screen" has nothing left to walk.
    const onScreen = visiblePaths()
    state.nodes = {}
    await Promise.all(onScreen.map((path) => load(path, { force: true })))
  }

  function expand(path) {
    state.expanded[path] = true
    if (!node(path).loaded) load(path)
  }

  function collapse(path) {
    delete state.expanded[path]
  }

  function toggle(path) {
    if (state.expanded[path]) collapse(path)
    else expand(path)
  }

  // Selection is a row click: a folder only toggles (an open folder is not a
  // thing ed-03 could load), a file is recorded — nothing else consumes it.
  // The entry travels with the path, so the consumer has its size and mtime
  // without a second request.
  function select(entry) {
    if (!entry) return
    state.focused = entry.path
    if (entry.dir) {
      toggle(entry.path)
      return
    }
    state.selected = {
      path: entry.path,
      name: entry.name,
      dir: false,
      size: entry.size ?? null,
      mtime: entry.mtime ?? null,
    }
  }

  function moveFocus(delta) {
    const paths = visiblePaths()
    const index = paths.indexOf(state.focused)
    if (index === -1) return
    const next = Math.min(Math.max(index + delta, 0), paths.length - 1)
    state.focused = paths[next]
  }

  function focusEdge(last) {
    const paths = visiblePaths()
    state.focused = paths[last ? paths.length - 1 : 0]
  }

  // The entry a path refers to, looked up in its parent's listing (the tree
  // has no global path index — the listings *are* the index). The root answers
  // with a synthetic directory row, since it always exists and is always open.
  function entryFor(path) {
    if (path === ROOT) {
      return { name: rootLabel(), path: ROOT, dir: true, children: true }
    }
    return state.nodes[parentOf(path)]?.entries.find((item) => item.path === path) ?? null
  }

  // The one keyboard model of the tree (↑ ↓ ← → Home End Enter/Space), owned
  // by the container so `visiblePaths()` stays the single source of truth for
  // what is on screen.
  function handleKey(event) {
    const path = state.focused
    const directory = path === ROOT || !!entryFor(path)?.dir
    if (event.key === 'ArrowDown') moveFocus(1)
    else if (event.key === 'ArrowUp') moveFocus(-1)
    else if (event.key === 'Home') focusEdge(false)
    else if (event.key === 'End') focusEdge(true)
    else if (event.key === 'ArrowRight') {
      if (directory && !state.expanded[path]) expand(path)
      else moveFocus(1)
    } else if (event.key === 'ArrowLeft') {
      if (directory && state.expanded[path]) collapse(path)
      else if (path !== ROOT) state.focused = parentOf(path)
    } else if (event.key === 'Enter' || event.key === ' ') {
      select(entryFor(path) ?? { path, dir: directory })
    } else {
      return
    }
    event.preventDefault()
  }

  // The visible root name: the workspace basename, so the header and the root
  // row always say *which* checkout is being browsed. The absolute path rides
  // in the row's `title`.
  function rootLabel() {
    const workspace = state.workspace
    if (!workspace) return 'workspace'
    const trimmed = workspace.replace(/[/\\]+$/, '')
    const index = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'))
    return index === -1 ? trimmed : trimmed.slice(index + 1)
  }

  return {
    state,
    ROOT,
    load,
    refresh,
    expand,
    collapse,
    toggle,
    select,
    entryFor,
    handleKey,
    rootLabel,
    parentOf,
    depthOf,
    visiblePaths,
  }
}
