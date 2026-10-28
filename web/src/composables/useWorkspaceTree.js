import { reactive } from 'vue'

// ---------------------------------------------------------------------------
// Workspace tree: the state behind WorkspaceNavigator.vue.
//
// One singleton, like every other composable: the opened branches, the focus
// and the selection live *here*, not in the component, so a tab switch
// (`v-show`) keeps the tree exactly as the user left it — and the code pane can
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
// triggers (the tree re-roots with it). A mutation touches one folder, so
// `refreshBranch(path)` is the sibling that re-lists exactly that listing.
//
// The tree is also where a name is typed: `editing` is the inline input's state
// (see TreeNameInput.vue), in two modes — `create` (a new name inside a folder)
// and `rename` (a new name for one row, which may name a *path* and so move the
// row elsewhere). `validateCreateName` / `validateRenameTarget` are the *fast*
// half of the answer — the shape of a name and whether the listing already
// holds it. Everything else (is this name protected?) is the server's rule and
// arrives as a refusal: there is deliberately no copy of the deny-list here.
//
// And one thing lives here besides the listings: **the subject of every tree
// command is the focused row** (`commandTarget()`). The keyboard chords, the
// context menu and the row actions all ask that one question, so a right-click
// and a key press cannot disagree about which row "this folder" means. It is
// *not* the same as `selected`, which means "the pane should open this file" —
// conflating them would make every folder click try to open a buffer.
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
  selected: null, // { path, name, dir, size, mtime } — the code pane's input
  workspace: '', // absolute path of the browsed workspace, from the server
  error: null,
  // The inline name input: `{ mode, parent, path, isDir, value, error }` while
  // the user is naming something, or null. `mode` is `'create'` (the name goes
  // into `parent`) or `'rename'` (the row at `path` is being renamed *within*
  // `parent`, so the input sits in the listing that shows it). It is here rather
  // than in the component because it is tree state — the row it belongs to, the
  // listing its name has to agree with, and the focus it owns outlive a re-render
  // (the navigator is inside a `v-show` view, so a half-typed name must survive a
  // tab switch). One object serves both modes: the widget, the focus rule and
  // the refusal-in-the-row rule are the same for a create and a rename, and a
  // second field would be a second answer to "what is being typed?".
  editing: null,
})

// The two halves of one path, joined the way the server spells them ("" is the
// root, so a root child is the bare name).
function childPath(parent, name) {
  return parent ? `${parent}/${name}` : name
}

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

// The paths whose *listing* is on screen — the ones refresh() has to re-fetch.
// Two differences from `visiblePaths()` matter: a file row is a row too, and
// the tree endpoint refuses a file with 400; a collapsed folder is on screen
// but its children are not, so its listing is not displayed and its next
// expand re-lists it anyway. What is left is the root plus every folder the
// user has opened — which is exactly what the refresh button promises.
function openDirPaths() {
  const out = [ROOT]
  const walk = (path) => {
    if (!state.expanded[path]) return
    for (const entry of state.nodes[path]?.entries ?? []) {
      if (entry.dir && state.expanded[entry.path]) {
        out.push(entry.path)
        walk(entry.path)
      }
    }
  }
  walk(ROOT)
  return out
}

// The fetch in flight per path, kept outside the reactive state so nobody
// observes a Promise. A second `load()` for the same path *joins* it rather
// than returning early: an awaiting caller then learns the listing instead of
// "someone else is busy", which is what reveal() below depends on.
const inflight = new Map()

// The create chord is answered by the create flow, not by this module: the tree
// owns the keyboard model (which key, in which folder) and `useEditorCreate`
// owns the request, its refusals and what success does to the buffers. The
// registry keeps that one-way — this module never imports the create half, so
// the tree stays a read model and the lazy Editor chunk stays lazy.
let createChord = null

export function registerCreateChord(handler) {
  createChord = typeof handler === 'function' ? handler : null
}

// Same one-way registry for the delete chord (`Delete`/`Shift+Delete`): this
// module says which key and on which row, and the buffer half — which owns the
// confirmation, the request and what success does to the open tabs — registers
// itself as the answer. The tree never imports the write half, so a cycle
// between the two files stays impossible.
let deleteChord = null

export function registerDeleteChord(handler) {
  deleteChord = typeof handler === 'function' ? handler : null
}

// …and the same for the rename chord (`F2`): the tree names the key and the row,
// `useEditorRename` owns the input, the request and re-pointing the buffers.
let renameChord = null

export function registerRenameChord(handler) {
  renameChord = typeof handler === 'function' ? handler : null
}

// …and for the refresh chord (`r`). This one needs no write half — re-listing is
// the tree's own job — but it does need the git marks, which the navigator owns,
// so the registry keeps that dependency out of this module: the answer re-lists
// the focused folder *and* re-reads which files are unstaged, exactly as the
// header's refresh button does for the whole tree.
let refreshChord = null

export function registerRefreshChord(handler) {
  refreshChord = typeof handler === 'function' ? handler : null
}

// True when `candidate` is `path` itself or something below it. The trailing
// slash is what keeps `src/x.pyw` out of `src/x.py`'s subtree. A missing
// candidate (no row focused, nothing selected) is not under anything.
function sameOrUnder(candidate, path) {
  if (!path || typeof candidate !== 'string') return false
  return candidate === path || candidate.startsWith(`${path}/`)
}

export function useWorkspaceTree() {
  // A path is fetched at most once concurrently (double clicks never
  // double-list).
  async function load(path = ROOT, { force = false } = {}) {
    const entry = node(path)
    if (entry.loaded && !force) return
    const running = inflight.get(path)
    // A joining caller waits for the running fetch instead of returning
    // "someone else is busy" — that is what reveal() below depends on. A
    // forced call still re-lists afterwards: after refresh() dropped the
    // cache, the fetch it joins writes into an orphaned entry.
    if (running) await running
    if (entry.loaded && !force) return
    entry.loading = true
    state.error = null
    const run = (async () => {
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
        inflight.delete(path)
      }
    })()
    inflight.set(path, run)
    await run
  }

  // One listing, re-fetched: what a mutation just made true about *one* folder.
  // `refresh()` answers a different question (the whole tree moved — the
  // working location changed, the refresh button was pressed), and its cost is
  // every opened branch, so it is not what a create/rename/delete should pay.
  //
  // An unloaded folder is skipped rather than listed: nothing on screen shows
  // it, and its next expand fetches it anyway. The focus is revalidated because
  // a re-list can drop the row that had it.
  async function refreshBranch(path) {
    if (!state.nodes[path]?.loaded) return
    await load(path, { force: true })
    revalidateFocus()
  }

  // One level up from a mutation: the listing that holds the *row of the folder*
  // the change happened in. A directory row's `children` flag comes from its
  // parent's listing, and `tree.py` promises it means "expanding will show
  // something" — so creating the first file inside a folder leaves that folder's
  // row, in the listing one level above, still claiming it has nothing to
  // expand. That is a lie about a branch the user may be staring at, and the
  // arrow is drawn from the flag.
  //
  // So: only when that flag currently says `false` (the one case that lies), and
  // only for a folder that is not in the root — where the root's own listing is
  // the one the mutation already re-listed. `refreshBranch` then skips the case
  // where nothing on screen shows that listing, so the usual cost is nothing.
  async function refreshListingOfParent(path) {
    const parent = parentOf(path)
    if (parent === ROOT) return
    const row = entryFor(parent)
    if (!row || row.children) return
    await refreshBranch(parentOf(parent))
  }

  // Drop the cache and re-list what is on screen: the refresh button, and the
  // hook every surface gets when the working location changes. The opened
  // branches stay open — a refresh is not a collapse.
  async function refresh() {
    // Snapshot the on-screen folders *before* the cache is dropped — after
    // that, the walk that answers "what is on screen" has nothing left to
    // walk. Folders only: a file row has no listing of its own and the tree
    // endpoint answers 400 for one.
    const openDirs = openDirPaths()
    state.nodes = {}
    await Promise.all(openDirs.map((path) => load(path, { force: true })))
    // A re-list can drop the focused file (deleted, or now hidden by policy),
    // which would leave the roving tabindex pointing at a row that is not
    // rendered any more.
    revalidateFocus()
  }

  function expand(path) {
    state.expanded[path] = true
    if (!node(path).loaded) load(path)
  }

  function collapse(path) {
    delete state.expanded[path]
    revalidateFocus()
  }

  // Is the row for this path actually on screen: its listing loaded, its entry
  // there, and every folder above it expanded?
  function isDisplayed(path) {
    if (path === ROOT) return true
    let current = path
    while (current !== ROOT) {
      if (!state.expanded[parentOf(current)] || !entryFor(current)) return false
      current = parentOf(current)
    }
    return true
  }

  // The roving tabindex has to be carried by a rendered row, or the tree cannot
  // be tabbed into and its arrow keys go nowhere (`moveFocus` cannot locate a
  // path that is not on screen). A focused path can stop being displayed under
  // the user: a refresh re-lists a folder and the file is gone, a reveal of a
  // tab's file fails, a collapse hides a branch. Then focus walks up to the
  // nearest row still on screen — the same rule `collapseAll` states.
  function revalidateFocus() {
    if (isDisplayed(state.focused)) return
    let current = state.focused
    while (current !== ROOT && !isDisplayed(current)) current = parentOf(current)
    state.focused = current
  }

  // Close every branch the user opened, in one click. The root itself stays
  // open — a collapsed root would leave a tree with no row left to re-open.
  // Listings keep their cache, so re-expanding a folder costs no request.
  // Focus lands on the root: it is the only row guaranteed to still be on
  // screen, and a focused path that vanished would leave the roving tabindex
  // with nothing in the tab order.
  function collapseAll() {
    state.expanded = { [ROOT]: true }
    state.focused = ROOT
  }

  function toggle(path) {
    if (state.expanded[path]) collapse(path)
    else expand(path)
  }

  // -- the inline name input -------------------------------------------------
  //
  // Ask for a name inside `parent` ("" is the workspace root). The branch is
  // opened and its listing fetched first, because the input renders *inside*
  // that listing and its "already in this folder" check reads that listing.
  //
  // Opening it is deliberately *not* awaited here: a request would put a pause
  // between the gesture and the caret, and `load()` joins the running fetch
  // anyway, so the listing arrives and the input's own name check re-reads it
  // when the name is submitted. A parent that is not a folder is refused rather
  // than asked about: there is no listing to type into.
  function startCreate(parent, isDir) {
    if (parent !== ROOT && !entryFor(parent)?.dir) return false
    expand(parent)
    state.focused = parent
    state.editing = { mode: 'create', parent, isDir: !!isDir, value: '', error: null }
    return true
  }

  // Ask for a new name for the row at `path`, in the listing that shows it.
  // The same `editing` state as a create, with `mode: 'rename'` and the row's
  // path remembered: the input is the *one* name widget, shared by both flows,
  // and the two modes differ only in what a submitted value means — a name for a new
  // thing, or the full new path of an existing one.
  //
  // The value is prefilled with the current name and the row's folder is opened
  // first, because the input renders inside that listing. The root row is not a
  // candidate: the working location itself is neither renamable nor listed in a
  // folder that could hold an input for it.
  function startRename(path) {
    if (!path) return false
    const entry = entryFor(path)
    if (!entry) return false
    const parent = parentOf(path)
    expand(parent)
    state.focused = path
    state.editing = {
      mode: 'rename',
      path,
      parent,
      isDir: !!entry.dir,
      value: entry.name,
      error: null,
    }
    return true
  }

  function cancelCreate() {
    state.editing = null
  }

  function updateCreateValue(value) {
    if (!state.editing) return
    state.editing.value = value
    // A new attempt is a new answer: the old refusal stops being about this text.
    state.editing.error = null
  }

  // The subject of every tree command: the focused row, as the three parts a
  // command needs — its path, its name, and the folder it lives in ("" for a
  // root child, and the root row itself answers with ROOT for both). Built from
  // `entryFor`, so it is always the listing's own answer rather than a path
  // string a caller reverse-engineered.
  //
  // The focused row can stop existing between two ticks (a re-list dropped it),
  // in which case there is no subject and every command that needs one declines.
  // Commands that take a *folder* (New file…) use the same answer: `dir` says
  // whether `path` is that folder or a neighbour of it.
  function commandTarget() {
    const path = state.focused
    const entry = entryFor(path)
    if (!entry) return null
    return { path, name: entry.name, dir: !!entry.dir, parent: parentOf(path) }
  }

  // Selection is a row click: a folder only toggles (an open folder is not a
  // thing a code pane could load), a file is recorded.
  // The entry travels with the path, so the consumer has its size and mtime
  // without a second request.
  //
  // EditorArea watches it and opens the file. Selection and the editor's active
  // buffer stay separate on purpose — a refused click (binary, over the cap)
  // selects a row without putting anything in a tab.
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

  // The reverse direction of the tree <-> tab-strip link: focusing a tab in
  // EditorArea selects the row of its file here.
  //
  // The tree is lazy, so that row usually does not exist yet: each listing on
  // the way down has to be fetched and each branch on the way opened. When a
  // listing *is* cached but does not carry the path, it may simply predate the
  // file (the WorkBench writes files), so that folder is re-listed once before
  // giving up.
  //
  // Failure — a denied path, a folder that refuses its listing, a file that
  // disappeared — leaves the selection exactly as it was: a tab the tree cannot
  // show must not highlight some other row. A reveal that finishes after a
  // newer one started is dropped for the same reason.
  //
  // `source: 'tab'` marks where the selection came from, so EditorArea's
  // watcher can skip it: replaying it as a click would be a redundant
  // open/focus round-trip back into the buffer that caused it.
  let revealSeq = 0

  async function reveal(path) {
    if (!path) return false
    const seq = ++revealSeq
    let current = ROOT
    for (const name of path.split('/')) {
      if (!state.nodes[current]?.loaded) await load(current)
      const child = childPath(current, name)
      let entry = entryFor(child)
      if (!entry) {
        await load(current, { force: true })
        entry = entryFor(child)
        if (!entry) return false
      }
      if (child === path) break // the last segment is the file's own row
      // A segment that is not a folder means the listings and the open buffers
      // disagree; there is no branch to open below it.
      if (!entry.dir) return false
      expand(child)
      current = child
    }
    if (seq !== revealSeq) return false
    const entry = entryFor(path)
    if (!entry) return false
    // The row exists now, so it also takes the tree's focus: the roving tabindex
    // lands where the eye is sent, and the row is inside the scroll container —
    // the navigator scrolls it into view.
    state.focused = path
    state.selected = {
      path,
      name: entry.name,
      dir: false,
      size: entry.size ?? null,
      mtime: entry.mtime ?? null,
      source: 'tab',
    }
    return true
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

  // Forget everything the tree knew about a path and its subtree, without
  // re-listing anything: what a rename does to the cache *before* the two
  // affected folders are re-fetched.
  //
  // The cache is keyed by folder path, so a renamed folder leaves three kinds of
  // stale key behind: its own listing, its descendants' listings, and the
  // `expanded` flags carried by all of them. Dropping them is what makes the
  // re-list afterwards the only source of the new names — an entry object that
  // still carried the old path would keep rendering a row that no longer exists.
  // Re-expansion is the caller's job (the rename flow re-opens the folder when
  // the moved thing was a folder), which is why the flags go with the entries
  // rather than surviving them.
  function dropBranch(path) {
    if (!path) return
    for (const key of Object.keys(state.nodes)) {
      if (sameOrUnder(key, path)) delete state.nodes[key]
    }
    for (const key of Object.keys(state.expanded)) {
      if (sameOrUnder(key, path)) delete state.expanded[key]
    }
    if (sameOrUnder(state.focused, path)) state.focused = parentOf(path)
    if (sameOrUnder(state.selected?.path, path)) state.selected = null
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

  // The one keyboard model of the tree:
  //
  //   ↑ ↓ ← → Home End   move the roving focus, open/close a branch
  //   Enter / Space      select (open a file, toggle a folder) — unchanged
  //   n / N              New file / New folder in the focused folder
  //   F2                 rename the focused row
  //   Delete             delete the focused row (asks first)
  //   r                  re-list the focused folder (the file's parent for a file)
  //
  // It lives in the composable, not the component, so `visiblePaths()` stays the
  // single source of truth for what is on screen, and every chord's subject is
  // `commandTarget()` — the focused row — which is also what the context menu
  // sets before it acts, so mouse and keyboard agree about "this row".
  //
  // A single letter is only safe here because the container is a `role="tree"`
  // widget that takes focus as a whole (one row of it carries the roving
  // tabindex): these letters mean something only while a *row* has the focus,
  // and nowhere else in the app — Monaco, the terminal, a chat box and the name
  // input all own their own keystrokes. The guards that make that true:
  //
  //   * `editing` is checked first — while the name input is on screen it owns
  //     the keyboard, and a chord that typed into itself is the bug this stops;
  //   * a modifier chord (Ctrl/Alt/Meta) is never one of ours, and a key the
  //     browser or an accessibility tool already means something else for is
  //     left alone by falling through the chain;
  //   * a dialog is never interrupted by a tree chord either — and that needs no
  //     code: these are `<dialog>`s opened with `showModal()`, which keeps the
  //     caret inside them, and this handler only ever runs for a key pressed
  //     while a row of *this* tree has the focus.
  //
  // The destructive/creating keys answer *through* the registries above (the
  // tree says which key on which row; the write halves own the request), and a
  // row that is the workspace root is never a candidate for rename or delete —
  // the working location itself has no name to change, no folder above it to
  // type one into, and nothing to delete it.
  //
  // The return value is the answer to "did the tree *own* this keystroke?", and
  // the navigator needs it: the container sees the keydowns of everything inside
  // it, including the inline name input, and it must move the caret to the
  // focused row only for the keys it acted on. Otherwise a single character
  // typed into that input would throw the caret out of it.
  function handleKey(event) {
    if (state.editing) return false
    if (event.ctrlKey || event.metaKey || event.altKey) return false
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
    } else if (event.key === 'n' || event.key === 'N') {
      // New file / new folder, in the focused folder — or, when a file row has
      // the focus, next to that file: the user's intent is "here", and "here" is
      // the row they were looking at. `createChord` is registered by the create
      // flow (useEditorCreate), which owns the request and its answers; this
      // model only says which key means which kind, and in which folder.
      createChord?.(event.shiftKey ? 'new-folder' : 'new-file', directory ? path : parentOf(path))
    } else if (event.key === 'F2') {
      // Rename the focused row — the tree's own chord, next to `Delete`: the
      // same one-way registry, the same reason (the keyboard model belongs to the
      // tree, the request belongs to the write half). The root is not a
      // candidate, for the delete's reason plus one: there is no listing above it
      // to type a new name into.
      if (path === ROOT) return false
      renameChord?.(path)
    } else if (event.key === 'Delete') {
      // Delete the focused row. The root is not a candidate: the working
      // location itself is never deletable, and offering it would only collect
      // a refusal. `deleteChord` (registered by the buffer half) asks the
      // user and then deletes — a destructive key is never the whole decision.
      // The *row* travels, not just the path: the confirmation names the name
      // and the kind the user was looking at.
      if (path === ROOT) return false
      deleteChord?.(entryFor(path) ?? { path, dir: directory, name: path })
    } else if (event.key === 'r' || event.key === 'R') {
      // Re-list the focused folder — or, for a file row, the folder that lists
      // it, which is the only listing that row can be wrong about. `refreshChord`
      // is the navigator's answer (folder listing + git marks), the same pair the
      // header's refresh button performs one level higher.
      refreshChord?.(directory ? path : parentOf(path))
    } else {
      return false
    }
    event.preventDefault()
    return true
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
    refreshBranch,
    refreshListingOfParent,
    expand,
    collapse,
    collapseAll,
    toggle,
    startCreate,
    startRename,
    cancelCreate,
    updateCreateValue,
    commandTarget,
    dropBranch,
    select,
    reveal,
    entryFor,
    handleKey,
    rootLabel,
    parentOf,
    depthOf,
    visiblePaths,
  }
}
