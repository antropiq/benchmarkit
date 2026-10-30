import { reactive } from 'vue'

import { scopeFor } from '../lib/search-scope'
import { registerReRootHandler, useWorkspaceTree } from './useWorkspaceTree'

// ---------------------------------------------------------------------------
// Workspace search: the state behind the Editor tab's search dialog.
//
// One singleton, like every other screen's: the query, the scope, the last
// answer, whether the dialog is up and the one queued reveal live *here*, not
// in the component, so closing the `<dialog>`, re-opening it, or switching to
// the WorkBench tab and back cannot lose them (§2.7). The dialog is the only
// thing that renders this state; the tree menu and the header icon only raise
// it, and the code pane consumes `pendingReveal`.
//
// That retention is the whole design: re-opening shows the previous query in
// the field, the previous scope in the label and the previous table, and
// nothing is re-fetched on re-open — an answer is only replaced by a *new*
// search. The cost is a stale hit, and it is accepted on purpose: clicking one
// produces the pane's existing refused-open answer (the 404 reason echoed with
// the path), so no staleness machinery is needed here.
//
// One blocking request per search, no stream and no cancel (D1): the scan is
// bounded server-side by the caps in `editorfs/constants`, so it finishes in
// well under a second and the only thing the client owes the user is a busy
// marker while it is in flight. The previous table *stays on screen* during a
// re-search — clearing it would flash an empty table at the person comparing
// two queries.
//
// The payload is the Contract, verbatim: `result` is what the server sent,
// unreshaped, and the dialog reads the same names (`matches`, `matchCount`,
// `footerNote`, `skipped`). The server's `detail` for a refusal is likewise
// kept word for word — the point of D5 is that `re`'s own message, caret
// position and all, reaches the user rather than a generic sentence.
//
// A *new working location* empties all of it. The results are paths relative to
// a root that no longer exists, so showing them under the new root's label is
// the lie §2.7 names, and the dialog closes with the tabs. That is the keyed
// registry the buffers are already forgotten through (`registerReRootHandler`),
// which is why this module reads the tree and never the Editor's write half:
// a search result reaches a buffer through the tree's *selection*, so the read
// model stays a read model.
// ---------------------------------------------------------------------------

const state = reactive({
  query: '',
  caseSensitive: false,
  regex: false,
  // { kind, label, path, isFile } from `scopeFor()`, or null until a row is the
  // subject. The kind/label answer the dialog's read-only "Search in:" line;
  // only `path` goes on the wire.
  scope: null,
  open: false,
  status: 'idle', // 'idle' | 'running' | 'done' | 'error'
  result: null,
  // { status, detail } for the inline line under the field. `status: 0` means
  // the request never reached the server — a refusal and an unreachable API
  // are two different things a user must not have to tell apart by guessing.
  error: null,
  // search-08's cue: { path, line, column }, applied by the pane once its
  // model holds the file (a reveal applied before then is erased by the
  // `model.setValue` that swaps the buffer).
  pendingReveal: null,
})

// The search in flight, stamped on send and checked on arrival. Two `run()`
// calls in a row is the ordinary case (the user presses Enter twice), and a
// response that answers an older query would put a table on screen for text
// that is no longer in the field. Same token trick `useWorkspaceTree` uses for
// its listings.
let generation = 0

export function useWorkspaceSearch() {
  const tree = useWorkspaceTree()

  // Raise the dialog on the row the gesture was made on — the `commandTarget()`
  // shape, so a folder's subtree, one file's content and the whole workspace are
  // the three cases the same rule already answers.
  //
  // It clears nothing. Not the query, not the result, not the status: the last
  // answer is the reason this state is here (§2.7 / D14).
  //
  // With no row focused there is no subject, so the command declines rather than
  // inventing a scope — the same answer every other tree command gives.
  function openFor(target) {
    const scope = scopeFor(target, tree.rootLabel)
    if (!scope) return false
    state.scope = scope
    state.open = true
    return true
  }

  // Escape and the backdrop both end up here. Only the dialog goes away: the
  // answer is kept, because closing a popup is not a search.
  function closeDialog() {
    state.open = false
  }

  // One search, one request. The four params are what the route declares, and
  // `URLSearchParams` is not a style choice: a needle containing `&`, `=` or `#`
  // has to arrive encoded or it reads as more than one param.
  async function run() {
    if (!state.scope) return
    const query = state.query.trim()
    if (!query) {
      // An empty needle is not a search, and the last answer is the one still on
      // screen — so this is 'idle', not an error and not a 'done' with nothing
      // in it. The server's 400 says the same thing; one rule, one owner, so no
      // sentence is invented here.
      state.status = 'idle'
      state.error = null
      return
    }

    const params = new URLSearchParams({
      path: state.scope.path,
      q: state.query,
      caseSensitive: String(state.caseSensitive),
      regex: String(state.regex),
    })
    const askedAt = ++generation
    state.status = 'running'
    state.error = null
    let res
    let body = null
    try {
      res = await fetch(`/api/fs/search?${params}`)
      body = await res.json().catch(() => null)
    } catch (exc) {
      if (askedAt !== generation) return
      state.status = 'error'
      // Named as never having arrived, so it cannot be mistaken for a refusal
      // of the query itself.
      state.error = { status: 0, detail: `The API could not be reached: ${exc}` }
      return
    }
    if (askedAt !== generation) return

    if (!res.ok) {
      state.status = 'error'
      state.error = { status: res.status, detail: body?.detail ?? '' }
      // `result` deliberately untouched: an invalid pattern shows the inline
      // error *and* keeps the table the user was reading. A failed search does
      // not destroy the last good one (§6 row 4).
      return
    }
    state.result = body
    state.error = null
    state.status = 'done'
  }

  // What a row click does, in the order §2.6 gives it: queue the reveal, then
  // publish the path as the tree's selection, which is what opens or focuses the
  // buffer. The dialog closes last — taking the hit is what closes it, not a
  // precondition of the hit (D13).
  //
  // The selection needs the tree's own trick for a path it has no listing of.
  // The tree is lazy, so a workspace-wide search almost always hits a collapsed
  // branch, `entryFor` answers null for it, and `select(null)` does nothing —
  // the naive form would silently open nothing for the common case. So a file
  // entry is synthesised from the path, exactly as the tree's Enter and Delete
  // chords do: `dir: false` routes it to "select a file" rather than "toggle a
  // folder", and `name` is real because `select()` copies it into `state.selected`.
  //
  // `tree.reveal()` is *not* the opener: it publishes with `source: 'tab'`,
  // which EditorArea's watcher skips as the echo of a buffer that caused it, so
  // it highlights a row without ever opening one. Highlighting happens by itself
  // once the buffer is active.
  function openHit(hit) {
    if (!hit?.path) return
    state.pendingReveal = { path: hit.path, line: hit.line, column: hit.column }
    const entry = tree.entryFor(hit.path) ?? {
      path: hit.path,
      name: hit.path.slice(hit.path.lastIndexOf('/') + 1),
      dir: false,
    }
    tree.select(entry)
    closeDialog()
  }

  // A working-location switch: every path in the payload is relative to a root
  // that no longer exists, so the answer, the query that produced it and the
  // scope that bounded it all go. Including `open` — this view is mounted with
  // the rest of the Editor, and a dialog left up over a root that does not
  // contain its own rows is the same lie in a louder form. The buffers' own
  // subscriber closes their tabs the same way.
  //
  // Registered from the factory body, like the buffer half does: the keyed
  // registry makes a re-registration (a remount re-runs `setup`) a replacement of
  // this holder's entry rather than a second copy of it.
  registerReRootHandler('workspaceSearch', () => {
    generation += 1 // an answer in flight belongs to the root that was just left
    state.query = ''
    state.caseSensitive = false
    state.regex = false
    state.scope = null
    state.open = false
    state.status = 'idle'
    state.result = null
    state.error = null
    state.pendingReveal = null
  })

  return { state, openFor, closeDialog, run, openHit }
}
