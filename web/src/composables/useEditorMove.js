import { reactive } from 'vue'

import { useEditorFiles } from './useEditorFiles'
import { useEditorNotices } from './useEditorNotices'
import { useGitStatus } from './useGitStatus'
import { registerReRootHandler, useWorkspaceTree } from './useWorkspaceTree'

// ---------------------------------------------------------------------------
// The batch move flow: marked rows, one destination folder, one request
// (`POST /api/fs/move-batch`).
//
// A sibling of `useEditorRename` and deliberately not a case inside it: a
// rename answers into the one row whose input renders *inside* a listing, while
// a batch has no single row — its subject is a list, and its answer reaches
// every holder of every path in that list. So the prompt lives here as
// composable state (`movePrompt`) rather than in a component: the dialog (mov-04)
// *renders* it and decides nothing, and a tab switch or a remount cannot drop a
// half-filled modal the way a half-typed name would be dropped with its listing.
//
// The same split of rules as the create and the rename:
//
//   * the *shape* of the typed destination is checked here, so the field answers
//     at once — and that shape is a **folder**: one workspace-relative path,
//     never a full target path. Renaming while moving stays `Rename…`'s job,
//     which is therefore the only place a basename rule lives;
//   * whether that folder is *allowed* is the server's rule. `.env`, `.git/…`, an
//     `ro` or `hidden` folder and a taken target name all come back as a refusal
//     in the policy's own wording, and nothing here grows a second list of
//     protected paths: no copy of the deny-list, no pre-check of writability.
//
// Two asymmetries are settled, not accidental:
//
//   * the client refuses **any** `.`/`..` segment while the server refuses only a
//     `..` that leaves the workspace (`notes/tmp/../archive` is legal there). A
//     field that resolved `..` would need a second path model in the browser to
//     explain what it had typed, and the prefill already hands the user a path
//     they can edit instead;
//   * the value is always **workspace-relative** and sent verbatim — like every
//     path the app holds, so a working-location switch needs nothing here. The
//     target is joined by the **server** (`to_parent + "/" + basename`), because
//     a client that joined it could hide a target the ancestor gate never saw.
//
// A destination *under one of the selected files* is impossible by construction —
// a folder row never joins a batch (`toggleMark` refuses it) — so no cycle check
// is written, and the invariant it would depend on belongs to the tree.
// ---------------------------------------------------------------------------

function parentOf(path) {
  const index = String(path ?? '').lastIndexOf('/')
  return index === -1 ? '' : String(path).slice(0, index)
}

// The shape half of a destination answer: null when the value may be sent, a
// sentence when it may not. `parent` is the batch's shared parent, which the
// prefill puts straight into the field — so the no-op case answers itself
// instead of costing a guaranteed 409 (every target would be its own source).
export function validateMoveDestination(value, { parent } = {}) {
  const text = (value ?? '').trim()
  if (!text) return 'A destination folder is needed'
  if (text.includes('\\')) return 'A folder path cannot contain a backslash'
  if (text.includes('\0')) return 'A folder path cannot contain that character'
  if (text.startsWith('/') || text.startsWith('~')) {
    return 'The destination must be relative to the working location'
  }
  if (text.split('/').some((segment) => segment === '.' || segment === '..')) {
    return 'The destination cannot contain . or ..'
  }
  // The prefill makes a single-file move open with this showing and the Move
  // button disabled: enabling it is the edit, and the sentence explains the
  // case for free.
  if (parent !== undefined && text === parent) return 'That is where they already are'
  return null
}

// The prompt the dialog renders: `{ paths, parent, value, error, busy }`, or
// null while nothing is being moved. `paths` is the batch as decided *when the
// menu opened* (`batchFor`), and `startMove` copies that list in at once — by the
// time a button is pressed the marks may have moved, and the dialog must not end
// up about a set that no longer exists.
const state = reactive({
  movePrompt: null,
})

// One batch at a time, for the create's and the rename's reason: a second Enter
// for one batch would make the second answer a 409 about the first.
let inFlight = null

export function useEditorMove() {
  const tree = useWorkspaceTree()
  const files = useEditorFiles()
  const git = useGitStatus()
  // The one write-feedback channel. The refusal is already in the field, so the
  // notice is not a second place to read it — it is the copy for a user whose
  // eyes are on the code pane, and the only confirmation a batch landed at all.
  const notices = useEditorNotices()

  // Ask where to move `batch` — the `{ paths, parent }` object `batchFor`
  // answered for the row the menu was opened on. The field is prefilled with the
  // sources' own parent: the common case becomes an edit rather than a path typed
  // from memory, and the no-op case shows its own sentence.
  function startMove(batch) {
    const paths = [...(batch?.paths ?? [])]
    if (!paths.length) return false
    const parent = batch.parent ?? parentOf(paths[0])
    state.movePrompt = {
      paths,
      parent,
      value: parent,
      error: null,
      busy: false,
    }
    return true
  }

  // Cancel keeps every path where it is: the destructive answer is the second
  // button, so Esc, the backdrop and Enter-on-first-focus all mean this.
  function cancelMove() {
    state.movePrompt = null
  }

  // A working-location switch: the prompt's paths are relative to a root that no
  // longer exists, which is the argument `reRoot` already makes for `selected`,
  // `editing` and the marks. Cancelled, not re-rooted — nothing is asked of the
  // new root. Registered from here, under its own key in the tree's *list* of
  // re-root subscribers (next to the buffers and the search), so this module
  // stays a writer the tree never imports.
  registerReRootHandler('editorMove', () => {
    cancelMove()
  })

  // Send the batch. One request: the all-or-nothing this feature promises cannot
  // be had from N renames fired here, the first already on disk when the fifth
  // is refused.
  //
  // On success, in the order the state matters in (§5.2): the records that name
  // each old path are re-pointed **first** (so no tick of the freshness probe
  // ever sees a tab whose path just vanished, which would mark it deleted and
  // offer to reload a file that was only *moved*), then the tree forgets each old
  // row, then the **two** listings the batch touched are re-fetched — once each,
  // not once per file — then one `git.load` and one notice.
  //
  // On refusal nothing moves: no record re-points, no folder re-lists, the marks
  // stay, and the field keeps what was typed, so the fix is an edit and not a
  // restart.
  async function submitMove() {
    const pending = state.movePrompt
    if (!pending) return { ok: false, reason: 'no-prompt' }
    if (inFlight) return inFlight

    const paths = [...pending.paths]
    const typed = String(pending.value ?? '').trim()
    const problem = validateMoveDestination(typed, { parent: pending.parent })
    if (problem) {
      pending.error = problem
      return { ok: false, reason: 'invalid-destination', text: problem }
    }

    const run = (async () => {
      pending.busy = true
      let res
      let body = null
      try {
        res = await fetch('/api/fs/move-batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paths, to_parent: typed }),
        })
        body = await res.json().catch(() => null)
      } catch (exc) {
        const text = `Could not move ${paths.length} files: the API could not be reached`
        pending.error = text
        pending.busy = false
        notices.notice('error', text, paths[0])
        return { ok: false, reason: 'network', paths, text }
      }

      if (!res.ok) {
        // The policy's words, "the folder is not there", "already there": the
        // server's rules in its own sentences. A taken target answers with the
        // {reason, path, dir} object rather than a sentence, so it is phrased
        // from the reason the server gave — the same shape a rename's refusal
        // takes, for the same reason.
        const detail = body?.detail
        const reason =
          typeof detail === 'string'
            ? detail
            : detail?.reason === 'exists'
              ? `${detail.path} is already ${detail.dir ? 'a folder' : 'a file'} on disk`
              : `the request was refused (${res.status})`
        // The notice names both ends the user typed: N files → destination.
        const text = `Could not move ${paths.length} files → ${typed}: ${reason}`
        // The field's own line stays the bare reason — it sits under an input
        // that names both ends already.
        pending.error = reason
        pending.busy = false
        notices.notice('error', text, typed)
        return { ok: false, reason: 'refused', paths, toParent: typed, text }
      }

      const moved = Array.isArray(body?.moved) ? body.moved : []
      const toParent = body?.to_parent ?? typed

      // The buffers first, for the reason above, then the tree forgets the row
      // each one used to have. `dropBranch` on a file is the cheap case: a file
      // has no subtree of its own.
      for (const item of moved) {
        files.repointPath(item.from, item.to)
        tree.dropBranch(item.from)
      }

      // The listings the batch changed: the folder each file left and the one it
      // arrived in — normally exactly **two**, whichever way N goes, which is the
      // whole efficiency the batch buys on this side. `refreshBranch` skips one
      // nothing on screen shows, which is right.
      const folders = new Set([toParent])
      for (const item of moved) folders.add(parentOf(item.from))
      await Promise.all([...folders].map((folder) => tree.refreshBranch(folder)))

      // And one level higher at *both* ends: the folder the files moved into may
      // have been listed as an empty branch (its row above needs the arrow back),
      // and the one they left may now be empty (its row above needs the arrow
      // gone). `children` is a promise about that row, so each side checks it —
      // with one path per end, which is what the rename passes: every `from`
      // shares one parent and every `to` shares `toParent`, so one entry each
      // answers for the whole set.
      const bothEnds = [moved[0]?.from, moved[moved.length - 1]?.to].filter(Boolean)
      for (const path of bothEnds) {
        await tree.refreshListingOfParent(path)
      }

      // The prompt goes before anything else: a modal over a tree that is being
      // re-listed underneath it is the worst of both, which is the reason the
      // delete confirmation gives for clearing first.
      state.movePrompt = null
      // The batch is spent: its rows are elsewhere now, and a set of marks
      // pointing at paths no listing shows is how a batch outlives its own rows.
      tree.clearMarks()
      // A move is a git event, and the batch is one event.
      void git.load({ force: true })
      // **One** notice for the whole batch: N confirmations would bury the tab
      // strip, and the paths are in the payload if a detail view ever wants them.
      notices.notice(
        'ok',
        `Moved ${moved.length} ${moved.length === 1 ? 'file' : 'files'} → ${toParent}`,
        toParent,
      )
      // The destination is opened so the user sees where the rows went, then the
      // last moved row is revealed — the row the eye is sent to.
      tree.expand(toParent)
      await tree.reveal(moved[moved.length - 1]?.to)
      return { ok: true, moved, toParent }
    })()
    inFlight = run
    try {
      return await run
    } finally {
      inFlight = null
    }
  }

  return { state, startMove, cancelMove, submitMove }
}
