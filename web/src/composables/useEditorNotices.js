import { reactive } from 'vue'

// ---------------------------------------------------------------------------
// The Editor's write-feedback channel: a small stack of notices, shown at the
// bottom-centre of the app.
//
// It exists because `useEditorFiles.state.message` — the one line above the
// code pane — is the right shape for a *refused open* and the wrong shape for a
// write. A failed save is news about work the user still holds, so a click on
// another tab must not eat the reason; and a refused rename or delete happens
// while the user is looking at the *tree*, which the pane's message line does
// not reach. This channel answers both: it is dismissible, it does not clear on
// the next click, and it is rendered as a fixed stack so it is visible from
// whichever half of the Editor the eyes are on.
//
// Three rules hold it together:
//
//   * **dedupe** — the same (tone, text, path) inside a short window is one
//     notice, not a stack of them, so anything that can be asked repeatedly (a
//     held chord, the 5 s freshness tick) makes noise at most once;
//   * **a refusal never times out** — only `ok` expires, because a refusal that
//     vanishes on a timer is the bug this channel exists to avoid;
//   * **a success answers the failure it was about** — `dismissForPath` drops
//     the error line once that path's write lands, so the stack never claims a
//     buffer is still unwritten.
//
// Nothing here decides *what* a refusal means. The callers pass the server's
// own `detail` verbatim, prefixed by what was attempted, and this layer only
// decides how long it stays and how many of them are on screen.
// ---------------------------------------------------------------------------

// How long a confirmation stays up on its own. `error` and `warn` never do:
// they leave when the user dismisses them or when the thing they are about is
// fixed.
export const NOTICE_TTL_MS = 4000

// Inside this window, an identical notice is silence rather than a new row.
// Longer than the freshness poll (FRESHNESS_POLL_MS) so a tick re-reporting the
// same news cannot add one per tick, short enough that a genuinely repeated
// failure much later is reported again.
export const NOTICE_DEDUPE_WINDOW_MS = 8000

// The stack's size: three lines is what a bottom-centre stack can hold without
// covering the pane, and once newer refusals are up the oldest news is the
// least actionable of them.
export const MAX_VISIBLE_NOTICES = 3

const state = reactive({
  notices: [], // [{ id, tone: 'error'|'warn'|'ok', text, path, at }]
})

let nextId = 1
const timers = new Map()

function sameNews(item, tone, text, path) {
  return item.tone === tone && item.text === text && (item.path ?? null) === path
}

function dismiss(id) {
  const index = state.notices.findIndex((item) => item.id === id)
  if (index !== -1) state.notices.splice(index, 1)
  const timer = timers.get(id)
  if (timer !== undefined) {
    clearTimeout(timer)
    timers.delete(id)
  }
}

function clear() {
  for (const item of [...state.notices]) dismiss(item.id)
}

// Raise one notice, or say nothing when it is the same news as one already on
// screen. Returns the notice that stands, so a caller can compare if it cares.
function notice(tone, text, path = null) {
  const at = Date.now()
  const target = path ?? null
  const existing = state.notices.find((item) => sameNews(item, tone, text, target))
  if (existing && at - existing.at <= NOTICE_DEDUPE_WINDOW_MS) return existing

  const item = { id: nextId, tone, text, path: target, at }
  nextId += 1
  state.notices.push(item)
  // Oldest first: what the user has already seen (and not acted on) is the
  // first thing to make room.
  while (state.notices.length > MAX_VISIBLE_NOTICES) dismiss(state.notices[0].id)
  if (tone === 'ok') timers.set(item.id, setTimeout(() => dismiss(item.id), NOTICE_TTL_MS))
  return item
}

// True when `candidate` is the path itself or below it — the same prefix rule
// the buffer records use, so "the news about this path" means the same thing in
// both layers.
function sameOrUnder(candidate, path) {
  if (typeof candidate !== 'string' || typeof path !== 'string') return false
  return candidate === path || candidate.startsWith(`${path}/`)
}

// A write that landed answers the refusal it was fighting: drop the error/warn
// lines about that path (or its subtree) rather than leave a stack of them
// describing work that is safely on disk. `ok` lines are left alone — a
// confirmation about a path is not made false by a later write.
//
// `tone` narrows that to one kind, which is what lets a probe that found the
// disk back in agreement silence *its* warning without swallowing an unrelated
// save refusal about the same file.
function dismissForPath(path, tone = null) {
  if (!path) return
  for (const item of [...state.notices]) {
    if (item.tone === 'ok') continue
    if (tone !== null && item.tone !== tone) continue
    if (sameOrUnder(item.path, path)) dismiss(item.id)
  }
}

// A rename re-points the refusals that name the old path — those are about
// buffers the user still holds, and those buffers now live at the new path. A
// *confirmation* of a past mutation keeps naming the path it happened at, which
// the move just un-named, so it goes away instead of lying.
function repointNotices(from, to) {
  if (!from || !to) return
  for (const item of [...state.notices]) {
    if (!sameOrUnder(item.path, from)) continue
    if (item.tone === 'ok') {
      dismiss(item.id)
      continue
    }
    state.notices[state.notices.indexOf(item)] = {
      ...item,
      path: to + item.path.slice(from.length),
    }
  }
}

// The refusal a path currently carries, if any. The read-only badge quotes it
// so the pane can answer "why did Ctrl+S do nothing?" from where the user is
// already looking.
function noticeForPath(path) {
  return state.notices.find((item) => item.tone !== 'ok' && item.path === path) ?? null
}

export function useEditorNotices() {
  return { state, notice, dismiss, clear, dismissForPath, repointNotices, noticeForPath }
}
