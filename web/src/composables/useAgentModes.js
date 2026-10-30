import { reactive } from 'vue'

// ---------------------------------------------------------------------------
// Agent modes: the persona WorkBench answers as, and what it may reach.
//
// One fetch (`GET /api/agent-modes`, am-06) fills this singleton: the dropdown
// in the composer, the hub panels' gray-out and the `agentMode` key of every
// chat payload all read *here*, so the same verdict drives all three. The list
// is the workspace's — `.nojae/agent-modes.json`, edited outside the app — so
// nothing here holds a mode id: the app-owned mode is recognised by
// `editable: false`, which is data from the server. That is deliberate — the
// id that mode has is the server's to know, and `web/src` must never contain it
// (a later test in this lot holds that line).
//
// Every decision lives in this file as a plain function, so a Settings editor
// for modes (a later lot) can reuse the rules instead of re-deriving them in a
// component. The rules that judge a *hypothetical* mode take it as an argument
// (`agentModeLockedFor`), so they stay testable without a mounted app.
// ---------------------------------------------------------------------------

const state = reactive({
  modes: [], // [{ id, label, mcptools, lsptools, editable }] — dropdown order
  current: null, // the mode id in play; null until the list loaded
  default: null, // what the server says this workspace's default is
  loaded: false, // false = the fetch failed; send() then omits agentMode
  locked: false, // D7: the current mode owns a contract and the thread turned
  findings: [], // validation findings about the modes file, if any
  source: null, // { path, exists, loaded, version } — where the list came from
  notice: null, // one quiet line the composer shows under the select
})

// A second fetch already in flight (tab entry racing a workspace switch).
let inFlight = null

// ---------------------------------------------------------------------------
// Pure rules
// ---------------------------------------------------------------------------

// Turn-count probe, injected by useChat the way `registerChatBusyGuard` injects
// the batch flag: useAgentModes must not import useChat (useChat imports this
// module), and the thread's length is the only fact the D7 lock needs.
let turnCounter = null

export function registerAgentModeTurnCounter(hasTurns) {
  turnCounter = typeof hasTurns === 'function' ? hasTurns : null
}

function threadHasTurns() {
  return !!turnCounter?.()
}

// The mode object for an id (or for `state.current` when called with nothing).
export function modeById(id) {
  const wanted = id ?? state.current
  return state.modes.find((mode) => mode.id === wanted) ?? null
}

export function currentMode() {
  return modeById(state.current)
}

// `true` when the current mode switches that hub off, so the panel is disabled.
//
// Gated on `loaded` as much as on the flags: with a stale list the payload
// carries no `agentMode`, so the **server's** default answers the turn and this
// client cannot know what it gates. Gray-out then would be a lie, so the panels
// behave exactly as they did before the feature (no gate, no reason line).
export function modeBlocksHub(hub) {
  const mode = state.loaded ? currentMode() : null
  return !!mode && mode[`${hub}tools`] === false
}

export function hubGateReason(hub) {
  const mode = currentMode()
  const label = hub === 'mcp' ? 'MCP' : 'LSP'
  return mode
    ? `${label} tools are not available in ${mode.label} mode — switch mode, or change them in Settings`
    : ''
}

// The select's `title`: the one place the tool gating is explained to the user.
export function modeHint(id) {
  const mode = modeById(id)
  if (!mode) return ''
  const hubs = []
  if (mode.mcptools) hubs.push('MCP')
  if (mode.lsptools) hubs.push('language servers')
  const tools = hubs.length
    ? `${hubs.join(' and ')} available`
    : 'no tools'
  return `${mode.label} — ${tools}`
}

// The `<select>`'s title: the hint, qualified by whatever currently stops the
// user acting on it. Kept here (not in the component) because it is the one
// place D7 and D13 are explained to a human, and both rules are this module's.
export function modeSelectTitle({ batchActive = false } = {}) {
  const hint = modeHint(state.current)
  if (!hint) return ''
  if (batchActive) return `${hint} — a running batch picks its own mode`
  if (state.locked) {
    return `${hint} — this conversation's mode is fixed once it has a turn`
  }
  return hint
}

// D7: a mode is locked when it is **not** editable (it owns a contract — the
// lot conventions) *and* the thread already has a turn. Entering is always
// allowed; leaving a locked mode is what the guard refuses.
export function agentModeLockedFor(mode, hasTurns) {
  return !!mode && !mode.editable && !!hasTurns
}

// Recomputed on every turn (a send, a reset, a session load) rather than
// remembered: the lock is a function of the thread, not a flag someone sets.
export function syncAgentModeLock(hasTurns) {
  state.locked = agentModeLockedFor(currentMode(), hasTurns ?? threadHasTurns())
}

// The D7 guard, as a predicate so `setAgentMode` and the `<select>`'s option
// list ask the same question: a locked thread keeps its mode.
export function modeOptionDisabled(id) {
  return state.locked && id !== state.current
}

// Whether the whole `<select>` is inert: D7's lock, or a running lot whose mode
// the runner pinned (D13). A mid-`streaming` switch is *not* in here — it is
// allowed and simply cannot reach the turn already in flight.
//
// The batch half is a *rendering* rule; the behaviour lives in useBatch, which
// resolves `LOT_TASK_MODE_ID` once per run into `state.pinnedMode` and reads it
// per task turn. So the select is disabled while a lot runs for the honest
// reason: changing it can neither move the running lot nor the next task of it,
// and leaving it live would advertise a power the dropdown does not have over
// a lot.
export function modeSelectDisabled(batchActive = false) {
  return state.locked || !!batchActive
}

// The guard is **client-side convenience, not enforcement**: the server accepts
// an `agentMode` change at any point in a thread, including this one, because a
// resumed session JSON may legitimately want a different persona for its next
// turn. This exists to stop an accidental click, not to police the user.
export function setAgentMode(id) {
  if (state.locked) return // leaving a locked mode: refused
  const next = modeById(id)
  if (!next) return // never the server's list, never a guess
  state.current = next.id
  // Entering a contract-owning mode on a thread that already turned locks at
  // once — no second click needed to see the lock.
  syncAgentModeLock(threadHasTurns())
}

// What `send()` puts in `payload.agentMode`: null means *omit the key*, so the
// server's own default applies — the same rule (and the same reason) as the
// `mcpEnabled` / `lspEnabled` allowlists in useChat's payload.
export function agentModePayloadId() {
  return state.loaded && state.current ? state.current : null
}

// One quiet line about a hand-edited file the server had problems with. Not a
// modal and not a banner: the file belongs to the user and lives outside the
// app, and a warning that cannot be dismissed would nag on every turn.
export function findingsLine() {
  if (!state.findings.length) return null
  const n = state.findings.length
  const path = state.source?.path ?? '.nojae/agent-modes.json'
  return `${n} problem${n === 1 ? '' : 's'} in the workspace agent-modes file — see ${path}`
}

// The notice line is about one *loaded session*, so a new conversation retires
// it. `findingsLine()` needs no such reset: the file's problems are the
// workspace's, and they come back with the next fetch of the list.
export function clearAgentModeNotice() {
  state.notice = null
}

// The mode a loaded session file asked for, judged against *this* workspace.
// An unknown id falls back to the workspace default — never to a different
// persona silently, and never by refusing to load the conversation (the same
// non-blocking rule the server applies to the file itself, D11).
export function applySessionAgentMode(rawId) {
  const wanted = typeof rawId === 'string' ? rawId.trim() : ''
  if (!wanted) {
    // A v3 file carries no mode: leave `current` at the workspace default.
    state.notice = null
    if (!state.current && state.default) state.current = state.default
    syncAgentModeLock(threadHasTurns())
    return null
  }
  if (!state.loaded) {
    // Nothing trustworthy to judge it against, and `send()` is already omitting
    // the key — so the server's default answers this conversation. Said out
    // loud, because a saved mode that silently does nothing reads as a bug.
    state.notice = `The saved session asked for agent mode "${wanted}", but this workspace's mode list could not be read — the server's default applies.`
    syncAgentModeLock(threadHasTurns())
    return null
  }
  if (modeById(wanted)) {
    state.current = wanted
    state.notice = null
    syncAgentModeLock(threadHasTurns())
    return wanted
  }
  state.current = state.default
  state.notice = `The saved session asked for agent mode "${wanted}", which this workspace does not have — using ${state.default ?? 'the workspace default'}.`
  syncAgentModeLock(threadHasTurns())
  return null
}

// ---------------------------------------------------------------------------
// The fetch
// ---------------------------------------------------------------------------

function acceptPayload(body, { reseed }) {
  const modes = Array.isArray(body?.modes) ? body.modes : []
  if (!modes.length) throw new Error('no modes in the response')
  state.modes = modes.map((mode) => ({
    id: mode.id,
    label: mode.label ?? mode.id,
    mcptools: mode.mcptools !== false,
    lsptools: mode.lsptools !== false,
    editable: mode.editable !== false,
  }))
  state.findings = Array.isArray(body.findings) ? body.findings : []
  state.source = body.source ?? null
  state.default =
    typeof body.default === 'string' && modeById(body.default)
      ? body.default
      : state.modes[0].id
  // A workspace switch re-seeds unconditionally: an id that existed in the old
  // root's file probably does not exist in the new one, and the alternative is
  // a 422 on every send. A plain refresh (tab entry) keeps the user's choice
  // whenever the mode is still offered.
  if (reseed || !modeById(state.current)) state.current = state.default
  // Leaving the root retires a notice about a session loaded from it. A plain
  // refresh keeps it: the conversation it describes is still on screen.
  if (reseed) state.notice = null
  state.loaded = true
  syncAgentModeLock(threadHasTurns())
}

// Load the workspace's modes. `reseed: true` (the working-location switch, and
// the first load) also moves `current` to the new default.
//
// Failure **hides nothing and changes nothing**: `loaded` goes false, `current`
// stays as it was, and `send()` omits `agentMode` — so the server answers with
// its own default, exactly what a client predating the feature gets. A failed
// dropdown must never silently downgrade anybody to another persona, which is
// also why the previous list survives a failed *refresh*: the composer keeps
// showing the mode the last turn actually used.
export async function loadAgentModes({ reseed = false } = {}) {
  if (inFlight) return inFlight
  inFlight = (async () => {
    try {
      const res = await fetch('/api/agent-modes')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      acceptPayload(await res.json(), { reseed })
    } catch {
      state.loaded = false
      state.findings = []
      state.notice = null
    } finally {
      inFlight = null
    }
  })()
  return inFlight
}

export function agentModesLoaded() {
  return state.loaded
}

export function useAgentModes() {
  return { state, loadAgentModes, setAgentMode, currentMode, modeHint }
}
