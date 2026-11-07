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
//
// That editor landed below (eds-05): the dialog's rules over a **draft
// document** — the editable copy of what `GET /api/agent-modes/document`
// answered — plus its two calls. The dropdown's rules above are untouched, and
// the save feeds the dropdown through `acceptPayload`, so a saved document
// reloads the list without a second round-trip (D9). No cap is spelled here:
// every number the form enforces arrives in the document payload's `caps`.
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

// ---------------------------------------------------------------------------
// The modes editor (eds-05): the dialog's rules over a *draft document*.
//
// `GET /api/agent-modes/document` answers *what my file says* (D2 — the file's
// own entries verbatim, broken ones shown with their finding, plus the `mtime`
// token the save compares). `createAgentModesDraft` turns that payload into the
// dialog's editable copy; everything below judges **that draft** as plain
// functions, so the component renders answers instead of re-deriving rules, and
// every rule stays testable without a mounted app.
//
// No cap is spelled in this half either: every number the form enforces arrives
// in the payload's `caps` (the mode count, the id pattern and length, the label
// cap, the persona byte cap, the persona line budget). The label cap is
// deliberately absent from the form rules — a label over it is the server's
// trim, not a form error (D3: "it is presentation, and the loader already
// treats it that way").
// ---------------------------------------------------------------------------

function capsOf(draft) {
  return draft?.caps ?? {}
}

function byteLengthOf(text) {
  return new TextEncoder().encode(text).length
}

// The slug a new mode's id starts as (D5): lowercase, then letters/digits/
// '_'/'-', within the payload's id cap, made unique against the ids already in
// the document. `caps` is the draft's own `caps` object — the numbers come from
// the payload, never a literal here (a missing cap means "no client-side cap",
// which the server still enforces on save).
export function slugForModeId(label, takenIds = [], caps = null) {
  const maxIdChars = Number.isFinite(caps?.maxIdChars) ? caps.maxIdChars : null
  let slug = String(label ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^[-_]+|[-_]+$/g, '')
  if (!slug) slug = 'mode' // an empty slug is not an id; a neutral one is editable
  if (!/^[a-z]/.test(slug)) slug = `m-${slug}` // the pattern wants a lowercase start
  if (maxIdChars !== null && slug.length > maxIdChars) slug = slug.slice(0, maxIdChars)
  const taken = new Set(takenIds ?? [])
  if (!taken.has(slug)) return slug
  // Uniqueness: a numeric suffix, and the base gives up its tail for it — the
  // result never exceeds the cap, because an over-cap id is what the validator
  // drops, and the form would then refuse its own Create.
  for (let n = 2; ; n += 1) {
    const suffix = `-${n}`
    const base =
      maxIdChars === null ? slug : slug.slice(0, Math.max(1, maxIdChars - suffix.length))
    const candidate = `${base}${suffix}`
    if (!taken.has(candidate)) return candidate
  }
}

// The draft's ids, minus one **entry**'s (matched by identity, not by key: a
// new mode whose typed id collides with an existing one must still be told it
// collides) — the "taken" set a slug or a typed id is judged against.
function takenKeys(draft, exceptEntry = null) {
  return (draft?.entries ?? [])
    .filter((entry) => entry !== exceptEntry)
    .map((entry) => entry.key)
}

// Slug generation for one draft: the label, and every id the document already
// holds, under the draft's own caps.
export function slugForDraft(draft, label) {
  return slugForModeId(label, takenKeys(draft), capsOf(draft))
}

// Whether an entry's id field is editable: **only while the mode is new** (D5).
// Renaming an existing id is a delete plus an add and silently orphans every
// saved session that names it, so this is a function's answer, not a template's
// `:disabled`.
export function idEditableFor(entry) {
  return !!entry && entry.isNew === true && entry.hasForm === true
}

// Why a *new* mode's id is not usable in this document, or null when it is:
// the payload's pattern, the payload's length cap, and uniqueness against the
// ids already declared. Existing entries are not judged here — their key came
// from the file and their id field is read-only (D5).
export function idValidityProblem(draft, entry) {
  if (!idEditableFor(entry)) return null
  const caps = capsOf(draft)
  const id = entry.key
  if (typeof id !== 'string' || !id) return 'An id is required.'
  if (Number.isFinite(caps.maxIdChars) && id.length > caps.maxIdChars) {
    return `The id must be at most ${caps.maxIdChars} characters.`
  }
  if (typeof caps.idPattern === 'string' && new RegExp(caps.idPattern).test(id) === false) {
    return `The id must match ${caps.idPattern} — a lowercase letter, then letters, digits, '_' or '-'.`
  }
  if (takenKeys(draft, entry).includes(id)) {
    return `The id "${id}" is already used by another mode.`
  }
  return null
}

// The persona's two counters. The **byte** one is hard (over the payload's cap
// is invalid and blocks Save — the loader drops such an entry whole); the
// **line** one is soft: it warns against the payload's budget and never blocks,
// because it is a prose style rule, not a file rule.
export function personaByteCount(text) {
  return byteLengthOf(typeof text === 'string' ? text : '')
}

export function personaLineCount(text) {
  const value = typeof text === 'string' ? text : ''
  return value === '' ? 0 : value.split('\n').length
}

export function personaOverByteCap(draft, text) {
  const cap = capsOf(draft).maxPersonaBytes
  return Number.isFinite(cap) && personaByteCount(text) > cap
}

// The small label under the textarea: how much of the line budget is left, and
// the warning once it is spent. Null when the payload states no budget.
export function personaLineBudget(draft, text) {
  const budget = capsOf(draft).maxPersonaLines
  if (!Number.isFinite(budget)) return null
  const lines = personaLineCount(text)
  return {
    budget,
    lines,
    remaining: Math.max(0, budget - lines),
    over: lines > budget,
    warning:
      lines > budget
        ? `The persona runs ${lines} lines, over the ${budget}-line budget this app writes personas to — a style budget, not a file rule.`
        : null,
  }
}

// The draft's own over-the-cap persona as a finding, in the loader's shape
// (pointer + message): the form blocks Save on it before the server would drop
// the entry (D3 — the caps are enforced *in the form*, not discovered through a
// refused save).
function draftPersonaFinding(draft, entry) {
  if (!personaOverByteCap(draft, entry.personaText)) return null
  const cap = capsOf(draft).maxPersonaBytes
  const size = personaByteCount(entry.personaText)
  return {
    kind: 'error',
    message:
      `agents.${entry.key}.systemMessage: is ${size} bytes, over the ${cap}-byte cap — ` +
      'this text is re-sent on every turn of the conversation',
  }
}

// The mode-count cap: the form stops being valid **past** the cap rather than
// the save being refused by the server later.
export function modeCountProblem(draft) {
  const cap = capsOf(draft).maxModes
  const count = draft?.entries?.length ?? 0
  if (!Number.isFinite(cap) || count <= cap) return null
  return `The document holds ${count} modes, over the ${cap} this file may hold — delete one before adding or saving.`
}

// Whether Create would take the draft past the cap: the button is inert *at* the
// cap, so the user never discovers the cap through a blocked Save.
export function createDisabledFor(draft) {
  const cap = capsOf(draft).maxModes
  return Number.isFinite(cap) && (draft?.entries?.length ?? 0) >= cap
}

// Which field of an entry a finding talks about, out of the pointer it names
// (`agents.code.systemMessage: …` → 'systemMessage'). A finding naming the
// entry itself (`agents.code: …`) has no field, and neither has one naming a
// key the vocabulary does not know — that is an unknown key, which the form
// cannot fix.
function findingField(entry, message) {
  const prefix = `agents.${entry.key}.`
  if (!String(message ?? '').startsWith(prefix)) return null
  const field = String(message).slice(prefix.length).split(':')[0]
  return ['label', 'mcptools', 'lsptools', 'systemMessage'].includes(field) ? field : null
}

// Is this file-level finding about the file's `default` key? The loader's
// sentences about that key always name it quoted.
function isDefaultFinding(message) {
  return String(message ?? '').includes('"default"')
}

// The draft's own `default` rule, mirroring what the save refuses: a document
// that declares modes must set a default naming one of them. A payload finding
// about the default is resolved by the Common tab's choice — that is the one
// file-level finding of its kind the dialog can fix — so this check, not the
// stale payload finding, is what holds Save once entries exist.
function draftDefaultProblem(draft) {
  const entries = draft?.entries ?? []
  if (!entries.length) return null
  const keys = entries.map((entry) => entry.key)
  if (keys.includes(draft.default)) return null
  return `The default mode must name one of the modes this document declares (${keys.join(', ')}).`
}

// D3's blocking findings: every `error` kind, about an entry or the file as a
// whole, that the draft has **not** fixed. A `note` (a file that named a
// reserved id by hand) never holds Save — the save is what cleans it up (D4).
//
// "Fixed" is judged against the draft's current values:
// * a finding on a field the form can only produce validly for (label, the two
//   switches) is resolved by the draft holding that field at all;
// * a persona finding is re-measured against the payload's byte cap — editing
//   the text down under it resolves it, leaving it over keeps blocking, and a
//   persona shape no textarea can hold (an empty list, a list with a non-string)
//   keeps its loader finding;
// * an unknown key, an entry with no form, and any file-level finding except
//   the default ones are not fixable in-form: they block until fixed or
//   deleted, exactly as D3 says.
export function blockingFindings(draft) {
  const blocks = []
  for (const entry of draft?.entries ?? []) {
    if (!entry.hasForm) {
      // An unrepresentable entry is visible and blocks: Delete is its only way
      // out, and a save that quietly dropped it is the loss D2 exists to prevent.
      for (const finding of entry.findings ?? []) {
        if (finding.kind === 'error') blocks.push(finding)
      }
      continue
    }
    for (const finding of entry.findings ?? []) {
      if (finding.kind !== 'error') continue
      const field = findingField(entry, finding.message)
      if (field === 'label' || field === 'mcptools' || field === 'lsptools') continue
      if (field === 'systemMessage' && entry.personaText !== null) continue // re-measured below
      blocks.push(finding) // an unknown key, a persona shape, or a finding naming no field
    }
    const persona = draftPersonaFinding(draft, entry)
    if (persona) blocks.push(persona)
    // D3's third cap: the id. A *new* mode's id is the one the form can still
    // make wrong (an existing entry's key came from the file and its field is
    // read-only), so an id the payload's pattern or length cap refuses blocks
    // Save here rather than becoming a save the server sends back.
    const idProblem = idValidityProblem(draft, entry)
    if (idProblem) blocks.push({ kind: 'error', message: idProblem })
  }
  const hasEntries = (draft?.entries?.length ?? 0) > 0
  for (const finding of draft?.findings ?? []) {
    if (finding.kind !== 'error') continue
    if (hasEntries && isDefaultFinding(finding.message)) continue // the Common tab answers it
    blocks.push(finding)
  }
  const defaultProblem = draftDefaultProblem(draft)
  if (defaultProblem) blocks.push({ kind: 'error', message: defaultProblem })
  const countProblem = modeCountProblem(draft)
  if (countProblem) blocks.push({ kind: 'error', message: countProblem })
  return blocks
}

// The one answer the dialog renders for its Save button, so the rule is not
// re-derived in the component: blocking findings, or a save already in flight.
export function saveDisabledFor(draft, { saving = false } = {}) {
  return saving || blockingFindings(draft).length > 0
}

// Which findings one entry's tab shows: the loader's own sentences naming it,
// plus the draft's over-the-cap persona finding (the same shape, produced by
// `blockingFindings`'s re-measurement). A tab renders these verbatim next to
// the field each one names — the dialog adds nothing to the loader's voice and
// filters nothing out, which is what keeps a hand-written entry that names a
// reserved id *readable* even though it gets no tab of its own (D4).
export function entryFindingsFor(draft, entry) {
  if (!entry) return []
  const own = Array.isArray(entry.findings) ? entry.findings : []
  const persona = entry.hasForm ? draftPersonaFinding(draft, entry) : null
  return persona ? [...own, persona] : own
}

// Whether *this entry* is one of the things holding Save: a loader error it
// carries, the draft's over-the-cap persona, or a new mode's id the payload
// refuses. The dialog asks this to decide which tab to open on (D3: a finding
// opens its entry's tab) and which tabs to mark, so the answer stays the
// composable's rather than a template reading findings by kind.
export function entryBlocksSave(draft, entry) {
  if (!entry) return false
  if (entry.findings?.some((finding) => finding.kind === 'error')) return true
  if (entry.hasForm && personaOverByteCap(draft, entry.personaText)) return true
  return idValidityProblem(draft, entry) !== null
}

// Which findings the dialog lists as the **file's** own: every file-level
// finding it produced, `error` and `note` alike. A note is listed but never
// holds Save (`blockingFindings` decides that); an error is listed *and* holds
// it — with one exception the listing shares with `blockingFindings`: a finding
// about the file's `default` is answered by the Common tab's choice, so once the
// draft names a declared mode it is listed **no longer**, and the lines on screen
// and the Save button obey one verdict. The two kinds are separated here rather
// than in the template, so the component renders the composable's answer instead
// of sorting findings by kind itself.
export function fileFindingsFor(draft) {
  const list = Array.isArray(draft?.findings) ? draft.findings : []
  const hasEntries = (draft?.entries?.length ?? 0) > 0
  const errors = list.filter(
    (finding) =>
      finding.kind === 'error' &&
      !(hasEntries && isDefaultFinding(finding.message)),
  )
  const notes = list.filter((finding) => finding.kind !== 'error')
  const extra = [draftDefaultProblem(draft), modeCountProblem(draft)].filter(Boolean)
  return { errors: [...errors, ...extra.map((message) => ({ kind: 'error', message }))], notes }
}

// Which field of the form one of an entry's findings talks about (`'label'`,
// `'mcptools'`, `'lsptools'`, `'systemMessage'`, or null when it names the entry
// itself or a key the vocabulary does not know — an unknown key, which the form
// cannot fix). The dialog puts each finding next to the field it names, and this
// is the loader-pointer reading that decides "next to", so it stays here rather
// than becoming a string test in the template.
export function findingFieldFor(entry, message) {
  return entry ? findingField(entry, message) : null
}

// A tab's caption: the Label field it currently holds, falling back to the id
// when the label is empty — the loader's own rule (an empty label is the id),
// and the reason a caption follows the Label field live rather than the file's
// original text. An entry with no form has no label to follow: its raw key is
// its caption, which is what shows the user *what the file says*.
export function tabCaptionFor(entry) {
  if (!entry) return ''
  const label = typeof entry.label === 'string' ? entry.label.trim() : ''
  return label || String(entry.key ?? '')
}

// Dirty tracking: whether the draft differs from what was read. Cancel needs no
// confirm when it is not; Save needs nothing when it is not either (the save
// would rewrite the same bytes — allowed, but pointless to advertise).
function draftSnapshot(draft) {
  return JSON.stringify({
    entries: draft.entries ?? [],
    default: draft.default ?? null,
    readmePresent: draft.readmePresent === true,
    readme: draft.readme ?? null,
  })
}

export function draftIsDirty(draft) {
  return draftSnapshot(draft) !== (draft?.baseline ?? '')
}

// D3's absent-file case: the dialog says plainly that saving will create the
// file, because there is nothing there to lose.
export function draftWillCreateFile(draft) {
  return draft?.source?.exists !== true
}

// The persona as the textarea holds it: a string verbatim, a list joined with
// newlines (the loader's own join, so the round-trip is byte-identical), null
// or absent as empty ("use the configured system prompt"). A shape no form can
// hold (an empty list, a list with a non-string) leaves `personaText` null and
// keeps the raw value for the save body — its finding is what the user sees.
function personaOf(fields) {
  const raw = 'systemMessage' in fields ? fields.systemMessage : undefined
  if (typeof raw === 'string') return { text: raw, raw }
  if (Array.isArray(raw) && raw.length > 0 && raw.every((line) => typeof line === 'string')) {
    return { text: raw.join('\n'), raw }
  }
  return { text: raw === undefined || raw === null ? '' : null, raw }
}

// Turn the document payload into the dialog's editable draft: the file's own
// entries in the file's own order (nothing added — the app-owned mode is absent
// from the payload entirely, D4), each with its fields, its findings and
// `hasForm`. For an absent file the built-in document is the draft (D3's one
// legitimate case) and `draftWillCreateFile` says so.
export function createAgentModesDraft(payload) {
  const defaults = payload?.defaults ?? {}
  const source = payload?.source ?? {}
  // The built-in document is the draft for **one** case (D3): the file is
  // absent, so there is nothing there to lose. A file that exists but holds no
  // usable entries — unparseable, `agents` not an object, every entry dropped —
  // drafts as *empty*, because offering the defaults there would be the loss
  // this read exists to prevent, and its findings are what the dialog shows.
  const absent = source.exists !== true
  const declared = Array.isArray(payload?.entries) ? payload.entries : []
  const rawEntries = absent && !declared.length
    ? Object.entries(defaults.agents ?? {}).map(([key, fields]) => ({
        key,
        hasForm: true,
        fields: fields ?? {},
        findings: [],
      }))
    : declared
  const entries = rawEntries.map((entry) => {
    const fields = entry.fields ?? {}
    const persona = personaOf(fields)
    return {
      key: entry.key,
      hasForm: entry.hasForm !== false,
      isNew: false,
      label: typeof fields.label === 'string' ? fields.label : '',
      mcptools: fields.mcptools === true,
      lsptools: fields.lsptools === true,
      personaText: persona.text,
      personaNull: persona.raw === null,
      personaAbsent: persona.raw === undefined,
      findings: entry.findings ?? [],
      rawEntry: entry,
    }
  })
  const keys = entries.map((entry) => entry.key)
  const declaredDefault =
    typeof payload?.default === 'string' && keys.includes(payload.default)
      ? payload.default
      : null
  const draft = {
    source,
    caps: payload?.caps ?? {},
    defaults,
    findings: Array.isArray(payload?.findings) ? payload.findings : [],
    entries,
    default:
      declaredDefault ??
      (typeof defaults.default === 'string' ? defaults.default : null) ??
      keys[0] ??
      '',
    readmePresent: 'readme' in (payload ?? {}),
    readme: payload?.readme,
    mtime: source.mtime ?? null,
  }
  draft.baseline = draftSnapshot(draft)
  return draft
}

// Create agent (D5): append the tab immediately and focus it — **nothing is
// written**; the new mode exists only in the draft until Save. Its id is the
// slug of its label, editable only while it is new (`idEditableFor`).
export function addModeToDraft(draft, label) {
  const entry = {
    key: slugForDraft(draft, label),
    hasForm: true,
    isNew: true,
    label: String(label ?? ''),
    mcptools: false,
    lsptools: false,
    personaText: '',
    personaNull: false,
    personaAbsent: true,
    findings: [],
    rawEntry: null,
  }
  draft.entries.push(entry)
  return entry
}

// Delete exists (resolved open question 1): per-tab, and refused for the mode
// that is currently the default — a default must name a declared mode.
export function deleteDisabledFor(draft, key) {
  return draft?.default === key
}

export function deleteModeFromDraft(draft, key) {
  if (deleteDisabledFor(draft, key)) return false
  const index = draft.entries.findIndex((entry) => entry.key === key)
  if (index === -1) return false
  draft.entries.splice(index, 1)
  return true
}

// The save body: D6's shape from the draft. `version` and `readme` are not sent
// at all (the writer takes the version from the app's constant and the readme
// from the file on disk); `agents` is the draft's entries in the draft's order;
// an entry keeps only the vocabulary keys it actually had, so a key the file
// never wrote does not appear because the form rendered a default for it. An
// unrepresentable entry travels **verbatim** — the save refuses it (which is
// right: it must be deleted first) rather than the client inventing a shape.
function saveBodyOf(draft) {
  const agents = {}
  for (const entry of draft.entries ?? []) {
    if (!entry.hasForm) {
      agents[entry.key] =
        entry.rawEntry && 'fields' in entry.rawEntry ? entry.rawEntry.fields : entry.rawEntry
      continue
    }
    const fields = {}
    if (entry.label !== '') fields.label = entry.label
    fields.mcptools = entry.mcptools === true
    fields.lsptools = entry.lsptools === true
    if (entry.personaText === null) {
      fields.systemMessage = entry.rawEntry?.fields?.systemMessage ?? null
    } else if (entry.personaText !== '') {
      fields.systemMessage = entry.personaText
    } else if (!entry.personaAbsent) {
      // The file said "use the configured prompt" (`null`) or held an empty
      // persona: the draft keeps saying it, in the same shape.
      fields.systemMessage = entry.personaNull ? null : ''
    }
    agents[entry.key] = fields
  }
  return { default: draft.default ?? '', agents }
}

// ---------------------------------------------------------------------------
// The two calls
// ---------------------------------------------------------------------------

// The dialog's own transport state — what a *component* renders, kept separate
// from the dropdown's so a failed dialog read can never disturb the composer
// (the same non-destructive rule `loadAgentModes` follows).
const editorState = reactive({
  open: false, // the dialog is open; a 409 keeps it open (D7)
  loading: false, // the document read is in flight
  saving: false, // a save is in flight
  error: null, // a transport-level failure, said out loud
  conflict: null, // D7's named conflict: { path, diskMtime, documentMtime }
  invalidFindings: [], // a server 422's findings, should the form ever miss one
})

function describeError(body, fallback) {
  const detail = body?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => item?.msg ?? 'invalid')
    if (messages.length) return messages.join(' · ')
  }
  return fallback
}

// The document read (`GET /api/agent-modes/document`). Failure leaves the
// dropdown exactly as it was — this call touches only its own state — and says
// so through `editorState.error`. On success it answers the editable draft.
export async function fetchAgentModesDocument() {
  editorState.loading = true
  editorState.error = null
  editorState.conflict = null
  editorState.invalidFindings = []
  try {
    const res = await fetch('/api/agent-modes/document')
    const body = await res.json().catch(() => null)
    if (!res.ok) {
      editorState.error = describeError(body, `Could not read the agent-modes file (${res.status}).`)
      return null
    }
    return createAgentModesDraft(body)
  } catch (exc) {
    editorState.error = `Could not reach the API: ${exc}`
    return null
  } finally {
    editorState.loading = false
  }
}

// D9's reset decision, as a plain rule: after a save, does the selection move
// to the newly set default? It moves — unless the current mode is **locked**
// (the modes lot's D7: a thread that entered the app-owned mode keeps it) or a
// lot is running (its mode is pinned at launch and a save cannot move it). When
// it is skipped, the returned sentence is what the quiet line under the select
// says, through the notice slot the composer already renders.
export function resetDecisionAfterSave({ batchActive = false } = {}) {
  if (batchActive) {
    return { move: false, reason: 'a running lot picks its mode at launch' }
  }
  if (state.locked) {
    return { move: false, reason: "this conversation's mode is fixed once it has a turn" }
  }
  return { move: true, reason: null }
}

// The whole-document save (`POST /api/agent-modes/save`), carrying the `mtime`
// token the read returned. On success the response — the new verdict in the
// dropdown's own shape — feeds the singleton through `acceptPayload`, so the
// list reloads **without a second round-trip** (D9), and the selection moves to
// the newly set default unless `resetDecisionAfterSave` says otherwise (a
// deleted-while-selected mode lands on the default too, which is what reseed
// does). On a 409 the draft is left untouched and the conflict is *named* —
// both `mtime` numbers — with the dialog staying open; nothing is merged and
// nothing is overwritten (D7, and reload-and-re-apply is out of this lot).
export async function saveAgentModesDocument(draft, { batchActive = false } = {}) {
  if (editorState.saving) return false
  editorState.saving = true
  editorState.error = null
  editorState.conflict = null
  editorState.invalidFindings = []
  try {
    const res = await fetch('/api/agent-modes/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document: saveBodyOf(draft), mtime: draft?.mtime ?? null }),
    })
    const body = await res.json().catch(() => null)
    if (!res.ok) {
      const detail = body?.detail
      if (res.status === 409 && detail && typeof detail === 'object') {
        editorState.conflict = {
          path: detail.path ?? '',
          diskMtime: detail.diskMtime ?? null,
          documentMtime: detail.documentMtime ?? null,
        }
      } else if (res.status === 422 && detail && typeof detail === 'object' && Array.isArray(detail.findings)) {
        // The form should have blocked this save; if it did not, the server's
        // findings are shown rather than swallowed.
        editorState.invalidFindings = detail.findings
      } else {
        editorState.error = describeError(body, `Could not save the agent-modes file (${res.status}).`)
      }
      return false
    }
    // The draft is now what the disk holds: a fresh token and a clean baseline.
    draft.mtime = body?.source?.mtime ?? null
    draft.baseline = draftSnapshot(draft)
    const decision = resetDecisionAfterSave({ batchActive })
    const kept = state.current
    try {
      acceptPayload(body, { reseed: true })
      if (!decision.move) {
        if (modeById(kept)) {
          // The lock and the lot pin survive the reload: the list changed, the
          // selection did not, and the quiet line says why.
          state.current = kept
          syncAgentModeLock(threadHasTurns())
          state.notice = `The agent-modes file was saved — the selection did not move: ${decision.reason}.`
        } else {
          // The saved document deleted the very mode the lock (or the lot) was
          // holding. There is nothing to keep, and pointing `current` at a mode
          // this workspace no longer has would be a 422 on the next turn; the
          // same out-loud fall-back `applySessionAgentMode` uses for an id this
          // workspace does not have.
          state.current = state.default
          syncAgentModeLock(threadHasTurns())
          state.notice = `The agent-modes file was saved, and it no longer declares the mode "${kept}" this conversation was using — using ${state.default ?? 'the workspace default'}.`
        }
      }
    } catch {
      // The bytes landed; only the dropdown's reload *of them* failed, and that
      // is the failure `loadAgentModes` already absorbs: the previous list
      // stands, `send()` omits the key so the server's default answers, and the
      // next refresh picks the new verdict up. Saying "could not save" here
      // would be a lie about a file that was saved.
      state.loaded = false
    }
    return true
  } catch (exc) {
    editorState.error = `Could not reach the API: ${exc}`
    return false
  } finally {
    editorState.saving = false
  }
}

export function useAgentModes() {
  return { state, editor: editorState, loadAgentModes, setAgentMode, currentMode, modeHint }
}
