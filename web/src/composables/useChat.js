import { nextTick, reactive } from 'vue'
import { renderMarkdown, TOOL_LINE_MARKER } from '../lib/markdown'
import { createReasoningBuffer } from '../lib/reasoning-buffer'
import {
  agentModePayloadId,
  applySessionAgentMode,
  clearAgentModeNotice,
  loadAgentModes,
  registerAgentModeTurnCounter,
  syncAgentModeLock,
} from './useAgentModes'

const state = reactive({
  messages: [], // { role, raw, html, streaming, images } in conversation order
  phase: 'idle', // idle | waiting | reasoning | answering
  error: null,
  // Composer draft mirrored as rendered markdown so WorkBenchView can preview the
  // exact HTML that send() will attach to the user message (no re-parse).
  draft: { raw: '', html: '', renderedRaw: '' },
  // Image picked for the next message: { dataUrl, name, size } | null.
  pendingImage: null,
  // Chat session clock (consumed by useSessionStats, rendered by the app bar's
  // SessionClock): ms timestamp of when the running session started
  // (0 = idle, so the clock reads 00:00:00).
  // Covers the whole request — waiting, thinking and every MCP/LSP tool
  // round — and is cleared when the reply ends, is stopped or errors out.
  session: { startedAt: 0 },
  // The reasoning panel's Live View (toggled in WorkBenchFocusGutter.vue): off
  // keeps the whole thinking history on screen, which is the default; on lets
  // the panel paint only what its box can hold. Panel render state only — the
  // flows stay in `block.text`, so a save or a copy never loses bytes.
  liveView: false,
  // A pause request is armed but its `pause_ack` has not landed yet: the ⏸
  // button blinks and is disabled, and the composer stays shut (the ack is
  // what reopens it). Cleared by the ack, by finish(), and by reset().
  pausePending: false,
  // Conversation-scoped MCP toggles (independent from the SettingsDialog):
  // the Settings-activated connectors snapshotted on tab entry, each ON by
  // default, frozen once the first message is accepted by the server.
  mcp: {
    connectors: [], // { id, label, status } — only connectors active in Settings
    enabled: {}, // id -> boolean (this conversation only, never persisted)
    loaded: false, // GET /api/mcp snapshot succeeded (panel + filter usable)
    locked: false, // sticky: first accepted send locks the toggles
    restarting: false, // POST /api/mcp/restart in flight (button stays busy)
  },
  // Conversation-scoped LSP toggles — the exact MCP slice pattern, fed by
  // GET /api/lsp instead. Shared `locked` flag semantics: the first accepted
  // send freezes both panels until restart resets them.
  lsp: {
    servers: [], // { id, label, status } — only language servers active in Settings
    enabled: {}, // id -> boolean (this conversation only, never persisted)
    loaded: false, // GET /api/lsp snapshot succeeded (panel + filter usable)
    locked: false, // sticky: first accepted send locks the toggles (as mcp)
    restarting: false, // POST /api/lsp/restart in flight (button stays busy)
  },
})

let controller = null
let renderScheduled = false
let activeMessage = null
let draftTimer = null
// Streamed thinking, coalesced into the panel timeline once per frame (see
// lib/reasoning-buffer). Module-level: one reply streams at a time.
const reasoningBuffer = createReasoningBuffer()
// Explicit Stop discards the partial reply; connection drops keep it.
let userStopped = false
// Pause's local fallback (when POST /api/chat/pause itself is unreachable)
// also aborts the fetch, but must NOT discard: the whole point of Pause is
// that the partial turn stays in the conversation. Distinguishes that abort
// from Stop's.
let userPaused = false

// Must mirror the backend ChatImageUrl validation (routes.py).
export const IMAGE_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif']
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024

export function humanFileSize(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`
  return `${bytes} B`
}

function clearDraft() {
  state.draft.raw = ''
  state.draft.html = ''
  state.draft.renderedRaw = ''
}

function setDraft(text) {
  state.draft.raw = text
  if (draftTimer) clearTimeout(draftTimer)
  draftTimer = setTimeout(() => {
    draftTimer = null
    const trimmed = state.draft.raw.trim()
    state.draft.renderedRaw = trimmed
    state.draft.html = trimmed ? renderMarkdown(trimmed) : ''
  }, 200)
}

// Composer textarea, registered by WorkBenchInput so draft insertions (memory
// bullets) can focus it after appending.
let draftInputEl = null

function registerDraftInput(el) {
  draftInputEl = el ?? null
}

// Appends a memory bullet's text to the draft (never overwrites): joined to
// the existing draft with a blank line when there is one. Goes through
// setDraft, so the markdown preview updates on the usual debounce.
function appendDraft(text) {
  const value = text ?? ''
  if (!value) return
  const existing = state.draft.raw
  setDraft(existing.trim() ? `${existing}\n\n${value}` : value)
  nextTick(() => {
    if (draftInputEl && document.activeElement !== draftInputEl) {
      draftInputEl.focus()
    }
  })
}

function flushDraftHtml(trimmed) {
  if (draftTimer) {
    clearTimeout(draftTimer)
    draftTimer = null
  }
  if (state.draft.renderedRaw === trimmed && state.draft.html) {
    return state.draft.html
  }
  return renderMarkdown(trimmed)
}

// Validates and reads a picked file into state.pendingImage (replacing any
// previous attachment). Returns false (with state.error set) when rejected.
function attachImage(file) {
  if (!file) return false
  state.error = null
  if (!IMAGE_MIME_TYPES.includes(file.type)) {
    state.error = `Unsupported image type: ${file.type || file.name} (png, jpeg, webp or gif).`
    return false
  }
  if (file.size > MAX_IMAGE_BYTES) {
    state.error = `Image too large (${humanFileSize(file.size)}, max ${humanFileSize(MAX_IMAGE_BYTES)}).`
    return false
  }
  const reader = new FileReader()
  reader.onload = () => {
    state.pendingImage = {
      dataUrl: String(reader.result),
      name: file.name || 'image',
      size: file.size,
    }
  }
  reader.onerror = () => {
    state.error = 'Could not read the selected image.'
  }
  reader.readAsDataURL(file)
  return true
}

function clearImage() {
  state.pendingImage = null
}

// Everything a conversation needs freshly snapshotted: the agent-mode list and
// both hub allowlists. The three are per-workspace and per-settings, so every
// entry point into a conversation (boot, the WorkBench tab's watch, a restart)
// calls this one function instead of picking a subset — which is what keeps the
// mode dropdown from being the one thing that stayed stale.
export async function loadChatHubs() {
  loadAgentModes()
  await Promise.all([loadChatMcp(), loadChatLsp()])
}

// Snapshot the Settings-activated connectors for a new conversation. Never
// touches a locked (already-started) conversation; on failure the panel
// stays hidden and send() omits the mcpEnabled filter, so chat keeps
// working with every activated connector — a failed load must not silently
// disable connectors the user never got to toggle. Toggle choices survive
// the refresh (restart, tab re-entry and MCP settings saves): previously
// switched-off connectors come back off, freshly appearing ones default to
// ON. Exported for useBenchmark's saveMcp() so saving the MCP settings
// refreshes the panel right away (no cycle: useChat imports useBenchmark
// nowhere).
export async function loadChatMcp() {
  if (state.mcp.locked) return
  try {
    const res = await fetch('/api/mcp')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const body = await res.json()
    state.mcp.connectors = (body.connectors || [])
      .filter((connector) => connector.active)
      .map(({ id, label, status }) => ({ id, label, status }))
    const previous = state.mcp.enabled
    const enabled = {}
    for (const connector of state.mcp.connectors) {
      enabled[connector.id] =
        connector.id in previous ? previous[connector.id] : true
    }
    state.mcp.enabled = enabled
    state.mcp.loaded = true
  } catch {
    state.mcp.connectors = []
    state.mcp.enabled = {}
    state.mcp.loaded = false
  }
}

// A running batch is the one caller-owned reason a conversation-scoped hub
// change must be refused while `phase` is momentarily 'idle' (the window
// between two task turns). The flag itself belongs to useBatch — this hook is
// how it reaches the guards below without useChat importing useBatch: the
// dependency direction stays acyclic (useBatch -> useChat, never the reverse).
let busyGuard = null

export function registerChatBusyGuard(isBusy) {
  busyGuard = typeof isBusy === 'function' ? isBusy : null
}

// True when a reply is streaming, or a batch owns the conversation.
function chatBusy() {
  return state.phase !== 'idle' || !!busyGuard?.()
}

// The Live View's switch (§3.2), owned here rather than by whichever component
// draws the control: the flag is conversation state and the lock is this
// composable's own busy rule, so a mode can never be switched mid-reply from a
// second place that reads a different streaming flag.
function toggleLiveView() {
  if (chatBusy()) return
  state.liveView = !state.liveView
}

// How many turns this conversation already holds — the fact the agent-mode lock
// (D7) is a rule about. Injected into useAgentModes rather than imported: the
// dependency direction stays acyclic (useChat -> useAgentModes, never back).
registerAgentModeTurnCounter(() => state.messages.length > 0)

function setChatMcp(id, on) {
  if (state.mcp.locked || chatBusy()) return
  if (!(id in state.mcp.enabled)) return
  state.mcp.enabled[id] = !!on
}

// Conversation-scoped twin of loadChatMcp, for the LSP language servers
// (GET /api/lsp). Same rules: never touches a locked conversation, a failed
// load hides the panel and send() omits the lspEnabled filter (chat keeps
// working with every Settings-activated server), and previous on/off
// choices survive the refresh. Exported for useBenchmark's saveLsp().
export async function loadChatLsp() {
  if (state.lsp.locked) return
  try {
    const res = await fetch('/api/lsp')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const body = await res.json()
    state.lsp.servers = (body.servers || [])
      .filter((server) => server.active)
      .map(({ id, label, status }) => ({ id, label, status }))
    const previous = state.lsp.enabled
    const enabled = {}
    for (const server of state.lsp.servers) {
      enabled[server.id] = server.id in previous ? previous[server.id] : true
    }
    state.lsp.enabled = enabled
    state.lsp.loaded = true
  } catch {
    state.lsp.servers = []
    state.lsp.enabled = {}
    state.lsp.loaded = false
  }
}

function setChatLsp(id, on) {
  if (state.lsp.locked || chatBusy()) return
  if (!(id in state.lsp.enabled)) return
  state.lsp.enabled[id] = !!on
}

// Restart a hub: tear every session down and rebuild it from a fresh scan of
// its declaration file, so a hand-edited ``externals.json``/``lsp_servers.json``
// (new entry, changed command, another ``workspaceRoot``) takes effect without
// restarting the web app. Awaited by the server, so on return the statuses in
// the refreshed snapshot are real. Errors never throw: they land in
// ``state.error`` (the chat error line) and, for a failing entry, in its own
// status — the panel still refreshes so the user sees which one broke.
// The rows a panel currently shows (`connectors` for MCP, `servers` for LSP),
// refreshed in place from a hub payload of the same shape the GET endpoints
// return.
function applyHubStatuses(kind, body) {
  const shown = kind === 'mcp' ? state.mcp.connectors : state.lsp.servers
  const fresh = kind === 'mcp' ? body?.connectors : body?.servers
  if (!Array.isArray(fresh)) return
  const byId = Object.fromEntries(fresh.map((row) => [row.id, row]))
  for (const row of shown) {
    const update = byId[row.id]
    if (update) row.status = update.status
  }
}

async function restartHub(kind) {
  const hub = state[kind]
  if (hub.restarting || chatBusy()) return
  hub.restarting = true
  try {
    const res = await fetch(`/api/${kind}/restart`, { method: 'POST' })
    const body = await res.json().catch(() => null)
    if (!res.ok) {
      state.error = body?.detail || `Restart failed (HTTP ${res.status})`
    } else {
      state.error = null
      // A started conversation has a locked (frozen) toggle set, so the
      // loaders below would skip it and leave the status dots stale — and
      // watching those dots turn green is the point of the button. The
      // restart response already carries live statuses, so paint them.
      applyHubStatuses(kind, body)
    }
  } catch {
    state.error = 'Restart failed: the server did not answer'
  } finally {
    hub.restarting = false
    // Re-snapshot either way: a refused restart leaves the hub untouched but
    // still lets the panel refresh, and a successful one shows the new ids
    // and statuses. A locked conversation keeps its frozen toggles.
    if (kind === 'mcp') loadChatMcp()
    else loadChatLsp()
  }
}

function scheduleMarkdownRender() {
  if (renderScheduled) return
  renderScheduled = true
  requestAnimationFrame(() => {
    renderScheduled = false
    if (activeMessage)
      activeMessage.html = renderMarkdown(activeMessage.raw, { mermaid: true })
  })
}

// Client-side mirror of the backend's estimate_completion_tokens() heuristic
// (~4 chars/token): feeds the live tok/s badge and the abort fallbacks.
function estimateTokens(count) {
  return Math.ceil(Math.max(count, 0) / 4)
}

// Characters this reply generated and the client can actually see: answer text
// (`raw`, minus the tool-line sentinels it only carries for rendering) plus
// streamed thinking. Tool-call arguments never reach the client, so they are
// only ever counted through the server's exact `usage` figures.
function generatedChars(message) {
  const answer = (message.raw ?? '').split(TOOL_LINE_MARKER).join('').length
  return answer + (message.reasoningChars ?? 0)
}

// Generated text produced AFTER the last `usage` baseline: what the live
// badges and the abort fallback must still add on top of the exact figures.
function estimateGeneratedTail(message) {
  const generated = generatedChars(message) - (message.usageGeneratedChars || 0)
  return estimateTokens(generated)
}

// Every generated chunk (answer text or thinking delta) opens/extends the
// generation window, mirroring the backend (see stream_chat): thinking deltas
// are generated tokens too, so a reasoning phase is neither free to the
// numerator nor to the tok/s denominator — which is what keeps the speed badge
// ticking while the model thinks instead of freezing it on the previous reply.
function noteGenerated(assistant) {
  const now = Date.now()
  if (!assistant.windowStartAt) assistant.windowStartAt = now
  assistant.lastTokenAt = now
}

// Closes the open generation window so what follows it (tool execution, the
// re-prefill of the next round) stays out of the measured seconds. Called on
// `tool_call` / `usage` and when the reply ends; the next generated chunk
// opens a fresh window.
function closeGenerationWindow(assistant) {
  if (!assistant.windowStartAt) return
  assistant.generatedSeconds +=
    Math.max((assistant.lastTokenAt || 0) - assistant.windowStartAt, 0) / 1000
  assistant.windowStartAt = 0
}

// Seconds of generation measured client-side: every closed window plus the one
// still open, ended at the last chunk seen — so an idle gap inside the stream
// never inflates the live denominator.
function measuredSeconds(message) {
  const open = message.windowStartAt
    ? Math.max((message.lastTokenAt || 0) - message.windowStartAt, 0) / 1000
    : 0
  return (message.generatedSeconds || 0) + open
}

// Live tok/s of a streaming reply, as `{ value, estimated }` or null.
// Preferred basis: the server's cumulative exact figures from the last `usage`
// frame (completion tokens — thinking and tool arguments included — over the
// summed decode seconds), extended by whatever arrived since: those tokens are
// char-estimated (flagging the result), and their seconds come from the
// client-measured generation windows, so the tool loop's dead time stays out
// of the rate. With no `usage` in the payload at all the whole reply is
// measured client-side — answer text AND thinking over their own windows.
export function liveTokenRate(message) {
  if (!message) return null
  const seconds = measuredSeconds(message)
  if (message.usageTokens > 0 && message.usageDecode > 0) {
    const tail = estimateGeneratedTail(message)
    const tailSeconds = Math.max(seconds - (message.usageGeneratedSeconds || 0), 0)
    const secs = message.usageDecode + tailSeconds
    if (secs <= 0) return null
    return {
      value: (message.usageTokens + tail) / secs,
      estimated: tail > 0,
    }
  }
  const tokens = estimateTokens(generatedChars(message))
  if (seconds <= 0 || tokens <= 0) return null
  return { value: tokens / seconds, estimated: true }
}

export function conversationMarkdown(messages) {
  return messages
    .map((m) => {
      const parts = []
      if (m.raw.trim()) parts.push(m.raw)
      for (const image of m.images ?? []) parts.push(`*[image: ${image.name}]*`)
      return parts.length ? `## ${m.role}\n\n${parts.join('\n\n')}` : null
    })
    .filter(Boolean)
    .join('\n\n')
}

// OpenAI content shape for one message: plain string when text-only (the
// backend and litellm keep the classic form), multimodal parts once an
// image is attached (image-only messages omit the text part entirely).
function messagePayload(message) {
  const images = message.images ?? []
  // The rendering-only tool-line sentinel never reaches the model.
  let raw = message.raw.split(TOOL_LINE_MARKER).join('')
  // A turn the user interrupted must not read to the model as a finished
  // answer: the note is added here (payload-build time) so the bubble the user
  // reads stays clean while the model sees where the fragment came from. The
  // same serializer writes saved sessions, so a loaded fragment keeps the note
  // — what does not survive a reload is the `paused` flag and its caption.
  if (message.paused) raw += '\n\n[interrupted by the user]'
  if (!images.length) return { role: message.role, content: raw }
  const parts = []
  if (raw.trim()) parts.push({ type: 'text', text: raw })
  for (const image of images) {
    parts.push({ type: 'image_url', image_url: { url: image.dataUrl } })
  }
  return { role: message.role, content: parts }
}

// ---------------------------------------------------------------------------
// Save / load a session as a JSON file (frontend only).
//
// The file is a thin envelope around the OpenAI user/assistant message list —
// exactly the shape POST /api/chat accepts — so continuing a loaded
// conversation is free: every send already re-posts state.messages mapped
// through messagePayload(). A user turn with attached images keeps them as
// standard OpenAI `image_url` parts, so the base64 data URL round-trips and the
// next send re-posts it to the model. Tool-loop turns and the per-turn badges
// are still never written (the browser only holds _preview-capped tool
// summaries, so writing them back would be fabrication; the system prompt is
// rebuilt server-side per send).
//
// Each image part carries one benchmarkit-only extra key, `name` (the original
// file name, needed for the restored thumbnail's alt/title): the backend's
// `ChatContentPart` uses extra="ignore", so that key is dropped on the way to
// the model.
//
// The one extra field an assistant turn may carry is `reasoning` — the streamed
// thinking flows of that reply, one entry per reasoning block of the
// WorkBenchToolPanel timeline (a tool_call frame splits one flow from the next
// round's). It lives beside `content`, never inside it: thinking is not
// conversation text, and putting it in `content` would change what the model
// sees on the next send. It is panel render state only — send() builds its
// history through messagePayload(), which reads `raw`/`images` alone, so a
// restored flow can never reach the model.
// ---------------------------------------------------------------------------

export const SESSION_FILE_FORMAT = 'benchmarkit-chat-session'
// v2 added the optional per-assistant-turn `reasoning` list, v3 started saving
// the attached images instead of stripping them, v4 records the conversation's
// agent mode in the envelope. Older files simply carry no image parts and no
// `agentMode`, so the loader needs no version branch either way.
export const SESSION_FILE_VERSION = 4
// Stands in for a loaded image part whose data URL is unusable (a hand-edited
// or truncated file), so an image-only turn keeps a non-empty content (an
// empty one would 422 the next send).
export const SESSION_IMAGE_PLACEHOLDER =
  '[image from the saved session could not be restored]'
// Must mirror the backend ChatImageUrl validation (routes.py).
const SESSION_IMAGE_DATA_URL =
  /^data:image\/(png|jpeg|webp|gif);base64,([A-Za-z0-9+/]+={0,2})$/

// Thinking flows of one assistant reply, in panel order: every
// `{ kind: 'reasoning' }` block of its timeline as one string, blank ones
// dropped. Kept as a list (not joined) so the panel timeline can be rebuilt
// with its original block boundaries — and so a v2 file stays readable by hand.
function reasoningFlows(message) {
  return (message.blocks ?? [])
    .filter((block) => block.kind === 'reasoning')
    .map((block) => block.text ?? '')
    .filter((text) => text.trim())
}

export function conversationJson(messages) {
  const items = []
  for (const message of messages) {
    // The very serializer a live send uses, so the file holds exactly what the
    // model was shown: text parts plus base64 `image_url` parts for an attached
    // image. Each image part then gets its file `name` (see the block header),
    // which the backend drops on its way to the model.
    const payload = messagePayload(message)
    const parts = typeof payload.content === 'string' ? [] : payload.content
    const hasText =
      typeof payload.content === 'string'
        ? Boolean(payload.content.trim())
        : parts.some((part) => part.type === 'text' && part.text.trim())
    const imageParts = parts.filter((part) => part.type === 'image_url')
    // A turn with neither text nor an image carries nothing worth restoring.
    if (!hasText && !imageParts.length) continue
    const images = message.images ?? []
    imageParts.forEach((part, i) => {
      if (images[i]) part.name = images[i].name || 'image'
    })
    if (message.role === 'assistant') {
      // Omitted entirely when the reply streamed no thinking, so files stay as
      // small as v1 ones for non-reasoning models.
      const flows = reasoningFlows(message)
      if (flows.length) payload.reasoning = flows
    }
    items.push(payload)
  }
  // No `stats` block on purpose: tok/s and ctx badges are client-only render
  // state, and the loader stays tolerant of a future version adding one back.
  //
  // `agentMode` is the one piece of *conversation* state the envelope carries
  // (am-07): the mode is not a message and never belongs in the list, but a
  // saved thread was written by one persona, and re-opening it in another would
  // change what its next turn means. Omitted when the mode list is not loaded
  // (or never was): a file without the key is a v3 file's shape, and the loader
  // then leaves the workspace's default alone.
  const agentMode = agentModePayloadId()
  return {
    format: SESSION_FILE_FORMAT,
    version: SESSION_FILE_VERSION,
    ...(agentMode ? { agentMode } : {}),
    messages: items,
  }
}

// One loaded image part -> the client image shape `{ dataUrl, name, size }`, or
// null when the part carries no usable data URL: the caller then inserts the
// placeholder text instead, so the turn stays postable. A well-formed but
// oversized image fails loudly instead — the backend would 422 the very same
// payload on the next send, a far more confusing place to hit the error.
function normalizeLoadedImage(part, index) {
  const url = part?.image_url?.url
  if (typeof url !== 'string') return null
  const match = SESSION_IMAGE_DATA_URL.exec(url)
  if (!match) return null
  let base64
  try {
    base64 = atob(match[2])
  } catch {
    return null
  }
  // atob yields exactly one character per decoded byte, so its length *is* the
  // decoded size — the same bound the backend checks with a real b64decode.
  const bytes = base64.length
  if (bytes > MAX_IMAGE_BYTES) {
    throw new Error(
      `Message ${index + 1} carries a ${humanFileSize(bytes)} image (max ${humanFileSize(MAX_IMAGE_BYTES)}).`,
    )
  }
  const name =
    typeof part.name === 'string' && part.name.trim() ? part.name.trim() : 'image'
  return { dataUrl: url, name, size: bytes }
}

// Content of one loaded turn, split into where the client keeps each part:
// `{ text, images }` — text parts joined into `raw`, image parts restored as
// `images` entries (an unusable one becomes the placeholder text). Image parts
// are only ever restored onto user turns: save() never writes them elsewhere,
// and messagePayload() would turn them into a multimodal `content` the backend
// rejects for assistant turns ("assistant messages must use plain text
// content"), so a hand-written file gets the placeholder instead.
function normalizeLoadedContent(content, role, index) {
  if (typeof content === 'string') return { text: content, images: [] }
  if (!Array.isArray(content)) return { text: '', images: [] }
  const texts = []
  const images = []
  for (const part of content) {
    if (part?.type === 'image_url') {
      const image = normalizeLoadedImage(part, index)
      if (!image || role === 'assistant') texts.push(SESSION_IMAGE_PLACEHOLDER)
      else images.push(image)
    } else if (typeof part?.text === 'string') {
      texts.push(part.text)
    }
  }
  return { text: texts.join('\n'), images }
}

// Optional per-assistant-turn thinking flows of a loaded turn: a list of
// strings (v2) — a lone string is accepted too, for hand-written files.
// Returns [] when the key is absent (every v1 file) or empty; throws when the
// key holds anything else, so a malformed file fails loudly instead of
// restoring half a timeline.
function normalizeLoadedReasoning(reasoning, index) {
  if (reasoning === undefined || reasoning === null) return []
  const flows = typeof reasoning === 'string' ? [reasoning] : reasoning
  if (!Array.isArray(flows)) {
    throw new Error(
      `Message ${index + 1} has a "reasoning" field that is neither a string nor a list of strings.`,
    )
  }
  const texts = []
  for (const flow of flows) {
    if (typeof flow !== 'string') {
      throw new Error(
        `Message ${index + 1} has a "reasoning" entry that is not a string.`,
      )
    }
    if (flow.trim()) texts.push(flow)
  }
  return texts
}

// Validates a session file's text and returns its turns as
// `{ messages: [{ role, content, images, reasoning }], agentMode }` (`images`
// being the restored `{ dataUrl, name, size }` entries of that turn, and
// `agentMode` the mode id the envelope carried — null for a v3 file, a bare
// array, or an envelope without that key). Throws with a precise
// message: silently dropping a system/tool turn would change the
// conversation's meaning, so an invalid file must fail loudly. Unknown extra
// keys are ignored, and a bare `[{role, content}, ...]` array is accepted for
// hand-written files.
export function parseSessionFile(text) {
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('Not a valid JSON file.')
  }
  let items = null
  let agentMode = null
  if (Array.isArray(parsed)) {
    items = parsed
  } else if (parsed && typeof parsed === 'object') {
    if (parsed.format && parsed.format !== SESSION_FILE_FORMAT) {
      throw new Error(
        `Unknown session format "${parsed.format}" (expected "${SESSION_FILE_FORMAT}").`,
      )
    }
    if (!Array.isArray(parsed.messages)) {
      throw new Error('Not a chat session file: no "messages" list found.')
    }
    items = parsed.messages
    // A mode id of the wrong shape is tolerated rather than fatal: this key is
    // conversation state the parser has no opinion about, and the rule that
    // judges it ("is this a mode the *current* workspace offers?") belongs to
    // useAgentModes, where a v3 file with no key keeps the workspace default.
    if (typeof parsed.agentMode === 'string' && parsed.agentMode.trim()) {
      agentMode = parsed.agentMode.trim()
    }
  } else {
    throw new Error('Not a chat session file: expected a JSON object or array.')
  }

  const messages = []
  for (const [index, item] of items.entries()) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new Error(`Message ${index + 1} is not an object.`)
    }
    if (item.tool_calls) {
      throw new Error(
        `Message ${index + 1} carries tool_calls: saved sessions store conversation text only.`,
      )
    }
    if (item.role !== 'user' && item.role !== 'assistant') {
      throw new Error(
        `Message ${index + 1} has role ${JSON.stringify(item.role ?? '')}: only "user" and "assistant" turns can be loaded.`,
      )
    }
    const { text: content, images } = normalizeLoadedContent(item.content, item.role, index)
    // A turn is usable when it has text or a restorable image.
    if (!content.trim() && !images.length) continue
    // A user turn never carries thinking: the key is ignored there (unknown
    // extra keys are tolerated by design).
    const reasoning =
      item.role === 'assistant' ? normalizeLoadedReasoning(item.reasoning, index) : []
    messages.push({ role: item.role, content, images, reasoning })
  }
  if (!messages.length) throw new Error('Session file holds no usable message.')
  return { messages, agentMode }
}

function describeError(body, fallback) {
  const detail = body?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      const field = Array.isArray(item?.loc)
        ? item.loc.filter((part) => part !== 'body').join('.')
        : ''
      return field ? `${field}: ${item?.msg ?? 'invalid'}` : item?.msg ?? 'invalid'
    })
    if (messages.length) return messages.join(' · ')
  }
  return fallback
}

function applyEvent(name, data, assistant) {
  switch (name) {
    case 'token': {
      if (state.phase !== 'answering') state.phase = 'answering'
      // Tool-loop continuation: the first token after a tool_result opens a
      // fresh line prefixed with the tool icon (sentinel replaced at render
      // time by lib/markdown.js, after a blank line) so each post-tool
      // message starts on its own line in WorkBenchView instead of running onto
      // the previous round's text. Flag consumed once per tool round (the
      // round's later tokens keep flowing on that icon line).
      if (assistant.pendingStreamQuote) {
        assistant.pendingStreamQuote = false
        assistant.raw += assistant.raw ? `\n\n${TOOL_LINE_MARKER}` : TOOL_LINE_MARKER
      }
      assistant.raw += data.text ?? ''
      noteGenerated(assistant)
      scheduleMarkdownRender()
      break
    }
    case 'done':
      assistant.tokens = data.tokens ?? 0
      assistant.took = data.took ?? 0
      assistant.ttft = data.ttft ?? 0
      // Cumulative decode seconds across all tool-loop rounds: the accurate
      // tok/s denominator once tools ran (took - ttft would include tool
      // execution and inter-round gaps). 0 on payloads from an old server.
      assistant.decode = data.decode ?? 0
      assistant.tokensEstimated = !!data.tokensEstimated
      // Context occupancy of the FINAL round: the figure to compare against
      // the server window: it never accumulates across rounds.
      if (typeof data.contextTokens === 'number' && data.contextTokens > 0) {
        assistant.contextTokens = data.contextTokens
        assistant.contextEstimated = !!data.contextEstimated
      }
      break
    case 'pause_ack':
      // A late ack (the reply had already terminated on `done`/`error`, or
      // Restart wiped the conversation while the pause was in flight): the turn
      // is closed, so this frame must not re-open it.
      if (!assistant.streaming) break
      // Terminal frame of a user-interrupted reply. Its exact figures arrived
      // as the `usage` frame just before it (the tool-loop sites reuse the
      // pre-tool `usage`), so all that is left is the measurement bookkeeping:
      // freeze the window, flag the fragment (caption + the payload note that
      // tells the model this turn is unfinished) and reopen the composer —
      // which is what hides Stop, shows Send and takes ⏸ out of the row.
      closeGenerationWindow(assistant)
      // The `usage` frame that precedes the ack holds the cumulative exact
      // figures for everything generated (the round is never reported by any
      // other frame), so they become the message's final numbers: finish()'s
      // char-based fallback would otherwise flag a server-reported count as
      // estimated, and the tok/s badge needs `decode` to stay exact.
      if (assistant.usageTokens > 0) {
        assistant.tokens = assistant.usageTokens
        assistant.tokensEstimated = false
        if (assistant.usageDecode > 0) assistant.decode = assistant.usageDecode
      }
      assistant.paused = true
      // Paused before anything was generated (still prefilling): an empty
      // assistant turn would sit in the thread and be re-posted, so it goes —
      // the same guard the non-OK response path uses.
      if (!assistant.raw) {
        state.messages.splice(state.messages.indexOf(assistant), 1)
      }
      state.pausePending = false
      state.phase = 'idle'
      break
    case 'tool_call':
      assistant.tools.push({
        id: data.id ?? `${data.tool}-${assistant.tools.length}`,
        tool: data.tool ?? 'tool',
        server: data.server ?? '',
        preview: data.argsPreview ?? '',
        state: 'running',
      })
      // Same reactive object in the panel timeline, at its chronological
      // slot: reasoning text flows around the badges, tool_result state
      // flips still re-render the badge in place.
      // The round's generation is over: close the client-side window so the
      // tool execution (which can take tens of seconds) never enters the
      // live tok/s denominator.
      closeGenerationWindow(assistant)
      // Draw buffered thinking before the badge: a delta still in the buffer
      // arrived before this call, so it must keep its earlier slot in the
      // timeline. One flush per tool round, not per chunk.
      reasoningBuffer.flush()
      assistant.blocks.push({
        kind: 'tool',
        tool: assistant.tools[assistant.tools.length - 1],
      })
      // Server-authoritative tool-loop position for the iteration gauge.
      if (data.iteration) assistant.mcpIteration = data.iteration
      if (data.maxIterations) assistant.mcpMax = data.maxIterations
      break
    case 'usage':
      // Cumulative completion tokens (tool-call arguments included) and
      // decode seconds as soon as a tool round has finished generating:
      // keeps the side-panel badges live while the MCP tool loop is still
      // running. usageRawLen snapshots the answer text already streamed so
      // later estimates only count the characters generated after this
      // exact baseline (never double-counted, never missed).
      assistant.usageTokens = data.tokens ?? assistant.usageTokens
      assistant.usageDecode = data.decode ?? assistant.usageDecode
      // Close the window opened by this round's chunks before snapshotting, so
      // both baselines match the server's exact figures: what arrives after
      // this frame is the only thing later estimates add on top (answer text
      // AND thinking — never double-counted, never missed).
      closeGenerationWindow(assistant)
      assistant.usageGeneratedChars = generatedChars(assistant)
      assistant.usageGeneratedSeconds = measuredSeconds(assistant)
      // Occupancy of the round that just ended: the tool loop refreshes the
      // context badge between rounds, right when generation stops.
      if (typeof data.contextTokens === 'number' && data.contextTokens > 0) {
        assistant.contextTokens = data.contextTokens
        assistant.contextEstimated = !!data.contextEstimated
      }
      break
    case 'tool_result': {
      const tool = assistant.tools.find((item) => item.id === data.id)
      if (tool) {
        tool.state = data.ok ? 'ok' : 'error'
        tool.resultPreview = data.preview ?? ''
      }
      // Arm the tool-icon line prefix: the next round's first streamed token
      // starts on a fresh line in WorkBenchView (see the 'token' case).
      assistant.pendingStreamQuote = true
      break
    }
    case 'reasoning':
      // Never downgrade to reasoning once answer text started.
      if (state.phase === 'waiting' || state.phase === 'reasoning') {
        state.phase = 'reasoning'
      }
      break
    case 'reasoning_delta': {
      // Thinking text, streamed live. The text is buffered and lands in the
      // panel timeline once per animation frame (see lib/reasoning-buffer):
      // chunks arrive at a network-set rate, and each write into the reactive
      // block re-cuts the panel's rows and re-runs its autoscroll.
      const text = data.text ?? ''
      if (!text) break
      reasoningBuffer.append(assistant.blocks, text)
      // Thinking is generated text the client DOES see: it feeds the live
      // tok/s numerator and keeps the measurement window open, so a long
      // reasoning phase ticks the speed badge instead of freezing it.
      // This counter stays synchronous on purpose — a `usage` frame snapshots
      // an exact baseline out of it, so deferring it would count the buffered
      // tail twice.
      assistant.reasoningChars += text.length
      noteGenerated(assistant)
      break
    }
    case 'error':
      state.error = data.message || 'Chat request failed.'
      // A server-cut reply keeps its partial text in the conversation (unlike
      // an explicit Stop, which discards it) and that fragment is re-posted on
      // the next send — so the message itself carries the marker, not just the
      // transient error line.
      if (data.truncated) assistant.truncated = true
      break
    default:
      break
  }
}

function processFrame(frame, assistant) {
  let name = null
  let data = null
  for (const line of frame.split('\n')) {
    if (line.startsWith('event: ')) name = line.slice(7).trim()
    else if (line.startsWith('data: ')) {
      try {
        data = JSON.parse(line.slice(6))
      } catch {
        /* skip malformed frame data */
      }
    }
  }
  if (name && data) applyEvent(name, data, assistant)
}

function finish(assistant) {
  // Nothing may read a timeline that is still missing its last chunks: session
  // saving serializes these blocks, and it does not wait for a frame.
  reasoningBuffer.flush()
  if (activeMessage)
    activeMessage.html = renderMarkdown(activeMessage.raw, { mermaid: true })
  activeMessage = null
  // Stream ended without an exact usage report (stop/abort/error): keep the
  // conversation totals stable — exact cumulative tokens from the last
  // `usage` event (covers every completed tool round, reasoning and tool
  // arguments included) plus the ~4 chars/token heuristic on the answer
  // text AND thinking streamed since then, flagged as estimated.
  if (!assistant.tokens) {
    const carried = assistant.usageTokens || 0
    const tail = estimateGeneratedTail(assistant)
    if (carried + tail > 0) {
      assistant.tokens = carried + tail
      assistant.tokensEstimated = true
    }
  }
  // Freeze the measurement window: a finished reply's live rate is never
  // recomputed, and the badge falls back to the server's exact figures.
  closeGenerationWindow(assistant)
  // Stop/abort mid tool-execution: no tool_result will ever arrive, so never
  // leave a badge stuck on the pulsing "running" state in the tool panel.
  for (const tool of assistant.tools ?? []) {
    if (tool.state === 'running') tool.state = 'error'
  }
  assistant.streaming = false
  state.phase = 'idle'
  state.session.startedAt = 0
  controller = null
  // A pause whose `pause_ack` never arrives (the server died mid-round, or the
  // fallback abort ended the stream): the reply is over either way, so the ⏸
  // must stop blinking. The per-message `paused` flag is left alone — it only
  // drives a caption.
  state.pausePending = false
}

async function readStream(response, assistant) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      if (frame.trim()) processFrame(frame, assistant)
      boundary = buffer.indexOf('\n\n')
    }
  }
}

// One streamed turn, end to end: push the two bubbles, open the SSE request
// and run finish() in its finally. Both entry points call it — `send()` for an
// interactive message, `sendBatchTurn()` for one batch task — because the
// pause/truncation/measurement logic below has already been the source of bugs
// once and must never be forked.
//
// `history` is the message list posted to the server and is the ONLY thing
// that varies between the two: an interactive send sends the whole accumulated
// thread, a batch task sends its one instruction (B6 — the LLM context is reset
// per task, while the displayed transcript keeps accumulating). The client
// never sends a system message: `core/chat.py` stays the only place one is
// created.
// `agentMode` (the last entry): the mode **this turn** runs in. An interactive
// send passes the composer's selection; a batch task turn passes the runner's
// pin (D13) — a lot is one job executed under one set of rules, and the
// dropdown the user may be touching meanwhile is not part of it. `undefined`
// means "ask the server for its default" (the pre-feature shape), which is also
// what `useBatch` passes when its pin could not be resolved.
async function startTurn({
  userMessage,
  history,
  batchContext = null,
  agentMode = undefined,
}) {
  userStopped = false
  // Fresh turn: a previous turn's pause must not survive into it (its
  // fragment already carries its own `paused` flag, which is per message).
  userPaused = false
  state.error = null
  state.messages.push(userMessage)
  state.messages.push({
    role: 'assistant',
    raw: '',
    html: '',
    streaming: true,
    tools: [], // MCP tool calls: { id, tool, server, preview, state, resultPreview }
    blocks: [], // panel timeline: { kind: 'reasoning', text } | { kind: 'tool', tool }
    // true between a tool_result and the next round's first answer token:
    // the token case then opens a tool-icon line in WorkBenchView.
    pendingStreamQuote: false,
    tokens: 0,
    took: 0,
    ttft: 0,
    tokensEstimated: false,
    // Client-side generation timing (see noteGenerated/measuredSeconds):
    // summed seconds of the closed windows, plus the open window's bounds.
    generatedSeconds: 0,
    windowStartAt: 0,
    lastTokenAt: 0,
    mcpIteration: 0, // server tool-loop position (gauge)
    mcpMax: 0, // MAX_TOOL_ITERATION reported by the server
    usageTokens: 0, // cumulative completion tokens per `usage` event
    usageDecode: 0, // cumulative decode seconds per `usage` event
    reasoningChars: 0, // chars of streamed thinking (live tok/s numerator)
    usageGeneratedChars: 0, // generated chars at the last `usage` baseline
    usageGeneratedSeconds: 0, // client-measured seconds at that same baseline
    decode: 0, // cumulative decode seconds from the `done` event
    contextTokens: 0, // LAST round's prompt + completion = context occupancy
    contextEstimated: false, // occupancy figure involves a char estimate
    // Set by a `truncated` error frame: the server ended the stream before
    // the reply was finished, so this turn is a fragment (see applyEvent).
    truncated: false,
    // Set by the `pause_ack` frame: the user interrupted this turn, so its
    // fragment is deliberately kept in the conversation and the payload says
    // so (display state, like `truncated` — never written to session JSON).
    paused: false,
  })
  // The thread now has a turn — the second half of the D7 rule: a
  // contract-owning mode (the server's `editable: false`) becomes unleaveable
  // from here on. Queried rather than asserted, so the lock can never disagree
  // with the message list it is a rule about.
  syncAgentModeLock()
  // Keep the reactive proxy so token mutations re-render.
  const assistant = state.messages[state.messages.length - 1]
  activeMessage = assistant

  state.phase = 'waiting'
  state.session.startedAt = Date.now()
  controller = new AbortController()
  try {
    const payload = { messages: history }
    if (state.mcp.loaded) {
      // Conversation-scoped allowlist: only the toggles currently on may
      // contribute tools (intersected with the Settings-activated ids
      // server-side). Omitted when the snapshot failed to load.
      payload.mcpEnabled = Object.keys(state.mcp.enabled).filter(
        (id) => state.mcp.enabled[id],
      )
    }
    if (state.lsp.loaded) {
      // Same conversation-scoped filter for the LSP language servers.
      payload.lspEnabled = Object.keys(state.lsp.enabled).filter(
        (id) => state.lsp.enabled[id],
      )
    }
    // The agent mode for this turn (am-07). Omitted when the mode list failed
    // to load, so the server's own default applies — the same rule as the hub
    // allowlists above, for the same reason: a dropdown that could not be
    // filled must not silently hand the conversation to a different persona.
    // `agentMode === undefined` → the composer's selection; a resolved id (the
    // batch runner's pin) → that id, whatever the select shows; `null` → omit
    // the key so the server's default applies.
    const modeId = agentMode === undefined ? agentModePayloadId() : agentMode
    if (modeId) {
      payload.agentMode = modeId
    }
    if (batchContext) {
      // The lot's big picture (batch mode): the server appends it to its own
      // system message, so it rides on every task turn without ever becoming
      // a visible bubble — which is what survives the per-task context reset.
      payload.batchContext = batchContext
    }
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => null)
      state.error = describeError(body, `Chat request failed (${res.status}).`)
      if (!assistant.raw) {
        state.messages.splice(state.messages.indexOf(assistant), 1)
      }
      finish(assistant)
      return
    }
    // Accepted send: the conversation has started, so the MCP and LSP
    // toggles freeze until the restart button resets them (sticky by
    // design — edit-and-resend of the last message does not unlock them
    // again).
    state.mcp.locked = true
    state.lsp.locked = true
    await readStream(res, assistant)
  } catch (exc) {
    if (exc?.name !== 'AbortError') {
      state.error = `Could not reach the API: ${exc}`
    }
    // Connection drop: keep whatever partial answer arrived; an explicit
    // Stop is handled in the finally block below.
  } finally {
    // A paused reply keeps its fragment (Pause's whole point), except when
    // nothing was generated: the fallback abort has no `pause_ack` to drop
    // the empty turn, so it goes here.
    const discard = (userStopped && !userPaused) || (userPaused && !assistant.raw)
    if (discard) {
      // Stop discards the partial reply entirely (including the empty
      // placeholder and its MCP tool badges). The user message stays so it
      // can be edited and re-sent; the server-side stream is aborted.
      const index = state.messages.indexOf(assistant)
      if (index !== -1) state.messages.splice(index, 1)
    }
    finish(assistant)
  }
}

export function useChat() {
  async function send() {
    const trimmed = state.draft.raw.trim()
    const pendingImage = state.pendingImage
    // An interactive send is refused while a batch owns the conversation
    // (chatBusy, not just the phase: the composer is also shut in the idle
    // window between two task turns). sendBatchTurn below deliberately checks
    // the phase only — that window is exactly when it must be able to fire.
    if ((!trimmed && !pendingImage) || chatBusy()) return

    // Reuse the preview HTML rendered by setDraft; re-render only if Enter
    // beat the debounce (or nothing was ever rendered for this text).
    const html = flushDraftHtml(trimmed)
    clearDraft()
    const userMessage = {
      role: 'user',
      raw: trimmed,
      html,
      streaming: false,
      images: pendingImage ? [{ ...pendingImage }] : [],
    }
    state.pendingImage = null
    // The whole accumulated thread, the new turn last — exactly what send()
    // posted before startTurn was factored out.
    const history = [
      ...state.messages.map(messagePayload),
      messagePayload(userMessage),
    ]
    await startTurn({ userMessage, history })
  }

  // Batch mode's entry point: one task instruction, one turn. Returns whether
  // the turn was *started* (false on an empty instruction or a reply already
  // streaming) — the batch loop needs that flag instead of awaiting a phase
  // change that would never come. The payload deliberately carries this turn
  // alone: `state.messages` is never used to build a batch request, so task N+1
  // arrives with no trace of task N's transcript (B6).
  function sendBatchTurn(text, { batchContext = null, agentMode = undefined } = {}) {
    const trimmed = typeof text === 'string' ? text.trim() : ''
    if (!trimmed || state.phase !== 'idle') return false
    startTurn({
      userMessage: {
        role: 'user',
        raw: trimmed,
        html: renderMarkdown(trimmed),
        streaming: false,
        images: [],
      },
      history: [{ role: 'user', content: trimmed }],
      batchContext,
      agentMode,
    })
    return true
  }

  function stop() {
    userStopped = true
    if (controller) controller.abort()
  }

  // Non-destructive twin of stop(): the server closes the upstream stream at
  // its next message boundary and terminates the reply with `pause_ack`, which
  // keeps the partial turn in the conversation. One-way — the ack puts the app
  // back to idle, and the user's next Send (their comment, riding on top of the
  // interrupted turn) is the only way back into inference.
  async function pause() {
    // Blinking = waiting for the ack: not clickable, and nothing to pause once
    // the reply is over.
    if (state.phase === 'idle' || state.pausePending) return
    state.pausePending = true
    try {
      // No body: there is exactly one pausable reply (the server keeps the
      // single in-flight handle).
      await fetch('/api/chat/pause', { method: 'POST' })
    } catch {
      // Endpoint unreachable: fall back to a local abort that keeps the
      // fragment. The abort ends readStream, so finish() runs and clears
      // pausePending.
      userPaused = true
      // Flag the fragment here, since no `pause_ack` will ever do it: it gets
      // the caption and, more importantly, the payload note that tells the
      // model this turn is unfinished.
      if (activeMessage) activeMessage.paused = true
      if (controller) controller.abort()
    }
  }

  function editLast() {
    if (chatBusy()) return
    let index = -1
    for (let i = state.messages.length - 1; i >= 0; i -= 1) {
      if (state.messages[i].role === 'user') {
        index = i
        break
      }
    }
    if (index === -1) return
    setDraft(state.messages[index].raw)
    // Re-attach the message's image so it travels with the edited text.
    state.pendingImage = state.messages[index].images?.[0] ?? null
    // Drop the message and everything after it (its replies and tool badges)
    // so the edited text re-sends onto a clean conversation tail.
    state.messages.splice(index)
    // An edited-away thread can be empty, and the D7 lock is a rule about the
    // thread — so it is re-derived here too (queried, never asserted).
    syncAgentModeLock()
    state.error = null
  }

  // Replaces the conversation with the turns of a parsed session file (see
  // parseSessionFile). Returns `{ restored }`, or null when refused because a
  // reply is streaming — the "must not be running" rule lives here, not only
  // on the button, exactly as in editLast()/reset().
  function loadSession(parsed) {
    if (chatBusy()) return null
    const items = Array.isArray(parsed?.messages) ? parsed.messages : []
    if (!items.length) return { restored: 0 }

    // Rebuild the rich shape the view and the side panel render: the common
    // fields plus the assistant-only field set send() initialises, so the
    // badges (which read `tokens` / `contextTokens`) never see `undefined`.
    state.messages = items.map((item) => {
      const raw = item.content ?? ''
      const message = {
        role: item.role,
        raw,
        html: renderMarkdown(raw, { mermaid: true }),
        streaming: false,
        // Images restored from the file (already `{ dataUrl, name, size }`), so
        // the thumbnail renders again and the next send re-posts the image to
        // the model — and editLast() can re-attach it to the edited draft.
        images: item.images ?? [],
      }
      if (item.role !== 'assistant') return message
      // The loaded thinking flows become the reply's panel timeline: one
      // `{ kind: 'reasoning' }` block per flow, in file order (tool badges are
      // never saved, so nothing else belongs in `blocks`).
      const blocks = (item.reasoning ?? []).map((text) => ({ kind: 'reasoning', text }))
      return Object.assign(message, {
        tools: [],
        blocks,
        pendingStreamQuote: false,
        tokens: 0,
        took: 0,
        ttft: 0,
        decode: 0,
        contextTokens: 0,
        tokensEstimated: false,
        contextEstimated: false,
        // Client-side measurement + tool-loop bookkeeping (see noteGenerated):
        // zeroed, so a restored reply reports no rate and no gauge position.
        generatedSeconds: 0,
        windowStartAt: 0,
        lastTokenAt: 0,
        mcpIteration: 0,
        mcpMax: 0,
        usageTokens: 0,
        usageDecode: 0,
        reasoningChars: 0,
        usageGeneratedChars: 0,
        usageGeneratedSeconds: 0,
      })
    })

    // A loaded conversation counts as started: the MCP/LSP toggles freeze as
    // they would have after the first accepted send, and the loaders below
    // must not re-snapshot over them (they ignore a locked conversation).
    // No reset() here — it would wipe the messages just loaded.
    if (state.messages.length) {
      state.mcp.locked = true
      state.lsp.locked = true
    } else {
      loadChatMcp()
      loadChatLsp()
    }
    // The mode the file was saved with (v4 envelope) is conversation state too:
    // useAgentModes judges it against *this* workspace — an unknown id falls
    // back to the workspace default with one notice line, never a silent swap of
    // persona and never a refused load — and restores the D7 lock, because a
    // restored thread *is* that thread.
    applySessionAgentMode(parsed.agentMode)
    state.error = null
    state.pendingImage = null
    // state.session.startedAt stays 0: the stopwatch reads 00:00:00 for a
    // conversation that is not currently running a reply.
    return { restored: state.messages.length }
  }

  // `refetchHubs: false` is batch mode's between-task wipe: the conversation
  // is wiped but the conversation-scoped MCP/LSP allowlist must NOT be re
  // seeded, or a connector that appeared mid-lot would silently join the tool
  // set of the remaining tasks. The toggles stay locked for the whole lot.
  function reset({ refetchHubs = true } = {}) {
    if (controller) controller.abort()
    state.messages = []
    state.error = null
    state.pendingImage = null
    // A pause still in flight has no conversation to ack any more: clear the
    // blinking flag (and the per-reply fallback flag) so no ⏸ survives on a
    // conversation that no longer streams. The ack itself is a no-op then —
    // `assistant.streaming` is already false (the guard in applyEvent).
    state.pausePending = false
    userPaused = false
    // A cleared thread has no turns, so the D7 mode lock is released — before
    // the `refetchHubs` early return, next to the hub unlocks below: batch
    // mode's between-task wipe resets without refetching, and it unlocks too
    // (a lot never runs *as* the mode that authored it, D13).
    syncAgentModeLock(false)
    // The session-mode notice explained a conversation that no longer exists.
    clearAgentModeNotice()
    if (!refetchHubs) return
    // New conversation: MCP and LSP toggles unlock against freshly fetched
    // snapshots, keeping the previous on/off choices (the loaders seed the
    // new lists from them). The mode list is refreshed by the same call: a
    // modes file edited by hand while the lot ran (or while this tab sat
    // idle) takes effect on the next conversation.
    state.mcp.locked = false
    state.lsp.locked = false
    loadChatHubs()
  }

  return {
    state,
    send,
    // The same busy rule the guards in this composable use: a control that has
    // to lock while a reply streams reads this rather than watching `phase`, so
    // it cannot disagree with a send being refused.
    chatBusy,
    // The one switch of the Live View, next to the rule that locks it — a
    // component that draws the control calls this and never moves the flag.
    toggleLiveView,
    sendBatchTurn,
    stop,
    pause,
    reset,
    setDraft,
    appendDraft,
    registerDraftInput,
    editLast,
    loadSession,
    attachImage,
    clearImage,
    loadChatMcp,
    setChatMcp,
    restartMcp: () => restartHub('mcp'),
    loadChatLsp,
    setChatLsp,
    restartLsp: () => restartHub('lsp'),
    // The three together (modes + both hubs), for the entry points that want a
    // fresh conversation snapshot rather than one list.
    loadChatHubs,
  }
}
