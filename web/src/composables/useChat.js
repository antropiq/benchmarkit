import { nextTick, reactive } from 'vue'
import { renderMarkdown, TOOL_LINE_MARKER } from '../lib/markdown'

const state = reactive({
  messages: [], // { role, raw, html, streaming, images } in conversation order
  phase: 'idle', // idle | waiting | reasoning | answering
  error: null,
  // Composer draft mirrored as rendered markdown so WorkBenchView can preview the
  // exact HTML that send() will attach to the user message (no re-parse).
  draft: { raw: '', html: '', renderedRaw: '' },
  // Image picked for the next message: { dataUrl, name, size } | null.
  pendingImage: null,
  // Chat session stopwatch (rendered by ChatStopwatch): ms timestamp of when
  // the running session started (0 = idle, so the clock reads 00:00:00).
  // Covers the whole request — waiting, thinking and every MCP/LSP tool
  // round — and is cleared when the reply ends, is stopped or errors out.
  session: { startedAt: 0 },
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
// Explicit Stop discards the partial reply; connection drops keep it.
let userStopped = false

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

function setChatMcp(id, on) {
  if (state.mcp.locked || state.phase !== 'idle') return
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
  if (state.lsp.locked || state.phase !== 'idle') return
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
  if (hub.restarting || state.phase !== 'idle') return
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
  const raw = message.raw.split(TOOL_LINE_MARKER).join('')
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
// through messagePayload(). Only the conversation TEXT travels in `content`:
// base64 images, tool-loop turns and the per-turn badges are never written
// (the browser only holds _preview-capped tool summaries, so writing them back
// would be fabrication; the system prompt is rebuilt server-side per send).
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
// v2 added the optional per-assistant-turn `reasoning` list. Version-1 files
// simply carry no such key, so the loader needs no version branch.
export const SESSION_FILE_VERSION = 2
// Stands in for an image part dropped from a loaded file, so an image-only
// turn keeps a non-empty content (an empty one would 422 the next send).
export const SESSION_IMAGE_PLACEHOLDER = '[image stripped from the saved session]'

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
    // images: [] forces the text-only branch, so no base64 ever reaches the file.
    const payload = messagePayload({ ...message, images: [] })
    if (typeof payload.content !== 'string' || !payload.content.trim()) continue
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
  return {
    format: SESSION_FILE_FORMAT,
    version: SESSION_FILE_VERSION,
    messages: items,
  }
}

// Content of one loaded turn: a plain string, or an OpenAI parts array with
// text parts joined and image parts replaced by the placeholder.
function normalizeLoadedContent(content) {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    const parts = []
    for (const part of content) {
      if (part?.type === 'image_url') parts.push(SESSION_IMAGE_PLACEHOLDER)
      else if (typeof part?.text === 'string') parts.push(part.text)
    }
    return parts.join('\n')
  }
  return ''
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
// `{ messages: [{ role, content, reasoning }] }`. Throws with a precise
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
    const content = normalizeLoadedContent(item.content)
    if (!content.trim()) continue
    // A user turn never carries thinking: the key is ignored there (unknown
    // extra keys are tolerated by design).
    const reasoning =
      item.role === 'assistant' ? normalizeLoadedReasoning(item.reasoning, index) : []
    messages.push({ role: item.role, content, reasoning })
  }
  if (!messages.length) throw new Error('Session file holds no usable message.')
  return { messages }
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
      // Thinking text, streamed live: appended to the panel timeline. A
      // tool_call frame in between breaks the flow, so consecutive deltas
      // extend one block and the next round's thinking starts a new one —
      // matching the model's actual reasoning-then-act chronology.
      const text = data.text ?? ''
      if (!text) break
      const blocks = assistant.blocks
      const last = blocks[blocks.length - 1]
      if (last && last.kind === 'reasoning') last.text += text
      else blocks.push({ kind: 'reasoning', text })
      // Thinking is generated text the client DOES see: it feeds the live
      // tok/s numerator and keeps the measurement window open, so a long
      // reasoning phase ticks the speed badge instead of freezing it.
      assistant.reasoningChars += text.length
      noteGenerated(assistant)
      break
    }
    case 'error':
      state.error = data.message || 'Chat request failed.'
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

export function useChat() {
  async function send() {
    const trimmed = state.draft.raw.trim()
    const pendingImage = state.pendingImage
    if ((!trimmed && !pendingImage) || state.phase !== 'idle') return

    userStopped = false
    state.error = null
    // Reuse the preview HTML rendered by setDraft; re-render only if Enter
    // beat the debounce (or nothing was ever rendered for this text).
    const html = flushDraftHtml(trimmed)
    clearDraft()
    state.messages.push({
      role: 'user',
      raw: trimmed,
      html,
      streaming: false,
      images: pendingImage ? [{ ...pendingImage }] : [],
    })
    state.pendingImage = null
    const history = state.messages.map(messagePayload)
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
    })
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
      if (userStopped) {
        // Stop discards the partial reply entirely (including the empty
        // placeholder and its MCP tool badges). The user message stays so it
        // can be edited and re-sent; the server-side stream is aborted.
        const index = state.messages.indexOf(assistant)
        if (index !== -1) state.messages.splice(index, 1)
      }
      finish(assistant)
    }
  }

  function stop() {
    userStopped = true
    if (controller) controller.abort()
  }

  function editLast() {
    if (state.phase !== 'idle') return
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
    state.error = null
  }

  // Replaces the conversation with the turns of a parsed session file (see
  // parseSessionFile). Returns `{ restored }`, or null when refused because a
  // reply is streaming — the "must not be running" rule lives here, not only
  // on the button, exactly as in editLast()/reset().
  function loadSession(parsed) {
    if (state.phase !== 'idle') return null
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
        images: [],
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
    state.error = null
    state.pendingImage = null
    // state.session.startedAt stays 0: the stopwatch reads 00:00:00 for a
    // conversation that is not currently running a reply.
    return { restored: state.messages.length }
  }

  function reset() {
    if (controller) controller.abort()
    state.messages = []
    state.error = null
    state.pendingImage = null
    // New conversation: MCP and LSP toggles unlock against freshly fetched
    // snapshots, keeping the previous on/off choices (the loaders seed the
    // new lists from them).
    state.mcp.locked = false
    loadChatMcp()
    state.lsp.locked = false
    loadChatLsp()
  }

  return {
    state,
    send,
    stop,
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
  }
}
