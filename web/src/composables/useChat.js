import { nextTick, reactive } from 'vue'
import { renderMarkdown, TOOL_LINE_MARKER } from '../lib/markdown'

const state = reactive({
  messages: [], // { role, raw, html, streaming, images } in conversation order
  phase: 'idle', // idle | waiting | reasoning | answering
  error: null,
  // Composer draft mirrored as rendered markdown so ChatView can preview the
  // exact HTML that send() will attach to the user message (no re-parse).
  draft: { raw: '', html: '', renderedRaw: '' },
  // Image picked for the next message: { dataUrl, name, size } | null.
  pendingImage: null,
  // Conversation-scoped MCP toggles (independent from the SettingsDialog):
  // the Settings-activated connectors snapshotted on tab entry, each ON by
  // default, frozen once the first message is accepted by the server.
  mcp: {
    connectors: [], // { id, label, status } — only connectors active in Settings
    enabled: {}, // id -> boolean (this conversation only, never persisted)
    loaded: false, // GET /api/mcp snapshot succeeded (panel + filter usable)
    locked: false, // sticky: first accepted send locks the toggles
  },
  // Conversation-scoped LSP toggles — the exact MCP slice pattern, fed by
  // GET /api/lsp instead. Shared `locked` flag semantics: the first accepted
  // send freezes both panels until restart resets them.
  lsp: {
    servers: [], // { id, label, status } — only language servers active in Settings
    enabled: {}, // id -> boolean (this conversation only, never persisted)
    loaded: false, // GET /api/lsp snapshot succeeded (panel + filter usable)
    locked: false, // sticky: first accepted send locks the toggles (as mcp)
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

// Composer textarea, registered by ChatInput so draft insertions (memory
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
// (~4 chars/token), used for the live badges and abort fallbacks. Also used
// on whole message histories to estimate a round's prompt cost.
export function estimateCompletionTokens(text) {
  return Math.ceil(text.length / 4)
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
      // message starts on its own line in ChatView instead of running onto
      // the previous round's text. Flag consumed once per tool round (the
      // round's later tokens keep flowing on that icon line).
      if (assistant.pendingStreamQuote) {
        assistant.pendingStreamQuote = false
        assistant.raw += assistant.raw ? `\n\n${TOOL_LINE_MARKER}` : TOOL_LINE_MARKER
      }
      assistant.raw += data.text ?? ''
      const now = Date.now()
      if (!assistant.firstTokenAt) assistant.firstTokenAt = now
      assistant.lastTokenAt = now
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
      assistant.usageRawLen = assistant.raw.length
      assistant.usageAt = Date.now()
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
      // starts on a fresh line in ChatView (see the 'token' case).
      assistant.pendingStreamQuote = true
      break
    }
    case 'reasoning':
      // Never downgrade to reasoning once answer text started.
      if (state.phase === 'waiting' || state.phase === 'reasoning') {
        state.phase = 'reasoning'
      }
      // Deliberately NOT anchoring firstTokenAt here: the client-side live
      // estimate only ever sees answer text, so its numerator can never
      // include thinking tokens — anchoring the window on the thinking
      // start would divide answer-only tokens by thinking + answering time
      // and underestimate. The exact accounting (thinking tokens AND time)
      // arrives via the backend's usage/done events, which open the decode
      // window on reasoning deltas server-side.
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
  // text streamed since then, flagged as estimated.
  if (!assistant.tokens) {
    const carried = assistant.usageTokens || 0
    const tail = estimateCompletionTokens(
      assistant.raw.slice(assistant.usageRawLen || 0),
    )
    if (carried + tail > 0) {
      assistant.tokens = carried + tail
      assistant.tokensEstimated = true
    }
  }
  // Stop/abort mid tool-execution: no tool_result will ever arrive, so never
  // leave a badge stuck on the pulsing "running" state in the tool panel.
  for (const tool of assistant.tools ?? []) {
    if (tool.state === 'running') tool.state = 'error'
  }
  assistant.streaming = false
  state.phase = 'idle'
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
      // the token case then opens a tool-icon line in ChatView.
      pendingStreamQuote: false,
      tokens: 0,
      took: 0,
      ttft: 0,
      tokensEstimated: false,
      firstTokenAt: 0,
      lastTokenAt: 0,
      mcpIteration: 0, // server tool-loop position (gauge)
      mcpMax: 0, // MAX_TOOL_ITERATION reported by the server
      usageTokens: 0, // cumulative completion tokens per `usage` event
      usageDecode: 0, // cumulative decode seconds per `usage` event
      usageRawLen: 0, // raw.length at the last `usage` event (estimate baseline)
      usageAt: 0, // client clock at the last `usage` event (live-rate anchor)
      decode: 0, // cumulative decode seconds from the `done` event
      contextTokens: 0, // LAST round's prompt + completion = context occupancy
      contextEstimated: false, // occupancy figure involves a char estimate
    })
    // Keep the reactive proxy so token mutations re-render.
    const assistant = state.messages[state.messages.length - 1]
    activeMessage = assistant

    state.phase = 'waiting'
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
    attachImage,
    clearImage,
    loadChatMcp,
    setChatMcp,
    loadChatLsp,
    setChatLsp,
  }
}
