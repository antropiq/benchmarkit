import { nextTick, reactive } from 'vue'
import { renderMarkdown } from '../lib/markdown'

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
// (~4 chars/token), used for the live badges and abort fallbacks.
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
  if (!images.length) return { role: message.role, content: message.raw }
  const parts = []
  if (message.raw.trim()) parts.push({ type: 'text', text: message.raw })
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
      assistant.tokensEstimated = !!data.tokensEstimated
      break
    case 'tool_call':
      assistant.tools.push({
        id: data.id ?? `${data.tool}-${assistant.tools.length}`,
        tool: data.tool ?? 'tool',
        server: data.server ?? '',
        preview: data.argsPreview ?? '',
        state: 'running',
      })
      // Server-authoritative tool-loop position for the iteration gauge.
      if (data.iteration) assistant.mcpIteration = data.iteration
      if (data.maxIterations) assistant.mcpMax = data.maxIterations
      break
    case 'usage':
      // Cumulative completion tokens (tool-call arguments included) and
      // decode seconds after each tool round: keeps the side-panel badges
      // live while the MCP tool loop is still running.
      assistant.usageTokens = data.tokens ?? assistant.usageTokens
      assistant.usageDecode = data.decode ?? assistant.usageDecode
      break
    case 'tool_result': {
      const tool = assistant.tools.find((item) => item.id === data.id)
      if (tool) {
        tool.state = data.ok ? 'ok' : 'error'
        tool.resultPreview = data.preview ?? ''
      }
      break
    }
    case 'reasoning':
      // Never downgrade to reasoning once answer text started.
      if (state.phase === 'waiting' || state.phase === 'reasoning') {
        state.phase = 'reasoning'
      }
      // Reasoning opens the decode window, like the benchmark does — the
      // live tok/s estimate must count the thinking time from its start.
      if (!assistant.firstTokenAt) assistant.firstTokenAt = Date.now()
      break
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
  // conversation totals stable with the same ~4 chars/token heuristic the
  // backend uses, flagged as estimated.
  if (!assistant.tokens && assistant.raw) {
    assistant.tokens = estimateCompletionTokens(assistant.raw)
    assistant.tokensEstimated = true
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
      tokens: 0,
      took: 0,
      ttft: 0,
      tokensEstimated: false,
      firstTokenAt: 0,
      lastTokenAt: 0,
      mcpIteration: 0, // server tool-loop position (gauge)
      mcpMax: 0, // MAX_MCP_ITERATION reported by the server
      usageTokens: 0, // cumulative completion tokens per `usage` event
      usageDecode: 0, // cumulative decode seconds per `usage` event
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
      // Accepted send: the conversation has started, so the MCP toggles
      // freeze until the restart button resets them (sticky by design —
      // edit-and-resend of the last message does not unlock them again).
      state.mcp.locked = true
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
    // New conversation: MCP toggles unlock against a freshly fetched
    // snapshot, keeping the previous on/off choices (loadChatMcp seeds the
    // new connector list from them).
    state.mcp.locked = false
    loadChatMcp()
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
  }
}
