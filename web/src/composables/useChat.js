import { reactive } from 'vue'
import { renderMarkdown } from '../lib/markdown'

const state = reactive({
  messages: [], // { role, raw, html, streaming } in conversation order
  phase: 'idle', // idle | waiting | reasoning | answering
  error: null,
  // Composer draft mirrored as rendered markdown so ChatView can preview the
  // exact HTML that send() will attach to the user message (no re-parse).
  draft: { raw: '', html: '', renderedRaw: '' },
})

let controller = null
let renderScheduled = false
let activeMessage = null
let draftTimer = null
// Explicit Stop discards the partial reply; connection drops keep it.
let userStopped = false

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
    .filter((m) => m.raw.trim())
    .map((m) => `## ${m.role}\n\n${m.raw}`)
    .join('\n\n')
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
    if (!trimmed || state.phase !== 'idle') return

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
    })
    const history = state.messages.map(({ role, raw }) => ({ role, content: raw }))
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
    })
    // Keep the reactive proxy so token mutations re-render.
    const assistant = state.messages[state.messages.length - 1]
    activeMessage = assistant

    state.phase = 'waiting'
    controller = new AbortController()
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history }),
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
    // Drop the message and everything after it (its replies and tool badges)
    // so the edited text re-sends onto a clean conversation tail.
    state.messages.splice(index)
    state.error = null
  }

  function reset() {
    if (controller) controller.abort()
    state.messages = []
    state.error = null
  }

  return { state, send, stop, reset, setDraft, editLast }
}
