import { reactive } from 'vue'
import { renderMarkdown } from '../lib/markdown'

const state = reactive({
  messages: [], // { role, raw, html, streaming } in conversation order
  phase: 'idle', // idle | waiting | reasoning | answering
  error: null,
})

let controller = null
let renderScheduled = false
let activeMessage = null

function scheduleMarkdownRender() {
  if (renderScheduled) return
  renderScheduled = true
  requestAnimationFrame(() => {
    renderScheduled = false
    if (activeMessage)
      activeMessage.html = renderMarkdown(activeMessage.raw, { mermaid: true })
  })
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
    case 'token':
      if (state.phase !== 'answering') state.phase = 'answering'
      assistant.raw += data.text ?? ''
      scheduleMarkdownRender()
      break
    case 'reasoning':
      // Never downgrade to reasoning once answer text started.
      if (state.phase === 'waiting' || state.phase === 'reasoning') {
        state.phase = 'reasoning'
      }
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
  async function send(text) {
    const trimmed = (text ?? '').trim()
    if (!trimmed || state.phase !== 'idle') return

    state.error = null
    state.messages.push({
      role: 'user',
      raw: trimmed,
      html: renderMarkdown(trimmed),
      streaming: false,
    })
    const history = state.messages.map(({ role, raw }) => ({ role, content: raw }))
    state.messages.push({ role: 'assistant', raw: '', html: '', streaming: true })
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
      // Abort or connection drop: keep whatever partial answer arrived.
    } finally {
      finish(assistant)
    }
  }

  function stop() {
    if (controller) controller.abort()
  }

  return { state, send, stop }
}
