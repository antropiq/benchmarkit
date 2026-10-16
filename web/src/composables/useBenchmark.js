import { reactive } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'

marked.setOptions({ breaks: true, gfm: true })

const state = reactive({
  config: null,
  configError: null,
  startError: null,
  status: 'idle', // idle | running | finished | failed
  runId: null,
  totalTests: 0,
  current: 0,
  successes: 0,
  answered: 0,
  tokenRate: 0,
  failedLabel: 'Failed: none',
  entries: [], // model | question | finished | error entries in stream order
  clock: { mode: 'idle', baseMs: 0, startedAt: 0 }, // idle | ticking | checking | frozen
  totalRunningTime: 0,
  progress: 0,
  hasResults: false,
})

let eventSource = null
let lastSeq = -1
let answeredMsSoFar = 0
let renderScheduled = false
let activeQuestion = null

function renderMarkdown(raw) {
  return DOMPurify.sanitize(marked.parse(raw || ''))
}

function scheduleMarkdownRender() {
  if (renderScheduled) return
  renderScheduled = true
  requestAnimationFrame(() => {
    renderScheduled = false
    if (activeQuestion) activeQuestion.html = renderMarkdown(activeQuestion.raw)
  })
}

function resetUi() {
  state.entries = []
  state.current = 0
  state.successes = 0
  state.answered = 0
  state.tokenRate = 0
  state.failedLabel = 'Failed: none'
  state.progress = 0
  state.totalRunningTime = 0
  state.clock = { mode: 'idle', baseMs: 0, startedAt: 0 }
  activeQuestion = null
  answeredMsSoFar = 0
}

function applyEvent(event) {
  switch (event.event) {
    case 'run_started':
      resetUi()
      state.status = 'running'
      state.runId = event.runId
      state.totalTests = event.totalTests
      state.clock = { mode: 'ticking', baseMs: 0, startedAt: Date.now() }
      break
    case 'model_started':
      state.entries.push({ type: 'model', model: event.model })
      break
    case 'question_started':
      state.current = event.index
      activeQuestion = {
        type: 'question',
        index: event.index,
        number: event.number,
        model: event.model,
        question: event.question,
        raw: '',
        html: '',
        streaming: true,
        succeeded: null,
        took: 0,
        tokens: 0,
        version: 0,
      }
      state.entries.push(activeQuestion)
      // Keep the reactive proxy, not the raw object, so token mutations re-render.
      activeQuestion = state.entries[state.entries.length - 1]
      state.clock = {
        mode: 'ticking',
        baseMs: answeredMsSoFar,
        startedAt: Date.now(),
      }
      break
    case 'token':
      if (activeQuestion) {
        activeQuestion.raw += event.text
        activeQuestion.version += 1
        scheduleMarkdownRender()
      }
      break
    case 'answer_done':
      if (activeQuestion) {
        activeQuestion.took = event.took
        activeQuestion.tokens = event.tokens
      }
      answeredMsSoFar += event.took * 1000
      state.clock = { mode: 'checking', baseMs: answeredMsSoFar, startedAt: 0 }
      break
    case 'question_done': {
      if (activeQuestion) {
        activeQuestion.streaming = false
        activeQuestion.succeeded = event.succeeded
        activeQuestion.took = event.took
        activeQuestion.tokens = event.tokens
        activeQuestion.html = renderMarkdown(activeQuestion.raw)
        activeQuestion = null
      }
      state.failedLabel = event.failedLabel
      if (event.stats) {
        state.current = event.stats.current
        state.successes = event.stats.successes
        state.answered = event.stats.answered
        state.tokenRate = event.stats.tokenRate
      }
      state.progress = state.totalTests ? event.index / state.totalTests : 0
      state.clock = { mode: 'frozen', baseMs: answeredMsSoFar, startedAt: 0 }
      break
    }
    case 'run_finished':
      state.status = 'finished'
      state.totalRunningTime = event.totalRunningTime ?? answeredMsSoFar / 1000
      state.progress = 1
      state.entries.push({ type: 'finished' })
      state.hasResults = true
      state.clock = { mode: 'frozen', baseMs: answeredMsSoFar, startedAt: 0 }
      disconnect()
      break
    case 'run_error':
      state.entries.push({ type: 'error', message: event.message })
      break
    default:
      break
  }
}

function trackSeq(eventSourceEvent) {
  if (eventSourceEvent && eventSourceEvent.lastEventId) {
    const seq = parseInt(eventSourceEvent.lastEventId, 10)
    if (!Number.isNaN(seq)) lastSeq = seq
  }
}

const EVENT_NAMES = [
  'run_started',
  'model_started',
  'question_started',
  'token',
  'answer_done',
  'question_done',
  'run_finished',
  'run_error',
  'stream_end',
]

function disconnect() {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
}

function connect(since = -1) {
  disconnect()
  lastSeq = since
  eventSource = new EventSource(`/api/run/events?since=${since}`)
  for (const name of EVENT_NAMES) {
    eventSource.addEventListener(name, (e) => {
      trackSeq(e)
      if (name === 'stream_end') {
        // Server closed the stream; reconnect if the run is still going.
        if (state.status === 'running') {
          eventSource.close()
          eventSource = null
          reconnectViaSnapshot()
        }
        return
      }
      applyEvent(JSON.parse(e.data))
    })
  }
  eventSource.onerror = () => {
    if (state.status !== 'running') {
      disconnect()
      return
    }
    eventSource.close()
    eventSource = null
    setTimeout(reconnectViaSnapshot, 2000)
  }
}

async function reconnectViaSnapshot() {
  try {
    const res = await fetch('/api/run')
    if (!res.ok) return
    const snapshot = await res.json()
    resetUi()
    lastSeq = snapshot.lastSeq ?? -1
    for (const item of snapshot.events || []) applyEvent(item.event)
    state.status = snapshot.status
    if (snapshot.status === 'running') connect(lastSeq)
  } catch {
    // Server unreachable; keep the retry loop alive while a run is active.
    if (state.status === 'running') setTimeout(reconnectViaSnapshot, 2000)
  }
}

export function useBenchmark() {
  async function loadConfig() {
    state.configError = null
    try {
      const res = await fetch('/api/config')
      const body = await res.json()
      if (!res.ok) {
        state.configError =
          typeof body.detail === 'string'
            ? body.detail
            : JSON.stringify(body.detail)
        return
      }
      state.config = body
      state.hasResults = body.hasResults
      state.status = body.runStatus
      if (body.runStatus === 'running' || body.runStatus === 'finished' || body.runStatus === 'failed') {
        await reconnectViaSnapshot()
      }
    } catch (exc) {
      state.configError = `Could not reach the API: ${exc}`
    }
  }

  async function start() {
    state.startError = null
    try {
      const res = await fetch('/api/run', { method: 'POST' })
      const body = await res.json().catch(() => ({}))
      if (res.status === 409) {
        // Another run is active (e.g. started in another tab) — follow it.
        await reconnectViaSnapshot()
        return
      }
      if (!res.ok) {
        state.startError =
          typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
        return
      }
      connect()
    } catch (exc) {
      state.startError = `Could not reach the API: ${exc}`
    }
  }

  function running() {
    return state.status === 'running'
  }

  return { state, loadConfig, start, running }
}
