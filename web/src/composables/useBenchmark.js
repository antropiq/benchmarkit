import { reactive } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import { katexBrackets } from '../lib/markedKatexBrackets'

marked.setOptions({ breaks: true, gfm: true })
marked.use(markedKatex({ throwOnError: false, output: 'html' }))
marked.use(katexBrackets({ throwOnError: false }))

const state = reactive({
  config: null,
  configError: null,
  startError: null,
  status: 'idle', // idle | running | finished | failed | cancelled
  runId: null,
  stopping: false,
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
  settingsOpen: false,
  settings: null,
  settingsLoading: false,
  settingsSaving: false,
  settingsError: null,
  questionSets: [],
  questionSetsLoading: false,
  questionSetsError: null,
  benchmarkModels: [],
  benchmarkModelsLoading: false,
  benchmarkModelsError: null,
  verifierModels: [],
  verifierModelsLoading: false,
  verifierModelsError: null,
})

let eventSource = null
let lastSeq = -1
let answeredMsSoFar = 0
let renderScheduled = false
let activeQuestion = null
let cancelNoticeTimer = null

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

function resetToIdle() {
  disconnect()
  resetUi()
  state.status = 'idle'
  state.runId = null
  state.totalTests = 0
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
  state.stopping = false
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
      // The server warms the model up first; the clock only starts ticking
      // when the first question begins.
      state.clock = { mode: 'starting', baseMs: 0, startedAt: 0 }
      break
    case 'model_started':
      // The tested model lives in the app bar; no stream entry for it.
      // The backend now warms the model up, so hold the chronometer at
      // "Starting…" until question_started arrives.
      state.clock = { mode: 'starting', baseMs: answeredMsSoFar, startedAt: 0 }
      break
    case 'question_started':
      state.current = event.index
      activeQuestion = {
        type: 'question',
        index: event.index,
        number: event.number,
        model: event.model,
        question: event.question,
        questionHtml: renderMarkdown(event.question),
        raw: '',
        html: '',
        streaming: true,
        startedAt: Date.now(),
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
    case 'run_finished': {
      state.stopping = false
      if (event.cancelled) {
        // Revert the whole UI to its initial state; the cancelled notice
        // fades out on its own after a few seconds.
        resetUi()
        state.status = 'cancelled'
        state.entries.push({ type: 'cancelled' })
        clearTimeout(cancelNoticeTimer)
        cancelNoticeTimer = setTimeout(() => {
          const i = state.entries.findIndex((e) => e.type === 'cancelled')
          if (i !== -1) state.entries.splice(i, 1)
        }, 5000)
      } else {
        state.status = 'finished'
        state.totalRunningTime = event.totalRunningTime ?? answeredMsSoFar / 1000
        state.progress = 1
        state.entries.push({ type: 'finished' })
        state.hasResults = true
        state.clock = { mode: 'frozen', baseMs: answeredMsSoFar, startedAt: 0 }
      }
      disconnect()
      break
    }
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
      // A cancelled run must not replay its partial stream on load: the UI
      // stays in its initial state, like after the stop button was used.
      if (['running', 'finished', 'failed'].includes(body.runStatus)) {
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

  async function stop() {
    state.startError = null
    // Finished/failed runs keep the stop button enabled as a reset: clicking
    // it returns the UI to its initial state so a new run can start clean.
    if (state.status === 'finished' || state.status === 'failed') {
      resetToIdle()
      return
    }
    if (state.status !== 'running') return
    state.stopping = true
    try {
      const res = await fetch('/api/run/stop', { method: 'POST' })
      if (res.status === 409) {
        // Run already ended between the click and the request.
        await reconnectViaSnapshot()
        return
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        state.startError =
          typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      }
      // On success keep running until the run_finished(cancelled) event arrives.
    } catch (exc) {
      state.stopping = false
      state.startError = `Could not reach the API: ${exc}`
    }
  }

  function running() {
    return state.status === 'running'
  }

  function canStop() {
    return ['running', 'finished', 'failed'].includes(state.status)
  }

  async function openSettings() {
    if (state.status === 'running' || state.settingsLoading) return
    state.settingsError = null
    state.settingsLoading = true
    state.settingsOpen = true
    try {
      const res = await fetch('/api/settings')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.settingsError = describeError(body, `Could not load settings (${res.status})`)
        return
      }
      state.settings = body
      state.benchmarkModels = []
      state.verifierModels = []
      state.benchmarkModelsError = null
      state.verifierModelsError = null
      state.questionSets = []
      state.questionSetsLoading = false
      state.questionSetsError = null
      refreshModelLists()
      loadQuestionSets()
    } catch (exc) {
      state.settingsError = `Could not reach the API: ${exc}`
    } finally {
      state.settingsLoading = false
    }
  }

  function closeSettings() {
    state.settingsOpen = false
    state.settingsError = null
  }

  const MODEL_FIELDS = {
    benchmark: {
      serverUrl: () => state.settings?.serverUrl,
      current: () => state.settings?.endpoint,
      setCurrent: (value) => {
        state.settings.endpoint = value
      },
      models: 'benchmarkModels',
      loading: 'benchmarkModelsLoading',
      error: 'benchmarkModelsError',
    },
    verifier: {
      // Same fallback as the backend: an unset verifier URL reuses the
      // benchmark server URL.
      serverUrl: () =>
        state.settings?.verifier?.serverUrl || state.settings?.serverUrl,
      current: () => state.settings?.verifier?.endpoint,
      setCurrent: (value) => {
        state.settings.verifier.endpoint = value
      },
      models: 'verifierModels',
      loading: 'verifierModelsLoading',
      error: 'verifierModelsError',
    },
  }

  async function fetchModels(kind) {
    const fields = MODEL_FIELDS[kind]
    if (!fields || !state.settings || state[fields.loading]) return
    state[fields.error] = null
    const serverUrl = (fields.serverUrl() || '').trim()
    if (!serverUrl) {
      state[fields.models] = []
      return
    }
    state[fields.loading] = true
    try {
      const res = await fetch(`/api/models?serverUrl=${encodeURIComponent(serverUrl)}`)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state[fields.error] = describeError(
          body,
          `Could not load models (${res.status})`,
        )
        return
      }
      state[fields.models] = body.models || []
      if (!fields.current()) {
        fields.setCurrent(state[fields.models][0] || '')
      }
    } catch (exc) {
      state[fields.error] = `Could not reach the API: ${exc}`
    } finally {
      state[fields.loading] = false
    }
  }

  function refreshModelLists() {
    fetchModels('benchmark')
    fetchModels('verifier')
  }

  async function saveSettings() {
    if (!state.settings || state.settingsSaving) return
    state.settingsError = null
    state.settingsSaving = true
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.settings),
      })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.settingsError = describeError(
          body,
          `Could not save settings (${res.status})`,
        )
        return
      }
      state.settings = body
      state.settingsOpen = false
      await loadConfig()
    } catch (exc) {
      state.settingsError = `Could not reach the API: ${exc}`
    } finally {
      state.settingsSaving = false
    }
  }

  async function loadQuestionSets() {
    state.questionSetsLoading = true
    state.questionSetsError = null
    try {
      const res = await fetch('/api/question-sets')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.questionSetsError = describeError(
          body,
          `Could not load question sets (${res.status})`,
        )
        return
      }
      state.questionSets = body.questionSets || []
      if (!state.settings?.questionSet && body.current) {
        state.settings.questionSet = body.current
      }
    } catch (exc) {
      state.questionSetsError = `Could not reach the API: ${exc}`
    } finally {
      state.questionSetsLoading = false
    }
  }

  return {
    state,
    loadConfig,
    start,
    stop,
    running,
    canStop,
    openSettings,
    closeSettings,
    saveSettings,
    loadQuestionSets,
    fetchModels,
  }
}
