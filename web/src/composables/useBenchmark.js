import { reactive } from 'vue'
import { renderMarkdown } from '../lib/markdown'
import { loadChatLsp, loadChatMcp } from './useChat'

const state = reactive({
  config: null,
  configError: null,
  startError: null,
  status: 'idle', // idle | running | finished | failed | cancelled
  activeTab: 'benchmark', // benchmark | chat
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
  mcpInstalled: false,
  mcpConnectors: [],
  mcpEnabled: [], // connector ids ticked in the dialog, saved via saveMcp()
  mcpMaxIterations: 25,
  mcpLoading: false,
  mcpSaving: false,
  mcpError: null,
  mcpExternalsError: null, // externals.json validation errors from the backend
  lspInstalled: false, // pygls/lsprotocol importable server-side
  lspServers: [],
  lspEnabled: [], // server ids ticked in the dialog, saved via saveLsp()
  lspMaxIterations: 25, // shared MCP+LSP tool-loop cap (one knob)
  lspLoading: false,
  lspSaving: false,
  lspError: null,
  lspServersError: null, // lsp_servers.json validation errors from the backend
  // Working location both hubs point at (${BENCHMARKIT_PATH_LOCATION} in
  // externals.json and lsp_servers.json). ``pathLocation`` is what the input
  // shows/edits, ``pathLocationCurrent`` the value the server confirmed — the
  // Set button is enabled while the two differ.
  pathLocation: '',
  pathLocationCurrent: '',
  pathLocationLoading: false,
  pathLocationSaving: false,
  pathLocationError: null,
})

let eventSource = null
let lastSeq = -1
let answeredMsSoFar = 0
let renderScheduled = false
let activeQuestion = null
let cancelNoticeTimer = null

function scheduleMarkdownRender() {
  if (renderScheduled) return
  renderScheduled = true
  requestAnimationFrame(() => {
    renderScheduled = false
    if (activeQuestion)
      activeQuestion.html = renderMarkdown(activeQuestion.raw, { mermaid: true })
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

function applyTiming(entry, event) {
  entry.took = event.took
  // `tokens` stays completion-only (the tok/s numerator); `consumed` is the
  // prompt + completion figure shown to the user. Old servers without the
  // field: fall back to completion + prompt, exact whenever usage was
  // reported (and identical to `tokens` when it wasn't, promptTokens = 0).
  entry.tokens = event.tokens
  entry.consumed = event.consumed ?? event.tokens + (event.promptTokens ?? 0)
  entry.ttft = event.ttft ?? 0
  entry.genTook = event.genTook ?? 0
  entry.promptTokens = event.promptTokens ?? 0
  entry.tokensEstimated = !!event.tokensEstimated
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
        consumed: 0,
        ttft: 0,
        genTook: 0,
        promptTokens: 0,
        tokensEstimated: false,
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
        applyTiming(activeQuestion, event)
      }
      answeredMsSoFar += event.took * 1000
      state.clock = { mode: 'checking', baseMs: answeredMsSoFar, startedAt: 0 }
      break
    case 'question_done': {
      if (activeQuestion) {
        activeQuestion.streaming = false
        activeQuestion.succeeded = event.succeeded
        applyTiming(activeQuestion, event)
        activeQuestion.html = renderMarkdown(activeQuestion.raw, {
          mermaid: true,
        })
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

  function setTab(tab) {
    if (tab === 'chat' && state.status === 'running') return
    state.activeTab = tab
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
      state.mcpConnectors = []
      state.mcpEnabled = []
      state.mcpError = null
      state.mcpExternalsError = null
      state.lspServers = []
      state.lspEnabled = []
      state.lspError = null
      state.lspServersError = null
      state.pathLocation = ''
      state.pathLocationCurrent = ''
      state.pathLocationError = null
      refreshModelLists()
      loadQuestionSets()
      loadMcp()
      loadLsp()
      loadPathLocation()
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
      apiKey: () => state.settings?.apiKey,
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
      apiKey: () => state.settings?.verifier?.apiKey,
      current: () => state.settings?.verifier?.endpoint,
      setCurrent: (value) => {
        state.settings.verifier.endpoint = value
      },
      models: 'verifierModels',
      loading: 'verifierModelsLoading',
      error: 'verifierModelsError',
    },
  }

  async function fetchModels(kind, resetSelection = false) {
    const fields = MODEL_FIELDS[kind]
    if (!fields || !state.settings || state[fields.loading]) return
    state[fields.error] = null
    // User-initiated refresh: the current selection belongs to the previous
    // listing and is dropped immediately, before the new list is fetched.
    if (resetSelection) fields.setCurrent('')
    const serverUrl = (fields.serverUrl() || '').trim()
    if (!serverUrl) {
      state[fields.models] = []
      return
    }
    state[fields.loading] = true
    // Wipe the previous server's models so a refresh never shows stale options.
    state[fields.models] = []
    try {
      let url = `/api/models?serverUrl=${encodeURIComponent(serverUrl)}`
      const apiKey = (fields.apiKey() || '').trim()
      if (apiKey) url += `&apiKey=${encodeURIComponent(apiKey)}`
      const res = await fetch(url)
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state[fields.error] = describeError(
          body,
          `Could not load models (${res.status})`,
        )
        return
      }
      const models = body.models || []
      state[fields.models] = models
      // Keep the selection consistent with the listed models: a saved model
      // from a previous server would otherwise stay bound invisibly and be
      // saved again for a server that does not offer it.
      if (!models.includes(fields.current())) {
        fields.setCurrent(models[0] || '')
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

  async function loadLsp() {
    state.lspLoading = true
    state.lspError = null
    try {
      const res = await fetch('/api/lsp')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.lspError = describeError(body, `Could not load LSP servers (${res.status})`)
        return
      }
      state.lspInstalled = !!body.installed
      state.lspServers = body.servers || []
      // One shared counter: whichever section is opened last mirrors the
      // server-side value, so both inputs always show the truth.
      state.lspMaxIterations = body.maxIterations ?? 25
      state.mcpMaxIterations = body.maxIterations ?? 25
      state.lspServersError = body.serversError || null
      state.lspEnabled = state.lspServers
        .filter((server) => server.active)
        .map((server) => server.id)
    } catch (exc) {
      state.lspError = `Could not reach the API: ${exc}`
    } finally {
      state.lspLoading = false
    }
  }

  function toggleLsp(id, checked) {
    const enabled = new Set(state.lspEnabled)
    if (checked) enabled.add(id)
    else enabled.delete(id)
    // Keep declaration order so the saved list stays stable.
    state.lspEnabled = state.lspServers
      .map((server) => server.id)
      .filter((serverId) => enabled.has(serverId))
  }

  async function saveLsp() {
    if (state.lspSaving) return
    state.lspError = null
    state.lspSaving = true
    try {
      const res = await fetch('/api/lsp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: state.lspEnabled,
          maxIterations: Number(state.lspMaxIterations) || 25,
        }),
      })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.lspError = describeError(body, `Could not save LSP settings (${res.status})`)
        return
      }
      state.lspInstalled = !!body.installed
      state.lspServers = body.servers || []
      state.lspMaxIterations = body.maxIterations ?? 25
      state.mcpMaxIterations = body.maxIterations ?? 25
      state.lspServersError = body.serversError || null
      state.lspEnabled = state.lspServers
        .filter((server) => server.active)
        .map((server) => server.id)
      state.settingsOpen = false
      // Keep the Chat tab's conversation-scoped LSP panel in sync with the
      // just-saved activation set (kept-OFF switches survive; a locked
      // conversation keeps its frozen list — loadChatLsp is a no-op then).
      loadChatLsp()
    } catch (exc) {
      state.lspError = `Could not reach the API: ${exc}`
    } finally {
      state.lspSaving = false
    }
  }

  async function loadMcp() {
    state.mcpLoading = true
    state.mcpError = null
    try {
      const res = await fetch('/api/mcp')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.mcpError = describeError(body, `Could not load MCP connectors (${res.status})`)
        return
      }
      state.mcpInstalled = !!body.installed
      state.mcpConnectors = body.connectors || []
      state.mcpMaxIterations = body.maxIterations ?? 25
      state.lspMaxIterations = body.maxIterations ?? 25
      state.mcpExternalsError = body.externalsError || null
      state.mcpEnabled = state.mcpConnectors
        .filter((connector) => connector.active)
        .map((connector) => connector.id)
    } catch (exc) {
      state.mcpError = `Could not reach the API: ${exc}`
    } finally {
      state.mcpLoading = false
    }
  }

  function toggleMcp(id, checked) {
    const enabled = new Set(state.mcpEnabled)
    if (checked) enabled.add(id)
    else enabled.delete(id)
    // Keep discovery order so the saved list stays stable.
    state.mcpEnabled = state.mcpConnectors
      .map((connector) => connector.id)
      .filter((connectorId) => enabled.has(connectorId))
  }

  async function saveMcp() {
    if (state.mcpSaving) return
    state.mcpError = null
    state.mcpSaving = true
    try {
      const res = await fetch('/api/mcp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: state.mcpEnabled,
          maxIterations: Number(state.mcpMaxIterations) || 25,
        }),
      })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.mcpError = describeError(body, `Could not save MCP settings (${res.status})`)
        return
      }
      state.mcpInstalled = !!body.installed
      state.mcpConnectors = body.connectors || []
      state.mcpMaxIterations = body.maxIterations ?? 25
      state.lspMaxIterations = body.maxIterations ?? 25
      state.mcpExternalsError = body.externalsError || null
      state.mcpEnabled = state.mcpConnectors
        .filter((connector) => connector.active)
        .map((connector) => connector.id)
      state.settingsOpen = false
      // Keep the Chat tab's conversation-scoped MCP panel in sync with the
      // just-saved activation set (kept OFF switches survive; a locked
      // conversation keeps its frozen list — loadChatMcp is a no-op then).
      loadChatMcp()
    } catch (exc) {
      state.mcpError = `Could not reach the API: ${exc}`
    } finally {
      state.mcpSaving = false
    }
  }

  async function loadPathLocation() {
    state.pathLocationLoading = true
    state.pathLocationError = null
    try {
      const res = await fetch('/api/path-location')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.pathLocationError = describeError(
          body,
          `Could not load the working location (${res.status})`,
        )
        return
      }
      state.pathLocationCurrent = body.path || ''
      state.pathLocation = state.pathLocationCurrent
    } catch (exc) {
      state.pathLocationError = `Could not reach the API: ${exc}`
    } finally {
      state.pathLocationLoading = false
    }
  }

  // Apply a new working location: the endpoint sets the env var and restarts
  // both hubs in one call, so on return the MCP/LSP panels below already
  // carry the post-restart statuses of the tools pointed at the new folder.
  async function setPathLocation() {
    if (state.pathLocationSaving) return
    const next = (state.pathLocation || '').trim()
    if (!next || next === state.pathLocationCurrent) return
    state.pathLocationSaving = true
    state.pathLocationError = null
    try {
      const res = await fetch('/api/path-location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: next }),
      })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.pathLocationError = describeError(
          body,
          `Could not set the working location (${res.status})`,
        )
        return
      }
      // The server canonicalises the path (realpath), so adopt what it stored
      // rather than what was typed: field and Set button then agree.
      state.pathLocationCurrent = body.path || ''
      state.pathLocation = state.pathLocationCurrent
      // The hubs just restarted: refresh both panels so the tool lists and
      // status dots show the new location.
      if (body.mcp) {
        state.mcpInstalled = !!body.mcp.installed
        state.mcpConnectors = body.mcp.connectors || []
        state.mcpExternalsError = body.mcp.externalsError || null
        state.mcpEnabled = state.mcpConnectors
          .filter((connector) => connector.active)
          .map((connector) => connector.id)
        loadChatMcp()
      }
      if (body.lsp) {
        state.lspInstalled = !!body.lsp.installed
        state.lspServers = body.lsp.servers || []
        state.lspServersError = body.lsp.serversError || null
        state.lspEnabled = state.lspServers
          .filter((server) => server.active)
          .map((server) => server.id)
        loadChatLsp()
      }
    } catch (exc) {
      state.pathLocationError = `Could not reach the API: ${exc}`
    } finally {
      state.pathLocationSaving = false
    }
  }

  return {
    state,
    loadConfig,
    start,
    stop,
    running,
    setTab,
    canStop,
    openSettings,
    closeSettings,
    saveSettings,
    loadQuestionSets,
    fetchModels,
    loadMcp,
    toggleMcp,
    saveMcp,
    loadLsp,
    toggleLsp,
    saveLsp,
    loadPathLocation,
    setPathLocation,
  }
}
