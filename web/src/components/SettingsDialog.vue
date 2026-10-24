<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import ApiKeyField from './ApiKeyField.vue'
import { useBenchmark } from '../composables/useBenchmark'

const {
  state,
  closeSettings,
  saveSettings,
  fetchModels,
  loadQuestionSets,
  toggleMcp,
  saveMcp,
  toggleLsp,
  saveLsp,
  setPathLocation,
} = useBenchmark()

const dialogEl = ref(null)
const activeSection = ref('working-dir')

const SECTIONS = [
  { id: 'working-dir', label: 'Working directory' },
  { id: 'llm', label: 'LLM' },
  { id: 'question-set', label: 'Question set' },
  { id: 'sampling', label: 'Sampling params' },
  { id: 'verifier', label: 'Verifier LLM' },
  { id: 'mcp', label: 'MCP' },
  { id: 'lsp', label: 'LSP' },
]

// The Set button only makes sense while the field disagrees with what the
// server has applied — comparing against ``pathLocationCurrent`` (the value
// returned by GET/POST /api/path-location) rather than a snapshot taken here
// means a canonicalised path (server realpath) does not leave the button
// spuriously enabled.
const pathLocationChanged = computed(
  () => (state.pathLocation || '').trim() !== state.pathLocationCurrent,
)

const MCP_STATUS_CLASS = {
  running: 'green',
  starting: 'violet',
  error: 'red',
  stopped: 'gray',
}

const REASONING_OPTIONS = [
  { value: '', label: 'None' },
  { value: 'low', label: 'low' },
  { value: 'medium', label: 'medium' },
  { value: 'high', label: 'high' },
  { value: 'xhigh', label: 'xhigh' },
]

// Depends on settings too: the dialog only renders once the first fetch
// filled state.settings, which may land after settingsOpen flipped.
watch(
  () => [state.settingsOpen, state.settings],
  async () => {
    await nextTick()
    const dialog = dialogEl.value
    if (!dialog) return
    const open = state.settingsOpen && state.settings
    if (open && !dialog.open) {
      dialog.showModal()
    } else if (!open && dialog.open) {
      dialog.close()
    }
  },
)

function onBackdropClick(event) {
  if (event.target === dialogEl.value) closeSettings()
}

function onNativeClose() {
  if (state.settingsOpen) closeSettings()
}

function connectorDetail(connector) {
  if (connector.description) return connector.description
  if (connector.kind === 'local') {
    return Array.isArray(connector.command) ? connector.command.join(' ') : ''
  }
  return connector.endpoint || ''
}

// Language-server row: what it handles and where it looks, then its argv
// (secrets live in `env`, which the API never sends, so this is safe text).
function serverDetail(server) {
  const scope = [
    server.extensions?.length
      ? `handles ${server.extensions.join(' ')}`
      : 'no extensions declared',
    `root: ${server.workspaceRoot || '?'}`,
  ]
  const command = Array.isArray(server.command) ? server.command.join(' ') : ''
  return [server.description, scope.join(' · '), command]
    .filter(Boolean)
    .join('\n')
}
</script>

<template>
  <dialog
    v-if="state.settings"
    ref="dialogEl"
    class="settings-dialog"
    aria-labelledby="settings-title"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <article>
    <header>
      <h2 id="settings-title" class="settings-title">Settings</h2>
    </header>
    <p class="caption settings-hint">
      Saved to the server-side <code>.env</code> file and used by the next run.
      Leave optional fields empty to omit them.
    </p>

    <p v-if="state.settingsError" class="alert-error" role="alert">
      {{ state.settingsError }}
    </p>

    <p v-if="state.questionSetsError" class="alert-error" role="alert">
      {{ state.questionSetsError }}
    </p>

    <form class="settings-form" @submit.prevent>
      <div class="settings-body">
        <nav class="settings-nav" aria-label="Settings sections">
          <ul>
            <li v-for="section in SECTIONS" :key="section.id">
              <button
                type="button"
                :class="{ active: activeSection === section.id }"
                :aria-current="activeSection === section.id ? 'true' : undefined"
                @click="activeSection = section.id"
              >
                {{ section.label }}
              </button>
            </li>
          </ul>
        </nav>

        <div class="settings-editor">
          <section v-show="activeSection === 'working-dir'">
            <h3 class="settings-section-title">Working directory</h3>
            <p class="caption path-location-hint" id="path-location-help">
              The folder the MCP connectors and
              language servers operate on. Applied to both hubs at once; it
              resets to this install on the next server start.
            </p>
            <label>
              Path
              <span class="path-location-row">
                <input
                  v-model="state.pathLocation"
                  type="text"
                  spellcheck="false"
                  :disabled="state.pathLocationLoading"
                  aria-describedby="path-location-help"
                />
                <button
                  type="button"
                  class="path-location-set"
                  title="Point the MCP and LSP tools at this folder and reload them"
                  :disabled="
                    !pathLocationChanged ||
                    state.pathLocationSaving ||
                    state.pathLocationLoading
                  "
                  @click="setPathLocation"
                >
                  {{ state.pathLocationSaving ? 'Reloading…' : 'Set' }}
                </button>
              </span>
            </label>
            <p v-if="state.pathLocationError" class="alert-error" role="alert">
              {{ state.pathLocationError }}
            </p>
          </section>

          <section v-show="activeSection === 'llm'">
            <h3 class="settings-section-title">LLM</h3>
            <label>
              Server URL
              <input
                v-model="state.settings.serverUrl"
                type="url"
                placeholder="http://127.0.0.1:8080/v1"
                @change="fetchModels('benchmark', true)"
              />
            </label>
            <label>
              API key
              <ApiKeyField
                v-model="state.settings.apiKey"
                @change="fetchModels('benchmark', true)"
              />
            </label>
            <label>
              Model
              <select v-model="state.settings.endpoint">
                <option v-if="!state.benchmarkModels.length" value="" disabled>
                  {{ state.benchmarkModelsLoading ? 'Loading models…' : 'No models available' }}
                </option>
                <option v-for="model in state.benchmarkModels" :key="model" :value="model">
                  {{ model }}
                </option>
              </select>
            </label>
            <p v-if="state.benchmarkModelsError" class="caption alert-error" role="alert">
              {{ state.benchmarkModelsError }}
            </p>
            <label>
              System prompt
              <textarea
                v-model="state.settings.sysprompt"
                rows="4"
                placeholder="You are a helpful assistant."
              ></textarea>
            </label>
            <label class="settings-vision">
              <input
                type="checkbox"
                :checked="state.settings.supportsVision === '1'"
                @change="state.settings.supportsVision = $event.target.checked ? '1' : ''"
              />
              Endpoint supports image input (vision)
            </label>
            <p class="caption">
              Enables the image attach button in the WorkBench tab. Only check it
              when the model accepts image input; GIFs are sent as their first
              frame.
            </p>
            <label class="settings-vision">
              <input
                type="checkbox"
                :checked="state.settings.litellmDebug === '1'"
                @change="state.settings.litellmDebug = $event.target.checked ? '1' : ''"
              />
              Log LiteLLM internals (debug)
            </label>
            <p class="caption">
              Writes LiteLLM's own DEBUG output for every LLM call to the server
              console, so a failed request shows the real upstream error instead
              of a one-line summary. Applies to the next run or chat message —
              it floods the console with request payloads, so switch it back off
              once the failure is diagnosed.
            </p>
          </section>

          <section v-show="activeSection === 'question-set'">
            <h3 class="settings-section-title">Question set</h3>
            <p class="caption">
              The questions a benchmark run walks through: one
              <code>resources/&lt;name&gt;.json</code> file per set.
            </p>
            <label>
              Source
              <select
                v-model="state.settings.questionSet"
                :disabled="state.questionSetsLoading"
              >
                <option
                  v-for="set in state.questionSets"
                  :key="set.id"
                  :value="set.id"
                >
                  {{ set.label }} ({{ set.count }} questions)
                </option>
                <option v-if="!state.questionSets.length" value="" disabled>
                  {{ state.questionSetsLoading ? 'Loading question sets…' : 'No question sets available' }}
                </option>
              </select>
            </label>
          </section>

          <section v-show="activeSection === 'sampling'">
            <h3 class="settings-section-title">Sampling params for LLM</h3>
            <div class="settings-grid">
              <label>
                Temperature
                <input v-model="state.settings.temp" type="number" step="0.1" min="0" placeholder="0.2" />
              </label>
              <label>
                Top-p
                <input v-model="state.settings.topP" type="number" step="0.1" min="0" max="1" placeholder="0.95" />
              </label>
              <label>
                Top-k
                <input v-model="state.settings.topK" type="number" step="1" min="0" placeholder="20" />
              </label>
              <label>
                Min-p
                <input v-model="state.settings.minP" type="number" step="0.1" min="0" placeholder="0" />
              </label>
              <label>
                Repeat penalty
                <input v-model="state.settings.repeatPenalty" type="number" step="0.1" min="0" placeholder="1.0" />
              </label>
              <label>
                Presence penalty
                <input v-model="state.settings.presencePenalty" type="number" step="0.1" placeholder="0" />
              </label>
            </div>
            <label>
              Reasoning level
              <select v-model="state.settings.reasoning">
                <option v-for="option in REASONING_OPTIONS" :key="option.value" :value="option.value">
                  {{ option.label }}
                </option>
              </select>
            </label>
          </section>

          <section v-show="activeSection === 'verifier'">
            <h3 class="settings-section-title">Verifier LLM</h3>
            <label>
              Server URL
              <input
                v-model="state.settings.verifier.serverUrl"
                type="url"
                placeholder="http://127.0.0.1:8080/v1"
                @change="fetchModels('verifier', true)"
              />
            </label>
            <label>
              API key
              <ApiKeyField
                v-model="state.settings.verifier.apiKey"
                @change="fetchModels('verifier', true)"
              />
            </label>
            <label>
              Model
              <select v-model="state.settings.verifier.endpoint">
                <option v-if="!state.verifierModels.length" value="" disabled>
                  {{ state.verifierModelsLoading ? 'Loading models…' : 'No models available' }}
                </option>
                <option v-for="model in state.verifierModels" :key="model" :value="model">
                  {{ model }}
                </option>
              </select>
            </label>
            <p v-if="state.verifierModelsError" class="caption alert-error" role="alert">
              {{ state.verifierModelsError }}
            </p>
            <label>
              System prompt
              <textarea
                v-model="state.settings.verifier.sysprompt"
                rows="4"
                placeholder="You are a response expert analyser…"
              ></textarea>
            </label>
          </section>

          <section v-show="activeSection === 'mcp'">
            <h3 class="settings-section-title">MCP connectors</h3>
            <p class="caption">
              Every connector is declared in
              <code>mcp_connectors/externals.json</code>; connected ones start
              with the app and stop when it quits. Activated connectors add
              their tools to the WorkBench tab only — benchmark runs never use them.
              Toggling one updates its <code>enable</code> flag in that file and
              starts or stops it right away (local connectors launched through
              <code>npx</code> may take a few seconds to run).
            </p>
            <p v-if="!state.mcpInstalled" class="caption alert-error" role="alert">
              The mcp package is not installed. Run
              <code>pip install -e ".[mcp]"</code> and restart the app.
            </p>
            <p v-if="state.mcpExternalsError" class="caption alert-error mcp-externals-error" role="alert">
              {{ state.mcpExternalsError }}
            </p>
            <p v-if="state.mcpLoading" class="caption">Loading connectors…</p>
            <p v-else-if="!state.mcpConnectors.length" class="caption">
              Nothing declared in <code>mcp_connectors/externals.json</code>:
              add a <code>remote</code> or <code>local</code> entry to get
              connectors here.
            </p>
            <div v-for="connector in state.mcpConnectors" :key="connector.id" class="mcp-panel">
              <div class="mcp-panel-header">
                <label class="mcp-activate">
                  <input
                    type="checkbox"
                    role="switch"
                    :checked="state.mcpEnabled.includes(connector.id)"
                    :disabled="state.mcpLoading"
                    @change="toggleMcp(connector.id, $event.target.checked)"
                  />
                  {{ connector.label }}
                  <span v-if="connector.kind" class="mcp-external-tag">[{{ connector.kind }}]</span>
                </label>
                <span class="mcp-panel-badges">
                  <span class="badge" :class="MCP_STATUS_CLASS[connector.status]">
                    {{ connector.status }}
                  </span>
                </span>
              </div>
              <p class="caption mcp-description">{{ connectorDetail(connector) }}</p>
              <p v-if="connector.error" class="caption alert-error">{{ connector.error }}</p>
            </div>
            <label class="mcp-iterations">
              Max tool iterations
              <input
                v-model="state.mcpMaxIterations"
                type="number"
                min="1"
                max="500"
                placeholder="25"
              />
            </label>
            <p class="caption">
              Cap on the tool-calling loop of a single chat reply.
              Shared by all connectors and language servers
              (<code>MAX_TOOL_ITERATION</code>).
            </p>
          </section>

          <section v-show="activeSection === 'lsp'">
            <h3 class="settings-section-title">LSP language servers</h3>
            <p class="caption">
              Language servers start with the app (entries marked
              <code>lazy</code> spawn on their first tool call) and stop when
              it quits. Activated servers add their <code>lsp_*</code> tools
              to the WorkBench tab only — benchmark runs never use them. Toggling
              one updates <code>lsp_connectors/lsp_servers.json</code> and
              starts or stops it right away. Both gates must agree: the
              switch here and the entry's <code>enable</code> flag.
            </p>
            <p v-if="!state.lspInstalled" class="caption alert-error" role="alert">
              The pygls package is not installed. Run
              <code>pip install -e ".[lsp]"</code> and restart the app.
            </p>
            <p v-if="state.lspError" class="caption alert-error" role="alert">
              {{ state.lspError }}
            </p>
            <p v-if="state.lspServersError" class="caption alert-error mcp-externals-error" role="alert">
              {{ state.lspServersError }}
            </p>
            <p v-if="state.lspLoading" class="caption">Loading language servers…</p>
            <p v-else-if="!state.lspServers.length" class="caption">
              No language servers declared in
              <code>lsp_connectors/lsp_servers.json</code> (copy
              <code>lsp_servers.example.json</code> to get started).
            </p>
            <div v-for="server in state.lspServers" :key="server.id" class="mcp-panel">
              <div class="mcp-panel-header">
                <label class="mcp-activate">
                  <input
                    type="checkbox"
                    role="switch"
                    :checked="state.lspEnabled.includes(server.id)"
                    :disabled="state.lspLoading"
                    @change="toggleLsp(server.id, $event.target.checked)"
                  />
                  {{ server.label }}
                  <span v-if="server.lazy" class="mcp-external-tag">[lazy]</span>
                </label>
                <span class="mcp-panel-badges">
                  <span v-if="server.toolCount" class="badge mcp-port-badge"
                    :title="`tools offered by ${server.id}`">
                    {{ server.toolCount }} tools
                  </span>
                  <span class="badge" :class="MCP_STATUS_CLASS[server.status]">
                    {{ server.status }}
                  </span>
                </span>
              </div>
              <p class="caption mcp-description">{{ serverDetail(server) }}</p>
              <p v-if="server.error" class="caption alert-error">{{ server.error }}</p>
            </div>
            <label class="mcp-iterations">
              Max tool iterations
              <input
                v-model="state.lspMaxIterations"
                type="number"
                min="1"
                max="500"
                placeholder="25"
              />
            </label>
            <p class="caption">
              Cap on the tool-calling loop of a single chat reply, MCP
              connectors and language servers together
              (<code>MAX_TOOL_ITERATION</code> — the same knob as the MCP
              section).
            </p>
          </section>
        </div>
      </div>

      <footer class="settings-actions">
        <button class="secondary contrast" type="button" :disabled="state.settingsSaving" @click="closeSettings">
          Cancel
        </button>
        <button
          v-if="activeSection === 'mcp'"
          type="button"
          :disabled="state.mcpSaving || state.mcpLoading"
          @click="saveMcp"
        >
          {{ state.mcpSaving ? 'Saving…' : 'Save' }}
        </button>
        <button
          v-else-if="activeSection === 'lsp'"
          type="button"
          :disabled="state.lspSaving || state.lspLoading"
          @click="saveLsp"
        >
          {{ state.lspSaving ? 'Saving…' : 'Save' }}
        </button>
        <button v-else type="button" :disabled="state.settingsSaving || state.settingsLoading" @click="saveSettings">
          {{ state.settingsSaving ? 'Saving…' : 'Save' }}
        </button>
      </footer>
    </form>
    </article>
  </dialog>
</template>
