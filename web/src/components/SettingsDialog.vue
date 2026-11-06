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
} = useBenchmark()

const dialogEl = ref(null)
const activeSection = ref('llm')

const SECTIONS = [
  { id: 'llm', label: 'LLM' },
  { id: 'question-set', label: 'Question set' },
  { id: 'sampling', label: 'Sampling params' },
  { id: 'verifier', label: 'Verifier LLM' },
  { id: 'mcp', label: 'MCP' },
  { id: 'lsp', label: 'LSP' },
]

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

// The workspace settings file's own report (D8/D9): ``failed`` carries its
// sentence in ``error``, the still-holds-the-template-text state in
// ``message``. ``absent`` carries its own too: production can never reach it
// (the seed always creates the file), but the e2e suite's mode reads a
// hand-authored file the app never seeds, so a switched-to workspace really can
// have none — and empty fields with no explanation would read as an app bug.
// The server's sentence already names the file and its path, so the dialog
// shows it verbatim — adds nothing, translates nothing.
const workspaceSettingsNote = computed(() => {
  const block = state.workspaceSettings
  if (!block) return null
  if (block.state === 'failed') return block.error || null
  if (block.state === 'absent') return block.message || null
  if (block.state === 'ok' && !block.configured) return block.message || null
  return null
})

// The file this dialog saves group 1 to, with the workspace path the server
// reported — never a client-side guess (the same dialog means a different
// server per workspace, so the user must see *which* workspace).
const workspaceSettingsPath = computed(
  () => state.workspaceSettings?.path || '',
)
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
      The model connection is saved to this workspace's
      <code>{{ workspaceSettingsPath }}</code> — the file this server reads and
      writes; the verifier and the question set go to the server-side
      <code>.env</code>. Saving here applies right away; editing either file by
      hand needs a server restart. Leave optional fields empty to omit them.
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
          <section v-show="activeSection === 'llm'">
            <h3 class="settings-section-title">LLM</h3>
            <p v-if="workspaceSettingsNote" class="caption alert-error mcp-externals-error" role="alert">
              {{ workspaceSettingsNote }}
            </p>
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
            <p class="caption">
              Used verbatim as the system prompt of a benchmark run; in the
              WorkBench tab it is the fallback for an agent mode whose
              <code>systemMessage</code> is empty.
            </p>
            <label class="settings-vision">
              <input
                type="checkbox"
                v-model="state.settings.supportsVision"
              />
              Endpoint supports image input (vision)
            </label>
            <p class="caption">
              Enables the image attach button in the WorkBench tab. Only check it
              when the model accepts image input; GIFs are sent as their first
              frame.
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
              Every connector is declared in the workspace's
              <code>.nojae/mcp-settings.json</code> (seeded from
              <code>mcp_connectors/mcp-settings.default.json</code> when a folder
              becomes the working location); connected ones start
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
              Nothing declared in <code>.nojae/mcp-settings.json</code>:
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
              one updates <code>.nojae/lsp-settings.json</code> and
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
              <code>.nojae/lsp-settings.json</code> (copy the seeded
              <code>.nojae/lsp-settings.template.json</code> to get started).
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
