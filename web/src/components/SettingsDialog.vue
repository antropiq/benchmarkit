<script setup>
import { nextTick, ref, watch } from 'vue'
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
} = useBenchmark()

const dialogEl = ref(null)
const activeSection = ref('question-set')

const SECTIONS = [
  { id: 'question-set', label: 'Question set' },
  { id: 'llm', label: 'LLM' },
  { id: 'sampling', label: 'Sampling params' },
  { id: 'verifier', label: 'Verifier LLM' },
  { id: 'mcp', label: 'MCP' },
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
          <section v-show="activeSection === 'question-set'">
            <h3 class="settings-section-title">Question set</h3>
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
              Enables the image attach button in the Chat tab. Only check it
              when the model accepts image input; GIFs are sent as their first
              frame.
            </p>
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
              Connector servers start with the app and stop when it quits.
              Activated connectors add their tools to the Chat tab only —
              benchmark runs never use them.
            </p>
            <p v-if="!state.mcpInstalled" class="caption alert-error" role="alert">
              The mcp package is not installed. Run
              <code>pip install -e ".[mcp]"</code> (then
              <code>playwright install chromium</code>) and restart the app.
            </p>
            <p v-if="state.mcpLoading" class="caption">Loading connectors…</p>
            <p v-else-if="!state.mcpConnectors.length" class="caption">
              No connectors found in <code>mcp_connectors/</code> (expected
              <code>*_mcp.py</code> modules).
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
                </label>
                <span class="badge" :class="MCP_STATUS_CLASS[connector.status]">
                  {{ connector.status }}
                </span>
              </div>
              <p class="caption mcp-description">{{ connector.description }}</p>
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
              Shared by all connectors.
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
        <button v-else type="button" :disabled="state.settingsSaving || state.settingsLoading" @click="saveSettings">
          {{ state.settingsSaving ? 'Saving…' : 'Save' }}
        </button>
      </footer>
    </form>
    </article>
  </dialog>
</template>
