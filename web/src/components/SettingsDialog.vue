<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useBenchmark } from '../composables/useBenchmark'

const { state, closeSettings, saveSettings, fetchModels } = useBenchmark()

const dialogEl = ref(null)

const REASONING_OPTIONS = [
  { value: '', label: 'None' },
  { value: 'low', label: 'low' },
  { value: 'medium', label: 'medium' },
  { value: 'high', label: 'high' },
  { value: 'xhigh', label: 'xhigh' },
]

// The saved model must stay selectable even when the server no longer lists
// it (endpoint down, model renamed), so prepend the current value.
const benchmarkModelOptions = computed(() => {
  const options = [...state.benchmarkModels]
  const current = state.settings?.endpoint
  if (current && !options.includes(current)) options.unshift(current)
  return options
})

const verifierModelOptions = computed(() => {
  const options = [...state.verifierModels]
  const current = state.settings?.verifier?.endpoint
  if (current && !options.includes(current)) options.unshift(current)
  return options
})

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

    <form method="dialog" class="settings-form" @submit.prevent="saveSettings">
      <fieldset>
        <legend>Benchmark LLM</legend>
        <label>
          Server URL
          <input
            v-model="state.settings.serverUrl"
            type="url"
            placeholder="http://127.0.0.1:8080/v1"
            @change="fetchModels('benchmark')"
          />
        </label>
        <label>
          Model
          <div class="settings-model-row">
            <select v-model="state.settings.endpoint">
              <option v-if="!benchmarkModelOptions.length" value="" disabled>
                {{ state.benchmarkModelsLoading ? 'Loading models…' : 'No models available' }}
              </option>
              <option v-for="model in benchmarkModelOptions" :key="model" :value="model">
                {{ model }}
              </option>
            </select>
            <button
              type="button"
              class="secondary outline settings-model-refresh"
              :disabled="state.benchmarkModelsLoading"
              title="Reload models from server"
              aria-label="Reload models from server"
              @click="fetchModels('benchmark')"
            >
              ↻
            </button>
          </div>
        </label>
        <p v-if="state.benchmarkModelsError" class="caption alert-error" role="alert">
          {{ state.benchmarkModelsError }}
        </p>
        <div class="settings-grid">
          <label>
            Temperature
            <input v-model="state.settings.temp" type="number" step="any" min="0" placeholder="0.2" />
          </label>
          <label>
            Top-p
            <input v-model="state.settings.topP" type="number" step="any" min="0" max="1" placeholder="0.95" />
          </label>
          <label>
            Top-k
            <input v-model="state.settings.topK" type="number" step="1" min="0" placeholder="20" />
          </label>
          <label>
            Min-p
            <input v-model="state.settings.minP" type="number" step="any" min="0" placeholder="0" />
          </label>
          <label>
            Repeat penalty
            <input v-model="state.settings.repeatPenalty" type="number" step="any" min="0" placeholder="1.0" />
          </label>
          <label>
            Presence penalty
            <input v-model="state.settings.presencePenalty" type="number" step="any" placeholder="0" />
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
      </fieldset>

      <fieldset>
        <legend>Verifier</legend>
        <label>
          Server URL
          <input
            v-model="state.settings.verifier.serverUrl"
            type="url"
            placeholder="http://127.0.0.1:8080/v1"
            @change="fetchModels('verifier')"
          />
        </label>
        <label>
          Model
          <div class="settings-model-row">
            <select v-model="state.settings.verifier.endpoint">
              <option v-if="!verifierModelOptions.length" value="" disabled>
                {{ state.verifierModelsLoading ? 'Loading models…' : 'No models available' }}
              </option>
              <option v-for="model in verifierModelOptions" :key="model" :value="model">
                {{ model }}
              </option>
            </select>
            <button
              type="button"
              class="secondary outline settings-model-refresh"
              :disabled="state.verifierModelsLoading"
              title="Reload models from server"
              aria-label="Reload models from server"
              @click="fetchModels('verifier')"
            >
              ↻
            </button>
          </div>
        </label>
        <p v-if="state.verifierModelsError" class="caption alert-error" role="alert">
          {{ state.verifierModelsError }}
        </p>
        <label>
          System prompt
          <textarea
            v-model="state.settings.verifier.sysprompt"
            rows="5"
            placeholder="You are a response expert analyser…"
          ></textarea>
        </label>
      </fieldset>

      <footer class="settings-actions">
        <button class="secondary contrast" type="button" :disabled="state.settingsSaving" @click="closeSettings">
          Cancel
        </button>
        <button type="submit" :disabled="state.settingsSaving || state.settingsLoading">
          {{ state.settingsSaving ? 'Saving…' : 'Save' }}
        </button>
      </footer>
    </form>
    </article>
  </dialog>
</template>

<style scoped>
.settings-model-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.settings-model-row select {
  flex: 1;
  margin-bottom: 0;
}

.settings-model-refresh {
  width: auto;
  margin-bottom: 0;
  padding: 0.35rem 0.75rem;
}
</style>
