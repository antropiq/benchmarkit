<script setup>
import { nextTick, ref, watch } from 'vue'
import { useBenchmark } from '../composables/useBenchmark'

const { state, closeSettings, saveSettings, fetchModels, loadQuestionSets } = useBenchmark()

const dialogEl = ref(null)

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
      <fieldset>
        <legend>Question set</legend>
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
      </fieldset>

      <fieldset>
        <legend>Benchmark LLM</legend>
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
            rows="5"
            placeholder="You are a response expert analyser…"
          ></textarea>
        </label>
      </fieldset>

      <footer class="settings-actions">
        <button class="secondary contrast" type="button" :disabled="state.settingsSaving" @click="closeSettings">
          Cancel
        </button>
        <button type="button" :disabled="state.settingsSaving || state.settingsLoading" @click="saveSettings">
          {{ state.settingsSaving ? 'Saving…' : 'Save' }}
        </button>
      </footer>
    </form>
    </article>
  </dialog>
</template>

