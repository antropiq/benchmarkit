<script setup>
import { computed, onMounted } from 'vue'
import MetricCard from './components/MetricCard.vue'
import Chronometer from './components/Chronometer.vue'
import StartButton from './components/StartButton.vue'
import StreamPanel from './components/StreamPanel.vue'
import { useBenchmark } from './composables/useBenchmark'

const { state, loadConfig, start } = useBenchmark()

onMounted(loadConfig)

const config = computed(() => state.config)
const models = computed(() => config.value?.models ?? [])
const running = computed(() => state.status === 'running')
const canStart = computed(() => {
  const cfg = config.value
  return !!cfg && cfg.totalTests > 0 && cfg.models.length > 0 && !running.value
})
const scorePercent = computed(() =>
  state.answered ? `${Math.round((state.successes / state.answered) * 100)}%` : '0%',
)
</script>

<template>
  <main class="app-shell">
    <div class="brand-row">
      <div class="model-name">{{ models.join(', ') }}</div>
      <h1>🔬 Benchmark'it</h1>
    </div>

    <div class="main-grid">
      <div class="stats-col">
        <div style="display: flex; justify-content: center">
          <StartButton :disabled="!canStart" :running="running" @click="start" />
        </div>
        <p v-if="state.startError" class="error-line">{{ state.startError }}</p>
        <p v-if="state.configError" class="error-line">{{ state.configError }}</p>
        <p v-if="config" class="caption" style="text-align: center">
          {{ config.questionCount }} questions · {{ config.samplingSummary }}
        </p>
        <p v-if="config && !config.verifierConfigured" class="error-line">
          No verifier agent configured — every answer will be marked incorrect.
        </p>
        <p class="caption" style="text-align: center">⏱ Elapsed</p>
        <Chronometer :clock="state.clock" />
        <MetricCard
          label="⚡ overall token/s"
          :value="state.tokenRate.toFixed(1)"
        />
        <MetricCard
          label="Question in progress"
          :value="state.current ? String(state.current).padStart(2, '0') : '--'"
          :delta="`${state.current} of ${state.totalTests}`"
        />
        <MetricCard
          label="Score"
          :value="`${state.successes}/${state.answered}`"
          :delta="scorePercent"
        />
        <p class="caption">{{ state.failedLabel }}</p>
        <a
          v-if="state.hasResults"
          class="secondary"
          role="button"
          href="/api/results"
          download="results.json"
          style="justify-content: center"
        >⬇ Download results.json</a>
      </div>

      <div>
        <StreamPanel />
        <div class="progress-row">
          <progress
            :value="Math.round(state.progress * 100)"
            max="100"
            style="width: 100%"
          ></progress>
          <span class="caption mono">
            {{ state.current }}/{{ state.totalTests }}
          </span>
        </div>
      </div>
    </div>
  </main>
</template>
