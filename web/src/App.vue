<script setup>
import { computed, onMounted } from 'vue'
import AppBar from './components/AppBar.vue'
import SettingsDialog from './components/SettingsDialog.vue'
import MetricCard from './components/MetricCard.vue'
import Chronometer from './components/Chronometer.vue'
import StreamPanel from './components/StreamPanel.vue'
import { useBenchmark } from './composables/useBenchmark'

const { state, loadConfig, start, stop } = useBenchmark()

onMounted(loadConfig)

const config = computed(() => state.config)
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
  <AppBar
    :can-start="canStart"
    @start="start"
    @stop="stop"
  />

  <SettingsDialog />

  <main class="app-shell">
    <aside class="stats-col">
      <section class="panel run-summary" aria-label="Elapsed time">
        <p class="caption">⏱ Elapsed</p>
        <Chronometer :clock="state.clock" />
        <p v-if="config" class="caption run-meta">
          {{ config.questionCount }} questions · {{ config.samplingSummary }}
        </p>
      </section>

      <MetricCard
        label="⚡ Generation token/s"
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

      <p class="caption failed-label">{{ state.failedLabel }}</p>

      <p v-if="state.startError" class="alert-error" role="alert">{{ state.startError }}</p>
      <p v-if="state.configError" class="alert-error" role="alert">{{ state.configError }}</p>
      <p v-if="config && !config.verifierConfigured" class="alert-error" role="alert">
        No verifier agent configured — every answer will be marked incorrect.
      </p>

      <a
        v-if="state.hasResults"
        class="download-link"
        href="/api/results"
        download="results.json"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
        </svg>
        Download results.json
      </a>
    </aside>

    <section class="stream-col" aria-label="Answer stream">
      <StreamPanel />
      <div class="progress-row">
        <progress
          :value="Math.round(state.progress * 100)"
          max="100"
          aria-label="Run progress"
        ></progress>
        <span class="caption mono">
          {{ state.current }}/{{ state.totalTests }}
        </span>
      </div>
    </section>
  </main>
</template>
