<script setup>
import { computed, defineAsyncComponent, onMounted, ref, watch } from 'vue'
import AppBar from './components/AppBar.vue'
import SettingsDialog from './components/SettingsDialog.vue'
import BatchModeDialog from './components/BatchModeDialog.vue'
import MetricCard from './components/MetricCard.vue'
import RunControls from './components/RunControls.vue'
import Chronometer from './components/Chronometer.vue'
import StreamPanel from './components/StreamPanel.vue'
import WorkBenchView from './components/WorkBenchView.vue'
import { useBenchmark } from './composables/useBenchmark'
import { useTabTitle } from './composables/useTabTitle'

// Lazy on purpose: CodeMirror and xterm (ed-04 / ed-07) are heavy, and the
// Benchmark tab is the app's default screen — it must never download them.
// The Editor tab's first open pays the chunk instead.
const EditorView = defineAsyncComponent(() =>
  import('./components/EditorView.vue'),
)

// Mirrors the inference state into the browser tab title (⚡ / ✨), so a
// backgrounded tab still answers "is it generating?". Started here because App
// is where every composable's state gets mounted.
useTabTitle()

const { state, loadConfig, start, stop, loadPathLocation } = useBenchmark()

// The lazy chunk is only fetched once the tab has been opened: an async
// component behind a plain v-show resolves its import at boot, which would make
// the default screen pay for it. Once mounted the view stays mounted (v-show),
// so its open buffers and its terminal session survive a tab switch.
const editorMounted = ref(false)
watch(
  () => state.activeTab,
  (tab) => {
    if (tab === 'editor') editorMounted.value = true
  },
  { immediate: true },
)

onMounted(() => {
  loadConfig()
  // The app bar shows the active working location, so fetch it at boot too —
  // not only when the settings dialog opens.
  loadPathLocation()
})

const config = computed(() => state.config)
const onBenchmarkTab = computed(() => state.activeTab === 'benchmark')
const running = computed(() => state.status === 'running')
const canStop = computed(() =>
  ['running', 'finished', 'failed'].includes(state.status),
)
const canStart = computed(() => {
  const cfg = config.value
  return !!cfg && cfg.totalTests > 0 && cfg.models.length > 0 && !running.value
})
const scorePercent = computed(() =>
  state.answered ? `${Math.round((state.successes / state.answered) * 100)}%` : '0%',
)
</script>

<template>
  <AppBar />

  <SettingsDialog />

  <!-- Mounted here (not inside WorkBenchView) so the lot form survives a tab
       switch, exactly like the settings dialog. -->
  <BatchModeDialog />

  <main v-show="onBenchmarkTab" class="app-shell">
    <aside class="stats-col">
      <section class="panel run-summary" aria-label="Elapsed time">
        <p class="caption">⏱ Elapsed</p>
        <Chronometer :clock="state.clock" />
        <p v-if="config" class="caption run-meta">
          {{ config.questionCount }} questions · {{ config.samplingSummary }}
        </p>
        <RunControls
          :can-start="canStart"
          :running="running"
          :can-stop="canStop"
          :stopping="state.stopping"
          @start="start"
          @stop="stop"
        />
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

  <WorkBenchView v-show="state.activeTab === 'chat'" />

  <!-- v-show, like the other tabs: it is what keeps the Editor tab's open
       buffers, tree state and (from ed-07) its live terminal session alive
       across a peek at the Benchmark screen. The v-if only defers the first
       mount until the lazy chunk is worth fetching. -->
  <EditorView v-if="editorMounted" v-show="state.activeTab === 'editor'" />
</template>
