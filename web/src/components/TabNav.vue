<script setup>
import { computed } from 'vue'
import { useBenchmark } from '../composables/useBenchmark'
import { useBatch } from '../composables/useBatch'

const { state, setTab } = useBenchmark()
const { state: batch } = useBatch()

// A running batch pins the app to WorkBench (B11): every other tab goes inert,
// and setTab refuses the switch too, so a keyboard path cannot escape mid-lot.
const disabled = computed(() => state.status === 'running' || batch.active)
const tabTitle = computed(() =>
  batch.active
    ? 'Unavailable while a batch runs'
    : 'Available when the benchmark is idle',
)
// A benchmark run locks WorkBench out but not the Editor: browsing files is
// read-only and touches nothing the run is using. A running batch does pin
// this tab too — it is the app-wide pin, not a WorkBench rule — which is the
// same `disabled` flag, and setTab refuses the switch as well.
const editorDisabled = computed(() => batch.active)
const editorTitle = computed(() =>
  batch.active ? 'Unavailable while a batch runs' : 'Editor',
)
</script>

<template>
  <nav class="tab-nav" role="tablist" aria-label="Screens">
    <button
      class="tab-btn"
      role="tab"
      type="button"
      :class="{ active: state.activeTab === 'benchmark' }"
      :aria-selected="state.activeTab === 'benchmark'"
      :disabled="disabled"
      :title="batch.active ? 'Unavailable while a batch runs' : 'Benchmark'"
      @click="setTab('benchmark')"
    >
      Benchmark
    </button>
    <button
      class="tab-btn"
      role="tab"
      type="button"
      :class="{ active: state.activeTab === 'chat' }"
      :aria-selected="state.activeTab === 'chat'"
      :disabled="disabled"
      :title="state.status === 'running' ? tabTitle : 'WorkBench'"
      @click="setTab('chat')"
    >
      WorkBench
    </button>
    <button
      class="tab-btn"
      role="tab"
      type="button"
      :class="{ active: state.activeTab === 'editor' }"
      :aria-selected="state.activeTab === 'editor'"
      :disabled="editorDisabled"
      :title="editorTitle"
      @click="setTab('editor')"
    >
      Editor
    </button>
  </nav>
</template>
