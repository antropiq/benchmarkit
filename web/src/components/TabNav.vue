<script setup>
import { computed } from 'vue'
import { useBenchmark } from '../composables/useBenchmark'
import { useBatch } from '../composables/useBatch'

const { state, setTab } = useBenchmark()
const { state: batch } = useBatch()

// A running batch pins the app to WorkBench (B11): both tabs go inert, and
// setTab refuses the switch too, so a keyboard path cannot escape mid-lot.
const disabled = computed(() => state.status === 'running' || batch.active)
const tabTitle = computed(() =>
  batch.active
    ? 'Unavailable while a batch runs'
    : 'Available when the benchmark is idle',
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
  </nav>
</template>
