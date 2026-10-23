<script setup>
import { useBenchmark } from '../composables/useBenchmark'

const { state, setTab } = useBenchmark()
</script>

<template>
  <nav class="tab-nav" role="tablist" aria-label="Screens">
    <button
      class="tab-btn"
      role="tab"
      type="button"
      :class="{ active: state.activeTab === 'benchmark' }"
      :aria-selected="state.activeTab === 'benchmark'"
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
      :disabled="state.status === 'running'"
      :title="state.status === 'running' ? 'Available when the benchmark is idle' : 'WorkBench'"
      @click="setTab('chat')"
    >
      WorkBench
    </button>
  </nav>
</template>
