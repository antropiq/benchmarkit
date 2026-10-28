<script setup>
import { computed } from 'vue'
import { useBatch } from '../composables/useBatch'

// The batch meter, rendered under the stopwatch only while a batch is running
// (or just after it ended, so its outcome stays readable). Reuses the
// <progress> + caption pair already styled by `.progress-row`: GaugeBar's
// green→red gradient encodes a *level*, which a task counter has no opinion
// about, so the neutral <progress> is the honest choice.
const { state } = useBatch()

// While running, plus afterwards only when the lot ended on Stop or on a failed
// turn: that outcome stays worth reading, and the only way back to a launchable
// state is a Restart (which resets the status). A lot that simply finished does
// not keep a meter on screen.
const visible = computed(
  () => state.active || state.status === 'stopped' || state.status === 'failed',
)

// value is the highest completed task number, so a gap in the numbering still
// fills the bar honestly.
const value = computed(() => Math.min(state.index, state.total))

const caption = computed(() => `${state.prefix} · ${state.dir}`)

// The counter names the task being worked on, so it reads 1/3 while the first
// turn streams (not 0/3: `state.index` only counts turns that already ran to
// their end). Once no turn is in flight — Stop, or a failed lot — it falls back
// to that completed count, which is the number that stays honest there ("Stop
// on task 3 leaves the bar at 2/3").
const current = computed(() =>
  state.current ? state.current.number : Math.min(state.index, state.total),
)

const counter = computed(() => `${current.value}/${state.total} task`)

const ariaLabel = computed(() => {
  const task = state.current ? `: ${state.current.name}` : ''
  return `${state.prefix} batch — task ${current.value} of ${state.total}${task}`
})

// What each outcome means for the lot, in one line. Deliberately about how the
// *turn* ended, not whether the task succeeded: batch mode never reads a task
// body, so it cannot check the move into done/ that the instruction asks for.
// Re-opening the dialog re-lists the folder, and that is how reality shows.
const OUTCOME_LABEL = {
  done: 'turn finished',
  truncated: 'turn cut short by the server',
  capped: 'hit the tool-iteration cap',
  stopped: 'stopped',
  'not-started': 'turn never started',
  error: 'turn failed',
}

const title = computed(() => {
  const head =
    `Batch progress — turns of ${state.total} numbered task(s), ` +
    'not a verdict on the tasks themselves.'
  const notes = state.outcomes.map((outcome, i) => {
    const number = state.listing?.tasks?.[i]?.number ?? i + 1
    return `task ${number}: ${OUTCOME_LABEL[outcome] ?? outcome}`
  })
  if (state.status === 'failed') notes.push('batch aborted — a turn ended in error')
  if (state.status === 'stopped') notes.push('batch stopped by you')
  return [head, ...notes].join(' · ')
})
</script>

<template>
  <section v-if="visible" class="batch-progress" aria-label="Batch progress">
    <p class="caption">{{ caption }}</p>
    <div class="progress-row">
      <progress :value="value" :max="state.total" :aria-label="ariaLabel"></progress>
    </div>
    <p class="caption mono" :title="title">{{ counter }}</p>
  </section>
</template>
