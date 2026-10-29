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

// What each turn outcome means, in one line. The turn's own ending is only half
// the story: the loop also checks whether the task file left the lot folder
// (`moved`), which is the completion signal the retry runs on. Batch mode still
// reads no task body, so `moved: true` means "no longer listed in the lot", not
// "the work was good" — re-opening the dialog re-lists the folder, and that is
// how reality shows.
const OUTCOME_LABEL = {
  done: 'turn finished',
  truncated: 'turn cut short by the server',
  capped: 'hit the tool-iteration cap',
  stopped: 'stopped',
  'not-started': 'turn never started',
  error: 'turn failed',
}

// One entry per turn, so a task that was relaunched reads as several lines with
// the same task number — the retry count is what the user is owed.
function noteFor(record, i) {
  const number = record?.number ?? state.listing?.tasks?.[i]?.number ?? i + 1
  const attempt = record?.attempt ?? 0
  const label = OUTCOME_LABEL[record?.outcome] ?? record?.outcome
  const suffix = attempt > 0 ? ` (retry ${attempt})` : ''
  if (record?.moved === false) {
    return `task ${number}: ${label}${suffix} — task file still in the lot`
  }
  return `task ${number}: ${label}${suffix}`
}

const title = computed(() => {
  const head =
    `Batch progress — turns of ${state.total} numbered task(s); a task whose ` +
    'file is still in the lot folder is relaunched.'
  const notes = state.outcomes.map(noteFor)
  if (state.status === 'failed') {
    notes.push(
      'batch aborted — a turn failed, or a task stayed unmoved after the retries',
    )
  }
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
