<script setup>
// The two focalisation controls of the chat screen, in a third flex item of
// `.chat-shell` between the two columns (spec §3). Presentational by design:
// the state is `useChatLayout`'s (D1), read here as props and changed only by
// the two setters the parent wires to wf-01's toggles — this component decides
// nothing about the transition.
//
// It is never hidden by a state and never by `v-if` either: the gutter is the
// one surface all three focalisations keep, so it is where the way back always
// lives. Focus control on top, hide control below — the user's sketch, and the
// frequent gesture is the upper one.
//
// The marks are placeholders: a plain square and a chevron. wf-03 replaces them
// with the dashed glyphs and adds the state rules; the class names below are
// what its CSS targets.

import { FOCUS_REASONING } from '../composables/useChatLayout'

defineProps({
  focus: { type: String, required: true },
  leftHidden: { type: Boolean, required: true },
})

const emit = defineEmits(['toggle-focus', 'toggle-left-hidden'])
</script>

<template>
  <div class="chat-col-gutter">
    <button
      type="button"
      class="chat-gutter-btn"
      :aria-pressed="focus === FOCUS_REASONING"
      :aria-label="
        focus === FOCUS_REASONING
          ? 'Restore the tool column'
          : 'Show the reasoning only'
      "
      :title="
        focus === FOCUS_REASONING
          ? 'Restore the tool column'
          : 'Show the reasoning timeline alone'
      "
      @click="emit('toggle-focus')"
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <rect x="3" y="3" width="18" height="18" rx="2" />
      </svg>
    </button>
    <button
      type="button"
      class="chat-gutter-btn"
      :aria-pressed="leftHidden"
      :aria-label="leftHidden ? 'Show the tool column' : 'Hide the tool column'"
      :title="leftHidden ? 'Show the tool column' : 'Hide the tool column'"
      @click="emit('toggle-left-hidden')"
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path v-if="leftHidden" d="m9 6 6 6-6 6" />
        <path v-else d="m15 6-6 6 6 6" />
      </svg>
    </button>
  </div>
</template>
