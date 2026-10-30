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
// The marks are dashed squares and a dashed chevron (spec §4): the dash is the
// glyph, so its pattern is an attribute of each shape here, and the CSS of
// `.chat-gutter-mark` is what dims it at rest and takes it solid on hover.
// No size animation anywhere — the target must not move under the cursor.

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
        class="chat-gutter-mark"
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="butt"
        stroke-linejoin="miter"
        aria-hidden="true"
      >
        <!-- Two concentric dashed squares at rest: "this surface can be
             maximised". Once focused the inner one is gone, so the mark reads
             "restore" without reusing the maximise motif (D11) — and the outer
             square keeps the exact same box, so the glyph does not jump.
             Hiding an SVG child with `v-if` is not D2's problem: D2 is about
             unmounting the reasoning panel, this is a path in a mark. -->
        <!-- The pattern is in viewBox units, and each one divides its own
             square's perimeter (64 and 32), so no dash is cut short at a
             corner. The inner square is half the side, so it takes half the
             period: at 14px both marks then show dashes of the same length. -->
        <rect x="4" y="4" width="16" height="16" stroke-dasharray="5 3" />
        <rect
          v-if="focus !== FOCUS_REASONING"
          x="8"
          y="8"
          width="8"
          height="8"
          stroke-dasharray="4 4"
        />
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
        class="chat-gutter-mark"
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="butt"
        stroke-linejoin="miter"
        aria-hidden="true"
      >
        <!-- Same trait as the squares, and the direction says what the click
             does: pointing left while the left column is on screen (it will
             fold away), right while it is out of the flow (it will come back).
             Both limbs together measure ~17 units, so a 5.6-unit period fits
             three dashes along the chevron exactly — at 14px that is a countable
             mark, not mud. -->
        <path
          v-if="leftHidden"
          d="m9 6 6 6-6 6"
          stroke-dasharray="3.4 2.2"
        />
        <path v-else d="m15 6-6 6 6 6" stroke-dasharray="3.4 2.2" />
      </svg>
    </button>
  </div>
</template>
