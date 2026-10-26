<script setup>
import { useSessionSound } from '../composables/useSessionSound'

// Speaker on / speaker muted: the audible twin of the ⚡/✨ tab glyph, so the
// chime that answers "is it still generating?" is switchable from the same bar
// the glyph is trusted in. Turning it back on plays the clip once — that click
// is also the one moment the browser is guaranteed to let it through.
const { state, toggle } = useSessionSound()
</script>

<template>
  <button
    class="icon-btn"
    type="button"
    :aria-pressed="state.enabled"
    :aria-label="state.enabled ? 'Mute the session-ended sound' : 'Unmute the session-ended sound'"
    :title="state.enabled ? 'Session sound: on' : 'Session sound: muted'"
    @click="toggle"
  >
    <svg v-if="state.enabled" width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
      aria-hidden="true">
      <path d="M11 5 6 9H2v6h4l5 4z" />
      <path d="M15.5 8.5a5 5 0 0 1 0 7" />
      <path d="M19 5a9 9 0 0 1 0 14" />
    </svg>
    <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
      aria-hidden="true">
      <path d="M11 5 6 9H2v6h4l5 4z" />
      <path d="m22 9-6 6" />
      <path d="m16 9 6 6" />
    </svg>
  </button>
</template>
