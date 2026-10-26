import { reactive, watch } from 'vue'
import { useBenchmark } from './useBenchmark'
import { useBatch } from './useBatch'
import { useChat } from './useChat'
import { inferenceRunning } from '../lib/inference-state'
// The clip is a build asset, not a server route: Vite hashes it into
// `web/dist/assets/` (like the fonts in `src/fonts/`), so nothing on the
// backend plays or serves audio and `resources/` stays question sets only.
import sessionEndedUrl from '../assets/sounds/session_ended.mp3'

// Audible twin of the tab-title glyph: the same edge, heard instead of seen.
// When the app goes back to "no inference in flight" — the moment the tab
// switches from ⚡ to ✨ — the browser tab is usually the only thing still on
// screen, so a short chime is what actually tells the user their conversation
// is over (and a benchmark run finished).
//
// Deliberately one edge on `inferenceRunning()` rather than a watch on
// useChat.state.phase: the shared predicate keeps ⚡/✨ and the chime telling
// the same story, and it means a batch lot chimes once when the lot is done,
// not once per task turn (the phase is 'idle' between turns while a task is
// still owed). The flip back to idle caused by Stop/Pause chimes too — same
// rule, and it costs nothing to keep one predicate instead of two.

const STORAGE_KEY = 'benchmarkit:sound'

// The clip is mastered loud enough to cut through, so it is turned down here
// rather than re-encoded — a notification has to be audible, not startling.
const PLAYBACK_VOLUME = 0.6

function storedEnabled() {
  try {
    // Only an explicit "0" mutes: a first run (and a storage-less private
    // mode) plays.
    return window.localStorage.getItem(STORAGE_KEY) !== '0'
  } catch {
    return true
  }
}

const state = reactive({ enabled: storedEnabled() })

// Whether the busy->idle watcher has been registered (see useSessionSound).
let watcherArmed = false

// Built on first use, so a session that never reaches an idle edge and a
// muted session never fetch the clip.
let audio = null

function element() {
  if (!audio) {
    audio = new Audio(sessionEndedUrl)
    audio.volume = PLAYBACK_VOLUME
    audio.preload = 'auto'
  }
  return audio
}

// Browsers gate audio playback on "this page has been interacted with", and
// the chime is fired by a watcher, not by a click handler — so Safari (and a
// strict Chrome) can refuse it. The gate lifts for an element whose play()
// was once called from inside a real gesture, so the first click or keypress
// does exactly that, muted and for zero duration.
let armed = false

function unlock() {
  const media = element()
  media.muted = true
  media
    .play()
    .then(() => {
      media.pause()
      media.currentTime = 0
      media.muted = false
      window.removeEventListener('pointerdown', unlock)
      window.removeEventListener('keydown', unlock)
    })
    // Still gated (an automated browser, a blocked autoplay setting): leave the
    // listeners in place so the next gesture tries again.
    .catch(() => {
      media.muted = false
    })
}

function armUnlock() {
  if (armed) return
  armed = true
  window.addEventListener('pointerdown', unlock)
  window.addEventListener('keydown', unlock)
}

export function playSessionEnded() {
  if (!state.enabled) return
  const media = element()
  // Rewind first: two ends close together each get the whole clip instead of
  // its fading tail.
  media.currentTime = 0
  // A rejected promise is only the autoplay gate refusing us; there is nothing
  // to recover from here, and an unhandled rejection would bury real errors.
  media.play().catch(() => {})
}

export function useSessionSound() {
  const { state: benchmark } = useBenchmark()
  const { state: chat } = useChat()
  const { state: batch } = useBatch()

  armUnlock()

  // Registered once for the whole app (App.vue starts it, SoundToggle reads the
  // same singleton): N components asking for the toggle must not ring N clips
  // on one edge.
  if (!watcherArmed) {
    watcherArmed = true
    watch(
      () => inferenceRunning({ benchmark, chat, batch }),
      (running, wasRunning) => {
        if (!wasRunning || running) return
        playSessionEnded()
      },
    )
  }

  function toggle() {
    state.enabled = !state.enabled
    try {
      window.localStorage.setItem(STORAGE_KEY, state.enabled ? '1' : '0')
    } catch {
      // Storage disabled: the toggle still works for this session.
    }
    // Enabling is a click, so it is the one moment playback is guaranteed to
    // be allowed: use it to let the user hear what they just switched on.
    if (state.enabled) playSessionEnded()
  }

  return { state, toggle }
}
