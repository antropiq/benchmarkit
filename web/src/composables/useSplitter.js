import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

const STORAGE_KEY = 'benchmarkit:chatSplit'
// Enough room for the whole input column stack: the badges row, the action
// buttons and the session stopwatch under them (plus the memory bullets and
// a one-line textarea) — anything smaller would clip the stopwatch.
const MIN_INPUT_HEIGHT = 124
const MIN_OUTPUT_HEIGHT = 160
const KEYBOARD_STEP = 24

function loadSaved() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    const value = raw === null ? NaN : Number.parseInt(raw, 10)
    return Number.isFinite(value) && value > 0 ? value : null
  } catch {
    return null
  }
}

function save(value) {
  try {
    if (value === null) window.localStorage.removeItem(STORAGE_KEY)
    else window.localStorage.setItem(STORAGE_KEY, String(value))
  } catch {
    // Private mode / storage disabled: the split still works for the session.
  }
}

/**
 * Vertical splitter between the chat output (top) and the input row
 * (bottom) of `.chat-shell`. Dragging the separator sets a pixel
 * flex-basis on the bottom group; the top group absorbs the rest.
 * The size persists to localStorage; double-click resets the default
 * 4:1 CSS ratio.
 */
export function useSplitter(shellRef, bottomRef) {
  const inputHeight = ref(loadSaved())
  const dragging = ref(false)
  let startY = 0
  let startHeight = 0

  function bottomEl() {
    const value = bottomRef.value
    return value?.$el ?? value
  }

  const topStyle = computed(() =>
    inputHeight.value ? { flex: '1 1 auto' } : null,
  )
  const bottomStyle = computed(() =>
    inputHeight.value ? { flex: `0 0 ${inputHeight.value}px` } : null,
  )

  function clamp(height) {
    const shell = shellRef.value
    const available = shell ? shell.clientHeight : Number.POSITIVE_INFINITY
    const max = Math.max(MIN_INPUT_HEIGHT, available - MIN_OUTPUT_HEIGHT)
    return Math.min(Math.max(height, MIN_INPUT_HEIGHT), max)
  }

  function apply(height) {
    inputHeight.value = clamp(height)
  }

  function onPointerDown(event) {
    const bottom = bottomEl()
    if (!bottom) return
    event.preventDefault()
    dragging.value = true
    startY = event.clientY
    startHeight = bottom.getBoundingClientRect().height
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function onPointerMove(event) {
    if (!dragging.value) return
    apply(startHeight - (event.clientY - startY))
  }

  function onPointerUp(event) {
    if (!dragging.value) return
    dragging.value = false
    const target = event.currentTarget
    if (target.hasPointerCapture?.(event.pointerId)) {
      target.releasePointerCapture(event.pointerId)
    }
    save(inputHeight.value)
  }

  function onKeyDown(event) {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault()
      const step = event.key === 'ArrowUp' ? KEYBOARD_STEP : -KEYBOARD_STEP
      const shell = shellRef.value
      const current =
        inputHeight.value ?? (shell ? shell.clientHeight / 5 : MIN_INPUT_HEIGHT)
      apply(current + step)
      save(inputHeight.value)
    } else if (event.key === 'Home') {
      event.preventDefault()
      inputHeight.value = null
      save(null)
    }
  }

  function onDoubleClick() {
    inputHeight.value = null
    save(null)
  }

  function onResize() {
    const shell = shellRef.value
    // While the Chat tab is hidden (App.vue toggles tabs with v-show), the
    // shell has no layout and clientHeight is 0. Clamping the restored
    // height against a 0-height shell collapses it to the minimum, which is
    // what made the saved split position look lost after a page refresh.
    if (!inputHeight.value || !shell || shell.clientHeight === 0) return
    inputHeight.value = clamp(inputHeight.value)
  }

  // A ResizeObserver catches window resizes and the moment the shell gains
  // layout again when the Chat tab becomes visible — no resize event fires
  // for the latter, and the first observer callback doubles as the initial
  // check, so the restored height is validated as soon as it can be.
  let observer = null
  onMounted(() => {
    const shell = shellRef.value
    if (typeof ResizeObserver !== 'undefined' && shell) {
      observer = new ResizeObserver(onResize)
      observer.observe(shell)
    } else {
      window.addEventListener('resize', onResize)
    }
  })
  onBeforeUnmount(() => {
    if (observer) observer.disconnect()
    else window.removeEventListener('resize', onResize)
  })

  nextTick(onResize)

  return {
    dragging,
    topStyle,
    bottomStyle,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onKeyDown,
    onDoubleClick,
  }
}
