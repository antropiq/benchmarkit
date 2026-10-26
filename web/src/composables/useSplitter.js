import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

// Defaults of the chat splitter, kept as defaults so the original two-argument
// call — `useSplitter(shell, input)` in WorkBenchView — is unchanged.
const CHAT_STORAGE_KEY = 'benchmarkit:chatSplit'
// Enough room for the whole input column stack: the badges row, the action
// buttons and the session stopwatch under them (plus the memory bullets and
// a one-line textarea) — anything smaller would clip the stopwatch.
const CHAT_MIN_RESIZED = 124
const CHAT_MIN_OTHER = 160
const KEYBOARD_STEP = 24
// The fraction of the shell a keyboard nudge starts from when no size has been
// dragged yet — roughly the untouched CSS ratio of each shell.
const DEFAULT_FRACTION = 1 / 5

function loadSaved(storageKey) {
  try {
    const raw = window.localStorage.getItem(storageKey)
    const value = raw === null ? NaN : Number.parseInt(raw, 10)
    return Number.isFinite(value) && value > 0 ? value : null
  } catch {
    return null
  }
}

function save(storageKey, value) {
  try {
    if (value === null) window.localStorage.removeItem(storageKey)
    else window.localStorage.setItem(storageKey, String(value))
  } catch {
    // Private mode / storage disabled: the split still works for the session.
  }
}

/**
 * Draggable splitter inside a two-pane shell.
 *
 * `orientation` names the separator bar itself, exactly as `aria-orientation`
 * does, and with it which pane the drag measures:
 *   'horizontal' (default) — the bar spans the shell; the pane *after* it (the
 *   bottom one, `resizedRef`) grows upwards: ↑↓ keys, `row-resize` cursor.
 *   'vertical' — the bar stands between the panes; the pane *before* it (the
 *   left one, `resizedRef`) grows to the right: ←→ keys, `col-resize` cursor.
 *
 * Options: `storageKey` (localStorage entry), `minResized` / `minOther` (the
 * two clamped minima), `maxRatio` (0 = unbounded; 0.5 caps the pane at half
 * the shell) and `sizeVar`. With `sizeVar` set to a CSS custom-property name,
 * the size reaches the layout through `shellStyle` on the shell — which a grid
 * template consumes via `var(…)` — instead of the flex-basis pair
 * `topStyle` / `bottomStyle` (which stay null).
 *
 * The size persists to localStorage; double-click or `Home` resets it to the
 * shell's own CSS default.
 */
export function useSplitter(shellRef, resizedRef, options = {}) {
  const {
    orientation = 'horizontal',
    storageKey = CHAT_STORAGE_KEY,
    minResized = CHAT_MIN_RESIZED,
    minOther = CHAT_MIN_OTHER,
    maxRatio = 0,
    sizeVar = null,
  } = options

  // A vertical bar measures width; a horizontal one measures height.
  const vertical = orientation === 'vertical'

  const size = ref(loadSaved(storageKey))
  const dragging = ref(false)
  let startCoord = 0
  let startSize = 0

  function resizedEl() {
    const value = resizedRef.value
    return value?.$el ?? value
  }

  function extent(el) {
    if (!el) return 0
    return vertical
      ? el.getBoundingClientRect().width
      : el.getBoundingClientRect().height
  }

  function shellExtent() {
    const shell = shellRef.value
    if (!shell) return 0
    // The panes live inside the shell's padding, so the space they can be
    // given is the content box — clientWidth/Height alone would count it twice.
    const inside = vertical ? shell.clientWidth : shell.clientHeight
    const style = window.getComputedStyle(shell)
    const pad = vertical
      ? parseFloat(style.paddingLeft) + parseFloat(style.paddingRight)
      : parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)
    return inside - (Number.isFinite(pad) ? pad : 0)
  }

  // Grid shells take the size as a custom property on the shell itself, so the
  // template keeps naming its tracks; flex shells get the basis pair instead.
  const shellStyle = computed(() =>
    size.value && sizeVar ? { [sizeVar]: `${size.value}px` } : null,
  )
  const topStyle = computed(() =>
    !sizeVar && size.value ? { flex: '1 1 auto' } : null,
  )
  const bottomStyle = computed(() =>
    !sizeVar && size.value ? { flex: `0 0 ${size.value}px` } : null,
  )

  function clamp(value) {
    // A hidden shell answers <= 0 (no layout at all): nothing to clamp against.
    const box = shellExtent()
    const available = box > 0 ? box : Number.POSITIVE_INFINITY
    let max = Math.max(minResized, available - minOther)
    if (maxRatio > 0) max = Math.min(max, available * maxRatio)
    return Math.min(Math.max(value, minResized), max)
  }

  function apply(value) {
    size.value = clamp(value)
  }

  function onPointerDown(event) {
    const resized = resizedEl()
    if (!resized) return
    event.preventDefault()
    dragging.value = true
    startCoord = vertical ? event.clientX : event.clientY
    startSize = extent(resized)
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function onPointerMove(event) {
    if (!dragging.value) return
    const coord = vertical ? event.clientX : event.clientY
    // The bottom pane grows towards the shell's origin, the left pane away
    // from it — the sign is all that differs between the two orientations.
    apply(vertical ? startSize + (coord - startCoord) : startSize - (coord - startCoord))
  }

  function onPointerUp(event) {
    if (!dragging.value) return
    dragging.value = false
    const target = event.currentTarget
    if (target.hasPointerCapture?.(event.pointerId)) {
      target.releasePointerCapture(event.pointerId)
    }
    save(storageKey, size.value)
  }

  function onKeyDown(event) {
    const growKey = vertical ? 'ArrowRight' : 'ArrowUp'
    const shrinkKey = vertical ? 'ArrowLeft' : 'ArrowDown'
    if (event.key === growKey || event.key === shrinkKey) {
      event.preventDefault()
      const step = event.key === growKey ? KEYBOARD_STEP : -KEYBOARD_STEP
      // With nothing dragged yet, start from the shell's own CSS default.
      const current =
        size.value ?? Math.max(shellExtent() * DEFAULT_FRACTION, minResized)
      apply(current + step)
      save(storageKey, size.value)
    } else if (event.key === 'Home') {
      event.preventDefault()
      size.value = null
      save(storageKey, null)
    }
  }

  function onDoubleClick() {
    size.value = null
    save(storageKey, null)
  }

  function onResize() {
    // While the tab is hidden (App.vue toggles tabs with v-show), the shell has
    // no layout and its extent is 0. Clamping the restored size against a
    // zero-size shell collapses it to the minimum, which is what made the saved
    // split position look lost after a page refresh.
    if (!size.value || shellExtent() <= 0) return
    size.value = clamp(size.value)
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
    size,
    shellStyle,
    topStyle,
    bottomStyle,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onKeyDown,
    onDoubleClick,
  }
}
