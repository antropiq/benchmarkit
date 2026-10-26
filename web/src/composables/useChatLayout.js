import { computed, ref } from 'vue'
import { useSplitter } from './useSplitter.js'

// The Chat screen's geometry, in one place, because its two splitters used to
// be one: the shell is now two columns, each a flex column with its own bar,
// so a bar sizes only its own column.
//
//   .chat-shell (row)
//     .chat-col--left   .chat-tool-panel + chat-col-resizer + .chat-tools-dock
//     .chat-col--right  .chat-output     + chat-resizer     + .chat-input
//
// Both splitters take the *column* as their shell, so each measures the
// column it lives in and sizes the pane after its bar: the composer row grows
// upwards, the dock grows downwards. Folding the dock is the one control that
// spans both — it takes the dock and its bar out of the flow, which hands
// their height to the timeline.
//
// A dock tall enough to hold two switches and their hub titles; the timeline
// above it keeps a usable window into the reasoning either way — 200 px being
// the timeline's own 160 px floor (see `.chat-tool-panel`) plus the bar and
// the column gaps, so the clamp and the CSS floor cannot both be honoured and
// the column overflow.
const DOCK_MIN = 104
const DOCK_MIN_OTHER = 200

const COLLAPSE_KEY = 'benchmarkit:chatToolsCollapsed'

function loadCollapsed() {
  try {
    return window.localStorage.getItem(COLLAPSE_KEY) === '1'
  } catch {
    return false
  }
}

export function useChatLayout(leftColRef, rightColRef, inputRef, dockRef) {
  const column = useSplitter(leftColRef, dockRef, {
    storageKey: 'benchmarkit:chatToolsColumn',
    minResized: DOCK_MIN,
    minOther: DOCK_MIN_OTHER,
  })

  // Composer row, unchanged in behaviour — only its bar got shorter.
  const composer = useSplitter(rightColRef, inputRef, {
    storageKey: 'benchmarkit:chatSplit',
  })

  // Folding is a flag, not a size: the dock's dragged height survives it
  // untouched, so expanding comes back at exactly the height it had — the view
  // merely stops applying that basis (and drops the bar) while folded. The
  // write-through keeps it durable however it is flipped (the dock's button
  // through v-model, a caller through the setter).
  const collapsed = ref(loadCollapsed())
  const dockCollapsed = computed({
    get: () => collapsed.value,
    set: (value) => {
      collapsed.value = value
      try {
        window.localStorage.setItem(COLLAPSE_KEY, value ? '1' : '0')
      } catch {
        // Storage disabled: the fold still works for the session.
      }
    },
  })

  return { column, composer, dockCollapsed }
}
