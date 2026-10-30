import { computed, ref } from 'vue'
import { useSplitter } from './useSplitter.js'

// The Chat screen's geometry, in one place, because its two splitters used to
// be one: the shell is a row of three children — two columns, each a flex
// column with its own bar, and the gutter between them — so a bar sizes only
// its own column.
//
//   .chat-shell (row)
//     .chat-col--left   .chat-tool-panel + chat-col-resizer + .chat-tools-dock
//     .chat-col-gutter  the two focalisation controls, hidden by no state
//     .chat-col--right  .chat-output     + chat-resizer     + .chat-input
//
// Both splitters take the *column* as their shell, so each measures the
// column it lives in and sizes the pane after its bar: the composer row grows
// upwards, the dock grows downwards.
//
// Three controls take elements out of the flow instead of sizing them: the
// dock's fold (its bar goes with it), and the two focalisations the gutter
// carries. `focus` picks which surface owns the screen and reaches the DOM as
// one state class on `.chat-shell` (`chat-shell--reasoning`,
// `chat-shell--conversation`) — the rules keyed on that name move a whole
// column, and no state reads or writes a splitter's stored size, so a dragged
// dock and a dragged composer come back exactly as they were left (see
// `dockCollapsed` below: a flag, not a size).
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

// Which of the two surfaces owns the screen. Three values on one axis, so
// "both at once, screen empty" is not a state this composable can be in (D1):
// the exclusivity lives in the type, not in a rule the CSS would have to keep.
export const FOCUS_NORMAL = 'normal'
export const FOCUS_REASONING = 'reasoning'
export const FOCUS_CONVERSATION = 'conversation'
const FOCUS_VALUES = [FOCUS_NORMAL, FOCUS_REASONING, FOCUS_CONVERSATION]

// `conversation` is a furnishing preference, of the same order as the dock
// fold, so it persists. `reasoning` deliberately does not (D4): reloading into
// it would boot the app with no composer — a screen with no way to speak to
// the model.
const LEFT_HIDDEN_KEY = 'benchmarkit:chatLeftHidden'

function loadLeftHidden() {
  try {
    return window.localStorage.getItem(LEFT_HIDDEN_KEY) === '1'
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

  // One owner arbitrates the axis (D1). Shape: `view` holds only the
  // non-persisted half ('normal' | 'reasoning'), and `conversation` is
  // *derived* from the persisted flag — so no caller can observe `reasoning`
  // and a hidden left column at the same time, and wf-03 sizes from `focus`.
  const hidden = ref(loadLeftHidden())
  const view = ref(FOCUS_NORMAL)
  const focus = computed(() => (hidden.value ? FOCUS_CONVERSATION : view.value))

  // Every change of the axis goes through this one setter: it moves the flag,
  // moves `view`, and writes the flag through — so `toggleLeftHidden` is
  // allowed to be a shortcut for it rather than a second decision.
  function setFocus(next) {
    if (!FOCUS_VALUES.includes(next)) return
    hidden.value = next === FOCUS_CONVERSATION
    view.value = next === FOCUS_REASONING ? FOCUS_REASONING : FOCUS_NORMAL
    try {
      window.localStorage.setItem(LEFT_HIDDEN_KEY, hidden.value ? '1' : '0')
    } catch {
      // Storage disabled: the layout still works for the session.
    }
  }

  // The control's own gesture: a second click on it restores, and from
  // `conversation` it uncovers the column and maximises in the same step —
  // the reasoning panel is not on screen while the column is out of the flow.
  function toggleFocus() {
    setFocus(focus.value === FOCUS_REASONING ? FOCUS_NORMAL : FOCUS_REASONING)
  }

  function toggleLeftHidden() {
    setFocus(hidden.value ? FOCUS_NORMAL : FOCUS_CONVERSATION)
  }

  // Read-only on purpose: writing the flag directly would bypass the
  // arbitration above and could persist a state the axis cannot represent.
  const leftHidden = computed(() => hidden.value)

  return {
    column,
    composer,
    dockCollapsed,
    focus,
    setFocus,
    toggleFocus,
    leftHidden,
    toggleLeftHidden,
  }
}
