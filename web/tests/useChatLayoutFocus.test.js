import { afterEach, describe, expect, it } from 'vitest'
import { ref } from 'vue'

import {
  FOCUS_CONVERSATION,
  FOCUS_NORMAL,
  FOCUS_REASONING,
  useChatLayout,
} from '../src/composables/useChatLayout'

// The screen's focalisation is one axis of three values (D1), and this file is
// where that is pinned down — no CSS and no template are involved yet, only the
// state the later tasks render. The splitters are part of the same composable,
// so their storage keys are asserted here too: a focus state that resized
// anything would silently destroy a dragged height (D3).
//
// The four arguments are the column/input/dock refs `useChatLayout` hands to
// `useSplitter`; null refs make both splitters inert (no shell to measure),
// which is what the tests want — the axis must not need a laid-out DOM.
// (The direct call does print Vue's "onMounted without a component instance"
// warning for those two splitters: harmless here, and the alternative — a
// throwaway component per test — would add a mount to every case without
// testing anything the axis owns.)
function layout() {
  return useChatLayout(ref(null), ref(null), ref(null), ref(null))
}

function pair(l) {
  return [l.focus.value, l.leftHidden.value]
}

const SPLIT_KEYS = ['benchmarkit:chatToolsColumn', 'benchmarkit:chatSplit']
const HIDDEN_KEY = 'benchmarkit:chatLeftHidden'
const COLLAPSE_KEY = 'benchmarkit:chatToolsCollapsed'

const STATES = [FOCUS_NORMAL, FOCUS_REASONING, FOCUS_CONVERSATION]

// What each state looks like from the outside, so the exclusivity table is one
// lookup rather than a hand-written expectation per transition.
const SHAPE = {
  [FOCUS_NORMAL]: [FOCUS_NORMAL, false],
  [FOCUS_REASONING]: [FOCUS_REASONING, false],
  [FOCUS_CONVERSATION]: [FOCUS_CONVERSATION, true],
}

let storageDescriptor = null

// `restoreMocks` does not undo a redefined `window.localStorage`, so this file
// owns the undo — otherwise the next file in the run inherits a throwing get.
function disableStorage() {
  storageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage')
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    get() {
      throw new Error('storage disabled')
    },
  })
}

afterEach(() => {
  if (storageDescriptor) {
    Object.defineProperty(window, 'localStorage', storageDescriptor)
    storageDescriptor = null
  }
  window.localStorage.clear()
})

describe('the focus axis', () => {
  it('reaches the three values and round-trips through the control', () => {
    const l = layout()
    expect(l.focus.value).toBe(FOCUS_NORMAL)

    l.toggleFocus()
    expect(l.focus.value).toBe(FOCUS_REASONING)
    l.toggleFocus()
    expect(l.focus.value).toBe(FOCUS_NORMAL)

    l.setFocus(FOCUS_CONVERSATION)
    expect(l.focus.value).toBe(FOCUS_CONVERSATION)
  })

  // All six ordered pairs: from any state to any other, the pair that comes
  // out is the target's own shape — never a leftover of the one left behind.
  it('never leaves two focalisations on at once', () => {
    for (const from of STATES) {
      for (const to of STATES) {
        if (from === to) continue
        window.localStorage.clear()
        const l = layout()
        l.setFocus(from)
        expect(pair(l)).toEqual(SHAPE[from])
        l.setFocus(to)
        expect(pair(l)).toEqual(SHAPE[to])
      }
    }
  })

  it('uncovers the column and maximises in one step from conversation', () => {
    const l = layout()
    l.setFocus(FOCUS_CONVERSATION)
    l.toggleFocus()
    expect(pair(l)).toEqual(SHAPE[FOCUS_REASONING])
  })

  it('ignores a value that is not on the axis', () => {
    const l = layout()
    l.toggleFocus()
    for (const junk of ['nonsense', undefined, null, '', 0, FOCUS_REASONING]) {
      l.setFocus(junk)
      expect(l.focus.value).toBe(FOCUS_REASONING)
      expect(l.leftHidden.value).toBe(false)
    }
  })
})

describe('persistence of the axis', () => {
  it('keeps conversation across a re-mount, and reasoning not', () => {
    const first = layout()
    first.toggleLeftHidden()
    expect(first.leftHidden.value).toBe(true)

    const second = layout()
    expect(second.leftHidden.value).toBe(true)
    expect(second.focus.value).toBe(FOCUS_CONVERSATION)

    const maximised = layout()
    maximised.setFocus(FOCUS_REASONING)
    expect(layout().focus.value).toBe(FOCUS_NORMAL)
  })

  it('flips the flag through the control and writes it through', () => {
    const l = layout()
    l.toggleLeftHidden()
    expect(window.localStorage.getItem(HIDDEN_KEY)).toBe('1')
    l.toggleLeftHidden()
    expect(window.localStorage.getItem(HIDDEN_KEY)).toBe('0')
    expect(pair(l)).toEqual(SHAPE[FOCUS_NORMAL])
  })

  it('leaves the splitter keys alone', () => {
    window.localStorage.setItem(SPLIT_KEYS[0], '320')
    window.localStorage.setItem(SPLIT_KEYS[1], '140')

    const l = layout()
    for (const from of STATES) {
      for (const to of STATES) {
        l.setFocus(from)
        l.setFocus(to)
        l.toggleFocus()
        l.toggleLeftHidden()
      }
    }

    expect(window.localStorage.getItem(SPLIT_KEYS[0])).toBe('320')
    expect(window.localStorage.getItem(SPLIT_KEYS[1])).toBe('140')
  })

  it('stays independent of the dock fold', () => {
    const l = layout()
    l.dockCollapsed.value = true
    expect(window.localStorage.getItem(COLLAPSE_KEY)).toBe('1')

    const before = l.dockCollapsed.value
    l.toggleFocus()
    l.toggleLeftHidden()
    l.toggleFocus()
    l.setFocus(FOCUS_NORMAL)

    expect(l.dockCollapsed.value).toBe(before)
    expect(window.localStorage.getItem(COLLAPSE_KEY)).toBe('1')
    // The dock's own fold never decided the focalisation, and vice versa.
    expect(pair(l)).toEqual(SHAPE[FOCUS_NORMAL])
  })

  it('works for the session when storage throws', () => {
    disableStorage()
    const l = layout()
    expect(l.focus.value).toBe(FOCUS_NORMAL)

    l.toggleFocus()
    expect(l.focus.value).toBe(FOCUS_REASONING)
    l.toggleLeftHidden()
    expect(pair(l)).toEqual(SHAPE[FOCUS_CONVERSATION])
    l.toggleFocus()
    expect(pair(l)).toEqual(SHAPE[FOCUS_REASONING])
    l.setFocus(FOCUS_NORMAL)
    expect(pair(l)).toEqual(SHAPE[FOCUS_NORMAL])
    l.dockCollapsed.value = true
    expect(l.dockCollapsed.value).toBe(true)
  })
})
