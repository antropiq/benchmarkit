import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

// jsdom has no ResizeObserver, and both the chat splitters and the reasoning
// panel create one on mount. A no-op is enough here: what this file asserts is
// that the gutter exists, is reachable and drives wf-01's axis — never a size.
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
}

import WorkBenchView from '../src/components/WorkBenchView.vue'
import {
  FOCUS_CONVERSATION,
  FOCUS_NORMAL,
  FOCUS_REASONING,
} from '../src/composables/useChatLayout'
import { useChat } from '../src/composables/useChat'

// The gutter is asserted through the real `WorkBenchView` rather than a
// component that only *could* render it: what matters is that the shell has a
// third child in the right place, wired to the composable's own state. That
// mount is cheap in jsdom (one panel with no lines, an inert composer), so no
// extraction is needed to keep it honest.
//
// The axis is read back off the mounted view (`wrapper.vm.focus`). Asking a
// second `useChatLayout()` for it would be wrong: the composable is a factory,
// not a singleton — only `localStorage` is shared — so a fresh call reports
// its own initial state and would never see what the clicks did.
const { state: chat, chatBusy } = useChat()

// `chatBusy` is the Live View's lock. The streaming half of it is `state.phase`,
// which is reactive and therefore what the button's `:disabled` re-evaluates on,
// so a case arms that one rather than a reply (an LLM call would make layout
// flaky) or the batch guard (a plain closure nothing re-reads).
function streamReply(on) {
  chat.phase = on ? 'reasoning' : 'idle'
}

function hubs({ mcp = false, lsp = false }) {
  chat.mcp.loaded = mcp
  chat.mcp.connectors = mcp ? [{ id: 'console', label: 'Console', on: true }] : []
  chat.lsp.loaded = lsp
  chat.lsp.servers = lsp ? [{ id: 'pyright', label: 'Pyright', on: true }] : []
}

function gutterOf(wrapper) {
  return wrapper.find('.chat-col-gutter')
}

function buttons(wrapper) {
  return gutterOf(wrapper).findAll('button')
}

beforeEach(() => {
  window.localStorage.clear()
  hubs({ mcp: true, lsp: true })
  // The Live View is singleton state, so a case that switched it leaves it
  // switched for the next one — the default on (live view) is what every other
  // case below reads the control's name against. The streaming flag is reset
  // the same way, so a case that armed it cannot leave the strip locked.
  chat.liveView = true
  streamReply(false)
})

afterEach(() => {
  window.localStorage.clear()
  hubs({})
  chat.liveView = true
  streamReply(false)
})

describe('the focus gutter', () => {
  it('is a third child of the shell, between the two columns', () => {
    const w = mount(WorkBenchView)
    const gutter = gutterOf(w)
    expect(gutter.exists()).toBe(true)

    // The first three children of `.chat-shell` in document order — the
    // lightbox `<dialog>` follows and is none of this task's business.
    const classes = Array.from(w.find('.chat-shell').element.children)
      .map((el) => el.className)
      .slice(0, 3)
    expect(classes).toEqual([
      'chat-col chat-col--left',
      'chat-col-gutter',
      'chat-col chat-col--right',
    ])
  })

  // The strip is three controls now: the two focalisations and, under them, the
  // Live View of the reasoning timeline — spec §3.1's "under the buttons that
  // are in the second column", and the same 28px target as the two above.
  it('renders exactly three buttons, focus on top, in that order', () => {
    const w = mount(WorkBenchView)
    const btns = buttons(w)
    expect(btns).toHaveLength(3)

    for (const btn of btns) {
      // Real <button type="button">s: that is what makes them tab-reachable
      // and keeps them from acting as submits. (Tab order itself is not
      // asserted: jsdom does not implement sequential focus navigation.)
      expect(btn.attributes().type).toBe('button')
      expect(btn.attributes()['aria-label']).toBeTruthy()
      // One size for all three, so a control under the other two cannot be
      // smaller than them.
      expect(btn.classes()).toEqual(['chat-gutter-btn'])
    }

    expect(btns[0].attributes()['aria-label']).toBe('Show the reasoning only')
    expect(btns[1].attributes()['aria-label']).toBe('Hide the tool column')
    // The third control's name says what the click does, never what was taken
    // out of the stream — and the live view is on at load, so the click is the
    // one that gives the history back.
    expect(btns[2].attributes()['aria-label']).toBe(
      'Show the whole reasoning history'
    )
  })

  it('drives the focus axis, and its aria-pressed follows', async () => {
    const w = mount(WorkBenchView)
    const [focusBtn, hideBtn] = buttons(w)

    expect(w.vm.focus).toBe(FOCUS_NORMAL)
    expect(focusBtn.attributes()['aria-pressed']).toBe('false')

    await focusBtn.trigger('click')
    expect(w.vm.focus).toBe(FOCUS_REASONING)
    expect(buttons(w)[0].attributes()['aria-pressed']).toBe('true')
    expect(buttons(w)[0].attributes()['aria-label']).toBe('Restore the tool column')

    await buttons(w)[0].trigger('click')
    expect(w.vm.focus).toBe(FOCUS_NORMAL)
    expect(w.vm.leftHidden).toBe(false)

    // The hide control owns the third value, and persists it (D4).
    await buttons(w)[1].trigger('click')
    expect(w.vm.focus).toBe(FOCUS_CONVERSATION)
    expect(w.vm.leftHidden).toBe(true)
    expect(buttons(w)[1].attributes()['aria-pressed']).toBe('true')
    expect(window.localStorage.getItem('benchmarkit:chatLeftHidden')).toBe('1')

    await buttons(w)[1].trigger('click')
    expect(w.vm.focus).toBe(FOCUS_NORMAL)
    expect(w.vm.leftHidden).toBe(false)
  })

  it('starts in the state storage describes, not a fresh one', async () => {
    window.localStorage.setItem('benchmarkit:chatLeftHidden', '1')
    const w = mount(WorkBenchView)
    const btns = buttons(w)
    expect(w.vm.focus).toBe(FOCUS_CONVERSATION)
    expect(btns[1].attributes()['aria-pressed']).toBe('true')
    expect(btns[1].attributes()['aria-label']).toBe('Show the tool column')

    await btns[1].trigger('click')
    expect(w.vm.focus).toBe(FOCUS_NORMAL)
    expect(window.localStorage.getItem('benchmarkit:chatLeftHidden')).toBe('0')
  })

  // The `chat-col--single` case (one hub or none): a gutter that vanished with
  // the dock would strand the user in a column they cannot get out of.
  it('survives a shell with no hubs at all', () => {
    hubs({})
    const w = mount(WorkBenchView)
    expect(w.find('.chat-col--left.chat-col--single').exists()).toBe(true)
    expect(buttons(w)).toHaveLength(3)
  })

  // The lock of §3.2: the control is inert while an inference owns the
  // conversation, and live again when it does not. `chatBusy` is the composable's
  // rule (a streaming reply or a batch), so this is the same verdict every other
  // inert control answers to — here armed through useBatch's guard, because a
  // case that streamed a reply would need an LLM and turn layout into flaky.
  it('locks the Live View while an inference runs, and releases it', async () => {
    streamReply(true)
    const w = mount(WorkBenchView)
    const liveBtn = () => buttons(w)[2]
    // The disabled attribute is the lock; the click being answered is not, so
    // the toggle is what the flag below reports. A tick first: the attribute is
    // a reactive binding, so it re-renders on the microtask after the flag.
    await nextTick()
    expect('disabled' in liveBtn().attributes()).toBe(true)
    expect(chatBusy(), 'the lock reads the composable rule').toBe(true)

    await liveBtn().trigger('click')
    // The click is answered and the flag never moves: the mode cannot be
    // switched mid-flux — and it is the load-time live view that stays put.
    expect(chat.liveView).toBe(true)

    streamReply(false)
    await nextTick()
    expect('disabled' in liveBtn().attributes()).toBe(false)
    expect(chatBusy()).toBe(false)

    await liveBtn().trigger('click')
    expect(chat.liveView).toBe(false)
    expect(liveBtn().attributes()['aria-pressed']).toBe('false')
    expect(liveBtn().attributes()['aria-label']).toBe(
      'Keep only the reasoning on screen'
    )
  })
})

// The geometry is four CSS rules (`.chat-shell--reasoning`,
// `.chat-shell--conversation`), and jsdom neither loads the theme nor lays
// anything out — the real widths are the Playwright spec's. What is asserted
// here is the other half of the contract: the class the CSS keys on is on the
// shell, in exactly one of the three states, so no rule can match two states at
// once, and the marks that say which state is active change with it.
describe('the state classes on the shell', () => {
  function shellClasses(w) {
    return w.find('.chat-shell').element.className
  }

  it('carries exactly one state class, in each of the three states', async () => {
    const w = mount(WorkBenchView)
    const [focusBtn, hideBtn] = buttons(w)

    // `normal` — neither class: the columns keep their own rules.
    expect(shellClasses(w)).not.toContain('chat-shell--reasoning')
    expect(shellClasses(w)).not.toContain('chat-shell--conversation')

    await focusBtn.trigger('click')
    expect(shellClasses(w)).toContain('chat-shell--reasoning')
    expect(shellClasses(w)).not.toContain('chat-shell--conversation')

    await focusBtn.trigger('click')
    expect(shellClasses(w)).not.toContain('chat-shell--reasoning')

    // The pair is the DOM mirror of D1: one axis, so the two names never
    // appear together and the screen can never be emptied by both rules.
    await hideBtn.trigger('click')
    expect(shellClasses(w)).toContain('chat-shell--conversation')
    expect(shellClasses(w)).not.toContain('chat-shell--reasoning')

    // From `conversation`, the focus control maximises in one step and the
    // column comes back with it.
    await buttons(w)[0].trigger('click')
    expect(shellClasses(w)).toContain('chat-shell--reasoning')
    expect(shellClasses(w)).not.toContain('chat-shell--conversation')
  })

  // `chat-col--single` is the column's own variant and a state never rewrites
  // it: both classes sit on the element at once and the CSS resolves them by
  // order (the state rules come later in theme.css).
  it('leaves chat-col--single alone when it enters reasoning', async () => {
    hubs({})
    const w = mount(WorkBenchView)
    await buttons(w)[0].trigger('click')

    expect(shellClasses(w)).toContain('chat-shell--reasoning')
    const left = w.find('.chat-col--left').element.className
    expect(left).toContain('chat-col--single')
  })

  // D8: exactly one element carries the state — the shell — so each state rule
  // has one ancestor to match against, and the columns never gain a class of
  // their own that a later rule would have to out-specify.
  it('puts the state on the shell and nowhere else', async () => {
    const w = mount(WorkBenchView)
    await buttons(w)[0].trigger('click')

    const carriesState = (n) =>
      n.classes().some((c) => c.startsWith('chat-shell--'))
    const parts = [
      w.find('.chat-shell'),
      w.find('.chat-col--left'),
      w.find('.chat-col--right'),
      gutterOf(w),
    ]
    const marked = parts.filter(carriesState)
    expect(marked).toHaveLength(1)
    expect(marked[0].classes()).toContain('chat-shell')
  })

  // The Live View's other half of the same contract: the panel's scroller is
  // the element that carries the mode, and it is the only one. `theme.css`
  // turns its `overflow-y` off on `.chat-tool-scroll--live`, so a scroller
  // without that class in live mode grows a vertical bar the moment the flow
  // is taller than the box — the reader could then drag the pruned head back
  // and end up off the bottom of the thinking. The real clipped geometry is the
  // Playwright spec's half (`.chat-tool-scroll--live` resolving to
  // `overflow-y: hidden`); what is pinned here is that the class the CSS keys
  // on is on that scroller, and moves with the mode.
  it('puts the live class on the panel scroller, and only in live mode', async () => {
    const w = mount(WorkBenchView)
    const scrollers = w.findAll('.chat-tool-scroll')
    // The scan must be seeing the panel: an empty result would otherwise pass
    // every case below for the wrong reason.
    expect(scrollers, 'the panel renders its scroller').toHaveLength(1)

    // On is the default: the panel holds only what its box can, so it clips.
    expect(scrollers[0].classes()).toContain('chat-tool-scroll')
    expect(scrollers[0].classes()).toContain('chat-tool-scroll--live')

    await buttons(w)[2].trigger('click')
    expect(chat.liveView).toBe(false)
    expect(w.find('.chat-tool-scroll').classes()).not.toContain('chat-tool-scroll--live')

    await buttons(w)[2].trigger('click')
    expect(chat.liveView).toBe(true)
    expect(w.find('.chat-tool-scroll').classes()).toContain('chat-tool-scroll--live')
  })
})

// The marks are glyphs of one generated Nerd face (see WorkBenchFocusGutter and
// the `UI_GLYPHS` table of tools/nerd_icons.py), so what is asserted here is the
// state in the character, not a drawn shape: whether a private-use code point is
// *displayed* is a font matter the browser layer answers (the Playwright spec
// measures the mark's box).
describe('the glyph marks', () => {
  function marks(wrapper) {
    return wrapper.findAll('.chat-gutter-mark')
  }

  it('draws the three marks as aria-hidden glyphs in labelled buttons', () => {
    const w = mount(WorkBenchView)
    const all = marks(w)
    expect(all).toHaveLength(3)
    for (const mark of all) {
      expect(mark.attributes()['aria-hidden']).toBe('true')
      // A private-use code point: a screen reader gets the button's label, never
      // the character, and the generated face is what gives the code point a
      // glyph at all.
      expect(mark.text()).toMatch(/^[\ue000-\uf8ff]$/)
    }
  })

  // D11: "restore" must not reuse the "maximise" motif, so the focus control
  // swaps fa-expand for fa-compress rather than losing one of two shapes.
  it('shows the restore mark once the reasoning owns the screen', async () => {
    const w = mount(WorkBenchView)
    const focusMark = () => marks(w)[0].text()
    const maximise = focusMark()

    await buttons(w)[0].trigger('click')
    const restore = focusMark()
    expect(restore).not.toBe(maximise)
    // One glyph each way: the box cannot move, only its face.
    expect(restore).toHaveLength(maximise.length)

    await buttons(w)[0].trigger('click')
    expect(focusMark()).toBe(maximise)
  })

  it('points the chevron at what the click will do', async () => {
    const w = mount(WorkBenchView)
    const chevron = () => marks(w)[1].text()

    // Column on screen: it is the one that will fold away, so the mark is
    // fa-chevron_left (U+F053).
    expect(chevron()).toBe('\uf053')

    await buttons(w)[1].trigger('click')
    // Column gone: the same control brings it back, so it is fa-chevron_right.
    expect(chevron()).toBe('\uf054')

    await buttons(w)[1].trigger('click')
    expect(chevron()).toBe('\uf053')
  })

  // The eye of the Live View keeps its own pair, and the strip is one face: the
  // code points of all three marks are the ones `UI_GLYPHS` subsets (asserted by
  // `python tools/nerd_icons.py check`, run from tests/test_file_icons.py).
  it('keeps the Live View mark on the same generated face', async () => {
    const w = mount(WorkBenchView)
    expect(marks(w)[2].text()).toBe('\uf070')

    await buttons(w)[2].trigger('click')
    expect(marks(w)[2].text()).toBe('\uf06e')
  })

  // The load-time default is the closed eye. `useChat`'s state is a singleton
  // that every case above has switched, so the app's own starting value is read
  // where it is declared — the shipped initial state, not what this process
  // happens to hold by now.
  it('loads with the live view on', () => {
    const source = readFileSync(
      join(resolve(), 'src', 'composables', 'useChat.js'),
      'utf8',
    )
    expect(source).toMatch(/liveView:\s*true,/)
  })
})
// wf-04's sweep — what the two tasks that shipped this could not see while
// each was in flight: a whole walk rather than one hop per case, and the walk
// read from the mounted shell (the state class *and* the pair behind it), so a
// class that lags one render behind the composable cannot pass.
describe('a walk through the shell, control by control', () => {
  function stateClasses(w) {
    return w
      .find('.chat-shell')
      .element.className.split(/\s+/)
      .filter((cls) => cls.startsWith('chat-shell--'))
  }

  // `normal` carries neither name, so the expected list is one per state.
  const SHELL_OF = {
    [FOCUS_NORMAL]: [],
    [FOCUS_REASONING]: ['chat-shell--reasoning'],
    [FOCUS_CONVERSATION]: ['chat-shell--conversation'],
  }

  it('shows exactly one state class at each step of a full cycle', async () => {
    const w = mount(WorkBenchView)
    // [which control, the state it must land on]: 0 is the focus control, 1
    // the hide control — normal → reasoning → normal → conversation →
    // reasoning → conversation → normal, always by the gesture a user has.
    const steps = [
      [0, FOCUS_REASONING],
      [0, FOCUS_NORMAL],
      [1, FOCUS_CONVERSATION],
      [0, FOCUS_REASONING],
      [1, FOCUS_CONVERSATION],
      [1, FOCUS_NORMAL],
    ]

    expect(stateClasses(w)).toEqual(SHELL_OF[FOCUS_NORMAL])
    for (const [which, expected] of steps) {
      await buttons(w)[which].trigger('click')
      // D1 on the DOM: never two names, never the wrong one.
      expect(stateClasses(w)).toEqual(SHELL_OF[expected])
      expect(w.vm.focus).toBe(expected)
      expect(w.vm.leftHidden).toBe(expected === FOCUS_CONVERSATION)
    }
  })

  // The gutter is the one surface no state hides, and it must not depend on
  // the dock either: with one hub or none the column is `--single`, the dock is
  // a bare strip, and the way back still has to be on screen in all three
  // states — a control that vanished with the panels would strand a column.
  it('keeps both controls in every state of a single-panel shell', async () => {
    for (const hub of [{ mcp: true }, { lsp: true }, {}]) {
      hubs(hub)
      window.localStorage.clear()
      const w = mount(WorkBenchView)
      expect(w.find('.chat-col--left').classes()).toContain('chat-col--single')

      // 0 is the focus control, 1 the hide control: reasoning by the upper
      // one, conversation and the way back to normal by the chevron.
      for (const [which, expected] of [
        [0, FOCUS_REASONING],
        [1, FOCUS_CONVERSATION],
        [1, FOCUS_NORMAL],
      ]) {
        await buttons(w)[which].trigger('click')
        expect(stateClasses(w)).toEqual(SHELL_OF[expected])
        expect(gutterOf(w).exists()).toBe(true)
        // Three controls survive every state of the narrow shell: the Live
        // View's switch rides the same strip, so a column that left the flow
        // cannot leave the switch behind either.
        expect(buttons(w)).toHaveLength(3)
      }
      w.unmount()
    }
  })
})

// D2's premise, asserted where it can be seen. jsdom neither loads the theme
// nor lays anything out, so it cannot answer "is this hidden by CSS?" — but it
// can read the shipped template, and a `v-if` on one of these elements is the
// mistake that would silently unmount the reasoning panel (line cache and
// ResizeObserver included) every time a state is entered. The real geometry is
// the Playwright spec's half of the same contract.
describe('nothing the states hide is unmounted', () => {
  const source = readFileSync(
    join(resolve(), 'src', 'components', 'WorkBenchView.vue'),
    'utf8',
  )

  // Every opening tag of the template, its attributes as one string. A `>`
  // inside an attribute value is the only thing that would fool this scan, and
  // the template has none — a conditional that this scan cannot see costs a
  // passing test, never a broken app, so the shape is stated rather than hid.
  const tags = [...source.matchAll(/<[A-Za-z][\w-]*\s[^>]*?>/gs)].map(
    (match) => match[0],
  )

  // The gutter, both columns and the panel: one of each in the shell, and
  // none of them conditional. `ref` is how the two column divs are named —
  // they are the only elements carrying it.
  for (const [name, isTag] of [
    ['the gutter', (tag) => tag.includes('<WorkBenchFocusGutter')],
    ['the reasoning panel', (tag) => tag.includes('<WorkBenchToolPanel')],
    ['the left column', (tag) => tag.includes('ref="leftCol"')],
    ['the right column', (tag) => tag.includes('ref="rightCol"')],
  ]) {
    it(`${name} is rendered unconditionally`, () => {
      const found = tags.filter(isTag)
      // The scan itself must be seeing the template: an empty result would
      // otherwise pass every case below for the wrong reason.
      expect(found, `${name} is not in the template`).toHaveLength(1)
      for (const tag of found) {
        expect(tag).not.toMatch(/\sv-if=/)
        expect(tag).not.toMatch(/\sv-show=/)
      }
    })
  }
})
