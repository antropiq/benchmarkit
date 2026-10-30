import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

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
const chat = useChat().state

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
})

afterEach(() => {
  window.localStorage.clear()
  hubs({})
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

  it('renders exactly two buttons, focus on top, in that order', () => {
    const w = mount(WorkBenchView)
    const btns = buttons(w)
    expect(btns).toHaveLength(2)

    for (const btn of btns) {
      // Real <button type="button">s: that is what makes them tab-reachable
      // and keeps them from acting as submits. (Tab order itself is not
      // asserted: jsdom does not implement sequential focus navigation.)
      expect(btn.attributes().type).toBe('button')
      expect(btn.attributes()['aria-label']).toBeTruthy()
    }

    expect(btns[0].attributes()['aria-label']).toBe('Show the reasoning only')
    expect(btns[1].attributes()['aria-label']).toBe('Hide the tool column')
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
    expect(buttons(w)).toHaveLength(2)
  })
})
