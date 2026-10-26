import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { inferenceRunning } from '../src/lib/inference-state'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useBatch } from '../src/composables/useBatch'
import { useChat } from '../src/composables/useChat'
import { playSessionEnded, useSessionSound } from '../src/composables/useSessionSound'

// Layer 1 (jsdom): when the end-of-session chime rings. The rule itself is the
// busy -> idle edge of `inferenceRunning()` — the same predicate the ⚡/✨ tab
// glyph uses — so both are asserted here instead of in a browser: the only
// browser-specific part (the autoplay gate) cannot be reproduced in jsdom.

// jsdom implements no media playback, so Audio is replaced with a recorder:
// the assertion is about *whether and how often* play() is reached, not about
// sound coming out of a speaker.
const played = []

class FakeAudio {
  constructor(src) {
    this.src = src
    this.volume = 1
    this.preload = 'none'
    this.muted = false
    this.currentTime = 0
  }
  play() {
    played.push({ src: this.src, muted: this.muted, volume: this.volume })
    return Promise.resolve()
  }
  pause() {}
}

vi.stubGlobal('Audio', FakeAudio)

const benchmark = useBenchmark().state
const chat = useChat().state
const batch = useBatch().state
const sound = useSessionSound()

function reset() {
  benchmark.status = 'idle'
  chat.phase = 'idle'
  batch.active = false
  sound.state.enabled = true
  played.length = 0
}

async function settle() {
  await nextTick()
  await nextTick()
}

describe('inferenceRunning: one predicate, two indicators', () => {
  it('is idle only when no run, no lot and no reply is in flight', () => {
    expect(
      inferenceRunning({ benchmark: { status: 'idle' }, chat: { phase: 'idle' }, batch: { active: false } }),
    ).toBe(false)
  })

  it('is busy for a running benchmark even with an idle WorkBench', () => {
    expect(
      inferenceRunning({ benchmark: { status: 'running' }, chat: { phase: 'idle' }, batch: { active: false } }),
    ).toBe(true)
  })

  it('stays busy across a batch turn gap', () => {
    expect(
      inferenceRunning({ benchmark: { status: 'idle' }, chat: { phase: 'idle' }, batch: { active: true } }),
    ).toBe(true)
  })
})

describe('useSessionSound: rings on the busy -> idle edge only', () => {
  beforeEach(reset)

  it('chimes when a streamed reply ends', async () => {
    chat.phase = 'answering'
    await settle()
    expect(played).toHaveLength(0)

    chat.phase = 'idle'
    await settle()
    expect(played).toHaveLength(1)
    expect(played[0].muted).toBe(false)
    expect(played[0].src).toContain('session_ended')
  })

  it('chimes once for a whole benchmark run', async () => {
    benchmark.status = 'running'
    await settle()
    benchmark.status = 'finished'
    await settle()
    expect(played).toHaveLength(1)
  })

  it('does not chime between two batch task turns', async () => {
    batch.active = true
    chat.phase = 'answering'
    await settle()
    chat.phase = 'idle' // task N done, another owed
    await settle()
    expect(played).toHaveLength(0)

    batch.active = false // lot over
    await settle()
    expect(played).toHaveLength(1)
  })

  it('stays silent while muted', async () => {
    sound.toggle()
    expect(sound.state.enabled).toBe(false)
    played.length = 0

    chat.phase = 'answering'
    await settle()
    chat.phase = 'idle'
    await settle()
    expect(played).toHaveLength(0)
  })

  it('is silent on boot: no chime without a preceding inference', async () => {
    await settle()
    expect(played).toHaveLength(0)
    playSessionEnded()
    expect(played).toHaveLength(1)
  })
})
