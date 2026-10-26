import { beforeEach, describe, expect, it } from 'vitest'

import { useSessionStats } from '../src/composables/useSessionStats'
import { useBenchmark } from '../src/composables/useBenchmark'
import { useBatch } from '../src/composables/useBatch'
import { useChat } from '../src/composables/useChat'

// Layer 1 (jsdom): the rules behind the app bar's shared meters — which session
// they describe, and when they read blank. The clock and the two counters are a
// composable rather than computeds inside AppBar because the Benchmark tab is
// meant to read the same figures (its own panel is a separate view), and the
// app can only ever have one session at a time (a run blocks WorkBench and a
// streaming reply blocks a run). What is asserted here is the resolution, not
// the pixels — the app bar's look stays an e2e concern.

const stats = useSessionStats()
const benchmark = useBenchmark().state
const batch = useBatch().state
const chat = useChat().state

function reset() {
  benchmark.status = 'idle'
  benchmark.clock = { mode: 'idle', baseMs: 0, startedAt: 0 }
  benchmark.tokenRate = 0
  benchmark.entries = []
  benchmark.config = { contextWindow: 8192 }
  batch.active = false
  batch.status = 'idle'
  chat.phase = 'idle'
  chat.messages = []
  chat.session.startedAt = 0
}

function runningChat() {
  chat.phase = 'answering'
  chat.session.startedAt = 1000
}

describe('useSessionStats: one clock and two counters, whichever session is live', () => {
  beforeEach(reset)

  it('reads blank with no run, no lot and no reply', () => {
    expect(stats.source.value).toBe(null)
    expect(stats.running.value).toBe(false)
    expect(stats.elapsedMs(5000)).toBe(0)
    expect(stats.rateText.value).toBe('— t/s')
    expect(stats.contextText.value).toBe('— ctx')
  })

  it('goes blank again when the session ends, instead of keeping the figures', () => {
    runningChat()
    chat.messages = [
      {
        role: 'assistant',
        streaming: false,
        tokens: 100,
        took: 4,
        ttft: 1,
        decode: 3,
        contextTokens: 4000,
        tokensEstimated: false,
        contextEstimated: false,
      },
    ]
    expect(stats.rateText.value).toBe('33.3 t/s')
    expect(stats.contextText.value).toBe('ctx 4k / 8.2k')

    chat.phase = 'idle'
    chat.session.startedAt = 0
    expect(stats.rateText.value).toBe('— t/s')
    expect(stats.contextText.value).toBe('— ctx')
  })

  it('counts a chat session from its startedAt, tool rounds included', () => {
    runningChat()
    expect(stats.source.value).toBe('chat')
    expect(stats.elapsedMs(4500)).toBe(3500)
    expect(stats.formatElapsed(stats.elapsedMs(4500))).toBe('00:00:03')
  })

  it('keeps a batch lot between two task turns, when the phase is back to idle', () => {
    runningChat()
    chat.phase = 'idle'
    batch.active = true
    expect(stats.source.value).toBe('chat')
    expect(stats.elapsedMs(3000)).toBe(2000)
  })

  it('prefers a benchmark run, and follows its accumulator and phases', () => {
    runningChat()
    benchmark.status = 'running'
    benchmark.clock = { mode: 'ticking', baseMs: 30_000, startedAt: 2000 }
    expect(stats.source.value).toBe('run')
    expect(stats.clockPlaceholder()).toBe(null)
    expect(stats.elapsedMs(8000)).toBe(36000)

    // Warm-up and the verifier call hold no readable elapsed time.
    benchmark.clock = { mode: 'starting', baseMs: 0, startedAt: 0 }
    expect(stats.clockPlaceholder()).toBe('Starting…')
    expect(stats.elapsedMs(8000)).toBe(0)
    benchmark.clock = { mode: 'checking', baseMs: 36_000, startedAt: 0 }
    expect(stats.clockPlaceholder()).toBe('Checking…')
    expect(stats.elapsedMs(8000)).toBe(36000)
  })

  it('reads the run rate only once the first question is over', () => {
    benchmark.status = 'running'
    expect(stats.rateText.value).toBe('— t/s')
    benchmark.tokenRate = 21.5
    expect(stats.rateText.value).toBe('21.5 t/s')
  })

  it('reads the context of the question in progress for a run', () => {
    benchmark.status = 'running'
    benchmark.entries = [
      { type: 'question', consumed: 1200, tokensEstimated: false },
      { type: 'question', consumed: 0, tokensEstimated: false },
    ]
    expect(stats.contextText.value).toBe('ctx 1.2k / 8.2k')
    expect(stats.contextClass.value).toBe('green')
  })

  it('flags the fill colour as the window fills up', () => {
    runningChat()
    chat.messages = [
      {
        role: 'assistant',
        streaming: false,
        tokens: 10,
        took: 2,
        ttft: 1,
        decode: 1,
        contextTokens: 6144, // 75% of the window: the warning band
        tokensEstimated: false,
        contextEstimated: false,
      },
    ]
    expect(stats.contextClass.value).toBe('violet')
    expect(stats.contextTitle.value).toContain('75%')
    chat.messages[0].contextTokens = 7400 // 90%: the window is nearly full
    expect(stats.contextClass.value).toBe('red')
  })
})
