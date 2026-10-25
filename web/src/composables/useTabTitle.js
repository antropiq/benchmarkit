import { watchEffect } from 'vue'
import { useBenchmark } from './useBenchmark'
import { useChat } from './useChat'
import { useBatch } from './useBatch'

// The browser tab is the only thing visible while the app's tab is in the
// background, so it carries the one bit that matters there: is an inference
// running right now? ⚡ in front of the title while one is, ✨ when none is.
//
// The base text is read from index.html's <title> once, at import time, so the
// prefix never stacks up and the title string is never duplicated here. The
// leading-glyph strip only guards a hot reload, where document.title may
// already hold a prefix written by the previous instance of this module.
const BASE_TITLE = document.title.replace(/^(?:[⚡✨]\s*)*/, '')

// The three ways this app spends time on an inference. Each owns its own
// composable, and all three are one-way reads from here, so the dependency
// graph stays acyclic (useBatch already imports useChat and useBenchmark).
//   * benchmark run — 'running' covers warm-up through the last verifier call;
//   * WorkBench reply — waiting, reasoning or answering (a batch turn too);
//   * batch lot — stays busy across the gap between two task turns, which is
//     when the chat phase is back to 'idle' but a task is still owed.
function inferenceRunning({ benchmark, chat, batch }) {
  if (benchmark.status === 'running') return true
  if (batch.active) return true
  return ['waiting', 'reasoning', 'answering'].includes(chat.phase)
}

export function useTabTitle() {
  const { state: benchmark } = useBenchmark()
  const { state: chat } = useChat()
  const { state: batch } = useBatch()

  // Mirrors the app bar's status pill (AppBar.vue reads the same three flags);
  // deliberately only two states, since a background tab is read at a glance.
  watchEffect(() => {
    document.title = inferenceRunning({ benchmark, chat, batch })
      ? `⚡ ${BASE_TITLE}`
      : `✨ ${BASE_TITLE}`
  })
}
