import { watchEffect } from 'vue'
import { useBenchmark } from './useBenchmark'
import { useChat } from './useChat'
import { useBatch } from './useBatch'
import { inferenceRunning } from '../lib/inference-state'

// The browser tab is the only thing visible while the app's tab is in the
// background, so it carries the one bit that matters there: is an inference
// running right now? ⚡ in front of the title while one is, ✨ when none is.
//
// The base text is read from index.html's <title> once, at import time, so the
// prefix never stacks up and the title string is never duplicated here. The
// leading-glyph strip only guards a hot reload, where document.title may
// already hold a prefix written by the previous instance of this module.
const BASE_TITLE = document.title.replace(/^(?:[⚡✨]\s*)*/, '')

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
