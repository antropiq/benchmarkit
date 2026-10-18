import { nextTick, ref, watch } from 'vue'

/**
 * Autoscroll a scrollable element while content grows, unless the user
 * scrolled away from the bottom. `source` is a watch getter that changes
 * whenever the content updates.
 */
export function useAutoScroll(elRef, source) {
  const stickToBottom = ref(true)

  function onScroll() {
    const el = elRef.value
    if (!el) return
    stickToBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 48
  }

  watch(source, async () => {
    if (!stickToBottom.value) return
    await nextTick()
    const el = elRef.value
    if (el) el.scrollTop = el.scrollHeight
  })

  return { onScroll, stickToBottom }
}
