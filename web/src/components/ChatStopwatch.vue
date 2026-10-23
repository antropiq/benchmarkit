<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useChat } from '../composables/useChat'

// Session stopwatch shown under the WorkBench action buttons: counts the
// elapsed time of the running chat request (waiting + reasoning + every tool
// round) and rests on 00:00:00 whenever the conversation is idle.
const { state } = useChat()

// Ticked on its own interval rather than on SSE events, so the digits keep
// advancing during a long tool call or a stalled first token as well.
const now = ref(Date.now())
let timer = null

onMounted(() => {
  timer = setInterval(() => {
    now.value = Date.now()
  }, 250)
})

onBeforeUnmount(() => clearInterval(timer))

const running = computed(() => state.session.startedAt > 0)

function format(ms) {
  const total = Math.max(0, Math.floor(ms / 1000))
  const parts = [Math.floor(total / 3600), Math.floor((total % 3600) / 60), total % 60]
  return parts.map((n) => String(n).padStart(2, '0')).join(':')
}

const display = computed(() =>
  running.value ? format(now.value - state.session.startedAt) : '00:00:00',
)
</script>

<template>
  <p
    class="chat-stopwatch"
    :class="{ 'chat-stopwatch--running': running }"
    role="timer"
    :aria-label="running ? 'Elapsed time of the chat session' : 'Chat session idle'"
    :title="
      running
        ? 'Elapsed time of the running chat session'
        : 'Idle — the stopwatch counts each running chat request'
    "
  >
    {{ display }}
  </p>
</template>
