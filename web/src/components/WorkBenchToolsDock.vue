<script setup>
import { computed } from 'vue'
import WorkBenchMcpPanel from './WorkBenchMcpPanel.vue'
import WorkBenchLspPanel from './WorkBenchLspPanel.vue'
import { useChat } from '../composables/useChat'
import { anyHub, hubsSummary } from '../lib/chat-hubs'

// Bottom of the Chat left column: the conversation-scoped MCP panel beside
// the LSP panel (the pair the input row used to carry), plus the control that
// folds them away so the reasoning timeline above gets the whole column.
//
// The fold belongs to the view, not to this component: folding changes how
// much height the column's splitter gives the timeline, so only the layout
// owner can answer to it. Hence the `collapsed` prop / `update:collapsed`
// pair rather than a local flag.
const props = defineProps({ collapsed: Boolean })
const emit = defineEmits(['update:collapsed'])

const { state } = useChat()

// With neither hub declared there is nothing to switch, so the dock is a
// plain strip: no body, and no button pretending it has one (the timeline
// then owns the column's height on its own, exactly as an empty column used
// to cost nothing in the input row).
const hasBody = computed(() => anyHub(state))
const expanded = computed(() => hasBody.value && !props.collapsed)

// Folded, the strip still answers the one question the panels answered: which
// hubs are switched on for this conversation. Counts rather than names — the
// folded strip is a single line.
const summary = computed(() => hubsSummary(state))
</script>

<template>
  <section
    class="chat-tools-dock"
    :class="{ 'chat-tools-dock--collapsed': !expanded }"
    aria-label="Conversation tools"
  >
    <div class="chat-tools-dock-head">
      <p class="caption chat-tools-dock-title">Tools</p>
      <span
        v-if="hasBody && !expanded"
        class="chat-tools-dock-summary"
        role="status"
        title="Tool hubs enabled for this conversation — expand to change them"
      >
        {{ summary }}
      </span>
      <button
        v-if="hasBody"
        class="icon-btn chat-tools-dock-toggle"
        type="button"
        :aria-expanded="expanded"
        :aria-controls="expanded ? 'chat-tools-dock-body' : undefined"
        :aria-label="expanded ? 'Hide the tool panels' : 'Show the tool panels'"
        :title="
          expanded
            ? 'Hide the tool panels — the reasoning timeline takes the column'
            : 'Show the MCP and LSP panels'
        "
        @click="emit('update:collapsed', expanded)"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path v-if="expanded" d="m6 9 6 6 6-6" />
          <path v-else d="m18 15-6-6-6 6" />
        </svg>
      </button>
    </div>
    <div v-if="expanded" id="chat-tools-dock-body" class="chat-tools-dock-body">
      <WorkBenchMcpPanel />
      <WorkBenchLspPanel />
    </div>
  </section>
</template>
