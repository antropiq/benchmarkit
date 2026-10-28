<script setup>
import { computed } from 'vue'
import { useBenchmark } from '../composables/useBenchmark'
import { useEditorNotices } from '../composables/useEditorNotices'

// The Editor's write-feedback stack: up to three dismissible lines at the
// bottom-centre of the app, answering "what happened to that write?" from
// wherever the eyes are — the tree or the code pane.
//
// It is teleported to `<body>` for the same reason the tree's context menu is
// (WorkspaceNavigator.vue): the panes it reports on are scroll containers with
// their own stacking, and a notice inside one of them gets clipped or buried.
// Bottom-centre because it must not cover the tab strip, the composer, or the
// row a right-click just opened.
//
// All of the rules — dedupe, the cap, which tones expire — live in
// useEditorNotices; this component only renders `state.notices` and gives each
// line its ×. `aria-live="polite"` on the container is what makes a refusal
// heard when focus is in the tree, and it is deliberately a *second* element
// from the pane's `role="alert"` message line: one element cannot serve both a
// refused open and a refused write.
const { state, dismiss } = useEditorNotices()

// Teleported content is outside App.vue's `v-show` on the Editor view, so the
// stack answers the same question the pane does: it is Editor feedback, and it
// has no business floating over the Benchmark table or the WorkBench composer.
// The notices themselves survive the switch (the composable owns them), so
// coming back to the tab shows exactly what was there.
const onEditorTab = computed(() => useBenchmark().state.activeTab === 'editor')
const visible = computed(() => onEditorTab.value && state.notices.length > 0)
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      class="editor-notice-stack"
      aria-live="polite"
      aria-label="Editor write feedback"
    >
      <p
        v-for="item in state.notices"
        :key="item.id"
        class="editor-notice"
        :class="`editor-notice--${item.tone}`"
      >
        <span class="editor-notice-text">{{ item.text }}</span>
        <button
          class="editor-notice-dismiss"
          type="button"
          aria-label="Dismiss this message"
          title="Dismiss"
          @click="dismiss(item.id)"
        >
          ×
        </button>
      </p>
    </div>
  </Teleport>
</template>
