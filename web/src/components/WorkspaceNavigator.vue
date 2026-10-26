<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import WorkspaceTreeItem from './WorkspaceTreeItem.vue'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { fileIcon } from '../lib/file-icons'
// The rows render Nerd Font glyphs (lib/file-icons). This CSS declares the
// "JetBrains Mono Nerd Icons" face those code points need; it is unicode-range-
// gated to exactly them, so the woff2 is only fetched once a glyph actually
// asks for it, and the file name beside it stays on the normal mono face.
import '../fonts/nerd-fonts.css'

// The Editor tab's file tree (ed-01): header with the workspace it browses,
// the recursive tree, and the footer that explains the omissions the connector
// policy causes. Self-contained on purpose — ed-02 drops it into the layout
// unchanged, and ed-03 only reads `state.selected` from the composable.
const {
  state,
  load,
  refresh,
  toggle,
  rootLabel,
  depthOf,
  ROOT,
  handleKey,
} = useWorkspaceTree()

onMounted(() => {
  // First paint: exactly one listing, whatever the size of the checkout.
  if (!state.nodes[ROOT]?.loaded) load(ROOT)
})

const refreshing = computed(() => !!state.nodes[ROOT]?.loading)
const rootTitle = computed(
  () =>
    `${state.workspace || 'workspace root'} — click to ${
      state.expanded[ROOT] ? 'collapse' : 'expand'
    }`,
)
// aria-level of the root row (its children are one deeper each).
const rootLevel = computed(() => depthOf(ROOT) + 1)
// The root is a folder like any other, so it carries the same glyph (open, as
// it always is at the top) — without it the first level of children would not
// line up under their parent's icon.
const rootIcon = computed(() => fileIcon('', { dir: true, open: !!state.expanded[ROOT] }))

const tree = ref(null)

// The roving tabindex decides which row is in the tab order; when a key moves
// the focus, the *DOM* focus follows it, so the visible caret and what a
// screen reader announces never lag behind the model. Rows carry their path in
// `data-path`, matched by identity here (a path may hold any character, so no
// selector escaping is attempted).
async function onKeydown(event) {
  handleKey(event)
  await nextTick()
  for (const candidate of tree.value?.querySelectorAll('.tree-row') ?? []) {
    if (candidate.dataset.path === state.focused) {
      candidate.focus()
      return
    }
  }
}
</script>

<template>
  <aside class="editor-nav" aria-label="Workspace files">
    <div class="editor-nav-head">
      <p class="caption editor-nav-title" :title="state.workspace">
        {{ rootLabel() }}
      </p>
      <button
        class="icon-btn editor-nav-refresh"
        type="button"
        :disabled="refreshing"
        aria-label="Refresh the file tree"
        title="Re-list every open folder (the working location may have moved)"
        @click="refresh"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
          <path d="M3 3v5h5" />
        </svg>
      </button>
    </div>

    <p v-if="state.error" class="alert-error editor-nav-error" role="alert">
      {{ state.error }}
    </p>

    <div
      ref="tree"
      class="editor-nav-tree"
      role="tree"
      aria-label="Workspace files"
      @keydown="onKeydown"
    >
      <div
        class="tree-row tree-row--root"
        :class="{ 'tree-row--focused': state.focused === ROOT }"
        role="treeitem"
        :aria-level="rootLevel"
        :aria-selected="state.selected === null && state.focused === ROOT"
        :aria-expanded="!!state.expanded[ROOT]"
        :title="rootTitle"
        :tabindex="state.focused === ROOT ? 0 : -1"
        data-path=""
        @focus="state.focused = ROOT"
        @click="toggle(ROOT)"
      >
        <span class="tree-chevron" :class="{ open: !!state.expanded[ROOT] }" aria-hidden="true">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="m9 18 6-6-6-6" />
          </svg>
        </span>
        <span class="tree-icon" aria-hidden="true">{{ rootIcon }}</span>
        <span class="tree-name">{{ rootLabel() }}</span>
      </div>

      <template v-if="state.expanded[ROOT]">
        <WorkspaceTreeItem
          v-for="(entry, index) in state.nodes[ROOT]?.entries ?? []"
          :key="entry.path"
          :entry="entry"
          :index="index"
          :setsize="state.nodes[ROOT]?.entries?.length ?? 0"
        />
        <p v-if="!state.nodes[ROOT]?.loading && !(state.nodes[ROOT]?.entries?.length)" class="caption editor-nav-empty">
          The working location is empty.
        </p>
      </template>
    </div>

    <p class="caption editor-nav-footer">
      Some files are hidden by the connector policy.
    </p>
  </aside>
</template>
