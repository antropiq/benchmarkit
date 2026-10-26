<script setup>
import { computed } from 'vue'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { fileIcon } from '../lib/file-icons'

// One row of the workspace tree, recursive: a directory renders its own row
// plus, when expanded, one item per entry of its cached listing. No tree
// library — the keyboard model and the ARIA attributes stay ours (the same
// reasoning as TabNav.vue's hand-written tablist).
const props = defineProps({
  entry: { type: Object, required: true },
  // Position within the parent's listing, for aria-posinset/aria-setsize.
  index: { type: Number, required: true },
  setsize: { type: Number, required: true },
})

const { state, select, depthOf } = useWorkspaceTree()

const path = computed(() => props.entry.path)
const isDir = computed(() => !!props.entry.dir)
const expanded = computed(() => !!state.expanded[path.value])
const focused = computed(() => state.focused === path.value)
const selected = computed(() => state.selected?.path === path.value)
const node = computed(() => state.nodes[path.value] ?? null)
const loading = computed(() => !!node.value?.loading)
const children = computed(() =>
  isDir.value && expanded.value ? (node.value?.entries ?? []) : [],
)
const chevron = computed(() => (isDir.value && props.entry.children ? true : false))
const level = computed(() => depthOf(path.value) + 1)
const rowTitle = computed(() =>
  isDir.value ? path.value || 'workspace root' : `${path.value} (${humanSize(props.entry.size)})`,
)
// One glyph per row, picked by file type (see lib/file-icons). An open branch
// gets the open-folder glyph, so the icon and the chevron never disagree.
const icon = computed(() =>
  fileIcon(props.entry.name, { dir: isDir.value, open: expanded.value }),
)

function humanSize(bytes) {
  if (bytes === null || bytes === undefined) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function onFocus() {
  state.focused = path.value
}
</script>

<template>
  <div
    class="tree-row"
    :class="[
      `depth-${level - 1}`,
      { 'tree-row--focused': focused, 'tree-row--selected': selected },
    ]"
    role="treeitem"
    :aria-level="level"
    :aria-selected="selected"
    :aria-expanded="isDir ? expanded : undefined"
    :aria-posinset="index + 1"
    :aria-setsize="setsize"
    :title="rowTitle"
    :tabindex="focused ? 0 : -1"
    :data-path="path"
    :data-dir="isDir ? 'true' : 'false'"
    :style="{ '--tree-depth': level - 1 }"
    @click="select(entry)"
    @focus="onFocus"
  >
    <span class="tree-chevron" :class="{ open: expanded }" aria-hidden="true">
      <svg v-if="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round">
        <path d="m9 18 6-6-6-6" />
      </svg>
    </span>
    <span class="tree-icon" aria-hidden="true">{{ icon }}</span>
    <span class="tree-name">{{ entry.name }}</span>
    <span v-if="loading && isDir" class="tree-spinner caption" aria-hidden="true">…</span>
  </div>

  <div v-if="isDir && expanded" class="tree-group" role="group">
    <WorkspaceTreeItem
      v-for="(child, childIndex) in children"
      :key="child.path"
      :entry="child"
      :index="childIndex"
      :setsize="children.length"
    />
    <p v-if="node?.truncated" class="caption tree-truncated">
      more entries are not listed
    </p>
    <p v-if="node?.error" class="alert-error tree-error" role="alert">
      {{ node.error }}
    </p>
  </div>
</template>
