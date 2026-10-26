<script setup>
import { computed } from 'vue'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { useGitStatus } from '../composables/useGitStatus'
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
const git = useGitStatus()

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

// The unstaged marker: a dot after the name, in git's own colours — amber for
// a modification, green for a file that is not tracked yet. A folder carries
// one when *anything below* it is unstaged, so a collapsed branch still hints
// at what it hides; that mark has no kind of its own.
//
// The row's `title` already says the path, so the marker's meaning goes in the
// `title` of its own span (which wins on hover), and a screen reader gets the
// same words from a visually-hidden label rather than from the colour.
const gitEntry = computed(() => git.statusFor(path.value))
const markKind = computed(() => {
  if (gitEntry.value) return gitEntry.value.kind
  return isDir.value && git.hasUnstagedBelow(path.value) ? 'below' : null
})
const MARK_LABEL = {
  modified: 'modified since the last commit',
  added: 'new since the last commit',
  untracked: 'not tracked by git',
  deleted: 'deleted since the last commit',
  renamed: 'renamed since the last commit',
  conflict: 'in conflict',
  below: 'contains modified or untracked files',
}
const markTitle = computed(() => MARK_LABEL[markKind.value] || '')
// The class carries the colour by kind, so CSS decides what git's colours are
// in each theme rather than a literal here.
const markClass = computed(() => (markKind.value ? `tree-mark--${markKind.value}` : ''))

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
    <!-- The unstaged marker: a dot whose colour carries the kind. It follows
         the name (the name is what needs qualifying) and never shrinks, so a
         long ellipsised label cannot hide it — unlike the chip's dirty dot,
         which sits first for the same reason. -->
    <span
      v-if="markKind"
      class="tree-mark"
      :class="markClass"
      :title="markTitle"
      aria-hidden="true"
    ></span>
    <span v-if="markKind" class="tree-mark-sr">{{ markTitle }}</span>
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
