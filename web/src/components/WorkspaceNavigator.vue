<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import WorkspaceTreeItem from './WorkspaceTreeItem.vue'
import { useWorkspaceTree } from '../composables/useWorkspaceTree'
import { useEditorFiles } from '../composables/useEditorFiles'
import { useGitStatus } from '../composables/useGitStatus'
import { useBenchmark } from '../composables/useBenchmark'
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
  collapseAll,
  rootLabel,
  depthOf,
  ROOT,
  handleKey,
} = useWorkspaceTree()

const git = useGitStatus()

onMounted(() => {
  // First paint: exactly one listing, whatever the size of the checkout.
  if (!state.nodes[ROOT]?.loaded) load(ROOT)
  // One status query for the whole work tree, alongside that single listing —
  // never one per row. A location that is not a repository answers `repo:
  // false` with nothing to mark, so the tree simply looks the same.
  git.load()
})

// Re-reading git is part of what the refresh button promises: the point of it
// is that the working location moved on disk (the WorkBench, the terminal, or
// an editor outside the app), and a file that changed there is exactly what
// the user wants to see marked.
async function refreshAll() {
  await refresh()
  git.load({ force: true })
}

// A marker is worth something only if it is current, and the moment it goes
// stale is the moment the user was elsewhere. Coming back to the Editor tab
// re-reads it — cheap (one `git status`) and invisible.
const bench = useBenchmark()
watch(
  () => bench.state.activeTab,
  (tab) => {
    if (tab === 'editor') git.load({ force: true })
  },
)

const refreshing = computed(() => !!state.nodes[ROOT]?.loading)
// The collapse button has nothing to do while only the root is open, so it
// says so instead of looking clickable.
const canCollapse = computed(
  () => Object.keys(state.expanded).some((path) => path !== ROOT),
)
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
// The root row's hint: any unstaged file anywhere in the browsed subtree.
const rootMarked = computed(() => git.state.entries.length > 0)

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

// The tree's focus can also move *without* the user being in the tree: focusing
// a tab in the strip reveals that file's row (useWorkspaceTree.reveal), which
// opens collapsed branches and may leave the row outside the scroll viewport.
// Scrolling it into view is all this does — calling `focus()` here would take
// the caret away from the chip the user just clicked. `block: nearest` is a
// no-op when the row is already on screen, so a click inside the tree never
// shifts it.
async function scrollFocusedIntoView() {
  await nextTick()
  for (const candidate of tree.value?.querySelectorAll('.tree-row') ?? []) {
    if (candidate.dataset.path === state.focused) {
      candidate.scrollIntoView({ block: 'nearest' })
      return
    }
  }
}

watch(() => state.focused, scrollFocusedIntoView)

// -- context menu -----------------------------------------------------------
//
// One menu for the whole tree, owned here (not by each recursive row): a
// right-click on any row bubbles its native `contextmenu` to the tree
// container, which reads the row's `data-path` + `data-dir` off the event
// target. Files get the menu; the root and folder rows do not (for now).
//
// The items:
//   * View diff… — only for an unstaged file git says it can compare, i.e. one
//     with a committed side (`comparable`). An untracked or newly added file is
//     marked in the tree but has nothing to diff *against*, so the item is
//     absent rather than disabled-and-lying. The tab it opens lives in the same
//     strip as the file tabs (useEditorFiles.openDiff).
//   * Close — only shown when a tab already holds this file; the same action
//     as the chip's × (useEditorFiles.closeFile). A dirty buffer closes and
//     discards: there is no save yet, so there is nothing to confirm.
//   * Copy relative path — the tree's paths are already workspace-relative, so
//     the value goes straight to the clipboard.
const files = useEditorFiles()
const menu = reactive({ open: false, x: 0, y: 0, path: null })
const menuCanClose = computed(() => !!menu.path && !!files.recordFor(menu.path))
// A folder row never opens this menu at all, so `menu.path` is always a file.
const menuCanDiff = computed(() => !!menu.path && git.isDiffable(menu.path))
const menuEl = ref(null)

function openMenu(event) {
  const row = event.target.closest?.('.tree-row')
  if (!row || row.dataset.dir === 'true') return
  const path = row.dataset.path
  if (!path) return
  event.preventDefault()
  closeMenu()
  // Place at the cursor, then clamp to the viewport once the menu's own size
  // is known (nextTick, after it renders).
  menu.x = event.clientX
  menu.y = event.clientY
  menu.path = path
  menu.open = true
  nextTick(() => {
    const el = menuEl.value
    if (!el) return
    const rect = el.getBoundingClientRect()
    const pad = 6
    let x = menu.x
    let y = menu.y
    if (x + rect.width > window.innerWidth - pad) x = window.innerWidth - pad - rect.width
    if (y + rect.height > window.innerHeight - pad) y = window.innerHeight - pad - rect.height
    menu.x = Math.max(pad, x)
    menu.y = Math.max(pad, y)
  })
}

function closeMenu() {
  if (!menu.open) return
  menu.open = false
  menu.path = null
}

function menuViewDiff() {
  if (menu.path) files.openDiff(menu.path)
  closeMenu()
}

function menuClose() {
  if (menu.path) files.closeFile(menu.path)
  closeMenu()
}

async function menuCopyPath() {
  if (menu.path) {
    try {
      await navigator.clipboard.writeText(menu.path)
    } catch {
      // A non-secure context or a denied permission: nothing else to do here,
      // and this pane has no toast system. Fail silently.
    }
  }
  closeMenu()
}

// Any click outside the menu, or Escape, dismisses it. Registered only while
// the menu is open so it never runs on the idle tree.
function onDocPointerDown(event) {
  if (menuEl.value && menuEl.value.contains(event.target)) return
  closeMenu()
}
function onDocKeydown(event) {
  if (event.key === 'Escape') closeMenu()
}

watch(
  () => menu.open,
  (open) => {
    if (open) {
      document.addEventListener('pointerdown', onDocPointerDown, true)
      document.addEventListener('keydown', onDocKeydown, true)
    } else {
      document.removeEventListener('pointerdown', onDocPointerDown, true)
      document.removeEventListener('keydown', onDocKeydown, true)
    }
  },
)
onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', onDocPointerDown, true)
  document.removeEventListener('keydown', onDocKeydown, true)
})
</script>

<template>
  <aside class="editor-nav" aria-label="Workspace files">
    <div class="editor-nav-head">
      <p class="caption editor-nav-title" :title="state.workspace">
        {{ rootLabel() }}
      </p>
      <div class="editor-nav-actions">
        <button
          class="icon-btn editor-nav-refresh"
          type="button"
          :disabled="refreshing"
          aria-label="Refresh the file tree"
          title="Re-list every open folder and re-read which files are unstaged"
          @click="refreshAll"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
            <path d="M3 3v5h5" />
          </svg>
        </button>
        <!-- Close every opened branch at once. Distinct from the root row's own
             chevron: this one leaves the root open and its listing cached, so
             nothing is re-fetched on the way back down. -->
        <button
          class="icon-btn editor-nav-collapse"
          type="button"
          :disabled="!canCollapse"
          aria-label="Close all opened folders in the file tree"
          title="Collapse every opened folder"
          @click="collapseAll"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M5 12h14" />
          </svg>
        </button>
      </div>
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
      @contextmenu="openMenu"
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
        <!-- The root is a folder like any other, so it carries the same hint:
             something in the work tree is unstaged, even in a branch that is
             closed right now. -->
        <span
          v-if="rootMarked"
          class="tree-mark tree-mark--below"
          title="contains modified or untracked files"
          aria-hidden="true"
        ></span>
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

    <!-- The file context menu, teleported to <body> so the tree's scroll
         container (`.editor-nav-tree`, which is `overflow: auto`) cannot clip
         it and the pane's stacking cannot bury it. Positioned fixed at the
         cursor (clamped in openMenu). `@contextmenu.prevent` here stops a
         right-click *inside* the menu from re-opening it. -->
    <Teleport to="body">
      <div
        v-if="menu.open"
        ref="menuEl"
        class="tree-context-menu"
        role="menu"
        :style="{ left: menu.x + 'px', top: menu.y + 'px' }"
        @contextmenu.prevent
      >
        <button
          v-if="menuCanDiff"
          class="tree-context-item"
          type="button"
          role="menuitem"
          @click="menuViewDiff"
        >
          View diff…
        </button>
        <button
          v-if="menuCanClose"
          class="tree-context-item"
          type="button"
          role="menuitem"
          @click="menuClose"
        >
          Close
        </button>
        <button
          class="tree-context-item"
          type="button"
          role="menuitem"
          @click="menuCopyPath"
        >
          Copy relative path
        </button>
      </div>
    </Teleport>
  </aside>
</template>
