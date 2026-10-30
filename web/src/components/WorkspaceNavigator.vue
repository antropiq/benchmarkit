<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import WorkspaceTreeItem from './WorkspaceTreeItem.vue'
import TreeNameInput from './TreeNameInput.vue'
import { registerRefreshChord, useWorkspaceTree } from '../composables/useWorkspaceTree'
import { TRASH_RETENTION_DAYS, useEditorFiles } from '../composables/useEditorFiles'
import { useEditorCreate } from '../composables/useEditorCreate'
import { useEditorRename } from '../composables/useEditorRename'
import { useGitStatus } from '../composables/useGitStatus'
import SearchDialog from './SearchDialog.vue'
import { useWorkspaceSearch } from '../composables/useWorkspaceSearch'
import { useBenchmark } from '../composables/useBenchmark'
import { useEditorMode } from '../composables/useEditorMode'
import { treeMenuItems } from '../lib/tree-menu'
import { fileIcon } from '../lib/file-icons'
import { isMarkdownPath } from '../lib/markdown-preview'
// The rows render Nerd Font glyphs (lib/file-icons). This CSS declares the
// "JetBrains Mono Nerd Icons" face those code points need; it is unicode-range-
// gated to exactly them, so the woff2 is only fetched once a glyph actually
// asks for it, and the file name beside it stays on the normal mono face.
import '../fonts/nerd-fonts.css'

// The Editor tab's file tree: header, the recursive tree, and the footer that
// explains the omissions the connector policy causes. Self-contained on
// purpose — the layout mounts it as it is, and the code pane only reads
// `state.selected` from the composable.
//
// The header names no workspace: the tree's own root row already carries the
// folder's name (and its absolute path in its `title`). What sits there instead
// is the code pane's mode badge — the tab strip to the right is a row that
// shrinks as buffers pile up, and this status belongs to the pane, not to one
// tab, so it is shown where there is always room for it.
const {
  state,
  load,
  refresh,
  refreshBranch,
  toggle,
  collapseAll,
  rootLabel,
  depthOf,
  entryFor,
  select,
  commandTarget,
  ROOT,
  handleKey,
} = useWorkspaceTree()

const git = useGitStatus()
// Template bindings are destructured, so they auto-unwrap like any other ref.
const { isEditable, badgeLabel, badgeTitle } = useEditorMode()

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

// The same promise for one folder: the `r` chord and a folder row's "Refresh"
// item. Both halves, because a re-listed row whose git marker is stale is only
// half refreshed — and the marks are one query for the whole work tree, so one
// folder's refresh costs exactly what the header button's costs.
async function refreshFolder(path) {
  await refreshBranch(path)
  git.load({ force: true })
}

// The tree names the key (useWorkspaceTree.handleKey); this owns the answer, for
// the reason the create/delete/rename chords are registered elsewhere too: the
// keyboard model belongs to the tree, the request belongs to whoever holds the
// other state it touches — here, the git marks.
registerRefreshChord((path) => refreshFolder(path))

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
// The header's Search button follows the collapse button's shape: disabled when
// there is nothing to act on. Its subject is the focused row, and `commandTarget()`
// is a function that answers null when that row stopped existing — calling it from
// the template would re-evaluate it on every render, so it is a computed here.
const canSearch = computed(() => commandTarget() !== null)
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
//
// The caret follows *only* for a keystroke the tree answered. This container
// sees the keydowns of everything inside it, and the inline name input lives in
// it: following the model after a key the input owned would throw the caret out
// of the half-typed name on the first letter — and out of it before `Escape`
// ever reached it, which is how a cancel appeared to do nothing.
async function onKeydown(event) {
  if (!handleKey(event)) return
  await nextTick()
  focusRow(state.focused)
}

// The row the model considers focused, by `data-path` (see the note above about
// paths needing no selector escaping). Returns false when no rendered row carries
// it — which is what `revalidateFocus` exists to prevent, but a key can land
// between a re-list and its revalidation.
function focusRow(path) {
  for (const candidate of tree.value?.querySelectorAll('.tree-row') ?? []) {
    if (candidate.dataset.path === path) {
      candidate.focus()
      return true
    }
  }
  return false
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
// target. **Every row kind gets one** — file, folder and the synthetic root —
// and the items come from one ordered table (`lib/tree-menu`) rather than from
// `v-if`s in the template, so the menu's contents and their order are data a
// test can read, and a new item is one row of that table.
//
// Which items apply is decided by the flags of the row, and an item is *absent*
// rather than shown-disabled-and-lying:
//
//   * New file… / New folder… / Refresh — a folder row and the root row (both
//     are folders a name goes into). They open the inline name input inside that
//     folder (useEditorCreate); the refusal of a protected name is the server's
//     and shows up in that row.
//   * View diff… — only for a file git says it can compare, i.e. one with a
//     committed side (`comparable`). An untracked or newly added file is marked
//     in the tree but has nothing to diff *against*. The tab it opens lives in
//     the same strip as the file tabs (useEditorFiles.openDiff).
//   * Open — a file row, and the same action as clicking it (`select`, which is
//     what EditorArea watches), so the menu and a left-click are one flow.
//   * Rename… / Delete — every row but the root. The root's row *is* the working
//     location: the server refuses renaming or deleting it outright, and there
//     is no listing above it to type a new name into — offering either would
//     only invite the refusal. Delete only opens the confirmation
//     (`useEditorFiles.requestDelete`); the trash move happens there, so this
//     menu never destroys anything itself.
//   * Close — only when a tab already holds this file; the same action as the
//     chip's × (`requestCloseTab` asks Save-and-close / Discard / Cancel for a
//     dirty buffer, so this item cannot throw work away either).
//   * Copy relative path — the tree's paths are already workspace-relative, so
//     the value goes straight to the clipboard. For the root that is the
//     workspace's absolute path, the only thing a reader can act on.
//   * Search… — every row, and this item only hands the row over: the query, the
//     scope and the results belong to `useWorkspaceSearch`, and `lib/search-scope`
//     turns the row into one of the three scopes (a folder's subtree, one file's
//     content, the whole workspace).
//
// A row that offers nothing at all gets no menu — one useless item is worse than
// no menu — which `openMenu` checks before it positions anything. `scope: 'row'`
// items are what decide that: Search is the floor every real row now has.
const files = useEditorFiles()
const creator = useEditorCreate()
const renamer = useEditorRename()
// Search raises the dialog (mounted at the end of this template) on the
// navigator's own subject. The query, the scope and the results are the
// composable's state, so this component only ever calls `openFor`.
const search = useWorkspaceSearch()
// `items` is the menu: the table's answer for the row that was clicked. Nothing
// else about the row is remembered — the subject is the tree's focus, which
// `openMenu` sets first, so `commandTarget()` is the same row for a menu item
// and for a key press.
const menu = reactive({ open: false, x: 0, y: 0, items: [] })
const menuEl = ref(null)

// The row kind plus the facts about it that make an item honest. `dir` counts
// the root as a folder (it is one); `mutable` is the root rule; the two "is
// there something to show?" flags are read from the same state the rows are
// drawn from, so a menu item cannot promise a tab or a diff that is not there.
// The list is built once, when the menu opens: a menu is a snapshot of the
// moment the pointer was put down, and an item appearing or vanishing under the
// cursor mid-click is how a menu misfires.
function menuFlags(path, isDir) {
  const isRoot = path === ROOT
  return {
    dir: isDir,
    isRoot,
    mutable: !isRoot,
    isMarkdown: !isDir && !isRoot && isMarkdownPath(path),
    diffable: !isDir && !isRoot && git.isDiffable(path),
    hasBuffer: !isDir && !isRoot && !!files.recordFor(path),
    trashRetentionDays: TRASH_RETENTION_DAYS,
  }
}

function openMenu(event) {
  const row = event.target.closest?.('.tree-row')
  if (!row) return
  // `data-path` is empty for the root row, so presence is tested on the
  // attribute, not on the value.
  const path = row.dataset.path
  if (path === undefined) return
  const items = treeMenuItems(menuFlags(path, row.dataset.dir === 'true'))
  if (!items.length) return // nothing honest to offer: no menu at all
  event.preventDefault()
  closeMenu()
  // The row becomes the tree's subject *before* the menu opens, so the mouse and
  // the keyboard agree about "this row" from the moment of the right-click: the
  // chords answer about `commandTarget()`, and so does every item below.
  state.focused = path
  menu.x = event.clientX
  menu.y = event.clientY
  menu.items = items
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
  menu.items = []
}

// Each item, keyed by its table id. Every one of them acts on `commandTarget()`
// — the focused row, which `openMenu` just set to the row that was clicked — so
// the menu has no private idea of what "this file" means.
const MENU_ACTIONS = {
  // The click's own flow: publish the row as the tree's selection, which is what
  // EditorArea watches to open a buffer. A refused open (binary, over the cap)
  // is then the pane's answer, the same one a left-click gets.
  open: (target) => select(entryFor(target.path)),
  // Search gets the whole row shape, not just its path: `lib/search-scope` is
  // what tells a folder's subtree from one file's content from the whole
  // workspace, and that answer needs `path` and `dir`.
  search: (target) => search.openFor(target),
  'new-file': (target) => creator.startCreate(target.path, false),
  'new-folder': (target) => creator.startCreate(target.path, true),
  'view-diff': (target) => files.openDiff(target.path),
  // Preview goes directly to the flow, like `view-diff` and unlike `open`:
  // publishing a tree selection would make EditorArea's watcher open the Monaco
  // buffer *beside* the preview tab. So `useWorkspaceTree` needs no new field.
  preview: (target) => files.openPreview(target.path),
  // Rename asks for a name; it does not move anything. The inline input opens in
  // the folder that shows the row, prefilled, and the move happens only when
  // that input is submitted — the same one-gesture-never-decides shape as the
  // delete item, one level milder.
  rename: (target) => renamer.startRename(target.path),
  // Delete asks, it does not delete: `requestDelete` raises the confirmation with
  // the row, and the tree's re-list happens when the move lands. The row travels
  // (name + kind) because that is what the user looked at.
  delete: (target) => files.requestDelete(entryFor(target.path) ?? target),
  close: (target) => files.closeFile(target.path),
  refresh: (target) => refreshFolder(target.path),
  'copy-path': copyMenuPath,
}

function runMenuItem(id) {
  const target = commandTarget()
  closeMenu()
  if (!target) return // the row stopped existing between the click and here
  MENU_ACTIONS[id]?.(target)
}

async function copyMenuPath(target) {
  // The root row has no relative path to copy — the only thing a reader can act
  // on there is the workspace's own absolute path.
  const value = target.path === ROOT ? state.workspace : target.path
  if (value) {
    try {
      await navigator.clipboard.writeText(value)
    } catch {
      // A non-secure context or a denied permission: nothing else to do here,
      // and this pane has no toast system. Fail silently.
    }
  }
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
      <span
        class="editor-mode-badge"
        :class="isEditable ? 'badge green' : 'badge gray'"
        :title="badgeTitle"
      >
        {{ badgeLabel }}
      </span>
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
        <!-- Search, the same command as the menu item, on the row the tree is
             about (§2.9). It is here rather than beside the tab strip because the
             dialog's scope label needs the navigator's own subject, and this
             button and `<SearchDialog />` below are the same pair for the same
             reason. Enabled only when a row is focused — with no subject the
             command declines, so the button says so first. -->
        <button
          class="icon-btn editor-nav-search"
          type="button"
          :disabled="!canSearch"
          aria-label="Search inside the workspace"
          title="Search inside this folder, this file, or the whole workspace"
          @click="search.openFor(commandTarget())"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.6-3.6" />
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
        data-dir="true"
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
        <!-- The name input, in the listing it belongs to — for the root, here;
             for a folder, in its own group (WorkspaceTreeItem). Keyed by mode,
             folder and row so a second New… or Rename… elsewhere re-mounts and
             re-focuses it. -->
        <TreeNameInput
          v-if="state.editing?.parent === ROOT"
          :key="`${state.editing.mode}:${ROOT}:${state.editing.path ?? ''}:${state.editing.isDir}`"
        />
        <p v-if="!state.nodes[ROOT]?.loading && !(state.nodes[ROOT]?.entries?.length) && !state.editing" class="caption editor-nav-empty">
          The working location is empty.
        </p>
      </template>
    </div>

    <p class="caption editor-nav-footer">
      Some files are hidden by the connector policy.
    </p>

    <!-- The tree's context menu, teleported to <body> so the tree's scroll
         container (`.editor-nav-tree`, which is `overflow: auto`) cannot clip
         it and the pane's stacking cannot bury it. Positioned fixed at the
         cursor (clamped in openMenu). `@contextmenu.prevent` here stops a
         right-click *inside* the menu from re-opening it. The buttons are the
         table's answer for this row, in its order — `v-for`, never a `v-if` per
         item, so what a row offers is one data question (`lib/tree-menu`). -->
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
          v-for="item in menu.items"
          :key="item.id"
          class="tree-context-item"
          :class="{ 'tree-context-item--danger': item.danger }"
          type="button"
          role="menuitem"
          :title="item.title"
          @click="runMenuItem(item.id)"
        >
          {{ item.label }}
        </button>
      </div>
    </Teleport>

    <!-- The search dialog, mounted beside the header button and the menu item
         that raise it (§2.9): both hand over the same `commandTarget()`, and a
         component mounted next to the state it reads is the only way the scope
         label can name the navigator's own subject without reaching into another
         component. It is never removed from the DOM — the composable keeps its
         answer, and the dialog's own watch reconciles `showModal()` with
         `state.open`. -->
    <SearchDialog />
  </aside>
</template>
