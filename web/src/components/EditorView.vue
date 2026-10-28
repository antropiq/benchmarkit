<script setup>
import { ref } from 'vue'
import WorkspaceNavigator from './WorkspaceNavigator.vue'
import EditorArea from './EditorArea.vue'
import EditorNoticeStack from './EditorNoticeStack.vue'
import EditorTerminal from './EditorTerminal.vue'
import { useSplitter } from '../composables/useSplitter.js'

// The Editor tab's shell: navigator on the left at full height, the code pane
// in the centre, the terminal strip along the bottom — a grid whose tracks are
// sized by the two splitters below. Each region owns its own state in a
// composable, so this component only lays them out and moves the borders.
//
// The navigator keeps its own state in useWorkspaceTree, so the whole view
// staying mounted (App.vue's v-show) is what preserves open files and, later,
// the terminal session across a peek at the Benchmark tab. A changed working
// location re-roots that tree from the save path itself (useBenchmark's
// setPathLocation), so this view owns no cross-tab wiring.
const shell = ref(null)
const nav = ref(null)
const terminal = ref(null)

// Sidebar: the pane *before* the bar grows to the right. 160 px at least, and
// never past half the shell, so the code pane always keeps a usable width.
const {
  dragging: sidebarDragging,
  shellStyle: sidebarStyle,
  onPointerDown: sidebarDown,
  onPointerMove: sidebarMove,
  onPointerUp: sidebarUp,
  onKeyDown: sidebarKey,
  onDoubleClick: sidebarReset,
} = useSplitter(shell, nav, {
  orientation: 'vertical',
  storageKey: 'benchmarkit:editorSidebar',
  minResized: 160,
  minOther: 240,
  maxRatio: 0.5,
  sizeVar: '--bi-editor-sidebar',
})

// Terminal: the pane *after* the bar grows upwards. 96 px is two lines plus
// the panel header; the code pane below it keeps 200 px.
const {
  dragging: terminalDragging,
  shellStyle: terminalStyle,
  onPointerDown: terminalDown,
  onPointerMove: terminalMove,
  onPointerUp: terminalUp,
  onKeyDown: terminalKey,
  onDoubleClick: terminalReset,
} = useSplitter(shell, terminal, {
  orientation: 'horizontal',
  storageKey: 'benchmarkit:editorTerminal',
  minResized: 96,
  minOther: 200,
  sizeVar: '--bi-editor-terminal',
})

// Narrow screens: the tree becomes a drawer, because the single-column stack
// would otherwise push the code pane below the fold.
const navOpen = ref(true)
</script>

<template>
  <div
    ref="shell"
    class="editor-shell"
    :class="{ 'nav-collapsed': !navOpen }"
    :style="[sidebarStyle, terminalStyle]"
  >
    <div ref="nav" class="editor-nav-col">
      <button
        class="editor-nav-toggle"
        type="button"
        :aria-expanded="navOpen"
        aria-controls="editor-nav"
        :title="navOpen ? 'Hide the file tree' : 'Show the file tree'"
        @click="navOpen = !navOpen"
      >
        Files
      </button>
      <div id="editor-nav" class="editor-nav-slot">
        <WorkspaceNavigator />
      </div>
    </div>

    <div
      class="pane-resizer pane-resizer--vertical editor-resizer-v"
      :class="{ dragging: sidebarDragging }"
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize the file tree (drag, arrow keys, double-click to reset)"
      tabindex="0"
      @pointerdown="sidebarDown"
      @pointermove="sidebarMove"
      @pointerup="sidebarUp"
      @pointercancel="sidebarUp"
      @keydown="sidebarKey"
      @dblclick="sidebarReset"
    ></div>

    <EditorArea class="editor-area-slot" />

    <div
      class="pane-resizer editor-resizer-h"
      :class="{ dragging: terminalDragging }"
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize the terminal (drag, arrow keys, double-click to reset)"
      tabindex="0"
      @pointerdown="terminalDown"
      @pointermove="terminalMove"
      @pointerup="terminalUp"
      @pointercancel="terminalUp"
      @keydown="terminalKey"
      @dblclick="terminalReset"
    ></div>

    <EditorTerminal ref="terminal" class="editor-terminal-slot" />

    <!-- The write-feedback stack. Mounted here rather than in EditorArea so it
         belongs to the Editor *screen* (tree, pane and terminal alike) instead
         of to one of its panes — it teleports to `<body>` anyway, and staying
         inside this lazy chunk keeps the Editor's weight where it is now. -->
    <EditorNoticeStack />
  </div>
</template>
