<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useWorkspaceSearch } from '../composables/useWorkspaceSearch'

// The search popup itself: a `<dialog>` + `showModal()` over `useWorkspaceSearch`
// state — the same shape EditorTreeDialogs.vue, BatchModeDialog.vue and
// RegistrationMemoryDlg.vue follow, and for the same reason. Nothing here is a
// second store: the query, the scope, the last answer and the busy marker are the
// composable's (§2.7), so closing the popup, switching to the WorkBench tab and
// back, or re-raising the dialog shows the *same* table without another request.
//
// There is deliberately no `v-if` on the `<dialog>`: unmounting it throws away the
// caret position in the field and any scrolling in the results, which is exactly
// what makes re-opening pleasant. The watch below reconciles the element with
// `state.open` instead.
//
// What this component *does* own is the presentation of the payload, and two
// rules live only here:
//
//   * **a snippet is a string, never markup** (D12). It is a line of somebody's
//     file, so it is interpolated and gets no HTML at all: no directive that
//     parses it, and no trip through the app's HTML renderer (that one renders
//     *rendered* HTML, and piping file bytes through a sanitiser here would
//     satisfy a rule about something else). The matched span is a `<span>`
//     wrapped around a *slice* of the string, so three text nodes carry it;
//   * **the footer sentence is the server's** (D8). That sentence is written once,
//     in `editorfs/search.py`, from the caps in `editorfs/constants`, and this
//     component renders it as it arrives — reading the payload's field exactly
//     once. Composing a count here would have to know those cap numbers a second
//     time, and the two would drift.
//
// Nothing raises this dialog yet — search-07 wires the menu row and the header
// icon — and nothing consumes `state.pendingReveal` yet (search-08). A row click
// calls `openHit`, which is the composable's own close (D13), so this component
// never closes the popup itself and the close keeps one owner.
const { state, closeDialog, run, openHit } = useWorkspaceSearch()

const dialogEl = ref(null)
const fieldEl = ref(null)

// The house pattern (EditorTreeDialogs.vue:152-161): the state decides, the
// element follows. The focus call is part of the same tick — §2.3's "one
// `<input>`, autofocused" is done here rather than trusted to the attribute,
// because the element is not re-mounted per open and the browser's own autofocus
// only runs when it is inserted.
watch(
  () => state.open,
  async (open) => {
    await nextTick()
    const dialog = dialogEl.value
    if (dialog) {
      if (open && !dialog.open) dialog.showModal()
      else if (!open && dialog.open) dialog.close()
    }
    if (open) fieldEl.value?.focus()
  },
)

// Esc and a click outside both end in `closeDialog()`, which keeps query, scope
// and result (D14). The native `@close` writes the state instead of leaving it
// disagreeing with the element — otherwise the next `openFor()` on the same scope
// would find `state.open` already true and never show the dialog again.
function onBackdropClick(event) {
  if (event.target === dialogEl.value) closeDialog()
}

function onNativeClose() {
  if (state.open) closeDialog()
}

// Enter in the field submits the search. There is no `<form>` in this dialog, so
// that is explicit rather than inherited: a bare input would otherwise do nothing,
// and a `<form>` added just to get Enter would navigate.
function onFieldEnter(event) {
  event.preventDefault()
  run()
}

const scope = computed(() => state.scope)

// §2.8: the kind is part of the sentence, because `docs/old` as a folder and
// `docs/old` as a file are two different searches and the label is read-only in
// both cases.
const SCOPE_KIND_LABEL = {
  folder: 'folder',
  file: 'file',
  workspace: 'whole workspace',
}
const scopeKindLabel = computed(() => SCOPE_KIND_LABEL[scope.value?.kind] ?? '')

const result = computed(() => state.result)
const matches = computed(() => result.value?.matches ?? [])
const running = computed(() => state.status === 'running')
// `state.error` is only the *current* answer while the status says so: a later
// search clears it, and a stale error line over a fresh table would be a lie.
const error = computed(() => (state.status === 'error' ? state.error : null))
// The one sentence about what the scan could not give. It is read straight off
// the payload (D8): the caps are backend numbers, so the dialog has no
// arithmetic of its own to be right about.
const footerLine = computed(() => result.value?.footerNote ?? '')
// "no matches" is an answer, so it belongs to a *finished* search. Before the
// first one the results area is simply empty, and while one runs the previous
// table (or nothing) stays where it was.
const noMatches = computed(() => state.status === 'done' && matches.value.length === 0)

// One row of the Contract, with the slice bounds the highlight needs.
//
// `column` is 1-based and names the *start of the match*; the snippet is that
// line trimmed at `MAX_SEARCH_SNIPPET_BYTES`, so the offset of the match inside
// the snippet is `column - 1` — clamped to the snippet's length, because a line
// cut at 240 characters can end *before* the match begins. The end is clamped too,
// and `matchLength: 0` is legal (a zero-width regex match, or that same trim): the
// highlight span is then empty rather than the whole line. A wrong slice puts the
// highlight on the wrong text, which is worse than no highlight at all.
const rows = computed(() =>
  matches.value.map((hit) => {
    const snippet = typeof hit.snippet === 'string' ? hit.snippet : ''
    const width = Number.isFinite(hit.matchLength) ? Math.max(hit.matchLength, 0) : 0
    const start = clampIndex(hit.column - 1, snippet.length)
    const end = clampIndex(start + width, snippet.length)
    return {
      hit,
      before: snippet.slice(0, start),
      mark: snippet.slice(start, end),
      after: snippet.slice(end),
    }
  }),
)

function clampIndex(value, length) {
  if (!Number.isFinite(value)) return 0
  return Math.min(Math.max(value, 0), length)
}
</script>

<template>
  <dialog
    ref="dialogEl"
    class="editor-search-dialog"
    aria-labelledby="editor-search-title"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <article>
      <header>
        <h2 id="editor-search-title">Search</h2>
      </header>

      <!-- 1. The scope, read-only and before the field (2.8): it is derived from
           the row the gesture was made on, so a typed scope would name a search
           the tree never offered. -->
      <!-- The space before the kind is in the text, not in CSS: the template
           compiler drops the whitespace between two elements on separate lines,
           and `docs/old(folder)` is what a screen reader would then say. -->
      <p class="caption editor-search-scope">
        Search in:
        <span class="editor-search-scope-path">{{ scope?.label }}</span>
        <span class="editor-search-scope-kind"> ({{ scopeKindLabel }})</span>
      </p>

      <!-- 2. The query, and the one button that asks for it. Enter in the field
           submits through `onFieldEnter`; there is no `<form>` anywhere in this
           dialog, so nothing can navigate. The field stays *editable* while a
           search runs — the button is what goes idle-disabled, so the next query
           can be typed before the current one has answered. -->
      <div class="editor-search-field">
        <label class="editor-search-field-label">
          Query
          <input
            ref="fieldEl"
            v-model="state.query"
            class="editor-search-query"
            type="text"
            autofocus
            spellcheck="false"
            autocomplete="off"
            placeholder="text to look for"
            :aria-describedby="running ? 'editor-search-status' : undefined"
            @keydown.enter="onFieldEnter"
          />
        </label>
        <button
          class="editor-search-submit"
          type="button"
          :disabled="running"
          title="Search again with this query (Enter)"
          @click="run"
        >
          Search
        </button>
      </div>

      <!-- 3. The two switches, on one line, both default off, and neither one
           disabled while a search runs: the answer to a *different* combination
           may be the next thing typed, and the composable's generation token
           already discards a late reply. The regex one names the engine the
           pattern runs in (2.3) — the answer did not come from a JS `RegExp`, and
           promising parity with one would be a lie about `\d`, `$1` and named
           groups. The `title` is on the label, because that is what is visible
           and hoverable: Pico positions the checkbox box itself absolutely. -->
      <div class="editor-search-switches">
        <label class="checkbox">
          <input v-model="state.caseSensitive" type="checkbox" />
          Case sensitive
        </label>
        <label class="checkbox" title="regex — Python syntax">
          <input v-model="state.regex" type="checkbox" />
          Regex
        </label>
      </div>

      <!-- A 400 for an invalid pattern is `re`'s own message, caret position and
           all (D5), shown here rather than in the notice stack: the fix is in the
           field just above, and a bottom-centre toast is nowhere near the caret
           that typed it. -->
      <p v-if="error" class="error-line editor-search-error" role="alert">
        {{ error.detail }}
      </p>
      <p
        id="editor-search-status"
        v-if="running"
        class="caption editor-search-busy"
        role="status"
        aria-live="polite"
      >
        searching…
      </p>

      <!-- 4. The results stay on screen through both an error and a re-search:
           the previous answer is the thing the new one is compared against, so
           blanking it would destroy the only context the query has. -->
      <div class="editor-search-results">
        <table v-if="rows.length" class="editor-search-table">
          <thead>
            <tr>
              <th scope="col">path</th>
              <th scope="col">line</th>
              <th scope="col">snippet</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, index) in rows"
              :key="`${row.hit.path}:${row.hit.line}:${row.hit.column}:${index}`"
              class="editor-search-row"
            >
              <td class="editor-search-cell-path">
                <!-- A real button, like the tab strip's chips: keyboard-reachable,
                     and it carries the path in `title` with no handler needed. -->
                <button
                  class="editor-search-hit"
                  type="button"
                  :title="`Open ${row.hit.path} at line ${row.hit.line}`"
                  @click="openHit(row.hit)"
                >
                  {{ row.hit.path }}
                </button>
              </td>
              <td class="editor-search-cell-line">{{ row.hit.line }}</td>
              <!-- D12: three text nodes, never markup built from the file's
                   bytes. The cell's `title` holds the same string (§2.4: the full
                   line lives in the row's title), and a `title` is an attribute,
                   not a subtree, so it cannot inject either. The awkward `>`
                   placement keeps the three nodes adjacent — a newline here would
                   put a space inside the snippet. -->
              <td class="editor-search-cell-snippet" :title="row.hit.snippet">
                <code class="editor-search-snippet"
                  >{{ row.before }}<span class="editor-search-mark">{{ row.mark }}</span
                  >{{ row.after }}</code
                >
              </td>
            </tr>
          </tbody>
        </table>
        <p v-else-if="noMatches" class="caption editor-search-empty">no matches</p>
      </div>

      <!-- 5. The footer belongs to `result`, which by definition is the last
           *successful* search — so when an error line is up, this sentence
           describes the table that is still on screen, not that error. The
           pairing looks like a bug and is not one: answer and footer came from
           the same request. -->
      <footer v-if="footerLine" class="caption editor-search-footer">{{ footerLine }}</footer>
    </article>
  </dialog>
</template>
