<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import {
  addModeToDraft,
  blockingFindings,
  createDisabledFor,
  deleteDisabledFor,
  deleteModeFromDraft,
  draftIsDirty,
  draftWillCreateFile,
  entryBlocksSave,
  entryFindingsFor,
  fetchAgentModesDocument,
  fileFindingsFor,
  findingFieldFor,
  idEditableFor,
  idValidityProblem,
  personaByteCount,
  personaLineBudget,
  saveAgentModesDocument,
  saveDisabledFor,
  slugForDraft,
  tabCaptionFor,
  useAgentModes,
} from '../composables/useAgentModes'

// The agent-modes editor (eds-06): one card that edits the workspace's modes
// file the way SettingsDialog edits the model connection — fields instead of
// JSON, the caps enforced on the way in, and the file's own problems named where
// they can be fixed.
//
// It decides **nothing**. Every rule this card renders is a plain function from
// `useAgentModes` (eds-05): the slug, the counters, the Save-disabled verdict,
// the Delete refusal, the tab caption, which finding belongs on which field. The
// component only turns those answers into fields, so a rule cannot drift from
// the dialog that shows it — and no cap is spelled here, because every number
// arrives in the document payload's `caps`.
//
// It opens the way the other cards open (watch + showModal + backdrop click +
// native-close sync, the SettingsDialog / RegistrationMemoryDlg / BatchModeDialog
// pattern), and reuses their two-column shape: a vertical nav beside the editor
// column, one tab per mode the **file** declares in the file's own order, plus
// Common (D5). The app-owned mode is absent from everything rendered here — it
// is not in the payload's entries, not in the default dropdown and not in any
// list (D4), and this file never names its id.
//
// The dialog is a browser surface: it reaches the file through the two
// purpose-built calls the composable owns (its document read and its whole-
// document save), never through a file-tree route and never through a path
// parameter — which is why no route path is spelled anywhere in this file.

const props = defineProps({
  open: { type: Boolean, default: false },
  // D10's enablement, passed in by whoever mounts this (eds-07): the pencil is
  // inert while a reply streams and while a lot runs. The dialog does not read
  // useChat or useBatch — the same acyclic rule useAgentModes itself follows.
  batchActive: { type: Boolean, default: false },
})
const emit = defineEmits(['close'])

const { editor } = useAgentModes()

const dialogEl = ref(null)
const draft = ref(null)
const activeTab = ref('common')
const createLabel = ref('')
const pendingDelete = ref(null) // one Delete awaiting its confirm, per tab

// The draft is this dialog's state, not the singleton's: a card that reads a
// broken file must leave the composer's dropdown exactly as it was.
const loaded = computed(() => draft.value !== null)
const entries = computed(() => draft.value?.entries ?? [])
const willCreate = computed(() => loaded.value && draftWillCreateFile(draft.value))
const dirty = computed(() => loaded.value && draftIsDirty(draft.value))
const fileFindings = computed(() => (loaded.value ? fileFindingsFor(draft.value) : { errors: [], notes: [] }))
// The count the footer states and the Save button obeys are the *same* verdict:
// `blockingFindings` is the composable's, including the draft's own re-measured
// persona/default/count findings and excluding a finding the form has fixed.
const blocking = computed(() => (loaded.value ? blockingFindings(draft.value).length : 0))

const saveBlocked = computed(() =>
  !loaded.value || editor.saving || saveDisabledFor(draft.value, { saving: editor.saving }),
)
const createBlocked = computed(() => !loaded.value || editor.saving || createDisabledFor(draft.value))

// The Common tab's default dropdown: the editable modes the *file* declares, in
// the file's order. It cannot select the app-owned mode because the payload
// never carries it, and it never lists an entry the dialog has no tab for.
const defaultableModes = computed(() => entries.value.filter((entry) => entry.hasForm))

const conflictLine = computed(() => {
  const conflict = editor.conflict
  if (!conflict) return null
  return (
    `The file changed on disk since this dialog read it, so nothing was written: ` +
    `${conflict.path || 'the agent-modes file'} now holds mtime ${conflict.diskMtime}, ` +
    `this dialog holds ${conflict.documentMtime}. Your draft is still here — close without ` +
    `saving to keep the file as it is, or edit it outside the app and reopen.`
  )
})

const serverFindings = computed(() => editor.invalidFindings)

// Open/fetch/close, following the SettingsDialog pattern. Opening *is* the read:
// the document is uncached on purpose (it is the save's baseline), so a card
// that has been open for ten minutes knows it.
//
// `immediate` because the pencil mounts this card with `open` already true (the
// composer holds the flag): a watch that only fires on a *change* would leave a
// card mounted while open never reading the file.
watch(
  () => [props.open, editor.loading],
  async () => {
    await nextTick()
    const dialog = dialogEl.value
    if (!dialog) return
    if (props.open && !dialog.open) {
      dialog.showModal()
      if (!loaded.value) await openDocument()
    } else if (!props.open && dialog.open) {
      dialog.close()
    }
  },
  { immediate: true },
)

async function openDocument() {
  activeTab.value = 'common'
  createLabel.value = ''
  pendingDelete.value = null
  draft.value = await fetchAgentModesDocument()
  // D3: a finding *opens its entry's tab*. So the card lands on the first entry
  // holding Save (`entryBlocksSave`, the composable's verdict), and only
  // otherwise on the first mode the file declares. With nothing declared, the
  // Common tab is the one that opens.
  const opened = draft.value
  if (!opened) return
  const broken = opened.entries.find((entry) => entryBlocksSave(opened, entry))
  focusTab(tabIdFor(broken ?? opened.entries[0] ?? null), /* focusFields = */ false)
}

function tabIdFor(entry) {
  return entry ? `entry:${entry.key}` : 'common'
}

function onBackdropClick(event) {
  if (event.target === dialogEl.value) cancel()
}

// Esc (or a programmatic close) is Cancel: it writes nothing, so a file that
// existed stays byte-identical and a file that did not exist still does not.
function onNativeClose() {
  if (props.open) emit('close')
}

function cancel() {
  pendingDelete.value = null
  draft.value = null
  emit('close')
}

// Create agent (D5): append the tab immediately and focus it. Nothing is
// written — the new mode exists only in this card's state until Save. Its id is
// the slug of its label, and it stays editable only while the mode is new.
async function createAgent() {
  if (!loaded.value || createBlocked.value) return
  const entry = addModeToDraft(draft.value, createLabel.value)
  focusTab(tabIdFor(entry), true)
}

// Focus what the user is about to type into: the first field of the section that
// is now shown. `v-show` keeps every section in the DOM, so the query is scoped
// to the active tab's own section — otherwise a Create would focus the *first*
// mode's Label instead of the new mode's.
async function focusTab(id, focusFields = true) {
  activeTab.value = id
  await nextTick()
  if (!focusFields) return
  const key = id === 'common' ? 'common' : id.slice('entry:'.length)
  // Matched through `dataset`, not an attribute selector: an entry key is a raw
  // JSON key from a hand-edited file, and one holding a quote or a bracket would
  // make such a selector throw. The focus is a convenience, never a crash.
  const section = [...(dialogEl.value?.querySelectorAll('section[data-tab]') ?? [])].find(
    (node) => node.dataset.tab === key,
  )
  section?.querySelector('input:not([disabled]), textarea:not([disabled])')?.focus()
}

// A new mode's id follows its Label field until the user edits it by hand: the
// slug is the composable's rule, and an id typed here is judged by the same one.
function onNewLabelChange(entry) {
  if (!idEditableFor(entry)) return
  entry.key = slugForDraft(draft.value, entry.label)
}

function idProblem(entry) {
  return loaded.value ? idValidityProblem(draft.value, entry) : null
}

function findingsOn(entry) {
  return loaded.value ? entryFindingsFor(draft.value, entry) : []
}

// The findings that name one field of this entry — the loader's own sentences,
// placed next to the field they talk about (D3). `findingFieldFor` reads the
// JSON pointer out of the message, so "next to" is the loader's fact and not a
// guess made here.
function fieldFindings(entry, field) {
  return findingsOn(entry).filter((finding) => findingFieldFor(entry, finding.message) === field)
}

// The findings that name no field this form has: an unknown key, a finding about
// the entry as a whole. They are listed at the end of the tab, because there is
// no field to put them on — and they are the ones the form cannot fix, only
// delete (D3).
function unfixedFindings(entry) {
  return findingsOn(entry).filter(
    (finding) => !['label', 'mcptools', 'lsptools', 'systemMessage'].includes(
      findingFieldFor(entry, finding.message),
    ),
  )
}

function personaCounters(entry) {
  const text = typeof entry.personaText === 'string' ? entry.personaText : ''
  return {
    bytes: personaByteCount(text),
    cap: draft.value?.caps?.maxPersonaBytes ?? null,
    budget: personaLineBudget(draft.value, text),
  }
}

// The built-in persona for this entry, or null when the app has none for it —
// which is what makes Reset appear only for an entry benchmarkit ships a persona
// for (a user-created mode has nothing to reset to).
function builtInPersona(entry) {
  const agents = draft.value?.defaults?.agents
  if (!agents || !Object.prototype.hasOwnProperty.call(agents, entry.key)) return null
  const raw = agents[entry.key]?.systemMessage
  if (typeof raw === 'string') return raw
  if (Array.isArray(raw) && raw.every((line) => typeof line === 'string')) return raw.join('\n')
  return null
}

function resetPersona(entry) {
  const persona = builtInPersona(entry)
  if (persona !== null) entry.personaText = persona
}

// Delete, with a confirm, refused for the mode that is currently the default (a
// default must name a declared mode). The refusal is the composable's answer;
// this card only puts it next to the button.
function askDelete(entry) {
  if (deleteDisabledFor(draft.value, entry.key)) return // the named refusal is already shown
  pendingDelete.value = entry.key
}

function confirmDelete() {
  const key = pendingDelete.value
  if (!key || !loaded.value) return
  deleteModeFromDraft(draft.value, key)
  pendingDelete.value = null
  if (activeTab.value === `entry:${key}`) activeTab.value = 'common'
}

function dismissDelete() {
  pendingDelete.value = null
}

function deleteRefusal(entry) {
  return deleteDisabledFor(draft.value, entry.key)
    ? 'This is the default mode — pick another default before deleting it.'
    : null
}

async function save() {
  if (!loaded.value || saveBlocked.value) return
  const ok = await saveAgentModesDocument(draft.value, { batchActive: props.batchActive })
  if (ok) cancel()
  // A failure keeps the card open with the reason named (a 409 conflict, a
  // server 422's findings, a transport error): nothing is merged, nothing is
  // overwritten and the draft survives.
}

// Cancel writes nothing: the draft is dropped, so a file that existed is
// byte-identical and a file that did not exist still does not.
function cancelWritesNothing() {
  return dirty.value ? 'Cancel discards these edits and writes nothing.' : 'Cancel writes nothing.'
}
</script>

<template>
  <dialog
    ref="dialogEl"
    class="agent-modes-dialog"
    aria-labelledby="agent-modes-title"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <article>
      <header>
        <h2 id="agent-modes-title" class="settings-title">Agent modes</h2>
      </header>
      <p class="caption agent-modes-hint">
        This workspace's modes are saved to
        <code>{{ draft?.source?.path || '.nojae/agent-modes.json' }}</code>. A mode
        replaces the persona sent to the model and can switch a tool hub off; it
        can only narrow what Settings already allows. Saving here applies right
        away; editing the file by hand needs a server restart.
      </p>

      <p v-if="editor.loading" class="caption" role="status">Reading the modes file…</p>
      <p v-if="editor.error" class="alert-error" role="alert">{{ editor.error }}</p>
      <p v-if="conflictLine" class="alert-error agent-modes-conflict" role="alert">{{ conflictLine }}</p>
      <p v-if="willCreate" class="caption agent-modes-creates-file" role="status">
        This workspace has no modes file yet — saving will create it, starting from
        benchmarkit's default modes.
      </p>

      <p v-if="fileFindings.errors.length" class="alert-error agent-modes-file-findings" role="alert">
        <span v-for="finding in fileFindings.errors" :key="finding.message" class="agent-modes-finding">
          {{ finding.message }}
        </span>
      </p>
      <p v-if="fileFindings.notes.length" class="caption agent-modes-file-notes">
        <span v-for="finding in fileFindings.notes" :key="finding.message" class="agent-modes-finding">
          {{ finding.message }}
        </span>
      </p>
      <p v-if="serverFindings.length" class="alert-error agent-modes-server-findings" role="alert">
        <span v-for="finding in serverFindings" :key="finding.message" class="agent-modes-finding">
          {{ finding.message }}
        </span>
      </p>

      <!-- The form is the *draft*'s rendering: with no draft (a read that
           failed, or one still in flight) there is nothing to edit, and the
           lines above are what the card says instead. -->
      <form v-if="loaded" class="settings-form agent-modes-form" @submit.prevent>
        <div class="settings-body agent-modes-body">
          <nav class="settings-nav agent-modes-nav" aria-label="Agent modes">
            <ul>
              <li>
                <button
                  type="button"
                  :class="{ active: activeTab === 'common', 'agent-modes-tab': true }"
                  :aria-current="activeTab === 'common' ? 'true' : undefined"
                  @click="activeTab = 'common'"
                >
                  Common
                </button>
              </li>
              <li v-for="entry in entries" :key="entry.key">
                <button
                  type="button"
                  :class="{
                    'agent-modes-tab': true,
                    active: activeTab === `entry:${entry.key}`,
                    'agent-modes-tab--broken': entryBlocksSave(draft, entry),
                  }"
                  :aria-current="activeTab === `entry:${entry.key}` ? 'true' : undefined"
                  @click="activeTab = `entry:${entry.key}`"
                >
                  {{ tabCaptionFor(entry) }}
                </button>
              </li>
            </ul>
            <label class="agent-modes-create">
              New agent label
              <input
                v-model="createLabel"
                type="text"
                spellcheck="false"
                placeholder="Reviewer"
                autocomplete="off"
                :disabled="createBlocked"
              />
            </label>
            <button
              class="agent-modes-create-button"
              type="button"
              :disabled="createBlocked"
              @click="createAgent"
            >
              + Create agent
            </button>
          </nav>

          <div class="settings-editor agent-modes-editor">
            <section data-tab="common" v-show="activeTab === 'common'">
              <h3 class="settings-section-title">Common</h3>
              <label>
                Default mode
                <select v-model="draft.default" :disabled="editor.saving">
                  <option v-if="!defaultableModes.length" value="" disabled>
                    This file declares no modes to default to
                  </option>
                  <option v-for="entry in defaultableModes" :key="entry.key" :value="entry.key">
                    {{ tabCaptionFor(entry) }}
                  </option>
                </select>
              </label>
              <p class="caption">
                The mode a chat turn answers as when the composer changes nothing.
                Create an agent above to add a mode to this file — it appears as a
                tab right away and reaches the file only when you Save.
              </p>
            </section>

            <section
              v-for="entry in entries"
              :key="entry.key"
              :data-tab="entry.key"
              v-show="activeTab === `entry:${entry.key}`"
            >
              <h3 class="settings-section-title">{{ tabCaptionFor(entry) }}</h3>

              <template v-if="!entry.hasForm">
                <!-- An entry with no fields to put in a form (its value is not an
                     object, its id is outside the pattern): the tab holds the
                     finding and a Delete only, so the file's own text is visible
                     and removable — Delete is the only way out of it. -->
                <p class="caption">
                  This entry has no fields to edit. It is listed because the file
                  declares it; saving cannot keep it.
                </p>
                <p
                  v-for="finding in findingsOn(entry)"
                  :key="finding.message"
                  class="alert-error agent-modes-field-finding"
                  role="alert"
                >
                  {{ finding.message }}
                </p>
              </template>

              <template v-else>
                <label>
                  Label
                  <input
                    v-model="entry.label"
                    type="text"
                    spellcheck="false"
                    autocomplete="off"
                    :disabled="editor.saving"
                    @change="onNewLabelChange(entry)"
                  />
                </label>
                <p
                  v-for="finding in fieldFindings(entry, 'label')"
                  :key="finding.message"
                  class="alert-error agent-modes-field-finding"
                  role="alert"
                >
                  {{ finding.message }}
                </p>
                <p class="caption">
                  The name the composer's dropdown shows. Empty falls back to the
                  id. A long label is trimmed by the app — it is presentation, not
                  a rule about the file.
                </p>

                <label v-if="idEditableFor(entry)">
                  Id
                  <input
                    v-model="entry.key"
                    type="text"
                    spellcheck="false"
                    autocomplete="off"
                    :disabled="editor.saving"
                  />
                </label>
                <p v-else class="caption agent-modes-id-readonly">
                  Id: <code>{{ entry.key }}</code> — an existing mode's id cannot be
                  renamed (that would orphan every saved session naming it; delete
                  and re-create instead).
                </p>
                <p v-if="idProblem(entry)" class="alert-error agent-modes-field-finding" role="alert">
                  {{ idProblem(entry) }}
                </p>

                <div class="agent-modes-switches">
                  <label class="agent-modes-switch">
                    <input type="checkbox" role="switch" v-model="entry.mcptools" :disabled="editor.saving" />
                    MCP tools
                  </label>
                  <label class="agent-modes-switch">
                    <input type="checkbox" role="switch" v-model="entry.lsptools" :disabled="editor.saving" />
                    LSP tools
                  </label>
                </div>
                <p
                  v-for="finding in [...fieldFindings(entry, 'mcptools'), ...fieldFindings(entry, 'lsptools')]"
                  :key="finding.message"
                  class="alert-error agent-modes-field-finding"
                  role="alert"
                >
                  {{ finding.message }}
                </p>
                <p class="caption agent-modes-gating-hint">
                  These switches narrow what the mode may reach — a hub that is off
                  in Settings stays off whatever this says.
                </p>

                <label>
                  System prompt
                  <textarea
                    v-model="entry.personaText"
                    rows="10"
                    spellcheck="false"
                    :disabled="editor.saving || entry.personaText === null"
                    placeholder="Leave empty to use the configured system prompt."
                  ></textarea>
                </label>
                <p
                  v-for="finding in fieldFindings(entry, 'systemMessage')"
                  :key="finding.message"
                  class="alert-error agent-modes-field-finding agent-modes-persona-finding"
                  role="alert"
                >
                  {{ finding.message }}
                </p>
                <p class="caption agent-modes-persona-count">
                  {{ personaCounters(entry).bytes }} / {{ personaCounters(entry).cap }} bytes
                </p>
                <p v-if="personaCounters(entry).budget" class="caption agent-modes-persona-lines">
                  remaining line cap: {{ personaCounters(entry).budget.remaining }}
                  ({{ personaCounters(entry).budget.budget }} lines is this app's
                  writing budget)
                </p>
                <p
                  v-if="personaCounters(entry).budget?.warning"
                  class="caption agent-modes-persona-warning"
                  role="status"
                >
                  {{ personaCounters(entry).budget.warning }}
                </p>
                <p class="caption">
                  Empty means "use the configured system prompt" — the Settings
                  one, which is what a mode with no persona answers with.
                </p>

                <button
                  v-if="builtInPersona(entry) !== null"
                  class="secondary contrast agent-modes-reset"
                  type="button"
                  :disabled="editor.saving"
                  @click="resetPersona(entry)"
                >
                  Reset this persona
                </button>
                <p v-if="builtInPersona(entry) !== null" class="caption">
                  Puts benchmarkit's default persona for this mode back in the
                  field. It is an edit, not a save: it still needs Save, and Cancel
                  discards it.
                </p>

                <p
                  v-for="finding in unfixedFindings(entry)"
                  :key="finding.message"
                  class="alert-error agent-modes-field-finding"
                  role="alert"
                >
                  {{ finding.message }}
                </p>
              </template>

              <div class="agent-modes-delete">
                <button
                  class="secondary contrast agent-modes-delete-button"
                  type="button"
                  :disabled="editor.saving || deleteDisabledFor(draft, entry.key)"
                  @click="askDelete(entry)"
                >
                  Delete this mode
                </button>
                <p v-if="deleteRefusal(entry)" class="caption alert-error agent-modes-delete-refusal" role="status">
                  {{ deleteRefusal(entry) }}
                </p>
                <div v-if="pendingDelete === entry.key" class="agent-modes-delete-confirm">
                  <!-- The confirm is asked *in place*, not by a nested dialog:
                       this card is already a modal. -->
                  <p class="caption" role="status">
                    Delete "{{ tabCaptionFor(entry) }}" from this file? Saving is what
                    removes it; Cancel writes nothing.
                  </p>
                  <button class="agent-modes-delete-yes" type="button" @click="confirmDelete">
                    Delete
                  </button>
                  <button class="secondary contrast" type="button" @click="dismissDelete">
                    Keep it
                  </button>
                </div>
              </div>
            </section>
          </div>
        </div>

        <footer class="settings-actions">
          <p v-if="blocking" class="caption agent-modes-blocking-note" role="status">
            {{ blocking }} problem{{ blocking === 1 ? '' : 's' }} in this document must
            be fixed or removed before it can be saved — no entry is dropped silently.
          </p>
          <button class="secondary contrast" type="button" :disabled="editor.saving" @click="cancel">
            Cancel
          </button>
          <p v-if="dirty" class="caption agent-modes-dirty-note" role="status">{{ cancelWritesNothing() }}</p>
          <button
            class="agent-modes-save"
            type="button"
            :disabled="saveBlocked"
            @click="save"
          >
            {{ editor.saving ? 'Saving…' : willCreate ? 'Create file' : 'Save' }}
          </button>
        </footer>
      </form>
    </article>
  </dialog>
</template>
