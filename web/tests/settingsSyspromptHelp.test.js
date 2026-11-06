import { beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import SettingsDialog from '../src/components/SettingsDialog.vue'
import { useBenchmark } from '../src/composables/useBenchmark'

// am-04 §9: agent modes move ``LLM_SYSPROMPT`` from *the* WorkBench prompt to
// the *fallback* WorkBench prompt, and a settings key whose value silently
// stopped being used is worse than one that moved. The key set itself does not
// change (``SETTING_KEYS``, ``_render_env()`` and ``.env.example`` all keep the
// one entry — pinned on the Python side by tests/test_api.py), so the sentence
// beside the field is what has to say what it now means.
//
// Read through the mounted dialog rather than a string constant: the point is
// that a user in the LLM section meets the caption under the textarea.

const { state } = useBenchmark()

function llmSettings() {
  return {
    serverUrl: 'http://127.0.0.1:8080/v1',
    apiKey: '',
    endpoint: 'model-a',
    sysprompt: 'Answer concisely.',
    // The payload shape since the workspace-file split: sampling fields are
    // numbers or null, the vision flag a boolean, and no debug field at all.
    temp: 0.2, topP: null, topK: null, minP: null, repeatPenalty: null, presencePenalty: null,
    reasoning: '', supportsVision: false, questionSet: 'legacy',
    verifier: { serverUrl: '', apiKey: '', endpoint: '', sysprompt: '' },
  }
}

function syspromptBlock(wrapper) {
  const label = wrapper.findAll('label').find((node) =>
    node.text().trim().startsWith('System prompt'),
  )
  if (!label) throw new Error('the LLM section has no System prompt field')
  // The caption *directly under this field* — the verifier section has a
  // System prompt textarea of its own with no such caption, and a sentence
  // borrowed from somewhere else in the dialog would not be the help text.
  const caption = label.element.nextElementSibling
  return `${caption.tagName} ${caption.className}: ${caption.textContent}`
}

function syspromptLabels(wrapper) {
  return wrapper.findAll('label').filter((node) =>
    node.text().trim().startsWith('System prompt'),
  )
}

describe('the LLM section says what System prompt is used for', () => {
  beforeEach(() => {
    state.settings = llmSettings()
  })

  it('names both the benchmark and the mode fallback', () => {
    const wrapper = mount(SettingsDialog)
    const raw = syspromptBlock(wrapper)
    const text = raw.replace(/\s+/g, ' ')

    expect(text).toMatch(/benchmark/)
    expect(text).toMatch(/WorkBench/)
    expect(text).toMatch(/fallback/)
    // It points at the file the escape hatch lives in, so the reader can find
    // it: an empty persona, not a checkbox.
    expect(text).toMatch(/systemMessage/)
    // A caption under *this* field, in the dialog's own help style.
    expect(text.startsWith('P caption:')).toBe(true)
    // The verifier's System prompt keeps its old meaning and gets no such
    // sentence: the second field is the one with nothing after it.
    const labels = syspromptLabels(wrapper)
    expect(labels).toHaveLength(2)
    expect(labels[1].element.nextElementSibling).toBeNull()
    wrapper.unmount()
  })
})
