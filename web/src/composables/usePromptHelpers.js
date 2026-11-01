import { reactive } from 'vue'

// Must mirror the backend constants (SLOT_COUNT, DEFAULT_COLOR,
// CONTENT_MAX_CHARS) in src/core/prompt_helpers.py — a hand mirror, so keep
// the three values in sync when either side changes.
export const PROMPT_HELPER_SLOT_COUNT = 9
export const PROMPT_HELPER_DEFAULT_COLOR = '#808080'
export const PROMPT_HELPER_CONTENT_MAX_CHARS = 32000

function emptySlot() {
  return { color: PROMPT_HELPER_DEFAULT_COLOR, content: '' }
}

const state = reactive({
  // Fixed-size slot list ({ color, content }) from /api/prompt-helpers; always
  // PROMPT_HELPER_SLOT_COUNT long so the bullets render before the first load.
  slots: Array.from({ length: PROMPT_HELPER_SLOT_COUNT }, emptySlot),
  loaded: false,
  saving: false,
  error: null,
})

function describeError(body, fallback) {
  const detail = body?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      const field = Array.isArray(item?.loc)
        ? item.loc.filter((part) => part !== 'body').join('.')
        : ''
      return field ? `${field}: ${item?.msg ?? 'invalid'}` : item?.msg ?? 'invalid'
    })
    if (messages.length) return messages.join(' · ')
  }
  return fallback
}

export function usePromptHelpers() {
  async function loadPromptHelpers() {
    try {
      const res = await fetch('/api/prompt-helpers')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.error = describeError(body, `Could not load memory bullets (${res.status}).`)
        return
      }
      if (Array.isArray(body?.slots) && body.slots.length) {
        state.slots = body.slots
      }
      state.loaded = true
    } catch (exc) {
      state.error = `Could not reach the API: ${exc}`
    }
  }

  // Persists one slot; on success the response replaces the whole slot list.
  // Returns an ok flag so the dialog can decide whether to
  // close (and append) or stay open with the error shown.
  async function savePromptHelper(index, color, content) {
    if (state.saving) return false
    state.error = null
    if (content.length > PROMPT_HELPER_CONTENT_MAX_CHARS) {
      state.error = `Content exceeds the ${PROMPT_HELPER_CONTENT_MAX_CHARS} character cap.`
      return false
    }
    state.saving = true
    try {
      const res = await fetch('/api/prompt-helpers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index, color, content }),
      })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.error = describeError(body, `Could not save memory bullet (${res.status}).`)
        return false
      }
      if (Array.isArray(body?.slots) && body.slots.length) {
        state.slots = body.slots
        state.loaded = true
      }
      return true
    } catch (exc) {
      state.error = `Could not reach the API: ${exc}`
      return false
    } finally {
      state.saving = false
    }
  }

  return { state, loadPromptHelpers, savePromptHelper }
}
