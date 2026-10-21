import { reactive } from 'vue'

// Must mirror the backend MEMORY_* constants (src/config.py).
export const MEMORY_SLOT_COUNT = 9
export const MEMORY_DEFAULT_COLOR = '#808080'
export const MEMORY_CONTENT_MAX_CHARS = 32000

function emptySlot() {
  return { color: MEMORY_DEFAULT_COLOR, content: '' }
}

const state = reactive({
  // Fixed-size slot list ({ color, content }) from /api/memory; always
  // MEMORY_SLOT_COUNT long so the bullets render before the first load.
  registration: Array.from({ length: MEMORY_SLOT_COUNT }, emptySlot),
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

export function useMemory() {
  async function loadMemory() {
    try {
      const res = await fetch('/api/memory')
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.error = describeError(body, `Could not load memory bullets (${res.status}).`)
        return
      }
      if (Array.isArray(body?.registration) && body.registration.length) {
        state.registration = body.registration
      }
      state.loaded = true
    } catch (exc) {
      state.error = `Could not reach the API: ${exc}`
    }
  }

  // Persists one slot; on success the response replaces the whole
  // registration. Returns an ok flag so the dialog can decide whether to
  // close (and append) or stay open with the error shown.
  async function saveMemory(index, color, content) {
    if (state.saving) return false
    state.error = null
    if (content.length > MEMORY_CONTENT_MAX_CHARS) {
      state.error = `Content exceeds the ${MEMORY_CONTENT_MAX_CHARS} character cap.`
      return false
    }
    state.saving = true
    try {
      const res = await fetch('/api/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index, color, content }),
      })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        state.error = describeError(body, `Could not save memory bullet (${res.status}).`)
        return false
      }
      if (Array.isArray(body?.registration) && body.registration.length) {
        state.registration = body.registration
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

  return { state, loadMemory, saveMemory }
}
