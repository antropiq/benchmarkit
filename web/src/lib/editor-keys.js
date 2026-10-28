// The Editor pane's window-level chords, decided apart from the listener that
// dispatches them — the same split `lib/tab-strip-wheel.js` makes for the
// strip: the component owns the DOM and the effects, this owns the question
// "which key, on which screen, with what modifiers — and so what?".
//
// It is worth a module because the answer is a table of preconditions a test
// can enumerate (another tab visible, no active buffer, the wrong modifiers)
// and because all three chords belong to *one* owner. Splitting `Ctrl+S` into a
// Monaco keybinding and `Ctrl+W` into this listener is how one of them quietly
// stops working: the pane is read-only while a reply streams, and a command
// registered on the editor is not reachable then. The window listener always is.

// Returns the action for this event, or null when the pane must not consume it
// at all — which is also the signal *not* to call `preventDefault()`, so the
// browser keeps its own behaviour for every chord this app does not own.
export function editorChord(event, context) {
  // The chord belongs to the Editor screen only: a Ctrl+W elsewhere must not
  // close a buffer the user is not looking at.
  if (context.activeTab !== 'editor') return null
  if (!(event.ctrlKey || event.metaKey)) return null
  // Alt is never part of these chords: accepting Ctrl+Alt+S would take a key
  // some window managers and input methods already own, and the app would never
  // know it had stolen it.
  if (event.altKey) return null

  const key = event.key.toLowerCase()
  if (key === 's') {
    // Ctrl+S saves the active buffer, Ctrl+Shift+S the whole dirty strip.
    return context.active ? { action: event.shiftKey ? 'save-all' : 'save' } : null
  }
  if (key === 'w' && !event.shiftKey) {
    // Whether the close needs a confirmation is the record's business
    // (`requestCloseTab` decides); this only says the gesture was meant here.
    // Ctrl+Shift+W is left alone: it means "close all" in the browsers this app
    // runs in, and taking a chord the app has no answer for is how a user loses
    // a whole window they meant to leave one tab in.
    return context.active ? { action: 'close' } : null
  }
  return null
}

// The single place the modifier test is written, so the listener reads as the
// intent rather than as a boolean expression: consume the event only for a
// chord this app owns, keeping the browser's Save-page / close-tab dialogs from
// appearing on top of the app's own answer.
export function applyEditorChord(event, context, handlers) {
  const chord = editorChord(event, context)
  if (!chord) return false
  event.preventDefault()
  handlers[chord.action]?.(event)
  return true
}
