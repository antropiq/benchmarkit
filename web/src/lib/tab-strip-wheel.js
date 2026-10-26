// Wheel-over-tab-strip scrolling, the way VS Code's editor tabs behave: a
// vertical notch scrolls the strip sideways, but only once its tabs overflow
// it. While they fit, there is nothing here to scroll, so the event is left
// alone and the pane under the pointer keeps its own vertical scrolling.
//
// Returns the pixels to add to the strip's `scrollLeft`, or null when the
// caller must not consume the event at all.
//
// `strip` is the scrolling element; it is measured, not assumed: the overflow
// test is what the strip itself reports at this moment.
export function wheelScrollDelta(strip, event) {
  if (!strip || strip.scrollWidth <= strip.clientWidth) return null
  // A trackpad already means sideways (`deltaX`); a mouse means it with the
  // vertical notch. Take whichever axis moved more, so a diagonal gesture is
  // read as the dominant one instead of adding the two.
  const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY
  if (!delta) return null
  // `deltaMode` says what the deltas are counted in: 0 = pixels, 1 = lines,
  // 2 = pages. Firefox reports lines, and treating 3 lines as 3 pixels makes
  // the wheel feel dead there, so each unit becomes pixels before scrolling.
  const unit = [1, 16, strip.clientWidth][event.deltaMode] ?? 1
  return delta * unit
}

// The listener itself: decide, consume, scroll. `passive: false` is what lets
// the preventDefault run — a passive listener cannot cancel, and without the
// cancel a notch over the tabs would also scroll the page behind them.
export function scrollStripWithWheel(strip, event) {
  const delta = wheelScrollDelta(strip, event)
  if (delta === null) return
  event.preventDefault()
  strip.scrollLeft += delta
}
