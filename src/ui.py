"""UI widgets that have no native Streamlit equivalent."""

from typing import Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_STREAM_BOX_CSS = """
<style>
div[data-testid="stColumn"]:last-child
  > div[data-testid="stVerticalBlock"]
  > div[data-testid="stLayoutWrapper"]
  > div[data-testid="stVerticalBlock"] {
  border: 3px solid #FFFFFF;
  background-color: transparent;
}
</style>
"""


def render_stream_box_style() -> None:
    """Give the response-stream container a transparent fill and 3px yellow border."""
    st.markdown(_STREAM_BOX_CSS, unsafe_allow_html=True)

_LAYOUT_CSS = """
<style>
div[data-testid="stHeadingWithActionElements"] {
  text-align: center;
}
[data-testid="stMain"] {
  overflow-y: hidden;
}
</style>
"""


def render_layout_style() -> None:
    """Center the brand heading and hide the main column scrollbar."""
    st.markdown(_LAYOUT_CSS, unsafe_allow_html=True)

_DIGITAL_STYLE = (
    "font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,monospace;"
    "font-size:1.75rem;font-weight:600;letter-spacing:0.08em;"
    "font-variant-numeric:tabular-nums;color:inherit;"
)

_CHRONOMETER_TEMPLATE = """
<div id="benchmark-chronometer" style="__STYLE__">00:00:00</div>
<script>
(function () {
  var el = document.getElementById("benchmark-chronometer");
  if (!el) { return; }
  var start = __START_MS__;
  var end = __END_MS__;
  function format(ms) {
    var total = Math.max(0, Math.floor(ms / 1000));
    var parts = [
      Math.floor(total / 3600),
      Math.floor((total % 3600) / 60),
      total % 60,
    ];
    return parts.map(function (n) {
      return String(n).padStart(2, "0");
    }).join(":");
  }
  function tick() {
    if (!el.isConnected) { clearInterval(timer); return; }
    var now = end === null ? Date.now() : Math.min(Date.now(), end);
    el.textContent = format(now - start);
    if (end !== null && now >= end) { clearInterval(timer); }
  }
  if (start === null) { return; }
  var timer = setInterval(tick, 250);
  tick();
})();
</script>
"""


def chronometer_html(
    started_at: Optional[float],
    finished_at: Optional[float] = None,
) -> str:
    """Return the chronometer markup, seeded from epoch-seconds timestamps.

    The clock ticks client-side, so it keeps running while the benchmark
    blocks the Streamlit script. ``finished_at`` freezes the display.
    """
    start_ms = int(started_at * 1000) if started_at else "null"
    end_ms = int(finished_at * 1000) if finished_at else "null"
    return (
        _CHRONOMETER_TEMPLATE
        .replace("__STYLE__", _DIGITAL_STYLE)
        .replace("__START_MS__", str(start_ms))
        .replace("__END_MS__", str(end_ms))
    )


def render_chronometer(
    slot: DeltaGenerator,
    started_at: Optional[float],
    finished_at: Optional[float] = None,
) -> None:
    """Render the chronometer into ``slot`` (an ``st.empty()`` placeholder)."""
    slot.html(
        chronometer_html(started_at, finished_at),
        unsafe_allow_javascript=True,
    )
