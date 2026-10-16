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
  border: 3px solid currentColor;
  background-color: transparent;
}
</style>
"""


def render_stream_box_style() -> None:
    """Give the response-stream container a transparent fill and a 3px border that follows the active theme's text color."""
    st.markdown(_STREAM_BOX_CSS, unsafe_allow_html=True)

_LAYOUT_CSS = """
<style>
div[data-testid="stHeadingWithActionElements"] {
  text-align: center;
}
[data-testid="stMain"] h3 {
  text-align: center;
}
.model-name {
  text-align: center;
  font-size: 0.75rem;
  color: var(--secondaryTextColor);
}
[data-testid="stMain"] {
  overflow-y: hidden;
}
</style>
"""


def render_layout_style() -> None:
    """Center the brand heading and model name; hide the main column scrollbar."""
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
  var base = __BASE_MS__;
  var live = __LIVE_MS__;
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
    el.textContent = format(base + (Date.now() - live));
  }
  if (live === null) {
    el.textContent = format(base);
    return;
  }
  var timer = setInterval(tick, 250);
  tick();
})();
</script>
"""


def chronometer_html(
    elapsed_seconds: float,
    live_since: Optional[float] = None,
) -> str:
    """Return the chronometer markup for a cumulative answer time.

    ``elapsed_seconds`` is the sum of the per-question response times already
    completed. When ``live_since`` (epoch seconds) is given, the clock keeps
    ticking client-side from it — so it runs while the LLM streams an answer —
    otherwise the display is frozen at ``elapsed_seconds``.
    """
    base_ms = int(elapsed_seconds * 1000)
    live_ms = int(live_since * 1000) if live_since else "null"
    return (
        _CHRONOMETER_TEMPLATE
        .replace("__STYLE__", _DIGITAL_STYLE)
        .replace("__BASE_MS__", str(base_ms))
        .replace("__LIVE_MS__", str(live_ms))
    )


def render_token_rate(slot: DeltaGenerator, tokens_per_second: float) -> None:
    """Show the overall tokens-per-second rate below the chronometer."""
    slot.caption(f":material/speed: overall token/s {tokens_per_second:.1f}")


_CHECKING_TEMPLATE = """
<div style="__STYLE__">Checking&hellip;</div>
"""


def render_checking(slot: DeltaGenerator) -> None:
    """Show a static ``Checking...`` placeholder while the verifier runs.

    Replaces the chronometer so verification time is visibly excluded from
    the elapsed answer counter.
    """
    slot.html(
        _CHECKING_TEMPLATE.replace("__STYLE__", _DIGITAL_STYLE),
        unsafe_allow_javascript=False,
    )


def render_chronometer(
    slot: DeltaGenerator,
    elapsed_seconds: float,
    live_since: Optional[float] = None,
) -> None:
    """Render the chronometer into ``slot`` (an ``st.empty()`` placeholder)."""
    slot.html(
        chronometer_html(elapsed_seconds, live_since),
        unsafe_allow_javascript=True,
    )
