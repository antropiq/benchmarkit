"""Static configuration of the in-house LLM client: timeouts and wire constants.

The tuning numbers live here as *module attributes* (same idea as
``mcphub.constants`` / ``lsphub.constants``): the code that reads them resolves
them through this module at call time, so a test that needs another timeout or
another body limit patches one attribute here instead of a literal in the module
under test.
"""

from typing import Optional

# The frame an OpenAI-compatible server sends last. It is transport framing,
# so it ends the read loop and nothing else: it never counts as evidence that
# the answer finished (D3 -- that is a chunk's ``finish_reason`` alone).
DONE_SENTINEL = "[DONE]"

# Every SSE event line of a chat-completion stream starts with this.
DATA_PREFIX = "data:"

# Sent as the bearer token when no key is configured. The header itself is
# still sent: that is what the endpoint sees today and it accepts it.
API_KEY_PLACEHOLDER = "none"

# A dead endpoint fails fast; a live stream is never given a read deadline.
# A stalled prefill or a model load (warm-up exists precisely because that wait
# is long) must stay what it is -- a slow first token -- and must not surface as
# a *truncated answer*, which is the confusion this client exists to remove
# (D10). ``None`` means "no read timeout" to httpx.
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT: Optional[float] = None
WRITE_TIMEOUT = 30.0
POOL_TIMEOUT = 10.0

# How much of an unparseable error body reaches the UI. The status and the
# upstream ``error.message`` are what an operator acts on; the raw body is the
# fallback, and a full HTML error page is not a diagnosis.
ERROR_BODY_LIMIT = 500

# The logger the debug switch (``LLM_DEBUG``) turns on: this package's request
# payloads and stream outcomes go under this one name.
DEBUG_LOGGER_NAME = "benchmarkit.llm"
