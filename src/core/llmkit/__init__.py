"""The in-house LLM client: the one place in the app that speaks HTTP to a
model endpoint.

Core-only transport: no FastAPI, no event bus, no ``.env`` read — a caller
passes ``model`` / ``server_url`` / ``api_key`` / ``sampling`` in, exactly as
it passed them to the streamed call this package replaces. Layout mirrors
``core/mcphub`` and ``core/lsphub``:

* :mod:`llmkit.constants` -- the tuning numbers (timeouts, wire sentinels)
* :mod:`llmkit.sse` -- SSE lines to parsed chat-completion chunk dicts
* :mod:`llmkit.client` -- the two call shapes, the error type, the URL rules
"""

from .client import (
    ChatStream,
    LLMRequestError,
    chat_completion,
    chat_completions_url,
    models_url,
    stream_completion,
)

__all__ = [
    "ChatStream",
    "LLMRequestError",
    "chat_completion",
    "chat_completions_url",
    "models_url",
    "stream_completion",
]
