"""Shared double and chunk builders for the in-house LLM client (llmh-01).

Nothing here imports ``src.core.llmkit`` -- that package is built by the later
tasks of this lot. What is pinned is the seam all of them are written against:
one streamed completion as an **iterable of plain chunk dicts** (D2) carrying
its own end state (``saw_finish_reason``, ``finish_reason``, ``usage``) plus a
``close()`` -- the shape ``llmkit.client.ChatStream`` will have, and the shape
the stubs in ``tests/test_llm.py`` / ``tests/test_chat.py`` are re-pointed to,
so the double and the real client are held to one interface from the start.

Two rules keep the double worth trusting:

* the end state is computed **while iterating**, never scanned up front -- a
  stream nobody drained must not claim a finish reason;
* ``[DONE]`` is transport framing and never counts as one (D3).

Hermetic by construction: the frames are handed to the constructor, so no test
using this module reaches a network or reads ``.env``.
"""

from typing import Any, Dict, Iterator, List, Optional, Sequence

# The wire's end-of-stream frame. ``llmkit.constants.DONE_SENTINEL`` (llmh-02)
# is the production twin of this value; it is spelled out here because this
# file must not depend on a package that does not exist yet.
DONE_FRAME = "[DONE]"


# --- chunk builders ----------------------------------------------------------
#
# The frames an OpenAI-compatible server puts on the wire, as the dicts the
# client yields. Names match the helpers they replace in ``tests/test_llm.py``
# and ``tests/test_chat.py``, so re-pointing those stubs is textual.


def content_chunk(text: str) -> Dict[str, Any]:
    """One answer delta."""
    return {"choices": [{"delta": {"content": text}}]}


def reasoning_chunk(text: str) -> Dict[str, Any]:
    """One thinking delta (llama.cpp's ``delta.reasoning_content``)."""
    return {"choices": [{"delta": {"reasoning_content": text}}]}


def tool_call_chunk(
    index: int = 0,
    call_id: Optional[str] = None,
    name: Optional[str] = None,
    arguments: Optional[str] = None,
) -> Dict[str, Any]:
    """One tool-call delta, as streamed in fragments across chunks."""
    call: Dict[str, Any] = {"index": index}
    if call_id:
        call["id"] = call_id
    function: Dict[str, Any] = {}
    if name:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    if function:
        call["function"] = function
    return {"choices": [{"delta": {"tool_calls": [call]}}]}


def usage_chunk(
    prompt: Optional[int] = None,
    completion: Optional[int] = None,
    total: Optional[int] = None,
) -> Dict[str, Any]:
    """A usage-only frame: no choices, only the token block."""
    usage: Dict[str, Any] = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if total is not None:
        usage["total_tokens"] = total
    return {"choices": [], "usage": usage}


def finish_chunk(
    reason: Optional[str] = "stop",
    content: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The frame that carries the finish reason (and, with ``include_usage``,
    usually the usage block too). ``reason=None`` is a frame that names no
    reason -- which is what a cut stream ends on."""
    choice: Dict[str, Any] = {
        "delta": {} if content is None else {"content": content},
        "finish_reason": reason,
    }
    chunk: Dict[str, Any] = {"choices": [choice]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


class FakeStream:
    """Stand-in for one streamed completion (``ChatStream``).

    ``chunks`` is the list of frames the server sent; a :data:`DONE_FRAME`
    entry ends iteration without being yielded, exactly as the SSE layer will.
    ``error_after`` cuts the stream after that many delivered chunks -- the
    upstream disconnect every truncation check exists for -- and ``error``
    replaces the exception it raises.

    ``close_error`` makes ``close()`` fail. That is deliberately *stricter*
    than ``ChatStream.close()``, which never raises: the cancel paths wrap the
    close in a best-effort try/except, and only a close that actually fails
    exercises that.
    """

    def __init__(
        self,
        chunks: Sequence[Any],
        close_error: Optional[BaseException] = None,
        error_after: Optional[int] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        self._chunks: List[Any] = list(chunks)
        self._close_error = close_error
        self._error_after = error_after
        self._error = error or ConnectionError("the server cut the stream")
        self._closed = False
        self._finish_reason: Optional[str] = None
        self._saw_finish_reason = False
        self._usage: Optional[Dict[str, Any]] = None
        self.close_calls = 0

    # --- end state: only what the delivered chunks actually said ------------

    @property
    def saw_finish_reason(self) -> bool:
        return self._saw_finish_reason

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        return self._usage

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        delivered = 0
        for chunk in self._chunks:
            if self._error_after is not None and delivered >= self._error_after:
                raise self._error
            if chunk == DONE_FRAME:
                return
            # Recorded as the chunk is handed over -- not before (the loop
            # never saw it) and not after (a generator resumes only on the
            # next pull, so a consumer that stops on the finish chunk would
            # leave the flag unset).
            self._record(chunk)
            yield chunk
            delivered += 1

    def _record(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        for choice in chunk.get("choices") or []:
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            if reason and self._finish_reason is None:
                self._finish_reason = reason
                self._saw_finish_reason = True
        usage = chunk.get("usage")
        if usage:
            self._usage = usage

    def close(self) -> None:
        self.close_calls += 1   # every call is recorded: close() is a no-op twice over
        self._closed = True
        if self._close_error is not None:
            raise self._close_error
