"""Server-Sent-Events lines -> chat-completion chunk dicts.

The mechanism half of the package; the numbers it reads live in
:mod:`llmkit.constants`. It never sees a socket: the caller hands it an
iterable of *decoded* lines -- what a streaming response's line iterator
hands over -- and gets back chunk dicts. No FastAPI, no event bus, no HTTP
client of any kind, and it records nothing: whether the stream carried a
finish reason is the stream object's fact to keep, not this parser's (D3).

The rules are the ones the live server exercises (spec §2.2): the space after
``data:`` is optional, ``: keep-alive`` is a comment rather than an event,
blank lines are frame separators, and ``data: [DONE]`` ends the stream.
"""

import json
from typing import Any, Dict, Iterable, Iterator, Optional

from . import constants


class SSEFrameError(ValueError):
    """A ``data:`` frame whose JSON payload does not parse.

    Named and raised rather than skipped on purpose (D3): a half-read frame
    means the connection died mid-write, and the lines that follow it describe
    nothing. Skipping it would hide a cut stream behind a frame that never
    carried a finish reason -- the exact lie the truncation check exists to
    catch. Both consuming loops surface ``str(exc)`` in their error line, so
    the message names what broke.
    """


def _data_payload(raw: str) -> Optional[str]:
    """The ``data:`` payload of one frame, or None when it carries no data.

    A blank line, a comment line (``: keep-alive`` and anything else prefixed
    with ``:``) and any other SSE field (``event:``, ``id:``) carry no payload.
    The space after the colon is optional, so the payload is stripped.
    """
    line = raw.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith(constants.DATA_PREFIX):
        return None
    return line[len(constants.DATA_PREFIX):].strip()


def parse_frame(raw: str) -> Optional[Dict[str, Any]]:
    """One SSE frame -> a chunk dict.

    ``None`` for everything that is not an event carrying a JSON object: a
    blank line, a comment line (``: keep-alive`` and anything else prefixed
    with ``:``), a field line that is not ``data:`` (``event:``, ``id:``), the
    ``data: [DONE]`` sentinel, or a ``data:`` payload that is not a JSON object.

    Raises :class:`SSEFrameError` when a ``data:`` payload looks like a chunk
    but does not parse.
    """
    payload = _data_payload(raw)
    if not payload or payload == constants.DONE_SENTINEL:
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SSEFrameError(
            f"the model endpoint sent a malformed streaming frame: {exc} "
            f"(frame: {payload[:80]!r})"
        ) from exc

    if not isinstance(parsed, dict):
        # A JSON scalar or array in a data frame is not a chat-completion
        # chunk: it carries nothing a caller could use, and it is not a
        # half-read frame either.
        return None
    return parsed


def iter_chunks(lines: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Yield chunk dicts until ``[DONE]``, ignoring everything else.

    The sentinel ends iteration and is not yielded, and nothing is recorded for
    it: it is transport framing, so it neither becomes a chunk nor a finish
    reason (D3) -- recording is the stream object's job. A frame whose JSON
    does not parse raises through here rather than being skipped.
    """
    for raw in lines:
        if _data_payload(raw) == constants.DONE_SENTINEL:
            return
        chunk = parse_frame(raw)
        if chunk is not None:
            yield chunk

