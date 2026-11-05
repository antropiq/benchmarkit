"""Neutralising delimiter-wrapped control tokens in outgoing message content.

Every benchmark answer, tool result and user message rides back into the
next prompt as message content, and a chat-template control token is plain
text at the HTTP surface: whoever gets one into a message can close the
current turn early or forge a block the server's template will honour.
This module is the single place that says such a run never leaves the app.
Both call shapes in :mod:`llmkit.client` pass their ``messages`` through
:func:`sanitize_messages` before the request body is built, so benchmark,
verifier, warm-up, chat and batch are all covered at the one HTTP seam --
no caller can forget it, because no caller can skip it.

The neutralisation is removal of every token-shaped run -- a name wrapped
in the two-pipe delimiters the major chat templates (ChatML and friends)
use for their control tokens -- repeated to a fixed point: deleting an
inner run can splice the surrounding text into a complete outer run, so a
single pass is not enough. What survives is text no server template can
re-form into a control token, and nothing else is touched: benign content,
down to a lonely unterminated delimiter, passes through byte for byte.

Every string a server template renders out of a message is walked, not
just ``content``: the assistant turns a tool loop feeds back carry a
``tool_calls`` function ``name`` and ``arguments`` that the template
interpolates raw, ``reasoning_content`` rides into the prompt wherever a
thinking continuation sends it, and ``name`` is interpolated by the
templates that address functions through it. A run smuggled through any
of those closes the turn or forges a block exactly as one in content
does; guarding content alone guards only the obvious door.

Removal, not escaping: an escaped run is still a shape a template's own
re-sanitiser might undo, and the app has no use for token-shaped text in
content -- a benchmark question or a verifier payload carrying one is an
attack, not an accident. The delimiters here are assembled from single
characters through ``re.escape``, so this source file never carries a
complete delimiter pair: the run this module exists to remove must not
appear in the code that removes it.
"""

import re
from typing import Any, Dict, List

_PIPE = "|"

_OPEN_RUN = re.escape(f"<{_PIPE}")
_CLOSE_RUN = re.escape(f"{_PIPE}>")

# A control token as a server's template spells it: a name between the two
# pipe delimiters, the name itself holding no delimiter character. A run
# with a delimiter inside its name is not the shape any template tokenises
# as one control token, so it is not what has to go.
TOKEN_RUN = re.compile(f"{_OPEN_RUN}[^<>{_PIPE}]*{_CLOSE_RUN}")


def neutralize(text: str) -> str:
    """Remove every token-shaped run from ``text`` until none is left.

    Removal can splice: deleting an inner run joins the text around it
    into a complete outer run, so the pass repeats to a fixed point. Each
    pass only ever deletes characters, so the loop terminates.
    """
    while True:
        stripped = TOKEN_RUN.sub("", text)
        if stripped == text:
            return text
        text = stripped


def sanitize_content(content: Any) -> Any:
    """Sanitize one message's content, leaving the caller's value untouched.

    A string is sanitized outright; a list of content parts has the
    ``text`` of each part sanitized while other parts and keys pass
    through; anything else -- ``None`` for a tool-call-only assistant
    turn, or a shape this layer does not interpret -- passes through for
    the server to judge, exactly as the client does with the rest of the
    body. The caller's object is returned unchanged when nothing
    matched, so the common path copies nothing.
    """
    if isinstance(content, str):
        return neutralize(content)
    if isinstance(content, list):
        parts: List[Any] = []
        changed = False
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text = neutralize(part["text"])
                if text != part["text"]:
                    part = {**part, "text": text}
                    changed = True
            parts.append(part)
        return parts if changed else content
    return content


# The plain-string fields a template interpolates besides ``content``.
# Missing keys and non-string values are left alone -- this layer removes
# runs, it does not validate message shapes.
_INTERPOLATED_FIELDS = ("name", "reasoning_content")


def _sanitize_tool_call(call: Any) -> Any:
    """Sanitize a tool-call's function name and argument text.

    The arguments are the text the server feeds back through its template
    on the next round, so a run in them is a run in the prompt. Removal
    can make an argument blob that was malformed around the run no longer
    parse as JSON -- the correct trade: a tool call built around a control
    token is corrupt either way, and smuggling the token is the worse
    outcome. Unchanged calls keep their identity.
    """
    if not isinstance(call, dict):
        return call
    function = call.get("function")
    if not isinstance(function, dict):
        return call
    name = function.get("name")
    arguments = function.get("arguments")
    clean_name = neutralize(name) if isinstance(name, str) else name
    clean_arguments = (
        neutralize(arguments) if isinstance(arguments, str) else arguments
    )
    if clean_name is name and clean_arguments is arguments:
        return call
    new_function = dict(function)
    new_function["name"] = clean_name
    new_function["arguments"] = clean_arguments
    new_call = dict(call)
    new_call["function"] = new_function
    return new_call


def _sanitize_message(message: Any) -> Any:
    """Sanitize every template-rendered string of one message.

    Returns the original object when nothing matched, so a caller that
    compares identity can tell an untouched message from a copied one.
    """
    if not isinstance(message, dict):
        return message
    cleaned = dict(message)
    if "content" in message:
        cleaned["content"] = sanitize_content(message["content"])
    for field in _INTERPOLATED_FIELDS:
        value = message.get(field)
        if isinstance(value, str):
            cleaned[field] = neutralize(value)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        cleaned["tool_calls"] = [_sanitize_tool_call(call) for call in calls]
    if cleaned == message:
        return message
    return cleaned


def sanitize_messages(messages: List[Dict]) -> List[Dict]:
    """A sanitized copy of ``messages``, ready for the request body.

    Every string a server template renders is walked -- ``content``, the
    ``tool_calls`` function fields, ``reasoning_content``, ``name``. Only
    a message whose text actually changed is copied; every other entry
    keeps its identity, and no input dict or list is ever mutated: the
    caller's conversation history is also what the UI renders, and the UI
    shows what the user actually typed. The outer list is always fresh,
    so the body can never alias the caller's list.
    """
    return [_sanitize_message(message) for message in messages]


# The longest candidate run that could still be a control token: the names
# a server template tokenizes are short, so a candidate whose span has grown
# past this bound is declared dead and its text released. Without it, one
# unterminated run opening in ordinary prose holds back every character
# after it until the stream ends -- prevention the user experiences as a
# stalled answer. What it costs is exact: only a run longer than the bound
# could cross into the browser, and the one-way :func:`neutralize` that
# guards outgoing prompts still removes a run of any length.
_MAX_TOKEN_LEN = 64


def _open_live(text: str, p: int) -> bool:
    """Whether a token-shaped run could still open at ``text[p]``.

    ``text`` is assumed run-free (a fixed-point :func:`neutralize` has
    run over it); the question is about any completion of it by later
    text. A name character of ``>``, or a pipe that cannot close, is dead
    for good -- no future text and no removal can make a run open there.
    A nested open inside the name keeps the outer alive: if the nested
    run later completes, removal erases it and the outer name continues
    through the gap -- that is precisely the splice this module exists
    for. Treating a undecided end as live is the safe direction: a false
    live only holds text back, a false dead would emit half a run.
    """
    if text[p + 1:p + 2] == "":
        return True  # a lone opening character: the pipe has not arrived
    if text[p + 1:p + 2] != _PIPE:
        return False
    q = p + 2
    while q < len(text):
        char = text[q]
        if char == ">":
            return False
        if char == _PIPE:
            following = text[q + 1:q + 2]
            if following == "" or following == ">":
                return True
            return False
        if char == "<":
            if q + 1 >= len(text):
                return True  # nested undecided
            return _open_live(text, q)
        q += 1
    return True


class StreamNeutralizer:
    """The streaming twin of :func:`neutralize`, for a chunk at a time.

    A run straddles chunk borders routinely -- a server writing the
    literal characters hands them over in several pieces -- so a
    per-chunk ``neutralize`` pass would happily emit the first half of
    one. The rule: text may be handed over only once no run can ever
    complete through it. Each ``feed`` re-runs removal over the held
    tail plus the new text, then releases up to the earliest *live* open
    (see :func:`_open_live`) and holds the rest. Benign text streams
    through with zero hold-back -- a ``<`` that cannot open a run is
    inert the moment its neighbour says so -- and the held remainder is
    provably run-free: it opens with a run opening whose closer never
    arrived, and a fixed-point ``neutralize`` guarantees no complete run
    is anywhere in it. The hold is bounded by ``_MAX_TOKEN_LEN`` -- the
    newest text a token could still complete through is kept, and anything
    older is released -- so a stray unterminated opening delays at most
    that many characters instead of freezing the rest of the reply.

    Order is never disturbed: only a prefix older than every live open
    is ever released.
    """

    def __init__(self) -> None:
        self._held = ""

    def feed(self, text: str) -> str:
        """The part of the stream that is now provably run-free."""
        self._held += text
        cleaned = neutralize(self._held)
        scan = 0
        boundary = len(cleaned)
        while scan < len(cleaned):
            start = cleaned.find("<", scan)
            if start < 0:
                break
            if _open_live(cleaned, start):
                boundary = start
                break
            scan = start + 1
        if len(cleaned) - boundary > _MAX_TOKEN_LEN:
            # The live candidate has outgrown any real control token, so it
            # was not one: release all but the newest window a run could
            # still complete through. Order is untouched -- this only ever
            # moves the boundary forward.
            boundary = len(cleaned) - _MAX_TOKEN_LEN
        self._held = cleaned[boundary:]
        return cleaned[:boundary]

    def flush(self) -> str:
        """The held remainder, when the stream has ended."""
        held, self._held = self._held, ""
        return held
