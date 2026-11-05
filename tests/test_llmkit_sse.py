"""The SSE layer of the in-house LLM client (llmh-02, spec §4.1/§4.2).

Every rule asserted here is one the live server exercises (spec §2.2): the
optional space after ``data:``, ``: keep-alive`` comment lines between frames,
``data: [DONE]`` last, a final frame carrying ``finish_reason`` **and** the
``usage`` block plus a llama.cpp ``timings`` object. Unparseable JSON raising
instead of being skipped is D3: a half-read frame means the connection died,
and skipping it would hide a cut stream behind a frame that never carried a
finish reason.

Hermetic by construction: the parser never sees a socket, so every case here
is a plain iterable of decoded lines -- no network, no ``.env``.
"""

import json

import pytest

from src.core.llmkit import constants
from src.core.llmkit.sse import SSEFrameError, iter_chunks, parse_frame
from tests.llmkit_fakes import (
    DONE_FRAME,
    content_chunk,
    finish_chunk,
    reasoning_chunk,
    usage_chunk,
)


def frame(payload) -> str:
    """One ``data:`` line, exactly as the server writes it (space included)."""
    return "data: " + json.dumps(payload)


def done_frame() -> str:
    return "data: " + DONE_FRAME


class TestParseFrameDataPrefix:
    def test_data_prefix_with_a_space_parses_into_a_chunk(self):
        assert parse_frame(frame(content_chunk("hel"))) == content_chunk("hel")

    def test_data_prefix_without_a_space_parses_into_a_chunk(self):
        # The space after the colon is optional, so both spellings must reach
        # the same dict -- a proxy that trims it must not break the run.
        raw = "data:" + json.dumps(content_chunk("hel"))
        assert parse_frame(raw) == content_chunk("hel")

    def test_a_frame_terminated_by_a_newline_parses(self):
        # httpx's line iterator may hand the line over with its terminator.
        assert parse_frame(frame(content_chunk("o")) + "\n") == content_chunk("o")

    def test_an_empty_data_payload_carries_no_chunk(self):
        assert parse_frame("data:") is None
        assert parse_frame("data: ") is None


class TestParseFrameSilence:
    """Everything that is not an event carrying a JSON object."""

    def test_a_keep_alive_comment_is_not_an_event(self):
        # The live server pushes these between frames while it works.
        assert parse_frame(": keep-alive") is None

    def test_any_colon_prefixed_line_is_a_comment(self):
        for raw in (": ping", ":", ":no-spaces-here"):
            assert parse_frame(raw) is None

    def test_a_blank_line_is_a_frame_separator_not_a_chunk(self):
        assert parse_frame("") is None
        assert parse_frame("\n") is None
        assert parse_frame("   ") is None

    def test_other_sse_fields_carry_no_chunk(self):
        for raw in ("event: message", "id: 42", "retry: 1000"):
            assert parse_frame(raw) is None


class TestParseFrameDone:
    """D3: ``[DONE]`` is transport framing -- never a chunk, never a signal."""

    def test_done_is_not_a_chunk(self):
        assert parse_frame(done_frame()) is None

    def test_done_without_the_space_is_not_a_chunk(self):
        assert parse_frame("data:" + DONE_FRAME) is None


class TestParseFrameChunks:
    def test_a_usage_only_frame_with_no_choices_parses(self):
        # ``stream_options: {"include_usage": true}`` earns exactly this frame
        # shape: no choices at all, only the token block.
        chunk = usage_chunk(prompt=11, completion=3, total=14)
        assert parse_frame(frame(chunk)) == chunk

    def test_the_real_final_frame_passes_through_untouched(self):
        # Measured shape: finish_reason *and* usage in one frame, plus the
        # llama.cpp ``timings`` object the app ignores. Unknown keys must
        # survive the parse -- the parser's job is to read framing, not to
        # vet the body.
        final = finish_chunk(
            "stop",
            content="",
            usage={
                "prompt_tokens": 42,
                "completion_tokens": 7,
                "total_tokens": 49,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        )
        final["timings"] = {"predicted_ms": 233.1, "predicted_per_second": 30.05}
        parsed = parse_frame(frame(final))
        assert parsed is not None  # a real final frame is a chunk, not framing
        assert parsed == final
        assert parsed["timings"]["predicted_per_second"] == 30.05
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_a_json_payload_that_is_not_an_object_carries_no_chunk(self):
        # Not a half-read frame (it parsed cleanly) and not a chunk either.
        assert parse_frame('data: [1, 2, 3]') is None
        assert parse_frame('data: "hello"') is None
        assert parse_frame("data: 42") is None


class TestParseFrameRejectsGarbage:
    """A ``data:`` payload that should be a chunk but does not parse raises."""

    def test_unparseable_json_raises(self):
        with pytest.raises(SSEFrameError):
            parse_frame("data: {not json}")

    def test_a_half_read_frame_raises(self):
        # The case D3 is about: the connection died mid-write, so the frame
        # stops in the middle. Skipping it would end the run on a fragment
        # that looks finished at the chunk level.
        with pytest.raises(SSEFrameError):
            parse_frame('data: {"choices": [{"delta": {"content": "half an')

    def test_the_message_says_what_broke(self):
        # Both consuming loops put ``str(exc)`` straight into the UI's error
        # line (D8), so the sentence has to name the transport fault.
        with pytest.raises(SSEFrameError) as excinfo:
            parse_frame('data: {"choices": [oops]}')
        assert "malformed" in str(excinfo.value)
        assert "oops" in str(excinfo.value)

    def test_it_is_a_value_error(self):
        # Callers catch ``Exception`` either way; the base keeps the raise
        # honest for anyone catching a malformed-value family.
        assert issubclass(SSEFrameError, ValueError)


class TestIterChunks:
    def test_a_whole_stream_parses_in_order(self):
        lines = [
            frame(content_chunk("Hello")),
            "",
            frame(content_chunk(" there")),
            done_frame(),
        ]
        assert [c["choices"][0]["delta"]["content"] for c in iter_chunks(lines)] == [
            "Hello",
            " there",
        ]

    def test_done_ends_iteration_and_yields_no_chunk(self):
        lines = [frame(content_chunk("a")), done_frame(), frame(content_chunk("b"))]
        assert list(iter_chunks(lines)) == [content_chunk("a")]

    def test_comments_and_blank_lines_are_skipped_between_frames(self):
        lines = [
            ": keep-alive",
            "",
            frame(reasoning_chunk("think")),
            ": keep-alive",
            frame(usage_chunk(prompt=5, completion=1, total=6)),
            done_frame(),
            ": keep-alive",
        ]
        chunks = list(iter_chunks(lines))
        assert chunks == [
            reasoning_chunk("think"),
            usage_chunk(prompt=5, completion=1, total=6),
        ]

    def test_a_stream_cut_before_done_still_yields_every_frame(self):
        # No sentinel and no exception: the iterable simply runs out. Whether
        # the answer finished is the stream object's record of finish reasons
        # (D3), decided from the chunks -- never from the missing frame.
        lines = [frame(content_chunk("half an "))]
        assert list(iter_chunks(lines)) == [content_chunk("half an ")]

    def test_an_empty_iterable_yields_nothing(self):
        assert list(iter_chunks([])) == []

    def test_a_malformed_frame_raises_through_iteration(self):
        lines = [frame(content_chunk("a")), "data: {oops", done_frame()]
        with pytest.raises(SSEFrameError):
            list(iter_chunks(lines))

    def test_iteration_is_lazy(self):
        # A socket hands lines over as they arrive, so a parser that drained
        # the iterable first would block the loop until the stream ended.
        pulled = []

        def lines():
            for raw in [frame(content_chunk("a")), "data: {oops", frame(content_chunk("b"))]:
                pulled.append(raw)
                yield raw

        chunks = iter_chunks(lines())
        assert next(chunks) == content_chunk("a")
        assert len(pulled) == 1
        with pytest.raises(SSEFrameError):
            next(chunks)
        assert len(pulled) == 2


class TestConstants:
    """§4.1: the tuning numbers, pinned where they are read from."""

    def test_the_wire_constants(self):
        assert constants.DONE_SENTINEL == "[DONE]"
        assert constants.DATA_PREFIX == "data:"
        assert constants.API_KEY_PLACEHOLDER == "none"
        assert constants.DEBUG_LOGGER_NAME == "benchmarkit.llm"

    def test_there_is_no_read_deadline_on_a_live_stream(self):
        # D10: a stalled prefill or a model load must stay a slow first token,
        # never a *truncated answer* -- a transport setting masquerading as a
        # model failure is the confusion this client exists to remove.
        assert constants.READ_TIMEOUT is None
        assert constants.CONNECT_TIMEOUT == 10.0
        assert constants.WRITE_TIMEOUT == 30.0
        assert constants.POOL_TIMEOUT == 10.0

    def test_the_error_body_limit_is_a_cap_not_a_license_to_dump(self):
        assert constants.ERROR_BODY_LIMIT == 500

    def test_constants_are_read_through_the_module_at_call_time(self, monkeypatch):
        # They live in ``constants`` as module attributes precisely so they can
        # be redirected (the tests do exactly this). A copy bound into ``sse``
        # would go stale under such a patch.
        monkeypatch.setattr(constants, "DONE_SENTINEL", "@END@")
        assert parse_frame("data: @END@") is None
        # The value it no longer recognises is now just a payload that does not
        # parse: proof the check consulted the module at call time rather than
        # a name bound at import.
        with pytest.raises(SSEFrameError):
            parse_frame(done_frame())
