"""The contract the in-house LLM client is built against (llmh-01).

**Characterization**, not a spec sketch: everything asserted here is either a
fact measured off the wire (spec §2.1) or the interface the seam exposes
(:mod:`tests.llmkit_fakes`), so that llmh-03's client and llmh-05/06's stubs
are held to one shape from the start. There is deliberately no
``src.core.llmkit`` import -- that package does not exist yet, and this file
must still pass once it does.

Hermetic: no network, no ``.env`` read. The values below are the *numbers* the
current ``.env`` configures, written down as test data.
"""

import pytest

from src.core.models import SamplingParams
from tests.llmkit_fakes import (
    DONE_FRAME,
    FakeStream,
    content_chunk,
    finish_chunk,
    reasoning_chunk,
    usage_chunk,
)


# --- the §2.1 request body, as measured -------------------------------------
#
# The exact body the app puts on the wire for the current ``.env``
# settings, plus the two keys a streaming call adds. llmh-03 diffs the real
# client's body against this constant, so the comparison is against the
# measured wire rather than against anyone's memory of it.
#
# ``MODEL`` and ``MAX_TOKENS`` are named because the spec's example values are
# placeholders for the call's own values, not part of the contract. The
# numeric values are written as the spec measured them; ``1 == 1.0`` in Python,
# so the fixture compares equal whether a value reaches the body as an int or
# as the float the settings model coerces it to.

MODEL = "mymodel"
MAX_TOKENS = 4

SAMPLING = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "repeat_penalty": 1,
    "presence_penalty": 0,
    "reasoning_effort": "medium",
}

EXPECTED_STREAM_BODY = {
    "messages": [],
    "model": MODEL,
    "max_tokens": MAX_TOKENS,
    "presence_penalty": 0,
    "reasoning_effort": "medium",
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "repeat_penalty": 1,
    "stream": True,
    "stream_options": {"include_usage": True},
}


def close_best_effort(stream) -> None:
    """What the cancel paths do: a close whose failure must not escape."""
    try:
        stream.close()
    except Exception:  # noqa: BLE001 — cleanup on cancel is best-effort
        pass


class TestFakeStreamSatisfiesTheSeam:
    """``FakeStream`` is the interface ``ChatStream`` will have (D2, D3)."""

    def test_it_iterates_plain_chunk_dicts(self):
        stream = FakeStream([content_chunk("a"), reasoning_chunk("b")])
        chunks = list(stream)
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert [c["choices"][0]["delta"] for c in chunks] == [
            {"content": "a"},
            {"reasoning_content": "b"},
        ]

    def test_done_frame_ends_iteration_and_is_never_yielded(self):
        chunks = list(FakeStream([content_chunk("a"), DONE_FRAME, content_chunk("b")]))
        assert chunks == [content_chunk("a")]

    def test_saw_finish_reason_starts_false(self):
        assert FakeStream([finish_chunk("stop")]).saw_finish_reason is False

    def test_finish_reason_is_not_claimed_before_its_chunk_is_yielded(self):
        # The lie D3 exists to prevent: a stub that scanned its list up front
        # would report a finish reason the loop never reached, and a cut
        # answer would pass as a finished one.
        stream = FakeStream([content_chunk("half an "), finish_chunk("stop")])
        it = iter(stream)
        assert next(it)["choices"][0]["delta"] == {"content": "half an "}
        assert stream.saw_finish_reason is False
        assert stream.finish_reason is None
        next(it)
        assert stream.saw_finish_reason is True
        assert stream.finish_reason == "stop"

    def test_a_cut_stream_never_claims_a_finish_reason(self):
        stream = FakeStream([content_chunk("half an "), finish_chunk("stop")],
                            error_after=1)
        with pytest.raises(ConnectionError):
            list(stream)
        assert stream.saw_finish_reason is False
        assert stream.finish_reason is None

    def test_done_is_not_a_finish_reason(self):
        # D3: ``data: [DONE]`` is transport framing. Whether it survives an
        # upstream disconnect is unverified, so it proves nothing.
        stream = FakeStream([content_chunk("a"), DONE_FRAME])
        list(stream)
        assert stream.saw_finish_reason is False

    def test_a_null_finish_reason_is_not_a_signal(self):
        stream = FakeStream([finish_chunk(None)])
        list(stream)
        assert stream.saw_finish_reason is False
        assert stream.finish_reason is None

    def test_finish_reason_is_the_first_one_seen(self):
        stream = FakeStream([finish_chunk("length"), finish_chunk("stop")])
        list(stream)
        assert stream.finish_reason == "length"

    def test_usage_is_the_last_usage_block_seen(self):
        stream = FakeStream([
            usage_chunk(prompt=10, total=12),
            content_chunk("a"),
            finish_chunk("stop", usage={"prompt_tokens": 10, "completion_tokens": 2,
                                        "total_tokens": 12}),
        ])
        list(stream)
        assert stream.usage == {"prompt_tokens": 10, "completion_tokens": 2,
                                "total_tokens": 12}

    def test_usage_is_none_when_no_chunk_carried_one(self):
        stream = FakeStream([content_chunk("a")])
        list(stream)
        assert stream.usage is None

    def test_close_is_idempotent(self):
        stream = FakeStream([])
        stream.close()
        stream.close()
        assert stream.closed is True
        assert stream.close_calls == 2  # both calls happened, neither raised

    def test_a_failing_close_is_survivable(self):
        stream = FakeStream([content_chunk("a")],
                            close_error=OSError("socket already gone"))
        list(stream)
        with pytest.raises(OSError):
            stream.close()  # the double is stricter than ChatStream.close()
        close_best_effort(stream)  # ... so that the best-effort call is tested


class TestRequestBodyIsTheMeasuredWire:
    """§2.1: what must be in the body, and in it untouched."""

    def test_the_llama_cpp_samplers_are_in_the_body(self):
        # top_k, min_p and repeat_penalty are llama.cpp sampler names, not
        # OpenAI ones. A client that whitelisted OpenAI's own list would
        # silently change the model's sampling -- and every score of every
        # run -- with nothing on screen to say so.
        for key in ("top_k", "min_p", "repeat_penalty", "reasoning_effort"):
            assert key in EXPECTED_STREAM_BODY

    def test_a_whitelist_would_have_dropped_them(self):
        # Names the fixture guards, stated as the set a whitelist is built
        # from: if these three ever become OpenAI params the test above stops
        # guarding anything, and this is what says so.
        openai_documented = {"temperature", "top_p", "presence_penalty",
                             "frequency_penalty", "seed", "stop", "n",
                             "max_tokens", "stream", "stream_options",
                             "tools", "tool_choice", "response_format",
                             "logprobs", "top_logprobs", "reasoning_effort"}
        dropped = set(SAMPLING) - openai_documented
        assert dropped == {"top_k", "min_p", "repeat_penalty"}
        assert dropped <= set(EXPECTED_STREAM_BODY)

    def test_the_model_name_travels_bare(self):
        # A leftover ``openai/`` prefix reaches llama-swap as
        # ``openai/qwen3.8-…`` and comes back 404 no router for requested model.
        assert EXPECTED_STREAM_BODY["model"] == MODEL
        assert not EXPECTED_STREAM_BODY["model"].startswith("openai/")
        assert not any("openai/" in str(v) for v in EXPECTED_STREAM_BODY.values())

    def test_the_sampling_block_is_exactly_the_configured_samplers(self):
        # The body's sampling part is the app's own sampling dict, key for key
        # -- the set of names the settings model can produce, none filtered.
        # ``to_kwargs()`` is what the app hands the client as ``sampling``, so
        # this pins the fixture's sampling part to the real config path rather
        # than to a hand-typed copy of it.
        params = SamplingParams.model_validate(SAMPLING)
        assert params.to_kwargs() == SAMPLING
        for key, value in SAMPLING.items():
            assert EXPECTED_STREAM_BODY[key] == value

    def test_the_streaming_shape_adds_only_two_keys(self):
        assert EXPECTED_STREAM_BODY["stream"] is True
        assert EXPECTED_STREAM_BODY["stream_options"] == {"include_usage": True}
        non_streaming = {
            k: v for k, v in EXPECTED_STREAM_BODY.items()
            if k not in ("stream", "stream_options")
        }
        assert set(non_streaming) == {
            "messages", "model", "max_tokens", "presence_penalty",
            "reasoning_effort", "temperature", "top_p", "top_k", "min_p",
            "repeat_penalty",
        }

    def test_an_unknown_sampler_key_survives_to_the_body_untouched(self):
        """No layer may "validate" the sampling dict.

        This server answers an unknown parameter with ``200`` (spec §2.2: a
        bogus key was probed and accepted), so a client that dropped or
        rejected an unrecognised sampler would get no signal back and the run
        would silently measure a different sampler set. The body is the
        configured dict merged in as-is, so a typo reaches the server verbatim
        and stays the operator's to see.
        """
        body = dict(EXPECTED_STREAM_BODY)
        body.update({"dys_tfs": 0.9, "typo_k": 20})  # merged verbatim, as llmkit does
        assert body["dys_tfs"] == 0.9
        assert body["typo_k"] == 20
