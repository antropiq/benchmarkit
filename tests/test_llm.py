import threading
from types import SimpleNamespace
from typing import Optional

import pytest

from src.core.llm import (
    RunCancelled,
    TRUNCATION_MESSAGE,
    _extract_reasoning,
    _extract_total_tokens,
    _extract_usage,
    call_llm_and_stream,
    close_llm_stream,
    safe_get_choice_content,
    saw_finish_reason,
    warmup_model,
)
# The chunk builders come from the shared double (``tests/llmkit_fakes``) so the
# stubs and ``llmkit.client.ChatStream`` are held to one interface.
from tests.llmkit_fakes import (
    FakeStream,
    content_chunk,
    finish_chunk,
    reasoning_chunk,
    usage_chunk,
)


class TestCallLlmAndStreamTiming:
    def _run(self, monkeypatch, chunks, clock_values):
        clock = list(clock_values)

        def fake_monotonic():
            return clock.pop(0)

        monkeypatch.setattr("src.core.llm.time.monotonic", fake_monotonic)
        monkeypatch.setattr(
            "src.core.llm.stream_completion", lambda **kwargs: iter(chunks)
        )
        events = []
        result = call_llm_and_stream(
            "m", "http://server:8080/v1",
            {"uuid": "u", "question": "q", "expected": "e"}, events.append,
        )
        assert clock == []  # every monotonic() call consumed a tick
        return result, events

    def test_usage_chunk_arrival_does_not_extend_timing(self, monkeypatch):
        # t0, first content, last content, stream-end (never used as took).
        chunks = [content_chunk("a"), content_chunk("bc"),
                  usage_chunk(prompt=20, completion=5, total=25)]
        result, _ = self._run(monkeypatch, chunks, [0.0, 1.0, 3.0, 100.0])

        assert result.text == "abc"
        assert result.took == 3.0   # t_last - t0, ignoring the 100.0 drain
        assert result.ttft == 1.0
        assert result.gen_duration == 2.0
        assert result.completion_tokens == 5
        assert result.prompt_tokens == 20
        assert result.tokens_estimated is False

    def test_missing_usage_estimates_tokens(self, monkeypatch):
        chunks = [content_chunk("hello world!!!")]  # 14 chars -> 4 tokens
        result, _ = self._run(monkeypatch, chunks, [0.0, 0.5, 2.0])

        assert result.completion_tokens == 4
        assert result.tokens_estimated is True
        assert result.prompt_tokens == 0

    def test_completion_derived_from_total_minus_prompt(self, monkeypatch):
        chunks = [content_chunk("x"),
                  usage_chunk(prompt=10, total=18)]
        result, _ = self._run(monkeypatch, chunks, [0.0, 0.2, 1.0])

        assert result.completion_tokens == 8
        assert result.tokens_estimated is False

    def test_empty_stream_has_zero_generation_window(self, monkeypatch):
        result, _ = self._run(monkeypatch, [], [0.0, 5.0])

        assert result.text == ""
        assert result.took == 5.0
        assert result.ttft == 0.0
        assert result.gen_duration == 0.0
        assert result.completion_tokens == 0
        assert result.tokens_estimated is False

    def test_call_failure_returns_zeroed_result(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("connect failed")

        clock = [0.0, 0.25]
        monkeypatch.setattr("src.core.llm.time.monotonic", lambda: clock.pop(0))
        monkeypatch.setattr("src.core.llm.stream_completion", boom)
        events = []
        result = call_llm_and_stream(
            "m", "http://server:8080/v1",
            {"uuid": "u", "question": "q", "expected": "e"}, events.append,
        )

        assert result.text == ""
        assert result.took == 0.25
        assert result.gen_duration == 0.0
        assert events[0]["event"] == "run_error"

    def test_emits_token_events_in_order(self, monkeypatch):
        chunks = [content_chunk("he"), content_chunk("llo")]
        result, events = self._run(monkeypatch, chunks, [0.0, 0.1, 0.2, 0.3])

        assert [e["text"] for e in events if e["event"] == "token"] == ["he", "llo"]

    def test_reasoning_deltas_anchor_the_generation_window(self, monkeypatch):
        # Prefill ends, reasoning streams first, then the answer: the decode
        # window must start at the first reasoning token, not the first
        # content token, or the rate is inflated by the reasoning wait.
        chunks = [
            reasoning_chunk("thinking…"),
            reasoning_chunk("more…"),
            content_chunk("4"),
            content_chunk("2"),
            usage_chunk(prompt=20, completion=40, total=60),
        ]
        result, events = self._run(
            monkeypatch, chunks, [0.0, 9.9, 11.0, 12.04, 16.5, 100.0]
        )

        assert result.text == "42"
        assert result.ttft == 9.9
        assert result.took == 16.5
        assert result.gen_duration == 16.5 - 9.9
        assert result.completion_tokens == 40
        assert [e["text"] for e in events if e["event"] == "token"] == ["4", "2"]

    def test_reasoning_only_stream_has_zero_answer_text(self, monkeypatch):
        chunks = [reasoning_chunk("only "), reasoning_chunk("thinking")]
        result, events = self._run(monkeypatch, chunks, [0.0, 1.0, 2.0, 3.0])

        assert result.text == ""
        assert result.gen_duration == 1.0
        assert [e["event"] for e in events if e["event"] == "token"] == []


class TestStreamTruncationDetection:
    """A stream the server cut short must not look like a finished answer.

    The chunks of a cut answer and of a finished one are indistinguishable, so
    the only honest signal is the stream's own ``saw_finish_reason``, which the
    client sets while iterating (``llmkit.client.ChatStream``).
    """

    def _run(self, monkeypatch, generator):
        monkeypatch.setattr(
            "src.core.llm.stream_completion", lambda **kwargs: generator
        )
        events = []
        result = call_llm_and_stream(
            "m", "http://server:8080/v1",
            {"uuid": "u", "question": "q", "expected": "e"}, events.append,
        )
        return result, events

    def test_plain_iterable_without_the_attributes_is_not_flagged(self, monkeypatch):
        # Fails open: a stream shape we do not recognise must never raise a
        # false alarm on a complete answer (this is what every other test in
        # this file feeds the loop).
        result, events = self._run(
            monkeypatch, iter([content_chunk("whole answer")])
        )
        assert result.truncated is False
        assert [e["event"] for e in events] == ["token"]

    def test_no_finish_reason_ever_received_flags_the_answer(self, monkeypatch):
        result, events = self._run(
            monkeypatch,
            FakeStream([content_chunk("half an "), content_chunk("answer")]),
        )
        assert result.truncated is True
        # The fragment is still returned (it is what the user should read), but
        # the run is told it is not a completed answer.
        assert result.text == "half an answer"
        errors = [e for e in events if e["event"] == "run_error"]
        assert [e["message"] for e in errors] == [TRUNCATION_MESSAGE]

    def test_received_finish_reason_leaves_the_answer_alone(self, monkeypatch):
        result, events = self._run(
            monkeypatch,
            FakeStream([content_chunk("done"), finish_chunk("stop")]),
        )
        assert result.truncated is False
        assert [e["event"] for e in events] == ["token"]

    def test_a_late_finish_reason_counts_as_received(self, monkeypatch):
        # The reason may arrive on any chunk, not the first: the flag is a
        # whole-stream fact, settled only once the loop has drained it.
        result, _ = self._run(
            monkeypatch,
            FakeStream([
                content_chunk("ok"),
                content_chunk(" still going"),
                finish_chunk("length"),
            ]),
        )
        assert result.truncated is False

    def test_reasoning_only_cut_is_flagged(self, monkeypatch):
        # A thinking model cut mid-reasoning streams an empty answer: gating on
        # the answer buffer would let that turn pass as a clean empty answer.
        result, events = self._run(
            monkeypatch, FakeStream([reasoning_chunk("thinking hard")])
        )
        assert result.truncated is True
        assert any(e["event"] == "run_error" for e in events)

    def test_empty_stream_is_not_flagged(self, monkeypatch):
        # Nothing was generated, so there is nothing to call incomplete.
        result, events = self._run(monkeypatch, FakeStream([]))
        assert result.truncated is False
        assert events == []

    def test_saw_finish_reason_directly(self):
        assert saw_finish_reason(object()) is True  # attribute absent
        assert saw_finish_reason(None) is True
        # Drained streams only: the flag is set while iterating.
        drained = FakeStream([finish_chunk("tool_calls")])
        list(drained)
        assert saw_finish_reason(drained) is True
        cut = FakeStream([content_chunk("half")])
        list(cut)
        assert saw_finish_reason(cut) is False


class TestCloseLlmStream:
    def test_none_generator_is_noop(self):
        close_llm_stream(None)

    def test_closes_the_stream_once(self):
        stream = FakeStream([content_chunk("x")])
        close_llm_stream(stream)
        assert stream.close_calls == 1

    def test_swallows_close_errors(self):
        stream = FakeStream([], close_error=RuntimeError("socket already gone"))
        close_llm_stream(stream)
        assert stream.closed

    def test_missing_close_is_noop(self):
        close_llm_stream(iter([]))


class TestCallLlmCancellation:
    def _run_cancelled(self, monkeypatch, gen):
        cancel = threading.Event()
        cancel.set()
        monkeypatch.setattr("src.core.llm.stream_completion", lambda **kwargs: gen)
        events = []
        with pytest.raises(RunCancelled):
            call_llm_and_stream(
                "m", "http://server:8080/v1",
                {"uuid": "u", "question": "q", "expected": "e"}, events.append,
                cancel_event=cancel,
            )
        return events

    def test_cancel_closes_the_stream(self, monkeypatch):
        stream = FakeStream([content_chunk("x")])
        events = self._run_cancelled(monkeypatch, stream)
        assert events == []
        assert stream.closed

    def test_cancel_with_a_stream_that_has_no_close_is_not_an_error(self, monkeypatch):
        # A stub-shaped stream (a bare iterable) has no socket to close; the
        # cancel must still surface as RunCancelled, not as an AttributeError.
        events = self._run_cancelled(monkeypatch, iter([content_chunk("x")]))
        assert events == []


class TestSafeGetChoiceContent:
    def test_dict_with_delta_content(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        assert safe_get_choice_content(chunk) == "hello"

    def test_dict_missing_delta_returns_empty(self):
        chunk = {"choices": [{}]}
        assert safe_get_choice_content(chunk) == ""

    def test_dict_with_no_choices_returns_empty(self):
        chunk = {}
        assert safe_get_choice_content(chunk) == ""

    def test_non_dict_chunk_returns_empty(self):
        assert safe_get_choice_content("string") == ""
        assert safe_get_choice_content(123) == ""

    def test_dict_with_empty_choices_returns_empty(self):
        chunk = {"choices": [], "usage": {"total_tokens": 12}}
        assert safe_get_choice_content(chunk) == ""


class TestExtractTotalTokens:
    def test_dict_chunk_with_usage(self):
        chunk = {"choices": [], "usage": {"total_tokens": 42}}
        assert _extract_total_tokens(chunk) == 42

    def test_non_dict_chunk_returns_none(self):
        assert _extract_total_tokens("not a chunk") is None

    def test_chunk_without_usage_returns_none(self):
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        assert _extract_total_tokens(chunk) is None

    def test_zero_total_returns_none(self):
        assert _extract_total_tokens({"usage": {"total_tokens": 0}}) is None


class TestExtractUsage:
    def test_dict_chunk_with_full_usage(self):
        chunk = {"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                           "total_tokens": 15}}
        assert _extract_usage(chunk) == {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }

    def test_missing_fields_are_none(self):
        chunk = {"usage": {"completion_tokens": 4}}
        assert _extract_usage(chunk) == {
            "prompt_tokens": None, "completion_tokens": 4, "total_tokens": None,
        }

    def test_chunk_without_usage_returns_none(self):
        assert _extract_usage({"choices": [{"delta": {"content": "hi"}}]}) is None

    def test_empty_usage_returns_none(self):
        assert _extract_usage({"usage": {}}) is None

    def test_null_usage_returns_none(self):
        assert _extract_usage({"usage": None}) is None

    def test_non_dict_chunk_returns_none(self):
        assert _extract_usage("not a chunk") is None


class TestChunksAreReadAsMappings:
    """The extractors read the *mapping protocol*, not ``isinstance(chunk, dict)``.

    The attribute branch (``chunk.choices[0].delta.content``) is gone for good
    (D2). What stays is a ``.get`` read, which a plain chunk dict and a non-dict
    chunk object that answers ``.get`` both satisfy — so the loop reads a chunk
    the same way whatever produced it. That is what a dict-only guard would
    break: a chunk carrying its fields through ``.get`` alone would read as
    empty and hand the caller an answer that ends on a tidy ``done``, which is
    the lie the truncation rules exist to prevent — pinned here by the non-dict
    stub below, since a dict-only read passes every dict-feeding test in this
    file.
    """

    class MappingChunk:
        """Not a dict, but answers ``.get`` with the same fields."""

        def __init__(self, fields):
            self._fields = fields

        def get(self, name, default=None):
            return self._fields.get(name, default)

    class AttributeChunk:
        """The deleted shape: attributes only, no mapping protocol."""

        def __init__(self):
            delta = SimpleNamespace(content="world", reasoning_content="thinking")
            self.choices = [SimpleNamespace(delta=delta)]
            self.usage = None

    def test_a_non_dict_mapping_chunk_is_still_read(self):
        chunk = self.MappingChunk({"choices": [{"delta": {"content": "hi"}}]})
        assert safe_get_choice_content(chunk) == "hi"
        # The usage block is a nested object of its own, not a dict, exactly as
        # a server's nested block arrives: the read is a ``.get``, not a type.
        usage = self.MappingChunk(
            {"usage": self.MappingChunk({"total_tokens": 9, "prompt_tokens": 4})})
        assert _extract_total_tokens(usage) == 9
        reasoning = self.MappingChunk(
            {"choices": [{"delta": {"reasoning_content": "deep"}}]}
        )
        assert _extract_reasoning(reasoning) == "deep"

    def test_the_attribute_branch_is_gone(self):
        # Attributes without a mapping protocol read as absent: no code path
        # reaches chunk.choices[0].delta.content any more.
        chunk = self.AttributeChunk()
        assert safe_get_choice_content(chunk) == ""
        assert _extract_total_tokens(chunk) is None


class TestCallLlmApiKey:
    def _captured_kwargs(self, monkeypatch, **call_kwargs):
        calls = []
        monkeypatch.setattr(
            "src.core.llm.stream_completion",
            lambda **kwargs: calls.append(kwargs) or iter([]),
        )
        events = []
        call_llm_and_stream(
            "m", "http://server:8080/v1",
            {"uuid": "u", "question": "q", "expected": "e"}, events.append,
            **call_kwargs,
        )
        assert len(calls) == 1
        return calls[0]

    def test_defaults_to_none_api_key(self, monkeypatch):
        assert self._captured_kwargs(monkeypatch)["api_key"] == "none"

    def test_uses_configured_api_key(self, monkeypatch):
        kwargs = self._captured_kwargs(monkeypatch, api_key="sk-42")
        assert kwargs["api_key"] == "sk-42"

    def test_passes_the_bare_model_and_server_url_to_the_seam(self, monkeypatch):
        # The lot's second invariant: no ``openai/`` prefix ever reaches the
        # request body. The URL-building and header rules are asserted once in
        # ``tests/test_llmkit_client.py``, not here.
        kwargs = self._captured_kwargs(monkeypatch)
        assert kwargs["model"] == "m"
        assert kwargs["server_url"] == "http://server:8080/v1"
        assert "api_base" not in kwargs


class TestVerifyResponseApiKey:
    def _verify(self, monkeypatch, api_key, content: Optional[str] = "true"):
        from src.core.llm import verify_response
        from src.core.models import VerifierAgent

        calls = []

        def fake_chat_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": content}}]}

        monkeypatch.setattr("src.core.llm.chat_completion", fake_chat_completion)
        verifier = VerifierAgent(endpoint="judge", serverUrl="http://v:8080/v1",
                                 apiKey=api_key)
        events = []
        ok = verify_response(verifier, {"question": "q", "expected": "e"}, "a",
                             events.append)
        return ok, calls[0], events

    def test_uses_verifier_api_key(self, monkeypatch):
        ok, kwargs, _ = self._verify(monkeypatch, "sk-judge")
        assert ok is True
        assert kwargs["api_key"] == "sk-judge"

    def test_falls_back_to_none_api_key(self, monkeypatch):
        _, kwargs, _ = self._verify(monkeypatch, "")
        assert kwargs["api_key"] == "none"

    def test_judge_model_and_server_url_reach_the_seam_bare(self, monkeypatch):
        _, kwargs, _ = self._verify(monkeypatch, "sk-judge")
        assert kwargs["model"] == "judge"
        assert kwargs["server_url"] == "http://v:8080/v1"

    def test_null_content_is_a_false_verdict_not_a_crash(self, monkeypatch):
        # A reasoning-only judge answer cut by max_tokens comes back with
        # ``content: null`` (spec §2.2): the answer is not graded, and the run
        # is not told anything went wrong with the transport.
        ok, _, events = self._verify(monkeypatch, "sk-judge", content=None)
        assert ok is False
        assert [e["event"] for e in events] == []


class TestWarmupModel:
    def test_sends_minimal_non_streaming_request(self, monkeypatch):
        calls = []

        def fake_chat_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": ""}}]}

        monkeypatch.setattr("src.core.llm.chat_completion", fake_chat_completion)
        warmup_model("my-model", "http://server:8080/v1")

        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["model"] == "my-model"
        assert kwargs["server_url"] == "http://server:8080/v1"
        assert kwargs["max_tokens"] == 1
        assert "stream" not in kwargs
        assert "stream_options" not in kwargs

    def test_uses_none_api_key_by_default(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "src.core.llm.chat_completion", lambda **kwargs: calls.append(kwargs)
        )
        warmup_model("my-model", "http://server:8080/v1")
        assert calls[0]["api_key"] == "none"

    def test_passes_configured_api_key(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "src.core.llm.chat_completion", lambda **kwargs: calls.append(kwargs)
        )
        warmup_model("my-model", "http://server:8080/v1", api_key="sk-42")
        assert calls[0]["api_key"] == "sk-42"

    def test_swallows_errors(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("model not found")

        monkeypatch.setattr("src.core.llm.chat_completion", boom)
        warmup_model("missing", "http://server:8080/v1")

    def test_cancelled_before_call_skips_request(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "src.core.llm.chat_completion", lambda **kwargs: called.append(kwargs)
        )
        cancel = threading.Event()
        cancel.set()
        warmup_model("my-model", "http://server:8080/v1", cancel)
        assert called == []
