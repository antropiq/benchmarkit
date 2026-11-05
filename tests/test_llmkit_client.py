"""The transport layer of the in-house LLM client (llmh-03, spec §4.3).

Everything here runs through ``httpx.MockTransport``: the client's own
request building, streaming and teardown are exercised for real, and only
the socket is a handler. The assertions are step 5's list -- the measured
§2.1 body against llmh-01's fixture, the placeholder bearer header, the
bare model name, the sampler keys verbatim, the ``tools`` rule, the
chunk-level finish-reason recording (D3), the error surface that names the
status and the upstream message with the debug flag off (D8), the close
semantics, and the one URL rule for both endpoint shapes (D9).

Hermetic by construction: a call whose test installed no handler fails the
test instead of reaching a network, and nothing here reads ``.env``.
"""

import json
import logging
from typing import Any, Dict, List

import httpx
import pytest

from src.core.bench import process_question
from src.core.job import RunJob
from src.core.llm import TRUNCATION_MESSAGE, call_llm_and_stream
from src.core.models import EndpointsConfig
from src.core.llmkit import client as client_module
from src.core.llmkit import constants
from src.core.llmkit.client import (
    ChatStream,
    LLMRequestError,
    chat_completion,
    chat_completions_url,
    models_url,
    stream_completion,
)
from tests.llmkit_fakes import (
    CLOSE_DELIM,
    OPEN_DELIM,
    PIPE,
    content_chunk,
    finish_chunk,
    special_token,
    usage_chunk,
)
from tests.test_llmkit_contract import (
    EXPECTED_STREAM_BODY,
    MAX_TOKENS,
    MODEL,
    SAMPLING,
)

SERVER = "http://model.test:8080/v1"
BARE_HOST = "http://model.test:8080"
URL = "http://model.test:8080/v1/chat/completions"


# --- the MockTransport seam --------------------------------------------------
#
# The client owns its httpx client per call (D11), so what a test injects is
# the transport, not the client: the code under test still builds the real
# request, streams the real way and closes the real way.


class _Wire:
    """Every ``httpx.Client`` the client builds answers from ``serve``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.requests: List[httpx.Request] = []
        self.clients: List[httpx.Client] = []
        self.client_kwargs: List[Dict[str, Any]] = []
        self.handler = None
        self._real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", self._factory)

    def _factory(self, *args, **kwargs):
        self.client_kwargs.append(dict(kwargs))
        kwargs["transport"] = httpx.MockTransport(self._dispatch)
        client = self._real_client(*args, **kwargs)
        self.clients.append(client)
        return client

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self.handler is not None, "this test installed no handler"
        return self.handler(request)

    @property
    def request(self) -> httpx.Request:
        assert len(self.requests) == 1, "expected exactly one request"
        return self.requests[0]

    def json_body(self) -> Dict[str, Any]:
        return json.loads(self.request.content)

    def serve_stream(
        self,
        chunks: List[Dict[str, Any]],
        done: bool = True,
        status: int = 200,
    ) -> None:
        """Answer with an SSE body, as the live server writes one."""
        lines = [": keep-alive"]  # the live server emits comment lines
        for chunk in chunks:
            lines.append("data: " + json.dumps(chunk))
            lines.append("")  # blank line as frame separator
        if done:
            lines.append("data: " + "[DONE]")
            lines.append("")
        body = "\n".join(lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status,
                headers={"content-type": "text/event-stream"},
                content=body,
            )

        self.handler = handler

    def serve_json(self, payload: Any, status: int = 200) -> None:
        self.serve_raw(json.dumps(payload), status, "application/json")

    def serve_raw(
        self, body: str, status: int = 200, content_type: str = "text/plain"
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status,
                headers={"content-type": content_type},
                content=body.encode("utf-8"),
            )

        self.handler = handler


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    return _Wire(monkeypatch)


def start(**overrides: Any) -> ChatStream:
    """One streamed call with the contract fixture's own values."""
    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "server_url": SERVER,
        "messages": [],
    }
    kwargs.update(overrides)
    return stream_completion(**kwargs)


# --- the §2.1 request body, against llmh-01's fixture -------------------------


class TestRequestBodyIsTheMeasuredWire:
    def test_the_streaming_body_equals_the_measured_wire(self, wire: _Wire):
        # The fixture's ``max_tokens`` is one of the call's own values (the
        # other is the model name); on the streaming shape the caller's
        # ``sampling`` dict is the route any call value takes, so it travels
        # that way here. Everything else in the body the client itself puts
        # there -- and the equality says it puts *exactly* that, nothing
        # dropped and nothing invented.
        wire.serve_stream([])
        stream = start(sampling=dict(SAMPLING, max_tokens=MAX_TOKENS))
        list(stream)
        assert wire.json_body() == EXPECTED_STREAM_BODY

    def test_the_stream_shape_adds_stream_and_include_usage(self, wire: _Wire):
        wire.serve_stream([])
        list(start(sampling=dict(SAMPLING)))
        body = wire.json_body()
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        # Without a max_tokens in the caller's dict none appears: the client
        # adds the two streaming keys and nothing else.
        assert "max_tokens" not in body

    def test_the_post_goes_to_the_chat_completions_url(self, wire: _Wire):
        wire.serve_stream([])
        list(start())
        assert wire.request.method == "POST"
        assert str(wire.request.url) == chat_completions_url(SERVER) == URL

    def test_the_model_name_reaches_the_body_bare(self, wire: _Wire):
        # A leftover ``openai/`` prefix reaches llama-swap as
        # ``openai/qwen3.8-...`` and comes back 404 no router for model.
        wire.serve_stream([])
        list(start(model="qwen3.8-flash-next-iq3_xxs"))
        body = wire.json_body()
        assert body["model"] == "qwen3.8-flash-next-iq3_xxs"
        assert not any("openai/" in str(v) for v in body.values())

    def test_an_unknown_sampler_key_appears_verbatim(self, wire: _Wire):
        # The lot's first invariant: the sampling dict is merged unexamined.
        # This server answers a bogus key with 200 (spec §2.2), so a client
        # that filtered it would change the sampling with nothing to see.
        wire.serve_stream([])
        list(start(sampling={"dys_tfs": 0.9, "typo_k": 20, "top_k": 20}))
        body = wire.json_body()
        assert body["dys_tfs"] == 0.9
        assert body["typo_k"] == 20
        assert body["top_k"] == 20


class TestAuthorizationHeader:
    def test_bearer_none_is_sent_with_no_key_configured(self, wire: _Wire):
        # Spec §2.1: the header goes out even with the placeholder -- that is
        # what the endpoint sees today and it accepts it.
        wire.serve_stream([])
        list(start())
        assert wire.request.headers["authorization"] == "Bearer none"

    def test_an_empty_key_falls_back_to_the_placeholder(self, wire: _Wire):
        wire.serve_stream([])
        list(start(api_key="  "))
        assert wire.request.headers["authorization"] == "Bearer none"

    def test_a_configured_key_is_sent_as_bearer(self, wire: _Wire):
        wire.serve_stream([])
        list(start(api_key="sk-local"))
        assert wire.request.headers["authorization"] == "Bearer sk-local"


class TestToolsKeyRule:
    """``tools`` is absent rather than empty (the rule in tests/test_chat.py)."""

    def test_no_tools_key_when_tools_is_none(self, wire: _Wire):
        wire.serve_stream([])
        list(start(tools=None))
        assert "tools" not in wire.json_body()

    def test_no_tools_key_when_tools_is_an_empty_list(self, wire: _Wire):
        wire.serve_stream([])
        list(start(tools=[]))
        assert "tools" not in wire.json_body()

    def test_tools_present_when_non_empty(self, wire: _Wire):
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        wire.serve_stream([])
        list(start(tools=tools))
        assert wire.json_body()["tools"] == tools


# --- the content sanitizer at the seam -------------------------------------------


class TestMessageContentIsSanitizedAtTheSeam:
    """No delimiter-wrapped control-token run leaves the app in a body.

    Both call shapes pass their ``messages`` through ``llmkit.sanitize``
    before the body is built, so benchmark, verifier, warm-up, chat and
    batch are covered here and nowhere else.
    """

    def test_a_control_token_never_reaches_the_stream_request_body(
        self, wire: _Wire
    ):
        wire.serve_stream([])
        list(start(messages=[{"role": "user",
                              "content": f"q: {special_token('evil')} answer"}]))
        assert wire.json_body()["messages"][0]["content"] == "q:  answer"

    def test_a_spliced_run_never_reaches_the_wire(self, wire: _Wire):
        # The fixed point, checked on the bytes that would go out:
        # removing the inner run must not splice an outer one together.
        wire.serve_stream([])
        smuggled = f"a{OPEN_DELIM}im_{special_token('x')}start{CLOSE_DELIM}b"
        list(start(messages=[{"role": "user", "content": smuggled}]))
        assert wire.json_body()["messages"][0]["content"] == "ab"

    def test_the_non_streaming_shape_sanitizes_too(self, wire: _Wire):
        # The verifier and warm-up ride this shape; the pass is not the
        # streaming shape's private favour.
        wire.serve_json({})
        chat_completion(model=MODEL, server_url=SERVER,
                        messages=[{"role": "user",
                                   "content": special_token("verdict") + "yes"}])
        assert wire.json_body()["messages"][0]["content"] == "yes"

    def test_the_caller_s_messages_are_not_mutated(self, wire: _Wire):
        # The same history object is what the UI renders: it shows what
        # the user typed, while the wire carries the sanitized copy.
        wire.serve_stream([])
        original = "keep " + special_token("evil")
        messages = [{"role": "user", "content": original}]
        list(start(messages=messages))
        assert messages[0]["content"] == original
        assert wire.json_body()["messages"][0]["content"] == "keep "

    def test_delimiters_that_form_no_run_survive_to_the_body(self, wire: _Wire):
        # Sanitizing must not become stripping: a pipe inside the name is
        # not what any template tokenises as one control token, and an
        # unterminated half of a pair is not a run either. Such text --
        # and a benchmark question carrying it -- reaches the server
        # byte for byte.
        wire.serve_stream([])
        text = (
            f"compare a{OPEN_DELIM}b{PIPE}c{CLOSE_DELIM}d "
            f"and x{OPEN_DELIM}y{PIPE}w and z{CLOSE_DELIM}v"
        )
        list(start(messages=[{"role": "user", "content": text}]))
        assert wire.json_body()["messages"][0]["content"] == text

    def test_a_run_in_tool_call_arguments_never_reaches_the_wire(self, wire: _Wire):
        # The tool loop re-posts the assistant's own tool_calls each
        # round, and the template interpolates the function text raw --
        # an argument run smuggles a turn close as surely as content.
        wire.serve_stream([])
        list(start(messages=[{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": f'{{"q": "{special_token("im")}"}}',
                },
            }],
        }]))
        fn = wire.json_body()["messages"][0]["tool_calls"][0]["function"]
        assert fn["arguments"] == '{"q": ""}'
        assert fn["name"] == "search"

    def test_the_logged_payload_is_the_sanitized_body(
        self,
        wire: _Wire,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # The debug payload is the body as it goes out -- so what the
        # operator sees is what the server receives, token or not.
        monkeypatch.setattr(client_module, "debug_enabled", lambda: True)
        wire.serve_json({})
        with caplog.at_level(logging.DEBUG, logger=constants.DEBUG_LOGGER_NAME):
            chat_completion(model=MODEL, server_url=SERVER,
                            messages=[{"role": "user",
                                       "content": special_token("evil") + "ok"}])
        text = "\n".join(record.getMessage() for record in caplog.records)
        assert '"ok"' in text
        assert OPEN_DELIM not in text


# --- what the stream records (D3) ----------------------------------------------


class TestStreamRecording:
    def test_a_late_finish_reason_flips_the_flag_as_its_chunk_is_yielded(
        self, wire: _Wire
    ):
        wire.serve_stream(
            [content_chunk("half an "), content_chunk("answer"), finish_chunk("stop")]
        )
        stream = start()
        it = iter(stream)
        next(it)
        next(it)
        assert stream.saw_finish_reason is False
        assert stream.finish_reason is None
        next(it)  # the finish chunk: recorded as it is handed over
        assert stream.saw_finish_reason is True
        assert stream.finish_reason == "stop"

    def test_a_stream_ending_on_done_with_no_finish_chunk_is_still_cut(
        self, wire: _Wire
    ):
        # D3: ``data: [DONE]`` is transport framing. Whether it survives an
        # upstream disconnect is unverified, so it proves nothing.
        wire.serve_stream([content_chunk("half an ")], done=True)
        stream = start()
        list(stream)
        assert stream.saw_finish_reason is False
        assert stream.finish_reason is None

    def test_a_null_finish_reason_is_not_a_signal(self, wire: _Wire):
        wire.serve_stream([finish_chunk(None)])
        stream = start()
        list(stream)
        assert stream.saw_finish_reason is False
        assert stream.finish_reason is None

    def test_finish_reason_is_the_first_non_null_one_seen(self, wire: _Wire):
        wire.serve_stream([finish_chunk("length"), finish_chunk("stop")])
        stream = start()
        list(stream)
        assert stream.finish_reason == "length"

    def test_usage_is_the_last_usage_block_seen(self, wire: _Wire):
        wire.serve_stream(
            [
                usage_chunk(prompt=10, total=12),
                finish_chunk("stop", usage={"prompt_tokens": 10,
                                            "completion_tokens": 2,
                                            "total_tokens": 12}),
            ]
        )
        stream = start()
        list(stream)
        assert stream.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }

    def test_usage_is_none_when_no_chunk_carried_one(self, wire: _Wire):
        wire.serve_stream([content_chunk("a")])
        stream = start()
        list(stream)
        assert stream.usage is None


# --- the socket's own teardown --------------------------------------------------


class _RaisingClose:
    """A response or client whose close fails -- the socket is already gone."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise OSError("socket already gone")


class TestStreamTeardown:
    def test_exhausting_the_iteration_closes_the_client(self, wire: _Wire):
        wire.serve_stream([content_chunk("a"), finish_chunk("stop")])
        stream = start()
        list(stream)
        assert stream.closed is True
        assert wire.clients[-1].is_closed

    def test_close_mid_stream_returns_and_closes_the_client(self, wire: _Wire):
        # The cancel path: close() is called blind, and llama.cpp must see
        # the socket die instead of streaming into it nobody reads.
        wire.serve_stream([content_chunk("a"), content_chunk("b")])
        stream = start()
        it = iter(stream)
        next(it)
        stream.close()
        assert stream.closed is True
        assert wire.clients[-1].is_closed

    def test_close_twice_is_a_no_op(self):
        client, response = _RaisingClose(), _RaisingClose()
        stream = ChatStream(client, response)  # type: ignore[arg-type]
        stream.close()
        stream.close()
        assert stream.closed is True
        assert client.close_calls == 1  # the second call never got through
        assert response.close_calls == 1

    def test_a_failing_underlying_close_never_propagates(self):
        client, response = _RaisingClose(), _RaisingClose()
        stream = ChatStream(client, response)  # type: ignore[arg-type]
        stream.close()  # the cancel path calls this blind; it must not raise
        assert stream.closed is True
        assert client.close_calls == 1
        assert response.close_calls == 1

    def test_a_refused_call_closes_its_client_before_raising(self, wire: _Wire):
        wire.serve_json({"error": {"message": "no router for requested model"}},
                        status=404)
        with pytest.raises(LLMRequestError):
            start()
        assert wire.clients[-1].is_closed


# --- the error surface (D8) -----------------------------------------------------


class TestErrorSurface:
    def test_a_404_raises_naming_status_and_upstream_message_with_debug_off(
        self, wire: _Wire
    ):
        # The exact llama-swap answer to a prefixed model name. The debug flag
        # is *off* here on purpose (the hook's own default): str() is what the
        # two consuming loops emit into the UI's error line, so the diagnosis
        # has to be complete without the flag -- the flag only adds the payload
        # on top (D8).
        assert client_module.debug_enabled() is False
        wire.serve_json(
            {"error": {"message": "no router for requested model",
                       "type": "invalid_request_error"}},
            status=404,
        )
        with pytest.raises(LLMRequestError) as excinfo:
            start()
        error = excinfo.value
        assert error.status == 404
        assert error.message == "no router for requested model"
        assert error.url == URL
        text = str(error)
        assert "404" in text
        assert "no router for requested model" in text

    def test_a_non_json_error_body_is_included_raw_truncated(
        self, wire: _Wire, monkeypatch: pytest.MonkeyPatch
    ):
        # A test that wants another body limit patches the constant, never a
        # literal in the module under test.
        monkeypatch.setattr(constants, "ERROR_BODY_LIMIT", 24)
        wire.serve_raw("<html><body>502 Bad Gateway</body>" + "x" * 5000,
                       status=503)
        with pytest.raises(LLMRequestError) as excinfo:
            start()
        message = excinfo.value.message
        assert message == "<html><body>502 Bad Gate"  # exactly the limit
        assert len(message) == 24
        assert "xxxx" not in message

    def test_an_error_body_that_does_not_decode_never_raises_a_decode_error(
        self, wire: _Wire
    ):
        wire.serve_raw("", status=500)
        with pytest.raises(LLMRequestError) as excinfo:
            start()
        assert excinfo.value.status == 500

    def test_the_non_streaming_shape_raises_the_same_error(self, wire: _Wire):
        wire.serve_json({"error": {"message": "context length exceeded"}},
                        status=400)
        with pytest.raises(LLMRequestError) as excinfo:
            chat_completion(model=MODEL, server_url=SERVER, messages=[])
        assert excinfo.value.status == 400
        assert "context length exceeded" in str(excinfo.value)


# --- the non-streaming shape -----------------------------------------------------


class TestChatCompletion:
    def test_it_returns_the_parsed_body(self, wire: _Wire):
        payload = {"choices": [{"message": {"content": "True"}}]}
        wire.serve_json(payload)
        assert chat_completion(model=MODEL, server_url=SERVER, messages=[]) == payload

    def test_the_body_carries_no_stream_keys(self, wire: _Wire):
        wire.serve_json({})
        chat_completion(model=MODEL, server_url=SERVER, messages=[])
        body = wire.json_body()
        assert "stream" not in body
        assert "stream_options" not in body
        assert "max_tokens" not in body

    def test_max_tokens_and_sampling_reach_the_body(self, wire: _Wire):
        wire.serve_json({})
        chat_completion(model=MODEL, server_url=SERVER, messages=[],
                        max_tokens=1, sampling=dict(SAMPLING))
        body = wire.json_body()
        assert body["max_tokens"] == 1
        for key, value in SAMPLING.items():
            assert body[key] == value

    def test_it_posts_to_the_same_url_with_the_bearer_header(self, wire: _Wire):
        wire.serve_json({})
        chat_completion(model=MODEL, server_url=SERVER, messages=[])
        assert wire.request.method == "POST"
        assert str(wire.request.url) == URL
        assert wire.request.headers["authorization"] == "Bearer none"


# --- the URL rule (D9) ------------------------------------------------------------


class TestUrlRuleIsOneRule:
    @pytest.mark.parametrize(
        "base",
        [SERVER, SERVER + "/", BARE_HOST, BARE_HOST + "/", f"  {BARE_HOST}  "],
    )
    def test_both_builders_follow_one_rule_for_both_shapes(self, base: str):
        # ``SERVER_URL`` usually already ends in ``/v1`` (it is the historical
        # api_base); a bare host gets it added. One rule, asserted for both
        # URL shapes so the two can never drift.
        assert chat_completions_url(base) == f"{SERVER}/chat/completions"
        assert models_url(base) == f"{SERVER}/models"

    def test_the_output_is_always_a_model_endpoint_path(self):
        # The write-surface guard formalises the wider rule in llmh-04; the
        # builders spell theirs here: whatever the base, the path is the
        # endpoint's own under ``/v1``, never an app route's shape.
        for base in (SERVER, BARE_HOST):
            assert chat_completions_url(base).endswith("/v1/chat/completions")
            assert models_url(base).endswith("/v1/models")


# --- the timeout (D10) -------------------------------------------------------------


class TestTimeoutComesFromConstants:
    def test_the_call_timeout_is_the_constants_one(self, wire: _Wire):
        wire.serve_json({})
        chat_completion(model=MODEL, server_url=SERVER, messages=[])
        timeout = wire.client_kwargs[-1]["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == pytest.approx(constants.CONNECT_TIMEOUT)
        assert timeout.read is None  # D10: no read deadline on a live stream
        assert timeout.write == pytest.approx(constants.WRITE_TIMEOUT)
        assert timeout.pool == pytest.approx(constants.POOL_TIMEOUT)

    def test_a_patched_constant_reaches_the_next_call(
        self, wire: _Wire, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(constants, "CONNECT_TIMEOUT", 3.5)
        wire.serve_stream([])
        list(start())
        assert wire.client_kwargs[-1]["timeout"].connect == 3.5


# --- the debug hook (payload half of D8) --------------------------------------------


class TestDebugPayloadLogging:
    def test_the_hook_defaults_to_off_and_nothing_is_logged(
        self, wire: _Wire, caplog: pytest.LogCaptureFixture
    ):
        assert client_module.debug_enabled() is False
        wire.serve_json({})
        with caplog.at_level(logging.DEBUG, logger=constants.DEBUG_LOGGER_NAME):
            chat_completion(model=MODEL, server_url=SERVER, messages=[])
        assert caplog.records == []

    def test_the_hook_on_logs_the_request_line_and_payload_but_never_the_key(
        self,
        wire: _Wire,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(client_module, "debug_enabled", lambda: True)
        wire.serve_json({})
        with caplog.at_level(logging.DEBUG, logger=constants.DEBUG_LOGGER_NAME):
            chat_completion(model=MODEL, server_url=SERVER, messages=[],
                            api_key="sk-secret")
        text = "\n".join(record.getMessage() for record in caplog.records)
        assert URL in text
        assert MODEL in text
        assert '"model"' in text  # the JSON payload itself
        assert "Authorization" not in text
        assert "sk-secret" not in text


# --- truncation end-to-end through the real transport (llmh-10, spec §6.3.6) ----
#
# The AGENTS.md invariant -- *a truncated benchmark answer is failed and never
# sent to the verifier* -- is usually asserted per layer: a fake stream that
# raises, a stub that reports no finish reason. Asserted below is the whole
# chain with no stub in the middle: the bytes a cut server would have written
# go through the real SSE parse, the real stream recording, the real benchmark
# answer loop and the real bench decision about the verifier. Only the socket
# is a handler, and the verifier is a spy that records whether it was reached.
# Spec §6.3.6 names this the one check that must be repeated after any change
# to ``sse.py``.


class TestTruncationReachesTheVerdictThroughTheRealParse:
    QUESTION = {"uuid": "u1", "question": "q1", "expected": "a1"}

    def serve_cut_answer(self, wire: _Wire, done: bool) -> None:
        """Answer chunks, then the stream dies: never a finish_reason frame."""
        wire.serve_stream(
            [content_chunk("Paris is "), content_chunk("the capital")],
            done=done,
        )

    def ask(self):
        """One benchmark answer over whatever the wire serves."""
        events: List[Dict[str, Any]] = []
        result = call_llm_and_stream(
            MODEL,
            SERVER,
            dict(self.QUESTION),
            events.append,
            dict(SAMPLING, max_tokens=MAX_TOKENS),
            api_key="none",
        )
        return result, events

    @pytest.mark.parametrize("done", [False, True], ids=["cut socket", "done frame"])
    def test_a_cut_stream_is_truncated_and_its_error_is_emitted(
        self, wire: _Wire, done: bool
    ):
        # D3 earns its keep here: whether the server got as far as writing
        # ``data: [DONE]`` changes nothing -- framing is not a finish reason,
        # so both shapes of the cut stream are truncation.
        self.serve_cut_answer(wire, done)
        result, events = self.ask()

        assert result.truncated is True
        assert result.text == "Paris is the capital"  # the fragment is kept
        errors = [e for e in events if e["event"] == "run_error"]
        assert [e["message"] for e in errors] == [TRUNCATION_MESSAGE]
        # The fragment still streamed as tokens: the card shows what arrived,
        # and the error that follows it is what marks it incomplete.
        streamed = "".join(e["text"] for e in events if e["event"] == "token")
        assert streamed == result.text

    @pytest.mark.parametrize("done", [False, True], ids=["cut socket", "done frame"])
    def test_the_fragment_never_reaches_the_verifier_and_the_question_fails(
        self, wire: _Wire, done: bool, monkeypatch: pytest.MonkeyPatch
    ):
        # The bench's own decision, taken on the answer the real parse
        # produced: no verifier call at all, and the question in the failed list.
        self.serve_cut_answer(wire, done)
        judged: List[str] = []

        def spy(verifier, question, response_text, emit):
            judged.append(response_text)
            return True

        monkeypatch.setattr("src.core.bench.verify_response", spy)
        config = EndpointsConfig(serverUrl=SERVER, endpoint=MODEL, **SAMPLING)
        job = RunJob()
        events: List[Dict[str, Any]] = []

        process_question(MODEL, dict(self.QUESTION), 1, 0, config, job, events.append)

        assert judged == []
        assert job.get_failed_questions() == [1]
        answer_done = next(e for e in events if e["event"] == "answer_done")
        assert answer_done["truncated"] is True

    def test_a_stream_that_did_finish_is_not_truncated(self, wire: _Wire):
        # The same chain, one frame different: a chunk that carries a
        # finish_reason. Without this case the tests above could be passing
        # because everything reads as truncated.
        wire.serve_stream([
            content_chunk("Paris is the capital"),
            finish_chunk("stop", usage={"prompt_tokens": 10,
                                        "completion_tokens": 4,
                                        "total_tokens": 14}),
        ])
        result, events = self.ask()

        assert result.truncated is False
        assert [e for e in events if e["event"] == "run_error"] == []
        # And the measurement that rides on a finished answer is still the
        # server's, not a re-count (the timing/token rules this lot must not
        # move): completion tokens only, prompt tokens carried separately.
        assert result.completion_tokens == 4
        assert result.prompt_tokens == 10
        assert result.tokens_estimated is False
        assert result.ttft >= 0.0
        assert result.took >= result.ttft
