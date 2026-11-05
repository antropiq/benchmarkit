"""The two call shapes, the error type and the URL rules.

The mechanism half of the package that speaks HTTP to a model endpoint
(spec §4.3): one POST to ``{server_url}/chat/completions``, streaming or
not. The tuning numbers it reads live in :mod:`llmkit.constants`; the SSE
frames are decoded by :mod:`llmkit.sse`. Like everything under ``core/`` it
imports no FastAPI, emits no events and never reads ``.env``: the caller
passes ``model`` / ``server_url`` / ``api_key`` / ``sampling`` in, exactly as
it passed them to the call this package replaces.

Two invariants hold here and nowhere else may they bend (spec §2.1):

* the ``sampling`` dict is merged into the request body **verbatim** --
  ``top_k``, ``min_p`` and ``repeat_penalty`` are llama.cpp sampler names,
  and a client that whitelisted OpenAI's own list would silently change the
  model's sampling, and every score of every run;
* the model name travels **bare** -- no provider prefix is ever prepended,
  because ``openai/qwen3.8-...`` reaches llama-swap as a 404.

One ``httpx.Client`` per call, owned and closed by the returned stream
(D11): nothing in here holds a reference across requests.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional

import httpx

from . import constants
from .sse import iter_chunks

logger = logging.getLogger(constants.DEBUG_LOGGER_NAME)


def debug_enabled() -> bool:
    """Whether the request payload is logged at DEBUG (the payload half of D8).

    A module-level hook, off by default: the debug switch itself lives in
    ``config`` (``LLM_DEBUG``), which is a layer this core package must not
    import. The debug owner re-points this callable at its own predicate; the
    failed-call diagnosis itself does not wait for it -- it is in
    :meth:`LLMRequestError.__str__`, visible with the flag off.
    """
    return False


class LLMRequestError(Exception):
    """A call the server refused with a non-2xx answer.

    Carries the status, the upstream message and the URL. ``str()`` names
    both the status and the upstream message (D8): both consuming loops do
    ``except Exception as exc`` and emit ``str(exc)`` straight into the UI's
    error line, so the message is the surface that matters, and it must be
    complete with the debug flag off. The debug flag only adds the request
    payload on top.
    """

    def __init__(self, status: int, message: str, url: str) -> None:
        self.status = status
        self.message = message
        self.url = url
        super().__init__(
            f"the model endpoint answered HTTP {status} for {url}: {message}"
        )


def _v1_base(server_url: str) -> str:
    """The ``/v1`` root of a server URL, appended when it is not there yet.

    ``SERVER_URL`` usually already ends in ``/v1`` (it is the historical
    ``api_base``); a bare host gets it added. One rule for both URL shapes
    (D9), so the two builders below cannot drift.
    """
    url = server_url.strip().rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def chat_completions_url(server_url: str) -> str:
    """The chat-completions endpoint for a server URL."""
    return f"{_v1_base(server_url)}/chat/completions"


def models_url(server_url: str) -> str:
    """The OpenAI-compatible models listing for a server URL."""
    return f"{_v1_base(server_url)}/models"


def _timeout() -> httpx.Timeout:
    """The D10 timeout, read through ``constants`` at call time.

    A short connect timeout (a dead endpoint fails fast) and no read
    deadline: a slow model load must stay a slow first token, never surface
    as a truncated answer.
    """
    return httpx.Timeout(
        constants.CONNECT_TIMEOUT,
        read=constants.READ_TIMEOUT,
        write=constants.WRITE_TIMEOUT,
        pool=constants.POOL_TIMEOUT,
    )


def _headers(api_key: str) -> Dict[str, str]:
    """The bearer header, sent even when the key is the placeholder.

    That is what the endpoint sees today and it accepts it (spec §2.1). The
    key never reaches a log line from here.
    """
    key = (api_key or "").strip() or constants.API_KEY_PLACEHOLDER
    return {"Authorization": f"Bearer {key}"}


def _error_message(response: httpx.Response) -> str:
    """What the upstream said, however it said it.

    An OpenAI-shaped ``{"error": {"message": ...}}`` body contributes its
    message; anything else contributes its raw text truncated to
    ``ERROR_BODY_LIMIT`` (a full HTML error page is not a diagnosis). Never
    raises: a body that does not decode is exactly the case the truncation
    fallback covers.
    """
    try:
        raw = response.text
    except Exception:  # noqa: BLE001 — undecodable body: fall back to nothing
        raw = ""
    raw = (raw or "").strip()
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        elif isinstance(error, str) and error.strip():
            return error.strip()
    return raw[: constants.ERROR_BODY_LIMIT] or "(empty error body)"


def _request_error(response: httpx.Response, url: str) -> LLMRequestError:
    return LLMRequestError(
        status=response.status_code,
        message=_error_message(response),
        url=url,
    )


def _log_request(url: str, model: str, body: Dict[str, Any]) -> None:
    """The debug half of D8: the request line and the JSON payload.

    The payload is the body as it goes out -- headers are not part of it, so
    the ``Authorization`` header and the key can never reach the log from
    here.
    """
    if not debug_enabled():
        return
    logger.debug("POST %s (model %s)", url, model)
    logger.debug("request payload: %s", json.dumps(body, default=str))


class ChatStream:
    """One streamed completion: an iterable of chunk dicts plus its end state.

    Iteration hands over the chunk dicts :mod:`llmkit.sse` parses and
    records as it yields (never before: the loop has not seen a chunk it did
    not yield). ``saw_finish_reason`` is a chunk-level fact only (D3): a
    ``[DONE]`` that arrives with no finish-reason chunk leaves it ``False``,
    so a stream the server cut still looks cut.

    The socket's own teardown is :meth:`close` -- response and client,
    idempotent, never raising (the cancel path calls it blind). Iteration
    that ends, exhausts or raises-through closes the client too, so a stream
    that dies mid-read never leaks its connection. (This is unrelated to the
    process-tree teardown ``core/proctree.py`` owns.)
    """

    def __init__(self, client: httpx.Client, response: httpx.Response) -> None:
        self._client = client
        self._response = response
        self._closed = False
        self._finish_reason: Optional[str] = None
        self._saw_finish_reason = False
        self._usage: Optional[Dict[str, Any]] = None

    @property
    def finish_reason(self) -> Optional[str]:
        """The first non-null ``finish_reason`` any yielded chunk carried."""
        return self._finish_reason

    @property
    def saw_finish_reason(self) -> bool:
        """Whether any yielded chunk carried one (D3: chunk-level only)."""
        return self._saw_finish_reason

    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        """The last usage block any yielded chunk carried, if any."""
        return self._usage

    @property
    def closed(self) -> bool:
        """Whether the response and client have been torn down."""
        return self._closed

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        try:
            for chunk in iter_chunks(self._response.iter_lines()):
                self._record(chunk)
                yield chunk
        finally:
            self.close()

    def _record(self, chunk: Dict[str, Any]) -> None:
        for choice in chunk.get("choices") or []:
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            if reason and self._finish_reason is None:
                self._finish_reason = reason
                self._saw_finish_reason = True
        usage = chunk.get("usage")
        if usage:
            self._usage = usage

    def close(self) -> None:
        """Close response and client. Idempotent, never raises."""
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        except Exception:  # noqa: BLE001 — teardown must not mask the caller's error
            pass
        finally:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass


def stream_completion(
    *,
    model: str,
    server_url: str,
    messages: List[Dict],
    api_key: str = constants.API_KEY_PLACEHOLDER,
    sampling: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> ChatStream:
    """Start one streamed chat completion and return its :class:`ChatStream`.

    The body is the measured wire (spec §2.1): ``stream`` plus
    ``stream_options.include_usage``, the ``sampling`` dict merged in
    verbatim (no whitelist, no rename, no drop), and ``tools`` added **only
    when non-empty** -- no ``tools`` key at all rather than ``tools: []``.
    A non-2xx answer raises :class:`LLMRequestError` before any chunk is
    handed over; the stream object owns the client and closes it.
    """
    url = chat_completions_url(server_url)
    body: Dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if sampling:
        body.update(sampling)
    if tools:
        body["tools"] = tools
    _log_request(url, model, body)

    client = httpx.Client(timeout=_timeout())
    response: Optional[httpx.Response] = None
    try:
        request = client.build_request(
            "POST", url, json=body, headers=_headers(api_key)
        )
        response = client.send(request, stream=True)
        if not response.is_success:
            response.read()  # the error body must be whole before it is read
            raise _request_error(response, url)
    except Exception:
        if response is not None:
            try:
                response.close()
            except Exception:  # noqa: BLE001 — the caller's error wins
                pass
        client.close()
        raise
    return ChatStream(client, response)


def chat_completion(
    *,
    model: str,
    server_url: str,
    messages: List[Dict],
    api_key: str = constants.API_KEY_PLACEHOLDER,
    max_tokens: Optional[int] = None,
    sampling: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One non-streaming chat completion; returns the parsed body.

    Used by warm-up (``max_tokens=1``) and the verifier call. Any non-2xx
    raises :class:`LLMRequestError` naming the status and the upstream
    message.
    """
    url = chat_completions_url(server_url)
    body: Dict[str, Any] = {"messages": messages, "model": model}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if sampling:
        body.update(sampling)
    _log_request(url, model, body)

    with httpx.Client(timeout=_timeout()) as client:
        response = client.post(url, json=body, headers=_headers(api_key))
    if not response.is_success:
        raise _request_error(response, url)
    return response.json()
