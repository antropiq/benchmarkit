"""Streaming chat with the benchmark-configured LLM.

Pure loop + event emission, like ``core/llm.py``: takes an ``emit``
callable, no FastAPI or browser concerns. Reasoning/thinking deltas are
never emitted as text — a single ``reasoning`` event flags that the model
is thinking — so the chat UI can show an indicator instead.

When ``tools`` and ``tool_executor`` are provided (MCP connectors active
for the Chat tab), the reply runs an OpenAI-style tool-calling loop: each
iteration streams the model, executes any ``tool_calls`` it produced via
the injected executor, feeds the results back, and repeats — up to
``max_tool_iterations`` — until a plain answer arrives.
"""

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from litellm import completion

from src.core.llm import (
    Emit,
    _extract_reasoning,
    _extract_usage,
    close_llm_stream,
    safe_get_choice_content,
)
from src.core.models import DEFAULT_LLM_SYSPROMPT
from src.core.utils import estimate_completion_tokens

# Executor bridge onto the MCP manager: name + args in, tool text out.
ToolExecutor = Callable[[str, Dict[str, Any]], str]

# Some tool-capable chat templates make models reluctant to call tools
# unless the prompt says they are available; appended (never into the
# shared LLM_SYSPROMPT) only while MCP tools are actually passed.
TOOL_NOTICE = (
    "\n\nYou can call the provided tools (browser/console) when they help "
    "answer the user."
)


def _collect_tool_call_deltas(chunk: Any, acc: Dict[int, Dict[str, Any]]) -> None:
    """Accumulate streamed tool-call deltas (they can split across chunks)."""
    if isinstance(chunk, dict):
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        deltas = delta.get("tool_calls") if isinstance(delta, dict) else None
    else:
        delta = getattr(chunk.choices[0], "delta", None) if getattr(chunk, "choices", None) else None
        deltas = getattr(delta, "tool_calls", None) if delta is not None else None
    for call in deltas or []:
        is_dict = isinstance(call, dict)
        index = (call.get("index") if is_dict else getattr(call, "index", None)) or 0
        entry = acc.setdefault(index, {"id": None, "name": "", "arguments": ""})
        call_id = call.get("id") if is_dict else getattr(call, "id", None)
        if call_id:
            entry["id"] = call_id
        function = call.get("function") if is_dict else getattr(call, "function", None)
        if function is None:
            continue
        name = function.get("name") if is_dict else getattr(function, "name", None)
        if name:
            entry["name"] += name
        arguments = (
            function.get("arguments") if is_dict else getattr(function, "arguments", None)
        )
        if arguments:
            entry["arguments"] += arguments


def _preview(value: str, limit: int = 200) -> str:
    return value if len(value) <= limit else value[:limit] + "\n... [truncated]"


def _content_text(message: Dict[str, Any]) -> str:
    """Text of a message's content, "" for multimodal (list) content parts."""
    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def stream_chat(
    messages: List[Dict[str, Any]],
    model_name: str,
    server_url: str,
    emit: Emit,
    sampling: Optional[Dict[str, Any]] = None,
    cancel_event: Optional[threading.Event] = None,
    api_key: str = "none",
    sysprompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_executor: Optional[ToolExecutor] = None,
    tool_servers: Optional[Dict[str, str]] = None,
    max_tool_iterations: int = 25,
) -> None:
    """Stream one chat reply, emitting ``token`` / ``reasoning`` / ``done``.

    Message ``content`` may be a plain string or a list of OpenAI-style
    multimodal parts (``text`` / ``image_url`` data URLs); parts are passed
    through to litellm untouched.

    With MCP tools active also emits ``tool_call`` / ``tool_result`` events
    per executed tool call (``tool_call`` carries ``iteration`` and
    ``maxIterations`` for the UI gauge) plus a cumulative ``usage`` event
    (``tokens`` / ``decode`` seconds) after each tool round. Terminal event
    is exactly one of ``done`` or ``error`` (unless the call was cancelled —
    then the client is gone and nothing is emitted).
    """
    if not messages or messages[0].get("role") != "system":
        messages = [
            {"role": "system", "content": sysprompt or DEFAULT_LLM_SYSPROMPT},
            *messages,
        ]

    use_tools = bool(tools) and tool_executor is not None
    if use_tools and "tool" not in _content_text(messages[0]).lower():
        # Replace (never mutate) the caller's system message.
        messages[0] = {
            "role": "system",
            "content": _content_text(messages[0]) + TOOL_NOTICE,
        }
    sampling_kwargs: Dict[str, Any] = dict(sampling or {})
    # litellm gates reasoning_effort to OpenAI reasoning models only;
    # allow it through so custom servers (llama.cpp) receive it.
    allowed: Optional[list] = (
        ["reasoning_effort"] if "reasoning_effort" in sampling_kwargs else None
    )

    t0 = time.monotonic()
    answer_buffer = ""
    completion_tokens = 0
    last_usage: Optional[Dict[str, Optional[int]]] = None
    reasoning_emitted = False
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    decode_seconds = 0.0
    capped = False

    iteration = 0
    while True:
        kwargs: Dict[str, Any] = {
            "model": f"openai/{model_name}",
            "api_key": api_key,
            "messages": messages,
            "api_base": server_url,
            "stream": True,
            "stream_options": {"include_usage": True},
            "allowed_openai_params": allowed,
            **sampling_kwargs,
        }
        if use_tools:
            kwargs["tools"] = tools
        try:
            response_generator = completion(**kwargs)
        except Exception as exc:
            emit({"event": "error", "message": f"LLM call failed: {exc}"})
            return

        iteration_content = ""
        iteration_completion = 0
        iteration_usage: Optional[Dict[str, Optional[int]]] = None
        tool_calls: Dict[int, Dict[str, Any]] = {}
        iter_first: Optional[float] = None
        iter_last: Optional[float] = None
        try:
            for chunk in response_generator:
                if cancel_event is not None and cancel_event.is_set():
                    close_llm_stream(response_generator)
                    return
                usage = _extract_usage(chunk)
                if usage is not None:
                    iteration_usage = usage
                    if usage["completion_tokens"]:
                        iteration_completion = usage["completion_tokens"]
                _collect_tool_call_deltas(chunk, tool_calls)
                content = safe_get_choice_content(chunk)
                reasoning = _extract_reasoning(chunk)
                if reasoning and not reasoning_emitted:
                    reasoning_emitted = True
                    emit({"event": "reasoning"})
                # Reasoning deltas open and extend the decode window (like the
                # benchmark loop in core/llm.py): they are generated tokens too,
                # so tok/s must not pretend the thinking phase was free.
                if not content and not reasoning:
                    continue
                arrival = time.monotonic()
                if t_first is None:
                    t_first = arrival
                t_last = arrival
                if iter_first is None:
                    iter_first = arrival
                iter_last = arrival
                if not content:
                    continue
                iteration_content += content
                answer_buffer += content
                emit({"event": "token", "text": content})
        except Exception as exc:
            emit({"event": "error", "message": f"Stream error: {exc}"})
            return

        completion_tokens += iteration_completion
        if iter_first is not None and iter_last is not None:
            decode_seconds += iter_last - iter_first
        if iteration_usage is not None:
            last_usage = iteration_usage

        if not use_tools or not tool_calls:
            break

        iteration += 1
        messages.append(
            {
                "role": "assistant",
                "content": iteration_content or None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for _, call in sorted(tool_calls.items())
                ],
            }
        )
        for _, call in sorted(tool_calls.items()):
            tool_name = call["name"]
            try:
                tool_args = json.loads(call["arguments"]) if call["arguments"] else {}
                invalid_json = None
            except json.JSONDecodeError as exc:
                tool_args = {}
                invalid_json = exc
            try:
                args_preview = json.dumps(tool_args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_preview = str(tool_args)
            emit(
                {
                    "event": "tool_call",
                    "id": call["id"],
                    "tool": tool_name,
                    "server": (tool_servers or {}).get(tool_name, ""),
                    "argsPreview": _preview(args_preview),
                    "iteration": iteration,
                    "maxIterations": max_tool_iterations,
                }
            )
            if invalid_json is not None:
                # Return a clear error so the model self-corrects next iteration.
                result_text = (
                    f"Error: the arguments for '{tool_name}' are not valid JSON. "
                    f"Fix the syntax and retry. Details: {invalid_json}"
                )
                ok = False
            else:
                try:
                    result_text = tool_executor(tool_name, tool_args)
                    ok = True
                except Exception as exc:  # noqa: BLE001 — feed failure back to the model
                    result_text = f"Error calling {tool_name}: {exc}"
                    ok = False
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result_text}
            )
            emit(
                {
                    "event": "tool_result",
                    "id": call["id"],
                    "ok": ok,
                    "preview": _preview(result_text, 400),
                }
            )
        if cancel_event is not None and cancel_event.is_set():
            return
        # Cumulative usage of the completed round (tool-call argument tokens
        # included): lets the chat UI update its token badges while the tool
        # loop is still running instead of waiting for the final `done`.
        emit(
            {
                "event": "usage",
                "tokens": completion_tokens,
                "decode": decode_seconds,
            }
        )
        if iteration >= max_tool_iterations:
            capped = True
            break

    if capped:
        note = (
            f"\n\n(maximum of {max_tool_iterations} tool iterations reached — stopping)"
        )
        answer_buffer += note
        emit({"event": "token", "text": note})

    took = (t_last or time.monotonic()) - t0
    ttft = t_first - t0 if t_first is not None else 0.0
    tokens_estimated = False
    if not completion_tokens:
        if last_usage is not None:
            total = last_usage.get("total_tokens")
            prompt = last_usage.get("prompt_tokens")
            if total and prompt:
                completion_tokens = max(total - prompt, 0)
        if not completion_tokens:
            completion_tokens = estimate_completion_tokens(answer_buffer)
            tokens_estimated = completion_tokens > 0

    emit({
        "event": "done",
        "took": took,
        "ttft": ttft,
        "tokens": completion_tokens or 0,
        "tokensEstimated": tokens_estimated,
    })
