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


def _collect_tool_call_deltas(chunk: Any, acc: Dict[int, Dict[str, Any]]) -> str:
    """Accumulate streamed tool-call deltas (they can split across chunks).

    Returns the argument text carried by this chunk ("" when it streams no
    tool-call deltas) — argument fragments are generated tokens too, so the
    caller both anchors the decode window on them and folds them into the
    last-resort char-based token estimate.
    """
    arg_text = ""
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
            arg_text += str(arguments)
    return arg_text


def _preview(value: str, limit: int = 200) -> str:
    return value if len(value) <= limit else value[:limit] + "\n... [truncated]"


def _content_text(message: Dict[str, Any]) -> str:
    """Text of a message's content, "" for multimodal (list) content parts."""
    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def _prompt_estimate(messages: List[Dict[str, Any]]) -> int:
    """Char-based (~4 chars/token) estimate of one round's prompt cost.

    Fallback for rounds whose server reported no usage at all: the message
    history as it stands at the round's start is exactly what the request
    sends as prompt (the server-side system prompt beyond it is not visible
    here, so the figure is a floor — flagged through the estimated badge).
    """
    return estimate_completion_tokens("".join(_content_text(m) for m in messages))


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
    ``maxIterations`` for the UI gauge) plus a cumulative ``usage`` event as
    soon as each tool round has finished generating — before the tool
    executes, so the chat badges tick when generation stops instead of
    waiting for the tool to return. The terminal ``done`` carries the
    cumulative ``decode`` seconds (sum of the per-iteration generation
    windows: tool-call argument deltas included, tool execution and
    inter-round re-prefill gaps excluded) so the client can compute an
    accurate tok/s with tools in play. ``tokens``/``usage.tokens`` count
    **completion tokens only** (the tok/s numerator); ``consumed``/
    ``usage.consumed`` count **tokens actually consumed** — prompt +
    completion summed per round, since every tool round re-processes the
    whole history (``consumedEstimated`` flags char-based fallbacks, e.g.
    servers reporting no usage at all). Terminal event is exactly one of
    ``done`` or ``error`` (unless the call was cancelled — then the client
    is gone and nothing is emitted).
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
    # Every generated character seen (answer + reasoning + tool-call
    # arguments): the corpus for the last-resort char-based token estimate
    # when the server reports no usage at all.
    generated_chars = ""
    completion_tokens = 0
    # Sum of (total - prompt) per round — each round's total covers its full
    # prompt, so the per-round differences add up to the loop's completion
    # tokens; used only when no round reported completion_tokens itself.
    usage_derived_completion = 0
    # "Consumed" tokens = prompt + completion, summed PER ROUND: every tool
    # round re-sends and re-processes the whole history, so each round's own
    # prompt counts again (a round's server-reported total_tokens is exactly
    # that round's consumed figure). Completion tokens and the tok/s rate
    # stay completion-only — prompt processing is prefill, not decode.
    consumed_tokens = 0
    consumed_estimated = False
    reasoning_emitted = False
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    decode_seconds = 0.0
    capped = False

    iteration = 0
    while True:
        # Prompt cost of this round, estimated from the exact messages the
        # request is about to send — used only if the server reports no
        # usable usage for the round (floor: invisible server-side template
        # tokens are not in it, hence the estimated flag).
        round_prompt_est = _prompt_estimate(messages)
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
        iteration_chars = ""
        iteration_completion = 0
        iteration_usage: Optional[Dict[str, Optional[int]]] = None
        iteration_usage_from_completion = False
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
                        iteration_usage_from_completion = True
                tool_args_text = _collect_tool_call_deltas(chunk, tool_calls)
                content = safe_get_choice_content(chunk) or ""
                reasoning = _extract_reasoning(chunk)
                if reasoning and not reasoning_emitted:
                    reasoning_emitted = True
                    emit({"event": "reasoning"})
                # Content, reasoning and tool-call argument deltas all open
                # and extend the decode window (like the benchmark loop in
                # core/llm.py): they are all generated tokens counted in
                # completion_tokens, so neither the thinking phase nor a
                # tool-call-only round may look free to the tok/s maths.
                if not content and not reasoning and not tool_args_text:
                    continue
                arrival = time.monotonic()
                if t_first is None:
                    t_first = arrival
                t_last = arrival
                if iter_first is None:
                    iter_first = arrival
                iter_last = arrival
                generated_chars += content + reasoning + tool_args_text
                iteration_chars += content + reasoning + tool_args_text
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
        round_consumed_est = round_prompt_est + max(
            iteration_completion, estimate_completion_tokens(iteration_chars)
        )
        if iteration_usage is not None:
            total = iteration_usage.get("total_tokens")
            prompt = iteration_usage.get("prompt_tokens")
            if total:
                # The server's per-round total covers its full prompt plus
                # generation: the exact consumed figure for this round.
                consumed_tokens += total
            elif prompt:
                # Exact prompt without a total: keep it, add the completion
                # (or its char estimate — then flagged).
                consumed_tokens += prompt + (
                    iteration_completion
                    or estimate_completion_tokens(iteration_chars)
                )
                consumed_estimated = consumed_estimated or not iteration_completion
            else:
                # Prompt unseen (completion-only or empty usage): estimate
                # the round's prompt from the request we sent — flagged,
                # never presented as a server-reported count.
                consumed_tokens += round_consumed_est
                consumed_estimated = True
            if not iteration_usage_from_completion and total and prompt:
                # Only rounds that omitted completion_tokens contribute their
                # (total - prompt) delta to the fallback sum — a round that did
                # report it is already counted in completion_tokens.
                usage_derived_completion += max(total - prompt, 0)
        else:
            consumed_tokens += round_consumed_est
            consumed_estimated = True

        if not use_tools or not tool_calls:
            break

        iteration += 1
        # This round's generation is over: report cumulative usage *before*
        # executing the tool(s) (which can take tens of seconds), so the
        # chat token badge ticks as soon as generation stops. Emitted before
        # the post-tool cancellation check too: a stop during the tool run
        # still leaves the client badges on the exact figure of everything
        # generated so far.
        emit(
            {
                "event": "usage",
                "tokens": completion_tokens,
                "decode": decode_seconds,
                "consumed": consumed_tokens,
                "consumedEstimated": consumed_estimated,
            }
        )
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
        # Servers that omit completion_tokens still report total/prompt per
        # round: the per-round differences summed cover every tool round
        # (last_usage alone would keep only the final round's share).
        if usage_derived_completion:
            completion_tokens = usage_derived_completion
        else:
            # Char heuristic over everything generated: answer text,
            # reasoning and tool-call arguments alike.
            completion_tokens = estimate_completion_tokens(generated_chars)
            tokens_estimated = completion_tokens > 0

    emit({
        "event": "done",
        "took": took,
        "ttft": ttft,
        # Cumulative decode seconds across all rounds — the tok/s denominator
        # once tools ran (took - ttft would wrongly include tool execution
        # and inter-round gaps). Equals took - ttft for a single round.
        "decode": decode_seconds,
        "tokens": completion_tokens or 0,
        "tokensEstimated": tokens_estimated,
        # Consumed tokens (prompt + completion) summed over every round of
        # the loop — what the conversation actually cost, as opposed to
        # `tokens` (completion only, the tok/s numerator).
        "consumed": consumed_tokens,
        "consumedEstimated": consumed_estimated,
    })
