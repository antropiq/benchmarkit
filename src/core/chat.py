"""Streaming chat with the benchmark-configured LLM.

Pure loop + event emission, like ``core/llm.py``: takes an ``emit``
callable, no FastAPI or browser concerns. Reasoning/thinking deltas are
streamed as ``reasoning_delta`` text (the Chat tab's left panel shows the
thinking timeline) while a single ``reasoning`` event flags that the model
started thinking (the app-bar status pill). Reasoning text is never answer
text: it never reaches ``token`` events nor the assistant message content
sent back to the model.

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

from src.core.llm import (
    Emit,
    TRUNCATION_EVENT_FLAG,
    TRUNCATION_MESSAGE,
    _extract_reasoning,
    _extract_usage,
    close_llm_stream,
    safe_get_choice_content,
    saw_finish_reason,
)
from src.core.llmkit import stream_completion
from src.core.llmkit.sanitize import StreamNeutralizer, neutralize
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

# The generation-side twin of llmkit's input sanitizer. Sanitizing the
# prompt cannot help when the user asks *about* the control tokens: the
# model generates a raw run, the serving runtime tokenizes it as a single
# end-of-turn token, and the sampler cuts this very reply mid-sentence --
# an EOG stop the app cannot even flag as truncation, because at the chunk
# level it looks like a finished answer. The prevention that works is
# telling the model to cite the tokens by name or with spaced-out
# delimiter characters instead of reproducing one. A recorded break showed
# where the model slips: prose follows the rule, then a fenced code block
# "showing the real thing" emits one -- so the notice names code fences
# and examples explicitly. Chat-only by design: the benchmark must keep
# measuring the model's raw behaviour, and it builds its prompts in
# core/llm.py, never here. The notice must never contain a raw run itself
# -- a test guards that with the sanitizer.
CONTROL_TOKEN_NOTICE = (
    "\n\nWhen discussing chat-template control tokens (im_start, im_end "
    "and their like), never write one in its raw delimiter-wrapped form: "
    "the serving template reads it as a turn boundary and cuts this very "
    "reply mid-sentence. This binds everywhere in the reply -- prose, code "
    "blocks, fenced examples, template renderings -- with no exemption "
    "for showing what the real thing looks like. Refer to them by name, "
    "or separate the delimiter characters with spaces (e.g. '< | im_start "
    "| >', and ' < | im_start | >user' inside a template example) so the "
    "text describes the token without becoming one."
)


def batch_instructions_block(text: str) -> str:
    """The lot's big picture rendered as a system-message block.

    Framed the way ``config.agent_instructions_block`` frames AGENTS.md: the
    wrapper says this describes a lot worked through *one task at a time*, so a
    big-picture file that reads "implement the editor tab" is taken as context
    rather than as an instruction to do the whole lot in this turn.
    """
    return (
        "The following is the big-picture description of the task lot being "
        "worked through, one task at a time. It is context, not a task: do "
        "only the task you are asked for now.\n\n" + text.strip()
    )


def _collect_tool_call_deltas(chunk: Any, acc: Dict[int, Dict[str, Any]]) -> str:
    """Accumulate streamed tool-call deltas (they can split across chunks).

    Returns the argument text carried by this chunk ("" when it streams no
    tool-call deltas) — argument fragments are generated tokens too, so the
    caller both anchors the decode window on them and folds them into the
    last-resort char-based token estimate.
    """
    arg_text = ""
    deltas = None
    if isinstance(chunk, dict):
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        deltas = delta.get("tool_calls") if isinstance(delta, dict) else None
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


def _emit_paused(
    emit: Emit,
    usage: Optional[Dict[str, Any]] = None,
) -> None:
    """Terminate a user-interrupted reply: exact usage when still missing,
    then the ``pause_ack`` terminal frame.

    ``usage`` is the cumulative figure the stream path needs (its round is
    folded but never reported), so the chat badges read the server's numbers
    for the partial turn instead of falling back to the client's ~4
    chars/token estimate. The tool-loop sites pass nothing: the same figures
    went out as the pre-tool ``usage`` frame. ``pause_ack`` is what reopens
    the composer.
    """
    if usage is not None:
        emit({"event": "usage", **usage})
    emit({"event": "pause_ack", "partial": True})


def stream_chat(
    messages: List[Dict[str, Any]],
    model_name: str,
    server_url: str,
    emit: Emit,
    sampling: Optional[Dict[str, Any]] = None,
    cancel_event: Optional[threading.Event] = None,
    pause_event: Optional[threading.Event] = None,
    api_key: str = "none",
    sysprompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_executor: Optional[ToolExecutor] = None,
    tool_servers: Optional[Dict[str, str]] = None,
    max_tool_iterations: int = 25,
    agent_instructions: Optional[str] = None,
    mode_instructions: Optional[str] = None,
    batch_instructions: Optional[str] = None,
) -> None:
    """Stream one chat reply, emitting ``token`` / ``reasoning`` / ``done``.

    Message ``content`` may be a plain string or a list of OpenAI-style
    multimodal parts (``text`` / ``image_url`` data URLs); parts are passed
    through to the LLM client untouched.

    Thinking streams as ``reasoning_delta`` (``text``) frames, preceded by
    the one-shot ``reasoning`` flag frame.

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
    **completion tokens only** (the tok/s numerator). ``contextTokens``/
    ``usage.contextTokens`` carry the LAST (resp. just-ended) round's
    prompt + generation — the context-window occupancy figure, which can
    exceed nothing but is NOT summed across rounds; ``contextEstimated``
    flags it when a fallback estimate was involved.
    Terminal event is exactly one of ``done``, ``error`` or ``pause_ack``
    (unless the call was cancelled — then the client is gone and nothing is
    emitted). ``pause_ack`` (carrying ``partial: True``) closes a reply the
    user interrupted through ``pause_event``; when the interrupted round has
    not been reported yet, a cumulative ``usage`` frame rides with the ack so
    the chat badges keep the server's exact figures. Precedence is cancel >
    error > pause — an ``error`` already emitted is never followed by a pause
    ack, and a pause request that arrives while a round was already finishing
    honestly still ends on ``done``. A pause never travels to the client as a
    truncation: the interrupt itself closed the upstream stream, so no
    ``finish_reason`` was expected and the truncation check is skipped.

    ``agent_instructions`` (when non-empty) is appended to the system message
    after the configured prompt, so a conversation starts already carrying the
    project's own instructions. It is deliberately a separate argument rather
    than pre-concatenated into ``sysprompt``: ``TOOL_NOTICE`` below is skipped
    when the prompt already mentions tools, and instruction files are full of
    the word — keeping them apart lets that check judge only the prompt the
    user configured. Like the notice above, it is appended to the text of the
    system message (the Chat API only ever sends plain-text system messages).

    ``mode_instructions`` (an agent mode's own generated blocks) rides between
    the two below — text only; this function never learns what a mode is.

    ``batch_instructions`` (batch mode's lot big picture) is appended *after*
    the project instructions and the mode blocks, so the system message reads
    sysprompt → tool notice → control-token notice → AGENTS.md → mode blocks
    → big picture: the project's rules stay the closest thing to the
    conversation, and the lot description — framed as context below, not as a
    task — is the furthest. It is re-sent on every task turn because the
    per-task context reset means nothing else carries it.
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
    # Every chat turn carries the control-token notice (it is gated on
    # nothing -- the danger is the model generating a raw run, not the
    # conversation carrying one). Placed after the TOOL_NOTICE word check
    # above so its wording never reaches that heuristic, and before the
    # instruction blocks so it stays part of the base behaviour.
    messages[0] = {
        "role": "system",
        "content": _content_text(messages[0]) + CONTROL_TOKEN_NOTICE,
    }
    if agent_instructions:
        # Appended last, so the notice heuristic above never sees the
        # instructions' own wording.
        messages[0] = {
            "role": "system",
            "content": _content_text(messages[0]) + "\n\n" + agent_instructions,
        }
    if mode_instructions:
        # The active mode's own standing instructions (the reserved Todo
        # builder carries the batch-lot conventions). After the project
        # instructions and before the lot big picture: AGENTS.md stays the
        # closest thing to the conversation, and the lot description stays
        # the furthest. It is a separate argument from agent_instructions
        # for the same reason those are separate: the TOOL_NOTICE word
        # check above must judge prompt-like text only.
        messages[0] = {
            "role": "system",
            "content": _content_text(messages[0]) + "\n\n" + mode_instructions,
        }
    if batch_instructions and batch_instructions.strip():
        # The lot's big picture closes the system message (see the docstring):
        # framed as context, so a model that reads "implement the editor tab"
        # in it does not start doing the whole lot on task 1.
        messages[0] = {
            "role": "system",
            "content": _content_text(messages[0])
            + "\n\n"
            + batch_instructions_block(batch_instructions),
        }
    sampling_kwargs: Dict[str, Any] = dict(sampling or {})

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
    # Context occupancy of the LAST round: that round's prompt + generation,
    # i.e. exactly what the server slotted into its window (template and tool
    # schemas included). Never accumulated across rounds: every round
    # re-processes the history, but only the latest round's figure says how
    # full the context window is.
    context_tokens = 0
    context_estimated = False
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
        try:
            response_generator: Any = stream_completion(
                model=model_name,
                server_url=server_url,
                messages=messages,
                api_key=api_key,
                sampling=sampling_kwargs,
                # The client omits an empty/``None`` ``tools``, so a tool-less
                # request still carries no ``tools`` key rather than ``[]``.
                tools=tools if use_tools else None,
            )
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
        paused = False
        # The display twin of the seam sanitizer: a token-shaped run can
        # straddle chunk borders, so what reaches the panels (and the
        # client's accumulated history) is filtered stream-by-stream,
        # never frame-by-frame. Raw text still feeds the accumulators
        # and the estimates -- those are measurement, not display -- and
        # the re-posted message is cleaned again at the seam anyway.
        answer_display = StreamNeutralizer()
        reasoning_display = StreamNeutralizer()

        def flush_display() -> None:
            # Hand over what the filters held back, before any terminal
            # frame of the round: a held benign partial (an unterminated
            # delimiter) is the user's text and must not vanish silently.
            left = reasoning_display.flush()
            if left:
                emit({"event": "reasoning_delta", "text": left})
            left = answer_display.flush()
            if left:
                emit({"event": "token", "text": left})

        try:
            for chunk in response_generator:
                if cancel_event is not None and cancel_event.is_set():
                    close_llm_stream(response_generator)
                    return
                # Cancel wins over pause (checked first, at every site).
                # Reasoning deltas and tool-call arguments are chunks of this
                # same loop, so this one check covers them for free.
                if pause_event is not None and pause_event.is_set():
                    close_llm_stream(response_generator)
                    paused = True
                    break
                usage = _extract_usage(chunk)
                if usage is not None:
                    iteration_usage = usage
                    if usage["completion_tokens"]:
                        iteration_completion = usage["completion_tokens"]
                        iteration_usage_from_completion = True
                tool_args_text = _collect_tool_call_deltas(chunk, tool_calls)
                content = safe_get_choice_content(chunk) or ""
                reasoning = _extract_reasoning(chunk)
                if reasoning:
                    if not reasoning_emitted:
                        reasoning_emitted = True
                        emit({"event": "reasoning"})
                    shown = reasoning_display.feed(reasoning)
                    if shown:
                        emit({"event": "reasoning_delta", "text": shown})
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
                shown = answer_display.feed(content)
                if shown:
                    emit({"event": "token", "text": shown})
        except Exception as exc:
            flush_display()
            emit({"event": "error", "message": f"Stream error: {exc}"})
            return

        # The chunk loop is over for every path that reaches here (finish,
        # pause break): hand the filters' held remainder over before any
        # terminal or per-round frame -- truncation error, pause ack,
        # pre-tool usage, done -- so the panels and the client's history
        # see the full display text of the round.
        flush_display()

        # The stream ended with no chunk ever carrying a finish_reason: the
        # server cut this round short (a disconnecting upstream behind
        # llama.cpp does exactly this) and the chunks of a cut answer look
        # like the chunks of a finished one. Stop here rather than falling
        # through: a half-streamed tool call would otherwise be executed with
        # truncated arguments — and the model would be told *its* JSON was
        # invalid, blaming it for a transport cut — while the fragment would
        # be re-posted as a finished assistant turn on the next send.
        # Checked after the loop: the stream only records the reason while it
        # is iterated.
        #
        # A pause disqualifies that reading: the interrupt itself closed the
        # stream, so no finish_reason was ever expected — and a flag armed a
        # moment before the stream died must not be reported as a server cut
        # either. Precedence stays cancel > error > pause.
        # ``saw_finish_reason`` is only meaningful now that the generator is
        # drained, which is why the raced ordering is settled here rather than
        # inside the loop: an interrupted round always ends on the ack, a round
        # that finished honestly (``done``) is never rewritten by a late flag.
        finish_seen = saw_finish_reason(response_generator)
        if (
            not paused
            and pause_event is not None
            and pause_event.is_set()
            and not finish_seen
        ):
            # The flag was armed while the last chunks were already arriving:
            # treat it as the interrupt that ended this round.
            paused = True
        if not paused and iteration_chars and not finish_seen:
            emit(
                {
                    "event": "error",
                    "message": TRUNCATION_MESSAGE,
                    TRUNCATION_EVENT_FLAG: True,
                }
            )
            return

        completion_tokens += iteration_completion
        if iter_first is not None and iter_last is not None:
            decode_seconds += iter_last - iter_first
        round_context_est = round_prompt_est + max(
            iteration_completion, estimate_completion_tokens(iteration_chars)
        )
        # Per-round figure: this round's prompt + generation, which is
        # ``context_tokens`` — how full the server's window is right now
        # (the figure of the LAST round only, never the sum over rounds).
        round_estimated = False
        if iteration_usage is not None:
            total = iteration_usage.get("total_tokens")
            prompt = iteration_usage.get("prompt_tokens")
            if total:
                # The server's per-round total covers its full prompt plus
                # generation: the exact occupancy figure for this round.
                round_context = total
            elif prompt:
                # Exact prompt without a total: keep it, add the completion
                # (or its char estimate — then flagged).
                round_context = prompt + (
                    iteration_completion
                    or estimate_completion_tokens(iteration_chars)
                )
                round_estimated = not iteration_completion
            else:
                # Prompt unseen (completion-only or empty usage): estimate
                # the round's prompt from the request we sent — flagged,
                # never presented as a server-reported count.
                round_context = round_context_est
                round_estimated = True
            if not iteration_usage_from_completion and total and prompt:
                # Only rounds that omitted completion_tokens contribute their
                # (total - prompt) delta to the fallback sum — a round that did
                # report it is already counted in completion_tokens.
                usage_derived_completion += max(total - prompt, 0)
        else:
            round_context = round_context_est
            round_estimated = True
        context_tokens = round_context
        context_estimated = round_estimated

        # A pause that landed inside the chunk loop ends the reply here: this
        # round's figures were just folded into the cumulative totals but are
        # never reported by any other frame, so the usage rides with the ack.
        # Checked *before* the tool-loop decision below, so a round interrupted
        # while its tool calls were still streaming never executes them: the
        # client cannot re-post a tool_calls turn, so that fragment is dropped.
        if paused:
            _emit_paused(
                emit,
                {
                    "tokens": completion_tokens,
                    "decode": decode_seconds,
                    "contextTokens": context_tokens,
                    "contextEstimated": context_estimated,
                },
            )
            return

        # `tool_executor is None` only narrows for the type checker: use_tools
        # already implies the executor is set.
        if not use_tools or tool_executor is None or not tool_calls:
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
                # Occupancy of the round that just ended (not cumulative):
                # what the server had in its window for that round.
                "contextTokens": context_tokens,
                "contextEstimated": context_estimated,
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
            # Pause asked for while the tool loop runs: a tool already running
            # is allowed to finish (Stop covers the hard case); only the
            # *next* call is skipped. The figures went out as the pre-tool
            # `usage` frame, so the ack needs no usage of its own.
            if pause_event is not None and pause_event.is_set():
                _emit_paused(emit)
                return
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
                    # The display name and argument preview reach the chat
                    # panel too: they are sanitized for the browser, while
                    # the executor and the re-posted message keep the raw
                    # text (the seam strips that path on the next round).
                    "tool": neutralize(tool_name),
                    "server": (tool_servers or {}).get(tool_name, ""),
                    "argsPreview": _preview(neutralize(args_preview)),
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
                    # Tool output is untrusted text rendered in the panel:
                    # the preview is run-free for the browser, the message
                    # re-posted to the model is stripped at the seam.
                    "preview": _preview(neutralize(result_text), 400),
                }
            )
        if cancel_event is not None and cancel_event.is_set():
            return
        # Between rounds: a pause asked for during a tool run (which was
        # allowed to finish) lands here instead of starting another round.
        # No usage frame — that round's figures went out before the first tool
        # executed.
        if pause_event is not None and pause_event.is_set():
            _emit_paused(emit)
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
        # Context occupancy of the FINAL round (prompt + generation — what
        # the server slotted into its window, template and tool schemas
        # included). It does not accumulate across rounds, so it is the
        # figure to compare against the server's context window
        # (`meta.n_ctx` from /v1/models).
        "contextTokens": context_tokens,
        "contextEstimated": context_estimated,
    })
