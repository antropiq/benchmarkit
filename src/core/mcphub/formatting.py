"""MCP tool/result objects -> the shapes the chat tool loop expects.

Pure text-level conversion, no client and no I/O: the OpenAI function-calling
definitions handed to litellm, and the plain-text rendering of a
``CallToolResult`` fed back as the tool turn.
"""

from typing import Any, Dict, List


def tools_to_openai_format(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert MCP tool objects into OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


# One content block of a CallToolResult: normally a typed object, but a plain
# dict carries the same fields when the result came back un-modelled.
def _content_text(item: Any) -> str:
    """The ``text`` field of one content block, whatever its shape."""
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(getattr(item, "text", ""))


def extract_tool_result_text(result: Any, max_length: int = 120_000) -> str:
    """Extract plain text from an MCP CallToolResult, truncating if needed."""
    if getattr(result, "is_error", False):
        error_texts = [
            _content_text(item)
            for item in result.content
            if hasattr(item, "text") or (isinstance(item, dict) and "text" in item)
        ]
        return "\n".join(error_texts) or "[Tool execution failed — no details.]"

    text_parts: List[str] = []
    non_text_types: List[str] = []
    for item in result.content:
        if hasattr(item, "text"):
            text_parts.append(item.text)
        elif isinstance(item, dict) and "text" in item:
            text_parts.append(item["text"])
        else:
            content_type = (
                getattr(item, "type", None)
                or (item.get("type") if isinstance(item, dict) else None)
                or type(item).__name__
            )
            non_text_types.append(str(content_type))

    if text_parts:
        text = "\n".join(text_parts)
        if non_text_types:
            summary = ", ".join(sorted(set(non_text_types)))
            text += f"\n[Note: tool also returned non-text content of types: {summary}]"
    elif non_text_types:
        summary = ", ".join(sorted(set(non_text_types)))
        text = f"[Tool returned no text output. Content types received: {summary}]"
    else:
        text = "[Tool returned empty response.]"

    if len(text) > max_length:
        text = text[:max_length] + "\n... [truncated]"
    return text
