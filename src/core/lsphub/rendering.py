"""Result rendering: LSP structures -> compact, line-numbered text.

What a tool call hands back to the model is text, and these helpers are the
whole contract: 1-based positions for readability (the input schemas ask for
0-based), capped output, and an explicit "suppressed" note instead of a
silent cut. They touch no session state, so they are testable on fixtures.
"""

from pathlib import Path
from typing import Any, List

from .constants import MAX_RESULT_CHARS


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def uri_to_path(uri: str) -> str:
    from pygls import uris as lsp_uris

    return lsp_uris.to_fs_path(uri) or uri


def relative(root: Path, uri: str) -> str:
    """Workspace-relative path for a file: URI (falls back to the raw path)."""
    path = Path(uri_to_path(uri))
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def severity_name(severity: Any) -> str:
    return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(
        int(severity) if severity is not None else 1, "error"
    )


def render_diagnostics(
    root: Path, diagnostics: List[Any], limit: int, default_uri: str = ""
) -> str:
    """``path:line:col severity: message (code)`` lines.

    ``publishDiagnostics`` carries the URI on the *notification*, not on each
    diagnostic, so the owning document's URI is passed in as ``default_uri``.
    """
    if not diagnostics:
        return "No diagnostics."
    lines: List[str] = []
    for diagnostic in diagnostics[:limit]:
        start = diagnostic.range.start
        code = getattr(diagnostic, "code", None)
        source = getattr(diagnostic, "source", None)
        suffix = ""
        if code is not None:
            suffix = f" ({source}{code})" if source else f" ({code})"
        message = " ".join(str(diagnostic.message).split())
        lines.append(
            f"{relative(root, getattr(diagnostic, 'uri', '') or default_uri)}"
            f":{start.line + 1}:{start.character + 1} "
            f"{severity_name(getattr(diagnostic, 'severity', None))}: "
            f"{message}{suffix}"
        )
    hidden = len(diagnostics) - len(lines)
    if hidden > 0:
        lines.append(f"[{hidden} more diagnostics suppressed]")
    return "\n".join(lines)


def location_line(root: Path, location: Any) -> str:
    # A Location carries uri/range; a LocationLink (definition/declaration
    # responses) carry target_uri/target_range instead.
    uri = getattr(location, "uri", None) or getattr(location, "target_uri", None)
    range_ = getattr(location, "range", None) or getattr(location, "target_range", None)
    if uri is None or range_ is None:
        return f"<unparsed location: {type(location).__name__}>"
    start = range_.start
    return f"{relative(root, uri)}:{start.line + 1}:{start.character + 1}"


def render_locations(root: Path, locations: Any, limit: int = 40) -> str:
    if not locations:
        return "No results."
    # Some servers (pyright) come back through cattrs as a tuple rather than
    # a list; treat any list/tuple as a collection of locations.
    if not isinstance(locations, (list, tuple)):
        locations = [locations]
    lines = [location_line(root, location) for location in locations[:limit]]
    if len(locations) > limit:
        lines.append(f"[{len(locations) - limit} more results suppressed]")
    return "\n".join(lines)


def symbol_kind_name(kind: Any) -> str:
    from lsprotocol import types as lsp

    try:
        return lsp.SymbolKind(int(kind)).name.lower()
    except (TypeError, ValueError):
        return "symbol"


def render_document_symbols(symbols: Any, limit: int = 200) -> str:
    if not symbols:
        return "No symbols."
    lines: List[str] = []

    def walk(items: Any, depth: int) -> None:
        for symbol in items:
            if len(lines) >= limit:
                return
            name = getattr(symbol, "name", "?")
            kind = symbol_kind_name(getattr(symbol, "kind", None))
            location = getattr(symbol, "location", None)
            range_ = getattr(symbol, "range", None)
            if location is not None:
                line = location.range.start.line + 1
            elif range_ is not None:
                line = range_.start.line + 1
            else:
                line = 0
            lines.append(f"{'  ' * depth}{kind} {name} (line {line})")
            children = getattr(symbol, "children", None)
            if children:
                walk(children, depth + 1)

    walk(symbols, 0)
    if len(lines) >= limit:
        lines.append("[more symbols suppressed]")
    return "\n".join(lines)


def render_hover(hover: Any) -> str:
    if hover is None or getattr(hover, "contents", None) is None:
        return "No hover information."
    from lsprotocol import types as lsp

    contents = hover.contents
    parts: List[str] = []
    if isinstance(contents, str):
        parts.append(contents)
    elif isinstance(contents, lsp.MarkupContent):
        parts.append(contents.value)
    elif isinstance(contents, lsp.MarkedStringWithLanguage):
        value = contents.value
        language = contents.language
        parts.append(f"```{language}\n{value}\n```" if language else value)
    else:
        for item in contents or []:
            if isinstance(item, str):
                parts.append(item)
            else:
                value = getattr(item, "value", "")
                language = getattr(item, "language", "")
                parts.append(f"```{language}\n{value}\n```" if language else value)
    text = "\n".join(part for part in parts if part)
    return text or "No hover information."


def render_completion(result: Any, limit: int) -> str:
    from lsprotocol import types as lsp

    items = result.items if isinstance(result, lsp.CompletionList) else result
    if not items:
        return "No completions."
    lines: List[str] = []
    for item in items[:limit]:
        label = getattr(item, "label", "?")
        detail = getattr(item, "detail", "") or ""
        lines.append(f"{label} — {detail}" if detail else str(label))
    if len(items) > limit:
        lines.append(f"[{len(items) - limit} more completions suppressed]")
    return "\n".join(lines)


def render_signature_help(help_: Any) -> str:
    if help_ is None or not getattr(help_, "signatures", None):
        return "No signature help."
    lines: List[str] = []
    for signature in help_.signatures:
        label = str(getattr(signature, "label", ""))
        documentation = getattr(signature, "documentation", None)
        if documentation is not None:
            value = documentation.value if hasattr(documentation, "value") else documentation
            text = " ".join(str(value).split())
            if text:
                label += f"\n  {text}"
        lines.append(label)
    return "\n".join(lines) or "No signature help."


def render_symbols(root: Path, symbols: Any, limit: int = 40) -> str:
    if not symbols:
        return "No symbols."
    lines: List[str] = []
    for symbol in symbols[:limit]:
        name = getattr(symbol, "name", "?")
        kind = symbol_kind_name(getattr(symbol, "kind", None))
        location = getattr(symbol, "location", None)
        where = location_line(root, location) if location is not None else "?"
        lines.append(f"{where} {kind} {name}")
    if len(symbols) > limit:
        lines.append(f"[{len(symbols) - limit} more symbols suppressed]")
    return "\n".join(lines)


def apply_whole_line_edits(text: str, edits: List[Any]) -> str:
    """Apply TextEdits to a copy of ``text`` (for the formatting diff preview).

    Applied bottom-up so the offsets of the not-yet-applied edits stay valid.
    Never touches the file on disk.
    """
    lines = text.splitlines(keepends=True)
    ordered = sorted(
        edits,
        key=lambda edit: (edit.range.start.line, edit.range.start.character),
        reverse=True,
    )
    for edit in ordered:
        start, end = edit.range.start, edit.range.end
        replacement = edit.new_text if isinstance(edit.new_text, str) else str(edit.new_text)
        if start.line >= len(lines):
            lines.append("\n")
        end_line = min(end.line, len(lines) - 1)
        prefix = "".join(lines[: start.line]) + (
            lines[start.line][: start.character] if start.line < len(lines) else ""
        )
        suffix = (lines[end_line][end.character :] if end_line < len(lines) else "") + "".join(
            lines[end_line + 1 :]
        )
        text = prefix + replacement + suffix
        lines = text.splitlines(keepends=True)
    return text


def render_formatting(relative_path: str, text: str, edits: Any) -> str:
    """Formatting edits rendered as a unified diff (never applied to disk)."""
    import difflib

    if not edits:
        return "Already formatted (no edits proposed)."
    usable = [
        edit
        for edit in edits
        if getattr(edit, "range", None) is not None and edit.new_text is not None
    ]
    if not usable:
        return "Already formatted (no edits proposed)."
    proposed = apply_whole_line_edits(text, usable)
    diff = difflib.unified_diff(
        text.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile=relative_path,
        tofile=relative_path,
        n=3,
    )
    rendered = "".join(diff)
    if not rendered:
        return "Already formatted (no visible change)."
    return rendered if rendered.endswith("\n") else rendered + "\n"


def language_id_for_suffix(suffix: str) -> str:
    """LSP language id for a file suffix ("" when unknown)."""
    return {
        ".py": "python",
        ".pyi": "python",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascriptreact",
        ".json": "json",
        ".vue": "vue",
    }.get(suffix.lower(), "")
