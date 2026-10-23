"""Document sync and the LSP requests behind each tool action.

These are the calls themselves — didOpen/didChange bookkeeping, then one
request per catalogue action, its result handed to ``rendering`` for the
text the model sees. They take the live session (and, for the pull-vs-push
diagnostics choice, nothing else) and keep no state of their own; the
manager owns the policy check that produced ``path`` and the loop the
timeouts run on.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict

from . import rendering
from .constants import DEFAULT_MAX_DIAGNOSTICS, DIAGNOSTICS_WAIT_SECONDS, REQUEST_TIMEOUT
from .session import ServerSession


async def sync_document(session: ServerSession, path: Path) -> str:
    """didOpen/didChange the file with its current content, return its URI."""
    from lsprotocol import types as lsp
    from pygls import uris as lsp_uris

    uri = lsp_uris.from_fs_path(str(path)) or "file://" + str(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    store = session.documents
    language_id = session.info.get("languageId") or rendering.language_id_for_suffix(
        path.suffix
    )
    version = store.next_version(uri)
    if store.version(uri) is None:
        store.add(uri, version, text, language_id)
        session.client.text_document_did_open(
            lsp.DidOpenTextDocumentParams(
                text_document=lsp.TextDocumentItem(
                    uri=uri,
                    language_id=language_id or "plaintext",
                    version=version,
                    text=text,
                )
            )
        )
    else:
        store.update(uri, version, text)
        session.client.text_document_did_change(
            lsp.DidChangeTextDocumentParams(
                text_document=lsp.VersionedTextDocumentIdentifier(
                    uri=uri, version=version
                ),
                content_changes=[
                    lsp.TextDocumentContentChangeWholeDocument(text=text)
                ],
            )
        )
    return uri


def position(arguments: Dict[str, Any]) -> Any:
    from lsprotocol import types as lsp

    line = arguments.get("line")
    character = arguments.get("character")
    if not isinstance(line, int) or not isinstance(character, int) or line < 0:
        raise RuntimeError(
            "line and character must be non-negative integers (0-based, as "
            "reported by lsp_*_diagnostics)"
        )
    return lsp.Position(line=line, character=max(int(character), 0))


async def fetch_diagnostics(session: ServerSession, root: Path, uri: str) -> str:
    """Pull diagnostics when the server advertises 3.17 pull support, else
    wait for the publishDiagnostics push on the inbox."""
    from lsprotocol import types as lsp

    limit = session.info.get("maxDiagnostics") or DEFAULT_MAX_DIAGNOSTICS
    if session.pull_diagnostics:
        report = await asyncio.wait_for(
            session.client.text_document_diagnostic_async(
                lsp.DocumentDiagnosticParams(
                    text_document=lsp.TextDocumentIdentifier(uri=uri)
                )
            ),
            timeout=REQUEST_TIMEOUT,
        )
        items = list(getattr(report, "items", None) or [])
        related = getattr(report, "related_documents", None) or {}
        for related_report in related.values():
            items.extend(getattr(related_report, "items", None) or [])
        return rendering.truncate(rendering.render_diagnostics(root, items, limit, uri))
    version = session.documents.version(uri)
    pushed = await session.inbox.wait(uri, version, DIAGNOSTICS_WAIT_SECONDS)
    if pushed is None:
        # Nothing ever arrived for this document: an empty answer would be
        # a lie, the server may simply still be indexing.
        return (
            "No diagnostics published by the server within "
            f"{DIAGNOSTICS_WAIT_SECONDS:.0f}s — the file may be clean, or "
            "the server is still indexing; call this tool again if in doubt."
        )
    return rendering.truncate(rendering.render_diagnostics(root, pushed, limit, uri))


async def workspace_symbols(session: ServerSession, arguments: Dict[str, Any]) -> str:
    """The one action that needs no document: a whole-workspace symbol query."""
    from lsprotocol import types as lsp

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return "Error: query is required"
    result = await asyncio.wait_for(
        session.client.workspace_symbol_async(lsp.WorkspaceSymbolParams(query=query)),
        timeout=REQUEST_TIMEOUT,
    )
    symbols = result if isinstance(result, list) else getattr(result, "items", None)
    root = Path(session.info["workspaceRoot"]).resolve()
    return rendering.truncate(rendering.render_symbols(root, symbols or []))


async def run_action(
    session: ServerSession, action: str, arguments: Dict[str, Any], path: Path
) -> str:
    """One catalogue action against an already policy-checked file."""
    from lsprotocol import types as lsp

    root = Path(session.info["workspaceRoot"]).resolve()
    uri = await sync_document(session, path)
    document = lsp.TextDocumentIdentifier(uri=uri)

    if action == "diagnostics":
        return await fetch_diagnostics(session, root, uri)
    if action == "hover":
        result = await asyncio.wait_for(
            session.client.text_document_hover_async(
                lsp.HoverParams(text_document=document, position=position(arguments))
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return rendering.truncate(rendering.render_hover(result))
    if action == "definition":
        result = await asyncio.wait_for(
            session.client.text_document_definition_async(
                lsp.DefinitionParams(text_document=document, position=position(arguments))
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return rendering.truncate(rendering.render_locations(root, result))
    if action == "references":
        result = await asyncio.wait_for(
            session.client.text_document_references_async(
                lsp.ReferenceParams(
                    text_document=document,
                    position=position(arguments),
                    context=lsp.ReferenceContext(
                        include_declaration=bool(arguments.get("includeDeclaration", True))
                    ),
                )
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return rendering.truncate(rendering.render_locations(root, result))
    if action == "completion":
        limit = arguments.get("limit") or 20
        result = await asyncio.wait_for(
            session.client.text_document_completion_async(
                lsp.CompletionParams(text_document=document, position=position(arguments))
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return rendering.truncate(rendering.render_completion(result, int(limit)))
    if action == "document_symbols":
        result = await asyncio.wait_for(
            session.client.text_document_document_symbol_async(
                lsp.DocumentSymbolParams(text_document=document)
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return rendering.truncate(rendering.render_document_symbols(result))
    if action == "signature_help":
        result = await asyncio.wait_for(
            session.client.text_document_signature_help_async(
                lsp.SignatureHelpParams(text_document=document, position=position(arguments))
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return rendering.truncate(rendering.render_signature_help(result))
    if action == "formatting":
        result = await asyncio.wait_for(
            session.client.text_document_formatting_async(
                lsp.DocumentFormattingParams(
                    text_document=document,
                    options=lsp.FormattingOptions(tab_size=4, insert_spaces=True),
                )
            ),
            timeout=REQUEST_TIMEOUT,
        )
        text = session.documents.text(uri) or path.read_text(
            encoding="utf-8", errors="replace"
        )
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path)
        return rendering.truncate(rendering.render_formatting(relative, text, result or []))
    return f"Error: unsupported LSP action '{action}'"
