"""The client side of the conversation: what the hub answers a server.

pygls 2.x calls a feature handler with a single ``params`` argument, and
tsserver/Volar are the noisiest here: an unanswerable
``workspace/configuration`` or ``client/registerCapability`` is what makes or
breaks the TS/Vue servers. Kept apart from the manager so these handlers can
be read as one contract; they reach the live session only through the
``sessions`` mapping and log through ``append_log``, so a restart simply
re-binds a fresh set.
"""

from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .constants import (
    PROJECT_CONFIG_NAMES,
    TSSERVER_PROJECT_INFO,
    TSSERVER_REQUEST,
    TSSERVER_RESPONSE,
)
from .session import ServerSession
from .tsserver import (
    parse_tsserver_request,
    tsserver_project_info,
    tsserver_response_params,
)


def client_capabilities() -> Any:
    """What the hub declares it supports. Deliberately explicit: Pyright
    only publishes diagnostics when the client advertises
    ``publishDiagnostics``, and only asks ``workspace/configuration`` when
    the client declares ``configuration``."""
    from lsprotocol import types as lsp

    return lsp.ClientCapabilities(
        workspace=lsp.WorkspaceClientCapabilities(
            apply_edit=False,
            configuration=True,
            workspace_folders=True,
            did_change_configuration=lsp.DidChangeConfigurationClientCapabilities(
                dynamic_registration=True
            ),
        ),
        text_document=lsp.TextDocumentClientCapabilities(
            publish_diagnostics=lsp.PublishDiagnosticsClientCapabilities(
                related_information=True,
                version_support=True,
            ),
            hover=lsp.HoverClientCapabilities(
                content_format=[lsp.MarkupKind.Markdown, lsp.MarkupKind.PlainText]
            ),
            # The hub renders both shapes (Location and LocationLink),
            # so it can accept links. Declaring False made Volar refuse
            # to answer definition requests for Vue components and log
            # "textDocument.definition.linkSupport is not supported by
            # the client but could be used by the server".
            definition=lsp.DefinitionClientCapabilities(
                link_support=True,
            ),
            references=lsp.ReferenceClientCapabilities(),
            document_symbol=lsp.DocumentSymbolClientCapabilities(
                hierarchical_document_symbol_support=True
            ),
            completion=lsp.CompletionClientCapabilities(),
            signature_help=lsp.SignatureHelpClientCapabilities(),
            formatting=lsp.DocumentFormattingClientCapabilities(),
            diagnostic=lsp.DiagnosticClientCapabilities(),
        ),
        window=lsp.WindowClientCapabilities(work_done_progress=True),
    )


def register_features(
    client: Any,
    info: Dict[str, Any],
    log_path: Path,
    sessions: Mapping[str, ServerSession],
    append_log: Callable[[Path, str], None],
    project_config_for: Callable[[ServerSession, str], Optional[str]],
) -> None:
    """Answer the server->client messages the hub must not crash on."""

    @client.feature("textDocument/publishDiagnostics")
    def on_diagnostics(params: Any) -> None:
        # Resolved through the manager so a restart re-binds a fresh inbox.
        session = sessions.get(info["id"])
        if session is None:
            return
        session.inbox.publish(
            params.uri, getattr(params, "version", None), list(params.diagnostics)
        )

    @client.feature("workspace/configuration")
    def on_configuration(params: Any) -> List[Any]:
        """One answer per requested section: ``null`` means 'not set'.

        The LSP spec's ``ConfigurationResponse`` is a per-item value that
        is ``null`` when nothing is configured. Answering ``{}`` instead
        (as this handler used to) is not a no-op: a server doing the
        idiomatic ``getConfiguration('x') ?? []`` keeps the ``{}`` because
        it is not nullish, and then iterates it — which is exactly how
        Volar dies on ``html.customData`` with ``TypeError: paths is not
        iterable``, taking the whole process down on the first Vue
        completion request. ``null`` makes every such fallback land on
        its own default.
        """
        answers: List[Any] = []
        for item in getattr(params, "items", None) or []:
            section = getattr(item, "section", None) or (
                item.get("section") if isinstance(item, dict) else None
            )
            if section in info.get("settings", {}):
                answers.append(info["settings"][section])
            else:
                answers.append(None)
        return answers

    @client.feature("client/registerCapability")
    def on_register_capability(params: Any) -> None:
        return None

    @client.feature("client/unregisterCapability")
    def on_unregister_capability(params: Any) -> None:
        return None

    @client.feature("window/workDoneProgress/create")
    def on_progress_create(params: Any) -> None:
        return None

    def on_log_message(params: Any) -> None:
        append_log(log_path, getattr(params, "message", params))

    client.feature("window/logMessage")(on_log_message)
    client.feature("$/logTrace")(on_log_message)
    # Vendor extensions we deliberately ignore (pygls logs an
    # "unknown method" warning for each otherwise).
    for ignored in (
        "$/progress",
        "$/setTrace",
        "$/log",
        "$/typescriptVersion",
        "language/status",
        "language/action",
        "textDocument/publishWarning",
        "textDocument/publishLink",
        "$/publishDiagnostics",
    ):
        with suppress(Exception):
            client.feature(ignored)(lambda params: None)

    # Volar's tsserver relay. It owns no tsserver: it asks *us* for the
    # answers a VS Code extension host would get from its tsserver child
    # process, and awaits them inside ``getLanguageService()``. Refusing
    # to answer is worse than answering wrongly — the awaited promise
    # never settles, so every request touching that file times out.
    #
    # The hub does not run a tsserver (that would double the RAM bill and
    # duplicate the ``typescript`` server). It answers the one command
    # that gates everything else, ``_vue:projectInfo``, with the
    # workspace's own tsconfig/jsconfig; Volar then builds its language
    # service itself and serves documentSymbol/hover/completion/diagnostics
    # from Vue's own semantic model. Commands it still asks for are
    # component-detail helpers (props/slots/directives/meta) — answered
    # ``null`` so their awaited promise settles and Vue degrades those
    # features instead of hanging the whole session.
    @client.feature(TSSERVER_REQUEST)
    def on_tsserver_request(params: Any) -> None:
        session = sessions.get(info["id"])
        if session is None:
            return
        parsed = parse_tsserver_request(params)
        if parsed is None:
            append_log(
                log_path, f"[hub] unrecognised {TSSERVER_REQUEST} params: {params!r}"
            )
            return
        request_id, command, args = parsed
        session.ts_relay_counts[command] = session.ts_relay_counts.get(command, 0) + 1
        if command == TSSERVER_PROJECT_INFO:
            file_name = ""
            if isinstance(args, dict) and isinstance(args.get("file"), str):
                file_name = args["file"]
            config_file = project_config_for(session, file_name)
            result: Any = tsserver_project_info(file_name, config_file)
            if config_file is None:
                # A null projectInfo is Volar's "no project found" signal:
                # it falls back to an inferred project that still gives
                # Vue semantics for a config-less workspace.
                append_log(
                    log_path,
                    f"[hub] {TSSERVER_PROJECT_INFO}: no "
                    + "/".join(PROJECT_CONFIG_NAMES)
                    + f" under {info['workspaceRoot']} — answering null "
                    "(Volar will use its inferred project)",
                )
        else:
            result = None
        if command != TSSERVER_PROJECT_INFO and session.ts_relay_counts[command] == 1:
            # Log each degraded command once. Without this, a caller who
            # gets "No hover information." out of lsp_vue_hover has no
            # way to learn that the answer came from the hub rather than
            # from a real tsserver — the relay's limitation is otherwise
            # invisible.
            append_log(
                log_path,
                f"[hub] {command}: answered null (the hub runs no tsserver, so "
                f"{command.removeprefix('_vue:')} is unavailable; the "
                "'typescript' server covers plain .js/.ts files)",
            )
        # pygls 2.x exposes no client-side send_notification: the raw
        # JSON-RPC layer is ``client.protocol.notify``. The method is not
        # in lsprotocol's METHOD_TO_TYPES, so it goes out verbatim as a
        # generic notification rather than being restructured.
        try:
            client.protocol.notify(
                TSSERVER_RESPONSE, tsserver_response_params(request_id, result)
            )
        except Exception as exc:  # noqa: BLE001 — a dead pipe must not kill the handler
            append_log(log_path, f"[hub] {TSSERVER_RESPONSE} failed: {exc}")
