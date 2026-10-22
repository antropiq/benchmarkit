"""A ~100-line stdlib language server used by the LSP hub tests.

Speaks just enough LSP over stdio JSON-RPC to exercise every part of the
hub: the ``initialize`` handshake, ``workspace/configuration`` (a
server->client request the hub must answer), ``didOpen``/``didChange`` with
versioned documents, a ``publishDiagnostics`` push, and the
``hover``/``definition``/``documentSymbol`` requests. It also exercises the
failure paths the hub has to survive: a deliberate crash mode
(``--crash-after``) and a deliberate init failure (``--fail-init``).

``--relay`` switches it into Volar mode: it asks its client for tsserver
answers over the private ``tsserver/request`` protocol and blocks every
per-document request until the answer arrives, which is how the hub's
tsserver relay is tested.

Run with ``sys.executable fake_langserver.py [flags]``.
"""

import json
import sys

CAPABILITIES = {
    "hoverProvider": True,
    "definitionProvider": True,
    "documentSymbolProvider": True,
    "textDocumentSync": 1,  # full
}

# Opt-in Volar mode (``--relay``): the fake behaves like vue-language-server
# and asks its client for tsserver answers over 'tsserver/request', refusing
# to answer any textDocument/* until the matching 'tsserver/response' has
# arrived. That hang is the exact bug the hub's relay fixes, and without this
# mode no test in the suite can observe it. ``--relay-unknown`` additionally
# sends a malformed request, to prove the hub survives an unparsable payload.
PROJECT_INFO = "_vue:projectInfo"
COMPONENT_PROPS = "_vue:getComponentProps"

VOLAR_CAPABILITIES = {
    "documentSymbolProvider": True,
    "hoverProvider": True,
    "textDocumentSync": 1,
}


def main() -> int:
    fail_init = "--fail-init" in sys.argv
    relay_mode = "--relay" in sys.argv
    relay_unknown = "--relay-unknown" in sys.argv
    crash_after = None
    for index, arg in enumerate(sys.argv):
        if arg == "--crash-after" and index + 1 < len(sys.argv):
            crash_after = int(sys.argv[index + 1])

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    opened = {}
    request_id = 0
    request_count = 0
    # Volar's own numbering: an int counter, answered out-of-band.
    ts_request_id = 0
    ts_answered = {}          # relay id -> the answer the client sent back
    ts_pending = []           # ids asked for but not answered yet
    project_ready = set()     # uris whose projectInfo the client answered
    project_answers = {}      # uri -> the projectInfo answer for it
    pending_project = {}      # relay id -> uri waiting on that answer
    queued_symbols = []       # documentSymbol requests blocked on a project
    texts = {}                # uri -> current text (for a deferred publish)
    configuration_ok = []     # set once a workspace/configuration answer survived

    def ask_client(command, args):
        """Send one 'tsserver/request' exactly the way Volar does.

        The single *array* argument is what makes vscode-jsonrpc wrap it
        again: on the wire the params end up as ``[[id, command, args]]``.
        """
        nonlocal ts_request_id
        ts_request_id += 1
        ts_pending.append(ts_request_id)
        send({"jsonrpc": "2.0", "method": "tsserver/request",
              "params": [[ts_request_id, command, args]]})
        return ts_request_id


    def send(message):
        body = json.dumps(message).encode("utf-8")
        stdout.write(b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        stdout.flush()

    def reply(msg_id, result):
        send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def notify(method, params):
        nonlocal request_id
        send({"jsonrpc": "2.0", "method": method, "params": params})

    def publish(uri, version, text):
        """Push diagnostics for one document: one per line containing 'BUG'."""
        diagnostics = []
        for line_no, line in enumerate(text.splitlines()):
            if "BUG" in line:
                diagnostics.append(
                    {
                        "range": {
                            "start": {"line": line_no, "character": line.index("BUG")},
                            "end": {"line": line_no, "character": line.index("BUG") + 3},
                        },
                        "severity": 1,
                        "message": "found a BUG here",
                        "code": "fake001",
                        "source": "fakeserver",
                    }
                )
        notify(
            "textDocument/publishDiagnostics",
            {"uri": uri, "version": version, "diagnostics": diagnostics},
        )

    def ask_project(uri):
        """Ask the client which project a document belongs to (Volar's gate).

        Returns the relay id. Until it is answered, this document is stuck:
        that is the real Volar behaviour reproduced faithfully.
        """
        relay_id = ask_client(PROJECT_INFO, {"file": uri, "needFileNameList": False})
        pending_project[relay_id] = uri
        return relay_id

    def flush_document(uri):
        """Answer whatever this document's blocked requests were waiting for."""
        if uri not in project_ready:
            return
        publish(uri, opened.get(uri), texts.get(uri, ""))
        answer = project_answers.get(uri)
        for queued in [q for q in queued_symbols if q["uri"] == uri]:
            reply(
                queued["id"],
                [
                    {
                        "name": "Thing",
                        "kind": 5,
                        "range": {
                            "start": {"line": 4, "character": 0},
                            "end": {"line": 8, "character": 0},
                        },
                        "selectionRange": {
                            "start": {"line": 4, "character": 0},
                            "end": {"line": 4, "character": 5},
                        },
                    },
                    # Proof that the hub's projectInfo answer arrived intact:
                    # a symbol named after the config file it pointed at.
                    {
                        "name": "config:" + str((answer or {}).get("configFileName")
                                                if isinstance(answer, dict) else answer),
                        "kind": 5,
                        "range": {
                            "start": {"line": 9, "character": 0},
                            "end": {"line": 10, "character": 0},
                        },
                        "selectionRange": {
                            "start": {"line": 9, "character": 0},
                            "end": {"line": 9, "character": 6},
                        },
                    },
                ],
            )
            queued_symbols.remove(queued)

    def read_exact(count):
        """Pipes short-read: loop until ``count`` bytes arrived or EOF."""
        chunks = []
        while count:
            chunk = stdin.read(count)
            if not chunk:
                break
            chunks.append(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    while True:
        length = None
        while True:
            header = stdin.readline()
            if not header:
                return 0
            if header in (b"\r\n", b"\n"):
                break  # end of the header block
            if header.startswith(b"Content-Length:"):
                length = int(header.split(b":")[1].strip())
        if length is None:
            continue
        raw = read_exact(length)
        if not raw:
            return 0
        message = json.loads(raw)
        method = message.get("method")
        params = message.get("params") or {}
        msg_id = message.get("id")

        if method == "tsserver/response":
            # Exactly Volar's handler: onNotification('tsserver/response',
            # ([id, res]) => …). It destructures params, so a client that
            # sends the pair flat (params=[id, res]) hands it an int and the
            # answer is silently dropped. Being as strict as the real server
            # is what turns a wrong hub shape into a failing test instead of
            # a pass.
            if not (isinstance(params, list) and len(params) == 1
                    and isinstance(params[0], list) and len(params[0]) >= 2):
                continue
            request_id_, answer = params[0][0], params[0][1]
            ts_answered[request_id_] = answer
            uri = pending_project.pop(request_id_, None)
            if uri is not None:
                project_ready.add(uri)
                project_answers[uri] = answer
                flush_document(uri)
            continue

        if relay_mode and method is None and "result" in message:
            # A reply to one of THIS server's own requests (the fake asked
            # workspace/configuration on 'initialized'). In relay mode the
            # fake then does what Volar does with it:
            #   const paths = await getConfiguration('html.customData') ?? [];
            #   for (const path of paths) …
            # In JavaScript only arrays (and other iterables) work in a
            # for..of, so a plain object answer is fatal there even though a
            # Python dict would iterate fine — hence the explicit type check
            # rather than a try/except around a loop.
            answer = (message.get("result") or [None])[0]
            if answer is not None and not isinstance(answer, list):
                sys.stderr.write(
                    f"configuration answer not iterable: {type(answer).__name__} "
                    f"is not an array\n"
                )
                sys.stderr.flush()
                return 4
            configuration_ok.append(True)
            continue

        if msg_id is not None and method not in ("initialize", "shutdown", "exit"):
            request_count += 1
            if crash_after is not None and request_count >= crash_after:
                # Die mid-conversation (never answering request N): the hub
                # must surface this as a status and fail the pending call.
                sys.stderr.write("fake server crashing\n")
                sys.stderr.flush()
                return 3

        if method == "initialize":
            if fail_init:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32001,
                            "message": "Could not find a valid TypeScript installation",
                        },
                    }
                )
                continue
            reply(
                msg_id,
                {
                    "capabilities": VOLAR_CAPABILITIES if relay_mode else CAPABILITIES,
                    "serverInfo": {"name": "fake", "version": "1"},
                },
            )
            continue
        if method == "initialized":
            # Server->client request: the hub must answer it or a real server
            # would stall. Deliberately noisy, like tsserver.
            request_id += 1
            send(
                {
                    "jsonrpc": "2.0",
                    "id": f"s{request_id}",
                    "method": "workspace/configuration",
                    "params": {"items": [{"scopeUri": None, "section": "fake"}]},
                }
            )
            continue
        if method == "textDocument/didOpen":
            document = params["textDocument"]
            opened[document["uri"]] = document["version"]
            texts[document["uri"]] = document["text"]
            if relay_mode:
                # Volar cannot analyse a .vue file until its client answers
                # which project it belongs to: nothing is published yet.
                ask_project(document["uri"])
                if relay_unknown:
                    send({"jsonrpc": "2.0", "method": "tsserver/request",
                          "params": ["not-an-int", 7]})
                continue
            publish(document["uri"], document["version"], document["text"])
            continue
        if method == "textDocument/didChange":
            # Like a real server: re-analyse and re-publish for the new
            # version, so the hub's wait-for-push step always resolves.
            document = params["textDocument"]
            opened[document["uri"]] = document["version"]
            texts[document["uri"]] = params["contentChanges"][0]["text"]
            if relay_mode:
                if document["uri"] in project_ready:
                    flush_document(document["uri"])
                continue
            publish(document["uri"], document["version"],
                    params["contentChanges"][0]["text"])
            continue
        if method == "textDocument/hover":
            position = params["position"]
            reply(
                msg_id,
                {
                    "contents": {"kind": "markdown", "value": f"hover at {position['line']}"},
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                },
            )
            continue
        if method == "textDocument/definition":
            reply(
                msg_id,
                [
                    {
                        "uri": params["textDocument"]["uri"],
                        "range": {
                            "start": {"line": 2, "character": 0},
                            "end": {"line": 2, "character": 3},
                        },
                    }
                ],
            )
            continue
        if method == "textDocument/documentSymbol":
            uri = params["textDocument"]["uri"]
            if relay_mode and uri not in project_ready:
                # The await that never settles in real Volar: the request is
                # never answered until the projectInfo relay question is.
                queued_symbols.append({"id": msg_id, "uri": uri})
                continue
            reply(
                msg_id,
                [
                    {
                        "name": "Thing",
                        "kind": 5,
                        "range": {
                            "start": {"line": 4, "character": 0},
                            "end": {"line": 8, "character": 0},
                        },
                        "selectionRange": {
                            "start": {"line": 4, "character": 0},
                            "end": {"line": 4, "character": 5},
                        },
                    }
                ],
            )
            continue
        if method == "shutdown":
            reply(msg_id, None)
            continue
        if method == "exit":
            return 0
        if msg_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"unknown method {method}"},
                }
            )


if __name__ == "__main__":
    sys.exit(main())
