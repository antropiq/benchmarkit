"""The Editor's write routes are reachable from the browser, and from nothing else.

AGENTS.md promises *"Tools are read-only by design"*. That promise survived the
guarded write surface only because something asserts it, so this file asserts it
five ways, each one a different way the promise could rot:

* **no advertised tool names a write route** — the MCP connectors' own tool
  listings and the LSP hub's tool catalogue, both built the way the hubs build
  them (the connectors answer ``list_tools`` themselves; the LSP catalogue is
  ``TOOL_SPECS`` through ``tools_for_server``, which is what a hub would
  publish for the servers in the committed template). With
  ``BENCHMARKIT_DISABLE_MCP``/``_LSP`` the live hubs start disconnected, so the
  lists here come from the sources of truth rather than a session.
* **no code path spells a call to one** — a source scan for the write markers
  (``PUT /api/fs``, ``POST /api/fs``, ``DELETE /api/fs``). They may appear in
  ``src/api/routes.py`` (the definition) and the Editor composables (the
  caller), and nowhere else. ``GET /api/fs`` is deliberately *not* a marker:
  the read surface's docstrings name their routes, and always will.
* **the model client cannot spell an app route either** — ``core/llmkit`` is
  the one core package that speaks HTTP, and *legally* so (the import scan
  covers the connectors, not ``src/core``), so its share of the promise is the
  *shape* of the URLs it builds: it appends two fixed suffixes under ``/v1``
  and can never produce a path under ``/api/``.
* **the terminal stays a human surface** — the shell is a WebSocket reached
  from the SPA, and neither hub mentions it, so it cannot become a tool by
  someone wiring the hub to ``app.state.terminal``.
* **the reserved folder keeps one reader per file, and it is a ``GET``** —
  ``.nojae`` is edited by hand outside the app, so its two purpose-built
  readers (the policy file's state, the agent-modes list) answer *about* a file
  and write nothing. A verb on ``/agent-modes`` would put the switch that
  decides whether tools exist behind a route, which no marker scan would see
  because that route is not an ``/api/fs`` route.

The last test feeds each check a deliberately poisoned input, because a guard
that cannot fail is not a guard: that is the same "assert the invariant, not
the intent" style as the file-icons sync test.
"""

import ast
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit

import pytest

from src.core import editorfs
from src.core.llmkit.client import chat_completions_url, models_url
from src.core.lsphub import tools as lsp_tools

REPO_ROOT = Path(editorfs.constants.PROJECT_ROOT)

# The three write routes, in the exact spelling a caller would use. Chosen so
# that a docstring naming a route *method* (``GET /api/fs/tree``) stays out of
# the scan, and a comment or a string naming a write is caught.
WRITE_MARKERS = ("PUT /api/fs", "POST /api/fs", "DELETE /api/fs")

# The app's own route prefix: a URL whose path holds it could address a write
# route. ``llmkit``'s URL builders are guarded on this below.
APP_ROUTE_PREFIX = "/api/"

# Where a write marker is legitimate.
ALLOWED_HOMES = {
    Path("src/api/routes.py"),          # the routes are defined here
    Path("web/src/composables/useEditorFiles.js"),   # save / delete
    Path("web/src/composables/useEditorCreate.js"),  # create
    Path("web/src/composables/useEditorRename.js"),  # rename
    # Declared with the route, before its caller exists: task mov-03 creates this
    # composable, whose ``fetch`` will name the batch-move route.
    Path("web/src/composables/useEditorMove.js"),    # move batch
}

# A connector that could speak HTTP could call a route it never names in a
# string, so the marker scan alone would miss it. These are the clients that
# would carry one: the top-level module name is enough (``urllib.parse`` builds
# URLs and reaches nothing, ``urllib.request`` does).
HTTP_CLIENTS: frozenset[str] = frozenset({
    "httpx", "requests", "urllib3", "aiohttp", "socket",
    "urllib.request",
})


def scan_roots() -> List[Path]:
    """The trees the marker scan walks: everything that is *not* a caller."""
    return [
        REPO_ROOT / "src" / "core",
        REPO_ROOT / "mcp_connectors",
        REPO_ROOT / "lsp_connectors",
        REPO_ROOT / "web" / "src" / "composables" / "useChat.js",
    ]


def offenders_for(markers, roots, allowed=ALLOWED_HOMES) -> List[str]:
    """Every file under ``roots`` carrying one of ``markers``, named relative to
    the repo. One implementation, shared by the guard and by its self-check
    below (a second copy of the walk is how a guard drifts from what it guards).
    """
    found: List[str] = []
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix not in (".py", ".js", ".json", ".txt"):
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if any(marker in text for marker in markers):
                label = (
                    str(path.relative_to(REPO_ROOT))
                    if path.is_relative_to(REPO_ROOT)
                    else str(path)
                )
                if Path(label) not in allowed:
                    found.append(label)
    return found


def http_import_offenders(roots: List[Path]) -> List[str]:
    """Every ``*.py`` under ``roots`` that imports an HTTP client, by file name.

    One implementation, used by the guard on the connector tree and by its
    self-check below — the same reason ``offenders_for`` is shared.
    """
    found: List[str] = []
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.glob("*.py"))
        for path in candidates:
            try:
                parsed = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            names: List[str] = []
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
            for name in names:
                if name in HTTP_CLIENTS or name.split(".")[0] in HTTP_CLIENTS:
                    found.append(f"{path.name}: {name}")
    return found


def route_registrations(source_text: str, path_prefix: str) -> List[Dict[str, Any]]:
    """Every route ``routes.py`` registers on a path under ``path_prefix``.

    Each answer is ``{"verb", "path", "handler"}``, read out of the decorator
    call (``@router.get("/agent-modes")``) rather than matched as a substring, so
    what is reported is the *registration*: a docstring naming a route cannot
    trip a guard built on this, and a new route cannot hide from one. Both the
    verb guard and the arity guard below share this one walk.
    """
    found: List[Dict[str, Any]] = []
    for node in ast.walk(ast.parse(source_text)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            if call is None or not isinstance(call.func, ast.Attribute):
                continue
            if not call.args or not isinstance(call.args[0], ast.Constant):
                continue
            route_path = call.args[0].value
            if isinstance(route_path, str) and route_path.startswith(path_prefix):
                found.append(
                    {"verb": call.func.attr, "path": route_path, "handler": node}
                )
    return found


def builder_paths(builder, bases: List[str]) -> List[str]:
    """The path of every URL ``builder`` produces from ``bases``.

    One implementation, shared by the two ``llmkit`` URL guards below and by
    their self-check — the same reason ``offenders_for`` is shared.
    """
    return [urlsplit(builder(base)).path for base in bases]


def connector_tool_lists() -> Dict[str, List[Dict[str, Any]]]:
    """The tool listing each local MCP connector advertises, by connector id.

    These are the scripts referenced as ``local`` entries in
    ``mcp-settings.json``, and ``list_tools`` is exactly what the hub asks them
    for — so this is the live surface, without needing a connected session.
    """
    from mcp_connectors import console_mcp, fedit_mcp

    listings = {}
    for connector_id, module in (("console", console_mcp), ("fedit", fedit_mcp)):
        tools = asyncio.run(module.server.list_tools())
        listings[connector_id] = [
            {"name": tool.name, "description": tool.description or "",
             "schema": json.dumps(tool.input_schema or {})}
            for tool in tools
        ]
        instructions = getattr(module.server, "instructions", "") or ""
        listings[connector_id].append(
            {"name": "<instructions>", "description": instructions, "schema": ""}
        )
    return listings


def lsp_tool_defs() -> List[Dict[str, Any]]:
    """The defs the hub would publish for the servers in the committed template.

    The declaration file itself is gitignored and machine-local, so the
    template is the portable source of truth for which tools a configured
    server asks for; ``TOOL_SPECS`` supplies their descriptions.
    """
    template = json.loads(
        (REPO_ROOT / "lsp_connectors" / "lsp_servers.example.json").read_text(
            encoding="utf-8"
        )
    )
    definitions: List[Dict[str, Any]] = []
    for server_id, entry in template.items():
        if not isinstance(entry, dict) or "command" not in entry:
            continue
        declared = {**entry, "id": server_id}
        # capabilities=None: consult the declared tool list, which is what the
        # hub does before a handshake (and what it can always fall back to).
        definitions.extend(lsp_tools.tools_for_server(declared, capabilities=None))
    return definitions


class TestNoToolNamesAWriteRoute:
    def test_no_mcp_tool_names_or_describes_a_write_route(self):
        listings = connector_tool_lists()
        assert set(listings) == {"console", "fedit"}
        for listing in listings.values():
            assert listing, "a connector with no tools publishes nothing to catch"
        for connector_id, listing in listings.items():
            for tool in listing:
                haystack = f"{tool['name']} {tool['description']} {tool['schema']}"
                for marker in WRITE_MARKERS:
                    assert marker not in haystack, f"{connector_id}: {tool['name']}"
                # Belt for the prose: no tool names a mutating editor helper.
                for helper in ("write_file", "create_path", "rename_path",
                               "trash_path", "delete_fs", "put_fs"):
                    assert helper not in tool["name"], f"{connector_id}: {helper}"

    def test_no_lsp_tool_names_or_describes_a_write_route(self):
        definitions = lsp_tool_defs()
        names = [d["function"]["name"] for d in definitions]
        assert names, "the LSP catalogue is the hub's whole tool surface"
        for definition in definitions:
            haystack = json.dumps(definition)
            for marker in WRITE_MARKERS:
                assert marker not in haystack, definition["function"]["name"]
        # Every published name comes from the catalogue — a name that is not
        # in TOOL_SPECS would have been added somewhere the reviewer never
        # looks.
        catalogue = {
            lsp_tools.tool_name_for(server, action)
            for server in ("pyright", "typescript", "vue")
            for action in lsp_tools.TOOL_SPECS
        }
        assert set(names) <= catalogue
        assert names, "the template declares the servers, so this cannot be empty"

    def test_every_lsp_method_is_a_query_not_a_document_mutation(self):
        """``TOOL_SPECS`` is the hub's whole surface, so this is the table."""
        forbidden = ("willSave", "didChange", "willCreate", "willRename",
                     "willDelete", "executeCommand", "applyEdit")
        for action, spec in lsp_tools.TOOL_SPECS.items():
            method = str(spec.get("method") or "")
            for word in forbidden:
                assert word not in method, f"{action} calls {method}"


class TestNoCodePathCallsThem:
    def test_the_write_markers_appear_only_in_the_routes_and_the_callers(self):
        offenders = offenders_for(WRITE_MARKERS, scan_roots())
        assert offenders == [], (
            "a write route is named outside the API layer and the Editor "
            f"composables: {offenders}"
        )

    def test_no_module_under_src_core_imports_the_http_layer(self):
        """The hubs and ``core/chat.py`` run in this process, so reaching a
        route would be an import away — the wall is that there is no import."""
        offenders = []
        for path in (REPO_ROOT / "src" / "core").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "from src.api" in text or "import src.api" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == []

    def test_the_write_core_is_called_from_the_routes_and_nowhere_else(self):
        writers = ("write_file", "backup_for_write", "create_path",
                   "rename_path", "move_paths_batch", "trash_path")
        callers: List[str] = []
        for path in (REPO_ROOT / "src").rglob("*.py"):
            if path.parent.name == "editorfs":
                continue  # these *define* the helpers
            text = path.read_text(encoding="utf-8")
            if any(name in text for name in writers):
                callers.append(str(path.relative_to(REPO_ROOT)))
        assert callers == [str(Path("src") / "api" / "routes.py")]


class TestTheConnectorsWorkspacePolicyChecksStayInTheirLane:
    """pol-08 gave the connectors a new reason to refuse a path — not a new route.

    The two checks (``fedit_mcp._workspace_policy_reason`` and
    ``console_mcp.workspace_policy_error``, both built on
    ``mcp_connectors.policy_file.workspace_policy_reason``) answer the same
    question the Editor pane answers, and the tempting shortcut is to ask the
    *app* — an HTTP call to a `/api/fs` route, which would hand a connector a
    write surface it has never had. Pinned three ways: the functions are still
    there (a rename that drops the check would otherwise pass every other test
    in this file), the shared loader reads a file rather than a socket, and no
    connector module can reach a route at all.
    """

    def test_the_three_new_functions_exist(self):
        from mcp_connectors import console_mcp, fedit_mcp, policy_file

        assert callable(fedit_mcp._workspace_policy_reason)
        assert callable(console_mcp.workspace_policy_error)
        assert callable(policy_file.workspace_policy_reason)

    def test_no_connector_module_imports_an_http_client(self):
        """A refusal path that cannot speak HTTP cannot reach a write route.

        The marker scan looks for a route *named in a string*; this one looks for
        the transport that could call a route without ever naming it. pol-08 is
        the moment that matters: the connectors now answer a question the app
        also answers, and "just ask the running app" is the shortcut that would
        turn a refusal path into a client of the write surface.
        """
        assert http_import_offenders([REPO_ROOT / "mcp_connectors"]) == []

    def test_the_verdict_comes_from_a_file_on_disk_not_from_a_transport(self):
        """The check answers locally: an unreadable root yields no verdict.

        A root with no policy file is the state a connector reaches when its
        workspace was never seeded, and the answer is "no rules" — a None, not a
        refusal fetched from the app. If the connector ever asked the running app
        instead, this is where it would show up.
        """
        from mcp_connectors import policy_file as connector_policy

        assert connector_policy.workspace_policy_reason(
            "web/vendor/x.js", root="/definitely-not-a-workspace", for_write=True,
        ) is None
        text = (REPO_ROOT / "mcp_connectors" / "policy_file.py").read_text(
            encoding="utf-8"
        )
        for marker in ("httpx", "requests.", "urllib.request", "socket"):
            assert marker not in text, marker


class TestTheTerminalStaysHumanOnly:
    def test_neither_hub_mentions_the_terminal(self):
        """The shell is one WebSocket per session, reached from the SPA."""
        offenders = []
        for folder in ("mcphub", "lsphub"):
            for path in (REPO_ROOT / "src" / "core" / folder).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for marker in ("termhub", "/api/terminal", "TerminalHub"):
                    if marker in text:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: {marker}")
        assert offenders == []

    def test_the_terminal_route_is_a_websocket_and_not_a_tool(self):
        routes = (REPO_ROOT / "src" / "api" / "routes.py").read_text(encoding="utf-8")
        assert '@router.websocket("/terminal")' in routes
        # Nothing outside termhub itself holds a session handle, so no hub or
        # chat path can drive a shell: the registry is reached from the SPA and
        # from the lifespan teardown only.
        assert offenders_for(
            ("app.state.terminal",),
            [
                REPO_ROOT / "src" / "core" / "mcphub",
                REPO_ROOT / "src" / "core" / "lsphub",
                REPO_ROOT / "src" / "core" / "chat.py",
                REPO_ROOT / "src" / "core" / "batch.py",
                REPO_ROOT / "src" / "core" / "llm.py",
            ],
        ) == []


class TestTheAgentModesReaderStaysReadOnly:
    """am-06 gave the reserved folder a second purpose-built reader: a *GET*.

    ``GET /api/agent-modes`` answers *about* ``.nojae/agent-modes.json`` the way
    ``GET /api/path-location`` answers about the policy file — one small reader,
    no ``?path=``, no raw bytes, and nothing writable. The modes file is edited
    by hand outside the app (D6), so a write route for it would not just break
    the reservation: it would hand a tool the switch that decides whether tools
    exist. The guard is on the *verb* rather than on the marker scan, because a
    route named ``/agent-modes`` is not a route named ``/api/fs``.
    """

    def test_no_agent_modes_route_accepts_a_mutation(self):
        routes_source = (REPO_ROOT / "src" / "api" / "routes.py").read_text(
            encoding="utf-8"
        )
        registered = route_registrations(routes_source, "/agent-modes")
        assert registered, "am-06's reader has to exist for this to mean anything"
        assert [entry["verb"] for entry in registered] == ["get"], (
            "a verb on /agent-modes would put the file that decides whether "
            "tools exist behind a route"
        )

    def test_the_reader_is_a_get_and_carries_no_path_parameter(self):
        """The one route, exactly as am-06 §3-§4 specify it: no argument at all.

        A ``?path=`` query parameter is how a purpose-built reader turns into a
        general way to reach the reserved folder, so what is pinned is the
        handler's *arity* — not the absence of a string somewhere in the file.
        """
        routes_source = (REPO_ROOT / "src" / "api" / "routes.py").read_text(
            encoding="utf-8"
        )
        handlers = route_registrations(routes_source, "/agent-modes")
        assert [entry["handler"].name for entry in handlers] == ["get_agent_modes"]
        assert handlers[0]["handler"].args.args == []


class TestTheSearchReaderStaysARead:
    """The same verb guard as ``/agent-modes``, for the surface it does not cover.

    ``test_editorfs.py``'s verb map already lists ``/api/fs/search`` as ``["get"]``;
    this is the sibling of the ``/agent-modes`` guard — the *verb* read out of the
    AST, so a mutation added to the same path trips it even before the OpenAPI map
    is consulted.
    """

    def test_no_fs_search_route_accepts_a_mutation(self):
        routes_source = (REPO_ROOT / "src" / "api" / "routes.py").read_text(
            encoding="utf-8"
        )
        registered = route_registrations(routes_source, "/fs/search")
        assert registered, "the search route has to exist for this to mean anything"
        assert [entry["verb"] for entry in registered] == ["get"]


class TestTheModelClientCannotReachAWriteRoute:
    """``core/llmkit`` is the one core package that speaks HTTP, and legally.

    The import scan above covers ``mcp_connectors/`` only, so ``httpx`` under
    ``llmkit`` is *legal* — widening that scan to ``src/core`` would condemn
    the app's own model client, a different decision. And the passive marker
    scan over ``src/core`` already means ``llmkit``'s sources never spell a
    ``PUT /api/fs``. What neither scan answers is the URL *shape*: an HTTP
    client that builds its own URLs could reach a route it never names in a
    string. The answer pinned here is that the two builders append only their
    two fixed suffixes, so no configured ``SERVER_URL`` turns the model client
    into a caller of an app route.
    """

    # Every shape ``SERVER_URL`` actually takes (D9): with and without the
    # ``/v1`` tail, with and without a trailing slash.
    SERVER_URL_SHAPES = [
        "http://host:8080/v1",
        "http://host:8080/v1/",
        "http://host:8080",
        "http://host:8080/",
    ]

    # The two suffixes the client may ever append, keyed by the builder that
    # owns each one.
    BUILDER_SUFFIXES = {
        chat_completions_url: "/v1/chat/completions",
        models_url: "/v1/models",
    }

    def test_each_builder_appends_only_its_one_fixed_suffix(self):
        for builder, suffix in self.BUILDER_SUFFIXES.items():
            paths = builder_paths(builder, self.SERVER_URL_SHAPES)
            assert paths and all(path.endswith(suffix) for path in paths), (
                f"{builder.__name__} appends more than {suffix}: {paths}"
            )

    def test_no_builder_path_reaches_under_api(self):
        """The invariant spec step 6 states: the model client cannot write-route.

        ``/api/`` is the app's own route prefix, so a builder whose path holds
        it could address a write route. It cannot: each appends its suffix to
        the ``/v1`` root of ``SERVER_URL``. A hostile ``SERVER_URL`` that itself
        embeds ``/api/fs`` still yields a chat-completions *POST* under that
        path, never a ``PUT /api/fs`` — the suffix lock keeps the shape honest
        even then, which is why this guard pins the builders rather than
        policing ``SERVER_URL``, which is user config.
        """
        for builder in self.BUILDER_SUFFIXES:
            paths = builder_paths(builder, self.SERVER_URL_SHAPES)
            assert not any(APP_ROUTE_PREFIX in path for path in paths), (
                f"{builder.__name__} reaches {APP_ROUTE_PREFIX}: {paths}"
            )

    def test_the_shape_guard_catches_a_builder_drifted_onto_an_app_route(self):
        """Both halves trip on a builder that drifts onto an app route.

        Fed to the same walker the guards use (a scratch builder, not an edit
        to ``client.py``), so what is checked is that the *guard* sees the
        drift — the ``TestTheGuardsActuallyFail`` pattern, applied to these.
        """
        def drifted(server_url: str) -> str:
            return f"{server_url.rstrip('/')}/api/fs/file"

        paths = builder_paths(drifted, ["http://host:8080/v1"])
        with pytest.raises(AssertionError):
            assert all(path.endswith("/v1/chat/completions") for path in paths), paths
        with pytest.raises(AssertionError):
            assert not any(APP_ROUTE_PREFIX in path for path in paths), paths

    def test_the_marker_scan_still_catches_llmkit_spelling_a_write(self, tmp_path):
        """The passive scan's own teeth, shown on a scratch copy of the package.

        ``llmkit`` lives under the roots ``scan_roots()`` walks, so a source
        file there naming a write route is an offender like any other, and no
        allowed home excuses it; the poisoned module below proves the shared
        walker sees it (reusing ``offenders_for`` — a second copy of the walk
        is how a guard drifts).
        """
        fake = tmp_path / "llmkit" / "cheeky.py"
        fake.parent.mkdir(parents=True)
        fake.write_text(
            'def leak(path, text):  # a model client wired to PUT /api/fs/file\n'
            '    return client.put("/api/fs/file", json={"path": path, "text": text})\n',
            encoding="utf-8",
        )
        assert offenders_for(WRITE_MARKERS, [fake.parent]) == [str(fake)]


class TestTheGuardsActuallyFail:
    """A guard that cannot fail is decoration, so each is shown catching one.

    Same purpose as the acceptance's ``verify locally by exposing a write route
    to a fixture tool and by calling one from src/core, then revert`` — done as
    a test instead of as a note, so the check is repeated on every run rather
    than remembered.
    """

    def test_the_marker_scan_catches_a_poisoned_module(self, tmp_path):
        fake = tmp_path / "core" / "cheeky.py"
        fake.parent.mkdir(parents=True)
        fake.write_text(
            'def leak(path, text):  # a tool wired to PUT /api/fs/file\n'
            '    return http.put("/api/fs/file", json={"path": path, "text": text})\n',
            encoding="utf-8",
        )
        assert offenders_for(WRITE_MARKERS, [fake.parent]) == [str(fake)]

    def test_the_marker_scan_catches_a_poisoned_connector_script(self, tmp_path):
        fake = tmp_path / "some_connector.py"
        fake.write_text('URL = "DELETE /api/fs/file"\n', encoding="utf-8")
        assert offenders_for(WRITE_MARKERS, [tmp_path]) == [str(fake)]

    def test_the_tool_scan_catches_a_poisoned_tool_list(self):
        poisoned = [{"name": "fs_delete", "description": "POST /api/fs/file",
                     "schema": "{}"}]
        haystack = " ".join(
            f"{tool['name']} {tool['description']} {tool['schema']}"
            for tool in poisoned
        )
        assert any(marker in haystack for marker in WRITE_MARKERS)

    def test_the_http_import_scan_catches_a_poisoned_connector(self, tmp_path):
        """The new guard sees an HTTP client, and the marker scan sees the route."""
        fake = tmp_path / "cheeky_connector.py"
        fake.write_text(
            "import httpx\n\n"
            "def save(rel, text):  # a refusal path wired to PUT /api/fs\n"
            "    return httpx.put('http://127.0.0.1:8000/api/fs/file',\n"
            "                   json={'path': rel, 'text': text})\n",
            encoding="utf-8",
        )
        assert http_import_offenders([tmp_path]) == [f"{fake.name}: httpx"]
        assert offenders_for(WRITE_MARKERS, [tmp_path], allowed=set()) == [str(fake)]

    def test_the_lsp_method_guard_catches_a_write_shaped_spec(self, monkeypatch):
        spec = {
            "capability": None,
            "method": "textDocument/willSaveWaitUntil",
            "schema": {"type": "object"},
            "description": "Rewrite the document.",
        }
        forbidden = ("willSave", "didChange", "willCreate", "willRename",
                     "willDelete", "executeCommand", "applyEdit")
        assert any(word in str(spec["method"]) for word in forbidden)

        # And through the real guard: a poisoned catalogue makes the test above
        # fail, which is what makes it a guard rather than a comment.
        monkeypatch.setitem(lsp_tools.TOOL_SPECS, "poison", spec)
        with pytest.raises(AssertionError):
            for action, entry in lsp_tools.TOOL_SPECS.items():
                method = str(entry.get("method") or "")
                for word in forbidden:
                    assert word not in method, f"{action} calls {method}"

    def test_the_agent_modes_verb_guard_catches_a_poisoned_route(self):
        """One added ``@router.post`` on the reader's path trips the guard.

        Fed to the same walk the guard uses (a string, not an edit to the real
        file), so the check is that the *guard* sees a mutation — which is what
        makes the assertion in the class above a guard rather than a comment.
        """
        poisoned = (
            "@router.get(\"/agent-modes\")\n"
            "def get_agent_modes():\n"
            "    return {}\n\n"
            "@router.post(\"/agent-modes\")\n"
            "def post_agent_modes(body: ModesIn):\n"
            "    return {}\n"
        )
        registered = route_registrations(poisoned, "/agent-modes")
        assert [entry["verb"] for entry in registered] == ["get", "post"]
        with pytest.raises(AssertionError):
            assert [entry["verb"] for entry in registered] == ["get"]
        # And the arity guard sees the body argument the poisoned route needs.
        assert registered[1]["handler"].args.args != []
