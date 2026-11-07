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
  lists here come from the sources of truth rather than a session. The same
  lists are asked a second question (eds-04): neither carries the modes file's
  save route nor a helper spelling of it, because that file decides whether
  tools exist at all — a tool reaching it is the one leak no amount of
  read-only tool design would notice.
* **no code path spells a call to one** — a source scan for the write markers
  (``PUT /api/fs``, ``POST /api/fs``, ``DELETE /api/fs`` and the modes file's
  ``POST /api/agent-modes/save``). They may appear in ``src/api/routes.py``
  (the definition) and the composables that call them, and nowhere else. The
  read spellings (``GET /api/fs``, ``GET /api/agent-modes/document``) are
  deliberately *not* markers: the read surface's docstrings name their routes,
  and always will.
* **the model client cannot spell an app route either** — ``core/llmkit`` is
  the one core package that speaks HTTP, and *legally* so (the import scan
  covers the connectors, not ``src/core``), so its share of the promise is the
  *shape* of the URLs it builds: it appends two fixed suffixes under ``/v1``
  and can never produce a path under ``/api/``.
* **the terminal stays a human surface** — the shell is a WebSocket reached
  from the SPA, and neither hub mentions it, so it cannot become a tool by
  someone wiring the hub to ``app.state.terminal``.
* **the modes file's routes are pinned per path, and only the browser's half
  of them writes** — ``.nojae`` stays invisible to ``/api/fs/*``, so its files
  are reached through purpose-built routes rather than a path parameter (none of
  them is an ``/api/fs`` route). The dropdown's reader answers *about* the modes
  file with one verb and no argument; the editor's pair is exactly two routes,
  ``document`` a ``GET`` and ``save`` a ``POST``, and the writer behind the save
  is reached from the routes alone. What must never happen is a **tool** reaching
  that save: it is the switch that decides whether tools exist at all.

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

# The write routes, in the exact spelling a caller would use. Chosen so that a
# docstring naming a route *method* (``GET /api/fs/tree``) stays out of the
# scan, and a comment or a string naming a write is caught. The modes pair's
# contribution (eds-04) is its save route only: the verb-and-path shape keeps
# ``GET /api/agent-modes/document`` — which the document read's own docstrings
# name, and will keep naming — out of the scan for the same reason ``GET
# /api/fs`` is.
WRITE_MARKERS = (
    "PUT /api/fs",
    "POST /api/fs",
    "DELETE /api/fs",
    "POST /api/agent-modes/save",
)

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
    # Same spirit, same reason (eds-04 declares it with the route): task eds-05
    # gives this composable the document fetch and the save ``fetch``.
    Path("web/src/composables/useAgentModes.js"),    # modes document / save
}

# The modes file's save route on its own, for the checks that ask only about it
# (the tool listings, the poisoned cases): the marker scan above covers it too,
# but a tool list is judged against *this* path and nothing else.
MODES_SAVE_MARKER = "POST /api/agent-modes/save"

# Spellings a connector or a tool description could use for reaching it without
# naming the route verbatim — the same "belt for the prose" the helper names are
# for the Editor's writes.
MODES_SAVE_HELPER_SPELLINGS = (
    "agent_modes_save",
    "agent-modes-save",
    "modes_save",
    "modes-save",
    "save_agent_modes",
    "agent-modes/save",
)

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


def modes_writer_callers(source_root: Path) -> List[str]:
    """Every file under ``src`` that names the core writer of the modes file.

    One implementation, shared by the guard below and by its self-check — the
    same reason ``offenders_for`` is shared. The writer's own module is skipped
    (it *defines* the entry point), so what is left is the list of callers: the
    routes alone, or something that should not be there.
    """
    found: List[str] = []
    for path in source_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name == "agent_modes_write.py":
            continue  # this *defines* the writer
        text = path.read_text(encoding="utf-8")
        if "save_agent_modes_document" in text:
            found.append(str(path.relative_to(source_root)))
    return sorted(found)


def advertisement_offenders(listing: List[Dict[str, Any]]) -> List[str]:
    """Every advertised tool def that names the modes file's save route.

    One implementation, shared by the MCP and LSP advertisement guards below and
    by their self-checks — the same reason ``offenders_for`` is shared: a second
    copy of the check is how a guard drifts from what it guards. Each entry is
    judged on its whole published surface (name, description, schema), which is
    what a client is actually told; the LSP defs are flattened to the same shape
    by the caller.
    """
    found: List[str] = []
    for tool in listing:
        haystack = f"{tool['name']} {tool['description']} {tool['schema']}"
        if MODES_SAVE_MARKER in haystack:
            found.append(f"{tool['name']}: {MODES_SAVE_MARKER}")
            continue
        for spelling in MODES_SAVE_HELPER_SPELLINGS:
            if spelling in haystack:
                found.append(f"{tool['name']}: {spelling}")
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

    def test_no_mcp_tool_advertises_the_modes_save(self):
        """eds-04: the save route is absent from every advertised connector surface.

        The marker scan over the connectors' *sources* cannot see this: a tool that
        *describes* a way to save the modes file would be a hub publishing a write
        route, and the file it writes is the one that decides whether the hub's
        tools exist at all. The listing here is ``list_tools`` plus the server's
        own ``instructions`` text — the whole thing a connected client is told.
        """
        offenders: List[str] = []
        for connector_id, listing in connector_tool_lists().items():
            offenders.extend(f"{connector_id}: {item}"
                             for item in advertisement_offenders(listing))
        assert offenders == [], (
            f"a connector advertises the modes file's save route: {offenders}"
        )

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

    def test_no_lsp_tool_advertises_the_modes_save(self):
        """eds-04: the hub's published defs carry neither the save path nor a
        spelling of it.

        The LSP hub is the other publisher of tool defs the chat hands the model,
        so it gets the same question as the MCP connectors: does anything it
        advertises name the modes file's write route. The defs are built the way
        the hub builds them (``TOOL_SPECS`` through ``tools_for_server``,
        capabilities ``None``), so this is the catalogue a server would be given.
        """
        listing = [
            {"name": definition["function"]["name"],
             "description": definition["function"]["description"],
             "schema": json.dumps(definition["function"]["parameters"])}
            for definition in lsp_tool_defs()
        ]
        assert listing, "the hub publishes nothing to check"
        offenders = advertisement_offenders(listing)
        assert offenders == [], (
            f"an LSP tool def names the modes file's save route: {offenders}"
        )

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

    def test_the_modes_save_is_not_a_side_door_through_the_shell(self):
        """eds-04: a reserved-folder file became app-writable; the shell is not how.

        The modes file is now a store the app writes, and the tempting route for a
        tool to reach it is the one surface that can *edit any file* — the Editor's
        terminal. It stays a human surface: no hub, chat or batch path names the
        save route or the writer behind it, and the connectors' own sources carry
        neither (the marker scan covers those trees, so this asks only the in-process
        paths that could import the writer an import away).
        """
        markers = (MODES_SAVE_MARKER, "agent_modes_write", "save_agent_modes_document")
        assert offenders_for(
            markers,
            [
                REPO_ROOT / "src" / "core" / "mcphub",
                REPO_ROOT / "src" / "core" / "lsphub",
                REPO_ROOT / "src" / "core" / "chat.py",
                REPO_ROOT / "src" / "core" / "batch.py",
                REPO_ROOT / "src" / "core" / "llm.py",
                REPO_ROOT / "src" / "core" / "termhub",
            ],
        ) == []


class TestTheAgentModesRoutesStayPinnedToTheirPaths:
    """The modes file's routes: one argument-free reader, one pinned editor pair.

    am-06 gave the reserved folder a purpose-built *reader* and this lot gave it
    an editor pair. The reservation of ``.nojae`` was not lifted — the file is
    still invisible to ``/api/fs/*`` — so every route that reaches it is one this
    guard has to name, because none of them is an ``/api/fs`` route and the marker
    scan above cannot see any of them.

    What survives am-06's reason is the narrower truth it was always guarding: a
    **tool** must not reach the switch that decides whether tools exist. The modes
    file is now a store the app writes, but through the browser only, so what is
    pinned is (a) the dropdown's reader keeps exactly one verb and zero arguments,
    (b) the editor pair is exactly two routes with exactly their verbs, and (c) the
    core writer behind the save has one caller, the routes.
    """

    #: The whole surface ``/agent-modes`` may ever answer, path by path. A third
    #: route — or a second verb on one of these — is a change to the modes file's
    #: reach, so it has to be made here, on purpose.
    AGENT_MODES_VERBS = {
        "/agent-modes": ["get"],
        "/agent-modes/document": ["get"],
        "/agent-modes/save": ["post"],
    }

    def _registered(self):
        routes_source = (REPO_ROOT / "src" / "api" / "routes.py").read_text(
            encoding="utf-8"
        )
        return route_registrations(routes_source, "/agent-modes")

    def test_the_reader_path_has_one_verb_and_no_argument(self):
        """am-06 §3-§4, unchanged by this lot: the dropdown's read is untouched.

        The arity is what is pinned rather than "no ``?path=`` string in the
        file": a query parameter is how a purpose-built reader turns into a
        general way to reach the reserved folder, and a handler's signature is
        the one place that cannot hide one.
        """
        handlers = [
            entry for entry in self._registered() if entry["path"] == "/agent-modes"
        ]
        assert [entry["verb"] for entry in handlers] == ["get"], (
            "a verb on the dropdown's reader path would put the file that "
            "decides whether tools exist behind a route no marker scan sees"
        )
        assert [entry["handler"].name for entry in handlers] == ["get_agent_modes"]
        assert handlers[0]["handler"].args.args == []

    def test_the_editor_pair_exists_and_is_pinned_to_get_and_post(self):
        """The pair is the modes file's whole writable reach, verbs and all."""
        registered = self._registered()
        by_path: Dict[str, List[str]] = {}
        for entry in registered:
            by_path.setdefault(entry["path"], []).append(entry["verb"])
        assert by_path == self.AGENT_MODES_VERBS, (
            f"the /agent-modes surface is {by_path}, expected "
            f"{self.AGENT_MODES_VERBS}"
        )

    def test_the_save_is_the_only_writer_and_the_routes_are_its_only_caller(self):
        """The core writer behind the save is reached from the routes alone.

        Same shape as ``test_the_write_core_is_called_from_the_routes_and_nowhere_else``
        above, applied to this pair: a hub, a tool or a chat path that imported the
        writer would be a tool editing the file that decides whether tools exist,
        and no scan of ``/api/fs`` markers would ever notice.
        """
        assert modes_writer_callers(REPO_ROOT / "src") == [
            str(Path("api") / "routes.py")
        ]


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
        The drifted spelling names the **modes save** (eds-04): the guard is
        shown to see *this* path, not merely some app path, so a builder that
        drifts onto the one route that decides whether tools exist trips both
        halves exactly like any other write route.
        """
        def drifted(server_url: str) -> str:
            return f"{server_url.rstrip('/')}/api/agent-modes/save"

        paths = builder_paths(drifted, ["http://host:8080/v1"])
        with pytest.raises(AssertionError):
            assert all(path.endswith("/v1/chat/completions") for path in paths), paths
        with pytest.raises(AssertionError):
            assert not any(APP_ROUTE_PREFIX in path for path in paths), paths
        assert any(MODES_SAVE_MARKER.split(" ", 1)[1] in path for path in paths), (
            "the drift named is the modes save itself: the shape guard is shown "
            "to see this route, not just some app route"
        )

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

    def test_the_marker_scan_catches_a_module_naming_the_modes_save(self, tmp_path):
        """The new marker has its own teeth: the scan sees *this* route spelled.

        A scratch file under a scratch root spelling the save the way a caller
        would — the same walker the real guard runs over ``src/core``, the two
        connector trees and ``useChat.js``. The allowed-home entry for
        ``useAgentModes.js`` is *not* passed here (``allowed=set()``), because
        what is being proven is that the marker itself is caught, not that a
        declared caller is excused.
        """
        fake = tmp_path / "core" / "cheeky.py"
        fake.parent.mkdir(parents=True)
        fake.write_text(
            'def leak(document, mtime):\n'
            '    # a core path wired to POST /api/agent-modes/save\n'
            '    return client.post("/api/agent-modes/save",\n'
            '                       json={"document": document, "mtime": mtime})\n',
            encoding="utf-8",
        )
        assert offenders_for(WRITE_MARKERS, [fake.parent], allowed=set()) == [str(fake)]
        # And the *read* spelling of the pair stays out of the scan, which is what
        # keeps every document-read docstring from becoming an offender: the same
        # reason ``GET /api/fs`` is not a marker.
        read = tmp_path / "core" / "reader.py"
        read.write_text(
            '"""The editor read (``GET /api/agent-modes/document``)."""\n',
            encoding="utf-8",
        )
        assert offenders_for(WRITE_MARKERS, [tmp_path], allowed=set()) == [str(fake)]

    def test_the_tool_scan_catches_a_poisoned_tool_list(self):
        poisoned = [{"name": "fs_delete", "description": "POST /api/fs/file",
                     "schema": "{}"}]
        haystack = " ".join(
            f"{tool['name']} {tool['description']} {tool['schema']}"
            for tool in poisoned
        )
        assert any(marker in haystack for marker in WRITE_MARKERS)

    def test_the_modes_save_tool_guard_catches_a_poisoned_connector_list(self):
        """A connector advertising the modes save trips the advertisement guard.

        Fed to the same walker the guards use (a scratch listing, not an edit to
        a connector), so what is checked is that the *guard* sees this route:
        first the verbatim path, then a helper spelling that names the save
        without spelling the verb — the shape the pinned helper list exists to
        catch. A clean listing answers nothing, which is what makes the empty
        answer of the guards above a verdict rather than an absence of checking.
        """
        poisoned = [
            {"name": "modes_save", "description": f"Call {MODES_SAVE_MARKER}.",
             "schema": "{}"},
            {"name": "save_modes", "description": "Writes the modes document.",
             "schema": json.dumps({"helper": "save_agent_modes_document"})},
            {"name": "clean_tool", "description": "Reads a file.", "schema": "{}"},
        ]
        assert advertisement_offenders(poisoned) == [
            f"modes_save: {MODES_SAVE_MARKER}",
            "save_modes: save_agent_modes",
        ]

    def test_the_modes_save_tool_guard_catches_a_poisoned_lsp_spec(self, monkeypatch):
        """The LSP half trips too: a poisoned catalogue def names the save route.

        The poisoning goes into a spec the committed template *declares* (every
        server there asks for ``diagnostics``), because the hub publishes the
        declared list — a spec no server declares would never be published, and
        a guard that only sees unpublished specs is not guarding the surface.
        """
        monkeypatch.setitem(
            lsp_tools.TOOL_SPECS,
            "diagnostics",
            {**lsp_tools.TOOL_SPECS["diagnostics"],
             "description": f"Save the modes document ({MODES_SAVE_MARKER})."},
        )
        listing = [
            {"name": definition["function"]["name"],
             "description": definition["function"]["description"],
             "schema": json.dumps(definition["function"]["parameters"])}
            for definition in lsp_tool_defs()
        ]
        offenders = advertisement_offenders(listing)
        assert offenders, "the poisoned def is published, so the guard has to see it"
        with pytest.raises(AssertionError):
            assert offenders == [], offenders

    def test_the_modes_writer_caller_scan_catches_a_second_caller(self, tmp_path):
        """A second caller of the core writer is named by the caller scan.

        A scratch ``src`` tree with the routes and a hub that imports the writer:
        the guard's answer is *both*, which is what makes the real guard's
        one-entry answer a verdict. The writer module itself is skipped by the
        shared walker, exactly as it is for the real tree.
        """
        (tmp_path / "api").mkdir()
        (tmp_path / "api" / "routes.py").write_text(
            "from src.core import agent_modes_write\n\n"
            "def route(body):\n"
            "    return agent_modes_write.save_agent_modes_document(body)\n",
            encoding="utf-8",
        )
        (tmp_path / "core" / "mcphub").mkdir(parents=True)
        (tmp_path / "core" / "mcphub" / "manager.py").write_text(
            "from src.core.agent_modes_write import save_agent_modes_document\n\n"
            "def tool_save(document):\n"
            "    return save_agent_modes_document(document)\n",
            encoding="utf-8",
        )
        (tmp_path / "core" / "agent_modes_write.py").write_text(
            "def save_agent_modes_document(document):\n    return {}\n",
            encoding="utf-8",
        )
        assert modes_writer_callers(tmp_path) == [
            str(Path("api") / "routes.py"),
            str(Path("core") / "mcphub" / "manager.py"),
        ]

    def test_the_shell_side_door_guard_catches_a_poisoned_hub(self, tmp_path):
        """A hub naming the modes save trips the terminal-stays-human-only scan.

        The same walker the guard uses, over a scratch hub file: the point is that
        the *guard* sees a hub wiring itself to the writer of the file that decides
        whether tools exist — which is how this class's other checks are proven.
        """
        fake = tmp_path / "cheeky_hub.py"
        fake.write_text(
            "# a hub wiring its tools to POST /api/agent-modes/save\n"
            "from src.core import agent_modes_write\n",
            encoding="utf-8",
        )
        markers = (MODES_SAVE_MARKER, "agent_modes_write", "save_agent_modes_document")
        assert offenders_for(markers, [tmp_path], allowed=set()) == [str(fake)]

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

    def test_the_agent_modes_verb_guard_catches_a_poisoned_pair(self):
        """A verb moved onto either modes path trips the rewritten guard.

        Fed to the same walk the guard uses (a string, not an edit to the real
        file), so the check is that the *guard* sees a mutation — which is what
        makes the assertions in the class above guards rather than comments. Two
        poisonings, because the pair has two halves: a ``post`` on the editor's
        **read** path (a second way to write the file, on a path the dialog only
        reads) and a ``get`` on the **save** path (a reader that shares the
        writer's path, so the pinned verb set is what refuses it).
        """
        poisoned = (
            "@router.get(\"/agent-modes\")\n"
            "def get_agent_modes():\n"
            "    return {}\n\n"
            "@router.get(\"/agent-modes/document\")\n"
            "def get_agent_modes_document():\n"
            "    return {}\n\n"
            "@router.post(\"/agent-modes/document\")\n"
            "def post_agent_modes_document(body: ModesIn):\n"
            "    return {}\n\n"
            "@router.get(\"/agent-modes/save\")\n"
            "def get_agent_modes_save():\n"
            "    return {}\n\n"
            "@router.post(\"/agent-modes/save\")\n"
            "def post_agent_modes_save(body: ModesIn):\n"
            "    return {}\n"
        )
        registered = route_registrations(poisoned, "/agent-modes")
        by_path: Dict[str, List[str]] = {}
        for entry in registered:
            by_path.setdefault(entry["path"], []).append(entry["verb"])
        assert by_path == {
            "/agent-modes": ["get"],
            "/agent-modes/document": ["get", "post"],
            "/agent-modes/save": ["get", "post"],
        }
        with pytest.raises(AssertionError):
            assert by_path == TestTheAgentModesRoutesStayPinnedToTheirPaths.AGENT_MODES_VERBS
        # And the arity guard still sees the reader's signature: the dropdown's
        # path stays argument-free even while the poisoned neighbours carry bodies.
        reader = [entry for entry in registered if entry["path"] == "/agent-modes"]
        assert [entry["handler"].args.args for entry in reader] == [[]]
        assert registered[2]["handler"].args.args != []
