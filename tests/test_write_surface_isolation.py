"""The Editor's write routes are reachable from the browser, and from nothing else.

AGENTS.md promises *"Tools are read-only by design"*. That promise survived the
guarded write surface only because something asserts it, so this file asserts
it three ways, each one a different way the promise could rot:

* **no advertised tool names a write route** — the MCP connectors' own tool
  listings and the LSP hub's tool catalogue, both built the way the hubs build
  them (the connectors answer ``list_tools`` themselves; the LSP catalogue is
  ``TOOL_SPECS`` through ``tools_for_server``, which is what a hub would
  publish for the servers in the committed template). With
  ``BENCHMARKIT_DISABLE_MCP``/``_LSP`` the live hubs start disconnected, so
  the lists here come from the sources of truth rather than a session.
* **no code path spells a call to one** — a source scan for the write markers
  (``PUT /api/fs``, ``POST /api/fs``, ``DELETE /api/fs``). They may appear in
  ``src/api/routes.py`` (the definition) and the Editor composables (the
  caller), and nowhere else. ``GET /api/fs`` is deliberately *not* a marker:
  the read surface's docstrings name their routes, and always will.
* **the terminal stays a human surface** — the shell is a WebSocket reached
  from the SPA, and neither hub mentions it, so it cannot become a tool by
  someone wiring the hub to ``app.state.terminal``.

The last test feeds each check a deliberately poisoned input, because a guard
that cannot fail is not a guard: that is the same "assert the invariant, not
the intent" style as the file-icons sync test.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.core import editorfs
from src.core.lsphub import tools as lsp_tools

REPO_ROOT = Path(editorfs.constants.PROJECT_ROOT)

# The three write routes, in the exact spelling a caller would use. Chosen so
# that a docstring naming a route *method* (``GET /api/fs/tree``) stays out of
# the scan, and a comment or a string naming a write is caught.
WRITE_MARKERS = ("PUT /api/fs", "POST /api/fs", "DELETE /api/fs")

# Where a write marker is legitimate.
ALLOWED_HOMES = {
    Path("src/api/routes.py"),          # the routes are defined here
    Path("web/src/composables/useEditorFiles.js"),   # save / delete
    Path("web/src/composables/useEditorCreate.js"),  # create
    Path("web/src/composables/useEditorRename.js"),  # rename
}


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


def connector_tool_lists() -> Dict[str, List[Dict[str, Any]]]:
    """The tool listing each local MCP connector advertises, by connector id.

    These are the scripts referenced as ``local`` entries in
    ``externals.json``, and ``list_tools`` is exactly what the hub asks them
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
                   "rename_path", "trash_path")
        callers: List[str] = []
        for path in (REPO_ROOT / "src").rglob("*.py"):
            if path.parent.name == "editorfs":
                continue  # these *define* the helpers
            text = path.read_text(encoding="utf-8")
            if any(name in text for name in writers):
                callers.append(str(path.relative_to(REPO_ROOT)))
        assert callers == [str(Path("src") / "api" / "routes.py")]


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
