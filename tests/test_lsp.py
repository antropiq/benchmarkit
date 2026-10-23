"""Tests for language-server discovery, the LSP hub, its API and chat wiring.

Mirrors tests/test_mcp.py. The hub is exercised against
``tests/fake_langserver.py`` (a stdlib stdio JSON-RPC server), never against
a real pyright/tsserver — exactly like the MCP tests never spawn real
connectors.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.core.lsphub import constants as lsp_constants
from src.core.lsphub import (
    LspManager,
    action_of_tool_name,
    all_servers,
    TOOL_SPECS,
    find_project_config,
    parse_tsserver_request,
    tsserver_project_info,
    tsserver_response_params,
    load_lsp_servers,
    set_lsp_enabled,
    tool_name_for,
    tools_for_server,
)

from src.core import mcphub
from src.core.mcphub import tools_to_openai_format

FAKE_SERVER = str(Path(__file__).parent / "fake_langserver.py")
# The loader requires an existing workspaceRoot; the repo itself always does.
REPO_ROOT = str(Path(FAKE_SERVER).resolve().parents[2])


def _entry(command: Any, **overrides: Any) -> Dict[str, Any]:
    entry = {
        "enable": True,
        "extensions": [".py"],
        "command": command,
        "workspaceRoot": REPO_ROOT,
    }
    entry.update(overrides)
    return entry


# Values are typed loosely on purpose: the validation tests feed deliberately
# malformed entries (a bare string, ...) to check the loader rejects them.
def _servers_doc(**entries: Any) -> str:
    return json.dumps(entries, indent=2)


# ---------------------------------------------------------------------------
# Config discovery / validation
# ---------------------------------------------------------------------------


class TestLoadLspServers:
    def test_missing_file_yields_nothing(self, tmp_path):
        assert load_lsp_servers(tmp_path / "nope.json") == ([], [])

    def test_invalid_json_is_one_file_level_error(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text("{oops", encoding="utf-8")
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert len(errors) == 1 and "not readable JSON" in errors[0]

    def test_top_level_must_be_an_object(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text("[1, 2]", encoding="utf-8")
        _servers, errors = load_lsp_servers(path)
        assert len(errors) == 1 and "object of servers" in errors[0]

    def test_valid_entry_is_normalized(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                pyright=_entry(
                    ["pyright-langserver", "--stdio"],
                    label="Pyright",
                    description="Python.",
                    extensions=[".PY", ".py"],
                    tools=["hover", "bogus_action"],
                    env={"PATH": "/x"},
                    initializationOptions={"typeCheckingMode": "standard"},
                    unknown_key="ignored",
                )
            ),
            encoding="utf-8",
        )
        [server], errors = load_lsp_servers(path)
        assert errors == []
        assert server["id"] == "pyright"
        assert server["label"] == "Pyright"
        assert server["enable"] is True
        assert server["lazy"] is False
        assert server["extensions"] == [".py"]  # deduplicated, lowercased
        assert server["command"] == ["pyright-langserver", "--stdio"]
        assert server["env"] == {"PATH": "/x"}
        assert server["tools"] == ["hover", "bogus_action"]  # filtered at synthesis
        assert server["maxDiagnostics"] == 40
        assert server["idleShutdownSeconds"] == 300
        assert "unknown_key" not in server

    def test_command_string_is_parsed_like_externals(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry("vue-language-server --stdio --tsdk=/ts/lib")),
            encoding="utf-8",
        )
        [server], errors = load_lsp_servers(path)
        assert errors == []
        assert server["command"] == [
            "vue-language-server",
            "--stdio",
            "--tsdk=/ts/lib",
        ]

    def test_bad_entry_skipped_with_note_but_others_survive(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                good=_entry(["x"]),
                bad_command=_entry([]),
                bad_ext=_entry(["x"], extensions=["py"]),
                bad_root=_entry(["x"], workspaceRoot="relative/path"),
                bad_env=_entry(["x"], env=["not", "a", "map"]),
                bad_init=_entry(["x"], initializationOptions=[1]),
                bad_tools=_entry(["x"], tools=["hover", 4]),
                bad_max=_entry(["x"], maxDiagnostics=0),
                bad_idle=_entry(["x"], idleShutdownSeconds=-1),
                not_object="just a string",
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert [server["id"] for server in servers] == ["good"]
        assert len(errors) == 9
        assert all("'bad_" in error or "not_object" in error for error in errors)
        assert all(error.endswith("— skipped") for error in errors)

    def test_workspace_root_must_exist(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot=str(tmp_path / "gone"))),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "workspaceRoot does not exist" in errors[0]

    def test_denied_workspace_root_is_skipped(self, tmp_path, monkeypatch):
        deny = tmp_path / "restricted-paths.txt"
        deny.write_text(f"{tmp_path}/secret\n# comment\n", encoding="utf-8")
        monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", deny)
        secret = tmp_path / "secret"
        secret.mkdir()  # exists — the deny-list, not existence, must reject it
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot=str(secret))),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "deny-list" in errors[0]

    def test_env_references_are_expanded_everywhere(self, tmp_path, monkeypatch):
        """command, env, workspaceRoot and the option objects all expand.

        ``workspaceRoot`` is expanded *before* the existence/deny-list checks,
        so a per-machine path can be written as one ``${VAR}`` and the entry
        still loads on every checkout. A value that expands to something with
        a space stays a single argv element (no shell re-splitting).
        """
        root = tmp_path / "ws dir"
        root.mkdir()
        monkeypatch.setenv("LSP_TEST_BIN", str(tmp_path / "bin"))
        monkeypatch.setenv("LSP_TEST_ROOT", str(root))
        monkeypatch.setenv("LSP_TEST_PATH", "/usr/bin:/bin")
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                srv=_entry(
                    ["${LSP_TEST_BIN}/langserver", "--stdio", "--root=${LSP_TEST_ROOT}"],
                    env={"PATH": "${LSP_TEST_PATH}"},
                    workspaceRoot="${LSP_TEST_ROOT}",
                    initializationOptions={"tsdk": "${LSP_TEST_ROOT}/lib", "n": 3, "on": True},
                    settings={"html.format": {"indentScriptTags": False}},
                )
            ),
            encoding="utf-8",
        )
        [server], errors = load_lsp_servers(path)
        assert errors == []
        assert server["command"] == [
            f"{tmp_path / 'bin'}/langserver",
            "--stdio",
            f"--root={root}",
        ]
        assert server["workspaceRoot"] == str(root)
        assert server["env"] == {"PATH": "/usr/bin:/bin"}
        assert server["initializationOptions"] == {"tsdk": f"{root}/lib", "n": 3, "on": True}

    def test_unresolved_env_reference_reports_and_explains_the_shape_error(
        self, tmp_path, monkeypatch
    ):
        """One note names the unset variable, and it comes before the error it causes."""
        monkeypatch.delenv("LSP_TEST_NOPE", raising=False)
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(srv=_entry(["x"], workspaceRoot="${LSP_TEST_NOPE}/web")),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path)
        assert servers == []
        assert "LSP_TEST_NOPE" in errors[0] and "not set in the environment" in errors[0]
        assert "workspaceRoot must be an absolute path" in errors[1]

    def test_reserved_id_and_underscore_comment_keys(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            _servers_doc(
                taken=_entry(["x"]),
                _comment="not a server",
            ),
            encoding="utf-8",
        )
        servers, errors = load_lsp_servers(path, reserved_ids={"taken"})
        assert servers == []
        assert errors == ["lsp_servers.json: 'taken': id is reserved — skipped"]

    def test_all_servers_reads_the_local_declaration_file(self):
        # lsp_servers.json is gitignored (per-machine, may hold env secrets),
        # so this only runs where one is configured — like the real app.
        if not lsp_constants.LSP_SERVERS_PATH.is_file():
            pytest.skip("no local lsp_connectors/lsp_servers.json on this machine")
        servers, errors = all_servers()
        assert errors == []
        assert servers
        assert all("env" in server for server in servers)


class TestSetLspEnabled:
    def test_writes_enable_back_and_preserves_everything_else(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text(
            json.dumps(
                {
                    "zeta": _entry(["z"], label="Z"),
                    "alpha": _entry(["a"], label="A"),
                    "_note": "keep me",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert set_lsp_enabled({"alpha": True, "zeta": False}, path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["alpha"]["enable"] is True
        assert data["zeta"]["enable"] is False
        assert list(data) == ["zeta", "alpha", "_note"]  # entry order kept
        assert data["_note"] == "keep me"
        assert data["alpha"]["label"] == "A"

    def test_unknown_id_is_ignored_and_missing_file_is_a_noop(self, tmp_path):
        assert set_lsp_enabled({"ghost": True}, tmp_path / "nope.json") == []
        path = tmp_path / "lsp_servers.json"
        path.write_text(_servers_doc(real=_entry(["x"])), encoding="utf-8")
        assert set_lsp_enabled({"ghost": True}, path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "ghost" not in data

    def test_unparsable_file_is_reported_and_left_untouched(self, tmp_path):
        path = tmp_path / "lsp_servers.json"
        path.write_text("{oops", encoding="utf-8")
        errors = set_lsp_enabled({"a": True}, path)
        assert len(errors) == 1 and "Could not update" in errors[0]
        assert path.read_text(encoding="utf-8") == "{oops"


# ---------------------------------------------------------------------------
# Tool synthesis
# ---------------------------------------------------------------------------


class TestToolSynthesis:
    def test_declared_tools_are_namespaced_and_openai_shaped(self):
        server = {"id": "pyright", "tools": ["diagnostics", "hover"]}
        defs = tools_for_server(server)
        assert [d["function"]["name"] for d in defs] == [
            "lsp_pyright_diagnostics",
            "lsp_pyright_hover",
        ]
        # Matches the exact shape mcphub.tools_to_openai_format() produces,
        # so the routes merge is trivial.
        assert set(defs[0]) == {"type", "function"}
        assert set(defs[0]["function"]) == {"name", "description", "parameters"}

    def test_undeclared_tools_default_to_the_full_catalogue(self):
        server = {"id": "s", "tools": None}
        names = [d["function"]["name"] for d in tools_for_server(server)]
        assert names == [f"lsp_s_{action}" for action in TOOL_SPECS]

    def test_unknown_declared_names_are_dropped(self):
        defs = tools_for_server({"id": "s", "tools": ["hover", "write_file"]})
        assert [d["function"]["name"] for d in defs] == ["lsp_s_hover"]

    def test_capabilities_intersect_the_catalogue(self):
        caps = SimpleNamespace(hover_provider=True, definition_provider=None)
        server = {"id": "s", "tools": ["hover", "definition", "diagnostics"]}
        defs = tools_for_server(server, caps)
        # diagnostics has no capability (push/pull) so it is always offered;
        # the declared order is what survives.
        assert [d["function"]["name"] for d in defs] == [
            "lsp_s_hover",
            "lsp_s_diagnostics",
        ]

    def test_name_round_trip(self):
        assert tool_name_for("vue", "hover") == "lsp_vue_hover"
        assert action_of_tool_name("vue", "lsp_vue_document_symbols") == "document_symbols"
        assert action_of_tool_name("vue", "execute_command") == ""

    def test_openai_format_compatibility(self):
        # A def from the hub and one from the MCP hub must be interchangeable.
        [lsp_def] = tools_for_server({"id": "s", "tools": ["hover"]})
        [mcp_def] = tools_to_openai_format(
            [SimpleNamespace(name="execute_command", description="d", input_schema={})]
        )
        assert set(lsp_def) == set(mcp_def)
        assert set(lsp_def["function"]) == set(mcp_def["function"])


# ---------------------------------------------------------------------------
# LspManager lifecycle against the fake language server
# ---------------------------------------------------------------------------


def _fake_server_entry(tmp_path: Path, server_id: str = "fake", *flags: str,
                       lazy: bool = False) -> Dict[str, Any]:
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return {
        "id": server_id,
        "label": server_id,
        "description": "",
        "command": [sys.executable, FAKE_SERVER, *flags],
        "workspaceRoot": str(root),
        "extensions": [".py"],
        "languageId": "python",
        "enable": True,
        "lazy": lazy,
        "initializationOptions": {},
        "tools": ["diagnostics", "hover", "definition", "document_symbols"],
        "env": {},
        "maxDiagnostics": 10,
        "idleShutdownSeconds": 0,
    }


@pytest.fixture
async def hub(tmp_path):
    """LspManager whose private hub thread is up, torn down after the test.

    Started with no servers declared (``start_all`` boots the thread), then
    each test declares its fake server and starts it on the hub thread —
    the same path the app uses, so ``call_tool_sync`` bridges correctly.
    """
    manager = LspManager([], logs_dir=tmp_path / "logs")
    await manager.start_all()
    try:
        yield manager
    finally:
        await manager.stop_all()


def _declare(hub: LspManager, entry: Dict[str, Any]) -> None:
    hub._servers[entry["id"]] = {**entry, "status": "stopped", "error": None,
                                 "last_used": 0.0}


async def _run(hub: LspManager, coroutine: Any) -> None:
    """Run one hub coroutine on the hub's own thread (like the API does)."""
    # ``start_all`` (the hub fixture) always set the loop before this runs.
    assert hub._loop is not None
    await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(coroutine, hub._loop)
    )


async def _start(hub: LspManager, entry: Dict[str, Any]) -> None:
    _declare(hub, entry)
    await _run(hub, hub._start_one(entry["id"]))


class TestLspLifecycle:
    @pytest.mark.anyio
    async def test_handshake_then_running_with_capability_tools(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        info = hub._servers["fake"]
        assert info["status"] == "running", info["error"]
        assert info["error"] is None
        # Only the capabilities the fake server advertises became tools:
        # references/completion/formatting/signature_help/workspace_symbols
        # are absent.
        tools, mapping = hub.active_tool_defs(["fake"])
        assert [d["function"]["name"] for d in tools] == [
            "lsp_fake_diagnostics",
            "lsp_fake_hover",
            "lsp_fake_definition",
            "lsp_fake_document_symbols",
        ]
        assert set(mapping.values()) == {"fake"}
        assert hub.server_status("fake") == "running"
        assert hub.server_status("nope") is None

    @pytest.mark.anyio
    async def test_stop_is_idempotent_and_restart_respawns(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        first_pid = hub._sessions["fake"].process.pid
        await _run(hub, hub._stop_one("fake"))
        assert hub.server_status("fake") == "stopped"
        await _run(hub, hub._stop_one("fake"))  # idempotent
        assert hub.server_status("fake") == "stopped"
        await _run(hub, hub._restart_one("fake"))
        assert hub.server_status("fake") == "running"
        assert hub._sessions["fake"].process.pid != first_pid

    @pytest.mark.anyio
    async def test_restart_rereads_the_declaration_file(self, hub, tmp_path):
        # The whole point of hub restart: an edited lsp_servers.json (here, a
        # different workspaceRoot and a changed tool list) applies without
        # restarting the web app.
        first = _fake_server_entry(tmp_path, "fake")
        await _start(hub, first)
        old_process = hub._sessions["fake"].process
        (Path(first["workspaceRoot"]) / "in_root1.py").write_text(
            "x = 1\n", encoding="utf-8"
        )
        other = tmp_path / "other_root"
        other.mkdir()
        (other / "in_root2.py").write_text("x = 1\n", encoding="utf-8")
        second = {**first, "workspaceRoot": str(other),
                  "tools": ["diagnostics"]}

        await hub.restart([second], ["fake"])

        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        assert hub._sessions["fake"].process.pid != old_process.pid
        for _ in range(100):
            if old_process.returncode is not None:
                break
            await asyncio.sleep(0.05)
        assert old_process.returncode is not None
        # The new workspaceRoot is in force: its file resolves, the old
        # root's file is gone.
        assert hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "in_root2.py"}
        ) == "No diagnostics."
        assert hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "in_root1.py"}
        ).startswith("Error: not a file:")
        # Tool defs come from the new entry too (hover dropped).
        tools, _mapping = hub.active_tool_defs(["fake"])
        assert [d["function"]["name"] for d in tools] == ["lsp_fake_diagnostics"]

    @pytest.mark.anyio
    async def test_restart_can_drop_and_add_servers(self, hub, tmp_path):
        await _start(hub, _fake_server_entry(tmp_path, "gone"))
        assert hub.server_status("gone") == "running"
        incoming = _fake_server_entry(tmp_path, "fresh")

        await hub.restart([incoming], ["fresh"])

        assert hub.server_status("gone") is None  # no longer declared
        assert hub.server_status("fresh") == "running", hub._servers["fresh"]["error"]

    @pytest.mark.anyio
    async def test_restart_respects_the_enabled_gate(self, hub, tmp_path):
        # A dormant file stays dormant: declared but not enabled means no
        # spawn (the two-gate rule the startup lifespan applies).
        entry = _fake_server_entry(tmp_path)
        await hub.restart([entry], [])
        assert hub.server_status("fake") == "stopped"
        assert "fake" not in hub._sessions

    @pytest.mark.anyio
    async def test_initialize_failure_lands_in_status_verbatim(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--fail-init")
        await _start(hub, entry)
        info = hub._servers["fake"]
        assert info["status"] == "error"
        assert "Could not find a valid TypeScript installation" in info["error"]
        assert "fake" not in hub._sessions

    @pytest.mark.anyio
    async def test_spawn_failure_lands_in_status(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake")
        entry["command"] = ["/nonexistent/langserver"]
        await _start(hub, entry)
        assert hub._servers["fake"]["status"] == "error"
        assert "Could not spawn" in hub._servers["fake"]["error"]

    @pytest.mark.anyio
    async def test_lazy_server_spawns_on_first_tool_call(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, lazy=True)
        _declare(hub, entry)
        await _run(hub, hub._start_all_matching({"fake"}))
        assert hub.server_status("fake") == "stopped"
        # Still represented in the catalogue, from the declared tools.
        tools, _mapping = hub.active_tool_defs(["fake"])
        assert tools
        hub.call_tool_sync("fake", "lsp_fake_hover",
                           {"path": "x.py", "line": 0, "character": 0})
        assert hub.server_status("fake") == "running"

    @pytest.mark.anyio
    async def test_crashed_server_surfaces_as_error_status(self, hub, tmp_path):
        # The fake dies instead of answering its first post-init request
        # (hover): the pending call fails fast (pygls cancels it on process
        # exit) and the exit watcher turns the server's status into 'error'
        # with the log tail, instead of a stale 'running'.
        entry = _fake_server_entry(tmp_path, "fake", "--crash-after", "1")
        await _start(hub, entry)
        assert hub.server_status("fake") == "running"
        (Path(entry["workspaceRoot"]) / "x.py").write_text("x = 1\n", encoding="utf-8")
        result = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "x.py", "line": 0, "character": 0}
        )
        assert result.startswith("Error:")
        for _ in range(100):
            if hub.server_status("fake") == "error":
                break
            await asyncio.sleep(0.05)
        assert hub.server_status("fake") == "error"
        assert "exited with code" in hub._servers["fake"]["error"]

    @pytest.mark.anyio
    async def test_close_session_terminates_the_process(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        process = hub._sessions["fake"].process
        await _run(hub, hub._close_session("fake"))
        for _ in range(100):
            if process.returncode is not None:
                break
            await asyncio.sleep(0.05)
        assert process.returncode is not None


class TestLspToolCalls:
    @pytest.mark.anyio
    async def test_diagnostics_open_then_receive_the_push(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "a.py").write_text(
            "clean line\nx = 1  # BUG here\n", encoding="utf-8"
        )
        result = hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "a.py"}
        )
        assert "a.py:2:10 error: found a BUG here (fakeserverfake001)" in result

    @pytest.mark.anyio
    async def test_clean_file_reports_no_diagnostics(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "b.py").write_text("fine\n", encoding="utf-8")
        result = hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "b.py"}
        )
        assert result == "No diagnostics."

    @pytest.mark.anyio
    async def test_didChange_republish_analyses_current_content(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        document = Path(entry["workspaceRoot"]) / "c.py"
        document.write_text("fine\n", encoding="utf-8")
        assert hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "c.py"}
        ) == "No diagnostics."
        document.write_text("fine\nBUG crept in\n", encoding="utf-8")
        result = hub.call_tool_sync(
            "fake", "lsp_fake_diagnostics", {"path": "c.py"}
        )
        assert "c.py:2:1 error: found a BUG here" in result

    @pytest.mark.anyio
    async def test_hover_and_definition_and_symbols(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "d.py").write_text("x = 1\n", encoding="utf-8")
        hover = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "d.py", "line": 3, "character": 0}
        )
        assert hover == "hover at 3"
        definition = hub.call_tool_sync(
            "fake", "lsp_fake_definition", {"path": "d.py", "line": 0, "character": 0}
        )
        assert definition == "d.py:3:1"
        symbols = hub.call_tool_sync(
            "fake", "lsp_fake_document_symbols", {"path": "d.py"}
        )
        assert symbols == "class Thing (line 5)"

    @pytest.mark.anyio
    async def test_path_policy(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        root = Path(entry["workspaceRoot"])
        (root / "notes.txt").write_text("hi", encoding="utf-8")
        outside = root.parent / "outside.py"
        outside.write_text("x = 1  # BUG\n", encoding="utf-8")

        escaped = hub.call_tool_sync("fake", "lsp_fake_diagnostics",
                                     {"path": str(outside)})
        assert escaped.startswith("Error: path must live inside the workspace root")
        wrong_ext = hub.call_tool_sync("fake", "lsp_fake_diagnostics",
                                       {"path": "notes.txt"})
        assert wrong_ext.startswith("Error: .txt is not handled by fake")
        missing = hub.call_tool_sync("fake", "lsp_fake_diagnostics",
                                     {"path": "nope.py"})
        assert missing.startswith("Error: not a file:")
        empty = hub.call_tool_sync("fake", "lsp_fake_diagnostics", {"path": ""})
        assert empty.startswith("Error: path is required")

    @pytest.mark.anyio
    async def test_unknown_or_unsupported_call_shapes(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        assert hub.call_tool_sync("ghost", "lsp_ghost_hover", {}) == (
            "Error: language server 'ghost' is not declared"
        )
        (Path(entry["workspaceRoot"]) / "e.py").write_text("x\n", encoding="utf-8")
        bad_position = hub.call_tool_sync(
            "fake", "lsp_fake_hover", {"path": "e.py", "line": -1, "character": 0}
        )
        assert "non-negative integers" in bad_position


# ---------------------------------------------------------------------------
# The tsserver relay (Volar's private 'tsserver/request' protocol)
# ---------------------------------------------------------------------------


class TestTsserverRelayShapes:
    """Pure-shape tests: the protocol is doubly nested and easy to get wrong.

    Getting either direction wrong is silent — Volar just never resolves its
    ``getLanguageService()`` promise — so the shapes are pinned here as well
    as they can be against the fake server below.
    """

    def test_request_is_unwrapped_from_its_positional_argument_array(self):
        parsed = parse_tsserver_request(
            [[1, "_vue:projectInfo", {"file": "/w/src/App.vue"}]]
        )
        assert parsed == (1, "_vue:projectInfo", {"file": "/w/src/App.vue"})

    def test_request_flat_and_object_forms_are_also_accepted(self):
        assert parse_tsserver_request([2, "_vue:getComponentNames", None]) == (
            2, "_vue:getComponentNames", None,
        )
        assert parse_tsserver_request(
            {"id": 3, "command": "_vue:projectInfo", "args": {"file": "x"}}
        ) == (3, "_vue:projectInfo", {"file": "x"})

    def test_unrecognisable_request_is_reported_not_raised(self):
        # A payload the hub cannot make sense of must not kill the session:
        # it is logged and skipped.
        assert parse_tsserver_request(["not-an-int", 7]) is None
        assert parse_tsserver_request(None) is None
        assert parse_tsserver_request([]) is None

    def test_response_pairs_the_id_inside_the_params_array(self):
        # Volar's handler is onNotification('tsserver/response', ([id, res]) …)
        # so the pair must be nested one level inside params.
        assert tsserver_response_params(4, {"a": 1}) == [[4, {"a": 1}]]
        assert tsserver_response_params(5, None) == [[5, None]]

    def test_project_config_prefers_tsconfig_over_jsconfig(self, tmp_path):
        (tmp_path / "jsconfig.json").write_text("{}", encoding="utf-8")
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
        assert find_project_config(tmp_path) == str(tmp_path / "tsconfig.json")

    def test_project_config_missing_is_none(self, tmp_path):
        assert find_project_config(tmp_path) is None

    def test_project_info_answer_names_the_config_as_external(self):
        info = tsserver_project_info("/w/a.vue", "/w/tsconfig.json")
        # The config path was supplied, so an answer object is expected.
        assert info is not None
        assert info["configFileName"] == "/w/tsconfig.json"
        assert info["projectKind"] == "external"
        assert info["languageServiceEnabled"] is True

    def test_project_info_without_a_config_is_an_explicit_null(self):
        assert tsserver_project_info("/w/a.vue", None) is None

    @pytest.mark.anyio
    async def test_config_walk_stops_at_the_workspace_root(self, hub, tmp_path):
        # A config above the root must never leak into the answer: the root
        # is the boundary the hub promises not to read past.
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]).parent / "tsconfig.json").write_text(
            "{}", encoding="utf-8"
        )
        session = hub._sessions["fake"]
        nested = Path(entry["workspaceRoot"]) / "src" / "components" / "A.vue"
        assert hub._project_config_for(session, str(nested)) is None

    @pytest.mark.anyio
    async def test_config_is_found_from_a_nested_file(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path)
        await _start(hub, entry)
        root = Path(entry["workspaceRoot"])
        (root / "jsconfig.json").write_text("{}", encoding="utf-8")
        session = hub._sessions["fake"]
        nested = root / "src" / "components" / "A.vue"
        assert hub._project_config_for(session, str(nested)) == str(
            root / "jsconfig.json"
        )
        # A file URI is normalised too (Volar sends a plain path, but the
        # hub should not be fooled by one that is not).
        assert hub._project_config_for(session, nested.as_uri()) == str(
            root / "jsconfig.json"
        )


class TestTsserverRelayAgainstTheFakeServer:
    """The regression that matters: without the relay, every call hangs.

    ``--relay`` makes the fake block ``documentSymbol`` and suppress the
    diagnostics publish until its client answers the ``_vue:projectInfo``
    question it asks on ``didOpen`` — the real Volar behaviour.
    """

    @pytest.mark.anyio
    async def test_relay_answer_unblocks_symbols_and_diagnostics(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        root = Path(entry["workspaceRoot"])
        (root / "tsconfig.json").write_text("{}", encoding="utf-8")
        (root / "v.py").write_text("clean line\nx = 1  # BUG\n", encoding="utf-8")
        await _start(hub, entry)

        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert "Thing" in symbols
        # The fake echoes back the configFileName the hub answered with, so
        # this also proves the response survived the wire in the right shape.
        assert f"config:{root / 'tsconfig.json'}" in symbols

        diagnostics = hub.call_tool_sync("fake", "lsp_fake_diagnostics", {"path": "v.py"})
        assert "v.py:2:10 error: found a BUG here" in diagnostics
        assert hub._sessions["fake"].ts_relay_counts.get("_vue:projectInfo") == 1

    @pytest.mark.anyio
    async def test_relay_answers_null_when_the_workspace_has_no_config(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        (Path(entry["workspaceRoot"]) / "v.py").write_text("x = 1  # BUG\n", encoding="utf-8")
        await _start(hub, entry)
        # No tsconfig/jsconfig: the hub answers null, which is Volar's own
        # "use the inferred project" signal — the call must still resolve.
        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert "config:None" in symbols
        tail = hub.log_tail("fake")
        assert "answering null" in tail

    @pytest.mark.anyio
    async def test_unparsable_relay_request_is_logged_and_survived(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay", "--relay-unknown")
        root = Path(entry["workspaceRoot"])
        (root / "jsconfig.json").write_text("{}", encoding="utf-8")
        (root / "v.py").write_text("x = 1  # BUG\n", encoding="utf-8")
        await _start(hub, entry)
        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        # The garbage request must not have broken the session.
        assert "Thing" in symbols
        assert "unrecognised tsserver/request" in hub.log_tail("fake")

    @pytest.mark.anyio
    async def test_started_session_records_a_log_line(self, hub, tmp_path):
        # A quiet server otherwise leaves no log file at all, which is when
        # the log endpoint is wanted most.
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        await _start(hub, entry)
        tail = hub.log_tail("fake")
        assert "[hub] fake running" in tail
        assert "no tsconfig/jsconfig" in tail

    @pytest.mark.anyio
    async def test_configuration_answers_are_null_not_empty_objects(self, hub, tmp_path):
        # The hub used to answer workspace/configuration with {} per item.
        # Volar does `getConfiguration('html.customData') ?? []` and then
        # iterates it: {} is not nullish and not iterable, so the language
        # service plugin threw 'TypeError: paths is not iterable' and the
        # whole vue-language-server process died on the first completion
        # request. In relay mode the fake applies that same logic and exits
        # with code 4 if the hub ever answers {} again.
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "v.py").write_text("x = 1\n", encoding="utf-8")
        hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert hub.server_status("fake") == "running", hub._servers["fake"]["error"]
        assert "not iterable" not in hub.log_tail("fake")

    @pytest.mark.anyio
    async def test_declared_settings_are_returned_verbatim(self, hub, tmp_path):
        entry = _fake_server_entry(tmp_path, "fake", "--relay")
        entry["settings"] = {"html.customData": ["/custom/data.json"]}
        await _start(hub, entry)
        (Path(entry["workspaceRoot"]) / "v.py").write_text("x = 1\n", encoding="utf-8")
        symbols = hub.call_tool_sync("fake", "lsp_fake_document_symbols", {"path": "v.py"})
        assert "Thing" in symbols


# ---------------------------------------------------------------------------
# /api/lsp endpoints
# ---------------------------------------------------------------------------


class FakeLspManager:
    installed = True

    def __init__(self, status: Optional[List[Dict[str, Any]]] = None) -> None:
        self._status = status or []
        self.started: List[str] = []
        self.stopped: List[str] = []
        self.restarts: List[Any] = []

    async def restart(self, servers, enabled_ids=None) -> None:
        self.restarts.append((servers, enabled_ids))

    def status(self):
        return self._status

    def active_tool_defs(self, ids):
        return [], {}

    def call_tool_sync(self, sid, name, args, timeout=30.0):
        return "ok"

    def server_status(self, sid):
        return next((s["status"] for s in self._status if s["id"] == sid), None)

    def start_server(self, sid):
        self.started.append(sid)

    def stop_server(self, sid):
        self.stopped.append(sid)

    def log_tail(self, sid, chars=4000):
        return f"log tail of {sid}"


@pytest.fixture
def lsp_client(monkeypatch, tmp_path):
    from src import config as config_module

    for key in config_module.SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
    # Point the hub at a temp declaration file (fake servers), so the real
    # lsp_connectors/lsp_servers.json stays untouched.
    servers_dir = tmp_path / "lsp_connectors"
    servers_dir.mkdir()
    workspace = tmp_path / "web"
    workspace.mkdir()
    servers_path = servers_dir / "lsp_servers.json"
    servers_path.write_text(
        _servers_doc(
            pyright=_entry(
                [sys.executable, FAKE_SERVER],
                label="Pyright (Python)",
                env={"SECRET": "do-not-leak"},
            ),
            vue=_entry(
                [sys.executable, FAKE_SERVER],
                label="Vue",
                extensions=[".vue"],
                enable=False,
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lsp_constants, "LSP_CONNECTORS_DIR", servers_dir)
    monkeypatch.setattr(lsp_constants, "LSP_SERVERS_PATH", servers_path)
    monkeypatch.setattr(lsp_constants, "LSP_RESTRICTED_PATHS", servers_dir / "restricted-paths.txt")
    # Keep the MCP hub off the developer's own externals.json: no connector is
    # declared in these tests unless they write that file themselves.
    monkeypatch.setattr(
        mcphub.constants, "EXTERNALS_PATH", servers_dir / "externals.json"
    )
    monkeypatch.setattr(
        routes,
        "load_endpoints_config",
        lambda: (
            SimpleNamespace(
                serverUrl="http://x/v1", endpoint="m", model_list=lambda: ["m"],
                to_kwargs=lambda: {}, apiKey="", sysprompt=None,
            ),
            None,
        ),
    )
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, app


class TestLspEndpoints:
    def test_get_lists_servers_with_stopped_status(self, lsp_client):
        client, _ = lsp_client
        body = client.get("/api/lsp").json()
        assert [s["id"] for s in body["servers"]] == ["pyright", "vue"]
        assert body["servers"][0]["status"] == "stopped"
        assert body["servers"][0]["active"] is False
        assert body["servers"][1]["active"] is False  # enable: false
        assert body["maxIterations"] == 25

    def test_get_never_leaks_env_or_initialization_secrets(self, lsp_client):
        client, _ = lsp_client
        text = client.get("/api/lsp").text
        assert "do-not-leak" not in text
        assert "SECRET" not in text

    def test_active_requires_both_gates(self, lsp_client, monkeypatch):
        import os

        client, _ = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        body = client.get("/api/lsp").json()
        by_id = {s["id"]: s for s in body["servers"]}
        assert by_id["pyright"]["active"] is True
        assert by_id["vue"]["active"] is False  # enable flag still false
        monkeypatch.setenv("LSP_ENABLED", "vue")
        body = client.get("/api/lsp").json()
        assert {s["id"]: s["active"] for s in body["servers"]} == {
            "pyright": False,
            "vue": False,
        }

    def test_post_persists_to_env_file_and_writes_enable_back(self, lsp_client):
        client, _ = lsp_client
        response = client.post(
            "/api/lsp", json={"enabled": ["pyright"], "maxIterations": 7}
        )
        assert response.status_code == 200
        body = response.json()
        assert [s["id"] for s in body["servers"] if s["active"]] == ["pyright"]
        assert body["maxIterations"] == 7
        assert os.environ["LSP_ENABLED"] == "pyright"
        assert os.environ["MAX_TOOL_ITERATION"] == "7"
        from src import config as config_module

        file_values = dotenv_values(config_module.ENV_PATH)
        assert file_values["LSP_ENABLED"] == "pyright"
        # enable flag written back for every declared server, both gates move.
        data = json.loads(lsp_constants.LSP_SERVERS_PATH.read_text(encoding="utf-8"))
        assert data["pyright"]["enable"] is True
        assert data["vue"]["enable"] is False
        assert data["pyright"]["env"] == {"SECRET": "do-not-leak"}  # keys kept

    def test_post_empty_list_deactivates_all(self, lsp_client):
        client, _ = lsp_client
        client.post("/api/lsp", json={"enabled": ["pyright", "vue"]})
        response = client.post("/api/lsp", json={"enabled": []})
        assert response.status_code == 200
        assert "LSP_ENABLED" not in os.environ
        assert not [s for s in response.json()["servers"] if s["active"]]
        data = json.loads(lsp_constants.LSP_SERVERS_PATH.read_text(encoding="utf-8"))
        assert data["pyright"]["enable"] is False

    def test_post_unknown_server_returns_422(self, lsp_client):
        client, _ = lsp_client
        response = client.post("/api/lsp", json={"enabled": ["ghost"]})
        assert response.status_code == 422
        assert "ghost" in response.text

    def test_post_invalid_max_iterations_returns_422(self, lsp_client):
        client, _ = lsp_client
        assert client.post("/api/lsp", json={"maxIterations": 0}).status_code == 422
        assert client.post("/api/lsp", json={"maxIterations": 9999}).status_code == 422

    def test_post_during_a_run_returns_409(self, lsp_client):
        client, app = lsp_client
        app.state.run_state.status = "running"
        response = client.post("/api/lsp", json={"enabled": ["pyright"]})
        assert response.status_code == 409

    def test_post_reconciles_running_servers(self, lsp_client):
        client, app = lsp_client
        hub = FakeLspManager(
            [{"id": "pyright", "status": "running", "error": None},
             {"id": "vue", "status": "stopped", "error": None}]
        )
        app.state.lsp = hub
        # vue asked for: start it; pyright stays (already running).
        client.post("/api/lsp", json={"enabled": ["pyright", "vue"]})
        assert hub.started == ["vue"]
        assert hub.stopped == []
        # pyright dropped while running: stop it.
        client.post("/api/lsp", json={"enabled": ["vue"]})
        assert hub.stopped == ["pyright"]

    def test_merges_manager_status_into_the_payload(self, lsp_client):
        client, app = lsp_client
        app.state.lsp = FakeLspManager(
            [{"id": "pyright", "status": "error", "error": "boom", "toolCount": 3}]
        )
        by_id = {s["id"]: s for s in client.get("/api/lsp").json()["servers"]}
        assert by_id["pyright"]["status"] == "error"
        assert by_id["pyright"]["error"] == "boom"
        assert by_id["pyright"]["toolCount"] == 3
        assert by_id["vue"]["status"] == "stopped"

    def test_restart_rereads_the_file_and_reapplies_the_gate(
        self, lsp_client, monkeypatch
    ):
        client, app = lsp_client
        # The hub under test is a fake, so spawning is allowed here: the
        # autouse switch would otherwise blank the scan (tested separately).
        monkeypatch.delenv("BENCHMARKIT_DISABLE_LSP", raising=False)
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        hub = FakeLspManager()
        app.state.lsp = hub

        response = client.post("/api/lsp/restart")

        assert response.status_code == 200
        body = response.json()
        assert {server["id"] for server in body["servers"]} == {"pyright", "vue"}
        assert body["serversError"] is None
        # Fresh file scan + the two-gate intersection (vue is enable:false).
        assert len(hub.restarts) == 1
        servers, enabled_ids = hub.restarts[0]
        assert [server["id"] for server in servers] == ["pyright", "vue"]
        assert enabled_ids == ["pyright"]

    def test_restart_keeps_a_dormant_file_dormant(self, lsp_client, monkeypatch):
        client, app = lsp_client
        monkeypatch.delenv("BENCHMARKIT_DISABLE_LSP", raising=False)
        monkeypatch.delenv("LSP_ENABLED", raising=False)  # env gate closed
        hub = FakeLspManager()
        app.state.lsp = hub
        client.post("/api/lsp/restart")
        servers, enabled_ids = hub.restarts[0]
        assert [server["id"] for server in servers] == ["pyright", "vue"]
        assert enabled_ids == []

    def test_restart_honours_the_disable_switch(self, lsp_client, monkeypatch):
        # BENCHMARKIT_DISABLE_LSP keeps the endpoints alive but spawns
        # nothing — the same rule as the app lifespan (tests rely on it).
        client, app = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        hub = FakeLspManager()
        app.state.lsp = hub
        client.post("/api/lsp/restart")
        servers, enabled_ids = hub.restarts[0]
        assert servers == []
        assert enabled_ids == []

    def test_restart_during_a_run_returns_409(self, lsp_client):
        client, app = lsp_client
        hub = FakeLspManager()
        app.state.lsp = hub
        app.state.run_state.status = "running"
        response = client.post("/api/lsp/restart")
        assert response.status_code == 409
        assert hub.restarts == []
        app.state.run_state.status = "idle"

    def test_restart_without_a_hub_returns_503(self, lsp_client):
        client, app = lsp_client
        app.state.lsp = None
        assert client.post("/api/lsp/restart").status_code == 503

    def test_log_endpoint_returns_the_tail(self, lsp_client):
        client, app = lsp_client
        app.state.lsp = FakeLspManager()
        body = client.get("/api/lsp/log", params={"server": "pyright"}).json()
        assert body == {"server": "pyright", "log": "log tail of pyright"}
        assert client.get("/api/lsp/log").status_code == 404


class TestActiveServerIds:
    def test_both_gates_must_agree(self, lsp_client, monkeypatch):
        monkeypatch.setenv("LSP_ENABLED", "pyright,vue")
        assert routes._active_server_ids() == ["pyright"]
        monkeypatch.delenv("LSP_ENABLED")
        assert routes._active_server_ids() == []


# ---------------------------------------------------------------------------
# /api/chat wiring: the merged tool loop
# ---------------------------------------------------------------------------


def _lsp_def(name: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": f"d {name}", "parameters": {}},
    }


class TestChatLspWiring:
    def _capture_stream(self, monkeypatch, probe: tuple = ()):
        """Stub ``stream_chat`` capturing the merged tool-loop kwargs.

        ``probe`` names are the tools the stub executes through the merged
        router (only tools actually offered by a hub may be probed).
        """
        captured: Dict[str, Any] = {}

        def fake_stream(messages, model, server_url, emit, sampling=None,
                        cancel_event=None, api_key="none", sysprompt=None,
                        tools=None, tool_executor=None, tool_servers=None,
                        max_tool_iterations=25, agent_instructions=None):
            captured["tools"] = tools
            captured["tool_servers"] = tool_servers
            captured["max"] = max_tool_iterations
            if tool_executor:
                for tool_name in probe:
                    captured[f"exec_{tool_name}"] = tool_executor(tool_name, {})
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake_stream)
        return captured

    def test_lsp_only_keeps_one_shared_iteration_cap(self, lsp_client, monkeypatch):
        client, app = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        monkeypatch.setenv("MAX_TOOL_ITERATION", "9")

        class Hub(FakeLspManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

            def call_tool_sync(self, sid, name, args, timeout=30.0):
                return f"{sid}:{name}"

        app.state.lsp = Hub()
        captured = self._capture_stream(monkeypatch, ("lsp_pyright_hover",))
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert app.state.lsp.ids == ["pyright"]
        assert [d["function"]["name"] for d in captured["tools"]] == ["lsp_pyright_hover"]
        assert captured["tool_servers"] == {"lsp_pyright_hover": "pyright"}
        assert captured["max"] == 9
        assert captured["exec_lsp_pyright_hover"] == "pyright:lsp_pyright_hover"

    def test_mcp_and_lsp_merge_into_one_loop(self, lsp_client, monkeypatch):
        client, app = lsp_client
        monkeypatch.setenv("LSP_ENABLED", "pyright")
        # The MCP connector comes from the declaration file, whose ``enable``
        # flag is the only activation source.
        externals_path = mcphub.constants.EXTERNALS_PATH
        externals_path.parent.mkdir(parents=True, exist_ok=True)
        externals_path.write_text(
            json.dumps({"console_mcp": {"type": "local", "enable": True, "command": ["x"]}}),
            encoding="utf-8",
        )

        mcp_tools = [_lsp_def("execute_command")]

        class McpHub(FakeLspManager):
            def active_tool_defs(self, ids):
                return mcp_tools, {"execute_command": "console_mcp"}

            def call_tool_sync(self, sid, name, args, timeout=30.0):
                return f"mcp:{name}"

        class LspHub(FakeLspManager):
            def active_tool_defs(self, ids):
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

            def call_tool_sync(self, sid, name, args, timeout=30.0):
                return f"lsp:{name}"

        app.state.mcp = McpHub()
        app.state.lsp = LspHub()
        captured = self._capture_stream(
            monkeypatch, ("execute_command", "lsp_pyright_hover")
        )
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert [d["function"]["name"] for d in captured["tools"]] == [
            "execute_command",
            "lsp_pyright_hover",
        ]
        assert captured["tool_servers"] == {
            "execute_command": "console_mcp",
            "lsp_pyright_hover": "pyright",
        }
        # The router sends each tool to its own hub.
        assert captured["exec_execute_command"] == "mcp:execute_command"
        assert captured["exec_lsp_pyright_hover"] == "lsp:lsp_pyright_hover"

    def test_body_filter_cannot_revive_a_settings_disabled_server(
        self, lsp_client, monkeypatch
    ):
        client, app = lsp_client

        class Hub(FakeLspManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

        hub = Hub()
        app.state.lsp = hub
        self._capture_stream(monkeypatch)
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "lspEnabled": ["pyright"],  # user toggle on...
            },
        )
        assert response.status_code == 200
        # ...but LSP_ENABLED is empty, so no server id ever reaches the hub.
        assert hub.ids == []

    def test_body_filter_intersects_the_activated_set(self, lsp_client, monkeypatch):
        client, app = lsp_client
        import os

        os.environ["LSP_ENABLED"] = "pyright,vue"

        class Hub(FakeLspManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return ([_lsp_def("lsp_pyright_hover")],
                        {"lsp_pyright_hover": "pyright"})

        hub = Hub()
        app.state.lsp = hub
        self._capture_stream(monkeypatch)
        client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}],
                  "lspEnabled": ["vue"]},
        )
        # vue is Settings-active but toggled off for this conversation;
        # pyright is on in Settings but not listed in the body filter.
        assert hub.ids == []


# ---------------------------------------------------------------------------
# The app lifespan must never spawn language servers in tests
# (conftest sets BENCHMARKIT_DISABLE_LSP, mirroring the MCP rule).
# ---------------------------------------------------------------------------


def test_lifespan_starts_no_servers_without_the_env_switch(lsp_client):
    _client, app = lsp_client
    # create_app's lifespan ran with BENCHMARKIT_DISABLE_LSP=1 (autouse
    # conftest fixture): an empty LspManager, so nothing was ever spawned.
    assert app.state.lsp.status() == []
