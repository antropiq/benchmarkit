"""Tests for MCP connector discovery, settings API, and the chat tool loop."""

import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.config import mcp_enabled_ids, max_mcp_iteration
from src.core import chat as chat_core
from src.core import mcp_hub
from src.core.mcp_hub import (
    McpManager,
    all_connectors,
    connector_environment,
    discover_connectors,
    extract_tool_result_text,
    load_externals,
    set_externals_enabled,
    tools_to_openai_format,
)

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _write_connector(root: Path, name: str, docstring: Optional[str]) -> None:
    body = f'"""{docstring}"""\n' if docstring is not None else ""
    (root / f"{name}.py").write_text(body, encoding="utf-8")


class TestDiscoverConnectors:
    def test_reads_docstring_first_paragraph(self, tmp_path):
        _write_connector(
            tmp_path,
            "console_mcp",
            "Console MCP Server - Allows LLMs.\n\nSecond paragraph ignored.\n",
        )
        [connector] = discover_connectors(tmp_path)
        assert connector["id"] == "console_mcp"
        assert connector["label"] == "Console"
        assert connector["description"] == "Console MCP Server - Allows LLMs."
        assert (
            mcp_hub.AUTO_PORT_BASE
            <= connector["port"]
            < mcp_hub.AUTO_PORT_BASE + mcp_hub.AUTO_PORT_SPAN
        )
        assert connector["endpoint"] == f"http://127.0.0.1:{connector['port']}/mcp"

    def test_ignores_other_files_and_broken_modules(self, tmp_path):
        _write_connector(tmp_path, "playwright_mcp", "Browser.")
        (tmp_path / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "helper.py").write_text("", encoding="utf-8")
        (tmp_path / "broken_mcp.py").write_text("def (:", encoding="utf-8")
        ids = [c["id"] for c in discover_connectors(tmp_path)]
        assert ids == ["playwright_mcp"]  # unparsable modules are skipped

    def test_missing_directory_returns_empty(self, tmp_path):
        assert discover_connectors(tmp_path / "nope") == []

    def test_allocation_is_deterministic(self, tmp_path):
        _write_connector(tmp_path, "alpha_mcp", "A.")
        _write_connector(tmp_path, "beta_mcp", "B.")
        first = [c["port"] for c in discover_connectors(tmp_path)]
        second = [c["port"] for c in discover_connectors(tmp_path)]
        assert first == second
        assert len(set(first)) == 2  # distinct connectors, distinct ports

    def test_preferred_port_survives_hash_seed_change(self):
        # hash() is randomised per process via PYTHONHASHSEED — the port
        # derivation must use a stable digest, so two interpreter runs
        # with different seeds must agree on the preferred port.
        project_root = str(Path(mcp_hub.__file__).resolve().parents[2])
        code = (
            f"import sys; sys.path.insert(0, {project_root!r}); "
            "from src.core.mcp_hub import _preferred_port; "
            "print(_preferred_port('console_mcp'))"
        )
        outputs = []
        for seed in ("1", "2"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            outputs.append(result.stdout.strip())
        assert outputs[0] == outputs[1] == str(mcp_hub._preferred_port("console_mcp"))

    def test_no_registry_entry_is_required_for_a_new_connector(self, tmp_path):
        _write_connector(tmp_path, "foo_mcp", "Fresh connector, no port declared.")
        [connector] = discover_connectors(tmp_path)
        assert connector["port"]  # never 0 for a parsed built-in
        assert connector["endpoint"] == f"http://127.0.0.1:{connector['port']}/mcp"

    def test_allocated_ports_never_collide_within_a_run(self, tmp_path):
        for index in range(mcp_hub.AUTO_PORT_SPAN + 2):
            _write_connector(tmp_path, f"c{index:03d}_mcp", "x.")
        ports = [c["port"] for c in discover_connectors(tmp_path)]
        allocated = [port for port in ports if port]
        assert len(allocated) == len(set(allocated))
        assert all(
            mcp_hub.AUTO_PORT_BASE <= port < mcp_hub.AUTO_PORT_BASE + mcp_hub.AUTO_PORT_SPAN
            for port in allocated
        )
        # More connectors than range slots: the surplus reports port 0.
        assert len(ports) - len(allocated) >= 2

    def test_foreign_occupancy_shifts_the_allocation(self, tmp_path):
        _write_connector(tmp_path, "solo_mcp", "Solo connector.")
        [before] = discover_connectors(tmp_path)
        hog = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            hog.bind(("127.0.0.1", before["port"]))
            [after] = discover_connectors(tmp_path)
        finally:
            hog.close()
        assert after["port"] != before["port"]
        assert (
            mcp_hub.AUTO_PORT_BASE
            <= after["port"]
            < mcp_hub.AUTO_PORT_BASE + mcp_hub.AUTO_PORT_SPAN
        )


# ---------------------------------------------------------------------------
# externals.json loading and write-back
# ---------------------------------------------------------------------------


def _write_externals(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


class TestLoadExternals:
    def test_missing_file_is_empty_and_silent(self, tmp_path):
        assert load_externals(tmp_path / "externals.json") == ([], [])

    def test_parses_remote_and_local_entries(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "context7": {
                    "type": "remote",
                    "enable": True,
                    "url": "https://mcp.context7.com/mcp",
                    "headers": {"CONTEXT7_API_KEY": "ctx7sk-1"},
                },
                "chrome-devtools": {
                    "type": "local",
                    "command": ["npx", "-y", "chrome-devtools-mcp@latest"],
                },
            },
        )
        connectors, errors = load_externals(path)
        assert errors == []
        by_id = {c["id"]: c for c in connectors}
        remote = by_id["context7"]
        assert remote["external"] is True and remote["kind"] == "remote"
        assert remote["enable"] is True
        assert remote["endpoint"] == "https://mcp.context7.com/mcp"
        assert remote["headers"] == {"CONTEXT7_API_KEY": "ctx7sk-1"}
        local = by_id["chrome-devtools"]
        assert local["enable"] is False  # unset enable defaults to off
        assert local["command"] == ["npx", "-y", "chrome-devtools-mcp@latest"]
        assert local["endpoint"] == ""

    def test_string_command_is_split_and_args_appended(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {"devtools": {"type": "local", "command": "npx -y 'chrome devtools'", "args": ["--stdio"]}},
        )
        connectors, errors = load_externals(path)
        assert errors == []
        assert connectors[0]["command"] == ["npx", "-y", "chrome devtools", "--stdio"]

    def test_invalid_json_reports_and_yields_nothing(self, tmp_path):
        path = tmp_path / "externals.json"
        path.write_text("{oops", encoding="utf-8")
        connectors, errors = load_externals(path)
        assert connectors == [] and len(errors) == 1 and "JSON" in errors[0]

    def test_non_object_root_reports(self, tmp_path):
        path = tmp_path / "externals.json"
        path.write_text("[]", encoding="utf-8")
        _, errors = load_externals(path)
        assert "object" in errors[0]

    def test_bad_entries_skipped_with_notes(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "no_type": {"enable": True},
                "bad_remote": {"type": "remote", "url": "ftp://x"},
                "bad_local": {"type": "local"},
                "bad_headers": {"type": "remote", "url": "http://x", "headers": "nope"},
                "console_mcp": {"type": "remote", "url": "http://x"},
            },
        )
        connectors, errors = load_externals(path, reserved_ids={"console_mcp"})
        assert connectors == []
        assert len(errors) == 5
        assert any("collides" in note for note in errors)

    def test_valid_entries_survive_bad_neighbours(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "oops": {"type": "sideways"},
                "good": {"type": "remote", "url": "https://x/mcp", "description": "Docs."},
            },
        )
        connectors, errors = load_externals(path)
        assert [c["id"] for c in connectors] == ["good"]
        assert connectors[0]["description"] == "Docs."
        assert len(errors) == 1


class TestSetExternalsEnabled:
    def test_missing_file_is_noop(self, tmp_path):
        path = tmp_path / "externals.json"
        assert set_externals_enabled({"a": True}, path) == []
        assert not path.exists()

    def test_flips_flags_and_preserves_everything_else(self, tmp_path):
        path = _write_externals(
            tmp_path / "externals.json",
            {
                "a": {"type": "remote", "url": "https://a", "enable": False, "custom": "keep"},
                "b": {"type": "local", "command": ["x"], "enable": True},
                "untouched": {"type": "remote", "url": "https://u", "enable": True},
            },
        )
        assert set_externals_enabled({"a": True, "b": False}, path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["a"] == {"type": "remote", "url": "https://a", "enable": True, "custom": "keep"}
        assert data["b"]["enable"] is False
        assert data["untouched"]["enable"] is True

    def test_unknown_id_is_not_added(self, tmp_path):
        path = _write_externals(tmp_path / "externals.json", {})
        set_externals_enabled({"ghost": True}, path)
        assert json.loads(path.read_text(encoding="utf-8")) == {}

    def test_broken_file_is_left_untouched(self, tmp_path):
        path = tmp_path / "externals.json"
        path.write_text("{oops", encoding="utf-8")
        errors = set_externals_enabled({"a": True}, path)
        assert errors and "Could not update" in errors[0]
        assert path.read_text(encoding="utf-8") == "{oops"


# ---------------------------------------------------------------------------
# Tool formatting helpers (ported from the agentic-tool runner)
# ---------------------------------------------------------------------------


class TestToolHelpers:
    def test_tools_to_openai_format(self):
        tool = SimpleNamespace(
            name="execute_command",
            description="Run a command",
            input_schema={"type": "object", "properties": {}},
        )
        [entry] = tools_to_openai_format([tool])
        assert entry["function"]["name"] == "execute_command"
        assert entry["function"]["description"] == "Run a command"

    def test_extract_tool_result_text_joins_and_truncates(self):
        result = SimpleNamespace(
            is_error=False,
            content=[SimpleNamespace(text="a"), SimpleNamespace(text="b")],
        )
        assert extract_tool_result_text(result) == "a\nb"
        big = SimpleNamespace(is_error=False, content=[SimpleNamespace(text="x" * 100)])
        assert extract_tool_result_text(big, max_length=10).endswith("[truncated]")

    def test_extract_tool_result_error_flag(self):
        result = SimpleNamespace(is_error=True, content=[SimpleNamespace(text="boom")])
        assert "boom" in extract_tool_result_text(result)


# ---------------------------------------------------------------------------
# McpManager (no real processes or clients)
# ---------------------------------------------------------------------------


def _manager(connector_ids=("console_mcp",)) -> McpManager:
    return McpManager(discover_connectors_stub(connector_ids))


def discover_connectors_stub(connector_ids) -> List[Dict[str, Any]]:
    # Explicit literal ports: the manager tests never exercise discovery or
    # allocation, they only exercise McpManager bookkeeping.
    connectors = []
    for index, cid in enumerate(connector_ids):
        port = 8940 + index
        connectors.append(
            {
                "id": cid,
                "label": cid,
                "description": "",
                "port": port,
                "endpoint": f"http://127.0.0.1:{port}/mcp",
            }
        )
    return connectors


class TestMcpManager:
    def test_active_tool_defs_only_running_connectors(self):
        manager = _manager(("console_mcp", "playwright_mcp"))
        manager._tools["console_mcp"] = [
            SimpleNamespace(name="execute_command", description="", input_schema=None)
        ]
        manager._tools["playwright_mcp"] = [
            SimpleNamespace(name="browser_navigate", description="", input_schema=None)
        ]
        tools, mapping = manager.active_tool_defs(["console_mcp", "playwright_mcp"])
        assert tools == []  # nothing marked running yet
        manager._connectors["console_mcp"]["status"] = "running"
        manager._connectors["playwright_mcp"]["status"] = "error"
        tools, mapping = manager.active_tool_defs(["console_mcp", "playwright_mcp"])
        assert [t["function"]["name"] for t in tools] == ["execute_command"]
        assert mapping == {"execute_command": "console_mcp"}

    def test_status_lists_all_connectors(self):
        manager = _manager(("console_mcp", "playwright_mcp"))
        status = {s["id"]: s for s in manager.status()}
        assert status["console_mcp"]["status"] == "stopped"
        assert status["playwright_mcp"]["port"] == 8941

    def test_stop_all_is_idempotent_without_start(self):
        manager = _manager()
        import asyncio

        asyncio.run(manager.stop_all())
        asyncio.run(manager.stop_all())  # second call must not raise


class TestStartBuiltinPortHandoff:
    def test_subprocess_receives_the_allocated_port(self, tmp_path, monkeypatch):
        # The hub is the port authority: the child must be launched with
        # --port set to the hub-allocated value, never left to its own
        # module default.
        import asyncio

        manager = McpManager(
            discover_connectors_stub(("console_mcp",)), logs_dir=tmp_path
        )
        # Only the lazy ``from mcp import Client`` inside _start_builtin needs
        # a module; patching after __init__ keeps find_spec() untouched.
        monkeypatch.setitem(sys.modules, "mcp", SimpleNamespace(Client=object))
        captured: List[List[str]] = []

        async def fake_exec(*args, **kwargs):
            captured.append([str(arg) for arg in args])
            return SimpleNamespace(returncode=1)  # exit -> retry loop breaks at once

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        info = manager._connectors["console_mcp"]
        asyncio.run(manager._start_builtin("console_mcp", info))

        assert captured, "connector subprocess was never spawned"
        argv = captured[0]
        assert f"--port={info['port']}" in argv
        assert "--host=127.0.0.1" in argv
        assert info["status"] == "error"  # fake process exited immediately
        assert "exited" in (info["error"] or "")


def _external_manager() -> McpManager:
    return McpManager(
        [
            {
                "id": "console_mcp",
                "label": "Console",
                "description": "",
                "port": 9999,
                "endpoint": "http://127.0.0.1:9999/mcp",
                "external": False,
                "kind": "builtin",
            },
            {
                "id": "on_ext",
                "label": "on_ext",
                "description": "",
                "port": 0,
                "endpoint": "https://example.com/mcp",
                "command": None,
                "external": True,
                "kind": "remote",
                "enable": True,
                "headers": {"SECRET": "leaky"},
                "env": {"TOKEN": "leaky"},
            },
            {
                "id": "off_ext",
                "label": "off_ext",
                "description": "",
                "port": 0,
                "endpoint": "https://example.com/other",
                "command": None,
                "external": True,
                "kind": "remote",
                "enable": False,
                "headers": {},
                "env": {},
            },
        ]
    )


class TestManagerExternals:
    def test_status_exposes_kind_and_never_secrets(self):
        manager = _external_manager()
        entries = manager.status()
        by_id = {entry["id"]: entry for entry in entries}
        assert by_id["on_ext"]["kind"] == "remote"
        assert by_id["on_ext"]["external"] is True
        assert by_id["console_mcp"]["kind"] == "builtin"
        for entry in entries:
            assert "headers" not in entry
            assert "env" not in entry
        assert "leaky" not in repr(entries)

    def test_start_all_skips_disabled_externals(self):
        import asyncio

        manager = _external_manager()
        started: List[str] = []

        async def fake_start(connector_id: str) -> None:
            started.append(connector_id)

        manager._installed = True
        manager._start_one = fake_start
        asyncio.run(manager.start_all())
        asyncio.run(manager.stop_all())
        assert sorted(started) == ["console_mcp", "on_ext"]

    def test_start_stop_connector_are_noops_before_start(self):
        manager = _external_manager()
        manager.start_connector("on_ext")  # must not raise without a loop
        manager.stop_connector("on_ext")
        assert manager.connector_status("on_ext") == "stopped"
        assert manager.connector_status("ghost") is None


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------


class TestMcpEnvHelpers:
    def test_mcp_enabled_ids_parses_and_dedupes(self, monkeypatch):
        monkeypatch.setenv("MCP_ENABLED", "console_mcp, playwright_mcp,,console_mcp,")
        assert mcp_enabled_ids() == ["console_mcp", "playwright_mcp"]

    def test_mcp_enabled_ids_empty(self, monkeypatch):
        monkeypatch.delenv("MCP_ENABLED", raising=False)
        assert mcp_enabled_ids() == []

    def test_max_mcp_iteration_default_and_fallbacks(self, monkeypatch):
        monkeypatch.delenv("MAX_MCP_ITERATION", raising=False)
        assert max_mcp_iteration() == 25
        monkeypatch.setenv("MAX_MCP_ITERATION", "7")
        assert max_mcp_iteration() == 7
        monkeypatch.setenv("MAX_MCP_ITERATION", "nope")
        assert max_mcp_iteration() == 25
        monkeypatch.setenv("MAX_MCP_ITERATION", "0")
        assert max_mcp_iteration() == 1


class TestConnectorEnvironment:
    def test_relative_temp_vars_never_reach_the_connector(self, tmp_path):
        # The sampling parameter TEMP=1 from .env must not be taken by the
        # Node playwright driver as its temp directory (ENOENT mkdtemp '1/…').
        env = connector_environment(
            tmp_path, {"TEMP": "1", "TMP": "rel", "TMPDIR": "/var/tmp"}
        )
        assert "TEMP" not in env
        assert "TMP" not in env
        assert env["TMPDIR"].startswith("/")

    def test_absolute_temp_vars_are_kept_and_pythonpath_prepended(self, tmp_path):
        env = connector_environment(
            tmp_path, {"TEMP": "/fast/tmp", "PYTHONPATH": "/other"}
        )
        assert env["TEMP"] == "/fast/tmp"
        assert env["PYTHONPATH"].startswith(f"{tmp_path}:/other")

    def test_defaults_to_python_tempdir(self, tmp_path):
        import tempfile

        env = connector_environment(tmp_path, {})
        assert env["TMPDIR"] == tempfile.gettempdir()


# ---------------------------------------------------------------------------
# /api/mcp endpoints
# ---------------------------------------------------------------------------


class FakeManager:
    installed = True

    def __init__(self, status: Optional[List[Dict[str, Any]]] = None) -> None:
        self._status = status or []

    def status(self):
        return self._status

    def active_tool_defs(self, ids):
        return [], {}

    def call_tool_sync(self, cid, name, args):
        return "ok"


@pytest.fixture
def mcp_client(monkeypatch, tmp_path):
    from src import config as config_module

    for key in config_module.SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
    # Point discovery at a temp folder holding two stub connectors, so the
    # real mcp_connectors/ dir and any developer externals.json stay untouched.
    connectors_dir = tmp_path / "mcp_connectors"
    connectors_dir.mkdir()
    _write_connector(connectors_dir, "console_mcp", "Console MCP Server.")
    _write_connector(connectors_dir, "playwright_mcp", "Playwright MCP Server.")
    monkeypatch.setattr(mcp_hub, "MCP_CONNECTORS_DIR", connectors_dir)
    monkeypatch.setattr(mcp_hub, "EXTERNALS_PATH", connectors_dir / "externals.json")
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


class TestMcpEndpoints:
    def test_get_lists_connectors_with_stopped_status(self, mcp_client):
        client, _ = mcp_client
        body = client.get("/api/mcp").json()
        assert [c["id"] for c in body["connectors"]] == [
            "console_mcp",
            "playwright_mcp",
        ]
        assert body["connectors"][0]["status"] == "stopped"
        assert body["connectors"][0]["active"] is False
        assert body["maxIterations"] == 25

    def test_get_merges_manager_status(self, mcp_client):
        client, app = mcp_client
        app.state.mcp = FakeManager(
            [{"id": "console_mcp", "status": "running", "error": None}]
        )
        body = client.get("/api/mcp").json()
        assert body["installed"] is True
        by_id = {c["id"]: c for c in body["connectors"]}
        assert by_id["console_mcp"]["status"] == "running"
        assert by_id["playwright_mcp"]["status"] == "stopped"

    def test_post_persists_to_env_file_and_environ(self, mcp_client):
        import os

        from dotenv import dotenv_values

        from src import config as config_module

        client, _ = mcp_client
        response = client.post(
            "/api/mcp", json={"enabled": ["console_mcp"], "maxIterations": 7}
        )
        assert response.status_code == 200
        body = response.json()
        assert [c["id"] for c in body["connectors"] if c["active"]] == ["console_mcp"]
        assert body["maxIterations"] == 7
        assert os.environ["MCP_ENABLED"] == "console_mcp"
        assert os.environ["MAX_MCP_ITERATION"] == "7"
        file_values = dotenv_values(config_module.ENV_PATH)
        assert file_values["MCP_ENABLED"] == "console_mcp"
        assert file_values["MAX_MCP_ITERATION"] == "7"

    def test_post_unknown_connector_returns_422(self, mcp_client):
        client, _ = mcp_client
        response = client.post("/api/mcp", json={"enabled": ["nope_mcp"]})
        assert response.status_code == 422
        assert "nope_mcp" in response.text

    def test_post_invalid_max_iterations_returns_422(self, mcp_client):
        client, _ = mcp_client
        assert client.post("/api/mcp", json={"maxIterations": 0}).status_code == 422
        assert client.post("/api/mcp", json={"maxIterations": 9999}).status_code == 422

    def test_post_empty_list_deactivates_all(self, mcp_client):
        import os

        client, _ = mcp_client
        client.post("/api/mcp", json={"enabled": ["console_mcp"], "maxIterations": 5})
        response = client.post("/api/mcp", json={"enabled": [], "maxIterations": 5})
        assert response.status_code == 200
        # Empty selection writes a commented-out line and clears the env key.
        assert "MCP_ENABLED" not in os.environ
        assert not [c for c in response.json()["connectors"] if c["active"]]


# ---------------------------------------------------------------------------
# External connectors through the API
# ---------------------------------------------------------------------------


EXTERNALS_DOC = {
    "context7": {
        "type": "remote",
        "enable": True,
        "url": "https://mcp.context7.com/mcp",
        "headers": {"CONTEXT7_API_KEY": "ctx7sk-secret-value"},
    },
    "chrome-devtools": {
        "type": "local",
        "enable": False,
        "command": ["npx", "-y", "chrome-devtools-mcp@latest"],
    },
}


class RecordingManager:
    installed = True

    def __init__(self, statuses: Dict[str, str]) -> None:
        self._statuses = dict(statuses)
        self.started: List[str] = []
        self.stopped: List[str] = []

    def status(self):
        return [
            {"id": cid, "status": status, "error": None}
            for cid, status in self._statuses.items()
        ]

    def active_tool_defs(self, ids):
        return [], {}

    def connector_status(self, connector_id):
        return self._statuses.get(connector_id)

    def start_connector(self, connector_id):
        self.started.append(connector_id)
        self._statuses[connector_id] = "starting"

    def stop_connector(self, connector_id):
        self.stopped.append(connector_id)
        self._statuses[connector_id] = "stopped"


class TestExternalConnectorEndpoints:
    def _externals_file(self, tmp_path, doc=None) -> Path:
        return _write_externals(
            tmp_path / "mcp_connectors" / "externals.json",
            EXTERNALS_DOC if doc is None else doc,
        )

    def test_get_lists_externals_with_enable_as_active(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, _ = mcp_client
        body = client.get("/api/mcp").json()
        ids = [c["id"] for c in body["connectors"]]
        # Built-ins first (discovery order), then externals sorted by id.
        assert ids == ["console_mcp", "playwright_mcp", "chrome-devtools", "context7"]
        by_id = {c["id"]: c for c in body["connectors"]}
        assert by_id["console_mcp"]["external"] is False
        remote = by_id["context7"]
        assert remote["external"] is True and remote["kind"] == "remote"
        assert remote["active"] is True  # enable: true
        assert by_id["chrome-devtools"]["active"] is False  # enable: false
        assert by_id["chrome-devtools"]["command"] == [
            "npx",
            "-y",
            "chrome-devtools-mcp@latest",
        ]

    def test_get_never_leaks_external_secrets(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, _ = mcp_client
        response = client.get("/api/mcp")
        assert "ctx7sk-secret-value" not in response.text
        assert "CONTEXT7_API_KEY" not in response.text

    def test_post_writes_enable_back_but_not_into_mcp_enabled(self, mcp_client, tmp_path):
        import os

        from dotenv import dotenv_values

        from src import config as config_module

        path = self._externals_file(tmp_path)
        client, _ = mcp_client
        response = client.post(
            "/api/mcp", json={"enabled": ["console_mcp", "context7"], "maxIterations": 9}
        )
        assert response.status_code == 200
        # Externals never enter MCP_ENABLED — only the built-in stays.
        assert os.environ["MCP_ENABLED"] == "console_mcp"
        assert "context7" not in dotenv_values(config_module.ENV_PATH)["MCP_ENABLED"]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["context7"]["enable"] is True
        assert data["chrome-devtools"]["enable"] is False
        assert data["context7"]["url"] == "https://mcp.context7.com/mcp"  # preserved
        by_id = {c["id"]: c for c in response.json()["connectors"]}
        assert by_id["context7"]["active"] is True

    def test_post_enabling_only_one_external_disables_the_other(self, mcp_client, tmp_path):
        path = self._externals_file(tmp_path)
        client, _ = mcp_client
        client.post("/api/mcp", json={"enabled": ["chrome-devtools"]})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["chrome-devtools"]["enable"] is True
        assert data["context7"]["enable"] is False

    def test_post_unknown_connector_still_422_with_externals(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, _ = mcp_client
        response = client.post("/api/mcp", json={"enabled": ["ghost"]})
        assert response.status_code == 422

    def test_corrupt_file_surfaces_externals_error_and_survives_post(self, mcp_client, tmp_path):
        path = tmp_path / "mcp_connectors" / "externals.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{oops", encoding="utf-8")
        client, _ = mcp_client
        body = client.get("/api/mcp").json()
        assert "not readable JSON" in body["externalsError"]
        response = client.post("/api/mcp", json={"enabled": ["console_mcp"], "maxIterations": 9})
        assert response.status_code == 200
        assert "not readable JSON" in response.json()["externalsError"]
        assert path.read_text(encoding="utf-8") == "{oops"

    def test_enable_start_and_disable_stop_are_reconciled(self, mcp_client, tmp_path):
        self._externals_file(tmp_path)
        client, app = mcp_client
        hub = RecordingManager(
            {
                "console_mcp": "running",
                "playwright_mcp": "running",
                "context7": "running",
                "chrome-devtools": "stopped",
            }
        )
        app.state.mcp = hub
        client.post(
            "/api/mcp",
            json={"enabled": ["console_mcp", "playwright_mcp", "context7", "chrome-devtools"]},
        )
        assert hub.started == ["chrome-devtools"]  # newly enabled
        assert hub.stopped == []

        client.post("/api/mcp", json={"enabled": ["console_mcp", "playwright_mcp"]})
        # Both externals now go down: context7 was running, chrome starting.
        assert hub.stopped == ["chrome-devtools", "context7"]
        assert hub.started == ["chrome-devtools"]  # unchanged set not restarted

    def test_chat_uses_mcp_enabled_plus_enabled_externals(self, mcp_client, tmp_path, monkeypatch):
        import os

        self._externals_file(tmp_path)
        monkeypatch.setenv("MCP_ENABLED", "console_mcp")
        assert routes._active_connector_ids() == ["console_mcp", "context7"]


# ---------------------------------------------------------------------------
# stream_chat tool loop
# ---------------------------------------------------------------------------


def content_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def tool_call_chunk(call_id, name, arguments, index=0):
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                }
            }
        ]
    }


def usage_chunk(completion=None, prompt=None, total=None):
    usage: Dict[str, int] = {}
    if completion is not None:
        usage["completion_tokens"] = completion
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if total is not None:
        usage["total_tokens"] = total
    return {"choices": [], "usage": usage}


class TestChatToolLoop:
    def _run(self, monkeypatch, completions, executor=None, **kwargs):
        calls: List[List[Dict[str, Any]]] = []
        generators = iter(completions)

        def fake_completion(**call_kwargs):
            calls.append(call_kwargs["messages"])
            self.tools_arg = call_kwargs.get("tools")
            return iter(next(generators))

        monkeypatch.setattr("src.core.chat.completion", fake_completion)
        events: List[Dict[str, Any]] = []
        chat_core.stream_chat(
            [{"role": "user", "content": "run it"}],
            "model-a",
            "http://server:8080/v1",
            events.append,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "execute_command", "description": "",
                                 "parameters": {}},
                }
            ],
            tool_executor=executor or (lambda name, args: "42"),
            tool_servers={"execute_command": "console_mcp"},
            **kwargs,
        )
        return events, calls

    def test_tool_call_then_final_answer(self, monkeypatch):
        seen_args = {}

        def executor(name, args):
            seen_args["call"] = (name, args)
            return "42"

        events, calls = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", '{"command": "ls"}'),
                 usage_chunk(completion=8)],
                [content_chunk("The answer is 42"), usage_chunk(completion=5)],
            ],
            executor=executor,
        )
        names = [e["event"] for e in events]
        # `usage` closes the round's *generation*, so it precedes the tool
        # execution: the badges tick as soon as generation stops.
        assert names == ["usage", "tool_call", "tool_result", "token", "done"]
        assert seen_args["call"] == ("execute_command", {"command": "ls"})
        tool_call = events[1]
        assert tool_call["tool"] == "execute_command"
        assert tool_call["server"] == "console_mcp"
        assert tool_call["iteration"] == 1
        assert tool_call["maxIterations"] == 25
        assert events[2]["ok"] is True
        # Cumulative usage of the completed round, before the final answer.
        assert events[0]["tokens"] == 8
        # usage.contextTokens: the just-ended round's prompt + completion
        # (its total_tokens, char-estimated here since the stub reports
        # completion only) — the round's own context occupancy figure.
        assert events[0]["contextTokens"] >= 8
        assert events[0]["contextEstimated"] is True
        # done carries the cumulative decode window for an accurate tok/s.
        assert events[-1]["decode"] >= 0
        # Second request carries the assistant tool_calls + tool result turn.
        second = calls[1]
        assistant = [m for m in second if m.get("tool_calls")]
        assert assistant and assistant[0]["tool_calls"][0]["function"]["name"] == (
            "execute_command"
        )
        tool_turn = [m for m in second if m.get("role") == "tool"]
        assert tool_turn[0]["content"] == "42"
        assert events[-1]["tokens"] == 13  # summed across iterations

    def test_split_tool_call_deltas_reassembled(self, monkeypatch):
        events, _ = self._run(
            monkeypatch,
            [
                [
                    tool_call_chunk("c1", "execute", "", index=0),
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "_command",
                                                "arguments": '{"a": ',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    tool_call_chunk(None, None, '1}', index=0),
                ],
                [content_chunk("done")],
            ],
            executor=lambda name, args: "x",
        )
        call = next(e for e in events if e["event"] == "tool_call")
        assert call["tool"] == "execute_command"
        assert call["id"] == "c1"

    def test_invalid_tool_args_feedback(self, monkeypatch):
        events, calls = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{oops")],
                [content_chunk("fixed")],
            ],
        )
        result = next(e for e in events if e["event"] == "tool_result")
        assert result["ok"] is False
        tool_turn = [m for m in calls[1] if m.get("role") == "tool"][0]
        assert "not valid JSON" in tool_turn["content"]

    def test_executor_exception_becomes_tool_result_error(self, monkeypatch):
        def boom(name, args):
            raise RuntimeError("connector down")

        events, _ = self._run(
            monkeypatch,
            [[tool_call_chunk("c1", "execute_command", "{}")], [content_chunk("ok")]],
            executor=boom,
        )
        result = next(e for e in events if e["event"] == "tool_result")
        assert result["ok"] is False
        assert "connector down" in result["preview"]

    def test_iteration_and_usage_events_across_rounds(self, monkeypatch):
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(completion=8)],
                [tool_call_chunk("c2", "execute_command", "{}"),
                 usage_chunk(completion=5)],
                [content_chunk("done")],
            ],
            max_tool_iterations=10,
        )
        names = [e["event"] for e in events]
        assert names == [
            "usage", "tool_call", "tool_result",
            "usage", "tool_call", "tool_result",
            "token", "done",
        ]
        calls = [e for e in events if e["event"] == "tool_call"]
        assert [e["iteration"] for e in calls] == [1, 2]
        assert all(e["maxIterations"] == 10 for e in calls)
        usages = [e for e in events if e["event"] == "usage"]
        assert [e["tokens"] for e in usages] == [8, 13]  # cumulative
        assert all(e["decode"] >= 0 for e in usages)
        # Occupancy is NOT a cumulative sum: each usage event carries its
        # OWN round's prompt + completion, never a running total — and is
        # char-estimated here, since the stub reports completion only.
        contexts = [e["contextTokens"] for e in usages]
        assert all(e["contextEstimated"] is True for e in usages)  # no totals
        # Each figure covers its OWN round (round 1: 8 generated, round 2:
        # 5 — not the cumulative 13), plus that round's estimated prompt.
        assert all(c > t for c, t in zip(contexts, [8, 5]))

    def test_context_tokens_is_last_round_not_cumulative(self, monkeypatch):
        # Both rounds report full usage: contextTokens is ONLY the final
        # round's total — the figure comparable to the server's context
        # window, never the sum of the rounds' (re-billed) totals.
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(prompt=100, completion=8, total=108)],
                [content_chunk("done"),
                 usage_chunk(prompt=120, completion=5, total=125)],
            ],
        )
        usage_event = next(e for e in events if e["event"] == "usage")
        assert usage_event["contextTokens"] == 108
        assert usage_event["contextEstimated"] is False
        done = events[-1]
        assert done["contextTokens"] == 125
        assert done["contextEstimated"] is False

    def test_iteration_cap_stops_with_note(self, monkeypatch):
        forever = [[tool_call_chunk(f"c{i}", "execute_command", "{}")] for i in range(5)]
        forever.append([content_chunk("never reached")])
        events, calls = self._run(
            monkeypatch,
            forever,
            max_tool_iterations=2,
        )
        # Two iterations of one tool call each, then the cap note.
        tool_calls = [e for e in events if e["event"] == "tool_call"]
        assert len(tool_calls) == 2
        assert tool_calls[-1]["iteration"] == 2  # gauge reads max on the cap round
        assert len(calls) == 2
        note = next(e for e in events if e["event"] == "token")
        assert "tool iterations reached" in note["text"]
        assert events[-1]["event"] == "done"

    def test_cancel_between_iterations_stops_silently(self, monkeypatch):
        cancel = threading.Event()

        def executor(name, args):
            cancel.set()  # user stopped while the tool was running
            return "42"

        events, calls = self._run(
            monkeypatch,
            [[tool_call_chunk("c1", "execute_command", "{}")], [content_chunk("x")]],
            executor=executor,
            cancel_event=cancel,
        )
        assert all(
            e["event"] in ("usage", "tool_call", "tool_result") for e in events
        )
        assert "done" not in [e["event"] for e in events]
        assert len(calls) == 1

    def test_context_is_round_total_with_full_usage(self, monkeypatch):
        # Each round reports usage with a total: its occupancy figure is
        # exactly that round's total (prompt re-billed every round, because
        # each tool round re-sends the whole history). Round totals here:
        # 18, then 25, then 25 — never the 68 sum across rounds.
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(completion=8, prompt=10, total=18)],
                [tool_call_chunk("c2", "execute_command", "{}"),
                 usage_chunk(completion=5, prompt=20, total=25)],
                [content_chunk("done"), usage_chunk(completion=0, prompt=25,
                                                   total=25)],
            ],
        )
        usages = [e for e in events if e["event"] == "usage"]
        assert [e["contextTokens"] for e in usages] == [18, 25]
        assert all(e["contextEstimated"] is False for e in usages)
        done = events[-1]
        assert done["tokens"] == 13
        assert done["contextTokens"] == 25
        assert done["contextEstimated"] is False

    def test_tool_call_arguments_extend_decode_window(self, monkeypatch):
        # A pure tool-call round streams no content and no reasoning: the
        # argument deltas must still open/extend the decode window (their
        # tokens are in completion_tokens), and the cumulative done.decode
        # must sum the rounds' windows without any tool-execution time.
        clock = [
            0.0,   # t0
            1.0,   # first tool-call argument delta (round 1 opens)
            3.0,   # second argument delta (round 1 closes at 2.0s)
            20.0,  # first content delta of the final round (opens)
            22.0,  # second content delta (final round closes at 2.0s)
        ]
        monkeypatch.setattr("src.core.chat.time.monotonic", lambda: clock.pop(0))
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", '{"a"'),
                 tool_call_chunk("c1", None, ': 1}')],
                [content_chunk("He"), content_chunk("llo")],
            ],
            # The 17 s gap between t=3 and t=20 stands for the tool run +
            # re-prefill: the clock is never sampled there, so none of it
            # may leak into the decode window.
            executor=lambda name, args: "42",
        )
        assert clock == []
        done = events[-1]
        assert done["decode"] == 4.0  # (3-1) + (22-20), tool time excluded
        assert done["took"] == 22.0  # wall clock does include the tool run
        assert done["ttft"] == 1.0

    def test_reasoning_deltas_interleave_with_tool_frames(self, monkeypatch):
        # The Chat tab's left panel renders a chronological timeline: the
        # panel relies on reasoning_delta frames arriving between the rounds'
        # tool_call frames, in stream order.
        events, _ = self._run(
            monkeypatch,
            [
                [{"choices": [{"delta": {"reasoning_content": "plan"}}]},
                 tool_call_chunk("c1", "execute_command", "{}")],
                [{"choices": [{"delta": {"reasoning_content": "again"}}]},
                 tool_call_chunk("c2", "execute_command", "{}")],
                [content_chunk("ok")],
            ],
        )
        names = [e["event"] for e in events]
        assert names == [
            "reasoning",            # one-shot flag (first thinking delta)
            "reasoning_delta",      # round 1 flow
            "usage",
            "tool_call", "tool_result",
            "reasoning_delta",      # round 2 flow, after the tool round
            "usage",
            "tool_call", "tool_result",
            "token", "done",
        ]
        texts = [e["text"] for e in events if e["event"] == "reasoning_delta"]
        assert texts == ["plan", "again"]

    def test_usage_precedes_tool_execution(self, monkeypatch):
        # The usage frame is emitted when the round's generation ends, so a
        # slow tool cannot freeze the token badge for its whole execution.
        order = []

        def executor(name, args):
            order.append("execute")
            return "42"

        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}")],
                [content_chunk("ok")],
            ],
            executor=executor,
        )
        names = [e["event"] for e in events]
        assert names.index("usage") < names.index("tool_result")

    def test_multi_round_usage_tokens_summed_without_completion(self, monkeypatch):
        # Servers omitting completion_tokens: the per-round (total - prompt)
        # deltas must add up across rounds — the last round's usage alone
        # would drop the tool rounds' tokens.
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "{}"),
                 usage_chunk(prompt=10, total=18)],
                [tool_call_chunk("c2", "execute_command", "{}"),
                 usage_chunk(prompt=20, total=25)],
                [content_chunk("done"), usage_chunk(prompt=27, total=32)],
            ],
        )
        done = events[-1]
        assert done["tokens"] == 8 + 5 + 5  # per-round deltas summed
        assert done["tokensEstimated"] is False

    def test_char_estimate_covers_tool_arguments_without_usage(self, monkeypatch):
        # No usage anywhere: the fallback estimate counts answer text AND
        # tool-call argument text (16 + 16 chars here -> 8 tokens).
        events, _ = self._run(
            monkeypatch,
            [
                [tool_call_chunk("c1", "execute_command", "x" * 16)],
                [content_chunk("y" * 16)],
            ],
        )
        done = events[-1]
        assert done["tokens"] == 8
        assert done["tokensEstimated"] is True

    def test_tool_notice_appended_to_system_prompt(self, monkeypatch):
        events, calls = self._run(monkeypatch, [[content_chunk("hi")]])
        assert chat_core.TOOL_NOTICE.strip() in calls[0][0]["content"]
        assert calls[0][0]["role"] == "system"

    def test_tool_notice_skips_prompt_that_already_mentions_tools(self, monkeypatch):
        events, calls = self._run(
            monkeypatch,
            [[content_chunk("hi")]],
            sysprompt="You may use your tools freely.",
        )
        assert calls[0][0]["content"] == "You may use your tools freely."

    def test_no_tools_keeps_single_shot_shape(self, monkeypatch):
        monkeypatch.setattr(
            "src.core.chat.completion",
            lambda **kwargs: iter([content_chunk("hi")])
            if "tools" not in kwargs
            else iter([]),
        )
        events: List[Dict[str, Any]] = []
        chat_core.stream_chat(
            [{"role": "user", "content": "hi"}], "m", "http://s", events.append
        )
        assert [e["event"] for e in events] == ["token", "done"]


# ---------------------------------------------------------------------------
# /api/chat wiring
# ---------------------------------------------------------------------------


class TestChatMcpWiring:
    def test_active_connectors_pass_tools_to_stream_chat(self, mcp_client, monkeypatch):
        client, app = mcp_client
        import os

        os.environ["MCP_ENABLED"] = "console_mcp"

        tools = [
            {"type": "function", "function": {"name": "execute_command",
                                              "description": "", "parameters": {}}}
        ]

        class Hub(FakeManager):
            def active_tool_defs(self, ids):
                self.ids = ids
                return tools, {"execute_command": "console_mcp"}

            def call_tool_sync(self, cid, name, args):
                return f"{cid}:{name}"

        hub = Hub()
        app.state.mcp = hub

        captured: Dict[str, Any] = {}

        def fake_stream(messages, model, server_url, emit, sampling=None,
                        cancel_event=None, api_key="none", sysprompt=None,
                        tools=None, tool_executor=None, tool_servers=None,
                        max_tool_iterations=25):
            captured["tools"] = tools
            captured["tool_servers"] = tool_servers
            captured["max"] = max_tool_iterations
            if tool_executor:
                captured["exec"] = tool_executor("execute_command", {})
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake_stream)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert hub.ids == ["console_mcp"]
        assert captured["tools"] == tools
        assert captured["tool_servers"] == {"execute_command": "console_mcp"}
        assert captured["exec"] == "console_mcp:execute_command"

    def test_nothing_enabled_keeps_call_shape(self, mcp_client, monkeypatch):
        client, app = mcp_client
        monkeypatch.setattr(app.state, "mcp", None)
        captured = {}

        def fake_stream(messages, model, server_url, emit, sampling=None,
                        cancel_event=None, api_key="none", sysprompt=None):
            captured["ok"] = True
            emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
                  "tokensEstimated": False})

        monkeypatch.setattr(routes, "stream_chat", fake_stream)
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert captured["ok"] is True
