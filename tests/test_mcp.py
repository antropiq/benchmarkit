"""Tests for MCP connector discovery, settings API, and the chat tool loop."""

import json
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
from src.core.mcp_hub import (
    McpManager,
    connector_environment,
    discover_connectors,
    extract_tool_result_text,
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
        assert connector["port"] == 9999
        assert connector["endpoint"] == "http://127.0.0.1:9999/mcp"

    def test_ignores_other_files_and_broken_modules(self, tmp_path):
        _write_connector(tmp_path, "playwright_mcp", "Browser.")
        (tmp_path / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "helper.py").write_text("", encoding="utf-8")
        (tmp_path / "broken_mcp.py").write_text("def (:", encoding="utf-8")
        ids = [c["id"] for c in discover_connectors(tmp_path)]
        assert ids == ["playwright_mcp"]  # unparsable modules are skipped

    def test_missing_directory_returns_empty(self, tmp_path):
        assert discover_connectors(tmp_path / "nope") == []


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
    from src.core.mcp_hub import CONNECTOR_PORTS

    return [
        {
            "id": cid,
            "label": cid,
            "description": "",
            "port": CONNECTOR_PORTS.get(cid, 0),
            "endpoint": f"http://127.0.0.1:{CONNECTOR_PORTS.get(cid, 0)}/mcp",
        }
        for cid in connector_ids
    ]


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
        assert status["playwright_mcp"]["port"] == 8934

    def test_stop_all_is_idempotent_without_start(self):
        manager = _manager()
        import asyncio

        asyncio.run(manager.stop_all())
        asyncio.run(manager.stop_all())  # second call must not raise


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
    monkeypatch.setattr(
        routes,
        "discover_connectors",
        lambda: [
            {
                "id": "console_mcp",
                "label": "Console",
                "description": "Console MCP Server.",
                "port": 9999,
                "endpoint": "http://127.0.0.1:9999/mcp",
            },
            {
                "id": "playwright_mcp",
                "label": "Playwright",
                "description": "Playwright MCP Server.",
                "port": 8934,
                "endpoint": "http://127.0.0.1:8934/mcp",
            },
        ],
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
        assert names == ["tool_call", "tool_result", "token", "done"]
        assert seen_args["call"] == ("execute_command", {"command": "ls"})
        tool_call = events[0]
        assert tool_call["tool"] == "execute_command"
        assert tool_call["server"] == "console_mcp"
        assert events[1]["ok"] is True
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

    def test_iteration_cap_stops_with_note(self, monkeypatch):
        forever = [[tool_call_chunk(f"c{i}", "execute_command", "{}")] for i in range(5)]
        forever.append([content_chunk("never reached")])
        events, calls = self._run(
            monkeypatch,
            forever,
            max_tool_iterations=2,
        )
        # Two iterations of one tool call each, then the cap note.
        assert len([e for e in events if e["event"] == "tool_call"]) == 2
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
        assert all(e["event"] in ("tool_call", "tool_result") for e in events)
        assert "done" not in [e["event"] for e in events]
        assert len(calls) == 1

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
