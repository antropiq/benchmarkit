import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.models import EndpointsConfig, VerifierAgent
from src.api import routes


def _config() -> EndpointsConfig:
    return EndpointsConfig(
        serverUrl="http://localhost:8080/v1",
        endpoint="model-a",
        verifierAgent=VerifierAgent(endpoint="judge"),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    app = create_app()

    monkeypatch.setattr(
        routes,
        "load_endpoints_config",
        lambda: (_config(), None),
    )
    monkeypatch.setattr(
        routes,
        "load_questions",
        lambda: [
            {"uuid": "q1", "question": "one?", "expected": "yes"},
            {"uuid": "q2", "question": "two?", "expected": "no"},
        ],
    )
    results = tmp_path / "results.json"
    monkeypatch.setattr(routes, "RESULTS_PATH", results)
    monkeypatch.setattr("src.core.utils.RESULTS_PATH", results)
    # The context-window probe hits the configured LLM server: keep the
    # suite hermetic; the real function is unit-tested directly.
    monkeypatch.setattr(
        routes, "_context_window", lambda server_url, api_key, model: None
    )
    # Context-managed so the app event loop lives across requests and
    # background run tasks can progress between polls.
    with TestClient(app) as test_client:
        yield test_client


def _wait_until(
    client: TestClient, predicate: Callable[[Dict[str, Any]], bool], timeout: float = 5.0
) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = client.get("/api/run").json()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"condition not met in {timeout}s: {snapshot}")


class TestConfigEndpoint:
    def test_returns_expected_shape(self, client: TestClient):
        body = client.get("/api/config").json()
        assert body["models"] == ["model-a"]
        assert body["questionCount"] == 2
        assert body["totalTests"] == 2
        assert body["verifierConfigured"] is True
        assert body["runStatus"] == "idle"
        assert body["samplingSummary"] == "defaults"
        assert body["supportsVision"] is False
        # Hermetic fixture stubs the probe: the key must always be present.
        assert body["contextWindow"] is None

    def test_supports_vision_reflects_config(self, client, monkeypatch):
        monkeypatch.setattr(
            routes,
            "load_endpoints_config",
            lambda: (
                EndpointsConfig(
                    serverUrl="http://x/v1", endpoint="m", supports_vision=True
                ),
                None,
            ),
        )
        assert client.get("/api/config").json()["supportsVision"] is True

    def test_invalid_config_returns_422(self, client, monkeypatch):
        from pydantic import ValidationError

        def invalid():
            try:
                EndpointsConfig.model_validate({"reasoning": "bogus"})
            except ValidationError as exc:
                return None, exc
            raise AssertionError("expected validation error")

        monkeypatch.setattr(routes, "load_endpoints_config", invalid)
        response = client.get("/api/config")
        assert response.status_code == 422


class TestContextWindow:
    """Unit tests for the /api/config n_ctx probe (real function, stubbed httpx)."""

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _patch(self, monkeypatch, payload=None, error=None):
        captured = {}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url, headers=None):
                captured["url"] = url
                captured["headers"] = headers
                if error is not None:
                    raise error
                return TestContextWindow._Response(payload)

        monkeypatch.setattr(routes.httpx, "Client", FakeClient)
        return captured

    def test_reads_n_ctx_from_the_matching_model_entry(self, monkeypatch):
        captured = self._patch(
            monkeypatch,
            {
                "data": [
                    {"id": "other", "meta": {"n_ctx": 4096}},
                    {"id": "model-a", "meta": {"n_ctx": 262144}},
                ]
            },
        )
        got = routes._context_window("http://ctx-test:9/v1", "sk-1", "model-a")
        assert got == 262144
        assert captured["url"] == "http://ctx-test:9/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer sk-1"}

    def test_falls_back_to_first_entry_when_model_absent(self, monkeypatch):
        self._patch(monkeypatch, {"data": [{"id": "x", "meta": {"n_ctx": 8192}}]})
        assert routes._context_window("http://ctx-first:9/v1", "", "model-a") == 8192

    def test_missing_meta_or_bad_server_is_none(self, monkeypatch):
        self._patch(monkeypatch, {"data": [{"id": "model-a"}]})
        assert routes._context_window("http://ctx-nometa:9/v1", "", "model-a") is None
        self._patch(monkeypatch, None, error=httpx.ConnectError("boom"))
        assert routes._context_window("http://ctx-down:9/v1", "", "model-a") is None

    def test_success_is_cached_across_failures(self, monkeypatch):
        self._patch(monkeypatch, {"data": [{"id": "m", "meta": {"n_ctx": 32768}}]})
        assert routes._context_window("http://ctx-cache:9/v1", "", "m") == 32768
        self._patch(monkeypatch, None, error=httpx.ConnectError("gone"))
        # Same (url, model) key: the stale success answers while the server is away.
        assert routes._context_window("http://ctx-cache:9/v1", "", "m") == 32768
        # A different key has no cache to fall back on.
        assert routes._context_window("http://ctx-cache:9/v1", "", "other") is None

    def test_empty_server_url_is_none(self, monkeypatch):
        assert routes._context_window("", "none", "model-a") is None


def patch_bench(monkeypatch, run_fn=None, block: Optional[threading.Event] = None):
    def fake(questions, config, job, emit, cancel_event=None):
        if block is not None:
            block.wait()
        for text in ("Hel", "lo"):
            emit({"event": "token", "text": text})
        job.add_result("model-a", "q1", True, "Hello", 1.0, 10)
        job.add_result("model-a", "q2", False, "nope", 2.0, 20)
        job.current_index = 2
        emit({"event": "run_finished", "totalRunningTime": 3.0, "totalTests": 2})

    monkeypatch.setattr(routes, "run_benchmark", run_fn or fake)


class TestRunLifecycle:
    def test_start_streams_events_and_finishes(self, client, monkeypatch):
        patch_bench(monkeypatch)
        started = client.post("/api/run")
        assert started.status_code == 202
        run_id = started.json()["runId"]

        snapshot = _wait_until(client, lambda s: s["status"] != "running")
        assert snapshot["status"] == "finished"
        assert snapshot["runId"] == run_id
        names = [item["event"]["event"] for item in snapshot["events"]]
        assert names[0] == "run_started"
        assert "token" in names
        assert names[-1] == "run_finished"

    def test_second_run_while_running_returns_409(self, client, monkeypatch):
        gate = threading.Event()
        patch_bench(monkeypatch, block=gate)
        try:
            assert client.post("/api/run").status_code == 202
            _wait_until(client, lambda s: s["status"] == "running")
            assert client.post("/api/run").status_code == 409
        finally:
            gate.set()
        _wait_until(client, lambda s: s["status"] != "running")

    def test_run_failure_emits_run_error(self, client, monkeypatch):
        def boom(questions, config, job, emit, cancel_event=None):
            raise RuntimeError("kaboom")

        patch_bench(monkeypatch, run_fn=boom)
        client.post("/api/run")
        snapshot = _wait_until(client, lambda s: s["status"] != "running")
        assert snapshot["status"] == "failed"
        assert snapshot["error"] == "kaboom"
        names = [item["event"]["event"] for item in snapshot["events"]]
        assert "run_error" in names

    def test_stop_cancels_running_run(self, client, monkeypatch):
        from src.core.llm import RunCancelled

        entered = threading.Event()

        def slow(questions, config, job, emit, cancel_event=None):
            entered.set()
            deadline = time.time() + 5
            while time.time() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    raise RunCancelled()
                time.sleep(0.02)
            raise AssertionError("run was not cancelled in time")

        patch_bench(monkeypatch, run_fn=slow)
        assert client.post("/api/run").status_code == 202
        assert entered.wait(5)
        assert client.post("/api/run/stop").status_code == 202

        snapshot = _wait_until(client, lambda s: s["status"] != "running")
        assert snapshot["status"] == "cancelled"
        last = snapshot["events"][-1]["event"]
        assert last["event"] == "run_finished"
        assert last.get("cancelled") is True

    def test_stop_without_run_returns_409(self, client: TestClient):
        assert client.post("/api/run/stop").status_code == 409

    def test_missing_models_returns_400(self, client, monkeypatch):
        empty = EndpointsConfig(serverUrl="http://x/v1")
        monkeypatch.setattr(routes, "load_endpoints_config", lambda: (empty, None))
        response = client.post("/api/run")
        assert response.status_code == 400


class TestResultsEndpoint:
    def test_missing_results_returns_404(self, client: TestClient):
        assert client.get("/api/results").status_code == 404

    def test_downloads_results(self, client: TestClient):
        # The fixture patched routes.RESULTS_PATH; write through the same path.
        routes.RESULTS_PATH.write_text('{"results": []}', encoding="utf-8")
        response = client.get("/api/results")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


class TestEventsEndpoint:
    def test_events_replay_after_finished_run(self, client, monkeypatch):
        patch_bench(monkeypatch)
        client.post("/api/run")
        _wait_until(client, lambda s: s["status"] != "running")

        with client.stream("GET", "/api/run/events") as response:
            assert response.status_code == 200
            body = response.read().decode("utf-8")
        assert "event: run_started" in body
        assert "event: run_finished" in body
        assert "event: stream_end" in body

    def test_events_without_run_returns_404(self, client: TestClient):
        assert client.get("/api/run/events").status_code == 404


class TestSettingsEndpoint:
    @pytest.fixture(autouse=True)
    def isolated_settings_env(self, monkeypatch, tmp_path):
        from src import config as config_module

        for key in config_module.SETTING_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")

    @staticmethod
    def _payload(**overrides):
        payload = {
            "serverUrl": "http://target:8080/v1",
            "apiKey": "secret-key",
            "endpoint": "model-a",
            "sysprompt": "Answer briefly.",
            "temp": "0.2",
            "topP": "",
            "topK": "",
            "minP": "",
            "repeatPenalty": "",
            "presencePenalty": "",
            "reasoning": "low",
            "verifier": {
                "serverUrl": "http://judge:8080/v1",
                "apiKey": "judge-key",
                "endpoint": "judge",
                "sysprompt": 'Say "true" or "false" only.',
            },
        }
        payload.update(overrides)
        return payload

    def test_get_returns_env_values(self, client, monkeypatch):
        monkeypatch.setenv("SERVER_URL", "http://x/v1")
        monkeypatch.setenv("SERVER_URL_API_KEY", "sk-target")
        monkeypatch.setenv("LLM_SYSPROMPT", "Answer concisely.")
        monkeypatch.setenv("VERIFIER_AGENT_SYSPROMPT", "verify!")
        monkeypatch.setenv("VERIFIER_AGENT_SERVERURL_API_KEY", "sk-judge")
        body = client.get("/api/settings").json()
        assert body["serverUrl"] == "http://x/v1"
        assert body["apiKey"] == "sk-target"
        assert body["endpoint"] == ""
        assert body["sysprompt"] == "Answer concisely."
        assert body["verifier"]["sysprompt"] == "verify!"
        assert body["verifier"]["apiKey"] == "sk-judge"

    def test_get_empty_when_nothing_set(self, client):
        from src.core.models import DEFAULT_LLM_SYSPROMPT

        body = client.get("/api/settings").json()
        assert body["serverUrl"] == ""
        assert body["apiKey"] == ""
        # Unset LLM_SYSPROMPT resolves to the built-in default, like QUESTION_SET.
        assert body["sysprompt"] == DEFAULT_LLM_SYSPROMPT
        assert body["verifier"] == {
            "serverUrl": "",
            "apiKey": "",
            "endpoint": "",
            "sysprompt": "",
        }

    def test_save_writes_file_and_environ(self, client):
        from dotenv import dotenv_values

        from src import config as config_module

        response = client.post("/api/settings", json=self._payload())
        assert response.status_code == 200
        assert response.json()["endpoint"] == "model-a"

        import os

        assert os.environ["SERVER_URL"] == "http://target:8080/v1"
        assert os.environ["SERVER_URL_API_KEY"] == "secret-key"
        assert os.environ["LLM_SYSPROMPT"] == "Answer briefly."
        assert os.environ["VERIFIER_AGENT_SERVERURL_API_KEY"] == "judge-key"
        assert os.environ["REASONING"] == "low"
        assert "TOP_P" not in os.environ

        file_values = dotenv_values(config_module.ENV_PATH)
        assert file_values["SERVER_URL"] == "http://target:8080/v1"
        assert file_values["SERVER_URL_API_KEY"] == "secret-key"
        assert file_values["LLM_SYSPROMPT"] == "Answer briefly."
        assert file_values["VERIFIER_AGENT_SERVERURL_API_KEY"] == "judge-key"
        # The quoted sysprompt must survive the .env round-trip verbatim.
        assert file_values["VERIFIER_AGENT_SYSPROMPT"] == 'Say "true" or "false" only.'
        assert file_values.get("TOP_P") is None
        text = config_module.ENV_PATH.read_text(encoding="utf-8")
        assert "#TOP_P=" in text

    def test_save_removes_emptied_optional_key(self, client, monkeypatch):
        from src import config as config_module

        monkeypatch.setenv("TEMP", "0.5")
        response = client.post("/api/settings", json=self._payload(temp=""))
        assert response.status_code == 200
        import os

        assert "TEMP" not in os.environ
        text = config_module.ENV_PATH.read_text(encoding="utf-8")
        assert "#TEMP=" in text

    def test_save_normalizes_supports_vision(self, client):
        import os

        from dotenv import dotenv_values

        from src import config as config_module

        response = client.post(
            "/api/settings", json=self._payload(supportsVision="true")
        )
        assert response.status_code == 200
        assert response.json()["supportsVision"] == "1"
        assert os.environ["SUPPORTS_VISION"] == "1"
        assert dotenv_values(config_module.ENV_PATH)["SUPPORTS_VISION"] == "1"

    def test_save_normalizes_litellm_debug_and_applies_it(self, client, monkeypatch):
        """The debug switch round-trips and is pushed onto the live process."""
        import os

        from dotenv import dotenv_values

        from src import config as config_module

        applied = []
        monkeypatch.setattr(
            config_module, "apply_litellm_debug", lambda: applied.append(True)
        )
        response = client.post(
            "/api/settings", json=self._payload(litellmDebug="true")
        )
        assert response.status_code == 200
        # "true" from a hand-written client becomes the canonical .env form.
        assert response.json()["litellmDebug"] == "1"
        assert os.environ["LITELLM_DEBUG"] == "1"
        assert dotenv_values(config_module.ENV_PATH)["LITELLM_DEBUG"] == "1"
        # Saving re-applies the switch, so no restart is needed to get debug
        # output on the console.
        assert applied == [True]

    def test_save_unsets_blank_supports_vision(self, client, monkeypatch):
        import os

        from src import config as config_module

        monkeypatch.setenv("SUPPORTS_VISION", "1")
        response = client.post("/api/settings", json=self._payload(supportsVision=""))
        assert response.status_code == 200
        assert "SUPPORTS_VISION" not in os.environ
        text = config_module.ENV_PATH.read_text(encoding="utf-8")
        assert "#SUPPORTS_VISION=" in text

    def test_save_emptied_llm_sysprompt_falls_back_to_default(self, client, monkeypatch):
        from src import config as config_module
        from src.core.models import DEFAULT_LLM_SYSPROMPT

        assert client.post("/api/settings", json=self._payload()).status_code == 200
        response = client.post("/api/settings", json=self._payload(sysprompt=""))
        assert response.status_code == 200
        import os

        assert "LLM_SYSPROMPT" not in os.environ
        text = config_module.ENV_PATH.read_text(encoding="utf-8")
        assert "#LLM_SYSPROMPT=" in text
        # An unset LLM_SYSPROMPT is reported back as the built-in default.
        assert response.json()["sysprompt"] == DEFAULT_LLM_SYSPROMPT

    def test_save_invalid_value_returns_422(self, client):
        from src import config as config_module

        response = client.post("/api/settings", json=self._payload(temp="abc"))
        assert response.status_code == 422
        assert not config_module.ENV_PATH.exists()

    def test_save_invalid_reasoning_returns_422(self, client):
        response = client.post("/api/settings", json=self._payload(reasoning="ultra"))
        assert response.status_code == 422

    def test_save_while_running_returns_409(self, client):
        client.app.state.run_state.status = "running"
        response = client.post("/api/settings", json=self._payload())
        assert response.status_code == 409
        client.app.state.run_state.status = "idle"

    def test_saved_settings_are_used_by_config_endpoint(self, client, monkeypatch):
        from src import config as config_module

        monkeypatch.setattr(
            routes, "load_endpoints_config", config_module.load_endpoints_config
        )
        assert client.post("/api/settings", json=self._payload()).status_code == 200
        body = client.get("/api/config").json()
        assert body["models"] == ["model-a"]
        assert "temperature=0.2" in body["samplingSummary"]
        assert body["verifierConfigured"] is True

    def test_save_accepts_json_numbers_and_nulls(self, client):
        import os

        payload = {
            "serverUrl": "http://target:8080/v1",
            "endpoint": "model-a",
            "temp": 0.2,
            "topK": 20,
            "topP": None,
            "verifier": None,
            "unknownKey": "ignored",
        }
        response = client.post("/api/settings", json=payload)
        assert response.status_code == 200
        assert os.environ["TEMP"] == "0.2"
        assert os.environ["TOP_K"] == "20"
        assert "TOP_P" not in os.environ
        assert os.environ["ENDPOINT"] == "model-a"

    def test_save_reasoning_none_is_omitted(self, client):
        import os

        payload = self._payload(reasoning="none")
        response = client.post("/api/settings", json=payload)
        assert response.status_code == 200
        assert "REASONING" not in os.environ


class TestModelsEndpoint:
    class _FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                import httpx

                raise httpx.ConnectError("boom")

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, captured, response):
            self.captured = captured
            self.response = response

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            self.captured["url"] = url
            self.captured["headers"] = headers or {}
            self.response.raise_for_status()
            return self.response

    def _patch_httpx(self, monkeypatch, response):
        captured: Dict[str, Any] = {}
        fake_cls = self._FakeClient(captured, response)

        def factory(*args, **kwargs):
            return fake_cls

        monkeypatch.setattr(routes.httpx, "AsyncClient", factory)
        return captured

    def test_lists_models_from_v1_url(self, client, monkeypatch):
        payload = {"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}, {"nope": 1}]}
        captured = self._patch_httpx(monkeypatch, self._FakeResponse(payload))
        body = client.get(
            "/api/models", params={"serverUrl": "http://llm:8080/v1"}
        ).json()
        assert body == {"models": ["a", "b"]}
        assert captured["url"] == "http://llm:8080/v1/models"
        assert captured["headers"] == {}

    def test_sends_authorization_header_with_api_key(self, client, monkeypatch):
        captured = self._patch_httpx(monkeypatch, self._FakeResponse({"data": []}))
        client.get(
            "/api/models",
            params={"serverUrl": "http://llm:8080/v1", "apiKey": "sk-42"},
        )
        assert captured["headers"] == {"Authorization": "Bearer sk-42"}

    def test_blank_api_key_sends_no_header(self, client, monkeypatch):
        captured = self._patch_httpx(monkeypatch, self._FakeResponse({"data": []}))
        client.get(
            "/api/models",
            params={"serverUrl": "http://llm:8080/v1", "apiKey": "  "},
        )
        assert captured["headers"] == {}

    def test_appends_v1_models_for_base_url(self, client, monkeypatch):
        captured = self._patch_httpx(monkeypatch, self._FakeResponse({"data": []}))
        response = client.get(
            "/api/models", params={"serverUrl": "http://llm:8080/"}
        )
        assert response.json() == {"models": []}
        assert captured["url"] == "http://llm:8080/v1/models"

    def test_empty_url_returns_422(self, client):
        assert client.get("/api/models").status_code == 422
        assert client.get("/api/models", params={"serverUrl": "  "}).status_code == 422

    def test_non_http_scheme_returns_422(self, client):
        response = client.get(
            "/api/models", params={"serverUrl": "file:///etc/passwd"}
        )
        assert response.status_code == 422

    def test_upstream_failure_returns_502(self, client, monkeypatch):
        self._patch_httpx(monkeypatch, self._FakeResponse({}, status_code=500))
        response = client.get(
            "/api/models", params={"serverUrl": "http://llm:8080/v1"}
        )
        assert response.status_code == 502


class _RecordingHub:
    """Minimal hub double: records restarts, answers what the payload needs."""

    installed = True

    def __init__(self) -> None:
        self.restarts: List[Any] = []

    async def restart(self, *args) -> None:
        self.restarts.append(args)

    def status(self) -> List[Dict[str, Any]]:
        return []


@pytest.fixture
def hub_client(client, monkeypatch, tmp_path):
    """``client`` with recording hubs and the declaration files redirected.

    ``BENCHMARKIT_PATH_LOCATION`` is cleared so each test starts from the boot
    default (the install dir) rather than a value another test left behind.
    """
    from src.core import lsphub
    from src.core import mcphub

    monkeypatch.delenv("BENCHMARKIT_PATH_LOCATION", raising=False)
    empty = tmp_path / "externals.json"
    empty.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mcphub.constants, "EXTERNALS_PATH", empty)
    servers_dir = tmp_path / "lsp_connectors"
    servers_dir.mkdir()
    # Hermetic copy of the workspace-root deny-list: the real file's contents
    # belong to the developer's machine, /etc is the stand-in tests assert on.
    (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
    monkeypatch.setattr(lsphub.constants, "LSP_CONNECTORS_DIR", servers_dir)
    monkeypatch.setattr(
        lsphub.constants, "LSP_SERVERS_PATH", servers_dir / "lsp_servers.json"
    )
    monkeypatch.setattr(
        lsphub.constants, "LSP_RESTRICTED_PATHS", servers_dir / "restricted-paths.txt"
    )
    client.app.state.mcp = _RecordingHub()
    client.app.state.lsp = _RecordingHub()
    return client


class TestPathLocationEndpoint:
    def test_get_defaults_to_the_install_dir(self, hub_client):
        from src.config import default_path_location

        body = hub_client.get("/api/path-location").json()
        assert body["key"] == "BENCHMARKIT_PATH_LOCATION"
        assert body["path"] == default_path_location()
        assert body["default"] == default_path_location()

    def test_post_sets_the_env_and_restarts_both_hubs(self, hub_client, tmp_path):
        target = tmp_path / "other-project"
        target.mkdir()
        response = hub_client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        assert os.environ["BENCHMARKIT_PATH_LOCATION"] == str(target.resolve())
        assert response.json()["path"] == str(target.resolve())
        assert len(hub_client.app.state.mcp.restarts) == 1
        assert len(hub_client.app.state.lsp.restarts) == 1
        # Both hub payloads ride back in the response so the dialog can
        # repaint its panels without a second round-trip.
        assert "connectors" in response.json()["mcp"]
        assert "servers" in response.json()["lsp"]

    def test_post_canonicalises_symlinks_and_dots(self, hub_client, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        response = hub_client.post("/api/path-location", json={"path": f"{link}/."})
        assert response.status_code == 200
        assert response.json()["path"] == str(real.resolve())

    def test_rejects_missing_relative_empty_and_denied(self, hub_client, tmp_path):
        before = os.environ.get("BENCHMARKIT_PATH_LOCATION")
        assert hub_client.post("/api/path-location", json={"path": ""}).status_code == 422
        assert hub_client.post("/api/path-location", json={"path": "rel"}).status_code == 422
        assert (
            hub_client.post(
                "/api/path-location", json={"path": str(tmp_path / "ghost")}
            ).status_code
            == 422
        )
        assert (
            hub_client.post("/api/path-location", json={"path": "/etc"}).status_code
            == 422
        )
        # Nothing was touched: the hubs never saw a restart, env is unchanged.
        assert os.environ.get("BENCHMARKIT_PATH_LOCATION") == before
        assert hub_client.app.state.mcp.restarts == []
        assert hub_client.app.state.lsp.restarts == []

    def test_refused_during_a_run_leaves_env_untouched(self, hub_client, tmp_path):
        before = os.environ.get("BENCHMARKIT_PATH_LOCATION")
        hub_client.app.state.run_state.status = "running"
        try:
            response = hub_client.post("/api/path-location", json={"path": str(tmp_path)})
        finally:
            hub_client.app.state.run_state.status = "idle"
        assert response.status_code == 409
        assert os.environ.get("BENCHMARKIT_PATH_LOCATION") == before

