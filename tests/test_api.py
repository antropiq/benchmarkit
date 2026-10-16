import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
            "endpoint": "model-a",
            "temp": "0.2",
            "topP": "",
            "topK": "",
            "minP": "",
            "repeatPenalty": "",
            "presencePenalty": "",
            "reasoning": "low",
            "verifier": {
                "serverUrl": "http://judge:8080/v1",
                "endpoint": "judge",
                "sysprompt": 'Say "true" or "false" only.',
            },
        }
        payload.update(overrides)
        return payload

    def test_get_returns_env_values(self, client, monkeypatch):
        monkeypatch.setenv("SERVER_URL", "http://x/v1")
        monkeypatch.setenv("VERIFIER_AGENT_SYSPROMPT", "verify!")
        body = client.get("/api/settings").json()
        assert body["serverUrl"] == "http://x/v1"
        assert body["endpoint"] == ""
        assert body["verifier"]["sysprompt"] == "verify!"

    def test_get_empty_when_nothing_set(self, client):
        body = client.get("/api/settings").json()
        assert body["serverUrl"] == ""
        assert body["verifier"] == {"serverUrl": "", "endpoint": "", "sysprompt": ""}

    def test_save_writes_file_and_environ(self, client):
        from dotenv import dotenv_values

        from src import config as config_module

        response = client.post("/api/settings", json=self._payload())
        assert response.status_code == 200
        assert response.json()["endpoint"] == "model-a"

        import os

        assert os.environ["SERVER_URL"] == "http://target:8080/v1"
        assert os.environ["REASONING"] == "low"
        assert "TOP_P" not in os.environ

        file_values = dotenv_values(config_module.ENV_PATH)
        assert file_values["SERVER_URL"] == "http://target:8080/v1"
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

        async def get(self, url):
            self.captured["url"] = url
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
