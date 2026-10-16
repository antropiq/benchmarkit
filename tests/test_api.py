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
    def fake(questions, config, job, emit):
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
        def boom(questions, config, job, emit):
            raise RuntimeError("kaboom")

        patch_bench(monkeypatch, run_fn=boom)
        client.post("/api/run")
        snapshot = _wait_until(client, lambda s: s["status"] != "running")
        assert snapshot["status"] == "failed"
        assert snapshot["error"] == "kaboom"
        names = [item["event"]["event"] for item in snapshot["events"]]
        assert "run_error" in names

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
