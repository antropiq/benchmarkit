import json
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
from src.core import agent_modes
from src.core import agent_modes_defaults as modes_defaults
from src.core import batch as batch_core
from src.core import editorfs
from src.core import prompt_helpers


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
        from tests.conftest import neutralise_workspace_settings

        for key in config_module.SETTING_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
        # The ``.env`` wipe never reached the workspace store: a settings test
        # must also own the ``.nojae/nojae-settings.json`` its POST writes and
        # its GET reads (it is group 1's home since the split). The write runs
        # before the app boots, so the boot seed finds an existing file and
        # leaves this one alone (exclusive create, D11).
        neutralise_workspace_settings(monkeypatch, tmp_path)

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

    def test_get_returns_values_from_both_stores(self, client, monkeypatch):
        """Group 1 answers from the workspace file, groups 2/3 from ``.env``."""
        monkeypatch.setenv("VERIFIER_AGENT_SYSPROMPT", "verify!")
        monkeypatch.setenv("VERIFIER_AGENT_SERVERURL_API_KEY", "sk-judge")
        body = client.get("/api/settings").json()
        # The fixture's own workspace file (conftest helper), not the env.
        assert body["serverUrl"] == "http://test-server:8080/v1"
        assert body["endpoint"] == "test-model"
        # The state block beside the values (D9's report point): one shape to
        # switch on, carrying the same sentence a chat request would refuse with.
        state = body["workspaceSettings"]
        assert set(state) == {"state", "path", "error", "configured", "message"}
        assert state["state"] == "ok"
        assert state["configured"] is True
        assert body["verifier"]["sysprompt"] == "verify!"
        assert body["verifier"]["apiKey"] == "sk-judge"

    def test_get_answers_native_json_types_for_group_one(self, client):
        """The payload types ws-05 reads: numbers are numbers, never strings.

        The dotenv wart D6 ends here: the file holds ``0.2`` and the dialog is
        answered ``0.2``, while the vision flag is a real boolean.
        """
        body = client.get("/api/settings").json()
        assert body["temp"] == 0.2
        assert not isinstance(body["temp"], str)
        assert body["topK"] == 20
        assert body["supportsVision"] is False
        assert body["questionSet"] == "legacy"

    def test_get_empty_when_nothing_set(self, client):
        from src.core import workspace_settings
        from src.core.models import DEFAULT_LLM_SYSPROMPT

        os.remove(workspace_settings.workspace_settings_path())
        workspace_settings.invalidate_workspace_settings()
        body = client.get("/api/settings").json()
        assert body["serverUrl"] == ""
        assert body["apiKey"] == ""
        # An absent member means what the spec key map says: no sampling
        # number at all (null), and the prompt falls back to the built-in
        # default — which is what the model then receives.
        assert body["temp"] is None
        assert body["supportsVision"] is False
        assert body["sysprompt"] == DEFAULT_LLM_SYSPROMPT
        assert body["workspaceSettings"]["state"] == "absent"
        assert body["verifier"] == {
            "serverUrl": "",
            "apiKey": "",
            "endpoint": "",
            "sysprompt": "",
        }

    def test_save_writes_group_one_to_the_workspace_file_and_two_to_env(
        self, client
    ):
        from dotenv import dotenv_values

        from src import config as config_module
        from src.core import workspace_settings

        response = client.post("/api/settings", json=self._payload())
        assert response.status_code == 200
        assert response.json()["endpoint"] == "model-a"
        assert response.json()["temp"] == 0.2
        assert response.json()["topP"] is None

        # The workspace file holds group 1 and nothing of the verifier's.
        path = workspace_settings.workspace_settings_path()
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        assert stored["serverUrl"] == "http://target:8080/v1"
        assert stored["serverApiKey"] == "secret-key"
        assert stored["endpoint"] == "model-a"
        assert stored["sysprompt"] == "Answer briefly."
        assert stored["temperature"] == 0.2
        assert stored["reasoning"] == "low"
        # A cleared numeric field leaves its member **absent** in the file —
        # not ``null``, not ``""``, not the value the file held before.
        assert "topP" not in stored
        assert "topK" not in stored
        assert "SUPPORTS_VISION" not in stored
        with open(path, encoding="utf-8") as handle:
            assert "judge" not in handle.read()

        # .env holds the verifier and nothing of group 1's.
        file_values = dotenv_values(config_module.ENV_PATH)
        assert file_values["VERIFIER_AGENT_SERVERURL_API_KEY"] == "judge-key"
        # The quoted sysprompt must survive the .env round-trip verbatim.
        assert file_values["VERIFIER_AGENT_SYSPROMPT"] == 'Say "true" or "false" only.'
        text = config_module.ENV_PATH.read_text(encoding="utf-8")
        assert "target:8080" not in text
        assert "secret-key" not in text
        assert "TEMPERATURE" not in text
        assert "SUPPORTS_VISION" not in text

        # The live environment carries the verifier half as its only managed
        # names: ``load_settings`` — the managed ``.env`` view — cannot even
        # answer a group-1 key any more (a leftover in the ambient environment
        # is dead bytes, D2, and reading it is the config suite's test).
        assert os.environ["VERIFIER_AGENT_SERVERURL_API_KEY"] == "judge-key"
        assert "SERVER_URL" not in config_module.load_settings()
        assert "REASONING" not in config_module.load_settings()
        assert "TOP_P" not in config_module.load_settings()

    def test_save_cleared_field_removes_a_member_the_file_held(self, client):
        """The dialog owns every group-1 member, so clearing one clears it.

        ``save_workspace_settings`` carries members over when they are not
        given — right for ``post_lsp``, wrong here. The fixture's file holds
        the template's ``temperature: 0.2``; a save with an empty ``temp``
        must leave the member absent, and the next GET must answer ``null``.
        """
        from src.core import workspace_settings

        response = client.post("/api/settings", json=self._payload(temp=""))
        assert response.status_code == 200
        with open(workspace_settings.workspace_settings_path(), encoding="utf-8") as f:
            stored = json.load(f)
        assert "temperature" not in stored
        assert response.json()["temp"] is None
        # The members the dialog does not own survived the rebuild.
        assert stored["maxToolIteration"] == 300
        assert stored["lspEnabled"] == []

    def test_save_supports_vision_is_a_native_boolean(self, client):
        from src.core import workspace_settings

        # Interim until ws-05 sends a real boolean: the checkbox still sends
        # the .env spellings, and they become a real ``true`` in the file.
        response = client.post(
            "/api/settings", json=self._payload(supportsVision="true")
        )
        assert response.status_code == 200
        assert response.json()["supportsVision"] is True
        with open(workspace_settings.workspace_settings_path(), encoding="utf-8") as f:
            assert json.load(f)["supportsVision"] is True

        # An empty box omits the member — absent means false, not ``"0"``.
        response = client.post(
            "/api/settings", json=self._payload(supportsVision="")
        )
        assert response.status_code == 200
        assert response.json()["supportsVision"] is False
        with open(workspace_settings.workspace_settings_path(), encoding="utf-8") as f:
            assert "supportsVision" not in json.load(f)

    def test_llm_debug_survives_a_save_that_never_mentions_it(
        self, client, monkeypatch
    ):
        """The dialog no longer carries ``LLM_DEBUG`` (spec section 7) — and a
        save must not switch it off.

        The trap this pins: ``_settings_env_values`` once *always* emitted the
        key, and the ``{**load_settings(), **env_values}`` merge let the
        emitted ``""`` win over the stored value — so every settings save
        commented ``LLM_DEBUG`` out and popped it from the environment. The key
        is hand-edited like ``BENCHMARKIT_PYTHON_ROOT`` now; a save that never
        mentions it re-applies the value the file still has.
        """
        import os

        from dotenv import dotenv_values

        from src import config as config_module

        monkeypatch.setenv(config_module.LLM_DEBUG_KEY, "1")
        applied = []
        monkeypatch.setattr(
            config_module, "apply_llm_debug", lambda: applied.append(True)
        )
        response = client.post("/api/settings", json=self._payload())
        assert response.status_code == 200
        # Stored, live in the environment, and re-applied to the process —
        # without the dialog ever naming the key.
        assert dotenv_values(config_module.ENV_PATH)[config_module.LLM_DEBUG_KEY] == "1"
        assert os.environ[config_module.LLM_DEBUG_KEY] == "1"
        assert applied == [True]
        # The field is gone from the API in both directions.
        assert "llmDebug" not in response.json()
        assert "llmDebug" not in client.get("/api/settings").json()

    def test_save_unticked_debug_clears_the_legacy_key_too(self, client, monkeypatch):
        """A save ends the fallback, because the file it writes no longer holds it.

        Boot loaded the legacy key from a hand-written .env; the next save must
        switch the logging off *now*, not only after a restart that never reads
        that key again — the dialog no longer carries the switch, so "a save"
        is any settings save, not one that unticks a box.
        """
        import os

        from src import config as config_module

        monkeypatch.setenv(config_module.LEGACY_LLM_DEBUG_KEY, "1")
        response = client.post("/api/settings", json=self._payload())
        assert response.status_code == 200
        assert config_module.LEGACY_LLM_DEBUG_KEY not in os.environ
        assert "LLM_DEBUG" not in os.environ
        assert config_module.llm_debug_enabled() is False
        text = config_module.ENV_PATH.read_text(encoding="utf-8")
        assert "#LLM_DEBUG=" in text
        assert config_module.LEGACY_LLM_DEBUG_KEY not in text

    def test_save_emptied_sysprompt_answers_the_default(self, client, monkeypatch):
        from src import config as config_module
        from src.core import workspace_settings
        from src.core.models import DEFAULT_LLM_SYSPROMPT

        assert client.post("/api/settings", json=self._payload()).status_code == 200
        response = client.post("/api/settings", json=self._payload(sysprompt=""))
        assert response.status_code == 200
        with open(workspace_settings.workspace_settings_path(), encoding="utf-8") as f:
            assert "sysprompt" not in json.load(f)
        text = config_module.ENV_PATH.read_text(encoding="utf-8")
        assert "LLM_SYSPROMPT" not in text
        assert "LLM_SYSPROMPT" not in config_module.load_settings()
        # An absent sysprompt is reported back as the built-in default, which
        # is what the model then receives.
        assert response.json()["sysprompt"] == DEFAULT_LLM_SYSPROMPT

    def test_save_invalid_value_returns_422(self, client):
        """The gate runs before **either** write (spec section 5)."""
        from src import config as config_module
        from src.core import workspace_settings

        path = workspace_settings.workspace_settings_path()
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        response = client.post("/api/settings", json=self._payload(temp="abc"))
        assert response.status_code == 422
        assert not config_module.ENV_PATH.exists()
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == before

    def test_save_invalid_reasoning_returns_422(self, client):
        """A bad ``reasoning`` reaches **neither** disk: one gate, both stores."""
        from src import config as config_module
        from src.core import workspace_settings

        path = workspace_settings.workspace_settings_path()
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        response = client.post("/api/settings", json=self._payload(reasoning="ultra"))
        assert response.status_code == 422
        assert not config_module.ENV_PATH.exists()
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == before

    def test_save_while_running_returns_409(self, client):
        client.app.state.run_state.status = "running"
        response = client.post("/api/settings", json=self._payload())
        assert response.status_code == 409
        client.app.state.run_state.status = "idle"

    def test_saved_settings_are_used_by_config_endpoint(self, client, monkeypatch):
        """Read-your-write across the split: the POST writes what the model
        call reads (spec section 5, the acceptance this whole task exists for)."""
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
        from src.core import workspace_settings

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
        # Numbers arrive as strings through the tolerant mixin and are stored
        # as native JSON again — the file never holds "0.2" or "20".
        with open(workspace_settings.workspace_settings_path(), encoding="utf-8") as f:
            stored = json.load(f)
        assert stored["temperature"] == 0.2
        assert stored["topK"] == 20
        assert "topP" not in stored
        assert stored["endpoint"] == "model-a"
        assert "ENDPOINT" not in routes.load_settings()

    def test_save_reasoning_none_is_omitted(self, client):
        from src.core import workspace_settings

        payload = self._payload(reasoning="none")
        response = client.post("/api/settings", json=payload)
        assert response.status_code == 200
        with open(workspace_settings.workspace_settings_path(), encoding="utf-8") as f:
            assert "reasoning" not in json.load(f)
        assert response.json()["reasoning"] == ""

    def test_get_reflects_the_second_workspace_after_a_switch(self, hub_client, tmp_path):
        """D4's cache invalidation at API level: two roots, one per file.

        The first workspace's saved values must not answer after the app bar
        has moved to the second one — the cached read is keyed per resolved
        path and the switch invalidates it (ws-02 placed the seed +
        invalidation, this proves the settings API follows).
        """
        from tests.conftest import write_workspace_settings

        first = tmp_path / "ws-first"
        first.mkdir()
        write_workspace_settings(first, serverUrl="http://one.example/v1")
        second = tmp_path / "ws-second"
        second.mkdir()
        write_workspace_settings(second, serverUrl="http://two.example/v1")

        assert (
            hub_client.post("/api/path-location", json={"path": str(first)}).status_code
            == 200
        )
        assert hub_client.get("/api/settings").json()["serverUrl"] == "http://one.example/v1"

        assert (
            hub_client.post(
                "/api/path-location", json={"path": str(second)}
            ).status_code
            == 200
        )
        body = hub_client.get("/api/settings").json()
        assert body["serverUrl"] == "http://two.example/v1"
        assert body["workspaceSettings"]["path"].startswith(str(second))


class TestSettingsEndpointInTestMode:
    """The e2e switch at API level (T-D3, T-D5): the dialog reads and refuses.

    With ``BENCHMARKIT_TEST_SETTINGS`` truthy the file **in use** is the suite's
    own hand-authored ``.nojae/nojae-test-settings.json``, so ``GET /api/settings``
    names that path and a group-1 save is refused — the dialog cannot rewrite the
    fixture the next spec depends on.
    """

    @pytest.fixture(autouse=True)
    def in_test_mode(self, monkeypatch, tmp_path):
        from src import config as config_module
        from src.core import workspace_settings as ws
        from tests.conftest import neutralise_workspace_settings

        for key in config_module.SETTING_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
        neutralise_workspace_settings(monkeypatch, tmp_path,
                                      serverUrl="http://dev-live:9/v1")
        # The switch, flipped per test (T-D2): the production file above now
        # answers nothing, because the file in use is the test one.
        monkeypatch.setenv(ws.TEST_SETTINGS_KEY, "1")
        ws.invalidate_workspace_settings()

    def test_get_reports_the_test_file_path(self, client):
        from src.core import workspace_settings as ws

        body = client.get("/api/settings").json()
        state = body["workspaceSettings"]
        assert state["path"].endswith(ws.WORKSPACE_TEST_SETTINGS_FILENAME)
        # The app never seeds the test file, so a workspace in this mode really
        # can be `absent` — the state production cannot reach.
        assert state["state"] == "absent"
        assert state["message"]
        assert ws.WORKSPACE_TEST_SETTINGS_FILENAME in state["message"]
        # The production file's values answer nothing (T-D3, no fallback).
        assert body["serverUrl"] == ""

    def test_a_group_one_save_is_refused_and_neither_file_changes(self, client):
        from src import config as config_module
        from src.core import workspace_settings as ws

        production = ws.workspace_settings_file_path(str(config_module.path_location()))
        with open(production, encoding="utf-8") as handle:
            before = handle.read()
        test_file = Path(ws.workspace_settings_path())
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('{"serverUrl": "http://suite:9/v1"}\n', encoding="utf-8")

        payload = TestSettingsEndpoint._payload()
        response = client.post("/api/settings", json=payload)
        assert response.status_code == 409
        assert ws.WORKSPACE_TEST_SETTINGS_FILENAME in response.json()["detail"]

        with open(production, encoding="utf-8") as handle:
            assert handle.read() == before
        assert test_file.read_text(encoding="utf-8") == '{"serverUrl": "http://suite:9/v1"}\n'
        # And the refusal lands before the .env write too: one refusal, no
        # half-applied save.
        assert not config_module.ENV_PATH.exists()


class TestEnvTrioParity:
    """AGENTS.md's trio rule, as a test: ``SETTING_KEYS``, ``_render_env()``
    and ``.env.example`` move together — ws-04 shrank all three the same way,
    and none of them may name a group-1 key any more.
    """

    # The names the model connection used to live under in ``.env`` — dead
    # bytes now (spec D2), rendered by nothing.
    GROUP_ONE_ENV_NAMES = (
        "SERVER_URL",
        "SERVER_URL_API_KEY",
        "ENDPOINT",
        "LLM_SYSPROMPT",
        "TEMPERATURE",
        "TOP_P",
        "TOP_K",
        "MIN_P",
        "REPEAT_PENALTY",
        "PRESENCE_PENALTY",
        "REASONING",
        "SUPPORTS_VISION",
        "LSP_ENABLED",
        "MAX_TOOL_ITERATION",
    )

    @staticmethod
    def _managed_keys(text: str):
        """The ``KEY=``/``#KEY=`` names a file or render mentions, line by line."""
        names = set()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("#"):
                line = line[1:]
            name, sep, _value = line.partition("=")
            if sep and name.replace("_", "").isalnum() and name == name.upper():
                names.add(name)
        return names

    def test_every_managed_key_is_rendered_and_exemplified(self):
        from src.config import PROJECT_ROOT, SETTING_KEYS, _render_env

        rendered = _render_env({key: "x" for key in SETTING_KEYS})
        assert self._managed_keys(rendered) == set(SETTING_KEYS)
        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert set(SETTING_KEYS) <= self._managed_keys(example)

    def test_no_group_one_key_in_any_of_the_three(self):
        from src.config import PROJECT_ROOT, SETTING_KEYS, _render_env

        rendered = _render_env({})
        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        for name in self.GROUP_ONE_ENV_NAMES:
            assert name not in SETTING_KEYS
            assert name not in self._managed_keys(rendered)
            assert name not in self._managed_keys(example)

    def test_the_pointer_comment_names_the_moved_store(self):
        from src.config import PROJECT_ROOT, _render_env

        for text in (
            _render_env({}),
            (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8"),
        ):
            assert ".nojae/nojae-settings.json" in text
            assert "nojae-settings.default.json" in text

    def test_required_setting_keys_is_empty(self):
        from src.config import REQUIRED_SETTING_KEYS

        assert REQUIRED_SETTING_KEYS == ()


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
    # The switch seeds the new root, and seeding is where the one-time lift
    # looks at the old app-level file. Point that lookup at nothing: this
    # checkout's ``resources/memory_registration.json`` belongs to the machine
    # and would make the seeded state ``migrated`` instead of ``created``.
    monkeypatch.setattr(
        prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", tmp_path / "no-legacy.json"
    )
    empty = tmp_path / "mcp-settings.json"
    empty.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mcphub.constants, "MCP_SETTINGS_PATH", empty)
    servers_dir = tmp_path / "lsp_connectors"
    servers_dir.mkdir()
    # Hermetic copy of the workspace-root deny-list: the real file's contents
    # belong to the developer's machine, /etc is the stand-in tests assert on.
    (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
    monkeypatch.setattr(lsphub.constants, "LSP_CONNECTORS_DIR", servers_dir)
    monkeypatch.setattr(
        lsphub.constants, "LSP_SETTINGS_PATH", servers_dir / "lsp-settings.json"
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
        # am-03/mbl-03: the read half reports each reserved-folder file in the
        # shape the settings dialog renders. A GET seeds nothing, so the install
        # dir answers whatever boot left there — the point is the shape, not the
        # state (that belongs to the seeding suites).
        modes = body["agentModes"]
        assert set(modes) == {"state", "path", "error"}
        assert modes["path"].endswith(
            agent_modes.POLICY_DIRNAME + os.sep + agent_modes.AGENT_MODES_FILENAME
        )
        # mbl-03: the prompt-helpers file is the third block of that answer, and
        # read-only like the other two.
        helpers = body["promptHelpers"]
        assert set(helpers) == {"state", "path", "error"}
        assert helpers["path"].endswith(
            prompt_helpers.POLICY_DIRNAME
            + os.sep
            + prompt_helpers.PROMPT_HELPERS_FILENAME
        )

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
        # am-03: the switch seeds the new root's agent-modes file and says so,
        # beside the policy block (the full seeding behaviour is
        # tests/test_agent_modes_seed.py; this pins the payload key).
        modes = response.json()["agentModes"]
        assert modes["state"] == "created"
        assert modes["error"] is None
        # mbl-03: the prompt-helpers file is seeded at the same moment, and its
        # own state rides back in the same response.
        helpers = response.json()["promptHelpers"]
        assert set(helpers) == {"state", "path", "error"}
        assert helpers["state"] == "created"
        assert helpers["error"] is None
        assert helpers["path"].endswith(
            prompt_helpers.POLICY_DIRNAME
            + os.sep
            + prompt_helpers.PROMPT_HELPERS_FILENAME
        )

    @pytest.mark.requires_symlink(
        "the working-location switch stores the realpath, so a location spelled "
        "through a link and through `.` lands on one canonical directory"
    )
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

    def test_the_deny_list_expands_at_read_time_for_the_switch_too(
        self, hub_client, tmp_path, monkeypatch
    ):
        r"""Consumer 4 of ``restricted-paths.txt`` honours ``${NAME}`` as well.

        The route calls the same ``restricted_lsp_roots()`` the hub and the tree
        call, so the one expansion at read time reaches all three: a line naming
        this profile's credential store refuses a working location inside it, and
        refuses nothing outside it.
        """
        from src.core import lsphub

        monkeypatch.setenv("BENCHMARKIT_TEST_PROFILE", str(tmp_path / "home"))
        store = tmp_path / "home" / ".ssh"
        store.mkdir(parents=True)
        list_file = tmp_path / "restricted-paths.txt"
        list_file.write_text("${BENCHMARKIT_TEST_PROFILE}/.ssh\n", encoding="utf-8")
        monkeypatch.setattr(lsphub.constants, "LSP_RESTRICTED_PATHS", list_file)

        refused = hub_client.post("/api/path-location", json={"path": str(store)})
        assert refused.status_code == 422
        assert "deny-list" in refused.json()["detail"]
        assert hub_client.app.state.mcp.restarts == []
        assert hub_client.app.state.lsp.restarts == []

        other = tmp_path / "project"
        other.mkdir()
        ok = hub_client.post("/api/path-location", json={"path": str(other)})
        assert ok.status_code == 200

    @pytest.mark.case_sensitive_volume(
        "the deny-list names the *other* spelling of the target folder, which "
        "has to stay a name that does not exist for its ``realpath`` to keep "
        "the text; on a volume that folds case the two spellings are one "
        "folder and the exact half of the pair is not the pair's other answer"
    )
    def test_the_deny_list_compare_folds_for_the_root_being_switched_to(
        self, hub_client, tmp_path, monkeypatch
    ):
        r"""win-06 §4: the switch asks the same helper as the tree, with the fold.

        A ``restricted-paths.txt`` line naming one spelling of a folder refuses
        the other spelling of it on a volume where they are one folder — and
        refuses nothing where they are two. The probe consulted is the one for
        the root *being switched to* (``routes.py``), so both halves run against
        this target folder; the exact half is the Linux answer, unchanged.
        """
        from src.core.editorfs import fscase
        from src.core import lsphub

        target = tmp_path / "project"
        target.mkdir()
        list_file = tmp_path / "restricted-paths.txt"
        # The other spelling, a path that does not exist here, so its realpath
        # keeps the text as written.
        list_file.write_text(f"{tmp_path / 'Project'}\n", encoding="utf-8")
        monkeypatch.setattr(lsphub.constants, "LSP_RESTRICTED_PATHS", list_file)

        monkeypatch.setattr(fscase, "case_folded", lambda _root: True)
        refused = hub_client.post("/api/path-location", json={"path": str(target)})
        assert refused.status_code == 422
        assert "deny-list" in refused.json()["detail"]
        assert hub_client.app.state.mcp.restarts == []

        monkeypatch.setattr(fscase, "case_folded", lambda _root: False)
        accepted = hub_client.post("/api/path-location", json={"path": str(target)})
        assert accepted.status_code == 200
        assert os.environ["BENCHMARKIT_PATH_LOCATION"] == str(target.resolve())
        assert len(hub_client.app.state.mcp.restarts) == 1


    def test_refused_during_a_run_leaves_env_untouched(self, hub_client, tmp_path):
        before = os.environ.get("BENCHMARKIT_PATH_LOCATION")
        hub_client.app.state.run_state.status = "running"
        try:
            response = hub_client.post("/api/path-location", json={"path": str(tmp_path)})
        finally:
            hub_client.app.state.run_state.status = "idle"
        assert response.status_code == 409
        assert os.environ.get("BENCHMARKIT_PATH_LOCATION") == before


class TestHubListingsAreModeBlind:
    """am-05 §4: the hub panels answer the *global* state, never per mode.

    A user who ticks connectors in Settings and picks Chat gets no tools, and
    the panels are where they would look for the reason. The answer is *not* a
    per-mode listing — that would make Settings a function of the dropdown and
    two routes would report two truths. So the mode question is answered once,
    by ``GET /api/agent-modes`` (am-06) carrying ``mcptools``/``lsptools``, and
    these routes gain nothing at all: same payload, and no mode parameter.
    """

    def test_hub_listings_take_no_mode_parameter(self, hub_client):
        spec = hub_client.get("/openapi.json").json()["paths"]
        for route in ("/api/mcp", "/api/lsp"):
            operation = spec[route]["get"]
            names = {p["name"] for p in operation.get("parameters", [])}
            assert names == set(), names

    def test_hub_listing_payload_has_no_mode_dependent_field(self, hub_client):
        # The payload keys are exactly the pre-mode set: the panel's ``active``
        # flag is the Settings verdict, which a mode can only narrow, so nothing
        # in here can depend on which mode the composer happens to hold.
        mcp = hub_client.get("/api/mcp").json()
        lsp = hub_client.get("/api/lsp").json()
        assert set(mcp) == {
            "installed", "connectors", "maxIterations", "externalsError",
        }
        assert set(lsp) == {
            "installed", "servers", "maxIterations", "serversError",
        }

    def test_the_mode_flags_are_reported_once_by_the_mode_reader(self, hub_client):
        # One route answers the mode question: the two booleans are on the mode
        # objects am-01 resolves, which is all the frontend needs to gray a
        # panel out — so no hub route has to know about modes.
        assert set(("mcptools", "lsptools")) <= set(agent_modes.MODE_KEYS)
        resolved = agent_modes.resolve_modes()["modes"]
        assert resolved, "the built-in defaults must always resolve"
        for mode in resolved:
            assert isinstance(mode.mcptools, bool)
            assert isinstance(mode.lsptools, bool)
        # And the hub listings carry no mode anywhere (keys pinned above).
        assert "mode" not in str(hub_client.get("/api/mcp").json()).lower()
        assert "mode" not in str(hub_client.get("/api/lsp").json()).lower()


def _modes_document(agents: Dict[str, Any], default: str = "code") -> Dict[str, Any]:
    """One well-formed modes document, spelled with its JSON keys."""
    return {
        "version": agent_modes.AGENT_MODES_VERSION,
        "default": default,
        "agents": agents,
    }


def _write_modes(root: Path, document: Dict[str, Any]) -> Path:
    """Write ``<root>/.nojae/agent-modes.json`` by hand (the app never does)."""
    folder = root / agent_modes.POLICY_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / agent_modes.AGENT_MODES_FILENAME
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def mode_root(client, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A scratch working location with no modes file, cache forgotten.

    Every agent-modes test needs a root it owns: the alternative is answering
    about the developer's real ``.nojae``, whose contents belong to their
    machine. The invalidation on both sides keeps another test's cached entry
    from being read here, and this test's from being read there.
    """
    root = tmp_path / "mode-workspace"
    root.mkdir()
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
    agent_modes.invalidate_agent_modes_cache()
    try:
        yield root
    finally:
        agent_modes.invalidate_agent_modes_cache()


class TestAgentModesEndpoint:
    """am-06 §3: ``GET /api/agent-modes`` is the app's only reader of the file."""

    def test_with_no_file_the_four_built_in_modes_answer(self, client, mode_root):
        response = client.get("/api/agent-modes")
        assert response.status_code == 200
        body = response.json()
        assert [mode["id"] for mode in body["modes"]] == [
            modes_defaults.CHAT_MODE_ID, modes_defaults.CODE_MODE_ID,
            modes_defaults.ARCHITECT_MODE_ID, modes_defaults.TODO_BUILDER_MODE_ID,
        ]
        assert [mode["label"] for mode in body["modes"]] == [
            modes_defaults.CHAT_LABEL, modes_defaults.CODE_LABEL,
            modes_defaults.ARCHITECT_LABEL, modes_defaults.TODO_BUILDER_LABEL,
        ]
        # D9 as data, not as an id the client hardcodes: the app-owned mode is
        # last and the only one not editable.
        assert body["modes"][-1]["editable"] is False
        assert all(mode["editable"] for mode in body["modes"][:-1])
        assert body["default"] == modes_defaults.CODE_MODE_ID
        assert body["findings"] == []
        # am-03 seeds on demand only: a GET never leaves a file behind.
        assert body["source"] == {
            "path": str(mode_root / agent_modes.POLICY_DIRNAME
                        / agent_modes.AGENT_MODES_FILENAME),
            "exists": False,
            "loaded": False,
            "version": None,
        }
        assert not (mode_root / agent_modes.POLICY_DIRNAME).exists()

    def test_the_two_booleans_are_the_hub_gate_the_mode_reports(
        self, client, mode_root
    ):
        # am-05 §4: these four flags are all the frontend needs to gray a hub
        # panel out, so they have to match the gate ``_chat_tool_kwargs`` applies.
        client.get("/api/agent-modes")
        resolved = {mode.id: mode for mode in agent_modes.resolve_modes()["modes"]}
        body = client.get("/api/agent-modes").json()
        for mode in body["modes"]:
            assert mode["mcptools"] is resolved[mode["id"]].mcptools
            assert mode["lsptools"] is resolved[mode["id"]].lsptools

    def test_a_mode_object_carries_exactly_the_five_published_fields(
        self, client, mode_root
    ):
        agent_modes.ensure_agent_modes_file(str(mode_root))
        for mode in client.get("/api/agent-modes").json()["modes"]:
            assert set(mode) == {"id", "label", "mcptools", "lsptools", "editable"}

    def test_the_seeded_file_is_reported_as_loaded_with_no_findings(
        self, client, mode_root
    ):
        state = agent_modes.ensure_agent_modes_file(str(mode_root))
        assert state["state"] == "created"
        body = client.get("/api/agent-modes").json()
        assert body["source"]["exists"] is True
        assert body["source"]["loaded"] is True
        assert body["source"]["version"] == agent_modes.AGENT_MODES_VERSION
        assert body["findings"] == [], "the seeded file must load cleanly"
        assert body["default"] == modes_defaults.CODE_MODE_ID

    def test_a_hand_broken_file_still_answers_200_and_names_the_pointer(
        self, client, mode_root
    ):
        # D11 on the HTTP edge: the dropdown has to render *and* say which
        # pointer is wrong, so the payload is the defaults plus the findings —
        # a 4xx/5xx would hide the fix (am-00's open question, settled here).
        _write_modes(mode_root, _modes_document({
            "chat": {"label": "Chat", "mcptools": False, "lsptools": False},
            "code": {"label": "Code", "mcptools": True, "lsptools": True},
            "architect": {"label": "Architect", "mcptool": True},
        }))
        response = client.get("/api/agent-modes")
        assert response.status_code == 200
        body = response.json()
        assert len(body["findings"]) == 1
        finding = body["findings"][0]
        assert set(finding) == {"kind", "message"}
        assert finding["kind"] == "error"
        assert "agents.architect.mcptool" in finding["message"]
        assert "unknown key" in finding["message"]
        # The good entries stand; the bad one is gone; the app-owned mode is
        # still appended.
        assert [mode["id"] for mode in body["modes"]] == [
            "chat", "code", modes_defaults.TODO_BUILDER_MODE_ID,
        ]

    def test_the_payload_carries_no_persona_and_no_generated_blocks(
        self, client, mode_root
    ):
        # §3's deliberate absences, checked on the serialised payload rather
        # than on the dict, so a refactor that adds the fields back by accident
        # fails here instead of in review.
        agent_modes.ensure_agent_modes_file(str(mode_root))
        blob = json.dumps(client.get("/api/agent-modes").json())
        assert "systemMessage" not in blob
        assert "instructions" not in blob
        # And the persona text itself is nowhere, under either spelling.
        assert modes_defaults.CODE_PERSONA[0] not in blob
        assert batch_core.DEFAULT_LOT_DIR not in blob

    def test_the_route_is_a_reader_and_leaves_nothing_behind(
        self, client, mode_root
    ):
        before = sorted(p.name for p in mode_root.iterdir())
        for _ in range(3):
            assert client.get("/api/agent-modes").status_code == 200
        assert sorted(p.name for p in mode_root.iterdir()) == before

    def test_the_route_takes_no_query_parameter(self, client, mode_root):
        # §4: it answers *about* the file, and a ``?path=`` would turn the one
        # purpose-built reader into a general way to reach the reserved folder.
        spec = client.get("/openapi.json").json()["paths"]
        operation = spec["/api/agent-modes"]["get"]
        assert {p["name"] for p in operation.get("parameters", [])} == set()
        assert set(operation["responses"]) == {"200"}

    def test_the_reserved_mode_is_listed_whatever_the_file_says(
        self, client, mode_root
    ):
        # Two different files, both list it (and last, and not editable): the
        # app-owned mode cannot be removed, renamed or redefined by a file.
        _write_modes(mode_root, _modes_document(
            {"planner": {"label": "Planner", "mcptools": True}},
            default="planner",
        ))
        first = client.get("/api/agent-modes").json()
        assert [mode["id"] for mode in first["modes"]] == [
            "planner", modes_defaults.TODO_BUILDER_MODE_ID,
        ]
        assert first["default"] == "planner"

        _write_modes(mode_root, _modes_document({
            "chat": {"label": "My Chat", "mcptools": False, "lsptools": False},
            modes_defaults.TODO_BUILDER_MODE_ID: {"label": "Mine"},
        }, default="chat"))
        # Only the note about the reserved entry: the file's own modes loaded,
        # and its ``default`` names one of them (the loader's own rules).
        second = client.get("/api/agent-modes").json()
        assert [mode["id"] for mode in second["modes"]] == [
            "chat", modes_defaults.TODO_BUILDER_MODE_ID,
        ]
        # The file's entry is ignored, benchmarkit's definition answers (D9),
        # and the note says so instead of failing the request.
        builder = second["modes"][-1]
        assert builder["label"] == modes_defaults.TODO_BUILDER_LABEL
        assert builder["editable"] is False
        assert [finding["kind"] for finding in second["findings"]] == ["note"]
        assert "cannot be set here" in second["findings"][0]["message"]

    def test_the_dropdown_order_is_the_files_order(self, client, mode_root):
        _write_modes(mode_root, _modes_document({
            "zulu": {"label": "Zulu"},
            "alpha": {"label": "Alpha"},
            "mike": {"label": "Mike"},
        }, default="mike"))
        body = client.get("/api/agent-modes").json()
        assert [mode["id"] for mode in body["modes"]] == [
            "zulu", "alpha", "mike", modes_defaults.TODO_BUILDER_MODE_ID,
        ]
        assert body["default"] == "mike"


def _modes_file(root: Path) -> Path:
    """The scratch root's modes file, built from the module's own names."""
    return root / agent_modes.POLICY_DIRNAME / agent_modes.AGENT_MODES_FILENAME


@pytest.fixture
def mode_backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private backup folder, so a modes save never reads the app's real chain.

    The save route hands the writer ``editorfs.backup_for_write``, which resolves
    ``BACKUP_DIR`` at call time — the same pin ``tests/test_editorfs_write.py``'s
    ``backups`` fixture uses, so this suite asserts the one chain.
    """
    folder = tmp_path / "app-backups"
    monkeypatch.setattr(editorfs.constants, "BACKUP_DIR", folder)
    return folder


class TestAgentModesDocumentEndpoint:
    """eds-03: ``GET /api/agent-modes/document`` is the editor's read (D2)."""

    def test_it_takes_no_query_parameter(self, client, mode_root):
        # The reader's rule, unchanged: a purpose-built reader for one file, never
        # a way to reach the reserved folder through a parameter.
        spec = client.get("/openapi.json").json()["paths"]
        operation = spec["/api/agent-modes/document"]["get"]
        assert {p["name"] for p in operation.get("parameters", [])} == set()

    def test_an_absent_file_answers_the_defaults_draft_and_a_null_mtime(
        self, client, mode_root
    ):
        response = client.get("/api/agent-modes/document")
        assert response.status_code == 200
        body = response.json()
        assert body["entries"] == []
        assert body["default"] is None
        assert "readme" not in body, "an absent file has no readme to preserve"
        assert body["findings"] == []
        assert body["source"] == {
            "path": str(_modes_file(mode_root)),
            "exists": False,
            "loaded": False,
            "mtime": None,
        }
        # D3's one legitimate draft: the built-in document, and the caps the form
        # enforces — no cap is spelled in the client, so they travel with the file.
        assert body["defaults"]["agents"]
        assert body["caps"] == {
            "maxModes": agent_modes.MAX_MODES,
            "maxIdChars": agent_modes.MAX_MODE_ID_CHARS,
            "idPattern": agent_modes.MODE_ID_PATTERN,
            "maxLabelChars": agent_modes.MAX_MODE_LABEL_CHARS,
            "maxPersonaBytes": agent_modes.MAX_MODE_SYSTEM_MESSAGE_BYTES,
            "maxPersonaLines": agent_modes.MAX_MODE_PERSONA_LINES,
        }
        # A GET never seeds (the same rule as the dropdown's reader).
        assert not _modes_file(mode_root).exists()

    def test_the_files_own_entries_answer_in_the_files_order_with_personas(
        self, client, mode_root
    ):
        _write_modes(mode_root, {
            "version": agent_modes.AGENT_MODES_VERSION,
            "readme": ["Hand-written prose.", "Preserved verbatim."],
            "default": "zulu",
            "agents": {
                "zulu": {"label": "Zulu", "mcptools": True, "lsptools": False,
                         "systemMessage": "Zulu persona line."},
                "alpha": {"label": "Alpha", "mcptools": False, "lsptools": True,
                          "systemMessage": ["Alpha line one.", "Alpha line two."]},
            },
        })
        body = client.get("/api/agent-modes/document").json()
        assert [entry["key"] for entry in body["entries"]] == ["zulu", "alpha"]
        zulu, alpha = body["entries"]
        # Persona verbatim, in whichever JSON form the file wrote it — never joined.
        assert zulu["fields"]["systemMessage"] == "Zulu persona line."
        assert alpha["fields"]["systemMessage"] == ["Alpha line one.", "Alpha line two."]
        assert zulu["hasForm"] is True and zulu["findings"] == []
        assert body["default"] == "zulu"
        assert body["readme"] == ["Hand-written prose.", "Preserved verbatim."]
        assert body["findings"] == []
        assert body["source"]["exists"] is True
        assert body["source"]["loaded"] is True
        # The token a save compares against, in the string spelling the read hands out.
        assert body["source"]["mtime"] == str(
            _modes_file(mode_root).stat().st_mtime_ns
        )

    def test_a_hand_broken_file_lists_the_broken_entry_with_its_finding(
        self, client, mode_root
    ):
        # D2's contrast with the dropdown's reader, on the same file: the reader
        # drops the entry (three modes answer), this read shows it (four entries),
        # because a dialog fed by the reader would save back three.
        _write_modes(mode_root, _modes_document({
            "chat": {"label": "Chat", "mcptools": False, "lsptools": False},
            "code": {"label": "Code", "mcptools": True, "lsptools": True},
            "architect": {"label": "Architect", "mcptool": True},
        }))
        response = client.get("/api/agent-modes/document")
        assert response.status_code == 200
        body = response.json()
        assert [entry["key"] for entry in body["entries"]] == [
            "chat", "code", "architect",
        ]
        broken = body["entries"][2]
        assert broken["hasForm"] is True, "the tab is a normal one; the finding holds Save"
        assert len(broken["findings"]) == 1
        finding = broken["findings"][0]
        assert set(finding) == {"kind", "message"}
        assert finding["kind"] == "error"
        assert "agents.architect.mcptool" in finding["message"]
        assert body["findings"] == [finding]
        # The reader, same file, same moment: the entry is gone from its list.
        assert [mode["id"] for mode in client.get("/api/agent-modes").json()["modes"]] == [
            "chat", "code", modes_defaults.TODO_BUILDER_MODE_ID,
        ]

    def test_an_entry_with_no_fields_to_edit_is_still_listed(self, client, mode_root):
        # D3's raw-key tab: a value that is not an object and a key outside the
        # pattern both get a tab with Delete only — visible, which is what keeps a
        # save from quietly dropping them.
        _write_modes(mode_root, _modes_document({
            "code": {"label": "Code"},
            "Broken Id": "not an object",
        }))
        body = client.get("/api/agent-modes/document").json()
        assert [entry["key"] for entry in body["entries"]] == ["code", "Broken Id"]
        odd = body["entries"][1]
        assert odd["hasForm"] is False
        assert odd["fields"] == {}
        assert [finding["kind"] for finding in odd["findings"]] == ["error"]

    def test_the_document_payload_carries_no_app_owned_entry(self, client, mode_root):
        # D4: the dialog never learns the reserved id. A hand-written entry for it
        # keeps today's shape — a note naming the pointer, no tab — and the note is
        # server data, not something to filter out.
        agent_modes.ensure_agent_modes_file(str(mode_root))
        body = client.get("/api/agent-modes/document").json()
        assert [entry["key"] for entry in body["entries"]] == [
            modes_defaults.CHAT_MODE_ID, modes_defaults.CODE_MODE_ID,
            modes_defaults.ARCHITECT_MODE_ID,
        ]

        _write_modes(mode_root, _modes_document({
            "chat": {"label": "My Chat"},
            modes_defaults.TODO_BUILDER_MODE_ID: {"label": "Mine"},
        }, default="chat"))
        second = client.get("/api/agent-modes/document").json()
        assert [entry["key"] for entry in second["entries"]] == ["chat"]
        assert [finding["kind"] for finding in second["findings"]] == ["note"]
        assert "cannot be set here" in second["findings"][0]["message"]

    def test_the_document_read_is_not_cached(self, client, mode_root):
        # The save compares against this token, so a cached mtime would defeat the
        # conflict guard: a hand-edit between two reads has to move it.
        _write_modes(mode_root, _modes_document({"code": {"label": "Code"}}))
        first = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        _write_modes(mode_root, _modes_document({
            "code": {"label": "Code"}, "extra": {"label": "Extra"},
        }, default="extra"))
        second = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        assert first != second


class TestAgentModesSaveEndpoint:
    """eds-03: ``POST /api/agent-modes/save`` is the whole-document write."""

    def test_a_save_writes_the_file_and_answers_the_new_verdict(
        self, client, mode_root, mode_backups
    ):
        _write_modes(mode_root, _modes_document({"code": {"label": "Code"}}))
        token = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        response = client.post("/api/agent-modes/save", json={
            "document": {
                "default": "reviewer",
                "agents": {
                    "code": {"label": "Code", "mcptools": True, "lsptools": True},
                    "reviewer": {"label": "Reviewer", "mcptools": False,
                                 "lsptools": False,
                                 "systemMessage": ["Review carefully."]},
                },
            },
            "mtime": token,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["saved"] is True and body["created"] is False
        assert body["backedUp"] is True
        # D9: the new verdict in the dropdown's shape (the app-owned mode appended
        # last like every other verdict), so no second round-trip.
        assert [mode["id"] for mode in body["modes"]] == [
            "code", "reviewer", modes_defaults.TODO_BUILDER_MODE_ID,
        ]
        assert body["default"] == "reviewer"
        assert body["findings"] == []
        assert body["source"]["exists"] is True
        assert body["source"]["mtime"] == str(_modes_file(mode_root).stat().st_mtime_ns)
        # What landed on disk, and the bytes it replaced.
        written = json.loads(_modes_file(mode_root).read_text(encoding="utf-8"))
        assert list(written["agents"]) == ["code", "reviewer"]
        assert written["version"] == agent_modes.AGENT_MODES_VERSION
        chain = editorfs.files.list_backups(Path(agent_modes.agent_modes_path()))
        assert len(chain) == 1
        assert json.loads(Path(chain[0]).read_text(encoding="utf-8"))["agents"] == {
            "code": {"label": "Code"}
        }

    def test_a_save_of_an_absent_file_creates_it_exclusively_with_the_readme(
        self, client, mode_root, mode_backups
    ):
        response = client.post("/api/agent-modes/save", json={
            "document": {"default": "code",
                         "agents": {"code": {"label": "Code"}}},
            "mtime": None,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["created"] is True and body["backedUp"] is False
        written = json.loads(_modes_file(mode_root).read_text(encoding="utf-8"))
        assert written["readme"], "a created file gets the built-in readme"
        assert not editorfs.files.list_backups(Path(agent_modes.agent_modes_path()))

    def test_a_save_against_a_moved_file_is_409_carrying_both_numbers(
        self, client, mode_root, mode_backups
    ):
        _write_modes(mode_root, _modes_document({"code": {"label": "Code"}}))
        stale = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        _write_modes(mode_root, _modes_document({
            "code": {"label": "Code"}, "someone": {"label": "Else"},
        }, default="someone"))
        before = _modes_file(mode_root).read_text(encoding="utf-8")

        response = client.post("/api/agent-modes/save", json={
            "document": {"default": "code", "agents": {"code": {"label": "Mine"}}},
            "mtime": stale,
        })
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["reason"] == "stale"
        assert detail["diskMtime"] == str(_modes_file(mode_root).stat().st_mtime_ns)
        assert detail["documentMtime"] == stale
        assert _modes_file(mode_root).read_text(encoding="utf-8") == before
        assert not editorfs.files.list_backups(Path(agent_modes.agent_modes_path()))

    def test_a_save_declaring_a_reserved_id_is_422_and_the_disk_is_untouched(
        self, client, mode_root, mode_backups
    ):
        _write_modes(mode_root, _modes_document({"code": {"label": "Code"}}))
        before = _modes_file(mode_root).read_text(encoding="utf-8")
        token = client.get("/api/agent-modes/document").json()["source"]["mtime"]

        response = client.post("/api/agent-modes/save", json={
            "document": {
                "default": "code",
                "agents": {
                    "code": {"label": "Code"},
                    modes_defaults.TODO_BUILDER_MODE_ID: {"label": "Mine"},
                },
            },
            "mtime": token,
        })
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["reason"] == "invalid"
        assert any(modes_defaults.TODO_BUILDER_MODE_ID in item["message"]
                   for item in detail["findings"])
        assert _modes_file(mode_root).read_text(encoding="utf-8") == before

    def test_a_save_of_a_document_the_loader_would_drop_is_422(
        self, client, mode_root, mode_backups
    ):
        # D3's server-side half: what the reader would drop, the writer refuses —
        # naming the JSON pointer rather than a Pydantic field path.
        _write_modes(mode_root, _modes_document({"code": {"label": "Code"}}))
        token = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        response = client.post("/api/agent-modes/save", json={
            "document": {
                "default": "code",
                "agents": {"code": {"label": "Code", "mcptool": True}},
            },
            "mtime": token,
        })
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["findings"][0]["kind"] == "error"
        assert "agents.code.mcptool" in detail["findings"][0]["message"]
        assert json.loads(_modes_file(mode_root).read_text(encoding="utf-8")) == {
            "version": agent_modes.AGENT_MODES_VERSION,
            "default": "code",
            "agents": {"code": {"label": "Code"}},
        }

    def test_a_save_over_bytes_the_reader_refuses_is_422(
        self, client, mode_root, mode_backups
    ):
        # D3: a file-level finding holds the Save button down, and a server that
        # wrote over those bytes would undo that.
        folder = mode_root / agent_modes.POLICY_DIRNAME
        folder.mkdir(parents=True, exist_ok=True)
        _modes_file(mode_root).write_text("{not json", encoding="utf-8")
        # The token the document read hands out for those bytes: the guard has to
        # match before the writer can say anything about what is inside.
        token = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        response = client.post("/api/agent-modes/save", json={
            "document": {"default": "code", "agents": {"code": {"label": "Code"}}},
            "mtime": token,
        })
        assert response.status_code == 422
        assert "is not valid JSON" in response.json()["detail"]["findings"][0]["message"]
        assert _modes_file(mode_root).read_text(encoding="utf-8") == "{not json"
        assert not editorfs.files.list_backups(Path(agent_modes.agent_modes_path()))

    def test_a_malformed_body_is_a_422_never_a_500(self, client, mode_root):
        # The transport half of the validation discipline: FastAPI's own 422. The
        # token is the string the document read published, so a body holding a
        # number or a bool for it is refused here rather than reaching the guard.
        assert client.post("/api/agent-modes/save", json={"nope": 1}).status_code == 422
        assert client.post("/api/agent-modes/save", json={
            "document": ["not", "an", "object"],
        }).status_code == 422
        assert client.post("/api/agent-modes/save", json={
            "document": {"default": "code", "agents": {}}, "mtime": 12345,
        }).status_code == 422
        assert client.post("/api/agent-modes/save", json={
            "document": {"default": "code", "agents": {}}, "mtime": True,
        }).status_code == 422
        assert not _modes_file(mode_root).exists()

    def test_a_save_while_a_chat_reply_streams_is_409(
        self, client, mode_root, mode_backups
    ):
        # The one place the streaming rule is written, applied to this pair like
        # every other browser write: a reply is the only other writer.
        _write_modes(mode_root, _modes_document({"code": {"label": "Code"}}))
        token = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        before = _modes_file(mode_root).read_text(encoding="utf-8")
        cancel_event = threading.Event()
        routes._arm_chat_handle(cancel_event, threading.Event())
        try:
            response = client.post("/api/agent-modes/save", json={
                "document": {"default": "code", "agents": {"code": {"label": "Mine"}}},
                "mtime": token,
            })
            assert response.status_code == 409
            assert "while a chat reply is streaming" in response.json()["detail"]
        finally:
            routes._clear_chat_handle(cancel_event)
        assert _modes_file(mode_root).read_text(encoding="utf-8") == before

    def test_a_save_is_not_gated_during_a_benchmark_run(
        self, client, mode_root, mode_backups
    ):
        # Pinned so nobody adds the settings routes' run-check here by analogy: a
        # run never consults a mode, so gating it would be a rule with no race
        # behind it.
        _write_modes(mode_root, _modes_document({"code": {"label": "Code"}}))
        token = client.get("/api/agent-modes/document").json()["source"]["mtime"]
        client.app.state.run_state.status = "running"
        try:
            response = client.post("/api/agent-modes/save", json={
                "document": {
                    "default": "code",
                    "agents": {"code": {"label": "Code"},
                               "reviewer": {"label": "Reviewer"}},
                },
                "mtime": token,
            })
        finally:
            client.app.state.run_state.status = "idle"
        assert response.status_code == 200
        assert [mode["id"] for mode in response.json()["modes"]] == [
            "code", "reviewer", modes_defaults.TODO_BUILDER_MODE_ID,
        ]

    def test_the_document_read_leaves_nothing_behind(self, client, mode_root):
        before = sorted(p.name for p in mode_root.iterdir())
        for _ in range(3):
            assert client.get("/api/agent-modes/document").status_code == 200
        assert sorted(p.name for p in mode_root.iterdir()) == before

    def test_the_modes_file_stays_unreachable_through_the_editor_surface(
        self, client, mode_root
    ):
        # The pair is a purpose-built route for one file; the reservation stands.
        agent_modes.ensure_agent_modes_file(str(mode_root))
        rel = f"{agent_modes.POLICY_DIRNAME}/{agent_modes.AGENT_MODES_FILENAME}"
        read = client.get("/api/fs/file", params={"path": rel})
        assert read.status_code == 404
        assert "does not exist" in read.json()["detail"]
        assert "systemMessage" not in read.text

        tree = client.get("/api/fs/tree", params={"path": agent_modes.POLICY_DIRNAME})
        assert tree.status_code in (404, 422)
        assert "entries" not in tree.json()

        # A mutation is the one answer that *names* the rule (the same shape
        # ``tests/test_agent_modes_seed.py`` pins for a create), and it changes
        # nothing: the seeded bytes are still there, unedited.
        before = _modes_file(mode_root).read_text(encoding="utf-8")
        save = client.put("/api/fs/file", json={"path": rel, "text": "x\n"})
        assert save.status_code == 422
        assert agent_modes.POLICY_DIRNAME in save.json()["detail"]
        assert _modes_file(mode_root).read_text(encoding="utf-8") == before


class _ChatHubDouble:
    """Hub double answering ``active_tool_defs`` from the ids it is given."""

    installed = True

    def __init__(self) -> None:
        self.requested_ids: List[List[str]] = []

    def status(self) -> List[Dict[str, Any]]:
        return []

    def active_tool_defs(self, ids):
        self.requested_ids.append(list(ids))
        tools = [{"type": "function", "function": {"name": f"use_{cid}"}}
                 for cid in ids]
        return tools, {f"use_{cid}": cid for cid in ids}

    def call_tool_sync(self, cid, name, args, timeout=30.0):
        return "ok"



@pytest.fixture
def chat_stub(client, monkeypatch: pytest.MonkeyPatch):
    """``client`` with a live MCP hub and a ``stream_chat`` that records kwargs.

    The recording shape mirrors ``tests/test_chat.py``'s stubs: the five named
    instruction/prompt keywords plus whatever tool kwargs the route passes, so
    the *absence* of a key is as observable as its value — which is the whole
    point of the tool-gate assertions below.
    """
    hub = _ChatHubDouble()
    client.app.state.mcp = hub
    client.app.state.lsp = _ChatHubDouble()
    monkeypatch.setattr(routes, "_active_connector_ids", lambda: ["console"])
    monkeypatch.setattr(routes, "_active_server_ids", lambda: [])
    seen: Dict[str, Any] = {}
    asked: List[Optional[str]] = []

    real_resolve = agent_modes.resolve_mode

    def spy_resolve(mode_id):
        asked.append(mode_id)
        return real_resolve(mode_id)

    def fake(messages, model, server_url, emit, sampling=None, cancel_event=None,
             pause_event=None, api_key="none", sysprompt=None,
             agent_instructions=None, mode_instructions=None, **tools):
        seen["sysprompt"] = sysprompt
        if mode_instructions is not None:
            seen["mode_instructions"] = mode_instructions
        seen.update(tools)
        emit({"event": "done", "took": 0.0, "ttft": 0.0, "tokens": 0,
              "tokensEstimated": False})

    monkeypatch.setattr(routes.agent_modes, "resolve_mode", spy_resolve)
    monkeypatch.setattr(routes, "stream_chat", fake)
    return client, seen, hub, asked


def _chat(client, **extra):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return client.post("/api/chat", json=body)


class TestChatAgentModeField:
    """am-06 §1-§2: ``ChatIn.agentMode`` selects a mode, and a typo refuses.

    The field and the gate are the same task because a client that could pick a
    mode which changed the *prompt* but not the *tool set* would get half a mode
    — so both edges are asserted here, through the hub double's allowlist.
    """

    def test_an_unknown_mode_is_422_naming_it_and_listing_the_real_ones(
        self, chat_stub, mode_root
    ):
        client, seen, _hub, _asked = chat_stub
        response = _chat(client, agentMode="nope")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "'nope'" in detail
        for mode_id in ("chat", "code", "architect",
                        modes_defaults.TODO_BUILDER_MODE_ID):
            assert mode_id in detail
        # Nothing streamed: the worker never ran.
        assert seen == {}

    def test_a_refused_mode_does_not_own_the_pause_handle(
        self, chat_stub, mode_root
    ):
        # §2: the 422 is raised *before* ``_arm_chat_handle``, so a request that
        # will never stream cannot leave a stale handle for the next one.
        client, _seen, _hub, _asked = chat_stub
        assert routes._CHAT_HANDLE is None
        assert _chat(client, agentMode="nope").status_code == 422
        assert routes._CHAT_HANDLE is None
        assert _chat(client).status_code == 200

    def test_a_mode_id_of_the_wrong_shape_is_422_from_the_model(
        self, chat_stub, mode_root
    ):
        # The *shape* check lives in the model (cheap, static); existence lives
        # in ``resolve_mode``. An id that could never be a mode id never even
        # reaches the file lookup.
        client, seen, _hub, _asked = chat_stub
        for bad in ("Nope", "9lives", "has space", "x" * 33):
            assert _chat(client, agentMode=bad).status_code == 422, bad
        assert seen == {}
        # It is the *shape* only: none of these reached the loader, so a
        # ``Literal``/enum would not have been needed either — and none of them
        # can be repaired by editing the file, unlike the 422 above.
        assert _asked == []

    def test_an_empty_agent_mode_asks_for_the_default_instead_of_refusing(
        self, chat_stub, mode_root
    ):
        # ``resolve_mode`` reads empty as "the file's default"; a client that
        # sends the field empty gets that, not a shape error.
        client, _seen, _hub, asked = chat_stub
        assert _chat(client, agentMode="   ").status_code == 200
        assert asked == [None]

    def test_no_agent_mode_uses_the_file_default_and_the_old_tool_kwargs(
        self, chat_stub, mode_root
    ):
        # A client predating the feature: the seeded default is ``code``, whose
        # gate is open, so the tool arguments are exactly what they were before
        # modes existed. They are *not* identical on ``sysprompt`` — from am-04
        # the default mode supplies a persona, which is the point of the
        # feature — so the persona is asserted separately, never as prose.
        client, seen, hub, asked = chat_stub
        agent_modes.ensure_agent_modes_file(str(mode_root))
        assert _chat(client).status_code == 200
        assert asked == [None]
        assert set(seen) == {
            "sysprompt", "tools", "tool_executor", "tool_servers",
            "max_tool_iterations",
        }
        assert hub.requested_ids == [["console"]]
        assert seen["tool_servers"] == {"use_console": "console"}
        # …identical in *content* too, not just in keys: the four tool arguments
        # an old client got are the four this one gets.
        assert seen["tools"] == [
            {"type": "function", "function": {"name": "use_console"}}
        ]
        assert callable(seen["tool_executor"])
        assert seen["max_tool_iterations"] == routes.max_tool_iteration()
        # The mode itself, asserted separately from the prose that carries it.
        default = agent_modes.resolve_mode(None)[0]
        assert default.id == modes_defaults.CODE_MODE_ID
        assert seen["sysprompt"] == default.system_message

    def test_the_selected_mode_narrows_the_tool_set_as_well_as_the_prompt(
        self, chat_stub, mode_root
    ):
        client, seen, hub, asked = chat_stub
        agent_modes.ensure_agent_modes_file(str(mode_root))
        assert _chat(client, agentMode="chat").status_code == 200
        assert asked == ["chat"]
        # D3: no ``tools`` key at all (never ``tools=[]``), and the hub was
        # narrowed to an empty allowlist rather than skipped.
        assert "tools" not in seen
        assert "tool_executor" not in seen
        assert hub.requested_ids == [[]]
        assert seen["sysprompt"] == agent_modes.resolve_mode("chat")[0].system_message

        seen.clear()
        assert _chat(client, agentMode="code").status_code == 200
        assert "tools" in seen
        assert hub.requested_ids == [[], ["console"]]

    def test_the_app_owned_mode_is_reachable_and_its_blocks_arrive(
        self, chat_stub, mode_root
    ):
        # The whole reason the reserved mode is published: a user can pick it.
        client, seen, _hub, asked = chat_stub
        response = _chat(client, agentMode=modes_defaults.TODO_BUILDER_MODE_ID)
        assert response.status_code == 200
        assert asked == [modes_defaults.TODO_BUILDER_MODE_ID]
        assert seen["mode_instructions"] == "\n\n".join(
            modes_defaults.todo_builder_mode().instructions
        )
        assert modes_defaults.lot_conventions_block() in seen["mode_instructions"]

    def test_a_selected_mode_is_resolved_against_the_file_as_it_is_now(
        self, chat_stub, mode_root
    ):
        # Hot edit through the HTTP edge: the same client, two files, two
        # personas — no restart and no second cache.
        client, seen, _hub, _asked = chat_stub
        _write_modes(mode_root, _modes_document(
            {"planner": {"label": "Planner", "systemMessage": "PERSONA ONE"}},
            default="planner",
        ))
        assert _chat(client, agentMode="planner").status_code == 200
        assert seen["sysprompt"] == "PERSONA ONE"
        _write_modes(mode_root, _modes_document(
            {"planner": {"label": "Planner", "systemMessage": "PERSONA TWO"}},
            default="planner",
        ))
        assert _chat(client, agentMode="planner").status_code == 200
        assert seen["sysprompt"] == "PERSONA TWO"
