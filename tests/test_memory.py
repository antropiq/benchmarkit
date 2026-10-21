import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api.server import create_app


@pytest.fixture
def resources_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate resources/ into an empty per-test directory."""
    resources = tmp_path / "resources"
    resources.mkdir()
    monkeypatch.setattr(config, "RESOURCES_DIR", resources)
    return resources


@pytest.fixture
def client(resources_dir: Path) -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _read_file(resources_dir: Path) -> dict:
    path = resources_dir / config.MEMORY_REGISTRATION_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def _defaults() -> list:
    return config.default_memory_registrations()


class TestGetMemory:
    def test_creates_defaults_when_file_missing(self, client, resources_dir):
        body = client.get("/api/memory").json()
        assert body["registration"] == _defaults()
        assert len(body["registration"]) == 9
        assert _read_file(resources_dir) == {"registration": _defaults()}

    def test_corrupt_file_is_normalized(self, client, resources_dir):
        path = resources_dir / config.MEMORY_REGISTRATION_FILENAME
        path.write_text("this is not json {{{", encoding="utf-8")
        body = client.get("/api/memory").json()
        assert body["registration"] == _defaults()
        assert _read_file(resources_dir) == {"registration": _defaults()}

    def test_missing_registration_key_is_normalized(self, client, resources_dir):
        path = resources_dir / config.MEMORY_REGISTRATION_FILENAME
        path.write_text(json.dumps({"questions": []}), encoding="utf-8")
        body = client.get("/api/memory").json()
        assert body["registration"] == _defaults()
        assert _read_file(resources_dir) == {"registration": _defaults()}

    def test_short_array_is_padded(self, client, resources_dir):
        path = resources_dir / config.MEMORY_REGISTRATION_FILENAME
        path.write_text(
            json.dumps({"registration": [{"color": "#123456", "content": "keep"}]}),
            encoding="utf-8",
        )
        body = client.get("/api/memory").json()
        assert body["registration"][0] == {"color": "#123456", "content": "keep"}
        assert body["registration"][1:] == _defaults()[1:]
        assert len(_read_file(resources_dir)["registration"]) == 9

    def test_long_array_is_truncated_and_bad_color_coerced(self, client, resources_dir):
        slots = [
            {"color": "bogus", "content": "first"},
            *[{"color": config.MEMORY_DEFAULT_COLOR, "content": ""} for _ in range(9)],
        ]
        path = resources_dir / config.MEMORY_REGISTRATION_FILENAME
        path.write_text(json.dumps({"registration": slots}), encoding="utf-8")
        body = client.get("/api/memory").json()
        assert len(body["registration"]) == 9
        assert body["registration"][0] == {
            "color": config.MEMORY_DEFAULT_COLOR,
            "content": "first",
        }
        on_disk = _read_file(resources_dir)["registration"]
        assert len(on_disk) == 9
        assert on_disk[0]["color"] == config.MEMORY_DEFAULT_COLOR


class TestPostMemory:
    def test_round_trip_persists(self, client, resources_dir):
        response = client.post(
            "/api/memory",
            json={"index": 2, "color": "#ff0000", "content": "hello"},
        )
        assert response.status_code == 200
        assert response.json()["registration"][2] == {
            "color": "#ff0000",
            "content": "hello",
        }
        body = client.get("/api/memory").json()
        assert body["registration"][2] == {"color": "#ff0000", "content": "hello"}
        assert _read_file(resources_dir)["registration"][2] == {
            "color": "#ff0000",
            "content": "hello",
        }
        untouched = _defaults()
        for index, slot in enumerate(_read_file(resources_dir)["registration"]):
            if index != 2:
                assert slot == untouched[index]

    def test_empty_content_clears_slot_but_keeps_color(self, client, resources_dir):
        client.post("/api/memory", json={"index": 0, "color": "#00ff00", "content": "x"})
        response = client.post(
            "/api/memory", json={"index": 0, "color": "#00ff00", "content": ""}
        )
        assert response.status_code == 200
        assert response.json()["registration"][0] == {
            "color": "#00ff00",
            "content": "",
        }

    def test_rejects_bad_index(self, client, resources_dir):
        for index in (9, -1, 100):
            response = client.post(
                "/api/memory", json={"index": index, "color": "#ff0000", "content": "x"}
            )
            assert response.status_code == 422
        assert client.get("/api/memory").json()["registration"] == _defaults()

    def test_rejects_bad_color(self, client, resources_dir):
        for color in ("red", "#12345", "#1234567", "#GGGGGG", ""):
            response = client.post(
                "/api/memory", json={"index": 0, "color": color, "content": "x"}
            )
            assert response.status_code == 422
        assert client.get("/api/memory").json()["registration"] == _defaults()

    def test_rejects_oversized_content_accepts_cap(self, client, resources_dir):
        oversized = "x" * (config.MEMORY_CONTENT_MAX_CHARS + 1)
        response = client.post(
            "/api/memory", json={"index": 0, "color": "#ff0000", "content": oversized}
        )
        assert response.status_code == 422
        capped = "x" * config.MEMORY_CONTENT_MAX_CHARS
        response = client.post(
            "/api/memory", json={"index": 0, "color": "#ff0000", "content": capped}
        )
        assert response.status_code == 200

    def test_slot_never_removed(self, client, resources_dir):
        client.post("/api/memory", json={"index": 4, "color": "#0000ff", "content": "y"})
        client.post("/api/memory", json={"index": 99, "color": "#0000ff", "content": "y"})
        registration = client.get("/api/memory").json()["registration"]
        assert len(registration) == 9
        assert registration[4] == {"color": "#0000ff", "content": "y"}


class TestQuestionSetExclusion:
    def test_memory_file_never_listed_as_question_set(self, client, resources_dir):
        client.get("/api/memory")
        (resources_dir / "other.json").write_text(
            json.dumps({"questions": [{"uuid": "1", "question": "q?", "expected": "a"}]}),
            encoding="utf-8",
        )
        ids = [entry["id"] for entry in config.list_question_sets()]
        assert "memory_registration" not in ids
        assert "other" in ids
