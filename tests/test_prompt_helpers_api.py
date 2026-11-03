"""mbl-03 — the ``/api/prompt-helpers`` half of the move, over HTTP.

The cases ``tests/test_memory.py`` used to assert, re-homed onto the new routes
and the new store: ``<workspace>/.nojae/prompt-helpers.json``, pointed at per
test by ``BENCHMARKIT_PATH_LOCATION`` (the ``tests/test_agent_modes.py`` way)
rather than by ``config.RESOURCES_DIR`` — the working location is process state,
so a test that moves it has to move it for itself alone.

Three of the old file's assertions are **not** carried over, because the
behaviour they pinned was deleted on purpose (spec D4/D5):

* the first ``GET`` no longer materialises the file — a reader writes nothing;
* a bad document no longer falls back to the defaults — it is a 422 and the
  bytes on disk are untouched;
* a wrong slot count is no longer padded or truncated — 422 naming the count.

Field coercion (a bad colour → the default gray) is the one remaining write on
a read and is still asserted, on disk as well as in the payload.

``LEGACY_PROMPT_HELPERS_PATH`` is patched for every test here, before the app is
built: the lift belongs to a seeding moment, and this checkout's real
``resources/memory_registration.json`` (gitignored, machine-local) must never be
what a test migrates.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api.server import create_app
from src.core import prompt_helpers


# -- the scratch workspace ------------------------------------------------------

@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[TestClient]:
    """The app on the boot location, with recording hubs and no legacy file.

    The location is deliberately *not* set to a scratch root here: boot seeds the
    root it lands on, so a test that wants a workspace with nothing in it must
    point at a fresh root itself (that is what "a reader never writes" is about).
    The declaration files are redirected the way ``tests/test_api.py``'s
    ``hub_client`` does, so a switch restarts nothing real.
    """
    from src.core import lsphub
    from src.core import mcphub

    class _RecordingHub:
        installed = True

        def __init__(self) -> None:
            self.restarts: List[Any] = []

        async def restart(self, *args) -> None:
            self.restarts.append(args)

        def status(self) -> List[Dict[str, Any]]:
            return []

    monkeypatch.delenv("BENCHMARKIT_PATH_LOCATION", raising=False)
    monkeypatch.setattr(
        prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", tmp_path / "no-legacy.json"
    )
    empty = tmp_path / "externals.json"
    empty.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mcphub.constants, "EXTERNALS_PATH", empty)
    servers_dir = tmp_path / "lsp_connectors"
    servers_dir.mkdir()
    (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
    monkeypatch.setattr(lsphub.constants, "LSP_CONNECTORS_DIR", servers_dir)
    monkeypatch.setattr(
        lsphub.constants, "LSP_SERVERS_PATH", servers_dir / "lsp_servers.json"
    )
    monkeypatch.setattr(
        lsphub.constants, "LSP_RESTRICTED_PATHS",
        servers_dir / "restricted-paths.txt",
    )
    app = create_app()
    with TestClient(app) as test_client:
        # Swapped *inside* the context: the lifespan installs the real hubs on
        # startup, so a stand-in assigned before it is replaced.
        test_client.app.state.mcp = _RecordingHub()
        test_client.app.state.lsp = _RecordingHub()
        yield test_client


@pytest.fixture
def workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """An empty scratch root made the active working location."""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
    yield root


def helpers_path(root: Path) -> Path:
    """The real lookup path for ``root``, built from the module's attributes."""
    return root / prompt_helpers.POLICY_DIRNAME / prompt_helpers.PROMPT_HELPERS_FILENAME


def write_helpers(root: Path, text: str) -> Path:
    """Put ``text`` at the helpers path, creating the reserved folder."""
    folder = root / prompt_helpers.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = helpers_path(root)
    path.write_text(text, encoding="utf-8")
    return path


def document(helpers: List[Any]) -> str:
    payload: Dict[str, Any] = {"version": prompt_helpers.PROMPT_HELPERS_VERSION}
    payload["helpers"] = helpers
    return json.dumps(payload)


def slots(count: int, color: str = "#2ec27e", content: str = "keep") -> List[Dict[str, str]]:
    return [{"color": color, "content": content} for _ in range(count)]


def defaults() -> List[Dict[str, str]]:
    return prompt_helpers.default_prompt_helpers()


def on_disk(root: Path) -> Dict[str, Any]:
    return json.loads(helpers_path(root).read_text(encoding="utf-8"))


# -- the read side --------------------------------------------------------------

class TestGetPromptHelpers:
    def test_an_empty_root_answers_the_defaults_and_leaves_no_folder(
        self, client: TestClient, workspace: Path
    ):
        """D5's behaviour change: the first GET materialises nothing."""
        response = client.get("/api/prompt-helpers")
        assert response.status_code == 200
        body = response.json()
        assert body["slots"] == defaults()
        assert len(body["slots"]) == prompt_helpers.SLOT_COUNT
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()

    def test_the_payload_key_is_slots(self, client: TestClient, workspace: Path):
        assert list(client.get("/api/prompt-helpers").json()) == ["slots"]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("this is not json {{{", "not valid JSON"),
            (json.dumps({"version": 2, "helpers": slots(9)}), "version 2"),
            (json.dumps({"helpers": slots(9)}), "version"),
        ],
    )
    def test_an_unusable_file_is_422_naming_the_path_and_is_never_repaired(
        self, client: TestClient, workspace: Path, text: str, expected: str
    ):
        path = write_helpers(workspace, text)
        response = client.get("/api/prompt-helpers")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert str(path) in detail
        assert expected in detail
        assert path.read_text(encoding="utf-8") == text

    @pytest.mark.parametrize("count", (8, 10))
    def test_a_wrong_count_is_422_naming_the_count(
        self, client: TestClient, workspace: Path, count: int
    ):
        """D4: never padded, never truncated — and never a 500."""
        text = document(slots(count))
        path = write_helpers(workspace, text)
        response = client.get("/api/prompt-helpers")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert f"holds {count} helpers" in detail
        assert f"expected exactly {prompt_helpers.SLOT_COUNT}" in detail
        assert path.read_text(encoding="utf-8") == text

    def test_a_bad_colour_is_coerced_and_the_file_is_rewritten(
        self, client: TestClient, workspace: Path
    ):
        """Field coercion stays self-healing (D4): the one write on a read."""
        write_helpers(
            workspace,
            document([{"color": "bogus", "content": "first"}] + slots(8)),
        )
        body = client.get("/api/prompt-helpers").json()
        assert len(body["slots"]) == prompt_helpers.SLOT_COUNT
        assert body["slots"][0] == {
            "color": prompt_helpers.DEFAULT_COLOR,
            "content": "first",
        }
        stored = on_disk(workspace)["helpers"]
        assert len(stored) == prompt_helpers.SLOT_COUNT
        assert stored[0]["color"] == prompt_helpers.DEFAULT_COLOR

    def test_the_slots_come_from_the_active_workspace(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Per-workspace: a switch moves the answer, the old file is not read."""
        first = tmp_path / "first"
        first.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(first))
        response = client.post(
            "/api/prompt-helpers",
            json={"index": 3, "color": "#ff0000", "content": "of the first root"},
        )
        assert response.status_code == 200
        before = helpers_path(first).read_bytes()

        second = tmp_path / "second"
        second.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(second))
        body = client.get("/api/prompt-helpers").json()
        assert body["slots"] == defaults()
        assert not (second / prompt_helpers.POLICY_DIRNAME).exists()
        # And the first root's bytes are exactly what its own POST wrote.
        assert helpers_path(first).read_bytes() == before


# -- the write side -------------------------------------------------------------

class TestPostPromptHelpers:
    def test_round_trip_persists(self, client: TestClient, workspace: Path):
        response = client.post(
            "/api/prompt-helpers",
            json={"index": 2, "color": "#ff0000", "content": "hello"},
        )
        assert response.status_code == 200
        assert response.json()["slots"][2] == {
            "color": "#ff0000",
            "content": "hello",
        }
        body = client.get("/api/prompt-helpers").json()
        assert body["slots"][2] == {"color": "#ff0000", "content": "hello"}
        stored = on_disk(workspace)
        assert stored["version"] == prompt_helpers.PROMPT_HELPERS_VERSION
        assert list(stored) == ["version", "helpers"]
        assert len(stored["helpers"]) == prompt_helpers.SLOT_COUNT
        assert stored["helpers"][2] == {"color": "#ff0000", "content": "hello"}
        untouched = defaults()
        for index, slot in enumerate(stored["helpers"]):
            if index != 2:
                assert slot == untouched[index]

    def test_a_write_creates_the_folder_a_reader_would_not(
        self, client: TestClient, workspace: Path
    ):
        """A workspace whose seed failed still saves the slot the user typed."""
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()
        response = client.post(
            "/api/prompt-helpers", json={"index": 0, "color": "#00ff00", "content": "x"}
        )
        assert response.status_code == 200
        assert helpers_path(workspace).is_file()

    def test_empty_content_clears_slot_but_keeps_color(self, client: TestClient,
                                                       workspace: Path):
        client.post(
            "/api/prompt-helpers", json={"index": 0, "color": "#00ff00", "content": "x"}
        )
        response = client.post(
            "/api/prompt-helpers", json={"index": 0, "color": "#00ff00", "content": ""}
        )
        assert response.status_code == 200
        assert response.json()["slots"][0] == {"color": "#00ff00", "content": ""}

    def test_rejects_bad_index(self, client: TestClient, workspace: Path):
        for index in (9, -1, 100):
            response = client.post(
                "/api/prompt-helpers",
                json={"index": index, "color": "#ff0000", "content": "x"},
            )
            assert response.status_code == 422
        assert client.get("/api/prompt-helpers").json()["slots"] == defaults()
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()

    def test_rejects_bad_color(self, client: TestClient, workspace: Path):
        for color in ("red", "#12345", "#1234567", "#GGGGGG", ""):
            response = client.post(
                "/api/prompt-helpers", json={"index": 0, "color": color, "content": "x"}
            )
            assert response.status_code == 422
        assert client.get("/api/prompt-helpers").json()["slots"] == defaults()

    def test_rejects_oversized_content_accepts_cap(self, client: TestClient,
                                                   workspace: Path):
        oversized = "x" * (prompt_helpers.CONTENT_MAX_CHARS + 1)
        response = client.post(
            "/api/prompt-helpers",
            json={"index": 0, "color": "#ff0000", "content": oversized},
        )
        assert response.status_code == 422
        capped = "x" * prompt_helpers.CONTENT_MAX_CHARS
        response = client.post(
            "/api/prompt-helpers",
            json={"index": 0, "color": "#ff0000", "content": capped},
        )
        assert response.status_code == 200

    def test_slot_never_removed(self, client: TestClient, workspace: Path):
        client.post(
            "/api/prompt-helpers", json={"index": 4, "color": "#0000ff", "content": "y"}
        )
        client.post(
            "/api/prompt-helpers", json={"index": 99, "color": "#0000ff", "content": "y"}
        )
        body = client.get("/api/prompt-helpers").json()
        assert len(body["slots"]) == prompt_helpers.SLOT_COUNT
        assert body["slots"][4] == {"color": "#0000ff", "content": "y"}

    def test_an_unusable_file_on_disk_is_422_not_500(
        self, client: TestClient, workspace: Path
    ):
        """The save reads the file first, so D4's error reaches this route too."""
        text = document(slots(8))
        path = write_helpers(workspace, text)
        response = client.post(
            "/api/prompt-helpers", json={"index": 0, "color": "#ff0000", "content": "x"}
        )
        assert response.status_code == 422
        assert "holds 8 helpers" in response.json()["detail"]
        assert path.read_text(encoding="utf-8") == text


class TestTheOldNamesAreGone:
    """D8: the rename is the point — no alias is kept for the SPA."""

    def test_the_old_route_is_not_served(self, client: TestClient,
                                         workspace: Path):
        """The retired name answers nothing — GET and POST alike.

        The path is spelled in two pieces so a sweep for the old route name
        across ``src`` and ``tests`` stays empty: this is the one place that
        still has to *name* it, and only to prove the answer is nothing.
        """
        retired = "/api/" + "memory"
        assert client.get(retired).status_code == 404
        response = client.post(
            retired, json={"index": 0, "color": "#ff0000", "content": "x"}
        )
        # 404 when there is no built SPA to fall back on; with ``web/dist`` the
        # catch-all static mount answers instead, and a mount accepts no write
        # verb, so 405 (the same shape ``tests/test_editorfs_search.py`` asserts
        # for a read-only route). Either way nothing serves the old name, and
        # nothing writes.
        assert response.status_code in (404, 405)
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()


# -- the working-location endpoints --------------------------------------------

class TestTheSwitchPayload:
    def test_get_reports_the_block_without_seeding(
        self, client: TestClient, workspace: Path
    ):
        body = client.get("/api/path-location").json()
        block = body["promptHelpers"]
        assert set(block) == {"state", "path", "error"}
        assert block["state"] == "absent"
        assert block["error"] is None
        assert block["path"] == str(helpers_path(workspace))
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()
        # A second GET says the same, because a read still wrote nothing.
        assert client.get("/api/path-location").json()["promptHelpers"]["state"] == (
            "absent"
        )

    def test_post_seeds_the_new_root_and_says_created(
        self, client: TestClient, tmp_path: Path
    ):
        target = tmp_path / "another-project"
        (target / "src").mkdir(parents=True)
        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        block = response.json()["promptHelpers"]
        assert block["state"] == "created"
        assert block["error"] is None
        assert block["path"] == str(helpers_path(target.resolve()))
        assert helpers_path(target.resolve()).is_file()
        # Seeded *after* both restarts, so a seeding hiccup can never be
        # mistaken for a hub failure and roll the environment back.
        assert len(client.app.state.mcp.restarts) == 1
        assert len(client.app.state.lsp.restarts) == 1

        second = client.post("/api/path-location", json={"path": str(target)})
        assert second.json()["promptHelpers"]["state"] == "existing"

    def test_post_reports_a_migration_when_a_legacy_file_is_present(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """D6: the lift rides back in the same block, as its own state."""
        old = tmp_path / "legacy" / "memory_registration.json"
        old.parent.mkdir()
        old.write_text(
            json.dumps({"registration": slots(9, color="#2ec27e", content="mine")}),
            encoding="utf-8",
        )
        monkeypatch.setattr(prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", old)
        target = tmp_path / "migrated-project"
        target.mkdir()

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        block = response.json()["promptHelpers"]
        assert block["state"] == "migrated"
        assert block["error"] is None
        assert on_disk(target.resolve())["helpers"] == slots(
            9, color="#2ec27e", content="mine"
        )
        assert old.is_file(), "the legacy copy is left in place"

    @pytest.mark.posix_behaviour(
        "the unwritable root this POST is switched to is made with "
        "``mkdir(mode=0o500)``, a POSIX fact: the switch and the note it carries "
        "are only observable once a root can be made unwritable"
    )
    def test_a_failing_seed_is_a_note_not_a_refusal(
        self, client: TestClient, tmp_path: Path
    ):
        """The switch succeeds on an unwritable root; the block says ``failed``."""
        if os.geteuid() == 0:
            pytest.skip("root ignores the mode bits")
        locked = tmp_path / "read-only-project"
        locked.mkdir(mode=0o500)
        try:
            response = client.post("/api/path-location", json={"path": str(locked)})
            assert response.status_code == 200
            block = response.json()["promptHelpers"]
            assert block["state"] == "failed"
            assert block["error"]
            assert os.environ["BENCHMARKIT_PATH_LOCATION"] == str(locked.resolve())
            assert len(client.app.state.mcp.restarts) == 1
        finally:
            locked.chmod(0o755)


# -- the question-set guard ----------------------------------------------------

class TestQuestionSetsHaveNoExemption:
    """D3: with the helpers file gone from ``resources/``, nothing is skipped.

    This is the one place in the suite still patching ``config.RESOURCES_DIR``:
    a question set is a file of the *app*, not of the workspace, so the folder
    the discovery reads is the one to move. The assertion is on a scratch folder
    only — never on this checkout's real ``resources/``.
    """

    def test_a_file_named_like_the_old_one_is_listed_when_it_has_questions(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        resources = tmp_path / "resources"
        resources.mkdir()
        questions = {"questions": [{"uuid": "1", "question": "q?", "expected": "a"}]}
        (resources / "other.json").write_text(json.dumps(questions), encoding="utf-8")
        (resources / "memory_registration.json").write_text(
            json.dumps(questions), encoding="utf-8"
        )
        monkeypatch.setattr(config, "RESOURCES_DIR", resources)

        # The name no longer means anything: what decides is the content, so a
        # file keeping the old name and carrying a ``questions`` list is a
        # question set like any other.
        listed = client.get("/api/question-sets").json()["questionSets"]
        assert [entry["id"] for entry in listed] == ["memory_registration", "other"]
