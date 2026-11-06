"""ws-02 — seeding ``.nojae/nojae-settings.json`` at the two moments.

The workspace's model-connection file arrives the way the policy, modes, helpers
and MCP declaration files do: at the two moments a folder becomes the active
working location (boot, and ``POST /api/path-location``), by exclusive create from
the committed template, never by a reader. This suite has the shape of
``tests/test_mcp_settings_seed.py`` — which is its twin in the one respect no
other seed file shares: **the seed must land before the hub restarts**, because a
restart reads what the seed writes.

The promises asserted here:

* **a new location gets the template verbatim** — boot and switch each leave
  ``nojae-settings.default.json``'s bytes at
  ``<root>/.nojae/nojae-settings.json``;
* **never overwrites** — a workspace's own server, key and sampling survive every
  later seed byte for byte;
* **the seed is a note, never a failure** — an unreadable template answers
  ``failed``, the app still boots, the switch still answers **200** and the
  environment still moves;
* **the seed precedes ``_restart_lsp_hub()``** (spec D10) — the file, and *which*
  root's file it is, is on disk at the moment the hub is restarted;
* **the switch invalidates the cache** (spec D4) — the new workspace's file
  answers afterwards, not the previous browse's cached ``absent``;
* **a seeded file reports "not configured"** (spec D8) naming its path — valid
  JSON, unusable values, and no transport call.

Hermetic in the folder's own style: every test points
``BENCHMARKIT_PATH_LOCATION`` at a throw-away ``tmp_path`` and builds the path
under test from ``workspace_settings.POLICY_DIRNAME`` /
``WORKSPACE_SETTINGS_FILENAME`` — never a literal ``.nojae/nojae-settings.json``,
and never this checkout's own ``.nojae``.

Two cases ws-02 could not write are here, added by ws-06 once the app really read
the file: the D10 ordering asserted end to end (a switch into a workspace whose
two LSP gates both name ``pyright`` has the hub asked to start it), and the D9
refusal — a malformed file answers a chat and a benchmark request with the
sentence naming the path, never a 200 with a guessed model.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core import lsphub
from src.core import workspace_settings


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch working location with no settings file yet.

    The module's cache is cleared around each test: it is keyed on the resolved
    settings path, so a test that moves the location must not answer from an entry
    a previous test made about its file.
    """
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    workspace_settings.invalidate_workspace_settings()
    yield tmp_path
    workspace_settings.invalidate_workspace_settings()


def seeded_path(root: Path) -> Path:
    """The real lookup path for ``root``, built from the module's attributes."""
    return (
        root
        / workspace_settings.POLICY_DIRNAME
        / workspace_settings.WORKSPACE_SETTINGS_FILENAME
    )


def template_bytes() -> bytes:
    return workspace_settings.WORKSPACE_SETTINGS_TEMPLATE.read_bytes()


def configured_document(
    server_url: str = "http://127.0.0.1:8080/v1", **members: Any
) -> str:
    """A settings document the reader calls *configured* — real values, no prose.

    Built from the template so it always holds exactly the approved members, with
    the two instruction sentences replaced by values.
    """
    raw = json.loads(template_bytes().decode("utf-8"))
    raw["serverUrl"] = server_url
    raw["endpoint"] = "qwen3-32b"
    raw.update(members)
    return json.dumps(raw, indent=2) + "\n"


def malformed_document() -> str:
    """Valid JSON is not required here — this is the document that is not."""
    return '{"serverUrl": "http://127.0.0.1:8080/v1",\n'


# -- the seed itself -------------------------------------------------------------


class TestTheSeedLands:
    def test_it_creates_the_file_byte_for_byte_from_the_template(
        self, workspace: Path
    ):
        result = workspace_settings.ensure_workspace_settings_file(str(workspace))
        assert result["state"] == "created"
        assert result["error"] is None
        assert result["path"] == str(seeded_path(workspace))
        assert seeded_path(workspace).read_bytes() == template_bytes()

    def test_never_overwrites_a_hand_edited_file(self, workspace: Path):
        seeded_path(workspace).parent.mkdir(parents=True)
        handwritten = configured_document("http://127.0.0.1:9999/v1")
        seeded_path(workspace).write_text(handwritten, encoding="utf-8")
        stamp = (seeded_path(workspace).stat().st_mtime_ns,
                 seeded_path(workspace).stat().st_size)

        result = workspace_settings.ensure_workspace_settings_file(str(workspace))
        assert result["state"] == "existing"
        assert result["error"] is None
        assert seeded_path(workspace).read_text(encoding="utf-8") == handwritten
        assert (
            seeded_path(workspace).stat().st_mtime_ns,
            seeded_path(workspace).stat().st_size,
        ) == stamp

    def test_two_calls_in_a_row_have_one_winner(self, workspace: Path):
        states = [
            workspace_settings.ensure_workspace_settings_file(str(workspace))["state"]
            for _ in range(2)
        ]
        assert states == ["created", "existing"]

    def test_a_folder_at_the_path_is_failed_not_created(self, workspace: Path):
        seeded_path(workspace).parent.mkdir(parents=True)
        seeded_path(workspace).mkdir()

        result = workspace_settings.ensure_workspace_settings_file(str(workspace))
        assert result["state"] == "failed"
        assert result["error"]

    def test_a_missing_template_is_a_note_with_no_half_written_file(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            workspace_settings,
            "WORKSPACE_SETTINGS_TEMPLATE",
            workspace / "gone.default.json",
        )
        result = workspace_settings.ensure_workspace_settings_file(str(workspace))
        assert result["state"] == "failed"
        assert result["error"]
        assert not seeded_path(workspace).exists()


# -- a seeded workspace is valid JSON and unusable (D8) ---------------------------


class TestASeededWorkspaceReportsItselfNotConfigured:
    def test_the_seeded_file_is_ok_but_not_configured(self, workspace: Path):
        workspace_settings.ensure_workspace_settings_file(str(workspace))
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "ok"
        assert state["configured"] is False
        # The D8 sentence names the file the user has to fill in.
        assert state["message"] is not None
        assert str(seeded_path(workspace)) in state["message"]

    def test_and_the_two_instruction_members_are_the_named_ones(
        self, workspace: Path
    ):
        workspace_settings.ensure_workspace_settings_file(str(workspace))
        message = workspace_settings.workspace_settings_state()["message"] or ""
        assert '"serverUrl"' in message
        assert '"endpoint"' in message

    def test_the_values_are_handed_back_unchanged(self, workspace: Path):
        """Reporting is not dialling: the loader still answers the template's text."""
        workspace_settings.ensure_workspace_settings_file(str(workspace))
        loaded = workspace_settings.load_workspace_settings()
        template = json.loads(template_bytes().decode("utf-8"))
        assert loaded["serverUrl"] == template["serverUrl"]
        assert loaded["endpoint"] == template["endpoint"]

    def test_a_file_holding_real_values_is_configured(self, workspace: Path):
        seeded_path(workspace).parent.mkdir(parents=True)
        seeded_path(workspace).write_text(configured_document(), encoding="utf-8")
        workspace_settings.invalidate_workspace_settings()
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "ok"
        assert state["configured"] is True
        assert state["message"] is None


# -- the two cases the earlier tasks could not write (ws-06) -----------------------


class _StartRecordingHub:
    """Hub double that records which servers a restart was asked to bring up.

    The real ``LspManager`` turns a restart into starts along one fixed chain —
    ``restart(servers, enabled_ids)`` → ``start_enabled`` → one start per id — so
    this double mirrors only that forwarding and records both ends of it. That is
    what makes it the end-to-end form of D10 rather than a restatement of it: the
    id can only reach ``started`` if the *single* restart the switch performed
    already carried it in ``enabled_ids``, which is only true because the seed
    landed first. A hub restarted before the seed would have been handed ``[]``
    (no file, so no ``lspEnabled``), and a second restart afterwards is exactly
    what "no restart in between" rules out.

    No language server is spawned: ``BENCHMARKIT_DISABLE_LSP`` is what keeps real
    servers out of the suite, and the test below clears it only so the *scan*
    runs — the hub under test is this object, the way ``tests/test_lsp.py``'s
    ``FakeLspManager`` stands in for the hub.
    """

    installed = True

    def __init__(self) -> None:
        self.restarts: List[Any] = []
        self.started: List[str] = []
        self.stopped: List[str] = []

    async def restart(self, servers, enabled_ids=None) -> None:
        self.restarts.append((servers, enabled_ids))
        # The hub's own forwarding, mirrored (see the class docstring).
        for server_id in enabled_ids or []:
            self.start_server(server_id)

    def start_server(self, server_id: str) -> None:
        self.started.append(server_id)

    def stop_server(self, server_id: str) -> None:
        self.stopped.append(server_id)

    def server_status(self, server_id: str) -> Optional[str]:
        return None

    def status(self) -> List[Dict[str, Any]]:
        return []


def lsp_declaration(workspace_root: str) -> str:
    """A ``lsp-settings.json`` declaring one enabled ``pyright`` entry.

    The command never runs — the hub is a fake — so it only has to *validate*: a
    non-empty command list and an absolute ``workspaceRoot`` that exists and is
    not on the deny-list.
    """
    return json.dumps(
        {
            "pyright": {
                "enable": True,
                "extensions": [".py"],
                "command": [sys.executable, "-c", "pass"],
                "workspaceRoot": workspace_root,
            }
        },
        indent=2,
    )


class TestASwitchStartsTheWorkspacesOwnLanguageServer:
    """D10 end to end: both gates of the *new* workspace answer on the switch.

    This is the half ws-02 could only state as a recorder — there the assertion
    was that the file was on disk when the hub was restarted. Here the target
    workspace declares ``pyright`` with ``enable: true`` **and** lists it in
    ``lspEnabled``, so the one restart the switch performs has to carry it.
    """

    def test_the_switch_hands_the_hub_the_enabled_server(
        self, seed_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        client, _ = seed_client
        # The scan has to run for the two gates to be consulted at all; the hub
        # is a fake, so lifting the switch spawns nothing (the same move
        # ``tests/test_lsp.py``'s restart cases make).
        monkeypatch.delenv("BENCHMARKIT_DISABLE_LSP", raising=False)

        target = tmp_path / "lsp-workspace"
        reserved = target / workspace_settings.POLICY_DIRNAME
        reserved.mkdir(parents=True)
        (reserved / lsphub.constants.LSP_SETTINGS_FILENAME).write_text(
            lsp_declaration(str(target)), encoding="utf-8"
        )
        # The workspace's own settings file, already listing the server, so the
        # seed answers ``existing`` and the restart reads these bytes.
        (reserved / workspace_settings.WORKSPACE_SETTINGS_FILENAME).write_text(
            configured_document(lspEnabled=["pyright"]), encoding="utf-8"
        )
        client.app.state.lsp = _StartRecordingHub()

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200

        hub = client.app.state.lsp
        assert len(hub.restarts) == 1, "the switch restarted the hub more than once"
        _servers, enabled_ids = hub.restarts[0]
        assert enabled_ids == ["pyright"]
        # ... and the hub was asked to bring that server up, in that one restart.
        assert hub.started == ["pyright"]
        # The other gate is asserted too: the declaration was scanned, so this
        # is the new workspace's own file rather than an empty scan.
        assert [server["id"] for server in hub.restarts[0][0]] == ["pyright"]

    def test_the_restart_reads_the_file_rather_than_a_stale_absent(
        self, seed_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The same start, reached only through the switch's own invalidation.

        The target is **browsed first**, so its path is cached as ``absent``, and
        only then given its settings file. The switch's seed answers ``existing``
        and writes nothing, so it never reaches the invalidation inside the
        seeder: the only thing that can put ``["pyright"]`` in front of the
        restart is the ``invalidate_workspace_settings()`` that sits on the
        seed's side of it. Remove that line and this is the test that says so.
        """
        from src import config as config_module

        client, _ = seed_client
        monkeypatch.delenv("BENCHMARKIT_DISABLE_LSP", raising=False)

        target = tmp_path / "browsed-then-owned"
        reserved = target / workspace_settings.POLICY_DIRNAME
        reserved.mkdir(parents=True)
        (reserved / lsphub.constants.LSP_SETTINGS_FILENAME).write_text(
            lsp_declaration(str(target)), encoding="utf-8"
        )
        monkeypatch.setenv(config_module.PATH_LOCATION_KEY, str(target))
        assert client.get("/api/path-location").json()["workspaceSettings"][
            "state"
        ] == "absent"
        # The file appears while the location is still that root — a write of
        # this kind invalidates nothing, so the cached ``absent`` is still live.
        (reserved / workspace_settings.WORKSPACE_SETTINGS_FILENAME).write_text(
            configured_document(lspEnabled=["pyright"]), encoding="utf-8"
        )
        client.app.state.lsp = _StartRecordingHub()

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        assert response.json()["workspaceSettings"]["state"] == "existing"

        hub = client.app.state.lsp
        assert len(hub.restarts) == 1
        assert hub.restarts[0][1] == ["pyright"]
        assert hub.started == ["pyright"]


# -- the two entry points, through the app ---------------------------------------


class _RecordingHub:
    """Hub double that snapshots the settings file at the moment of the restart.

    D10 is a statement about *ordering*, so the recorder is the whole test: it
    captures what ``.nojae/nojae-settings.json`` holds for the root the switch is
    moving to at the instant ``_restart_lsp_hub()`` reaches the hub. No real
    language server is involved — the honest signal is that the value was on disk
    before the hub was asked, which is what "has ``pyright`` running immediately"
    reduces to. The end-to-end form of that (a live server, a real switch, a tool
    call) is the lot's manual check, not a test.
    """

    installed = True

    def __init__(self) -> None:
        self.restarts: List[Any] = []
        self.settings_seen: List[Optional[bytes]] = []
        self.settings_exists: List[bool] = []

    async def restart(self, *args) -> None:
        self.restarts.append(args)
        path = Path(workspace_settings.workspace_settings_path())
        self.settings_exists.append(path.is_file())
        self.settings_seen.append(path.read_bytes() if path.is_file() else None)

    def status(self) -> List[Dict[str, Any]]:
        return []


@pytest.fixture
def seed_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A live app on a scratch root, with no settings file anywhere yet."""
    from src import config as config_module
    from src.core import mcphub, prompt_helpers

    for key in config_module.SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(
        prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", tmp_path / "no-legacy.json"
    )
    # Unpin both declaration lookups: the switch is what moves them here.
    monkeypatch.setattr(mcphub.constants, "MCP_SETTINGS_PATH", None)
    monkeypatch.setattr(lsphub.constants, "LSP_SETTINGS_PATH", None)
    # Hermetic LSP side (the switch restarts that hub too): the machine's own
    # deny-list must not be read.
    servers_dir = tmp_path / "lsp_connectors"
    servers_dir.mkdir()
    (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
    monkeypatch.setattr(lsphub.constants, "LSP_CONNECTORS_DIR", servers_dir)
    monkeypatch.setattr(
        lsphub.constants, "LSP_RESTRICTED_PATHS", servers_dir / "restricted-paths.txt"
    )
    root = tmp_path / "boot-root"
    root.mkdir()
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
    workspace_settings.invalidate_workspace_settings()
    with TestClient(create_app()) as client:
        client.app.state.mcp = _RecordingHub()
        client.app.state.lsp = _RecordingHub()
        yield client, root
    workspace_settings.invalidate_workspace_settings()


class TestTheTwoEntryPoints:
    def test_boot_leaves_the_file_behind(self, seed_client):
        """A new location gets the file with the template's bytes."""
        client, root = seed_client
        assert seeded_path(root).is_file()
        assert seeded_path(root).read_bytes() == template_bytes()
        body = client.get("/api/path-location").json()
        assert body["workspaceSettings"]["state"] == "ok"
        # A seeded workspace is reported, never dialled: valid JSON, unusable.
        assert body["workspaceSettings"]["configured"] is False
        assert str(seeded_path(root)) in body["workspaceSettings"]["message"]

    def test_boot_survives_an_unreadable_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A seed that cannot happen is a note in the log, never a boot failure."""
        from src import config as config_module
        from src.core import prompt_helpers

        for key in config_module.SETTING_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
        monkeypatch.setattr(
            prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", tmp_path / "no-legacy.json"
        )
        monkeypatch.setattr(
            workspace_settings,
            "WORKSPACE_SETTINGS_TEMPLATE",
            tmp_path / "gone.default.json",
        )
        root = tmp_path / "boot-root"
        root.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
        with TestClient(create_app()):
            pass
        assert not seeded_path(root).exists()

    def test_get_never_seeds(self, seed_client, tmp_path: Path, monkeypatch):
        client, _ = seed_client
        other = tmp_path / "browsed"
        other.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(other))
        for _ in range(2):
            body = client.get("/api/path-location").json()
            assert body["workspaceSettings"]["state"] == "absent"
            assert body["workspaceSettings"]["path"] == str(seeded_path(other))
        assert not seeded_path(other).exists()
        assert not other.joinpath(workspace_settings.POLICY_DIRNAME).exists()

    def test_switch_seeds_the_file(self, seed_client, tmp_path: Path):
        client, _ = seed_client
        target = tmp_path / "fresh-root"
        target.mkdir()

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        expanded = os.path.realpath(str(target))
        assert response.json()["workspaceSettings"] == {
            "state": "created",
            "path": os.path.join(
                expanded,
                workspace_settings.POLICY_DIRNAME,
                workspace_settings.WORKSPACE_SETTINGS_FILENAME,
            ),
            "error": None,
        }
        assert seeded_path(target).read_bytes() == template_bytes()

    def test_switch_seeds_before_the_hub_restarts(
        self, seed_client, tmp_path: Path
    ):
        """D10: the file is on disk when the LSP hub is restarted.

        ``lspEnabled`` is read *during* that restart, so a seed after it would
        leave the new workspace on zero language servers until the next restart.
        """
        client, _ = seed_client
        target = tmp_path / "fresh-root"
        target.mkdir()

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        lsp = client.app.state.lsp
        assert lsp.settings_exists, "the LSP hub was not restarted"
        assert lsp.settings_exists[-1] is True
        assert lsp.settings_seen[-1] == template_bytes()

    def test_the_file_the_restart_sees_is_the_new_roots(
        self, seed_client, tmp_path: Path
    ):
        """The other half of D10: *which* file was on disk.

        The target already holds its own settings file, so the seed answers
        ``existing`` and writes nothing — the bytes the restart sees can only be
        the new root's own, never the previous workspace's and never the template.
        """
        client, _ = seed_client
        target = tmp_path / "owned-root"
        seeded_path(target).parent.mkdir(parents=True)
        own = configured_document("http://127.0.0.1:1234/v1")
        seeded_path(target).write_text(own, encoding="utf-8")

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        assert response.json()["workspaceSettings"]["state"] == "existing"
        lsp = client.app.state.lsp
        assert lsp.settings_seen[-1] == own.encode("utf-8")

    def test_the_switch_invalidates_the_settings_cache(
        self, seed_client, tmp_path: Path, monkeypatch
    ):
        """D4: after a switch the *new* workspace's file answers.

        Browsing the target first caches ``absent`` for its path (a GET never
        seeds); the file then appears by hand, and the switch — whose seed answers
        ``existing`` and so writes nothing, and so never reaches the seeder's own
        invalidation — must still forget the stale entry.
        """
        client, _ = seed_client
        target = tmp_path / "late-root"
        target.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(target))
        assert client.get("/api/path-location").json()["workspaceSettings"][
            "state"
        ] == "absent"

        seeded_path(target).parent.mkdir(parents=True)
        seeded_path(target).write_text(configured_document(), encoding="utf-8")

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        assert response.json()["workspaceSettings"]["state"] == "existing"
        body = client.get("/api/path-location").json()["workspaceSettings"]
        assert body["state"] == "ok"
        assert body["configured"] is True

    def test_a_failing_seed_is_a_note_not_a_refusal(
        self, seed_client, tmp_path: Path, monkeypatch
    ):
        client, _ = seed_client
        monkeypatch.setattr(
            workspace_settings,
            "WORKSPACE_SETTINGS_TEMPLATE",
            tmp_path / "gone.default.json",
        )
        target = tmp_path / "fresh-root"
        target.mkdir()

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        block = response.json()["workspaceSettings"]
        assert block["state"] == "failed"
        assert block["error"]
        # The switch itself succeeded: the environment moved and the hubs
        # restarted — a seeding hiccup is a note, never a rollback.
        assert os.environ["BENCHMARKIT_PATH_LOCATION"] == os.path.realpath(
            str(target)
        )
        assert len(client.app.state.lsp.restarts) >= 1
        assert client.app.state.lsp.settings_exists[-1] is False


# -- a malformed file refuses a chat and a benchmark request (D9) -----------------


class TestAMalformedFileRefusesEveryModelRequest:
    """D9 as a refusal, not just as a state field.

    The app boots on a broken file (a workspace is still browsable), so the
    question is what a model request does with it. Three answers were possible
    and two of them are wrong: a 200 that benchmarked on invented defaults would
    measure the defaults rather than the model, and a transport error quoting the
    template's instruction sentence would put a sentence in the user's face where
    a path belongs. The one the app gives is the server's own sentence — the same
    string ``GET /api/settings`` reports — naming the file and its path.

    The client is ``seed_client`` with nothing patched over the loader: this asks
    the real routes through the real ``load_endpoints_config()``, which is the
    seam ws-03 cut over.
    """

    def _write(self, root: Path, text: str) -> Path:
        path = seeded_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        workspace_settings.invalidate_workspace_settings()
        return path

    def test_a_chat_request_refuses_with_the_servers_sentence(self, seed_client):
        client, root = seed_client
        path = self._write(root, malformed_document())

        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "not valid JSON" in detail
        assert str(path) in detail
        # Not a transport error quoting a template sentence: the instruction text
        # never leaves the file.
        assert "Put here the url" not in detail

    def test_a_benchmark_request_refuses_with_the_same_sentence(self, seed_client):
        client, root = seed_client
        path = self._write(root, malformed_document())

        response = client.post("/api/run")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "not valid JSON" in detail
        assert str(path) in detail

    def test_a_seeded_file_refuses_a_chat_request_too(self, seed_client):
        """D8's turn at the same seam: valid JSON, unusable values, no call."""
        client, root = seed_client
        response = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert str(seeded_path(root)) in detail
        assert "Put here the url" not in detail

