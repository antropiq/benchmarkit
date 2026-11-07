"""Seeding ``.nojae/mcp-settings.json`` when a workspace becomes active.

The declaration file joined the reserved folder's family, so this suite has
the shape of ``tests/test_agent_modes_seed.py``: scratch workspaces on disk,
the helpers exercised directly, and the two entry points checked through the
app. What differs is only what a *missing* file costs — a workspace with no
declaration has no MCP connectors — and the one asymmetry of this seed: the
MCP restart *reads* the very file the switch seeds, so the seed must land
first.

The promises this change exists to keep:

* **never overwrites** — one ``O_CREAT | O_EXCL``, so a hand-edited
  declaration survives every later seed byte for byte;
* **the seed is the template verbatim** — the ``${NAME}`` references are
  resolved on every read (``expand_env``), never substituted at seed time,
  which is what keeps one file correct across switches and machines;
* **``${PROJECT_ROOT}`` answers from the app's own computation** — a stray
  ``PROJECT_ROOT`` in the environment cannot redirect connectors;
  ``${BENCHMARKIT_PYTHON_ROOT}`` answers from ``.env`` like any managed key,
  and an unset one is a note, not a blank;
* **the switch seeds before it restarts** — a fresh root comes up holding
  ``console`` and ``fedit``, not zero connectors;
* **a failure is a note** and **a reader never writes** — same as the policy
  and modes seeds.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.mcphub import constants, seed
from src.core.mcphub.discovery import load_externals, set_externals_enabled
from src.core.mcphub.validation import expand_env


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch working location with no declaration file yet."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    yield tmp_path


def seeded_path(root: Path) -> Path:
    return root / constants.POLICY_DIRNAME / constants.MCP_SETTINGS_FILENAME


class TestTheSeedLands:
    def test_it_creates_the_file_byte_for_byte_from_the_template(
        self, workspace: Path
    ):
        result = seed.ensure_mcp_settings_file(str(workspace))
        assert result["state"] == "created"
        assert result["error"] is None
        assert result["path"] == str(seeded_path(workspace))
        assert seeded_path(workspace).read_bytes() == (
            constants.MCP_SETTINGS_TEMPLATE.read_bytes()
        )

    def test_the_seeded_file_loads_as_the_two_shipped_connectors(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The point of seeding: a fresh workspace's hub is console + fedit."""
        monkeypatch.setenv("BENCHMARKIT_PYTHON_ROOT", "/opt/py/python")
        seed.ensure_mcp_settings_file(str(workspace))
        connectors, errors = load_externals()
        assert errors == []
        assert [connector["id"] for connector in connectors] == ["console", "fedit"]
        console = connectors[0]
        # The declaration spells a path by joining tokens with ``/``, and
        # ``${PROJECT_ROOT}`` answers as the install root's own text, so the
        # resolved script element is that text followed by the template's
        # slash. Expansion is substitution, never a path rewrite (the rule
        # ``tests/test_mcp.py`` pins), so the expected element is spelled the
        # way the template spells it — same as the expansion test below.
        from src.config import PROJECT_ROOT

        assert console["command"] == [
            "/opt/py/python",
            f"{PROJECT_ROOT}/mcp_connectors/console_mcp.py",
            "--root-dir",
            str(workspace),
            "--transport",
            "stdio",
        ]

    def test_the_template_is_not_a_working_file(self):
        """The committed template keeps its tokens and ships exactly two entries."""
        raw = constants.MCP_SETTINGS_TEMPLATE.read_text(encoding="utf-8")
        assert "${BENCHMARKIT_PYTHON_ROOT}" in raw
        assert "${PROJECT_ROOT}" in raw
        assert "${BENCHMARKIT_PATH_LOCATION}" in raw
        assert list(json.loads(raw)) == ["console", "fedit"]

    def test_never_overwrites_a_hand_edited_declaration(self, workspace: Path):
        seeded_path(workspace).parent.mkdir(parents=True)
        handwritten = '{"console": {"type": "local", "enable": false, "command": ["x"]}}\n'
        seeded_path(workspace).write_text(handwritten, encoding="utf-8")

        result = seed.ensure_mcp_settings_file(str(workspace))
        assert result["state"] == "existing"
        assert seeded_path(workspace).read_text(encoding="utf-8") == handwritten

    def test_a_folder_at_the_path_is_failed_not_created(self, workspace: Path):
        seeded_path(workspace).parent.mkdir(parents=True)
        seeded_path(workspace).mkdir()

        result = seed.ensure_mcp_settings_file(str(workspace))
        assert result["state"] == "failed"
        assert "folder" in (result["error"] or "")

    def test_a_missing_template_is_a_note_with_no_half_written_file(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            constants, "MCP_SETTINGS_TEMPLATE", workspace / "gone.default.json"
        )
        result = seed.ensure_mcp_settings_file(str(workspace))
        assert result["state"] == "failed"
        assert not seeded_path(workspace).exists()

    def test_a_reader_never_writes(self, workspace: Path):
        absent = seed.mcp_settings_file_state(str(workspace))
        assert absent["state"] == "absent"
        assert not (workspace / constants.POLICY_DIRNAME).exists()

        seed.ensure_mcp_settings_file(str(workspace))
        assert seed.mcp_settings_file_state(str(workspace))["state"] == "existing"


class TestTheInterpolation:
    def test_project_root_answers_from_the_app_not_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A stray ``PROJECT_ROOT`` must not redirect connectors elsewhere."""
        monkeypatch.setenv("PROJECT_ROOT", "/someone/elses/checkout")
        from src.config import PROJECT_ROOT

        expanded, missing = expand_env("${PROJECT_ROOT}/mcp_connectors/fedit_mcp.py")
        assert missing == []
        assert expanded == f"{PROJECT_ROOT}/mcp_connectors/fedit_mcp.py"

    def test_python_root_answers_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("BENCHMARKIT_PYTHON_ROOT", "/opt/py/python")
        expanded, missing = expand_env("${BENCHMARKIT_PYTHON_ROOT}")
        assert (expanded, missing) == ("/opt/py/python", [])

    def test_an_unset_python_root_is_a_note_not_a_blank(self, monkeypatch):
        monkeypatch.delenv("BENCHMARKIT_PYTHON_ROOT", raising=False)
        expanded, missing = expand_env("${BENCHMARKIT_PYTHON_ROOT}")
        assert expanded == "${BENCHMARKIT_PYTHON_ROOT}"
        assert missing == ["BENCHMARKIT_PYTHON_ROOT"]

    def test_the_workspace_switch_moves_the_expanded_root_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """One seeded file, two roots: ``--root-dir`` follows the location."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        for root in (first, second):
            root.mkdir()
            monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
            seed.ensure_mcp_settings_file(str(root))

        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(first))
        connectors, _ = load_externals()
        assert connectors[0]["command"][3] == str(first)

        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(second))
        connectors, _ = load_externals()
        assert connectors[0]["command"][3] == str(second)


class TestThePinStillPins:
    def test_the_pin_wins_over_the_workspace(self, workspace: Path, monkeypatch):
        pinned = workspace / "elsewhere" / "pinned.json"
        pinned.parent.mkdir()
        pinned.write_text(
            json.dumps({"only": {"type": "local", "enable": True, "command": ["x"]}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(constants, "MCP_SETTINGS_PATH", pinned)

        assert constants.mcp_settings_path() == pinned
        connectors, errors = load_externals()
        assert [connector["id"] for connector in connectors] == ["only"]
        assert errors == []
        assert set_externals_enabled({"only": False}) == []
        assert json.loads(pinned.read_text(encoding="utf-8"))["only"]["enable"] is False

    def test_unpinned_the_workspace_file_answers(self, workspace: Path):
        assert constants.MCP_SETTINGS_PATH is None
        assert constants.mcp_settings_path() == seeded_path(workspace)


# -- the two entry points, through the app ---------------------------------------


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
def seed_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A live app on a scratch root, declaration paths unpinned, no legacy lift."""
    from src import config as config_module
    from src.core import lsphub, prompt_helpers

    for key in config_module.SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(
        prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", tmp_path / "no-legacy.json"
    )
    # Unpin the declaration lookup: these tests assert the resolver follows the
    # *working location*, which a pin would defeat.
    monkeypatch.setattr(constants, "MCP_SETTINGS_PATH", None)
    # Hermetic LSP side (the switch restarts that hub too): the machine's own
    # deny-list and server file must not be read.
    servers_dir = tmp_path / "lsp_connectors"
    servers_dir.mkdir()
    (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
    monkeypatch.setattr(lsphub.constants, "LSP_CONNECTORS_DIR", servers_dir)
    monkeypatch.setattr(
        lsphub.constants, "LSP_SETTINGS_PATH", servers_dir / "lsp-settings.json"
    )
    monkeypatch.setattr(
        lsphub.constants, "LSP_RESTRICTED_PATHS", servers_dir / "restricted-paths.txt"
    )
    root = tmp_path / "boot-root"
    root.mkdir()
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
    with TestClient(create_app()) as client:
        client.app.state.mcp = _RecordingHub()
        client.app.state.lsp = _RecordingHub()
        yield client, root


class TestTheTwoEntryPoints:
    def test_boot_leaves_the_declaration_behind(self, seed_client):
        client, root = seed_client
        assert seeded_path(root).is_file()
        body = client.get("/api/path-location").json()
        assert body["mcpSettings"]["state"] == "existing"

    def test_get_never_seeds(self, seed_client, tmp_path: Path, monkeypatch):
        client, _ = seed_client
        other = tmp_path / "browsed"
        other.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(other))
        for _ in range(2):
            body = client.get("/api/path-location").json()
            assert body["mcpSettings"]["state"] == "absent"
        assert not seeded_path(other).exists()
        assert not other.joinpath(constants.POLICY_DIRNAME).exists()

    def test_switch_seeds_before_the_hub_restarts(
        self, seed_client, tmp_path: Path, monkeypatch
    ):
        """The order that the whole move depends on: fresh root, real connectors.

        With the MCP test switch lifted, ``_restart_mcp_hub`` hands the hub the
        live file scan; the recording hub captures it. If the seed ran after
        the restart, the new root would come up with zero connectors — the
        exact failure the old "all seeds after the restarts" order would hit
        once the declaration moved inside ``.nojae``.
        """
        client, _ = seed_client
        monkeypatch.delenv("BENCHMARKIT_DISABLE_MCP")
        target = tmp_path / "fresh-root"
        target.mkdir()

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        body = response.json()
        expanded = os.path.realpath(str(target))
        assert body["mcpSettings"] == {
            "state": "created",
            "path": os.path.join(expanded, constants.POLICY_DIRNAME,
                                 constants.MCP_SETTINGS_FILENAME),
            "error": None,
        }
        handed = last_restart_connectors(client)
        assert [connector["id"] for connector in handed] == ["console", "fedit"]
        # Every entry carries an unresolved ${BENCHMARKIT_PYTHON_ROOT} note —
        # honest, and proof the expansion ran on the seeded bytes.
        assert "BENCHMARKIT_PYTHON_ROOT" in (body["mcp"]["externalsError"] or "")
        assert os.environ["BENCHMARKIT_PATH_LOCATION"] == expanded

    def test_switch_reports_an_existing_declaration_untouched(
        self, seed_client, tmp_path: Path
    ):
        client, _ = seed_client
        target = tmp_path / "owned-root"
        (target / constants.POLICY_DIRNAME).mkdir(parents=True)
        handwritten = "{}\n"
        seeded_path(target).write_text(handwritten, encoding="utf-8")

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        assert response.json()["mcpSettings"]["state"] == "existing"
        assert seeded_path(target).read_text(encoding="utf-8") == handwritten

    def test_a_failing_seed_is_a_note_not_a_refusal(self, seed_client, tmp_path: Path):
        client, _ = seed_client
        target = tmp_path / "locked-root"
        target.mkdir()
        (target / constants.POLICY_DIRNAME).mkdir()
        seeded_path(target).mkdir()  # a folder where the file belongs

        response = client.post("/api/path-location", json={"path": str(target)})
        assert response.status_code == 200
        assert response.json()["mcpSettings"]["state"] == "failed"
        # The switch itself succeeded: the environment moved and the hubs
        # restarted — a seeding hiccup is a note, never a rollback.
        assert os.environ["BENCHMARKIT_PATH_LOCATION"] == os.path.realpath(
            str(target)
        )
        assert len(client.app.state.mcp.restarts) >= 1


def last_restart_connectors(client) -> List[Dict[str, Any]]:
    """The connector list the last hub restart was handed."""
    restarts = client.app.state.mcp.restarts
    assert restarts, "the MCP hub was not restarted"
    return list(restarts[-1][0])
