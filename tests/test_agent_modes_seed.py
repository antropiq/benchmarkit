"""am-03 — seeding ``.nojae/agent-modes.json`` when a workspace becomes active.

The modes file arrives the way pol-05's policy file does, so this suite has the
shape of ``tests/test_editorfs_policy_seed.py``: scratch workspaces on disk, the
helpers exercised directly, and the two entry points checked through the app.
What differs is only what a *missing* file costs — a workspace with no modes
file chats on the built-in modes — so an unseeded root is a shrug rather than a
broken editor.

The four promises this task exists to keep:

* **never overwrites** — one ``O_CREAT | O_EXCL``, so a hand-edited modes file
  survives and two callers have one winner;
* **the seed and the loader are the same bytes** — what
  :func:`ensure_agent_modes_file` writes is am-02's document, and the loader
  accepts it with zero findings;
* **a failure is a note** — an unwritable root answers ``"failed"`` and raises
  nothing, and the switch still succeeds;
* **a reader never writes** — ``agent_modes_file_state`` and
  ``GET /api/path-location`` report the file without leaving one behind.
"""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core import agent_modes
from src.core import agent_modes_defaults as defaults
from src.core.editorfs import constants as editorfs_constants
from src.core.editorfs import policy_file, policy_seed
from src.core import prompt_helpers


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch working location with no modes file yet, both caches clean."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    agent_modes.invalidate_agent_modes_cache()
    policy_file.invalidate_policy_cache()
    yield tmp_path
    agent_modes.invalidate_agent_modes_cache()
    policy_file.invalidate_policy_cache()


def seeded_path(root: Path) -> Path:
    return root / agent_modes.POLICY_DIRNAME / agent_modes.AGENT_MODES_FILENAME


class TestTheSeedLands:
    def test_it_creates_the_file_and_the_loader_accepts_it(self, workspace: Path):
        result = agent_modes.ensure_agent_modes_file(str(workspace))
        assert result["state"] == "created"
        assert result["error"] is None
        assert result["path"] == str(seeded_path(workspace))
        assert seeded_path(workspace).is_file()

        document = json.loads(seeded_path(workspace).read_text(encoding="utf-8"))
        assert document["version"] == agent_modes.AGENT_MODES_VERSION
        assert isinstance(document["readme"], list) and document["readme"]
        assert list(document["agents"]) == ["chat", "code", "architect"]

    def test_resolve_modes_on_the_seeded_root_is_the_four_modes_no_findings(
        self, workspace: Path
    ):
        """The point of seeding: a fresh workspace's dropdown is the designed one."""
        agent_modes.ensure_agent_modes_file(str(workspace))
        payload = agent_modes.resolve_modes()
        assert payload["findings"] == []
        assert payload["source"]["loaded"] is True
        assert payload["default"] == defaults.CODE_MODE_ID
        assert [mode.id for mode in payload["modes"]] == [
            "chat", "code", "architect", "todoBuilder",
        ]

    @pytest.mark.posix_behaviour(
        "the readable-file half is the 0o644 the seeder asks for: mode bits and "
        "the uid that may ignore them are POSIX facts (the folder-is-one-level "
        "half above is platform-free and still runs)"
    )
    def test_the_folder_is_one_level_and_the_file_is_readable(self, workspace: Path):
        """``0o644`` — this file is meant to be opened and edited by hand (D6)."""
        agent_modes.ensure_agent_modes_file(str(workspace))
        assert (workspace / agent_modes.POLICY_DIRNAME).is_dir()
        if os.geteuid() != 0:
            assert os.access(seeded_path(workspace), os.R_OK)

    def test_an_existing_nojae_folder_is_not_touched(self, workspace: Path):
        """The policy seed usually creates the folder a moment earlier."""
        folder = workspace / agent_modes.POLICY_DIRNAME
        folder.mkdir()
        (folder / "keepme.txt").write_text("mine\n", encoding="utf-8")

        assert agent_modes.ensure_agent_modes_file(str(workspace))["state"] == "created"
        assert (folder / "keepme.txt").read_text(encoding="utf-8") == "mine\n"

    def test_the_reserved_folder_is_the_only_new_entry(self, workspace: Path):
        """Acceptance: a seeded workspace gains the file and nothing else."""
        agent_modes.ensure_agent_modes_file(str(workspace))
        assert sorted(os.listdir(workspace)) == sorted(["src", agent_modes.POLICY_DIRNAME])
        assert os.listdir(workspace / agent_modes.POLICY_DIRNAME) == [
            agent_modes.AGENT_MODES_FILENAME
        ]


class TestItNeverOverwrites:
    def test_a_hand_edited_file_survives_the_next_seed(self, workspace: Path):
        agent_modes.ensure_agent_modes_file(str(workspace))
        path = seeded_path(workspace)
        edited = path.read_text(encoding="utf-8").replace(
            '"version": 1', '"version": 1, "readme": ["mine"]'
        )
        # Deliberately *shorter* as well as different, so a re-write would move
        # the size and cannot hide behind an unchanged mtime.
        path.write_text(edited, encoding="utf-8")
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)

        result = agent_modes.ensure_agent_modes_file(str(workspace))
        assert result["state"] == "existing"
        assert result["error"] is None
        assert path.read_text(encoding="utf-8") == edited
        assert (path.stat().st_mtime_ns, path.stat().st_size) == stamp

    def test_two_calls_in_a_row_have_one_winner(self, workspace: Path):
        states = [
            agent_modes.ensure_agent_modes_file(str(workspace))["state"]
            for _ in range(2)
        ]
        assert states == ["created", "existing"]

    def test_the_race_is_closed_by_the_flag_not_by_a_check(self, workspace: Path):
        """A folder standing at the modes path is the hostile version of that
        race: the exclusive open refuses it and nothing in the tree is deleted."""
        folder = workspace / agent_modes.POLICY_DIRNAME
        folder.mkdir()
        (folder / agent_modes.AGENT_MODES_FILENAME).mkdir()

        result = agent_modes.ensure_agent_modes_file(str(workspace))
        assert result["state"] == "failed"
        assert result["error"]
        assert (folder / agent_modes.AGENT_MODES_FILENAME).is_dir()

    def test_the_create_is_exclusive_and_the_folder_is_one_level(self):
        """The three non-negotiables as a source rule, not a trust statement.

        Parsed rather than grepped: the module's own prose *names* ``makedirs``
        to say it is not used, and a docstring must not fail a check whose point
        is what the code calls.
        """
        tree = ast.parse(Path(agent_modes.__file__).read_text(encoding="utf-8"))
        called = {
            f"{node.func.value.id}.{node.func.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }
        # ``os.open`` with the exclusive flag, ``os.mkdir`` one level, and no
        # check-then-write TOCTOU anywhere in the module.
        assert "os.mkdir" in called
        assert not {"os.makedirs", "os.path.isfile", "os.path.exists"} & called
        assert any(
            "O_CREAT" in ast.dump(node) and "O_EXCL" in ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        )


class TestTheSeedRoundTripsThroughTheLoader:
    """am-03 §1: the written file and the read modes may not drift."""

    def test_the_rendered_text_is_byte_for_byte_what_is_written(self, workspace: Path):
        agent_modes.ensure_agent_modes_file(str(workspace))
        assert seeded_path(workspace).read_text(encoding="utf-8") == (
            defaults.default_agent_modes_text()
        )

    def test_the_seeded_bytes_are_the_document_they_came_from(self, workspace: Path):
        agent_modes.ensure_agent_modes_file(str(workspace))
        assert json.loads(seeded_path(workspace).read_text(encoding="utf-8")) == (
            defaults.DEFAULT_AGENTS_DOCUMENT
        )

    def test_a_write_that_fails_leaves_no_half_written_modes_file(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No-disk must not mean a permanently broken workspace.

        The file the failed call created is removed: absence is the state the
        loader reads as "the built-in defaults", where a fragment would be an
        error finding of this call's own making.
        """
        real_fdopen = os.fdopen

        def exploding(fd, *args, **kwargs):
            os.close(fd)
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, "fdopen", exploding)
        result = agent_modes.ensure_agent_modes_file(str(workspace))
        assert result["state"] == "failed"
        assert "space" in result["error"]
        assert not seeded_path(workspace).exists()
        # And the loader is unbothered: no findings, the built-in modes answer.
        payload = agent_modes.resolve_modes()
        assert payload["findings"] == []
        assert payload["source"]["exists"] is False

        monkeypatch.setattr(os, "fdopen", real_fdopen)
        assert agent_modes.ensure_agent_modes_file(str(workspace))["state"] == "created"


class TestAFailureIsANote:
    @pytest.mark.posix_behaviour(
        "an unwritable root is made with ``mkdir(mode=0o500)``: without POSIX "
        "mode bits there is no root the seeder refuses to write into"
    )
    def test_an_unwritable_root_answers_failed_and_raises_nothing(self, tmp_path: Path):
        """``chmod 0o500`` — a read-only mount or a foreign-owned root."""
        if os.geteuid() == 0:
            pytest.skip("root ignores the mode bits")
        locked = tmp_path / "read-only-project"
        locked.mkdir(mode=0o500)
        try:
            result = agent_modes.ensure_agent_modes_file(str(locked))
            assert result["state"] == "failed"
            assert result["error"]
            assert result["path"].endswith(agent_modes.AGENT_MODES_FILENAME)
            assert not (locked / agent_modes.POLICY_DIRNAME).exists()
        finally:
            locked.chmod(0o755)

    def test_a_missing_root_answers_failed_rather_than_creating_a_chain(
        self, tmp_path: Path
    ):
        result = agent_modes.ensure_agent_modes_file(str(tmp_path / "never-made"))
        assert result["state"] == "failed"
        assert not (tmp_path / "never-made").exists()

    def test_the_public_call_never_raises_whatever_the_root_is(self, tmp_path: Path):
        """The whole function is guarded: a boot or a switch may not die on it."""
        for root in ("", str(tmp_path / "x" / "y" / "z"), "\x00bad"):
            result = agent_modes.ensure_agent_modes_file(root)
            assert result["state"] in ("created", "existing", "failed")


class TestTheReaderDoesNotWrite:
    def test_file_state_reports_absent_without_creating_anything(
        self, workspace: Path
    ):
        result = agent_modes.agent_modes_file_state(str(workspace))
        assert result["state"] == "absent"
        assert result["path"] == str(seeded_path(workspace))
        assert result["error"] is None
        assert not (workspace / agent_modes.POLICY_DIRNAME).exists()

    def test_and_existing_after_a_seed(self, workspace: Path):
        agent_modes.ensure_agent_modes_file(str(workspace))
        assert agent_modes.agent_modes_file_state(str(workspace))["state"] == "existing"

    def test_a_folder_at_the_modes_path_is_not_reported_as_absent(self, workspace: Path):
        """The loader calls that a file-level error; the reader says the same."""
        folder = workspace / agent_modes.POLICY_DIRNAME
        folder.mkdir()
        (folder / agent_modes.AGENT_MODES_FILENAME).mkdir()
        result = agent_modes.agent_modes_file_state(str(workspace))
        assert result["state"] == "failed"
        assert "folder" in result["error"]


class TestTheSeedInvalidatesTheCache:
    def test_the_first_read_after_a_seed_sees_the_file(self, workspace: Path):
        """The part that is easy to forget: the *absent* answer was cached."""
        assert agent_modes.resolve_modes()["source"]["exists"] is False
        agent_modes.ensure_agent_modes_file(str(workspace))
        payload = agent_modes.resolve_modes()
        assert payload["source"]["exists"] is True
        assert payload["source"]["loaded"] is True


class TestTheEndpoints:
    """Both call sites, and the payload shape the dialog renders."""

    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """``client`` with recording hubs and hermetic declaration files.

        The pattern ``tests/test_editorfs_policy_seed.py`` and ``tests/test_api.py``
        use for the hub endpoints: the real ``mcp-settings.json``/``lsp-settings.json``
        of the developer's machine must not decide what a restart does here.
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

        app = create_app()
        monkeypatch.delenv("BENCHMARKIT_PATH_LOCATION", raising=False)
        empty = tmp_path / "mcp-settings.json"
        empty.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(mcphub.constants, "MCP_SETTINGS_PATH", empty)
        servers_dir = tmp_path / "lsp_connectors"
        servers_dir.mkdir()
        (servers_dir / "restricted-paths.txt").write_text("/etc\n", encoding="utf-8")
        monkeypatch.setattr(lsphub.constants, "LSP_CONNECTORS_DIR", servers_dir)
        monkeypatch.setattr(
            lsphub.constants, "LSP_SETTINGS_PATH", servers_dir / "lsp-settings.json"
        )
        monkeypatch.setattr(
            lsphub.constants, "LSP_RESTRICTED_PATHS",
            servers_dir / "restricted-paths.txt",
        )
        with TestClient(app) as test_client:
            # Swapped *inside* the context: the lifespan installs the real hubs
            # on startup, so a stand-in assigned before it is replaced and a
            # restart would go to a live hub.
            test_client.app.state.mcp = _RecordingHub()
            test_client.app.state.lsp = _RecordingHub()
            yield test_client

    def test_boot_seeds_both_files_of_the_reserved_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The first of am-03's two entry points: a brand-new workspace opened
        at boot ends up with exactly the files of the reserved folder — the
        policy, the modes, the prompt helpers, the MCP declaration, the LSP
        template (never the LSP declaration, which is hand-written), and the
        workspace settings — and nothing else."""
        from src.core import lsphub, mcphub, workspace_settings

        target = tmp_path / "booted-project"
        target.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(target))
        create_app()
        folder = target / agent_modes.POLICY_DIRNAME
        assert seeded_path(target).is_file()
        assert policy_seed.policy_file_path(str(target)) == str(
            folder / editorfs_constants.POLICY_FILENAME
        )
        assert (folder / editorfs_constants.POLICY_FILENAME).is_file()
        assert (
            folder / prompt_helpers.PROMPT_HELPERS_FILENAME
        ).is_file()
        assert (
            folder / mcphub.constants.MCP_SETTINGS_FILENAME
        ).is_file()
        assert (
            folder / lsphub.constants.LSP_SETTINGS_TEMPLATE_FILENAME
        ).is_file()
        assert not (folder / lsphub.constants.LSP_SETTINGS_FILENAME).is_file()
        assert (
            folder / workspace_settings.WORKSPACE_SETTINGS_FILENAME
        ).is_file()
        assert sorted(name.name for name in folder.iterdir()) == sorted(
            [
                editorfs_constants.POLICY_FILENAME,
                agent_modes.AGENT_MODES_FILENAME,
                prompt_helpers.PROMPT_HELPERS_FILENAME,
                mcphub.constants.MCP_SETTINGS_FILENAME,
                lsphub.constants.LSP_SETTINGS_TEMPLATE_FILENAME,
                workspace_settings.WORKSPACE_SETTINGS_FILENAME,
            ]
        )
        assert sorted(os.listdir(target)) == sorted(
            [agent_modes.POLICY_DIRNAME]
        )

    def test_post_to_a_scratch_dir_seeds_it_then_reports_existing(
        self, client: TestClient, tmp_path: Path
    ):
        target = tmp_path / "another-project"
        (target / "src").mkdir(parents=True)

        first = client.post("/api/path-location", json={"path": str(target)})
        assert first.status_code == 200
        modes = first.json()["agentModes"]
        assert modes == {
            "state": "created",
            "path": str(
                target.resolve()
                / agent_modes.POLICY_DIRNAME
                / agent_modes.AGENT_MODES_FILENAME
            ),
            "error": None,
        }
        assert seeded_path(target).is_file()
        # The seed happens *after* both restarts, so a hub failure can never be
        # mistaken for one and roll the environment back.
        assert len(client.app.state.mcp.restarts) == 1
        assert len(client.app.state.lsp.restarts) == 1

        second = client.post("/api/path-location", json={"path": str(target)})
        assert second.json()["agentModes"]["state"] == "existing"

    def test_the_switch_leaves_the_new_root_readable_through_the_loader(
        self, client: TestClient, tmp_path: Path
    ):
        """am-03 §4: the modes cache must not answer the new root with the old
        root's absence."""
        target = tmp_path / "switched-project"
        target.mkdir()
        client.post("/api/path-location", json={"path": str(target)})
        payload = agent_modes.resolve_modes()
        assert payload["source"]["exists"] is True
        assert payload["findings"] == []

    @pytest.mark.posix_behaviour(
        "the unwritable root this POST is switched to is made with "
        "``mkdir(mode=0o500)``, a POSIX fact: the switch and the note it carries "
        "are only observable once a root can be made unwritable"
    )
    def test_post_to_an_unwritable_root_still_switches_and_says_so(
        self, client: TestClient, tmp_path: Path
    ):
        if os.geteuid() == 0:
            pytest.skip("root ignores the mode bits")
        locked = tmp_path / "read-only-project"
        locked.mkdir(mode=0o500)
        try:
            response = client.post("/api/path-location", json={"path": str(locked)})
            assert response.status_code == 200
            modes = response.json()["agentModes"]
            assert modes["state"] == "failed"
            assert modes["error"]
            # The switch itself is untouched: the location moved and the hubs ran.
            assert os.environ["BENCHMARKIT_PATH_LOCATION"] == str(locked.resolve())
            assert len(client.app.state.mcp.restarts) == 1
        finally:
            locked.chmod(0o755)

    def test_get_carries_the_modes_block_without_seeding(
        self, client: TestClient, tmp_path: Path
    ):
        target = tmp_path / "already-seeded"
        target.mkdir()
        client.post("/api/path-location", json={"path": str(target)})
        os.environ["BENCHMARKIT_PATH_LOCATION"] = str(target)
        try:
            body = client.get("/api/path-location").json()
            assert body["key"] == "BENCHMARKIT_PATH_LOCATION"
            assert body["path"] == str(target)
            # The POST seeded it, so GET reports it — and GET itself wrote
            # nothing: the same answer on a second GET proves nothing changed.
            assert body["agentModes"]["state"] == "existing"
            assert body["agentModes"]["path"] == str(seeded_path(target))
            again = client.get("/api/path-location").json()
            assert again["agentModes"]["state"] == "existing"
            assert again["policy"]["state"] == "existing"
        finally:
            os.environ.pop("BENCHMARKIT_PATH_LOCATION", None)

    def test_a_location_that_was_never_posted_reports_absent(
        self, client: TestClient, tmp_path: Path
    ):
        """The read half with no POST in front of it: the settings dialog may
        open a hundred times without leaving a folder in the browsed project."""
        target = tmp_path / "only-browsed"
        target.mkdir()
        os.environ["BENCHMARKIT_PATH_LOCATION"] = str(target)
        try:
            body = client.get("/api/path-location").json()
            assert body["path"] == str(target)
            assert body["agentModes"]["state"] == "absent"
            assert not (target / agent_modes.POLICY_DIRNAME).exists()
        finally:
            os.environ.pop("BENCHMARKIT_PATH_LOCATION", None)


class TestTheSeededFileIsUnreachableFromTheEditor:
    """The reserved folder is reserved for both files in it, not just the policy.

    The seed *creates* ``.nojae/agent-modes.json``; the reservation is enforced
    on the gate rather than by a rule a file could undo, so a config folder that
    appeared in the tree after the app wrote it would be the leak D5 exists to
    prevent. An observation of a hidden path answers absent (404); a mutation is
    the one answer that names the rule.
    """

    @pytest.fixture
    def client(self, workspace: Path):
        agent_modes.ensure_agent_modes_file(str(workspace))
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_the_modes_file_is_not_readable(self, client: TestClient):
        response = client.get(
            "/api/fs/file",
            params={
                "path": (
                    f"{agent_modes.POLICY_DIRNAME}/{agent_modes.AGENT_MODES_FILENAME}"
                )
            },
        )
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]
        assert "systemMessage" not in response.text
        assert "architect" not in response.text

    def test_the_folder_itself_is_not_listable(self, client: TestClient):
        response = client.get(
            "/api/fs/tree", params={"path": agent_modes.POLICY_DIRNAME}
        )
        assert response.status_code in (404, 422)
        assert "entries" not in response.json()

    def test_a_mutation_of_it_is_refused(self, client: TestClient, workspace: Path):
        response = client.post(
            "/api/fs/create",
            json={
                "path": f"{agent_modes.POLICY_DIRNAME}/agent-modes-copy.json",
                "dir": False,
            },
        )
        assert response.status_code == 422
        assert agent_modes.POLICY_DIRNAME in response.json()["detail"]
        assert not (
            workspace / agent_modes.POLICY_DIRNAME / "agent-modes-copy.json"
        ).exists()


class TestTheCallSitesAreTheTwoEntryPoints:
    """am-03 §2: seeding belongs at the two moments a workspace becomes active.

    ``config.path_location()`` is a per-request *reader*, and a reader that
    writes is how a stray folder turns up in a project someone only meant to
    browse — so ``src/config.py`` may not call the seeder at all, and the boot
    call belongs in the app that boots (``config`` sits at the bottom of the
    import graph and may not import ``core``).
    """

    REPO = Path(agent_modes.__file__).resolve().parents[2]

    def sources(self) -> List[Path]:
        return sorted(
            path
            for path in (self.REPO / "src").rglob("*.py")
            if "__pycache__" not in path.parts
        )

    def test_config_never_seeds_modes(self):
        text = (self.REPO / "src" / "config.py").read_text(encoding="utf-8")
        assert "ensure_agent_modes_file" not in text
        assert "agent_modes" not in text

    def test_the_two_call_sites_are_the_boot_and_the_switch(self):
        seeder = Path(agent_modes.__file__)
        callers = [
            path.relative_to(self.REPO).as_posix()
            for path in self.sources()
            if "ensure_agent_modes_file(" in path.read_text(encoding="utf-8")
            and path != seeder
        ]
        assert callers == ["src/api/routes.py", "src/api/server.py"]

    def test_the_boot_wrapper_is_shaped_like_the_policy_one(self):
        """``_seed_agent_modes`` sits beside ``_seed_workspace_policy`` and
        guards the same way, so a seed failure cannot take the boot down."""
        from src.api import server

        source = Path(server.__file__).read_text(encoding="utf-8")
        for name in ("_seed_workspace_policy", "_seed_agent_modes"):
            assert f"def {name}(" in source
            assert callable(getattr(server, name))
        assert server._seed_agent_modes.__module__ == "src.api.server"

    def test_the_reserved_folder_is_out_of_source_control(self):
        """Acceptance: the seeded file must not turn up in ``git status`` — and
        it is the *folder* that is ignored, never a line for this one file."""
        lines = [
            line.strip()
            for line in (self.REPO / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert ".nojae/" in lines
        assert not any("agent-modes" in line for line in lines)
