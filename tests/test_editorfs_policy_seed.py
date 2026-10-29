"""Tests for pol-05 — seeding a workspace's default policy file.

A workspace gets ``.nojae/read-write-policy.json`` **when it becomes active**
(boot, and the settings switch), so the syntax arrives as a worked example and
the first listing of a new root is already filtered. The tests therefore run
against scratch workspaces on disk — the file, the folder and the cache key are
all real paths — and the pair that must never drift is asserted directly: what
the seed writes and what pol-04's loader reads are the same rules.

The four promises this task exists to keep, each asserted more than one way:

* **never overwrites** — the create is exclusive (``O_CREAT | O_EXCL``), so a
  hand-edited policy survives and two callers have one winner;
* **a failure is a note** — an unwritable root answers ``"failed"`` with a
  message and raises nothing, and the switch still succeeds;
* **a reader never writes** — ``GET /api/path-location`` reports the file
  without leaving one behind;
* **the reserved folder stays unreachable** — the seeded file is not browsable
  from the Editor, on the reservation rather than on any rule.
"""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from mcp_connectors import policy_shared as shared

from src.api.server import create_app
from src.core.editorfs import constants, policy_file, policy_modes, policy_seed


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch working location with no policy file yet, cache clean."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    policy_file.invalidate_policy_cache()
    yield tmp_path
    policy_file.invalidate_policy_cache()


def seeded_path(root: Path) -> Path:
    return root / constants.POLICY_DIRNAME / constants.POLICY_FILENAME


def mode(rel: str) -> str:
    return policy_modes.mode_for(rel).mode


class TestTheSeedLands:
    def test_it_creates_the_file_and_it_is_valid_json(self, workspace: Path):
        result = policy_seed.ensure_policy_file(str(workspace))
        assert result["state"] == "created"
        assert result["error"] is None
        assert result["path"] == str(seeded_path(workspace))
        assert seeded_path(workspace).is_file()

        document = json.loads(seeded_path(workspace).read_text(encoding="utf-8"))
        assert document["version"] == constants.POLICY_VERSION
        assert isinstance(document["policies"], list) and document["policies"]
        assert isinstance(document["readme"], list) and document["readme"]

    def test_the_folder_is_one_level_and_the_file_is_readable(self, workspace: Path):
        """``0o644`` — the user is meant to open this file and edit it by hand."""
        policy_seed.ensure_policy_file(str(workspace))
        assert (workspace / constants.POLICY_DIRNAME).is_dir()
        if os.geteuid() != 0:
            assert os.access(seeded_path(workspace), os.R_OK)

    def test_a_second_call_is_existing_and_leaves_the_bytes_alone(
        self, workspace: Path
    ):
        policy_seed.ensure_policy_file(str(workspace))
        path = seeded_path(workspace)
        sentinel = path.read_text(encoding="utf-8").replace(
            '"version": 1', '"version": 1, "readme": ["mine"]'
        ).replace('{"path": ".git"', '{"path": "notes.md"')
        # The edit also *shrinks* the document, so a re-write would change the
        # size as well as the mtime and cannot hide behind either.
        path.write_text(sentinel, encoding="utf-8")
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)

        result = policy_seed.ensure_policy_file(str(workspace))
        assert result["state"] == "existing"
        assert result["error"] is None
        assert path.read_text(encoding="utf-8") == sentinel
        assert (path.stat().st_mtime_ns, path.stat().st_size) == stamp

    def test_the_race_is_closed_by_the_flag_not_by_a_check(self, workspace: Path):
        """A name that exists by the time of the write still loses, and no
        ``isfile()`` guard is involved: the seed is one ``O_EXCL`` syscall.

        A folder at the policy path is the most hostile version of that race —
        the exclusive open refuses it, and the seed must report rather than
        delete anything in the user's tree.
        """
        folder = workspace / constants.POLICY_DIRNAME
        folder.mkdir()
        (folder / constants.POLICY_FILENAME).mkdir()

        result = policy_seed.ensure_policy_file(str(workspace))
        assert result["state"] == "failed"
        assert result["error"]
        assert (folder / constants.POLICY_FILENAME).is_dir(), "nothing was deleted"

    def test_two_calls_in_a_row_have_one_winner(self, workspace: Path):
        states = [
            policy_seed.ensure_policy_file(str(workspace))["state"] for _ in range(2)
        ]
        assert states == ["created", "existing"]


class TestTheSeedRoundTripsThroughTheLoader:
    """pol-05 §4: the written file and the read rules may not drift."""

    def test_the_rendered_text_is_byte_for_byte_what_is_written(self, workspace: Path):
        policy_seed.ensure_policy_file(str(workspace))
        assert seeded_path(workspace).read_text(encoding="utf-8") == (
            policy_seed.default_policy_text()
        )

    def test_the_loader_accepts_it_without_a_single_finding(self, workspace: Path):
        policy_seed.ensure_policy_file(str(workspace))
        assert policy_file.workspace_findings() == ()
        assert policy_file.policy_blocking_error() is None
        assert len(policy_file.workspace_rules()) == len(policy_seed.DEFAULT_POLICY_RULES)

    def test_mode_for_answers_for_the_seeded_rules_immediately(
        self, workspace: Path
    ):
        """The seam pol-06 wires, asserted one step early — and it is also the
        test that the seed *invalidated* the cache: the loader had this
        workspace cached as "no policy file" a moment ago."""
        (workspace / "src" / "__pycache__").mkdir()
        (workspace / "web" / "node_modules" / "react").mkdir(parents=True)
        (workspace / ".git").mkdir()
        policy_seed.ensure_policy_file(str(workspace))

        assert mode(".git/config") == shared.HIDDEN
        assert mode("src/__pycache__/main.pyc") == shared.HIDDEN
        assert mode("web/node_modules/react/index.js") == shared.READ_ONLY
        assert mode("src/main.py") == shared.READ_WRITE

        verdict = policy_modes.mode_for("web/node_modules/react/index.js")
        assert verdict.level == shared.LEVEL_WORKSPACE
        assert verdict.pattern == "**/node_modules"
        assert verdict.reason == "browse the docs, edit nothing"

    def test_the_rules_the_seed_names_are_the_ones_pol_00_decided(self):
        """D6/D7: ``.git`` is a level-B rule (so the user *can* change it), and
        dependency/build trees are ``ro`` while pure derived noise is hidden."""
        by_pattern = {rule["path"]: rule["mode"] for rule in policy_seed.DEFAULT_POLICY_RULES}
        assert by_pattern[".git"] == shared.HIDDEN
        assert by_pattern["**/__pycache__"] == shared.HIDDEN
        assert by_pattern["**/node_modules"] == shared.READ_ONLY
        assert by_pattern["**/.venv"] == shared.READ_ONLY
        assert by_pattern["**/dist"] == shared.READ_ONLY
        # Every name the deleted hardcoded ignore-list used to hold is here.
        for name in (".git", "**/__pycache__", "**/node_modules", "**/.venv", "**/dist", "**/.pytest_cache"):
            assert name in by_pattern, name

    def test_the_readme_teaches_the_one_syntax_pol_00_documents(self):
        text = "\n".join(policy_seed.DEFAULT_POLICY_README)
        assert "LAST match wins" in text
        assert '{"path": "**/node_modules", "mode": "ro"}' in text
        assert '{"path": "**/node_modules/**/README.md", "mode": "rw"}' in text
        assert constants.POLICY_DIRNAME in text

    def test_a_write_that_fails_leaves_no_half_written_policy(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No-disk is the ``"failed"`` case, and it must not mean D4 forever.

        The file the failed call created is removed, because a *broken* policy
        file is an error finding that refuses every write in that workspace
        (pol-04 §5) — a mistake this call made must not lock the project.
        Absence is the state pol-04 reads as "no rules", so the workspace is
        back where it was and the next attempt can succeed.
        """
        real_fdopen = os.fdopen

        def exploding(fd, *args, **kwargs):
            os.close(fd)
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, "fdopen", exploding)
        result = policy_seed.ensure_policy_file(str(workspace))
        assert result["state"] == "failed"
        assert "space" in result["error"]
        assert not seeded_path(workspace).exists()
        assert policy_file.workspace_findings() == ()
        assert policy_file.policy_blocking_error() is None

        # The empty reserved folder is left alone, and the next call lands.
        monkeypatch.setattr(os, "fdopen", real_fdopen)
        assert policy_seed.ensure_policy_file(str(workspace))["state"] == "created"



class TestAFailureIsANote:
    def test_an_unwritable_root_answers_failed_and_raises_nothing(self, tmp_path: Path):
        """``chmod 0o500`` — a read-only mount or a foreign-owned root."""
        if os.geteuid() == 0:
            pytest.skip("root ignores the mode bits")
        locked = tmp_path / "read-only-project"
        locked.mkdir(mode=0o500)
        try:
            result = policy_seed.ensure_policy_file(str(locked))
            assert result["state"] == "failed"
            assert result["error"]
            assert result["path"].endswith(constants.POLICY_FILENAME)
            assert not (locked / constants.POLICY_DIRNAME).exists()
        finally:
            locked.chmod(0o755)

    def test_a_missing_root_answers_failed_rather_than_creating_a_chain(
        self, tmp_path: Path
    ):
        """The contract is that the root exists: ``os.mkdir`` is one level, and
        a parent that is not there is a symptom, not something to paper over."""
        result = policy_seed.ensure_policy_file(str(tmp_path / "never-made"))
        assert result["state"] == "failed"
        assert not (tmp_path / "never-made").exists()

    def test_the_public_call_never_raises_whatever_the_path_is(self, tmp_path: Path):
        """The whole function is guarded: a boot or a switch may not die on it."""
        for root in ("", str(tmp_path / "x" / "y" / "z"), "\x00bad"):
            result = policy_seed.ensure_policy_file(root)
            assert result["state"] in ("created", "existing", "failed")


class TestTheReaderDoesNotWrite:
    def test_policy_file_state_reports_absent_without_creating_anything(
        self, workspace: Path
    ):
        result = policy_seed.policy_file_state(str(workspace))
        assert result["state"] == "absent"
        assert result["path"] == str(seeded_path(workspace))
        assert not (workspace / constants.POLICY_DIRNAME).exists()

    def test_and_existing_after_a_seed(self, workspace: Path):
        policy_seed.ensure_policy_file(str(workspace))
        assert policy_seed.policy_file_state(str(workspace))["state"] == "existing"


class TestTheEndpoints:
    """Both call sites, and the payload shape the dialog renders."""

    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """``client`` with recording hubs and hermetic declaration files.

        The pattern ``tests/test_api.py`` uses for the hub endpoints: the real
        ``externals.json``/``lsp_servers.json`` of the developer's machine must
        not decide what a restart does here.
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
            lsphub.constants, "LSP_RESTRICTED_PATHS", servers_dir / "restricted-paths.txt"
        )
        with TestClient(app) as test_client:
            # Swapped *inside* the context, as ``tests/test_api.py``'s
            # ``hub_client`` does: the lifespan installs the real hubs on
            # startup, so a stand-in assigned before it is replaced and a
            # restart would go to a live hub.
            test_client.app.state.mcp = _RecordingHub()
            test_client.app.state.lsp = _RecordingHub()
            yield test_client

    def test_boot_seeds_the_location_it_chose(self, tmp_path: Path, monkeypatch):
        """The first of pol-05's two entry points, and the one that gives
        benchmarkit its own ``.nojae`` on first launch."""
        target = tmp_path / "booted-project"
        target.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(target))
        create_app()
        assert seeded_path(target).is_file()

    def test_post_to_a_scratch_dir_seeds_it_then_reports_existing(
        self, client: TestClient, tmp_path: Path
    ):
        target = tmp_path / "another-project"
        (target / "src").mkdir(parents=True)

        first = client.post("/api/path-location", json={"path": str(target)})
        assert first.status_code == 200
        policy = first.json()["policy"]
        assert policy["state"] == "created"
        assert policy["error"] is None
        assert policy["path"] == str(
            target.resolve() / constants.POLICY_DIRNAME / constants.POLICY_FILENAME
        )
        assert seeded_path(target).is_file()
        # The seed happens *after* both restarts, so a hub failure can never be
        # mistaken for one and roll the environment back (routes.py:1237).
        assert len(client.app.state.mcp.restarts) == 1
        assert len(client.app.state.lsp.restarts) == 1

        second = client.post("/api/path-location", json={"path": str(target)})
        assert second.json()["policy"]["state"] == "existing"

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
            policy = response.json()["policy"]
            assert policy["state"] == "failed"
            assert policy["error"]
            # The switch itself is untouched: the location moved and the hubs ran.
            assert os.environ["BENCHMARKIT_PATH_LOCATION"] == str(locked.resolve())
            assert len(client.app.state.mcp.restarts) == 1
        finally:
            locked.chmod(0o755)

    def test_get_carries_the_policy_block_without_seeding(
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
            assert body["policy"]["state"] == "existing"
            assert body["policy"]["path"] == str(seeded_path(target))
            assert client.get("/api/path-location").json()["policy"]["state"] == "existing"
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
            assert body["policy"]["state"] == "absent"
            assert not (target / constants.POLICY_DIRNAME).exists()
        finally:
            os.environ.pop("BENCHMARKIT_PATH_LOCATION", None)


class TestTheSeededFolderIsUnreachableFromTheEditor:
    """pol-05 §6.7 — and the reason seeding could not ship on its own.

    The seed *creates* ``.nojae/read-write-policy.json``, and the reservation is
    enforced on the gate rather than by a rule a file could undo: a config folder
    that appeared in the tree, or in ``git status``, after the app wrote it would
    be the leak D5 exists to prevent. Since pol-06 the surfaces answer from the
    verdict, so the shape of the refusal is the one every hidden path gets —
    absent to anything that looks (404, ``exists: false``), and a named 422 to
    anything that would change it.
    """

    @pytest.fixture
    def client(self, workspace: Path):
        policy_seed.ensure_policy_file(str(workspace))
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_nojae_is_absent_from_the_listing(self, client: TestClient):
        names = [entry["name"] for entry in client.get("/api/fs/tree").json()["entries"]]
        assert names == ["src"]

    def test_the_policy_file_is_not_readable(self, client: TestClient):
        response = client.get(
            "/api/fs/file",
            params={"path": f"{constants.POLICY_DIRNAME}/{constants.POLICY_FILENAME}"},
        )
        # pol-06 §2 settled the number: an *observation* of a hidden path answers
        # 404, because a refusal naming the rule would confirm the file is there.
        # The path itself is echoed back — that is the request the caller made,
        # not an admission about the disk, and the reservation is not a secret
        # (the folder is named by the app, not written by the user). What may not
        # appear under any circumstances: the file's contents.
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]
        assert "readme" not in response.text
        assert "policies" not in response.text

    def test_the_folder_itself_is_not_listable_as_a_directory(self, client: TestClient):
        response = client.get("/api/fs/tree", params={"path": constants.POLICY_DIRNAME})
        assert response.status_code in (404, 422)
        assert "entries" not in response.json()

    def test_the_freshness_probe_answers_absent_for_it(self, client: TestClient):
        """The probe only *looks*, and a look at a hidden name is absence (D8).

        The number flipped in pol-06 §2 on purpose: under the old gate a refusal
        was what kept the probe from confirming a secret was there, and under a
        policy the pane never receives the name from a listing that pruned it.
        """
        response = client.get(
            "/api/fs/stat",
            params={"path": f"{constants.POLICY_DIRNAME}/{constants.POLICY_FILENAME}"},
        )
        assert response.status_code == 200
        assert response.json()["paths"][0] == {
            "path": f"{constants.POLICY_DIRNAME}/{constants.POLICY_FILENAME}",
            "exists": False,
            "mtime": None,
            "size": None,
            "dir": False,
        }

    def test_a_mutation_of_it_is_refused(self, client: TestClient, workspace: Path):
        response = client.post(
            "/api/fs/create",
            json={"path": f"{constants.POLICY_DIRNAME}/notes.md", "dir": False},
        )
        # A mutation is the one answer that never becomes 404: 404 is the green
        # light for ``O_CREAT | O_EXCL`` (pol-00 D8).
        assert response.status_code == 422
        assert constants.POLICY_DIRNAME in response.json()["detail"]
        assert not (workspace / constants.POLICY_DIRNAME / "notes.md").exists()

    def test_the_refusal_survives_a_broken_policy_file(
        self, workspace: Path, client: TestClient
    ):
        """The reservation is not a rule, so a JSON typo cannot open the folder
        that holds the JSON (pol-04's asymmetry, asserted at the surface)."""
        seeded_path(workspace).write_text('{"version": 1, "policies": [', encoding="utf-8")
        policy_file.invalidate_policy_cache()
        response = client.get("/api/fs/tree", params={"path": constants.POLICY_DIRNAME})
        assert response.status_code in (404, 422)
        assert "entries" not in response.json()

    def test_the_reserved_name_is_patchable_for_the_surfaces_too(
        self, workspace: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(constants, "POLICY_DIRNAME", ".policydir")
        assert (
            client.get(
                "/api/fs/tree", params={"path": constants.POLICY_DIRNAME}
            ).status_code
            in (404, 422)
        )


class TestTheCallSitesAreTheTwoEntryPoints:
    """The seeding rule pol-05 §2 draws, pinned rather than trusted.

    Seeding belongs at the *two moments a workspace becomes active* — boot and
    the switch — and nowhere else. ``config.path_location()`` is a per-request
    reader, and a reader that writes is how a stray folder turns up in a project
    someone only meant to browse, so ``src/config.py`` may not call the seeder
    at all (and may not import ``editorfs``, which imports ``path_location``
    back — config sits at the bottom of the dependency graph).
    """

    REPO = Path(constants.__file__).resolve().parents[3]

    def sources(self) -> List[Path]:
        return sorted(
            path
            for path in (self.REPO / "src").rglob("*.py")
            if "__pycache__" not in path.parts
        )

    def test_config_never_seeds(self):
        text = (self.REPO / "src" / "config.py").read_text(encoding="utf-8")
        assert "ensure_policy_file" not in text
        assert "policy_seed" not in text

    def test_the_two_call_sites_are_the_boot_and_the_switch(self):
        callers = [
            path.relative_to(self.REPO).as_posix()
            for path in self.sources()
            if "ensure_policy_file(" in path.read_text(encoding="utf-8")
            and path.name != "policy_seed.py"
        ]
        assert callers == ["src/api/routes.py", "src/api/server.py"]

    def test_the_reserved_folder_is_out_of_source_control(self):
        """Acceptance: the seeded file must not turn up in ``git status``."""
        lines = (self.REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".nojae/" in [line.strip() for line in lines]

    def test_no_editorfs_write_route_or_gate_reads_the_seeded_text_back(self):
        """The seeder writes; only pol-04's loader reads.

        Two readers of one file is how the two halves of a policy disagree, and
        the same argument makes exactly *one* place allowed to create it.
        """
        # Parsed, not grepped: a docstring that *points at* the seeder is prose
        # and must not fail this; an import hidden inside a function must not
        # pass it (the same reasoning as pol-04's stdlib-only check).
        offenders: List[str] = []
        for name in ("tree.py", "files.py", "mutate.py", "gitstatus.py", "policy.py"):
            path = Path(constants.__file__).parent / name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "policy_seed"
                ):
                    offenders.append(name)
                elif isinstance(node, ast.Import):
                    if any(alias.name.endswith("policy_seed") for alias in node.names):
                        offenders.append(name)
        assert offenders == []

