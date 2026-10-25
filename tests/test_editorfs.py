"""Tests for the Editor tab's file surface: the tree endpoint and its policy.

The workspace is always a throw-away ``tmp_path`` pointed at by
``BENCHMARKIT_PATH_LOCATION`` (the pattern ``tests/test_batch.py`` uses), so
nothing here reads the repository's own checkout. Caps and the ignored-names
set are exercised by patching ``editorfs.constants`` — the module attributes
exist for exactly that.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core import editorfs


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small tree, pointed at as the working location.

    ``BENCHMARKIT_PATH_LOCATION`` is monkeypatched, so the value is restored
    after every test.
    """
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    (tmp_path / "src" / "core").mkdir(parents=True)
    (tmp_path / "src" / "config.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "core" / "job.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    (tmp_path / "empty-dir").mkdir()
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    (tmp_path / ".venv").mkdir()
    return tmp_path


@pytest.fixture
def client(workspace: Path):
    with TestClient(create_app()) as test_client:
        yield test_client


def tree(client: TestClient, path: Optional[str] = None):
    params: Dict[str, str] = {} if path is None else {"path": path}
    return client.get("/api/fs/tree", params=params)


def names(payload: dict) -> List[str]:
    return [entry["name"] for entry in payload["entries"]]


class TestRootListing:
    def test_dirs_first_then_files_each_alphabetical(self, client):
        payload = tree(client).json()
        assert payload["path"] == ""
        assert payload["truncated"] is False
        assert payload["entries"], "the root should not be empty"
        dirs = [entry for entry in payload["entries"] if entry["dir"]]
        files = [entry for entry in payload["entries"] if not entry["dir"]]
        assert [entry["name"] for entry in payload["entries"]] == (
            [entry["name"] for entry in dirs] + [entry["name"] for entry in files]
        )
        assert [entry["name"] for entry in dirs] == sorted(
            (entry["name"] for entry in dirs), key=str.lower
        )
        assert [entry["name"] for entry in files] == ["README.md"]

    def test_paths_are_workspace_relative(self, workspace: Path, client):
        payload = tree(client, "src").json()
        assert payload["path"] == "src"
        assert payload["workspace"] == str(workspace.resolve())
        by_name = {entry["name"]: entry for entry in payload["entries"]}
        assert by_name["core"]["path"] == "src/core"
        assert by_name["config.py"]["path"] == "src/config.py"

    def test_directories_carry_mtime_and_children_flag(self, client):
        by_name = {entry["name"]: entry for entry in tree(client).json()["entries"]}
        assert by_name["src"]["children"] is True
        assert isinstance(by_name["src"]["mtime"], float)
        assert by_name["empty-dir"]["children"] is False
        assert "size" not in by_name["src"]

    def test_files_carry_size_and_mtime(self, client):
        readme = next(e for e in tree(client).json()["entries"] if e["dir"] is False)
        assert readme["name"] == "README.md"
        assert readme["size"] == len("# hi\n")
        assert isinstance(readme["mtime"], float)
        assert "children" not in readme

    def test_ignored_names_are_absent(self, client):
        assert not {"node_modules", ".venv"} & set(names(tree(client).json()))

    def test_ignored_set_is_patchable_through_constants(
        self, client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(editorfs.constants, "IGNORED_DIR_NAMES", frozenset({"src"}))
        assert "src" not in names(tree(client).json())

    def test_only_ignored_children_counts_as_childless(
        self, workspace: Path, client
    ):
        """``children`` uses the same two filters as the listing itself."""
        (workspace / "mostly-cache" / "__pycache__").mkdir(parents=True)
        by_name = {entry["name"]: entry for entry in tree(client).json()["entries"]}
        assert by_name["mostly-cache"]["children"] is False
        assert tree(client, "mostly-cache").json()["entries"] == []


class TestPathLocks:
    @pytest.mark.parametrize(
        "raw", ["../etc", "a/../../b", "/etc", "~/", "..", "src/../../etc"]
    )
    def test_escape_attempts_are_422(self, client, raw: str):
        assert tree(client, raw).status_code == 422

    def test_absolute_path_detail_names_the_rule(self, client):
        assert "relative to the workspace" in tree(client, "/etc").json()["detail"]

    def test_nul_byte_is_refused_by_the_textual_lock(self):
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.normalise_rel("a\0b")
        assert caught.value.status_code == 422

    def test_traversal_inside_the_workspace_stays_legal(self, client):
        payload = tree(client, "src/../src").json()
        assert payload["path"] == "src"
        assert "config.py" in names(payload)

    def test_symlink_to_outside_is_listed_but_not_followed(
        self, workspace: Path, client
    ):
        outside = workspace.parent / "outside-target"
        outside.mkdir(exist_ok=True)
        (outside / "secret.txt").write_text("no\n", encoding="utf-8")
        (workspace / "link").symlink_to(outside)
        entry = next(e for e in tree(client).json()["entries"] if e["name"] == "link")
        assert entry["dir"] is False, "a symlinked directory is never descended into"
        # Walking through it is refused by the realpath prefix lock.
        response = tree(client, "link")
        assert response.status_code == 422
        assert "outside the working location" in response.json()["detail"]
        assert tree(client, "link/secret.txt").status_code == 422

    def test_denied_names_are_refused_and_never_listed(
        self, workspace: Path, client
    ):
        (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (workspace / ".ssh").mkdir()
        (workspace / ".ssh" / "id_ed25519").write_text("priv\n", encoding="utf-8")
        assert tree(client, ".env").status_code == 422
        assert tree(client, ".ssh").status_code == 422
        assert tree(client, ".ssh/id_ed25519").status_code == 422
        # Denied siblings are omitted, not marked — no lock icon, no leak.
        listed = names(tree(client).json())
        assert ".env" not in listed and ".ssh" not in listed

    def test_workspace_root_on_the_lsp_deny_list_is_422(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The tree inherits the guard ``POST /path-location`` already applies."""
        denied = tmp_path / "denied-root"
        (denied / "src").mkdir(parents=True)
        list_file = tmp_path / "restricted.txt"
        list_file.write_text(f"{denied}\n", encoding="utf-8")
        monkeypatch.setattr("src.core.lsphub.constants.LSP_RESTRICTED_PATHS", list_file)
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(denied))
        with TestClient(create_app()) as denied_client:
            response = denied_client.get("/api/fs/tree")
        assert response.status_code == 422
        assert "deny-list" in response.json()["detail"]


class TestMissingAndFile:
    def test_missing_directory_is_404(self, client):
        response = tree(client, "nope")
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]

    def test_file_path_is_400(self, client):
        response = tree(client, "README.md")
        assert response.status_code == 400
        assert "not a directory" in response.json()["detail"]


class TestEntryCap:
    @pytest.mark.parametrize("extra", [5, 0])
    def test_cap_is_exact_and_only_then_truncated(
        self, workspace: Path, client, monkeypatch: pytest.MonkeyPatch, extra: int
    ):
        monkeypatch.setattr(editorfs.constants, "MAX_TREE_ENTRIES_PER_DIR", 10)
        base = len(tree(client).json()["entries"])
        for index in range(10 - base + extra):
            (workspace / f"file-{index:02d}.txt").write_text("x", encoding="utf-8")
        payload = tree(client).json()
        assert len(payload["entries"]) == 10
        assert payload["truncated"] is (extra > 0)


class TestStatEndpoint:
    def test_reports_mtime_size_and_absence(self, client):
        response = client.get(
            "/api/fs/stat", params={"path": ["README.md", "gone.txt"]}
        )
        assert response.status_code == 200
        by_path = {item["path"]: item for item in response.json()["paths"]}
        assert by_path["README.md"]["exists"] is True
        assert by_path["README.md"]["size"] == len("# hi\n")
        assert isinstance(by_path["README.md"]["mtime"], float)
        assert by_path["gone.txt"] == {
            "path": "gone.txt",
            "exists": False,
            "mtime": None,
            "size": None,
            "dir": False,
        }

    def test_denied_path_is_a_refusal_not_exists_false(
        self, workspace: Path, client
    ):
        """Otherwise the probe would leak that a secret file exists."""
        (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
        assert client.get("/api/fs/stat", params={"path": ".env"}).status_code == 422


class TestReadOnlySurface:
    def test_no_write_route_exists_under_api_fs(self):
        """This lot's read-only promise, written as a test (decision D4).

        Read through the OpenAPI paths rather than ``app.routes``: the router is
        included lazily, so the flat path map is the one view that lists every
        endpoint the app actually answers.
        """
        paths = create_app().openapi()["paths"]
        fs = {path: sorted(ops) for path, ops in paths.items() if path.startswith("/api/fs")}
        assert fs, "the /api/fs routes should be registered"
        for path, ops in fs.items():
            assert ops == ["get"], f"{path} exposes {ops}, not GET only"

    def test_editorfs_exposes_no_write_helper(self):
        for name in ("write_file", "save_file", "delete_path", "create_file"):
            assert not hasattr(editorfs, name)
