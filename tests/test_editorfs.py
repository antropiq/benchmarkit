"""Tests for the Editor tab's file surface: the tree, the file read and their
shared policy.

The workspace is always a throw-away ``tmp_path`` pointed at by
``BENCHMARKIT_PATH_LOCATION`` (the pattern ``tests/test_batch.py`` uses), so
nothing here reads the repository's own checkout. Caps and the ignored-names
set are exercised by patching ``editorfs.constants`` — the module attributes
exist for exactly that.
"""

import os
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


def read(client: TestClient, path: str):
    return client.get("/api/fs/file", params={"path": path})


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


class TestReadEndpoint:
    """``GET /api/fs/file`` — every rule of the read route, mirrored by a test."""

    def test_text_read_payload(self, workspace: Path, client):
        response = read(client, "src/config.py")
        assert response.status_code == 200
        payload = response.json()
        assert payload["path"] == "src/config.py"
        assert payload["text"] == "x = 1\n"
        assert payload["size"] == 6
        assert payload["mtime"] == (workspace / "src" / "config.py").stat().st_mtime
        assert payload["language"] == "python"
        assert payload["binary"] is False
        assert payload["truncated"] is False
        assert payload["encoding"] == "utf-8"

    @pytest.mark.parametrize(
        ("name", "language"),
        [
            ("a.py", "python"), ("a.pyi", "python"),
            ("a.js", "javascript"), ("a.mjs", "javascript"),
            ("a.cjs", "javascript"), ("a.jsx", "javascript"),
            ("a.ts", "typescript"), ("a.tsx", "typescript"),
            ("a.vue", "vue"), ("a.json", "json"), ("a.jsonc", "json"),
            ("a.md", "markdown"), ("a.html", "html"), ("a.css", "css"),
            ("a.scss", "scss"), ("a.toml", "toml"), ("a.yaml", "yaml"),
            ("a.yml", "yaml"), ("a.sh", "shell"),
            # Unknown extensions and dotfiles never get a guessed grammar.
            ("Dockerfile", "plaintext"), ("Makefile", "plaintext"),
            (".gitignore", "plaintext"), ("a.xyz", "plaintext"),
        ],
    )
    def test_extension_table(self, workspace: Path, client, name: str, language: str):
        (workspace / name).write_text("hello\n", encoding="utf-8")
        assert read(client, name).json()["language"] == language

    @pytest.mark.parametrize(
        "rel",
        [
            ".env",                       # secrets/config file by prefix
            ".env.local",                 # …any spelling of it
            "deploy/id_ed25519",          # a credential file
            "keys/server.pem",            # a private key by extension
            ".git/config",                # version-control internals
            ".ssh/config",                # a secret store
        ],
    )
    def test_denied_files_are_refused_and_name_the_rule(
        self, workspace: Path, client, rel: str
    ):
        """The same gate as the tree: the UI never asks for what the connectors
        refuse (decision D3), and the body says which rule fired."""
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("SECRET=never-shown\n", encoding="utf-8")
        response = read(client, rel)
        assert response.status_code == 422
        assert "connector path policy" in response.json()["detail"]
        assert "never-shown" not in response.text

    def test_connector_sources_use_the_same_read_flavour(
        self, workspace: Path, client
    ):
        """``mcp_connectors/`` is read-denied *for writes* only, so the editor
        reads its sources exactly as the tree already lists them — and a secret
        *name* inside that folder stays refused. Refusing the folder for reads
        too would make the read endpoint disagree with the tree, which decision
        D3 exists to prevent."""
        scripts = workspace / "mcp_connectors"
        scripts.mkdir()
        (scripts / "console_mcp.py").write_text("# the policy itself\n", encoding="utf-8")
        (scripts / "id_ed25519").write_text("priv\n", encoding="utf-8")
        assert read(client, "mcp_connectors/console_mcp.py").status_code == 200
        # …and it is a listed file, so the tree promised it.
        listed = names(tree(client, "mcp_connectors").json())
        assert "console_mcp.py" in listed
        denied = read(client, "mcp_connectors/id_ed25519")
        assert denied.status_code == 422
        assert "credential file" in denied.json()["detail"]

    @pytest.mark.parametrize("raw", ["../etc/hosts", "/etc/hosts", "~/x", "..", "a/../../b"])
    def test_escape_attempts_are_422(self, client, raw: str):
        assert read(client, raw).status_code == 422

    def test_symlink_to_outside_is_422(self, workspace: Path, client, tmp_path: Path):
        outside = tmp_path.parent / "hosts-target"
        outside.mkdir(exist_ok=True)
        (outside / "hosts").write_text("127.0.0.1\n", encoding="utf-8")
        (workspace / "hosts").symlink_to(outside / "hosts")
        response = read(client, "hosts")
        assert response.status_code == 422
        assert "outside the working location" in response.json()["detail"]

    def test_over_the_size_cap_is_413_naming_the_size(
        self, workspace: Path, client
    ):
        cap = editorfs.constants.MAX_EDITOR_BYTES
        (workspace / "big.py").write_bytes(b"x" * (cap + 1))
        response = read(client, "big.py")
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert str(cap + 1) in detail and str(cap) in detail

    def test_nul_in_the_head_is_binary_without_text(self, workspace: Path, client):
        (workspace / "blob.txt").write_bytes(b"abc\x00def\n")
        payload = read(client, "blob.txt").json()
        assert payload["binary"] is True
        assert payload["size"] == 8
        assert "text" not in payload
        assert payload["language"] == "plaintext"

    def test_invalid_utf8_is_binary(self, workspace: Path, client):
        (workspace / "latin.txt").write_bytes(b"caf\xe9\n")
        payload = read(client, "latin.txt").json()
        assert payload["binary"] is True
        assert "text" not in payload

    def test_crlf_survives_exactly(self, workspace: Path, client):
        (workspace / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
        payload = read(client, "crlf.py")
        assert payload.json()["text"] == "a = 1\r\nb = 2\r\n"
        assert payload.json()["size"] == 14

    @staticmethod
    def _read_past_the_cap(
        client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch,
        name: str, body: bytes, cap: int, stale_size: int,
    ) -> dict:
        """``size`` from the stat is under the cap, the bytes are not.

        The real cause of that state is a write racing the read (the model
        growing a file between the two), which is not a race a test should
        win — so the *first* ``os.stat`` of that file is answered with a
        stale size. Everything else goes through the unchanged policy.
        """
        monkeypatch.setattr(editorfs.constants, "MAX_EDITOR_BYTES", cap)
        (workspace / name).write_bytes(body)
        real_stat = os.stat
        state = {"stale": True}

        def first_stat(path, *args, **kwargs):
            info = real_stat(path, *args, **kwargs)
            if state["stale"] and str(path).endswith(name):
                state["stale"] = False
                return os.stat_result(
                    (info.st_mode, info.st_ino, info.st_dev, info.st_nlink,
                     info.st_uid, info.st_gid, stale_size,
                     info.st_atime, info.st_mtime, info.st_ctime)
                )
            return info

        monkeypatch.setattr(os, "stat", first_stat)
        response = read(client, name)
        assert response.status_code == 200
        return response.json()

    def test_a_read_past_the_cap_is_flagged_not_silent(
        self, workspace: Path, client, monkeypatch: pytest.MonkeyPatch
    ):
        """``truncated`` + the offset, the way ``tokensEstimated`` flags an
        estimate: a partial is never presented as a whole."""
        payload = self._read_past_the_cap(
            client, workspace, monkeypatch, "growing.py", b"hello world",
            cap=4, stale_size=4,
        )
        assert payload["truncated"] is True
        assert payload["truncatedAt"] == 4
        assert payload["text"] == "hell"
        assert payload["size"] == 11  # the size on disk, not of the slice

    def test_cap_cut_mid_character_is_not_called_binary(
        self, workspace: Path, client, monkeypatch: pytest.MonkeyPatch
    ):
        """Six bytes of two-byte characters in a three-byte slice: the cut
        lands inside a character, and that is truncation, not a binary file."""
        payload = self._read_past_the_cap(
            client, workspace, monkeypatch, "accent.py", "ééé".encode("utf-8"),
            cap=3, stale_size=3,
        )
        assert payload["binary"] is False
        assert payload["truncated"] is True
        assert payload["text"] == "é"

    def test_missing_is_404_directory_is_400(self, client):
        missing = read(client, "nope.py")
        assert missing.status_code == 404
        assert "does not exist" in missing.json()["detail"]
        directory = read(client, "src")
        assert directory.status_code == 400
        assert "not a file" in directory.json()["detail"]


class TestSurfaceShape:
    """Which verbs ``/api/fs`` answers, and what the package exposes.

    Read through the OpenAPI paths rather than ``app.routes``: the router is
    included lazily, so the flat path map is the one view that lists every
    endpoint the app actually answers.

    The read routes stay GET-only — that half of the surface has not moved.
    Mutating routes are an *allowlist*: the save, the create, the rename/move
    and the delete (a ``DELETE`` on ``/fs/file``, which trashes rather than
    unlinks) — all browser-only, and a new one has to appear here on purpose.
    No hub, tool or connector-facing route may ever add another; the
    browser-only half of that promise is asserted in
    ``tests/test_editorfs_write.py``.
    """

    def test_every_read_route_is_get_and_the_mutations_are_the_named_ones(self):
        paths = create_app().openapi()["paths"]
        fs = {path: sorted(ops) for path, ops in paths.items() if path.startswith("/api/fs")}
        assert fs == {
            "/api/fs/tree": ["get"],
            "/api/fs/file": ["delete", "get", "put"],
            "/api/fs/create": ["post"],
            "/api/fs/stat": ["get"],
            "/api/fs/status": ["get"],
            "/api/fs/diff": ["get"],
            "/api/fs/rename": ["post"],
        }

    def test_editorfs_exposes_the_guarded_write_and_nothing_beyond_it(self):
        """The names the package exports, and the ones it must not grow.

        ``write_file`` is the guarded save; the create/delete/rename halves live
        in ``editorfs.mutate`` and are re-exported through the package because
        the routes call them. What stays forbidden is anything *beyond* those
        four — a second spelling of a write, or a helper that skips the gate.
        """
        for name in ("write_file", "create_path", "trash_path", "rename_path"):
            assert hasattr(editorfs, name)
        for name in ("save_file", "delete_file", "move_path", "write_text", "unlink"):
            assert not hasattr(editorfs, name)
