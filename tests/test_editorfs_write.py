"""Tests for the Editor tab's guarded write surface: save, create, and gates.

Same hermetic shape as ``tests/test_editorfs.py``: the workspace is a
throw-away ``tmp_path`` pointed at by ``BENCHMARKIT_PATH_LOCATION``, caps and
the backup folder are patched through ``editorfs.constants``, and the
autouse fixtures in ``tests/conftest.py`` keep the hubs asleep. One test per
rule of the write path — the gate, the mtime guard, the chat-stream gate, the
size caps, the backup chain, the never-overwrite rule of a create — plus the
cross-cutting rows that only make sense stated once for all four routes, and
the promise that no tool can reach any of it (the wider form of which lives in
``tests/test_write_surface_isolation.py``).
"""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.core import editorfs


@pytest.fixture
def backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private backup folder, so a test never reads the app's real chain."""
    folder = tmp_path / "app-backups"
    monkeypatch.setattr(editorfs.constants, "BACKUP_DIR", folder)
    return folder


@pytest.fixture
def trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private trash folder, so a delete never touches the app's real one."""
    folder = tmp_path / "app-trash"
    monkeypatch.setattr(editorfs.constants, "TRASH_DIR", folder)
    return folder


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small tree, pointed at as the working location."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    (tmp_path / "src" / "core").mkdir(parents=True)
    (tmp_path / "src" / "config.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "core" / "job.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(workspace: Path):
    with TestClient(create_app()) as test_client:
        yield test_client


def save(client: TestClient, path: str, text: str,
         if_unmodified_since: Optional[float] = None):
    body: Dict[str, object] = {"path": path, "text": text}
    if if_unmodified_since is not None:
        body["ifUnmodifiedSince"] = if_unmodified_since
    return client.put("/api/fs/file", json=body)


def token(client: TestClient, path: str) -> float:
    """The mtime the read route handed out for ``path`` — the save guard's token."""
    response = client.get("/api/fs/file", params={"path": path})
    assert response.status_code == 200
    return response.json()["mtime"]


class TestSaveWrites:
    def test_save_writes_the_bytes_and_answers_new_size_and_mtime(
        self, workspace: Path, client, backups: Path
    ):
        payload = save(client, "src/config.py", "x = 42\n", token(client, "src/config.py"))
        assert payload.status_code == 200
        body = payload.json()
        assert body["path"] == "src/config.py"
        assert body["size"] == len("x = 42\n")
        assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == "x = 42\n"
        assert body["mtime"] == (workspace / "src" / "config.py").stat().st_mtime

    def test_a_second_save_rebaselines_on_the_answered_mtime(
        self, workspace: Path, client, backups: Path
    ):
        """The token round-trip: the payload's mtime is the next save's token."""
        first = save(client, "README.md", "one\n", token(client, "README.md"))
        assert first.status_code == 200
        second = save(client, "README.md", "two\n", first.json()["mtime"])
        assert second.status_code == 200
        assert (workspace / "README.md").read_text(encoding="utf-8") == "two\n"

    def test_crlf_round_trips_byte_for_byte(
        self, workspace: Path, client, backups: Path
    ):
        """No normalisation anywhere: what the editor holds is what lands."""
        (workspace / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
        before = (workspace / "crlf.py").read_bytes()
        body = client.get("/api/fs/file", params={"path": "crlf.py"}).json()
        assert body["text"] == "a = 1\r\nb = 2\r\n"
        response = save(client, "crlf.py", body["text"], body["mtime"])
        assert response.status_code == 200
        assert (workspace / "crlf.py").read_bytes() == before

    def test_a_save_keeps_the_file_mode(self, workspace: Path, client, backups: Path):
        """An atomic replace must not turn a 0755 script into 0600."""
        script = workspace / "run.sh"
        script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        script.chmod(0o755)
        response = save(client, "run.sh", "#!/bin/sh\necho bye\n", token(client, "run.sh"))
        assert response.status_code == 200
        assert script.stat().st_mode & 0o7777 == 0o755

    def test_no_temp_file_is_left_behind(
        self, workspace: Path, client, backups: Path
    ):
        save(client, "src/config.py", "x = 9\n", token(client, "src/config.py"))
        leftovers = [p.name for p in (workspace / "src").iterdir() if p.suffix == ".tmp"]
        assert leftovers == []


class TestMtimeGuard:
    def test_a_stale_token_is_409_carrying_both_numbers(
        self, workspace: Path, client, backups: Path
    ):
        """The whole point of the guard: disk moved, so nothing is overwritten."""
        stale = token(client, "src/config.py")
        (workspace / "src" / "config.py").write_text("changed by the model\n", encoding="utf-8")
        disk_mtime = (workspace / "src" / "config.py").stat().st_mtime

        response = save(client, "src/config.py", "my edit\n", stale)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["reason"] == "stale"
        assert detail["path"] == "src/config.py"
        assert detail["diskMtime"] == disk_mtime
        assert detail["bufferMtime"] == stale
        assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == (
            "changed by the model\n"
        )

    def test_the_epsilon_tolerates_a_coarse_filesystem(
        self, client, workspace: Path, backups: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A token a tick old is not a conflict — the guard compares with slack."""
        fresh = token(client, "src/config.py")
        monkeypatch.setattr(editorfs.constants, "MTIME_GUARD_EPSILON", 3600.0)
        (workspace / "src" / "config.py").write_text("moved\n", encoding="utf-8")
        assert save(client, "src/config.py", "x\n", fresh).status_code == 200

    def test_no_token_skips_the_guard_but_is_logged(
        self, client, workspace: Path, backups: Path, caplog: pytest.LogCaptureFixture
    ):
        """Absent means "no expectation", not "refuse" — and it names a client bug."""
        response = save(client, "src/config.py", "x = 7\n")
        assert response.status_code == 200
        assert "ifUnmodifiedSince" in caplog.text

    def test_a_save_after_a_save_does_not_look_stale(
        self, client, workspace: Path, backups: Path
    ):
        """The freshly re-stat'ed mtime is inside the epsilon of itself."""
        first = save(client, "README.md", "a\n", token(client, "README.md"))
        assert first.status_code == 200
        assert save(client, "README.md", "b\n", first.json()["mtime"]).status_code == 200


class TestWriteGate:
    @pytest.mark.parametrize(
        "raw", ["../etc/hosts", "/etc/hosts", "~/x", "..", "a/../../b", "src/../../etc"]
    )
    def test_escape_attempts_are_422_naming_the_rule(self, client, raw: str):
        response = save(client, raw, "x")
        assert response.status_code == 422
        assert response.json()["detail"]

    def test_nul_byte_is_refused(self, client):
        response = save(client, "a\0b", "x")
        assert response.status_code == 422
        assert "NUL" in response.json()["detail"]

    # (path, is it readable?) — the two flavours of one list, not two lists.
    # ``mcp_connectors/`` is the entry that differs: reading a connector's
    # source is legitimate debugging, rewriting the policy that constrains the
    # agent is not. Everything else here is a secret store and is denied both
    # ways, which the read suite already asserts.
    DENIED = [
        ("mcp_connectors/console_mcp.py", True),   # the connectors' own sources
        (".git/config", False),                    # version-control internals
        (".env", False),                           # the app's own secrets
        (".env.local", False),
        ("deploy/id_rsa", False),                  # a credential file
        ("keys/server.pem", False),                # a private key by extension
        (".ssh/config", False),                    # a secret store
    ]

    @pytest.mark.parametrize(("rel", "readable"), DENIED)
    def test_write_denied_paths_are_refused_naming_the_write_rule(
        self, workspace: Path, client, rel: str, readable: bool
    ):
        """Refused for writing, in the deny-list's own words, disk untouched —
        and the read answer for the same path does not move."""
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original\n", encoding="utf-8")

        response = save(client, rel, "overwritten\n")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "connector path policy" in detail
        assert "never write to it" in detail
        assert target.read_text(encoding="utf-8") == "original\n"

        read_response = client.get("/api/fs/file", params={"path": rel})
        if readable:
            assert read_response.status_code == 200
            assert read_response.json()["text"] == "original\n"
        else:
            assert read_response.status_code == 422
            assert "never read from it" in read_response.json()["detail"]

    def test_mcp_connectors_is_readable_but_never_writable(
        self, workspace: Path, client
    ):
        """The one pair that proves the two flavours are different lists."""
        scripts = workspace / "mcp_connectors"
        scripts.mkdir()
        (scripts / "console_mcp.py").write_text("# the policy itself\n", encoding="utf-8")
        assert client.get(
            "/api/fs/file", params={"path": "mcp_connectors/console_mcp.py"}
        ).status_code == 200
        response = save(client, "mcp_connectors/console_mcp.py", "# rewritten\n")
        assert response.status_code == 422
        assert "'mcp_connectors'" in response.json()["detail"]

    def test_a_denied_parent_refuses_an_innocent_leaf(
        self, workspace: Path, client
    ):
        """The ancestor chain: a folder whose *name* is a secret file."""
        hidden = workspace / ".env.local"
        hidden.mkdir()
        (hidden / "notes.txt").write_text("original\n", encoding="utf-8")
        response = save(client, ".env.local/notes.txt", "overwritten\n")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "secrets/config file" in detail
        assert (hidden / "notes.txt").read_text(encoding="utf-8") == "original\n"

    def test_a_symlink_inside_the_workspace_pointing_outside_is_422(
        self, workspace: Path, client, tmp_path: Path
    ):
        """The realpath lock, on the write side: the link is inside, its
        target is not, so the bytes that would be written are outside too."""
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir(exist_ok=True)
        (outside / "hosts").write_text("127.0.0.1\n", encoding="utf-8")
        (workspace / "hosts").symlink_to(outside / "hosts")
        response = save(client, "hosts", "10.0.0.1\n")
        assert response.status_code == 422
        assert "outside the working location" in response.json()["detail"]
        assert (outside / "hosts").read_text(encoding="utf-8") == "127.0.0.1\n"


class TestSizeCaps:
    def test_an_over_cap_payload_is_413_and_the_disk_is_untouched(
        self, workspace: Path, client, backups: Path
    ):
        cap = editorfs.constants.MAX_EDITOR_BYTES
        (workspace / "pad.py").write_bytes(b"x" * 10)
        before = (workspace / "pad.py").read_bytes()
        response = save(client, "pad.py", "y" * (cap + 1), token(client, "pad.py"))
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert str(cap + 1) in detail and str(cap) in detail
        assert (workspace / "pad.py").read_bytes() == before

    def test_an_over_cap_file_on_disk_is_413_even_with_a_fresh_token(
        self, workspace: Path, client, backups: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A buffer we never fully read must never be written back."""
        monkeypatch.setattr(editorfs.constants, "MAX_EDITOR_BYTES", 32)
        (workspace / "grown.py").write_bytes(b"z" * 100)
        response = save(client, "grown.py", "small\n")
        assert response.status_code == 413
        assert "100 bytes" in response.json()["detail"]
        assert (workspace / "grown.py").read_bytes() == b"z" * 100

    def test_a_multibuffer_payload_is_measured_in_bytes_not_characters(
        self, workspace: Path, client, backups: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(editorfs.constants, "MAX_EDITOR_BYTES", 4)
        (workspace / "accent.txt").write_text("a\n", encoding="utf-8")
        # "éé" is 2 characters but 4 UTF-8 bytes; "ééé" is 6 and is refused.
        assert save(client, "accent.txt", "éé").status_code == 200
        assert save(client, "accent.txt", "ééé").status_code == 413


class TestBackups:
    def test_a_save_backs_the_previous_bytes_up(
        self, workspace: Path, client, backups: Path
    ):
        assert save(client, "README.md", "new\n", token(client, "README.md")).status_code == 200
        copies = editorfs.files.list_backups(workspace / "README.md")
        assert len(copies) == 1
        assert Path(copies[0]).read_text(encoding="utf-8") == "# hi\n"

    def test_a_second_save_keeps_two_copies(
        self, workspace: Path, client, backups: Path
    ):
        resolved = workspace / "README.md"
        first = save(client, "README.md", "one\n", token(client, "README.md"))
        save(client, "README.md", "two\n", first.json()["mtime"])
        copies = editorfs.files.list_backups(resolved)
        assert len(copies) == 2
        assert sorted(Path(c).read_text(encoding="utf-8") for c in copies) == ["# hi\n", "one\n"]

    def test_the_keep_is_a_patchable_per_file_count(
        self, workspace: Path, client, backups: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(editorfs.constants, "BACKUP_KEEP", 2)
        resolved = workspace / "README.md"
        next_mtime = token(client, "README.md")
        for index in range(5):
            answer = save(client, "README.md", f"v{index}\n", next_mtime)
            next_mtime = answer.json()["mtime"]
        assert len(editorfs.files.list_backups(resolved)) == 2

    def test_a_backup_of_an_untouched_file_is_not_born_expired(
        self, workspace: Path, client, backups: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """``copy2`` inherits the source mtime; the sweep would eat the copy."""
        resolved = workspace / "README.md"
        os.utime(resolved, (0, 0))
        assert save(client, "README.md", "new\n").status_code == 200
        copies = editorfs.files.list_backups(resolved)
        assert copies
        assert Path(copies[0]).stat().st_mtime > 0

    def test_the_age_sweep_only_removes_old_copies(
        self, backups: Path, workspace: Path, client
    ):
        save(client, "README.md", "new\n")
        assert editorfs.files.prune_old_backups(older_than_days=30) == 0
        assert editorfs.files.prune_old_backups(older_than_days=0) == 0
        assert len(list(backups.glob("*.orig"))) == 1


class TestGateHelpers:
    """The two shapes a later mutation needs, checked where they are defined.

    A create targets a name that is not on disk, and a save/rename/delete
    targets one that must be — so the gate has two extra entry points, and the
    404 wording lives in one of them rather than in each future route.
    """

    def test_the_exists_shape_answers_404_for_an_absent_path(self, workspace: Path):
        normalised, resolved = editorfs.resolve_existing_in_workspace("README.md")
        assert (normalised, resolved) == ("README.md", workspace / "README.md")
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_existing_in_workspace("gone.md")
        assert caught.value.status_code == 404
        assert "does not exist" in caught.value.detail

    def test_the_exists_shape_uses_the_write_flavour(self, workspace: Path):
        (workspace / ".ssh").mkdir()
        (workspace / ".ssh" / "config").write_text("Host *\n", encoding="utf-8")
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_existing_in_workspace(".ssh/config")
        assert caught.value.status_code == 422
        assert "never write to it" in caught.value.detail

    def test_the_create_shape_resolves_the_parent_not_the_leaf(
        self, workspace: Path
    ):
        """A new name has no leaf to check, so its folder is the gate."""
        normalised, parent, name = editorfs.resolve_parent_in_workspace("src/new.py")
        assert (normalised, name) == ("src/new.py", "new.py")
        assert parent == workspace / "src"

        _, root_parent, root_name = editorfs.resolve_parent_in_workspace("top.txt")
        assert (root_name, root_parent) == ("top.txt", workspace)

    def test_the_create_shape_refuses_a_missing_or_occupied_parent(
        self, workspace: Path
    ):
        # 409, not 404: the name may be fine while the folder around it is not
        # there, and nothing is created on the way to that answer.
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_parent_in_workspace("nope/new.py")
        assert caught.value.status_code == 409
        assert "Parent folder does not exist" in caught.value.detail

        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_parent_in_workspace("README.md/new.py")
        assert caught.value.status_code == 400

        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_parent_in_workspace("")
        assert caught.value.status_code == 400

    def test_the_create_shape_gates_the_parent_with_the_write_flavour(
        self, workspace: Path
    ):
        (workspace / "mcp_connectors").mkdir()
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_parent_in_workspace("mcp_connectors/new.py")
        assert caught.value.status_code == 422
        assert "'mcp_connectors'" in caught.value.detail

    def test_the_chain_helper_reports_each_level(self, workspace: Path):
        chain = editorfs.resolve_write_target("src/core/job.py")
        assert [normalised for normalised, _ in chain] == [
            "src", "src/core", "src/core/job.py",
        ]

    def test_the_two_predicates_are_the_two_flavours_of_one_list(self):
        """``listable``/``writable`` answer the same question the gate asks, in
        boolean form — and they differ exactly where the lists do."""
        from pathlib import PurePosixPath

        scripts = PurePosixPath("/w/mcp_connectors/console_mcp.py")
        assert editorfs.listable(scripts) is True
        assert editorfs.writable(scripts) is False

        for name in (".git/config", ".ssh/id_rsa", ".env", "secrets/id_ed25519"):
            path = PurePosixPath(f"/w/{name}")
            assert editorfs.listable(path) is False, name
            assert editorfs.writable(path) is False, name

        ordinary = PurePosixPath("/w/src/config.py")
        assert editorfs.listable(ordinary) is True
        assert editorfs.writable(ordinary) is True


def create(client: TestClient, path: str, *, is_dir: bool = False):
    return client.post("/api/fs/create", json={"path": path, "dir": is_dir})


class TestCreate:
    def test_a_new_file_is_empty_and_answers_its_own_mtime(
        self, workspace: Path, client
    ):
        response = create(client, "src/new.py")
        assert response.status_code == 200
        body = response.json()
        assert body["path"] == "src/new.py"
        assert body["dir"] is False
        assert body["size"] == 0
        created = workspace / "src" / "new.py"
        assert created.is_file()
        assert created.read_bytes() == b""
        assert body["mtime"] == created.stat().st_mtime
        # The parent was not recreated on the way, and nothing else moved.
        assert sorted(p.name for p in (workspace / "src").iterdir()) == [
            "config.py", "core", "new.py",
        ]

    def test_a_new_folder_is_a_directory(self, workspace: Path, client):
        response = create(client, "src/new-dir", is_dir=True)
        assert response.status_code == 200
        assert response.json()["dir"] is True
        assert (workspace / "src" / "new-dir").is_dir()

    def test_a_name_at_the_workspace_root_needs_no_parent(
        self, workspace: Path, client
    ):
        assert create(client, "scratch.md").status_code == 200
        assert (workspace / "scratch.md").is_file()

    # The worst bug this route could have is truncating a file that was already
    # there, so every shape of "the name is taken" answers 409 and leaves the
    # bytes alone.
    def test_creating_over_a_file_is_409_and_changes_nothing(
        self, workspace: Path, client
    ):
        response = create(client, "README.md")
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["reason"] == "exists"
        assert detail["path"] == "README.md"
        assert detail["dir"] is False
        assert (workspace / "README.md").read_text(encoding="utf-8") == "# hi\n"

    def test_creating_over_a_folder_is_409_naming_it_as_a_folder(
        self, workspace: Path, client
    ):
        for is_dir in (True, False):
            response = create(client, "src", is_dir=is_dir)
            assert response.status_code == 409
            assert response.json()["detail"]["dir"] is True
        assert (workspace / "src").is_dir()

    def test_creating_over_a_dangling_symlink_is_409(
        self, workspace: Path, client
    ):
        """A symlink whose target is gone has no ``stat`` — the *name* is taken."""
        (workspace / "gone").symlink_to(workspace / "never-existed")
        response = create(client, "gone")
        assert response.status_code == 409
        assert response.json()["detail"]["reason"] == "exists"
        assert (workspace / "gone").is_symlink()
        assert not (workspace / "never-existed").exists()

    def test_a_missing_parent_is_409_naming_it_and_creates_no_chain(
        self, workspace: Path, client
    ):
        response = create(client, "nope/deeper/new.py")
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "Parent folder does not exist" in detail
        assert "nope" in detail
        assert not (workspace / "nope").exists()

    def test_a_parent_that_is_a_file_is_400(self, workspace: Path, client):
        response = create(client, "README.md/new.py")
        assert response.status_code == 400
        assert "not a folder" in response.json()["detail"]

    # The leaf name and its folder go through the same write-flavour list the
    # save goes through — one list, no second copy of it.
    DENIED_NAMES = [
        ("mcp_connectors/new.py", "the connectors' own sources"),
        (".git/config", "version-control internals"),
        (".env", "the app's own secrets"),
        (".env.local", "a secrets/config file"),
        ("deploy/id_rsa", "a credential file"),
        ("keys/server.pem", "a private key by extension"),
        ("sub/.ssh/y", "a secret store below an ordinary folder"),
    ]

    @pytest.mark.parametrize(("rel", "why"), DENIED_NAMES)
    def test_a_denied_name_is_422_naming_the_rule(
        self, workspace: Path, client, rel: str, why: str
    ):
        response = create(client, rel)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "connector path policy" in detail
        assert "never write to it" in detail
        assert not os.path.lexists(workspace / rel.split("/")[0])

    def test_a_denied_folder_name_refuses_a_whole_new_subtree(
        self, workspace: Path, client
    ):
        response = create(client, ".git", is_dir=True)
        assert response.status_code == 422
        assert not (workspace / ".git").exists()

    def test_a_leaf_carrying_a_slash_is_a_path_so_the_parent_rule_answers(
        self, client
    ):
        """The route takes a *path*, so a name with a `/` in it is a nested one.

        The client-side validator is what stops such a name from being sent at
        all; server-side there is nothing special left to say — the folder above
        it does not exist, and that is the refusal.
        """
        response = create(client, "a/b")
        assert response.status_code == 409
        assert "Parent folder does not exist" in response.json()["detail"]

    @pytest.mark.parametrize("raw", ["../escape.txt", "/etc/x", "~/x", "a/../../b"])
    def test_an_escape_attempt_is_422(self, client, raw: str):
        assert create(client, raw).status_code == 422

    def test_a_name_the_tree_hides_is_created_and_stays_hidden(
        self, workspace: Path, client
    ):
        """``IGNORED_DIR_NAMES`` is a read rule, not a write rule — both halves.

        Creating ``node_modules`` is legal, and the tree then does not show it.
        From the browser that looks like the create failed, so the pair is
        asserted together: the folder is on disk, the listing omits it.
        """
        assert create(client, "node_modules", is_dir=True).status_code == 200
        assert (workspace / "node_modules").is_dir()
        entries = client.get("/api/fs/tree").json()["entries"]
        assert "node_modules" not in [entry["name"] for entry in entries]

    def test_the_race_is_closed_by_the_open_flag_not_by_the_check(
        self, workspace: Path, client, monkeypatch: pytest.MonkeyPatch
    ):
        """A name that appears between the check and the open still loses.

        ``O_CREAT | O_EXCL`` (and ``mkdir``) fail on an existing name, so the
        pre-check is only an early answer — the guard is the syscall. Forcing the
        check to report the name as free is how a test reaches that path.
        """
        (workspace / "src" / "racer.py").write_text("do not touch me\n", encoding="utf-8")
        monkeypatch.setattr(os.path, "lexists", lambda _path: False)
        response = create(client, "src/racer.py")
        assert response.status_code == 409
        assert response.json()["detail"]["reason"] == "exists"
        assert (workspace / "src" / "racer.py").read_text(encoding="utf-8") == (
            "do not touch me\n"
        )

    def test_an_empty_name_is_400_and_a_missing_field_is_422(self, client):
        assert create(client, "").status_code == 400
        assert client.post("/api/fs/create", json={"dir": True}).status_code == 422

    def test_the_created_file_is_readable_and_savable_from_there(
        self, workspace: Path, client, backups: Path
    ):
        """The whole loop: the create's ``mtime`` is the save's next token."""
        created = create(client, "src/loop.py").json()
        assert client.get("/api/fs/file", params={"path": "src/loop.py"}).json()[
            "text"
        ] == ""
        response = save(client, "src/loop.py", "print(1)\n", created["mtime"])
        assert response.status_code == 200
        assert (workspace / "src" / "loop.py").read_text(encoding="utf-8") == "print(1)\n"


def rename(client: TestClient, frm: str, to: str):
    return client.post("/api/fs/rename", json={"from": frm, "to": to})


class TestRename:
    """``POST /api/fs/rename``: one path, two of them, one gate on each end."""

    def test_renaming_a_file_in_place_moves_the_name_and_keeps_the_bytes(
        self, workspace: Path, client, backups: Path
    ):
        before = (workspace / "src" / "config.py").stat()
        response = rename(client, "src/config.py", "src/settings.py")
        assert response.status_code == 200
        body = response.json()
        assert body["from"] == "src/config.py"
        assert body["to"] == "src/settings.py"
        assert body["dir"] is False
        assert body["size"] == before.st_size
        # A rename does not touch content, so the mtime is the file's own: an
        # open buffer of it stays fresh and its save token stays valid.
        assert body["mtime"] == before.st_mtime

        moved = workspace / "src" / "settings.py"
        assert (workspace / "src" / "config.py").exists() is False
        assert moved.read_text(encoding="utf-8") == "x = 1\n"
        assert moved.stat().st_mtime == before.st_mtime

    def test_moving_a_file_to_another_existing_folder(self, workspace: Path, client, backups: Path):
        """Same route, different parent — a move is a rename with a new folder."""
        response = rename(client, "src/config.py", "src/core/config.py")
        assert response.status_code == 200
        assert (workspace / "src" / "core" / "config.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (workspace / "src" / "config.py").exists() is False

    def test_renaming_a_folder_moves_the_whole_tree(
        self, workspace: Path, client, backups: Path
    ):
        response = rename(client, "src", "source")
        assert response.status_code == 200
        assert response.json()["dir"] is True
        assert (workspace / "src").exists() is False
        assert (workspace / "source" / "config.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (workspace / "source" / "core" / "job.py").read_text(encoding="utf-8") == "y = 2\n"

    def test_a_renamed_folder_stays_readable_through_its_new_path(
        self, workspace: Path, client, backups: Path
    ):
        """The read routes answer about the moved file with no help from here."""
        rename(client, "src", "source")
        body = client.get("/api/fs/file", params={"path": "source/config.py"}).json()
        assert body["text"] == "x = 1\n"
        assert client.get("/api/fs/tree", params={"path": "source"}).json()["path"] == "source"

    # The both-ends rule, from the direction that a leaf-only gate would miss:
    # an innocent source moving into a protected place.
    @pytest.mark.parametrize(
        "to",
        [
            ".git/hooks/pre-commit",
            "mcp_connectors/policy_patch.py",
            ".env",
            "keys/id_ed25519",
            "deploy/server.pem",
            "sub/.ssh/config",
        ],
    )
    def test_a_denied_target_is_422_and_nothing_moved(
        self, workspace: Path, client, to: str
    ):
        response = rename(client, "src/config.py", to)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "connector path policy" in detail
        assert "never write to it" in detail
        assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"
        assert not os.path.lexists(workspace / to.split("/")[0])

    def test_a_denied_target_folder_refuses_a_whole_tree_move(
        self, workspace: Path, client
    ):
        """The source is a folder, so every moved path would be denied too."""
        response = rename(client, "src", "mcp_connectors/src")
        assert response.status_code == 422
        assert "'mcp_connectors'" in response.json()["detail"]
        assert (workspace / "src" / "config.py").is_file()

    def test_an_innocent_name_under_a_denied_folder_is_refused(
        self, workspace: Path, client
    ):
        """The leaf name is fine; its *parent* is the protected one."""
        (workspace / ".env.local").mkdir()
        response = rename(client, "README.md", ".env.local/notes.txt")
        assert response.status_code == 422
        assert "secrets/config file" in response.json()["detail"]
        assert (workspace / "README.md").is_file()

    def test_a_denied_source_is_refused_by_its_own_end(self, workspace: Path, client):
        """The other direction: ``to`` is innocent, ``from`` is not."""
        (workspace / ".git").mkdir()
        (workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        response = rename(client, ".git/config", "config.txt")
        assert response.status_code == 422
        assert "never write to it" in response.json()["detail"]
        assert (workspace / ".git" / "config").is_file()
        assert not (workspace / "config.txt").exists()

    # An occupied target is where ``os.rename`` would silently replace a file.
    def test_an_occupied_target_file_is_409_and_both_paths_survive(
        self, workspace: Path, client
    ):
        response = rename(client, "README.md", "src/config.py")
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["reason"] == "exists"
        assert detail["path"] == "src/config.py"
        assert detail["dir"] is False
        assert (workspace / "README.md").read_text(encoding="utf-8") == "# hi\n"
        assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_an_occupied_target_folder_is_409_naming_it(self, workspace: Path, client):
        """``os.rename`` moves a directory *into* an empty one; here neither moves."""
        (workspace / "empty").mkdir()
        response = rename(client, "src", "empty")
        assert response.status_code == 409
        assert response.json()["detail"]["dir"] is True
        assert (workspace / "src" / "config.py").is_file()
        assert (workspace / "empty").is_dir()

    def test_an_occupied_target_dangling_symlink_is_409(self, workspace: Path, client):
        (workspace / "gone").symlink_to(workspace / "never-existed")
        response = rename(client, "README.md", "gone")
        assert response.status_code == 409
        assert response.json()["detail"]["reason"] == "exists"
        assert (workspace / "gone").is_symlink()
        assert (workspace / "README.md").is_file()

    def test_a_missing_target_parent_is_409_naming_it(self, workspace: Path, client):
        """No implicit ``mkdir``: a typo in a typed path is a refusal, not a tree."""
        response = rename(client, "src", "src/deeper/x")
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "Parent folder does not exist" in detail
        assert "src/deeper" in detail
        assert (workspace / "src" / "config.py").is_file()
        assert not (workspace / "src" / "deeper").exists()

    def test_a_target_parent_that_is_a_file_is_400(self, workspace: Path, client):
        response = rename(client, "README.md", "src/config.py/x")
        assert response.status_code == 400

    def test_from_equal_to_is_400(self, workspace: Path, client):
        for raw in ("src/config.py", "./src/config.py", "src/./config.py"):
            response = rename(client, raw, "src/config.py")
            assert response.status_code == 400
            assert "same path" in response.json()["detail"]
        assert (workspace / "src" / "config.py").is_file()

    def test_a_case_only_rename_succeeds(self, workspace: Path, client):
        """Not the same normalised path here, so it is a real rename."""
        assert rename(client, "README.md", "README.MD").status_code == 200
        assert (workspace / "README.MD").read_text(encoding="utf-8") == "# hi\n"

    def test_a_folder_cannot_move_into_itself(self, workspace: Path, client):
        response = rename(client, "src", "src/core/src")
        assert response.status_code == 409
        assert "inside itself" in response.json()["detail"]
        assert (workspace / "src" / "core" / "job.py").is_file()

    def test_moving_a_folder_next_to_itself_by_a_similar_name_is_legal(
        self, workspace: Path, client
    ):
        """The prefix rule needs the trailing ``/``: ``src`` → ``src2`` is a
        sibling, not a descendant, and so is ``src/sub`` → ``src/sub2``."""
        (workspace / "src" / "sub").mkdir()
        (workspace / "src" / "sub" / "deep.py").write_text("deep\n", encoding="utf-8")

        assert rename(client, "src", "src2").status_code == 200
        assert (workspace / "src2" / "sub" / "deep.py").is_file()

        assert rename(client, "src2/sub", "src2/sub2").status_code == 200
        assert (workspace / "src2" / "sub2" / "deep.py").read_text(encoding="utf-8") == "deep\n"

    def test_the_working_location_cannot_be_renamed(self, workspace: Path, client):
        for raw in ("", "."):
            response = rename(client, raw, "nope")
            assert response.status_code == 400
            assert "working location itself cannot be renamed" in response.json()["detail"]
        assert (workspace / "README.md").is_file()

    def test_renaming_a_symlink_is_refused_and_the_link_is_untouched(
        self, workspace: Path, client
    ):
        (workspace / "target.py").write_text("the real file\n", encoding="utf-8")
        (workspace / "shortcut.py").symlink_to(workspace / "target.py")

        response = rename(client, "shortcut.py", "shortcut2.py")
        assert response.status_code == 400
        assert "symbolic link" in response.json()["detail"]
        assert (workspace / "shortcut.py").is_symlink()
        assert not (workspace / "shortcut2.py").exists()

    def test_a_link_pointing_outside_is_refused_by_the_gate(
        self, workspace: Path, client, tmp_path: Path
    ):
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir(exist_ok=True)
        (outside / "hosts").write_text("127.0.0.1\n", encoding="utf-8")
        (workspace / "hosts").symlink_to(outside / "hosts")

        response = rename(client, "hosts", "hosts.local")
        assert response.status_code == 422
        assert "outside the working location" in response.json()["detail"]
        assert (workspace / "hosts").is_symlink()
        assert (outside / "hosts").read_text(encoding="utf-8") == "127.0.0.1\n"

    def test_a_missing_source_is_404(self, workspace: Path, client):
        response = rename(client, "nope.py", "other.py")
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]

    @pytest.mark.parametrize(
        "raw", ["../etc/hosts", "/etc/hosts", "~/x", "src/../../etc"]
    )
    def test_an_escape_attempt_on_either_end_is_422(self, client, raw: str):
        assert rename(client, raw, "x").status_code == 422
        assert rename(client, "README.md", raw).status_code == 422

    def test_a_nul_in_either_path_is_422(self, client):
        assert "NUL" in rename(client, "a\0b", "x").json()["detail"]
        assert "NUL" in rename(client, "README.md", "x\0y").json()["detail"]

    def test_a_cross_device_move_is_refused_rather_than_copied(
        self, workspace: Path, client, monkeypatch: pytest.MonkeyPatch
    ):
        """No copy+delete fallback: it is not atomic and it loses the mode."""
        def cross_device(_frm: str, _to: str) -> None:
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr(os, "rename", cross_device)
        response = rename(client, "src/config.py", "config.py")
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "different filesystems" in detail
        assert (workspace / "src" / "config.py").is_file()
        assert not (workspace / "config.py").exists()

    def test_a_rename_writes_no_backup(self, workspace: Path, client, backups: Path):
        """Deliberate, not an oversight: the content did not change, so a copy
        of it buys nothing — the trash and git status already describe the move."""
        rename(client, "src/config.py", "src/settings.py")
        if backups.exists():
            assert list(backups.glob("*.orig")) == []
        assert editorfs.files.list_backups(workspace / "src" / "settings.py") == []
        # The old name leaves no chain either: a rename has nothing to keep.
        assert editorfs.files.list_backups(workspace / "src" / "config.py") == []

    def test_a_renamed_file_is_savable_at_its_new_path_with_its_old_token(
        self, workspace: Path, client, backups: Path
    ):
        """The round trip the acceptance names: the token, the path and the
        buffer all still agree after a move."""
        token_value = token(client, "src/config.py")
        assert rename(client, "src/config.py", "src/settings.py").status_code == 200
        response = save(client, "src/settings.py", "x = 42\n", token_value)
        assert response.status_code == 200
        assert (workspace / "src" / "settings.py").read_text(encoding="utf-8") == "x = 42\n"

    def test_malformed_bodies_are_422_not_500(self, client):
        assert client.post("/api/fs/rename", json={"to": "x"}).status_code == 422
        assert client.post("/api/fs/rename", json={"from": "README.md"}).status_code == 422


def trash_path(client: TestClient, path: str):
    return client.request("DELETE", "/api/fs/file", params={"path": path})


class TestDeleteToTrash:
    """``DELETE /api/fs/file``: a move, never an ``unlink``."""

    def test_deleting_a_file_moves_it_and_names_the_batch(
        self, workspace: Path, client, trash: Path
    ):
        response = trash_path(client, "src/config.py")
        assert response.status_code == 200
        body = response.json()
        assert body["path"] == "src/config.py"
        assert body["dir"] is False
        assert body["entries"] == 0

        assert not (workspace / "src" / "config.py").exists()
        batch = trash / Path(body["trashedTo"]).name
        trashed = batch / "src" / "config.py"
        assert trashed.read_text(encoding="utf-8") == "x = 1\n"
        # The batch is named where the confirmation can quote it, and the digest
        # is keyed on the resolved path (two mount spellings, one directory).
        assert batch.name.startswith(editorfs.mutate.trash_digest(workspace / "src" / "config.py"))

    def test_deleting_a_folder_keeps_the_layout_under_the_batch(
        self, workspace: Path, client, trash: Path
    ):
        response = trash_path(client, "src")
        assert response.status_code == 200
        body = response.json()
        assert body["dir"] is True
        # config.py, core/, core/job.py — the server's own walk, which is the
        # number the confirmation dialog repeated.
        assert body["entries"] == 3

        assert not (workspace / "src").exists()
        batch = trash / Path(body["trashedTo"]).name
        assert (batch / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (batch / "src" / "core" / "job.py").read_text(encoding="utf-8") == "y = 2\n"

    def test_a_folder_holding_a_denied_path_is_refused_and_survives_whole(
        self, workspace: Path, client, trash: Path
    ):
        """The rule the whole subtree check exists for: the *innocent* name is
        not what decides — every descendant is checked first, all-or-nothing."""
        (workspace / "src" / ".env").write_text("SECRET=1\n", encoding="utf-8")
        response = trash_path(client, "src")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "never write to it" in detail
        # The refusal names the offending descendant, not the innocent parent.
        assert "inside src" in detail
        assert (workspace / "src" / "config.py").is_file()
        assert (workspace / "src" / "core" / "job.py").is_file()
        assert (workspace / "src" / ".env").read_text(encoding="utf-8") == "SECRET=1\n"
        # Nothing at all was moved.
        assert not trash.exists()

    def test_a_folder_holding_the_connectors_sources_is_refused(
        self, workspace: Path, client, trash: Path
    ):
        scripts = workspace / "project" / "mcp_connectors"
        scripts.mkdir(parents=True)
        (scripts / "console_mcp.py").write_text("# policy\n", encoding="utf-8")
        response = trash_path(client, "project")
        assert response.status_code == 422
        assert "'mcp_connectors'" in response.json()["detail"]
        assert (scripts / "console_mcp.py").is_file()

    def test_a_denied_path_deleted_directly_is_422(self, workspace: Path, client, trash: Path):
        (workspace / ".git").mkdir()
        (workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        response = trash_path(client, ".git/config")
        assert response.status_code == 422
        assert "never write to it" in response.json()["detail"]
        assert (workspace / ".git" / "config").is_file()
        assert not trash.exists()

    def test_the_working_location_itself_cannot_be_deleted(
        self, workspace: Path, client, trash: Path
    ):
        """``""`` and ``.`` both name the root, and the root is never moved."""
        for raw in ("", "."):
            response = trash_path(client, raw)
            assert response.status_code == 400
            assert "working location itself cannot be deleted" in response.json()["detail"]
        assert (workspace / "README.md").is_file()
        assert not trash.exists()

    def test_deleting_a_symlink_is_refused_and_the_link_is_untouched(
        self, workspace: Path, client, trash: Path
    ):
        """A link inside the workspace, so the realpath lock lets it through —
        and it is still refused: unlinking a link is a human decision, and
        moving one into the trash would silently break it."""
        (workspace / "link-target.py").write_text("the real file\n", encoding="utf-8")
        (workspace / "shortcut.py").symlink_to(workspace / "link-target.py")

        response = trash_path(client, "shortcut.py")
        assert response.status_code == 400
        assert "symbolic link" in response.json()["detail"]
        assert (workspace / "shortcut.py").is_symlink()
        assert (workspace / "link-target.py").read_text(encoding="utf-8") == "the real file\n"
        assert not trash.exists()

    def test_a_link_pointing_outside_is_refused_by_the_gate(
        self, workspace: Path, client, trash: Path, tmp_path: Path
    ):
        """The same shape as the save's: the gate sees the *target*, outside."""
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir(exist_ok=True)
        (outside / "hosts").write_text("127.0.0.1\n", encoding="utf-8")
        (workspace / "hosts").symlink_to(outside / "hosts")

        response = trash_path(client, "hosts")
        assert response.status_code == 422
        assert "outside the working location" in response.json()["detail"]
        assert (workspace / "hosts").is_symlink()
        assert (outside / "hosts").read_text(encoding="utf-8") == "127.0.0.1\n"
        assert not trash.exists()

    def test_a_folder_holding_a_link_to_a_denied_target_is_refused(
        self, workspace: Path, client, trash: Path
    ):
        """The subtree walk does not *follow* links, but it does judge them:
        a link to ``~/.ssh`` inside an innocent folder is a route to a secret."""
        (workspace / "deploy").mkdir()
        (workspace / "deploy" / "key").symlink_to(workspace / ".ssh" / "id_rsa")
        response = trash_path(client, "deploy")
        assert response.status_code == 422
        assert "never write to it" in response.json()["detail"]
        assert (workspace / "deploy" / "key").is_symlink()
        assert not trash.exists()

    @pytest.mark.parametrize(
        "raw", ["../etc/hosts", "/etc/hosts", "~/x", "src/../../etc"]
    )
    def test_an_escape_attempt_is_422(self, client, raw: str):
        assert trash_path(client, raw).status_code == 422

    def test_a_nul_in_the_path_is_422(self, client):
        response = trash_path(client, "a\0b")
        assert response.status_code == 422
        assert "NUL" in response.json()["detail"]

    def test_an_absent_path_is_404_not_a_quiet_success(
        self, workspace: Path, client, trash: Path
    ):
        """The caller's view of the tree is wrong, and it has to be told."""
        response = trash_path(client, "nope.py")
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]
        assert not trash.exists()

    def test_a_second_delete_of_the_same_path_is_404(
        self, workspace: Path, client, trash: Path
    ):
        assert trash_path(client, "README.md").status_code == 200
        response = trash_path(client, "README.md")
        assert response.status_code == 404
        assert len(list(trash.iterdir())) == 1

    def test_two_deletes_make_two_batches(self, workspace: Path, client, trash: Path):
        first = trash_path(client, "README.md").json()["trashedTo"]
        second = trash_path(client, "src/config.py").json()["trashedTo"]
        assert first != second
        assert len(list(trash.iterdir())) == 2

    def test_same_named_files_from_different_parents_keep_their_own_batch(
        self, workspace: Path, client, trash: Path
    ):
        """The relative layout inside each batch is what prevents a collision:
        ``a/f.py`` and ``b/f.py`` are never one file."""
        (workspace / "a").mkdir()
        (workspace / "b").mkdir()
        (workspace / "a" / "f.py").write_text("from a\n", encoding="utf-8")
        (workspace / "b" / "f.py").write_text("from b\n", encoding="utf-8")

        first = trash_path(client, "a/f.py").json()["trashedTo"]
        second = trash_path(client, "b/f.py").json()["trashedTo"]
        assert (trash / Path(first).name / "a" / "f.py").read_text(encoding="utf-8") == "from a\n"
        assert (trash / Path(second).name / "b" / "f.py").read_text(encoding="utf-8") == "from b\n"

    def test_the_batch_is_unique_even_when_the_stamp_collides(
        self, workspace: Path, client, trash: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A fixed clock stands in for the rare same-tick pair of deletes; a
        second batch must not be created *inside* the first."""
        class Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 1, 1, 12, 0, 0, 0)

        monkeypatch.setattr("src.core.editorfs.mutate.datetime", Frozen)
        first = trash_path(client, "README.md").json()["trashedTo"]
        second = trash_path(client, "src/config.py").json()["trashedTo"]
        assert first != second
        assert len(list(trash.iterdir())) == 2

    def test_retention_sweeps_an_old_batch_and_keeps_a_young_one(
        self, workspace: Path, client, trash: Path
    ):
        old = trash / Path(trash_path(client, "README.md").json()["trashedTo"]).name
        young = trash / Path(trash_path(client, "src/config.py").json()["trashedTo"]).name
        os.utime(old, (0, 0))

        swept = editorfs.sweep_trash(older_than_days=30)
        assert swept == 1
        assert not old.exists()
        assert young.is_dir()
        # The sweep never reaches outside its own directory: the workspace is
        # where it started, and the young copy is still recoverable.
        assert (young / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (workspace / "src" / "config.py").exists() is False

    def test_a_retention_window_of_zero_disables_the_sweep(
        self, workspace: Path, client, trash: Path
    ):
        assert trash_path(client, "README.md").status_code == 200
        os.utime(next(iter(trash.iterdir())), (0, 0))
        assert editorfs.sweep_trash(older_than_days=0) == 0
        assert len(list(trash.iterdir())) == 1

    def test_a_delete_never_touches_the_read_surface(
        self, workspace: Path, client, trash: Path
    ):
        """The read routes keep answering as before, for the survivors."""
        assert trash_path(client, "README.md").status_code == 200
        assert client.get("/api/fs/file", params={"path": "src/config.py"}).json()[
            "text"
        ] == "x = 1\n"
        assert client.get("/api/fs/tree").json()["path"] == ""


class TestChatStreamGate:
    def test_a_save_while_a_reply_streams_is_409(
        self, client, workspace: Path, backups: Path
    ):
        """The reply may be editing these files through its tools right now."""
        cancel_event = threading.Event()
        routes._arm_chat_handle(cancel_event, threading.Event())
        try:
            response = save(client, "src/config.py", "racing edit\n")
            assert response.status_code == 409
            assert "while a chat reply is streaming" in response.json()["detail"]
        finally:
            routes._clear_chat_handle(cancel_event)
        assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_the_same_slot_the_pause_endpoint_uses(self, client):
        """Not a second flag: one in-flight reply, one answer."""
        cancel_event = threading.Event()
        pause_event = threading.Event()
        routes._arm_chat_handle(cancel_event, pause_event)
        try:
            assert client.post("/api/chat/pause").status_code == 200
        finally:
            routes._clear_chat_handle(cancel_event)
        assert save(client, "src/config.py", "x\n").status_code == 200

    def test_a_create_while_a_reply_streams_is_409(
        self, client, workspace: Path
    ):
        """The reply may be writing this very folder through its tools."""
        cancel_event = threading.Event()
        routes._arm_chat_handle(cancel_event, threading.Event())
        try:
            for is_dir in (False, True):
                response = create(client, "src/new-thing", is_dir=is_dir)
                assert response.status_code == 409
                assert "while a chat reply is streaming" in response.json()["detail"]
        finally:
            routes._clear_chat_handle(cancel_event)
        assert not (workspace / "src" / "new-thing").exists()

    def test_a_rename_while_a_reply_streams_is_409(
        self, client, workspace: Path, backups: Path
    ):
        """The reply may be editing the source *or* the folder it lands in."""
        cancel_event = threading.Event()
        routes._arm_chat_handle(cancel_event, threading.Event())
        try:
            for frm, to in (("README.md", "notes.md"), ("src", "source")):
                response = rename(client, frm, to)
                assert response.status_code == 409
                assert "while a chat reply is streaming" in response.json()["detail"]
        finally:
            routes._clear_chat_handle(cancel_event)
        assert (workspace / "README.md").is_file()
        assert (workspace / "src" / "config.py").is_file()
        assert not (workspace / "notes.md").exists()
        assert not (workspace / "source").exists()
        assert not backups.exists()

    def test_a_delete_while_a_reply_streams_is_409(
        self, client, workspace: Path, trash: Path
    ):
        """The reply may be reading the very file being removed."""
        cancel_event = threading.Event()
        routes._arm_chat_handle(cancel_event, threading.Event())
        try:
            for path in ("README.md", "src"):
                response = trash_path(client, path)
                assert response.status_code == 409
                assert "while a chat reply is streaming" in response.json()["detail"]
        finally:
            routes._clear_chat_handle(cancel_event)
        assert (workspace / "README.md").is_file()
        assert (workspace / "src" / "config.py").is_file()
        assert not trash.exists()


class TestMissingAndShape:
    def test_saving_a_missing_path_is_404(self, client):
        response = save(client, "nope.py", "x\n")
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]

    def test_saving_a_directory_is_400(self, client):
        response = save(client, "src", "x\n")
        assert response.status_code == 400
        assert "not a file" in response.json()["detail"]

    def test_a_body_without_text_is_a_422_from_fastapi(self, client):
        """Malformed in, 422 out — never a 500 out of the core."""
        assert client.put("/api/fs/file", json={"nope": 1}).status_code == 422

    def test_a_missing_path_field_is_422(self, client):
        assert client.put("/api/fs/file", json={"text": "x"}).status_code == 422


class TestWritesStayBrowserOnly:
    """The promise AGENTS.md makes about the tools, asserted rather than trusted.

    The wider "no tool can reach a write route" suite — the connectors' live
    tool listings, the LSP catalogue, the marker scan across the source tree,
    and the terminal's boundary — is ``tests/test_write_surface_isolation.py``.
    What lives here is the half that belongs to the write surface itself: who
    is allowed to *call* the core, and that core cannot reach the HTTP layer.
    """

    def test_the_write_helpers_are_called_from_the_routes_and_nowhere_else(self):
        """Static guard: the only caller of the write core is the ``/api/fs`` router.

        The write core is the whole mutating half — ``write_file`` for a buffer,
        ``create_path`` for a new name, ``rename_path`` for a move,
        ``trash_path`` for a removal — so a scan of ``src`` for those names
        answers "who can write" without trusting the prose. The package's own
        files are skipped (they *define* them).
        """
        source_root = Path(editorfs.__file__).resolve().parents[2]  # ``src``
        writers = (
            "write_file",
            "backup_for_write",
            "create_path",
            "rename_path",
            "trash_path",
        )
        callers: List[str] = []
        for path in source_root.rglob("*.py"):
            if path.parent.name == "editorfs":
                continue
            text = path.read_text(encoding="utf-8")
            if any(name in text for name in writers):
                callers.append(str(path.relative_to(source_root)))
        assert callers == [str(Path("api") / "routes.py")]

    def test_no_core_module_imports_the_http_layer(self):
        """Core owns logic, the API owns transport — and that is the wall.

        The hubs and ``core/chat.py`` run inside the same process, so reaching
        a route would be an import away, not a network call. No module under
        ``src/core`` imports ``src.api``, so no tool-execution path can call a
        write route either.
        """
        source_root = Path(editorfs.__file__).resolve().parents[2]  # ``src``
        offenders = []
        for path in (source_root / "core").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "from src.api" in text or "import src.api" in text:
                offenders.append(str(path.relative_to(source_root)))
        assert offenders == []


# ---------------------------------------------------------------------------
# The cross-cutting matrix.
#
# Each route has its own suite above. The properties below are stated once for
# the whole write family, so a fifth write route added later has a table to
# slot into — a rule only one route asserts is a rule the next one forgets.
# ---------------------------------------------------------------------------

ESCAPES = ["../etc/hosts", "/etc/hosts", "~/x", "a/../../b", "src/../../etc", "a\0b"]

# (path, a fragment of the deny-list's own sentence for it)
DENIED = [
    (".env", "secrets/config file"),
    (".git/config", "inside '.git'"),
    ("deploy/id_rsa", "credential file"),
    ("keys/server.pem", "private key"),
    ("mcp_connectors/new_entry.py", "inside 'mcp_connectors'"),
]


def call_write(client: TestClient, which: str, path: str, extra: Dict[str, Any] = None):
    """One mutation of ``path`` through the route named by ``which``.

    The matrix's single point: a property asserted over every route is written
    once here, so adding a route means adding a branch and nothing else.
    """
    if which == "save":
        return save(client, path, "matrix\n", extra["token"] if extra else None)
    if which == "create":
        return create(client, path)
    if which == "rename-src":
        return rename(client, path, "innocent-target")
    if which == "rename-dst":
        return rename(client, "README.md", path)
    if which == "delete":
        return trash_path(client, path)
    raise AssertionError(f"unknown write kind {which!r}")  # a matrix hole, not a typo


WRITE_KINDS = ["save", "create", "rename-src", "rename-dst", "delete"]


class TestWriteMatrix:
    """Every route, the same refusals, in the same words."""

    @pytest.mark.parametrize("kind", WRITE_KINDS)
    @pytest.mark.parametrize("raw", ESCAPES)
    def test_an_escaped_path_is_422_on_every_route(self, client, kind: str, raw: str):
        assert call_write(client, kind, raw).status_code == 422

    @pytest.mark.parametrize("kind", WRITE_KINDS)
    @pytest.mark.parametrize(("rel", "phrase"), DENIED)
    def test_the_write_flavour_deny_list_refuses_on_every_route(
        self, workspace: Path, client, kind: str, rel: str, phrase: str
    ):
        """The same list, the same sentence, whichever route was asked.

        ``mcp_connectors`` is the entry the two flavours disagree on, and the
        refusal must still be the write flavour's ("write to") here: every one
        of these calls is a write whatever the read answer for the path is.
        """
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("original\n", encoding="utf-8")

        response = call_write(client, kind, rel)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "connector path policy" in detail
        assert "never write to it" in detail
        assert phrase in detail, (kind, rel, detail)
        assert target.read_text(encoding="utf-8") == "original\n"

    @pytest.mark.parametrize("kind", ["create", "rename", "delete"])
    def test_no_route_silently_replaces_or_reduces_what_is_already_there(
        self, workspace: Path, client, trash: Path, backups: Path, kind: str
    ):
        """The property the three path routes share: nothing is lost on the way.

        ``os.mkdir``/``O_EXCL``, ``os.rename`` and ``shutil.move`` would each
        have accepted one of these calls with data loss, so for create and
        rename the row asserts the **refusal** (409, and the occupied bytes
        still there), and for delete it asserts the **consequence**: the moved
        copy is byte-for-byte what was on disk and its neighbour is untouched.
        """
        if kind == "create":
            response = create(client, "src/config.py")
            assert response.status_code == 409
            assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == (
                "x = 1\n"
            )
        elif kind == "rename":
            response = rename(client, "README.md", "src/config.py")
            assert response.status_code == 409
            assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == (
                "x = 1\n"
            )
            assert (workspace / "README.md").read_text(encoding="utf-8") == "# hi\n"
        else:
            response = trash_path(client, "src/config.py")
            assert response.status_code == 200
            batch = trash / Path(response.json()["trashedTo"]).name
            assert (batch / "src" / "config.py").read_bytes() == b"x = 1\n"
            # Everything that was *not* asked about is exactly where it was.
            assert (workspace / "src" / "core" / "job.py").read_text(
                encoding="utf-8"
            ) == "y = 2\n"
            assert (workspace / "README.md").read_text(encoding="utf-8") == "# hi\n"

    @pytest.mark.parametrize("kind", ["save", "create", "rename"])
    def test_each_route_answers_what_a_restat_finds(
        self, workspace: Path, client, trash: Path, backups: Path, kind: str
    ):
        """The payload's mtime/size is the disk's, not a number the core made up.

        The client re-baselines on these, so a payload that drifted from the
        disk would look like a conflict on the caller's next save.
        """
        if kind == "save":
            answer = save(client, "README.md", "new body\n", token(client, "README.md"))
            path = "README.md"
        elif kind == "create":
            answer = create(client, "src/made.py")
            path = "src/made.py"
        else:
            answer = rename(client, "README.md", "moved.md")
            path = "moved.md"
        assert answer.status_code == 200
        body = answer.json()
        stat = (workspace / path).stat()
        assert body["mtime"] == stat.st_mtime
        assert body["size"] == stat.st_size
        # And the same numbers the stat probe reports, so the two can never
        # disagree about what the buffer's token should be.
        probed = client.get("/api/fs/stat", params={"path": path}).json()["paths"][0]
        assert probed["mtime"] == body["mtime"]
        assert probed["size"] == body["size"]

    def test_a_save_that_failed_at_the_replace_leaves_no_temp_behind(
        self, workspace: Path, client, backups: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The durability half of the atomic write, on the failure path.

        The temp holds a full copy of the buffer; nothing would ever look for
        it, so a save that died between writing it and renaming it must have
        taken it with it.
        """
        def refuse(*_args, **_kwargs):
            raise OSError(1, "Operation not permitted")

        monkeypatch.setattr(os, "replace", refuse)
        # The core has no answer for a failed replace, so it raises and the
        # route turns it into a 500 — TestClient surfaces the exception instead
        # of the response. What is asserted is the cleanup on that path.
        with pytest.raises(OSError):
            save(client, "src/config.py", "half-written\n",
                 token(client, "src/config.py"))
        leftovers = [
            entry.name for entry in (workspace / "src").iterdir()
            if entry.name.startswith(".config.py.")
        ]
        assert leftovers == []
        assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"
        # The one thing the temp held that is not on disk anywhere else is the
        # pre-save copy, and it is where the backup chain puts it.
        assert len(editorfs.files.list_backups(workspace / "src" / "config.py")) == 1


class TestTrashIsNotASideDoor:
    """The trash sits *inside the browsed tree*, and that is honest.

    ``TRASH_DIR`` and ``BACKUP_DIR`` are under the app repo's ``logs/`` — the
    same folder the tree browses whenever the working location is the install
    directory, which is the value the app seeds at boot. So the write surface
    adds ``logs/trash`` and ``logs/editor_backups`` to what a user can see.
    That is deliberate, not an oversight: the working location may be
    ``$HOME``, so a trash *inside it* would be a second copy of the user's
    home, and putting the artefacts outside the browsed tree would hide from
    ``git status`` the very moves the user just made. ``logs/`` is gitignored
    already, for the same argument ``mcp_connectors/fedit_mcp.py`` makes about
    ``logs/fedit_backups``.

    What has to hold — and what this class asserts — is that being browsable
    does not make it readable: the trash cannot become a side door around the
    read gate, because the gate judges the *name*, wherever it sits.
    """

    @pytest.fixture
    def repo_like(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A workspace whose trash lives inside itself (the boot shape)."""
        root = tmp_path / "repo"
        (root / "logs" / "trash").mkdir(parents=True)
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
        monkeypatch.setattr(editorfs.constants, "TRASH_DIR", root / "logs" / "trash")
        monkeypatch.setattr(editorfs.constants, "BACKUP_DIR", root / "logs" / "editor_backups")
        return root

    # The batch name a delete would produce, minus the digest — a planted one
    # is fine, because the question is what the gate says about a *name*, not
    # how the file got there.
    PLANTED = "000000000000_planted-batch"

    @pytest.fixture
    def planted_batch(self, repo_like: Path) -> Path:
        """``logs/trash/<batch>/.env`` — a denied name inside the trash."""
        batch = repo_like / "logs" / "trash" / self.PLANTED
        batch.mkdir()
        (batch / ".env").write_text("SECRET=planted\n", encoding="utf-8")
        return batch

    @pytest.fixture
    def repo_client(self, repo_like: Path):
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_the_default_artefact_dirs_are_inside_the_default_browsed_tree(self):
        """The shape the test below stands in for, asserted on the real defaults.

        ``BACKUP_DIR``/``TRASH_DIR`` are children of ``PROJECT_ROOT`` — so a
        workspace left at its boot default browses them. Naming this here keeps
        the arrangement visible instead of inherited.
        """
        root = Path(editorfs.constants.PROJECT_ROOT)
        assert editorfs.constants.BACKUP_DIR == root / "logs" / "editor_backups"
        assert editorfs.constants.TRASH_DIR == root / "logs" / "trash"

    def test_a_trashed_copy_of_a_refused_name_is_still_refused_to_read(
        self, repo_client, planted_batch: Path
    ):
        """A denied name inside the trash is still denied — the read gate judges
        the name, not the folder it happens to sit in.

        A delete can never *produce* this (the subtree rule below refuses the
        delete before anything moves), so the state is planted: an older build's
        copy, or a human's own ``mv``. The promise is that browsing the trash
        leaks nothing the original path would have kept private.
        """
        response = repo_client.get(
            "/api/fs/file", params={"path": f"logs/trash/{self.PLANTED}/.env"}
        )
        assert response.status_code == 422
        assert "never read from it" in response.json()["detail"]

        # The tree omits it rather than marking it: the branch is browsable and
        # its name is harmless, but a denied entry is never published at all.
        listing = repo_client.get("/api/fs/tree", params={"path": "logs/trash"}).json()
        assert self.PLANTED in [entry["name"] for entry in listing["entries"]]
        inner = repo_client.get(
            "/api/fs/tree", params={"path": f"logs/trash/{self.PLANTED}"}
        ).json()["entries"]
        assert inner == []  # an empty-looking branch, not a list of secrets

    def test_a_delete_of_a_batch_holding_a_denied_name_is_refused_whole(
        self, repo_client, planted_batch: Path
    ):
        """The trash cannot launder a protected path either.

        The same subtree rule that protects an innocent folder holding ``.env``
        protects one that already sits in the trash: the route answers 422 and
        nothing moves, so a denied name can only reach the trash from outside
        the route — never through it.
        """
        response = repo_client.request(
            "DELETE", "/api/fs/file", params={"path": f"logs/trash/{self.PLANTED}"}
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "never write to it" in detail
        assert self.PLANTED in detail
        assert (planted_batch / ".env").is_file()
        assert editorfs.trash_batches() == [planted_batch]


class TestReadSurfaceUnmoved:
    """The read half keeps every promise ``tests/test_editorfs.py`` asserts."""

    def test_read_routes_still_answer(self, client, workspace: Path):
        body = client.get("/api/fs/file", params={"path": "src/config.py"}).json()
        assert body["text"] == "x = 1\n"
        assert body["language"] == "python"
        assert body["binary"] is False
        assert client.get("/api/fs/tree").json()["path"] == ""
        assert client.get("/api/fs/stat", params={"path": "README.md"}).json()["paths"][0][
            "exists"
        ] is True

    def test_a_denied_path_still_answers_the_read_flavour(self, workspace: Path, client):
        """The read refusal keeps its own wording — only writes say "write to"."""
        (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
        response = client.get("/api/fs/file", params={"path": ".env"})
        assert response.status_code == 422
        assert "never read from it" in response.json()["detail"]
