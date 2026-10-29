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

What the gate answers with, since pol-06: a **policy verdict**, from two levels.
So the refusals below are arranged by writing rules (``write_policy``), and every
one of them names the level, the mode and the pattern that fired — the sentence
is the point as much as the status code, because it is what the pane shows and
what tells the user which line of their own file to change. A **mutation never
answers 404**: absence is the green light for ``O_CREAT | O_EXCL``, so a hidden
name is a 422 that says so (pol-00 D8).
"""

import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.server import create_app
from src.core import editorfs
from src.core.editorfs import builtin_policy, policy_file, policy_seed


def write_policy(root: Path, rules: List[Dict[str, str]]) -> Path:
    """Give ``root`` exactly ``rules``, the way its owner would have edited them."""
    folder = root / editorfs.constants.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = folder / editorfs.constants.POLICY_FILENAME
    path.write_text(
        json.dumps(
            {"version": editorfs.constants.POLICY_VERSION, "policies": rules},
            indent=2,
        ),
        encoding="utf-8",
    )
    policy_file.invalidate_policy_cache()
    return path


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
    # The sweep prunes the legacy ``logs/trash`` too, so that folder is pinned
    # to a temp path that does not exist: a test asserting about batches must
    # never list — let alone expire — the real repo's copies.
    monkeypatch.setattr(
        editorfs.constants,
        "LEGACY_TRASH_DIR",
        tmp_path / "legacy-trash-absent",
    )
    return folder


@pytest.fixture
def legacy_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin the *legacy* folder on its own, for the un-pinned-trash tests.

    ``sweep_trash`` prunes both folders, so a test that lets the trash resolve
    to a real workspace must still keep the sweep away from the app repo's own
    ``logs/trash`` — which is exactly what it used to be written into.
    """
    folder = tmp_path / "legacy-trash"
    monkeypatch.setattr(editorfs.constants, "LEGACY_TRASH_DIR", folder)
    return folder


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small tree, pointed at as the working location, with the seeded policy.

    The file pol-05 leaves behind when a folder becomes the working location is
    what makes these answers the *default* ones — ``.git`` hidden, dependency
    folders read-only — and the workspace is foreign (not the install directory),
    so benchmarkit's own level A is dormant here.
    """
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    (tmp_path / "src" / "core").mkdir(parents=True)
    (tmp_path / "src" / "config.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "core" / "job.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    policy_seed.ensure_policy_file(str(tmp_path))
    policy_file.invalidate_policy_cache()
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

    # ``(rule, path, is the path readable?)`` — the two modes, and the two
    # answers that go with them. ``ro`` is a *write* rule, so the read stays
    # open and the save does not; ``hidden`` is both, so the read answers absent
    # and the write answers 422 (never the 404 a create would treat as a free
    # name). The seeded file already carries the ``.git`` and ``node_modules``
    # rows; the rest are written by the test.
    LOCKED = [
        # (rules to write, path to try, readable?, fragment of the message)
        ([], ".git/config", False, '".git"'),               # from the seed
        ([], "node_modules/left-pad/index.js", True, '"**/node_modules"'),
        ([{"path": "**/.ssh", "mode": "hidden"}], ".ssh/config", False, '"**/.ssh"'),
        ([{"path": "deploy/id_rsa", "mode": "hidden"}], "deploy/id_rsa", False,
         '"deploy/id_rsa"'),
        ([{"path": "keys", "mode": "ro"}], "keys/server.pem", True, '"keys"'),
        ([{"path": ".env.local", "mode": "ro"}], ".env.local", True, '".env.local"'),
    ]

    @pytest.mark.parametrize(("rules", "rel", "readable", "quoted"), LOCKED)
    def test_a_locked_path_is_refused_naming_the_rule_that_matched(
        self, workspace: Path, client, rules: List[Dict[str, str]], rel: str,
        readable: bool, quoted: str,
    ):
        """422 naming the level, the mode and the pattern — disk untouched."""
        if rules:
            write_policy(workspace, rules)
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original\n", encoding="utf-8")

        response = save(client, rel, "overwritten\n")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "workspace policy" in detail, detail
        assert quoted in detail, detail
        assert target.read_text(encoding="utf-8") == "original\n"

        read_response = client.get("/api/fs/file", params={"path": rel})
        if readable:
            assert read_response.status_code == 200
            assert read_response.json()["text"] == "original\n"
        else:
            assert read_response.status_code == 404
            assert "does not exist" in read_response.json()["detail"]

    def test_an_ro_folder_is_browsable_and_readable_but_never_writable(
        self, workspace: Path, client
    ):
        """The pair that proves ``ro`` is one mode with two answers."""
        (workspace / "mcp_connectors").mkdir()
        (workspace / "mcp_connectors" / "console_mcp.py").write_text(
            "# the policy itself\n", encoding="utf-8"
        )
        write_policy(workspace, [{"path": "**/mcp_connectors", "mode": "ro"}])
        assert client.get(
            "/api/fs/file", params={"path": "mcp_connectors/console_mcp.py"}
        ).status_code == 200
        response = save(client, "mcp_connectors/console_mcp.py", "# rewritten\n")
        assert response.status_code == 422
        assert '"**/mcp_connectors"' in response.json()["detail"]

    def test_a_folder_rule_refuses_an_innocent_descendant(
        self, workspace: Path, client
    ):
        """The subtree rule, on a leaf whose own name says nothing.

        ``.env.local`` is locked by a rule about *it*, and the matcher's subtree
        rule is what reaches ``notes.txt`` — so the refusal quotes the folder's
        pattern, which is the line the user would go and change.
        """
        write_policy(workspace, [{"path": ".env.local", "mode": "ro"}])
        hidden = workspace / ".env.local"
        hidden.mkdir()
        (hidden / "notes.txt").write_text("original\n", encoding="utf-8")
        response = save(client, ".env.local/notes.txt", "overwritten\n")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert '".env.local"' in detail
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

    def test_the_exists_shape_refuses_a_read_only_leaf(self, workspace: Path):
        """``node_modules`` is ``ro`` in the seeded file — no test rule needed."""
        (workspace / "node_modules" / "left-pad" / "index.js").write_text(
            "x\n", encoding="utf-8"
        )
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_existing_in_workspace("node_modules/left-pad/index.js")
        assert caught.value.status_code == 422
        assert '"**/node_modules"' in caught.value.detail

    def test_the_exists_shape_refuses_a_hidden_leaf_by_name_not_by_absence(
        self, workspace: Path
    ):
        """D8, at the gate: a mutation of a hidden path is 422 naming the rule.

        The read gate answers 404 for the same path — that asymmetry lives here,
        in one branch, rather than in each route that could get it wrong.
        """
        (workspace / ".git").mkdir(exist_ok=True)
        (workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_existing_in_workspace(".git/config")
        assert caught.value.status_code == 422
        assert '".git"' in caught.value.detail

        # And the read gate of the very same path is the 404 that hides it.
        with pytest.raises(editorfs.WorkspacePathError) as read_gate:
            editorfs.resolve_in_workspace(".git/config")
        assert read_gate.value.status_code == 404

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

    def test_the_create_shape_gates_the_parent_with_the_write_verdict(
        self, workspace: Path
    ):
        """An ``ro`` folder may not be added to, and the message says which rule."""
        (workspace / "mcp_connectors").mkdir()
        write_policy(workspace, [{"path": "**/mcp_connectors", "mode": "ro"}])
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_parent_in_workspace("mcp_connectors/new.py")
        assert caught.value.status_code == 422
        assert '"**/mcp_connectors"' in caught.value.detail

    def test_the_create_shape_refuses_a_hidden_name_before_the_parent_is_statted(
        self, workspace: Path
    ):
        """The leaf check happens on the name, so no answer confirms the name is free.

        The parent here also does not exist: without the pre-``stat`` leaf check,
        this would be the 409 "parent folder does not exist", which is exactly the
        green light for ``O_CREAT | O_EXCL`` that D8 forbids.
        """
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_parent_in_workspace(".git/HEAD")
        assert caught.value.status_code == 422
        assert '".git"' in caught.value.detail
        assert not (workspace / ".git" / "HEAD").exists()

    def test_the_chain_helper_reports_each_level(self, workspace: Path):
        chain = editorfs.resolve_write_target("src/core/job.py")
        assert [normalised for normalised, _ in chain] == [
            "src", "src/core", "src/core/job.py",
        ]

    def test_a_content_write_and_a_listing_write_are_two_different_questions(
        self, workspace: Path
    ):
        """The split pol-06 draws, and the carve-out it buys.

        :func:`resolve_write_target` insists every folder above the name is
        read-write — that is what D3 means by an ``ro`` folder whose own listing
        may not change. A save replaces bytes in a file that is already there and
        changes no listing at all, so it asks the leaf alone (:func:`resolve_save_target`).
        That is the only way a later rule can carve one file back to ``rw`` under
        a folder rule, and this asserts both halves of that in one workspace.
        """
        write_policy(workspace, [
            {"path": "**/node_modules", "mode": "ro"},
            {"path": "**/node_modules/**/README.md", "mode": "rw"},
        ])
        pkg = workspace / "node_modules" / "left-pad"
        (pkg / "README.md").write_text("docs\n", encoding="utf-8")
        (pkg / "index.js").write_text("x\n", encoding="utf-8")

        # The carve-out saves…
        assert editorfs.resolve_save_target("node_modules/left-pad/README.md")[0] == (
            "node_modules/left-pad/README.md"
        )
        # …and its sibling, which no later rule touched, is refused by the folder.
        with pytest.raises(editorfs.WorkspacePathError) as caught:
            editorfs.resolve_save_target("node_modules/left-pad/index.js")
        assert caught.value.status_code == 422
        assert '"**/node_modules"' in caught.value.detail

        # The listing itself stays locked: adding a name there is refused even
        # though a *file* inside that folder is individually writable.
        with pytest.raises(editorfs.WorkspacePathError) as listing:
            editorfs.resolve_parent_in_workspace("node_modules/left-pad/new.js")
        assert listing.value.status_code == 422
        assert '"**/node_modules"' in listing.value.detail


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

    # ``(rules, path, pattern quoted in the refusal)`` — the leaf name and every
    # folder above it go through the same verdict the save does; one policy, no
    # second copy of it. ``.git`` needs no rule: the seeded file already hides it.
    LOCKED_NAMES = [
        ([{"path": "**/mcp_connectors", "mode": "ro"}], "mcp_connectors/new.py",
         '"**/mcp_connectors"'),
        ([], ".git/config", '".git"'),
        ([{"path": "**/.env**", "mode": "hidden"}], ".env", '"**/.env**"'),
        ([{"path": "**/.env**", "mode": "hidden"}], ".env.local/notes.txt",
         '"**/.env**"'),
        ([{"path": "deploy/id_rsa", "mode": "hidden"}], "deploy/id_rsa",
         '"deploy/id_rsa"'),
        ([{"path": "keys", "mode": "ro"}], "keys/server.pem", '"keys"'),
        ([{"path": "**/.ssh", "mode": "hidden"}], "sub/.ssh/y", '"**/.ssh"'),
    ]

    @pytest.mark.parametrize(("rules", "rel", "quoted"), LOCKED_NAMES)
    def test_a_locked_name_is_422_naming_the_rule(
        self, workspace: Path, client, rules: List[Dict[str, str]], rel: str,
        quoted: str,
    ):
        if rules:
            write_policy(workspace, rules)
        response = create(client, rel)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert quoted in detail, detail
        assert "workspace policy" in detail
        # Nothing landed: not the leaf, and not a locked folder on the way to it
        # — a stray ``.ssh`` or ``keys`` directory would be worse than the file
        # this call was refused for.
        assert not os.path.lexists(workspace / rel)
        for locked in (".env.local", "deploy", "keys", "sub", ".ssh"):
            assert not (workspace / locked).exists(), locked

    def test_a_hidden_folder_name_refuses_a_whole_new_subtree(
        self, workspace: Path, client
    ):
        """The name is the rule's subject, so nothing below it may be made."""
        write_policy(workspace, [{"path": "**/.private", "mode": "hidden"}])
        response = create(client, ".private", is_dir=True)
        assert response.status_code == 422
        assert not (workspace / ".private").exists()
        response = create(client, ".private/deep/inner.txt")
        assert response.status_code == 422
        assert not (workspace / ".private").exists()

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

    def test_a_name_the_policy_locks_cannot_be_made(
        self, workspace: Path, client
    ):
        """The old "created but invisible" pair is gone: one policy, one answer.

        There used to be *two* mechanisms — the tree's hardcoded ignore-list (a
        read rule) and the connector deny-list (a write rule) — and they
        disagreed about ``node_modules``: the folder could be created and then
        vanished from the listing, so the browser saw a create that had not
        happened. Under one verdict that disagreement cannot exist: the seeded
        file marks that name ``ro``, so the create is **refused** and the answer
        says which rule. A user who wants the folder changes the rule, and the
        message names the file to change it in.
        """
        response = create(client, "vendor/node_modules", is_dir=True)
        assert response.status_code == 422
        assert '"**/node_modules"' in response.json()["detail"]
        assert not (workspace / "vendor" / "node_modules").exists()
        # A name no rule mentions is created normally and listed normally.
        assert create(client, "vendor", is_dir=True).status_code == 200
        assert create(client, "vendor/pkgs", is_dir=True).status_code == 200
        assert "pkgs" in [
            entry["name"]
            for entry in client.get(
                "/api/fs/tree", params={"path": "vendor"}
            ).json()["entries"]
        ]

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
    # ``(rules, target, pattern the refusal quotes)`` — the both-ends rule, from
    # the direction a leaf-only gate would miss: an innocent source moving into a
    # folder the policy hides or locks. ``.git`` needs no rule (the seeded file
    # hides it); a ``ro`` target refuses at the *destination*, and the message
    # names the line the user wrote.
    LOCKED_TARGETS = [
        ([], ".git/hooks/pre-commit", '".git"'),
        ([{"path": "**/mcp_connectors", "mode": "ro"}], "mcp_connectors/policy_patch.py",
         '"**/mcp_connectors"'),
        ([{"path": "**/.env**", "mode": "hidden"}], ".env", '"**/.env**"'),
        ([{"path": "keys", "mode": "ro"}], "keys/id_ed25519", '"keys"'),
        ([{"path": "deploy", "mode": "hidden"}], "deploy/server.pem", '"deploy"'),
        ([{"path": "**/.ssh", "mode": "hidden"}], "sub/.ssh/config", '"**/.ssh"'),
    ]

    @pytest.mark.parametrize(("rules", "to", "quoted"), LOCKED_TARGETS)
    def test_a_locked_target_is_422_and_nothing_moved(
        self, workspace: Path, client, rules: List[Dict[str, str]], to: str,
        quoted: str,
    ):
        if rules:
            write_policy(workspace, rules)
        response = rename(client, "src/config.py", to)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "workspace policy" in detail, detail
        assert quoted in detail, detail
        assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"
        assert not os.path.lexists(workspace / to)

    def test_a_locked_target_folder_refuses_a_whole_tree_move(
        self, workspace: Path, client
    ):
        """The source is a folder, so every moved path would be locked too."""
        write_policy(workspace, [{"path": "**/mcp_connectors", "mode": "ro"}])
        (workspace / "mcp_connectors").mkdir()
        response = rename(client, "src", "mcp_connectors/src")
        assert response.status_code == 422
        assert '"**/mcp_connectors"' in response.json()["detail"]
        assert (workspace / "src" / "config.py").is_file()

    def test_an_innocent_name_under_a_locked_folder_is_refused(
        self, workspace: Path, client
    ):
        """The leaf name is fine; its *parent* is the one the rule names."""
        write_policy(workspace, [{"path": ".env.local", "mode": "ro"}])
        (workspace / ".env.local").mkdir()
        response = rename(client, "README.md", ".env.local/notes.txt")
        assert response.status_code == 422
        assert '".env.local"' in response.json()["detail"]
        assert (workspace / "README.md").is_file()

    def test_a_locked_source_is_refused_by_its_own_end(self, workspace: Path, client):
        """The other direction — and D3's rename-out case: ``to`` is innocent.

        No new code enforces that: a move *out* of a locked folder rewrites that
        folder's directory entries, so the source end of the gate is already the
        rule, which is why both ends were gated from the start.
        """
        (workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        response = rename(client, ".git/config", "config.txt")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert '".git"' in detail
        assert "hidden" in detail
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

    def test_a_folder_holding_a_hidden_file_is_refused_and_survives_whole(
        self, workspace: Path, client, trash: Path
    ):
        """The rule the whole subtree check exists for: the *innocent* name is
        not what decides — every descendant is checked first, all-or-nothing.

        The descendant here is one the tree would never even have shown, which is
        the same protection from the other side: an innocent folder cannot be
        deleted *through* a hidden file it happens to hold.
        """
        write_policy(workspace, [
            {"path": "**/.env**", "mode": "hidden"},
            {"path": "**/vendor", "mode": "ro"},
        ])
        (workspace / "src" / ".env").write_text("SECRET=1\n", encoding="utf-8")
        response = trash_path(client, "src")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert '"**/.env**"' in detail
        # The refusal names the offending descendant, not the innocent parent.
        assert "inside src" in detail
        assert (workspace / "src" / "config.py").is_file()
        assert (workspace / "src" / "core" / "job.py").is_file()
        assert (workspace / "src" / ".env").read_text(encoding="utf-8") == "SECRET=1\n"
        # Nothing at all was moved.
        assert not trash.exists()

    def test_a_folder_holding_a_read_only_subtree_is_refused(
        self, workspace: Path, client, trash: Path
    ):
        """``ro`` means the subtree does not move — visible in the tree or not.

        The row that used to be about the connectors' own sources (a name the
        deny-list carried) is now about a workspace rule: same refusal, and it
        quotes the line the user wrote.
        """
        write_policy(workspace, [
            {"path": "**/.env**", "mode": "hidden"},
            {"path": "**/vendor", "mode": "ro"},
        ])
        scripts = workspace / "project" / "vendor"
        scripts.mkdir(parents=True)
        (scripts / "left-pad.js").write_text("# vendored\n", encoding="utf-8")
        response = trash_path(client, "project")
        assert response.status_code == 422
        assert '"**/vendor"' in response.json()["detail"]
        assert (scripts / "left-pad.js").is_file()

    def test_a_hidden_path_deleted_directly_is_422(
        self, workspace: Path, client, trash: Path
    ):
        """A mutation never answers 404, and this is the row that says so.

        The read gate answers "does not exist" for the very same path; the delete
        says which rule hid it, because an absent-sounding answer is what a
        caller would take as permission to create the name again.
        """
        (workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        response = trash_path(client, ".git/config")
        assert response.status_code == 422
        assert '".git"' in response.json()["detail"]
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

    def test_a_folder_holding_a_link_to_a_locked_target_is_refused(
        self, workspace: Path, client, trash: Path
    ):
        """The subtree walk does not *follow* links, but it does judge them:
        a link to a locked path inside an innocent folder is a route to it."""
        write_policy(workspace, [{"path": "**/.ssh", "mode": "hidden"}])
        (workspace / "deploy").mkdir()
        (workspace / "deploy" / "key").symlink_to(workspace / ".ssh" / "id_rsa")
        response = trash_path(client, "deploy")
        assert response.status_code == 422
        assert '"**/.ssh"' in response.json()["detail"]
        assert (workspace / "deploy" / "key").is_symlink()
        assert not trash.exists()

    def test_a_link_pointing_outside_the_workspace_is_refused_by_the_subtree_walk(
        self, workspace: Path, client, trash: Path, tmp_path: Path
    ):
        """No policy of this workspace reaches it, so nothing says it may move."""
        outside = tmp_path.parent / f"{tmp_path.name}-elsewhere"
        outside.mkdir(exist_ok=True)
        (outside / "notes.txt").write_text("x\n", encoding="utf-8")
        (workspace / "project").mkdir()
        (workspace / "project" / "link").symlink_to(outside / "notes.txt")
        response = trash_path(client, "project")
        assert response.status_code == 422
        assert "outside the working location" in response.json()["detail"]
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


class TestTrashBelongsToTheWorkspace:
    """The trash is the *active* workspace's own ``.nojae/trash``, per call.

    The old placement was one module-level constant under ``logs/``: resolved
    once at import, shared by every workspace, so deleting in a foreign project
    wrote bytes inside the benchmarkit checkout. These tests are the proof of
    the fix — the folder is derived from ``workspace_root()`` at call time, so a
    settings switch moves the trash with the workspace, and each root's batches
    are invisible to the other.
    """

    def test_the_default_resolves_to_the_active_workspaces_reserved_folder(
        self, workspace: Path, legacy_trash: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No pin, no patch: ``<root>/.nojae/trash``, named from the constants."""
        monkeypatch.setattr(editorfs.constants, "TRASH_DIR", None)
        resolved = editorfs.trash_dir()
        assert resolved == Path(
            workspace / editorfs.constants.POLICY_DIRNAME
            / editorfs.constants.TRASH_SUBDIR
        )
        # And the reserved name is the *policy* module's, so a test that moves
        # the reserved folder moves the trash with it — no second literal.
        assert editorfs.constants.POLICY_DIRNAME == ".nojae"

    def test_a_delete_without_a_pin_files_into_the_workspaces_own_trash(
        self, workspace: Path, client, legacy_trash: Path,
        monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(editorfs.constants, "TRASH_DIR", None)
        response = trash_path(client, "src/config.py")
        assert response.status_code == 200
        batch = workspace / ".nojae" / "trash" / Path(response.json()["trashedTo"]).name
        assert (batch / "src" / "config.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_trashedTo_is_workspace_relative(
        self, workspace: Path, client, legacy_trash: Path,
        monkeypatch: pytest.MonkeyPatch
    ):
        """``.nojae/trash/<digest>_<stamp>`` — the frame every other payload uses.

        The label names a folder the pane deliberately cannot open, so the
        confirmation quotes it as a path to ``cd`` to in a terminal; a client
        joining it onto the working location lands on the batch.
        """
        monkeypatch.setattr(editorfs.constants, "TRASH_DIR", None)
        body = trash_path(client, "src/config.py").json()
        assert body["trashedTo"].startswith(".nojae/trash/")
        assert not Path(body["trashedTo"]).is_absolute()
        assert (workspace / body["trashedTo"]).is_dir()

    def test_a_switched_in_workspace_gets_its_own_trash(
        self, tmp_path: Path, legacy_trash: Path,
        monkeypatch: pytest.MonkeyPatch
    ):
        """Defect 2, head-on: two roots, two trashes, and neither sees the other.

        One client spans the switch (as a settings change does), so the second
        delete *must* land under the second root and must not see — let alone
        sweep — the first root's batches.
        """
        monkeypatch.setattr(editorfs.constants, "TRASH_DIR", None)
        first = tmp_path / "one"
        second = tmp_path / "two"
        for root in (first, second):
            root.mkdir()
            (root / "f.txt").write_text("content\n", encoding="utf-8")
            policy_seed.ensure_policy_file(str(root))
        policy_file.invalidate_policy_cache()

        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(first))
        with TestClient(create_app()) as client:
            body = trash_path(client, "f.txt").json()
            assert (first / body["trashedTo"]).is_dir()
            assert editorfs.trash_batches()[0].is_relative_to(first / ".nojae")

            # The switch: the very next delete files into the *new* root, and
            # the old root's batch is not this workspace's to list or sweep.
            monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(second))
            policy_file.invalidate_policy_cache()
            body = trash_path(client, "f.txt").json()
            assert (second / body["trashedTo"]).is_dir()
            batches = editorfs.trash_batches()
            assert len(batches) == 1
            assert batches[0].is_relative_to(second / ".nojae")
            assert not any(batch.is_relative_to(first / ".nojae") for batch in batches)

        assert (first / ".nojae" / "trash").is_dir()
        assert (second / ".nojae" / "trash").is_dir()

    def test_the_legacy_logs_trash_is_swept_but_never_written_to(
        self, workspace: Path, client, trash: Path,
        legacy_trash: Path
    ):
        """``logs/trash`` keeps nothing writable, but its batches still expire."""
        # The `trash` pin keeps the sweep's *primary* folder private; the
        # `legacy_trash` fixture does the same for the legacy one, so a test
        # never lists — let alone expires — the real app repo's copies.
        legacy = legacy_trash
        old = legacy / "000000000000_old-batch"
        old.mkdir(parents=True)
        (old / "gone.txt").write_text("x\n", encoding="utf-8")
        os.utime(old, (0, 0))
        young = legacy / "111111111111_young-batch"
        young.mkdir()

        assert trash_path(client, "README.md").status_code == 200
        # The delete's batch went to the workspace's own trash (pinned by the
        # `trash` fixture), never to the legacy folder — which the sweep only
        # *drains*: the over-age batch is gone, the young one stays.
        assert len(list(trash.iterdir())) == 1
        assert not old.exists()
        assert young.is_dir()
        assert [item.name for item in legacy.iterdir()] == [young.name]


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

# ``(path, a fragment of the rule the refusal quotes)``. The seeded file carries
# the ``.git`` and ``**/node_modules`` rows; the test writes the rest. Every
# write route has to refuse every one of them, in the same words — the pattern
# being quoted is what makes a refusal actionable instead of a shrug.
LOCKED = [
    (".env", '"**/.env**"'),
    (".git/config", '".git"'),
    ("deploy/id_rsa", '"deploy/id_rsa"'),
    ("keys/server.pem", '"keys"'),
    ("mcp_connectors/new_entry.py", '"**/mcp_connectors"'),
]

POLICY_FOR_LOCKED = [
    {"path": ".git", "mode": "hidden"},
    {"path": "**/.env**", "mode": "hidden"},
    {"path": "deploy/id_rsa", "mode": "hidden"},
    {"path": "keys", "mode": "ro"},
    {"path": "**/mcp_connectors", "mode": "ro"},
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
    @pytest.mark.parametrize(("rel", "quoted"), LOCKED)
    def test_a_locked_path_is_refused_on_every_route_in_the_same_words(
        self, workspace: Path, client, kind: str, rel: str, quoted: str
    ):
        """One verdict, asked once, whichever route was asked.

        A ``hidden`` row and an ``ro`` row both arrive here: every one of these
        calls is a **mutation**, so both answer 422 naming the rule, and neither
        is allowed the 404 that a read of the hidden one answers. The pattern
        travels through the message because the refusal has to point at the line
        of ``.nojae/read-write-policy.json`` that decides it.
        """
        write_policy(workspace, POLICY_FOR_LOCKED)
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("original\n", encoding="utf-8")

        response = call_write(client, kind, rel)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "workspace policy" in detail, (kind, rel, detail)
        assert quoted in detail, (kind, rel, detail)
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
    """The trash lives *inside the workspace* — and the workspace cannot open it.

    A delete files into ``<workspace>/.nojae/trash``, so the copy of what the
    user deleted sits in the tree the pane browses. The old placement under the
    app repo's ``logs/`` needed level A to hide it, which only worked in
    benchmarkit's own checkout. The new placement is hidden by the *reservation*
    (pol-00 D5): unconditional, in every workspace, whatever the policy file
    says, checked on the component list before any ``stat``. So the guarantee is
    stronger than the one it replaces — there is no batch to browse at all.

    What this class asserts is exactly that: no route can *address* the trash.
    A read of a path inside it answers 404 (indistinguishable from absence,
    D8), the tree never lists the reserved folder, and a delete of anything
    under it is a 422 naming the reservation — so the new location is not a
    side door around the gate for a hidden name, and a batch can never be
    trashed into itself.
    """

    @pytest.fixture
    def repo_like(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A workspace whose trash resolves to its own reserved folder."""
        root = tmp_path / "repo"
        (root / ".nojae" / "trash").mkdir(parents=True)
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
        monkeypatch.setattr(editorfs.constants, "TRASH_DIR", None)
        monkeypatch.setattr(
            editorfs.constants, "LEGACY_TRASH_DIR", tmp_path / "legacy-trash-absent"
        )
        monkeypatch.setattr(editorfs.constants, "BACKUP_DIR", root / "logs" / "editor_backups")
        # The rule that used to be the connector floor's ``.env`` name check:
        # here it is a rule of this workspace, which is what makes the promise
        # below about the *gate* rather than about a built-in list.
        write_policy(root, [{"path": "**/.env**", "mode": "hidden"}])
        return root

    # The batch name a delete would produce, minus the digest — a planted one
    # is fine, because the question is what the gate says about a *name*, not
    # how the file got there.
    PLANTED = "000000000000_planted-batch"

    @pytest.fixture
    def planted_batch(self, repo_like: Path) -> Path:
        """``.nojae/trash/<batch>/.env`` — a hidden name inside the trash.

        Planted on disk, because no route could have written it: that is the
        point of the reservation. It stands in for a batch a delete *would*
        have made, and for a human's own ``mv``.
        """
        batch = repo_like / ".nojae" / "trash" / self.PLANTED
        batch.mkdir()
        (batch / ".env").write_text("SECRET=planted\n", encoding="utf-8")
        return batch

    @pytest.fixture
    def repo_client(self, repo_like: Path):
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_the_default_trash_is_the_workspaces_reserved_folder(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The shape every test here stands in for, asserted on the real default.

        With no pin, the trash is ``<workspace>/.nojae/trash`` — and at the boot
        default (workspace = install dir) that is inside the browsed tree, which
        is fine precisely because the reserved folder is not listable.
        ``BACKUP_DIR`` stays where the ``fedit`` symmetry lives; the legacy
        folder's own batches are the sweep test's business.
        """
        root = Path(editorfs.constants.PROJECT_ROOT)
        # Pinned rather than inherited: the resolution is what is asserted, so
        # the location this process happens to have booted on must not decide it.
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
        assert editorfs.workspace_root() == os.path.realpath(str(root))
        assert editorfs.constants.TRASH_DIR is None
        assert editorfs.trash_dir() == (
            root / editorfs.constants.POLICY_DIRNAME / editorfs.constants.TRASH_SUBDIR
        )
        assert editorfs.constants.BACKUP_DIR == root / "logs" / "editor_backups"

    def test_a_trashed_copy_of_a_hidden_name_is_still_absent_to_a_read(
        self, repo_client, planted_batch: Path
    ):
        """Nothing about a batch is observable, secret or otherwise.

        The path inside the reserved folder is refused one level earlier than
        the old shape: the verdict does not have to judge ``.env`` inside it,
        because the *component* hit answers first — and it answers a look with
        absence (D8), exactly as a hidden path does everywhere else.
        """
        response = repo_client.get(
            "/api/fs/file", params={"path": f".nojae/trash/{self.PLANTED}/.env"}
        )
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]
        assert "SECRET=planted" not in response.text

        # The reserved folder is not a browsable branch with a hidden entry: it
        # is absent from the root listing, and every path under it 404s.
        listing = repo_client.get("/api/fs/tree").json()
        assert ".nojae" not in [entry["name"] for entry in listing["entries"]]
        for rel in (".nojae", ".nojae/trash", f".nojae/trash/{self.PLANTED}"):
            assert repo_client.get(
                "/api/fs/tree", params={"path": rel}
            ).status_code == 404, rel
        assert (
            repo_client.get("/api/fs/stat", params={"path": ".nojae/trash"}).json()[
                "paths"
            ][0]["exists"]
            is False
        )

    def test_a_delete_of_a_batch_holding_a_hidden_name_is_refused_whole(
        self, repo_client, planted_batch: Path
    ):
        """The trash cannot launder a protected path, and cannot be deleted.

        The reservation answers before the name is looked for, so the batch —
        hidden content and all — cannot be addressed by the delete route at
        all: 422 naming the reserved folder, and nothing moves.
        """
        for rel in (".nojae", ".nojae/trash", f".nojae/trash/{self.PLANTED}"):
            response = repo_client.request(
                "DELETE", "/api/fs/file", params={"path": rel}
            )
            assert response.status_code == 422, rel
            assert ".nojae" in response.json()["detail"], rel
        assert (planted_batch / ".env").is_file()
        assert editorfs.trash_batches() == [planted_batch]

    def test_a_batch_is_never_trashed_into_itself(
        self, repo_client, repo_like: Path, planted_batch: Path
    ):
        """The ``shutil.move`` "directory into itself" hazard cannot arise.

        Nothing under the root contains the trash path (only the trash names
        it, and that is unreachable), and no folder holding it can be deleted —
        so the destination is never inside the source. Asserted on a real
        delete of the folder beside the trash.
        """
        response = repo_client.request("DELETE", "/api/fs/file", params={"path": "src"})
        assert response.status_code == 200
        assert (repo_like / ".nojae" / "trash" / self.PLANTED / ".env").is_file()
        batch = repo_like / response.json()["trashedTo"]
        assert (batch / "src" / "app.py").is_file()


class TestPolicyFileIsBroken:
    """D4 at the surface: writes off, everything else on, until the file loads."""

    def test_every_write_route_is_refused_with_pol_04s_message(
        self, workspace: Path, client, trash: Path, backups: Path
    ):
        """The message is pol-04's whole answer: the file, and what is wrong in it."""
        path = write_policy(workspace, [{"path": "**/node_modules", "mode": "hidden"}])
        path.write_text('{"version": 1, "policies": [{"path": "src", "mode": "hidd"}',
                        encoding="utf-8")
        policy_file.invalidate_policy_cache()

        expected = str(path)
        for which, extra in (("save", {"token": None}), ("create", None),
                             ("rename-src", None), ("rename-dst", None),
                             ("delete", None)):
            response = call_write(client, which, "src/config.py", extra)
            assert response.status_code == 422, which
            detail = response.json()["detail"]
            assert "did not load" in detail, (which, detail)
            assert expected in detail, which

    def test_a_read_still_answers_so_the_file_can_be_diagnosed(
        self, workspace: Path, client
    ):
        """Refusing the read too would leave nothing to look at the mistake with."""
        path = write_policy(workspace, [{"path": "src", "mode": "hidden"}])
        path.write_text('{"version": 1, "policies": [', encoding="utf-8")
        policy_file.invalidate_policy_cache()

        assert client.get("/api/fs/file", params={"path": "README.md"}).status_code == 200
        assert client.get("/api/fs/tree").status_code == 200
        assert save(client, "README.md", "no\n").status_code == 422
        # The rules that *would* have protected a folder are not applied either:
        # an unfiltered tree is the honest state of an unloaded policy.
        assert "src" in [
            entry["name"] for entry in client.get("/api/fs/tree").json()["entries"]
        ]

    def test_fixing_the_file_opens_the_writes_again(
        self, workspace: Path, client, backups: Path
    ):
        write_policy(workspace, [{"path": "src", "mode": "ro"}])
        assert save(client, "README.md", "blocked by nothing\n").status_code == 200


class TestBuiltInRulesInBenchmarksOwnWorkspace:
    """Level A at the routes: live only while the workspace *is* the install dir.

    The same paths, the same routes, two workspaces: benchmarkit editing itself
    refuses them with the app's own wording, and a project that merely *has*
    those names treats them as ordinary files. That is the whole difference
    between a rule about who owns a file and a rule about its name.
    """

    @pytest.fixture
    def app_like(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A scratch folder standing in for the install directory.

        Both sides of :func:`is_self_workspace` are read at call time, so moving
        ``constants.PROJECT_ROOT`` beside the location makes a self workspace out
        of a temp folder — which is what lets this test exist at all instead of
        editing the real checkout.
        """
        root = tmp_path / "app"
        (root / "src").mkdir(parents=True)
        (root / "src" / "config.py").write_text("x = 1\n", encoding="utf-8")
        (root / ".env").write_text("ENDPOINT=x\n", encoding="utf-8")
        (root / "logs" / "trash").mkdir(parents=True)
        (root / "web" / "dist").mkdir(parents=True)
        (root / "web" / "dist" / "index.html").write_text("<html></html>\n",
                                                          encoding="utf-8")
        monkeypatch.setattr(editorfs.constants, "PROJECT_ROOT", root)
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
        policy_file.invalidate_policy_cache()
        assert builtin_policy.is_self_workspace() is True
        yield root
        policy_file.invalidate_policy_cache()

    @pytest.fixture
    def app_client(self, app_like: Path):
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_the_apps_own_env_and_logs_are_writable_by_nobody(
        self, app_like: Path, app_client, trash: Path, backups: Path
    ):
        """``.env`` and ``logs/…`` are the app's own, and the refusal says so."""
        assert builtin_policy.is_self_workspace() is True
        response = save(app_client, ".env", "ENDPOINT=evil\n")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "benchmarkit's own rules" in detail
        assert '"**/.env**"' in detail
        assert (app_like / ".env").read_text(encoding="utf-8") == "ENDPOINT=x\n"

        response = save(app_client, "logs/trash/x", "planted\n")
        assert response.status_code == 422
        assert "benchmarkit's own rules" in response.json()["detail"]

    def test_the_apps_own_build_output_is_readable_but_not_writable(
        self, app_like: Path, app_client, backups: Path
    ):
        """Level A's ``ro`` rows: vite's output is browsable and never hand-edited."""
        assert app_client.get(
            "/api/fs/file", params={"path": "web/dist/index.html"}
        ).status_code == 200
        response = save(app_client, "web/dist/index.html", "<html>mine</html>\n")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "benchmarkit's own rules" in detail
        assert "vite build output" in detail

    def test_the_same_paths_in_a_foreign_workspace_are_ordinary(
        self, workspace: Path, client, backups: Path
    ):
        """The pair that makes the condition real rather than decorative."""
        assert builtin_policy.is_self_workspace() is False
        (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (workspace / "logs").mkdir()
        (workspace / "logs" / "notes.txt").write_text("a log\n", encoding="utf-8")

        assert save(client, ".env", "SECRET=2\n").status_code == 200
        assert save(client, "logs/notes.txt", "edited\n").status_code == 200
        assert (workspace / ".env").read_text(encoding="utf-8") == "SECRET=2\n"
        assert (workspace / "logs" / "notes.txt").read_text(encoding="utf-8") == (
            "edited\n"
        )


class TestReadOnlyFolderBlocksEveryWrite:
    """D3 in full: an ``ro`` folder cannot be deleted, emptied, renamed or added to.

    Five routes, five refusals, and the rename is the one that reads like a new
    rule and is not: both ends were always gated, which is why moving *out* of a
    locked folder is refused by its own source end.
    """

    def test_each_write_under_or_around_an_ro_folder_is_refused(
        self, workspace: Path, client, trash: Path, backups: Path
    ):
        pkg = workspace / "node_modules" / "left-pad"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "index.js").write_text("x\n", encoding="utf-8")

        # save: refused by the folder's own rule.
        assert save(client, "node_modules/left-pad/index.js", "y\n").status_code == 422
        # create inside: refused, and nothing was added.
        assert create(client, "node_modules/left-pad/new.js").status_code == 422
        assert not (pkg / "new.js").exists()
        # delete: refused, naming the locked descendant.
        response = trash_path(client, "node_modules/left-pad")
        assert response.status_code == 422
        assert '"**/node_modules"' in response.json()["detail"]
        assert (pkg / "index.js").is_file()
        assert not trash.exists()
        # rename-out: refused at the source end.
        assert rename(client, "node_modules/left-pad", "vendored").status_code == 422
        assert (pkg / "index.js").is_file()
        # rename-in: refused at the destination end.
        assert rename(client, "README.md", "node_modules/README.md").status_code == 422
        assert (workspace / "README.md").is_file()

    def test_a_carve_out_saves_one_file_and_not_its_sibling(
        self, workspace: Path, client, backups: Path
    ):
        """The last-match-wins idiom, at the route that uses it.

        The README is individually ``rw`` under a folder rule; its sibling is not,
        and is refused by the folder. This is the row pol-00's example file is
        written to teach.
        """
        write_policy(workspace, [
            {"path": "**/node_modules", "mode": "ro"},
            {"path": "**/node_modules/**/README.md", "mode": "rw"},
        ])
        pkg = workspace / "node_modules" / "left-pad"
        (pkg / "README.md").write_text("docs\n", encoding="utf-8")
        (pkg / "index.js").write_text("x\n", encoding="utf-8")

        assert save(client, "node_modules/left-pad/README.md", "new docs\n").status_code == 200
        assert (pkg / "README.md").read_text(encoding="utf-8") == "new docs\n"

        response = save(client, "node_modules/left-pad/index.js", "tampered\n")
        assert response.status_code == 422
        assert '"**/node_modules"' in response.json()["detail"]
        assert (pkg / "index.js").read_text(encoding="utf-8") == "x\n"


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

    def test_a_hidden_path_answers_a_read_as_absent_and_a_write_as_a_refusal(
        self, workspace: Path, client
    ):
        """The asymmetry, on one path, in one place: 404 to look, 422 to touch.

        This is the pair the old suite stated as two wordings of one refusal
        ("read from" / "write to"). The policy answers differently on purpose —
        an observation of a hidden name must look like absence, a mutation must
        name the rule — and every write route keeps that promise (the matrix).
        """
        write_policy(workspace, [{"path": "**/.env**", "mode": "hidden"}])
        (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
        read = client.get("/api/fs/file", params={"path": ".env"})
        assert read.status_code == 404
        assert "SECRET=1" not in read.text

        write = save(client, ".env", "overwritten\n")
        assert write.status_code == 422
        assert '"**/.env**"' in write.json()["detail"]
        assert (workspace / ".env").read_text(encoding="utf-8") == "SECRET=1\n"
