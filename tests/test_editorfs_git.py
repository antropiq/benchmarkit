"""Tests for the git half of the Editor tab's file surface: ``/api/fs/status``
(which rows to mark) and ``/api/fs/diff`` (the two sides of one file).

Every test builds its own throw-away repository under ``tmp_path`` and points
``BENCHMARKIT_PATH_LOCATION`` at it, so nothing here reads the suite's own
checkout. The git runs are hermetic: the global/system config is ignored and
the identity comes from the environment, so a developer's ``~/.gitconfig``
(signing, ``core.autocrlf``, a rename threshold) cannot change an answer.

The policy rows here assert one rule, not two (pol-00 D7 deleted the tree's
hardcoded ignored-names set): a path whose verdict is ``hidden`` gets no row. An
``ro`` folder's dirty files **do** — they are visible in the tree, and a marker
with no row (or a row with no marker) would be the disagreement to avoid.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core import editorfs
from src.core.editorfs import policy_file, policy_seed


def write_policy(root: Path, rules: List[Dict[str, str]]) -> Path:
    """Give ``root`` exactly ``rules`` — the only way to change visibility now."""
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


def git(repo: Path, *args: str) -> str:
    """One git command in ``repo``, failing the test on a non-zero exit."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    proc = subprocess.run(
        ["git", *args], cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert proc.returncode == 0, f"git {' '.join(args)}: {proc.stderr.decode()}"
    return proc.stdout.decode()


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A committed two-file repository, pointed at as the working location."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "keep.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("x\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "seed")
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(ws: Path):
    with TestClient(create_app()) as test_client:
        yield test_client


def status(client: TestClient) -> Dict:
    response = client.get("/api/fs/status")
    assert response.status_code == 200, response.text
    return response.json()


def entries_by_path(payload: Dict) -> Dict[str, Dict]:
    return {entry["path"]: entry for entry in payload["entries"]}


@pytest.fixture
def plain_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A client whose working location holds no repository anywhere.

    ``GIT_CEILING_DIRECTORIES`` is what makes that claim about *this* folder
    rather than about the machine: it stops ``git`` from walking out of
    ``tmp_path``, so a checkout somewhere above the pytest baseline (or a
    developer's ``$TMPDIR`` inside one) cannot make the "no repository" answer
    depend on where the suite happened to run.
    """
    plain = tmp_path / "no-repo"
    plain.mkdir()
    (plain / "file.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(plain))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    with TestClient(create_app()) as test_client:
        yield test_client


class TestNoRepository:
    def test_a_plain_folder_is_not_an_error(self, plain_client):
        payload = status(plain_client)
        assert payload == {
            "repo": False,
            "root": None,
            "atToplevel": True,
            "entries": [],
            "truncated": False,
        }

    def test_the_diff_endpoint_says_so_rather_than_404(self, plain_client):
        response = plain_client.get("/api/fs/diff", params={"path": "file.txt"})
        assert response.status_code == 504
        assert "not a git repository" in response.json()["detail"]


class TestStatusClassification:
    def test_clean_work_tree_reports_nothing(self, client, ws):
        payload = status(client)
        assert payload["repo"] is True
        assert payload["root"] == str(ws.resolve())
        assert payload["atToplevel"] is True
        assert payload["entries"] == []

    def test_modified_tracked_file_is_comparable(self, client, ws):
        (ws / "keep.py").write_text("a = 1\nB = 2\nc = 3\n", encoding="utf-8")
        entry = entries_by_path(status(client))["keep.py"]
        assert entry == {"path": "keep.py", "kind": "modified", "comparable": True}

    def test_untracked_file_is_marked_but_not_comparable(self, client, ws):
        (ws / "fresh.txt").write_text("new\n", encoding="utf-8")
        entry = entries_by_path(status(client))["fresh.txt"]
        assert entry == {"path": "fresh.txt", "kind": "untracked", "comparable": False}

    def test_staged_only_change_is_not_unstaged(self, client, ws):
        (ws / "keep.py").write_text("a = 9\n", encoding="utf-8")
        git(ws, "add", "keep.py")
        assert status(client)["entries"] == []

    def test_staged_then_edited_again_is_unstaged(self, client, ws):
        (ws / "keep.py").write_text("a = 9\n", encoding="utf-8")
        git(ws, "add", "keep.py")
        (ws / "keep.py").write_text("a = 10\n", encoding="utf-8")
        entry = entries_by_path(status(client))["keep.py"]
        assert entry["kind"] == "modified"
        assert entry["comparable"] is True

    def test_deleted_file_is_comparable_against_head(self, client, ws):
        (ws / "keep.py").unlink()
        entry = entries_by_path(status(client))["keep.py"]
        assert entry == {"path": "keep.py", "kind": "deleted", "comparable": True}

    def test_entries_are_sorted_by_path(self, client, ws):
        (ws / "zeta.txt").write_text("z\n", encoding="utf-8")
        (ws / "alpha.txt").write_text("a\n", encoding="utf-8")
        (ws / "pkg" / "mid.py").write_text("m\n", encoding="utf-8")
        paths = [entry["path"] for entry in status(client)["entries"]]
        assert paths == sorted(paths)
        assert paths == ["alpha.txt", "pkg/mid.py", "zeta.txt"]

    def test_rename_is_detected_with_its_old_name(self, client, ws):
        git(ws, "mv", "keep.py", "moved.py")
        (ws / "moved.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")
        entry = entries_by_path(status(client))["moved.py"]
        assert entry == {
            "path": "moved.py",
            "kind": "renamed",
            "comparable": True,
            "oldPath": "keep.py",
        }

    def test_status_entry_cap_is_patchable_and_honest(self, ws, client, monkeypatch):
        for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
            (ws / name).write_text("x\n", encoding="utf-8")
        monkeypatch.setattr(editorfs.constants, "MAX_STATUS_ENTRIES", 2)
        payload = status(client)
        assert len(payload["entries"]) == 2
        assert payload["truncated"] is True


class TestStatusFollowsTheTreePolicy:
    """The status list may never name a file the tree refuses to show."""

    def test_a_hidden_paths_row_is_dropped(self, ws, client):
        """One omission rule, and it is the one the tree uses.

        ``.env`` and ``.ssh`` are hidden by rules written here rather than by the
        connector floor's name lists — which is what makes the pairing exact: the
        row disappears with the tree row, because they consult one verdict.
        """
        write_policy(ws, [
            {"path": "**/.ssh", "mode": "hidden"},
            {"path": "**/.env**", "mode": "hidden"},
        ])
        (ws / ".ssh").mkdir()
        (ws / ".ssh" / "notes.txt").write_text("public\n", encoding="utf-8")
        (ws / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
        assert status(client)["entries"] == []
        # The very same call lists a name no rule hides, so the emptiness above
        # is the policy and not a broken query.
        (ws / "outer.txt").write_text("shown\n", encoding="utf-8")
        assert [entry["path"] for entry in status(client)["entries"]] == ["outer.txt"]

    def test_a_read_only_folders_dirty_file_keeps_its_row(self, ws, client):
        """``ro`` is a write rule: the tree shows the file, so status marks it."""
        write_policy(ws, [{"path": "**/vendor", "mode": "ro"}])
        (ws / "vendor").mkdir()
        (ws / "vendor" / "lib.js").write_text("x\n", encoding="utf-8")
        git(ws, "add", "-A")
        git(ws, "commit", "-qm", "vendor")
        (ws / "vendor" / "lib.js").write_text("changed\n", encoding="utf-8")
        assert [entry["path"] for entry in status(client)["entries"]] == ["vendor/lib.js"]
        listed = [
            entry["name"]
            for entry in client.get("/api/fs/tree", params={"path": "vendor"}).json()[
                "entries"
            ]
        ]
        assert listed == ["lib.js"]

    def test_visibility_is_the_workspaces_file_to_change(
        self, ws, client, monkeypatch: pytest.MonkeyPatch
    ):
        """Replaces the old patch-the-constant test: the rule is the mechanism.

        ``pkg`` is listed while no rule names it, and its rows vanish the moment a
        rule hides it — with the same file written either way, so what changed is
        the policy and not the tree.
        """
        (ws / "pkg" / "mod.py").write_text("changed\n", encoding="utf-8")
        assert "pkg/mod.py" in [entry["path"] for entry in status(client)["entries"]]

        write_policy(ws, [{"path": "pkg", "mode": "hidden"}])
        assert status(client)["entries"] == []

    def test_the_reserved_folder_never_draws_a_marker_even_untracked(
        self, ws, client
    ):
        """The wrinkle the new trash placement accepts, pinned rather than hoped.

        A *foreign* repo has no `.nojae/` in its own `.gitignore` (that entry
        lives in benchmarkit's), and the app will not edit a repository it was
        pointed at — so a terminal `git status` there shows `.nojae/` as
        untracked. The app must not: the reservation is unconditional, and every
        path git reported is re-filtered through the verdict, so neither the
        policy file nor a trashed batch inside that folder can become a row the
        tree refuses to show.
        """
        # Seeding is what writes `.nojae/read-write-policy.json` here, and a
        # delete's batch would land in `.nojae/trash` — both untracked.
        policy_seed.ensure_policy_file(str(ws))
        (ws / editorfs.constants.POLICY_DIRNAME / "trash" / "batch").mkdir(parents=True)
        (ws / "shown.txt").write_text("x\n", encoding="utf-8")

        # The raw query really does report the folder — this repo has no
        # `.gitignore` naming it, which is exactly the foreign-workspace wrinkle.
        assert editorfs.constants.POLICY_DIRNAME in git(ws, "status", "--porcelain")

        paths = [entry["path"] for entry in status(client)["entries"]]
        shared = editorfs.constants.policy_shared
        assert editorfs.mode_for(editorfs.constants.POLICY_DIRNAME).mode == shared.HIDDEN
        assert not any(
            part == editorfs.constants.POLICY_DIRNAME for path in paths
            for part in path.split("/")
        )
        assert "shown.txt" in paths


class TestSubdirectoryWorkspace:
    """The working location may be a folder inside a larger checkout."""

    @pytest.fixture
    def inner(self, ws: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        (ws / "outer.txt").write_text("outer\n", encoding="utf-8")
        git(ws, "add", "outer.txt")
        git(ws, "commit", "-qm", "outer")
        (ws / "outer.txt").write_text("outer changed\n", encoding="utf-8")
        (ws / "pkg" / "mod.py").write_text("x changed\n", encoding="utf-8")
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(ws / "pkg"))
        return ws / "pkg"

    def test_paths_stay_workspace_relative(self, client, inner):
        payload = status(client)
        assert payload["atToplevel"] is False
        assert list(entries_by_path(payload)) == ["mod.py"]

    def test_changes_elsewhere_in_the_repo_are_not_listed(self, client, inner):
        assert not any("outer" in entry["path"] for entry in status(client)["entries"])

    def test_the_diff_resolves_the_prefix_both_ways(self, client, inner):
        response = client.get("/api/fs/diff", params={"path": "mod.py"})
        assert response.status_code == 200
        body = response.json()
        assert body["path"] == "mod.py"
        assert body["original"] == "x\n"
        assert body["modified"] == "x changed\n"

    def test_a_path_outside_the_workspace_is_still_422(self, client, inner):
        response = client.get("/api/fs/diff", params={"path": "../outer.txt"})
        assert response.status_code == 422


class TestDiffPayload:
    def test_modified_file_gets_both_sides(self, client, ws):
        (ws / "keep.py").write_text("a = 1\nB = 2\nc = 3\n", encoding="utf-8")
        body = client.get("/api/fs/diff", params={"path": "keep.py"}).json()
        assert body["original"] == "a = 1\nb = 2\nc = 3\n"
        assert body["modified"] == "a = 1\nB = 2\nc = 3\n"
        assert body["status"] == "modified"
        assert body["baseline"] == "head"
        assert body["language"] == "python"
        assert body["binary"] is False
        assert body["truncated"] is False

    def test_deleted_file_keeps_the_committed_side(self, client, ws):
        (ws / "keep.py").unlink()
        body = client.get("/api/fs/diff", params={"path": "keep.py"}).json()
        assert body["status"] == "deleted"
        assert body["original"] == "a = 1\nb = 2\nc = 3\n"
        assert body["modified"] == ""

    def test_rename_falls_back_to_the_stored_blob(self, client, ws):
        git(ws, "mv", "keep.py", "moved.py")
        (ws / "moved.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")
        body = client.get("/api/fs/diff", params={"path": "moved.py"}).json()
        assert body["status"] == "renamed"
        assert body["renamedFrom"] == "keep.py"
        assert body["original"] == "a = 1\nb = 2\nc = 3\n"
        assert body["modified"].endswith("d = 4\n")

    def test_untracked_file_has_no_committed_side(self, client, ws):
        (ws / "fresh.txt").write_text("new\n", encoding="utf-8")
        body = client.get("/api/fs/diff", params={"path": "fresh.txt"}).json()
        assert body["status"] == "untracked"
        assert body["original"] == ""
        assert body["modified"] == "new\n"

    def test_a_file_added_to_the_index_is_reported_as_such(self, client, ws):
        (ws / "staged.txt").write_text("staged\n", encoding="utf-8")
        git(ws, "add", "staged.txt")
        body = client.get("/api/fs/diff", params={"path": "staged.txt"}).json()
        assert body["status"] == "added"
        assert body["original"] == ""

    def test_an_unchanged_file_answers_clean(self, client, ws):
        body = client.get("/api/fs/diff", params={"path": "keep.py"}).json()
        assert body["status"] == "clean"
        assert body["original"] == body["modified"]

    def test_the_answer_comes_from_git_not_from_the_browser(self, client, ws):
        """A marker that went stale cannot make a committed file look changed."""
        (ws / "keep.py").write_text("a = 42\n", encoding="utf-8")
        git(ws, "add", "keep.py")
        git(ws, "commit", "-qm", "absorbed")
        body = client.get("/api/fs/diff", params={"path": "keep.py"}).json()
        assert body["status"] == "clean"
        assert body["original"] == "a = 42\n"

    def test_binary_on_disk_answers_like_the_file_endpoint(self, client, ws):
        (ws / "keep.py").write_bytes(b"\x00\x01not text\n")
        body = client.get("/api/fs/diff", params={"path": "keep.py"}).json()
        assert body["binary"] is True
        assert "original" not in body
        assert "modified" not in body

    def test_binary_in_head_answers_binary_too(self, client, ws):
        (ws / "blob.dat").write_bytes(b"\x00\x02one\n")
        git(ws, "add", "blob.dat")
        git(ws, "commit", "-qm", "blob")
        (ws / "blob.dat").write_bytes(b"\x00\x03two\n")
        body = client.get("/api/fs/diff", params={"path": "blob.dat"}).json()
        assert body["binary"] is True

    def test_a_hidden_path_is_absent_exactly_as_the_file_endpoint_is(
        self, client, ws
    ):
        """A diff is an observation, so it answers absence — never the rule (D8).

        The pair this replaces stated the same rule as a 422 naming the
        deny-list; the policy inverts the number on purpose, and the two read
        endpoints stay identical to each other either way.
        """
        write_policy(ws, [{"path": "**/.env**", "mode": "hidden"}])
        (ws / ".env").write_text("A=1\n", encoding="utf-8")
        response = client.get("/api/fs/diff", params={"path": ".env"})
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]
        assert "A=1" not in response.text

    def test_a_read_only_path_diffs_normally(self, client, ws):
        """``ro`` says nothing may change, not that nothing may be seen."""
        write_policy(ws, [{"path": "keep.py", "mode": "ro"}])
        (ws / "keep.py").write_text("a = 1\nB = 2\nc = 3\n", encoding="utf-8")
        response = client.get("/api/fs/diff", params={"path": "keep.py"})
        assert response.status_code == 200
        assert response.json()["modified"] == "a = 1\nB = 2\nc = 3\n"

    @pytest.mark.parametrize("raw", ["../outside", "/etc/hosts", "~/x", ".."])
    def test_the_same_three_locks_apply(self, client, ws, raw: str):
        assert client.get("/api/fs/diff", params={"path": raw}).status_code == 422

    def test_a_missing_file_answers_404_like_the_file_read(self, client, ws):
        response = client.get("/api/fs/diff", params={"path": "nope.py"})
        assert response.status_code == 404

    def test_the_size_cap_applies_to_the_disk_side(self, ws, client, monkeypatch):
        (ws / "keep.py").write_text("a = 1\nbig\n", encoding="utf-8")
        monkeypatch.setattr(editorfs.constants, "MAX_EDITOR_BYTES", 4)
        response = client.get("/api/fs/diff", params={"path": "keep.py"})
        assert response.status_code == 413
        assert "over the 4-byte cap" in response.json()["detail"]


class TestOddPathNames:
    """Names are byte-exact through ``-z`` and literal through the pathspec."""

    def test_spaces_and_non_ascii_survive_the_round_trip(self, client, ws):
        (ws / "with space.txt").write_text("one\n", encoding="utf-8")
        git(ws, "add", "with space.txt")
        git(ws, "commit", "-qm", "space")
        (ws / "with space.txt").write_text("two\n", encoding="utf-8")
        (ws / "caf\u00e9.md").write_text("caf\u00e9\n", encoding="utf-8")

        paths = list(entries_by_path(status(client)))
        assert "with space.txt" in paths
        assert "caf\u00e9.md" in paths

        body = client.get("/api/fs/diff", params={"path": "with space.txt"}).json()
        assert body["original"] == "one\n"
        assert body["modified"] == "two\n"

    def test_a_name_that_looks_like_a_glob_is_the_file_itself(self, client, ws):
        (ws / "a*b.txt").write_text("one\n", encoding="utf-8")
        git(ws, "add", "a*b.txt")
        git(ws, "commit", "-qm", "glob")
        (ws / "a*b.txt").write_text("two\n", encoding="utf-8")
        body = client.get("/api/fs/diff", params={"path": "a*b.txt"}).json()
        assert body["status"] == "modified"
        assert body["original"] == "one\n"
        assert body["modified"] == "two\n"


class TestGitFailuresAreHonest:
    def test_a_missing_git_binary_is_504_not_an_empty_list(
        self, client, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("PATH", "")
        response = client.get("/api/fs/status")
        assert response.status_code == 504
        assert "git is not available" in response.json()["detail"]

    def test_a_timeout_is_504(self, client, monkeypatch: pytest.MonkeyPatch):
        def expire(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=1)

        monkeypatch.setattr(editorfs.gitstatus.subprocess, "run", expire)
        response = client.get("/api/fs/status")
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]

    def test_the_query_never_touches_the_index_on_disk(self, ws, monkeypatch):
        """``GIT_OPTIONAL_LOCKS=0`` keeps even ``status`` a pure read."""
        seen: List[Dict[str, str]] = []
        real_run = subprocess.run

        def spy(args, **kwargs):
            seen.append(kwargs.get("env") or {})
            return real_run(args, **kwargs)

        monkeypatch.setattr(editorfs.gitstatus.subprocess, "run", spy)
        (ws / "keep.py").write_text("a = 7\n", encoding="utf-8")
        assert editorfs.git_status()["entries"]

        assert seen, "no git command was run"
        assert all(env.get("GIT_OPTIONAL_LOCKS") == "0" for env in seen)
