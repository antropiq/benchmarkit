"""mbl-02 — seeding ``.nojae/prompt-helpers.json``, and the one-time lift.

The helpers file arrives the way the policy and modes files do (spec D5): at the
two moments a workspace becomes active, by exclusive create, never by a reader.
So this suite has the shape of ``tests/test_agent_modes_seed.py`` — scratch
workspaces on disk, the helpers exercised directly — and adds the one thing only
this file has: the **lift** from ``resources/memory_registration.json`` (spec
D6), which happens on the branch where the create succeeded and nowhere else.

The promises asserted here:

* **a reader never writes** — ``prompt_helpers_file_state`` answers ``absent``
  and leaves no ``.nojae`` folder behind (the change of behaviour: the old
  app-level reader materialised the file on the first GET);
* **never overwrites** — one ``O_CREAT | O_EXCL``, so a hand-edited file
  survives, and the legacy file is never opened for it;
* **the seed and the loader are the same bytes** — what
  :func:`ensure_prompt_helpers_file` writes round-trips through
  :func:`load_prompt_helpers` as the nine defaults;
* **the lift is one-shot and strict** — copied only when the legacy list holds
  exactly nine entries, otherwise ``"migration-skipped"`` naming the count;
* **the legacy file is left alone** — no delete, no move, no trash, and its
  bytes are identical after a migration;
* **a failure is a note** — an unwritable root answers ``"failed"`` and raises
  nothing.

Every test points ``LEGACY_PROMPT_HELPERS_PATH`` at a scratch file (never the
developer's real ``resources/memory_registration.json``, which is gitignored and
machine-local) — the repo's ``mcphub.constants`` patching convention.
"""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from src.core import prompt_helpers


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch working location with no helpers file and no legacy file."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    # The lift must not read this checkout's real app-level file, whose contents
    # belong to the machine running the suite.
    monkeypatch.setattr(
        prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", tmp_path / "no-legacy.json"
    )
    yield tmp_path


def seeded_path(root: Path) -> Path:
    """The real lookup path for ``root``, built from the module's attributes."""
    return root / prompt_helpers.POLICY_DIRNAME / prompt_helpers.PROMPT_HELPERS_FILENAME


def legacy_file(root: Path, entries: List[Any]) -> Path:
    """A scratch legacy document holding ``entries`` under its old key."""
    path = root / "legacy" / "memory_registration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"registration": entries}, ensure_ascii=False), encoding="utf-8"
    )
    return path


def use_legacy(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Point the module's legacy lookup at ``path``."""
    monkeypatch.setattr(prompt_helpers, "LEGACY_PROMPT_HELPERS_PATH", path)


def slot(color: str, content: str) -> Dict[str, str]:
    return {"color": color, "content": content}


def nine_filled() -> List[Dict[str, str]]:
    return [
        slot(f"#{index:02x}{index:02x}{index:02x}", f"prompt {index}")
        for index in range(prompt_helpers.SLOT_COUNT)
    ]


class TestTheSeedLands:
    def test_it_creates_the_file_with_the_version_and_nine_defaults(
        self, workspace: Path
    ):
        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "created"
        assert result["error"] is None
        assert result["path"] == str(seeded_path(workspace))
        assert seeded_path(workspace).is_file()

        document = json.loads(seeded_path(workspace).read_text(encoding="utf-8"))
        assert document["version"] == prompt_helpers.PROMPT_HELPERS_VERSION
        assert document["helpers"] == prompt_helpers.default_prompt_helpers()
        assert len(document["helpers"]) == prompt_helpers.SLOT_COUNT

    def test_the_seeded_bytes_round_trip_through_the_loader(self, workspace: Path):
        """The pair that may not drift: the seeder's text is what the loader reads.

        No coercion and no error, and the read leaves the file alone — the
        defaults the seed writes are exactly the defaults the loader accepts.
        """
        prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        before = seeded_path(workspace).read_bytes()
        assert prompt_helpers.load_prompt_helpers() == (
            prompt_helpers.default_prompt_helpers()
        )
        assert seeded_path(workspace).read_bytes() == before

    @pytest.mark.posix_behaviour(
        "the readable-file half is the 0o644 the seeder asks for: mode bits and "
        "the uid that may ignore them are POSIX facts (the folder-is-one-level "
        "half above is platform-free and still runs)"
    )
    def test_the_file_is_0644_and_the_folder_is_one_level(self, workspace: Path):
        prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert (workspace / prompt_helpers.POLICY_DIRNAME).is_dir()
        if os.geteuid() != 0:
            assert seeded_path(workspace).stat().st_mode & 0o777 == 0o644
        assert list((workspace / prompt_helpers.POLICY_DIRNAME).iterdir()) == [
            seeded_path(workspace)
        ]

    def test_an_existing_nojae_folder_is_not_touched(self, workspace: Path):
        """The policy and modes seeds usually create the folder earlier."""
        folder = workspace / prompt_helpers.POLICY_DIRNAME
        folder.mkdir()
        (folder / "keepme.txt").write_text("mine\n", encoding="utf-8")

        assert (
            prompt_helpers.ensure_prompt_helpers_file(str(workspace))["state"]
            == "created"
        )
        assert (folder / "keepme.txt").read_text(encoding="utf-8") == "mine\n"

    def test_the_rendered_text_is_byte_for_byte_what_is_written(self, workspace: Path):
        prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert seeded_path(workspace).read_text(encoding="utf-8") == (
            prompt_helpers.default_prompt_helpers_text()
        )

    def test_a_write_that_fails_leaves_no_half_written_file(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No-disk must not mean a permanently broken workspace."""
        real_fdopen = os.fdopen

        def exploding(fd, *args, **kwargs):
            os.close(fd)
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, "fdopen", exploding)
        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "failed"
        assert "space" in result["error"]
        assert not seeded_path(workspace).exists()
        # And the reader is unbothered: the nine defaults, nothing raised.
        assert prompt_helpers.load_prompt_helpers() == (
            prompt_helpers.default_prompt_helpers()
        )

        monkeypatch.setattr(os, "fdopen", real_fdopen)
        assert (
            prompt_helpers.ensure_prompt_helpers_file(str(workspace))["state"]
            == "created"
        )


class TestItNeverOverwrites:
    def test_a_hand_edited_file_survives_the_next_seed(self, workspace: Path):
        prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        path = seeded_path(workspace)
        edited = json.dumps(
            {
                "version": prompt_helpers.PROMPT_HELPERS_VERSION,
                "helpers": [slot("#123456", "mine")]
                + prompt_helpers.default_prompt_helpers()[1:],
            }
        )
        # Deliberately *shorter* as well as different, so a re-write would move
        # the size and cannot hide behind an unchanged mtime.
        path.write_text(edited, encoding="utf-8")
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)

        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "existing"
        assert result["error"] is None
        assert path.read_text(encoding="utf-8") == edited
        assert (path.stat().st_mtime_ns, path.stat().st_size) == stamp

    def test_two_calls_in_a_row_have_one_winner(self, workspace: Path):
        states = [
            prompt_helpers.ensure_prompt_helpers_file(str(workspace))["state"]
            for _ in range(2)
        ]
        assert states == ["created", "existing"]

    def test_a_folder_at_the_file_path_answers_failed(self, workspace: Path):
        """The hostile version of that race: the exclusive open refuses it."""
        folder = workspace / prompt_helpers.POLICY_DIRNAME
        folder.mkdir()
        (folder / prompt_helpers.PROMPT_HELPERS_FILENAME).mkdir()

        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "failed"
        assert result["error"]
        assert "folder" in result["error"]
        assert (folder / prompt_helpers.PROMPT_HELPERS_FILENAME).is_dir()

    def test_the_create_is_exclusive_and_the_folder_is_one_level(self):
        """The non-negotiables as a source rule, not a trust statement.

        Parsed rather than grepped: the module's prose *names* ``makedirs`` to
        say it is not used, and a docstring must not fail a check whose point is
        what the code calls.
        """
        tree = ast.parse(Path(prompt_helpers.__file__).read_text(encoding="utf-8"))
        called = {
            f"{node.func.value.id}.{node.func.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }
        assert "os.mkdir" in called
        assert not {"os.makedirs", "os.path.isfile", "os.path.exists"} & called
        assert any(
            "O_CREAT" in ast.dump(node) and "O_EXCL" in ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        )
        # The legacy copy is read, never moved: destroying the only copy of
        # somebody's prompts is not this function's business (spec D6).
        assert not {"os.remove", "os.replace", "os.rename"} & called
        assert "shutil" not in {
            alias.name
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }


class TestTheReaderDoesNotWrite:
    def test_file_state_reports_absent_without_creating_anything(
        self, workspace: Path
    ):
        result = prompt_helpers.prompt_helpers_file_state(str(workspace))
        assert result["state"] == "absent"
        assert result["path"] == str(seeded_path(workspace))
        assert result["error"] is None
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()

    def test_and_existing_after_a_seed(self, workspace: Path):
        prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert prompt_helpers.prompt_helpers_file_state(str(workspace))["state"] == (
            "existing"
        )

    def test_a_folder_at_the_helpers_path_is_not_reported_as_absent(
        self, workspace: Path
    ):
        folder = workspace / prompt_helpers.POLICY_DIRNAME
        folder.mkdir()
        (folder / prompt_helpers.PROMPT_HELPERS_FILENAME).mkdir()
        result = prompt_helpers.prompt_helpers_file_state(str(workspace))
        assert result["state"] == "failed"
        assert "folder" in result["error"]

    def test_reading_an_absent_file_answers_the_defaults_and_writes_nothing(
        self, workspace: Path
    ):
        """The behaviour change this task keeps: the first GET materialises nothing."""
        assert prompt_helpers.load_prompt_helpers() == (
            prompt_helpers.default_prompt_helpers()
        )
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()


class TestTheLift:
    def test_nine_legacy_entries_are_copied_in_order(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        old = legacy_file(workspace, nine_filled())
        use_legacy(monkeypatch, old)
        before = old.read_bytes()

        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "migrated"
        assert result["path"] == str(seeded_path(workspace))
        document = json.loads(seeded_path(workspace).read_text(encoding="utf-8"))
        assert document["version"] == prompt_helpers.PROMPT_HELPERS_VERSION
        assert document["helpers"] == nine_filled()
        assert list(document) == ["version", "helpers"]

        # And the legacy file is *left in place*, byte-identical (D6).
        assert old.read_bytes() == before

    def test_a_padded_legacy_list_lifts_as_nine(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The real shape on a working machine: five prompts, four empty slots."""
        filled = [slot("#2ec27e", f"prompt {index}") for index in range(5)]
        old = legacy_file(workspace, filled + prompt_helpers.default_prompt_helpers()[5:])
        use_legacy(monkeypatch, old)

        assert prompt_helpers.ensure_prompt_helpers_file(str(workspace))["state"] == (
            "migrated"
        )
        entries = json.loads(seeded_path(workspace).read_text(encoding="utf-8"))["helpers"]
        assert [entry["content"] for entry in entries[:5]] == [
            entry["content"] for entry in filled
        ]
        assert entries[5:] == prompt_helpers.default_prompt_helpers()[5:]

    def test_the_lift_is_one_shot_for_that_workspace(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        old = legacy_file(workspace, nine_filled())
        use_legacy(monkeypatch, old)
        assert prompt_helpers.ensure_prompt_helpers_file(str(workspace))["state"] == (
            "migrated"
        )
        path = seeded_path(workspace)
        edited = path.read_text(encoding="utf-8").replace("prompt 0", "hand edited")
        path.write_text(edited, encoding="utf-8")
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)

        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "existing"
        assert path.read_text(encoding="utf-8") == edited
        assert (path.stat().st_mtime_ns, path.stat().st_size) == stamp

    def test_no_lift_over_an_existing_file(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Both paths present: the workspace's own bytes win, never re-opened."""
        folder = workspace / prompt_helpers.POLICY_DIRNAME
        folder.mkdir()
        sentinel = json.dumps(
            {
                "version": prompt_helpers.PROMPT_HELPERS_VERSION,
                "helpers": [slot("#ff0000", "sentinel")]
                + prompt_helpers.default_prompt_helpers()[1:],
            }
        )
        seeded_path(workspace).write_text(sentinel, encoding="utf-8")
        size = seeded_path(workspace).stat().st_size
        old = legacy_file(workspace, nine_filled())
        use_legacy(monkeypatch, old)

        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "existing"
        assert seeded_path(workspace).read_text(encoding="utf-8") == sentinel
        assert seeded_path(workspace).stat().st_size == size

    def test_a_legacy_list_that_will_not_lift(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Seven entries are neither padded nor truncated: the new file is the
        nine defaults, and the answer names the count and the path (D6)."""
        old = legacy_file(workspace, nine_filled()[:7])
        use_legacy(monkeypatch, old)
        before = old.read_bytes()

        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "migration-skipped"
        assert "7" in result["error"]
        assert str(old) in result["error"]
        assert result["path"] == str(seeded_path(workspace))
        assert json.loads(seeded_path(workspace).read_text(encoding="utf-8"))[
            "helpers"
        ] == prompt_helpers.default_prompt_helpers()
        assert old.read_bytes() == before

    def test_a_legacy_file_that_was_never_a_helper_store_is_not_a_case(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Junk, or no ``registration`` list: no state for it, just ``created``."""
        for text in ("not json at all", "[]", '{"helpers": []}'):
            old = workspace / "legacy.json"
            old.write_text(text, encoding="utf-8")
            use_legacy(monkeypatch, old)
            (workspace / prompt_helpers.POLICY_DIRNAME).mkdir(exist_ok=True)
            seeded_path(workspace).unlink(missing_ok=True)
            assert prompt_helpers.ensure_prompt_helpers_file(str(workspace))[
                "state"
            ] == "created"
        assert json.loads(seeded_path(workspace).read_text(encoding="utf-8"))[
            "helpers"
        ] == prompt_helpers.default_prompt_helpers()

    def test_malformed_legacy_fields_are_normalised(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Nine entries with bad colours still lift — coercion, not an error."""
        raw = [slot("bogus", "one")] + nine_filled()[1:]
        old = legacy_file(workspace, raw)
        use_legacy(monkeypatch, old)

        assert prompt_helpers.ensure_prompt_helpers_file(str(workspace))["state"] == (
            "migrated"
        )
        assert prompt_helpers.load_prompt_helpers()[0] == {
            "color": prompt_helpers.DEFAULT_COLOR,
            "content": "one",
        }

    def test_a_write_that_fails_after_the_lift_leaves_no_fragment(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The migrated write truncates first, so a failure must not persist."""
        old = legacy_file(workspace, nine_filled())
        use_legacy(monkeypatch, old)
        real_write_text = Path.write_text

        def exploding(self, *args, **kwargs):
            if self == seeded_path(workspace):
                raise OSError(28, "No space left on device")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", exploding)
        result = prompt_helpers.ensure_prompt_helpers_file(str(workspace))
        assert result["state"] == "failed"
        assert not seeded_path(workspace).exists()
        assert old.is_file(), "the legacy copy is never the casualty"

        monkeypatch.setattr(Path, "write_text", real_write_text)
        assert prompt_helpers.ensure_prompt_helpers_file(str(workspace))["state"] == (
            "migrated"
        )

    def test_the_loader_never_looks_at_the_legacy_path(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """One lookup, one truth: the read side has no legacy fallback (D6)."""
        old = legacy_file(workspace, nine_filled())
        use_legacy(monkeypatch, old)
        assert prompt_helpers.load_prompt_helpers() == (
            prompt_helpers.default_prompt_helpers()
        )
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()


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
            result = prompt_helpers.ensure_prompt_helpers_file(str(locked))
            assert result["state"] == "failed"
            assert result["error"]
            assert result["path"].endswith(prompt_helpers.PROMPT_HELPERS_FILENAME)
            assert not (locked / prompt_helpers.POLICY_DIRNAME).exists()
        finally:
            locked.chmod(0o755)

    def test_a_missing_root_answers_failed_rather_than_creating_a_chain(
        self, tmp_path: Path
    ):
        result = prompt_helpers.ensure_prompt_helpers_file(str(tmp_path / "never-made"))
        assert result["state"] == "failed"
        assert not (tmp_path / "never-made").exists()

    def test_the_public_call_never_raises_whatever_the_root_is(self, tmp_path: Path):
        """The whole function is guarded: a boot may not die on it."""
        states = ("created", "existing", "migrated", "migration-skipped", "failed")
        for root in ("", str(tmp_path / "x" / "y" / "z"), "\x00bad"):
            assert prompt_helpers.ensure_prompt_helpers_file(root)["state"] in states


class TestTheCallSites:
    """mbl-02 §4: seeding belongs at the moments a workspace becomes active.

    ``config.path_location()`` is a per-request *reader*, so it may not call the
    seeder at all. The callers are exactly the two moments a folder becomes the
    working location: boot (``api/server.py``) and the switch
    (``POST /api/path-location``) — the set below is that pair, so a third
    caller showing up is what this catches.
    """

    REPO = Path(prompt_helpers.__file__).resolve().parents[2]

    def sources(self) -> List[Path]:
        return sorted(
            path
            for path in (self.REPO / "src").rglob("*.py")
            if "__pycache__" not in path.parts
        )

    def test_config_never_seeds_helpers(self):
        text = (self.REPO / "src" / "config.py").read_text(encoding="utf-8")
        assert "ensure_prompt_helpers_file" not in text
        assert "prompt_helpers" not in text

    def test_the_seeders_are_boot_and_the_switch(self):
        seeder = Path(prompt_helpers.__file__)
        callers = {
            path.relative_to(self.REPO).as_posix()
            for path in self.sources()
            if "ensure_prompt_helpers_file(" in path.read_text(encoding="utf-8")
            and path != seeder
        }
        # The two moments a folder becomes the working location (spec D5): boot,
        # and the switch in ``POST /api/path-location`` (mbl-03).
        assert callers == {"src/api/server.py", "src/api/routes.py"}

    def test_the_boot_wrapper_is_shaped_like_the_other_two(self):
        """``_seed_prompt_helpers`` sits beside the policy and modes seeds and
        guards the same way, so a seed failure cannot take the boot down."""
        from src.api import server

        source = Path(server.__file__).read_text(encoding="utf-8")
        for name in (
            "_seed_workspace_policy",
            "_seed_agent_modes",
            "_seed_prompt_helpers",
        ):
            assert f"def {name}(" in source
            assert callable(getattr(server, name))
        assert server._seed_prompt_helpers.__module__ == "src.api.server"

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
        assert not any("prompt-helpers" in line for line in lines)
