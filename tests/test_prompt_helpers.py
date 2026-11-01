"""mbl-01 — ``core/prompt_helpers.py``: the constants, the loader, the writer.

Everything runs against a throw-away ``tmp_path`` pointed at by
``BENCHMARKIT_PATH_LOCATION`` — the way ``tests/test_agent_modes.py`` does it,
because the working location is process state and a test that sets it has to set
it per test. The file itself is written at
``root/.nojae/prompt-helpers.json`` through the module's own
``POLICY_DIRNAME`` / ``PROMPT_HELPERS_FILENAME`` attributes, so a rename cannot
silently stop these tests exercising the real path.

Nothing is wired to a route yet (mbl-03 does that), so every assertion here is on
the module's own accessors. Seeding and the legacy lift (mbl-02) are not tested
here — they do not exist yet.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from src.core import prompt_helpers


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch workspace, set as the active working location."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    yield tmp_path


def helpers_path(root: Path) -> Path:
    """The real lookup path for ``root``, built from the module's attributes."""
    return root / prompt_helpers.POLICY_DIRNAME / prompt_helpers.PROMPT_HELPERS_FILENAME


def write_helpers(root: Path, text: str) -> Path:
    """Put ``text`` at the helpers path, creating the reserved folder."""
    folder = root / prompt_helpers.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = helpers_path(root)
    path.write_text(text, encoding="utf-8")
    return path


def document(helpers: List[Any], *, version: Any = prompt_helpers.PROMPT_HELPERS_VERSION,
             omit_version: bool = False) -> str:
    """One helpers document as JSON, its list passed through untouched.

    The malformed cases hand it entries that are not objects and a ``version``
    that is not the accepted one — exactly what the loader has to notice — so the
    helper does not vouch for what it is given.
    """
    payload: Dict[str, Any] = {}
    if not omit_version:
        payload["version"] = version
    payload["helpers"] = helpers
    return json.dumps(payload)


def slots(count: int, color: str = "#2ec27e", content: str = "keep") -> List[Dict[str, str]]:
    return [{"color": color, "content": content} for _ in range(count)]


class TestAbsentFile:
    def test_absent_file_answers_defaults_and_writes_nothing(self, workspace: Path):
        """The read side never creates the folder (D5)."""
        assert prompt_helpers.load_prompt_helpers() == prompt_helpers.default_prompt_helpers()
        assert not (workspace / prompt_helpers.POLICY_DIRNAME).exists()

    def test_absent_folder_answers_defaults_and_writes_nothing(self, workspace: Path):
        """A workspace with no helpers file is a normal state, not a finding."""
        loaded = prompt_helpers.load_prompt_helpers()
        assert len(loaded) == prompt_helpers.SLOT_COUNT
        assert loaded == prompt_helpers.default_prompt_helpers()
        assert list(workspace.iterdir()) == []


class TestWrongCountIsAnError:
    """D4: eight or ten entries raise, and the bytes on disk stay untouched."""

    @pytest.mark.parametrize("count", (0, 8, 10))
    def test_wrong_count_raises_naming_the_count(self, workspace: Path, count: int):
        path = write_helpers(workspace, document(slots(count)))
        before = path.read_text(encoding="utf-8")
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        message = str(exc.value)
        assert f"holds {count} helpers" in message
        assert f"expected exactly {prompt_helpers.SLOT_COUNT}" in message
        assert str(path) in message
        assert path.read_text(encoding="utf-8") == before

    def test_no_pad_no_truncate_no_rewrite(self, workspace: Path):
        path = write_helpers(workspace, document(slots(10)))
        before = path.read_bytes()
        with pytest.raises(prompt_helpers.PromptHelpersFileError):
            prompt_helpers.load_prompt_helpers()
        assert path.read_bytes() == before


class TestUnusableDocument:
    """A file that cannot be used is an error naming the path, never a guess."""

    def test_bad_json(self, workspace: Path):
        path = write_helpers(workspace, "this is not json {{{")
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        assert str(path) in str(exc.value)
        assert path.read_text(encoding="utf-8") == "this is not json {{{"

    def test_missing_version(self, workspace: Path):
        write_helpers(workspace, document(slots(prompt_helpers.SLOT_COUNT), omit_version=True))
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        assert "version" in str(exc.value)

    def test_unknown_version(self, workspace: Path):
        write_helpers(workspace, document(slots(prompt_helpers.SLOT_COUNT), version=2))
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        assert "2" in str(exc.value)

    @pytest.mark.parametrize("version", (True, "1", "one", None, 1.0))
    def test_a_version_that_is_not_the_accepted_number(self, workspace: Path, version: Any):
        """``true`` is an ``int`` in Python, and it is not a version (D4)."""
        write_helpers(workspace, document(slots(prompt_helpers.SLOT_COUNT), version=version))
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        assert str(workspace / prompt_helpers.POLICY_DIRNAME) in str(exc.value)

    def test_helpers_not_a_list(self, workspace: Path):
        write_helpers(workspace, json.dumps({"version": 1, "helpers": {"color": "#ffffff"}}))
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        assert "helpers" in str(exc.value)

    def test_unreadable_file(self, workspace: Path, monkeypatch: pytest.MonkeyPatch):
        """An ``OSError`` on the read is the same error, not a 500."""
        write_helpers(workspace, document(slots(prompt_helpers.SLOT_COUNT)))

        def refuse(*args, **kwargs):
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(Path, "read_text", refuse)
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        assert "cannot be read" in str(exc.value)

    def test_document_not_an_object(self, workspace: Path):
        write_helpers(workspace, json.dumps(slots(prompt_helpers.SLOT_COUNT)))
        with pytest.raises(prompt_helpers.PromptHelpersFileError) as exc:
            prompt_helpers.load_prompt_helpers()
        assert prompt_helpers.POLICY_DIRNAME in str(exc.value)

    def test_a_folder_standing_at_the_path(self, workspace: Path):
        (workspace / prompt_helpers.POLICY_DIRNAME).mkdir()
        helpers_path(workspace).mkdir()
        with pytest.raises(prompt_helpers.PromptHelpersFileError):
            prompt_helpers.load_prompt_helpers()


class TestFieldCoercion:
    """D4's last paragraph: a bad *field* is still healed, and the fix is stored."""

    def test_bad_color_falls_back_to_gray_and_is_rewritten(self, workspace: Path):
        path = write_helpers(
            workspace,
            document([{"color": "bogus", "content": "first"}]
                     + slots(prompt_helpers.SLOT_COUNT - 1)),
        )
        loaded = prompt_helpers.load_prompt_helpers()
        assert loaded[0] == {"color": prompt_helpers.DEFAULT_COLOR, "content": "first"}
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["helpers"][0]["color"] == prompt_helpers.DEFAULT_COLOR

    def test_non_string_content_becomes_empty(self, workspace: Path):
        path = write_helpers(
            workspace,
            document([{"color": "#123456", "content": 42}]
                     + slots(prompt_helpers.SLOT_COUNT - 1)),
        )
        loaded = prompt_helpers.load_prompt_helpers()
        assert loaded[0] == {"color": "#123456", "content": ""}
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["helpers"][0]["content"] == ""

    def test_non_object_entry_is_coerced(self, workspace: Path):
        write_helpers(
            workspace,
            document(["just a string"] + slots(prompt_helpers.SLOT_COUNT - 1)),
        )
        loaded = prompt_helpers.load_prompt_helpers()
        assert loaded[0] == {"color": prompt_helpers.DEFAULT_COLOR, "content": ""}

    def test_coercion_is_idempotent(self, workspace: Path):
        write_helpers(
            workspace,
            document([{"color": "bogus", "content": "first"}]
                     + slots(prompt_helpers.SLOT_COUNT - 1)),
        )
        once = prompt_helpers.load_prompt_helpers()
        second_bytes = helpers_path(workspace).read_bytes()
        assert prompt_helpers.load_prompt_helpers() == once
        assert helpers_path(workspace).read_bytes() == second_bytes

    def test_a_clean_file_is_left_alone(self, workspace: Path):
        path = write_helpers(workspace, document(slots(prompt_helpers.SLOT_COUNT)))
        before = path.read_bytes()
        prompt_helpers.load_prompt_helpers()
        assert path.read_bytes() == before


class TestSeededLiteralRoundTrips:
    """The pairing test: the bytes a seeder will write parse through this loader."""

    def test_default_text_loads_with_no_error(self, workspace: Path):
        path = helpers_path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt_helpers.default_prompt_helpers_text(), encoding="utf-8")
        assert prompt_helpers.load_prompt_helpers() == prompt_helpers.default_prompt_helpers()

    def test_default_text_carries_the_version_and_the_helpers_key(self):
        document_body = json.loads(prompt_helpers.default_prompt_helpers_text())
        assert document_body["version"] == prompt_helpers.PROMPT_HELPERS_VERSION
        assert len(document_body["helpers"]) == prompt_helpers.SLOT_COUNT

    def test_default_list_is_byte_for_byte_what_config_ships_today(self):
        """The move changes the file, not a single character of its content.

        Pinned to literals rather than to this module's own constants on
        purpose: the values are the point — nine gray empty slots, exactly what
        the feature shipped before it moved into the workspace (mbl-03 deletes
        the old ``config`` helper outright, so there is nothing left to compare to).
        """
        assert prompt_helpers.default_prompt_helpers() == [
            {"color": "#808080", "content": ""} for _ in range(9)
        ]
        assert (prompt_helpers.DEFAULT_COLOR, prompt_helpers.CONTENT_MAX_CHARS) == (
            "#808080", 32000,
        )
        assert prompt_helpers.COLOR_RE.pattern == r"^#[0-9a-fA-F]{6}$"


class TestSave:
    def test_save_sets_one_slot_and_keeps_nine(self, workspace: Path):
        write_helpers(workspace, document(slots(prompt_helpers.SLOT_COUNT)))
        saved = prompt_helpers.save_prompt_helper(3, "#0000ff", "x")
        assert len(saved) == prompt_helpers.SLOT_COUNT
        assert saved[3] == {"color": "#0000ff", "content": "x"}
        on_disk = json.loads(helpers_path(workspace).read_text(encoding="utf-8"))
        assert on_disk["version"] == prompt_helpers.PROMPT_HELPERS_VERSION
        assert len(on_disk["helpers"]) == prompt_helpers.SLOT_COUNT
        assert on_disk["helpers"][3] == {"color": "#0000ff", "content": "x"}

    def test_save_creates_the_reserved_folder_it_needs(self, workspace: Path):
        """A **write** may create the folder; a reader (above) must not.

        This is the ``POST`` half of the story the loader's D5 rule sets up: the
        user typed a prompt, so it is saved rather than refused over a folder
        a boot-time seed did not leave behind.
        """
        saved = prompt_helpers.save_prompt_helper(0, "#ff0000", "hello")
        assert saved[0] == {"color": "#ff0000", "content": "hello"}
        assert prompt_helpers.load_prompt_helpers()[0] == {"color": "#ff0000", "content": "hello"}

    def test_save_refuses_a_wrong_count_without_touching_it(self, workspace: Path):
        path = write_helpers(workspace, document(slots(8)))
        before = path.read_bytes()
        with pytest.raises(prompt_helpers.PromptHelpersFileError):
            prompt_helpers.save_prompt_helper(0, "#ff0000", "hello")
        assert path.read_bytes() == before

    def test_save_clears_content_and_keeps_color(self, workspace: Path):
        prompt_helpers.save_prompt_helper(5, "#00ff00", "y")
        saved = prompt_helpers.save_prompt_helper(5, "#00ff00", "")
        assert saved[5] == {"color": "#00ff00", "content": ""}


class TestPathConstruction:
    def test_path_is_the_reserved_folder_under_the_active_root(self, workspace: Path):
        expected = os.path.join(
            str(workspace), prompt_helpers.POLICY_DIRNAME,
            prompt_helpers.PROMPT_HELPERS_FILENAME,
        )
        assert prompt_helpers.prompt_helpers_path() == expected

    def test_explicit_root_path_does_not_move_with_the_location(self, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path / "one"))
        other = str(tmp_path / "two")
        assert prompt_helpers.prompt_helpers_file_path(other) == os.path.join(
            other, prompt_helpers.POLICY_DIRNAME, prompt_helpers.PROMPT_HELPERS_FILENAME
        )

    def test_a_patched_filename_moves_the_lookup(self, workspace: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(prompt_helpers, "PROMPT_HELPERS_FILENAME", "renamed.json")
        assert prompt_helpers.prompt_helpers_path().endswith("renamed.json")
        assert prompt_helpers.load_prompt_helpers() == prompt_helpers.default_prompt_helpers()

    def test_folder_name_is_not_redeclared_here(self):
        assert prompt_helpers.POLICY_DIRNAME == ".nojae"

    def test_resources_dir_plays_no_part(self, workspace: Path,
                                         monkeypatch: pytest.MonkeyPatch):
        from src import config

        monkeypatch.setattr(config, "RESOURCES_DIR", workspace / "elsewhere-resources")
        assert prompt_helpers.prompt_helpers_path() == os.path.join(
            str(workspace), prompt_helpers.POLICY_DIRNAME,
            prompt_helpers.PROMPT_HELPERS_FILENAME,
        )


class TestSelfCheck:
    def test_self_check_passes_at_import(self):
        prompt_helpers._self_check()

    def test_self_check_rejects_a_color_that_its_own_pattern_refutes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(prompt_helpers, "DEFAULT_COLOR", "not-a-color")
        with pytest.raises(ValueError):
            prompt_helpers._self_check()

    def test_self_check_rejects_a_non_integer_slot_count(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(prompt_helpers, "SLOT_COUNT", "9")
        with pytest.raises(ValueError):
            prompt_helpers._self_check()

    def test_self_check_rejects_a_slot_count_the_bullet_row_does_not_draw(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(prompt_helpers, "SLOT_COUNT", 10)
        with pytest.raises(ValueError):
            prompt_helpers._self_check()

    def test_self_check_rejects_an_empty_filename(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(prompt_helpers, "PROMPT_HELPERS_FILENAME", "")
        with pytest.raises(ValueError):
            prompt_helpers._self_check()
