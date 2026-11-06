"""ws-01 — ``core/workspace_settings.py``: the constants, the seed, the reader, the writer.

Everything runs against a throw-away ``tmp_path`` pointed at by
``BENCHMARKIT_PATH_LOCATION`` — the way ``tests/test_agent_modes.py`` and
``tests/test_prompt_helpers.py`` do it, because the working location is process
state and a test that sets it has to set it per test. The file itself is written
and read at ``root/.nojae/nojae-settings.json`` through the module's own
``POLICY_DIRNAME`` / ``WORKSPACE_SETTINGS_FILENAME`` /
``WORKSPACE_SETTINGS_TEMPLATE`` attributes, so a rename cannot silently stop
these tests exercising the real path — and nothing here ever touches this
checkout's own ``.nojae``.

Nothing is wired to a route yet (ws-04 does that) and nothing imports the module
yet (ws-03 does that), so every assertion here is on the module's own accessors.
The boot seed and the switch seed (ws-02) are not tested here either.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from src.core import workspace_settings


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch workspace, set as the active working location.

    The module's cache is cleared around each test: it is keyed on the resolved
    settings path, and a test that points the location somewhere else must not
    answer from an entry made about a previous test's file.
    """
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    workspace_settings.invalidate_workspace_settings()
    yield tmp_path
    workspace_settings.invalidate_workspace_settings()


def settings_path(root: Path) -> Path:
    """The real lookup path for ``root``, built from the module's attributes."""
    return root / workspace_settings.POLICY_DIRNAME / workspace_settings.WORKSPACE_SETTINGS_FILENAME


def write_settings(root: Path, text: str) -> Path:
    """Put ``text`` at the settings path, creating the reserved folder."""
    folder = root / workspace_settings.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = settings_path(root)
    path.write_text(text, encoding="utf-8")
    return path


def document(**members: Any) -> str:
    """A settings document as JSON, its members passed through untouched.

    The malformed cases hand it a misspelled member and a value of the wrong
    JSON type — exactly what the loader has to notice — so the helper does not
    vouch for what it is given.
    """
    return json.dumps(members)


def configured(**overrides: Any) -> str:
    """A document with real connection values, so the D8 flag answers configured.

    The template's ``serverUrl``/``endpoint`` are instruction text, so a test
    that wants a file the loader calls *configured* has to replace them; the
    defaults here are what the rest of the suite can then assume.
    """
    payload: Dict[str, Any] = {
        "serverUrl": "http://127.0.0.1:8080/v1",
        "serverApiKey": "",
        "endpoint": "qwen3-32b",
        "supportsVision": False,
        "sysprompt": "You are a helpful assistant.",
        "temperature": 0.2,
        "topP": 0.95,
        "topK": 20,
        "minP": 0,
        "repeatPenalty": 1.0,
        "presencePenalty": 0,
        "reasoning": "off",
        "lspEnabled": [],
        "maxToolIteration": 300,
    }
    payload.update(overrides)
    return json.dumps(payload, indent=2)


def read_document(root: Path) -> Dict[str, Any]:
    """The settings file as written on disk (the bytes, not the loader's view)."""
    return json.loads(settings_path(root).read_text(encoding="utf-8"))


def instruction_member(name: str) -> str:
    """The template's own text for ``name`` — read, never re-typed.

    The D8 tests compare against the template's string rather than a copy, so a
    reworded template cannot leave them asserting a sentence that no longer
    matches what the seed writes.
    """
    raw = json.loads(
        workspace_settings.WORKSPACE_SETTINGS_TEMPLATE.read_text(encoding="utf-8")
    )
    value = raw[name]
    assert isinstance(value, str) and value
    return value


class TestThePathPair:
    def test_the_explicit_root_and_the_active_root_agree(self, workspace: Path):
        """``workspace_settings_path()`` is the active root's file."""
        assert workspace_settings.workspace_settings_path() == str(
            workspace_settings.workspace_settings_file_path(str(workspace))
        )

    def test_a_patched_filename_and_a_switched_root_both_move_the_lookup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The path is built from module attributes per call, not frozen at import."""
        monkeypatch.setattr(
            workspace_settings, "WORKSPACE_SETTINGS_FILENAME", "renamed.json"
        )
        assert workspace_settings.workspace_settings_path().endswith("renamed.json")
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path / "elsewhere"))
        assert workspace_settings.workspace_settings_path().startswith(
            str(tmp_path / "elsewhere")
        )

    def test_the_state_path_is_the_real_lookup(self, workspace: Path):
        state = workspace_settings.workspace_settings_state()
        assert state["path"] == str(settings_path(workspace))


class TestTheSeeder:
    def test_a_fresh_root_gets_the_templates_bytes(self, workspace: Path):
        """``ensure_*`` on an empty root is a byte-for-byte copy of the template."""
        result = workspace_settings.ensure_workspace_settings_file(str(workspace))
        assert result["state"] == "created"
        assert result["path"] == str(settings_path(workspace))
        assert result["error"] is None
        assert settings_path(workspace).read_bytes() == (
            workspace_settings.WORKSPACE_SETTINGS_TEMPLATE.read_bytes()
        )

    def test_a_second_call_reports_existing_and_touches_nothing(self, workspace: Path):
        """An existing file is never overwritten — a workspace's own values stay."""
        workspace_settings.ensure_workspace_settings_file(str(workspace))
        path = settings_path(workspace)
        marker = configured(serverUrl="http://kept:9/v1")
        path.write_text(marker, encoding="utf-8")

        result = workspace_settings.ensure_workspace_settings_file(str(workspace))
        assert result["state"] == "existing"
        assert path.read_text(encoding="utf-8") == marker

    def test_the_seeded_document_is_one_the_loader_accepts(
        self, workspace: Path
    ):
        """The seeder/loader pairing: what is written parses ``ok``, not ``failed``.

        D8 means a seeded file is *unusable*, not *malformed* — it must be state
        ``ok`` (with ``configured`` False), never the ``failed`` of D9.
        """
        workspace_settings.ensure_workspace_settings_file(str(workspace))
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "ok"

    def test_a_missing_template_answers_failed_and_leaves_no_file(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A template the install cannot read is a note, and no half-seeded file."""
        monkeypatch.setattr(
            workspace_settings,
            "WORKSPACE_SETTINGS_TEMPLATE",
            workspace / "gone.default.json",
        )
        result = workspace_settings.ensure_workspace_settings_file(str(workspace))
        assert result["state"] == "failed"
        assert result["error"]
        assert not settings_path(workspace).exists()

    @pytest.mark.posix_behaviour(
        "an unwritable root is made with ``mkdir(mode=0o500)``: without POSIX "
        "mode bits there is no root the seeder refuses to write into"
    )
    def test_an_unwritable_root_answers_failed_and_raises_nothing(
        self, tmp_path: Path
    ):
        if os.geteuid() == 0:
            pytest.skip("root ignores the mode bits")
        locked = tmp_path / "read-only-project"
        locked.mkdir(mode=0o500)
        try:
            result = workspace_settings.ensure_workspace_settings_file(str(locked))
            assert result["state"] == "failed"
            assert result["error"]
            assert result["path"].endswith(
                workspace_settings.WORKSPACE_SETTINGS_FILENAME
            )
            assert not (locked / workspace_settings.POLICY_DIRNAME).exists()
        finally:
            locked.chmod(0o755)

    def test_the_public_call_never_raises_whatever_the_root_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The whole function is guarded: a boot or a switch may not die on it.

        Run with the cwd moved into ``tmp_path``: the empty root resolves
        against it, so even the "created" branch of that loop lands in the
        scratch folder and never in this checkout.
        """
        monkeypatch.chdir(tmp_path)
        for root in ("", str(tmp_path / "x" / "y" / "z"), "\x00bad"):
            result = workspace_settings.ensure_workspace_settings_file(root)
            assert result["state"] in ("created", "existing", "failed")


class TestTheReaderAbsent:
    def test_absent_answers_empty_and_writes_nothing(self, workspace: Path):
        """A reader never creates the reserved folder in a project it only browses."""
        assert workspace_settings.load_workspace_settings() == {}
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "absent"
        assert not (workspace / workspace_settings.POLICY_DIRNAME).exists()


class TestTheReaderMalformed:
    def test_broken_json_fails_naming_the_path_and_keeps_its_bytes(
        self, workspace: Path
    ):
        broken = '{ "serverUrl": "http://x/v1", '
        path = write_settings(workspace, broken)
        assert workspace_settings.load_workspace_settings() == {}
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "failed"
        assert str(path) in state["error"]
        # The bytes are the evidence of what went wrong: the reader does not rewrite.
        assert path.read_text(encoding="utf-8") == broken

    def test_a_non_object_document_fails(self, workspace: Path):
        write_settings(workspace, "[1, 2, 3]")
        assert workspace_settings.workspace_settings_state()["state"] == "failed"

    def test_an_unknown_member_fails_naming_it(self, workspace: Path):
        """D9: ``EndpointsConfig``'s ``extra=\"ignore\"`` would swallow this typo."""
        write_settings(workspace, configured(temperture=0.2))
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "failed"
        assert "temperture" in state["error"]

    @pytest.mark.parametrize("member,bad", [
        ("temperature", "0.2"),   # string where a number belongs
        ("temperature", None),    # null is not a value this file may hold
        ("temperature", True),    # a boolean is not a number here
        ("topK", 2.5),            # a fraction is not an integer
        ("supportsVision", "1"),  # the old dotenv truthy string
        ("lspEnabled", "pyright"),  # a bare string is not an array
        ("lspEnabled", ["pyright", ""]),  # an array entry cannot be blank
        ("maxToolIteration", "300"),
    ])
    def test_a_member_of_the_wrong_type_fails_naming_it(
        self, workspace: Path, member: str, bad: Any
    ):
        write_settings(
            workspace,
            document(**{member: bad, "serverUrl": "http://x/v1", "endpoint": "m"}),
        )
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "failed"
        assert member in state["error"]

    def test_a_wrong_type_is_reported_without_the_bytes_being_repaired(
        self, workspace: Path
    ):
        """A string ``temperature`` stays a string: the reader never coerces (D6)."""
        raw = configured(temperature="0.2")
        path = write_settings(workspace, raw)
        assert workspace_settings.load_workspace_settings() == {}
        assert path.read_text(encoding="utf-8") == raw


class TestTheSeedTextFlag:
    def test_a_freshly_seeded_file_is_ok_but_not_configured(self, workspace: Path):
        """D8: valid JSON, unusable — reported, never dialled into a transport call."""
        workspace_settings.ensure_workspace_settings_file(str(workspace))
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "ok"
        assert state["configured"] is False
        assert state["message"]
        assert str(settings_path(workspace)) in state["message"]

    def test_the_message_names_whichever_member_is_still_the_template(
        self, workspace: Path
    ):
        write_settings(
            workspace,
            document(
                serverUrl="http://real/v1",
                endpoint=instruction_member("endpoint"),
            ),
        )
        state = workspace_settings.workspace_settings_state()
        assert state["configured"] is False
        assert "endpoint" in state["message"]
        assert "serverUrl" not in state["message"]

    def test_real_values_make_it_configured(self, workspace: Path):
        write_settings(workspace, configured())
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "ok"
        assert state["configured"] is True
        assert state["message"] is None

    def test_a_malformed_file_is_never_reported_as_configured(self, workspace: Path):
        """The ``failed`` flag is independent of ``configured``: broken is not ready."""
        write_settings(workspace, "{ not json")
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "failed"
        assert state["configured"] is False


class TestTheTypedAccessor:
    def test_native_types_come_back_native(self, workspace: Path):
        """D6: the accessor answers ``True`` / ``0.2`` / ``["pyright"]``, not strings."""
        write_settings(
            workspace,
            configured(supportsVision=True, temperature=0.7, lspEnabled=["pyright"],
                       maxToolIteration=12),
        )
        assert workspace_settings.workspace_setting("supportsVision") is True
        assert workspace_settings.workspace_setting("temperature") == 0.7
        assert workspace_settings.workspace_setting("lspEnabled") == ["pyright"]
        assert workspace_settings.workspace_setting("maxToolIteration") == 12

    def test_an_absent_member_answers_none_not_a_default(self, workspace: Path):
        """The caller owns the default (spec section 5), so the accessor invents none."""
        write_settings(
            workspace, document(serverUrl="http://x/v1", endpoint="m")
        )
        assert workspace_settings.workspace_setting("temperature") is None

    def test_a_failed_file_answers_no_member(self, workspace: Path):
        write_settings(workspace, configured(temperture=0.2))
        assert workspace_settings.workspace_setting("temperature") is None


class TestTheWriter:
    def test_a_partial_save_updates_one_member_and_keeps_the_other(
        self, workspace: Path
    ):
        """A save writes only what it is given; a valid member stays (spec section 5)."""
        write_settings(workspace, configured(serverUrl="http://keep/v1", temperature=0.2))
        workspace_settings.save_workspace_settings({"temperature": 0.7})

        stored = read_document(workspace)
        assert stored["temperature"] == 0.7
        assert stored["serverUrl"] == "http://keep/v1"
        # And the loader answers the new value: the save invalidated the cache.
        assert workspace_settings.workspace_setting("temperature") == 0.7
        assert workspace_settings.load_workspace_settings()["temperature"] == 0.7

    def test_a_saved_array_is_native_both_on_disk_and_through_the_accessor(
        self, workspace: Path
    ):
        write_settings(workspace, configured())
        workspace_settings.save_workspace_settings(
            {"lspEnabled": ["pyright", "typescript"]}
        )
        assert read_document(workspace)["lspEnabled"] == ["pyright", "typescript"]
        assert workspace_settings.workspace_setting("lspEnabled") == [
            "pyright",
            "typescript",
        ]

    def test_an_unknown_member_raises_and_leaves_the_file_untouched(
        self, workspace: Path
    ):
        raw = configured()
        path = write_settings(workspace, raw)
        with pytest.raises(workspace_settings.WorkspaceSettingsFileError):
            workspace_settings.save_workspace_settings({"temperture": 0.2})
        assert path.read_text(encoding="utf-8") == raw

    def test_a_wrong_typed_value_raises_and_names_the_member(self, workspace: Path):
        write_settings(workspace, configured())
        with pytest.raises(workspace_settings.WorkspaceSettingsFileError) as caught:
            workspace_settings.save_workspace_settings({"temperature": "0.2"})
        assert "temperature" in str(caught.value)

    def test_a_save_creates_the_folder_when_the_file_is_absent(self, workspace: Path):
        """A workspace whose seed failed must still be saveable from the dialog."""
        workspace_settings.save_workspace_settings(
            {"serverUrl": "http://fresh/v1", "endpoint": "m"}
        )
        assert read_document(workspace)["serverUrl"] == "http://fresh/v1"
        assert workspace_settings.workspace_settings_state()["state"] == "ok"

    def test_a_save_over_a_broken_file_repairs_it_to_the_given_members(
        self, workspace: Path
    ):
        """The dialog is how a hand-typed typo is fixed, so a save is not blocked by it."""
        write_settings(workspace, "{ not json at all")
        workspace_settings.save_workspace_settings(
            {"serverUrl": "http://fixed/v1", "endpoint": "m"}
        )
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "ok"
        assert state["configured"] is True
        assert read_document(workspace)["serverUrl"] == "http://fixed/v1"

    def test_the_written_bytes_are_ones_the_loader_reads(self, workspace: Path):
        write_settings(workspace, configured())
        workspace_settings.save_workspace_settings({"sysprompt": "héllo"})
        # Round-trip through the file, not just the in-memory cache.
        workspace_settings.invalidate_workspace_settings()
        assert workspace_settings.workspace_setting("sysprompt") == "héllo"
        # ensure_ascii=False: the accent stays a character, not an escape.
        assert "héllo" in settings_path(workspace).read_text(encoding="utf-8")


class TestTheCache:
    def test_a_load_is_cached_until_invalidation(self, workspace: Path):
        """The file is read on every chat turn, so a second call must not re-read it.

        ``load_*`` answers the entry the first call made: a change on disk with no
        invalidation (a hand-edit, which is D4's case) is not seen until the cache
        is cleared.
        """
        write_settings(workspace, configured(temperature=0.2))
        assert workspace_settings.workspace_setting("temperature") == 0.2
        write_settings(workspace, configured(temperature=0.9))
        # Still the cached 0.2: the loader did not re-read the changed bytes.
        assert workspace_settings.workspace_setting("temperature") == 0.2
        workspace_settings.invalidate_workspace_settings()
        assert workspace_settings.workspace_setting("temperature") == 0.9


class TestTheStateVocabulary:
    def test_the_state_dict_carries_the_agreed_fields(self, workspace: Path):
        """ws-04's dialog switches on one shape: {state, path, error, configured, message}."""
        state = workspace_settings.workspace_settings_state()
        assert set(state) == {"state", "path", "error", "configured", "message"}
        assert state["state"] in ("ok", "absent", "failed")

    def test_a_folder_where_the_file_belongs_fails(self, workspace: Path):
        (workspace / workspace_settings.POLICY_DIRNAME).mkdir(exist_ok=True)
        settings_path(workspace).mkdir()
        state = workspace_settings.workspace_settings_state()
        assert state["state"] == "failed"
        assert workspace_settings.load_workspace_settings() == {}
