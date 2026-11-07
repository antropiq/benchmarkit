"""eds-02 — ``core/agent_modes_write.py``: the whole-document **writer**.

The loader (``core/agent_modes.py``) is forgiving on purpose; this module is the
modes editor's save and it is **strict**: what the loader would drop, a save
refuses (eds-00 D3). Everything here runs against a ``tmp_path`` workspace whose
``.nojae/agent-modes.json`` the test writes itself, and the writer is called
directly — no route yet (eds-03 adds the two), so what is pinned is the module's
own contract: the refusals it raises, the bytes it lands, the guard it refuses a
stale save with, and the backup it keeps.

The promises this suite exists to keep:

* **a save round-trips** — what the writer writes, :func:`resolve_modes` reads
  back with zero findings, in the same order, with the persona byte for byte;
* **the file's own shape survives** — ``version`` from the app's constant,
  ``readme`` preserved verbatim (or absent, or the built-in one for a created
  file), ``agents`` in the body's order;
* **the persona's JSON form is chosen so the reader accepts it back** — a list
  only when no line is blank, a string otherwise, the text never rewritten;
* **nothing is written on a refusal** — a reserved id, an entry the loader would
  drop, a body over the mode cap, a ``default`` naming no declared mode, a file
  whose bytes the reader itself refuses: the disk keeps what it had;
* **a stale save is a conflict carrying both numbers** (D7), and an exclusive
  create treats a file that appeared mid-save as one too;
* **a save keeps the bytes it replaced** (D8) in the app's own backup chain.

The backup is the **injected** callable the module documents: the tests hand it
the Editor's own ``backup_for_write`` — the same object eds-03's route will hand
it — and locate the copies the way ``tests/test_editorfs_write.py::TestBackups``
does, so this suite asserts the one chain rather than a second one.
"""

import ast
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

from src.core import agent_modes, editorfs
from src.core import agent_modes_defaults as defaults
from src.core import agent_modes_write as writer

from tests.test_agent_modes import (
    GOOD_AGENTS,
    STANDIN_DOCUMENT,
    entry,
    install_standin_defaults,
    modes_json,
)


def body(agents: Optional[Dict[str, Any]] = None,
         *,
         default: Any = "code",
         version: Any = None,
         readme: Any = None,
         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One save body, in the shape the dialog sends.

    ``agents`` is passed through untouched rather than rebuilt: half of the
    refusal suite hands it entries the loader would drop, and a helper that
    vouched for them inside it could not express the cases it exists to feed.
    ``version``/``readme`` are here only so a test can show that the writer
    ignores them (D6).
    """
    document: Dict[str, Any] = {}
    if version is not None:
        document["version"] = version
    if readme is not None:
        document["readme"] = readme
    if extra:
        document.update(extra)
    if default is not None:
        document["default"] = default
    if agents is not None:
        document["agents"] = copy.deepcopy(agents)
    return document


# -- the fixtures ---------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch workspace, active, with am-02's content stood in."""
    install_standin_defaults(monkeypatch)
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    agent_modes.invalidate_agent_modes_cache()
    yield tmp_path
    agent_modes.invalidate_agent_modes_cache()


@pytest.fixture
def backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The app's backup folder, moved out of ``logs/`` and into this test's."""
    folder = tmp_path / "app-backups"
    monkeypatch.setattr(editorfs.constants, "BACKUP_DIR", folder)
    return folder


def write_modes(root: Path, text: str) -> Path:
    """The modes file as a hand-editor would leave it."""
    folder = root / agent_modes.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = folder / agent_modes.AGENT_MODES_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def modes_path(root: Path) -> Path:
    return root / agent_modes.POLICY_DIRNAME / agent_modes.AGENT_MODES_FILENAME


def token_of(root: Path) -> str:
    """The token :func:`read_agent_modes_document` publishes for this file.

    Every test that asks for one has a file (or a folder) at the path — the absent
    case is expressed by passing ``None`` to :func:`save` directly.
    """
    token = agent_modes.read_agent_modes_document()["source"]["mtime"]
    assert token is not None, f"no mtime token for {root}"
    return token


def save(document: Any, token: Any = None) -> Dict[str, Any]:
    """Call the writer the way eds-03's route will: body, token, backup callable."""
    return writer.save_agent_modes_document(
        document, token, backup=editorfs.files.backup_for_write
    )


def disk_document(root: Path) -> Dict[str, Any]:
    return json.loads(modes_path(root).read_text(encoding="utf-8"))


def folder_names(root: Path) -> List[str]:
    """Everything the reserved folder holds — a leftover temp file shows up here."""
    folder = root / agent_modes.POLICY_DIRNAME
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def refused(call, kind: type) -> Any:
    """Run ``call``, expect ``kind``, and hand back the exception for inspection."""
    with pytest.raises(kind) as caught:
        call()
    return caught.value


# -- the round-trip -------------------------------------------------------------

class TestRoundTrip:
    def test_a_saved_document_reads_back_with_no_findings(self, workspace: Path):
        answer = save(body(GOOD_AGENTS), None)

        payload = agent_modes.resolve_modes()
        assert payload["findings"] == []
        assert payload["source"]["loaded"] is True
        assert [mode.id for mode in payload["modes"]][:-1] == ["chat", "code", "architect"]
        assert payload["default"] == "code"
        assert answer["created"] is True

    def test_the_persona_text_survives_byte_for_byte(self, workspace: Path):
        agents = {"scribe": entry(label="Scribe", mcptools=True,
                                  systemMessage="  first line\n  second line  ")}
        save(body(agents, default="scribe"), None)

        mode = next(m for m in agent_modes.resolve_modes()["modes"] if m.id == "scribe")
        assert mode.system_message == "  first line\n  second line  "

    def test_the_saved_bytes_are_a_document_the_loader_accepts(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        save(body(GOOD_AGENTS), token_of(workspace))

        modes, declared, findings, usable = agent_modes._validated_document(
            disk_document(workspace), "<on-disk>"
        )
        assert (usable, findings) == (True, [])
        assert [mode.id for mode in modes] == ["chat", "code", "architect"]
        assert declared == "code"

    def test_a_second_save_of_the_same_body_changes_nothing_but_the_mtime(
        self, workspace: Path
    ):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        first = token_of(workspace)
        save(body(GOOD_AGENTS), first)
        bytes_after = modes_path(workspace).read_bytes()

        second = save(body(GOOD_AGENTS), token_of(workspace))
        assert modes_path(workspace).read_bytes() == bytes_after
        assert second["source"]["mtime"] != first

    def test_the_file_order_is_the_bodys_order(self, workspace: Path):
        agents = {
            "zeta": entry(label="Zeta"),
            "alpha": entry(label="Alpha"),
            "mid": entry(label="Mid"),
        }
        save(body(agents, default="mid"), None)

        assert list(disk_document(workspace)["agents"]) == ["zeta", "alpha", "mid"]
        assert [mode.id for mode in agent_modes.resolve_modes()["modes"]][:-1] == [
            "zeta", "alpha", "mid"
        ]


class TestPersonaFormOnDisk:
    def test_a_persona_with_no_blank_line_lands_as_a_list(self, workspace: Path):
        agents = {"code": entry(systemMessage="one line\ntwo lines")}
        save(body(agents), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] == [
            "one line", "two lines"
        ]

    def test_a_persona_with_a_blank_line_lands_as_a_string(self, workspace: Path):
        agents = {"code": entry(systemMessage="one line\n\ntwo lines")}
        save(body(agents), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] == (
            "one line\n\ntwo lines"
        )

    def test_a_blank_line_round_trips_through_the_reader(self, workspace: Path):
        """The exact bug the list form would commit: a file the reader refuses."""
        text = "line one\n\nline two\n\nline three"
        save(body({"code": entry(systemMessage=text)}, default="code"), None)

        payload = agent_modes.resolve_modes()
        assert payload["findings"] == []
        mode = next(m for m in payload["modes"] if m.id == "code")
        assert mode.system_message == text

    def test_a_body_carrying_the_list_form_of_a_blank_line_persona_lands_as_a_string(
        self, workspace: Path
    ):
        """The document read reports a persona in the form the file wrote, so the
        body can come back holding a list — and a list whose text has a blank line is
        what a textarea produces. D6's rule is about the text, not the shape sent."""
        lines = ["Answer briefly.", "", "Never touch the workspace."]
        save(body({"code": entry(systemMessage=lines)}, default="code"), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] == (
            "\n".join(lines)
        )
        mode = next(m for m in agent_modes.resolve_modes()["modes"] if m.id == "code")
        assert mode.system_message == "\n".join(lines)

    def test_a_body_carrying_the_list_form_of_a_clean_persona_lands_as_a_list(
        self, workspace: Path
    ):
        lines = ["Answer briefly.", "Never touch the workspace."]
        save(body({"code": entry(systemMessage=lines)}, default="code"), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] == lines

    def test_a_line_of_spaces_forces_the_string_form_rather_than_vanishing(
        self, workspace: Path
    ):
        text = "one\n   \ntwo"
        save(body({"code": entry(systemMessage=text)}, default="code"), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] == text
        assert agent_modes.resolve_modes()["findings"] == []

    def test_a_string_and_a_list_persona_land_as_the_same_bytes(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        token = token_of(workspace)
        save(body({"code": entry(systemMessage="a\nb")}, default="code"), token)
        as_written = modes_path(workspace).read_bytes()

        save(body({"code": entry(systemMessage=["a", "b"])}, default="code"),
             token_of(workspace))
        assert modes_path(workspace).read_bytes() == as_written

    def test_an_empty_persona_lands_as_an_empty_string_not_an_empty_list(
        self, workspace: Path
    ):
        save(body({"code": entry(systemMessage="")}, default="code"), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] == ""
        assert agent_modes.resolve_modes()["findings"] == []

    def test_a_null_persona_keeps_saying_use_the_configured_prompt(self, workspace: Path):
        save(body({"code": entry(label="Code", systemMessage=None)}, default="code"), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] is None
        payload = agent_modes.resolve_modes()
        assert payload["findings"] == []
        assert next(m for m in payload["modes"] if m.id == "code").system_message == ""

    def test_a_persona_with_no_lines_at_all_is_not_written_as_an_empty_list(
        self, workspace: Path
    ):
        save(body({"code": entry(systemMessage=[])}, default="code"), None)

        assert disk_document(workspace)["agents"]["code"]["systemMessage"] == ""
        assert agent_modes.resolve_modes()["findings"] == []


class TestTheShapeOfTheWrittenFile:
    def test_the_version_in_the_body_is_ignored_and_the_constant_lands(
        self, workspace: Path
    ):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        save(body(GOOD_AGENTS, version=99), token_of(workspace))

        assert disk_document(workspace)["version"] == agent_modes.AGENT_MODES_VERSION

    def test_a_body_with_no_version_is_still_written_with_one(self, workspace: Path):
        answer = save(body(GOOD_AGENTS, version=None), None)
        assert answer["created"] is True
        assert disk_document(workspace)["version"] == agent_modes.AGENT_MODES_VERSION

    def test_the_readme_of_the_file_is_preserved_verbatim(self, workspace: Path):
        readme = ["hand-written line", "another one", "", "  spaced  "]
        write_modes(workspace, modes_json(agents=GOOD_AGENTS, readme=readme))
        save(body(GOOD_AGENTS, readme=["the dialog invented this"]), token_of(workspace))

        assert disk_document(workspace)["readme"] == readme

    def test_a_readme_written_as_a_string_stays_a_string(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS, readme="one prose line"))
        save(body(GOOD_AGENTS), token_of(workspace))

        assert disk_document(workspace)["readme"] == "one prose line"

    def test_a_null_readme_stays_null_rather_than_becoming_the_builtin_one(
        self, workspace: Path
    ):
        """An absent ``readme`` and a ``null`` one are two documents (D6), and the
        writer keeps them two — the built-in lines belong to a *created* file only."""
        write_modes(workspace, modes_json(agents=GOOD_AGENTS, extra={"readme": None}))
        save(body(GOOD_AGENTS), token_of(workspace))

        document = disk_document(workspace)
        assert document["readme"] is None

    def test_a_file_with_no_readme_gains_none(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        save(body(GOOD_AGENTS), token_of(workspace))

        assert "readme" not in disk_document(workspace)

    def test_a_created_file_gets_the_builtin_readme(self, workspace: Path):
        save(body(GOOD_AGENTS), None)

        document = disk_document(workspace)
        assert isinstance(document["readme"], list)
        assert document["readme"] == STANDIN_DOCUMENT["readme"]

    def test_the_written_bytes_end_with_a_newline_and_keep_the_prose_readable(
        self, workspace: Path
    ):
        save(body({"code": entry(label="Kod", systemMessage="žluťoučký kůň")}), None)

        raw = modes_path(workspace).read_bytes()
        assert raw.endswith(b"\n")
        assert "žluťoučký kůň".encode("utf-8") in raw  # ensure_ascii=False

    def test_the_written_key_order_is_the_built_in_documents(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS, readme=["x"]))
        save(body(GOOD_AGENTS), token_of(workspace))

        assert list(disk_document(workspace)) == ["version", "readme", "default", "agents"]

    def test_an_entry_keeps_only_the_keys_the_vocabulary_knows(self, workspace: Path):
        agents = {"code": entry(label="Code", mcptools=True, lsptools=False,
                                systemMessage="ship it")}
        save(body(agents), None)

        assert list(disk_document(workspace)["agents"]["code"]) == [
            "label", "mcptools", "lsptools", "systemMessage"
        ]

    def test_a_label_is_written_as_the_form_holds_it_not_pre_trimmed(self, workspace: Path):
        long_label = "  a label with room to spare and then some more  "
        save(body({"code": entry(label=long_label)}), None)

        assert disk_document(workspace)["agents"]["code"]["label"] == long_label
        shown = next(m.label for m in agent_modes.resolve_modes()["modes"]
                     if m.id == "code")
        assert shown == long_label.strip()[: agent_modes.MAX_MODE_LABEL_CHARS]


# -- the refusals (D3's server-side half, and D4) -------------------------------

class TestRefusalsLeaveTheDiskAlone:
    def test_a_body_declaring_a_reserved_id_is_refused_and_nothing_lands(
        self, workspace: Path
    ):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = modes_path(workspace).read_bytes()
        agents = {**copy.deepcopy(GOOD_AGENTS), "todoBuilder": entry(label="Mine")}

        error = refused(lambda: save(body(agents), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert error.status_code == 422
        assert any("agents.todoBuilder" in item["message"]
                   for item in error.findings)
        assert modes_path(workspace).read_bytes() == before
        assert folder_names(workspace) == [agent_modes.AGENT_MODES_FILENAME]

    @pytest.mark.parametrize(
        "key, bad, expected",
        [
            ("mcptools", entry(mcptools="yes"), "agents.code.mcptools"),
            ("lsptools", entry(lsptools=1), "agents.code.lsptools"),
            ("label", entry(label=42), "agents.code.label"),
            ("systemMessage", entry(systemMessage={"a": 1}), "agents.code.systemMessage"),
            ("systemMessage", entry(systemMessage=["ok", 7]), "agents.code.systemMessage"),
            ("mcptool", entry(mcptool=True), "agents.code.mcptool"),
        ],
        ids=["tool-flag-string", "tool-flag-number", "label-number",
             "persona-object", "persona-list-holding-a-number", "unknown-key"],
    )
    def test_each_invalid_entry_kind_is_refused_naming_its_pointer(
        self, workspace: Path, key: str, bad: Dict[str, Any], expected: str
    ):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = modes_path(workspace).read_bytes()
        agents = {**copy.deepcopy(GOOD_AGENTS), "code": bad}

        error = refused(lambda: save(body(agents), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert error.status_code == 422
        assert [expected] == [
            item["message"].split(":", 1)[0] for item in error.findings
            if item["message"].startswith(expected)
        ]
        assert modes_path(workspace).read_bytes() == before

    def test_the_refusal_sentences_are_the_loaders_own(self, workspace: Path):
        """The dialog shows what the reader would have said, not a new sentence — and
        it says it about every problem, including the ones the reader *repairs*."""
        agents = {**copy.deepcopy(GOOD_AGENTS), "code": entry(mcptools="yes")}
        text = modes_json(agents=agents)
        write_modes(workspace, text)
        loader_says = [f.message for f in agent_modes.resolve_modes()["findings"]
                       if f.kind == agent_modes.KIND_ERROR]
        assert loader_says  # the entry finding plus the default it leaves behind

        error = refused(lambda: save(json.loads(text), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert [item["message"] for item in error.findings] == loader_says

    def test_an_entry_that_is_not_an_object_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        agents = {**copy.deepcopy(GOOD_AGENTS), "code": "just a string"}

        error = refused(lambda: save(body(agents), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert any(item["message"].startswith("agents.code:") for item in error.findings)
        assert modes_path(workspace).read_text(encoding="utf-8").count("just a string") == 0

    def test_a_key_outside_the_id_pattern_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        agents = {**copy.deepcopy(GOOD_AGENTS), "Bad Id": entry(label="Nope")}

        error = refused(lambda: save(body(agents), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert any("Bad Id" in item["message"] for item in error.findings)
        assert modes_path(workspace).read_bytes() == (
            modes_json(agents=GOOD_AGENTS).encode("utf-8")
        )

    def test_a_mode_id_over_the_length_cap_is_refused(self, workspace: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(agent_modes, "MAX_MODE_ID_CHARS", 4)
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body({"longer_id": entry(label="Code")},
                                          default="longer_id"),
                                     token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert error.findings[0]["message"].startswith("agents.longer_id:")
        assert "longer than 4 characters" in error.findings[0]["message"]

    def test_the_mode_cap_is_refused_and_it_is_patchable(self, workspace: Path,
                                                         monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(agent_modes, "MAX_MODES", 2)
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = modes_path(workspace).read_bytes()

        error = refused(lambda: save(body(GOOD_AGENTS), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert "more than the 2 mode(s)" in error.findings[0]["message"]
        assert error.findings[0]["message"].startswith("agents:")
        assert modes_path(workspace).read_bytes() == before

    def test_a_default_naming_no_declared_mode_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body(GOOD_AGENTS, default="ghost"),
                                     token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert '"default" is "ghost"' in error.findings[0]["message"]
        assert disk_document(workspace)["default"] == "code"

    def test_a_default_naming_the_reserved_mode_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body(GOOD_AGENTS, default="todoBuilder"),
                                     token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert '"default" is "todoBuilder"' in error.findings[0]["message"]
        assert disk_document(workspace)["default"] == "code"

    def test_a_missing_default_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body(GOOD_AGENTS, default=None),
                                     token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert 'has no "default" key' in error.findings[0]["message"]

    def test_an_unknown_top_level_key_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body(GOOD_AGENTS, extra={"agent": {}}),
                                     token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert "unknown top-level key" in error.findings[0]["message"]

    def test_a_body_that_is_not_an_object_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = modes_path(workspace).read_bytes()

        refused(lambda: save(["not", "a", "document"], token_of(workspace)),
                writer.AgentModesDocumentInvalid)

        assert modes_path(workspace).read_bytes() == before

    def test_agents_that_is_not_an_object_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body(None, extra={"agents": []}),
                                     token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert '"agents" is a list' in error.findings[0]["message"]

    def test_a_persona_over_the_byte_cap_is_refused_naming_the_cap(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(agent_modes, "MAX_MODE_SYSTEM_MESSAGE_BYTES", 32)
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(
            body({"code": entry(systemMessage="x" * 40)}, default="code"),
            token_of(workspace)),
            writer.AgentModesDocumentInvalid)

        assert "over the 32 bytes cap" in error.findings[0]["message"]

    def test_every_problem_in_one_body_is_reported_at_once(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        agents = {
            "chat": entry(mcptools="yes", label=7),
            "code": entry(lsptools=None),
        }

        error = refused(lambda: save(body(agents, default="ghost"), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        pointers = [item["message"] for item in error.findings]
        assert len(pointers) >= 4
        assert any("agents.chat.mcptools" in line for line in pointers)
        assert any("agents.chat.label" in line for line in pointers)
        assert any("agents.code.lsptools" in line for line in pointers)
        assert any('"default" is "ghost"' in line for line in pointers)

    def test_a_file_that_declared_a_reserved_id_by_hand_is_cleaned_up_by_a_save(
        self, workspace: Path
    ):
        """D4's edge case: the *file* naming one is a note, not a blocking finding,
        and the save is what fixes it — because the body cannot have carried it."""
        agents = {**copy.deepcopy(GOOD_AGENTS), "todoBuilder": entry(label="Hand")}
        write_modes(workspace, modes_json(agents=agents))
        assert agent_modes.resolve_modes()["findings"]  # the note is there

        answer = save(body(GOOD_AGENTS), token_of(workspace))

        assert "todoBuilder" not in disk_document(workspace)["agents"]
        assert answer["findings"] == []
        assert [mode["id"] for mode in answer["modes"]] == [
            "chat", "code", "architect", "todoBuilder"
        ]

    def test_a_refused_save_leaves_no_temp_file_and_no_half_written_file(
        self, workspace: Path
    ):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        agents = {**copy.deepcopy(GOOD_AGENTS), "code": entry(mcptools="yes")}

        refused(lambda: save(body(agents), token_of(workspace)),
                writer.AgentModesDocumentInvalid)

        assert folder_names(workspace) == [agent_modes.AGENT_MODES_FILENAME]
        assert agent_modes.resolve_modes()["findings"] == []


class TestTheFileOnDiskIsNotOverwrittenWhenTheReaderRefusesIt:
    """D3's file-level findings hold the Save button down; the server agrees."""

    def test_a_file_that_is_not_valid_JSON_stays_where_it_is(self, workspace: Path):
        path = write_modes(workspace, '{"version": 1, "agents": ')
        before = path.read_bytes()

        error = refused(lambda: save(body(GOOD_AGENTS), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert "is not valid JSON" in error.findings[0]["message"]
        assert path.read_bytes() == before

    def test_a_file_whose_version_this_app_does_not_understand_is_kept(
        self, workspace: Path
    ):
        text = modes_json(agents=GOOD_AGENTS, version=2)
        path = write_modes(workspace, text)

        error = refused(lambda: save(body(GOOD_AGENTS), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert "version 1 only" in error.findings[0]["message"]
        assert path.read_text(encoding="utf-8") == text

    def test_a_file_whose_agents_is_not_an_object_is_kept(self, workspace: Path):
        path = write_modes(workspace, modes_json(agents=None, extra={"agents": "nope"}))
        before = path.read_bytes()

        error = refused(lambda: save(body(GOOD_AGENTS), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert '"agents" is a string' in error.findings[0]["message"]
        assert path.read_bytes() == before

    def test_a_folder_at_the_path_says_so_rather_than_creating_over_it(
        self, workspace: Path
    ):
        """The document read stamps whatever stands at the path, so a folder's own
        token passes the guard and this is the refusal the save answers with."""
        folder = workspace / agent_modes.POLICY_DIRNAME
        folder.mkdir(exist_ok=True)
        (folder / agent_modes.AGENT_MODES_FILENAME).mkdir()

        error = refused(lambda: save(body(GOOD_AGENTS), token_of(workspace)),
                        writer.AgentModesDocumentInvalid)

        assert "is a folder, not a file" in error.findings[0]["message"]
        assert (folder / agent_modes.AGENT_MODES_FILENAME).is_dir()

    def test_an_unchanged_file_with_a_broken_entry_is_still_writable(
        self, workspace: Path
    ):
        """Entry-level problems are the dialog's to fix; the file is not unusable."""
        broken = {**copy.deepcopy(GOOD_AGENTS), "architect": entry(mcptools="yes")}
        write_modes(workspace, modes_json(agents=broken))
        token = token_of(workspace)

        save(body(GOOD_AGENTS), token)

        assert agent_modes.resolve_modes()["findings"] == []


# -- the conflict guard (D7) ----------------------------------------------------

class TestTheMtimeGuard:
    def test_a_stale_token_is_refused_carrying_both_numbers(self, workspace: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        stale = token_of(workspace)
        monkeypatch.setattr(agent_modes, "_file_stamp",
                            lambda path: (int(stale) + 5_000_000, 1234))

        error = refused(lambda: save(body(GOOD_AGENTS), stale),
                        writer.AgentModesSaveConflict)

        assert error.status_code == 409
        detail = error.detail_object
        assert detail["reason"] == "stale"
        assert detail["path"] == str(modes_path(workspace))
        assert detail["diskMtime"] == str(int(stale) + 5_000_000)
        assert detail["documentMtime"] == stale

    def test_the_bytes_the_user_put_them_in_stay_on_disk(self, workspace: Path):
        text = modes_json({"hand": entry(label="Hand-written", systemMessage="mine")},
                          default="hand")
        path = write_modes(workspace, text)
        token = token_of(workspace)
        path.write_text(modes_json({"other": entry(label="Someone else")},
                                   default="other"), encoding="utf-8")

        refused(lambda: save(body(GOOD_AGENTS), token), writer.AgentModesSaveConflict)

        assert path.read_text(encoding="utf-8") == modes_json(
            {"other": entry(label="Someone else")}, default="other")
        assert agent_modes.resolve_modes()["default"] == "other"

    def test_a_conflict_writes_no_backup_and_leaves_no_temp_file(self, workspace: Path,
                                                                 backups: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        token = token_of(workspace)
        os.utime(modes_path(workspace), ns=(int(token) + 1, int(token) + 1))

        refused(lambda: save(body(GOOD_AGENTS), token), writer.AgentModesSaveConflict)

        assert not backups.exists()
        assert folder_names(workspace) == [agent_modes.AGENT_MODES_FILENAME]

    def test_a_token_of_none_against_a_file_that_exists_is_a_conflict(
        self, workspace: Path
    ):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body(GOOD_AGENTS), None),
                        writer.AgentModesSaveConflict)

        assert error.detail_object["diskMtime"] is not None
        assert error.detail_object["documentMtime"] is None

    def test_a_matching_token_is_accepted(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        answer = save(body(GOOD_AGENTS), token_of(workspace))

        assert answer["saved"] is True
        assert answer["created"] is False

    def test_a_body_naming_a_token_that_is_not_a_number_is_refused(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        error = refused(lambda: save(body(GOOD_AGENTS), "not-a-timestamp"),
                        writer.AgentModesDocumentInvalid)

        assert error.findings[0]["message"].startswith("mtime:")

    def test_the_answer_of_a_save_carries_the_new_token(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = token_of(workspace)
        answer = save(body(GOOD_AGENTS), before)

        assert answer["source"]["mtime"] == token_of(workspace)
        assert answer["source"]["mtime"] != before
        assert save(body(GOOD_AGENTS), answer["source"]["mtime"])["saved"] is True


class TestExclusiveCreate:
    def test_no_file_means_the_file_is_created(self, workspace: Path):
        answer = save(body(GOOD_AGENTS), None)

        assert answer["created"] is True
        assert modes_path(workspace).is_file()
        assert agent_modes.resolve_modes()["source"]["exists"] is True

    def test_a_file_that_appeared_after_the_read_is_a_conflict_not_an_overwrite(
        self, workspace: Path
    ):
        someone_else = modes_json({"theirs": entry(label="Theirs")}, default="theirs")
        write_modes(workspace, someone_else)

        error = refused(lambda: save(body(GOOD_AGENTS), None),
                        writer.AgentModesSaveConflict)

        assert error.status_code == 409
        assert error.detail_object["documentMtime"] is None
        assert error.detail_object["diskMtime"] is not None
        assert modes_path(workspace).read_text(encoding="utf-8") == someone_else

    def test_the_reserved_folder_is_created_one_level_deep(self, workspace: Path):
        save(body(GOOD_AGENTS), None)

        folder = workspace / agent_modes.POLICY_DIRNAME
        assert folder.is_dir()
        assert (folder / agent_modes.AGENT_MODES_FILENAME).is_file()

    def test_a_missing_root_is_a_refusal_not_a_folder_chain(self, workspace: Path,
                                                            monkeypatch: pytest.MonkeyPatch):
        """One ``os.mkdir`` level: a root whose parent is missing is a symptom, not
        something to paper over by building the chain."""
        gone = workspace / "moved-away"
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(gone))
        agent_modes.invalidate_agent_modes_cache()

        error = refused(lambda: save(body(GOOD_AGENTS), None),
                        writer.AgentModesWriteRefused)

        assert error.status_code == 500
        assert "cannot be created" in error.detail
        assert not gone.exists()

    def test_a_created_file_needs_no_backup(self, workspace: Path, backups: Path):
        answer = save(body(GOOD_AGENTS), None)

        assert answer["backedUp"] is False
        assert not backups.exists()


# -- the backup (D8) ------------------------------------------------------------

class TestTheBackup:
    def test_a_save_backs_the_bytes_it_replaced_up(self, workspace: Path,
                                                   backups: Path):
        path = write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = path.read_bytes()

        answer = save(body(GOOD_AGENTS), token_of(workspace))

        assert answer["backedUp"] is True
        copies = editorfs.files.list_backups(path)
        assert len(copies) == 1
        assert Path(copies[0]).read_bytes() == before

    def test_a_second_save_keeps_two_copies(self, workspace: Path, backups: Path):
        path = write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        first = save(body(GOOD_AGENTS), token_of(workspace))
        save(body({"solo": entry(label="Solo")}, default="solo"),
             first["source"]["mtime"])

        copies = editorfs.files.list_backups(path)
        assert len(copies) == 2
        # Oldest first: what the hand-edited file held, then what the first save wrote.
        assert Path(copies[0]).read_bytes() == modes_json(agents=GOOD_AGENTS).encode("utf-8")
        assert b'"architect"' in Path(copies[1]).read_bytes()
        assert b'"solo"' in path.read_bytes()

    def test_the_backup_chain_is_keyed_on_the_modes_file_path(self, workspace: Path,
                                                              backups: Path):
        path = write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        save(body(GOOD_AGENTS), token_of(workspace))

        assert backups.is_dir()
        key = editorfs.files.backup_key(path)
        assert [p.name for p in backups.glob(f"{key}_*.orig")]

    def test_the_previous_document_is_recoverable_from_the_chain(self, workspace: Path):
        path = write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = path.read_bytes()
        save(body({"solo": entry(label="Solo")}, default="solo"), token_of(workspace))

        copy_of_before = Path(editorfs.files.list_backups(path)[0])
        path.write_bytes(copy_of_before.read_bytes())
        payload = agent_modes.resolve_modes()
        assert payload["findings"] == []
        assert [mode.id for mode in payload["modes"]][:-1] == [
            "chat", "code", "architect"
        ]


# -- the answer the save gives back (D9) ----------------------------------------

class TestTheSaveAnswer:
    def test_it_carries_the_verdict_in_the_readers_shape(self, workspace: Path):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        answer = save(body(GOOD_AGENTS), token_of(workspace))

        published = agent_modes.resolve_modes()
        assert answer["modes"] == [
            {"id": mode.id, "label": mode.label, "mcptools": mode.mcptools,
             "lsptools": mode.lsptools, "editable": mode.editable}
            for mode in published["modes"]
        ]
        assert answer["default"] == published["default"]
        assert answer["findings"] == [{"kind": f.kind, "message": f.message}
                                      for f in published["findings"]]
        assert answer["source"]["path"] == str(modes_path(workspace))
        assert answer["source"]["exists"] is True
        assert answer["source"]["loaded"] is True

    def test_the_app_owned_mode_is_in_the_answer_and_last(self, workspace: Path):
        answer = save(body(GOOD_AGENTS), None)

        assert answer["modes"][-1]["editable"] is False
        assert answer["modes"][-1]["id"] == agent_modes.RESERVED_MODE_IDS[0]

    def test_the_new_default_is_the_one_the_body_named(self, workspace: Path):
        answer = save(body(GOOD_AGENTS, default="architect"), None)

        assert answer["default"] == "architect"
        assert agent_modes.resolve_modes()["default"] == "architect"

    def test_the_cache_is_invalidated_by_the_save(self, workspace: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        agent_modes.resolve_modes()  # fill the cache with the old verdict
        calls: List[int] = []
        real_open = open

        def counting(*args: Any, **kwargs: Any):
            calls.append(1)
            return real_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", counting)
        save(body(GOOD_AGENTS), token_of(workspace))
        monkeypatch.undo()

        assert [mode.id for mode in agent_modes.resolve_modes()["modes"]][:-1] == [
            "chat", "code", "architect"
        ]


# -- the module's own boundaries ------------------------------------------------

class TestTheWriterIsStillCoreOnly:
    def test_it_imports_neither_the_api_layer_nor_editorfs(self):
        source = Path(writer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        offenders = [
            name for name in imported
            if name.split(".")[0] in {"fastapi", "src.api", "litellm", "llmkit"}
            or name.split(".")[-1] in {"editorfs", "mcphub", "lsphub", "termhub",
                                        "chat", "bench", "batch"}
        ]
        assert offenders == [], source

    def test_the_loader_gained_no_write_because_of_this_module(self):
        """The write half lives here, so ``agent_modes.py`` stays a reader."""
        loader = Path(agent_modes.__file__).read_text(encoding="utf-8")
        assert "agent_modes_write" not in loader
        assert "backup_for_write" not in loader

    def test_the_writer_names_no_editorfs_write_helper(self):
        """The backup is injected, so this file stays out of the caller scan."""
        source = Path(writer.__file__).read_text(encoding="utf-8")
        for name in ("write_file", "backup_for_write", "create_path",
                     "rename_path", "move_paths_batch", "trash_path"):
            assert name not in source

    def test_the_writer_reuses_the_loaders_validator_rather_than_a_second_one(self):
        source = Path(writer.__file__).read_text(encoding="utf-8")
        assert "_validated_document" in source
        assert "def _validate_body" not in source

    def test_the_exclusive_create_lives_in_exactly_one_function(self):
        tree = ast.parse(Path(writer.__file__).read_text(encoding="utf-8"))
        openers = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and "O_CREAT" in ast.dump(node)
        }
        assert openers == {"_create_exclusively"}

    def test_it_never_creates_a_folder_chain(self):
        """The seeder's rule, read off the calls rather than off the prose."""
        tree = ast.parse(Path(writer.__file__).read_text(encoding="utf-8"))
        made = [node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"makedirs", "mkdir"}]
        assert made == ["mkdir"]


# -- the real content, and the poisoned guard -----------------------------------

@pytest.fixture
def real_workspace(tmp_path: Path,
                   monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch workspace with **no stand-in**: the seeded words answer here."""
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    agent_modes.invalidate_agent_modes_cache()
    yield tmp_path
    agent_modes.invalidate_agent_modes_cache()


class TestTheSeededDocumentRoundTripsThroughTheWriter:
    def test_saving_the_seeded_bytes_back_changes_nothing(self, real_workspace: Path):
        write_modes(real_workspace, defaults.default_agent_modes_text())
        token = token_of(real_workspace)

        save(json.loads(defaults.default_agent_modes_text()), token)

        assert modes_path(real_workspace).read_text(encoding="utf-8") == (
            defaults.default_agent_modes_text()
        )
        assert agent_modes.resolve_modes()["findings"] == []

    def test_the_built_in_document_is_a_valid_body(self, real_workspace: Path):
        answer = save(copy.deepcopy(defaults.DEFAULT_AGENTS_DOCUMENT), None)

        assert answer["created"] is True
        assert answer["findings"] == []
        assert [mode["id"] for mode in answer["modes"]] == [
            "chat", "code", "architect", "todoBuilder"
        ]

    def test_a_persona_reset_to_the_builtin_one_round_trips(self, real_workspace: Path):
        write_modes(real_workspace, modes_json(
            {"code": entry(label="Rewritten", systemMessage="my own words")},
            default="code"))
        token = token_of(real_workspace)
        document = copy.deepcopy(defaults.DEFAULT_AGENTS_DOCUMENT)

        save(document, token)

        payload = agent_modes.resolve_modes()
        assert payload["findings"] == []
        mode = next(m for m in payload["modes"] if m.id == defaults.CODE_MODE_ID)
        assert mode.system_message == "\n".join(defaults.CODE_PERSONA)


class TestTheGuardCanFail:
    """A guard that cannot fail is not a guard (the isolation suite's style)."""

    def test_a_writer_that_drops_an_invalid_entry_instead_of_refusing_is_caught(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Poison the writer: ignore the validator and let the save through."""
        monkeypatch.setattr(writer, "_body_findings", lambda document, path: [])
        write_modes(workspace, modes_json(agents=GOOD_AGENTS))

        answer = save(body({"code": entry(mcptools="yes")}, default="code"),
                      token_of(workspace))

        assert answer["saved"] is True
        # The proof: the file the writer produced is one the reader drops an entry for.
        payload = agent_modes.resolve_modes()
        assert [f.message for f in payload["findings"]
                if f.kind == agent_modes.KIND_ERROR]

    def test_a_writer_that_skipped_the_backup_is_caught(self, workspace: Path,
                                                        backups: Path):
        path = write_modes(workspace, modes_json(agents=GOOD_AGENTS))
        before = path.read_bytes()

        save(body(GOOD_AGENTS), token_of(workspace))

        assert editorfs.files.list_backups(path)
        assert Path(editorfs.files.list_backups(path)[0]).read_bytes() == before
        assert backups.is_dir()
