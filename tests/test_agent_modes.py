"""am-01 — ``core/agent_modes.py``: the vocabulary, the loader, the cache.

Everything runs against a ``tmp_path`` workspace whose
``.nojae/agent-modes.json`` the test writes itself, so the file, the reserved
folder and the cache key are real paths on disk — and nothing is wired to a route
or a model yet (am-04 and am-06 do that), so every assertion here is on the
loader's two accessors: :func:`resolve_modes` and :func:`resolve_mode`.

The fixtures install a stand-in for am-02's ``agent_modes_defaults`` behind
``agent_modes.DEFAULTS_MODULE_NAME`` — the attribute that seam exists for. The
stand-in has the shape am-02 specifies (three personas, one reserved mode, a
document whose ``default`` is ``code``), because what these tests pin is the
machinery: the rules, the findings, the fallbacks and the cache. ``TestDefaults``
at the end drops the stand-in and runs the same loader against the **real**
content module — the round-trip guard and the prose guards, which are the tests
that cannot mean anything against a stand-in.
"""

import ast
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

from src.core import agent_modes, batch
from src.core import chat as chat_core
from src.core import agent_modes_defaults as defaults
from src.core.models import DEFAULT_LLM_SYSPROMPT
from src.core.agent_modes import AgentMode


def modes_json(
    agents: Optional[Dict[str, Any]] = None,
    *,
    version: Any = agent_modes.AGENT_MODES_VERSION,
    default: Any = "code",
    readme: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Build one modes document as JSON.

    ``agents`` goes through untouched rather than being rebuilt: half of the
    malformed-file suite hands it entries that are not objects at all, which is
    exactly what the loader has to notice. A helper that vouched for the entries
    inside it could not express the cases it exists to feed.
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
        document["agents"] = agents
    return json.dumps(document, indent=2)


def entry(**fields: Any) -> Dict[str, Any]:
    """One mode entry, spelled with its JSON keys."""
    return fields


GOOD_AGENTS: Dict[str, Any] = {
    "chat": entry(label="Chat", mcptools=False, lsptools=False,
                  systemMessage="never touch the workspace"),
    "code": entry(label="Code", mcptools=True, lsptools=True,
                  systemMessage="implement and prove it"),
    "architect": entry(label="Architect", mcptools=True, lsptools=False,
                       systemMessage="write a spec"),
}


# -- the stand-in for am-02's content module -----------------------------------

STANDIN_DOCUMENT: Dict[str, Any] = {
    "version": agent_modes.AGENT_MODES_VERSION,
    "readme": ["the vocabulary explains itself from inside the file"],
    "default": "code",
    "agents": {
        "chat": entry(label="Chat", mcptools=False, lsptools=False,
                      systemMessage=["Answer; do not touch the workspace."]),
        "code": entry(label="Code", mcptools=True, lsptools=True,
                      systemMessage="Implement, and prove it."),
        "architect": entry(label="Architect", mcptools=True, lsptools=False),
    },
}

STANDIN_TODO_BUILDER = AgentMode(
    id="todoBuilder",
    label="Todo builder",
    system_message="Write the lot; never implement it.",
    mcptools=True,
    lsptools=False,
    editable=False,
    instructions=("the lot conventions, rendered from core/batch.py",),
)


def install_standin_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a stand-in ``agent_modes_defaults`` behind the module's own seam."""
    module = types.ModuleType(agent_modes.DEFAULTS_MODULE_NAME)
    module.DEFAULT_AGENTS_DOCUMENT = STANDIN_DOCUMENT  # type: ignore[attr-defined]

    def default_modes() -> List[AgentMode]:
        # am-02's rule for the fallback: it runs *this* module's validator
        # rather than hand-building AgentModes, so the seeded document and the
        # fallback cannot drift apart.
        modes, _, findings, usable = agent_modes._validated_document(
            json.loads(json.dumps(STANDIN_DOCUMENT)), "<stand-in>"
        )
        assert usable and not findings, findings
        return modes

    module.default_modes = default_modes  # type: ignore[attr-defined]
    module.todo_builder_mode = lambda: STANDIN_TODO_BUILDER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, agent_modes.DEFAULTS_MODULE_NAME, module)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch workspace, set as the active location, with am-02 stood in."""
    install_standin_defaults(monkeypatch)
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    agent_modes.invalidate_agent_modes_cache()
    yield tmp_path
    agent_modes.invalidate_agent_modes_cache()


def write_modes(root: Path, text: str) -> Path:
    folder = root / agent_modes.POLICY_DIRNAME
    folder.mkdir(exist_ok=True)
    path = folder / agent_modes.AGENT_MODES_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def resolved() -> Dict[str, Any]:
    return agent_modes.resolve_modes()


def ids() -> List[str]:
    return [mode.id for mode in resolved()["modes"]]


def errors() -> List[str]:
    return [f.message for f in resolved()["findings"]
            if f.kind == agent_modes.KIND_ERROR]


def notes() -> List[str]:
    return [f.message for f in resolved()["findings"]
            if f.kind == agent_modes.KIND_NOTE]


def mode_of(mode_id: str) -> AgentMode:
    for mode in resolved()["modes"]:
        if mode.id == mode_id:
            return mode
    raise AssertionError(f"no mode {mode_id!r} in {ids()}")


class TestAbsentFile:
    """D11's ordinary case: no file is not a problem, it is the unseeded state."""

    def test_no_findings_and_the_built_in_modes_answer(self, workspace: Path):
        payload = resolved()
        assert payload["findings"] == []
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert payload["default"] == "code"

    def test_source_says_absent_and_not_loaded(self, workspace: Path):
        source = resolved()["source"]
        assert source["exists"] is False
        assert source["loaded"] is False
        assert source["version"] is None
        assert source["path"] == str(
            workspace / agent_modes.POLICY_DIRNAME / agent_modes.AGENT_MODES_FILENAME
        )

    def test_the_path_follows_the_active_workspace(self, workspace: Path,
                                                   monkeypatch):
        """The lookup starts at the *active* root, so a switch moves it."""
        assert agent_modes.agent_modes_path() == str(
            workspace / agent_modes.POLICY_DIRNAME / agent_modes.AGENT_MODES_FILENAME
        )
        other = workspace.parent / "other-root"
        other.mkdir()
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(other))
        assert agent_modes.agent_modes_path().startswith(str(other))

    def test_the_reserved_mode_is_last_and_not_editable(self, workspace: Path):
        builder = mode_of("todoBuilder")
        assert builder is STANDIN_TODO_BUILDER
        assert builder.editable is False
        assert builder.instructions  # only the app-owned mode ever carries blocks
        assert all(mode.editable for mode in resolved()["modes"][:-1])

    def test_the_built_in_fallback_modes_are_the_validated_document(self,
                                                                    workspace: Path):
        """am-02's rule, asserted through the seam: no hand-built second copy."""
        fallback = agent_modes._fallback_modes()
        assert [mode.id for mode in fallback] == ["chat", "code", "architect"]
        assert [mode.label for mode in fallback] == ["Chat", "Code", "Architect"]
        # ``chat`` really is the closed mode, and the document's own persona text
        # is what reaches the model — not the validator inventing one.
        assert fallback[0].system_message == "Answer; do not touch the workspace."
        assert fallback[0].mcptools is False and fallback[0].lsptools is False


class TestWellFormedFile:
    def test_every_declared_mode_survives_with_no_findings(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS))
        assert resolved()["findings"] == []
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert resolved()["source"] == {
            "path": str(agent_modes.agent_modes_path()),
            "exists": True,
            "loaded": True,
            "version": agent_modes.AGENT_MODES_VERSION,
        }

    def test_the_file_order_is_the_returned_order(self, workspace: Path):
        """A dict's insertion order is the dropdown's order — pinned, not trusted."""
        shuffled = {key: GOOD_AGENTS[key] for key in ("architect", "chat", "code")}
        write_modes(workspace, modes_json(shuffled, default="architect"))
        assert ids() == ["architect", "chat", "code", "todoBuilder"]
        assert resolved()["default"] == "architect"

    def test_a_declared_mode_replaces_the_built_in_one(self, workspace: Path):
        write_modes(workspace, modes_json({
            "code": entry(label="My coder", mcptools=False, lsptools=False,
                          systemMessage="only my persona"),
        }))
        code = mode_of("code")
        assert (code.label, code.mcptools, code.lsptools) == ("My coder", False, False)
        assert code.system_message == "only my persona"
        assert resolved()["findings"] == []
        assert ids() == ["code", "todoBuilder"]

    def test_the_built_in_document_round_trips_through_the_loader(self, workspace: Path):
        """The written file and the read one must agree, not agree by luck.

        The ``policy_seed.py`` argument, reused: the bytes am-03 will write and
        the fallback this module answers with come from one object, so rendering
        that object as JSON and reading it back has to produce **no findings at
        all** and the same modes. am-02 repeats this against its real document —
        here it is the stand-in's, and it catches the same class of bug: a key
        the seeder writes that the validator does not accept.
        """
        rendered = json.dumps(STANDIN_DOCUMENT, indent=2) + "\n"
        write_modes(workspace, rendered)
        payload = resolved()
        assert payload["findings"] == []
        assert payload["source"]["loaded"] is True
        assert payload["default"] == STANDIN_DOCUMENT["default"]
        declared = [mode.id for mode in payload["modes"]
                    if mode.id not in agent_modes.RESERVED_MODE_IDS]
        assert declared == list(STANDIN_DOCUMENT["agents"])
        # The two paths are one answer: what the file says and what a broken
        # file would be replaced by are the *same objects*, so they cannot drift.
        assert [mode for mode in payload["modes"] if mode.editable] == (
            agent_modes._fallback_modes())
        for mode_id, fields in STANDIN_DOCUMENT["agents"].items():
            mode = mode_of(mode_id)
            assert mode.label == fields.get("label", mode_id)
            assert mode.mcptools is fields.get("mcptools", False)
            assert mode.lsptools is fields.get("lsptools", False)
            persona = fields.get("systemMessage", "")
            expected = persona if isinstance(persona, str) else "\n".join(persona)
            assert mode.system_message == expected

    def test_readme_of_either_form_and_a_reserved_default_are_accepted(self,
                                                                       workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, readme=["one", "two"]))
        assert resolved()["findings"] == []
        write_modes(workspace, modes_json(GOOD_AGENTS, readme="one string"))
        assert resolved()["findings"] == []
        write_modes(workspace, modes_json(GOOD_AGENTS, readme=42))
        assert resolved()["findings"] == []

    def test_the_agent_mode_fields_are_the_ones_the_route_needs(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS))
        code = mode_of("code")
        assert (code.id, code.label, code.system_message, code.mcptools,
                code.lsptools, code.editable, code.instructions) == (
            "code", "Code", "implement and prove it", True, True, True, ())


class TestFileLevelFailureFallsBackToDefaults:
    """The rules that question the vocabulary, so nothing in the file stands."""

    def test_a_syntax_error_gives_the_defaults_and_one_error(self, workspace: Path):
        write_modes(workspace, '{"version": 1, "agents": {"code": {')
        payload = resolved()
        assert payload["source"]["loaded"] is False
        assert payload["source"]["exists"] is True
        assert len(payload["findings"]) == 1
        assert "not valid JSON" in errors()[0]
        assert agent_modes.agent_modes_path() in errors()[0]
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert payload["default"] == "code"

    @pytest.mark.parametrize("body", ['"chat"', "[1, 2]", "null", "7"])
    def test_a_root_that_is_not_an_object_refuses_the_file(self, workspace: Path,
                                                           body: str):
        write_modes(workspace, body)
        assert resolved()["source"]["loaded"] is False
        assert any("not an object" in message for message in errors())
        assert ids() == ["chat", "code", "architect", "todoBuilder"]

    def test_agents_that_is_not_an_object_refuses_the_file(self, workspace: Path):
        write_modes(workspace, modes_json(None, extra={"agents": [1, 2]}))
        assert resolved()["source"]["loaded"] is False
        assert any('"agents" is a list' in message for message in errors())
        assert ids() == ["chat", "code", "architect", "todoBuilder"]

    def test_a_missing_version_refuses_the_file(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, version=None))
        assert resolved()["source"]["loaded"] is False
        assert any('no "version" key' in message and "expected 1" in message
                   for message in errors())
        assert ids() == ["chat", "code", "architect", "todoBuilder"]

    def test_another_version_refuses_the_file(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, version=2))
        assert resolved()["source"]["loaded"] is False
        assert any('"version" is 2' in message and "version 1 only" in message
                   for message in errors())
        # Not one mode from the file survives: the shape they are written in is
        # the thing that was in question.
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert mode_of("chat").system_message == "Answer; do not touch the workspace."

    def test_a_version_written_as_a_boolean_is_not_a_number(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, version=True))
        assert resolved()["source"]["loaded"] is False
        assert any("not a number" in message for message in errors())

    def test_a_folder_where_the_file_belongs_names_the_path(self, workspace: Path):
        folder = workspace / agent_modes.POLICY_DIRNAME
        folder.mkdir()
        (folder / agent_modes.AGENT_MODES_FILENAME).mkdir()
        payload = resolved()
        assert payload["source"]["loaded"] is False
        # Something *is* standing there, so ``exists`` must not send the user
        # hunting for a missing file when the finding says it is a folder.
        assert payload["source"]["exists"] is True
        assert any("is a folder, not a file" in message for message in errors())
        assert ids() == ["chat", "code", "architect", "todoBuilder"]

    def test_non_utf8_bytes_are_a_finding_not_a_crash(self, workspace: Path):
        path = workspace / agent_modes.POLICY_DIRNAME
        path.mkdir()
        (path / agent_modes.AGENT_MODES_FILENAME).write_bytes(
            b'{"version": 1, "agents": {"cod\xc3": {}}'
        )
        assert any("is not UTF-8 text" in message for message in errors())
        assert resolved()["source"]["loaded"] is False


class TestEntryLevelFailureDropsOneMode:
    """One bad entry costs the user that mode, never the other three."""

    def test_a_bad_entry_leaves_its_siblings_and_the_file_usable(self, workspace: Path):
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(mcptools=1, systemMessage="implement")
        write_modes(workspace, modes_json(agents))
        assert ids() == ["chat", "architect", "todoBuilder"]
        assert resolved()["source"]["loaded"] is True

    def test_an_entry_is_dropped_as_a_unit_not_field_by_field(self, workspace: Path):
        """The mode is *absent*, not present-with-defaults."""
        agents = dict(GOOD_AGENTS)
        # The two flags are perfectly good here; only the persona is broken.
        agents["code"] = entry(mcptools=True, lsptools=True, systemMessage=["ok", 3])
        write_modes(workspace, modes_json(agents, default="chat"))
        assert "code" not in ids()
        assert all(mode.id != "code" for mode in resolved()["modes"])

    def test_an_unknown_mode_key_names_the_pointer_and_the_vocabulary(self,
                                                                      workspace: Path):
        agents = dict(GOOD_AGENTS)
        agents["architect"] = entry(mcptool=True, systemMessage="spec")
        write_modes(workspace, modes_json(agents, default="chat"))
        message = next(m for m in errors() if "mcptool" in m)
        assert message.startswith("agents.architect.mcptool:")
        assert "unknown key" in message
        assert "label, mcptools, lsptools, systemMessage" in message
        assert "architect" not in ids()

    def test_an_unknown_top_level_key_is_an_error(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, extra={"agent": {}}))
        assert any("unknown top-level key 'agent'" in message for message in errors())
        # D12's own example: a top-level typo does not void the keys it sits next to.
        assert ids() == ["chat", "code", "architect", "todoBuilder"]

    @pytest.mark.parametrize("value", [1, 0, "yes", "false", [], {}, None])
    def test_a_tool_flag_that_is_not_a_boolean_drops_the_mode(self, workspace: Path,
                                                              value: Any):
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(mcptools=value, systemMessage="implement")
        write_modes(workspace, modes_json(agents, default="chat"))
        assert "code" not in ids()
        message = next(m for m in errors() if m.startswith("agents.code.mcptools:"))
        assert "not a boolean" in message

    def test_a_missing_tool_flag_is_false_and_not_a_finding(self, workspace: Path):
        write_modes(workspace, modes_json(
            {"plain": entry(systemMessage="no flags at all")}, default="plain",
        ))
        plain = mode_of("plain")
        assert (plain.mcptools, plain.lsptools) == (False, False)
        assert resolved()["findings"] == []

    @pytest.mark.parametrize("shape", [
        {"a": 1}, [1, 2], ["ok", "  ", "fine"], True, 42, [], ["ok", ""],
    ])
    def test_a_system_message_that_is_neither_form_drops_the_mode(self, workspace: Path,
                                                                  shape: Any):
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(mcptools=True, systemMessage=shape)
        write_modes(workspace, modes_json(agents, default="chat"))
        assert "code" not in ids()
        assert any(m.startswith("agents.code.systemMessage:") for m in errors())

    def test_an_oversized_system_message_drops_the_mode(self, workspace: Path):
        big = "x" * (agent_modes.MAX_MODE_SYSTEM_MESSAGE_BYTES + 1)
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(mcptools=True, systemMessage=big)
        write_modes(workspace, modes_json(agents, default="chat"))
        assert "code" not in ids()
        message = next(m for m in errors() if m.startswith("agents.code.systemMessage:"))
        assert "16 KiB" in message
        assert "every turn" in message

    def test_the_byte_cap_is_patchable_and_counts_utf8_bytes(self, workspace: Path,
                                                             monkeypatch):
        # The cap is a module attribute and is read per call, so patching it
        # patches the rule everywhere — including for the built-in fallback
        # personas, which is why this one sits above the stand-in's 35 bytes.
        monkeypatch.setattr(agent_modes, "MAX_MODE_SYSTEM_MESSAGE_BYTES", 40)
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(systemMessage="é" * 21)  # 21 characters, 42 bytes
        write_modes(workspace, modes_json(agents, default="chat"))
        assert "code" not in ids()
        message = next(m for m in errors() if m.startswith("agents.code.systemMessage:"))
        assert "is 42 bytes, over the 40 bytes cap" in message
        # Under the same cap, the persona that fits is untouched.
        assert mode_of("chat").system_message == "never touch the workspace"
        assert "chat" in ids() and "architect" in ids()

    def test_a_label_that_is_not_a_string_drops_the_mode(self, workspace: Path):
        for bad in (["Chat"], 7, None, True):
            agents = dict(GOOD_AGENTS)
            agents["chat"] = entry(label=bad, mcptools=False)
            write_modes(workspace, modes_json(agents, default="code"))
            agent_modes.invalidate_agent_modes_cache()
            assert "chat" not in ids(), bad
            assert any(m.startswith("agents.chat.label:") for m in errors())

    def test_a_missing_label_falls_back_to_the_id_with_no_finding(self, workspace: Path):
        write_modes(workspace, modes_json(
            {"helper": entry(mcptools=True)}, default="helper",
        ))
        assert mode_of("helper").label == "helper"
        assert resolved()["findings"] == []

    def test_a_blank_label_falls_back_to_the_id_too(self, workspace: Path):
        write_modes(workspace, modes_json(
            {"helper": entry(label="   ", mcptools=True)}, default="helper",
        ))
        assert mode_of("helper").label == "helper"
        assert resolved()["findings"] == []

    def test_an_over_long_label_is_trimmed_not_refused(self, workspace: Path):
        write_modes(workspace, modes_json(
            {"helper": entry(label="h" * 200, mcptools=True)}, default="helper",
        ))
        assert mode_of("helper").label == "h" * agent_modes.MAX_MODE_LABEL_CHARS
        assert resolved()["findings"] == []

    @pytest.mark.parametrize("mode_id", ["Chat", "9 lives", "a b", "_x", "mode.name", "élan"])
    def test_a_mode_id_outside_the_pattern_is_dropped(self, workspace: Path,
                                                      mode_id: str):
        agents = dict(GOOD_AGENTS)
        agents[mode_id] = entry(mcptools=True)
        write_modes(workspace, modes_json(agents, default="code"))
        assert mode_id not in ids()
        message = next(m for m in errors() if m.startswith(f"agents.{mode_id}:"))
        assert "mode id" in message
        assert agent_modes.MODE_ID_PATTERN in message

    def test_a_long_mode_id_is_dropped_by_the_length_cap(self, workspace: Path,
                                                         monkeypatch):
        monkeypatch.setattr(agent_modes, "MAX_MODE_ID_CHARS", 6)
        agents = dict(GOOD_AGENTS)
        agents["seventh"] = entry(mcptools=True)
        write_modes(workspace, modes_json(agents))
        assert "seventh" not in ids()
        assert any("longer than 6 characters" in message for message in errors())

    def test_the_id_pattern_is_patchable(self, workspace: Path, monkeypatch):
        monkeypatch.setattr(agent_modes, "MODE_ID_PATTERN", r"^[A-Z][A-Za-z0-9_-]*$")
        write_modes(workspace, modes_json({"Review": entry(mcptools=True)},
                                          default="Review"))
        assert ids() == ["Review", "todoBuilder"]
        assert resolved()["findings"] == []

    def test_an_entry_that_is_not_an_object_names_the_vocabulary(self, workspace: Path):
        agents = dict(GOOD_AGENTS)
        agents["code"] = "implement it"
        write_modes(workspace, modes_json(agents, default="chat"))
        message = next(m for m in errors() if m.startswith("agents.code:"))
        assert "is a string, not an object" in message
        assert "label, mcptools, lsptools, systemMessage" in message
        assert "code" not in ids()

    def test_every_problem_in_one_entry_is_reported_at_once(self, workspace: Path):
        """One pass through the findings fixes the file."""
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(mcptool=1, lsptools="yes", systemMessage=["ok", 5])
        write_modes(workspace, modes_json(agents, default="chat"))
        pointers = [m.split(":")[0] for m in errors() if m.startswith("agents.code")]
        assert pointers == [
            "agents.code.mcptool",
            "agents.code.lsptools",
            "agents.code.systemMessage",
        ]


class TestDefaultKey:
    def test_a_missing_default_is_an_error_and_the_built_in_one_answers(
        self, workspace: Path
    ):
        write_modes(workspace, modes_json(GOOD_AGENTS, default=None))
        assert any('no "default" key' in message for message in errors())
        assert resolved()["default"] == "code"
        assert resolved()["default"] in ids()
        assert ids() == ["chat", "code", "architect", "todoBuilder"]

    def test_a_default_that_is_not_a_string_is_an_error(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, default=["code"]))
        assert any('"default" is a list' in message for message in errors())
        assert resolved()["default"] == "code"

    def test_a_default_naming_a_dropped_mode_is_refused_but_the_file_stands(
        self, workspace: Path
    ):
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(mcptools=1)  # the mode the default names is broken
        write_modes(workspace, modes_json(agents, default="code"))
        assert "code" not in ids()
        assert resolved()["source"]["loaded"] is True
        assert any('"default" is "code"' in message for message in errors())
        # The answer is a mode the dropdown can actually offer — here that is the
        # built-in default again, which only survives because the fallback modes
        # are used: the file's own list no longer holds ``code``.
        assert resolved()["default"] in ids()

    def test_a_default_naming_the_reserved_mode_is_refused_and_repaired(self,
                                                                        workspace: Path):
        """The default names a mode this file declares; the app-owned ones are not
        settable here (D9), so this is the same mistake as naming nothing."""
        write_modes(workspace, modes_json(GOOD_AGENTS, default="todoBuilder"))
        assert any('"default" is "todoBuilder"' in message for message in errors())
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert resolved()["default"] == "code"
        mode, _ = agent_modes.resolve_mode(None)
        assert mode is not None and mode.id == "code"

    def test_the_answered_default_is_always_selectable(self, workspace: Path):
        """Every declared mode refused: the defaults answer, and so does theirs."""
        write_modes(workspace, modes_json({"Chat": entry(mcptools=True)},
                                          default="Chat"))
        payload = resolved()
        assert payload["default"] in ids()
        assert agent_modes.resolve_mode(None)[0] is not None

    def test_the_default_is_a_value_not_a_reordering(self, workspace: Path):
        write_modes(workspace, modes_json(
            {"zeta": entry(mcptools=True), "alpha": entry(mcptools=True)},
            default="alpha",
        ))
        assert ids() == ["zeta", "alpha", "todoBuilder"]
        assert resolved()["default"] == "alpha"


class TestReservedMode:
    def test_a_file_that_declares_it_gets_a_note_and_the_app_definition(self,
                                                                        workspace: Path):
        agents = dict(GOOD_AGENTS)
        agents["todoBuilder"] = entry(label="Mine", mcptools=False, lsptools=False,
                                      systemMessage="my own lot rules")
        write_modes(workspace, modes_json(agents))
        assert len(notes()) == 1
        assert notes()[0].startswith("agents.todoBuilder:")
        assert "defined by benchmarkit and cannot be set here" in notes()[0]
        assert errors() == []
        builder = mode_of("todoBuilder")
        assert builder is STANDIN_TODO_BUILDER
        assert builder.label == "Todo builder"
        assert builder.mcptools is True
        assert ids().count("todoBuilder") == 1
        assert ids()[-1] == "todoBuilder"

    def test_the_note_keeps_the_other_modes_and_the_file_loaded(self, workspace: Path):
        """D11 again: the fix is to delete the entry, and it must not cost the rest."""
        agents = dict(GOOD_AGENTS)
        agents["todoBuilder"] = entry(mcptools=1, label="Mine")
        write_modes(workspace, modes_json(agents))
        assert resolved()["source"]["loaded"] is True
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert len(notes()) == 1

    def test_its_presence_does_not_depend_on_the_file(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS))
        assert ids()[-1] == "todoBuilder"
        write_modes(workspace, modes_json({"solo": entry(mcptools=True)}, default="solo"))
        assert ids()[-1] == "todoBuilder"
        os.remove(agent_modes.agent_modes_path())
        agent_modes.invalidate_agent_modes_cache()
        assert ids()[-1] == "todoBuilder"

    def test_max_modes_does_not_count_it(self, workspace: Path, monkeypatch):
        monkeypatch.setattr(agent_modes, "MAX_MODES", 1)
        write_modes(workspace, modes_json(GOOD_AGENTS, default="chat"))
        assert ids() == ["chat", "todoBuilder"]
        assert any("declares more than the 1 mode(s)" in message for message in errors())

    def test_the_overflow_is_dropped_in_file_order(self, workspace: Path, monkeypatch):
        monkeypatch.setattr(agent_modes, "MAX_MODES", 2)
        agents = {f"m{i}": entry(mcptools=True) for i in range(5)}
        write_modes(workspace, modes_json(agents, default="m0"))
        assert ids() == ["m0", "m1", "todoBuilder"]
        message = next(m for m in errors() if "declares more than the 2 mode(s)" in m)
        assert '"m2", "m3", "m4"' in message


class TestSystemMessageForms:
    def test_a_string_and_a_list_of_lines_agree_byte_for_byte(self, workspace: Path):
        lines = ["You are working in Code mode.", "- Use the tools.", "- Prove it."]
        write_modes(workspace, modes_json(
            {"code": entry(mcptools=True, systemMessage=lines)}, default="code"))
        list_text = mode_of("code").system_message
        agent_modes.invalidate_agent_modes_cache()
        write_modes(workspace, modes_json(
            {"code": entry(mcptools=True, systemMessage="\n".join(lines))},
            default="code"))
        agent_modes.invalidate_agent_modes_cache()
        string_text = mode_of("code").system_message
        assert list_text.encode("utf-8") == string_text.encode("utf-8")
        assert list_text == "\n".join(lines)

    def test_an_empty_or_absent_persona_means_the_configured_prompt(self,
                                                                    workspace: Path):
        write_modes(workspace, modes_json({
            "blank": entry(label="Blank", mcptools=True, systemMessage=""),
            "nullish": entry(label="Null", mcptools=True, systemMessage=None),
            "absent": entry(label="Absent", mcptools=True),
        }, default="blank"))
        assert all(mode_of(key).system_message == ""
                   for key in ("blank", "nullish", "absent"))
        assert resolved()["findings"] == []

    def test_the_persona_text_is_not_stripped_or_rewritten(self, workspace: Path):
        persona = "  Indented first line.\n\n  A blank line in between.  "
        write_modes(workspace, modes_json(
            {"code": entry(systemMessage=persona)}, default="code"))
        assert mode_of("code").system_message == persona


class TestResolveMode:
    def test_none_answers_the_file_default(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, default="architect"))
        mode, findings = agent_modes.resolve_mode(None)
        assert mode is not None and mode.id == "architect"
        assert findings == []

    @pytest.mark.parametrize("wanted", ["chat", "code", "architect", "todoBuilder"])
    def test_every_listed_mode_is_reachable_by_id(self, workspace: Path, wanted: str):
        mode, _ = agent_modes.resolve_mode(wanted)
        assert mode is not None
        assert mode.id == wanted

    def test_an_unknown_id_is_none_and_raises_nothing(self, workspace: Path):
        mode, findings = agent_modes.resolve_mode("nope")
        assert mode is None
        assert findings == []

    def test_no_fall_back_to_the_default_for_an_unknown_id(self, workspace: Path):
        """D14 lives with the caller: this function must not supply the answer."""
        write_modes(workspace, modes_json(GOOD_AGENTS, default="code"))
        mode, _ = agent_modes.resolve_mode("nope")
        assert mode is None
        mode, _ = agent_modes.resolve_mode("TODO")
        assert mode is None

    def test_an_unknown_id_stays_unknown_when_it_was_dropped_from_the_file(
        self, workspace: Path
    ):
        agents = dict(GOOD_AGENTS)
        agents["code"] = entry(mcptools=1)
        write_modes(workspace, modes_json(agents, default="chat"))
        mode, findings = agent_modes.resolve_mode("code")
        assert mode is None
        # The caller answering 422 also gets the findings about *why* the mode the
        # user is asking for is not in the list.
        assert any(m.startswith("agents.code.mcptools:") for m in
                   [f.message for f in findings])

    def test_the_reserved_mode_is_reachable(self, workspace: Path):
        mode, _ = agent_modes.resolve_mode("todoBuilder")
        assert mode is not None
        assert mode is STANDIN_TODO_BUILDER
        assert mode.instructions

    def test_the_findings_about_the_file_ride_along(self, workspace: Path):
        write_modes(workspace, '{"version": 1, "agents": {')
        mode, findings = agent_modes.resolve_mode("code")
        assert mode is not None and mode.id == "code"
        assert any("not valid JSON" in finding.message for finding in findings)

    def test_whitespace_and_empty_string_mean_the_default(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS, default="chat"))
        for raw in ("", "   ", None):
            mode, _ = agent_modes.resolve_mode(raw)
            assert mode is not None and mode.id == "chat"


class TestCache:
    """One read per changed file: every request asks, and must not re-read."""

    def test_an_unchanged_file_is_read_once_per_workspace(self, workspace: Path,
                                                          monkeypatch):
        write_modes(workspace, modes_json(GOOD_AGENTS))
        agent_modes.invalidate_agent_modes_cache()
        reads: List[str] = []
        real_open = open

        def counting(file: Any, *args: Any, **kwargs: Any):
            if str(file).endswith(".json"):
                reads.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(agent_modes, "open", counting, raising=False)
        for _ in range(5):
            assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert reads == [agent_modes.agent_modes_path()]

    def test_an_edit_at_the_same_size_with_a_new_mtime_is_seen(self, workspace: Path):
        """Same length, different modes — a size-only cache would miss it."""
        path = write_modes(workspace, modes_json(
            {"code": entry(label="One", mcptools=True)}, default="code"))
        assert mode_of("code").label == "One"
        second = modes_json({"code": entry(label="Two", mcptools=True)}, default="code")
        path.write_text(second, encoding="utf-8")
        assert len(second) == len(
            modes_json({"code": entry(label="One", mcptools=True)}, default="code")
        )
        future = path.stat().st_atime_ns + 1_000_000_000
        os.utime(path, ns=(future, future))
        assert mode_of("code").label == "Two"

    def test_a_second_root_gets_its_own_modes_not_the_firsts(self, workspace: Path,
                                                             tmp_path_factory,
                                                             monkeypatch):
        """The regression the root in the cache key exists to prevent.

        The two files are made to **collide on the stamp** — ``"alpha"`` and
        ``"betab"`` give them the same size, and the second file's ``mtime_ns``
        is set to the first's — so a cache keyed on ``(mtime, size)`` alone has
        nothing left to tell them apart.
        """
        first = write_modes(workspace, modes_json(
            {"alpha": entry(label="Alpha", mcptools=True)}, default="alpha"))
        assert ids() == ["alpha", "todoBuilder"]

        second_root: Path = tmp_path_factory.mktemp("other-root")
        stamp = (first.stat().st_mtime_ns, first.stat().st_size)
        monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(second_root))
        second_file = write_modes(second_root, modes_json(
            {"betab": entry(label="Betab", mcptools=True)}, default="betab"))
        assert second_file.stat().st_size == stamp[1]
        os.utime(second_file, ns=(stamp[0], stamp[0]))
        assert (second_file.stat().st_mtime_ns, second_file.stat().st_size) == stamp
        assert ids() == ["betab", "todoBuilder"]

    def test_a_patch_of_the_filename_invalidates_rather_than_answers_stale(
        self, workspace: Path, monkeypatch
    ):
        write_modes(workspace, modes_json(
            {"only": entry(label="Mine", mcptools=True)}, default="only"))
        assert ids() == ["only", "todoBuilder"]
        monkeypatch.setattr(agent_modes, "AGENT_MODES_FILENAME", "modes-other.json")
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert resolved()["source"]["exists"] is False
        assert resolved()["findings"] == []

    def test_an_absent_file_is_cached_too(self, workspace: Path, monkeypatch):
        """One ``stat`` per call is the whole cost of an unseeded workspace."""
        stats: List[str] = []
        real_stat = os.stat

        def counting(path: Any, *args: Any, **kwargs: Any):
            if str(path).endswith(".json"):
                stats.append(str(path))
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(agent_modes.os, "stat", counting)
        for _ in range(4):
            agent_modes.resolve_modes()
        assert stats == [agent_modes.agent_modes_path()] * 4

    def test_the_cache_is_dropped_whole_past_its_bound(self, workspace: Path,
                                                       tmp_path_factory, monkeypatch):
        monkeypatch.setattr(agent_modes, "_CACHE_MAX_ROOTS", 2)
        for index in range(4):
            root = workspace if index == 0 else tmp_path_factory.mktemp(f"root{index}")
            monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
            agent_modes.resolve_modes()
            assert len(agent_modes._CACHE) <= 2
        # The last root still answers correctly after the drops: the bound costs
        # a re-read, never an answer.
        assert ids() == ["chat", "code", "architect", "todoBuilder"]

    def test_invalidation_forgets_every_workspace(self, workspace: Path):
        write_modes(workspace, modes_json(GOOD_AGENTS))
        agent_modes.resolve_modes()
        assert agent_modes._CACHE
        agent_modes.invalidate_agent_modes_cache()
        assert agent_modes._CACHE == {}

    def test_a_file_deleted_after_a_read_answers_the_defaults(self, workspace: Path):
        write_modes(workspace, modes_json({"only": entry(mcptools=True)},
                                          default="only"))
        assert ids() == ["only", "todoBuilder"]
        os.remove(agent_modes.agent_modes_path())
        assert ids() == ["chat", "code", "architect", "todoBuilder"]
        assert resolved()["source"]["exists"] is False


class TestTheReaderIsAlone:
    """The module's own boundaries, pinned rather than asserted in prose.

    The modes file has exactly **one** reader in the app process: there is no
    connector-side twin of it (unlike the path policy, whose parser lives in
    ``mcp_connectors`` because the two stdio connectors answer that question in
    their own processes). A mode changes only what the model is *told*, and every
    connector receives its instructions through the prompt it is called with — so
    a second reader would be a second answer, not a symmetry.
    """

    def test_no_connector_reads_the_modes_file(self):
        connectors = Path(agent_modes.__file__).resolve().parents[2] / "mcp_connectors"
        offenders = [
            path.name
            for path in connectors.glob("*.py")
            if "agent_modes" in path.read_text(encoding="utf-8")
            or "agent-modes" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_the_loader_imports_no_transport_no_model_and_no_hub(self):
        """Pure core: FastAPI, litellm and the hubs belong to other tasks."""
        tree = ast.parse(Path(agent_modes.__file__).read_text(encoding="utf-8"))
        forbidden = ("fastapi", "litellm", "src.api", "mcphub", "lsphub", "termhub",
                     "editorfs", "chat", "bench", "batch")
        offenders: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    alias.name for alias in node.names
                    if alias.name.split(".")[-1] in forbidden
                    or alias.name.startswith(tuple(f"{f}." for f in forbidden))
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".")[-1] in forbidden:
                    offenders.append(module)
        assert offenders == []

    def test_the_loader_writes_nothing(self):
        """Seeding is am-03's exclusive-create job; a reader stays a reader."""
        source = Path(agent_modes.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        writers = {"write_text", "write_bytes", "mkdir", "makedirs", "unlink",
                   "rename", "replace", "remove", "rmdir", "chmod", "symlink"}
        # am-03 put the seeder *in this module* (so the seeded bytes and the
        # loader can never drift), so the rule is per-function, not per-file:
        # only the seeding half may touch the disk, and everything the loader
        # walks — ``_current``, ``_load``, ``_read_text``, the accessors — stays
        # read-only. A write appearing in one of those is the mistake this pins:
        # a per-request reader that creates a folder in a browsed project.
        seeding_functions = {
            "ensure_agent_modes_file", "_ensure_agent_modes",
            "agent_modes_file_path", "_seeded_text", "_state_dict",
            "_is_regular_file", "_discard_seeded_file",
        }
        offenders: List[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in seeding_functions:
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    if call.func.attr in writers:
                        offenders.append(f"{node.name}.{call.func.attr}")
                    if call.func.attr == "open":
                        offenders.append(f"{node.name}.open")
        assert offenders == []
        # The exclusive-create flag belongs to exactly one function, and there is
        # no check-then-write TOCTOU anywhere in the module.
        open_flags = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and "O_CREAT" in ast.dump(node)
        }
        assert open_flags == {"_ensure_agent_modes"}
        assert "os.path.isfile" not in source

    def test_the_seeder_is_reachable_but_the_reader_is_the_default(self):
        """The public read side never names the write side."""
        for name in ("resolve_modes", "resolve_mode", "_current", "_load"):
            body = next(
                node for node in ast.walk(ast.parse(
                    Path(agent_modes.__file__).read_text(encoding="utf-8")))
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            assert "ensure_agent_modes" not in ast.dump(body)

    def test_the_benchmark_never_consults_a_mode(self):
        """The invariant that survives every later task in this lot.

        A benchmark run has ``LLM_SYSPROMPT`` and that is enough (am-00): modes
        are a WorkBench concern. ``core/chat.py`` will take a *string* from the
        mode (am-04) without learning what a mode is, so it too stays off this
        list — the seam is ``routes.py``, and no core module reaches the loader.
        """
        core = Path(agent_modes.__file__).resolve().parent
        for name in ("bench.py", "llm.py", "chat.py", "job.py", "batch.py"):
            # One scan, shared with am-08's poisoned case: an inline copy of it
            # here once missed ``from src.core import agent_modes`` (the loader
            # is in the alias, not the module), which is exactly the import this
            # test exists to refuse.
            found = _mode_imports((core / name).read_text(encoding="utf-8"))
            assert found == [], f"{name} reaches the modes loader: {found}"


class TestConstantsAreSane:
    """The caps are the feature's edges: each one is a number someone can read."""

    def test_the_documented_values(self):
        assert agent_modes.AGENT_MODES_FILENAME == "agent-modes.json"
        assert agent_modes.AGENT_MODES_VERSION == 1
        assert agent_modes.MAX_MODES == 16
        assert agent_modes.MAX_MODE_ID_CHARS == 32
        assert agent_modes.MAX_MODE_LABEL_CHARS == 40
        assert agent_modes.MAX_MODE_SYSTEM_MESSAGE_BYTES == 16 * 1024
        assert agent_modes.RESERVED_MODE_IDS == ("todoBuilder",)
        assert agent_modes.MODE_ID_PATTERN == r"^[a-z][A-Za-z0-9_-]*$"
        assert agent_modes.POLICY_DIRNAME == ".nojae"

    def test_the_folder_name_is_the_policy_module_s_not_a_second_constant(self):
        """Two constants for one reserved folder is how it stops being one folder."""
        from mcp_connectors import policy_shared

        assert agent_modes.POLICY_DIRNAME is policy_shared.POLICY_DIRNAME

    def test_the_reserved_id_is_a_valid_mode_id(self):
        """Otherwise a reserved name would reach the declared path by accident."""
        import re

        for reserved in agent_modes.RESERVED_MODE_IDS:
            assert agent_modes._mode_id_problem(reserved) is None
            assert re.match(agent_modes.MODE_ID_PATTERN, reserved)

    def test_the_two_key_sets_do_not_overlap(self):
        assert not set(agent_modes.TOP_LEVEL_KEYS) & set(agent_modes.MODE_KEYS)

    def test_the_self_check_refuses_a_cap_typed_as_a_string(self):
        """The loader promises never to raise about a file; a bad *cap* is different."""
        original = agent_modes.MAX_MODES
        agent_modes.MAX_MODES = "16"
        try:
            with pytest.raises(ValueError, match="MAX_MODES"):
                agent_modes._self_check()
        finally:
            agent_modes.MAX_MODES = original
        agent_modes._self_check()


# -- am-02: the content module, against the real loader -------------------------

#: Every mention of a mode inside persona prose, as ``"<Label> mode"``. The test
#: that uses it asks that each such name is one the code actually defines.
MODE_MENTION_RE = re.compile(r"\b([A-Z][A-Za-z]*(?: [a-z][A-Za-z]*)?) mode\b")

PERSONA_TUPLES = (
    ("CHAT_PERSONA", defaults.CHAT_MODE_ID),
    ("CODE_PERSONA", defaults.CODE_MODE_ID),
    ("ARCHITECT_PERSONA", defaults.ARCHITECT_MODE_ID),
    ("TODO_BUILDER_PERSONA", defaults.TODO_BUILDER_MODE_ID),
)


def persona_lines(attribute: str) -> List[str]:
    """The lines of one persona, looked up by its module attribute name."""
    return list(getattr(defaults, attribute))


@pytest.fixture
def real_workspace(tmp_path: Path,
                   monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch workspace with **no stand-in**: the real content answers here.

    The loader resolves the content module by name per call, so leaving
    ``sys.modules`` alone is enough to reach the module imported at the top of
    this file — the same seam, pointed at the real words.
    """
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(tmp_path))
    agent_modes.invalidate_agent_modes_cache()
    yield tmp_path
    agent_modes.invalidate_agent_modes_cache()


class TestSeededDocumentRoundTrips:
    """The one-object argument, finally paid off: seed == fallback == file.

    Against the stand-in this class would only prove the seam works. Here it
    proves the thing the whole design of am-02 exists for — the bytes a
    workspace is seeded with, and the modes a broken file falls back to, are the
    same definition read through the same validator.
    """

    def test_the_rendered_file_is_the_document_itself(self):
        assert json.loads(defaults.default_agent_modes_text()) == (
            defaults.DEFAULT_AGENTS_DOCUMENT)
        assert defaults.default_agent_modes_text().endswith("\n")

    def test_default_modes_is_the_document_validated_not_a_second_copy(self):
        document = json.loads(json.dumps(defaults.DEFAULT_AGENTS_DOCUMENT))
        modes, declared_default, findings, usable = agent_modes._validated_document(
            document, "<inline>"
        )
        assert usable is True
        assert findings == []
        assert defaults.default_modes() == modes
        assert declared_default == defaults.DEFAULT_AGENTS_DOCUMENT["default"]

    def test_the_seeded_bytes_load_with_no_findings(self, real_workspace: Path):
        write_modes(real_workspace, defaults.default_agent_modes_text())
        payload = resolved()
        assert payload["findings"] == []
        assert payload["source"]["loaded"] is True
        assert payload["default"] == "code"
        assert [mode.id for mode in payload["modes"]] == [
            "chat", "code", "architect", "todoBuilder"]

    def test_the_file_and_the_fallback_answer_identically(self, real_workspace: Path):
        """Write the seed, or break the file: the same four modes either way."""
        write_modes(real_workspace, defaults.default_agent_modes_text())
        from_file = resolved()["modes"]
        write_modes(real_workspace, "{ not json")
        agent_modes.invalidate_agent_modes_cache()
        from_defaults = resolved()["modes"]
        assert from_file == from_defaults

    def test_the_seeded_document_does_not_declare_the_reserved_mode(self):
        for reserved in agent_modes.RESERVED_MODE_IDS:
            assert reserved not in defaults.DEFAULT_AGENTS_DOCUMENT["agents"]

    def test_the_seeded_default_is_the_mode_that_behaves_as_the_app_always_has(self):
        """``code`` stays the default: the feature must change nobody's defaults."""
        assert defaults.DEFAULT_AGENTS_DOCUMENT["default"] == defaults.CODE_MODE_ID
        code = next(mode for mode in defaults.default_modes()
                    if mode.id == defaults.CODE_MODE_ID)
        assert (code.mcptools, code.lsptools) == (True, True)

    def test_the_document_holds_its_own_explanation(self):
        """The four things a first-time editor cannot infer from the keys (am-02 §1)."""
        readme = "\n".join(defaults.DEFAULT_AGENTS_DOCUMENT["readme"])
        assert readme == "\n".join(defaults.AGENT_MODES_README)
        assert "unknown key is refused" in readme
        assert "systemMessage" in readme and "list" in readme
        assert "LLM_SYSPROMPT" in readme
        assert "not 1/0" in readme
        reserved_note = '"todoBuilder" is defined by benchmarkit and cannot be set here'
        assert reserved_note in readme
        for key in agent_modes.TOP_LEVEL_KEYS:
            assert key in readme, f"the readme never names the top-level key {key}"


class TestResolvedContent:
    def test_a_missing_file_gives_the_three_defaults_plus_the_app_mode(
        self, real_workspace: Path
    ):
        assert not os.path.exists(agent_modes.agent_modes_path())
        payload = resolved()
        assert payload["findings"] == []
        assert [mode.id for mode in payload["modes"]] == [
            "chat", "code", "architect", "todoBuilder"]
        assert payload["modes"][-1].editable is False
        assert all(mode.editable for mode in payload["modes"][:-1])

    def test_only_the_reserved_mode_carries_generated_blocks(self,
                                                            real_workspace: Path):
        builder = mode_of("todoBuilder")
        assert len(builder.instructions) == 1
        assert builder.instructions[0].strip()
        for mode in resolved()["modes"]:
            if mode.id != defaults.TODO_BUILDER_MODE_ID:
                assert mode.instructions == ()

    def test_the_reserved_mode_is_the_app_definition_whatever_the_file_says(
        self, real_workspace: Path
    ):
        agents = dict(GOOD_AGENTS)
        agents["todoBuilder"] = entry(label="Mine", mcptools=False,
                                      systemMessage="my own lot rules")
        write_modes(real_workspace, modes_json(agents))
        builder = mode_of("todoBuilder")
        assert builder.id == defaults.TODO_BUILDER_MODE_ID
        assert builder.label == defaults.TODO_BUILDER_LABEL
        assert builder.editable is False
        assert "my own lot rules" not in builder.system_message
        assert any("cannot be set here" in note for note in notes())
        assert errors() == []

    def test_the_mode_flags_match_the_design(self, real_workspace: Path):
        """The coarse reach of each persona — the table am-00 D4 starts from."""
        expected = {
            "chat": (False, False),
            "code": (True, True),
            "architect": (True, False),
            "todoBuilder": (True, False),
        }
        for mode_id, flags in expected.items():
            mode = mode_of(mode_id)
            assert (mode.mcptools, mode.lsptools) == flags
            assert mode.system_message, f"{mode_id} has no persona"

    def test_every_persona_is_the_module_constant_join(self, real_workspace: Path):
        """The document, the constants and the resolved mode say one text."""
        lines_by_id = {
            mode_id: persona_lines(attribute)
            for attribute, mode_id in PERSONA_TUPLES
        }
        for mode_id, lines in lines_by_id.items():
            assert mode_of(mode_id).system_message == "\n".join(lines)


class TestLotConventionsBlock:
    """The block that cannot be allowed to go stale: the runner enforces these."""

    def test_it_quotes_the_live_batch_constants(self):
        block = defaults.lot_conventions_block()
        assert batch.DEFAULT_LOT_DIR in block
        assert batch.PREFIX_PATTERN in block
        assert str(batch.MAX_BATCH_TASKS) in block
        assert str(batch.MAX_BATCH_LOT_BYTES) in block
        assert batch.DONE_SUBFOLDER in block
        for separator in batch.NUMBER_SEPARATORS:
            assert f"`{separator}`" in block

    def test_a_patched_constant_changes_the_text(self, monkeypatch):
        """The drift guard: pasting the numbers back in as literals fails here."""
        before = defaults.lot_conventions_block()
        monkeypatch.setattr(batch, "MAX_BATCH_TASKS", 7)
        monkeypatch.setattr(batch, "DEFAULT_LOT_DIR", "tasks")
        monkeypatch.setattr(batch, "DONE_SUBFOLDER", "completed")
        after = defaults.lot_conventions_block()
        assert after != before
        assert "At most 7 tasks" in after
        assert "`tasks/`" in after
        assert "`completed/`" in after
        # The values that were live when the *other* text was rendered are gone,
        # so the block is re-rendered per call rather than frozen at import.
        assert "`todos/`" not in after and "At most 100 tasks" not in after

    def test_the_reserved_mode_carries_it(self):
        assert defaults.todo_builder_mode().instructions == (
            defaults.lot_conventions_block(),)

    def test_the_persona_points_at_the_block_it_does_not_restate(self):
        """The persona says "the lot conventions below"; the block is below it."""
        assert "conventions below" in defaults.todo_builder_mode().system_message
        assert "MAX_BATCH_TASKS" not in "\n".join(defaults.TODO_BUILDER_PERSONA)


class TestPersonaProse:
    """Prose is the part of this feature a later edit rots without noticing."""

    def test_no_persona_is_over_half_the_byte_cap(self):
        cap = agent_modes.MAX_MODE_SYSTEM_MESSAGE_BYTES
        for attribute, mode_id in PERSONA_TUPLES:
            size = len("\n".join(persona_lines(attribute)).encode("utf-8"))
            assert size < cap // 2, f"{mode_id} persona is {size} bytes of {cap}"

    def test_an_editable_persona_stays_inside_its_line_budget(self):
        """The ~15-line budget am-02 §2 sets for the three modes a user rewrites.

        The app-owned persona is longer (it carries the whole lot discipline) and
        is bounded by the byte cap instead — the assertion above.
        """
        for attribute, mode_id in PERSONA_TUPLES:
            if mode_id == defaults.TODO_BUILDER_MODE_ID:
                continue
            lines = persona_lines(attribute)
            assert len(lines) <= 16, f"{mode_id} persona is {len(lines)} lines"

    def test_no_persona_claims_to_have_read_agents_md(self):
        """No persona names the instructions file — the block frames it.

        ``config.agent_instructions_block`` frames AGENTS.md as the project's own
        instructions, and every mode receives it, so a persona that names the
        *file* makes a claim about the working location it cannot verify and
        duplicates the thing the block already states. Whether a persona points
        at those instructions at all is its own business — a mode whose job is to
        answer without the workspace has no reason to mention them.
        """
        for attribute, mode_id in PERSONA_TUPLES:
            # The wrapped lines are one sentence, so whitespace is collapsed
            # before the phrase is looked for.
            joined = " ".join("\n".join(persona_lines(attribute)).split())
            assert "AGENTS.md" not in joined, f"{mode_id} persona names the file"

    def test_every_mode_mention_in_the_prose_is_a_label_the_code_defines(self):
        """A renamed label leaves stale prose behind; this is the tripwire."""
        labels = {
            defaults.CHAT_LABEL, defaults.CODE_LABEL, defaults.ARCHITECT_LABEL,
            defaults.TODO_BUILDER_LABEL,
        }
        mentioned = set()
        for attribute, _ in PERSONA_TUPLES:
            mentioned.update(MODE_MENTION_RE.findall(
                "\n".join(persona_lines(attribute))))
        assert mentioned, "no persona names its own mode — the framing is gone"
        assert mentioned <= labels, mentioned - labels

    def test_a_persona_never_hardcodes_a_mode_mention(self):
        """A *mode mention* in the prose interpolates the label constant.

        A mention — ``"<Label> mode"`` — is a claim about a dropdown entry, and a
        spelled-out one survives a rename as a lie. Ordinary English that happens
        to repeat a word ("markdown for structure") is not a mention and is not
        this rule's business, which is why the check looks for the phrase.

        The names are read off the **source**, not the live objects: an
        f-string's interpolated value would hide the literal either way, so the
        only honest version of this test is textual.
        """
        source = ast.parse(Path(defaults.__file__).read_text(encoding="utf-8"))
        wanted = {name for name, _ in PERSONA_TUPLES}
        tuples: Dict[str, ast.Tuple] = {}
        for node in ast.walk(source):
            if (not isinstance(node, ast.AnnAssign)
                    or not isinstance(node.value, ast.Tuple)):
                continue
            target = node.target
            if isinstance(target, ast.Name) and target.id in wanted:
                tuples[target.id] = node.value
        # A vacuous pass is worse than no test: these are annotated assignments,
        # and a refactor that drops the annotations must not silence the check.
        assert set(tuples) == wanted, sorted(wanted - set(tuples))

        labels = [
            defaults.CHAT_LABEL, defaults.CODE_LABEL, defaults.ARCHITECT_LABEL,
            defaults.TODO_BUILDER_LABEL,
        ]
        for attribute, tuple_node in tuples.items():
            for element in tuple_node.elts:
                if isinstance(element, ast.Constant):
                    literal = str(element.value)
                elif isinstance(element, ast.JoinedStr):
                    # The static text of an f-string: the parts around the braces.
                    literal = "".join(
                        part.value for part in element.values
                        if isinstance(part, ast.Constant)
                    )
                else:
                    continue
                for label in labels:
                    assert f"{label} mode" not in literal, (
                        f"{attribute} spells out the mention {label!r} mode; "
                        "interpolate the label constant instead")

    def test_the_readme_lines_are_short_enough_to_read_in_the_file(self):
        assert all(len(line) <= 80 for line in defaults.AGENT_MODES_README)
        assert len(defaults.AGENT_MODES_README) >= 4


class TestDefaultsModuleIsStillCoreOnly:
    """am-02 stays the words: no transport, no model, no second reader."""

    def test_it_reads_no_file_and_writes_none(self):
        source = Path(defaults.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        writers = {"write_text", "write_bytes", "mkdir", "makedirs", "unlink",
                   "open", "remove", "rename"}
        assert writers & called == set()
        assert "O_CREAT" not in source
        assert "ensure_agent_modes_file" not in source

    def test_it_imports_only_am_01_batch_and_the_standard_library(self):
        """The seam is am-01, the constants are ``core.batch``: nothing else.

        ``from src.core import agent_modes, batch`` is one ``ImportFrom`` of
        ``src.core``, so the pinned set names the package, not each submodule —
        the point is that no route, hub, model client or config module appears.
        """
        tree = ast.parse(Path(defaults.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert imported == {"json", "typing", "src.core", "src.core.agent_modes"}
        forbidden = ("fastapi", "litellm", "src.api", "mcphub", "lsphub",
                     "termhub", "editorfs", "config")
        assert not any(any(part in name for part in forbidden) for name in imported)

    def test_the_loader_is_the_only_thing_that_builds_the_shapes(self):
        """``AgentMode`` comes from am-01: no local copy of the dataclass."""
        source = Path(defaults.__file__).read_text(encoding="utf-8")
        assert "@dataclass" not in source
        assert defaults.AgentMode is agent_modes.AgentMode

    def test_the_words_have_exactly_one_home_in_the_tree(self):
        """No second copy of a persona line, a readme line or the conventions.

        The invariant am-00 states for this module: it is the only place the
        seeded bytes and the lot conventions live. A paste somewhere else is a
        second source of truth that drifts on the next edit, so the longest
        literal string the module itself holds may appear in this file and
        nowhere else under ``src/``.

        The probes come out of the module's **own source** rather than being
        typed into this test: a persona's first line is an f-string, so the
        rendered sentence is not a literal anywhere, and re-typing a probe here
        would put a second copy of the words in the tree — the thing measured.
        """
        own_path = Path(defaults.__file__).resolve()
        src_root = own_path.parent.parent
        own = own_path.relative_to(src_root).as_posix()

        own_source = own_path.read_text(encoding="utf-8")
        tree = ast.parse(own_source)
        # ``ast.get_docstring`` dedents, so a docstring is identified by the
        # **node** that is one — the bare string expression opening a body.
        doc_nodes = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        literals = sorted(
            (node.value for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and len(node.value) > 60 and id(node) not in doc_nodes),
            key=len,
            reverse=True,
        )
        # Implicit concatenation joins pieces across source lines, so keep only
        # what is findable in the module's own text — the same substring a reader
        # or a copy-paste would produce.
        probes = [value for value in literals if value in own_source][:8]
        assert probes, "the module holds no findable long literals to probe for"

        haystacks = {
            path.relative_to(src_root).as_posix(): path.read_text(encoding="utf-8")
            for path in list(src_root.rglob("*.py")) + list(src_root.rglob("*.json"))
            if "__pycache__" not in path.parts
        }
        assert own in haystacks
        for probe in probes:
            holders = [name for name, text in haystacks.items() if probe in text]
            assert holders == [own], f"{probe[:40]!r} also lives in {holders}"

    def test_its_self_check_refuses_a_persona_over_the_cap(self, monkeypatch):
        """The import-time guarantee is a real check, not a sentence."""
        monkeypatch.setattr(agent_modes, "MAX_MODE_SYSTEM_MESSAGE_BYTES", 64)
        try:
            with pytest.raises(ValueError, match="chat persona"):
                defaults._self_check()
        finally:
            monkeypatch.undo()
        defaults._self_check()


# -- am-08: the sweep — what only the assembled feature can show ---------------


def _mode_imports(source: str) -> List[str]:
    """Every import of the modes loader in one source text.

    An ``ast`` scan in the style of ``tests/test_write_surface_isolation.py``,
    pulled out as a function because the sweep runs it twice: on the real
    modules, which must be clean, and on a poisoned copy, which must **not** be.
    A guard nobody has watched fail is a sentence, not a check.

    Both spellings count. ``from src.core import agent_modes`` names the loader
    in the *alias*, not the module, so a scan that only inspects
    ``ImportFrom.module`` waves it through — the poison case below is what found
    that, and it is the spelling a contributor is most likely to write.
    """
    found: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
            "agent_modes"
        ):
            found.append(node.module or "")
        elif isinstance(node, ast.ImportFrom) and any(
            (alias.name == "agent_modes" or alias.name.startswith("agent_modes."))
            for alias in node.names
        ):
            found.extend(
                alias.name for alias in node.names
                if alias.name == "agent_modes" or alias.name.startswith("agent_modes.")
            )
        elif isinstance(node, ast.Import) and any(
            alias.name.endswith("agent_modes") for alias in node.names
        ):
            found.extend(alias.name for alias in node.names)
    return found


class _HubDouble:
    """A live hub answering ``active_tool_defs`` from the ids it was *given*."""

    installed = True

    def __init__(self) -> None:
        self.requested_ids: List[List[str]] = []

    def status(self):
        return []

    def active_tool_defs(self, ids):
        self.requested_ids.append(list(ids))
        tools = [{"type": "function", "function": {"name": f"use_{cid}"}}
                 for cid in ids]
        return tools, {f"use_{cid}": cid for cid in ids}

    def call_tool_sync(self, cid, name, args, timeout=30.0):
        return "ok"


@pytest.fixture
def swept_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real app on a scratch workspace holding the **seeded** modes file.

    The last promises of this lot are about the request that leaves the process,
    so this wires the three things separately-tested suites only ever wired one
    at a time: an HTTP client, an ``AGENTS.md`` at the working location, and the
    **real** ``core.chat.stream_chat`` with only ``completion`` stubbed — so what
    is asserted is the assembled system message and the assembled litellm
    kwargs, not the arguments one module handed the next.

    The modes file comes from :func:`agent_modes.ensure_agent_modes_file`
    (am-03's seeder), never from a document this test writes: a sweep that
    invents its own JSON would pass on a seed nobody ships.
    """
    from fastapi.testclient import TestClient

    from src.api import routes
    from src.api.server import create_app
    from src.core.models import EndpointsConfig

    root = tmp_path / "swept-project"
    root.mkdir()
    (root / "AGENTS.md").write_text(
        "# swept rules\n\nUse pytest for the suite.\n", encoding="utf-8"
    )
    monkeypatch.setenv("BENCHMARKIT_PATH_LOCATION", str(root))
    agent_modes.invalidate_agent_modes_cache()

    app = create_app()
    monkeypatch.setattr(
        routes,
        "load_endpoints_config",
        lambda: (
            EndpointsConfig(serverUrl="http://localhost:8080/v1",
                            endpoint="model-a"),
            None,
        ),
    )
    monkeypatch.setattr(routes, "RESULTS_PATH", tmp_path / "results.json")
    # Settings-level activation: *every* hub on, so the only thing that can
    # take tools away in the requests below is the mode (am-00's flagship).
    monkeypatch.setattr(routes, "_active_connector_ids", lambda: ["console"])
    monkeypatch.setattr(routes, "_active_server_ids", lambda: ["pyright"])

    captured: Dict[str, Any] = {}
    monkeypatch.setattr(routes, "stream_chat", chat_core.stream_chat)
    monkeypatch.setattr(
        "src.core.chat.completion",
        lambda **kwargs: captured.clear() or captured.update(kwargs) or iter([]),
    )

    try:
        with TestClient(app) as client:
            client.app.state.mcp = _HubDouble()
            client.app.state.lsp = _HubDouble()
            # Boot seeded it (am-03); calling the seeder here is the assertion
            # that the file in play *is* that document.
            assert agent_modes.ensure_agent_modes_file(str(root))["state"] in (
                "created", "existing",
            )
            agent_modes.invalidate_agent_modes_cache()

            def post(mode_id: Optional[str] = None):
                body: Dict[str, Any] = {
                    "messages": [{"role": "user", "content": "hi"}]
                }
                if mode_id is not None:
                    body["agentMode"] = mode_id
                return client.post("/api/chat", json=body)

            yield client, root, captured, post
    finally:
        agent_modes.invalidate_agent_modes_cache()


class TestTheWholeFeatureInOneRequest:
    """am-08 §4: the invariants that need am-01 … am-07 all standing at once."""

    def test_a_seeded_todo_builder_turn_carries_persona_agents_md_conventions(
        self, swept_client
    ):
        _client, root, captured, post = swept_client
        assert post(defaults.TODO_BUILDER_MODE_ID).status_code == 200
        content = captured["messages"][0]["content"]

        persona = "\n".join(defaults.TODO_BUILDER_PERSONA)
        conventions = defaults.lot_conventions_block()
        parts = [
            content.index(persona),
            content.index(f"project root is {root}"),
            content.index(conventions),
        ]
        assert parts == sorted(parts), (
            "persona → AGENTS.md → the mode's blocks is the order am-00 fixes"
        )
        assert len(set(parts)) == 3
        # D1 as an end-to-end fact, not a kwarg check: the mode's persona is the
        # *first* thing in the system message, and the built-in prompt is
        # nowhere in it.
        assert content.startswith(persona)
        assert DEFAULT_LLM_SYSPROMPT not in content
        # The conventions are the generated block, quoting the live constants.
        assert batch.DEFAULT_LOT_DIR in conventions

    def test_a_chat_mode_turn_sends_no_tools_and_keeps_the_project_block(
        self, swept_client
    ):
        # The other flagship of am-00, in the same request shape: a mode whose
        # flags are both false, on a workspace where **Settings** has both hubs
        # active. The mode narrows and never grants, and it narrows by taking
        # the schema away — no ``tools`` key at all — while AGENTS.md survives.
        client, root, captured, post = swept_client
        assert post("chat").status_code == 200
        assert "tools" not in captured
        assert "tool_executor" not in captured
        mcp_hub, lsp_hub = client.app.state.mcp, client.app.state.lsp
        assert mcp_hub.requested_ids == [[]]
        assert lsp_hub.requested_ids == [[]]
        content = captured["messages"][0]["content"]
        assert "# swept rules" in content
        assert f"project root is {root}" in content
        # No tool schema reached litellm, so ``stream_chat`` appends no
        # TOOL_NOTICE either: the prompt and the affordance stay in step (D3).
        assert chat_core.TOOL_NOTICE.strip() not in content

    def test_the_mode_gate_survives_a_second_turn_of_the_same_conversation(
        self, swept_client
    ):
        # Two turns, one thread, no second cache: the gate is decided per
        # request, so the mode of *this* turn is what litellm sees this turn.
        client, _root, captured, post = swept_client
        assert post("code").status_code == 200
        assert "tools" in captured
        assert client.app.state.mcp.requested_ids == [["console"]]
        captured.clear()
        assert post("chat").status_code == 200
        assert "tools" not in captured
        assert client.app.state.mcp.requested_ids == [["console"], []]

    def test_an_extra_mode_in_the_file_is_listed_and_selectable(
        self, swept_client
    ):
        """D8's point: the vocabulary is closed, the *modes* are not.

        A workspace can add a fourth agent with no code change — the sweep
        proves the whole path (loader → ``GET /api/agent-modes`` → the dropdown
        payload → ``POST /api/chat``) accepts something benchmarkit never heard
        of, and that its persona is the one that answers.
        """
        client, root, captured, post = swept_client
        path = (
            root / agent_modes.POLICY_DIRNAME / agent_modes.AGENT_MODES_FILENAME
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        document["agents"]["myhelper"] = {
            "label": "My helper",
            "mcptools": True,
            "lsptools": False,
            "systemMessage": ["MY HELPER PERSONA"],
        }
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        agent_modes.invalidate_agent_modes_cache()

        body = client.get("/api/agent-modes").json()
        assert [mode["id"] for mode in body["modes"]] == [
            "chat", "code", "architect", "myhelper",
            defaults.TODO_BUILDER_MODE_ID,
        ]
        assert body["findings"] == []

        assert post("myhelper").status_code == 200
        content = captured["messages"][0]["content"]
        assert content.startswith("MY HELPER PERSONA")
        # It reaches the workspace like ``code`` does, because it says so.
        assert "tools" in captured

    def test_the_benchmark_guard_fails_when_poisoned(self):
        """The scan above is a guard, so it has to be shown to bite.

        ``test_the_benchmark_never_consults_a_mode`` in ``TestTheReaderIsAlone``
        reads the same AST; here the same reader is handed an import that *does*
        reach the loader, and refuses it. An ``ast`` check that never fails on
        anything is indistinguishable from a check that never runs.
        """
        core = Path(agent_modes.__file__).resolve().parent
        for name in ("bench.py", "llm.py"):
            clean = (core / name).read_text(encoding="utf-8")
            assert _mode_imports(clean) == []
            for poison in (
                "from src.core import agent_modes\n",
                "from src.core.agent_modes import resolve_mode\n",
                "import src.core.agent_modes\n",
            ):
                assert _mode_imports(poison + clean) != [], (
                    f"{name} accepts {poison.strip()!r} unnoticed"
                )

    def test_the_readme_holds_the_seeded_file_verbatim(self):
        """Docs that disagree with a config file are worse than no docs.

        The README quotes the modes document the app seeds, so the quote is
        checked against the module that owns those bytes — parseable, and equal
        as data. A hand-retyped copy would pass a prose check and fail this one
        the day someone edits a persona.
        """
        readme = (
            Path(agent_modes.__file__).resolve().parents[2] / "README.md"
        ).read_text(encoding="utf-8")
        fenced = [
            block
            for block in readme.split("```json\n")
            if '"agents":' in block and '"systemMessage":' in block
        ]
        assert len(fenced) == 1, "the README should quote the seeded file once"
        quoted = json.loads(fenced[0].split("\n```", 1)[0])
        assert quoted == defaults.DEFAULT_AGENTS_DOCUMENT
        # …and the seeded *text*, byte for byte, so the indentation and the
        # list-vs-string forms in the docs are the shipped ones.
        assert defaults.default_agent_modes_text().rstrip("\n") in readme

    def test_the_readme_table_names_every_mode_the_feature_ships(self):
        """The one table a human reads must not fall behind the code."""
        readme = (
            Path(agent_modes.__file__).resolve().parents[2] / "README.md"
        ).read_text(encoding="utf-8")
        for mode in defaults.default_modes():
            assert f"| `{mode.id}` | {mode.label} |" in readme, mode.id
        reserved = defaults.todo_builder_mode()
        assert (
            f"| `{reserved.id}` | {reserved.label} | yes | no | **no**" in readme
        )
