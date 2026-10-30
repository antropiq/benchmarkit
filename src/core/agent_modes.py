"""Agent modes: what the WorkBench agent is *for*, read from one JSON file.

``<workspace>/.nojae/agent-modes.json`` names the personas the composer offers —
each one a system message plus two booleans saying whether the MCP and the LSP
hub reach the model at all. This module is the whole **read** side of that file:
it validates it, answers "which modes exist" and "what does mode *X* say", and
caches the answer per workspace. It talks to no model, no route and no browser;
am-04 (prompt assembly) and am-06 (the routes) consume what :func:`resolve_modes`
returns. The one exception is :func:`ensure_agent_modes_file` below (am-03): the
seeder lives here because it has to write exactly the bytes the loader reads, and
it runs only at the two moments a folder becomes the working location.

Two things this deliberately is **not**. It is not a hub: no live sessions, no
subprocesses, nothing hot-toggled — where ``mcphub``/``lsphub`` own processes, a
mode is a string and two booleans, so a hot edit costs one re-read. And it is not
the path policy: :mod:`editorfs.policy_file` blocks every write when its file is
broken, because an unloaded policy protects nothing, while a broken modes file
must never block a request — its cost is a wrong persona, not an unprotected
tree (am-00 D11). So this loader always answers with usable modes plus findings
saying what it noticed, and it decides nothing about what an *unknown mode id*
costs a caller: that belongs to :func:`resolve_mode`'s caller (the chat route
answers 422, D14).

Like the policy file, the modes file is **seeded** — written once when a folder
becomes the working location — and edited by hand outside the app from then on
(D6); the app only ever validates and reports. One asymmetry with it is
deliberate and easy to "fix" wrongly: the file declares the editable modes, and
:func:`resolve_modes` always appends one more, from :data:`RESERVED_MODE_IDS`.
That mode is benchmarkit's own, rendered by :mod:`core.agent_modes_defaults`
rather than parsed, because its instructions are the lot conventions the runner
enforces in :mod:`core.batch` — a JSON copy would be a second voice for a rule
the code already applies, and would let a workspace edit its way out of a lot it
cannot launch. Moving it into the file is the consolidation to argue against;
the reserved-mode tests in ``tests/test_agent_modes.py`` are the answer.
"""

import importlib
import json
import os
import re
import stat as stat_module
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.config import PROJECT_ROOT, path_location

# The reserved folder's name is not re-declared here: it comes from
# ``mcp_connectors/policy_shared``, with the same ``sys.path`` fallback
# :mod:`editorfs.constants` uses, because that module lives outside ``src`` so
# the stdio connectors can read it too. Two constants for one folder name is how
# the folder stops being one folder. (The *modes* file has no connector-side
# reader: modes shape what the model is told, and every connector gets its
# instructions through the prompt it is called with.)
try:
    from mcp_connectors import policy_shared as shared
except ImportError:  # pragma: no cover - depends on how the app is started
    _root = str(PROJECT_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from mcp_connectors import policy_shared as shared  # type: ignore[no-redef]

POLICY_DIRNAME: str = shared.POLICY_DIRNAME

# -- the constants (module attributes, patchable) -------------------------------

#: The modes file's name, beside ``read-write-policy.json`` in the reserved
#: folder. A module attribute so a test can move the lookup — and the cache
#: entry carries the name it was read under, so moving it invalidates.
AGENT_MODES_FILENAME = "agent-modes.json"

#: The only ``version`` this file is understood at; a mismatch is an error,
#: mirroring the policy file's rule (a different version means a different
#: vocabulary, and guessing at one is how a mode silently stops existing).
AGENT_MODES_VERSION = 1

#: Declared modes kept, in file order. The dropdown has to stay usable, and
#: every entry in it is persona text paid for on every turn of a thread.
MAX_MODES = 16

#: Bounds on one mode's own id and its display label.
MAX_MODE_ID_CHARS = 32
MAX_MODE_LABEL_CHARS = 40

#: Persona size, per mode, in UTF-8 bytes. This is the cap that keeps a persona
#: short: the text is re-sent on **every** turn of the conversation, so past it a
#: longer prompt is not a better agent, only a slower and dearer one.
MAX_MODE_SYSTEM_MESSAGE_BYTES = 16 * 1024

#: Modes benchmarkit owns: absent from the file, and a file that declares one is
#: answered with a note rather than an edit (D9). :func:`resolve_modes` appends
#: them, last, whatever the file says — no file can remove or redefine one.
RESERVED_MODE_IDS: Tuple[str, ...] = ("todoBuilder",)

#: The shape of a mode id. It is a JSON key, the value of ``ChatIn.agentMode``,
#: and the id the batch runner pins a task turn to — so it is matched against
#: this pattern per call, which keeps a patch of it meaningful.
MODE_ID_PATTERN = r"^[a-z][A-Za-z0-9_-]*$"

#: Top-level keys the file may carry, and the keys one mode may carry. An
#: unknown key at either level is an **error**, never ignored (D12): on a file
#: edited by hand, ``mcptool: true`` silently meaning "no MCP tools" is the
#: classic loss, and a closed vocabulary is what makes it visible instead. The
#: price is that a newer key needs this app updated first (am-00 D10).
TOP_LEVEL_KEYS: Tuple[str, ...] = ("version", "readme", "default", "agents")
MODE_KEYS: Tuple[str, ...] = ("label", "mcptools", "lsptools", "systemMessage")

#: Where the built-in content lives (am-02's module). Imported **by name, per
#: call**, never at module level: that module's ``default_modes()`` runs *this*
#: module's validator, so a static import in either direction is a cycle. The
#: name is also an attribute, so a test of the machinery here can swap in a
#: stand-in for the personas — am-01 owns the rules, am-02 owns the words.
DEFAULTS_MODULE_NAME = "src.core.agent_modes_defaults"


# -- the two shapes -------------------------------------------------------------

KIND_ERROR = "error"  # this mode or file rule was dropped; defaults cover the gap
KIND_NOTE = "note"    # nothing was dropped; the file says something odd


@dataclass(frozen=True)
class ModeFinding:
    """One thing the loader noticed about the modes file.

    ``message`` starts with the **JSON pointer** it came from
    (``agents.architect.mcptool``), because the file is hand-edited outside the
    app and has no preview of its own: a finding a user cannot trace to a line of
    their own JSON is a wall rather than a message.
    """

    kind: str
    message: str


@dataclass(frozen=True)
class AgentMode:
    """One mode as resolved: what the model is told, and what it may reach."""

    id: str
    label: str
    system_message: str              # "" = fall back to LLM_SYSPROMPT (D1)
    mcptools: bool
    lsptools: bool
    editable: bool = True            # False only for the app-owned mode (D9)
    instructions: Tuple[str, ...] = ()  # generated blocks, already rendered


# -- the message voice ----------------------------------------------------------

def _file_error(location: str, problem: str) -> ModeFinding:
    """A finding about the file as a whole, naming the file."""
    return ModeFinding(KIND_ERROR, f"The agent-modes file {location} {problem}.")


def _pointer_error(pointer: str, problem: str) -> ModeFinding:
    """A finding about one mode, naming the pointer the user has to edit."""
    return ModeFinding(KIND_ERROR, f"{pointer}: {problem}")


def _kind_of(value: Any) -> str:
    """What to call a JSON value in a message — the JSON word, not the Python one."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__  # pragma: no cover - json never produces this


def _names(keys: Tuple[str, ...]) -> str:
    return ", ".join(keys)


def _quoted_ids(ids: List[str]) -> str:
    """The ids a message is about, each in quotes — empty list says so."""
    if not ids:
        return "none"
    return ", ".join(f"\"{name}\"" for name in ids)


def _cap_of(num_bytes: int) -> str:
    """A byte cap spelled the way a human writes it in a JSON file (``16 KiB``)."""
    if num_bytes >= 1024 and num_bytes % 1024 == 0:
        return f"{num_bytes // 1024} KiB"
    return f"{num_bytes} bytes"


# -- the validator -------------------------------------------------------------

def _mode_id_problem(mode_id: Any) -> Optional[str]:
    """Why ``mode_id`` is not usable as a mode, or ``None`` when it is.

    Matched against :data:`MODE_ID_PATTERN` per call rather than through a regex
    compiled at import, so patching the pattern patches the rule.
    """
    if not isinstance(mode_id, str) or not mode_id:
        return "is not a mode id: a JSON key here must be a non-empty string"
    if len(mode_id) > MAX_MODE_ID_CHARS:
        return f"is a mode id longer than {MAX_MODE_ID_CHARS} characters"
    if re.match(MODE_ID_PATTERN, mode_id) is None:
        return (
            f"is not a usable mode id: it must match {MODE_ID_PATTERN} — a "
            "lowercase letter, then letters, digits, '_' or '-'"
        )
    return None


def _system_message_of(
    entry: Dict[str, Any], pointer: str
) -> Tuple[str, List[ModeFinding]]:
    """``(text, findings)`` for one mode's ``systemMessage``.

    A **string or a list of lines** (D8): the list form exists so a hand-edited
    persona stays readable inside JSON, and the two forms have to produce
    byte-identical text — the file that is seeded and the fallback used when it
    is broken are the same words, and they must not differ by a newline. An
    absent or ``null`` persona is legitimate and means "use the configured
    prompt" (D1), so it yields ``""`` with no finding.
    """
    value = entry.get("systemMessage")
    if value is None:
        return ("", [])
    if isinstance(value, str):
        text = value
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(line, str) and line.strip() for line in value)
    ):
        text = "\n".join(value)
    else:
        shape = (
            "holds an entry that is not a non-empty string; a list is one line "
            "per entry, so a paragraph break belongs in the string form"
            if isinstance(value, list)
            else "is a shape that cannot be read as a persona: use a string, or "
                 "a list of non-empty strings with one line per entry"
        )
        return ("", [_pointer_error(
            f"{pointer}.systemMessage",
            f"{_kind_of(value)} that {shape}",
        )])
    size = len(text.encode("utf-8"))
    if size > MAX_MODE_SYSTEM_MESSAGE_BYTES:
        return ("", [_pointer_error(
            f"{pointer}.systemMessage",
            f"is {size} bytes, over the {_cap_of(MAX_MODE_SYSTEM_MESSAGE_BYTES)} "
            "cap — this text is re-sent on every turn of the conversation",
        )])
    return (text, [])


def _flag_of(entry: Dict[str, Any], key: str, pointer: str) -> Tuple[bool, List[ModeFinding]]:
    """One of ``mcptools``/``lsptools``, as the strict boolean the file says.

    No truthiness coercion: ``1``, ``"yes"`` and ``0`` are exactly what a user
    reaching for another format types, and reading them as on/off is the guessing
    D12 refuses. **Absent means false** — a mode never grants itself reach it did
    not ask for, and Settings stays the authority (D4). An explicit ``null`` is
    *not* absence: it is a value the user typed where a boolean belongs, and the
    one key here where "I wrote nothing" and "I wrote null" must not answer the
    same way — unlike ``systemMessage``, where ``null`` is the documented way of
    saying "use the configured prompt" (D1).
    """
    if key not in entry:
        return (False, [])
    value = entry[key]
    if isinstance(value, bool):
        return (value, [])
    return (False, [_pointer_error(
        f"{pointer}.{key}",
        f"is {_kind_of(value)}, not a boolean: write \"{key}\": true or "
        f"\"{key}\": false",
    )])


def _one_mode(
    mode_id: str, entry: Any, pointer: str
) -> Tuple[Optional[AgentMode], List[ModeFinding]]:
    """Validate one declared mode and build it — or drop the **whole entry**.

    An entry is dropped as a unit, and that is the rule rather than a shortcut: a
    mode whose ``mcptools`` survived but whose ``systemMessage`` did not would be
    an agent with tools and no persona, which is exactly the thing this feature
    exists to prevent. Every problem in the entry is reported, so one pass
    through the findings fixes the file.
    """
    if not isinstance(entry, dict):
        return (None, [_pointer_error(
            pointer,
            f"is {_kind_of(entry)}, not an object with {_names(MODE_KEYS)}",
        )])

    findings: List[ModeFinding] = [
        _pointer_error(f"{pointer}.{key}", f"unknown key (expected {_names(MODE_KEYS)})")
        for key in sorted(entry)
        if key not in MODE_KEYS
    ]

    # ``label: null`` is the same mistake as ``label: 42``, not the same thing as
    # leaving the key out: an absent label is presentation the file declined to
    # state, a null one is a value where a string belongs.
    if "label" in entry:
        label = entry["label"]
        if not isinstance(label, str):
            findings.append(_pointer_error(
                f"{pointer}.label",
                f"is {_kind_of(label)}, not a string",
            ))

    mcptools, own = _flag_of(entry, "mcptools", pointer)
    findings.extend(own)
    lsptools, own = _flag_of(entry, "lsptools", pointer)
    findings.extend(own)
    system_message, own = _system_message_of(entry, pointer)
    findings.extend(own)

    if any(finding.kind == KIND_ERROR for finding in findings):
        return (None, findings)
    # A label is presentation: a missing, empty or over-long one is not a
    # finding, and the id is what the dropdown shows instead.
    declared = entry.get("label")
    shown = ""
    if isinstance(declared, str):
        shown = declared.strip()[:MAX_MODE_LABEL_CHARS]
    return (
        AgentMode(
            id=mode_id,
            label=shown or mode_id,
            system_message=system_message,
            mcptools=mcptools,
            lsptools=lsptools,
        ),
        findings,
    )


def _validated_document(
    document: Any, location: str
) -> Tuple[List[AgentMode], Optional[str], List[ModeFinding], bool]:
    """``(modes, default, findings, file_usable)`` for one parsed document.

    Two tiers of failure, and only one of them is the whole file:

    * the **file** is unusable — the built-in defaults answer for all of it —
      when the root is not an object, ``agents`` is not an object, or
      ``version`` is missing or names something else. Those three question the
      vocabulary itself, so nothing read through it can be trusted: a mode
      written for version 2's keys and accepted by version 1's rules is a mode
      with keys nobody asked for;
    * an **entry** is dropped for its own problems (see :func:`_one_mode`), and
      the entries around it stand: a typo in one persona must not cost the user
      the other three. An unknown **top-level** key sits in this tier too: it is
      reported as an error, but the keys beside it are spelled correctly and mean
      what they say, and the mistake that key usually stands for (``"agent"``
      where ``"agents"`` belongs) is caught by the ``agents`` row above anyway.

    Modes come back in **file order**, and that order is the dropdown's order: a
    JSON object keeps the order it was written in and Python preserves it, so
    ``agents`` reads as the user typed it. That is why nothing here sorts — a
    later "just sort it" refactor would silently reorder everyone's dropdown, so
    the tests assert the order instead of trusting it. The reserved modes are not
    added here; :func:`resolve_modes` appends them last.
    """
    if not isinstance(document, dict):
        return ([], None, [_file_error(
            location,
            f"holds {_kind_of(document)}, not an object with {_names(TOP_LEVEL_KEYS)}",
        )], False)

    findings: List[ModeFinding] = [
        _file_error(
            location,
            f"has an unknown top-level key {key!r}: this file understands "
            f"{_names(TOP_LEVEL_KEYS)}",
        )
        for key in sorted(document)
        if key not in TOP_LEVEL_KEYS
    ]
    version_findings = _version_findings(document.get("version"), location)
    findings.extend(version_findings)
    if version_findings:
        # Wrong or missing version: the entries below are not even looked at,
        # because the shape they are written in is the thing in question.
        return ([], None, findings, False)

    agents = document.get("agents")
    if not isinstance(agents, dict):
        findings.append(_file_error(
            location,
            f"\"agents\" is {_kind_of(agents)}, not an object of mode id to mode",
        ))
        return ([], None, findings, False)

    modes: List[AgentMode] = []
    overflow: List[str] = []
    for mode_id, entry in agents.items():
        pointer = f"agents.{mode_id}"
        if mode_id in RESERVED_MODE_IDS:
            # D9, a note rather than an error: the user *can* fix this one by
            # deleting the entry, and the fix must not cost them the modes that
            # were written correctly. benchmarkit's own definition is used.
            findings.append(ModeFinding(KIND_NOTE, (
                f"{pointer}: \"{mode_id}\" is defined by benchmarkit and cannot "
                "be set here; this entry is ignored"
            )))
            continue
        problem = _mode_id_problem(mode_id)
        if problem is not None:
            findings.append(_pointer_error(pointer, problem))
            continue
        mode, own = _one_mode(mode_id, entry, pointer)
        findings.extend(own)
        if mode is None:
            continue
        if len(modes) >= MAX_MODES:
            overflow.append(mode_id)
            continue
        modes.append(mode)

    if overflow:
        findings.append(_pointer_error(
            "agents",
            f"declares more than the {MAX_MODES} mode(s) this file may hold: "
            f"the {len(overflow)} past the cap ({_quoted_ids(overflow)}) are "
            "ignored",
        ))

    declared_default = document.get("default")
    usable_default: Optional[str] = None
    if declared_default is None:
        findings.append(_file_error(location, 'has no "default" key'))
    elif not isinstance(declared_default, str):
        findings.append(_file_error(
            location, f"\"default\" is {_kind_of(declared_default)}, not a mode id"
        ))
    elif declared_default not in {mode.id for mode in modes}:
        # A default naming a mode this file does not declare is the one mistake it
        # can make *about the whole conversation*, so it is said loudly — but it is
        # repaired on its own: the file's modes still stand and the built-in
        # default answers for this key. Declaring it is the only way to want it:
        # the app-owned modes cannot be configured here (D9), so a workspace that
        # means one is told the list this file can actually name.
        findings.append(_file_error(
            location,
            f"\"default\" is \"{declared_default}\", which is not one of the "
            f"modes it declares ({_quoted_ids([mode.id for mode in modes])}); "
            "a default has to name a mode this file declares — "
            + (
                "and this file declares none" if not modes else
                "benchmarkit's own modes are not settable here"
            ),
        ))
    else:
        usable_default = declared_default

    return (modes, usable_default, findings, True)


def _version_findings(version: Any, location: str) -> List[ModeFinding]:
    if version is None:
        return [_file_error(
            location, f"has no \"version\" key (expected {AGENT_MODES_VERSION})"
        )]
    # ``bool`` is an ``int`` in Python, and ``true`` is not a version.
    if isinstance(version, bool) or not isinstance(version, int):
        return [_file_error(
            location,
            f"\"version\" is {_kind_of(version)}, not a number "
            f"(expected {AGENT_MODES_VERSION})",
        )]
    if version != AGENT_MODES_VERSION:
        return [_file_error(
            location,
            f"\"version\" is {version}; this benchmarkit understands version "
            f"{AGENT_MODES_VERSION} only",
        )]
    return []


# -- the content this module does not own --------------------------------------

def _defaults() -> Any:
    """am-02's ``agent_modes_defaults``, resolved per call (see the constant).

    This is the one thing in the module allowed to raise: a missing content
    module is a broken installation, not a broken file, and no fallback of mine
    could be any less wrong than saying so. The contract of :func:`resolve_modes`
    — never raises — is about *the file*.
    """
    return importlib.import_module(DEFAULTS_MODULE_NAME)


def _fallback_modes() -> List[AgentMode]:
    """The built-in modes — what answers when the file cannot be trusted (D11).

    These are the *same* modes the seeder writes (am-02 renders both from one
    document), so a broken file costs the user their own edits and nothing else.
    """
    return list(_defaults().default_modes())


def _reserved_modes() -> List[AgentMode]:
    """The app-owned modes, appended last, whatever the file said (D9)."""
    return [_defaults().todo_builder_mode()]


def _fallback_default_id(modes: List[AgentMode], available: List[AgentMode]) -> str:
    """The default to answer with when the file's own was missing or refused.

    The built-in document's own ``default`` first (am-02's is ``code``, the mode
    that behaves as the app always has), then the first built-in mode. The
    result is checked against ``available`` — the list actually being answered
    with — because a default that names a mode nobody can select is worse than
    the first selectable one: :func:`resolve_mode` would report a mode that is
    not in the dropdown.
    """
    declared = getattr(_defaults(), "DEFAULT_AGENTS_DOCUMENT", None)
    wanted = ""
    if isinstance(declared, dict) and isinstance(declared.get("default"), str):
        wanted = declared["default"]
    ids = [mode.id for mode in modes]
    for candidate in (wanted, *ids):
        if candidate and any(mode.id == candidate for mode in available):
            return candidate
    return ids[0] if ids else ""


# -- the cache -----------------------------------------------------------------


class _Entry:
    """One workspace's resolved modes, plus what they were read at.

    ``stamp`` is ``(st_mtime_ns, st_size)`` of the modes file, or ``None`` when
    the file is absent — a state worth caching too, so a workspace the seeder
    could not reach pays one ``stat`` per request rather than one per lookup.
    ``names`` is the ``(dirname, filename)`` the entry was read under, so a patch
    of either reserved name invalidates the entry instead of answering from a
    file that is no longer the one being looked up.
    """

    __slots__ = (
        "stamp", "names", "exists", "loaded", "modes", "default", "findings",
    )

    def __init__(
        self,
        stamp: Optional[Tuple[int, int]],
        names: Tuple[str, str],
        exists: bool,
        loaded: bool,
        modes: Tuple[AgentMode, ...],
        default: str,
        findings: Tuple[ModeFinding, ...],
    ) -> None:
        self.stamp = stamp
        self.names = names
        self.exists = exists
        self.loaded = loaded
        self.modes = modes
        self.default = default
        self.findings = findings


# Keyed on the **workspace root**, for the three reasons the policy loader's key
# exists for (``editorfs/policy_file.py``): the working location is hot-swapped,
# two roots routinely hold same-sized config files (and a same-size edit is
# entirely possible), and the test suite patches the location per test. A stamp
# alone cannot tell one workspace's file from another's.
_CACHE: Dict[str, _Entry] = {}

# How many workspaces are remembered. A process switches location rarely, so this
# is a bound against long uptime and a curious operator, not a working set: past
# it the cache is dropped whole, which costs one re-read of a small JSON file and
# never an answer.
_CACHE_MAX_ROOTS = 32


def _active_root() -> str:
    """The working location, fully resolved — so the cache key and the lookup agree."""
    return os.path.realpath(path_location())


def agent_modes_path() -> str:
    """Where this workspace's modes file lives (absolute; it may not exist).

    Built per call out of module attributes, so a patched name or a switched
    location moves the path the same way it moves the cache.
    """
    return os.path.join(_active_root(), POLICY_DIRNAME, AGENT_MODES_FILENAME)


def _file_stamp(path: str) -> Optional[Tuple[int, int]]:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _read_text(path: str) -> Tuple[Optional[str], bool, List[ModeFinding]]:
    """``(text, something_stands_there, findings)`` for one modes file.

    Absence is **not** a finding: a workspace the seeder could not reach (am-03)
    or one that predates the feature is in a normal state and its user did
    nothing wrong. An unreadable file — permissions, a folder standing where the
    file belongs — is an error, because that they can fix. The middle value says
    whether *anything* stands at the path, so a folder or an unreadable file is
    not reported upstream as "there is no modes file here": that answer would
    send the user looking in the wrong place for the object the finding names.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return (handle.read(), True, [])
    except FileNotFoundError:
        return (None, False, [])
    except IsADirectoryError as exc:
        return (None, True, [_file_error(
            path, f"is a folder, not a file: {exc.strerror or exc}")])
    except NotADirectoryError as exc:
        return (None, True, [_file_error(
            path, f"cannot be reached: {exc.strerror or exc}")])
    except UnicodeDecodeError as exc:
        return (None, True, [_file_error(
            path, f"is not UTF-8 text: {exc.reason} (at byte {exc.start})"
        )])
    except OSError as exc:
        return (None, True, [_file_error(
            path, f"cannot be read: {exc.strerror or exc}")])


def _fallback_entry(names: Tuple[str, str], exists: bool,
                    findings: List[ModeFinding]) -> _Entry:
    """What a file that could not be used answers with: the built-in modes (D11).

    Chat over a typo in a file nobody can preview must keep working, so the
    defaults cover every gap and the findings — never a raised exception — say
    what was lost. ``loaded`` is False either way, and the findings are what
    tells an absent file (normal) from a broken one (the user's to fix).
    """
    modes = _fallback_modes()
    available = modes + _reserved_modes()
    return _Entry(
        stamp=None,
        names=names,
        exists=exists,
        loaded=False,
        modes=tuple(available),
        default=_fallback_default_id(modes, available),
        findings=tuple(findings),
    )


def _load(root: str, names: Tuple[str, str]) -> _Entry:
    """Read and validate one workspace's modes file — once per change."""
    dirname, filename = names
    path = os.path.join(root, dirname, filename)
    text, present, findings = _read_text(path)
    if text is None:
        return _fallback_entry(names, present, findings)

    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        findings.append(_file_error(
            path,
            f"is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
        ))
        return _fallback_entry(names, True, findings)

    modes, declared_default, own, usable = _validated_document(document, path)
    findings.extend(own)
    if not usable:
        return _fallback_entry(names, True, findings)

    if not modes:
        # Nothing the file declared survived. The file is fine — its entries are
        # not — so this is not ``loaded: False``; the findings name every entry
        # that was dropped, and the built-in modes keep the conversation working.
        modes = _fallback_modes()
    available = modes + _reserved_modes()
    default = declared_default or ""
    if default not in [mode.id for mode in available]:
        default = _fallback_default_id(modes, available)
    return _Entry(
        stamp=None,
        names=names,
        exists=True,
        loaded=True,
        modes=tuple(available),
        default=default,
        findings=tuple(findings),
    )


def _current() -> _Entry:
    """This workspace's entry, re-read only when its file changed."""
    root = _active_root()
    names = (POLICY_DIRNAME, AGENT_MODES_FILENAME)
    stamp = _file_stamp(os.path.join(root, *names))

    entry = _CACHE.get(root)
    if entry is not None and entry.stamp == stamp and entry.names == names:
        return entry

    entry = _load(root, names)
    entry.stamp = stamp
    if len(_CACHE) >= _CACHE_MAX_ROOTS:
        _CACHE.clear()
    _CACHE[root] = entry
    return entry


def invalidate_agent_modes_cache() -> None:
    """Forget every cached workspace. For tests and for a workspace switch."""
    _CACHE.clear()


# -- the accessors -------------------------------------------------------------

def resolve_modes() -> Dict[str, Any]:
    """Every mode usable now, in dropdown order, plus the file's findings.

    Returns ``{"default": str, "modes": [AgentMode…], "findings":
    [ModeFinding…], "source": {"path": str, "exists": bool, "loaded": bool,
    "version": Optional[int]}}``.

    Nothing about the **file** raises and nothing about it blocks: a missing,
    unreadable, unparseable or invalid file yields the built-in defaults plus the
    findings that say what happened (D11), while a file that loaded but carries
    bad entries yields the entries that were valid. ``source.loaded`` is ``True``
    only when a file was read **and** understood as this version understands it,
    so an absent file and a rejected one both answer ``False`` there and the
    findings — which the UI shows — say which of the two it was.

    ``readme`` is accepted at any shape and never checked: unlike the policy
    file's readme, this one is not rendered by the app (it is prose for the human
    editing the file by hand, and am-06's payload leaves it in the file), so a
    strange shape there costs nothing any surface has to display. A label longer
    than :data:`MAX_MODE_LABEL_CHARS` is trimmed rather than refused for the same
    reason: it is presentation, never correctness.
    """
    entry = _current()
    return {
        "default": entry.default,
        "modes": list(entry.modes),
        "findings": list(entry.findings),
        "source": {
            "path": os.path.join(_active_root(), *entry.names),
            "exists": entry.exists,
            "loaded": entry.loaded,
            "version": AGENT_MODES_VERSION if entry.loaded else None,
        },
    }


def resolve_mode(
    mode_id: Optional[str],
) -> Tuple[Optional[AgentMode], List[ModeFinding]]:
    """One mode by id. ``None``/empty → the file's default. Unknown → ``(None, …)``.

    The caller decides what an unknown id costs: the chat route answers **422**
    naming it (D14), and falling back to the default would answer the user's
    question with a different agent and no clue, so this function deliberately
    does not offer one. ``None`` rather than an exception because the caller is
    the SSE path, where an exception has already lost the client's stream.

    The findings about the *file* ride along, so a caller that reports the mode it
    used can say in the same breath that the file it came from has problems.
    """
    resolved = resolve_modes()
    findings: List[ModeFinding] = resolved["findings"]
    wanted = (mode_id or "").strip() or resolved["default"]
    for mode in resolved["modes"]:
        if mode.id == wanted:
            return (mode, findings)
    return (None, findings)


# -- the seeder (am-03) ---------------------------------------------------------
#
# The modes file arrives the way ``.nojae/read-write-policy.json`` does: written
# once, at the two moments a folder becomes the working location (boot, and the
# settings switch), and never again by the app. ``editorfs/policy_seed.py`` is
# where the reasoning for each of the three non-negotiables is written at length
# — this is the same three, applied to a second file in the same reserved folder:
#
# * **exclusive create** — one ``os.open(..., O_CREAT | O_EXCL)`` is the whole
#   check-and-create, so a hand-edited file is never clobbered and two
#   concurrent switches have one winner. ``EEXIST`` is the ``"existing"`` state,
#   not an error, and there is deliberately no ``isfile()`` guard in front of it
#   (that is a TOCTOU and buys nothing the flag already buys);
# * **``os.mkdir`` for ``.nojae``, one level, not ``makedirs``** — the root
#   exists by contract, and a reserved folder whose parent is missing is a
#   symptom worth failing on rather than a chain to paper over;
# * **never called from ``config.path_location()``** — that is a per-request
#   *reader*, and a reader that writes is how a stray folder turns up in a
#   project someone only meant to browse.
#
# It lives beside the loader rather than in its own module because the two must
# agree on the same bytes: :func:`_seeded_text` renders am-02's document, and the
# test suite asserts that text round-trips through :func:`_load` with no
# findings. A second file would make that pairing a cross-module import for
# nothing — unlike ``policy_seed``, which has a whole package of readers next to
# it and no loader of its own to sit in.

def agent_modes_file_path(root: str) -> str:
    """Where the modes file goes under ``root`` — one level of the reserved folder.

    ``agent_modes_path()`` is the *active* root (what the loader reads); this
    takes the root explicitly, because the switch seeds the folder it is moving
    to before ``path_location()`` is what the rest of the app sees.
    """
    return os.path.join(root, POLICY_DIRNAME, AGENT_MODES_FILENAME)


def _seeded_text() -> str:
    """The bytes the seeder writes — am-02's document, rendered once per call."""
    return _defaults().default_agent_modes_text()


def _state_dict(state: str, path: str, error: Optional[str] = None) -> Dict[str, Any]:
    """The one payload shape: what the settings dialog renders without a syscall.

    Same keys as ``policy_seed``'s answers (``created`` / ``existing`` /
    ``absent`` / ``failed``), so the client-side sentence is the same sentence.
    """
    return {"state": state, "path": path, "error": error}


def _is_regular_file(path: str) -> bool:
    """Whether ``path`` is an ordinary file (a folder there is a finding)."""
    try:
        return stat_module.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def _discard_seeded_file(path: str) -> None:
    """Best-effort removal of the file this call just created."""
    try:
        os.unlink(path)
    except OSError:
        pass


def ensure_agent_modes_file(root: str) -> Dict[str, Any]:
    """Create ``<root>/.nojae/agent-modes.json`` if absent. Never overwrites.

    Returns ``{"state": "created" | "existing" | "failed", "path": str,
    "error": Optional[str]}``.

    The whole function is one ``try``/``except``: a seed that cannot happen
    (read-only mount, a root owned by someone else, no disk) is a **NOTE** for
    the UI, never a reason to refuse opening a project. A workspace with no
    modes file still chats — :func:`resolve_modes` answers with the built-in
    defaults — so the cost of a failed seed is a dropdown nobody can customise,
    and the cost of refusing the switch is somebody else's project staying shut.
    """
    try:
        return _ensure_agent_modes(root)
    except Exception as exc:  # noqa: BLE001 — a seed is a note, never a failure
        return _state_dict("failed", agent_modes_file_path(root),
                           f"{type(exc).__name__}: {exc}")


def _ensure_agent_modes(root: str) -> Dict[str, Any]:
    path = agent_modes_file_path(root)
    folder = os.path.dirname(path)
    try:
        os.mkdir(folder, mode=0o755)
    except FileExistsError:
        pass
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))

    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if not _is_regular_file(path):
            # ``agent_modes._read_text`` reports a folder at this path as a
            # file-level error finding; the seeder says the same thing rather
            # than claiming a modes file that the loader rejects.
            return _state_dict(
                "failed", path, "a folder stands where the agent-modes file belongs"
            )
        return _state_dict("existing", path)
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_seeded_text())
    except OSError as exc:
        # The bytes did not land and the file is ours by construction (no one
        # else's bytes are in it), so it goes: absence is the state the loader
        # reads as "the built-in defaults", where a half-written file would be a
        # finding of its own about a mistake this call made.
        _discard_seeded_file(path)
        return _state_dict("failed", path, exc.strerror or str(exc))

    # The cached entry for this workspace was made a moment ago about a file
    # that did not exist, so forget it: the first read of a freshly seeded
    # workspace must see the seed. This is the same invalidation ``policy_seed``
    # does at the end of its ``_ensure``, and it is *after* the write on
    # purpose — invalidating before the seed would cache "absent" again.
    invalidate_agent_modes_cache()
    return _state_dict("created", path)


def agent_modes_file_state(root: str) -> Dict[str, Any]:
    """What a **reader** may say about the modes file, writing nothing.

    ``GET /api/path-location`` asks this instead of :func:`ensure_agent_modes_file`
    — the settings dialog opens far more often than a workspace is switched, and
    a read that leaves a folder in someone's project is the mistake the seeding
    rules forbid. ``"absent"`` is a normal answer here: a workspace whose seed
    failed, or one that predates the feature, chats on the built-in defaults.
    """
    path = agent_modes_file_path(root)
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return _state_dict("absent", path)
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))
    if not stat_module.S_ISREG(info.st_mode):
        return _state_dict("failed", path, "the agent-modes path is a folder, not a file")
    return _state_dict("existing", path)


# -- self-check ----------------------------------------------------------------

def _self_check() -> None:
    """Fail at import on what the rules above quietly assume.

    A cap typed as ``MAX_MODES = "16"`` would otherwise surface as a ``TypeError``
    inside a loader whose contract is to never raise, and a reserved id that does
    not match the pattern would mean a name reaching the declared path by accident.
    """
    for name in (
        "AGENT_MODES_VERSION",
        "MAX_MODES",
        "MAX_MODE_ID_CHARS",
        "MAX_MODE_LABEL_CHARS",
        "MAX_MODE_SYSTEM_MESSAGE_BYTES",
        "_CACHE_MAX_ROOTS",
    ):
        value = globals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"agent_modes.{name} must be a positive int, got {value!r}")
    for name in ("AGENT_MODES_FILENAME", "POLICY_DIRNAME", "MODE_ID_PATTERN",
                 "DEFAULTS_MODULE_NAME"):
        if not isinstance(globals()[name], str) or not globals()[name]:
            raise ValueError(f"agent_modes.{name} must be a non-empty string")
    pattern = re.compile(MODE_ID_PATTERN)
    if not isinstance(RESERVED_MODE_IDS, tuple) or not all(
        isinstance(item, str) and pattern.match(item) for item in RESERVED_MODE_IDS
    ):
        raise ValueError(
            "agent_modes.RESERVED_MODE_IDS must be a tuple of ids matching "
            f"{MODE_ID_PATTERN}, got {RESERVED_MODE_IDS!r}"
        )
    if set(TOP_LEVEL_KEYS) & set(MODE_KEYS):
        raise ValueError("agent_modes: a top-level key must not also be a mode key")


_self_check()
