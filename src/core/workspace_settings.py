"""The workspace's model connection: one JSON file per working location.

``<working location>/.nojae/nojae-settings.json`` holds everything a model call
needs — server, key, model id, system prompt, sampling, the vision flag, the
activated language servers and the tool-loop cap. Before this file those values
lived in the install's ``.env``, so both tabs talked to whatever the last save
left there and a workspace could not travel with its own model the way
``.nojae/mcp-settings.json`` travels with its connectors. This module is the
whole read and write side of that file: the constants, the path, the seeder, the
cached loader, the writer, the typed accessor, the member-set and type check and
the error type. It talks to no route, no model and no browser; the routes wrap
:func:`load_workspace_settings`, :func:`save_workspace_settings` and
:func:`workspace_settings_state`, and the seeder lives here for the reason
:mod:`core.agent_modes` and :mod:`core.prompt_helpers` both give — a seeder has
to write exactly the bytes its loader reads.

Two things this deliberately is **not**. It is not ``config.py``: that module
sits at the bottom of the dependency graph (``editorfs.policy`` and
``agent_modes`` import ``path_location`` *from* it), so a module that reads
``.nojae`` cannot live there without an import cycle. And it is not hot on a
hand-edit: the file is read at boot and on a workspace switch, and
:func:`invalidate_workspace_settings` is the only way a cached answer changes
(spec D4) — exactly ``.env``'s existing contract, and why the Settings dialog
says a hand-edit needs a restart.

The file joins ``.nojae``, and it is that folder's second store the app itself
writes (``prompt-helpers.json`` being the first): the Settings dialog writes
group 1 here, so the folder's "edited only outside the app" rule bends for it
the way it already bends for the prompt-helper store. What stays reserved is the
folder — hidden from the Editor and from the connectors, never from its writer
(spec D3, D5).

The two failure rules are why this module checks the file itself instead of
handing the mapping to ``EndpointsConfig`` and trusting it:

* **a malformed file is a named state, never a silent default** (D9) — broken
  JSON, an unknown member or a member of the wrong JSON type answers ``failed``
  with a sentence naming the path and the member. ``extra="ignore"`` would
  swallow ``"temperture"`` and benchmark a workspace at the default temperature;
* **a value that is still the template's instruction sentence is reported, not
  dialled** (D8) — a freshly seeded workspace is valid JSON and unusable, so the
  reader answers ``ok`` but *not configured*, and the caller says "not
  configured" with the file's path instead of making a transport call with a
  sentence.
"""

import json
import os
import stat as stat_module
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from src.config import PROJECT_ROOT, path_location

# The reserved folder's name is not re-declared here: it comes from
# ``mcp_connectors/policy_shared``, with the same ``sys.path`` fallback
# :mod:`core.prompt_helpers` uses, because that module lives outside ``src`` so
# the stdio connectors can read it too. Two constants for one folder name is how
# the folder stops being one folder.
try:
    from mcp_connectors import policy_shared as shared
except ImportError:  # pragma: no cover - depends on how the app is started
    _root = str(PROJECT_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from mcp_connectors import policy_shared as shared  # type: ignore[no-redef]

POLICY_DIRNAME: str = shared.POLICY_DIRNAME

# -- the constants (module attributes, patchable) -------------------------------

#: The settings file's name, beside ``mcp-settings.json`` and
#: ``agent-modes.json`` in the reserved folder. A module attribute so a test can
#: move the lookup — the repo's ``mcphub.constants`` convention.
WORKSPACE_SETTINGS_FILENAME = "nojae-settings.json"

#: The committed template the seed copies byte for byte, at the install root
#: beside ``.env.example``: it is the committed template of a settings file, not
#: a connector declaration (which is why it does not live in ``mcp_connectors``).
#: A module attribute for the same reason as the filename — the test that pins a
#: missing template moves it.
WORKSPACE_SETTINGS_TEMPLATE: Path = PROJECT_ROOT / "nojae-settings.default.json"

#: The approved member set, each with the JSON type it must hold (spec section
#: 2's key map). This is the one rule both sides consult: the reader refuses a
#: member outside it or of the wrong type (D9), and the writer refuses to
#: introduce one, so a caller cannot put on disk the typo the loader exists to
#: catch.
#:
#: ``float`` here means "a JSON number" and so accepts ``2`` as well as ``2.0``;
#: ``int`` means a JSON integer and rejects ``true`` (which Python reads as an
#: ``int``) as well as ``2.5``. ``lspEnabled`` is a list of strings because an
#: array that could hold prose would let a sentence become a language-server id.
WORKSPACE_SETTINGS_KEYS: Dict[str, type] = {
    "serverUrl": str,
    "serverApiKey": str,
    "endpoint": str,
    "supportsVision": bool,
    "sysprompt": str,
    "temperature": float,
    "topP": float,
    "topK": int,
    "minP": float,
    "repeatPenalty": float,
    "presencePenalty": float,
    "reasoning": str,
    "lspEnabled": list,
    "maxToolIteration": int,
}

#: The members whose template value is an instruction sentence rather than a
#: value (D8). ``sysprompt`` and ``reasoning`` are absent here on purpose: their
#: template text is a real, usable value, while these two cannot be sent
#: anywhere.
_SEED_TEXT_MEMBERS: Tuple[str, ...] = ("serverUrl", "endpoint")

#: The JSON word for each approved type, used by the sentences that name one.
_TYPE_WORDS: Dict[type, str] = {
    str: "a string",
    bool: "a boolean",
    float: "a number",
    int: "an integer",
    list: "an array of strings",
}


class WorkspaceSettingsFileError(Exception):
    """The workspace settings file cannot be used, or cannot be written.

    Carries the sentence naming the path and, when the complaint is about a
    member, the member and the type it expected. Raised by the **writer**
    (:func:`save_workspace_settings`), because a save that would leave an
    unusable file behind must not report success. The **reader** does not raise:
    it answers the same sentence as the ``error`` of
    :func:`workspace_settings_state`, because a broken file is something the
    dialog displays and a chat or benchmark request refuses with — not an
    exception to unwind.
    """


# -- the path ------------------------------------------------------------------

def workspace_settings_file_path(root: str) -> str:
    """Where the settings file goes under ``root`` — one level of the reserved folder.

    ``workspace_settings_path()`` is the *active* root (what the loader reads);
    this takes the root explicitly, because the workspace switch seeds the folder
    it is moving to before ``path_location()`` is what the rest of the app sees.
    """
    return os.path.join(root, POLICY_DIRNAME, WORKSPACE_SETTINGS_FILENAME)


def workspace_settings_path() -> str:
    """Where this workspace's settings file lives (absolute; it may not exist).

    Built per call out of module attributes, so a patched filename or a
    switched working location moves the lookup the same way.
    """
    return workspace_settings_file_path(path_location())


# -- the messages --------------------------------------------------------------

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


def _member_names() -> str:
    """The approved members, listed the way a message points at them."""
    return ", ".join(f"\"{name}\"" for name in WORKSPACE_SETTINGS_KEYS)


# -- the one validation rule (D9) ----------------------------------------------

def _value_matches(expected: type, value: Any) -> bool:
    """Whether ``value`` may stand for a member whose declared type is ``expected``.

    Strict on purpose: ``null`` is not a value a member of this file may hold,
    ``1`` is not what the user means where a boolean belongs, and a list of ids
    holds strings only. There is no coercion here because the point of native
    types (D6) is that what the file holds is what reaches pydantic — coercing
    at the door would be the dotenv wart, moved rather than removed.
    """
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        # ``True`` is an ``int`` in Python and is not an integer here — the trap
        # every other loader in the folder calls out by name.
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    if expected is list:
        return isinstance(value, list) and all(
            isinstance(item, str) and item.strip() for item in value
        )
    return False  # pragma: no cover - the table holds the five types above


def _problem_of(payload: Mapping[str, Any]) -> Optional[str]:
    """Why ``payload`` is not a usable set of members, or ``None`` when it is.

    One sentence at a time, in a fixed order (unknown members first, then the
    approved ones in table order), because the sentence goes into a state field
    and into a refusal: a list of everything wrong is for a linter with a
    preview pane, and this file has none — the user reads one sentence, fixes
    it, retries.
    """
    unknown = sorted(name for name in payload if name not in WORKSPACE_SETTINGS_KEYS)
    if unknown:
        listed = ", ".join(f"\"{name}\"" for name in unknown)
        return f"has an unknown member {listed}; it understands {_member_names()}"
    for name, expected in WORKSPACE_SETTINGS_KEYS.items():
        if name in payload and not _value_matches(expected, payload[name]):
            wanted = _TYPE_WORDS.get(expected, str(expected))
            return f"holds \"{name}\" as {_kind_of(payload[name])}, expected {wanted}"
    return None


def _mergeable_members(payload: Any) -> Dict[str, Any]:
    """The members of a current file that a save may carry over.

    Lenient by design, and that is the one place this module prefers repair to
    refusal: when the file on disk is broken (the JSON does not parse, a member
    is misspelled) the dialog still has to be able to *fix* it, because the file
    is invisible to the Editor and a hand-edit means leaving the app. So a save
    keeps whatever it can read of the current members and drops the rest, which
    turns a typo the reader refuses into a file the reader accepts.
    """
    if not isinstance(payload, dict):
        return {}
    return {
        name: value
        for name, value in payload.items()
        if name in WORKSPACE_SETTINGS_KEYS
        and _value_matches(WORKSPACE_SETTINGS_KEYS[name], value)
    }


# -- the template's instruction text (D8) --------------------------------------

def default_workspace_settings_text() -> str:
    """The template's text, read fresh through the module attribute.

    The one place that touches the template for the seed. Read at call time (no
    cache, no copy at import) so a patched ``WORKSPACE_SETTINGS_TEMPLATE`` — the
    test that pins a missing template — moves the lookup like every other path
    here.
    """
    return WORKSPACE_SETTINGS_TEMPLATE.read_text(encoding="utf-8")


def _template_instruction_values() -> Dict[str, str]:
    """The members whose template text is an instruction, mapped to that text.

    Read from the template rather than re-typed into a constant, so the sentence
    the seed writes and the sentence the reader matches cannot drift apart (D8).
    An unreadable or unparseable template answers ``{}``, which means "no seed
    text can be recognised": that is the install's own breakage, and it is not
    allowed to answer a real workspace's file as "not configured" on a guess.
    """
    try:
        document: Any = json.loads(default_workspace_settings_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    return {
        name: document[name]
        for name in _SEED_TEXT_MEMBERS
        if isinstance(document.get(name), str) and document[name]
    }


def _configured_of(path: str, payload: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    """``(configured, message)`` for a file that parsed cleanly (D8)."""
    instructions = _template_instruction_values()
    stale = [
        name
        for name in _SEED_TEXT_MEMBERS
        if instructions.get(name) and payload.get(name) == instructions[name]
    ]
    if not stale:
        return (True, None)
    listed = ", ".join(f"\"{name}\"" for name in stale)
    return (
        False,
        f"the workspace settings file {path} still holds the template's text "
        f"for {listed}; set them in the Settings dialog, or edit that file and "
        "restart the server",
    )


# -- the cache -----------------------------------------------------------------

class _Entry:
    """One workspace's parsed file, plus what may be said about it.

    A workspace's answer is cached per **resolved path**, so a switch to another
    location reads its own file rather than the previous one's. Unlike the modes
    and the policy loaders there is deliberately **no file stamp**: a hand-edit
    does not take effect until a restart (D4), so re-reading on a change is the
    behaviour this feature rejects, and :func:`invalidate_workspace_settings` is
    the only thing that changes an answer.
    """

    __slots__ = ("state", "path", "error", "configured", "message", "payload")

    def __init__(
        self,
        state: str,
        path: str,
        error: Optional[str] = None,
        configured: bool = False,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state = state
        self.path = path
        self.error = error
        self.configured = configured
        self.message = message
        self.payload: Dict[str, Any] = payload or {}


# How many workspaces are remembered. A process switches location rarely, so
# this is a bound against long uptime and a curious operator, not a working
# set: past it the cache is dropped whole, which costs one re-read of a small
# JSON file and never an answer.
_CACHE_MAX_PATHS = 32

_CACHE: Dict[str, _Entry] = {}


def invalidate_workspace_settings() -> None:
    """Forget every cached workspace's settings.

    Called by a settings save, by the seed once it has created a file, and by
    the working-location switch (spec D4) — and nowhere else. A hand-edit is not
    such a moment, which is why it waits for a restart.
    """
    _CACHE.clear()


# -- the read side -------------------------------------------------------------

def _is_regular_file(path: str) -> bool:
    """Whether ``path`` is an ordinary file (a folder there is a failure)."""
    try:
        return stat_module.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def _load(path: str) -> _Entry:
    """Read, check and interpret one workspace's settings file — once per change."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        # A reader never writes: creating the folder here is how a stray
        # ``.nojae`` turns up in a project someone only meant to browse, and the
        # seeding moments own the create.
        return _Entry(
            "absent",
            path,
            message=f"there is no workspace settings file at {path} yet",
        )
    except (OSError, UnicodeDecodeError) as exc:
        if not _is_regular_file(path):
            return _Entry(
                "failed", path, "the workspace settings path is a folder, not a file"
            )
        return _Entry(
            "failed",
            path,
            f"the workspace settings file {path} cannot be read: "
            f"{getattr(exc, 'strerror', None) or exc}",
        )

    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        return _Entry(
            "failed",
            path,
            f"the workspace settings file {path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})",
        )

    if not isinstance(document, dict):
        return _Entry(
            "failed",
            path,
            f"the workspace settings file {path} holds {_kind_of(document)}, "
            "not an object of members",
        )

    problem = _problem_of(document)
    if problem is not None:
        return _Entry("failed", path, f"the workspace settings file {path} {problem}")

    configured, message = _configured_of(path, document)
    return _Entry("ok", path, None, configured, message, dict(document))


def _current() -> _Entry:
    """This workspace's entry, re-read only when the cache was invalidated."""
    path = workspace_settings_path()
    entry = _CACHE.get(path)
    if entry is not None:
        return entry
    entry = _load(path)
    if len(_CACHE) >= _CACHE_MAX_PATHS:
        _CACHE.clear()
    _CACHE[path] = entry
    return entry


def load_workspace_settings() -> Dict[str, Any]:
    """This workspace's members as stored, with native JSON types (D6).

    A file that parses answers its mapping; anything else answers ``{}`` and
    says why through :func:`workspace_settings_state`, which the routes read in
    the same breath:

    * **absent** — ``{}``, state ``absent``, and nothing written;
    * **not valid JSON, not an object, an unknown member, or a member of the
      wrong JSON type** — ``{}``, state ``failed``, the sentence naming the path
      and the member. Never a silent default (D9): a caller that benchmarked on
      defaults here would measure the defaults, not the model.

    A member being absent here means "absent", which is not the same as "off" or
    "empty": the spec's key map documents what each member absent means, and the
    caller passes its own default (:func:`workspace_setting`).
    """
    return dict(_current().payload)


def workspace_setting(name: str) -> Any:
    """One member as the file stores it — ``None`` when the file does not hold it.

    The typed accessor group 1's readers use (``config.lsp_enabled_ids()`` and
    ``config.max_tool_iteration()``). It answers the native value — ``True``,
    ``0.2``, ``["pyright"]`` — and performs no coercion and no defaulting: what
    "absent" means differs per member and belongs to the caller, so a caller
    that wants a default passes its own (spec section 5).
    """
    return _current().payload.get(name)


def workspace_settings_state() -> Dict[str, Any]:
    """What may be said about this workspace's settings file, writing nothing.

    ``{"state": "ok" | "absent" | "failed", "path": str, "error":
    Optional[str], "configured": bool, "message": Optional[str]}`` — the
    folder's own state vocabulary (``prompt_helpers._state_dict``), so the dialog
    has one shape to switch on.

    ``configured`` is the D8 flag: ``True`` only when the file is usable **and**
    its ``serverUrl``/``endpoint`` are no longer the template's instruction text.
    ``message`` carries the sentence naming the path for the two states where
    the file is unusable without being broken (absent, still-seed text), while
    ``error`` carries it for ``failed`` — so a refusal always has one of the two
    to quote, and both name the file the user has to fix.
    """
    entry = _current()
    return {
        "state": entry.state,
        "path": entry.path,
        "error": entry.error,
        "configured": entry.configured,
        "message": entry.message,
    }


# -- the write side ------------------------------------------------------------

def _read_current(path: str) -> Any:
    """The current document, for the merge — ``None`` when there is nothing to merge.

    Absent, unreadable or unparseable all answer ``None``, which makes the save a
    write of exactly the members it was given: the file is broken or missing and
    the dialog is how the user fixes it (see :func:`_mergeable_members`).
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def save_workspace_settings(values: Mapping[str, Any]) -> None:
    """Write the given members to this workspace's file, keeping the others.

    The keys are the **JSON member names**: this module knows nothing about the
    Settings dialog's field names, and the dialog-field → member table lives in
    the routes (D7) — one mapping table, not a rewrite of the dialog.

    Only the members it is given are written; valid members it was not given are
    carried over from the file as it stands, so a partial save cannot blank a
    setting the dialog did not show. Then the cache is invalidated, which is what
    makes a dialog save live in the same request (D4).

    Raises :class:`WorkspaceSettingsFileError` — and writes nothing — when a key
    is not an approved member or a value is not of its member's JSON type: the
    same rule the reader applies, so a caller cannot put on disk the file the
    reader would refuse. An unwritable file raises the same error, because a save
    that did not land must not be reported as one that did.
    """
    problem = _problem_of(values)
    if problem is not None:
        raise WorkspaceSettingsFileError(
            f"the workspace settings save {problem} (nothing was written)"
        )

    path = workspace_settings_path()
    merged = _mergeable_members(_read_current(path))
    merged.update(values)
    text = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"

    try:
        os.mkdir(os.path.dirname(path), mode=0o755)
    except FileExistsError:
        pass
    except OSError as exc:
        raise WorkspaceSettingsFileError(
            f"the workspace settings file {path} cannot be written: "
            f"{exc.strerror or exc}"
        ) from exc

    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise WorkspaceSettingsFileError(
            f"the workspace settings file {path} cannot be written: "
            f"{exc.strerror or exc}"
        ) from exc

    invalidate_workspace_settings()


# -- seeding (exclusive create, one input) -------------------------------------

def _state_dict(state: str, path: str, error: Optional[str] = None) -> Dict[str, Any]:
    """The one payload shape: what the app bar renders without a syscall.

    Same keys as ``policy_seed``'s, ``agent_modes``' and ``mcphub.seed``'s answers
    (``created`` / ``existing`` / ``failed``), so the client-side sentence is the
    same sentence for every file in the reserved folder.
    """
    return {"state": state, "path": path, "error": error}


def _discard_seeded_file(path: str) -> None:
    """Best-effort removal of the file this call just created."""
    try:
        os.unlink(path)
    except OSError:
        pass


def ensure_workspace_settings_file(root: str) -> Dict[str, Any]:
    """Create ``<root>/.nojae/nojae-settings.json`` from the template if absent.

    Never overwrites: a workspace's own server, key and sampling survive every
    later seed byte for byte, exactly as a hand-edited ``mcp-settings.json`` does.

    Returns ``{"state": "created" | "existing" | "failed", "path": str, "error":
    Optional[str]}``. The whole function is one ``try``/``except``: a seed that
    cannot happen (read-only mount, a root owned by someone else, no disk, no
    template) is a **NOTE** for the UI, never a reason to refuse opening a
    project — and never a hub failure, which is why the switch seeds before the
    language-server restart (D10) yet still cannot make one fail.

    One input, one behaviour (D11): the template's bytes. There is no branch that
    seeds from ``.env``'s group-1 values, because "the file is a copy of the
    template, then yours" has to stay a sentence that is always true.
    """
    try:
        return _ensure_workspace_settings(root)
    except Exception as exc:  # noqa: BLE001 — a seed is a note, never a failure
        return _state_dict("failed", workspace_settings_file_path(root),
                           f"{type(exc).__name__}: {exc}")


def _ensure_workspace_settings(root: str) -> Dict[str, Any]:
    # The template first, the destination second: a template that cannot be read
    # must leave no half-seeded file behind at the workspace.
    try:
        data = default_workspace_settings_text().encode("utf-8")
    except OSError as exc:
        return _state_dict(
            "failed", str(WORKSPACE_SETTINGS_TEMPLATE),
            f"the workspace settings template is not readable: "
            f"{exc.strerror or exc}",
        )

    path = workspace_settings_file_path(root)
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
            # The loader reads a folder at this path as its own failure; the
            # seeder says the same thing rather than claiming a settings file the
            # loader would reject.
            return _state_dict(
                "failed", path,
                "a folder stands where the workspace settings file belongs"
            )
        return _state_dict("existing", path)
    except PermissionError:
        # Which error a **folder** at that path answers with is the platform's
        # choice: ``EEXIST`` on POSIX, ``EACCES`` on Windows — the exclusive open
        # cannot create over a directory there. Same condition, same sentence, so
        # the seeder says what stands there rather than quoting an errno.
        if not _is_regular_file(path):
            return _state_dict(
                "failed", path,
                "a folder stands where the workspace settings file belongs"
            )
        return _state_dict("failed", path, "the workspace settings path cannot be written")
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except OSError as exc:
        # The bytes did not land and the file is ours by construction, so it
        # goes: absence is the state the loader reads as "no settings yet", where
        # a half-written file would be a finding of its own about a mistake this
        # call made.
        _discard_seeded_file(path)
        return _state_dict("failed", path, exc.strerror or str(exc))

    # The cached entry for this workspace was made about a file that did not
    # exist, so forget it: the first read of a freshly seeded workspace must see
    # the seed. *After* the write on purpose — invalidating before the seed would
    # cache ``absent`` again.
    invalidate_workspace_settings()
    return _state_dict("created", path)


# -- self-check ----------------------------------------------------------------

def _self_check() -> None:
    """Fail at import on what the rules above quietly assume.

    The template and the member table are the same list written twice, on purpose
    (one is committed prose an operator reads, the other is the code's rule), so
    the check that they still agree is the whole point of this function: a member
    added to one and not the other is a file that seeds fine and then reports
    itself malformed, or a typo accepted silently.
    """
    for name in ("WORKSPACE_SETTINGS_FILENAME", "POLICY_DIRNAME"):
        value = globals()[name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"workspace_settings.{name} must be a non-empty string, got {value!r}"
            )
    if not isinstance(WORKSPACE_SETTINGS_KEYS, dict) or not WORKSPACE_SETTINGS_KEYS:
        raise ValueError(
            "workspace_settings.WORKSPACE_SETTINGS_KEYS must be a non-empty "
            f"member-to-type mapping, got {WORKSPACE_SETTINGS_KEYS!r}"
        )
    drifted = [
        f"{member}={declared!r}"
        for member, declared in WORKSPACE_SETTINGS_KEYS.items()
        if declared not in _TYPE_WORDS
    ]
    if drifted:
        raise ValueError(
            "workspace_settings.WORKSPACE_SETTINGS_KEYS holds a type "
            f"workspace_settings._TYPE_WORDS cannot name: {', '.join(drifted)}"
        )

    template = WORKSPACE_SETTINGS_TEMPLATE
    if not isinstance(template, Path):
        raise ValueError(
            f"workspace_settings.WORKSPACE_SETTINGS_TEMPLATE must be a Path, "
            f"got {template!r}"
        )
    try:
        raw: Any = json.loads(template.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            f"workspace_settings.WORKSPACE_SETTINGS_TEMPLATE ({template}) must "
            f"exist and parse as JSON: {getattr(exc, 'strerror', None) or exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"workspace_settings.WORKSPACE_SETTINGS_TEMPLATE ({template}) must "
            f"hold a JSON object, got {_kind_of(raw)}"
        )
    if set(raw) != set(WORKSPACE_SETTINGS_KEYS):
        missing = sorted(set(WORKSPACE_SETTINGS_KEYS) - set(raw))
        extra = sorted(set(raw) - set(WORKSPACE_SETTINGS_KEYS))
        raise ValueError(
            f"workspace_settings.WORKSPACE_SETTINGS_TEMPLATE ({template}) and "
            "workspace_settings.WORKSPACE_SETTINGS_KEYS must name the same "
            f"members: template only {extra}, table only {missing}"
        )
    problem = _problem_of(raw)
    if problem is not None:
        raise ValueError(
            f"workspace_settings.WORKSPACE_SETTINGS_TEMPLATE ({template}) "
            f"{problem} — the seed must write what the loader accepts"
        )
    # The two members whose template value is a real value rather than an
    # instruction (spec section 2's note): a seeded workspace is expected to run
    # agent turns out of the box, so these two are pinned here and not left to
    # the general type check above.
    max_iteration = raw["maxToolIteration"]
    if isinstance(max_iteration, bool) or not isinstance(max_iteration, int):
        raise ValueError(
            "workspace_settings.WORKSPACE_SETTINGS_TEMPLATE's "
            f"\"maxToolIteration\" must be an integer, got {max_iteration!r}"
        )
    if max_iteration < 1:
        raise ValueError(
            "workspace_settings.WORKSPACE_SETTINGS_TEMPLATE's "
            f"\"maxToolIteration\" must be at least 1, got {max_iteration}"
        )
    if not isinstance(raw["lspEnabled"], list):
        raise ValueError(
            "workspace_settings.WORKSPACE_SETTINGS_TEMPLATE's \"lspEnabled\" "
            f"must be a list, got {raw['lspEnabled']!r}"
        )


_self_check()

