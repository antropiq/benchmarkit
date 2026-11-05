"""The WorkBench prompt helpers: nine pre-written prompts, one per workspace.

``<workspace>/.nojae/prompt-helpers.json`` holds the text behind the coloured
bullets above the composer — each slot a colour and a prompt the user reaches
for often. This module is the whole read and write side of that file: the
constants, the path, the loader, the one-slot writer, the defaults and the
error type, plus the seeder and the one-time lift from the old app-level file.
It talks to no route, no model and no browser; the routes (mbl-03) wrap
:func:`load_prompt_helpers` and :func:`save_prompt_helper`, and the seeder
lives here for the reason :mod:`core.agent_modes` gives for the same choice — a
seeder has to write exactly the bytes its loader reads, so the two belong in
one file.

Two things this deliberately is **not**. It is not cached: ``agent_modes``
caches per root because it is consulted on every chat turn, while this file is
read when the chat tab opens and when the working location changes — nine small
strings do not earn a cache and its invalidation rules. And it is not in
``config.py``: that module sits at the bottom of the dependency graph
(``editorfs.policy`` and ``agent_modes`` import ``path_location`` *from* it),
so a module that reads ``.nojae`` cannot live there without an import cycle.

The file joins ``.nojae`` but not that folder's "edited only outside the app"
rule (spec D1): it is workspace *state the app owns*, not workspace config the
user declares, because the bullet picker **is** the feature. What stays
reserved is the folder, not the editing rights.
"""

import json
import os
import re
import stat as stat_module
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import PROJECT_ROOT, path_location

# The reserved folder's name is not re-declared here: it comes from
# ``mcp_connectors/policy_shared``, with the same ``sys.path`` fallback
# :mod:`core.agent_modes` uses, because that module lives outside ``src`` so
# the stdio connectors can read it too. Two constants for one folder name is
# how the folder stops being one folder.
try:
    from mcp_connectors import policy_shared as shared
except ImportError:  # pragma: no cover - depends on how the app is started
    _root = str(PROJECT_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from mcp_connectors import policy_shared as shared  # type: ignore[no-redef]

POLICY_DIRNAME: str = shared.POLICY_DIRNAME

# -- the constants (module attributes, patchable) -------------------------------

#: The helpers file's name, beside ``read-write-policy.json`` and
#: ``agent-modes.json`` in the reserved folder. A module attribute so a test
#: can move the lookup — the repo's ``mcphub.constants`` convention.
PROMPT_HELPERS_FILENAME = "prompt-helpers.json"

#: The only ``version`` this file is understood at; a mismatch is an error,
#: mirroring the other two files in the folder (a different version means a
#: different vocabulary, and guessing at one is how a slot stops existing).
PROMPT_HELPERS_VERSION = 1

#: How many slots the file holds. This is a **layout** contract, not a
#: preference: the bullet row is drawn as nine buttons, so a document with a
#: different count is answered as an error rather than repaired (spec D4).
SLOT_COUNT = 9

#: The colour an unfilled bullet is drawn in, and the cap on one slot's text.
DEFAULT_COLOR = "#808080"
CONTENT_MAX_CHARS = 32000

#: What counts as a colour: a 6-digit hex triplet, so a bad value falls back to
#: the gray above instead of reaching CSS as an invalid colour.
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class PromptHelpersFileError(Exception):
    """The helpers file exists but cannot be used, so nothing was answered.

    Carries the sentence the route turns into a 422: it names the path, and
    when the complaint is the slot count it names the count found. Raised
    rather than collected as a finding, because a wrong count is the one thing
    here that cannot be papered over: the app is the file's only writer, so a
    list of eight or ten entries means a hand edit or corruption, and padding
    or truncating it would overwrite the user's bytes to hide that.
    """


# -- the path ------------------------------------------------------------------

def prompt_helpers_file_path(root: str) -> str:
    """Where the helpers file goes under ``root`` — one level of the reserved folder.

    ``prompt_helpers_path()`` is the *active* root (what the loader reads); this
    takes the root explicitly, because the workspace switch seeds the folder it
    is moving to before ``path_location()`` is what the rest of the app sees.
    """
    return os.path.join(root, POLICY_DIRNAME, PROMPT_HELPERS_FILENAME)


def prompt_helpers_path() -> str:
    """Where this workspace's helpers file lives (absolute; it may not exist).

    Built per call out of module attributes, so a patched filename or a
    switched working location moves the lookup the same way.
    """
    return prompt_helpers_file_path(path_location())


# -- the defaults --------------------------------------------------------------

def default_prompt_helpers() -> List[Dict[str, str]]:
    """The fresh-state document: nine empty gray slots, nothing configured.

    Empty on purpose (spec D4): shipping example prompts would put one
    project's habits into every other project's tree, which is the app-level
    scoping this file exists to remove. A user's own prompts arrive through the
    one-time lift from the old app-level file.
    """
    return [{"color": DEFAULT_COLOR, "content": ""} for _ in range(SLOT_COUNT)]


def default_prompt_helpers_text() -> str:
    """The document a workspace is seeded with, rendered from the literal.

    Same shape as ``policy_seed.default_policy_text()`` and
    ``agent_modes_defaults.default_agent_modes_text()``: one ``json.dumps`` of
    the Python literal, indented with a trailing newline, so the bytes a seeder
    writes *are* the object the loader accepts rather than a second copy that
    happens to agree with it.
    """
    document = {
        "version": PROMPT_HELPERS_VERSION,
        "helpers": default_prompt_helpers(),
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


# -- seeding and the one-time lift (mbl-02) ------------------------------------

#: The app-level file this feature replaces. Module attribute (the
#: ``mcphub.constants`` convention) so a test can move the lookup — it must
#: never be opened for real by a test, which would be reading a gitignored file
#: of the developer's machine.
#:
#: Read **once**, at the moment a workspace is seeded (spec D6), and left in
#: place afterwards: no delete, no ``os.replace``, no ``.nojae/trash``. Nothing
#: else in the app reads this path again, and :func:`load_prompt_helpers` has no
#: fallback to it — one lookup, one truth.
LEGACY_PROMPT_HELPERS_PATH: Path = (
    PROJECT_ROOT / "resources" / "memory_registration.json"
)


def _state_dict(state: str, path: str, error: Optional[str] = None) -> Dict[str, Any]:
    """The one payload shape: what the app bar renders without a syscall.

    Same keys as ``policy_seed``'s and ``agent_modes``' answers (``created`` /
    ``existing`` / ``absent`` / ``failed``), so the client-side sentence is the
    same sentence — plus the two states only this file answers (``migrated`` /
    ``migration-skipped``).
    """
    return {"state": state, "path": path, "error": error}


def _is_regular_file(path: str) -> bool:
    """Whether ``path`` is an ordinary file (a folder there is a failure)."""
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


def _legacy_slots_or_none() -> Optional[List[Any]]:
    """The old app-level document's ``registration`` list, if there is a usable one.

    ``None`` means "there is no legacy file to lift": absent, unreadable, not
    JSON, or no ``registration`` list. There is deliberately no state for a file
    that was never a helper store (spec D6), and no error — a boot that cannot
    read its own install directory's leftover is not somebody else's project's
    problem.
    """
    try:
        document: Any = json.loads(
            LEGACY_PROMPT_HELPERS_PATH.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    raw = document.get("registration")
    return raw if isinstance(raw, list) else None


def _seeded_text() -> str:
    """The bytes the seeder writes — the same text the loader reads cleanly."""
    return default_prompt_helpers_text()


def _lift_legacy_helpers(path: str) -> Optional[Dict[str, Any]]:
    """Copy the old app-level slots into the file this call just created.

    Reached **only** on the branch where the exclusive create succeeded, which
    is what makes the lift one-shot: a workspace that already has its file —
    seeded last boot, or hand-edited — is never opened by the migration, so
    there is no clobbering and no second copy (spec D6).

    Answers the state dict for ``migrated`` / ``migration-skipped``, or ``None``
    when there is nothing to lift and the caller should report ``created``.
    """
    raw = _legacy_slots_or_none()
    if raw is None:
        return None
    if len(raw) != SLOT_COUNT:
        # Not migrated, and said out loud. D4's no-repair rule gets the stricter
        # version here: the legacy list is copied only when it holds exactly
        # nine entries, because the old path is never read again, so a silent
        # pad or truncate during the lift would stay invisible forever.
        return _state_dict(
            "migration-skipped",
            path,
            f"the legacy file {LEGACY_PROMPT_HELPERS_PATH} holds {len(raw)} "
            f"entries, expected exactly {SLOT_COUNT}; the new file keeps the "
            "default slots",
        )
    slots = [_normalize_slot(item) for item in raw]
    try:
        Path(path).write_text(_document_text(slots), encoding="utf-8")
    except OSError as exc:
        # Same rule as the seed's own failed write: the file is ours by
        # construction, so it goes rather than staying as a half-written
        # document, and the legacy copy is still untouched either way.
        _discard_seeded_file(path)
        return _state_dict(
            "failed", path, f"could not write the migrated helpers file: "
            f"{exc.strerror or exc}"
        )
    return _state_dict("migrated", path)


def ensure_prompt_helpers_file(root: str) -> Dict[str, Any]:
    """Create ``<root>/.nojae/prompt-helpers.json`` if absent. Never overwrites.

    Returns ``{"state": "created" | "existing" | "migrated" |
    "migration-skipped" | "failed", "path": str, "error": Optional[str]}``.

    The whole function is one ``try``/``except``: a seed that cannot happen
    (read-only mount, a root owned by someone else, no disk) is a **NOTE** for
    the UI, never a reason to refuse opening a project. A workspace with no
    helpers file still chats — :func:`load_prompt_helpers` answers the nine
    empty slots — so the cost of a failed seed is nine blank bullets.
    """
    try:
        return _ensure_prompt_helpers(root)
    except Exception as exc:  # noqa: BLE001 — a seed is a note, never a failure
        return _state_dict("failed", prompt_helpers_file_path(root),
                           f"{type(exc).__name__}: {exc}")


def _ensure_prompt_helpers(root: str) -> Dict[str, Any]:
    path = prompt_helpers_file_path(root)
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
            # The loader reports a folder at this path as its own error; the
            # seeder says the same thing rather than claiming a helpers file
            # that the loader would reject. The legacy path is not opened here:
            # an existing file means this workspace was seeded already, and the
            # lift belongs to that moment only.
            return _state_dict(
                "failed", path, "a folder stands where the prompt-helpers file belongs"
            )
        return _state_dict("existing", path)
    except PermissionError:
        # Which error a **folder** at that path answers with is the platform's
        # choice: ``EEXIST`` on POSIX, ``EACCES`` on Windows — the exclusive open
        # cannot create over a directory there. Same condition, same sentence, so
        # the seeder says what stands there rather than quoting an errno.
        if not _is_regular_file(path):
            return _state_dict(
                "failed", path, "a folder stands where the prompt-helpers file belongs"
            )
        return _state_dict("failed", path, "the prompt-helpers path cannot be written")
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_seeded_text())
    except OSError as exc:
        # The bytes did not land and the file is ours by construction, so it
        # goes: absence is the state the loader reads as "the nine default
        # slots", where a half-written file would be a finding of its own about
        # a mistake this call made.
        _discard_seeded_file(path)
        return _state_dict("failed", path, exc.strerror or str(exc))

    # No cache to invalidate, unlike ``agent_modes._ensure_agent_modes``: this
    # file is read on a tab switch and on a workspace switch, and nine small
    # strings do not earn a cache and its invalidation rules (spec D5).
    lifted = _lift_legacy_helpers(path)
    if lifted is not None:
        return lifted
    return _state_dict("created", path)


def prompt_helpers_file_state(root: str) -> Dict[str, Any]:
    """What a **reader** may say about the helpers file, writing nothing.

    ``GET /api/path-location`` asks this instead of :func:`ensure_prompt_helpers_file`
    — the working location is read far more often than a workspace is switched,
    and a read that leaves a folder in someone's project is the mistake the
    seeding rules forbid. ``"absent"`` is a normal answer here: a workspace whose
    seed failed, or one that predates the feature, chats on nine empty bullets.
    """
    path = prompt_helpers_file_path(root)
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return _state_dict("absent", path)
    except OSError as exc:
        return _state_dict("failed", path, exc.strerror or str(exc))
    if not stat_module.S_ISREG(info.st_mode):
        return _state_dict("failed", path, "the prompt-helpers path is a folder, not a file")
    return _state_dict("existing", path)


# -- the read and write sides --------------------------------------------------

def _normalize_slot(item: Any) -> Dict[str, str]:
    """Coerce one raw JSON entry to a ``{color, content}`` string pair.

    A missing or malformed color (anything but a 6-digit hex) falls back to
    the default gray so rendering never receives an invalid CSS color; a
    non-string content becomes empty.
    """
    raw_color = item.get("color") if isinstance(item, dict) else None
    raw_content = item.get("content") if isinstance(item, dict) else None
    color = (
        raw_color
        if isinstance(raw_color, str) and COLOR_RE.match(raw_color)
        else DEFAULT_COLOR
    )
    content = raw_content if isinstance(raw_content, str) else ""
    return {"color": color, "content": content}


def _document_text(slots: List[Dict[str, str]]) -> str:
    """The bytes this module writes: the versioned document, one shape only.

    The on-disk key is ``helpers`` (spec section 4). The route's payload key is
    ``slots`` (spec D8) — two different keys for two different surfaces, and
    the conversion belongs to the route, not to here.
    """
    document = {
        "version": PROMPT_HELPERS_VERSION,
        "helpers": slots,
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _write_prompt_helpers(slots: List[Dict[str, str]]) -> None:
    """Persist the normalized slots, creating one level for the reserved folder.

    The folder create is not the seeder's exclusive ``O_CREAT | O_EXCL``: this is
    a plain ``os.mkdir`` for the folder the file is about to live in, reached only
    from a **write** moment. It is here because ``POST /api/prompt-helpers`` on a
    workspace whose seed failed (or that predates the feature) must still save the
    slot the user typed — refusing the save over a missing folder would make the
    bullet picker depend on a boot-time step it cannot see. A reader never gets
    this far: :func:`load_prompt_helpers` returns the defaults on an absent file
    without calling this at all.
    """
    path = prompt_helpers_path()
    try:
        os.mkdir(os.path.dirname(path), mode=0o755)
    except FileExistsError:
        pass
    Path(path).write_text(_document_text(slots), encoding="utf-8")


def load_prompt_helpers() -> List[Dict[str, str]]:
    """Read this workspace's nine slots, writing nothing unless a field was coerced.

    Answers for each state of the file:

    * **absent** — the nine default slots, and no folder is created. A reader
      that writes is how a stray ``.nojae`` turns up in a project someone only
      meant to browse (the argument already written in ``policy_seed.py`` and
      ``agent_modes.py``); the seeding moments own the create.
    * **unreadable, not JSON, a missing or unknown ``version``, ``helpers`` not
      a list, or a list that does not hold exactly :data:`SLOT_COUNT`
      entries** — :class:`PromptHelpersFileError`, naming the path and the
      count. Never padded, never truncated, never rewritten: the bytes stay as
      they were, because they are the evidence of what went wrong.
    * **nine entries, one malformed field** — the normalized list, and the file
      rewritten. That is the one write on a read, and it is deliberate:
      coercing a colour in memory only would show a gray bullet whose stored
      value is ``bogus``, so the next save would re-commit the bad field.
      Field coercion is idempotent and the app is the only writer, so this
      write cannot contradict anyone.
    """
    path = prompt_helpers_path()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return default_prompt_helpers()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PromptHelpersFileError(
            f"the prompt-helpers file {path} cannot be read: "
            f"{getattr(exc, 'strerror', None) or exc}"
        ) from exc

    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PromptHelpersFileError(
            f"the prompt-helpers file {path} is not valid JSON: "
            f"{exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(document, dict):
        raise PromptHelpersFileError(
            f"the prompt-helpers file {path} is not a JSON object"
        )

    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        # ``bool`` is an ``int`` in Python, and ``true`` is not a version — the
        # same trap the modes loader calls out (``agent_modes._version_findings``).
        raise PromptHelpersFileError(
            f"the prompt-helpers file {path} has a version that is not a "
            f"number ({version!r}), expected {PROMPT_HELPERS_VERSION}"
        )
    if version != PROMPT_HELPERS_VERSION:
        raise PromptHelpersFileError(
            f"the prompt-helpers file {path} declares version {version!r}, "
            f"expected {PROMPT_HELPERS_VERSION}"
        )

    raw = document.get("helpers")
    if not isinstance(raw, list):
        raise PromptHelpersFileError(
            f"the prompt-helpers file {path} has no 'helpers' list"
        )
    if len(raw) != SLOT_COUNT:
        raise PromptHelpersFileError(
            f"the prompt-helpers file {path} holds {len(raw)} helpers, "
            f"expected exactly {SLOT_COUNT}"
        )

    slots = [_normalize_slot(item) for item in raw]
    if any(raw[index] != slots[index] for index in range(SLOT_COUNT)):
        _write_prompt_helpers(slots)
    return slots


def save_prompt_helper(index: int, color: str, content: str) -> List[Dict[str, str]]:
    """Write one slot and return the full normalized set of nine.

    No length guard here, and none is needed: :func:`load_prompt_helpers` has
    already raised on a document that does not hold exactly nine entries, and
    this replaces one element of that list, so it cannot emit another length
    (spec D4). A guard would be code for a state the file's only writer never
    produces.
    """
    slots = load_prompt_helpers()
    slots[index] = {"color": color, "content": content}
    _write_prompt_helpers(slots)
    return slots


# -- self-check ----------------------------------------------------------------

def _self_check() -> None:
    """Fail at import on what the rules above quietly assume.

    A ``SLOT_COUNT`` that no longer matches the row the UI draws, or a
    ``DEFAULT_COLOR`` that its own ``COLOR_RE`` rejects, are mistakes a later
    edit can make silently — and each one costs every workspace the bullets it
    shows.
    """
    for name in ("PROMPT_HELPERS_VERSION", "SLOT_COUNT", "CONTENT_MAX_CHARS"):
        value = globals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"prompt_helpers.{name} must be a positive int, got {value!r}"
            )
    for name in ("PROMPT_HELPERS_FILENAME", "POLICY_DIRNAME", "DEFAULT_COLOR"):
        value = globals()[name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"prompt_helpers.{name} must be a non-empty string, got {value!r}"
            )
    if not COLOR_RE.match(DEFAULT_COLOR):
        raise ValueError(
            f"prompt_helpers.DEFAULT_COLOR {DEFAULT_COLOR!r} does not match "
            "prompt_helpers.COLOR_RE"
        )
    # The count is pinned to the layout rather than checked against itself:
    # ``default_prompt_helpers()`` is built from ``SLOT_COUNT``, so comparing the
    # two proves nothing. The bullet row iterates its own mirrored constant
    # (``web/src/composables``), so raising here is the reminder that the row and
    # the file move together — nine buttons, nine slots (spec D4).
    if SLOT_COUNT != 9:
        raise ValueError(
            f"prompt_helpers.SLOT_COUNT is {SLOT_COUNT}, but the WorkBench bullet "
            "row draws nine slots — change the row and its own constant too"
        )


_self_check()
