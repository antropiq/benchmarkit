"""eds-02: the whole-document **writer** for the workspace's agent-modes file.

:mod:`core.agent_modes` is the read side of ``<working location>/.nojae/agent-modes.json``
and it is deliberately forgiving: a broken modes file must never block a chat turn,
so it answers with usable modes plus findings (am-00 D11). This module is the other
half — the modes editor's save (eds-00 D3/D6/D7/D8) — and a **writer may be strict**:
it refuses what its own reader would drop, because writing a file the reader refuses
is how an edit quietly loses a mode.

Why it is a module of its own rather than a write half inside ``agent_modes.py``:
the pre-save backup lives in ``editorfs.constants.BACKUP_DIR``, and
``tests/test_agent_modes.py`` forbids ``agent_modes.py`` from importing ``editorfs``
(a per-request reader stays free of the Editor's machinery). The seeder stays where
it is — the seeded bytes and the loader must remain one object — so the write half
sits beside them instead of dragging an Editor-side import into the reader.

**Where the backup call lives** (the choice eds-02 has to make and state): the
backup *callable* is **injected** by the caller — ``src/api/routes.py`` hands this
function the Editor's own pre-save backup helper, so one helper, one digest rule and
one retention sweep still cover every file this app replaces, while this module
imports neither ``editorfs`` nor the API layer. That keeps
``tests/test_write_surface_isolation.py::test_the_write_core_is_called_from_the_routes_and_nowhere_else``
green **unmodified**: that scan flags a core file which *names* one of the Editor's
write helpers, docstring included, and an injected callable is what keeps this file
out of its caller list. The alternative — importing the helper here and widening that
scan's allowlist — was rejected because the guard is cheaper to keep than to re-argue.

What a save does, in order (each step has a test beside it):

1. **Validate the body as a writer** (D3's server-side half): every finding the
   loader would produce for this document — an unknown key, a tool flag that is not
   a boolean, a persona over the byte cap, an id outside the pattern, an entry that
   is not an object, more entries than the mode cap, a ``default`` naming no declared
   mode — is a **refusal**, not a drop. The sentences are the loader's own, naming
   the JSON pointer, so the dialog can show what the reader would have said.
2. **Refuse a reserved id** (D4): a body declaring one of
   :data:`core.agent_modes.RESERVED_MODE_IDS` is refused outright. A file that names
   one *by hand* keeps today's shape — the reader answers a note, the save drops the
   entry — because the note is the file's problem, not the body's.
3. **Guard against a stale write** (D7): the body carries the ``mtime`` token the
   document read handed out; a disk ``mtime`` that has moved is a conflict carrying
   **both** numbers, as an object. Nothing is merged and nothing is overwritten.
4. **Refuse to overwrite bytes the reader itself refuses**: a file on disk that is
   not JSON, not an object, whose ``version`` this app does not understand, or whose
   ``agents`` is not an object stays exactly where it is (D3: such a finding holds
   the Save button down; a server that wrote over it would undo that).
5. **Write D6's shape**: ``version`` from the app's constant (never from the body),
   ``readme`` preserved verbatim when the file had one and absent when it had none —
   the built-in one when the save *creates* the file — ``agents`` in the body's
   order, ``default`` from the body. A persona lands as a **list only when no line
   is blank** and as a string otherwise, because a list holding a blank line is
   invalid to this app's own reader. The text itself is never stripped or rewritten.
6. **Back up what it replaces** (D8) through the injected helper, then replace
   atomically (temp file in the same directory + ``os.replace``). A save that
   refused wrote nothing and left no temp file behind.
7. **Create exclusively** when the file was absent: one ``os.mkdir`` level for the
   reserved folder (never ``makedirs``, same rule as the seeder) and
   ``O_CREAT | O_EXCL``, so a file that appeared between the read and the save is a
   conflict rather than something to overwrite.

Then it invalidates the modes cache and answers the **new** ``mtime`` plus the new
verdict in the shape ``GET /api/agent-modes`` publishes, so the composer's dropdown
can reload without a second round-trip (D9).

Like :mod:`core.batch`'s listing errors and unlike the Editor's own exceptions, the
refusals raised here are this module's own typed errors carrying the status their
route should answer — core owns the rule, ``routes.py`` owns the transport, and
nothing in here imports it.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core import agent_modes
from src.core.agent_modes import ModeFinding


# -- the refusals ---------------------------------------------------------------

class AgentModesSaveError(Exception):
    """A save this app refuses, carrying the status its route should answer.

    The status belongs to the core because *which* refusal it is forms the endpoint's
    contract: a stale token and an unwritable document are two different user
    messages, and a route that guessed them would be deciding file rules. Same shape
    as :class:`core.batch.BatchListingError`; :mod:`api.routes` only re-raises it as
    an ``HTTPException``.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class AgentModesDocumentInvalid(AgentModesSaveError):
    """The body is not a document this app can write faithfully (422).

    ``findings`` are the loader's own :class:`ModeFinding` objects rendered to
    ``{"kind", "message"}`` — the shape ``GET /api/agent-modes`` publishes — so the
    dialog can put each sentence next to the field it talks about instead of parsing
    a sentence this module invented.
    """

    def __init__(self, findings: List[ModeFinding]) -> None:
        self.findings = [{"kind": finding.kind, "message": finding.message}
                         for finding in findings]
        super().__init__(422, "; ".join(item["message"] for item in self.findings)
                         or "the document names nothing to save")
        self.detail_object: Dict[str, Any] = {"reason": "invalid",
                                              "findings": self.findings}


class AgentModesSaveConflict(AgentModesSaveError):
    """The 409 a moved ``mtime`` answers — carrying **both** numbers.

    Mirrors :class:`editorfs.WorkspaceSaveConflict`'s shape on purpose: ``detail`` is
    a sentence for a log, :attr:`detail_object` is what goes on the wire, because a
    conflict has to be *acted on* (the dialog stays open and names it) rather than
    read. Both numbers are strings: an ``st_mtime_ns`` does not survive a JSON float
    round-trip, and this is the same token :func:`core.agent_modes.read_agent_modes_document`
    handed out. ``None`` on either side means that side had no file to stamp.
    """

    def __init__(self, path: str, disk_mtime: Optional[str],
                 document_mtime: Optional[str]) -> None:
        super().__init__(
            409,
            f"{path} changed on disk since it was read "
            f"(disk {disk_mtime}, document {document_mtime})",
        )
        self.detail_object: Dict[str, Any] = {
            "reason": "stale",
            "path": path,
            "diskMtime": disk_mtime,
            "documentMtime": document_mtime,
        }


class AgentModesWriteRefused(AgentModesSaveError):
    """The bytes could not be put on disk (the write family's usual codes).

    Distinct from the two refusals above: those are about *what* to write and are the
    user's to fix, while this one is about the disk itself — a folder that cannot be
    created, a path that cannot be written, a write that failed halfway.
    """


# -- the body, validated as a writer --------------------------------------------


def _normalised_body(document: Any) -> Any:
    """The body with every persona in the JSON form this writer would put on disk.

    The document read reports a persona **in whichever form the file wrote it** (D2:
    verbatim, never joined), so the body can come back holding a list — including a
    list whose text contains a blank line, which is exactly what the dialog's textarea
    produces when the user writes a paragraph break. Validating that shape as it came
    would refuse it (the loader drops a list holding a blank entry) and then D6's rule
    — "the string form when a line is blank" — could never apply: the writer would be
    refusing the case it exists to write.

    So the **form** is settled first and the **text** is what gets validated, which has
    the useful consequence that validation runs over the document that is about to be
    written rather than over a different one. A list carrying a value that is not a
    string is left alone: no form choice can turn it into text, and the loader's own
    sentence is the refusal the dialog should show.
    """
    if not isinstance(document, dict):
        return document
    agents = document.get("agents")
    if not isinstance(agents, dict):
        return document

    entries: Dict[str, Any] = {}
    for mode_id, entry in agents.items():
        if isinstance(entry, dict) and "systemMessage" in entry:
            value = entry["systemMessage"]
            if isinstance(value, str) or (
                isinstance(value, list)
                and all(isinstance(line, str) for line in value)
            ):
                entries[mode_id] = {**entry, "systemMessage": _persona_for_disk(value)}
                continue
        entries[mode_id] = entry
    return {**document, "agents": entries}


def _reserved_entry_findings(document: Any) -> List[ModeFinding]:
    """D4's refusal: a **body** may not declare a mode benchmarkit owns.

    The reader answers this case with a *note* about the file and ignores the entry,
    and that stays right: a workspace that wrote one by hand loses nothing, and the
    save is what cleans it up. A body is a different thing — the dialog never shows
    the app-owned mode, so an entry for it here was not edited, it was invented. The
    sentence is the loader's own, minus "this entry is ignored", which is not what
    happens to a body.
    """
    agents = document.get("agents") if isinstance(document, dict) else None
    if not isinstance(agents, dict):
        return []
    return [
        agent_modes._pointer_error(
            f"agents.{mode_id}",
            f"\"{mode_id}\" is defined by benchmarkit and cannot be set here",
        )
        for mode_id in agents
        if mode_id in agent_modes.RESERVED_MODE_IDS
    ]


def _body_findings(document: Any, path: str) -> List[ModeFinding]:
    """Why this body cannot be written faithfully — or ``[]`` when it can.

    The validator is the **loader's own** :func:`core.agent_modes._validated_document`,
    so every refusal is the sentence the reader would have produced for the same
    entry, naming the same pointer. Two deliberate differences:

    * ``version`` is replaced by the app's constant before validating, because D6
      says the body's version is never written — refusing it would be a rule about a
      value this app has already decided to ignore;
    * every persona is validated **in the form the save will write** (see
      :func:`_normalised_body`), so what is refused is the text, never the JSON shape
      the dialog happened to send it in — and what is validated is the document that
      is about to be written.
    * every finding of kind ``error`` is a refusal here, including the ones the reader
      *repairs* (the overflow past the mode cap, a ``default`` naming no declared
      mode). A reader must keep the conversation working; a writer must not commit a
      document it would then have to repair.

    Notes are never refusals: nothing was dropped, and the reserved-id case is handled
    by :func:`_reserved_entry_findings` instead.
    """
    body = _normalised_body(document)
    candidate: Any = document
    if isinstance(body, dict):
        candidate = {**body, "version": agent_modes.AGENT_MODES_VERSION}

    _modes, _default, findings, usable = agent_modes._validated_document(candidate, path)
    refusals = [finding for finding in findings if finding.kind == agent_modes.KIND_ERROR]
    if not usable:
        # The vocabulary itself is what the body got wrong (root not an object,
        # ``agents`` not an object): nothing read through it can be trusted.
        return refusals
    return refusals + _reserved_entry_findings(body)


def _token_of(token: Any) -> Optional[int]:
    """The body's ``mtime`` token as the integer the read handed out, or ``None``.

    ``None`` is a real answer — it is what the document read publishes for a file that
    is absent, and it is what tells a save to *create* rather than to guard. Anything
    else that cannot be read as a number is the client's bug, said as a refusal naming
    the field rather than a silent "no expectation".
    """
    if token is None:
        return None
    if isinstance(token, str):
        try:
            return int(token.strip())
        except ValueError:
            pass
    elif isinstance(token, int) and not isinstance(token, bool):
        # ``bool`` is an ``int`` in Python, and ``true`` is not a timestamp — the same
        # trap the loader calls out for ``version``.
        return token
    raise AgentModesDocumentInvalid([agent_modes._pointer_error(
        "mtime",
        "cannot be read as an mtime token: it is the file's timestamp, as a number",
    )])


def _as_token(value: Optional[int]) -> Optional[str]:
    """One ``st_mtime_ns`` in the spelling a save compares against."""
    return None if value is None else str(value)


def _guard(path: str, disk_mtime: Optional[int], token: Optional[int]) -> None:
    """D7: the disk has to be the file the body was read from, or nothing is written.

    The comparison is exact (both sides are integer nanoseconds, no float round-trip),
    so it needs no epsilon — and a token of ``None`` against a file that exists is a
    conflict, not a free pass: the dialog read an absent file and someone else made
    one since.
    """
    if disk_mtime == token:
        return
    raise AgentModesSaveConflict(path, _as_token(disk_mtime), _as_token(token))


# -- the file on disk, read for what the writer must preserve -------------------

def _disk_document(path: str) -> Tuple[Optional[int], Optional[Any]]:
    """``(st_mtime_ns, parsed document)`` for the modes file, refusing what is unusable.

    Only the **file-level** verdict matters here, and it decides one thing: whether
    this app may write over these bytes at all. Entry-level problems are not looked at
    — the dialog fixes those in the form and the save replaces the entry whole — while
    a file that is not JSON, not an object, whose ``version`` this app does not
    understand, or whose ``agents`` is not an object stays exactly as it is. The
    findings quoted are the loader's own, so the refusal says what the dialog already
    showed the user.
    """
    stamp = agent_modes._file_stamp(path)
    mtime_ns = None if stamp is None else stamp[0]

    text, _present, findings = agent_modes._read_text(path)
    if findings:
        raise AgentModesDocumentInvalid(findings)
    if text is None:
        return (mtime_ns, None)  # absent: this save will create the file

    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentModesDocumentInvalid([agent_modes._file_error(
            path,
            f"is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
        )]) from exc

    _modes, _default, own, usable = agent_modes._validated_document(document, path)
    if not usable:
        raise AgentModesDocumentInvalid(
            [finding for finding in own if finding.kind == agent_modes.KIND_ERROR]
        )
    return (mtime_ns, document)


def _readme_to_write(disk_document: Optional[Any]) -> Tuple[bool, Any]:
    """D6's ``readme`` rule as a decision: ``(write_it, value)``.

    * the file had the key → preserve it **verbatim**, ``null`` included — it is prose
      for the human editing the file by hand, and this dialog is not its author;
    * the file had no such key → write none. An absent ``readme`` and a ``null`` one
      stay two different documents, which is why the document read reports them apart;
    * there was no file at all → the built-in readme, the same lines the seeder writes,
      because a created file is a new hand-editor's only explanation of the vocabulary.
    """
    if disk_document is None:
        declared = agent_modes._document_defaults().get("readme")
        return (declared is not None, declared)
    if "readme" in disk_document:
        return (True, disk_document["readme"])
    return (False, None)


# -- the persona's JSON form ----------------------------------------------------

def _persona_for_disk(value: Any) -> Any:
    """One validated ``systemMessage`` in the JSON form the reader accepts back.

    The **list form only when no line is blank**, the string form otherwise. That is
    not a style preference: the loader reads a list of *non-blank* strings and drops
    the whole entry for a list holding a blank one, so a writer that kept the list form
    for text containing an empty line would commit a file its own reader refuses — the
    exact bug the round-trip tests exist to catch. "Blank" means what the loader means
    (``line.strip()`` empty), so a line of spaces forces the string form too rather
    than being quietly deleted.

    The text is never stripped, joined or otherwise rewritten: the two forms produce
    byte-identical text through the loader, and that is the only rewriting allowed. A
    ``null`` persona (the documented "use the configured prompt") passes through as it
    came, and an empty one lands as the string ``""`` — never as ``[]``, which the
    loader refuses as an empty list.
    """
    if not isinstance(value, (str, list)):
        return value  # null: the key says "use LLM_SYSPROMPT", and it keeps saying that
    text = "\n".join(value) if isinstance(value, list) else value
    lines = text.split("\n")
    if lines and all(line.strip() for line in lines):
        return lines
    return text


def _document_text(body: Dict[str, Any], readme: Tuple[bool, Any]) -> str:
    """The bytes to write: D6's shape, in the order the built-in document uses.

    ``agents`` is written from the **body**, in the body's order — which is the
    dialog's order and therefore the dropdown's order — because validation already
    refused every entry the loader would have dropped. Values go out as they came
    except the persona, whose form is chosen by :func:`_persona_for_disk`. A label is
    written as the form holds it: the loader's trim-and-cap is what it applies when it
    *reads*, and a writer that pre-trimmed someone's text would be rewriting bytes the
    dialog never asked it to touch.
    """
    document: Dict[str, Any] = {"version": agent_modes.AGENT_MODES_VERSION}
    write_readme, readme_value = readme
    if write_readme:
        document["readme"] = readme_value
    document["default"] = body.get("default")

    agents: Dict[str, Any] = {}
    for mode_id, entry in body["agents"].items():
        fields: Dict[str, Any] = {}
        for key in agent_modes.MODE_KEYS:
            if key not in entry:
                continue
            fields[key] = (
                _persona_for_disk(entry[key]) if key == "systemMessage" else entry[key]
            )
        agents[mode_id] = fields
    document["agents"] = agents

    # ``ensure_ascii=False`` for the same reason the seeder renders that way: this file
    # stays editable by hand outside the app, and the personas are real prose.
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


# -- the two writes -------------------------------------------------------------

def _fsync_parent(path: str) -> None:
    """Durability hint for the directory whose entry just changed.

    The same reasoning as the Editor's atomic replace: a rename leaves a new *entry*
    behind, and that is what has to survive a crash. It is spelled here rather than
    imported because this module keeps its distance from ``editorfs`` (see the module
    docstring); failure is not a failure — the mutation already happened.
    """
    parent = os.path.dirname(path)
    try:
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _discard_temp(temp_path: str) -> None:
    """Best-effort removal of the half-written copy this call made."""
    try:
        os.unlink(temp_path)
    except OSError:
        pass


def _replace_atomically(path: str, payload: bytes) -> None:
    """``payload`` into a temp file in the target's directory, then rename over.

    Same-directory is the rule: a temp on another filesystem would turn
    ``os.replace`` into a copy, which is neither atomic nor cheap. Both fsyncs matter —
    the file's bytes, then the parent directory's entry. The temp never outlives a
    failure: it holds a full copy of the document and no one will ever look for it.
    """
    def refused(exc: OSError) -> AgentModesWriteRefused:
        return AgentModesWriteRefused(
            500, f"the agent-modes file {path} could not be written: {exc.strerror or exc}"
        )

    try:
        handle = tempfile.NamedTemporaryFile(
            dir=os.path.dirname(path),
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        )
    except OSError as exc:  # the folder itself will not take a file
        raise refused(exc) from exc

    temp_path = handle.name
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        # The bytes did not land where the reader looks for them, so the half-written
        # copy goes with them: the modes file keeps the bytes it had, and a stray temp
        # file in the reserved folder is something the user would have to wonder about.
        _discard_temp(temp_path)
        raise refused(exc) from exc
    except BaseException:
        _discard_temp(temp_path)
        raise
    _fsync_parent(path)


def _create_exclusively(path: str, payload: bytes) -> None:
    """The absent case: one ``os.mkdir`` level, then ``O_CREAT | O_EXCL``.

    The same three non-negotiables as the seeder's ``_ensure_agent_modes``: the
    exclusive open *is* the check-and-create, so a file that appeared between the
    document read and this save is a conflict rather than something to overwrite; the
    reserved folder is created one level deep, because a root whose parent is missing
    is a symptom worth failing on rather than a chain to paper over; and a write that
    failed halfway leaves no half-written file, because absence is the state the loader
    reads as "the built-in defaults" while a fragment would be a finding about a
    mistake this call made.
    """
    folder = os.path.dirname(path)
    try:
        os.mkdir(folder, mode=0o755)
    except FileExistsError:
        pass
    except OSError as exc:
        raise AgentModesWriteRefused(
            500, f"the reserved folder {folder} cannot be created: {exc.strerror or exc}"
        ) from exc

    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        # The race the exclusive open exists to catch: the read said "absent", and a
        # file is there now. Its bytes are someone else's and they stay.
        raise AgentModesSaveConflict(
            path, _as_token(_disk_mtime_ns(path)), None
        ) from exc
    except OSError as exc:
        raise AgentModesWriteRefused(
            500, f"the agent-modes file {path} cannot be created: {exc.strerror or exc}"
        ) from exc

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise AgentModesWriteRefused(
            500, f"the agent-modes file {path} could not be written: {exc.strerror or exc}"
        ) from exc
    _fsync_parent(path)


def _disk_mtime_ns(path: str) -> Optional[int]:
    """The disk's current ``st_mtime_ns``, or ``None`` when nothing stands there."""
    stamp = agent_modes._file_stamp(path)
    return None if stamp is None else stamp[0]


# -- the save -------------------------------------------------------------------

def save_agent_modes_document(
    document: Any,
    mtime_token: Any = None,
    *,
    backup: Callable[[Path], Any],
) -> Dict[str, Any]:
    """Write the whole modes document, or refuse without touching the disk.

    ``document`` is the body as parsed JSON — the dialog's entries in the dialog's
    order plus its ``default``; its ``version`` and its ``readme`` are **ignored** (D6:
    the version is the app's constant, the readme belongs to the file on disk).
    ``mtime_token`` is the token :func:`core.agent_modes.read_agent_modes_document`
    published (``None`` for a file that was absent). ``backup`` is the caller's pre-save
    backup callable — the Editor's own, handed over by the save route, which is what
    keeps one backup folder, one digest rule and one retention sweep for every file this
    app replaces (see the module docstring for why it is injected rather than imported).

    Returns ``{"saved": True, "created": bool, "backedUp": bool, "modes": [...],
    "default": str, "findings": [{"kind", "message"}…], "source": {..., "mtime": str}}``
    — the new verdict in the shape ``GET /api/agent-modes`` publishes (D9: the dropdown
    reloads from this answer instead of a second round-trip) plus the **new** ``mtime``
    token, so a dialog that stays open has a current one.

    Raises :class:`AgentModesDocumentInvalid` (422), :class:`AgentModesSaveConflict`
    (409) or :class:`AgentModesWriteRefused`; every refusal happens before a byte is
    written, and a failed write leaves no temp file and no half-written one.

    The modes cache is invalidated on a successful write — the step the seeder takes
    after its own — because the dropdown's cached verdict was made about the bytes this
    save just replaced.
    """
    path = agent_modes.agent_modes_path()

    findings = _body_findings(document, path)
    if findings:
        raise AgentModesDocumentInvalid(findings)

    token = _token_of(mtime_token)
    _guard(path, _disk_mtime_ns(path), token)

    _mtime_ns, disk_document = _disk_document(path)
    payload = _document_text(document, _readme_to_write(disk_document)).encode("utf-8")

    if disk_document is None and _disk_mtime_ns(path) is None:
        _create_exclusively(path, payload)
        created, backed_up = True, False
    else:
        # D8: the bytes this save replaces go into the app's backup chain first.
        backed_up = backup(Path(path)) is not None
        _replace_atomically(path, payload)
        created = False

    agent_modes.invalidate_agent_modes_cache()
    resolved = agent_modes.resolve_modes()
    return {
        "saved": True,
        "created": created,
        "backedUp": backed_up,
        "modes": [
            {
                "id": mode.id,
                "label": mode.label,
                "mcptools": mode.mcptools,
                "lsptools": mode.lsptools,
                "editable": mode.editable,
            }
            for mode in resolved["modes"]
        ],
        "default": resolved["default"],
        "findings": [{"kind": finding.kind, "message": finding.message}
                     for finding in resolved["findings"]],
        "source": {**resolved["source"],
                   "mtime": _as_token(_disk_mtime_ns(path))},
    }
