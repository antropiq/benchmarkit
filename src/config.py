"""Application configuration: .env loading, questions, and endpoint config."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

from src.core.models import DEFAULT_LLM_SYSPROMPT, EndpointsConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
RESOURCES_DIR = PROJECT_ROOT / "resources"

QUESTION_SET_DEFAULT = "legacy"

# Cap on tool-calling iterations for one chat reply; shared by MCP connectors
# and LSP language servers alike (one knob, no split-brain).
DEFAULT_MAX_TOOL_ITERATION = 25

# Managed .env keys, in the order they are written to the file.
SETTING_KEYS: Tuple[str, ...] = (
    "SERVER_URL",
    "SERVER_URL_API_KEY",
    "ENDPOINT",
    "LLM_SYSPROMPT",
    "TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "REPEAT_PENALTY",
    "PRESENCE_PENALTY",
    "REASONING",
    "LLM_DEBUG",
    "SUPPORTS_VISION",
    "QUESTION_SET",
    "VERIFIER_AGENT_SYSPROMPT",
    "VERIFIER_AGENT_SERVERURL",
    "VERIFIER_AGENT_SERVERURL_API_KEY",
    "VERIFIER_AGENT_ENDPOINT",
    "MAX_TOOL_ITERATION",
    "LSP_ENABLED",
    "BENCHMARKIT_PYTHON_ROOT",
)

# Names an operating system already owns, which a managed key may therefore
# never take.
#
# This joins the rule AGENTS.md states about a new or renamed key — edit
# ``SETTING_KEYS``, ``_render_env()`` and ``.env.example`` together — with a
# fourth demand: the name must be free of OS variables on *every* supported
# platform. ``load_dotenv_once`` below deliberately keeps dotenv's "existing
# environment wins" precedence, so a key that names an OS variable does not set
# a value at all: the OS value survives and the app reads something it never
# wrote. That is the whole ``TEMP`` story — on Windows ``TEMP`` is the temp
# folder, so a sampling parameter became a directory path (and a stray folder
# appeared in the working location). The check lives on the key list rather
# than inside the loader, because that precedence is the point and not a bug to
# paper over: the collision is prevented by never choosing such a name.
_RESERVED_ENV_NAMES: frozenset[str] = frozenset(
    {
        "TEMP",
        "TMP",
        "PATH",
        "HOME",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "SYSTEMROOT",
        "COMSPEC",
        "PROMPT",
        "OS",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMFILES",
        "COMPUTERNAME",
        "HOMEDRIVE",
        "HOMEPATH",
        "NUMBER_OF_PROCESSORS",
        "PATHEXT",
    }
)

# Keys always written to .env; the others are commented out when empty.
REQUIRED_SETTING_KEYS: Tuple[str, ...] = ("SERVER_URL", "ENDPOINT")

# Environment variable holding the *working location* the MCP connectors and
# language servers are pointed at. Both declaration files reference it
# (``--root-dir`` / ``workspaceRoot`` as ``${BENCHMARKIT_PATH_LOCATION}``), so
# one value moves every file-touching tool together.
#
# Deliberately NOT a managed ``SETTING_KEYS`` key: it is process state, not
# run configuration, and keeping it out of ``.env`` means ``save_settings``
# (which regenerates that file from a fixed template) can neither erase it nor
# resurrect a stale value at the next boot. The boot default below is the
# single persistence rule: a restart always lands on the install directory.
PATH_LOCATION_KEY = "BENCHMARKIT_PATH_LOCATION"


# Debug switch for the in-house LLM client. When set, ``benchmarkit.llm`` —
# the logger ``core/llmkit`` writes its request payloads to — is raised to
# DEBUG and its output reaches the server console, so a failed call can be read
# off that console: which URL was POSTed, with what payload. The *status and
# upstream body* of a refused call reach the UI's error line whether or not the
# flag is on (they are in the exception message itself); what the flag adds on
# top is the request payload.
#
# The value lives in ``.env`` (a managed ``SETTING_KEYS`` key) rather than in
# code so it can be flipped from the settings dialog, and it is applied to the
# live process at startup and on every save — no restart needed.
LLM_DEBUG_KEY = "LLM_DEBUG"

# The name this switch had while the calls went through litellm. It lives here
# only as the fallback in ``llm_debug_enabled`` (the precedent
# ``max_tool_iteration`` sets with ``MAX_MCP_ITERATION``), so a hand-written
# ``.env`` keeps working until the dialog rewrites that file from the template.
LEGACY_LLM_DEBUG_KEY = "LITELLM_DEBUG"

TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def env_flag(key: str) -> bool:
    """True when an env var is set to a truthy value (1/true/yes/on)."""
    return (os.environ.get(key) or "").strip().lower() in TRUTHY_VALUES


def normalize_env_flag(raw: Optional[str]) -> str:
    """Normalise a flag-ish submitted value to the ``.env`` form ("1" or "").

    Used by the settings API for the boolean-ish keys: any truthy spelling
    (1/true/yes/on) becomes the canonical ``1``, and anything else — including
    the explicit off spellings (0/false/no/off) and an empty field — becomes
    "", which ``save_settings`` writes as a commented-out line.
    """
    return "1" if (raw or "").strip().lower() in TRUTHY_VALUES else ""


def llm_debug_enabled() -> bool:
    """True when the in-house LLM client's debug logging was requested.

    ``LLM_DEBUG`` is the key; the legacy name above is still honoured while the
    new key is unset, so a ``.env`` written by hand before the rename keeps
    working until the settings dialog regenerates that file from its template.
    The new key wins when both are set.
    """
    # *Which* name answers, then one truthiness rule for both: the new key is
    # the one asked while it holds something, so an explicit "0" under it means
    # off even while the legacy name still says on.
    key = (
        LLM_DEBUG_KEY
        if (os.environ.get(LLM_DEBUG_KEY) or "").strip()
        else LEGACY_LLM_DEBUG_KEY
    )
    return env_flag(key)


class _DebugStreamHandler(logging.StreamHandler):
    """The one handler the debug switch owns, marked by its own class.

    Identified by class rather than remembered on the logger, so "exactly one of
    ours" holds by construction: an attach clears every handler of this class it
    finds, whoever attached it. Boot plus one attach per settings save would
    otherwise be free to stack one per save, printing each payload again.
    """


def _attach_debug_handler(logger: logging.Logger) -> None:
    """Give the client's logger exactly one stderr handler.

    Raising the level alone would log nothing: uvicorn configures handlers for
    its own loggers only, so a ``benchmarkit.*`` record has nowhere to go. While
    the switch is on the logger therefore carries a handler of its own, with
    ``propagate`` off so a deployment that *does* configure a root logger is not
    printed twice. ``core/llmkit`` deliberately attaches no handler, so this is
    the one place those payload lines reach the server console from — and the
    fresh one per call is what binds that console as it is now (a replaced
    ``sys.stderr`` included) rather than one built long before.
    """
    _detach_debug_handler(logger)
    handler = _DebugStreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def _detach_debug_handler(logger: logging.Logger) -> None:
    """Remove every handler of ours and restore propagation.

    Clears by class, so it also removes what an earlier boot attached: the
    switch never leaves more than one behind, and its off path is a no-op on a
    logger it never touched (``propagate`` stays as its other owner left it).
    """
    mine = [h for h in logger.handlers if isinstance(h, _DebugStreamHandler)]
    for handler in mine:
        logger.removeHandler(handler)
    if mine:
        logger.propagate = True


def apply_llm_debug() -> bool:
    """Raise the in-house LLM client's logger to DEBUG when ``LLM_DEBUG`` is on.

    Called at app startup (right after the .env is read) and again on every
    settings save, so flipping the switch takes effect without restarting the
    server. The off path is just as explicit: the level goes back to ``NOTSET``
    — which inherits WARNING from the root logger, exactly where it stood
    before the switch was ever on — and the handler comes off, so a process
    never stays chatty (nor silent) after the flag is cleared.

    What the flag adds is the *request payload* of every model call: the URL,
    the model, and the body as it goes out. Never the ``Authorization`` header,
    so the API key cannot reach the console from here. The status and upstream
    body of a *refused* call need no flag at all: they are in the exception
    message the two consuming loops emit into the UI's error line.

    It also points ``core/llmkit``'s debug hook at this predicate, so payload
    logging follows the switch: ``llmkit`` is core and must not import the
    layer that owns the setting, while ``config`` sits below both. Returns
    whether the switch is now on.
    """
    # Imported lazily and from the layer below (never from ``api``): this module
    # is also imported by code paths that never call an LLM.
    from src.core.llmkit import client as llmkit_client
    from src.core.llmkit import constants as llmkit_constants

    enabled = llm_debug_enabled()
    logger = logging.getLogger(llmkit_constants.DEBUG_LOGGER_NAME)
    if enabled:
        _attach_debug_handler(logger)
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.NOTSET)
        _detach_debug_handler(logger)
    llmkit_client.debug_enabled = llm_debug_enabled
    return enabled


def load_dotenv_once() -> None:
    """Load the .env file once; existing environment variables take precedence."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def default_path_location() -> str:
    """The install directory, used as the working-location boot default."""
    return str(PROJECT_ROOT)


def setup_path_location() -> None:
    """Ensure ``PATH_LOCATION_KEY`` is set, defaulting it to the install dir.

    Called at startup, *after* ``load_dotenv_once``: an exported variable (or
    one written in ``.env``) wins, which is what lets an operator pin the
    location without touching the code. Never overwrites, so it stays
    idempotent across the test suite and repeated app factories.
    """
    current = (os.environ.get(PATH_LOCATION_KEY) or "").strip()
    if not current:
        os.environ[PATH_LOCATION_KEY] = default_path_location()


def path_location() -> str:
    """Effective working location (the value the declaration files expand to).

    Reads ``os.environ`` at call time, so a value set through the API is what
    the next hub re-scan sees. Falls back to the boot default rather than
    returning empty, so the settings dialog always shows a real path.
    """
    return (os.environ.get(PATH_LOCATION_KEY) or "").strip() or default_path_location()


# Values the app provides for ``${NAME}`` references in the hub declaration
# files (see ``mcphub.validation.expand_env``). ``BENCHMARKIT_PATH_LOCATION``
# does not belong here — it is process state carried in the environment, which
# is exactly what lets a workspace switch move it. ``PROJECT_ROOT`` is
# deliberately *not* exported into the environment either: the app computes
# its own root, so the value is always right, and a generic name like
# ``PROJECT_ROOT`` exported process-wide would collide with whatever a shell,
# a CI runner or one of the connectors' own scripts calls its project root —
# the ``TEMP`` lesson in ``_RESERVED_ENV_NAMES``, seen from the other side:
# a name the world already uses cannot become an export of ours either. The
# value therefore reaches expansion from this computation alone, and a stray
# ``PROJECT_ROOT`` in the environment cannot redirect connectors away from
# this install.
def provided_env_values() -> Dict[str, str]:
    """The app-computed ``${NAME}`` values for hub declaration expansion.

    Read at call time through this module, so a patched ``PROJECT_ROOT`` (a
    test does exactly that) moves the answer — same convention as every other
    patchable name.
    """
    return {"PROJECT_ROOT": str(PROJECT_ROOT)}


# Project agent instructions: when present at the root of the working
# location, their text is appended to the Chat system prompt so a new
# conversation starts already knowing the project's conventions.
AGENT_INSTRUCTIONS_FILENAME = "AGENTS.md"


def agent_instructions_filename() -> str:
    """Name of the agent-instructions file looked for at the location root.

    Read through a function (not the constant) so a test can point the lookup
    at another filename without rewriting the module.
    """
    return AGENT_INSTRUCTIONS_FILENAME


def load_agent_instructions(location: Optional[str] = None) -> str:
    """Text of the working location's ``AGENTS.md``, "" when absent/unreadable.

    Resolved against ``BENCHMARKIT_PATH_LOCATION`` (the same folder the
    file-touching tools are pointed at) at call time, so moving the location
    through the settings dialog also moves where the instructions are looked
    for. Read per chat request rather than cached: editing ``AGENTS.md`` then
    takes effect on the very next message.
    """
    root = Path(location) if location else Path(path_location())
    try:
        return (root / agent_instructions_filename()).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def agent_instructions_block(text: Optional[str] = None) -> str:
    """The instructions rendered as a system-prompt block ("" when no text).

    The framing tells the model these are the project's own instructions —
    they win over generic wording for anything project-specific — and pins
    the project root to the working location, which is what makes the paths
    the instructions mention (``src/``, ``resources/``, ``tests/``) and the
    tool UIDs they describe resolve to the real project.
    """
    body = load_agent_instructions() if text is None else (text or "").strip()
    if not body:
        return ""
    return (
        "The current project root is {root}\n\n"
        "The following are this project's own instructions (AGENTS.md at that "
        "root). Follow them for anything project-specific; they take "
        "precedence over generic wording.\n\n{body}"
    ).format(root=path_location(), body=body)


def questions_path() -> Path:
    """Resolve the questions file path from the QUESTION_SET env var.

    Defaults to ``legacy`` which reads ``resources/legacy.json``.
    Resolved at call time so ``.env`` (loaded at app startup) takes effect.
    """
    name = os.environ.get("QUESTION_SET", QUESTION_SET_DEFAULT)
    return RESOURCES_DIR / f"{name}.json"


def list_question_sets() -> List[Dict[str, Any]]:
    """Discover available question sets in ``resources/``.

    Each ``*.json`` file must contain a ``"questions"`` key. Returns a list of
    ``{id, label, count}`` dicts sorted by filename.
    """
    sets: List[Dict[str, Any]] = []
    for path in sorted(RESOURCES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            questions = data.get("questions", [])
            if not isinstance(questions, list):
                continue
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        sets.append(
            {
                "id": path.stem,
                "label": path.stem.capitalize(),
                "count": len(questions),
            }
        )
    return sets


def load_questions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the benchmark questions from the configured question set."""
    if path is None:
        path = questions_path()
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


def load_endpoints_config() -> Tuple[Optional[EndpointsConfig], Optional[ValidationError]]:
    """Build and validate the endpoints config from the environment.

    Returns ``(config, None)`` on success or ``(None, error)`` when the
    environment is invalid.
    """
    try:
        return EndpointsConfig.from_env(), None
    except ValidationError as exc:
        return None, exc


def load_settings() -> Dict[str, str]:
    """Return the effective value of every managed .env key (empty if unset).

    ``os.environ`` already reflects the .env file loaded at startup (real
    environment variables keep precedence) and is refreshed on every save.
    ``QUESTION_SET`` and ``LLM_SYSPROMPT`` resolve to their defaults when unset.
    """
    defaults = {
        "QUESTION_SET": QUESTION_SET_DEFAULT,
        "LLM_SYSPROMPT": DEFAULT_LLM_SYSPROMPT,
        "MAX_TOOL_ITERATION": str(DEFAULT_MAX_TOOL_ITERATION),
    }
    return {
        key: os.environ.get(key) or defaults.get(key, "")
        for key in SETTING_KEYS
    }


def save_settings(values: Dict[str, str]) -> None:
    """Persist managed settings to the .env file and refresh ``os.environ``.

    Empty optional values are written as commented-out lines and removed from
    the environment; required keys are always written. The file is regenerated
    from a fixed template mirroring ``.env.example``. ``QUESTION_SET`` always
    resolves to ``QUESTION_SET_DEFAULT`` when unset, so it is never written
    as a commented-out line.
    """
    clean: Dict[str, str] = {
        key: (values.get(key) or "").strip() for key in SETTING_KEYS
    }
    # QUESTION_SET always has a value; default to "legacy" when unset.
    if not clean["QUESTION_SET"]:
        clean["QUESTION_SET"] = QUESTION_SET_DEFAULT
    ENV_PATH.write_text(_render_env(clean), encoding="utf-8")
    for key in SETTING_KEYS:
        if clean[key]:
            os.environ[key] = clean[key]
        else:
            os.environ.pop(key, None)
    # The fallback key goes with them: the regenerated file no longer holds it,
    # so an environment still carrying the legacy name the boot loaded would
    # keep the switch on through a save that unticked it. Dropping it here is
    # what makes the fallback last exactly as long as the hand-written .env it
    # exists for — no longer.
    if not clean[LLM_DEBUG_KEY]:
        os.environ.pop(LEGACY_LLM_DEBUG_KEY, None)
    # Re-apply the debug switch from the values just written, so ticking the
    # box in the settings dialog takes effect on the *current* process (the
    # next call is what gets the extra logging) instead of waiting for a
    # server restart.
    apply_llm_debug()


def _render_env(values: Dict[str, str]) -> str:
    """Render the .env file content from the managed values."""

    def line(key: str) -> str:
        value = values.get(key, "")
        if not value:
            return f"#{key}="
        return f"{key}={_encode_value(value)}"

    lines = [
        "# Benchmark'it configuration (managed by the settings dialog).",
        "# Real environment variables take precedence over the values defined here.",
        "",
        "# Target LLM endpoint",
        line("SERVER_URL"),
        line("SERVER_URL_API_KEY"),
        line("ENDPOINT"),
        "# Set to 1 when the model accepts image input (enables Chat image attach)",
        line("SUPPORTS_VISION"),
        line("LLM_SYSPROMPT"),
        "",
        "# Sampling parameters (optional)",
        line("TEMPERATURE"),
        line("TOP_P"),
        line("TOP_K"),
        line("MIN_P"),
        line("REPEAT_PENALTY"),
        line("PRESENCE_PENALTY"),
        line("REASONING"),
        line("QUESTION_SET"),
        "",
        "# LLM_DEBUG=1 logs every request's payload (URL, model, body) to the",
        "# server console at DEBUG, so a failed call can be read off it. The",
        "# refused call's status and upstream body reach the UI's error line",
        "# whether or not this is on; what the flag adds is the payload.",
        "# Verbose, so switch it back off once the failure has been read.",
        line("LLM_DEBUG"),
        "",
        "# Verifier agent (optional)",
        line("VERIFIER_AGENT_SYSPROMPT"),
        line("VERIFIER_AGENT_SERVERURL"),
        line("VERIFIER_AGENT_SERVERURL_API_KEY"),
        line("VERIFIER_AGENT_ENDPOINT"),
        "",
        "# MCP connectors (Chat tab only) are declared only in",
        "# .nojae/mcp-settings.json at the working location: seeded from",
        "# mcp_connectors/mcp-settings.default.json when a folder becomes the",
        "# working location, their per-entry ``enable`` flag is the single",
        "# source of truth, no .env key is involved.",
        "# LSP language servers (Chat tab only) — ids of the servers in",
        "# .nojae/lsp-settings.json to activate on top of their",
        "# per-entry ``enable`` flag (both must agree).",
        line("LSP_ENABLED"),
        "# Cap on the tool loop of one chat reply, MCP + LSP together.",
        line("MAX_TOOL_ITERATION"),
        "",
        "# Absolute path to the python interpreter the stdio connectors run",
        "# on (the console and fedit commands in .nojae/mcp-settings.json",
        "# expand ${BENCHMARKIT_PYTHON_ROOT} from it). Unset leaves their",
        "# commands unresolved — the MCP panel names the key.",
        line("BENCHMARKIT_PYTHON_ROOT"),
        "",
    ]
    return "\n".join(lines)


def max_tool_iteration() -> int:
    """Tool-iteration cap for one chat reply; invalid or unset values fall
    back to ``DEFAULT_MAX_TOOL_ITERATION``.

    The key was renamed ``MAX_TOOL_ITERATION`` because the cap covers the
    whole tool loop of one chat reply — MCP connectors and LSP language
    servers share the single counter (and the UI's progress gauge).
    ``MAX_MCP_ITERATION`` is still honoured so an existing .env keeps working.
    """
    raw = (
        os.environ.get("MAX_TOOL_ITERATION", "")
        or os.environ.get("MAX_MCP_ITERATION", "")
    ).strip()
    if not raw:
        return DEFAULT_MAX_TOOL_ITERATION
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_TOOL_ITERATION


def lsp_enabled_ids() -> List[str]:
    """Language-server ids activated for the Chat tab from ``LSP_ENABLED``.

    Comma-separated, deduplicated, order-preserving; empty entries ignored.
    A server is only actually started when this list contains its id *and*
    its ``enable`` flag is true in ``.nojae/lsp-settings.json`` — the env key
    lets a whole prepared config file stay dormant.
    """
    ids: List[str] = []
    for part in os.environ.get("LSP_ENABLED", "").split(","):
        part = part.strip()
        if part and part not in ids:
            ids.append(part)
    return ids


def _encode_value(value: str) -> str:
    """Quote a .env value when it contains characters dotenv would misread."""
    if any(char in value for char in '#"$\\ \n\t') or "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    return value
