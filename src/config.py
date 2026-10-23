"""Application configuration: .env loading, questions, and endpoint config."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

from src.core.models import DEFAULT_LLM_SYSPROMPT, EndpointsConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
RESOURCES_DIR = PROJECT_ROOT / "resources"

QUESTION_SET_DEFAULT = "legacy"

# Chat-tab memory bullets: fixed-size slot registration stored in
# ``resources/memory_registration.json`` (never a question set).
MEMORY_REGISTRATION_FILENAME = "memory_registration.json"
MEMORY_SLOT_COUNT = 9
MEMORY_DEFAULT_COLOR = "#808080"
MEMORY_CONTENT_MAX_CHARS = 32000

MEMORY_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# Cap on tool-calling iterations for one chat reply; shared by MCP connectors
# and LSP language servers alike (one knob, no split-brain).
DEFAULT_MAX_TOOL_ITERATION = 25

# Managed .env keys, in the order they are written to the file.
SETTING_KEYS: Tuple[str, ...] = (
    "SERVER_URL",
    "SERVER_URL_API_KEY",
    "ENDPOINT",
    "LLM_SYSPROMPT",
    "TEMP",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "REPEAT_PENALTY",
    "PRESENCE_PENALTY",
    "REASONING",
    "SUPPORTS_VISION",
    "QUESTION_SET",
    "VERIFIER_AGENT_SYSPROMPT",
    "VERIFIER_AGENT_SERVERURL",
    "VERIFIER_AGENT_SERVERURL_API_KEY",
    "VERIFIER_AGENT_ENDPOINT",
    "MAX_TOOL_ITERATION",
    "LSP_ENABLED",
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
        if path.name == MEMORY_REGISTRATION_FILENAME:
            continue
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


def memory_path() -> Path:
    """Path of the memory-bullet slot registration file."""
    return RESOURCES_DIR / MEMORY_REGISTRATION_FILENAME


def default_memory_registrations() -> List[Dict[str, str]]:
    """The fresh-state registration: empty gray slots, nothing configured."""
    return [
        {"color": MEMORY_DEFAULT_COLOR, "content": ""}
        for _ in range(MEMORY_SLOT_COUNT)
    ]


def _normalize_memory_slot(item: Any) -> Dict[str, str]:
    """Coerce one raw JSON entry to a ``{color, content}`` string pair.

    A missing or malformed color (anything but a 6-digit hex) falls back to
    the default gray so rendering never receives an invalid CSS color; a
    non-string content becomes empty.
    """
    raw_color = item.get("color") if isinstance(item, dict) else None
    raw_content = item.get("content") if isinstance(item, dict) else None
    color = (
        raw_color
        if isinstance(raw_color, str) and MEMORY_COLOR_RE.match(raw_color)
        else MEMORY_DEFAULT_COLOR
    )
    content = raw_content if isinstance(raw_content, str) else ""
    return {"color": color, "content": content}


def _write_memory_registrations(slots: List[Dict[str, str]]) -> None:
    """Persist the normalized registration (same simple write as save_settings)."""
    payload = {"registration": slots}
    memory_path().write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_memory_registrations() -> List[Dict[str, str]]:
    """Read the memory-bullet slots, self-healing the file to 9 normalized slots.

    A missing file, invalid JSON or a missing/malformed ``registration`` list
    falls back to the defaults; the normalized result is also written back
    whenever the file did not already contain exactly those 9 slots (pad /
    truncate, malformed entries coerced), so the file on disk is always valid.
    """
    path = memory_path()
    raw: Optional[List[Any]] = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("registration"), list):
            raw = data["registration"]
    except (json.JSONDecodeError, OSError, ValueError):
        raw = None
    slots = [_normalize_memory_slot(item) for item in (raw or [])]
    slots = (slots + default_memory_registrations())[:MEMORY_SLOT_COUNT]
    if (
        raw is None
        or len(raw) != MEMORY_SLOT_COUNT
        or any(raw[i] != slots[i] for i in range(MEMORY_SLOT_COUNT))
    ):
        _write_memory_registrations(slots)
    return slots


def save_memory_registration(index: int, color: str, content: str) -> List[Dict[str, str]]:
    """Write one memory-bullet slot and return the full normalized registration."""
    slots = load_memory_registrations()
    slots[index] = {"color": color, "content": content}
    _write_memory_registrations(slots)
    return slots


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
        line("TEMP"),
        line("TOP_P"),
        line("TOP_K"),
        line("MIN_P"),
        line("REPEAT_PENALTY"),
        line("PRESENCE_PENALTY"),
        line("REASONING"),
        line("QUESTION_SET"),
        "",
        "# Verifier agent (optional)",
        line("VERIFIER_AGENT_SYSPROMPT"),
        line("VERIFIER_AGENT_SERVERURL"),
        line("VERIFIER_AGENT_SERVERURL_API_KEY"),
        line("VERIFIER_AGENT_ENDPOINT"),
        "",
        "# MCP connectors (Chat tab only) are declared only in",
        "# mcp_connectors/externals.json: their per-entry ``enable`` flag is",
        "# the single source of truth, no .env key is involved.",
        "# LSP language servers (Chat tab only) — ids of the servers in",
        "# lsp_connectors/lsp_servers.json to activate on top of their",
        "# per-entry ``enable`` flag (both must agree).",
        line("LSP_ENABLED"),
        "# Cap on the tool loop of one chat reply, MCP + LSP together.",
        line("MAX_TOOL_ITERATION"),
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
    its ``enable`` flag is true in ``lsp_servers.json`` — the env key lets a
    whole prepared config file stay dormant.
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
