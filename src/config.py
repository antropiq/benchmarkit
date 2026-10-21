"""Application configuration: .env loading, questions, and endpoint config."""

import json
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

# Cap on tool-calling iterations for one chat reply; shared by all connectors.
DEFAULT_MAX_MCP_ITERATION = 25

# Cap on the agent instruction document injected into the chat system prompt.
# Deliberately far below the 2 MB the connectors accept elsewhere: this text is
# paid for on every chat round-trip, on top of the tool schemas.
AGENT_DOC_MAX_BYTES = 8192

# Document name read from the user config dir (see agent_doc_dir).
AGENT_DOC_FILENAME = "AGENTS.md"

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
    "MCP_ENABLED",
    "MAX_MCP_ITERATION",
)

# Keys always written to .env; the others are commented out when empty.
REQUIRED_SETTING_KEYS: Tuple[str, ...] = ("SERVER_URL", "ENDPOINT")


def load_dotenv_once() -> None:
    """Load the .env file once; existing environment variables take precedence."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


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
        "MAX_MCP_ITERATION": str(DEFAULT_MAX_MCP_ITERATION),
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
        "# MCP connectors (Chat tab only)",
        line("MCP_ENABLED"),
        line("MAX_MCP_ITERATION"),
        "",
    ]
    return "\n".join(lines)


def mcp_enabled_ids() -> List[str]:
    """Connector ids activated for the Chat tab from ``MCP_ENABLED``.

    Comma-separated, deduplicated, order-preserving; empty entries ignored.
    """
    ids: List[str] = []
    for part in os.environ.get("MCP_ENABLED", "").split(","):
        part = part.strip()
        if part and part not in ids:
            ids.append(part)
    return ids


def max_mcp_iteration() -> int:
    """Tool-iteration cap for one chat reply; invalid or unset values fall
    back to ``DEFAULT_MAX_MCP_ITERATION``."""
    raw = os.environ.get("MAX_MCP_ITERATION", "").strip()
    if not raw:
        return DEFAULT_MAX_MCP_ITERATION
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_MCP_ITERATION


def agent_doc_dir() -> Path:
    """User-scope config directory: ``$XDG_CONFIG_HOME/benchmarkit``.

    Falls back to ``~/.config/benchmarkit`` when ``XDG_CONFIG_HOME`` is unset
    or empty, per the XDG Base Directory specification. Resolved at call time
    so a changed environment takes effect without a restart.
    """
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if not base:
        base = str(Path.home() / ".config")
    return Path(base).expanduser() / "benchmarkit"


def agent_doc_path() -> Path:
    """Path of the chat agent instruction document.

    ``AGENT_DOC`` overrides the location outright (and may point at a
    nonexistent file to disable the feature). Otherwise the conventional
    ``AGENTS.md`` inside :func:`agent_doc_dir` is used.
    """
    explicit = os.environ.get("AGENT_DOC", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return agent_doc_dir() / AGENT_DOC_FILENAME


def load_agent_doc() -> str:
    """Read the user-scope agent instruction document, "" when unusable.

    Called on every chat request, so it must never raise: a missing, unreadable
    or empty document is reported as an empty string and chat proceeds normally.
    The content is capped at ``AGENT_DOC_MAX_BYTES`` (bytes, not characters) so
    an oversized document cannot inflate the system prompt; a document cut in the
    middle of a multi-byte character decodes to U+FFFD rather than failing.
    """
    path = agent_doc_path()
    try:
        if not path.is_file():
            return ""
        raw = path.read_bytes()[:AGENT_DOC_MAX_BYTES]
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace").strip()


def chat_sysprompt(base: str) -> str:
    """Compose the chat system prompt: the configured prompt plus the agent doc.

    Used by the Chat endpoint only -- a benchmark run keeps sending
    ``LLM_SYSPROMPT`` verbatim, so its measured behaviour stays comparable
    across runs no matter what this file contains. Returns ``base`` unchanged
    when no document is available.

    The framing text deliberately avoids the word ``tool``: ``stream_chat``
    appends its generic "you can call the provided tools" notice only when the
    system prompt does not mention tools, so wording it here would suppress
    that notice for every document, including one that says nothing about
    tools. Whether the notice is replaced is left to the document's own text.
    """
    doc = load_agent_doc()
    if not doc:
        return base
    return (
        f"{base}\n\n"
        "# Additional rules for this conversation\n"
        "These instructions come from the user's agent configuration file. "
        "Follow them unless they conflict with a direct instruction from the "
        "user.\n\n"
        f"{doc}\n"
    )


def _encode_value(value: str) -> str:
    """Quote a .env value when it contains characters dotenv would misread."""
    if any(char in value for char in '#"$\\ \n\t') or "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    return value
