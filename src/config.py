"""Application configuration: .env loading, questions, and endpoint config."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

from src.core.models import EndpointsConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "resources" / "questions.json"
ENV_PATH = PROJECT_ROOT / ".env"

# Managed .env keys, in the order they are written to the file.
SETTING_KEYS: Tuple[str, ...] = (
    "SERVER_URL",
    "ENDPOINT",
    "TEMP",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "REPEAT_PENALTY",
    "PRESENCE_PENALTY",
    "REASONING",
    "VERIFIER_AGENT_SYSPROMPT",
    "VERIFIER_AGENT_SERVERURL",
    "VERIFIER_AGENT_ENDPOINT",
)

# Keys always written to .env; the others are commented out when empty.
REQUIRED_SETTING_KEYS: Tuple[str, ...] = ("SERVER_URL", "ENDPOINT")


def load_dotenv_once() -> None:
    """Load the .env file once; existing environment variables take precedence."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """Load the benchmark questions from disk."""
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
    """
    return {key: os.environ.get(key, "") for key in SETTING_KEYS}


def save_settings(values: Dict[str, str]) -> None:
    """Persist managed settings to the .env file and refresh ``os.environ``.

    Empty optional values are written as commented-out lines and removed from
    the environment; required keys are always written. The file is regenerated
    from a fixed template mirroring ``.env.example``.
    """
    clean: Dict[str, str] = {
        key: (values.get(key) or "").strip() for key in SETTING_KEYS
    }
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
        line("ENDPOINT"),
        "# Optional comma-separated list of models (ENDPOINT takes precedence)",
        "#ENDPOINTS=model-a,model-b",
        "",
        "# Sampling parameters (optional)",
        line("TEMP"),
        line("TOP_P"),
        line("TOP_K"),
        line("MIN_P"),
        line("REPEAT_PENALTY"),
        line("PRESENCE_PENALTY"),
        line("REASONING"),
        "",
        "# Verifier agent (optional)",
        line("VERIFIER_AGENT_SYSPROMPT"),
        line("VERIFIER_AGENT_SERVERURL"),
        line("VERIFIER_AGENT_ENDPOINT"),
        "",
    ]
    return "\n".join(lines)


def _encode_value(value: str) -> str:
    """Quote a .env value when it contains characters dotenv would misread."""
    if any(char in value for char in '#"$\\ \n\t') or "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    return value
