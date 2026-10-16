"""Application configuration: .env loading, questions, and endpoint config."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

from src.core.models import EndpointsConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "resources" / "questions.json"
ENV_PATH = PROJECT_ROOT / ".env"


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
