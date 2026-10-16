import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from src.core.job import RunJob

LLMResult = Tuple[str, float, int]  # (response_text, took, tokens_used)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results.json"


def format_time(seconds: float) -> str:
    """Return a string HH:MM:SS for given seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_results(job: "RunJob", path: Path = RESULTS_PATH) -> str:
    """Save benchmark results to a JSON file and return the serialized data.

    Raises on write failure so the caller can surface the error.
    """
    data = json.dumps(job.to_output(), indent=2)
    path.write_text(data, encoding="utf-8")
    return data
