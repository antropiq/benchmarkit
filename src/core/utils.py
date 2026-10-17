import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.core.job import RunJob

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results.json"


@dataclass
class LLMResult:
    """Timing and token counts for a single streamed answer.

    All times are measured with ``time.monotonic()``:
    ``took`` spans request start to the last content chunk (TTFT included),
    ``ttft`` spans request start to the first content chunk, and
    ``gen_duration`` spans first to last content chunk — the window used for
    generation (decode) tokens per second.
    """

    text: str = ""
    took: float = 0.0
    ttft: float = 0.0
    gen_duration: float = 0.0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    prompt_tokens: int = 0


def estimate_completion_tokens(text: str) -> int:
    """Approximate generated-token count from text when usage is unavailable.

    Heuristic only (~4 chars/token); results using it are flagged as
    estimated so they are never mistaken for server-reported counts.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 4)


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
