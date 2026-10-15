import os
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

ReasoningEffort = Literal["off", "low", "medium", "high", "xhigh"]

SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "reasoning_effort",
)


class VerifierAgent(BaseModel):
    """Verifier LLM settings from the ``verifierAgent`` config block."""

    model_config = ConfigDict(extra="ignore")

    sysprompt: str = ""
    serverUrl: str = ""
    endpoint: str = ""


class SamplingParams(BaseModel):
    """Sampling parameters mapped to litellm/OpenAI-compatible names.

    Config keys (``temp``, ``top-p``, ...) are accepted as aliases; values
    may be strings or numbers and are coerced by pydantic.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    temperature: Optional[float] = Field(default=None, alias="temp")
    top_p: Optional[float] = Field(default=None, alias="top-p")
    top_k: Optional[int] = Field(default=None, alias="top-k")
    min_p: Optional[float] = Field(default=None, alias="min-p")
    repeat_penalty: Optional[float] = Field(default=None, alias="repeat-penalty")
    presence_penalty: Optional[float] = Field(default=None, alias="presence-penalty")
    reasoning_effort: Optional[ReasoningEffort] = Field(default=None, alias="reasoning")

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
            if not value:
                return None
        return value

    def to_kwargs(self) -> Dict[str, Any]:
        """Return only the sampling parameters as litellm kwargs."""
        return self.model_dump(include=set(SAMPLING_FIELDS), exclude_none=True)


class EndpointsConfig(SamplingParams):
    """Benchmark configuration built from environment variables."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    serverUrl: str = ""
    endpoint: Optional[Union[str, List[str]]] = None
    endpoints: Optional[Union[str, List[str]]] = None
    verifierAgent: Optional[VerifierAgent] = None

    @field_validator("endpoint", "endpoints", mode="before")
    @classmethod
    def _strip_endpoints(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, list):
            cleaned = [e.strip() for e in value if isinstance(e, str) and e.strip()]
            return cleaned or None
        return value

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "EndpointsConfig":
        """Build the configuration from environment variables.

        Recognized keys: ``SERVER_URL``, ``ENDPOINT``, ``ENDPOINTS``
        (comma-separated), ``TEMP``, ``TOP_P``, ``TOP_K``, ``MIN_P``,
        ``REPEAT_PENALTY``, ``PRESENCE_PENALTY``, ``REASONING`` and the
        ``VERIFIER_AGENT_*`` block (``SYSPROMPT``, ``SERVERURL``,
        ``ENDPOINT``).
        """
        source: Mapping[str, str] = os.environ if env is None else env

        def get(key: str) -> Optional[str]:
            value = source.get(key)
            if isinstance(value, str):
                value = value.strip()
            return value or None

        endpoints = get("ENDPOINTS")
        data: Dict[str, Any] = {
            "serverUrl": get("SERVER_URL") or "",
            "endpoint": get("ENDPOINT"),
            "endpoints": [e.strip() for e in endpoints.split(",")] if endpoints else None,
            "temp": get("TEMP"),
            "top-p": get("TOP_P"),
            "top-k": get("TOP_K"),
            "min-p": get("MIN_P"),
            "repeat-penalty": get("REPEAT_PENALTY"),
            "presence-penalty": get("PRESENCE_PENALTY"),
            "reasoning": get("REASONING"),
        }
        verifier = {
            "sysprompt": get("VERIFIER_AGENT_SYSPROMPT") or "",
            "serverUrl": get("VERIFIER_AGENT_SERVERURL") or "",
            "endpoint": get("VERIFIER_AGENT_ENDPOINT") or "",
        }
        if any(verifier.values()):
            data["verifierAgent"] = verifier
        return cls.model_validate(data)

    def model_list(self) -> List[str]:
        """Return the tested model names.

        Supports the single ``endpoint`` value and the comma-separated
        ``endpoints`` list; ``endpoint`` takes precedence.
        """
        for value in (self.endpoint, self.endpoints):
            if isinstance(value, str):
                return [value]
            if isinstance(value, list) and value:
                return list(value)
        return []
